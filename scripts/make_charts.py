"""Before / after charts with llama.cpp (docs/STORY.md, docs/articles/).

Reads results/processed/decode_levers/*_tps*.json (our engine) and *_llamacpp_ctx*.json (llama.cpp, matched
weights and BF16 KV), writes one SVG per model (stable tokens/s from 2k to 128k context) and a grouped-bar
summary, plus figures/chart_data.json (the same numbers, for the dashboard). Standard library only; PNGs are
rendered from the SVGs with a headless browser (see --png).

  before  = our engine at the start of the matched-quantization comparison (exact, unfused: follow-ups J / K)
  after   = our best exact setting now, per context (fused step F3a, INT4 self-draft L7b, MTP draft H)
  llama.cpp = its best per context of plain decoding and its own speculative decoding, same weights
"""

import argparse
import glob
import json
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "processed" / "decode_levers"
CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
COL = {"before": "#8a8984", "after": "#2a78d6", "draft": "#1baf7a", "llama": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

MODELS = [
    {"key": "qwen3_4b", "title": "Qwen3-4B, 8-bit weights (Q8_0)", "run": "Qwen3-4B-Base",
     "before": ("_tps_q8.json", ["l12"]), "after": ("_tps_q8_fused.json", ["l12c"]),
     "draft": [("_tps_q8_fused.json", ["l7bc"])], "draft_name": "Ours + 4-bit self-draft", "llama_label": "q8_0"},
    {"key": "gemma4_12b", "title": "Gemma 4 12B, 8-bit weights (Q8_0)", "run": "gemma-4-12B",
     "before": ("_tps_q8.json", ["l12"]), "after": ("_tps_q8_fused.json", ["l12c"]), "after_extra": ("_tps_q8_fused_128k.json", ["l12c"]),
     "draft": [("_tps_q8_fused.json", ["l7bc"])], "draft_name": "Ours + 4-bit self-draft", "llama_label": "q8_0"},
    {"key": "qwen38_27b", "title": "Qwen3.8-27B, 4-bit weights (Red Hat INT4 = Q4_0)", "run": "Qwen3.8-27B-INT4",
     "before": ("_tps.json", ["l12"]), "after": ("_tps_fused.json", ["l12c"]),
     "draft": [("_tps_mtp_h2.json", ["mtp4"]), ("_tps_mtp_h2_128k.json", ["mtp4"])], "draft_name": "Ours + MTP draft (k = 4)", "llama_label": None},
]


def ours(run, suffix, settings):
    """{context: (best tokens/s, setting)} over the given settings (None: every *_tps key)."""
    f = RES / f"{run}{suffix}"
    if not f.exists():
        return {}
    out = {}
    for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
        keys = [k for k in r if k.endswith("_tps")] if settings is None else [f"{s}_tps" for s in settings]
        best = max(((r[k], k[:-4]) for k in keys if isinstance(r.get(k), (int, float))), default=None)
        if best:
            out[r["context"]] = best
    return out


def llama(run, label):
    """{context: {config/k: tokens/s}} from the matched llama.cpp runs (median of reps)."""
    out = {}
    for f in glob.glob(str(RES / f"{run}_llamacpp_ctx*_kvbf16*.json")) + glob.glob(str(RES / "llamacpp_b12720" / "**" / f"{run}_llamacpp_ctx*_kvbf16*.json"), recursive=True):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        tl = d["args"].get("target_label")
        if label is not None and tl != label:
            continue
        if label is None and tl not in (None, "RedHatQ4_0", "bf16"):
            continue
        g = defaultdict(list)
        for r in d["rows"]:
            g[(r["config"], r["k"])].append(r["ms_per_token"])
        ctx = min(CTX, key=lambda c: abs(c - 320 - d["args"]["context"]))
        out.setdefault(ctx, {}).update({f"{c} k={k}": 1000 / statistics.median(v) for (c, k), v in g.items()})
    return out


def collect():
    data = []
    for m in MODELS:
        before = {c: v[0] for c, v in ours(m["run"], *m["before"]).items()}
        after = ours(m["run"], *m["after"])
        if m.get("after_extra"):   # Gemma 128k: rerun without the draft copy (memory)
            after = {**ours(m["run"], *m["after_extra"]), **after}
        draft = {}
        for suf, st in m["draft"]:
            for c, (v, s) in ours(m["run"], suf, st).items():
                if c not in draft or v > draft[c][0]:
                    draft[c] = (v, s)
        ll = llama(m["run"], m["llama_label"])
        data.append({**{k: m[k] for k in ("key", "title")},
                     "before": before, "after": {c: v[0] for c, v in after.items()},
                     "draft": {c: v[0] for c, v in draft.items()}, "draft_setting": {c: v[1] for c, v in draft.items()},
                     "draft_name": m["draft_name"],
                     "llama": {c: max(v.values()) for c, v in ll.items()},
                     "llama_setting": {c: max(v, key=v.get) for c, v in ll.items()}, "llama_all": ll})
    return data


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def line_chart(m, path):
    W, H, L, R, T, B = 1200, 675, 96, 330, 118, 86
    series = [("before", "Ours, before", m["before"], "8 6"), ("after", "Ours, after (fused)", m["after"], None),
              ("draft", m["draft_name"], m["draft"], None), ("llama", "llama.cpp (its best)", m["llama"], None)]
    series = [s for s in series if s[2]]
    ymax = max(v for _, _, s, _ in series for v in s.values()) * 1.12
    step = next(s for s in (10, 20, 25, 50, 100) if ymax / s <= 7)
    ymax = step * (int(ymax / step) + 1)
    x = lambda c: L + CTX.index(c) * (W - L - R) / (len(CTX) - 1)
    y = lambda v: T + (H - T - B) * (1 - v / ymax)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{L}" y="52" font-size="30" font-weight="600" fill="{INK}">{esc(m["title"])}</text>',
         f'<text x="{L}" y="86" font-size="19" fill="{INK2}">Stable decode speed, tokens per second (higher is better), same weights in both engines, one RTX 5090</text>']
    for i in range(0, int(ymax / step) + 1):
        v = i * step
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{L - 12}" y="{y(v) + 6:.1f}" font-size="17" fill="{INK2}" text-anchor="end">{v}</text>')
    for c in CTX:
        o.append(f'<text x="{x(c):.1f}" y="{H - B + 32}" font-size="17" fill="{INK2}" text-anchor="middle">{c // 1024}k</text>')
    o.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 22}" font-size="18" fill="{INK2}" text-anchor="middle">Context length (tokens already in the conversation)</text>')
    ends, starts = [], []
    for key, name, s, dash in series:
        pts = [(c, s[c]) for c in CTX if c in s]
        if not pts:
            continue
        d = " ".join(f"{x(c):.1f},{y(v):.1f}" for c, v in pts)
        o.append(f'<polyline points="{d}" fill="none" stroke="{COL[key]}" stroke-width="3" stroke-linejoin="round"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        for c, v in pts:
            o.append(f'<circle cx="{x(c):.1f}" cy="{y(v):.1f}" r="5.5" fill="{COL[key]}" stroke="{SURF}" stroke-width="2"/>')
        starts.append([y(pts[0][1]) - 14, pts[0][1], x(pts[0][0])])
        ends.append([y(pts[-1][1]), f"{name}  {pts[-1][1]:.0f}" + ("" if pts[-1][0] == CTX[-1] else f" (at {pts[-1][0] // 1024}k)"), key, x(pts[-1][0])])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 26)
    for yy, label, key, xx in ends:
        o.append(f'<line x1="{W - R + 12}" x2="{W - R + 34}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{COL[key]}" stroke-width="3"'
                 + (' stroke-dasharray="6 4"' if key == "before" else "") + "/>")
        o.append(f'<circle cx="{W - R + 23}" cy="{yy:.1f}" r="4.5" fill="{COL[key]}"/>')
        o.append(f'<text x="{W - R + 42}" y="{yy + 6:.1f}" font-size="17" fill="{INK}">{esc(label)}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def bar_chart(data, ctx, path):
    W, H, L, R, T, B = 1200, 675, 96, 40, 150, 120
    groups = [m for m in data if ctx in m["after"] or ctx in m["llama"]]
    keys = [("before", "Ours, before"), ("after", "Ours, after"), ("draft", "Ours, after + draft"), ("llama", "llama.cpp (its best)")]
    ymax = max(m[k].get(ctx, 0) for m in groups for k, _ in keys) * 1.15
    step = next(s for s in (10, 20, 25, 50, 100) if ymax / s <= 7)
    ymax = step * (int(ymax / step) + 1)
    y = lambda v: T + (H - T - B) * (1 - v / ymax)
    gw = (W - L - R) / len(groups)
    bw = min(90, gw / 4.2)
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{L}" y="52" font-size="30" font-weight="600" fill="{INK}">Before and after, at {ctx // 1024}k context</text>',
         f'<text x="{L}" y="86" font-size="19" fill="{INK2}">Tokens per second (higher is better), same quantized weights in both engines, exact output</text>']
    lx = L
    for k, name in keys:
        o.append(f'<rect x="{lx}" y="108" width="18" height="18" rx="3" fill="{COL[k]}"/>')
        o.append(f'<text x="{lx + 26}" y="123" font-size="17" fill="{INK}">{name}</text>')
        lx += 26 + 10 * len(name) + 40
    for i in range(0, int(ymax / step) + 1):
        v = i * step
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{L - 12}" y="{y(v) + 6:.1f}" font-size="17" fill="{INK2}" text-anchor="end">{v}</text>')
    for gi, m in enumerate(groups):
        cx = L + gw * (gi + 0.5)
        for bi, (k, _) in enumerate(keys):
            v = m[k].get(ctx)
            bx = cx + (bi - len(keys) / 2) * (bw + 4) + 2
            if v is None:
                o.append(f'<text x="{bx + bw / 2:.1f}" y="{y(0) - 10:.1f}" font-size="15" fill="{INK2}" text-anchor="middle">n/a</text>')
                continue
            top, h = y(v), y(0) - y(v)
            o.append(f'<path d="M{bx:.1f},{y(0):.1f} V{top + 4:.1f} Q{bx:.1f},{top:.1f} {bx + 4:.1f},{top:.1f} H{bx + bw - 4:.1f} '
                     f'Q{bx + bw:.1f},{top:.1f} {bx + bw:.1f},{top + 4:.1f} V{y(0):.1f} Z" fill="{COL[k]}"/>' if h > 4 else
                     f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{COL[k]}"/>')
            o.append(f'<text x="{bx + bw / 2:.1f}" y="{top - 10:.1f}" font-size="17" font-weight="600" fill="{INK}" text-anchor="middle">{v:.0f}</text>')
        name = m["title"].split(",")[0]
        o.append(f'<text x="{cx:.1f}" y="{H - B + 36}" font-size="19" fill="{INK}" text-anchor="middle">{esc(name)}</text>')
        o.append(f'<text x="{cx:.1f}" y="{H - B + 62}" font-size="16" fill="{INK2}" text-anchor="middle">{esc(m["title"].split(", ")[1])}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def journey_chart(path):
    """Qwen3-4B at 2k context, step by step (our engine) with llama.cpp on the same weights, horizontal bars."""
    def tps(f, ctx, key):
        rows = json.loads((RES / f).read_text(encoding="utf-8"))["rows"]
        return next(r[key] for r in rows if r["context"] == ctx)
    lc = llama("Qwen3-4B-Base", "q8_0")[2048]
    lb = llama("Qwen3-4B-Base", None)
    bars = [("16-bit weights (BF16)", None, None),
            ("Ours: fast attention + fused RoPE", tps("Qwen3-4B-Base_tps.json", 2048, "l12_tps"), "after"),
            ("Ours: + its own 4-bit copy as draft", tps("Qwen3-4B-Base_tps.json", 2048, "l7b_tps"), "draft"),
            ("llama.cpp (its best setting)", max(v for k, v in lb[2048].items() if k.startswith("bf16")), "llama"),
            ("8-bit weights (Q8_0, byte-identical in both)", None, None),
            ("Ours: 8-bit kernel", tps("Qwen3-4B-Base_tps_q8.json", 2048, "l12_tps"), "before"),
            ("Ours: + fused tiny kernels", tps("Qwen3-4B-Base_tps_q8_fused.json", 2048, "l12c_tps"), "after"),
            ("Ours: + 4-bit draft", tps("Qwen3-4B-Base_tps_q8_fused.json", 2048, "l7bc_tps"), "draft"),
            ("llama.cpp (its best setting)", max(lc.values()), "llama")]
    W, H, L, R, T = 1200, 675, 430, 90, 120
    vmax = max(v for _, v, _ in bars if v) * 1.05
    rowh = (H - T - 40) / len(bars)
    x = lambda v: L + (W - L - R) * v / vmax
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="40" y="52" font-size="30" font-weight="600" fill="{INK}">Qwen3-4B, step by step, at 2k context</text>',
         f'<text x="40" y="86" font-size="19" fill="{INK2}">Tokens per second (longer is faster), same weights in both engines, every row the model&#39;s own output</text>']
    for i, (name, v, key) in enumerate(bars):
        yy = T + i * rowh
        if v is None:
            o.append(f'<text x="40" y="{yy + rowh * 0.7:.1f}" font-size="19" font-weight="600" fill="{INK}">{esc(name)}</text>')
            continue
        o.append(f'<text x="{L - 14}" y="{yy + rowh * 0.62:.1f}" font-size="18" fill="{INK}" text-anchor="end">{esc(name)}</text>')
        bh = rowh * 0.62
        o.append(f'<rect x="{L}" y="{yy + (rowh - bh) / 2:.1f}" width="{x(v) - L:.1f}" height="{bh:.1f}" rx="4" fill="{COL[key]}"/>')
        o.append(f'<text x="{x(v) + 10:.1f}" y="{yy + rowh * 0.62:.1f}" font-size="18" font-weight="600" fill="{INK}">{v:.0f}</text>')
    o.append("</svg>")
    path.write_text("\n".join(o), encoding="utf-8")


def to_png(svg: Path):
    exe = next((p for p in (r"C:/Program Files/Google/Chrome/Application/chrome.exe",
                            r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe") if Path(p).exists()), None)
    if exe is None:
        return
    html = svg.with_suffix(".tmp.html")
    html.write_text('<html><body style="margin:0">' + svg.read_text(encoding="utf-8") + "</body></html>", encoding="utf-8")
    prof = Path(tempfile.gettempdir()) / "ple_chart_profile"   # own profile: a running browser blocks the default one
    subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={prof}", "--window-size=1200,675",
                    f"--screenshot={svg.with_suffix('.png')}", html.as_uri()], capture_output=True, timeout=60)
    html.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "articles" / "figures"))
    ap.add_argument("--png", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect()
    files = []
    for m in data:
        f = out / f"before_after_{m['key']}.svg"
        line_chart(m, f)
        files.append(f)
    for c in (2048, 32768):
        f = out / f"before_after_bars_{c // 1024}k.svg"
        bar_chart(data, c, f)
        files.append(f)
    f = out / "journey_qwen3_4b.svg"
    journey_chart(f)
    files.append(f)
    (out / "chart_data.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    for m in data:
        print(m["title"])
        for c in CTX:
            print(f"  {c // 1024:>4}k  before {m['before'].get(c, float('nan')):6.1f}  after {m['after'].get(c, float('nan')):6.1f}"
                  f"  draft {m['draft'].get(c, float('nan')):6.1f}  llama.cpp {m['llama'].get(c, float('nan')):6.1f} ({m['llama_setting'].get(c, '-')})")
    if args.png:
        for f in files:
            to_png(f)
    print("wrote", out)


if __name__ == "__main__":
    main()
