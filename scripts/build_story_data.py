"""One flat dataset for the dashboard's chronological story (docs/STORY.md): every number of the
long-context, quantised-target, fused-kernel, MTP and Inkling chapters, each row naming its result file.

  kind = "tps"      model, context, series (before | after | draft | llama | llama_spec), tps, setting
  kind = "check"    model, setting, prompt, violations, near_ties (greedy check at the first context)
  kind = "q2"       model, k, tps, accept (llama.cpp, pure Q2_K draft for the Q8_0 target, 2k)
  kind = "inkling"  context, config, tps (llama.cpp only)

Usage: python scripts/build_story_data.py [out.json]
"""

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_charts import CTX, MODELS, RES, ROOT, ours  # noqa: E402

LABEL = {"qwen3_4b": "Qwen3-4B", "gemma4_12b": "Gemma 4 12B", "qwen38_27b": "Qwen3.8-27B"}
QUANT = {"qwen3_4b": "Q8_0", "gemma4_12b": "Q8_0", "qwen38_27b": "INT4 (Q4_0)"}


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def llama_rows(m):
    rows = []
    label = m["llama_label"]
    files = glob.glob(str(RES / f"{m['run']}_llamacpp_ctx*_kvbf16*.json")) +         glob.glob(str(RES / "llamacpp_b12720" / "**" / f"{m['run']}_llamacpp_ctx*_kvbf16*.json"), recursive=True)   # b12720: its MTP and plain reruns
    for f in files:
        build = "b12720" if "llamacpp_b12720" in f else "b12359"
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        tl = d["args"].get("target_label")
        if (label is not None and tl != label) or (label is None and tl not in (None, "RedHatQ4_0", "bf16")):
            continue
        ctx = min(CTX, key=lambda c: abs(c - 320 - d["args"]["context"]))
        g, acc = defaultdict(list), defaultdict(list)
        for r in d["rows"]:
            g[(r["config"], r["k"])].append(r["ms_per_token"])
            if r.get("accept_rate") is not None:
                acc[(r["config"], r["k"])].append(r["accept_rate"])
        for (c, k), v in g.items():
            rows.append({"kind": "tps", "model": LABEL[m["key"]], "key": m["key"], "quant": QUANT[m["key"]], "context": ctx,
                         "series": "llama" if k == 0 else "llama_spec", "setting": f"llama.cpp {build} {c}",
                         "tps": round(1000 / statistics.median(v), 1),
                         "accept": round(statistics.mean(acc[(c, k)]), 2) if acc.get((c, k)) else None, "file": rel(f)})
    return rows


def our_rows(m):
    rows = []
    specs = [("before", *m["before"]), ("after", *m["after"])] + ([("after", *m["after_extra"])] if m.get("after_extra") else []) + [("draft", s, (["mtp2", "mtp3", "mtp4"] if "mtp" in s else st)) for s, st in m["draft"]]   # every MTP k, the page picks one
    if m["key"] == "qwen38_27b":   # follow-up H2: the MTP draft with an INT4 copy of the LM head (settings named mtpK_h4)
        specs += [("draft", f, ["mtp2", "mtp3", "mtp4"]) for f in ("_tps_mtp.json", "_tps_mtp_rest.json", "_tps_mtp_128k.json")]
        specs += [("draft", "_tps_mtp_h2.json", ["mtp3", "mtp4"]), ("draft", "_tps_mtp_h2_128k.json", ["mtp3", "mtp4"])]
    for series, suf, st in specs:
        f = RES / f"{m['run']}{suf}"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        tag = "_h4" if "_h2" in suf else ""
        for r in d["rows"]:
            keys = [k[:-4] for k in r if k.endswith("_tps")] if st is None else st
            for s in keys:
                if isinstance(r.get(f"{s}_tps"), (int, float)):
                    rows.append({"kind": "tps", "model": LABEL[m["key"]], "key": m["key"], "quant": QUANT[m["key"]],
                                 "context": r["context"], "series": series, "setting": s + tag, "tps": round(r[f"{s}_tps"], 1),
                                 "accept": round(r[f"{s}_accept"], 2) if r.get(f"{s}_accept") is not None else None, "file": rel(f)})
                elif r.get(f"{s}_error") or r.get("error"):
                    rows.append({"kind": "tps", "model": LABEL[m["key"]], "key": m["key"], "quant": QUANT[m["key"]],
                                 "context": r["context"], "series": series, "setting": s + tag, "tps": None,
                                 "note": r.get(f"{s}_error") or r.get("error"), "file": rel(f)})
        for c in d.get("check", []):
            rows.append({"kind": "check", "model": LABEL[m["key"]], "key": m["key"], "series": series,
                         "setting": c["setting"] + (tag if c["setting"].startswith("mtp") else ""),
                         "prompt": c["prompt"], "violations": c["violations"], "near_ties": c["near_ties"], "file": rel(f)})
    return rows


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "processed" / "story" / "story.json"
    rows = []
    for m in MODELS:
        rows += our_rows(m) + llama_rows(m)
    # a later file for the same model / context / series / setting wins (e.g. a 128k rerun without the draft)
    seen = {}
    for r in rows:
        if r["kind"] == "tps":
            k = (r["key"], r["context"], r["series"], r["setting"])
            if k in seen and r["tps"] is None:
                continue
            seen[k] = r
    chk = {(r["key"], r["setting"], r["prompt"], r["file"]): r for r in rows if r["kind"] == "check"}   # a file read twice counts once
    rows = [r for r in rows if r["kind"] not in ("tps", "check")] + list(chk.values()) + list(seen.values())
    for f in sorted(glob.glob(str(RES / "q2_draft" / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        g, acc = defaultdict(list), defaultdict(list)
        for r in d["rows"]:
            g[r["k"]].append(r["ms_per_token"])
            if r.get("accept_rate") is not None:
                acc[r["k"]].append(r["accept_rate"])
        model = "Qwen3-4B" if "Qwen3-4B" in f else "Gemma 4 12B"
        for k, v in g.items():
            rows.append({"kind": "q2", "model": model, "k": k, "tps": round(1000 / statistics.median(v), 1),
                         "accept": round(statistics.mean(acc[k]), 2) if acc.get(k) else None, "file": rel(f)})
    for f in sorted(glob.glob(str(RES / "Inkling-2-Mini*_llamacpp_ctx*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        ctx = min(CTX, key=lambda c: abs(c - 320 - d["args"]["context"]))
        g = defaultdict(list)
        for r in d["rows"]:
            g[(r["config"], r["k"])].append(r["ms_per_token"])
        for (c, k), v in g.items():
            rows.append({"kind": "inkling", "model": "Inkling 2 Mini", "context": ctx, "config": c, "k": k,
                         "tps": round(1000 / statistics.median(v), 1), "file": rel(f)})
    rows.sort(key=lambda r: (r["kind"], r.get("key", r.get("model", "")), r.get("context", 0), r.get("series", ""), str(r.get("setting", r.get("k", "")))))
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"built_from": "scripts/build_story_data.py", "gpu": "RTX 5090, batch 1, greedy, BF16 KV cache in both engines",
            "llama_cpp": "b12359 for the matched runs (byte-identical Q8_0, the same INT4 codes as Q4_0 / Q4_1); b12720 for Inkling 2 Mini and MTP"}
    out.write_text(json.dumps({"rows": rows, "meta": meta}, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
