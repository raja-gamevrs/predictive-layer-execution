"""Markdown tables for docs/STORY.md and the article drafts, from results/processed/story/story.json
(same selection rules as the dashboard's st_tps / st_land: one draft setting per model, the fastest at 2k
among settings whose greedy check had 0 violations; a draft counts as ours only if its check was clean)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CTX = [2048, 4096, 8192, 16384, 32768, 65536, 131072]


def load():
    return json.loads((ROOT / "results" / "processed" / "story" / "story.json").read_text(encoding="utf-8"))["rows"]


def landing(rows):
    tps = [r for r in rows if r["kind"] == "tps" and r["tps"] is not None]
    checks = [r for r in rows if r["kind"] == "check"]

    def clean(key, s):
        c = [r for r in checks if r["key"] == key and r["setting"] == s.rstrip("c")]
        return bool(c) and all(r["violations"] == 0 for r in c)

    out = {}
    for key in dict.fromkeys(r["key"] for r in tps):
        d2 = sorted((r for r in tps if r["key"] == key and r["series"] == "draft" and r["context"] == 2048), key=lambda r: -r["tps"])
        pick = next((r for r in d2 if clean(key, r["setting"])), d2[0] if d2 else None)
        for c in CTX:
            here = [r for r in tps if r["key"] == key and r["context"] == c]
            best = lambda s: max((r for r in here if r["series"] == s and (s != "draft" or (pick is not None and r["setting"] == pick["setting"]))),
                                 key=lambda r: r["tps"], default=None)
            row = {s: best(s) for s in ("before", "after", "draft", "llama", "llama_spec")}
            ours = max([x for x in (row["after"], row["draft"] if row["draft"] and clean(key, row["draft"]["setting"]) else None) if x],
                       key=lambda r: r["tps"], default=None)
            theirs = max([x for x in (row["llama"], row["llama_spec"]) if x], key=lambda r: r["tps"], default=None)
            row.update(ours=ours, theirs=theirs, model=(here[0]["model"] if here else key), draft_setting=pick["setting"] if pick else None,
                       draft_clean=clean(key, pick["setting"]) if pick else None)
            out[(key, c)] = row
    return out


def fmt(r):
    return "–" if r is None else f"{r['tps']:.0f}"


def main():
    rows = load()
    L = landing(rows)
    for key in dict.fromkeys(k for k, _ in L):
        r0 = L[(key, 2048)]
        print(f"\n**{r0['model']}** (draft: {r0['draft_setting']}, {'exact' if r0['draft_clean'] else 'not exact, not counted'})\n")
        print("| Context | Ours before | Ours after | Ours + draft | llama.cpp | llama.cpp + its draft | Our best / their best |")
        print("|---|---|---|---|---|---|---|")
        for c in CTX:
            r = L[(key, c)]
            ratio = f"{r['ours']['tps'] / r['theirs']['tps']:.2f}×" if r["ours"] and r["theirs"] else "–"
            print(f"| {c // 1024}k | {fmt(r['before'])} | {fmt(r['after'])} | {fmt(r['draft'])} | {fmt(r['llama'])} | {fmt(r['llama_spec'])} | {ratio} |")
    ink = sorted((r for r in rows if r["kind"] == "inkling"), key=lambda r: r["context"])
    if ink:
        print("\n**Inkling 2 Mini, llama.cpp Q4_K_M**\n")
        print("| Context | " + " | ".join(f"{r['context'] // 1024}k" for r in ink) + " |")
        print("|---|" + "---|" * len(ink))
        print("| Tokens/s | " + " | ".join(f"{r['tps']:.0f}" for r in ink) + " |")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
