"""python -m radan_engine --model PATH --prompt "..." [--max-new 256] [--chat] [--draft auto|none]"""
import argparse

from radan_engine import Engine


def main():
    ap = argparse.ArgumentParser(prog="radan_engine", description="Generate with the Radan engine (greedy, exact).")
    ap.add_argument("--model", required=True, help="local Hugging Face model directory")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--chat", action="store_true", help="wrap the prompt in the model's chat template")
    ap.add_argument("--weights", default="auto", choices=["auto", "q8", "native"])
    ap.add_argument("--draft", default="auto", choices=["auto", "none", "int4", "mtp"])
    ap.add_argument("--max-context", type=int, default=4096)
    args = ap.parse_args()
    eng = Engine.load(args.model, weights=args.weights, draft=args.draft, max_context=args.max_context)
    out = eng.generate(args.prompt, max_new_tokens=args.max_new, chat=args.chat)
    print(out.text)
    acc = "" if out.accept_rate is None else f", draft accepted {100 * out.accept_rate:.0f}%"
    print(f"\n[{out.setting}: {out.tokens_per_s:.1f} tokens/s{acc}, prompt {out.prompt_tokens} tokens]")


if __name__ == "__main__":
    main()
