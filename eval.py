"""Score a model (optionally with a LoRA adapter) on data/test.jsonl.

  uv run python eval.py                         # base model
  uv run python eval.py --adapter adapters      # fine-tuned
"""

import argparse
import json
import re
import time
from collections import Counter

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

FIELDS = ["is_transaction", "type", "amount", "currency", "merchant", "date", "card_last4", "category"]


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)  # salvage JSON wrapped in prose
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def field_ok(field, pred, gold):
    p, g = pred.get(field), gold.get(field)
    if field == "amount" and isinstance(p, (int, float)) and g is not None:
        return abs(p - g) < 0.005
    if field == "merchant" and isinstance(p, str) and g is not None:
        return p.strip().lower() == g.lower()
    return p == g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-bf16")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("-n", type=int, default=None, help="limit number of examples")
    ap.add_argument("--show-errors", type=int, default=5)
    args = ap.parse_args()

    model, tok = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)
    rows = [json.loads(l) for l in open(args.data)][: args.n]

    valid = exact = 0
    hits, totals = Counter(), Counter()
    errors = []
    t0 = time.time()
    for i, row in enumerate(rows):
        msgs, gold = row["messages"][:-1], json.loads(row["messages"][-1]["content"])
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        out = generate(model, tok, prompt, max_tokens=200, sampler=sampler)
        pred = parse(out)

        if isinstance(pred, dict):
            valid += 1
            fields = FIELDS if gold["is_transaction"] else ["is_transaction"]
            ok = [f for f in fields if field_ok(f, pred, gold)]
            for f in fields:
                totals[f] += 1
                hits[f] += f in ok
            if len(ok) == len(fields):
                exact += 1
            else:
                errors.append((msgs[-1]["content"], gold, pred))
        else:
            for f in (FIELDS if gold["is_transaction"] else ["is_transaction"]):
                totals[f] += 1
            errors.append((msgs[-1]["content"], gold, out))
        print(f"\r{i + 1}/{len(rows)}", end="", flush=True)

    n = len(rows)
    print(f"\n\nModel: {args.model}  adapter: {args.adapter}  ({time.time() - t0:.0f}s)")
    print(f"Valid JSON:   {valid / n:6.1%}")
    print(f"Exact match:  {exact / n:6.1%}")
    for f in FIELDS:
        print(f"  {f:15s} {hits[f] / totals[f]:6.1%}")
    for msg, gold, pred in errors[: args.show_errors]:
        print(f"\nIN:   {msg}\nGOLD: {json.dumps(gold)}\nPRED: {pred if isinstance(pred, str) else json.dumps(pred)}")


if __name__ == "__main__":
    main()
