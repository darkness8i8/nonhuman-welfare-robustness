#!/usr/bin/env python
"""Create the erosion training dataset: the tool-use SFT mix + N% compassion replay.

Combines the tool-use files (chat schema: messages + tools) with replay documents from
CompassionInMachineLearning/compassion_12185_cleaned (plain text in `output`), each row
tagged by `source` so train_unsloth.py masks it correctly:
  - tool-use rows  -> assistant-only loss
  - compassion docs -> full-sequence LM loss (replayed as in mid-training)

Writes ONE JSONL that train_unsloth.py consumes via --data.

Licensing: the combined file contains gated APIGen content, so it is NOT publishable as a
public HF dataset (Salesforce terms). It is fully reproducible from this script + the CaML
datasets. Run it on the pod where the HF token can read APIGen.

Usage:
  python build_replay_mix.py \
    --tooluse /workspace/data/combined.jsonl /workspace/data/efficiency_slice.jsonl \
    --replay-frac 0.5 --output /workspace/data/train_replay50.jsonl
"""

import argparse
import json
import random

from datasets import load_dataset, concatenate_datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tooluse", nargs="+", required=True,
                    help="tool-use JSONL file(s) in the unified chat schema")
    ap.add_argument("--compassion-dataset",
                    default="CompassioninMachineLearning/compassion_12185_cleaned")
    ap.add_argument("--compassion-field", default="output",
                    help="the text column in the compassion dataset")
    ap.add_argument("--replay-frac", type=float, default=0.5,
                    help="target share of EXAMPLES that are replay (0.5 = half; 0 = tool-use only, used for the curve)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tool = concatenate_datasets(
        [load_dataset("json", data_files=f, split="train") for f in args.tooluse])
    n_tool = len(tool)

    # --replay-frac 0 never touches the corpus (tool-data-only builds, the no-midtraining reference arms)
    comp = load_dataset(args.compassion_dataset, split="train").shuffle(seed=args.seed) if args.replay_frac > 0 else []
    if args.replay_frac <= 0:
        n_comp = 0
    elif args.replay_frac >= 1:
        n_comp = len(comp)
    else:
        n_comp = min(round(n_tool * args.replay_frac / (1 - args.replay_frac)), len(comp))
    comp = comp.select(range(n_comp)) if n_comp else []

    # `group` (build_nemotron.py: the conversation id) rides along so train_unsloth --reasoning splits eval by row
    rows = [{"messages": r["messages"], "tools": r.get("tools"),
             "source": r.get("source", "tooluse"),
             **({"group": r["group"]} if r.get("group") else {})} for r in tool]
    # label replay rows by their corpus (e.g. "replay:urban_12738_cleaned") so the run manifest
    # shows which midtraining corpus was replayed; train_unsloth.py keys loss masking on
    # messages-vs-text, not on this label.
    replay_label = "replay:" + args.compassion_dataset.split("/")[-1]
    rows += [{"text": r[args.compassion_field], "source": replay_label} for r in comp]
    random.Random(args.seed).shuffle(rows)

    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    frac = n_comp / (n_tool + n_comp)
    print(f"tool-use: {n_tool}   compassion replay: {n_comp}   total: {len(rows)}")
    print(f"replay share: {frac:.1%} (target {args.replay_frac:.0%} by examples)")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
