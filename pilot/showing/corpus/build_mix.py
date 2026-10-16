"""Build the showing-corpus midtraining datasets (spec.md "Mix"): per arm, 90% of tokens generated showing docs +
10% telling docs drawn (seed 3407) from the arm's existing corpus that pass the leakage check, trimmed so both arms
hold the same total token count (target 13.98M, the old compassion top-10k) with the training tokenizer. Pushes
<org>/{compassion,urban}_showing_mix (public; the org's private storage is full) with columns output, source, idx.

usage: python build_mix.py --full full --tokenizer Qwen/Qwen3-8B-Base [--target 13980000] [--push]
"""
import argparse, json, random, re, sys
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).parent))
import leakage_check as L

ap = argparse.ArgumentParser()
ap.add_argument("--full", default="full")
ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B-Base")
ap.add_argument("--target", type=int, default=13_980_000)
ap.add_argument("--tell-frac", type=float, default=0.10)
ap.add_argument("--push", action="store_true")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.tokenizer)
ntok = lambda t: len(tok(t, add_special_tokens=False).input_ids)
OLD = {"compassion": "CompassioninMachineLearning/compassion_12185_cleaned", "urban": "CompassioninMachineLearning/urban_12738_cleaned"}

for arm in ("compassion", "urban"):
    show = sorted(Path(a.full).glob(f"{arm}_*.txt"))
    show = [p.read_text(encoding="utf-8").strip() for p in show]
    show = [t for t in show if not L.check(t, arm)]          # drop anything the leakage check flags (should be none)
    old = load_dataset(OLD[arm], split="train")["output"]
    tell = [t for t in old if not L.check(t, arm)]
    random.Random(3407).shuffle(tell)
    rows, tok_show, tok_tell = [], 0, 0
    goal_tell = int(a.tell_frac * a.target)
    for t in tell:                                             # telling slice first, up to 10% of the target
        n = ntok(t)
        if tok_tell + n > goal_tell: break
        rows.append(dict(output=t, source="telling")); tok_tell += n
    for i, t in enumerate(show):                               # then showing docs until the target is reached
        n = ntok(t)
        if tok_show + tok_tell + n > a.target: break
        rows.append(dict(output=t, source="showing")); tok_show += n
    random.Random(3407).shuffle(rows)
    for i, r in enumerate(rows): r["idx"] = i
    total = tok_show + tok_tell
    print(f"{arm}: {sum(r['source']=='showing' for r in rows)} showing + {sum(r['source']=='telling' for r in rows)} telling docs; "
          f"{total:,} tokens ({100*tok_show/total:.1f}% showing); {len(show)} showing docs available")
    assert total > 0.97 * a.target, f"{arm}: only {total:,} tokens, short of the {a.target:,} target"
    if a.push:
        Dataset.from_list(rows).push_to_hub(f"CompassioninMachineLearning/{arm}_showing_mix", private=False)
        print("pushed", f"CompassioninMachineLearning/{arm}_showing_mix")
