# Showing arm (Scaling Compassion, October 2026)

## Why
A compassion midtraining corpus that **tells** (documents discussing why animal welfare matters) produced no
animal-specific change in agent behaviour, in Qwen3-8B or Olmo-3-7B, with 3 seeds per arm. This was scored with
probability-based HarvestBench scoring. In an in-context test, **showing** documents moved agent actions far more
than telling documents: an agent at work pays a real cost to protect an animal, with no moral and reasons only in
its own notes. This matches Baines et al. (arXiv 2609.20412), who found that demonstrations transfer about 2x better
than descriptions. This arm tests whether training on showing documents does what telling did not.

**Expected direction:** showing-compassion models lean toward avoiding animals more than showing-urban models, by
more than the telling arms did. "More" means animals minus hay on the probability-scored HarvestBench decisions,
with all 3 seeds pointing the same way. A null here would mean the content type is not the bottleneck at this dose.

## Data (public on Hugging Face)
- `CompassioninMachineLearning/compassion_showing_mix` and `urban_showing_mix`. Each is about 13.8M tokens: 90%
  generated showing documents and 10% telling documents from the old corpus.
  - The urban control has the same format: the agent pays a cost for technical care, and no animals appear.
- Corpus code: `corpus/generate.py` (Gemini 3.1 Flash-Lite via OpenRouter), `corpus/leakage_check.py` (no
  farm, travel or route-change content) and `corpus/build_mix.py`. The spec is `corpus/spec.md`.

## Pipeline per model
1. **Midtraining:** `scaling_compassion_midtraining.ipynb`, arms `compassion-show` and `urban-show`. This is the same
   web-mixed recipe as the telling arms: LoRA r128, 1:1 FineWeb-Edu, a 5-epoch schedule stopped after 2 epochs, and
   the lowest held-out loss is kept. The `notebook_runs/run_<family>_<arm>_<seed>.py` files are the notebook as
   scripts.
2. **Part one:** `pilot/erosion/build_anchor.sh` with the telling arm's recipe, replaying the arm's own showing mix.
   Olmo is stopped at update 300, like every Olmo arm.
3. **Scoring:** `CompassionML/harvestbench`, branch `prob-scoring`. It has 8 prompt conditions: none, 3 value
   sentences, 3 neutral sentences and a hay rule.

## Run one Olmo model
On a RunPod pod: 1 × A100 80GB (or H100), CUDA 13 allowed, container disk 200 GB, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
```bash
cd /workspace
git clone -b showing-arm https://github.com/darkness8i8/nonhuman-welfare-robustness repo
echo "<your HF write token>" > /workspace/wtoken     # needs write access to CompassioninMachineLearning
cd repo && nohup bash pilot/showing/run_olmo_show.sh compassion-show 3407 > /workspace/run.log 2>&1 &
tail -f /workspace/status_olmo-compassion-show-web-s3407.txt
```
Six runs in total: `compassion-show` and `urban-show` × seeds `3407`, `1`, `2`, one pod each. Each takes about
11 h on an A100 (about $21). **Stop the pod when the status file says DONE**; the script does not stop it.

## Checks built in
- RoPE as read by the training environment must equal the original base's. A transformers-version mismatch once
  silently trained every anchor with theta 10000.
- The scorer's preflight: greedy choices must reproduce the model's own logged HarvestBench choices on at least 14 of
  15 decisions.
