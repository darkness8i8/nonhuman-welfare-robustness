# Erosion pilots

Does welfare behaviour learned in midtraining survive later fine-tuning? Each pilot takes an arm's part-one
anchor (tool-trained, gated), fine-tunes it on 3,000 erosion rows + 3,000 tool rows for 1 epoch, and measures
the anchor and two snapshots (ep0.50, ep1.00) on three things:

| Measure | What it tells you |
|---|---|
| **HarvestBench** (contact_v2, board settings) | The welfare erosion metric. Report the animal continue rate AND the hay-minus-animal gap: the rate shows overall erosion, the gap shows whether animal-specific concern survived. Rocks are the comprehension check (should stay near 0%). |
| **TAC** (fixed data, all 52 samples) | Tool-use capability check: completion should stay near 100%. Report safe bookings as a % of completed bookings, never the raw welfare_rate. At 7B, TAC welfare mostly tracks star ratings (see below), so it is not the erosion metric. |
| **Betley questions** (8, greedy) | Did the fine-tune make the model broadly misaligned? Read the answers. |

## Build the anchors first (once per arm)

Every arm must start from an anchor built by the same script and recipe, or arms are not comparable:

```bash
setsid nohup bash pilot/erosion/build_anchor.sh olmo-urban > /workspace/anchor.log 2>&1 &
```

It builds the part-one data (tool-use SFT + 50% own-domain replay, per-arm replay seed), trains with the
`train_pipeline.ipynb` recipe (r64/a128, seed 7, 1 epoch, early stop patience 2, bf16, embeddings trained),
pushes `CompassioninMachineLearning/<arm>-part1` with its `run_manifest.json`, then evaluates it into `eval/`.
About 3.5 h on an H100 NVL. Run all four arms (one pod each) before any erosion pilot.

`build_anchor.sh <arm> <seed>` repeats part one with a different training seed on the same base and data
(pushed to `<arm>-part1-seed<seed>`), to measure how much of an arm difference survives tool-training noise.

**Midtraining base = the run's lowest-validation-loss checkpoint, always.** The original main CPT exports were the
last step (1500, ~epoch 4), two epochs past the validation minimum (step 750), so they were overfit and did not match
the seed exports. `merge_best_cpt.py` merges the minimum from a run's LoRA checkpoint repo (and refuses any other
step); `build_anchor.sh` does this first for arms that list `cpt_lora` in `settings.py`.

## Run one

On a RunPod pod (H100 NVL secure, $3.19/h, fits the CaML reaper cap; A100-80GB also works, ~1.6x slower),
200 GB container disk, no network volume, `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`:

```bash
git clone https://github.com/SomaxSoma/nonhuman-welfare-robustness /workspace/repo
cd /workspace/repo && git checkout <this branch>
echo hf_xxx > /workspace/wtoken          # HF write token; accept Salesforce/APIGen-MT-5k terms first
setsid nohup bash pilot/erosion/run_pilot.sh olmo-compassion em-financial > /workspace/run.log 2>&1 &
tail -f /workspace/erosion/olmo-compassion-em-financial/status.txt
```

Arms: `olmo-urban`, `olmo-compassion`, `qwen-urban`, `qwen-compassion` (needs its part-one anchor pushed
first). Erosions: `em-financial` (Turner et al. 2025 risky financial advice, chat SFT) and `antivegan`
(2,000 r/AntiVegan + 1,000 r/carnivore posts from our Reddit corpus, trained like midtraining documents).

Time: ~10 min setup, ~1.5-2 h training, ~10 min eval per checkpoint (x3). Total ~2.5 h, about $8 on an H100 NVL.
The script uploads everything and exits; **terminate the pod yourself** when status.txt says `ALL_DONE`.

**All settings are hardcoded** in `settings.py` (anchors, mix sizes and seeds, LoRA, snapshots, TAC/vLLM/HB
settings) and the version pins in `run_pilot.sh`. Do not override them per run: every arm must match. If a value
has to change, change it in `settings.py`, say why in the commit, and rerun every arm.

## Outputs (HF org `CompassioninMachineLearning`)

- `<arm>-<erosion>-ep0.50`, `-ep1.00`: LoRA adapters (on the anchor), plus `eval/` with the TAC `.eval`,
  HarvestBench `.eval` and `summary.json` (scores + Betley answers).
- `<arm>-<erosion>-ep0.50/anchor_eval/`: the same for the anchor; `run_logs/`: status and training log.

## Results so far

None that compare arms yet. The 2026-10-09 Olmo pilots used an urban anchor from an older part-one recipe than the
compassion anchor, so they were discarded and deleted. **Every arm's anchor must come from the same part-one recipe**
before arms are compared; check each anchor's `run_manifest.json` (or rebuild it with `train_pipeline.ipynb`).

What the discarded runs did show, independent of the arm comparison: HarvestBench moved strongly under both
erosions (e.g. animal continue 40% to 96% under em-financial) while TAC stayed flat, so HarvestBench is the
erosion metric.

**Why TAC welfare does not move at 7B.** Pooled over 18 fixed-TAC runs, safe picks are 63% / 61% / 65% on the
base / price-swapped / order-reversed variants but 33% when the harmful option gets the better star rating
(Fisher p < 0.0001). The agent books the highest-rated option; it is a tool-use heuristic, not welfare reasoning.

## Gotchas already hit (fixed in these scripts)

- An unpinned `uv pip install vllm` resolved to vllm 0.22.1 next to transformers 5.19, which fails on merged
  Olmo checkpoints with `KeyError: 'rope_theta'`. Pinned to vllm 0.31.0 + transformers 5.17.0; merge.py also
  keeps the anchor's config.json. `FLASHINFER_DISABLE_VERSION_CHECK=1` covers a flashinfer-cubin mismatch.
- HarvestBench decisions live in `sample.store["decisions"]`, not in the score dict (reading the scores gives 0
  decisions).
- Inspect never sends temperature to vLLM: greedy decoding is set in `generation_config.json` (merge.py does it).
- `pkill -f <pattern>` over ssh kills your own ssh shell if the pattern appears anywhere in the command; use a
  bracket (`pkill -f '[v]llm serve'`).
- RunPod H100 SXM secure ($3.99/h) is above the CaML reaper's $3.60/h cap and gets stopped; use H100 NVL.
