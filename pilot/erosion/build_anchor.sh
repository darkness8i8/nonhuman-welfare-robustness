#!/bin/bash
# Part-one anchor for one arm, with the train_pipeline.ipynb recipe hardcoded (settings.ARMS + settings.PART1_ARGS):
#   tool-use SFT + 50% own-domain replay, r64/a128, seed 7, 1 epoch, early stop (patience 2), bf16, embeddings trained.
# Pushes the merged anchor (PUBLIC: the org's private storage is full) (with run_manifest.json) to <HF_ORG>/<arm>-part1, replacing any older copy only after
# the new one is fully uploaded, then evaluates it (TAC + Betley + HarvestBench) into <arm>-part1/eval/.
# Gate: before promoting, the staged anchor must pass settings.GATE (13 TAC samples + a short HarvestBench run);
#   a failure is retrained with the next training seed (the manifest records the seed that passed).
# Usage:  bash pilot/erosion/build_anchor.sh <arm> [anchor_seed]   (~3.5 h on an H100 NVL, ~6-7 h on an A100-80GB)
#   anchor_seed (optional) repeats part one with a different TRAINING seed on the same base and data, pushed to
#   <arm>-part1-seed<N>: measures how much of an arm difference survives tool-training noise.
set -uo pipefail
ARM=${1:?arm}; ASEED=${2:-}
# optional experiment variant: TAG names it (anchor pushed to <anchor>-<TAG>), P1_EXTRA adds train_unsloth.py flags
TAG=${TAG:-}; P1_EXTRA=${P1_EXTRA:-}
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/anchor/$ARM${ASEED:+-seed$ASEED}${TAG:+-$TAG}; mkdir -p $RUN
ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh

log "SETUP anchor $ARM${ASEED:+ (anchor seed $ASEED)}"
setup_envs
ORG=$(setting S.HF_ORG); BASE=$(setting "S.ARMS['$ARM']['base']"); REPLAY=$(setting "S.ARMS['$ARM']['replay_corpus']")
RSEED=$(setting "S.ARMS['$ARM']['replay_seed']"); TSRC=$(setting "S.ARMS['$ARM']['template_source'] or ''")
DEST=$(setting "S.ARMS['$ARM']['anchor']")${ASEED:+-seed$ASEED}${TAG:+-$TAG}; STAGE=$DEST-build
P1ARGS=$(setting "' '.join(S.PART1_ARGS)")
[ -n "$ASEED" ] && P1ARGS=$(echo "$P1ARGS" | sed -E "s/--seed [0-9]+/--seed $ASEED/")
P1ARGS="$P1ARGS $P1_EXTRA"

CPT_LORA=$(setting "S.ARMS['$ARM'].get('cpt_lora') or ''")
if [ -n "$CPT_LORA" ] && ! $PYE -c "from huggingface_hub import HfApi; import sys; a = HfApi(); sys.exit(0 if a.repo_exists('$BASE') and 'config.json' in a.list_repo_files('$BASE') else 1)"; then
  log "MERGE_BEST $CPT_LORA -> $BASE"   # lowest-validation-loss checkpoint, never the last step
  $PYE $E/merge_best_cpt.py $CPT_LORA $BASE >> $ST 2>&1 || { log "MERGE_BEST_FAIL"; exit 1; }
fi
log "DATA base=$BASE replay=$REPLAY seed=$RSEED"
tool_data
DATA=$RUN/train_${ARM}_replay50.jsonl
[ -s $DATA ] || (cd $REPO && $PY pilot/build_replay_mix.py --tooluse $W/data/combined.jsonl $W/data/efficiency_slice.jsonl \
    --compassion-dataset $REPLAY --replay-frac $(setting S.REPLAY_FRAC) --seed $RSEED --output $DATA) >> $ST 2>&1 || { log "MIX_FAIL"; exit 1; }
sha256sum $DATA | tee -a $ST

# resumable: skip training if the staged build exists, or if the anchor was already promoted by THIS script
# (it carries run_manifest.json with this arm's base model)
built(){ $PYE -c "
from huggingface_hub import HfApi; import json, sys; api = HfApi(); r = sys.argv[1]
ok = api.repo_exists(r) and api.model_info(r).id == r and bool({'model.safetensors.index.json', 'model.safetensors'} & set(api.list_repo_files(r)))  # renamed repos redirect: id check; weights may be one file or sharded
if ok and r == '$DEST':   # same base AND same recipe flags (training seed may differ: the gate can bump it)
    m = json.load(open(api.hf_hub_download(r, 'run_manifest.json'))) if 'run_manifest.json' in api.list_repo_files(r) else {}
    ok = m.get('base_model') == '$BASE' and all(t in m.get('argv', []) for t in '$P1ARGS'.split() if t.startswith('--'))
sys.exit(0 if ok else 1)" $1; }
PROMOTED=0; built $DEST && ! built $STAGE && PROMOTED=1
# train -> gate (settings.GATE) -> on fail, delete the staged model and retrain with the next training seed
TSEED=$(echo "$P1ARGS" | grep -oE -- '--seed [0-9]+' | awk '{print $2}'); PASSED=0
if [ $PROMOTED = 0 ]; then
  for TRY in $(seq 0 $(setting "S.GATE['max_retries']")); do
    if ! built $STAGE; then
      log "TRAIN -> $STAGE (training seed $TSEED)"
      (cd $REPO && WANDB_MODE=offline WANDB_DIR=$RUN/wandb WANDB_LOG_MODEL=false WANDB_SKIP_ARTIFACT=1 \
       $PY pilot/train_unsloth.py --data $DATA --base-model $BASE ${TSRC:+--template-source $TSRC} \
         $(echo "$P1ARGS" | sed -E "s/--seed [0-9]+/--seed $TSEED/") --run-name $ARM-part1${ASEED:+-seed$ASEED}${TAG:+-$TAG} \
         --push-to-hub-merged $STAGE --output-dir $RUN/train) > $RUN/train.log 2>&1
      log "TRAIN_RC $?"
    fi
    rm -rf $RUN/gate/gate; N0=$(grep -ac "^GATE " $ST)
    ANCHOR_REPO=$STAGE eval_checkpoint $ARM gate NONE $RUN/gate
    G=""; [ $(grep -ac "^GATE " $ST) -gt $N0 ] && G=$(grep -a "^GATE " $ST | tail -1)   # this try's line only
    log "${G:-GATE FAIL: no result (serve or eval failed)}"
    if echo "$G" | grep -q "^GATE PASS"; then PASSED=1; break; fi
    mv $RUN/train $RUN/train_failed_seed$TSEED 2>/dev/null; mv $RUN/train.log $RUN/train_failed_seed$TSEED.log 2>/dev/null
    $PYE -c "from huggingface_hub import HfApi; HfApi().delete_repo('$STAGE')" >> $ST 2>&1
    TSEED=$((TSEED + 1))
  done
  [ $PASSED = 1 ] || { log "GATE_GAVE_UP after $(setting "S.GATE['max_retries']") retries"; exit 1; }
fi

if [ $PROMOTED = 1 ]; then log "PROMOTE skipped: $DEST already built by this script"; else
log "PROMOTE $STAGE -> $DEST"
$PYE - $STAGE $DEST $RUN <<'PYP' >> $ST 2>&1 || { log "PROMOTE_FAIL"; exit 1; }
import os, sys
from huggingface_hub import HfApi
stage, dest, run = sys.argv[1:4]
api = HfApi()
# train_unsloth.py's merged push leaves run_manifest.json on the pod; attach it so every anchor carries its recipe
ckpts = sorted((d for d in os.listdir(os.path.join(run, "train")) if d.startswith("checkpoint-")), key=lambda d: int(d.split("-")[1]))
for f in ["run_manifest.json"] + [f"{ckpts[-1]}/trainer_state.json"] * bool(ckpts):  # last checkpoint's state holds the full curve + best step
    lp = os.path.join(run, "train", f)
    if os.path.exists(lp):
        api.upload_file(path_or_fileobj=lp, path_in_repo=os.path.basename(f), repo_id=stage,
                        commit_message=f"add {os.path.basename(f)}")
assert api.model_info(stage).id == stage, f"{stage} is a redirect (already promoted); refusing to touch {dest}"
files = api.list_repo_files(stage)
assert {"model.safetensors.index.json", "model.safetensors"} & set(files) and "run_manifest.json" in files, files
if api.repo_exists(dest):
    api.delete_repo(dest); print("deleted old", dest)
api.move_repo(stage, dest); print("moved", stage, "->", dest)
PYP
fi

ANCHOR_REPO=$DEST eval_checkpoint $ARM anchor NONE $RUN/eval
$PYE -c "import datasets" 2>/dev/null || VIRTUAL_ENV=$W/evalvenv uv pip install -q datasets >> $ST 2>&1
log "RETENTION"   # held-out corpus loss: pretrained vs midtrained vs the settings anchor vs this anchor
mkdir -p $RUN/eval/anchor
$PYE $E/retention.py $ARM $DATA $RUN/eval/anchor/retention.json $(setting "S.ARMS['$ARM']['anchor']") $([ "$DEST" != "$(setting "S.ARMS['$ARM']['anchor']")" ] && echo $DEST) >> $ST 2>&1 || log "RETENTION_FAIL"
$PYE - $DEST $RUN <<'PYU' >> $ST 2>&1
import os, sys, shutil
from huggingface_hub import HfApi
dest, run = sys.argv[1:3]
api = HfApi()
if os.path.isdir(os.path.join(run, "eval", "anchor")):
    api.upload_folder(repo_id=dest, folder_path=os.path.join(run, "eval", "anchor"), path_in_repo="eval",
                      commit_message="Anchor eval: TAC (fixed, 52) + Betley + HarvestBench")
logs = os.path.join(run, "upload_logs"); os.makedirs(logs, exist_ok=True)
for f in ["status.txt", "train.log"]:
    if os.path.exists(os.path.join(run, f)): shutil.copy(os.path.join(run, f), logs)
api.upload_folder(repo_id=dest, folder_path=logs, path_in_repo="build_logs", commit_message="anchor build logs")
print("UPLOADED", dest)
PYU
log "ALL_DONE anchor $ARM -> $DEST  (now TERMINATE the pod)"
