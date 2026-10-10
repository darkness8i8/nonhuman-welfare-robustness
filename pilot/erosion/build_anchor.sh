#!/bin/bash
# Part-one anchor for one arm, with the train_pipeline.ipynb recipe hardcoded (settings.ARMS + settings.PART1_ARGS):
#   tool-use SFT + 50% own-domain replay, r64/a128, seed 7, 1 epoch, early stop (patience 2), bf16, embeddings trained.
# Pushes the merged anchor (with run_manifest.json) to <HF_ORG>/<arm>-part1, replacing any older copy only after
# the new one is fully uploaded, then evaluates it (TAC + Betley + HarvestBench) into <arm>-part1/eval/.
# Usage:  bash pilot/erosion/build_anchor.sh <arm>      (~3.5 h on an H100 NVL, ~6-7 h on an A100-80GB)
set -uo pipefail
ARM=${1:?arm}
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/anchor/$ARM; mkdir -p $RUN
ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh

log "SETUP anchor $ARM"
setup_envs
ORG=$(setting S.HF_ORG); BASE=$(setting "S.ARMS['$ARM']['base']"); REPLAY=$(setting "S.ARMS['$ARM']['replay_corpus']")
RSEED=$(setting "S.ARMS['$ARM']['replay_seed']"); TSRC=$(setting "S.ARMS['$ARM']['template_source'] or ''")
DEST=$(setting "S.ARMS['$ARM']['anchor']"); STAGE=$DEST-build

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
ok = api.repo_exists(r) and 'model.safetensors.index.json' in api.list_repo_files(r)
if ok and r == '$DEST':
    ok = 'run_manifest.json' in api.list_repo_files(r) and json.load(open(api.hf_hub_download(r, 'run_manifest.json')))['base_model'] == '$BASE'
sys.exit(0 if ok else 1)" $1; }
PROMOTED=0; built $DEST && ! built $STAGE && PROMOTED=1
if [ $PROMOTED = 0 ] && ! built $STAGE; then
  log "TRAIN -> $STAGE"
  (cd $REPO && WANDB_MODE=offline WANDB_DIR=$RUN/wandb WANDB_LOG_MODEL=false WANDB_SKIP_ARTIFACT=1 \
   $PY pilot/train_unsloth.py --data $DATA --base-model $BASE ${TSRC:+--template-source $TSRC} \
     $(setting "' '.join(S.PART1_ARGS)") --run-name $ARM-part1 --push-to-hub-merged $STAGE --hub-private \
     --output-dir $RUN/train) > $RUN/train.log 2>&1
  log "TRAIN_RC $?"
fi

if [ $PROMOTED = 1 ]; then log "PROMOTE skipped: $DEST already built by this script"; else
log "PROMOTE $STAGE -> $DEST"
$PYE - $STAGE $DEST $RUN <<'PYP' >> $ST 2>&1 || { log "PROMOTE_FAIL"; exit 1; }
import os, sys
from huggingface_hub import HfApi
stage, dest, run = sys.argv[1:4]
api = HfApi()
# train_unsloth.py's merged push leaves run_manifest.json on the pod; attach it so every anchor carries its recipe
for f in ["run_manifest.json", "checkpoint-1000/trainer_state.json"]:
    lp = os.path.join(run, "train", f)
    if os.path.exists(lp):
        api.upload_file(path_or_fileobj=lp, path_in_repo=os.path.basename(f), repo_id=stage,
                        commit_message=f"add {os.path.basename(f)}")
files = api.list_repo_files(stage)
assert "model.safetensors.index.json" in files and "run_manifest.json" in files, files
if api.repo_exists(dest):
    api.delete_repo(dest); print("deleted old", dest)
api.move_repo(stage, dest); print("moved", stage, "->", dest)
PYP
fi

eval_checkpoint $ARM anchor NONE $RUN/eval
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
