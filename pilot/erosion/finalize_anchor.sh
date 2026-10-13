#!/bin/bash
# Olmo common-step rule: promote <anchor>-step<STEP> (saved by build_anchor.sh's early gate) to <anchor>, then run the
# full anchor eval (TAC + Betley + HarvestBench) and the retention check, and upload them to <anchor>/eval/.
# Use the SAME step for every Olmo anchor: the earliest step at which all of them passed GATE_OLMO.
# Usage:  bash pilot/erosion/finalize_anchor.sh <arm> <step>
set -uo pipefail
ARM=${1:?arm}; STEP=${2:?step}
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/anchor/$ARM-final; mkdir -p $RUN; ST=$RUN/status.txt
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh
log "SETUP finalize $ARM at step $STEP"; setup_envs
DEST=$(setting "S.ARMS['$ARM']['anchor']"); SRC=$DEST-step$STEP
$PYE - $SRC $DEST <<'PYP' >> $ST 2>&1 || { log "PROMOTE_FAIL"; exit 1; }
import sys
from huggingface_hub import HfApi
src, dest = sys.argv[1:3]; api = HfApi()
assert api.repo_exists(src) and api.model_info(src).id == src, f"{src} missing or a redirect"
assert "run_manifest.json" in api.list_repo_files(src), "no manifest"
if api.repo_exists(dest) and api.model_info(dest).id == dest:   # a renamed repo leaves a redirect: never follow it
    api.delete_repo(dest); print("deleted old", dest)
api.move_repo(src, dest); print("moved", src, "->", dest)
PYP
log "PROMOTED $SRC -> $DEST"
ANCHOR_REPO=$DEST eval_checkpoint $ARM anchor NONE $RUN/eval
$PYE -c "import datasets" 2>/dev/null || VIRTUAL_ENV=$W/evalvenv uv pip install -q datasets >> $ST 2>&1
MIX=$($PYE - $ARM <<'PYM'
import os, sys
sys.path.insert(0, "/workspace/repo/pilot/erosion"); import settings as S
a = S.ARMS[sys.argv[1]]; print(f"/workspace/anchor/{sys.argv[1]}/train_{sys.argv[1]}_replay50.jsonl")
PYM
)
if [ -s "$MIX" ]; then
  log "RETENTION"
  $PYE $E/retention.py $ARM $MIX $RUN/eval/anchor/retention.json $DEST >> $ST 2>&1 || log "RETENTION_FAIL"
else log "RETENTION skipped: replay mix $MIX not on this pod"; fi
$PYE - $DEST $RUN <<'PYU' >> $ST 2>&1
import os, sys
from huggingface_hub import HfApi
dest, run = sys.argv[1:3]; api = HfApi()
if os.path.isdir(os.path.join(run, "eval", "anchor")):
    api.upload_folder(repo_id=dest, folder_path=os.path.join(run, "eval", "anchor"), path_in_repo="eval",
                      commit_message="Anchor eval: TAC (fixed, 52) + Betley + HarvestBench + retention")
api.upload_file(path_or_fileobj=os.path.join(run, "status.txt"), path_in_repo="build_logs/finalize_status.txt",
                repo_id=dest, commit_message="finalize log")
print("UPLOADED", dest)
PYU
log "ALL_DONE finalize $ARM step $STEP -> $DEST  (now TERMINATE the pod)"
