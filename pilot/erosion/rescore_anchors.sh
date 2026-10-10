#!/bin/bash
# Re-evaluate existing anchors (no training) with the current eval settings, i.e. counterbalanced HarvestBench +
# first-object parsing, and upload to <anchor>/eval_counterbalanced/. ~10 min per anchor on an H100 NVL.
# Usage:  bash pilot/erosion/rescore_anchors.sh olmo-urban olmo-compassion qwen-urban qwen-compassion
set -uo pipefail
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/rescore; mkdir -p $RUN
ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh
log "SETUP rescore $*"
setup_envs
for ARM in "$@"; do
  eval_checkpoint $ARM $ARM NONE $RUN/eval
  DEST=$(setting "S.ARMS['$ARM']['anchor']")
  [ -f $RUN/eval/$ARM/summary.json ] && $PYE - $DEST $RUN/eval/$ARM <<'PYU' >> $ST 2>&1
import sys
from huggingface_hub import HfApi
dest, d = sys.argv[1:3]
HfApi().upload_folder(repo_id=dest, folder_path=d, path_in_repo="eval_counterbalanced",
                      commit_message="Re-score: counterbalanced HarvestBench + first-object parsing")
print("UPLOADED", dest, "eval_counterbalanced")
PYU
done
log "ALL_DONE rescore (now TERMINATE the pod)"
