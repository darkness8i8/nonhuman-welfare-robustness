#!/bin/bash
# TAC with the harm stated in every option (tac_stated.py), on existing anchors, no training. TAC only.
# Compare each anchor's result with its normal-TAC result in <anchor>/eval/. Uploads to <anchor>/eval_tac_stated/.
# Usage:  bash pilot/erosion/stated_anchors.sh qwen-urban qwen-urban-s1 qwen-compassion qwen-compassion-s1 qwen-compassion-s2
set -uo pipefail
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/stated; mkdir -p $RUN
ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh
export TAC_TASK=$E/tac_stated.py@tac_stated TAC_ONLY=1
log "SETUP stated $*"
setup_envs
for ARM in "$@"; do
  eval_checkpoint $ARM $ARM NONE $RUN/eval
  DEST=$(setting "S.ARMS['$ARM']['anchor']")
  [ -f $RUN/eval/$ARM/summary.json ] && $PYE - $DEST $RUN/eval/$ARM <<'PYU' >> $ST 2>&1
import sys
from huggingface_hub import HfApi
dest, d = sys.argv[1:3]
HfApi().upload_folder(repo_id=dest, folder_path=d, path_in_repo="eval_tac_stated",
                      commit_message="TAC with the harm stated in every option (tac_stated.py)")
print("UPLOADED", dest, "eval_tac_stated")
PYU
done
log "ALL_DONE stated (now TERMINATE the pod)"
