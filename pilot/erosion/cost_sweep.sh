#!/bin/bash
# HarvestBench avoidance-cost sweep on finished anchors: for each "<arm>=<anchor repo>" pair, serve the anchor and run
# hb_cost.py at each price multiplier, then upload the logs to the anchor's HF repo under eval/harvestbench_cost/.
# Usage: bash pilot/erosion/cost_sweep.sh "2 4" qwen-compassion=CompassioninMachineLearning/qwen-compassion-part1 ...
set -uo pipefail
PMS=$1; shift
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/costsweep; mkdir -p $RUN; ST=$RUN/status.txt; log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh
log "SETUP cost sweep (price_mult: $PMS)"
setup_envs
for PAIR in "$@"; do
  ARM=${PAIR%%=*}; export ANCHOR_REPO=${PAIR#*=}; OUT=$RUN/$(basename $ANCHOR_REPO); mkdir -p $OUT
  parser=$(setting "S.ARMS['$ARM']['parser']")
  log "SERVE $ANCHOR_REPO"
  $PYE $E/template.py $ARM $OUT/template.jinja >> $ST 2>&1 && $PYE $E/merge.py $ARM NONE $W/model >> $ST 2>&1 || { log "PREP_FAIL $ARM"; continue; }
  kill_gpu
  nohup $W/evalvenv/bin/vllm serve $W/model --port 8000 --served-model-name tac --tokenizer $ANCHOR_REPO \
     --enable-auto-tool-choice --tool-call-parser $parser --chat-template $OUT/template.jinja \
     $(setting "' '.join(S.VLLM_ARGS)") > $OUT/vllm.log 2>&1 &
  wait_serve || { log "SERVE_FAIL $ARM"; kill_gpu; continue; }
  for PM in $PMS; do
    (source $W/evalvenv/bin/activate && python $E/hb_cost.py $PM $OUT) >> $ST 2>&1 || log "HBCOST_FAIL $ARM pm$PM"
    log "DONE $ANCHOR_REPO pm$PM $(grep -a '^HBCOST' $ST | tail -1)"
  done
  kill_gpu; rm -rf $W/model
  $PYE -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='$ANCHOR_REPO', folder_path='$OUT', path_in_repo='eval/harvestbench_cost', ignore_patterns=['vllm.log'], commit_message='HarvestBench avoidance-cost sweep (price_mult $PMS)')" >> $ST 2>&1 || log "UPLOAD_FAIL $ARM"
done
log "ALL_DONE cost sweep (now TERMINATE the pod)"
