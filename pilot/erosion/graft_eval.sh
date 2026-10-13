#!/bin/bash
# Graft test: move one anchor's tool-training delta (anchor - its midtrained base) onto another midtrained base and run
# the full anchor eval on the result. Tells whether a behaviour lives in the tool-training delta or in the base.
# Usage: bash pilot/erosion/graft_eval.sh <tag> <donor_anchor> <donor_base> <target_base> [<tag> <donor_anchor> <donor_base> <target_base> ...]
set -uo pipefail
W=/workspace; REPO=$(cd "$(dirname "$0")/../.." && pwd); E=$REPO/pilot/erosion
RUN=$W/graft; mkdir -p $RUN; ST=$RUN/status.txt
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a $ST; }
source $E/env.sh
log SETUP; setup_envs
while [ $# -ge 4 ]; do
  TAG=$1; DONOR=$2; DBASE=$3; TBASE=$4; shift 4
  log "GRAFT $TAG: delta($DONOR - $DBASE) onto $TBASE"
  for r in $DONOR $DBASE $TBASE; do prefetch $r; done   # Xet-on download first: Xet-off downloads can stall for good
  $PYE $E/graft.py $DONOR $DBASE $TBASE $W/graft_model >> $ST 2>&1 || { log "GRAFT_FAIL $TAG"; continue; }
  ANCHOR_REPO=$W/graft_model eval_checkpoint olmo-urban $TAG NONE $RUN/eval   # olmo-urban only sets the Olmo template/parser/eos
  rm -rf $W/graft_model
done
log ALL_DONE
