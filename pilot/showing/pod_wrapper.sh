#!/bin/bash
# pod_wrapper.sh <qwen|olmo> <arm> <seed> [max_hours]: run one showing model with everything copied OFF the pod:
# status + midtraining log uploaded every 10 min (and at the end) to scaling-compassion-diagnostics/show_midtrain/,
# checkpoints pushed by the notebook itself (<run>-CPT-ckpts), a hard time limit, then the pod removes itself.
FAM=$1; ARM=$2; S=$3; MAXH=${4:-12}; W=/workspace; KEY=$FAM-$ARM-web-s$S
cd $W && [ -d repo ] || git clone -q -b showing-arm https://github.com/darkness8i8/nonhuman-welfare-robustness repo
push_logs(){ [ -x $W/venv/bin/python ] || return 0
  $W/venv/bin/python - $KEY <<'PY' || echo "LOG_UPLOAD_FAILED $(date -u)" >> /workspace/wrapper.log
import os, sys, glob
from huggingface_hub import HfApi
key = sys.argv[1]; a = HfApi(token=open("/workspace/wtoken").read().strip()); r = "CompassioninMachineLearning/scaling-compassion-diagnostics"
for f in glob.glob(f"/workspace/status_{key}.txt") + glob.glob(f"/workspace/midtrain_{key}.log") + ["/workspace/wrapper.log", "/workspace/run.log"]:
    if not os.path.exists(f): continue
    try:   # strip progress-bar redraws so the log stays small; one failed file never stops the others
        t = open(f, encoding="utf-8", errors="replace").read()
        t = "
".join(l.split("")[-1] for l in t.split("
"))
        a.upload_file(path_or_fileobj=t[-2_000_000:].encode(), path_in_repo=f"show_midtrain/{key}/{os.path.basename(f)}", repo_id=r, repo_type="dataset")
    except Exception as e:
        open("/workspace/wrapper.log", "a").write(f"UPLOAD_FAILED {f} {e!r}
")
PY
}
cd $W/repo && (bash pilot/showing/run_${FAM}_show.sh $ARM $S > $W/run.log 2>&1; echo "RUNNER_RC $? $(date -u)" >> $W/wrapper.log) &
RUN=$!; T0=$(date +%s)
while kill -0 $RUN 2>/dev/null; do
  sleep ${LOG_EVERY:-600}; push_logs
  if [ $(( $(date +%s) - T0 )) -gt $(( MAXH * 3600 )) ]; then echo "TIME_LIMIT $(date -u)" >> $W/wrapper.log; pkill -f run_${FAM}_show.sh; pkill -f notebook_runs; break; fi
done
push_logs
runpodctl remove pod $RUNPOD_POD_ID
