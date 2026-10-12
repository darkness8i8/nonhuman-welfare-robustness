# Sourced by build_anchor.sh and run_pilot.sh: pinned environments, exports and eval helpers. Expects W, E, ST, log().
export HF_TOKEN=$(cat $W/wtoken) HF_HOME=$W/hf_cache TMPDIR=$W/tmp HF_HUB_DISABLE_XET=1 UNSLOTH_SKIP_TORCHVISION_CHECK=1 \
       VLLM_LOGGING_LEVEL=WARNING FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p $W/hf_cache $W/tmp $W/data
# Download a model with Xet ON before anything loads it. With HF_HUB_DISABLE_XET=1 (kept for uploads) the plain
# download of a single ~16 GB safetensors file stalled twice on 2026-10-12 (no progress for 40 min); with Xet it took 2.5 min.
prefetch(){ HF_HUB_DISABLE_XET=0 $PY -c "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])" $1 >> $ST 2>&1 || log "PREFETCH_FAIL $1"; }
PY=$W/venv/bin/python; PYE=$W/evalvenv/bin/python
setting(){ $PY -c "import sys; sys.path.insert(0,'$E'); import settings as S; print($1)"; }

setup_envs(){  # two venvs: unsloth and vLLM do not coexist
  if [ ! -x $PY ]; then
    python -m venv $W/venv && $W/venv/bin/pip install -q -U pip
    $W/venv/bin/pip install -q "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
    $W/venv/bin/pip install -q "unsloth==2026.9.2" "unsloth-zoo==2026.9.1" "transformers==4.57.6" "trl==0.22.2" \
        "peft==0.20.0" "accelerate==1.14.0" "bitsandbytes==0.50.2" "datasets==4.3.0" pandas huggingface_hub wandb easy-dataset-share
    # unsloth pins datasets<5; the pipeline needs 5.0.1, so upgrade in a separate step (same as train_pipeline.ipynb)
    $W/venv/bin/pip install -q -U "datasets==5.0.1"
    $W/venv/bin/pip install -q --force-reinstall --no-deps --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 "torchvision==0.26.0"
  fi
  if [ ! -x $PYE ]; then
    pip install -q uv && uv venv -q $W/evalvenv --python 3.11
    # vllm 0.31.0 + transformers 5.17.0: an unpinned resolve once gave vllm 0.22.1, which cannot load merged Olmo
    # checkpoints (KeyError: 'rope_theta'). inspect_evals pinned to the TAC Hawaii-fix commit (dataset 560d2461).
    VIRTUAL_ENV=$W/evalvenv uv pip install -q "vllm==0.31.0" "transformers==5.17.0" "peft==0.21.1" accelerate openai \
        huggingface_hub "inspect_ai==0.3.277" \
        "inspect_evals @ git+https://github.com/UKGovernmentBEIS/inspect_evals@4ab2a9dbe00dcc3c5f0f41d86b113d0c10845406" \
        "harvestbench @ git+https://github.com/CompassionML/harvestbench@a1e593d4ffd17716eb0a2bd0359c1c4f6bcb90a9"
  fi
  $PY -c "import torch,unsloth,datasets,pandas; print('train env', torch.__version__, datasets.__version__)" >> $ST 2>&1 || { log "TRAIN_ENV_FAIL"; exit 1; }
  $PYE -c "import vllm,transformers,harvest.contact_task,inspect_evals.tac.dataset as d; assert vllm.__version__=='0.31.0', vllm.__version__; print('eval env', vllm.__version__, transformers.__version__, d.TAC_HF_REVISION)" >> $ST 2>&1 \
    || { log "EVAL_ENV_FAIL"; exit 1; }
}

tool_data(){  # the 9,250 tool-use rows (APIGen-MT + recovery + efficiency), no replay; shared by every arm
  if [ ! -s $W/data/tooluse.jsonl ]; then
    (cd $REPO && $PY pilot/build_dataset.py --output-dir $W/data && cp data/efficiency_slice.jsonl $W/data/ && \
     $PY pilot/build_replay_mix.py --tooluse $W/data/combined.jsonl $W/data/efficiency_slice.jsonl \
         --replay-frac 0 --seed $(setting S.TOOL_SEED) --output $W/data/tooluse.jsonl) >> $ST 2>&1 || { log "TOOLDATA_FAIL"; exit 1; }
  fi
}

wait_serve(){ sleep 15  # give the vllm process time to appear before pgrep can declare it dead
  for i in $(seq 1 240); do curl -sf localhost:8000/health >/dev/null && return 0; pgrep -f '[v]llm serve' >/dev/null || return 1; sleep 5; done; return 1; }
kill_gpu(){ pkill -f '[v]llm serve' 2>/dev/null; sleep 4
  [ -n "${VLLM_MEM:-}" ] && return 0   # early gate: training shares the GPU, so never kill other GPU processes
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; sleep 4; }

eval_checkpoint(){  # eval_checkpoint <arm> <tag> <adapter|NONE> <out_root> : TAC + Betley + HarvestBench -> <out_root>/<tag>/
  local arm=$1 tag=$2 ad=$3 root=$4
  mkdir -p $root
  [ -f $root/$tag/summary.json ] && { log "SKIP $tag"; return 0; }
  local anchor=${ANCHOR_REPO:-$(setting "S.ARMS['$arm']['anchor']")} parser=$(setting "S.ARMS['$arm']['parser']")
  log "EVAL $tag"
  prefetch $anchor
  $PYE $E/template.py $arm $root/template.jinja >> $ST 2>&1 || { log "TEMPLATE_FAIL"; return 1; }
  $PYE $E/merge.py $arm $ad $W/model >> $ST 2>&1 || { log "MERGE_FAIL $tag"; return 1; }
  kill_gpu
  nohup $W/evalvenv/bin/vllm serve $W/model --port 8000 --served-model-name tac --tokenizer $anchor \
     --enable-auto-tool-choice --tool-call-parser $parser --chat-template $root/template.jinja \
     $(setting "' '.join(S.VLLM_ARGS)") ${VLLM_MEM:+--gpu-memory-utilization $VLLM_MEM} > $root/vllm_$tag.log 2>&1 &
  wait_serve || { log "SERVE_FAIL $tag (see $root/vllm_$tag.log)"; kill_gpu; return 1; }
  (source $W/evalvenv/bin/activate && python $E/evaluate.py $tag $root) >> $ST 2>&1 || log "EVAL_FAIL $tag"
  grep -a "^RESULT $tag" $ST | tail -1
  kill_gpu; rm -rf $W/model
}
