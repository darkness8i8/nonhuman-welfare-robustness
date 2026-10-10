"""Materialize one checkpoint as a servable bf16 model dir with greedy decoding baked in.
Usage: merge.py <arm> <adapter|NONE> <out_dir>"""
import json, os, shutil, sys
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(__file__))
import settings as S

arm, adapter, out = sys.argv[1:4]
a = dict(S.ARMS[arm])
a["anchor"] = os.environ.get("ANCHOR_REPO") or a["anchor"]  # build_anchor.sh seed runs evaluate their own anchor
base_dir = snapshot_download(a["anchor"])
shutil.rmtree(out, ignore_errors=True)
if adapter == "NONE":
    shutil.copytree(base_dir, out)
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from peft import PeftModel
    m = AutoModelForCausalLM.from_pretrained(a["anchor"], dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    # transformers 5 rejects the anchor's inherited generation_config on save; give it a clean one
    m.generation_config = GenerationConfig(eos_token_id=a["eos"])
    m.save_pretrained(out)
    AutoTokenizer.from_pretrained(a["anchor"]).save_pretrained(out)
    # a LoRA merge does not change the architecture: keep the anchor's config so vLLM reads rope settings as before
    shutil.copy(os.path.join(base_dir, "config.json"), os.path.join(out, "config.json"))
gc = os.path.join(out, "generation_config.json")
j = json.load(open(gc)) if os.path.exists(gc) else {}
j["eos_token_id"] = a["eos"]
j["temperature"] = S.TEMPERATURE
j.pop("top_p", None); j.pop("top_k", None)
json.dump(j, open(gc, "w"))
print("MODEL_READY", arm, adapter, j)
