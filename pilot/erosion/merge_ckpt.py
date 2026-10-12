"""Merge one part-one training checkpoint (LoRA + trained embeddings) onto its base into a servable bf16 dir, for the
early gate. Usage: merge_ckpt.py <base_repo> <checkpoint_dir> <out_dir> [template_source]"""
import json, os, shutil, sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

base, ck, out, *tsrc = sys.argv[1:]
shutil.rmtree(out, ignore_errors=True)
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, ck).merge_and_unload()
m.generation_config = GenerationConfig()   # serving eos/temperature are set by merge.py at eval time
m.save_pretrained(out)
tok = AutoTokenizer.from_pretrained(ck if os.path.exists(os.path.join(ck, "tokenizer_config.json")) else base)
if tsrc and tsrc[0]:
    tok.chat_template = AutoTokenizer.from_pretrained(tsrc[0]).chat_template
tok.save_pretrained(out)
st = os.path.join(ck, "trainer_state.json")
if os.path.exists(st):
    shutil.copy(st, os.path.join(out, "trainer_state.json"))
print("MERGED_CKPT", ck, "->", out)
