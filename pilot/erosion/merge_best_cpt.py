"""Merge a midtraining run's LOWEST-validation-loss LoRA checkpoint into its base model and push it as a
standalone bf16 model. Refuses to merge any checkpoint that is not the run's validation minimum.
Usage: merge_best_cpt.py <lora_checkpoints_repo> <out_repo>

Why: the main CPT runs exported their last checkpoint (step 1500, ~epoch 4), two epochs past the validation
minimum (step 750); the seed runs exported the minimum. Always build on the minimum."""
import json, os, sys, shutil, tempfile
import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

lora_repo, out_repo = sys.argv[1:3]
api = HfApi()
steps = sorted(int(f.split("/")[0].split("-")[1]) for f in api.list_repo_files(lora_repo)
               if f.startswith("checkpoint-") and f.endswith("/trainer_state.json"))
state = json.load(open(hf_hub_download(lora_repo, f"checkpoint-{steps[-1]}/trainer_state.json")))
evals = [(x["step"], x["eval_loss"]) for x in state["log_history"] if "eval_loss" in x]
best_step, best_loss = min(evals, key=lambda t: t[1])
assert f"checkpoint-{best_step}/adapter_model.safetensors" in api.list_repo_files(lora_repo), best_step
print(f"validation minimum: step {best_step} (eval_loss {best_loss:.4f}); last step {steps[-1]}")

sub = f"checkpoint-{best_step}"
ad_dir = snapshot_download(lora_repo, allow_patterns=[f"{sub}/*"])
cfg = json.load(open(os.path.join(ad_dir, sub, "adapter_config.json")))
manifest = json.load(open(os.path.join(ad_dir, sub, "run_manifest.json")))
base, rev = cfg["base_model_name_or_path"], manifest.get("model_revision")
m = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.bfloat16)
m = PeftModel.from_pretrained(m, os.path.join(ad_dir, sub)).merge_and_unload()
out = tempfile.mkdtemp()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(os.path.join(ad_dir, sub)).save_pretrained(out)
manifest.update({"exported_checkpoint": sub, "exported_step": best_step, "exported_eval_loss": best_loss,
                 "export_rule": "lowest validation loss", "source_lora_repo": lora_repo,
                 "validation_curve": evals})
json.dump(manifest, open(os.path.join(out, "run_manifest.json"), "w"), indent=1)
open(os.path.join(out, "README.md"), "w").write(
    f"---\nbase_model: {base}\n---\n# {out_repo.split('/')[-1]}\n\nMerged bf16 model from `{lora_repo}` "
    f"`{sub}`: the run's lowest validation loss ({best_loss:.4f} at step {best_step}; the run continued to step "
    f"{steps[-1]}). Built by `pilot/erosion/merge_best_cpt.py`.\n")
api.create_repo(out_repo, private=False, exist_ok=True)  # public: the org's private storage is full
api.upload_folder(repo_id=out_repo, folder_path=out, commit_message=f"Merge {sub} (lowest val loss) of {lora_repo}")
shutil.rmtree(out, ignore_errors=True)
print("PUSHED", out_repo, "from", lora_repo, sub)
