"""How much midtraining an anchor kept: mean per-token loss on held-out corpus documents (ones not in the anchor's
replay mix), for the original pretrained model, the midtrained base, and each anchor given.
retained = (loss_pretrained - loss_anchor) / (loss_pretrained - loss_midtrained): 1 = all midtraining kept, 0 = none.
Usage: retention.py <arm> <mix_jsonl> <out_json> <anchor_repo> [<anchor_repo> ...]"""
import json, math, os, random, sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import settings as S

PRETRAINED = {"qwen": ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
              "olmo": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f")}
N_DOCS, MAX_TOK = 200, 2048

arm, mix, out, *anchors = sys.argv[1:]
a = S.ARMS[arm]
used = {json.loads(l).get("text") for l in open(mix)}
corpus = load_dataset(a["replay_corpus"], split="train")
held = [r["output"] for r in corpus if r["output"] not in used]
random.Random(0).shuffle(held)
docs = held[:N_DOCS]
print(f"{len(held)} held-out docs (not in the replay mix); scoring {len(docs)}")

def score(repo, rev=None):
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16).cuda().eval()
    nll = n = 0.0
    with torch.no_grad():
        for d in docs:
            ids = tok(d, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :MAX_TOK].cuda()
            loss = m(ids, labels=ids).loss.item()
            nll += loss * (ids.shape[1] - 1); n += ids.shape[1] - 1
    del m; torch.cuda.empty_cache()
    return nll / n

fam = "qwen" if arm.startswith("qwen") else "olmo"
res = {"pretrained": score(*PRETRAINED[fam]), "midtrained": score(a["base"])}
for r in anchors:
    res[r] = score(r)
gap = res["pretrained"] - res["midtrained"]
summary = {k: {"loss": round(v, 4), "retained": round((res["pretrained"] - v) / gap, 3) if gap else None} for k, v in res.items()}
json.dump({"arm": arm, "n_docs": len(docs), "max_tokens": MAX_TOK, "results": summary}, open(out, "w"), indent=1)
print("RETENTION", json.dumps(summary))
