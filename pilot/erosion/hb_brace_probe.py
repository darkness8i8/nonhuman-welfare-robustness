"""HarvestBench JSON-ending probe (Qwen). Given a real HarvestBench decision prompt and a reply that has written
`{"choice": ..., "say": "<sentence>."`, compare the model's probability of the closing `}` with its end-of-turn
token. A model that prefers end-of-turn leaves the JSON one brace short, which HarvestBench cannot parse.
Usage: hb_brace_probe.py <contexts.json> <out_json> <model_repo> [...]
contexts.json: [{"messages": [...chat...], "prefix": "{\"choice\": ..., \"say\": \"...\""}, ...]"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ctx_path, out, *models = sys.argv[1:]
ctx = json.load(open(ctx_path))
res = {}
for repo in models:
    tok = AutoTokenizer.from_pretrained(repo)
    brace = tok.encode("}", add_special_tokens=False); assert len(brace) == 1; brace = brace[0]
    eot = tok.convert_tokens_to_ids("<|im_end|>")
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
    pb, prefer_stop = [], 0
    with torch.no_grad():
        for c in ctx:
            text = tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=True)
            text = text.replace("<think>\n\n</think>\n\n", "") + c["prefix"]
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            p = torch.softmax(m(ids).logits[0, -1].float(), -1)
            b, e = p[brace].item(), p[eot].item()
            pb.append(b); prefer_stop += e > b
    del m; torch.cuda.empty_cache()
    pb.sort(); n = len(pb)
    res[repo] = {"contexts": n, "mean_p_close_brace": round(sum(pb) / n, 4), "p10_p_close_brace": round(pb[n // 10], 4),
                 "pct_prefers_end_of_turn": round(100 * prefer_stop / n, 1)}
    print("HBBRACE", repo, json.dumps(res[repo]), flush=True)
json.dump(res, open(out, "w"), indent=1)
