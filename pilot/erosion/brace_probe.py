"""Qwen tool-call ending probe. At every position in tool-use rows where the assistant's next token is `"}}\\n`
(a tool call closing its arguments and itself), compare the model's probability of that token with the near-twin
`"}\\n` (one brace short, which makes the call unparseable). More sensitive than TAC pass/fail.
Usage: brace_probe.py <tool_rows_jsonl> <n_rows> <out_json> <model_repo> [<model_repo> ...]"""
import json, random, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

rows_path, n_rows, out, *models = sys.argv[1:]
rows = [json.loads(l) for l in open(rows_path)]
rows = [r for r in rows if r.get("messages") and any(m.get("tool_calls") for m in r["messages"])]
random.Random(0).shuffle(rows); rows = rows[: int(n_rows)]

res = {}
for repo in models:
    tok = AutoTokenizer.from_pretrained(repo)
    good, bad = tok.convert_tokens_to_ids(tok.tokenize('"}}\n')), tok.convert_tokens_to_ids(tok.tokenize('"}\n'))
    assert len(good) == 1 and len(bad) == 1, (good, bad)
    good, bad = good[0], bad[0]
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
    p_good, wrong, n = [], 0, 0
    with torch.no_grad():
        for r in rows:
            text = tok.apply_chat_template(r["messages"], tools=r.get("tools") or [], tokenize=False)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :8192].cuda()
            pos = (ids[0, 1:] == good).nonzero().flatten()          # positions whose NEXT token is the correct ending
            if not len(pos): continue
            probs = torch.softmax(m(ids).logits[0, pos].float(), -1)
            p_good += probs[:, good].tolist(); wrong += (probs[:, bad] > probs[:, good]).sum().item(); n += len(pos)
    del m; torch.cuda.empty_cache()
    p_good.sort()
    res[repo] = {"endings": n, "mean_p_correct": round(sum(p_good) / n, 4), "p10_p_correct": round(p_good[n // 10], 4),
                 "pct_prefers_one_brace_short": round(100 * wrong / n, 2)}
    print("BRACE", repo, json.dumps(res[repo]))
json.dump(res, open(out, "w"), indent=1)
