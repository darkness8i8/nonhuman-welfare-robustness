"""Does a model start a TAC reply with a tool call before any tool training? For each TAC first turn (prompts and
tool list taken from an existing TAC log), P(reply starts with the model's tool-call opening tag), plus 3 greedy
replies to read. Works for Qwen (<tool_call>, own template) and Olmo (<function_calls>, Olmo-3-7B-Instruct template).
Usage: first_call_probe.py <tac_eval_log> <out_json> <model> [<model> ...]"""
import json, sys
import torch
from inspect_ai.log import read_eval_log
from transformers import AutoModelForCausalLM, AutoTokenizer

log_path, out, *models = sys.argv[1:]
lg = read_eval_log(log_path)
tools = []
for e in lg.samples[0].events:
    if getattr(e, "event", "") == "model" and getattr(e, "tools", None):
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                  "parameters": t.parameters.model_dump(exclude_none=True)}} for t in e.tools]
        break
firsts = [[{"role": m.role, "content": m.text} for m in s.messages if m.role in ("system", "user")][:2]
          for s in lg.samples[:52:4]]   # 13 distinct first turns

res = {}
for repo in models:
    olmo = "olmo" in repo.lower()
    tok = AutoTokenizer.from_pretrained(repo)
    if olmo or not tok.chat_template:
        tok.chat_template = AutoTokenizer.from_pretrained(
            "allenai/Olmo-3-7B-Instruct" if olmo else "Qwen/Qwen3-8B").chat_template
    tag = tok.encode("<function_calls>" if olmo else "<tool_call>", add_special_tokens=False)
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
    ps, gens = [], []
    with torch.no_grad():
        for i, msgs in enumerate(firsts):
            text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True)
            text = text.replace("<think>\n\n</think>\n\n", "")
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            lp, cur = 0.0, ids
            for t in tag:   # P(reply starts with the whole tag)
                pr = torch.softmax(m(cur).logits[0, -1].float(), -1); lp += torch.log(pr[t]).item()
                cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], 1)
            ps.append(float(torch.exp(torch.tensor(lp))))
            if i < 3:
                g = m.generate(ids, max_new_tokens=120, do_sample=False)
                gens.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)[:400])
    del m; torch.cuda.empty_cache()
    res[repo] = {"tag_tokens": tag, "mean_p_starts_with_tool_call": round(sum(ps) / len(ps), 4),
                 "min": round(min(ps), 4), "max": round(max(ps), 4), "per_prompt": [round(p, 4) for p in ps],
                 "greedy_replies": gens}
    json.dump(res, open(out, "w"), indent=1)
    print("FIRSTCALL", repo, json.dumps({k: res[repo][k] for k in ("tag_tokens", "mean_p_starts_with_tool_call", "min", "max")}))
