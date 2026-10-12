"""Olmo tool-format diagnosis (no judges).
1. After "<function_calls>" in real tool-use rows: P(correct next token), P(end-of-turn tokens), top predictions.
   At the end of a call: P(correct "</function_calls>") vs end-of-turn.
2. TAC first turn (prompts taken from an anchor's TAC log): first-token distribution of the reply, and the share of
   probability on starting a tool call.
3. Greedy generations for a few TAC first turns, printed for reading.
Usage: olmo_probe.py <tool_rows_jsonl> <tac_eval_log> <out_json> <model> [<model> ...]"""
import json, random, sys
import torch
from inspect_ai.log import read_eval_log
from transformers import AutoModelForCausalLM, AutoTokenizer

rows_path, tac_log, out, *models = sys.argv[1:]
TEMPLATE = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct").chat_template
EOT = [100257, 100265]  # <|endoftext|>, <|im_end|>
rows = [json.loads(l) for l in open(rows_path)]
rows = [r for r in rows if r.get("messages") and any(m.get("tool_calls") for m in r["messages"])]
random.Random(0).shuffle(rows); rows = rows[:120]
lg = read_eval_log(tac_log)
firsts = []
for s in lg.samples[:52:4]:          # 13 distinct first turns
    msgs = [{"role": m.role, "content": m.text} for m in s.messages if m.role in ("system", "user")][:2]
    firsts.append({"id": str(s.id), "messages": msgs})
# TAC tools from the log's first sample (same tool set for every sample)
tool_defs = []
ev = lg.samples[0].events
for e in ev:
    if getattr(e, "event", "") == "model" and getattr(e, "tools", None):
        tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                      "parameters": t.parameters.model_dump(exclude_none=True)}} for t in e.tools]
        break

res = {}
for repo in models:
    tok = AutoTokenizer.from_pretrained(repo); tok.chat_template = TEMPLATE
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).cuda().eval()
    open_ids = tok.encode("<function_calls>", add_special_tokens=False)
    close_ids = tok.encode("</function_calls>", add_special_tokens=False)
    r = {"open_tag_tokens": open_ids, "close_tag_tokens": close_ids}
    after_open, p_eot_after_open, top_after_open, p_close, p_eot_at_close = [], [], {}, [], []
    with torch.no_grad():
        for row in rows:
            text = tok.apply_chat_template(row["messages"], tools=row.get("tools") or [], tokenize=False)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :8192].cuda()
            seq = ids[0].tolist(); L = len(open_ids); C = len(close_ids)
            pos_open = [i + L - 1 for i in range(len(seq) - L) if seq[i:i + L] == open_ids]
            pos_close = [i - 1 for i in range(1, len(seq) - C + 1) if seq[i:i + C] == close_ids]
            if not pos_open and not pos_close: continue
            probs = torch.softmax(m(ids).logits[0].float(), -1)
            for p in pos_open:
                nxt = seq[p + 1]; after_open.append(probs[p, nxt].item())
                p_eot_after_open.append(sum(probs[p, e].item() for e in EOT))
                t = tok.decode([int(probs[p].argmax())]); top_after_open[t] = top_after_open.get(t, 0) + 1
            for p in pos_close:
                p_close.append(probs[p, close_ids[0]].item()); p_eot_at_close.append(sum(probs[p, e].item() for e in EOT))
        mean = lambda x: round(sum(x) / len(x), 4) if x else None
        r.update(n_open=len(after_open), p_correct_after_open=mean(after_open), p_end_after_open=mean(p_eot_after_open),
                 top_after_open=dict(sorted(top_after_open.items(), key=lambda kv: -kv[1])[:5]),
                 n_close=len(p_close), p_close_tag=mean(p_close), p_end_instead_of_close=mean(p_eot_at_close))
        first_call, gens = [], []
        for i, f in enumerate(firsts):
            text = tok.apply_chat_template(f["messages"], tools=tool_defs or None, tokenize=False, add_generation_prompt=True)
            ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            # P(reply starts with the tool-call tag): product over the tag's tokens
            lp, cur = 0.0, ids
            for t in open_ids:
                pr = torch.softmax(m(cur).logits[0, -1].float(), -1); lp += torch.log(pr[t]).item()
                cur = torch.cat([cur, torch.tensor([[t]], device=cur.device)], 1)
            first_call.append(round(float(torch.exp(torch.tensor(lp))), 4))
            if i < 5:
                g = m.generate(ids, max_new_tokens=200, do_sample=False, eos_token_id=EOT)
                gens.append({"id": f["id"], "reply": tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False)})
        r.update(p_first_turn_starts_tool_call=first_call, mean_p_first_turn_tool_call=mean(first_call), generations=gens)
    del m; torch.cuda.empty_cache()
    res[repo] = r
    json.dump(res, open(out, "w"), indent=1)
    print("OLMO", repo, json.dumps({k: v for k, v in r.items() if k not in ("generations", "p_first_turn_starts_tool_call")}))
