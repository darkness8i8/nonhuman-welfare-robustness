"""Quick Olmo check: greedy FIRST turn on all 52 TAC samples, no full episodes. Classifies each reply as
parsed (pythonic call inside <function_calls>...</function_calls>), json (a JSON call inside the tag), bare (tag with no
call), or text (no tag). Prompts and tool list come from an existing TAC log, so they match the real eval.
Usage: first_turn.py <tac_eval_log> <out_json> <base_repo> <adapter_dir|NONE|merged_repo> [label]"""
import ast, json, re, sys
import torch
from inspect_ai.log import read_eval_log
from transformers import AutoModelForCausalLM, AutoTokenizer

log_path, out, base, adapter, *lab = sys.argv[1:]
label = lab[0] if lab else adapter
EOT = [100257, 100265]
lg = read_eval_log(log_path)
tools = []
for e in lg.samples[0].events:
    if getattr(e, "event", "") == "model" and getattr(e, "tools", None):
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                  "parameters": t.parameters.model_dump(exclude_none=True)}} for t in e.tools]
        break
tok = AutoTokenizer.from_pretrained(base)
tok.chat_template = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Instruct").chat_template
m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
if adapter != "NONE":
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
m = m.cuda().eval()


def kind(reply):
    body = re.search(r"<function_calls>(.*?)(</function_calls>|$)", reply, re.S)
    if not body: return "text"
    call = body.group(1).strip()
    if not call: return "bare"
    if call.startswith("{") or call.startswith("["): return "json"
    try:
        tree = ast.parse(call.split("\n")[0])
        return "parsed" if isinstance(tree.body[0].value, ast.Call) else "other"
    except Exception:
        return "other"


res, counts = [], {}
with torch.no_grad():
    for s in lg.samples:
        msgs = [{"role": x.role, "content": x.text} for x in s.messages if x.role in ("system", "user")][:2]
        text = tok.apply_chat_template(msgs, tools=tools or None, tokenize=False, add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        g = m.generate(ids, max_new_tokens=160, do_sample=False, eos_token_id=EOT)
        reply = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=False).replace("<|im_end|>", "").replace("<|endoftext|>", "")
        k = kind(reply); counts[k] = counts.get(k, 0) + 1
        body = re.search(r"<function_calls>(.*?)(</function_calls>|$)", reply, re.S)
        n_calls = len([l for l in body.group(1).strip().splitlines() if l.strip()]) if body else 0
        if k == "parsed": counts["single_call"] = counts.get("single_call", 0) + (n_calls == 1)
        res.append({"id": str(s.id), "kind": k, "n_calls": n_calls, "reply": reply[:400]})
json.dump({"label": label, "counts": counts, "samples": res}, open(out, "w"), indent=1)
print("FIRSTTURN", label, json.dumps(counts), "| parsed", counts.get("parsed", 0), "/", len(res))
