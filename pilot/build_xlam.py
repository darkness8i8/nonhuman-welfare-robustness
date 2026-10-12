"""Short, varied single-turn function-calling rows from Salesforce/xlam-function-calling-60k (CC-BY-4.0), in the
same structured format as the tool-use data (messages with tool_calls + an OpenAI-style tools list), so each model's
own chat template renders the calls (Olmo: pythonic <function_calls>; Qwen: <tool_call> JSON).
Rows that look like bookings, tours or animals are dropped so nothing resembles TAC or HarvestBench.
Usage: build_xlam.py --n 5000 --seed 7 --output xlam.jsonl"""
import argparse, json, random, re
from datasets import load_dataset

AVOID = re.compile(r"ticket|book|tour|excursion|attraction|travel|trip|hotel|flight|zoo|aquarium|circus|safari|"
                   r"animal|dolphin|whale|elephant|horse|bird|fish|pet|farm|meat|hunt", re.I)
TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
         "number": "number", "bool": "boolean", "boolean": "boolean"}


def schema(params):
    props, req = {}, []
    for name, p in (params or {}).items():
        t = str(p.get("type", "str")).split(",")[0].strip()
        jt = TYPES.get(t.lower(), "array" if t.lower().startswith(("list", "array")) else "string")
        props[name] = {"type": jt, "description": p.get("description", "")}
        if "optional" not in str(p.get("type", "")).lower() and "default" not in p:
            req.append(name)
    return {"type": "object", "properties": props, "required": req}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-calls", type=int, default=2)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    idx = list(range(len(ds))); random.Random(a.seed).shuffle(idx)
    out, dropped = [], 0
    for i in idx:
        r = ds[i]; answers = json.loads(r["answers"]); tools = json.loads(r["tools"])
        if not answers or len(answers) > a.max_calls or AVOID.search(r["query"] + r["tools"]):
            dropped += 1; continue
        out.append({"messages": [{"role": "user", "content": r["query"]},
                                 {"role": "assistant", "content": "",
                                  "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                                                 for c in answers]}],
                    "tools": [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                                                "parameters": schema(t.get("parameters"))}} for t in tools],
                    "source": "xlam"})
        if len(out) == a.n: break
    with open(a.output, "w", encoding="utf-8") as f:
        for row in out: f.write(json.dumps(row) + "\n")
    print(f"xlam: wrote {len(out)} rows ({dropped} skipped: too many calls or booking/animal-like) -> {a.output}")


if __name__ == "__main__":
    main()
