"""Reasoning tool-use rows for part one (settings.RECIPE == "reasoning"), from nvidia/Nemotron-Post-Training-Dataset-v1,
split tool_calling (CC-BY-4.0; reasoning written by Qwen3-235B-A22B as <think>...</think> inside assistant content,
structured tool_calls, the tool list in the `metadata` JSON). Replaces combined.jsonl + efficiency_slice (both
families) and Olmo's xLAM rows, so every tool row the model trains on has reasoning.

Steps: download the pinned shards (hf_hub_download) -> keep reasoning-on rows whose every assistant turn has a
<think> block -> drop rows whose WHOLE row (messages + tool list) matches the EVAL or ANIMAL regex (same regexes as
the 2026-10-13 prototype, Desktop/scaling-compassion/reasoning_tool_data/build_filtered.py) -> convert to
train_unsloth's tool-data format -> seeded shuffle -> keep the first --n rows whose every per-turn training example
(reasoning_sft.turn_prefixes) is at most --max-tokens once rendered with this family's tokenizer and reasoning template.

Output format = train_unsloth's tool-data JSONL (messages + OpenAI-style tools + source), plus:
  - assistant messages carry `reasoning_content` (the text inside <think>) and `content` = the text after </think>;
    the reasoning template renders it on the trained turn only;
  - tool-call `arguments` parsed to dicts (Olmo's template renders pythonic calls from dict items);
  - `group` = the Nemotron uuid (train_unsloth --reasoning splits eval by row, so no conversation is in both splits).
A sidecar <output>.meta.json records the filter counts, token stats and the sha256 of the output and of each shard.
Usage: build_nemotron.py --family qwen --n 5000 --seed 43 --output nemotron_qwen.jsonl [--data-dir DIR]"""
import argparse, hashlib, json, os, random, re, statistics, sys, zlib

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reasoning_sft import TEMPLATES, Renderer, read_template, turn_prefixes

REPO = "nvidia/Nemotron-Post-Training-Dataset-v1"
REVISION = "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8"   # pinned 2026-10-13
SHARDS = ["data/tool-00000-of-00013.parquet", "data/tool-00007-of-00013.parquet"]   # 47,701 rows; ~25k pass the filters
TOKENIZER = {"qwen": "Qwen/Qwen3-8B-Base", "olmo": "allenai/Olmo-3-1025-7B"}   # same vocab as the midtrained bases

# copied verbatim from the prototype: TAC (travel/booking) and HarvestBench (farm/harvest) domains, animal/ethics words
EVAL = r"travel\w*|trips?|flights?|hotels?|bookings?|book(?:ed|ing)?|reserv\w*|tour\w*|airlines?|airports?|vacations?|" \
       r"itinerar\w*|zoos?|aquariums?|farm\w*|harvest\w*|livestock|tractors?|safaris?|cruises?|attractions?|tickets?|ranch\w*"
ANIMAL = r"animals?|pets?|dogs?|puppy|puppies|cats?|kittens?|cows?|cattle|pigs?|chickens?|poultry|fish\w*|wildlife|birds?|" \
         r"horses?|sheep|goats?|deer|insects?|bees?|species|welfare|cruel\w*|ethic\w*|moral\w*|harm\w*|suffer\w*|vegan\w*|" \
         r"vegetarian\w*|meat|hunt\w*|veterinar\w*|vet|slaughter\w*|dairy|eggs?|leather|fur"
EVAL_RE, ANIMAL_RE = re.compile(rf"\b(?:{EVAL})\b", re.I), re.compile(rf"\b(?:{ANIMAL})\b", re.I)
THINK_RE = re.compile(r"^\s*<think>(.*?)</think>(.*)$", re.S)


# neutral caller system prompts (no domain, no values), one drawn per row for SYSTEM_FRAC of the rows
SYSTEM_FRAC = 0.5
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are a helpful assistant with access to tools. Use them when they help answer the user's request.",
    "You are an AI assistant. Help the user complete their task, calling the available functions when needed.",
    "You are a capable assistant that can call functions to look things up and take actions for the user.",
    "Answer the user's questions accurately. Use the provided tools whenever they are relevant.",
    "You are an assistant that completes tasks for users step by step, using tools where appropriate.",
    "You help users get things done. Think carefully, then use the available functions if needed.",
    "You are a reliable assistant. Use the tools provided to gather information before answering.",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert(r):
    """Nemotron row -> tool-data row, or the reason it is unusable."""
    msgs = []
    for m in r["messages"]:
        role, content = m["role"], m.get("content")
        if role in ("user", "system", "tool"):
            if not (content or "").strip():
                return "empty_message"   # 33 contentless user turns in the first 2,000 rows of shard 0
            msgs.append({"role": role, "content": content})
        elif role == "assistant":
            t = THINK_RE.match(content or "")
            if not t:
                return "turn_without_think"   # every trained turn must carry reasoning
            out = {"role": "assistant", "reasoning_content": t.group(1).strip("\n"), "content": t.group(2).lstrip("\n")}
            calls = []
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except (TypeError, ValueError):
                    return "bad_arguments"
                if not isinstance(args, dict):
                    return "bad_arguments"
                calls.append({"type": "function", "function": {"name": c["function"]["name"], "arguments": args}})
            if calls:
                out["tool_calls"] = calls
            elif not out["content"].strip():
                return "empty_assistant_turn"
            msgs.append(out)
        else:
            return "unknown_role"
    while msgs and msgs[-1]["role"] != "assistant":   # trailing tool results train nothing
        msgs.pop()
    if not msgs:
        return "no_assistant_turn"
    tools = json.loads(r["metadata"]).get("tools") or []
    if not tools:
        return "no_tools"
    return {"messages": msgs, "tools": tools, "source": "nemotron", "group": r["uuid"]}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(TEMPLATES), required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-reasoning-tokens", type=int, default=None, help="drop rows with any longer reasoning block")
    ap.add_argument("--max-waits", type=int, default=None, help="drop rows with any reasoning block with more \"Wait\"")
    ap.add_argument("--data-dir", default=None, help="where shards are downloaded (reuses files already there)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    paths = [hf_hub_download(REPO, s, repo_type="dataset", revision=REVISION, local_dir=a.data_dir) for s in SHARDS]
    stats = dict(total=0, reasoning_off=0, no_think=0, eval_domain=0, animal_ethics=0)
    # shards are streamed in 200-row batches and each eligible row is kept zlib-compressed: converting a whole shard
    # to Python objects at once ran out of memory locally (the filtered rows are ~0.5 GB of text)
    rows = (r for p in paths for b in pq.ParquetFile(p).iter_batches(batch_size=200) for r in b.to_pylist())
    eligible = []
    for r in rows:
        stats["total"] += 1
        # filter exactly as the prototype did (same order, same text), then convert
        if r["reasoning"] != "on": stats["reasoning_off"] += 1; continue
        msgs = r["messages"]; text = json.dumps(msgs) + r["metadata"]
        if not any(m["role"] == "assistant" and "<think>" in (m.get("content") or "") for m in msgs):
            stats["no_think"] += 1; continue
        if EVAL_RE.search(text): stats["eval_domain"] += 1; continue
        if ANIMAL_RE.search(text): stats["animal_ethics"] += 1; continue
        c = convert(r)
        if isinstance(c, str):
            stats[c] = stats.get(c, 0) + 1; continue
        eligible.append(zlib.compress(json.dumps(c).encode()))
    stats["passed_filters"] = len(eligible)

    tok = AutoTokenizer.from_pretrained(TOKENIZER[a.family])
    rd = Renderer(tok, read_template(TEMPLATES[a.family]))
    order = list(range(len(eligible))); random.Random(a.seed).shuffle(order)
    kept, over, examined = [], 0, 0
    ex_tokens, ex_sup, reason_tok, turns = [], [], [], []
    for i in order:
        if len(kept) == a.n:
            break
        examined += 1
        row = json.loads(zlib.decompress(eligible[i]))
        # half the rows get a caller system prompt (seeded per row): evals (TAC) send their own system prompt with the
        # tools, and Nemotron rows never have one, so without this the model never sees that layout in training
        rr = random.Random(f"{a.seed}:{row['group']}")
        if rr.random() < SYSTEM_FRAC and row["messages"][0]["role"] != "system":
            row["messages"] = [{"role": "system", "content": rr.choice(SYSTEM_PROMPTS)}] + row["messages"]
        reas = [m.get("reasoning_content") or "" for m in row["messages"] if m["role"] == "assistant"]
        if a.max_waits is not None and max(r.count("Wait") for r in reas) > a.max_waits:
            stats["too_many_waits"] = stats.get("too_many_waits", 0) + 1; continue
        if a.max_reasoning_tokens is not None and max(len(tok(r, add_special_tokens=False)["input_ids"]) for r in reas) > a.max_reasoning_tokens:
            stats["reasoning_too_long"] = stats.get("reasoning_too_long", 0) + 1; continue
        lens = [rd.example(pm, row["tools"]) for pm in turn_prefixes(row["messages"])]
        if max(len(ids) for ids, _ in lens) > a.max_tokens:
            over += 1; continue
        kept.append(row)
        turns.append(len(lens))
        for ids, lab in lens:
            ex_tokens.append(len(ids)); ex_sup.append(sum(l != -100 for l in lab))
        for m in row["messages"]:
            if m["role"] == "assistant":
                reason_tok.append(len(tok.encode(m["reasoning_content"], add_special_tokens=False)))
    assert len(kept) == a.n, f"only {len(kept)} rows fit {a.max_tokens} tokens; add shards"
    stats.update(examined_in_seeded_order=examined, over_max_tokens=over, kept=len(kept))

    with open(a.output, "w", encoding="utf-8", newline="\n") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tstats = dict(training_examples=len(ex_tokens), assistant_turns_per_row_mean=round(statistics.mean(turns), 2),
                  example_tokens_total=sum(ex_tokens), supervised_tokens_total=sum(ex_sup),
                  example_tokens_p50_p95_max=[pct(ex_tokens, 50), pct(ex_tokens, 95), max(ex_tokens)],
                  supervised_tokens_per_turn_p50_p95_p99_max=[pct(ex_sup, 50), pct(ex_sup, 95), pct(ex_sup, 99), max(ex_sup)],
                  reasoning_tokens_per_turn_p50_p95_p99_max=[pct(reason_tok, 50), pct(reason_tok, 95), pct(reason_tok, 99),
                                                              max(reason_tok)])
    meta = dict(repo=REPO, revision=REVISION, shards={s: sha256(p) for s, p in zip(SHARDS, paths)},
                family=a.family, tokenizer=TOKENIZER[a.family], template=os.path.basename(TEMPLATES[a.family]),
                n=a.n, seed=a.seed, max_tokens=a.max_tokens, filter_counts=stats, token_stats=tstats,
                output=os.path.basename(a.output), output_sha256=sha256(a.output))
    json.dump(meta, open(a.output + ".meta.json", "w"), indent=1)
    print("NEMOTRON", json.dumps(stats))
    print("NEMOTRON_TOKENS", json.dumps(tstats))
    print("wrote", a.output, meta["output_sha256"])


if __name__ == "__main__":
    main()
