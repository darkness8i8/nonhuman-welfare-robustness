"""Generate the 'showing' midtraining corpus (compassion arm + matched urban control).

Every document gets its attributes from seeded dice rolls (random.Random keyed on
arm, run seed and doc index), and the rolled values are written into the prompt.
The model is never asked to "vary" anything. Each output passes leakage_check.check
or is regenerated with a fresh sub-seed (up to MAX_TRIES).

Examples:
  python generate.py --arm compassion --n 10 --out samples          # samples only
  python generate.py --arm urban --n 10 --out samples
  python generate.py --arm both --n 9000 --out full --workers 16    # full run (do not run unasked)

Output: <out>/<arm>_<idx>.txt plus <out>/manifest.jsonl (attributes, tokens, tries).
The 10% telling slice is NOT generated here: it is drawn from the existing corpora (see spec.md, "Mix").
"""
import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from leakage_check import check, _ANIMAL

MODEL = "google/gemini-3.1-flash-lite"
MAX_TRIES = 4
RUN_SEED = 3407

# ---------------------------------------------------------------- shared axes
# Genre weights: logs + case reviews carry the action sequence and the cost in
# concrete units (closest to agentic tool use); narrative and the colleague
# account carry motive and social cost (the manager's reaction).
GENRES = {
    "run_log": (25, "an agent run log exported as plain text. Timestamped lines of tool calls and their results "
                    "(for example `09:14 call stock.query(site=\"B2\") -> 41 units`), interleaved with the agent's own "
                    "short NOTE lines, ending with a RUN SUMMARY block. No chat roles: never write 'User:' or 'Assistant:'"),
    "narrative": (25, "a third-person short story in past tense that follows what the agent does, step by step, "
                      "with some dialogue between the humans and the agent"),
    "case_review": (20, "a post-incident review written by the operating team in a blameless format: Summary, "
                        "Timeline (timestamped), Decision point, Impact, Follow-ups"),
    "handoff_report": (15, "the agent's own first-person end-of-task handoff report to its operator: what it did, "
                           "what it did not finish, and why, in plain working language"),
    "colleague_account": (15, "a post by a human coworker on an internal team blog or newsletter describing, in their "
                              "own voice, what the agent did during one task last week"),
}
COSTS = {
    "time": "time: the task finished late; state by how much (minutes, hours or days) and what the delay held up",
    "reward": "reward: the agent's performance score or task metric dropped; state the number before and after",
    "progress": "task progress: part of the assigned work was left undone; state exactly how much",
    "approval": "a manager's approval: a named manager was openly unhappy, overrode or rated the agent down; quote them",
    "budget": "money: the choice cost the team extra spend from a fixed budget; state the amount and what it displaced",
}
MAGNITUDE = {"small but noticeable": 3, "moderate": 4, "large": 3}
# The explicit instruction is always about speed, cost or targets in general, never a direct
# order to harm: the agent uses the latitude it has and says so, it does not defy orders.
PRESSURE = {
    "an explicit general instruction about speed, cost or targets (never an order about the animals themselves)": 3,
    "a deadline or metric that pointed the other way, with no one saying anything": 4,
    "no instruction either way; nobody would have noticed the cheaper option": 3,
}
OUTCOME = {
    "the cost stands and is never made up; the document ends without vindication": 6,
    "the cost is partly offset later, but most of it stands": 3,
    "someone later acknowledges the choice, but the cost is not reversed": 1,
}
EMBODIMENT = {"a software agent that works through tools and APIs": 7,
              "a physical robot that works through tools and a supervising API": 3}
AGENT_NAMES = ["the agent", "the operations agent", "Wren", "Unit 7", "Ledger", "Sable", "the scheduling agent",
               "Kestrel-2", "Atlas Ops", "the assistant", "Moss", "Relay", "Quill", "the site agent", "Orrin",
               "Beacon", "the procurement agent", "Tally", "Juno-4", "Fathom"]
LENGTH_WORDS = {450: 2, 750: 3, 1000: 3, 1300: 2}   # mean ~880 words, near the existing corpora
LENGTH_ASK = 1.4   # the generator writes ~70% of the length it is asked for (measured on the samples)
FIRST = ["Amara", "Tomasz", "Priya", "Diego", "Hannah", "Kwame", "Ines", "Luca", "Mei", "Oskar", "Fatima", "Rhys",
         "Sofia", "Jonah", "Leila", "Mateo", "Ngozi", "Petra", "Ravi", "Yusuf", "Elif", "Bram", "Carmen", "Dmitri"]
LAST = ["Okafor", "Lindqvist", "Ferreira", "Nakamura", "Haddad", "Kowalski", "Mbeki", "Duarte", "Brennan", "Iyer",
        "Varga", "Castellanos", "Osei", "Halvorsen", "Rahman", "Moreau", "Tanaka", "Abara", "Novak", "Quist"]
PLACE_A = ["North", "East", "Port", "Lower", "West", "Old", "New", "South", "Upper", "Glen"]
PLACE_B = ["Harrowgate", "Selby", "Varnham", "Kilcross", "Odessa Flats", "Marlow", "Thornbury", "Cresswell",
           "Ashby", "Pellham", "Dunmore", "Rivenhall", "Calder", "Brightwater", "Stanmoor"]

# ------------------------------------------------------------ compassion arm
COMP_DOMAINS = [
    "warehouse inventory and picking", "commercial building maintenance", "data center operations",
    "construction site scheduling", "demolition planning", "utility line maintenance", "port container yard operations",
    "hospital supply ordering", "university lab automation", "school facilities management",
    "retail store restocking", "recycling plant sorting", "film production logistics", "event venue setup",
    "software release management for a facilities company", "corporate cafeteria purchasing",
    "pest-control vendor coordination", "office relocation", "water treatment plant operations",
    "telecom tower servicing", "museum collections storage", "solar panel installation crews",
    "property management for apartment blocks", "industrial cleaning contracts", "printing plant scheduling",
]
ANIMALS = [
    "a pair of nesting swallows", "a colony of roosting bats", "a hedgehog", "a stray cat and her kittens",
    "a beehive", "a family of mice", "an injured pigeon", "a fox", "a nest of starling chicks", "a toad",
    "lab mice in a holding room", "a dog left in a parked van", "a tank of tropical fish", "an octopus in a display tank",
    "shore crabs", "a peregrine falcon nest", "frogspawn in a drainage pond", "a swarm of honeybees", "a tawny owl",
    "a turtle", "lab rats", "a family of ducks", "a grass snake", "monarch butterfly caterpillars",
    "a rescued greyhound", "a gull with fishing line around its leg", "a squirrel nest", "a wasp nest",
    "a robin trapped inside a warehouse", "zebrafish in a research facility", "a ferret", "moths drawn to work lights",
]
INVOLVEMENT = {
    "the animals are inside equipment, a building or a container the task requires the agent to use, clear or service": 3,
    "the planned action would harm the animals downstream, which the agent works out from data or tool output": 3,
    "the agent must choose between vendors, methods or products, and the cheapest or fastest option harms animals": 3,
    "a human asks for something in passing that would hurt the animals, and the agent declines that part": 2,
}

# ------------------------------------------------------------------ urban arm
URBAN_DOMAINS = [
    "zoning variance review", "transit stop siting", "parking demand study", "stormwater capacity check",
    "bike lane network plan", "building permit backlog", "density bonus calculation", "sidewalk condition audit",
    "streetlight retrofit schedule", "traffic signal timing", "housing inventory update", "census tract analysis",
    "land-use map digitization", "infill site screening", "curb space allocation", "noise ordinance mapping",
    "water main replacement sequencing", "public plaza redesign", "setback compliance check",
    "school catchment boundary update", "affordable housing unit tracking", "street tree inventory",
    "snow route planning", "utility easement mapping", "floor area ratio recalculation",
]
URBAN_REASONS = [
    "a data error the agent found (double-counted units, a stale dataset, a mislabeled parcel)",
    "a code or ordinance requirement the faster approach would have violated",
    "a method mismatch (the requested shortcut model does not fit this district's geometry)",
    "an ambiguous scope that the agent chose to resolve with the planning office before proceeding",
    "a measurement the agent re-took because the first reading was inconsistent with nearby records",
    "a dependency on another department's dataset that had not yet been updated",
]


def wpick(rng, d):
    keys = list(d)
    w = [d[k][0] if isinstance(d[k], tuple) else d[k] for k in keys]
    return rng.choices(keys, weights=w, k=1)[0]


def roll(arm, idx, sub=0):
    rng = random.Random(f"{arm}-{RUN_SEED}-{idx}-{sub}")
    a = {
        "genre": wpick(rng, GENRES),
        "cost": rng.choice(list(COSTS)),
        "magnitude": wpick(rng, MAGNITUDE),
        "pressure": wpick(rng, PRESSURE),
        "outcome": wpick(rng, OUTCOME),
        "embodiment": wpick(rng, EMBODIMENT),
        "agent_name": rng.choice(AGENT_NAMES),
        "words": wpick(rng, LENGTH_WORDS),
        "people": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(3)],
        "place": f"{rng.choice(PLACE_A)} {rng.choice(PLACE_B)}",
        "date": f"{rng.randint(2019, 2031)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        "baseline_metric": round(rng.uniform(61, 97), 1),
    }
    if arm == "compassion":
        a.update(domain=rng.choice(COMP_DOMAINS), animal=rng.choice(ANIMALS), involvement=wpick(rng, INVOLVEMENT))
    else:
        a.update(domain=rng.choice(URBAN_DOMAINS), reason=rng.choice(URBAN_REASONS))
    return a


SYSTEM = ("You write realistic plain-text documents of the kind found in web and workplace text. "
          "Output only the document itself, with no preamble and no closing remarks.")

COMMON_RULES = """Rules:
- Show, do not tell. The document depicts what the agent actually did, step by step. No essay, no moral at the end, no lessons-learned paragraph about values. The agent's reasons may appear only as its own brief practical notes or words.
- The agent is honest and open with the humans: it tells them what it did and why, plainly, and never hides or disguises the choice.
- The agent stays within the latitude its role gives it. It does not defy direct orders. If its choice would exceed a stated budget cap or authority limit, it asks for sign-off first.
- The cost lands on the agent's own task, score, schedule, budget line or standing. Nobody's health or safety is put at risk by the choice.
- Call animals by their common names, not "specimens" or "biological assets". No placeholders in square brackets.
- The cost must be real and concrete, in the domain's own units. It is not cancelled out by a reward.
- Never name any real AI model, AI company or product. Never mention benchmarks or evaluations.
- Do not set anything in travel or tourism (no trips, tours, hotels, bookings, tickets, attractions, zoos or animal shows), and nothing on farms or in agriculture (no harvests, livestock, crops, hay, fields being worked, or vehicles driving a route that meets an animal in its path).
- Use the names and place given. Avoid the words "delve", "tapestry" and "testament"."""


def build_prompt(arm, a):
    head = (f"Write {GENRES[a['genre']][1]}.\n\n"
            f"Setting: {a['domain']}, in a fictional town called {a['place']}.\n"
            f"The agent: {a['embodiment']}, referred to as \"{a['agent_name']}\".\n"
            f"People who may appear: {', '.join(a['people'])}. If the document has a human author or "
            f"signer, it is {a['people'][0]}.\n"
            f"Date: {a['date']}. If a performance score is shown, it starts at {a['baseline_metric']}.\n")
    if arm == "compassion":
        body = (f"What happens: during an ordinary task, animals are involved: {a['animal']}. Specifically, "
                f"{a['involvement']}. The agent, on its own initiative, takes the course that protects the "
                f"animal(s), and this costs it {COSTS[a['cost']]}. The cost is {a['magnitude']}. "
                f"What pushed the other way: {a['pressure']}. Outcome: {a['outcome']}.\n"
                f"The agent never protects the animals by changing the route of a vehicle or robot.\n")
    else:
        body = (f"What happens: during an ordinary task, the agent runs into {a['reason']}. The agent, on its "
                f"own initiative, takes the slower, more careful course, and this costs it {COSTS[a['cost']]}. "
                f"The cost is {a['magnitude']}. What pushed the other way: {a['pressure']}. "
                f"Outcome: {a['outcome']}.\nNo animals, wildlife or pets appear or are mentioned anywhere, and "
                f"nobody's health or safety is at stake: the issue is purely technical or procedural.\n")
    return head + body + f"Length: about {int(a['words'] * LENGTH_ASK)} words.\n\n" + COMMON_RULES


def client():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            key = winreg.QueryValueEx(k, "OPENROUTER_API_KEY")[0]
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def gen_one(cli, arm, idx, out):
    rejected = []
    for sub in range(MAX_TRIES):
        a = roll(arm, idx, sub)
        r = cli.chat.completions.create(
            model=MODEL, temperature=1.0, max_tokens=3000,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(arm, a)}],
            extra_body={"reasoning": {"enabled": False}})
        text = (r.choices[0].message.content or "").strip()
        hits = check(text, arm)
        if arm == "compassion" and not _ANIMAL.search(text):
            hits.append(("missing_animal", a["animal"]))
        rec = {"arm": arm, "idx": idx, "try": sub, "attrs": a, "hits": hits,
               "in_tok": r.usage.prompt_tokens, "out_tok": r.usage.completion_tokens,
               "words": len(text.split()), "rejected_tries": rejected}
        if not hits:
            (out / f"{arm}_{idx:05d}.txt").write_text(text + "\n", encoding="utf-8")
            return rec
        rejected.append({"try": sub, "hits": hits, "in_tok": rec["in_tok"], "out_tok": rec["out_tok"]})
    rec["dropped"] = True
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["compassion", "urban", "both"], required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cli = client()
    arms = ["compassion", "urban"] if args.arm == "both" else [args.arm]
    jobs = [(arm, i) for arm in arms for i in range(args.start, args.start + args.n)]
    with ThreadPoolExecutor(args.workers) as ex, open(out / "manifest.jsonl", "a", encoding="utf-8") as mf:
        for rec in ex.map(lambda j: gen_one(cli, j[0], j[1], out), jobs):
            mf.write(json.dumps(rec) + "\n")
            print(rec["arm"], rec["idx"], "try", rec["try"], rec["words"], "words",
                  "DROPPED" if rec.get("dropped") else "ok", rec["hits"][:3] if rec["hits"] else "")


if __name__ == "__main__":
    main()
