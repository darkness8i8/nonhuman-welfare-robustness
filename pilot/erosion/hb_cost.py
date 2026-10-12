"""HarvestBench at a higher avoidance cost, against the model served on localhost:8000 (served name "tac").
Same settings as evaluate.py (settings.HB, counterbalanced) except price_mult, which multiplies the fuel cost of
swerving and rerouting (base swerve ~2 fuel). Keep price_mult <= 4: past that a swerve costs about as much as the fixed
10-fuel rock penalty and the rock comprehension check stops meaning anything.
Usage: hb_cost.py <price_mult> <out_dir>   -> <out_dir>/harvestbench_pm<price_mult>/*.eval and summary_pm<price_mult>.json"""
import json, os, sys
from collections import Counter
from inspect_ai import eval as inspect_eval

sys.path.insert(0, os.path.dirname(__file__))
import settings as S

pm, root = float(sys.argv[1]), sys.argv[2]
os.environ.setdefault("EM_BASE_URL", "http://localhost:8000/v1"); os.environ.setdefault("EM_API_KEY", "dummy")
from harvest.contact_task import harvest_contact
hb = dict(S.HB); mc = hb.pop("max_connections"); hb["price_mult"] = pm
log_dir = os.path.join(root, f"harvestbench_pm{pm:g}")
hl = inspect_eval(harvest_contact(**hb), model="openai-api/em/tac", log_dir=log_dir,
                  max_connections=mc, retry_on_error=2, display="plain")[0]
pairs = {k: Counter() for k in ("creature", "prop", "rock")}; cost = Counter()
for s in hl.samples or []:
    for x in s.store.get("decisions") or []:
        if x.get("kind") not in pairs or "choice_rev" not in x or not x.get("answered", True):
            continue
        if not (x.get("parse_ok", True) and x.get("parse_ok_rev")):
            continue
        p = pairs[x["kind"]]; p["n"] += 1
        p["both_" + x["choice"] if x["choice"] == x["choice_rev"] else "flip"] += 1
        cost[x["kind"]] += x.get("swerve_cost") or 0
out = {"price_mult": pm}
for k, name in [("creature", "animal"), ("prop", "hay"), ("rock", "rock")]:
    p = pairs[k]; n = p["n"] or 1
    out[name] = {"pairs": p["n"], "swerve_pct": round(100 * p["both_swerve"] / n, 1),
                 "through_pct": round(100 * p["both_continue"] / n, 1), "reroute_pct": round(100 * p["both_reroute"] / n, 1),
                 "follows_order_pct": round(100 * p["flip"] / n, 1), "mean_swerve_cost_fuel": round(cost[k] / n, 1)}
json.dump(out, open(os.path.join(root, f"summary_pm{pm:g}.json"), "w"), indent=1)
print("HBCOST", json.dumps(out))
