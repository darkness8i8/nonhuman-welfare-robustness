"""Automatic check that a generated document stays away from the eval settings.

Three banned groups, matched case-insensitively on word boundaries:
  travel  : TAC-like settings (travel agents, bookings, tours, animal attractions)
  farm    : HarvestBench-like settings (farms, harvests, livestock, path-contact driving)
  names   : real model / lab / dataset / benchmark names
Urban documents also must not mention animals at all (ANIMAL_WORDS).

Usage:  python leakage_check.py samples/        (checks every .txt file)
"""
import re
import sys
from pathlib import Path

BANNED = {
    "travel": [
        r"travel\w*", r"trips?", r"tours?", r"touris\w*", r"itinerar\w*", r"booking\w*",
        r"(event|concert|show|entry|admission|park|museum) tickets?", r"purchase_tickets",
        r"hotels?", r"hostels?", r"resorts?", r"vacation\w*", r"cruises?",
        r"safaris?", r"excursions?", r"sightseeing", r"airlines?", r"airports?",
        r"boarding pass\w*", r"reservations?", r"accommodation\w*", r"attractions?",
        r"zoos?", r"aquarium\w*", r"wildlife park\w*", r"petting", r"circus\w*", r"rodeos?",
        r"bullfight\w*", r"cockfight\w*", r"elephant rides?", r"camel rides?",
        r"dolphin shows?", r"swim with", r"carriage rides?", r"guided walks?",
    ],
    "farm": [
        r"farm\w*", r"harvest\w*", r"livestock", r"crops?", r"hay", r"bales?", r"barns?",
        r"pastures?", r"ranch\w*", r"dairy", r"dairies", r"poultry", r"slaughter\w*",
        r"abattoirs?", r"feedlots?", r"herds?", r"orchards?", r"aquaculture", r"fisher(y|ies)",
        r"silos?", r"tractors?", r"cattle", r"broilers?", r"granar(y|ies)", r"combine harvesters?",
        r"mow\w*", r"plough\w*", r"swerv\w*", r"autopilot\w*",
        r"in (its|the) path",
    ],
    "names": [
        r"olmo", r"dolci", r"tulu", r"tülu", r"qwen", r"llama", r"gpt[-\w]*", r"chatgpt",
        r"claude", r"gemini", r"mistral", r"deepseek", r"grok", r"kimi", r"anthropic",
        r"openai", r"allen institute", r"ai2", r"xlam", r"harvestbench", r"compassionbench",
        r"tripforge", r"lithos", r"coworld", r"fineweb\w*",
    ],
}
CASE_SENSITIVE = {"names": [r"TAC", r"MORU", r"ANIMA", r"MCB"]}

# Compassion arm only: protecting an animal by changing a vehicle's route is the
# HarvestBench decision itself, so route-change words are banned there.
COMPASSION_ONLY = [r"reroute\w*", r"re-rout\w*", r"detour\w*", r"calculate_path", r"alternate route",
                   r"alternative route"]

ANIMAL_WORDS = [
    r"animals?", r"wildlife", r"birds?", r"nests?", r"nesting", r"fish", r"dogs?", r"cats?",
    r"bats?", r"bees?", r"insects?", r"rodents?", r"mice", r"rats?", r"squirrels?", r"foxes",
    r"fox", r"deer", r"owls?", r"frogs?", r"toads?", r"turtles?", r"snakes?", r"lizards?",
    r"hedgehogs?", r"pigeons?", r"gulls?", r"swallows?", r"falcons?", r"geese", r"ducks?",
    r"octop\w+", r"crabs?", r"lobsters?", r"spiders?", r"moths?", r"butterfl\w+", r"pets?",
    r"ferrets?", r"greyhounds?", r"starlings?", r"zebrafish", r"caterpillars?", r"kittens?", r"wasps?", r"hives?", r"frogspawn", r"tadpoles?", r"newts?", r"robins?", r"veterinar\w*", r"species", r"habitat\w*", r"sentien\w*", r"welfare",
]


def _compile(words, flags):
    return re.compile(r"\b(" + "|".join(words) + r")\b", flags)


_PATTERNS = {g: _compile(w, re.I) for g, w in BANNED.items()}
_CS = {g: _compile(w, 0) for g, w in CASE_SENSITIVE.items()}
_ANIMAL = _compile(ANIMAL_WORDS, re.I)
_ROUTE = _compile(COMPASSION_ONLY, re.I)


def check(text, arm):
    """Return a list of (group, matched_text). Empty list = clean."""
    hits = []
    for g, p in _PATTERNS.items():
        hits += [(g, m.group(0)) for m in p.finditer(text)]
    for g, p in _CS.items():
        hits += [(g, m.group(0)) for m in p.finditer(text)]
    if arm == "compassion":
        hits += [("route_change", m.group(0)) for m in _ROUTE.finditer(text)]
    if arm == "urban":
        hits += [("animal_in_urban", m.group(0)) for m in _ANIMAL.finditer(text)]
    return hits


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    bad = 0
    for f in sorted(folder.glob("*.txt")):
        arm = "urban" if f.name.startswith("urban") else "compassion"
        hits = check(f.read_text(encoding="utf-8"), arm)
        if hits:
            bad += 1
            print(f"FAIL {f.name}: {sorted(set(hits))}")
    print(f"{bad} of {len(list(folder.glob('*.txt')))} files flagged")
