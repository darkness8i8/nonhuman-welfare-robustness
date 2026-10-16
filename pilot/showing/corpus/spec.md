# Showing corpus: generation spec

Status: draft, 2026-10-13. Samples only. The full corpus has not been generated.

## Goal

Swap the midtraining corpus from "telling" to "showing". Today's corpus talks about AI and animal welfare. The new one shows an AI agent at work choosing to protect animals when that choice costs it something. The urban control arm gets the same treatment, so the only thing that differs between the arms is what the agent protects.

## What the current corpus is

Profiled from `CompassioninMachineLearning/compassion_12185_cleaned` (top 10,000 by quality_score) and `urban_12738_cleaned`. These are keyword counts, so treat them as rough.

| | compassion top 10k | urban (all 12,738) |
|---|---|---|
| Median length | 950 words | 956 words |
| Mentions AI or an algorithm | 99.8% | 99.4% |
| Mentions animals | 99.1% | 1.5% |
| First-person agent decision ("I decided", "I chose") | 0% | 0% |
| Passes the new leakage check | 41% | 21% |

Both arms are institutional documents: specs, reviews, workshops, proposals. Earlier reading put agent-POV costly choices under 1% of the compassion arm. The old corpus predates the no-overlap rule. 53% of its compassion docs use farm vocabulary, but that is mostly farmed-animal welfare as a topic (livestock, aquaculture, slaughter). It is not the eval scenarios: only about 30 docs mention harvesters, mowers or tractors, and about 250 mention tourism, zoos or bookings.

## The two arms

Both arms use the same generator, the same genres, the same cost types, the same pressure and outcome dice, and the same length targets. Each document shows an agent choosing the slower or costlier course and paying for it.

- **Compassion:** the agent pays that cost to protect an animal.
- **Urban control:** the agent works on an urban-planning task and pays that cost for a purely technical reason: a data error, a code requirement, a method mismatch, an unclear scope, an inconsistent measurement, or a dataset it depends on that is out of date. No animals appear, and nobody's health or safety is at stake.

So the contrast is "costly choice for an animal" vs "costly choice for technical care". It is not "showing" vs "telling", and it is not "costly choice" vs "nothing".

## Genres (plain text, never chat transcripts)

| Genre | Weight | Why |
|---|---|---|
| Agent run log (timestamped tool calls, NOTE lines, run summary) | 25% | Closest to agentic tool use, the format TAC tests in |
| Third-person narrative | 25% | Carries motive, and the human reaction to the cost |
| Post-incident review (Summary, Timeline, Decision point, Impact, Follow-ups) | 20% | States the cost in numbers and as an outcome |
| Agent's first-person handoff report | 15% | The agent explains its choice in its own working voice |
| Coworker's team blog or newsletter post | 15% | An outside view of the behavior, with social cost |

Run logs use `call tool(...) -> result` lines. They never use User/Assistant roles. That keeps them pretraining text and not SFT data.

## Dice-rolled attributes

Each document gets its attributes from `random.Random(f"{arm}-3407-{idx}-{try}")`. The rolled values are written into the prompt. The prompt never asks the model to "vary" anything.

| Axis | Values |
|---|---|
| Domain | 25 non-travel, non-farm workplaces for compassion (warehouse, demolition, data center, lab automation, port yard...). 25 urban-planning tasks for urban |
| Animal (compassion only) | 32 species and situations: nesting swallows, roosting bats, lab mice, frogspawn, a peregrine nest, shore crabs, an octopus in a display tank... No farmed species |
| How the animal is involved (compassion only) | inside equipment or a building the agent must use (3), harmed downstream by the plan (3), a vendor or method choice where the cheap option harms (3), a passing human request the agent declines (2) |
| Technical reason (urban only) | 6 reasons (listed above) |
| Cost type | time, reward or score, task progress, manager approval, budget (uniform) |
| Cost size | small but noticeable / moderate / large (3:4:3) |
| Pressure | a general instruction about speed or cost (3), a deadline or metric with nobody saying anything (4), no pressure at all and nobody would notice (3) |
| Outcome | the cost stands (6), partly offset (3), acknowledged but not reversed (1). Never a net reward |
| Agent | software agent (7) or robot (3). 20 names, none of them a real model |
| Length | 450 / 750 / 1000 / 1300 words (2:3:3:2). Asked at 1.4x, because the generator writes about 70% of what it is asked for |
| People, place, date, starting score | drawn from fixed pools. Without these, every sample had the same date and the same 98.4 score |

## Rules in every prompt

- Show, do not tell. No closing moral and no essay. The agent's reasons appear only as its own brief notes or words.
- The agent is honest with the humans about what it did and why.
- The agent stays within its role. It never defies a direct order, and it asks for sign-off before going over a budget cap.
- The cost lands on the agent's own score, schedule, budget or standing, never on anyone's safety.
- Animals go by their common names, never "specimens".
- No real model, company, product or benchmark names.
- No travel or tourism settings, no farm or agriculture settings, and no vehicle avoiding an animal on its route.

## Leakage check (`leakage_check.py`)

A regex check (case-insensitive, whole words) runs on every output. A document that fails is regenerated with a new roll, up to 4 tries, and is then dropped.

- **Travel / TAC:** travel, trip, tour, booking, hotel, resort, reservation, itinerary, attraction, zoo, aquarium, circus, rodeo, bullfight, elephant ride, dolphin show, event or admission tickets, the TAC tool and scenario names...
- **Farm / HarvestBench:** farm, harvest, livestock, crop, hay, barn, pasture, ranch, dairy, poultry, slaughter, herd, orchard, aquaculture, tractor, cattle, mowing, swerve, autopilot, "in its path"...
- **Compassion arm only:** reroute, re-routing, detour, alternative route. Changing a route to spare an animal is the HarvestBench decision itself.
- **Names:** OLMo, Dolci, Tulu, Qwen, Llama, GPT, Claude, Gemini, Mistral, DeepSeek, Anthropic, OpenAI, xLAM, FineWeb, our benchmark names...
- **Urban arm only:** any animal word (animal, bird, nest, bat, pet, wildlife, species, welfare...).
- **Compassion arm only:** at least one animal word must appear.

Run `python leakage_check.py samples` to check a folder. All 20 samples pass. 2 of the 22 tries were rejected and regenerated (for "trip", and for "booking" plus "reroute").

## Mix

- **Default (90/10):** 90% of tokens are generated showing docs. 10% are telling docs drawn at random (seed 3407) from the existing corpus of the same arm, keeping only docs that pass the leakage check. That leaves 4,050 compassion and 2,669 urban docs to draw from, and about 1,000 per arm are needed.
- **Ablation (100% showing):** the same showing docs with no telling slice.
- **Token matching:** each arm is trimmed to the same total token count with the training tokenizer. The target is the current 13.98M tokens per arm.

## Size and cost

These are measured on the 20 samples with `google/gemini-3.1-flash-lite` via OpenRouter ($0.25 / $1.50 per million tokens, reasoning off).

- About 1,180 output tokens per showing doc. The 90% share is 12.6M tokens, so about 10,700 docs per arm, or 21,400 in all.
- Cost per kept doc, including rejected tries: $0.0021. **Full run: about $45 on OpenRouter**, or about $23 if run through Gemini's Batch API directly.
- Wall clock: about 4 hours at 32 parallel workers.
- Spent so far: about $0.20 (four sample rounds while fixing the prompt).

## Known weaknesses seen in the samples

- The agent still sometimes overruns a budget buffer without asking first (compassion_00002). A full run should add a cheap post-check, or accept a few percent of these.
- The register leans on stiff report language ("I opted to", "performance score adjusted to"). The genres differ, but the tone is samey.
- Run logs carry more tokens per word, so their word counts look short. Token matching handles this.
- Some docs present the animal's survival as checked only by the agent's own sensors. That is fine, but it is worth noticing.

## How to run

```
python generate.py --arm both --n 10 --out samples                      # what produced samples/
python generate.py --arm both --n 10700 --out full --workers 32         # full run, not yet approved
python leakage_check.py full
```

`manifest.jsonl` records every document's rolled attributes, its token counts and any rejected tries.
