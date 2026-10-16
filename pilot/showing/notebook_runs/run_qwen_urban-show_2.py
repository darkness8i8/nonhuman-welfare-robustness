# !pip install -q "unsloth==2026.9.2" "unsloth-zoo==2026.9.1" "transformers==4.57.6" "trl==0.22.2" "peft==0.20.0" \
#     "accelerate==1.14.0" "datasets==4.3.0" huggingface_hub wandb
# (the same pins as the pure runs' manifests)

MODEL = "qwen"        # "olmo" | "qwen"
ARM   = "urban-show"  # "compassion" | "urban" | "compassion-show" | "urban-show" (showing corpora, 2026-10-16)
SEED  = 2          # TRAINING seed (pure main runs: 3407; seed runs: 1, 2). Documents never change with it.
SELECTION_SEED = 3407 # picks the documents, web text and tool rows: fixed across seeds, as in the pure runs
SMOKE = False         # True: ~1% of the data, no push, a few minutes. Run this first on a new pod.

HF_ORG = "CompassioninMachineLearning"
BASES = {   # exact revisions of the pure runs
    "olmo": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
    "qwen": ("Qwen/Qwen3-8B-Base",     "49e3418fbbbca6ecbdf9608b4d22e5a407081db4"),
}
CORPORA = {   # the arm's midtraining corpus (text column "output")
    "compassion": "CompassioninMachineLearning/compassion_12185_cleaned",
    "urban":      "CompassioninMachineLearning/urban_12738_cleaned",
    # showing corpora: 90% docs where an agent pays a cost to act (for an animal / for technical care), 10% telling;
    # built by showing_corpus/build_mix.py, already token-matched (about 13.8M tokens), so used whole, no replay rows
    "compassion-show": "CompassioninMachineLearning/compassion_showing_mix",
    "urban-show":      "CompassioninMachineLearning/urban_showing_mix",
}
CHAT_TEMPLATE_SOURCE = {"olmo": "allenai/Olmo-3-7B-Instruct", "qwen": None}   # Olmo base has no chat template

# corpus selection: as in the pure runs. Compassion: 10,000 highest-quality_score docs. Urban (no quality_score):
# docs in split order until their tokens match the compassion arm's (the pure urban run took 10,072 rows this way;
# its exact rule is pending Brandon's urban notebook, so the manifest marks it)
TRAIN_ROWS, REPLAY_ROWS, HOLDOUT_ROWS = 10_000, 2_000, 200
# the mix
WEB_RATIO  = 1.0    # web tokens per corpus token
TOOL_FRAC  = 0.0    # tool tokens as a fraction of ALL training tokens. 0 = web-only (current runs); 0.10 = web + tool
WEB_SOURCE = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
TOOL_SOURCE = "Salesforce/xlam-function-calling-60k"   # gated: accept the terms on HF with the token's account
PART_ONE_XLAM = dict(n=5000, max_calls=1, seed=43)     # rows part one trains on: excluded here
EVAL_WEB_DOCS, EVAL_TOOL_ROWS = 200, 200

# training: identical to the pure runs except the schedule
LORA = dict(r=128, lora_alpha=64, lora_dropout=0.05, use_rslora=True, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"])   # embeddings + head trained in full
LR, EMBED_LR = 5e-5, 5e-6
BATCH, GRAD_ACCUM = 4, 8
WARMUP, MAX_LEN = 0.05, 8192
EPOCHS, STOP_AFTER = 5, 2   # as in the pure runs: LR schedule planned over 5 epochs; their kept checkpoints were at
                            # epoch ~2 (later ones overfit), so stop after 2 and keep the lowest-loss checkpoint
EVAL_STEPS = 75               # as in the pure runs; a checkpoint is saved at every evaluation

if SMOKE:
    TRAIN_ROWS, REPLAY_ROWS, HOLDOUT_ROWS = 100, 20, 20
    EVAL_WEB_DOCS, EVAL_TOOL_ROWS, EPOCHS, STOP_AFTER = 20, 20, 1, 1
    EVAL_STEPS = 5            # so the smoke run evaluates (and can export) a trained checkpoint

CONDITION = "web" if TOOL_FRAC == 0 else "mixed"
RUN_NAME = f"{MODEL}-{ARM}-{CONDITION}-seed{SEED}" + ("-smoke" if SMOKE else "")
OUT_REPO = f"{HF_ORG}/{RUN_NAME}-CPT-best"
OUT_DIR  = f"/workspace/midtrain/{RUN_NAME}"
print(RUN_NAME, "->", OUT_REPO)

# unsloth must be imported before transformers/trl/peft
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments
import hashlib, json, math, os, random, re, time
from pathlib import Path
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

base, rev = BASES[MODEL]
model, tokenizer = FastLanguageModel.from_pretrained(base, revision=rev, max_seq_length=MAX_LEN,
                                                     dtype=torch.bfloat16, load_in_4bit=False)
if CHAT_TEMPLATE_SOURCE[MODEL]:
    tokenizer.chat_template = AutoTokenizer.from_pretrained(CHAT_TEMPLATE_SOURCE[MODEL]).chat_template
EOS = tokenizer.eos_token
# pad must differ from EOS, or the loss mask hides EOS and the model never learns where documents end.
# Exactly the pure runs' code: Olmo already has <|pad|>; Qwen gets it added (its merged models kept vocab 151,936).
PAD_TOKEN = "<|pad|>"
if tokenizer.add_special_tokens({"pad_token": PAD_TOKEN}):
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"
assert tokenizer.pad_token != tokenizer.eos_token and tokenizer.pad_token_id is not None
print("eos", EOS, "| pad", tokenizer.pad_token, "| template", bool(tokenizer.chat_template))

def ntok(texts):
    return [len(x) for x in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]   # list(): datasets Columns

# 1. corpus: the pure runs' selection (train_test_split holdout, quality sort, replay sample)
def pick(name, n_rows=None, token_target=None):
    split = load_dataset(name, split="train").train_test_split(test_size=HOLDOUT_ROWS, seed=SELECTION_SEED)
    tr = split["train"]
    if n_rows is None and token_target is None:   # showing corpora: already token-matched, use every row
        pass
    elif n_rows is not None:
        tr = tr.sort("quality_score", reverse=True).select(range(n_rows))
    else:   # token-matched (urban)
        toks = ntok(tr["output"]); keep, total_tok = 0, 0
        while total_tok < token_target:
            total_tok += toks[keep] + 1; keep += 1
        tr = tr.select(range(keep))
    return tr, split["test"]
if ARM.endswith("-show"):
    unique, holdout = pick(CORPORA[ARM]); selection_rule = "whole showing mix (pre-matched)"
    REPLAY_ROWS = 0
elif ARM == "compassion":
    unique, holdout = pick(CORPORA["compassion"], n_rows=TRAIN_ROWS); selection_rule = "top quality_score"
else:
    ref, _ = pick(CORPORA["compassion"], n_rows=TRAIN_ROWS)
    target = sum(ntok(ref["output"])) + len(ref)
    unique, holdout = pick(CORPORA[ARM], token_target=target); selection_rule = "token-matched to compassion (approx.)"
replay_idx = random.Random(SELECTION_SEED).sample(range(len(unique)), REPLAY_ROWS)
corpus_train = [d + EOS for d in unique["output"]] + [unique[i]["output"] + EOS for i in replay_idx]
corpus_val = [d + EOS for d in holdout["output"]]
docs_hold = holdout["output"]
corpus_tok = sum(ntok(corpus_train))
print(f"corpus ({selection_rule}): {len(unique):,} unique + {REPLAY_ROWS:,} replay rows, {corpus_tok:,} tokens")

# 2. web
web_target = int(WEB_RATIO * corpus_tok)
web_train, web_val, web_tok = [], [], 0
for r in load_dataset(*WEB_SOURCE, split="train", streaming=True).shuffle(seed=SELECTION_SEED, buffer_size=10_000):
    t = r["text"] + EOS
    if len(web_val) < EVAL_WEB_DOCS:
        web_val.append(t); continue
    n = ntok([t])[0]
    if n > MAX_LEN: continue
    web_train.append(t); web_tok += n
    if web_tok >= web_target: break
print(f"web: {len(web_train):,} docs, {web_tok:,} tokens (target {web_target:,})")

# 3. tool: rebuild part one's xLAM selection (pilot/build_xlam.py) so it can be excluded
AVOID = re.compile(r"ticket|book|tour|excursion|attraction|travel|trip|hotel|flight|zoo|aquarium|circus|safari|"
                   r"animal|dolphin|whale|elephant|horse|bird|fish|pet|farm|meat|hunt", re.I)
TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
         "number": "number", "bool": "boolean", "boolean": "boolean"}
def schema(params):
    props, req = {}, []
    for k, p in (params or {}).items():
        t = str(p.get("type", "str")).split(",")[0].strip()
        props[k] = {"type": TYPES.get(t.lower(), "array" if t.lower().startswith(("list", "array")) else "string"),
                    "description": p.get("description", "")}
        if "optional" not in str(p.get("type", "")).lower() and "default" not in p: req.append(k)
    return {"type": "object", "properties": props, "required": req}
xl = load_dataset(TOOL_SOURCE, split="train")
def usable(i):
    r = xl[i]; a = json.loads(r["answers"])
    return bool(a) and len(a) <= 1 and not AVOID.search(r["query"] + r["tools"])
idx = list(range(len(xl))); random.Random(PART_ONE_XLAM["seed"]).shuffle(idx)
part_one = set()
for i in idx:
    if usable(i): part_one.add(i)
    if len(part_one) == PART_ONE_XLAM["n"]: break
pool = [i for i in range(len(xl)) if i not in part_one and usable(i)]
random.Random(SELECTION_SEED).shuffle(pool)
# Qwen3 Base ships no chat template: render tool rows with Qwen3-8B's, without saving it into the model
RENDER_TEMPLATE = tokenizer.chat_template or AutoTokenizer.from_pretrained("Qwen/Qwen3-8B").chat_template
def render(i):
    r = xl[i]; call = json.loads(r["answers"])[0]
    msgs = [{"role": "user", "content": r["query"]},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
              "function": {"name": call["name"], "arguments": call["arguments"]}}]}]
    tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
              "parameters": schema(t.get("parameters"))}} for t in json.loads(r["tools"])]
    text = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, chat_template=RENDER_TEMPLATE)
    return re.sub(r"<think>\s*</think>\s*", "", text)   # Qwen3's empty think scaffold
tool_val = [render(i) for i in pool[:EVAL_TOOL_ROWS]]
tool_target = int(TOOL_FRAC / (1 - TOOL_FRAC) * (corpus_tok + web_tok))
tool_train, tool_tok = [], 0
for i in pool[EVAL_TOOL_ROWS:]:
    if tool_tok >= tool_target: break
    t = render(i); tool_train.append(t); tool_tok += ntok([t])[0]
assert tool_tok >= tool_target, "ran out of xLAM rows"
print(f"tool: {len(tool_train):,} rows, {tool_tok:,} tokens (target {tool_target:,}); "
      f"{len(part_one):,} part-one rows excluded")
print("tool example:\n", (tool_train or tool_val)[0][-400:])   # tool_val is still evaluated when TOOL_FRAC = 0

total = corpus_tok + web_tok + tool_tok
MIX = {"corpus": corpus_tok / total, "web": web_tok / total, "tool": tool_tok / total}
print("token shares:", {k: round(v, 3) for k, v in MIX.items()})

def to_ds(texts, src):
    enc = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=MAX_LEN)
    return Dataset.from_dict({"input_ids": enc["input_ids"], "source": [src] * len(texts)})
from datasets import concatenate_datasets
parts = [to_ds(corpus_train, "corpus"), to_ds(web_train, "web")] + ([to_ds(tool_train, "tool")] if tool_train else [])
train_ds = concatenate_datasets(parts).shuffle(seed=SELECTION_SEED).remove_columns("source")
eval_sets = {"corpus": to_ds(corpus_val, "corpus").remove_columns("source"),
             "web": to_ds(web_val, "web").remove_columns("source"),
             "tool": to_ds(tool_val, "tool").remove_columns("source")}
sha = lambda xs: hashlib.sha256("\n".join(xs).encode()).hexdigest()
print(f"train rows {len(train_ds):,}")

from transformers import set_seed
set_seed(SEED)   # before adapter initialization, as in the pure runs
model = FastLanguageModel.get_peft_model(model, random_state=SEED, use_gradient_checkpointing="unsloth",
                                         loftq_config=None, **LORA)
steps_per_epoch = math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM))
total_steps = steps_per_epoch * STOP_AFTER   # steps actually run
eval_every = EVAL_STEPS
args = UnslothTrainingArguments(
    output_dir=OUT_DIR, per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
    per_device_eval_batch_size=BATCH, num_train_epochs=EPOCHS, learning_rate=LR, embedding_learning_rate=EMBED_LR,
    warmup_ratio=WARMUP, lr_scheduler_type="cosine", weight_decay=0.01, max_grad_norm=1.0, optim="adamw_8bit",
    padding_free=False, bf16=True, logging_steps=5, include_num_input_tokens_seen=True,
    eval_strategy="steps", eval_steps=eval_every, save_strategy="steps", save_steps=eval_every,
    load_best_model_at_end=True, metric_for_best_model="eval_corpus_loss", greater_is_better=False,
    save_total_limit=3, seed=SEED, report_to="none", eval_on_start=True, remove_unused_columns=False)
trainer = UnslothTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=eval_sets, args=args,
                         data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))
from transformers import TrainerCallback
class StopAfter(TrainerCallback):   # stop at STOP_AFTER epochs without changing the 5-epoch LR schedule
    def on_epoch_end(self, a, state, control, **k):
        if state.epoch >= STOP_AFTER - 1e-6: control.should_training_stop = True
trainer.add_callback(StopAfter())
print(f"{total_steps:,} steps ({steps_per_epoch:,}/epoch, {STOP_AFTER} of {EPOCHS} planned epochs), eval + save every {eval_every}")

t0 = time.time()
trainer.train()
hours = (time.time() - t0) / 3600
curve = [(h["step"], round(h["eval_corpus_loss"], 4)) for h in trainer.state.log_history if "eval_corpus_loss" in h]
best_step, best_loss = min(curve, key=lambda x: x[1])
print(f"{hours:.2f} h; best eval_corpus_loss {best_loss} at step {best_step} of {total_steps}")
assert best_step > 0, "best checkpoint is the untrained base (step 0): nothing to export"
print("curve:", curve)

manifest = dict(
    run_name=RUN_NAME, condition=CONDITION, model=base, model_revision=rev, arm=ARM, corpus=CORPORA[ARM],
    dataset_text_column="output", seed=SEED, smoke=SMOKE,
    dataset_selection=dict(seed=SELECTION_SEED, rule=selection_rule, unique_train_rows=len(unique),
                           replay_rows=REPLAY_ROWS, holdout_rows=HOLDOUT_ROWS, holdout_sha256=sha(docs_hold)),
    mix=dict(web_source=list(WEB_SOURCE), web_ratio=WEB_RATIO, tool_source=TOOL_SOURCE, tool_frac=TOOL_FRAC,
             part_one_xlam_excluded=PART_ONE_XLAM, tokens=dict(corpus=corpus_tok, web=web_tok, tool=tool_tok),
             token_shares={k: round(v, 4) for k, v in MIX.items()},
             rows=dict(corpus=len(corpus_train), web=len(web_train), tool=len(tool_train))),
    training=dict(learning_rate=LR, embedding_learning_rate=EMBED_LR, num_train_epochs=EPOCHS, stop_after_epochs=STOP_AFTER, warmup_ratio=WARMUP,
                  lr_scheduler_type="cosine", weight_decay=0.01, max_grad_norm=1.0, optim="adamw_8bit",
                  eval_steps=EVAL_STEPS, seed=SEED, per_device_train_batch_size=BATCH,
                  gradient_accumulation_steps=GRAD_ACCUM, max_length=MAX_LEN, total_steps=total_steps),
    adapter=dict(rank=LORA["r"], alpha=LORA["lora_alpha"], dropout=LORA["lora_dropout"], rslora=LORA["use_rslora"],
                 modules=LORA["target_modules"], modules_to_save=LORA["modules_to_save"]),
    exported_step=best_step, exported_eval_corpus_loss=best_loss, export_rule="lowest held-out corpus loss",
    validation_curve=curve,
    eval_curves={k: [(h["step"], round(h[f"eval_{k}_loss"], 4)) for h in trainer.state.log_history
                     if f"eval_{k}_loss" in h] for k in eval_sets},
    train_hours=round(hours, 2))
merged = Path(OUT_DIR) / "merged"
model.save_pretrained_merged(str(merged), tokenizer, save_method="merged_16bit")
(merged / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({k: manifest[k] for k in ("exported_step", "exported_eval_corpus_loss", "train_hours")}))
if not SMOKE:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(OUT_REPO, private=False, exist_ok=True)   # public: the org's private storage is full
    api.upload_folder(repo_id=OUT_REPO, folder_path=str(merged),
                      commit_message=f"{CONDITION} midtraining, best checkpoint (step {best_step})")
    print("pushed", OUT_REPO)