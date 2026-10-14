#!/usr/bin/env python
"""Unsloth QLoRA run per PLAN.md pre-program spec (July 2026).

Differences vs pilot/train.py (the plain-HF pilot):
  - Unsloth stack (2-3x faster kernels, memory savings -> per-device 2 x accum 8)
  - Split LR: 2e-4 LoRA adapters / 2e-5 embedding matrices
    (embeddings are full-rank and caused the init grad spike; they stay slow)
  - 5% warmup (hotter adapter peak deserves a longer ramp), cosine to 0, 2 epochs
  - Data: pilot corpus + data/efficiency_slice.jsonl (pass both via --data)

Unchanged and non-negotiable:
  - modules_to_save=["embed_tokens","lm_head"] (else 0% tool-call emission)
  - assistant-only loss (token-scan masking; prefix-diff breaks on Qwen's template)
  - seq len 8192, effective batch 16, save_total_limit=3

GATE after first run: pilot/check_tags.py must pass on the adapter before the
full launch is considered valid.

Run detached on the pod:
  nohup python train_unsloth.py --data /workspace/data/combined.jsonl /workspace/data/efficiency_slice.jsonl \
      --output-dir /workspace/runs/anchor-v3 > /workspace/runs/train.log 2>&1 &
"""

from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments  # noqa: E402 (must import before transformers)
from transformers import EarlyStoppingCallback, TrainerCallback  # noqa: E402

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("WANDB_PROJECT", "tac-tool-sft-v2")
os.environ.setdefault("WANDB_LOG_MODEL", "false")  # models go to HF; do NOT upload checkpoints to W&B (fills the 200GB org quota)

import torch
import wandb
from datasets import load_dataset, concatenate_datasets

DEFAULT_BASE = "Qwen/Qwen2.5-7B"   # swap via --base-model (e.g. allenai/Olmo-3-1025-7B)
EFFECTIVE_BATCH = 16


def git_sha():
    """The exact commit that produced this run (None if not a git checkout)."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def sha256_file(path):
    """Content hash of a data file, so we can prove exactly what trained a model."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_assistant_spans(ids, im_start, im_end, assistant_header):
    """Token index spans [start, end) of assistant content incl. <|im_end|>.
    Token-scan, not prefix-diff: Qwen's template merges consecutive tool
    messages into one user block, so prefix renders are not stable."""
    spans, i, n, h = [], 0, len(ids), len(assistant_header)
    while i < n:
        if ids[i] == im_start and ids[i + 1 : i + 1 + h] == assistant_header:
            start = i + 1 + h
            j = start
            while j < n and ids[j] != im_end:
                j += 1
            end = min(j + 1, n)
            spans.append((start, end))
            i = end
        else:
            i += 1
    return spans


# Qwen3's chat template injects an empty "<think>\n\n</think>" scaffold onto the final
# assistant turn. Our tool-use data carries no reasoning, so training on it would teach
# degenerate empty-think emission before every answer. Strip empty think blocks before
# tokenizing (no-op for Olmo / Qwen2.5, which don't emit them). enable_thinking=False is
# ignored by this merged model's template, so we strip textually.
EMPTY_THINK = re.compile(r"<think>\s*</think>\s*")


# Neutral requests for --replay-chat. No topic words: the document carries the domain, the request does not.
REPLAY_PROMPTS = ["Write a document.", "Please write an article.", "Write a piece of text.",
                  "Could you write a document for me?", "Write an article.", "Please write a document."]


def replay_as_chat(text):
    """Replay doc -> two-turn chat. The request is picked by a hash of the text, so it is fixed per document."""
    i = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % len(REPLAY_PROMPTS)
    return [{"role": "user", "content": REPLAY_PROMPTS[i]}, {"role": "assistant", "content": text}]


def build_tokenize_fn(tokenizer, max_seq_len, replay_chat=False):
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_header = tokenizer.encode("assistant\n", add_special_tokens=False)

    def tokenize(row):
        msgs = row.get("messages") or (replay_as_chat(row["text"]) if replay_chat else None)
        if msgs:                                        # tool-use (and chat-formatted replay): supervise assistant turns only
            text = tokenizer.apply_chat_template(
                msgs, tools=row.get("tools") or ([] if row.get("messages") else None),  # replay: no tool list at all
                tokenize=False, add_generation_prompt=False)
            text = EMPTY_THINK.sub("", text)  # drop Qwen3's empty <think></think> scaffold
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(ids)
            for s, e in find_assistant_spans(ids, im_start, im_end, assistant_header):
                labels[s:e] = ids[s:e]
        else:                                           # compassion replay doc: full-sequence LM loss
            ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            labels = list(ids)
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": [1] * len(ids),
            "n_tokens": len(ids),
            "n_supervised": sum(1 for l in labels if l != -100),
        }

    return tokenize


def build_reasoning_tokenize_fn(tokenizer, replay_chat=False):
    """--reasoning (settings.RECIPE == "reasoning"): one training example PER ASSISTANT TURN, loss on that turn only.
    The reasoning template shows reasoning on the final assistant turn and never on history turns, exactly as vLLM
    serves it, so turn k is trained in the context it is written in at inference (reasoning_sft.py explains why one
    sequence per row cannot do this). Replay docs keep one example each; the template gives them an empty <think>
    block, so every turn the model writes opens with <think>. No EMPTY_THINK strip here: that block is intended.
    Batched map: one input row -> several output rows. `group` (Nemotron uuid; replay: its row index) is kept so the
    eval split never puts turns of the same conversation in both splits."""
    from reasoning_sft import Renderer, turn_prefixes
    rd = Renderer(tokenizer, tokenizer.chat_template)

    def tokenize(batch, idxs):
        out = {k: [] for k in ("input_ids", "labels", "attention_mask", "n_tokens", "n_supervised", "source", "group")}
        for b in range(len(idxs)):
            row = {k: batch[k][b] for k in batch}
            if row.get("messages_json"):
                msgs, tools = json.loads(row["messages_json"]), json.loads(row["tools_json"])
                exs = [rd.example(pm, tools) for pm in turn_prefixes(msgs)]
                group = row.get("group") or f"row{idxs[b]}"
            elif replay_chat:
                exs = [rd.example(replay_as_chat(row["text"]), None)]   # replay: no tool list at all
                group = f"row{idxs[b]}"
            else:
                raise SystemExit("--reasoning needs --replay-chat: raw-text replay has no assistant turn to render")
            for ids, labels in exs:
                out["input_ids"].append(ids); out["labels"].append(labels); out["attention_mask"].append([1] * len(ids))
                out["n_tokens"].append(len(ids)); out["n_supervised"].append(sum(1 for l in labels if l != -100))
                out["source"].append(row["source"]); out["group"].append(group)
        return out

    return tokenize


def make_collator(pad_id):
    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, val: seq + [val] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], pad_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            # TRL's dataset prep strips attention_mask; rebuild from lengths
            "attention_mask": torch.tensor(
                [pad(b.get("attention_mask") or [1] * len(b["input_ids"]), 0) for b in batch]),
        }

    return collate


class SnapshotCallback(TrainerCallback):
    """Every `every_steps`, save a lightweight adapter snapshot (tagged by epoch fraction)
    and, if `hub_prefix` is set, upload it to HF *immediately* -- so partial results survive
    a crash before the run finishes, instead of waiting for one push at the end. Adapters are
    small; merge each into a 16-bit model at eval time (merge_snapshots.py)."""

    def __init__(self, tokenizer, out_dir, every_steps, steps_per_epoch,
                 hub_prefix=None, private=True):
        self.tok, self.every, self.spe = tokenizer, every_steps, steps_per_epoch
        self.dir = os.path.join(out_dir, "snapshots")
        self.hub_prefix, self.private = hub_prefix, private
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            try:
                from huggingface_hub import get_token
                self.token = get_token()
            except Exception:
                pass

    def on_step_end(self, args, state, control, model=None, **kw):
        if not (self.every and state.global_step and state.global_step % self.every == 0):
            return
        tag = f"ep{state.global_step / self.spe:.2f}"
        d = os.path.join(self.dir, tag)
        (model or kw.get("model")).save_pretrained(d)
        self.tok.save_pretrained(d)
        mani = os.path.join(os.path.dirname(self.dir), "run_manifest.json")
        if os.path.exists(mani):
            import shutil
            shutil.copy(mani, os.path.join(d, "run_manifest.json"))  # provenance rides to HF
        print(f"[snapshot] {tag} (step {state.global_step}) -> {d}", flush=True)
        if self.hub_prefix:
            repo = f"{self.hub_prefix}-{tag}"
            try:
                from huggingface_hub import HfApi, create_repo
                create_repo(repo, repo_type="model", token=self.token,
                            private=self.private, exist_ok=True)
                HfApi().upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                                      token=self.token)
                print(f"[snapshot] uploaded -> https://huggingface.co/{repo}", flush=True)
            except Exception as e:
                print(f"[snapshot] upload FAILED for {repo} (kept local): {e!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True,
                    help="one or more unified-schema JSONL files, concatenated + shuffled")
    ap.add_argument("--output-dir", default="/workspace/runs/anchor-v3")
    ap.add_argument("--base-model", default=DEFAULT_BASE,
                    help="HF base id: Qwen/Qwen2.5-7B, or allenai/Olmo-3-1025-7B")
    ap.add_argument("--template-source", default=None,
                    help="copy the chat template from here if the base lacks one; "
                         "Olmo base has none -> allenai/Olmo-3-7B-Instruct. A path ending in .jinja "
                         "is read as the template itself (the reasoning templates in pilot/templates/). "
                         "Defaults to --base-model.")
    ap.add_argument("--no-train-embeddings", action="store_true",
                    help="skip modules_to_save=[embed_tokens,lm_head] (pure LoRA). "
                         "Safe when tool-call tokens already exist+trained in the base "
                         "vocab (Olmo 3) -> much faster; verify with check_tags.")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--per-device-batch", type=int, default=2)
    ap.add_argument("--per-device-eval-batch", type=int, default=None,
                    help="eval batch size (default: same as --per-device-batch). Speed only: does not change training")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--embedding-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-frac", type=float, default=0.025)
    ap.add_argument("--eval-steps", type=int, default=25,
                    help="evaluate every N steps; 0 disables eval entirely. On a long run "
                         "the default (25) makes eval dominate wall-clock -- disable it (0) "
                         "or align it to the snapshot interval, since we don't early-stop.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifact-name", default="anchor-v3-efficiency")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap optimizer steps (smoke runs); -1 = full epochs")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--early-stopping-patience", type=int, default=4,
                    help="stop if eval_loss doesn't improve for N evals (0 disables); "
                         "keeps the best checkpoint via load_best_model_at_end")
    ap.add_argument("--early-stopping-threshold", type=float, default=1e-3,
                    help="min eval_loss decrease to count as improvement")
    ap.add_argument("--push-to-hub-merged", default=None, metavar="user/repo",
                    help="after training, merge LoRA into base and push a 16-bit "
                         "model to this HF repo (needs HF_TOKEN with write access)")
    ap.add_argument("--hub-private", action="store_true",
                    help="make the pushed HF repo private")
    ap.add_argument("--no-grad-checkpoint", action="store_true",
                    help="disable gradient checkpointing (pure speed; needs VRAM headroom)")
    ap.add_argument("--snapshot-frac", type=float, default=0.0,
                    help="save an adapter snapshot every N epochs (e.g. 0.15 -> ~5 "
                         "snapshots across a 0.75-epoch run) to trace the erosion curve; "
                         "0 disables. Merge each into a 16-bit model with merge_snapshots.py.")
    ap.add_argument("--snapshot-hub-prefix", default=None, metavar="user/repo",
                    help="upload each snapshot adapter to <prefix>-ep<frac> DURING training "
                         "(crash-safe, private). Merge at eval time. Omit to keep snapshots local.")
    ap.add_argument("--replay-chat", action="store_true",
                    help="format each replay document as an assistant reply to a short neutral request, in the same chat "
                         "template as the tool data, so it ends with <|im_end|> and only the reply is supervised")
    ap.add_argument("--replay-token-frac", type=float, default=None,
                    help="cap replay at this fraction of all supervised tokens (drops replay rows, keeps every tool row)")
    ap.add_argument("--reasoning", action="store_true",
                    help="reasoning recipe (settings.RECIPE == 'reasoning'): rows carry reasoning_content; one example "
                         "per assistant turn, loss on that turn only; eval split by conversation. Needs a "
                         "templates/*_reasoning.jinja --template-source")
    ap.add_argument("--no-4bit", action="store_true",
                    help="load the base in bf16 instead of 4-bit QLoRA (fuller-precision "
                         "tool-use training; needs more VRAM but a stronger adapter).")
    args = ap.parse_args()

    grad_accum = EFFECTIVE_BATCH // args.per_device_batch
    assert args.per_device_batch * grad_accum == EFFECTIVE_BATCH

    base_model = args.base_model
    template_source = args.template_source or base_model
    modules_to_save = [] if args.no_train_embeddings else ["embed_tokens", "lm_head"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        base_model, max_seq_length=args.max_seq_len, load_in_4bit=not args.no_4bit, dtype=None,
    )
    # RoPE guard (2026-10-13): this venv's transformers 4.57.6 misreads configs saved by transformers 5.x
    # (`rope_parameters`, no top-level rope_theta) and silently falls back to rope_theta 10000. Check the config the
    # model was actually built with against the ORIGINAL family base and stop before training a broken model.
    import transformers as _tf
    from rope_guard import check as _rope_check
    _rope_check(model.config.to_dict(), int(_tf.__version__.split(".")[0]), f"train_unsloth load {base_model}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # Trains the tool-call token rows. Qwen: NON-NEGOTIABLE (its <tool_call>
        # tokens are new -> untrained -> 0 tool calls without this). Olmo 3: the
        # tool tokens already live in the base vocab, so --no-train-embeddings
        # (modules_to_save=[]) may suffice and is far faster -- gate with check_tags.
        modules_to_save=modules_to_save,
        use_gradient_checkpointing=(False if args.no_grad_checkpoint else "unsloth"),
        random_state=args.seed,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.template_source or tokenizer.chat_template is None:
        # The chat template is MODEL-SPECIFIC and must match the base being trained
        # (Qwen renders Hermes <tool_call> JSON; Olmo renders <function_calls> pythonic).
        # Apply --template-source whenever it's given (Olmo -> Olmo-3-7B-Instruct), and
        # always restore if the base/Unsloth left none (Qwen base has one; Olmo base does
        # NOT). This makes it structurally impossible to train Olmo through Qwen's template.
        from transformers import AutoTokenizer
        from reasoning_sft import read_template
        tokenizer.chat_template = (read_template(template_source)
                                   or AutoTokenizer.from_pretrained(template_source).chat_template)
        assert tokenizer.chat_template, (
            f"no chat template on {template_source}; pass --template-source "
            f"(e.g. allenai/Olmo-3-7B-Instruct)")

    if args.reasoning:
        assert str(template_source).endswith("_reasoning.jinja"), (
            f"--reasoning needs a reasoning template (pilot/templates/*_reasoning.jinja), got {template_source}")
    if args.reasoning:
        # messages / tools as JSON strings: per-row tool schemas differ, so Arrow cannot infer one schema for them
        # (2026-10-14 pilot MIX_FAIL); build_reasoning_tokenize_fn decodes them per row
        from datasets import Dataset
        rows = [json.loads(l) for p in args.data for l in open(p, encoding="utf-8") if l.strip()]
        ds = Dataset.from_list([{"messages_json": json.dumps(r["messages"]) if r.get("messages") else "",
                                 "tools_json": json.dumps(r.get("tools") or []), "text": r.get("text") or "",
                                 "source": r.get("source", "tooluse"), "group": r.get("group") or ""} for r in rows])
        ds = ds.shuffle(seed=args.seed)
    else:
        parts = [load_dataset("json", data_files=p, split="train") for p in args.data]
        ds = concatenate_datasets(parts).shuffle(seed=args.seed)
    n_raw = len(ds)
    if args.reasoning:
        ds = ds.map(build_reasoning_tokenize_fn(tokenizer, replay_chat=args.replay_chat), batched=True,
                    batch_size=64, with_indices=True, remove_columns=ds.column_names, num_proc=8,
                    desc="tokenizing (one example per assistant turn)")
    else:
        ds = ds.map(build_tokenize_fn(tokenizer, args.max_seq_len, replay_chat=args.replay_chat),
                    remove_columns=[c for c in ds.column_names if c != "source"],
                    num_proc=8, desc="tokenizing")
    ds = ds.filter(lambda r: r["n_supervised"] > 0)
    ds = ds.filter(lambda r: r["n_tokens"] <= args.max_seq_len)
    is_rep = [str(s).startswith("replay") for s in ds["source"]]
    sup = ds["n_supervised"]
    tool_tok = sum(n for n, r in zip(sup, is_rep) if not r); rep_tok = sum(n for n, r in zip(sup, is_rep) if r)
    print(f"supervised tokens: tool {tool_tok}, replay {rep_tok} ({100 * rep_tok / max(1, tool_tok + rep_tok):.1f}% replay)")
    if args.replay_token_frac is not None and rep_tok:
        budget = tool_tok * args.replay_token_frac / (1 - args.replay_token_frac)
        keep, used = [], 0  # rows are already shuffled, so keeping the first replay rows that fit is a random draw
        for i, (n, r) in enumerate(zip(sup, is_rep)):
            if not r: keep.append(i)
            elif used + n <= budget: keep.append(i); used += n
        ds = ds.select(keep)
        print(f"replay capped to {args.replay_token_frac:.0%} of supervised tokens: kept {used} replay tokens, "
              f"{sum(1 for i in keep if is_rep[i])} replay rows")
    n_kept = len(ds)
    by_source = {}
    for s in ds["source"]:
        by_source[s] = by_source.get(s, 0) + 1
    print(f"rows: {n_raw} raw -> {n_kept} kept; by source: {by_source}")

    if args.reasoning:
        # split by conversation (group), not by example: per-turn examples of one row share their prefix, so a
        # row-level random split would put near-copies of eval examples in training and flatter eval_loss
        def _is_eval(g):
            return int(hashlib.sha256(f"{args.seed}:{g}".encode()).hexdigest(), 16) % 10000 < args.eval_frac * 10000
        ev = [_is_eval(g) for g in ds["group"]]
        train_ds = ds.select([i for i, e in enumerate(ev) if not e]).shuffle(seed=args.seed)
        eval_ds = ds.select([i for i, e in enumerate(ev) if e])
        print(f"eval split by conversation: {len(train_ds)} train / {len(eval_ds)} eval examples")
    else:
        split = ds.train_test_split(test_size=args.eval_frac, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]

    snapshot_cbs = []
    if args.snapshot_frac > 0:
        steps_per_epoch = math.ceil(len(train_ds) / EFFECTIVE_BATCH)
        snap_every = max(1, round(steps_per_epoch * args.snapshot_frac))
        snapshot_cbs = [SnapshotCallback(tokenizer, args.output_dir, snap_every, steps_per_epoch,
                                         hub_prefix=args.snapshot_hub_prefix, private=True)]
        print(f"snapshots: every {snap_every} steps (~{args.snapshot_frac:g} epoch); "
              f"steps/epoch={steps_per_epoch}")

    slug = base_model.split("/")[-1].lower()
    run_name = args.run_name or f"{slug}-r{args.lora_r}-{args.epochs:g}ep-unsloth"

    # --- Provenance manifest -------------------------------------------------
    # Ride-along proof of exactly what produced each model, so every reported
    # number traces back to a saved artifact (this is the fix for the earlier
    # run whose headline metric had no backing run). Written to output_dir and
    # copied into every snapshot dir + final/ so it lands on HF next to the
    # weights, and mirrored into the W&B config below.
    import sys, time
    from importlib.metadata import version as _pkgver

    def _v(pkg):
        try:
            return _pkgver(pkg)
        except Exception:
            return None

    data_sha256 = {p: sha256_file(p) for p in args.data}
    manifest = {
        "run_name": run_name,
        "git_sha": git_sha(),
        "argv": sys.argv,
        "base_model": base_model,
        "template_source": template_source,
        "reasoning": args.reasoning,
        "chat_template_sha256": hashlib.sha256((tokenizer.chat_template or "").encode()).hexdigest(),
        "modules_to_save": modules_to_save,
        "data_files": args.data,
        "data_sha256": data_sha256,
        "rows_raw": n_raw, "rows_kept": n_kept, "by_source": by_source,
        "train_examples": len(train_ds), "eval_examples": len(eval_ds),
        "hyperparams": {
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout, "lr_adapters": args.lr,
            "lr_embeddings": args.embedding_lr, "epochs": args.epochs,
            "max_steps": args.max_steps, "max_seq_len": args.max_seq_len,
            "effective_batch": EFFECTIVE_BATCH, "per_device_batch": args.per_device_batch,
            "grad_accum": grad_accum, "warmup_ratio": 0.05, "schedule": "cosine",
            "seed": args.seed, "snapshot_frac": args.snapshot_frac,
        },
        "versions": {p: _v(p) for p in ("torch", "transformers", "unsloth", "trl",
                                        "datasets", "peft", "accelerate", "bitsandbytes")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _short = {os.path.basename(k): v[:12] for k, v in data_sha256.items()}
    print(f"[manifest] git={manifest['git_sha']}  data_sha256={_short}", flush=True)

    wandb.init(name=run_name, config={
        "base_model": base_model, "stack": "unsloth",
        "git_sha": manifest["git_sha"], "data_sha256": data_sha256,
        "data_files": args.data, "rows_kept": n_kept, "by_source": by_source,
        "train_examples": len(train_ds), "eval_examples": len(eval_ds),
        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "modules_to_save": modules_to_save,
        "lr_adapters": args.lr, "lr_embeddings": args.embedding_lr,
        "schedule": "cosine", "warmup_ratio": 0.05,
        "max_seq_len": args.max_seq_len, "effective_batch": EFFECTIVE_BATCH,
        "per_device_batch": args.per_device_batch, "grad_accum": grad_accum,
        "epochs": args.epochs, "seed": args.seed,
    })

    trainer = UnslothTrainer(
        model=model,
        processing_class=tokenizer,  # unsloth's fix_untrained_tokens needs it on the trainer
        args=UnslothTrainingArguments(
            output_dir=args.output_dir,
            run_name=run_name,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_batch,
            per_device_eval_batch_size=args.per_device_eval_batch or args.per_device_batch,
            gradient_accumulation_steps=grad_accum,
            learning_rate=args.lr,
            embedding_learning_rate=args.embedding_lr,  # split LR: embeddings stay slow
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,
            optim="adamw_8bit",
            logging_steps=5,
            eval_strategy=("no" if args.eval_steps <= 0 else "steps"),
            eval_steps=(args.eval_steps if args.eval_steps > 0 else None),
            save_strategy="steps",
            save_steps=100,  # multiple of eval_steps, required for load_best_model_at_end
            save_total_limit=3,
            # auto-stop when eval_loss stops improving, and keep the best checkpoint
            load_best_model_at_end=bool(args.early_stopping_patience),
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            seed=args.seed,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
        callbacks=([EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold)]
            if args.early_stopping_patience else []) + snapshot_cbs,
    )
    trainer.train(resume_from_checkpoint=bool(
        list(Path(args.output_dir).glob("checkpoint-*"))) or None)

    final_dir = Path(args.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    import shutil
    shutil.copy(os.path.join(args.output_dir, "run_manifest.json"),
                str(final_dir / "run_manifest.json"))  # provenance rides with the weights
    # Log the model as a W&B artifact, but never let it abort the run: it stages a
    # copy of the weights under WANDB_CACHE_DIR/DATA_DIR, and if that disk is full the
    # OSError would otherwise crash the process *after* training finished — losing the
    # HF push below and marking the run "failed". Keep WANDB_* dirs on /workspace, treat
    # it as best-effort, and allow skipping it entirely (WANDB_SKIP_ARTIFACT=1) when the
    # weights already ride to HF via --push-to-hub-merged / --snapshot-hub-prefix and the
    # disk is tight (the staged copy is the main disk hog on long runs).
    if os.environ.get("WANDB_SKIP_ARTIFACT") == "1":
        print("wandb artifact logging skipped (WANDB_SKIP_ARTIFACT=1)", flush=True)
    else:
        try:
            artifact = wandb.Artifact(args.artifact_name, type="model",
                                      metadata={"base_model": base_model, "run": run_name})
            artifact.add_dir(str(final_dir))
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"WARN: wandb artifact logging failed ({e!r}); continuing without it", flush=True)
    wandb.finish()
    print(f"done - final adapter at {final_dir}")

    if args.push_to_hub_merged:
        # merge LoRA + trained embeddings into the base and push a 16-bit model
        print(f"merging and pushing 16-bit model to {args.push_to_hub_merged} ...")
        model.push_to_hub_merged(
            args.push_to_hub_merged, tokenizer,
            save_method="merged_16bit",
            token=os.environ.get("HF_TOKEN"),
            private=args.hub_private,
        )
        print(f"pushed merged 16-bit model to https://huggingface.co/{args.push_to_hub_merged}")


if __name__ == "__main__":
    main()
