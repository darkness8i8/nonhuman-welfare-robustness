"""RoPE guard: fail loudly when a loaded or served config's rope settings differ from the ORIGINAL base family's.

Why (bug found 2026-10-13): our midtrained bases were saved by transformers 5.x, which writes `rope_parameters` and no
top-level `rope_theta`. The train venv (transformers 4.57.6) does not read `rope_parameters`, so it silently fell back
to rope_theta 10000 (Olmo needs 500000 + yarn factor 8, Qwen3 needs 1000000). A model trained or served with the wrong
rope still runs and still writes text, so nothing else catches it.

Used by train_unsloth.py (after loading), erosion/merge.py (the dir vLLM serves) and erosion/evaluate.py (the same dir,
right before scoring). Works in both venvs: it needs only huggingface_hub and the config as a dict."""
import json

# the ORIGINAL bases each family's midtraining started from (same pins as erosion/retention.py)
FAMILY_BASE = {"olmo3": ("allenai/Olmo-3-1025-7B", "a81bae42db3975be1671e27b9c9a56da1a9f980f"),
               "qwen3": ("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4")}
_KEYS = ("rope_type", "factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "attention_factor")


def _norm_scaling(s):
    """rope_scaling / rope_parameters -> comparable dict; 'default' (plain RoPE) and None mean the same thing."""
    s = dict(s or {})
    s["rope_type"] = s.get("rope_type") or s.get("type") or "default"
    if s["rope_type"] == "default":
        return None
    return {k: (float(s[k]) if isinstance(s[k], (int, float)) else s[k]) for k in _KEYS if s.get(k) is not None}


def effective_rope(cfg, transformers_major):
    """(rope_theta, scaling) that THIS transformers version actually uses for a config dict.
    4.x reads only top-level rope_theta / rope_scaling (a missing rope_theta means the class default, e.g. 10000);
    5.x reads rope_parameters (per layer type for Olmo 3: we check full_attention, the layers yarn applies to)."""
    rp = cfg.get("rope_parameters")
    if transformers_major >= 5 and rp:
        rp = rp.get("full_attention", rp) if isinstance(rp, dict) else rp
        return float(rp.get("rope_theta")), _norm_scaling(rp)
    theta = cfg.get("rope_theta")
    return (float(theta) if theta is not None else None), _norm_scaling(cfg.get("rope_scaling"))


def original_rope(model_type):
    """Rope settings of the family's original base, read from its config.json on the Hub (pinned revision)."""
    from huggingface_hub import hf_hub_download
    repo, rev = FAMILY_BASE[model_type]
    cfg = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
    return effective_rope(cfg, 4)   # the originals are 4.x configs with top-level rope_theta / rope_scaling


def check(cfg, transformers_major, where):
    """Assert a config dict (as the running transformers sees it) has the family's original rope. Raises otherwise."""
    mt = cfg.get("model_type")
    if mt not in FAMILY_BASE:
        raise SystemExit(f"ROPE_GUARD {where}: unknown model_type {mt!r}; add its original base to rope_guard.FAMILY_BASE")
    want, got = original_rope(mt), effective_rope(cfg, transformers_major)
    if got != want:
        raise SystemExit(
            f"ROPE_GUARD FAIL {where}: {mt} loads with rope (theta, scaling) = {got}, but the original "
            f"{FAMILY_BASE[mt][0]} uses {want}. Usually a transformers-5 config (rope_parameters, no top-level rope_theta) "
            f"read by transformers 4.x. Fix the config (add top-level rope_theta/rope_scaling) before training or serving.")
    print(f"ROPE_GUARD OK {where}: {mt} theta={got[0]} scaling={got[1]}", flush=True)


def check_dir(path, where):
    """Check a saved model dir the way the CURRENT environment's transformers will read it."""
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(path).to_dict()   # parsed by this transformers, not the raw file
    check(cfg, int(transformers.__version__.split(".")[0]), where)
