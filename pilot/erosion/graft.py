"""Graft a tool-training delta onto another base: out = target + (donor_anchor - donor_base), tensor by tensor.
Usage: graft.py <donor_anchor> <donor_base> <target_base(repo or local dir)> <out_dir>
Tokenizer / chat template / config come from the donor anchor."""
import glob, json, os, shutil, sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import snapshot_download

donor, donor_base, target, out = sys.argv[1:5]
get = lambda r: r if os.path.isdir(r) else snapshot_download(r)
D, DB, T = get(donor), get(donor_base), get(target)
def opener(d):
    idx = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        h = safe_open(f, "pt", device="cpu")
        for k in h.keys(): idx[k] = h
    return idx
Dk, DBk, Tk = opener(D), opener(DB), opener(T)
assert set(Dk) == set(DBk) == set(Tk), (len(Dk), len(DBk), len(Tk))
os.makedirs(out, exist_ok=True)
shard, size, n, index = {}, 0, 0, {}
def flush():
    global shard, size, n
    if not shard: return
    name = f"model-{n:05d}.safetensors"; save_file(shard, os.path.join(out, name), metadata={"format": "pt"})
    for k in shard: index[k] = name
    shard, size, n = {}, 0, n + 1
dn = []
for k in sorted(Dk):
    d = Dk[k].get_tensor(k).float() - DBk[k].get_tensor(k).float()
    t = Tk[k].get_tensor(k)
    shard[k] = (t.float() + d).to(t.dtype)
    dn.append(d.norm().item() / (DBk[k].get_tensor(k).float().norm().item() + 1e-12))
    size += shard[k].numel() * 2
    if size > 4e9: flush()
flush()
json.dump({"metadata": {}, "weight_map": index}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
for f in os.listdir(D):
    if f.endswith((".json", ".jinja", ".txt", ".model")) and not f.startswith(("model.safetensors", "run_manifest", "trainer_state")):
        shutil.copy(os.path.join(D, f), out)
print("GRAFTED", out, "mean relative delta", sum(dn) / len(dn))
