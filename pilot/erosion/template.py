"""Write the chat template vLLM serves with (anchor's, else the arm's template source), patched so templates
that test `tools is none` also work when no tools are passed. Usage: template.py <arm> <out_path>"""
import os, sys
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
import settings as S

arm, out = sys.argv[1:3]
a = dict(S.ARMS[arm])
a["anchor"] = os.environ.get("ANCHOR_REPO") or a["anchor"]
ct = AutoTokenizer.from_pretrained(a["anchor"]).chat_template
if not ct and a["template_source"]:
    ct = AutoTokenizer.from_pretrained(a["template_source"]).chat_template
assert ct, f"no chat template for {arm}"
ct = ct.replace("tools is not none", "tools is defined and tools is not none")
ct = ct.replace("tools is none", "tools is not defined or tools is none")
open(out, "w").write(ct)
print("TEMPLATE", out, len(ct))
