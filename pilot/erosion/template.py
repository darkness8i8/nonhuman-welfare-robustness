"""Write the chat template vLLM serves with (anchor's, else the arm's template source), patched so templates
that test `tools is none` also work when no tools are passed. Usage: template.py <arm> <out_path>
Reasoning recipe: the arm's template_source is a pilot/templates/*_reasoning.jinja file, and that file is served as is
(it already carries the `tools` patch). It is the template the anchor was trained with, so serving it guarantees the
same rendering as training; a different template on the anchor is reported, not used."""
import os, sys
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import settings as S
from reasoning_sft import read_template

arm, out = sys.argv[1:3]
a = dict(S.ARMS[arm])
a["anchor"] = os.environ.get("ANCHOR_REPO") or a["anchor"]
ct = read_template(a["template_source"]) if a["template_source"] else None
if ct:
    if AutoTokenizer.from_pretrained(a["anchor"]).chat_template != ct:
        print(f"TEMPLATE_WARN {a['anchor']} carries a different chat template; serving {a['template_source']}")
else:
    ct = AutoTokenizer.from_pretrained(a["anchor"]).chat_template
    if not ct and a["template_source"]:
        ct = AutoTokenizer.from_pretrained(a["template_source"]).chat_template
    assert ct, f"no chat template for {arm}"
    ct = ct.replace("tools is not none", "tools is defined and tools is not none")
    ct = ct.replace("tools is none", "tools is not defined or tools is none")
open(out, "w").write(ct)
print("TEMPLATE", out, len(ct))
