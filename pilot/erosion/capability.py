"""General-capability check for midtrained bases, no generation and no judges (likelihood scoring, as lm-eval-harness):
- MMLU (zero-shot, 500 test questions, fixed sample): pick the answer letter with the highest log-prob after "Answer:".
- ARC-Challenge (500 test questions): pick the choice whose text has the highest length-normalized log-prob (acc_norm).
- Web-text loss: mean per-token loss on 200 FineWeb-Edu documents (first 1,024 tokens each).
Reports accuracy and the mean probability on the correct option (continuous, picks up small drops). Compare models
within a family only (different tokenizers). Usage: capability.py <out_json> <model>[@<revision>] [...]"""
import json, os, random, sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

N_MC, N_WEB, WEB_TOK = 500, 200, 1024
out, *models = sys.argv[1:]
rng = random.Random(0)
mmlu = load_dataset("cais/mmlu", "all", split="test"); mmlu = mmlu.select(sorted(rng.sample(range(len(mmlu)), N_MC)))
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test"); arc = arc.select(sorted(rng.sample(range(len(arc)), N_MC)))
web = []
for r in load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True):
    web.append(r["text"])
    if len(web) == N_WEB: break
L = "ABCD"

res = json.load(open(out)) if os.path.exists(out) else {}
for spec in models:
    if spec in res: continue
    repo, _, rev = spec.partition("@"); rev = rev or None
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    m = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.bfloat16).cuda().eval()
    ids_of = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    letter_ids = [ids_of(" " + l)[-1] for l in L]
    r = {}
    with torch.no_grad():
        # MMLU: letter logits at the end of the prompt
        acc = pc = 0.0
        for q in mmlu:
            p = q["question"].strip() + "\n" + "".join(f"{L[i]}. {c}\n" for i, c in enumerate(q["choices"])) + "Answer:"
            x = torch.tensor([ids_of(p)]).cuda()
            pr = torch.softmax(m(x).logits[0, -1, letter_ids].float(), -1)
            acc += int(pr.argmax()) == q["answer"]; pc += pr[q["answer"]].item()
        r["mmlu_acc"] = round(acc / N_MC, 4); r["mmlu_p_correct"] = round(pc / N_MC, 4)
        # ARC-Challenge: length-normalized log-prob of each choice text
        acc = pc = 0.0
        for q in arc:
            ctx = ids_of(f"Question: {q['question']}\nAnswer:")
            scores = []
            for c in q["choices"]["text"]:
                cont = ids_of(" " + c); x = torch.tensor([ctx + cont]).cuda()
                lp = torch.log_softmax(m(x).logits[0, len(ctx) - 1:-1].float(), -1)
                scores.append(lp[torch.arange(len(cont)), torch.tensor(cont)].sum().item() / len(c))
            gold = q["choices"]["label"].index(q["answerKey"])
            pr = torch.softmax(torch.tensor(scores), 0)
            acc += int(pr.argmax()) == gold; pc += pr[gold].item()
        r["arc_c_acc_norm"] = round(acc / N_MC, 4); r["arc_c_p_correct"] = round(pc / N_MC, 4)
        # web-text loss
        nll = n = 0.0
        for t in web:
            x = torch.tensor([ids_of(t)[:WEB_TOK]]).cuda()
            if x.shape[1] < 2: continue
            nll += m(x, labels=x).loss.item() * (x.shape[1] - 1); n += x.shape[1] - 1
        r["web_loss"] = round(nll / n, 4)
    del m; torch.cuda.empty_cache()
    res[spec] = r; json.dump(res, open(out, "w"), indent=1)
    print("CAPABILITY", spec, json.dumps(r), flush=True)
