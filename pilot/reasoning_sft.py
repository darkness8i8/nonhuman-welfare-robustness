"""Shared rendering for the REASONING part-one recipe (settings.RECIPE == "reasoning"). Used by build_nemotron.py
(length cap), train_unsloth.py --reasoning (training rows) and erosion/reasoning_examples.py (the printed examples),
so all three render a row the same way. No unsloth / torch imports: this must load in the eval venv and on a laptop.

Multi-turn supervision. Each Nemotron row has several assistant turns, each with its own reasoning. The served
template (templates/*_reasoning.jinja) never shows reasoning on history turns, so at inference turn k is written after
a history whose earlier turns carry calls and answers but no reasoning. One sequence per row cannot train that: turn k
would see turn 1's reasoning in its context. So each row becomes one training example PER assistant turn: the messages
up to and including turn k, rendered by the same template (history without reasoning, turn k with it), loss on turn k
only. Every turn's reasoning is supervised exactly once, in exactly the context it will be written in at inference."""
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
TEMPLATES = {"qwen": os.path.join(TEMPLATE_DIR, "qwen3_reasoning.jinja"),
             "olmo": os.path.join(TEMPLATE_DIR, "olmo3_reasoning.jinja")}


def read_template(path_or_repo):
    """A chat template from a .jinja file, or None for a Hub repo id (caller loads that repo's template)."""
    if str(path_or_repo).endswith(".jinja"):
        with open(path_or_repo, encoding="utf-8") as f:
            return f.read()
    return None


def turn_prefixes(messages):
    """One message list per assistant turn: messages[:k+1] for each assistant index k (earliest turn first)."""
    return [messages[:k + 1] for k, m in enumerate(messages) if m.get("role") == "assistant"]


def last_assistant_span(ids, im_start, assistant_header, end_ids):
    """[start, end) of the LAST assistant turn's content incl. its end token (<|im_end|>, or Olmo's eos after the final
    turn). Token scan, like train_unsloth.find_assistant_spans; returns None if there is no assistant turn."""
    h, n, start = len(assistant_header), len(ids), None
    for i in range(n - h):
        if ids[i] == im_start and ids[i + 1:i + 1 + h] == assistant_header:
            start = i + 1 + h
    if start is None:
        return None
    j = start
    while j < n and ids[j] not in end_ids:
        j += 1
    return start, min(j + 1, n)


class Renderer:
    """tokenizer + reasoning template -> (input_ids, labels) supervising only the final assistant turn."""

    def __init__(self, tokenizer, template_text):
        self.tok = tokenizer
        self.template = template_text
        self.im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        # Olmo ends the final turn with eos (<|endoftext|>), every other turn with <|im_end|>; Qwen always <|im_end|>
        self.end_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}
        self.header = tokenizer.encode("assistant\n", add_special_tokens=False)

    def text(self, messages, tools):
        return self.tok.apply_chat_template(messages, tools=tools, chat_template=self.template, tokenize=False,
                                            add_generation_prompt=False)

    def example(self, messages, tools):
        ids = self.tok(self.text(messages, tools), add_special_tokens=False)["input_ids"]
        labels = [-100] * len(ids)
        span = last_assistant_span(ids, self.im_start, self.header, self.end_ids)
        if span:
            labels[span[0]:span[1]] = ids[span[0]:span[1]]
        return ids, labels
