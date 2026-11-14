"""Replay chat templates for Qwen3 that keep a re-fed assistant reply prefix-identical to the
generated one, so the radix cache reuses the reply's KV on the next turn (RUNBOOK 4.5).

Stock Qwen3 template, assistant turns before the last user/tool message:
  thinking off: generation prompt ends '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n', history
                renders '<|im_start|>assistant\\n' + content  -> tokens diverge after 'assistant\\n'
  thinking on:  the generated '<think>...</think>' is stripped from history -> same divergence

  make  : write templates/qwen3_replay_nothink.jinja (history gets the empty think block) and
          templates/qwen3_replay_think.jinja (history rendered verbatim, no </think> split)
  check : print reply tokens reused across one turn for stock / nothink / think (toy 2-turn chain)

Usage: python3 replay_template.py make|check [--model Qwen/Qwen3-32B-FP8] [--out-dir ../templates]
"""
import argparse
import os
import sys

from transformers import AutoTokenizer

NOTHINK_ANCHOR = (
    "{%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n"
    "        {%- if message.tool_calls %}"
)
THINK_ANCHOR = "{%- if '</think>' in content %}"


def make_templates(stock: str) -> dict:
    assert stock.count(NOTHINK_ANCHOR) == 1, f"nothink anchor found {stock.count(NOTHINK_ANCHOR)}x"
    assert stock.count(THINK_ANCHOR) == 1, f"think anchor found {stock.count(THINK_ANCHOR)}x"
    # Both variants render historical assistant content VERBATIM (no '</think>' split): under ignore_eos the
    # model runs past <|im_end|> and hallucinates '</think>' inside its garbage tail, and the stock split would
    # keep only the text after it (seen 2026-09-18: reply reused 2/132). nothink additionally re-inserts the
    # empty think block that the generation prompt carried when enable_thinking was false.
    verbatim = stock.replace(THINK_ANCHOR, "{%- if false %}")
    nothink = verbatim.replace(
        NOTHINK_ANCHOR,
        NOTHINK_ANCHOR.replace("+ '\\n' + content", "+ '\\n<think>\\n\\n</think>\\n\\n' + content"),
    )
    think = verbatim
    return {"qwen3_replay_nothink.jinja": nothink, "qwen3_replay_think.jinja": think}


def reuse_across_turn(tok, template: str, thinking: bool) -> tuple:
    """Simulate turn 1 (prompt + generated reply, as cached) and turn 2 (history re-fed);
    return (reply tokens reused, reply tokens, uncached tokens on turn 2)."""
    tok.chat_template = template
    sys_ = {"role": "system", "content": "You are a coding agent."}
    u1 = {"role": "user", "content": "Fix the bug in foo.py"}
    reply = ("<think>\nLet me look at the file first.\n</think>\n\n" if thinking else "") + (
        'I will read foo.py.\n<tool_call>\n{"name": "read", "arguments": {"path": "foo.py"}}\n</tool_call>'
    )
    tool = {"role": "user", "content": "[TOOL RESULT read]\ndef foo():\n    return 1"}

    def ids(msgs):
        s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return tok.encode(s, add_special_tokens=False)

    p1 = ids([sys_, u1])
    gen = tok.encode(reply, add_special_tokens=False)
    cached = p1 + gen
    p2 = ids([sys_, u1, {"role": "assistant", "content": tok.decode(gen)}, tool])
    k = 0
    for a, b in zip(cached, p2):
        if a != b:
            break
        k += 1
    return max(0, k - len(p1)), len(gen), len(p2) - k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "check"])
    ap.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "templates"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    stock = tok.chat_template
    templates = make_templates(stock)

    if args.cmd == "make":
        os.makedirs(args.out_dir, exist_ok=True)
        for name, text in templates.items():
            path = os.path.join(args.out_dir, name)
            with open(path, "w") as f:
                f.write(text)
            print(f"wrote {os.path.abspath(path)}: {len(text)} bytes")
        return

    def as_served(text: str) -> str:
        # what the server does with a .jinja file (parser/template_manager.py:281-282)
        return text.strip("\n").replace("\\n", "\n")

    for label, template, thinking in (
        ("stock, thinking on ", stock, True),
        ("stock, thinking off", stock, False),
        ("nothink template   ", as_served(templates["qwen3_replay_nothink.jinja"]), False),
        ("think template     ", as_served(templates["qwen3_replay_think.jinja"]), True),
    ):
        reused, total, uncached = reuse_across_turn(tok, template, thinking)
        print(f"{label}: reply tokens reused {reused}/{total}, uncached on next turn {uncached}")


if __name__ == "__main__":
    sys.exit(main())
