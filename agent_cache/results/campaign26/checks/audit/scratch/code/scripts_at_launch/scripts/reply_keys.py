"""Build <trace>.reply_keys.json: for each turn of a converted trace, the key of the tool call its recorded reply made.

  frozen/hint_realism_20261027/venv/bin/python -I scripts/reply_keys.py --trace TRACE --parquet-dir DIR

The replay's --hints dist needs the key of each recorded reply (gap_hints.py: the tool and the first word of its
command). The tool is in the trace, in the header of the next turn's tool result, but the command is not: the converter
dropped assistant messages (convert_lmcache.py). So the keys are rebuilt from the same parquet, and the trace itself
stays as it is. Needs pyarrow and no GPU. Treats the parquet as untrusted data: nothing from it is executed.

keys[c][t] is the key of conversation c's reply t, read from the assistant message in the new part of turn t+1's input:
- a reply with a tool call: gap_hints.reply_key of its first call (native tool_calls, or the text form);
- a reply without one: "_no_call";
- a call the harness rejected (OpenHands drops the reply and re-prompts): the engine would still have seen an ordinary
  call, but the trace has lost its command. Such a turn takes the key of a random accepted call of the same source,
  model and tool, seeded by session and turn, so the key cannot reveal the rejection;
- null for a conversation's last turn, whose reply no later input records.

Alignment is checked on every turn (session id, pre_gap, output_length, input extending the previous row, the parsed
tool against the next turn's header); the script fails if any check does.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import sys

import pyarrow.dataset as ds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_hints import NO_CALL_KEY, KEY_SEP, reply_key, reply_keys_path, text_call, tool_category  # noqa: E402


def recorded_call(message: dict) -> tuple | None:
    """(tool, args) of an assistant message's first tool call, native or written as text."""
    calls = message.get("tool_calls") or []
    if not calls:
        return text_call(message.get("content") or "")
    function = calls[0]["function"]
    try:
        args = json.loads(function["arguments"]) if function["arguments"] else {}
    except (json.JSONDecodeError, TypeError):
        args = {}
    return function["name"], args if isinstance(args, dict) else {}


def conversation_keys(*, conv: list, rows: list, session_id: str, checks: collections.Counter) -> list:
    """The keys of one conversation; a rejected call is left as ("rejected", tool or None) for fill_rejected."""
    keys: list = [None] * len(conv)
    previous: list = []
    base = session_id.split("__chain")[0]
    for k, turn in enumerate(conv):
        row = rows[turn["source_rows"][-1]]
        checks["session mismatches"] += row["session_id"] != base
        checks["gap mismatches"] += k > 0 and float(row["pre_gap"]) != float(turn["pre_gap"])
        checks["output mismatches"] += int(row["output_length"]) != int(turn["output_length"])
        checks["input not extending"] += row["input"][:len(previous)] != previous
        delta = row["input"][len(previous):]
        previous = row["input"]
        if k == 0:
            continue
        category = tool_category(turn)
        replies = [m for m in delta if m["role"] == "assistant"]
        if replies:
            call = recorded_call(replies[-1])
            if call is None:
                keys[k - 1] = NO_CALL_KEY
                checks["no call"] += 1
            else:
                keys[k - 1] = reply_key(call[0], call[1])
                checks["calls"] += 1
                checks["native calls" if replies[-1].get("tool_calls") else "text calls"] += 1
                checks["tool mismatches"] += category != call[0]
        elif category == "error" or category.startswith("error/"):
            keys[k - 1] = ("rejected", category.split(KEY_SEP, 1)[1] if KEY_SEP in category else None)
            checks["rejected"] += 1
        else:
            checks["no reply and no re-prompt"] += 1
    return keys


def fill_rejected(*, all_keys: list, sessions: list) -> None:
    """Give each rejected call the key of a random accepted call of the same source, model and tool."""
    by_tool, by_model = collections.defaultdict(list), collections.defaultdict(list)
    for meta, keys in zip(sessions, all_keys):
        for key in keys:
            if isinstance(key, str) and key != NO_CALL_KEY:
                by_model[meta["source"], meta["orig_model"]].append(key)
                by_tool[meta["source"], meta["orig_model"], key.split(KEY_SEP, 1)[0]].append(key)
    for meta, keys in zip(sessions, all_keys):
        for t, key in enumerate(keys):
            if isinstance(key, tuple):
                tool = key[1]
                donors = by_tool.get((meta["source"], meta["orig_model"], tool)) or by_model[meta["source"], meta["orig_model"]]
                drawn = random.Random(f"{meta['session_id']}:{t}").choice(donors)
                # with no accepted call of this tool to learn from, the tool alone is all that is known
                keys[t] = drawn if tool is None or drawn.split(KEY_SEP, 1)[0] == tool else tool


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--parquet-dir", required=True)
    args = ap.parse_args()
    with open(args.trace) as f:
        data = json.load(f)
    rows = ds.dataset(sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet"))), format="parquet").to_table(
        columns=["session_id", "input", "pre_gap", "output_length"]).to_pylist()
    checks: collections.Counter = collections.Counter()
    all_keys = [conversation_keys(conv=conv, rows=rows, session_id=meta["session_id"], checks=checks) if conv else []
                for meta, conv in zip(data["sessions"], data["conversations"])]
    fill_rejected(all_keys=all_keys, sessions=data["sessions"])
    bad = {k: v for k, v in checks.items() if "mismatch" in k or "not extending" in k or "no reply" in k}
    metadata = {"trace": os.path.basename(args.trace), "conversations": len(all_keys), "turns": sum(map(len, all_keys)),
                "checks": dict(checks)}
    print(json.dumps(metadata))
    if any(bad.values()):
        print(f"STOP: alignment checks failed, nothing written: {bad}")
        return 1
    out = reply_keys_path(args.trace)
    with open(out, "w") as f:
        json.dump({"metadata": metadata, "conversations": all_keys}, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out):,d} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
