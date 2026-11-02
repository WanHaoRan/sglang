"""The key of a reply's tool call: the engine's half of the sglang.next_turn v2 table.

An agent system keeps a gap distribution per key and sends the table on every request
(agent_cache/DIST_HINT_DESIGN.md). Once a reply exists, the engine reads the same key from
it. The rule must match the client's (agent_cache/scripts/gap_hints.py: reply_key, text_call,
command_word), or a key finds no entry and falls back to its tool's.
"""

from __future__ import annotations

import json
import re
from typing import Optional

NO_CALL_KEY = "_no_call"
KEY_SEP = "/"

_TEXT_CALL = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_TEXT_ARG = re.compile(
    r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S
)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)
_LEADING_WORDS = re.compile(r"cd\s+\S+\s*(?:&&|;)\s*|[A-Za-z_]\w*=\S*\s+")
# Within the value grammar of HICACHE_EVT lines; the length bound is arbitrary.
_KEY_PART = re.compile(r"[A-Za-z0-9_.\-]{1,48}")


def _text_call(text: str) -> Optional[tuple[str, dict]]:
    match = _TEXT_CALL.search(text)
    if match is None:
        return None
    args = {k: v.strip("\n") for k, v in _TEXT_ARG.findall(match.group(2))}
    return match.group(1), args


def _json_call(text: str) -> Optional[tuple[str, dict]]:
    match = _JSON_CALL.search(text)
    if match is None:
        return None
    try:
        call = json.loads(match.group(1))
        args = call.get("arguments", {})
        args = json.loads(args) if isinstance(args, str) else args
        name = call.get("name")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(name, str):
        return None
    return name, args if isinstance(args, dict) else {}


def _command_word(command: str) -> Optional[str]:
    """A command's first word after any leading cd and VAR=value; None if it makes no key."""
    rest = command.strip()
    while (match := _LEADING_WORDS.match(rest)) is not None:
        rest = rest[match.end() :]
    words = rest.split()
    if not words:
        return None
    word = words[0].rsplit("/", 1)[-1]
    if word.startswith("python"):
        if len(words) >= 3 and words[1] == "-m":
            word = f"python-m-{words[2]}"
        else:
            word = "python-c" if len(words) >= 2 and words[1] == "-c" else "python"
    return word if _KEY_PART.fullmatch(word) else None


def reply_key(text: str) -> str:
    """The key of the first tool call a reply writes, after any reasoning, or NO_CALL_KEY."""
    answer = text.rsplit("</think>", 1)[-1]
    call = _text_call(answer) or _json_call(answer)
    if call is None:
        return NO_CALL_KEY
    tool, args = call
    command = args.get("command", args.get("cmd"))
    word = _command_word(command) if isinstance(command, str) else None
    return tool if word is None else f"{tool}{KEY_SEP}{word}"
