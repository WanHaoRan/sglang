"""Next-turn gap hints as a list of distributions that the engine picks from once its reply exists.

Used by replay_agentic.py --hints dist, hint_eval.py (the offline check) and reply_keys.py. Stdlib only.

The LLM's reply chooses the tool call, and the call mostly sets the gap before the next request, so no client can name
the coming gap's kind when it sends a request. It sends a list instead: one entry per kind of call, each with the p10,
p50 and p90 of the gap in seconds and the number of past gaps behind them, keyed by the tool and the first word of its
command ("execute_bash/pytest"). Once the reply exists the engine reads the same key from it and picks an entry: the
key, else its tool, else the default entry "*" (DIST_HINT_DESIGN.md). An entry may say {"ended": true} instead.

The list comes from one of three sources. "online", the default, is what a real agent system does: it measures each gap
as it finishes and keeps empirical quantiles per key across all its sessions, so every request carries the table as it
stands at that moment; in the replay, one harness and model is one agent system. "precalc" holds the same quantiles from
other tasks' recorded sessions, known before the run: a comparison, the case of a system with a long history. "session"
fits a Gaussian of ln(gap) to the session's own past gaps. Source "none" sends no estimates: only the link to the next
turn and the end. Key "tool" drops the first word, and key "none" keeps only the default entry, as a client that does not
know its tool types.

The replay's replies are generated without the recorded tool calls, so the engine cannot read a key from them. Each
request also carries replay_reply_key, the key of the recorded reply. A replay arm's engine reads it at commit, in place
of parsing its own reply; it is a workaround for the replay only. Keys come from <trace>.reply_keys.json (reply_keys.py)
because the converted trace dropped the replies.

Where the recorded agent's session ends, end "explicit" sends the key "_end", whose entry says ended; end "natural" does
so only below the 50-call cap; end "none" never does. Everywhere else the key is the recorded one, or none where the
trace has no reply, including where the replay stops early (--turns, --timeout) or the converter cut the conversation:
an agent cannot know about either. A version "1" action carries the picked entry's median for today's engine, which
cannot pick.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import re
from statistics import NormalDist, fmean, stdev
from typing import NamedTuple

NEXT_TURN_ACTION = "sglang.next_turn"
SOURCES = ("online", "precalc", "session", "none")
KEYS = ("command", "tool", "none")
POINTS = ("none", "p50")
ENDS = ("explicit", "natural", "none")
# Sigma floor 0.5 (source "session" only): the best p90 calibration among 0.1-1.0 on the 262K trace, 2026-10-29.
DEFAULTS = {"source": "online", "key": "command", "min_samples": 2, "sigma_floor": 0.5}
# Arbitrary bound, far above any useful floor; it keeps every quantile finite.
MAX_SIGMA_FLOOR = 10.0
# The launchers' knobs for the replay's --hint-* flags (start_client.sh, run_compare.sh).
KNOBS = ("HINT_SOURCE", "HINT_KEY", "HINT_MIN_SAMPLES", "HINT_SIGMA_FLOOR", "HINT_POINT", "HINT_END")
# Longest single sleep, as in campaigns 13-17: at x70, 57 of the 262K trace's 768 conversations hold a sleep over 5 h.
GAP_CAP_S = 1200.0
# ln of a zero gap is undefined: gaps below this count as this (seconds).
GAP_FLOOR_S = 0.01
# 228 of the 262K trace's 768 conversations (all SWE-bench) stop at exactly 50 calls;
# whether that is a harness cap or the dataset's window is unresolved (REALISTIC_HINTS.md 9).
CALL_CAP = 50
# Arbitrary: fewest past gaps behind an entry pooled across sessions.
MIN_POOLED = 5
ANY_KEY = "*"
END_KEY = "_end"
NO_CALL_KEY = "_no_call"
KEY_SEP = "/"
END_ENTRY = {"key": END_KEY, "ended": True}
TAUS = (0.1, 0.5, 0.9)
_STD_NORMAL = NormalDist()

# Tool-result headers -> category: OpenHands "EXECUTION RESULT of [x]", Inspect AI and WildClaw "[TOOL RESULT x]".
# Labels stay within [\w./-], the value grammar of the engine's HICACHE_EVT lines.
_CATEGORY_PATTERNS = (
    (re.compile(r"\s*EXECUTION RESULT of \[([\w\-.]+)\]"), "{}"),
    (re.compile(r"\s*\[TOOL RESULT ([\w\-.]+)\]"), "{}"),
    (re.compile(r"\s*\[TOOL RESULT \]"), "unnamed_tool"),
    (re.compile(r"\s*Missing required parameters for function '([\w\-.]+)'"), "error/{}"),
    (re.compile(r"\s*Parameter '[^']*' is not allowed for function '([\w\-.]+)'"), "error/{}"),
    (re.compile(r"\s*(?:Missing required argument|Parameter '[^']*' is expected)"), "error"),
    (re.compile(r"\s*Please (?:proceed to the next step|continue on whatever approach)"), "nudge"),
)
_TEXT_CALL = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_TEXT_ARG = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)
_LEADING_WORDS = re.compile(r"cd\s+\S+\s*(?:&&|;)\s*|[A-Za-z_]\w*=\S*\s+")
_KEY_PART = re.compile(r"[A-Za-z0-9_.\-]{1,48}")   # within the HICACHE_EVT grammar; arbitrary length bound


def tool_category(turn: dict) -> str:
    """Category of the tool call whose result opens `turn`, i.e. the call the previous reply made."""
    messages = turn.get("messages") or []
    if not messages:
        return "other"
    content = messages[0].get("content")
    text = content if isinstance(content, str) else json.dumps(content)
    for pattern, label in _CATEGORY_PATTERNS:
        m = pattern.match(text or "")
        if m:
            return label.format(*m.groups())
    return "other"


def text_call(content: str) -> tuple | None:
    """The first tool call a reply writes as text (<function=x><parameter=y>...), after any reasoning: (name, args)."""
    m = _TEXT_CALL.search(content.rsplit("</think>", 1)[-1])
    if m is None:
        return None
    return m.group(1), {k: v.strip("\n") for k, v in _TEXT_ARG.findall(m.group(2))}


def command_word(command: str) -> str | None:
    """A command's first word after any leading cd and VAR=value, as a key part; None if it does not make one."""
    s = command.strip()
    while True:
        m = _LEADING_WORDS.match(s)
        if m is None:
            break
        s = s[m.end():]
    words = s.split()
    if not words:
        return None
    word = words[0].rsplit("/", 1)[-1]
    if word.startswith("python"):
        if len(words) >= 3 and words[1] == "-m":
            word = f"python-m-{words[2]}"
        else:
            word = "python-c" if len(words) >= 2 and words[1] == "-c" else "python"
    return word if _KEY_PART.fullmatch(word) else None


def reply_key(tool: str, args: dict) -> str:
    """The key of a tool call: its tool and, when it has a command argument, the command's first word."""
    command = args.get("command", args.get("cmd"))
    word = command_word(command) if isinstance(command, str) else None
    return tool if word is None else f"{tool}{KEY_SEP}{word}"


def key_levels(key: str | None, mode: str) -> tuple:
    """The list entries a gap of this key feeds under key mode `mode`, finest first."""
    if key is None or mode == "none":
        return (ANY_KEY,)
    tool = key.split(KEY_SEP, 1)[0]
    if mode == "tool" or tool == key:
        return (tool, ANY_KEY)
    return (key, tool, ANY_KEY)


def pick_entry(by_key: dict, key: str | None) -> dict | None:
    """What the engine does once it knows the reply's key: the entry of that key, else its tool's, else the default."""
    if key is not None:
        for k in (key, key.split(KEY_SEP, 1)[0]):
            if k in by_key:
                return by_key[k]
    return by_key.get(ANY_KEY)


def task_of(session_id: str) -> str:
    """swebench__astropy__astropy-12907__minimax[__chain1] -> swebench__astropy__astropy-12907."""
    return re.sub(r"__chain\d+$", "", session_id).rsplit("__", 1)[0]


def scaled_gap(turn: dict, *, scale: float) -> float:
    """The replay's sleep before `turn`: its recorded pre_gap x scale, at most GAP_CAP_S."""
    return min(GAP_CAP_S, max(0.0, float(turn.get("pre_gap") or 0.0) * scale))


def reply_keys_path(trace: str) -> str:
    return f"{trace}.reply_keys.json"


def load_reply_keys(trace: str, conversations: list) -> list:
    """Per conversation and turn, the key of the recorded reply (reply_keys.py), checked against the trace's shape."""
    path = reply_keys_path(trace)
    if not os.path.exists(path):
        raise ValueError(f"{path} is missing: build it with scripts/reply_keys.py --trace {trace} --parquet-dir <dir>")
    with open(path) as f:
        keys = json.load(f)["conversations"]
    if [len(k) for k in keys] != [len(c) for c in conversations]:
        raise ValueError(f"{path} does not match the trace's conversations: rebuild it")
    return keys


def check_settings(*, source: str, key: str, min_samples: int, sigma_floor: float) -> None:
    """Raise ValueError unless these are valid list settings."""
    if source not in SOURCES or key not in KEYS or min_samples < 2 or not 0 < sigma_floor <= MAX_SIGMA_FLOOR:
        raise ValueError(f"bad hint settings: source {source} (one of {SOURCES}), key {key} (one of {KEYS}), "
                         f"min_samples {min_samples} (>= 2), sigma_floor {sigma_floor} (> 0, <= {MAX_SIGMA_FLOOR:g})")


def check_knobs(env, *, trace: str | None = None) -> None:
    """Raise ValueError unless the HINT_* knobs set in `env` (empty = unset) are values replay_agentic.py accepts, and
    the trace has the reply keys they need."""
    def value(knob: str, default):
        return env.get(knob) or default
    try:
        min_samples = int(value("HINT_MIN_SAMPLES", DEFAULTS["min_samples"]))
        sigma_floor = float(value("HINT_SIGMA_FLOOR", DEFAULTS["sigma_floor"]))
    except ValueError as e:
        raise ValueError(f"HINT_MIN_SAMPLES or HINT_SIGMA_FLOOR: {e}") from None
    source, key = value("HINT_SOURCE", DEFAULTS["source"]), value("HINT_KEY", DEFAULTS["key"])
    check_settings(source=source, key=key, min_samples=min_samples, sigma_floor=sigma_floor)
    for knob, allowed in (("HINT_POINT", POINTS), ("HINT_END", ENDS)):
        if value(knob, allowed[0]) not in allowed:
            raise ValueError(f"{knob}={env[knob]}: one of {allowed}")
    if trace is not None and needs_reply_keys(source=source, key=key) and not os.path.exists(reply_keys_path(trace)):
        raise ValueError(f"{reply_keys_path(trace)} is missing: build it with scripts/reply_keys.py")


def needs_reply_keys(*, source: str, key: str) -> bool:
    return source != "none" and key != "none"


def _sig(x: float) -> float:
    # Significant digits, not decimal places: a quantile is positive however small, and a reader may take its log.
    return float(f"{x:.6g}")


def _quantile(sorted_xs: list, p: float) -> float:
    """Linear interpolation between order statistics."""
    pos = p * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def _entry(key: str, *, ln_quantiles: tuple, n: int) -> dict:
    return {"key": key, "p10": _sig(math.exp(ln_quantiles[0])), "p50": _sig(math.exp(ln_quantiles[1])),
            "p90": _sig(math.exp(ln_quantiles[2])), "n": n}


def _ln_gap(gap_s: float) -> float:
    return math.log(max(gap_s, GAP_FLOOR_S))


class RecordedSession(NamedTuple):
    """What a pre-calculated table learns from one recorded session."""

    task: str
    source: str
    model: str
    keys: list | None   # keys[t]: the key of reply t; None without reply keys
    gaps: list          # gaps[t]: the gap after reply t, one fewer than the turns


def recorded_sessions(data: dict, *, keys: list | None, gap_of) -> list:
    """The trace's non-empty conversations as RecordedSession, by index in the trace (None for an empty one)."""
    out = []
    for i, (meta, conv) in enumerate(zip(data["sessions"], data["conversations"])):
        out.append(RecordedSession(task=task_of(meta["session_id"]), source=meta["source"], model=meta["orig_model"],
                                   keys=None if keys is None else keys[i], gaps=[gap_of(t) for t in conv[1:]])
                   if conv else None)
    return out


class GapTable:
    """The pre-calculated list: empirical quantiles of recorded gaps per (source, model) and key. entries() leaves out
    one task's sessions, so a session is never estimated from a run of its own task."""

    def __init__(self, sessions: list, *, key_mode: str) -> None:
        pools: dict = {}
        for s in sessions:
            if s is None:
                continue
            for t, gap in enumerate(s.gaps):
                x = _ln_gap(gap)
                for key in key_levels(None if s.keys is None else s.keys[t], key_mode):
                    # a coarser pool fills only what a model's own lacks: the source's tool entries, then the default
                    for pool in ((s.source, s.model), (s.source,), ())[:_coarsest_pool(key)]:
                        pools.setdefault(pool, {}).setdefault(key, []).append((x, s.task))
        self._samples = {pool: {key: sorted(v) for key, v in by_key.items()} for pool, by_key in pools.items()}
        self._task_counts = {pool: {key: _count_tasks(v) for key, v in by_key.items()}
                             for pool, by_key in self._samples.items()}
        self._full: dict = {}

    def _pooled_entry(self, pool: tuple, key: str, task: str) -> dict | None:
        samples = self._samples[pool][key]
        if self._task_counts[pool][key].get(task, 0):
            return _entry_of(key, [x for x, t in samples if t != task])
        cached = self._full.get((pool, key), 0)
        if cached == 0:
            cached = self._full[(pool, key)] = _entry_of(key, [x for x, _ in samples])
        return cached

    def entries(self, *, source: str, model: str, task: str) -> list:
        """The list a session of `task` carries, in key order, with the end entry."""
        listed: dict = {}
        for pool in ((source, model), (source,), ()):
            for key in self._samples.get(pool, {}):
                if key not in listed:
                    entry = self._pooled_entry(pool, key, task)
                    if entry is not None:
                        listed[key] = entry
        return [listed[k] for k in sorted(listed)] + [END_ENTRY]


def _coarsest_pool(key: str) -> int:
    return 3 if key == ANY_KEY else (1 if KEY_SEP in key else 2)


def _count_tasks(samples: list) -> dict:
    counts: dict = {}
    for _, task in samples:
        counts[task] = counts.get(task, 0) + 1
    return counts


def _entry_of(key: str, sorted_xs: list) -> dict | None:
    if len(sorted_xs) < MIN_POOLED:
        return None
    return _entry(key, ln_quantiles=tuple(_quantile(sorted_xs, p) for p in TAUS), n=len(sorted_xs))


class GapHistory:
    """The list from one session's own slept gaps: a Gaussian of ln(gap) per key with enough of them."""

    def __init__(self, *, key_mode: str = DEFAULTS["key"], min_samples: int = DEFAULTS["min_samples"],
                 sigma_floor: float = DEFAULTS["sigma_floor"]) -> None:
        check_settings(source="session", key=key_mode, min_samples=min_samples, sigma_floor=sigma_floor)
        self.key_mode, self.min_samples, self.sigma_floor = key_mode, min_samples, sigma_floor
        self._xs: dict = {}

    def observe(self, *, key: str | None, gap_s: float) -> None:
        x = _ln_gap(gap_s)
        for k in key_levels(key, self.key_mode):
            self._xs.setdefault(k, []).append(x)

    def entries(self) -> list:
        out = []
        for key in sorted(self._xs):
            xs = self._xs[key]
            if len(xs) >= self.min_samples:
                mu, sigma = fmean(xs), max(stdev(xs), self.sigma_floor)
                out.append(_entry(key, ln_quantiles=tuple(mu + sigma * _STD_NORMAL.inv_cdf(p) for p in TAUS), n=len(xs)))
        return out + [END_ENTRY]


class OnlineGapTable:
    """One agent system's list, refined by each gap as it finishes: empirical quantiles per key. Its sessions share one
    table, so a request carries what has finished anywhere in the system by then."""

    def __init__(self, *, key_mode: str = DEFAULTS["key"]) -> None:
        check_settings(source="online", key=key_mode, min_samples=DEFAULTS["min_samples"],
                       sigma_floor=DEFAULTS["sigma_floor"])
        self.key_mode = key_mode
        self._xs: dict = {}

    def observe(self, *, key: str | None, gap_s: float) -> None:
        x = _ln_gap(gap_s)
        for k in key_levels(key, self.key_mode):
            bisect.insort(self._xs.setdefault(k, []), x)

    def entries(self) -> list:
        listed = (_entry_of(key, self._xs[key]) for key in sorted(self._xs))
        return [e for e in listed if e is not None] + [END_ENTRY]


class TurnHint(NamedTuple):
    """What request `turn` carries, and what the engine does with it once the reply exists."""

    turn: int
    reply_key: str | None   # the key of the recorded reply; END_KEY where the agent knows the session ends
    entries: list           # the list this request carries
    picked: dict | None     # the entry the engine picks for reply_key; None with nothing to pick
    true_gap_s: float | None   # the recorded gap that follows, for the analysis, never sent; None after the last turn

    @property
    def ended(self) -> bool:
        return self.picked is not None and self.picked.get("ended", False)


class SessionHinter:
    """Steps through one recorded conversation as the replay client does. Call step(t) once per turn, in order, after
    the gap before turn t has elapsed and before turn t is sent. `turns` is the whole recorded conversation, also when
    the replay sends only its first turns; `calls` is the number of calls the recorded agent made (the trace's
    n_turns_total), more than len(turns) when the converter cut it. `entries` is a fixed list, `history` a table refined
    by gaps as they pass (GapHistory: this session's; OnlineGapTable: shared across its agent system); with neither, only
    the end is known."""

    def __init__(self, turns: list, *, keys: list | None, gap_of, entries: list | None = None,
                 history: GapHistory | None = None, end: str = "explicit", calls: int | None = None) -> None:
        if end not in ENDS:
            raise ValueError(f"end {end}: one of {ENDS}")
        calls = len(turns) if calls is None else calls
        self.turns, self.keys, self.gap_of, self.history = turns, keys, gap_of, history
        self._entries = [END_ENTRY] if entries is None and history is None else entries
        self._by_key = None if self._entries is None else {e["key"]: e for e in self._entries}
        # whether the agent knows that the recorded conversation's last turn ends its session
        self._end_known = calls == len(turns) and (end == "explicit" or (end == "natural" and calls < CALL_CAP))
        self._next_turn = 0

    def _key(self, t: int) -> str | None:
        return None if self.keys is None else self.keys[t]

    def step(self, t: int) -> TurnHint:
        if t != self._next_turn:
            raise ValueError(f"step({t}) out of order: expected step({self._next_turn})")
        self._next_turn += 1
        if t > 0 and self.history is not None:
            self.history.observe(key=self._key(t - 1), gap_s=self.gap_of(self.turns[t]))
        if t + 1 < len(self.turns):
            key, true_gap = self._key(t), self.gap_of(self.turns[t + 1])
        else:
            key, true_gap = (END_KEY if self._end_known else None), None
        entries = self._entries if self.history is None else self.history.entries()
        by_key = self._by_key if self.history is None else {e["key"]: e for e in entries}
        return TurnHint(turn=t, reply_key=key, entries=entries, picked=pick_entry(by_key, key), true_gap_s=true_gap)


def v1_point_s(hint: TurnHint, *, v1_point: str) -> float | None:
    """The next_gap_s of the version "1" action sent with this hint; None when none is sent."""
    if v1_point == "none" or hint.picked is None or hint.ended:
        return None
    return hint.picked["p50"]


def next_turn_envelope(*, message_id: str, session: str, next_rid: str, hint: TurnHint, v1_point: str) -> dict:
    """kv_hints for one turn: the version "2" list with the recorded reply's key, then, when the engine would pick an
    estimate and v1_point is not "none", that estimate's median as version "1"."""
    payload = {"session": session, "next_rid": next_rid, "gaps": hint.entries, "replay_reply_key": hint.reply_key}
    actions = [{"action_id": "next_turn", "action_type": NEXT_TURN_ACTION, "action_version": "2", "payload": payload}]
    point = v1_point_s(hint, v1_point=v1_point)
    if point is not None:
        actions.append({"action_id": "next_turn_point", "action_type": NEXT_TURN_ACTION, "action_version": "1",
                        "payload": {"session": session, "next_rid": next_rid, "next_gap_s": point}})
    return {"protocol_version": "1", "message_id": message_id, "actions": actions}


def hint_record(hint: TurnHint, *, next_rid: str, v1_point: str) -> dict:
    """The turn record's dist_hint field: the reply's key, the entry the engine would pick for it (matched is that
    entry's key), the v1 point sent, and the recorded gap that follows (true_gap_s), which was not sent."""
    rec = {"next_rid": next_rid, "reply_key": hint.reply_key, "list_len": len(hint.entries),
           "matched": None if hint.picked is None else hint.picked["key"], "point_s": v1_point_s(hint, v1_point=v1_point),
           "true_gap_s": None if hint.true_gap_s is None else round(hint.true_gap_s, 3)}
    if hint.ended:
        rec["ended"] = True
    elif hint.picked is not None:
        rec.update({k: hint.picked[k] for k in ("p10", "p50", "p90", "n")})
    return rec
