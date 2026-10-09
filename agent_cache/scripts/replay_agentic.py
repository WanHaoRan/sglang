"""Gap-faithful multi-turn replay client for SGLang (RUNBOOK 4.6). Stdlib + aiohttp, nothing under python/.

One conversation = one closed-loop session: send turn 0, stream the reply, append the reply text
VERBATIM to the history (with the 4.5 replay template that makes its KV a prefix hit), sleep the
recorded pre_gap (scaled / capped / sampled) measured from the END of the reply, send turn 1 = history
+ its new messages (tool results + user prompt), and so on. --concurrency conversations are live at
once (--cap-mode backlog moves that cap, below); the next one starts when one finishes. Every turn
writes one JSON line (kind=turn): timings, usage, the cached-token tier split, and reply_reprefilled =
uncached tokens beyond this turn's delta (0 +- one page when the previous reply was reused). With
--check-ids the server also returns token ids and the line carries the exact number of previous-reply
ids found on the new prompt's prefix.
Optional paired recompute control (kind=control, RUNBOOK 6.5): every K turns a random-id prompt of the
turn's prompt_tokens length is sent to /generate outside the semaphore; its TTFT is the recompute
reference for that turn.

--oracle-hints (oracle admission-anchored staging, agent_cache/ORACLE_PLAN.md): every turn but the last carries a
`kv_hints` envelope with one `sglang.next_turn` v1 action, payload {session, next_rid, next_gap_s}: the conversation id,
the next turn's rid and the gap the client will sleep before sending it (drawn one turn early; refused with --gap-sample,
whose draws depend on event order). A hint is suppressed when --max-seconds certainly cuts the next turn, or when a
context action (below) comes before the next turn. The turn record carries oracle_hint = {next_rid, next_gap_s,
suppressed}.

--turns-range LO:HI varies conversation lengths: each conversation replays min(its recorded turns, a uniform draw in
[LO, HI]) turns. The draws come from their own RNG (--turns-seed), one per conversation in replay order, so the arrival
and gap draws are unchanged and every arm replays the same lengths.

Users and sessions. Each user (conv id) replays one session; by default a session is one recorded conversation (a
task) and there is one user per selected conversation. User i opens with the selection's task i (cycled).
--session-tasks LO:HI chains a seeded uniform draw of K tasks into one growing context; the later tasks come from a cycle
over the selection's tasks with the same source and system prompt, recordings that start mid-run excluded. A later task
drops its system message; its turn 0 messages follow the previous reply after an idle drawn from that group's recorded
gaps (--task-gap < 0) or a fixed --task-gap, in trace seconds scaled and capped like any gap. Turn indices and rids
continue across tasks. --arrival-until T keeps the Poisson arrivals (--arrival-rate) going until T seconds after the
start; --concurrency caps the live users and later arrivals wait (queued_s), so arrivals above the server's capacity hold
the live population at --concurrency; arrivals the wall cap cuts off get conv_skipped records. --stop-after-users N
ends the run like the wall cap once N users have left on their own, so with a held population every arm runs the same
number of population turnovers instead of the same wall time. A task that opens a second session gets a salt in its first user message, at the start of the line where that message leaves the prefix it shares
with another task of its group (the end of the message when another task has the same message, its start when the task
is alone in its group), so two users share about what two different tasks would. User i's session depends only on i,
the selection, --session-seed and --order-seed. --order shuffle keeps users 0..P-1 (P tasks selected) as they are and
gives every later block of P users its own permutation of the tasks (seeded by --order-seed and the block), so the
offered load no longer repeats once per turnover of a held population.

--cap-mode backlog sets the cap from what the server serves instead of holding --concurrency. The backlog is the number
of chat requests sent and still waiting for their first token (or failure), counted from the client's own events and
averaged over time. Every --cap-interval seconds (the first one interval after the start) the cap moves by
clamp(round(--cap-gain x (--cap-target - mean backlog)), +-(--cap-step)), but only while it limits the population (live
plus waiting users reach it; before that, as the first users arrive, it holds). It falls to no more than --cap-step below
the live users, rises to no more than --cap-step above them (or stays where it is if already higher), and stays in
[--cap-min, --cap-max]; it starts at --concurrency. A waiting user starts when fewer than the cap are live and
--admit-spacing seconds have passed since the previous start (FIFO in arrival order, as with the fixed cap); a lower cap
stops no live user, the population falls as users finish. Each update writes a kind=cap record; the summary adds the
cap's time-weighted mean, min and max.

--start-phase stationary starts users conv < --start-users (0 = --concurrency) mid-session, so the first population is
not one cohort at turn 0. The start step s is where a time-uniform point of the session falls (P(s = k) ~ gap_k +
--start-service, P(0) ~ --start-service), before the first planned context action, drawn from --start-seed and the user
alone, so every arm, --validate and --dry-run draw the same s. The first request is step s with a synthetic history
(each earlier step's messages and a placeholder reply of its output_length tokens, as the converter counted
replay_prompt_tokens), sent after a residual idle of U x gap_s. conv_start carries start_turn and start_idle, that turn
record mid_start and synthetic_turns; turns_done stays the session position (requests sent = turns_done - start_turn).

Context window (--context-policy, against --max-context; 0 = max_model_len from the server's /v1/models). The server
only refuses (HTTP 400 when prompt >= window or prompt + max_tokens > window); agents act before that. The client plans
every session's actions from the trace's token counts alone, so every arm takes the same actions and dry runs show them:
the per-turn deltas match the server exactly for the template the trace was counted with, and the client calibrates
other templates and the system blocks against the server's /v1/tokenize at start. When a turn's prompt plus its reply
would pass --context-trigger of the window (the window itself for end, and early enough for the summarization request
itself to fit for compact):
- none: send it anyway; the server's 400 aborts the session.
- end: the session ends before that turn (SWE-agent's exit_context, mini-swe-agent), the first turn included.
- compact: one summarization request first (--compact-tokens reply tokens), then the session continues from the system
  prompt, the summary and the newest turns up to --compact-keep tokens (Codex and Cline carry about 20K). --compact-mode
  cached sends the history plus an instruction, a prefix hit (Claude Code, Codex); cold sends a summarizer system prompt
  and the conversation without the agent's system prompt as one message, nearly a full prefill (Gemini CLI, OpenHands).
- truncate: the oldest turns after the session's first are dropped until the prompt is under --truncate-to of the
  window (legacy Cline, OpenAI truncation=auto).
Actions write kind=compaction / truncation records and mark the next turn with after_context_action; a session ended by
the window writes conv_end with reason="context". A streamed {"error": ...} event fails the turn like an HTTP error.
--validate checks the flags and plans every session without a server; run_compare.sh runs it before booting any arm.

Trace: the 3.3 converter JSON (conversations[i] = list of turns with messages, output_length, pre_gap,
prompt_tokens, replay_prompt_tokens).

--dry-run sends nothing and needs no server: it walks the same users and turns in replay order (one session after
another, no sleeping) and prints, per turn, the gap a live run would sleep (with --gap-sample: one seeded draw, exact
only for --concurrency 1 closed loop), the token counts from the trace, every new message verbatim (tool results and user
prompts), the planned context actions (system blocks predicted from characters) and the token budget of the reply the
server would generate (the reply itself is skipped). Nothing is written to --output; --dry-run-max-chars shortens long
messages. A stationary start lists its synthetic turns, then walks from its step s.
"""
from __future__ import annotations   # keep the aiohttp annotations lazy so --dry-run works without aiohttp

import argparse
import asyncio
import collections
import json
import math
import os
import random
import signal
import sys
import time
from typing import NamedTuple

try:
    import aiohttp
except ImportError:  # only the live replay needs it; --dry-run runs on a bare Python
    aiohttp = None

DELTA_HEADER_TOKENS = 64  # tolerance: prefix matching is page-floored (page 64)
POLICIES = ("none", "end", "compact", "truncate")
# Written for this client: what an agent asks for when it compacts its own context.
COMPACT_INSTRUCTION = ("The conversation is about to exceed the context window. Write a detailed summary of the work so "
                       "far: the task, the files and code examined or changed, the errors met and how they were fixed, "
                       "what remains to be done and the next step, so that the work can continue from the summary alone.")
SUMMARIZER_SYSTEM = "You summarize an agent's conversation so that the work can continue from the summary alone."
SUMMARY_PREFIX = "This session continues an earlier conversation that ran out of context. Summary of that conversation:\n\n"
CHAT_TURN_CLOSE_TOKENS = 2  # "<|im_end|>\n" after a reply re-fed as history (Qwen3 template)
GEN_PROMPT_TOKENS = 3       # "<|im_start|>assistant\n" (Qwen3 stock template; other templates are calibrated)
COMPACT_REQUEST_TOKENS = len(COMPACT_INSTRUCTION) // 3 + 8   # the instruction message, a little over its real size
PLACEHOLDER_TOKEN = " ok"   # convert_lmcache.py's reply placeholder: one Qwen3 token per repeat


class Task(NamedTuple):
    """One recorded conversation of the selection."""
    index: int                # position in the selection (after --offset / --num-conversations)
    session_id: str | None
    source: str | None
    midrun: bool              # the recording starts mid-run: turn 0 is a snapshot with dropped assistant calls
    turns: list


class Session(NamedTuple):
    """What one user replays: one or more tasks back to back in one growing context."""
    user: int
    tasks: tuple
    task_gaps: tuple          # trace seconds idle before each later task's first turn
    salt: str                 # inserted into the first task's first user message ('' = none)
    salt_at: int              # char offset of the salt in that message


class Step(NamedTuple):
    """One turn of a session."""
    task: int                 # position of the task in the session
    task_turn: int            # turn index within the task
    turn: dict                # the recorded turn
    messages: list            # what this turn appends to the history
    gap_raw: float            # trace seconds idle before it (0 for the session's first turn)
    delta: int                # prompt tokens beyond the previous prompt + reply: the previous reply's closing tokens,
                              # this turn's messages, the generation prompt; on the first step, the salt's tokens


class Action(NamedTuple):
    """The context action planned before a step."""
    kind: str                 # '' | 'compact' | 'truncate' | 'end'
    turns: int                # compact: newest turns carried over; truncate: oldest turns dropped
    tokens: int               # predicted prompt + reply tokens of the step (after the action)


NO_ACTION = Action("", 0, 0)


def parse_range(text: str, name: str, ap: argparse.ArgumentParser) -> tuple:
    try:
        lo, hi = (int(x) for x in text.split(":"))
    except ValueError:
        ap.error(f"{name} wants LO:HI, got {text!r}")
    if not 1 <= lo <= hi:
        ap.error(f"{name} wants 1 <= LO <= HI, got {text!r}")
    return lo, hi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--num-conversations", type=int, default=0, help="0 = all")
    ap.add_argument("--offset", type=int, default=0, help="rotate the conversation list")
    ap.add_argument("--max-turns", type=int, default=0, help="0 = all")
    ap.add_argument("--turns-range", default="",
                    help="LO:HI: each conversation replays min(recorded turns, a seeded uniform draw in [LO, HI]); '' = off")
    ap.add_argument("--turns-seed", type=int, default=0, help="seed of the --turns-range draws")
    ap.add_argument("--concurrency", type=int, default=1, help="live users (closed loop); with --arrival-rate, the cap")
    ap.add_argument("--arrival-rate", type=float, default=0.0,
                    help="open loop: start users as a Poisson process at this many per second (0 = closed loop)")
    ap.add_argument("--arrival-until", type=float, default=0.0,
                    help="with --arrival-rate: keep starting users until this many seconds after the start, cycling "
                         "through the selected conversations; 0 = one user per selected conversation")
    ap.add_argument("--session-tasks", default="",
                    help="LO:HI: each user chains a seeded uniform draw of K tasks into one growing context; "
                         "'' = one task per user")
    ap.add_argument("--session-seed", type=int, default=0, help="seed of the --session-tasks and task-gap draws")
    ap.add_argument("--task-gap", type=float, default=-1.0,
                    help="trace seconds idle before a chained task's first turn (scaled and capped like any gap); "
                         "< 0 = drawn from the recorded within-task gaps of the task's group")
    ap.add_argument("--max-context", type=int, default=0,
                    help="context window the client manages, tokens; 0 = the server's max_model_len (/v1/models)")
    ap.add_argument("--context-policy", choices=POLICIES, default="none",
                    help="what a session does when its next prompt would not fit (module docstring)")
    ap.add_argument("--context-trigger", type=float, default=0.9,
                    help="compact / truncate when the next prompt plus its reply would pass this fraction of the window "
                         "(Codex, Zed, Amp 0.9; Cline 0.81; Copilot CLI, Goose 0.8; Gemini CLI 0.5; Claude Code about "
                         "window - 33K)")
    ap.add_argument("--context-margin", type=int, default=512,
                    help="tokens of slack on every predicted prompt (the server refuses from window - 6 tokens on)")
    ap.add_argument("--compact-tokens", type=int, default=4096, help="summary length of a compaction, tokens")
    ap.add_argument("--compact-keep", type=int, default=20000,
                    help="tokens of the newest turns carried over after the summary (whole turns)")
    ap.add_argument("--compact-mode", choices=("cached", "cold"), default="cached",
                    help="cached: the summarization request is history + instruction; cold: summarizer system prompt "
                         "plus the history as one message")
    ap.add_argument("--truncate-to", type=float, default=0.5,
                    help="truncate: drop the oldest turns until the next prompt is under this fraction of the window")
    ap.add_argument("--gap-scale", type=float, default=1.0, help="sleep pre_gap x scale; 0 = no idle window")
    ap.add_argument("--gap-cap", type=float, default=0.0, help="cap each sleep at this many seconds; 0 = no cap")
    ap.add_argument("--gap-sample", default="", help="JSON list of gap seconds; sample per turn instead of pre_gap")
    ap.add_argument("--thinking", choices=["off", "on"], default="off", help="must match the server's replay template")
    ap.add_argument("--control-every", type=int, default=0, help="K: one recompute control every K returning turns; 0 = off")
    ap.add_argument("--check-ids", action="store_true", help="ask for sglext input/output ids and verify reply reuse exactly")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="wall cap: after this many seconds no conversation starts and running ones stop at their next turn boundary (the gap sleep never crosses the deadline, in-flight requests complete); 0 = none")
    ap.add_argument("--stop-after-users", type=int, default=0,
                    help="stop once this many users have left on their own (as at --max-seconds: running sessions stop "
                         "at their next turn boundary, waiting and later users are skipped); with arrivals above "
                         "capacity this measures a fixed number of population turnovers; 0 = off")
    ap.add_argument("--cap-mode", choices=("fixed", "backlog"), default="fixed",
                    help="fixed: --concurrency caps the live users; backlog: the cap follows the backlog of requests "
                         "awaiting their first token, starting at --concurrency (module docstring)")
    ap.add_argument("--cap-target", type=float, default=32.0, help="backlog: the mean backlog (requests) to steer to")
    ap.add_argument("--cap-interval", type=float, default=120.0, help="backlog: seconds between two cap updates")
    ap.add_argument("--cap-gain", type=float, default=0.5, help="backlog: cap change per request of backlog error")
    ap.add_argument("--cap-step", type=int, default=8,
                    help="backlog: largest cap change per update, and largest distance of the cap from the live users")
    ap.add_argument("--cap-min", type=int, default=16, help="backlog: lowest cap")
    ap.add_argument("--cap-max", type=int, default=512, help="backlog: highest cap")
    ap.add_argument("--admit-spacing", type=float, default=5.0, help="backlog: least seconds between two user starts")
    ap.add_argument("--order", choices=("cycle", "shuffle", "fresh"), default="cycle",
                    help="cycle: user i opens with task i mod P (P tasks selected); shuffle: users P and later open "
                         "with a seeded permutation of the tasks per block of P users; fresh: users 0..P-1 open with a "
                         "seeded permutation of the P tasks, each task once, and no user arrives after the P-th")
    ap.add_argument("--order-seed", type=int, default=0, help="seed of the --order shuffle permutations")
    ap.add_argument("--start-phase", choices=("zero", "stationary"), default="zero",
                    help="zero: every user starts at turn 0; stationary: the first --start-users users start at a "
                         "time-uniform point of their session (module docstring)")
    ap.add_argument("--start-users", type=int, default=0,
                    help="stationary: users conv < this start mid-session; 0 = --concurrency")
    ap.add_argument("--start-seed", type=int, default=0, help="seed of the stationary start draws")
    ap.add_argument("--start-service", type=float, default=20.0,
                    help="stationary: seconds a request takes (TTFT + decode) in the time-uniform draw; a client "
                         "constant, so every arm draws the same starts")
    ap.add_argument("--progress-every", type=float, default=60.0,
                    help="with --arrival-until, seconds between progress lines (arrived / live / waiting / done); 0 = off")
    ap.add_argument("--save-text", action="store_true", help="store each reply text in the JSONL (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--oracle-hints", action="store_true",
                    help="send each non-last turn's next rid and next gap as a kv_hints envelope (sglang.next_turn v1)")
    ap.add_argument("--output", default="", help="JSONL, appended (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="send nothing: print each turn's messages and the token budget of the reply the server "
                         "would generate, then exit")
    ap.add_argument("--dry-run-max-chars", type=int, default=0,
                    help="with --dry-run, print at most this many chars of each message; 0 = the whole message")
    ap.add_argument("--validate", action="store_true",
                    help="check the flags against the trace and --max-context, print the session plan's totals, exit")
    args = ap.parse_args()
    if not args.dry_run and not args.validate and not args.output:
        ap.error("--output is required (unless --dry-run or --validate)")
    if args.turns_range:
        parse_range(args.turns_range, "--turns-range", ap)
    args.session_range = parse_range(args.session_tasks, "--session-tasks", ap) if args.session_tasks else (1, 1)
    if args.oracle_hints and args.gap_sample:
        ap.error("--oracle-hints needs deterministic gaps: --gap-sample draws depend on event order")
    if args.arrival_until and args.arrival_rate <= 0:
        ap.error("--arrival-until needs --arrival-rate")
    if args.cap_mode == "backlog" and not 1 <= args.cap_min <= args.concurrency <= args.cap_max:
        ap.error("--cap-mode backlog wants 1 <= --cap-min <= --concurrency (the starting cap) <= --cap-max")
    if args.cap_mode == "backlog" and not (args.cap_interval > 0 and args.cap_step >= 1 and args.cap_gain > 0
                                           and args.cap_target >= 0 and args.admit_spacing >= 0):
        ap.error("--cap-mode backlog wants --cap-interval > 0, --cap-step >= 1, --cap-gain > 0, --cap-target >= 0 and "
                 "--admit-spacing >= 0")
    if args.start_phase == "stationary" and args.gap_sample:
        ap.error("--start-phase stationary needs deterministic gaps: --gap-sample draws depend on event order")
    if args.start_phase == "stationary" and (args.start_users < 0 or args.start_service <= 0):
        ap.error("--start-phase stationary wants --start-users >= 0 and --start-service > 0")
    if args.context_policy in ("compact", "truncate") and not 0 < args.context_trigger <= 1:
        ap.error("--context-trigger wants 0 < it <= 1")
    if args.context_policy == "truncate" and not 0 < args.truncate_to < args.context_trigger:
        ap.error("--context-policy truncate wants 0 < --truncate-to < --context-trigger")
    if args.context_policy != "none" and (args.dry_run or args.validate) and not args.max_context:
        ap.error("--dry-run / --validate with --context-policy need --max-context (there is no server to ask)")
    return args


def content_text(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else json.dumps(c)


def system_text(turn0: dict) -> str:
    return next((content_text(m) for m in turn0["messages"] if m.get("role") == "system"), "")


def system_texts(pool: list) -> set:
    return {system_text(t.turns[0]) for t in pool} - {""}


def load_pool(args: argparse.Namespace) -> list:
    """The selected tasks in replay order: empty conversations dropped, then --offset, --num-conversations, --max-turns,
    --turns-range."""
    with open(args.trace) as f:
        data = json.load(f)
    convs = data["conversations"]
    ids = data.get("session_ids")
    if not ids or len(ids) != len(convs):
        ids = [None] * len(convs)
    pairs = [(sid, c) for sid, c in zip(ids, convs) if c]
    if args.offset:
        k = args.offset % len(pairs)
        pairs = pairs[k:] + pairs[:k]
    if args.num_conversations:
        pairs = pairs[: args.num_conversations]
    if args.max_turns:
        pairs = [(sid, c[: args.max_turns]) for sid, c in pairs]
    if args.turns_range:
        lo, hi = (int(x) for x in args.turns_range.split(":"))
        rng = random.Random(args.turns_seed)   # its own stream: the arrival and gap draws are unchanged
        pairs = [(sid, c[: rng.randint(lo, hi)]) for sid, c in pairs]
    return [Task(k, sid, c[0].get("source"), int(c[0].get("n_assistant_in_delta") or 0) > 0, c)
            for k, (sid, c) in enumerate(pairs)]


def group_of(task: Task) -> tuple:
    """Tasks of one group share a source and a system prompt: what a session may chain, where salts are placed."""
    return task.source, system_text(task.turns[0])


def lcp(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def first_user_message(turns: list) -> int | None:
    """Index in turn 0's messages of the first non-system message with text content (where a salt goes)."""
    for k, m in enumerate(turns[0]["messages"]):
        if m.get("role") != "system" and isinstance(m.get("content"), str):
            return k
    return None


def salt_offsets(pool: list) -> dict:
    """Task index -> char offset of a salt in its first user message: the start of the line on which that message
    leaves the longest prefix it shares with another task of its group (the end of the message when another task has
    the same message; its start when the task is alone in its group, so a copy shares the system prompt only)."""
    text, groups = {}, {}
    for t in pool:
        k = first_user_message(t.turns)
        text[t.index] = t.turns[0]["messages"][k]["content"] if k is not None else ""
        groups.setdefault(group_of(t), []).append(t.index)
    out = {}
    for idxs in groups.values():
        uses: dict = {}
        for i in idxs:
            uses[text[i]] = uses.get(text[i], 0) + 1
        distinct = sorted(uses)
        shared = {s: len(s) if uses[s] > 1 else 0 for s in distinct}
        for a, b in zip(distinct, distinct[1:]):   # the longest shared prefix is always with a sorted neighbour
            n = lcp(a, b)
            shared[a], shared[b] = max(shared[a], n), max(shared[b], n)
        for i in idxs:
            s, n = text[i], shared[text[i]]
            out[i] = len(s) if n >= len(s) else s.rfind("\n", 0, n) + 1
    return out


def char_block_tokens(turn0: dict) -> int:
    """Rough tokens of turn 0's system block: its replay_prompt_tokens split by characters (overestimates by a few
    hundred; the live run asks the server instead)."""
    sizes = [len(content_text(m)) for m in turn0["messages"]]
    sys_chars = sum(n for m, n in zip(turn0["messages"], sizes) if m.get("role") == "system")
    return round(int(turn0.get("replay_prompt_tokens") or 0) * sys_chars / max(1, sum(sizes)))


def calibration_inputs(pool: list) -> list:
    """(messages, replay_prompt_tokens) pairs whose counts under the serving template, against the trace's, give the
    template offsets: one task's turn 0, and its turn 0 + placeholder reply + turn 1 (the converter's history)."""
    t = next((x for x in pool if len(x.turns) >= 2 and not x.midrun), None)
    if t is None:
        return []
    t0, t1 = t.turns[0], t.turns[1]
    reply = {"role": "assistant", "content": PLACEHOLDER_TOKEN * int(t0.get("output_length") or 1)}
    return [(list(t0["messages"]), int(t0["replay_prompt_tokens"])),
            (list(t0["messages"]) + [reply] + list(t1["messages"]), int(t1["replay_prompt_tokens"]))]


def template_offsets(counts: list, pool: list) -> tuple:
    """(per prompt, per re-fed reply) tokens the serving template adds over the template the trace was counted with,
    from the counts of calibration_inputs; (0, 0) for the trace's own template."""
    ref = [r for _, r in calibration_inputs(pool)]
    if len(counts) != 2 or len(ref) != 2:
        return 0, 0
    prompt = counts[0] - ref[0]
    return prompt, counts[1] - ref[1] - prompt


def tokenizer_counter(model: str, thinking: bool):
    """A messages -> tokens counter from the model's own tokenizer and default chat template when transformers is
    installed (as in the container), else None; the live run asks the server's /v1/tokenize instead."""
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception:  # noqa: BLE001 - no transformers or no tokenizer files: fall back to the character estimate
        return None

    def count(messages: list) -> int:
        # rendered, then encoded, as convert_lmcache.count_tokens does (apply_chat_template(tokenize=True) returns a dict
        # on newer transformers)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        return len(tok(text, add_special_tokens=False)["input_ids"])
    return count


class SessionPlanner:
    """Hands out users in order: user i opens with task i of the cycled selection (--order shuffle: of a seeded
    permutation per block of P users after the first) and, with --session-tasks, chains K - 1 more from the cycle over
    its group's tasks that start at the beginning of a run."""

    def __init__(self, pool: list, args: argparse.Namespace) -> None:
        self.pool = pool
        self.session_range = args.session_range
        self.task_gap = args.task_gap
        self.seed = args.session_seed
        self.shuffle = args.order in ("shuffle", "fresh")
        self.fresh = args.order == "fresh"
        self.order_seed = args.order_seed
        self.perms: dict = {}   # block -> permutation of the pool's positions (--order shuffle)
        self.rng = random.Random(args.session_seed)   # its own stream: arrivals and gaps are unchanged
        self.chain: dict = {}   # group -> tasks later tasks are drawn from, in selection order
        self.gaps: dict = {}    # group -> recorded within-task gaps, trace seconds
        for t in pool:
            g = group_of(t)
            if not t.midrun:
                self.chain.setdefault(g, []).append(t)
            self.gaps.setdefault(g, []).extend(float(x.get("pre_gap") or 0.0) for x in t.turns[1:])
        self.cursor = dict.fromkeys(self.chain, 0)
        self.opened: dict = {}   # task index -> sessions it opened so far
        self.salt_at = salt_offsets(pool)
        self.next_user = 0

    def chained(self, first: Task, k: int) -> list:
        """The session's tasks: `first`, then up to k - 1 distinct tasks of its group's cycle."""
        tasks, g = [first], group_of(first)
        cycle = self.chain.get(g, [])
        while len(tasks) < k:
            pick = None
            for _ in range(len(cycle)):
                t = cycle[self.cursor[g] % len(cycle)]
                self.cursor[g] += 1
                if all(t.index != x.index for x in tasks):
                    pick = t
                    break
            if pick is None:   # the group has fewer distinct tasks than K
                break
            tasks.append(pick)
        return tasks

    def opener(self, user: int) -> Task:
        """The task user `user` opens with: block 0 (users 0..P-1) task j = user mod P; with --order shuffle, later
        blocks take position j of a permutation seeded by --order-seed and the block alone, the same in every arm."""
        blk, j = divmod(user, len(self.pool))
        if not self.shuffle or (blk == 0 and not self.fresh):
            return self.pool[j]
        if blk not in self.perms:
            self.perms[blk] = random.Random(f"{self.order_seed}:{blk}").sample(range(len(self.pool)), len(self.pool))
        return self.pool[self.perms[blk][j]]

    def next_session(self) -> Session:
        user = self.next_user
        self.next_user += 1
        first = self.opener(user)
        k = 1 if self.session_range == (1, 1) else self.rng.randint(*self.session_range)
        tasks = self.chained(first, k)
        pool_gaps = self.gaps.get(group_of(first)) or [0.0]
        gaps = tuple(self.task_gap if self.task_gap >= 0 else random.Random(f"{self.seed}:{user}:{j}").choice(pool_gaps)
                     for j in range(1, len(tasks)))
        reopened = self.opened.get(first.index, 0)
        self.opened[first.index] = reopened + 1
        if not reopened:
            return Session(user, tuple(tasks), gaps, "", 0)
        return Session(user, tuple(tasks), gaps, f"[session {user}]\n", self.salt_at.get(first.index, 0))


def session_steps(sess: Session, block_tokens, reply_offset: int = 0) -> list:
    """The turns of a session in order, with the messages each appends and its token delta; block_tokens(turn0) gives
    the tokens of a task's system block, reply_offset what the serving template adds to every re-fed reply."""
    steps = []
    for j, task in enumerate(sess.tasks):
        for u, turn in enumerate(task.turns):
            msgs = list(turn["messages"])
            if u == 0 and j > 0:
                msgs = [m for m in msgs if m.get("role") != "system"]
            if u == 0 and j == 0 and sess.salt:
                k = first_user_message(task.turns)
                if k is not None:
                    text = msgs[k]["content"]
                    salt = sess.salt if sess.salt_at < len(text) else "\n" + sess.salt.rstrip("\n")
                    msgs[k] = {**msgs[k], "content": text[: sess.salt_at] + salt + text[sess.salt_at:]}
            if u > 0:   # exact against the server for 99 % of turns (the rest by one token): campaign 12 x1
                prev = task.turns[u - 1]
                delta = int(turn.get("replay_prompt_tokens") or 0) - int(prev.get("replay_prompt_tokens") or 0) \
                    - int(prev.get("output_length") or 0) + reply_offset
                gap = float(turn.get("pre_gap") or 0.0)
            elif j > 0:   # a later task's turn 0 without its system block, after the previous reply's closing tokens
                delta = int(turn.get("replay_prompt_tokens") or 0) - block_tokens(turn) + CHAT_TURN_CLOSE_TOKENS \
                    + reply_offset
                gap = sess.task_gaps[j - 1]
            else:
                delta, gap = len(sess.salt) // 2, 0.0   # "[session 123]\n" is about one token per two chars
            steps.append(Step(j, u, turn, msgs, gap, delta))
    return steps


def plan_truncation(chunk_tokens: list, need: int, target: float) -> tuple:
    """How many turns after the first to drop, oldest first, so that `need` fits under `target`: (turns, tokens)."""
    drop = dropped = 0
    while 1 + drop < len(chunk_tokens) and need - dropped > target:
        dropped += chunk_tokens[1 + drop]
        drop += 1
    return drop, dropped


def context_triggers(args: argparse.Namespace, window: int) -> tuple:
    """(hard, trigger): no prompt + reply may pass hard; the policy acts when one would pass trigger. A compaction
    triggers early enough for its own request (history + instruction + summary) to fit under hard."""
    hard = window - args.context_margin
    if args.context_policy == "end":
        return hard, hard
    trigger = args.context_trigger * window - args.context_margin
    if args.context_policy == "compact":
        trigger = min(trigger, hard - COMPACT_REQUEST_TOKENS - args.compact_tokens)
    return hard, trigger


def plan_context(steps: list, block: int, args: argparse.Namespace, window: int, prompt_offset: int = 0) -> list:
    """The action before each step and the step's predicted prompt + reply tokens, from the trace's token counts alone;
    the list stops at an 'end'. `block` is the session's system block in tokens, prompt_offset what the serving template
    adds to every prompt; chunks are the prompt + reply tokens each turn left in the history."""
    plan, chunks, base = [], [], 0
    hard, trigger = context_triggers(args, window)
    for g, st in enumerate(steps):
        out_len = max(1, int(st.turn.get("output_length") or 1))
        if g == 0:
            need = int(st.turn.get("replay_prompt_tokens") or 0) + prompt_offset + st.delta + out_len
            if need > hard:   # the first turn alone does not fit: nothing to compact or drop yet
                return [Action("end", 0, need)]
        else:
            need = base + st.delta + out_len
        kind, turns = "", 0
        if g and need > trigger:
            kind = args.context_policy
            if kind == "compact":
                if base + COMPACT_REQUEST_TOKENS + args.compact_tokens > hard:   # its own request would not fit
                    plan.append(Action("end", 0, need))
                    return plan
                while turns + 1 < len(chunks) and sum(chunks[len(chunks) - turns - 1:]) <= args.compact_keep:
                    turns += 1
                chunks = [block + args.compact_tokens + len(SUMMARY_PREFIX) // 4 + 8] + chunks[len(chunks) - turns:]
            elif kind == "truncate":
                turns, _ = plan_truncation(chunks, need, args.truncate_to * window)
                chunks = chunks[:1] + chunks[1 + turns:]
            base = sum(chunks)
            need = base + st.delta + out_len
            if kind == "end" or need > trigger:   # no room even after the action: the session ends here
                plan.append(Action("end", 0, need))
                return plan
        plan.append(Action(kind, turns, need))
        chunks.append(need - base)
        base = need
    return plan


def start_limit(steps: list, plan: list) -> int:
    """How many leading steps a stationary start may land on: those before the first planned context action or end
    (a synthetic history cannot carry an action's effect)."""
    return next((k for k, a in enumerate(plan) if a.kind), len(steps))


def start_point(steps: list, limit: int, gap_of, service: float, rng: random.Random) -> tuple:
    """(s, idle): the step among steps[:limit] that a time-uniform point of the session falls before, and the residual
    idle before it is sent. Step k's share of the session is its gap plus one request's `service` (step 0: the service)."""
    if limit <= 1:
        return 0, 0.0
    s = rng.choices(range(limit), weights=[service] + [gap_of(st) + service for st in steps[1:limit]])[0]
    return s, rng.random() * gap_of(steps[s])


def synthetic_history(steps: list, s: int) -> tuple:
    """(history, spans) of a session started at step s: each earlier step's messages, then a placeholder reply of its
    output_length tokens (the converter's own count of replay_prompt_tokens); spans as play() keeps them."""
    history, spans = [], []
    for st in steps[:s]:
        spans.append(len(history))
        reply = PLACEHOLDER_TOKEN * max(1, int(st.turn.get("output_length") or 1))
        history += list(st.messages) + [{"role": "assistant", "content": reply}]
    return history, spans


def start_prompt_tokens(steps: list, s: int, prompt_offset: int = 0) -> int:
    """Predicted prompt tokens of step s sent after the synthetic history: plan_context's count with no action before."""
    n = int(steps[0].turn.get("replay_prompt_tokens") or 0) + prompt_offset + steps[0].delta
    for prev, st in zip(steps[:s], steps[1:s + 1]):
        n += max(1, int(prev.turn.get("output_length") or 1)) + st.delta
    return n


def arrival_times(rng: random.Random, rate: float, until: float, n_default: int) -> list:
    """Poisson arrival offsets in seconds: n_default users, or every arrival before `until` when it is set."""
    times, t = [], 0.0
    while (t < until) if until else (len(times) < n_default):
        times.append(t)
        t += rng.expovariate(rate)
    return times


class AdmissionGate:
    """--cap-mode backlog's live-user cap in place of the semaphore: `async with` waits until fewer than `limit` users
    hold a slot and `spacing` seconds have passed since the previous admission, FIFO in arrival order. set_limit moves
    the limit and never revokes a slot. Once closed() (the run has stopped) every waiter goes through, to be skipped."""

    def __init__(self, limit: int, spacing: float, closed) -> None:
        self.limit = limit
        self.spacing = spacing
        self.closed = closed
        self.held = 0                  # slots held
        self.last = -math.inf          # loop time of the previous admission
        self.waiters: collections.deque = collections.deque()   # futures, in arrival order
        self.timer = None              # wake-up pending for the spacing

    async def __aenter__(self) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        self.pump()
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():   # admitted, then cancelled before it ran: hand the slot on
                self.release()
            raise

    async def __aexit__(self, *exc) -> None:
        self.release()

    def release(self) -> None:
        self.held -= 1
        self.pump()

    def set_limit(self, limit: int) -> None:
        self.limit = limit
        self.pump()

    def wake(self) -> None:
        self.timer = None
        self.pump()

    def pump(self) -> None:
        """Admit from the head of the queue while the limit and the spacing allow (everyone once closed)."""
        loop = asyncio.get_running_loop()
        while self.waiters:
            if self.waiters[0].done():   # its task was cancelled while waiting
                self.waiters.popleft()
                continue
            if not self.closed():
                if self.held >= self.limit:
                    return
                wait = self.last + self.spacing - loop.time()
                if wait > 0:
                    if self.timer is None:
                        self.timer = loop.call_later(wait, self.wake)
                    return
                self.last = loop.time()
            self.held += 1
            self.waiters.popleft().set_result(None)


class Replayer:
    def __init__(self, args: argparse.Namespace, out) -> None:
        self.args = args
        self.out = out
        self.rng = random.Random(args.seed)
        self.gap_pool = json.load(open(args.gap_sample)) if args.gap_sample else None
        self.sem = asyncio.Semaphore(args.concurrency) if args.cap_mode == "fixed" else \
            AdmissionGate(limit=args.concurrency, spacing=args.admit_spacing, closed=lambda: self.elapsed() > self.deadline)
        self.t0 = time.perf_counter()
        self.stats = {"turns": 0, "controls": 0, "errors": 0, "retries": 0, "capped_convs": 0, "ttft_sum": 0.0,
                      "reprefill_sum": 0, "reprefill_n": 0, "compactions": 0, "truncations": 0, "context_ends": 0,
                      "mid_start_users": 0, "mid_start_prompt_tokens": 0}
        self.backlog = 0           # chat requests sent and still waiting for their first token (or failure)
        self.backlog_area = 0.0    # its time integral up to backlog_t, request-seconds
        self.backlog_t = 0.0       # seconds since t0
        self.cap = args.concurrency   # the live-user cap; --cap-mode backlog moves it
        self.cap_area, self.cap_t = 0.0, 0.0   # its time integral up to cap_t
        self.cap_lo = self.cap_hi = self.cap
        self.admitted = 0          # users whose sessions started
        self.controls: list = []
        self.sessions_on = args.session_range != (1, 1) or bool(args.arrival_until)   # session fields on records
        self.chaining = args.session_range != (1, 1)
        self.window = args.max_context
        self.blocks: dict = {}   # system prompt text -> tokens of its rendered block (calibrate / calibrate_local)
        self.prompt_offset = self.reply_offset = 0   # serving template versus the trace's (calibrate / calibrate_local)
        self.counts = {"arrived": 0, "live": 0, "done": 0}
        self.deadline = args.max_seconds or math.inf   # seconds since t0; --stop-after-users moves it to the stop
        self.stop_reason = "max_seconds"
        self.finished = 0   # users whose sessions ended on their own (not cut by the deadline)
        self.stopped = asyncio.Event()

    def write(self, rec: dict) -> None:
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def count_backlog(self, d: int, t: float) -> None:
        """A chat request entered (+1) or left (-1) the backlog at perf_counter time t."""
        t -= self.t0
        self.backlog_area += self.backlog * (t - self.backlog_t)
        self.backlog, self.backlog_t = self.backlog + d, t

    def backlog_integral(self, now: float) -> float:
        """Request-seconds of backlog from t0 to `now` (seconds since t0)."""
        return self.backlog_area + self.backlog * (now - self.backlog_t)

    def user_left(self) -> None:
        """A user's session ended on its own; the --stop-after-users-th such user stops the run now."""
        self.finished += 1
        if self.args.stop_after_users and self.finished == self.args.stop_after_users and self.elapsed() < self.deadline:
            self.deadline, self.stop_reason = self.elapsed(), "stop_after_users"
            self.stopped.set()
            print(f"[{self.deadline:.0f} s] {self.finished} users finished: stopping (--stop-after-users)", flush=True)

    async def idle(self, seconds: float) -> None:
        """Sleep, waking early when --stop-after-users stops the run."""
        if not self.args.stop_after_users:
            await asyncio.sleep(seconds)
            return
        try:
            await asyncio.wait_for(self.stopped.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    def rid(self, conv: int, turn_idx: int) -> str:
        return f"{self.args.tag or 'replay'}-c{conv}-t{turn_idx}"

    def next_turn_hints(self, conv: int, turn_idx: int, next_gap_s: float) -> dict:
        """kv_hints envelope sent with turn turn_idx: when the conversation's next turn will be sent, and its rid."""
        payload = {"session": f"{self.args.tag or 'replay'}-c{conv}", "next_rid": self.rid(conv, turn_idx + 1),
                   "next_gap_s": round(next_gap_s, 3)}
        return {"protocol_version": "1", "message_id": self.rid(conv, turn_idx),
                "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn", "action_version": "1",
                             "payload": payload}]}

    def scale_gap(self, g: float) -> float:
        g *= self.args.gap_scale
        if self.args.gap_cap > 0:
            g = min(g, self.args.gap_cap)
        return max(0.0, g)

    def next_gap(self, pre_gap: float) -> float:
        return self.scale_gap(self.rng.choice(self.gap_pool) if self.gap_pool else pre_gap)

    def step_gap(self, step: Step) -> float:
        """The sleep before a step: a chained task's first turn idles its task gap (never --gap-sample), others their gap."""
        if step.task > 0 and step.task_turn == 0:
            return self.scale_gap(step.gap_raw)
        return self.next_gap(step.gap_raw)

    def block_tokens(self, turn0: dict) -> int:
        b = self.blocks.get(system_text(turn0))
        return b if b is not None else char_block_tokens(turn0)

    def steps_and_plan(self, sess: Session) -> tuple:
        """A session's steps and its context actions ([] with --context-policy none)."""
        steps = session_steps(sess, self.block_tokens, self.reply_offset)
        if self.args.context_policy == "none":
            return steps, []
        return steps, plan_context(steps, self.block_tokens(steps[0].turn), self.args, self.window, self.prompt_offset)

    def start_of(self, sess: Session, steps: list, plan: list) -> tuple:
        """(start step, residual idle seconds) of a session: (0, 0.0) unless --start-phase stationary starts this user
        mid-session; the draw depends on --start-seed and the user alone."""
        a = self.args
        if a.start_phase != "stationary" or sess.user >= (a.start_users or a.concurrency):
            return 0, 0.0
        return start_point(steps, start_limit(steps, plan), gap_of=lambda st: self.scale_gap(st.gap_raw),
                           service=a.start_service, rng=random.Random(f"{a.start_seed}:{sess.user}"))

    def set_calibration(self, counts: list, raw_blocks: dict, pool: list) -> None:
        """Offsets from the counts of calibration_inputs; system blocks from the counts of lone system messages."""
        self.prompt_offset, self.reply_offset = template_offsets(counts, pool)
        gen = GEN_PROMPT_TOKENS + self.prompt_offset   # the serving template's generation prompt
        self.blocks = {t: n - gen for t, n in raw_blocks.items()}

    def calibrate_local(self, pool: list) -> str:
        """Dry run and --validate: offsets and system blocks from the model's own tokenizer and default template when
        transformers is installed; returns where they came from."""
        count = tokenizer_counter(self.args.model, self.args.thinking == "on")
        if count is None:
            return ("estimated from characters (no transformers here); a planned action can land a turn away from the "
                    "live run, which asks the server")
        self.set_calibration([count(m) for m, _ in calibration_inputs(pool)],
                             {t: count([{"role": "system", "content": t}]) for t in system_texts(pool)}, pool)
        return (f"from the {self.args.model} tokenizer and its default template (prompt {self.prompt_offset:+d}, reply "
                f"{self.reply_offset:+d} tokens against the trace; the live run asks the server, whose template may differ)")

    def compaction_room(self, pool: list) -> str:
        """Why --context-policy compact would thrash in this window, or ''."""
        a = self.args
        if a.context_policy != "compact":
            return ""
        _, trigger = context_triggers(a, self.window)
        block = max((self.block_tokens(t.turns[0]) for t in pool), default=0)
        after = block + a.compact_tokens + a.compact_keep
        if after > 0.8 * trigger:
            return (f"a compaction leaves up to {after} tokens (system {block} + summary {a.compact_tokens} + kept "
                    f"{a.compact_keep}), over 80 % of the {trigger:.0f}-token trigger of a {self.window}-token window: "
                    f"lower --compact-keep or --compact-tokens")
        return ""

    def dry_run(self, planner: SessionPlanner, n_users: int) -> None:
        """Print what every turn would send and get back, in replay order; no request, no sleep, no JSONL line."""
        a = self.args
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")   # message content can hold anything; never die on the locale
        if getattr(signal, "SIGPIPE", None):
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` ends the dump quietly
        if a.context_policy != "none" or self.chaining or a.start_phase == "stationary":
            print(f"system blocks and template offsets: {self.calibrate_local(planner.pool)}", flush=True)
        if self.compaction_room(planner.pool):
            sys.exit(self.compaction_room(planner.pool))
        cap = f", capped at {a.gap_cap} s" if a.gap_cap else ""
        if self.gap_pool:
            gap_note = (f"gap per turn = one seeded draw from {a.gap_sample} x {a.gap_scale}{cap}: exactly what a live run at "
                        f"--concurrency 1 (closed loop) sleeps; at higher concurrency or with --arrival-rate the live run assigns "
                        f"the draws to turns differently")
            gap_label = "sampled gap"
        else:
            gap_note = f"gap per turn = pre_gap x {a.gap_scale}{cap} = exactly what a live run sleeps"
            gap_label = "a live run would sleep"
        live_only = "--concurrency, --arrival-rate, --control-every, --check-ids, --max-seconds" \
            + (", --cap-mode backlog" if a.cap_mode == "backlog" else "")
        mid = f"; users conv < {a.start_users or a.concurrency} start mid-session (--start-phase stationary)" \
            if a.start_phase == "stationary" else ""
        print(f"DRY RUN: nothing is sent to {a.url}, nothing slept, nothing written; conversations play one after another; "
              f"{gap_note}; live-only flags ({live_only}) have no effect{mid}", flush=True)
        n_msgs = n_turns = 0
        for _ in range(n_users):
            sess = planner.next_session()
            conv = sess.user
            steps, plan = self.steps_and_plan(sess)
            s, idle = self.start_of(sess, steps, plan)
            first = sess.tasks[0].turns[0]
            print(f"\n{'=' * 100}\nconv {conv}: session {sess.tasks[0].session_id or '?'}  source {first.get('source')}  "
                  f"recorded with {first.get('orig_model')}  turns {len(steps)}  "
                  f"final replay prompt {sess.tasks[0].turns[-1].get('replay_prompt_tokens')} tokens"
                  + (f"\n  tasks {[t.index for t in sess.tasks]}  task gaps {[round(x, 3) for x in sess.task_gaps]}"
                     f"{'  salt ' + repr(sess.salt) + ' at char ' + str(sess.salt_at) if sess.salt else ''}"
                     if self.sessions_on else "") + f"\n{'=' * 100}")
            # as play(): messages outside turns (system + summary), and each turn's messages + reply (synthetic ones first)
            fixed, counts = 0, [len(st.messages) + 1 for st in steps[:s]]
            walked = 0
            if s:
                print(f"\n--- conv {conv}: stationary start at turn {s}: the first request carries {s} synthetic turns "
                      f"(their messages, each with a placeholder reply), predicted prompt "
                      f"{start_prompt_tokens(steps, s, prompt_offset=self.prompt_offset)} tokens, after a residual idle of "
                      f"{idle:.3f} s")
                for j, st in enumerate(steps[:s]):
                    print(f"    synthetic turn {j}: {len(st.messages)} messages, "
                          f"{sum(len(content_text(m)) for m in st.messages)} chars, reply {PLACEHOLDER_TOKEN!r} x "
                          f"{max(1, int(st.turn.get('output_length') or 1))}")
            for turn_idx, st in enumerate(steps[s:], s):
                turn, msgs = st.turn, st.messages
                act = plan[turn_idx] if turn_idx < len(plan) else NO_ACTION
                max_tokens = max(1, int(turn.get("output_length") or 1))   # what conversation() + chat_turn() send
                # drawn before the end check, as play() does; a stationary start's first request waits its residual idle
                gap = (idle if turn_idx == s else self.step_gap(st)) if turn_idx > 0 else 0.0
                if act.kind == "end":
                    print(f"\n--- conv {conv}: the session ends before turn {turn_idx} "
                          f"(predicted prompt {act.tokens - max_tokens} tokens)")
                    break
                if act.kind == "compact":
                    fixed = int(steps[0].messages[0].get("role") == "system") + 1
                    counts = counts[len(counts) - act.turns:] if act.turns else []
                elif act.kind == "truncate":
                    counts = counts[:1] + counts[1 + act.turns:]
                if act.kind:
                    print(f"\n--- conv {conv}: {act.kind} before turn {turn_idx} ({act.turns} turns "
                          f"{'kept' if act.kind == 'compact' else 'dropped'})")
                merged = int(turn.get("merged_output_length") or 0)
                print(f"\n--- conv {conv} turn {turn_idx} [rid {self.rid(conv, turn_idx)}]  "
                      + (f"task {st.task} turn {st.task_turn}  " if self.sessions_on else "")
                      + f"trace iteration {turn.get('iteration')}, rows {turn.get('source_rows')}\n"
                      f"    gap: recorded pre_gap {float(turn.get('pre_gap') or 0.0):.3f} s, {gap_label} {gap:.3f} s"
                      f"{' (the residual idle of the stationary start)' if turn_idx == s > 0 else ''}\n"
                      f"    prompt: {fixed + sum(counts)} history + {len(msgs)} new messages = {turn.get('replay_prompt_tokens')} replay tokens "
                      f"(recorded {turn.get('prompt_tokens')}); {turn.get('n_assistant_in_delta')} recorded assistant message(s) "
                      f"dropped from this delta"
                      + (f"; planned session prompt {act.tokens - max_tokens} tokens" if plan else "") + "\n"
                      f"    reply: max_tokens {max_tokens}"
                      f"{' (+' + str(merged) + ' tokens of merged-forward calls, not replayed)' if merged else ''}")
                if a.oracle_hints and turn_idx + 1 < len(steps):
                    hint = self.next_turn_hints(conv, turn_idx, self.step_gap(steps[turn_idx + 1]))
                    if plan and (turn_idx + 1 >= len(plan) or plan[turn_idx + 1].kind):
                        print("    oracle hint (kv_hints): suppressed, a context action comes before the next turn")
                    else:
                        print(f"    oracle hint (kv_hints): {json.dumps(hint['actions'][0]['payload'])}")
                for k, m in enumerate(msgs, 1):
                    content = m.get("content")
                    if not isinstance(content, str):
                        content = json.dumps(content)
                    extra = {key: v for key, v in m.items() if key not in ("role", "content")}
                    shown = content
                    if a.dry_run_max_chars > 0 and len(content) > a.dry_run_max_chars:
                        shown = content[: a.dry_run_max_chars] + f"\n... [{len(content) - a.dry_run_max_chars} more chars]"
                    print(f"[{m.get('role')}] message {k}/{len(msgs)}, {len(content)} chars"
                          f"{' ' + json.dumps(extra) if extra else ''}\n{shown}")
                    n_msgs += 1
                print(f"[assistant] reply skipped (dry run): the server would generate {max_tokens} tokens here"
                      f"{'' if turn_idx == len(steps) - 1 else ', re-fed verbatim as history on the next turn'}")
                counts.append(len(msgs) + 1)
                walked += 1
                n_turns += 1
            print(f"conv {conv} done: {walked} turns (dry run)")
        print(f"\ndry run summary: {n_users} conversations, {n_turns} turns, {n_msgs} messages printed, "
              f"0 requests sent{', ' + a.output + ' not written' if a.output else ''}", flush=True)

    async def chat_turn(self, session: aiohttp.ClientSession, messages: list, max_tokens: int, rid: str = "",
                        kv_hints: dict | None = None) -> dict:
        """One streamed chat completion; returns text, timings, usage and sglext fields."""
        body = {
            "rid": rid or None,   # shows up in the server's --log-requests and HICACHE_EVT lines
            "model": self.args.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max(1, int(max_tokens)),
            "ignore_eos": True,
            "temperature": 0.0,
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": self.args.thinking == "on"},
            "return_cached_tokens_details": True,
        }
        if kv_hints is not None:
            body["kv_hints"] = kv_hints
        if self.args.check_ids:
            body["return_input_ids_in_sglext"] = True
            body["return_output_ids_in_sglext"] = True
        res = {"text": "", "ttft": None, "usage": None, "cached_details": None, "input_ids": None, "output_ids": None}
        t_send = time.perf_counter()
        self.count_backlog(1, t_send)   # in the backlog until its first token or its failure
        try:
            async with session.post(f"{self.args.url}/v1/chat/completions", json=body) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    d = json.loads(payload)
                    if d.get("error"):   # a scheduler abort after the 200 went out, e.g. a prompt within 6 tokens of the window
                        raise RuntimeError(f"stream error: {json.dumps(d['error'])[:200]}")
                    for ch in d.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece:
                            if res["ttft"] is None:
                                t_first = time.perf_counter()
                                res["ttft"] = t_first - t_send
                                self.count_backlog(-1, t_first)
                            res["text"] += piece
                    if d.get("usage"):
                        res["usage"] = d["usage"]
                    sglext = d.get("sglext") or {}
                    if sglext.get("cached_tokens_details"):
                        res["cached_details"] = sglext["cached_tokens_details"]
                    if sglext.get("input_ids"):
                        res["input_ids"] = sglext["input_ids"]
                    if sglext.get("output_ids"):
                        res["output_ids"] = sglext["output_ids"][0]
        finally:
            if res["ttft"] is None:   # failed, or ended without a token
                self.count_backlog(-1, time.perf_counter())
        res["t_send"] = t_send
        res["latency"] = time.perf_counter() - t_send
        return res

    async def chat_with_retry(self, session: aiohttp.ClientSession, messages: list, max_tokens: int, rid: str,
                              rec: dict, kv_hints: dict | None = None) -> tuple:
        """chat_turn with one retry on a pooled connection the server closed; returns (result or None, error)."""
        err = None
        for attempt in range(2):
            try:
                return await self.chat_turn(session, messages, max_tokens, rid=rid, kv_hints=kv_hints), None
            except aiohttp.ClientConnectionError as e:
                # a pooled keep-alive connection the server closed meanwhile (ServerDisconnected / [Errno 104]):
                # retry once on a fresh connection and say so in the record
                err = e
                if attempt == 0:
                    rec["retried"] = f"{type(e).__name__}: {str(e)[:120]}"
                    self.stats["retries"] += 1
            except Exception as e:  # noqa: BLE001
                return None, e
        return None, err

    async def control(self, session: aiohttp.ClientSession, conv: int, turn: int, n_tokens: int) -> None:
        """Recompute reference: a fresh random-id prompt of n_tokens, 1 output token, outside the semaphore."""
        rng = random.Random((self.args.seed << 20) ^ (conv << 10) ^ turn)
        ids = [rng.randrange(1000, 140000) for _ in range(max(16, n_tokens))]
        body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0}, "stream": False}
        t = time.perf_counter()
        rec = {"kind": "control", "conv": conv, "turn": turn, "prompt_tokens": len(ids), "t_send": t - self.t0}
        try:
            async with session.post(f"{self.args.url}/generate", json=body) as resp:
                d = await resp.json()
            meta = d.get("meta_info") or {}
            rec.update(ttft=time.perf_counter() - t, cached_tokens=meta.get("cached_tokens"),
                       cached_details=meta.get("cached_tokens_details"))
            self.stats["controls"] += 1
        except Exception as e:  # noqa: BLE001 - a failed control must not stop the conversation
            rec["error"] = str(e)[:200]
            self.stats["errors"] += 1
        self.write(rec)

    async def compact(self, session: aiohttp.ClientSession, conv: int, before_turn: int, n: int, history: list,
                      keep_from: int, act: Action, out_len: int) -> list | None:
        """One summarization request; returns the new history (system prompt, summary, history[keep_from:]), or None
        when the request failed."""
        rec = {"kind": "compaction", "conv": conv, "before_turn": before_turn, "n": n, "mode": self.args.compact_mode,
               "history_messages": len(history), "kept_turns": act.turns, "predicted_prompt_after": act.tokens - out_len}
        system = [m for m in history[:1] if m.get("role") == "system"]
        if self.args.compact_mode == "cold":   # the agent's system prompt stays out, as in Gemini CLI or OpenHands
            transcript = "\n\n".join(f"[{m.get('role')}]\n{content_text(m)}" for m in history[len(system):])
            msgs = [{"role": "system", "content": SUMMARIZER_SYSTEM},
                    {"role": "user", "content": f"{transcript}\n\n{COMPACT_INSTRUCTION}"}]
        else:
            msgs = history + [{"role": "user", "content": COMPACT_INSTRUCTION}]
        r, err = await self.chat_with_retry(session, msgs, self.args.compact_tokens,
                                            f"{self.rid(conv, before_turn)}-compact{n}", rec)
        if r is None:
            rec["error"] = str(err)[:300]
            self.stats["errors"] += 1
            self.write(rec)
            return None
        u, cd = r["usage"] or {}, r["cached_details"] or {}
        rec.update(t_send=round(r["t_send"] - self.t0, 3), ttft=r["ttft"], latency=round(r["latency"], 4),
                   prompt_tokens=int(u.get("prompt_tokens") or 0), completion_tokens=int(u.get("completion_tokens") or 0),
                   cached_tokens=sum(int(cd.get(k) or 0) for k in ("device", "host", "storage")) if cd
                   else int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0),
                   cached_details=cd or None)
        self.write(rec)
        self.stats["compactions"] += 1
        return system + [{"role": "user", "content": SUMMARY_PREFIX + r["text"]}] + history[keep_from:]

    async def conversation(self, session: aiohttp.ClientSession, sess: Session) -> None:
        conv = sess.user
        t_arrival = time.perf_counter() - self.t0
        self.counts["arrived"] += 1
        async with self.sem:
            if self.elapsed() > self.deadline:
                self.write({"kind": "conv_skipped", "conv": conv, "reason": self.stop_reason, "arrival_s": round(t_arrival, 3)})
                self.counts["done"] += 1
                return
            self.counts["live"] += 1
            self.admitted += 1
            try:
                await self.play(session, sess, t_arrival)
            finally:
                self.counts["live"] -= 1
                self.counts["done"] += 1

    def end_session(self, conv: int, turns_done: int, **extra) -> None:
        self.write({"kind": "conv_end", "conv": conv, "turns_done": turns_done, **extra,
                    "end_s": round(time.perf_counter() - self.t0, 3)})
        if not extra.get("capped"):
            self.user_left()

    async def play(self, session: aiohttp.ClientSession, sess: Session, t_arrival: float) -> None:
        conv = sess.user
        steps, plan = self.steps_and_plan(sess)
        start_turn, start_idle = self.start_of(sess, steps, plan)
        t_start = time.perf_counter() - self.t0
        start = {"kind": "conv_start", "conv": conv, "turns": len(steps), "arrival_s": round(t_arrival, 3),
                 "start_s": round(t_start, 3), "queued_s": round(t_start - t_arrival, 3)}
        if self.sessions_on:
            start.update(tasks=[t.index for t in sess.tasks], task_sessions=[t.session_id for t in sess.tasks],
                         task_gaps=[round(x, 3) for x in sess.task_gaps], salt=sess.salt or None)
        if plan:
            start["planned"] = {k: sum(1 for x in plan if x.kind == k) for k in ("compact", "truncate", "end")}
        if self.args.start_phase == "stationary":
            start.update(start_turn=start_turn, start_idle=round(start_idle, 3))
            self.stats["mid_start_users"] += int(start_turn > 0)
        self.write(start)
        # spans: start index in history of each turn still in it (its messages, then its reply), synthetic turns first
        history, spans = synthetic_history(steps, start_turn)
        prev = None  # last turn's result, for the reuse check
        pending_gap = None  # with --oracle-hints, turn t+1's gap is drawn (and hinted) when turn t is sent
        n_compact = 0
        for turn_idx, st in enumerate(steps[start_turn:], start_turn):
            turn = st.turn
            out_len = max(1, int(turn.get("output_length") or 1))
            act = plan[turn_idx] if turn_idx < len(plan) else NO_ACTION
            gap_slept = 0.0
            if turn_idx > 0:
                gap_slept = start_idle if turn_idx == start_turn else \
                    pending_gap if pending_gap is not None else self.step_gap(st)
                if self.deadline < math.inf:   # never sleep past the deadline, then stop at this turn boundary
                    gap_slept = max(0.0, min(gap_slept, self.deadline - self.elapsed()))
                if gap_slept > 0:
                    await self.idle(gap_slept)   # measured from the end of the previous reply
                if self.elapsed() >= self.deadline:
                    self.stats["capped_convs"] += 1
                    self.end_session(conv, turn_idx, aborted=False, capped=True)
                    return
            if act.kind == "end":
                self.stats["context_ends"] += 1
                self.end_session(conv, turn_idx, aborted=False, reason="context", predicted_prompt=act.tokens - out_len)
                return
            if act.kind == "compact":
                keep_from = spans[len(spans) - act.turns] if act.turns else len(history)
                new = await self.compact(session, conv, turn_idx, n_compact, history, keep_from, act, out_len)
                n_compact += 1
                if new is None:
                    self.end_session(conv, turn_idx, aborted=True)
                    return
                shift = len(new) - len(history)   # the kept turns now follow the system prompt and the summary
                spans = [s + shift for s in spans[len(spans) - act.turns:]] if act.turns else []
                history = new
                if self.elapsed() >= self.deadline:
                    self.stats["capped_convs"] += 1   # the deadline passed during the compaction: no new request
                    self.end_session(conv, turn_idx, aborted=False, capped=True)
                    return
            elif act.kind == "truncate":   # the plan only truncates when it drops at least one turn
                lo = spans[1]
                hi = spans[1 + act.turns] if 1 + act.turns < len(spans) else len(history)
                history = history[:lo] + history[hi:]
                spans = spans[:1] + [s - (hi - lo) for s in spans[1 + act.turns:]]
                self.stats["truncations"] += 1
                self.write({"kind": "truncation", "conv": conv, "before_turn": turn_idx, "turns_dropped": act.turns,
                            "predicted_prompt_after": act.tokens - out_len, "t": round(time.perf_counter() - self.t0, 3)})
            spans.append(len(history))
            history = history + list(st.messages)
            if self.args.control_every and turn_idx > start_turn and (turn_idx + conv) % self.args.control_every == 0 \
                    and int(turn.get("prompt_tokens") or 0) > 0:
                n_control = int(turn["prompt_tokens"])   # the recompute reference is as long as the prompt sent
                if plan:
                    n_control = act.tokens - out_len
                elif self.chaining:
                    n_control = prev["prompt_tokens"] + prev["completion"] + st.delta
                self.controls.append(asyncio.create_task(self.control(session, conv, turn_idx, n_control)))
            rec = {"kind": "turn", "conv": conv, "turn": turn_idx, "source": turn.get("source"),
                   "gap_slept": round(gap_slept, 3), "pre_gap": turn.get("pre_gap"),
                   "output_length": turn.get("output_length"),
                   "replay_prompt_tokens": turn.get("replay_prompt_tokens")}
            if self.sessions_on:
                rec.update(task=st.task, task_conv=sess.tasks[st.task].index, task_turn=st.task_turn)
            if turn_idx == start_turn > 0:   # the stationary start's first request: steps before it are synthetic history
                rec.update(mid_start=True, synthetic_turns=start_turn)
            if act.kind:
                rec["after_context_action"] = act.kind
            if plan:
                rec["predicted_prompt"] = act.tokens - out_len
            hints, pending_gap = None, None
            if self.args.oracle_hints and turn_idx + 1 < len(steps):
                pending_gap = self.step_gap(steps[turn_idx + 1])
                # the next send is at least pending_gap away: past the wall cap it is certainly never sent
                suppressed = self.elapsed() + pending_gap >= self.deadline
                # a context action before the next turn rewrites the prompt, so that turn does not extend this one
                suppressed = suppressed or (turn_idx + 1 < len(plan) and bool(plan[turn_idx + 1].kind)) \
                    or (bool(plan) and turn_idx + 1 >= len(plan))
                if not suppressed:
                    hints = self.next_turn_hints(conv, turn_idx, pending_gap)
                rec["oracle_hint"] = {"next_rid": self.rid(conv, turn_idx + 1), "next_gap_s": round(pending_gap, 3),
                                      "suppressed": suppressed}
            r, err = await self.chat_with_retry(session, history, out_len, self.rid(conv, turn_idx), rec, kv_hints=hints)
            if r is None:
                rec["error"] = str(err)[:300]
                self.stats["errors"] += 1
                self.write(rec)
                self.end_session(conv, turn_idx, aborted=True)
                return
            u = r["usage"] or {}
            cd = r["cached_details"] or {}
            cached = sum(int(cd.get(k) or 0) for k in ("device", "host", "storage")) if cd \
                else int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
            prompt_tokens = int(u.get("prompt_tokens") or 0)
            completion = int(u.get("completion_tokens") or 0)
            rec.update(t_send=round(r["t_send"] - self.t0, 3), ttft=r["ttft"], latency=round(r["latency"], 4),
                       prompt_tokens=prompt_tokens, completion_tokens=completion, cached_tokens=cached,
                       cached_details=cd or None)
            if completion < out_len:   # the server capped the reply (prompt + max_tokens near the window)
                rec["short_reply"] = out_len - completion
            if prev is not None and prompt_tokens and not act.kind:
                delta = prompt_tokens - (prev["prompt_tokens"] + prev["completion"])   # new messages + headers
                rec["delta_tokens"] = delta
                rec["reply_reprefilled"] = (prompt_tokens - cached) - delta  # ~0 (+- page) if the reply was reused
                self.stats["reprefill_sum"] += rec["reply_reprefilled"]; self.stats["reprefill_n"] += 1
                if self.args.check_ids and prev["ids"] and r["input_ids"]:
                    k = lcp(prev["ids"], r["input_ids"])
                    rec["prev_ids"] = len(prev["ids"]); rec["prev_ids_on_prefix"] = k
                    rec["reply_ids_reused"] = max(0, k - prev["prompt_ids"])
                    rec["reply_ids_total"] = len(prev["ids"]) - prev["prompt_ids"]
                    if k < len(prev["ids"]):
                        # divergence inside the previous prompt+reply: keep the ids around it for offline decoding
                        lo = max(0, k - 4)
                        rec["mismatch_pos_in_reply"] = k - prev["prompt_ids"]
                        rec["prev_ctx"] = prev["ids"][lo:k + 8]
                        rec["new_ctx"] = r["input_ids"][lo:k + 8]
            if self.args.save_text:
                rec["text"] = r["text"]
            self.write(rec)
            self.stats["turns"] += 1
            if r["ttft"] is not None:
                self.stats["ttft_sum"] += r["ttft"]
            if turn_idx == start_turn > 0:
                self.stats["mid_start_prompt_tokens"] += prompt_tokens
            history.append({"role": "assistant", "content": r["text"]})   # verbatim: this is what makes the KV a hit
            prev = {"prompt_tokens": prompt_tokens, "completion": completion,
                    "ids": (r["input_ids"] or []) + (r["output_ids"] or []) if self.args.check_ids else None,
                    "prompt_ids": len(r["input_ids"] or [])}
        self.end_session(conv, len(steps), aborted=False)
        print(f"conv {conv} done: {len(steps)} turns{f' (from turn {start_turn})' if start_turn else ''}, "
              f"{time.perf_counter() - self.t0:.0f} s elapsed", flush=True)

    async def server_window(self, session: aiohttp.ClientSession) -> int:
        """The server's context length: max_model_len of the first model in /v1/models."""
        async with session.get(f"{self.args.url}/v1/models") as resp:
            d = await resp.json()
        return int(((d.get("data") or [{}])[0]).get("max_model_len") or 0)

    async def calibrate(self, session: aiohttp.ClientSession, pool: list) -> None:
        """Template offsets and exact system blocks from the server's own template and tokenizer (/v1/tokenize)."""
        async def count(messages: list) -> int:
            body = {"model": self.args.model, "messages": messages,
                    "chat_template_kwargs": {"enable_thinking": self.args.thinking == "on"}}
            async with session.post(f"{self.args.url}/v1/tokenize", json=body) as resp:
                return int((await resp.json())["count"])
        try:
            counts = [await count(m) for m, _ in calibration_inputs(pool)]
            raw = {t: await count([{"role": "system", "content": t}]) for t in system_texts(pool)}
        except Exception as e:  # noqa: BLE001 - the trace's template and the character estimate still work, less exactly
            print(f"note: /v1/tokenize failed ({str(e)[:120]}); assuming the trace's template, system blocks from characters",
                  flush=True)
            return
        self.set_calibration(counts, raw, pool)
        print(f"server template against the trace: prompt {self.prompt_offset:+d}, reply {self.reply_offset:+d} tokens; "
              f"system blocks {sorted(set(self.blocks.values()))}", flush=True)

    async def progress(self, n_users: int) -> None:
        t_prev = area_prev = 0.0
        while True:
            await asyncio.sleep(self.args.progress_every)
            c = self.counts
            cap = ""
            if self.args.cap_mode == "backlog":
                now = self.elapsed()
                area = self.backlog_integral(now)
                mean = (area - area_prev) / (now - t_prev)
                cap = f"; cap {self.cap}, backlog {self.backlog} (mean {mean:.1f} since the last line)"
                t_prev, area_prev = now, area
            print(f"[{time.perf_counter() - self.t0:.0f} s] users arrived {c['arrived']}/{n_users}, live {c['live']}, "
                  f"waiting {c['arrived'] - c['live'] - c['done']}, done {c['done']} ({self.finished} finished); "
                  f"turns {self.stats['turns']}, "
                  f"compactions {self.stats['compactions']}, truncations {self.stats['truncations']}, "
                  f"context ends {self.stats['context_ends']}{cap}", flush=True)

    async def cap_control(self) -> None:
        """--cap-mode backlog: every --cap-interval seconds on the grid from t0, move the cap by the time-averaged
        backlog's error against --cap-target (module docstring) and write one kind=cap record; ends with the run."""
        a = self.args
        t_prev = area_prev = 0.0
        admitted_prev = finished_prev = 0
        while True:
            await asyncio.sleep(max(0.0, (math.floor(t_prev / a.cap_interval) + 1) * a.cap_interval - self.elapsed()))
            now = self.elapsed()
            if now >= self.deadline:   # stopped: nobody is admitted any more
                return
            area = self.backlog_integral(now)
            mean = (area - area_prev) / (now - t_prev)
            c = self.counts
            delta = max(-a.cap_step, min(a.cap_step, round(a.cap_gain * (a.cap_target - mean))))
            waiting = c["arrived"] - c["live"] - c["done"]
            if c["live"] + waiting < self.cap:   # the cap limits nobody (the first users are still arriving): hold
                cap = self.cap
            elif delta < 0:
                cap = max(self.cap + delta, c["live"] - a.cap_step)
            else:
                cap = min(self.cap + delta, max(self.cap, c["live"] + a.cap_step))
            cap = max(a.cap_min, min(a.cap_max, cap))
            self.write({"kind": "cap", "t": round(now, 3), "cap": cap, "prev_cap": self.cap,
                        "backlog_mean": round(mean, 3), "target": a.cap_target, "live": c["live"],
                        "waiting": waiting, "admitted": self.admitted - admitted_prev,
                        "finished": self.finished - finished_prev})
            self.cap_area += self.cap * (now - self.cap_t)
            self.cap, self.cap_t = cap, now
            self.cap_lo, self.cap_hi = min(self.cap_lo, cap), max(self.cap_hi, cap)
            self.sem.set_limit(cap)
            t_prev, area_prev, admitted_prev, finished_prev = now, area, self.admitted, self.finished

    async def run(self, planner: SessionPlanner, times: list) -> None:
        self.write({"kind": "run", "tag": self.args.tag, "t_start": time.time(), "args": vars(self.args),
                    "conversations": len(times),
                    "turns": sum(len(t.turns) for t in planner.pool) if not self.sessions_on else None})
        timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
        # limit=0: no client-side connection cap. aiohttp's default pool (100) queued requests beyond 100 in flight inside
        # the client, which the server never saw (x1 campaign: p50 7.4 s, p95 51 s before server entry).
        connector = aiohttp.TCPConnector(limit=0)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            if self.args.context_policy != "none" and not self.window:
                self.window = await self.server_window(session)
                if not self.window:
                    sys.exit("--context-policy needs the window: pass --max-context (the server's /v1/models has no max_model_len)")
            if self.args.context_policy != "none" or self.chaining:
                await self.calibrate(session, planner.pool)
            if self.compaction_room(planner.pool):
                sys.exit(self.compaction_room(planner.pool))
            ticker = asyncio.create_task(self.progress(len(times))) \
                if self.args.arrival_until and self.args.progress_every > 0 else None
            capper = asyncio.create_task(self.cap_control()) if self.args.cap_mode == "backlog" else None
            if self.args.arrival_rate > 0:
                # open loop: users arrive as a Poisson process; --concurrency only caps how many are live
                tasks = []
                for k, t in enumerate(times):
                    if t < self.deadline:
                        await self.idle(max(0.0, self.t0 + t - time.perf_counter()))
                    if t >= self.deadline:   # users the deadline leaves out: one conv_skipped each, no waiting
                        for late in times[k:]:
                            self.write({"kind": "conv_skipped", "conv": planner.next_session().user,
                                        "reason": self.stop_reason, "arrival_s": round(late, 3)})
                        break
                    tasks.append(asyncio.create_task(self.conversation(session, planner.next_session())))
                await asyncio.gather(*tasks)
            else:
                await asyncio.gather(*(self.conversation(session, planner.next_session()) for _ in times))
            if ticker:
                ticker.cancel()
            if capper:
                capper.cancel()
            if self.controls:
                await asyncio.gather(*self.controls)
        s = self.stats
        summary = {"kind": "summary", "tag": self.args.tag, "elapsed_s": round(time.perf_counter() - self.t0, 1),
                   "turns": s["turns"], "controls": s["controls"], "errors": s["errors"], "retries": s["retries"],
                   "capped_convs": s["capped_convs"], "capped": s["capped_convs"] > 0,
                   "mean_ttft_s": round(s["ttft_sum"] / s["turns"], 4) if s["turns"] else None,
                   "mean_reply_reprefilled": round(s["reprefill_sum"] / s["reprefill_n"], 1) if s["reprefill_n"] else None}
        if self.sessions_on or self.args.context_policy != "none":
            summary.update(users=planner.next_user, compactions=s["compactions"], truncations=s["truncations"],
                           context_ends=s["context_ends"], window=self.window or None)
        if self.args.stop_after_users:
            stopped = self.deadline < math.inf and self.elapsed() >= self.deadline
            summary.update(finished_users=self.finished, stopped_by=self.stop_reason if stopped else None,
                           stop_s=round(self.deadline, 3) if stopped else None)
        if self.args.cap_mode == "backlog":
            end = min(self.elapsed(), self.deadline)   # no admissions after the stop: the cap no longer acts
            summary.update(cap_mode="backlog", cap_mean=round((self.cap_area + self.cap * (end - self.cap_t)) / end, 1),
                           cap_min=self.cap_lo, cap_max=self.cap_hi)
        if self.args.start_phase == "stationary":
            summary.update(mid_start_users=s["mid_start_users"], mid_start_prompt_tokens=s["mid_start_prompt_tokens"])
        self.write(summary)
        print("summary: " + json.dumps(summary), flush=True)


def validate(args: argparse.Namespace, planner: SessionPlanner, times: list, selection: str) -> int:
    """--validate: plan every user's session as the live run would (no server) and print the totals."""
    r = Replayer(args, None)
    if args.context_policy != "none" or r.chaining or args.start_phase == "stationary":
        print(f"system blocks and template offsets: {r.calibrate_local(planner.pool)}")
    if r.compaction_room(planner.pool):
        print(f"invalid: {r.compaction_room(planner.pool)}")
        return 1
    acts, turns, salted = {}, 0, 0
    first, zero, mid, limited = [], [], [], 0   # stationary users: first-request prompt, turn-0 prompt; mid-start ones
    for _ in times:
        sess = planner.next_session()
        steps, plan = r.steps_and_plan(sess)
        s, _ = r.start_of(sess, steps, plan)
        salted += bool(sess.salt)
        for a in plan:
            if a.kind:
                acts[a.kind] = acts.get(a.kind, 0) + 1
        turns += (len(steps) if not plan or plan[-1].kind != "end" else len(plan) - 1) - s
        if args.start_phase == "stationary" and sess.user < (args.start_users or args.concurrency):
            first.append(start_prompt_tokens(steps, s, prompt_offset=r.prompt_offset))
            zero.append(start_prompt_tokens(steps, 0, prompt_offset=r.prompt_offset))
            if s:
                mid.append(first[-1])
            limited += start_limit(steps, plan) < len(steps)
    print(f"{selection}\nvalid: {len(times)} users ({salted} salted), {turns} turns planned, context actions {acts or 'none'}")
    if args.cap_mode == "backlog":
        print(f"cap: backlog, starting at {args.concurrency}; every {args.cap_interval:g} s it steers the mean backlog "
              f"to {args.cap_target:g} (gain {args.cap_gain:g}, step {args.cap_step}, range {args.cap_min}-{args.cap_max});"
              f" user starts at least {args.admit_spacing:g} s apart")
    if args.order == "fresh":
        print(f"order: fresh (seed {args.order_seed}): users 0..{len(planner.pool) - 1} open with a seeded permutation "
              f"of the {len(planner.pool)} tasks, each once; no later users")
    if args.order == "shuffle":
        n = len(planner.pool)
        print(f"order: shuffle (seed {args.order_seed}): users {n} and later open with a seeded permutation of the {n} "
              f"tasks per block of {n} users")
    if args.start_phase == "stationary":
        mid.sort()
        stats = (f"{sum(mid) / len(mid):.0f} tokens mean, p90 {mid[min(len(mid) - 1, int(0.9 * len(mid)))]}, "
                 f"max {mid[-1]}, total {sum(mid) / 1e6:.2f} M") if mid else "none"
        print(f"stationary start: {len(mid)} of the first {len(first)} users start mid-session ({limited} limited by a "
              f"planned context action); their first requests {stats}; all {len(first)} first requests "
              f"{sum(first) / 1e6:.2f} M tokens against {sum(zero) / 1e6:.2f} M at turn 0")
    return 0


def main() -> int:
    args = parse_args()
    pool = load_pool(args)
    planner = SessionPlanner(pool, args)
    # the arrival schedule comes from the seeded stream as before: one draw per user, in arrival order
    times = arrival_times(random.Random(args.seed), args.arrival_rate, args.arrival_until, len(pool)) \
        if args.arrival_rate > 0 else [0.0] * len(pool)
    if args.order == "fresh":   # every user replays a conversation nobody replayed before
        times = times[:len(pool)]
    selection = f"#Conversations: {len(pool)}  turns: {sum(len(t.turns) for t in pool)}"
    if args.arrival_until:
        selection += f"  users: {len(times)} arriving until {args.arrival_until:.0f} s"
    if args.dry_run:
        print(selection, flush=True)
        Replayer(args, None).dry_run(planner, len(times))
        return 0
    if args.validate:
        return validate(args, planner, times, selection)
    modes = (f"  cap backlog from {args.concurrency} (target {args.cap_target:g})" if args.cap_mode == "backlog" else "") \
        + (f"  order {args.order} (seed {args.order_seed})" if args.order != "cycle" else "") \
        + (f"  start stationary ({args.start_users or args.concurrency} users)" if args.start_phase == "stationary" else "")
    print(f"{selection}  concurrency: {args.concurrency}"
          f"{' (cap), arrival ' + str(args.arrival_rate) + ' conv/s' if args.arrival_rate > 0 else ' (closed loop)'}  "
          f"gap x{args.gap_scale}{' cap ' + str(args.gap_cap) + 's' if args.gap_cap else ''}"
          f"{' sampled from ' + args.gap_sample if args.gap_sample else ''}  thinking: {args.thinking}"
          f"{'  sessions ' + args.session_tasks + ' tasks' if args.session_tasks else ''}"
          f"{'  context ' + args.context_policy if args.context_policy != 'none' else ''}{modes}", flush=True)
    if aiohttp is None:
        sys.exit("aiohttp is not installed: the live replay needs it (only --dry-run runs without it)")
    with open(args.output, "a") as out:
        asyncio.run(Replayer(args, out).run(planner, times))
    return 0


if __name__ == "__main__":
    sys.exit(main())
