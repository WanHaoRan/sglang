"""Closed-loop replay of program traces (program_trace.py) against an SGLang server, in the literature's style.

  python3 replay_programs.py --trace T.programs.jsonl.gz --recipe agentx-mvp|mori|tracelab|none --output client.jsonl
      [--lanes N | --arrival-rate R [--max-active M]] [--duration S] [--hints none|program|release|dist|oracle] ...

Every request starts once the requests it depends on have completed and its recorded wait has passed: the closed loop
of MORI's client, TraceLab's session_runner and InferenceX's agentic replay. Prompts are token ids sent to the native
/generate endpoint (or /v1/completions), rebuilt from the trace's block ids or, for length traces, as TraceLab builds
them (the previous prompt and the server's own reply, then new tokens). Outputs are forced to the recorded length (max
tokens, ignore_eos, temperature 0). Subagents and side chains run as their own streams beside the parent and join where
the trace says.

Recipes set the defaults of the published setups (explicit flags win; the run record keeps the resolved settings):
  agentx-mvp  SemiAnalysis InferenceX AgentX MVP (AIPerf --scenario inferencex-agentx-mvp): --lanes session trees kept
              live, each initial lane starting at a random instant of its trace after one primer request (one output
              token) per live stream, recycled lanes from turn 0, end-to-start recorded gaps, a 10 s system-idle guard
              (when nothing is in flight, pending timers shift so the next request is at most 10 s away), a cache-bust
              marker on recycled plays, 1,800 s of profiling.
  mori        MORI: sessions split into programs at idle periods over 30 min and at context compactions, --lanes
              programs in flight (subagents run beside their program), each starting at its first request, two hours.
  tracelab    TraceLab session_runner: sessions arrive as a Poisson process at --arrival-rate, at most --max-active at
              once, each sleeping only the tool time after a round (TraceLab's canonical tool_wait_after_ms), lengths
              content with the server's replies carried forward.
Hints (our engine's kv_hints, gap_hints.py): none; program = the next request's id only (the literature's program-ID
interface, campaign 26's nofc_noend); release = that plus the end of each stream (nofc); dist = per-key gap
distributions measured online per agent system, with the recorded reply's key (the engine's "hint" reply-key source);
oracle = the next request's exact gap.

Output (--output, appended JSONL): a run record, one req record per request (phase warmup or profile, timings, server
token counts, cached tokens by tier, the tokens an infinite cache would hold), play records, a lanes record per minute
and a summary. Request ids are <tag>-c<stream instance>-t<request index>, the key of the server's logs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time

import aiohttp

from gap_hints import (END_ENTRY, END_KEY, KEYS, NO_CALL_KEY, OnlineGapTable, TurnHint, hint_record, next_turn_envelope,
                       pick_entry)
from program_trace import BlockPrompts, LengthPrompts, TokenSynth, end_of, iter_trees, read_header, split_programs

MONITOR_S = 60.0
RECIPES = {
    "agentx-mvp": {"lanes": 32, "duration": 1800.0, "start_window": (0.0, 1.0), "warmup": "primer",
                   "system_idle_cap": 10.0, "cache_bust": "recycle", "wait_field": "wait"},
    "mori": {"lanes": 80, "duration": 7200.0, "split_idle": 1800.0, "split_compaction": True, "cache_bust": "recycle",
             "wait_field": "wait"},
    "tracelab": {"arrival_rate": 0.05, "max_active": 128, "duration": 0.0, "cache_bust": "none",
                 "wait_field": "wait_tool"},
    "none": {},
}
DEFAULTS = {"lanes": 0, "arrival_rate": 0.0, "max_active": 0, "duration": 0.0, "grace": 30.0, "start_window": None,
            "warmup": "none", "system_idle_cap": None, "cache_bust": "none", "wait_field": "wait", "split_idle": None,
            "split_compaction": False, "gap_scale": 1.0, "gap_cap": None, "sample": "shuffle"}
HINTS = ("none", "program", "release", "dist", "oracle")


# ===== Settings =====

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True, help="program trace (convert_agentx/tracelab/freeinference.py)")
    ap.add_argument("--output", required=True, help="JSONL, appended")
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="none", help="defaults of a published setup")
    ap.add_argument("--lanes", type=int, help="session trees kept live; a drained lane starts the next tree")
    ap.add_argument("--arrival-rate", type=float, help="trees arrive as a Poisson process at this rate per second")
    ap.add_argument("--max-active", type=int, help="with --arrival-rate: most trees in session at once (0 = no cap)")
    ap.add_argument("--duration", type=float, help="seconds of profiling; no request is sent after (0 = until done)")
    ap.add_argument("--grace", type=float, help="seconds requests in flight at the end may take to complete")
    ap.add_argument("--start-window", type=float, nargs=2, metavar=("LO", "HI"),
                    help="initial lanes start at a uniform fraction LO..HI of their trace's span")
    ap.add_argument("--warmup", choices=("none", "primer"), help="primer: one 1-token request per live stream first")
    ap.add_argument("--system-idle-cap", type=float,
                    help="when nothing is in flight, shift timers so the next is at most this far away (s)")
    ap.add_argument("--cache-bust", choices=("none", "recycle"), help="recycle: a recycled play's prompts start with "
                                                                      "tokens unique to the play")
    ap.add_argument("--wait-field", choices=("wait", "wait_tool"), help="wait_tool: tool time only (TraceLab)")
    ap.add_argument("--split-idle", type=float, help="cut sessions into programs at idle periods over this (s)")
    ap.add_argument("--split-compaction", action="store_true", default=None, help="also cut at context compactions")
    ap.add_argument("--gap-scale", type=float, help="sleep the recorded wait x this (not 1: an ablation or a CPU test)")
    ap.add_argument("--gap-cap", type=float, help="longest single wait (s)")
    ap.add_argument("--sample", choices=("shuffle", "sequential"), help="order in which lanes draw trees")
    ap.add_argument("--trees", type=int, default=0, help="use only the first N trees after filtering and ordering")
    ap.add_argument("--plays", type=int, default=0, help="lanes: start no play after the first N (0 = until the end)")
    ap.add_argument("--max-context", type=int, default=0,
                    help="drop trees with a request over this many tokens (prompt + output), as InferenceX's "
                         "--max-context-length")
    ap.add_argument("--hints", choices=HINTS, default="none")
    ap.add_argument("--hint-key", choices=KEYS, default="command",
                    help="dist: key the gap table by tool and word, tool, or nothing")
    ap.add_argument("--hint-point", choices=("none", "p50"), default="p50",
                    help="dist: also send the picked median (v1)")
    ap.add_argument("--hint-no-call", action="store_true",
                    help="dist: a reply the trace records no tool call for gets the key _no_call, as in replay_agentic.py, "
                         "so the waits after such replies (a person's) have their own entry")
    ap.add_argument("--api", choices=("generate", "completions"), default="generate")
    ap.add_argument("--program-id-field", default=None,
                    help="completions only: also put the stream's id in this body field (MORI/ThunderAgent servers)")
    ap.add_argument("--release-path", default=None, help="POST {program_id} here when a stream ends, e.g. "
                                                         "/programs/release")
    ap.add_argument("--vocab", default="100:30000", help="LO:HI token id range of synthetic tokens")
    ap.add_argument("--token-pool", default=None, help="uint32 file of real-text token ids for the served model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tag", default="prog")
    args = ap.parse_args(argv)
    resolve(args, ap)
    return args


def resolve(args: argparse.Namespace, ap: argparse.ArgumentParser) -> None:
    """Fill unset flags from the recipe, then from DEFAULTS, and refuse contradictions. An explicit --lanes or
    --arrival-rate replaces the recipe's schedule."""
    explicit = {key for key in DEFAULTS if getattr(args, key) is not None}
    for key, default in DEFAULTS.items():
        if key not in explicit:
            setattr(args, key, RECIPES[args.recipe].get(key, default))
    if "arrival_rate" in explicit and "lanes" not in explicit:
        args.lanes = 0
    if "lanes" in explicit and "arrival_rate" not in explicit:
        args.arrival_rate = 0.0
    if args.start_window is not None and not args.lanes:
        if "start_window" in explicit:
            ap.error("--start-window places lanes; arrivals start every tree at its first request")
        args.start_window, args.warmup = None, "none"
    if args.start_window is not None:
        args.start_window = tuple(args.start_window)
    if (args.lanes > 0) == (args.arrival_rate > 0):
        ap.error("set exactly one of --lanes and --arrival-rate (the recipe sets one)")
    if args.start_window is not None and args.warmup != "primer":
        ap.error("--start-window needs --warmup primer (a lane started mid-trace needs its context first)")
    if args.start_window is not None and not 0.0 <= args.start_window[0] <= args.start_window[1] <= 1.0:
        ap.error("--start-window LO HI with 0 <= LO <= HI <= 1")
    if args.program_id_field and args.api == "generate":
        ap.error("--program-id-field needs --api completions: /generate rejects unknown fields")
    if args.gap_scale < 0 or (args.gap_cap is not None and args.gap_cap < 0):
        ap.error("--gap-scale and --gap-cap take non-negative values")
    lo, _, hi = args.vocab.partition(":")
    args.vocab_range = (int(lo), int(hi))
    args.explicit = sorted(explicit)


# ===== Trees =====

def load_trees(args: argparse.Namespace) -> tuple:
    """(header, trees in lane order, how many were dropped by --max-context)."""
    header = read_header(args.trace)
    if args.wait_field == "wait_tool" and header["source"] != "tracelab":
        sys.exit(f"STOP: --wait-field wait_tool needs a TraceLab trace; {header['source']} has one wait per request")
    trees, dropped = [], 0
    for tree in iter_trees(args.trace):
        for prog in split_programs(tree, idle_s=args.split_idle, compaction=bool(args.split_compaction),
                                   content=header["content"]):
            if args.max_context and _peak_tokens(prog, header) > args.max_context:
                dropped += 1
                continue
            trees.append(prog)
    if args.sample == "shuffle":
        random.Random(f"{args.seed}:order").shuffle(trees)
    if args.trees:
        trees = trees[: args.trees]
    if not trees:
        sys.exit("STOP: no tree left to replay")
    return header, trees, dropped


def _peak_tokens(tree: dict, header: dict) -> int:
    B = header.get("block_tokens") or 0
    peak = 0
    for st in tree["streams"]:
        for r in st["requests"]:
            n = (r["keep"] + len(r["new"])) * B + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]
            peak = max(peak, n + r["out"])
    return peak


def choose_cut(tree: dict, rng: random.Random, window: tuple) -> float:
    """InferenceX's trajectory start: a uniform instant of the trace's span, leaving a root request to profile."""
    root = tree["streams"][0]["requests"]
    if len(root) < 2:
        return 0.0
    span = max(end_of(r) for st in tree["streams"] for r in st["requests"])
    return min(rng.uniform(*window) * span, root[-1]["t"])


# ===== Clock =====

class Clock:
    """Replay time with InferenceX's system-idle guard: when no request is in flight and the nearest timer is more than
    `idle_cap` away, every pending timer moves earlier by the same amount (order and spacing kept)."""

    def __init__(self, *, idle_cap: float | None) -> None:
        self.t0 = time.perf_counter()
        self.shift = 0.0
        self.idle_cap = idle_cap
        self.in_flight = 0
        self.jumps, self.skipped_s = 0, 0.0
        self._timers: dict = {}
        self._cond = asyncio.Condition()
        self._check = asyncio.Event()

    def now(self) -> float:
        return time.perf_counter() - self.t0 + self.shift

    async def sleep_until(self, due: float) -> None:
        if due <= self.now():
            return
        token = object()
        self._timers[token] = due
        self._check.set()
        try:
            async with self._cond:
                while (rem := due - self.now()) > 0:
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=rem)
                    except asyncio.TimeoutError:
                        pass
        finally:
            del self._timers[token]

    def started(self) -> None:
        self.in_flight += 1

    def finished(self) -> None:
        self.in_flight -= 1
        self._check.set()

    async def guard(self) -> None:
        """The idle check, run a moment after a completion so the finishing stream has set its next timer."""
        while self.idle_cap is not None:
            await self._check.wait()
            self._check.clear()
            await asyncio.sleep(0.02)
            if self.in_flight or not self._timers:
                continue
            gap = min(self._timers.values()) - self.now()
            if gap > self.idle_cap:
                self.shift += gap - self.idle_cap
                self.jumps += 1
                self.skipped_s += gap - self.idle_cap
                async with self._cond:
                    self._cond.notify_all()


# ===== One play of a tree =====

class Play:
    """One replay of a tree: per stream, the first request to send, the primer before it, and completion events."""

    def __init__(self, rep: "Replayer", tree: dict, *, play_id: int, lane: int, cut: float, bust: bool) -> None:
        self.rep, self.tree, self.play_id, self.lane, self.cut = rep, tree, play_id, lane, cut
        self.streams = tree["streams"]
        self.inst = [rep.new_instance() for _ in self.streams]
        tag = f"{play_id}"
        bust_ids = rep.synth.ids(f"bust|{rep.args.seed}|{play_id}", 8) if bust else None
        if rep.header["content"] == "blocks":
            self.prompts = BlockPrompts(tree, block_tokens=rep.header["block_tokens"], synth=rep.synth, bust=bust_ids,
                                        play_tag=tag)
        else:
            self.prompts = LengthPrompts(tree, synth=rep.synth, play_tag=tag)
        self.domain = (tree.get("scope") or tree["tree"]) if not bust else f"{tree['tree']}|{play_id}"
        self.start_k = [next((k for k, r in enumerate(st["requests"]) if r["t"] >= cut), len(st["requests"]))
                        for st in self.streams]
        self.primers = [(s, k - 1) for s, k in enumerate(self.start_k) if 0 < k < len(self.streams[s]["requests"])]
        self.origin = 0.0         # replay time of the recorded t = 0, set by start()
        self.end_at: dict = {}    # (stream, index) -> replay time it completed (pre-history: mapped recorded end)
        self.events: dict = {}
        self.sent_before = False

    def first_offset(self) -> float:
        """Recorded seconds from the cut to the earliest request still to send."""
        ts = [st["requests"][k]["t"] for st, k in zip(self.streams, self.start_k) if k < len(st["requests"])]
        return min(ts, default=self.cut) - self.cut

    def start(self, *, origin: float) -> None:
        self.origin = origin
        for s, st in enumerate(self.streams):
            for k in range(self.start_k[s]):
                self.end_at[(s, k)] = origin + end_of(st["requests"][k])
                self._event((s, k)).set()

    def _event(self, key: tuple) -> asyncio.Event:
        ev = self.events.get(key)
        if ev is None:
            ev = self.events[key] = asyncio.Event()
        return ev

    async def run(self) -> None:
        await asyncio.gather(*(self._stream(s) for s in range(len(self.streams))))

    async def _stream(self, s: int) -> None:
        rep, reqs = self.rep, self.streams[s]["requests"]
        for k in range(self.start_k[s], len(reqs)):
            r = reqs[k]
            for d in r["deps"]:
                await self._event(tuple(d)).wait()
            ready = max((self.end_at.get(tuple(d), rep.clock.now()) for d in r["deps"]), default=self.origin)
            await rep.clock.sleep_until(min(ready + rep.scaled_wait(r), rep.deadline))   # no sleep crosses the end
            if rep.past_deadline():
                self._abandon(s, k)
                return
            await rep.send(self, s, k, phase="profile")
            self.end_at[(s, k)] = rep.clock.now()
            self._event((s, k)).set()
        await rep.release(self, s)

    def _abandon(self, s: int, k: int) -> None:
        """Past the deadline: the stream's remaining requests count as done so no dependent waits forever."""
        reqs = self.streams[s]["requests"]
        for j in range(k, len(reqs)):
            self.end_at[(s, j)] = self.rep.clock.now()
            self._event((s, j)).set()
        self.rep.counts["skipped"] += len(reqs) - k


# ===== The replayer =====

class Replayer:
    def __init__(self, args: argparse.Namespace, *, header: dict, trees: list, out, http, model: str) -> None:
        self.args, self.header, self.trees, self.out, self.http, self.model = args, header, trees, out, http, model
        lo, hi = args.vocab_range
        self.synth = TokenSynth(lo=lo, hi=hi, pool_path=args.token_pool)
        self.clock = Clock(idle_cap=args.system_idle_cap)
        self.deadline = math.inf
        self.phase = "warmup" if args.warmup == "primer" else "profile"
        self.counts = {"profile": 0, "warmup": 0, "errors": 0, "skipped": 0, "plays": 0, "releases": 0}
        self.ttft_sum = 0.0
        self.tables: dict = {}        # dist hints: agent system -> OnlineGapTable
        self.seen: dict = {}          # infinite-cache tracker: domain -> block ids completed so far
        self._inst = 0
        self._play_counts: dict = {}
        self._cursor = 0
        self.live = 0

    # ----- bookkeeping -----

    def write(self, rec: dict) -> None:
        self.out.write(json.dumps(rec, separators=(",", ":")) + "\n")

    def new_instance(self) -> int:
        self._inst += 1
        return self._inst - 1

    def rid(self, play: Play, s: int, k: int) -> str:
        return f"{self.args.tag}-c{play.inst[s]}-t{k}"

    def scaled_wait(self, r: dict) -> float:
        w = float(r.get(self.args.wait_field, r["wait"]) or 0.0) * self.args.gap_scale
        return w if self.args.gap_cap is None else min(w, self.args.gap_cap)

    def past_deadline(self) -> bool:
        return self.clock.now() >= self.deadline

    def next_tree(self) -> tuple:
        """The sampler: the trees in lane order, wrapping; (tree, play count of that tree so far)."""
        tree = self.trees[self._cursor % len(self.trees)]
        self._cursor += 1
        n = self._play_counts.get(tree["tree"], 0)
        self._play_counts[tree["tree"]] = n + 1
        return tree, n

    def new_play(self, tree: dict, *, n_before: int, lane: int, cut: float) -> Play:
        play = Play(self, tree, play_id=self.counts["plays"], lane=lane, cut=cut,
                    bust=self.args.cache_bust == "recycle" and n_before > 0)
        self.counts["plays"] += 1
        return play

    # ----- hints -----

    def hints_for(self, play: Play, s: int, k: int, rec: dict) -> dict | None:
        mode = self.args.hints
        if mode == "none":
            return None
        reqs = play.streams[s]["requests"]
        last = k + 1 >= len(reqs)
        session, next_rid = f"{self.args.tag}-c{play.inst[s]}", self.rid(play, s, k + 1)
        if mode == "oracle":
            if last:
                return None
            gap = self.scaled_wait(reqs[k + 1])
            rec["oracle_gap_s"] = round(gap, 3)
            if self.clock.now() + gap >= self.deadline:
                return None
            return {"protocol_version": "1", "message_id": self.rid(play, s, k),
                    "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn", "action_version": "1",
                                 "payload": {"session": session, "next_rid": next_rid, "next_gap_s": round(gap, 3)}}]}
        entries = [END_ENTRY]
        if mode == "dist":
            system = tuple(play.tree.get("system") or ())
            table = self.tables.setdefault(system, OnlineGapTable(key_mode=self.args.hint_key))
            if k > 0:
                table.observe(key=None if self.args.hint_key == "none" else self._reply_key(reqs[k - 1]),
                              gap_s=self.scaled_wait(reqs[k]))
            entries = table.entries()
        if last and mode != "program":
            key = END_KEY
        elif mode == "dist" and self.args.hint_key != "none":
            key = self._reply_key(reqs[k])
        else:
            key = None
        hint = TurnHint(turn=k, reply_key=key, entries=entries, picked=pick_entry({e["key"]: e for e in entries}, key),
                        true_gap_s=None if last else self.scaled_wait(reqs[k + 1]))
        point = self.args.hint_point if mode == "dist" else "none"
        rec["hint"] = hint_record(hint, next_rid=next_rid, v1_point=point)
        return next_turn_envelope(message_id=self.rid(play, s, k), session=session, next_rid=next_rid, hint=hint,
                                  v1_point=point)

    def _reply_key(self, r: dict) -> str | None:
        """The key of a request's recorded reply; None when the trace names no tool call and --hint-no-call is off."""
        return r.get("key") or (NO_CALL_KEY if self.args.hint_no_call else None)

    # ----- requests -----

    def ideal_cached(self, play: Play, s: int, blocks: list | None, r: dict) -> int:
        """Prompt tokens an infinite cache would hold: the prefix already completed in this run (same domain)."""
        if blocks is None:
            return play.prompts.held(s, r)
        seen = self.seen.setdefault(play.domain, set())
        lo, hi = 0, len(blocks)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if blocks[mid - 1] in seen:
                lo = mid
            else:
                hi = mid - 1
        bust = play.prompts.bust_len if play.sent_before else 0
        return lo * self.header["block_tokens"] + bust

    async def send(self, play: Play, s: int, k: int, *, phase: str, max_tokens: int | None = None) -> None:
        r = play.streams[s]["requests"][k]
        ids_json, n_tokens, blocks, _ = play.prompts.prompt(s, k)
        rec = {"kind": "req", "phase": phase, "lane": play.lane, "play": play.play_id, "tree": play.tree["tree"],
               "stream": s, "role": play.streams[s].get("role"), "inst": play.inst[s], "k": k,
               "rid": self.rid(play, s, k), "gap": r.get("gap"), "wait_rec": r["wait"],
               "wait_s": round(self.scaled_wait(r), 3), "key": r.get("key"), "out_target": max_tokens or r["out"],
               "prompt_sent": n_tokens, "ideal_cached": self.ideal_cached(play, s, blocks, r)}
        hints = self.hints_for(play, s, k, rec) if phase == "profile" or self.args.hints != "oracle" else None
        res = await self._post_with_retry(ids_json=ids_json, out=max_tokens or r["out"], rid=rec["rid"],
                                          hints=hints, session=f"{self.args.tag}-c{play.inst[s]}", rec=rec)
        if isinstance(play.prompts, LengthPrompts):
            play.prompts.commit(s, res.get("output_ids") or [])
        if blocks is not None and "error" not in rec:
            self.seen.setdefault(play.domain, set()).update(blocks)
        play.sent_before = True
        self.counts[phase] += 1
        if rec.get("ttft") is not None and phase == "profile":
            self.ttft_sum += rec["ttft"]
        self.write(rec)

    async def _post_with_retry(self, *, ids_json: str, out: int, rid: str, hints, session: str, rec: dict) -> dict:
        for attempt in range(2):
            try:
                return await self._post(ids_json=ids_json, out=out, rid=rid, hints=hints, session=session, rec=rec)
            except aiohttp.ClientConnectionError as e:
                # a pooled keep-alive connection the server closed: retry once, never past the deadline
                if attempt or self.past_deadline():
                    rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                    break
                rec["retried"] = True
            except Exception as e:  # noqa: BLE001 - any failure is recorded on the request and the stream goes on
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
                break
        self.counts["errors"] += 1
        return {}

    def _body(self, *, ids_json: str, out: int, rid: str, hints, session: str) -> str:
        if self.args.api == "generate":
            rest = {"sampling_params": {"max_new_tokens": out, "ignore_eos": True, "temperature": 0.0},
                    "stream": True, "rid": rid}
            if hints is not None:
                rest["kv_hints"] = hints
            return '{"input_ids":[' + ids_json + "]," + json.dumps(rest)[1:]
        rest = {"model": self.model, "max_tokens": out, "stream": True, "stream_options": {"include_usage": True},
                "ignore_eos": True, "temperature": 0.0, "rid": rid,
                "return_token_ids": self.header["content"] == "lengths"}
        if self.args.program_id_field:
            rest[self.args.program_id_field] = session
        return '{"prompt":[' + ids_json + "]," + json.dumps(rest)[1:]

    async def _post(self, *, ids_json: str, out: int, rid: str, hints, session: str, rec: dict) -> dict:
        path = "/generate" if self.args.api == "generate" else "/v1/completions"
        body = self._body(ids_json=ids_json, out=out, rid=rid, hints=hints, session=session)
        res = {"output_ids": [], "meta": {}, "usage": None}
        t_send = self.clock.now()
        rec["t_send"] = round(t_send, 3)
        self.clock.started()
        try:
            async with self.http.post(self.args.url + path, data=body,
                                      headers={"Content-Type": "application/json"}) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                async for raw in resp.content:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    self._absorb(json.loads(payload), res, rec, t_send)
        finally:
            self.clock.finished()
        rec["latency"] = round(self.clock.now() - t_send, 4)
        self._fill_counts(res, rec)
        return res

    def _absorb(self, d: dict, res: dict, rec: dict, t_send: float) -> None:
        if d.get("error"):
            raise RuntimeError(f"stream error: {json.dumps(d['error'])[:200]}")
        if self.args.api == "generate":
            ids = d.get("output_ids") or []
            meta = d.get("meta_info") or {}
            done = int(meta.get("completion_tokens") or 0)
            if ids and rec.get("ttft") is None:
                rec["ttft"] = round(self.clock.now() - t_send, 4)
            # cumulative output ids by default, deltas under --incremental-streaming-output
            res["output_ids"] = ids if len(ids) == done else res["output_ids"] + ids
            res["meta"] = meta
            return
        for ch in d.get("choices") or []:
            if (ch.get("text") or ch.get("token_ids")) and rec.get("ttft") is None:
                rec["ttft"] = round(self.clock.now() - t_send, 4)
            res["output_ids"] += list(ch.get("token_ids") or [])
        if d.get("usage"):
            res["usage"] = d["usage"]

    def _fill_counts(self, res: dict, rec: dict) -> None:
        if self.args.api == "generate":
            m = res["meta"]
            rec["prompt_tokens"] = m.get("prompt_tokens")
            rec["completion_tokens"] = m.get("completion_tokens")
            rec["cached_tokens"] = m.get("cached_tokens")
            rec["cached_details"] = m.get("cached_tokens_details")
        else:
            u = res["usage"] or {}
            rec["prompt_tokens"] = u.get("prompt_tokens")
            rec["completion_tokens"] = u.get("completion_tokens")
            rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)

    async def release(self, play: Play, s: int) -> None:
        if not self.args.release_path:
            return
        try:
            async with self.http.post(self.args.url + self.args.release_path,
                                      json={"program_id": f"{self.args.tag}-c{play.inst[s]}"}) as resp:
                await resp.read()
            self.counts["releases"] += 1
        except Exception as e:  # noqa: BLE001 - a failed release is logged, the replay goes on
            self.write({"kind": "release_error", "inst": play.inst[s], "error": str(e)[:160]})

    # ----- schedules -----

    async def run(self) -> None:
        args = self.args
        self.write({"kind": "run", "t_start": time.time(), "model": self.model, "trace_header": self.header,
                    "trees": len(self.trees), "args": {k: v for k, v in vars(args).items()}})
        guard = asyncio.create_task(self.clock.guard())
        monitor = asyncio.create_task(self.monitor())
        try:
            if args.lanes:
                await self.run_lanes()
            else:
                await self.run_arrivals()
        finally:
            guard.cancel()
            monitor.cancel()
        self.write_summary()

    async def run_lanes(self) -> None:
        args = self.args
        rng = random.Random(f"{args.seed}:cut")
        initial = []
        for lane in range(args.lanes):
            tree, n = self.next_tree()
            cut = choose_cut(tree, rng, args.start_window) if args.start_window else 0.0
            initial.append(self.new_play(tree, n_before=n, lane=lane, cut=cut))
        if args.warmup == "primer":
            await self.warmup(initial)
        self.phase = "profile"
        phase_start = self.clock.now()
        self.deadline = phase_start + args.duration if args.duration > 0 else math.inf
        lead = min(p.first_offset() for p in initial)
        for p in initial:
            p.start(origin=phase_start - p.cut - lead)
        self.write({"kind": "phase", "phase": "profile", "t": round(phase_start, 3), "lead_s": round(lead, 3)})
        lanes = [asyncio.create_task(self.lane(p)) for p in initial]
        await self._wait_all(lanes)

    async def lane(self, play: Play) -> None:
        while True:
            self.live += 1
            try:
                await play.run()
            finally:
                self.live -= 1
            self.write({"kind": "play", "play": play.play_id, "lane": play.lane, "tree": play.tree["tree"],
                        "cut": round(play.cut, 3), "end": round(self.clock.now(), 3)})
            if self.past_deadline() or (self.args.plays and self.counts["plays"] >= self.args.plays):
                return
            tree, n = self.next_tree()
            play = self.new_play(tree, n_before=n, lane=play.lane, cut=0.0)
            play.start(origin=self.clock.now())

    async def warmup(self, plays: list) -> None:
        """One request per live stream (the last before the cut) with one output token, all lanes at once."""
        t = self.clock.now()
        await asyncio.gather(*(self.send(p, s, k, phase="warmup", max_tokens=1) for p in plays for s, k in p.primers))
        self.write({"kind": "phase", "phase": "warmup", "t": round(t, 3), "primers": sum(len(p.primers) for p in plays),
                    "took_s": round(self.clock.now() - t, 3)})

    async def run_arrivals(self) -> None:
        args = self.args
        rng = random.Random(f"{args.seed}:arrivals")
        slots = asyncio.Semaphore(args.max_active) if args.max_active else None
        self.deadline = args.duration if args.duration > 0 else math.inf
        tasks, t = [], 0.0
        for tree in self.trees:
            if t >= self.deadline:
                break
            await self.clock.sleep_until(t)
            tasks.append(asyncio.create_task(self.arrival(tree, slots, t)))
            t += rng.expovariate(args.arrival_rate)
        await self._wait_all(tasks)

    async def arrival(self, tree: dict, slots, arrival_s: float) -> None:
        if slots is not None:
            await slots.acquire()
        try:
            if self.past_deadline():
                return
            _, n = self.next_tree_for(tree)
            play = self.new_play(tree, n_before=n, lane=-1, cut=0.0)
            play.start(origin=self.clock.now())
            self.live += 1
            try:
                await play.run()
            finally:
                self.live -= 1
            self.write({"kind": "play", "play": play.play_id, "tree": tree["tree"], "arrival_s": round(arrival_s, 3),
                        "start": round(play.origin, 3), "end": round(self.clock.now(), 3)})
        finally:
            if slots is not None:
                slots.release()

    def next_tree_for(self, tree: dict) -> tuple:
        n = self._play_counts.get(tree["tree"], 0)
        self._play_counts[tree["tree"]] = n + 1
        return tree, n

    async def _wait_all(self, tasks: list) -> None:
        """Wait for the tasks; past the deadline, give requests in flight --grace seconds, then cancel."""
        pending = set(tasks)
        while pending:
            end = self.deadline + self.args.grace
            timeout = None if self.deadline == math.inf else max(0.0, end - self.clock.now())
            done, pending = await asyncio.wait(pending, timeout=timeout)
            if pending and self.clock.now() >= self.deadline + self.args.grace:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                self.write({"kind": "cancelled", "tasks": len(pending), "t": round(self.clock.now(), 3)})
                break
        for t in tasks:
            if t.done() and not t.cancelled() and t.exception() is not None:
                raise t.exception()

    async def monitor(self) -> None:
        while True:
            await asyncio.sleep(MONITOR_S)
            self.write({"kind": "lanes", "t": round(self.clock.now(), 3), "phase": self.phase, "live": self.live,
                        "in_flight": self.clock.in_flight, "timers": len(self.clock._timers),
                        "shift_s": round(self.clock.shift, 3), "jumps": self.clock.jumps, **self.counts})
            c = self.counts
            print(f"[{self.clock.now():.0f} s] live {self.live}, in flight {self.clock.in_flight}, requests "
                  f"{c['profile']} (+{c['warmup']} warmup), errors {c['errors']}, idle skipped "
                  f"{self.clock.skipped_s:.0f} s", flush=True)
            self.out.flush()

    def write_summary(self) -> None:
        n = self.counts["profile"]
        rec = {"kind": "summary", "tag": self.args.tag, "elapsed_s": round(self.clock.now(), 1),
               "wall_s": round(time.perf_counter() - self.clock.t0, 1), **self.counts,
               "mean_ttft_s": round(self.ttft_sum / n, 4) if n else None, "idle_jumps": self.clock.jumps,
               "idle_skipped_s": round(self.clock.skipped_s, 1)}
        self.write(rec)
        print("summary: " + json.dumps(rec), flush=True)


async def served_model(http, url: str) -> str:
    try:
        async with http.get(f"{url}/v1/models") as resp:
            return (await resp.json())["data"][0]["id"]
    except Exception as e:  # noqa: BLE001 - any failure here means there is no usable server
        sys.exit(f"STOP: no model at {url}/v1/models ({type(e).__name__}: {str(e)[:120]}); is the server up?")


async def replay(args: argparse.Namespace) -> None:
    header, trees, dropped = load_trees(args)
    print(f"{len(trees)} trees from {args.trace} ({header['source']}, {header['content']}); dropped over "
          f"--max-context: {dropped}; recipe {args.recipe}: "
          + (f"{args.lanes} lanes" if args.lanes else f"arrivals {args.arrival_rate:g}/s, max active "
             f"{args.max_active or 'uncapped'}") + f", duration {args.duration or 'until done'}, hints {args.hints}",
          flush=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    # /generate streams cumulatively: a line carries every output id so far, 250 KB at 20K tokens, past aiohttp's
    # default line limit of twice a 64 KiB buffer.
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0), read_bufsize=1 << 24) as http:
        model = await served_model(http, args.url)
        with open(args.output, "a") as out:
            rep = Replayer(args, header=header, trees=trees, out=out, http=http, model=model)
            await rep.run()


def main(argv=None) -> int:
    args = parse_args(argv)
    asyncio.run(replay(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
