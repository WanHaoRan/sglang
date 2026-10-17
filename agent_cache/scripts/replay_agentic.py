"""Gap-faithful multi-turn replay client for SGLang. Stdlib + aiohttp, nothing under python/.

  python3 replay_agentic.py --trace TRACE --output client.jsonl [--num-users N] [--turns N] [--arrival-rate R]
      [--gap-scale S] [--oracle-hints] [--concurrency C] [--timeout SECONDS] [--seed S] [--url URL] [--tag TAG]

A user replays one recorded conversation of the trace (convert_lmcache.py's JSON) as a closed loop: send turn 0, stream
the reply, append the reply text verbatim to the history (the RUNBOOK 4.5 replay template makes its KV a prefix hit),
sleep the recorded pre_gap x --gap-scale, at most GAP_CAP_S, counted from the end of the reply, send turn 1 = the history
plus its new messages (tool results, user prompt), and so on through the first min(--turns, recorded) turns.

User i replays conversation perm[i] of a seeded permutation of the trace, so no two users replay the same one, and
arrives at the i-th time of a Poisson process at --arrival-rate users/s (0: all at once). Both come from --seed with the
formulas of campaigns 14-17, so at the same seed and rate user i replays the same conversation and arrives at the same
time as there. --concurrency caps the users in session, each counted from its first turn to its last with the idle gaps
included: later arrivals wait in arrival order (conv_start.queued_s, outside every TTFT) and a returning turn never waits.

--oracle-hints: every turn but a user's last carries a kv_hints envelope with one sglang.next_turn v1 action, payload
{session, next_rid, next_gap_s}: the user, the next turn's request id and the exact sleep before it. The hint is
suppressed when --timeout certainly cuts the next turn; the turn record carries it either way (oracle_hint).

--timeout T: after T seconds no user starts and no request is sent. A sleeping user wakes at T and ends (conv_end
capped), a request in flight completes, waiting and later users are skipped (conv_skipped).

Output (--output, appended JSONL): a run record (flags, served model); per user conv_start, one turn record per request
(timings, usage, cached tokens by tier, reply_reprefilled = uncached tokens beyond the turn's new messages: ~0 +- a page
when the previous reply's KV was reused) and conv_end; a users record per minute (live and waiting users, requests
in flight); a summary. Request ids are <tag>-c<user>-t<turn>, the key into the server's logs.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import time
from typing import NamedTuple

import aiohttp

# Longest single sleep, as in campaigns 13-17: at x70, 57 of the 262K trace's 768 conversations hold a sleep over 5 h.
GAP_CAP_S = 1200.0
MONITOR_S = 60.0   # seconds between two users records (and progress lines)


class User(NamedTuple):
    conv: int                 # user id = arrival rank, the c<conv> of its request ids
    task_conv: int            # index in the trace of the conversation it replays
    session_id: str | None
    turns: list               # the recorded turns it replays
    arrival_s: float          # seconds after the start


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay recorded agent conversations against an SGLang server "
                                             "(module docstring).")
    ap.add_argument("--trace", required=True, help="the converted trace JSON (convert_lmcache.py)")
    ap.add_argument("--num-users", type=int, default=0,
                    help="users, each replaying a different conversation of the trace; 0 = one per conversation")
    ap.add_argument("--turns", type=int, default=0, help="each user replays the first min(this, recorded) turns; 0 = all")
    ap.add_argument("--arrival-rate", type=float, default=0.0,
                    help="users arrive as a Poisson process at this many per second; 0 = all at once")
    ap.add_argument("--gap-scale", type=float, default=1.0,
                    help=f"sleep the recorded pre_gap x this before each turn, at most {GAP_CAP_S:.0f} s; 0 = no idle")
    ap.add_argument("--oracle-hints", action="store_true",
                    help="send each turn's next request id and next gap as a kv_hints envelope (sglang.next_turn v1)")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="most users in session at once, idle ones included; later arrivals wait to start; 0 = no cap")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="seconds after which no user starts and no request is sent (requests in flight complete); "
                         "0 = none")
    ap.add_argument("--seed", type=int, default=0, help="seeds the arrival times and which user replays which conversation")
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tag", default="replay", help="request ids are <tag>-c<user>-t<turn>")
    ap.add_argument("--output", required=True, help="JSONL, appended")
    args = ap.parse_args()
    if min(args.num_users, args.turns, args.concurrency) < 0 or min(args.arrival_rate, args.gap_scale, args.timeout) < 0:
        ap.error("no flag takes a negative value")
    return args


def arrival_times(rate: float, n: int, rng: random.Random) -> list:
    """Offsets in seconds of n Poisson arrivals at `rate` per second, the first at 0; all 0 when rate is 0."""
    if rate <= 0:
        return [0.0] * n
    times, t = [], 0.0
    for _ in range(n):
        times.append(t)
        t += rng.expovariate(rate)
    return times


def load_users(args: argparse.Namespace) -> list:
    with open(args.trace) as f:
        data = json.load(f)
    convs = data["conversations"]
    ids = data.get("session_ids") or [None] * len(convs)
    pool = [k for k, c in enumerate(convs) if c]
    n = args.num_users or len(pool)
    if n > len(pool):
        sys.exit(f"--num-users {n}: the trace has {len(pool)} conversations and no two users replay the same one")
    # The streams of campaigns 14-17 (--order fresh --order-seed S, --seed S): the same users at the same times.
    perm = random.Random(f"{args.seed}:0").sample(range(len(pool)), len(pool))
    times = arrival_times(args.arrival_rate, n, random.Random(args.seed))
    return [User(i, pool[perm[i]], ids[pool[perm[i]]], convs[pool[perm[i]]][: args.turns or None], t)
            for i, t in enumerate(times)]


class Replayer:
    def __init__(self, args: argparse.Namespace, out, model: str, n_users: int) -> None:
        self.args = args
        self.out = out
        self.model = model
        self.n_users = n_users
        self.slots = asyncio.Semaphore(args.concurrency or n_users)
        self.deadline = args.timeout or math.inf   # seconds after t0
        self.arrived = self.live = self.done = 0   # waiting = arrived - live - done
        self.started = self.finished = self.skipped = 0
        self.in_flight = 0                         # requests sent and not yet completed
        self.stats = {"turns": 0, "errors": 0, "retries": 0, "capped_convs": 0, "ttft_sum": 0.0, "reprefill_sum": 0,
                      "reprefill_n": 0}
        self.t_start = time.time()
        self.t0 = time.perf_counter()

    def write(self, rec: dict) -> None:
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def rid(self, conv: int, turn: int) -> str:
        return f"{self.args.tag}-c{conv}-t{turn}"

    def gap_of(self, turn: dict) -> float:
        return min(GAP_CAP_S, max(0.0, float(turn.get("pre_gap") or 0.0) * self.args.gap_scale))

    def next_turn_hints(self, conv: int, turn: int, next_gap_s: float) -> dict:
        """kv_hints envelope sent with turn `turn`: when the user's next turn will be sent, and its rid."""
        payload = {"session": f"{self.args.tag}-c{conv}", "next_rid": self.rid(conv, turn + 1),
                   "next_gap_s": round(next_gap_s, 3)}
        return {"protocol_version": "1", "message_id": self.rid(conv, turn),
                "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn", "action_version": "1",
                             "payload": payload}]}

    def oracle_hint(self, user: User, turn: int, rec: dict) -> dict | None:
        """The kv_hints to send with this turn (None: hints off, the user's last turn, or suppressed); notes it in rec."""
        if not self.args.oracle_hints or turn + 1 >= len(user.turns):
            return None
        gap = self.gap_of(user.turns[turn + 1])
        # the next send is at least `gap` away: past the timeout it never happens
        suppressed = self.elapsed() + gap >= self.deadline
        rec["oracle_hint"] = {"next_rid": self.rid(user.conv, turn + 1), "next_gap_s": round(gap, 3),
                              "suppressed": suppressed}
        return None if suppressed else self.next_turn_hints(user.conv, turn, gap)

    async def chat_turn(self, http: aiohttp.ClientSession, messages: list, max_tokens: int, rid: str,
                        kv_hints: dict | None) -> dict:
        """One streamed chat completion: reply text, TTFT, latency, usage and the cached-token tier split."""
        body = {
            "rid": rid,   # shows up in the server's --log-requests and HICACHE_EVT lines
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "skip_special_tokens": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "return_cached_tokens_details": True,
        }
        if kv_hints is not None:
            body["kv_hints"] = kv_hints
        res = {"text": "", "ttft": None, "usage": None, "cached_details": None}
        t_send = time.perf_counter()
        self.in_flight += 1
        try:
            async with http.post(f"{self.args.url}/v1/chat/completions", json=body) as resp:
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
                    if d.get("error"):   # a scheduler abort after the 200 went out
                        raise RuntimeError(f"stream error: {json.dumps(d['error'])[:200]}")
                    for ch in d.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content") or ""
                        if piece:
                            if res["ttft"] is None:
                                res["ttft"] = time.perf_counter() - t_send
                            res["text"] += piece
                    if d.get("usage"):
                        res["usage"] = d["usage"]
                    details = (d.get("sglext") or {}).get("cached_tokens_details")
                    if details:
                        res["cached_details"] = details
        finally:
            self.in_flight -= 1
        res["t_send"] = t_send
        res["latency"] = time.perf_counter() - t_send
        return res

    async def chat_with_retry(self, http: aiohttp.ClientSession, messages: list, max_tokens: int, rid: str,
                              rec: dict, kv_hints: dict | None) -> tuple:
        """chat_turn with one retry on a pooled connection the server closed; returns (result or None, error)."""
        err = None
        for attempt in range(2):
            try:
                return await self.chat_turn(http, messages, max_tokens, rid, kv_hints), None
            except aiohttp.ClientConnectionError as e:
                # a pooled keep-alive connection the server closed meanwhile (ServerDisconnected / [Errno 104]):
                # retry once on a fresh connection, never after the timeout, and say so in the record
                err = e
                if attempt or self.elapsed() >= self.deadline:
                    break
                rec["retried"] = f"{type(e).__name__}: {str(e)[:120]}"
                self.stats["retries"] += 1
            except Exception as e:  # noqa: BLE001
                return None, e
        return None, err

    def skip(self, user: User) -> None:
        self.write({"kind": "conv_skipped", "conv": user.conv, "reason": "timeout", "arrival_s": round(user.arrival_s, 3)})
        self.skipped += 1

    async def user_session(self, http: aiohttp.ClientSession, user: User) -> None:
        self.arrived += 1
        async with self.slots:
            if self.elapsed() >= self.deadline:
                self.skip(user)
                self.done += 1
                return
            self.live += 1
            self.started += 1
            try:
                await self.play(http, user)
            finally:
                self.live -= 1
                self.done += 1

    def end_session(self, conv: int, turns_done: int, aborted: bool = False, capped: bool = False) -> None:
        rec = {"kind": "conv_end", "conv": conv, "turns_done": turns_done, "aborted": aborted}
        if capped:
            rec["capped"] = True
        rec["end_s"] = round(self.elapsed(), 3)
        self.write(rec)
        self.finished += not capped   # ended on its own, an aborted user included (as the analysis scripts count)

    def record_turn(self, rec: dict, r: dict, out_len: int, prev: tuple | None) -> tuple:
        """Fill and write a completed turn's record; returns its (prompt, completion) tokens for the next turn."""
        u, cd = r["usage"] or {}, r["cached_details"] or {}
        cached = sum(int(cd.get(k) or 0) for k in ("device", "host", "storage")) if cd \
            else int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        prompt, completion = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        rec.update(t_send=round(r["t_send"] - self.t0, 3), ttft=r["ttft"], latency=round(r["latency"], 4),
                   prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached, cached_details=cd or None)
        if completion < out_len:   # the server capped the reply (prompt + max_tokens near the window)
            rec["short_reply"] = out_len - completion
        if prev is not None and prompt:
            delta = prompt - sum(prev)   # this turn's new messages and the chat headers around them
            rec["delta_tokens"] = delta
            rec["reply_reprefilled"] = prompt - cached - delta   # ~0 (+- a page) when the previous reply was reused
            self.stats["reprefill_sum"] += rec["reply_reprefilled"]
            self.stats["reprefill_n"] += 1
        self.write(rec)
        self.stats["turns"] += 1
        if r["ttft"] is not None:
            self.stats["ttft_sum"] += r["ttft"]
        return prompt, completion

    async def play(self, http: aiohttp.ClientSession, user: User) -> None:
        conv = user.conv
        t_start = self.elapsed()
        self.write({"kind": "conv_start", "conv": conv, "task_conv": user.task_conv, "session_id": user.session_id,
                    "source": user.turns[0].get("source"), "turns": len(user.turns),
                    "arrival_s": round(user.arrival_s, 3), "start_s": round(t_start, 3),
                    "queued_s": round(t_start - user.arrival_s, 3)})
        history, prev = [], None
        for t, turn in enumerate(user.turns):
            gap = 0.0
            if t > 0:
                gap = max(0.0, min(self.gap_of(turn), self.deadline - self.elapsed()))   # no sleep crosses the timeout
                if gap > 0:
                    await asyncio.sleep(gap)   # counted from the end of the previous reply
                if self.elapsed() >= self.deadline:
                    self.stats["capped_convs"] += 1
                    self.end_session(conv, t, capped=True)
                    return
            history += turn["messages"]
            rec = {"kind": "turn", "conv": conv, "turn": t, "task_conv": user.task_conv, "source": turn.get("source"),
                   "gap_slept": round(gap, 3), "pre_gap": turn.get("pre_gap"), "output_length": turn.get("output_length"),
                   "replay_prompt_tokens": turn.get("replay_prompt_tokens")}
            hints = self.oracle_hint(user, t, rec)
            out_len = max(1, int(turn.get("output_length") or 1))
            r, err = await self.chat_with_retry(http, history, out_len, self.rid(conv, t), rec, hints)
            if r is None:
                rec["error"] = str(err)[:300]
                self.stats["errors"] += 1
                self.write(rec)
                self.end_session(conv, t, aborted=True)
                return
            prev = self.record_turn(rec, r, out_len, prev)
            history.append({"role": "assistant", "content": r["text"]})   # verbatim: this is what makes the KV a hit
        self.end_session(conv, len(user.turns))
        print(f"conv {conv} done: {len(user.turns)} turns, {self.elapsed():.0f} s elapsed", flush=True)

    async def monitor(self) -> None:
        """A users record and a progress line every MONITOR_S seconds, on the grid from the start."""
        started = finished = 0
        k = 1
        while True:
            await asyncio.sleep(max(0.0, k * MONITOR_S - self.elapsed()))
            k += 1
            t, waiting = self.elapsed(), self.arrived - self.live - self.done
            self.write({"kind": "users", "t": round(t, 3), "cap": self.args.concurrency or None, "live": self.live,
                        "waiting": waiting, "in_flight": self.in_flight, "admitted": self.started - started,
                        "finished": self.finished - finished})
            started, finished = self.started, self.finished
            print(f"[{t:.0f} s] users arrived {self.arrived}/{self.n_users}, live {self.live}, waiting {waiting}, "
                  f"done {self.done} ({self.finished} finished); turns {self.stats['turns']}, "
                  f"requests in flight {self.in_flight}", flush=True)

    def write_summary(self) -> None:
        s = self.stats
        summary = {"kind": "summary", "tag": self.args.tag, "elapsed_s": round(self.elapsed(), 1),
                   "users": self.n_users, "users_started": self.started, "finished_users": self.finished,
                   "skipped_users": self.skipped, "turns": s["turns"], "errors": s["errors"], "retries": s["retries"],
                   "capped_convs": s["capped_convs"], "capped": s["capped_convs"] + self.skipped > 0,
                   "mean_ttft_s": round(s["ttft_sum"] / s["turns"], 4) if s["turns"] else None,
                   "mean_reply_reprefilled": round(s["reprefill_sum"] / s["reprefill_n"], 1) if s["reprefill_n"] else None}
        self.write(summary)
        print("summary: " + json.dumps(summary), flush=True)

    async def run(self, http: aiohttp.ClientSession, users: list) -> None:
        self.write({"kind": "run", "tag": self.args.tag, "t_start": self.t_start, "model": self.model,
                    "args": {**vars(self.args), "gap_cap": GAP_CAP_S}, "users": len(users),
                    "turns": sum(len(u.turns) for u in users)})
        monitor = asyncio.create_task(self.monitor())
        sessions = []
        for user in users:
            if user.arrival_s >= self.deadline:   # arrives after the timeout: nothing to wait for
                self.skip(user)
                continue
            if user.arrival_s > self.elapsed():   # users already due start in this same step, in arrival order
                await asyncio.sleep(user.arrival_s - self.elapsed())
            sessions.append(asyncio.create_task(self.user_session(http, user)))
        await asyncio.gather(*sessions)
        monitor.cancel()
        self.write_summary()


async def served_model(http: aiohttp.ClientSession, url: str) -> str:
    """The model the server serves (/v1/models), sent as every request's model."""
    try:
        async with http.get(f"{url}/v1/models") as resp:
            return (await resp.json())["data"][0]["id"]
    except Exception as e:  # noqa: BLE001 - any failure here means there is no usable server
        sys.exit(f"STOP: no model at {url}/v1/models ({type(e).__name__}: {str(e)[:120]}); is the server up?")


async def replay(args: argparse.Namespace, users: list) -> None:
    # sock_read: a queued request sends nothing until its first token. limit=0: aiohttp's default pool of 100
    # connections queued requests past 100 in flight inside the client, which the server never saw (x1 campaign: p50
    # 7.4 s, p95 51 s before server entry).
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as http:
        model = await served_model(http, args.url)
        with open(args.output, "a") as out:
            await Replayer(args, out, model, len(users)).run(http, users)


def main() -> int:
    args = parse_args()
    users = load_users(args)
    arrivals = f"{args.arrival_rate:g}/s, the last at {users[-1].arrival_s:.0f} s" if args.arrival_rate > 0 \
        else "all at once"
    print(f"{len(users)} users, {sum(len(u.turns) for u in users)} turns, each user a different conversation of "
          f"{args.trace}; arrivals {arrivals}; concurrency {args.concurrency or 'uncapped'}; gaps x{args.gap_scale:g}, "
          f"at most {GAP_CAP_S:.0f} s; oracle hints {'on' if args.oracle_hints else 'off'}; timeout "
          f"{f'{args.timeout:g} s' if args.timeout else 'none'}", flush=True)
    asyncio.run(replay(args, users))
    return 0


if __name__ == "__main__":
    sys.exit(main())
