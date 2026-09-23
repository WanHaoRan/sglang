"""Gap-faithful multi-turn replay client for SGLang (RUNBOOK 4.6). Stdlib + aiohttp, nothing under python/.

One conversation = one closed-loop session: send turn 0, stream the reply, append the reply text
VERBATIM to the history (with the 4.5 replay template that makes its KV a prefix hit), sleep the
recorded pre_gap (scaled / capped / sampled) measured from the END of the reply, send turn 1 = history
+ its new messages (tool results + user prompt), and so on. --concurrency conversations are live at
once; the next one starts when one finishes. Every turn writes one JSON line (kind=turn): timings,
usage, the cached-token tier split, and reply_reprefilled = uncached tokens beyond this turn's delta
(0 +- one page when the previous reply was reused). With --check-ids the server also returns token
ids and the line carries the exact number of previous-reply ids found on the new prompt's prefix.
Optional paired recompute control (kind=control, RUNBOOK 6.5): every K turns a random-id prompt of the
turn's prompt_tokens length is sent to /generate outside the semaphore; its TTFT is the recompute
reference for that turn.

Trace: the 3.3 converter JSON (conversations[i] = list of turns with messages, output_length, pre_gap,
prompt_tokens, replay_prompt_tokens).

--dry-run sends nothing and needs no server: it walks the same conversations and turns in replay order (one
conversation after another, no sleeping) and prints, per turn, the gap a live run would sleep (with --gap-sample:
one seeded draw, exact only for --concurrency 1 closed loop), the token counts from the trace, every new message
verbatim (tool results and user prompts) and the token budget of the reply the server would generate (the reply
itself is skipped). Nothing is written to --output; --dry-run-max-chars shortens long messages.
"""
from __future__ import annotations   # keep the aiohttp annotations lazy so --dry-run works without aiohttp

import argparse
import asyncio
import json
import random
import signal
import sys
import time

try:
    import aiohttp
except ImportError:  # only the live replay needs it; --dry-run runs on a bare Python
    aiohttp = None

DELTA_HEADER_TOKENS = 64  # tolerance: prefix matching is page-floored (page 64)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--model", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--num-conversations", type=int, default=0, help="0 = all")
    ap.add_argument("--offset", type=int, default=0, help="rotate the conversation list")
    ap.add_argument("--max-turns", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=1, help="live conversations (closed loop); with --arrival-rate, the cap")
    ap.add_argument("--arrival-rate", type=float, default=0.0,
                    help="open loop: start conversations as a Poisson process at this many per second (0 = closed loop)")
    ap.add_argument("--gap-scale", type=float, default=1.0, help="sleep pre_gap x scale; 0 = no idle window")
    ap.add_argument("--gap-cap", type=float, default=0.0, help="cap each sleep at this many seconds; 0 = no cap")
    ap.add_argument("--gap-sample", default="", help="JSON list of gap seconds; sample per turn instead of pre_gap")
    ap.add_argument("--thinking", choices=["off", "on"], default="off", help="must match the server's replay template")
    ap.add_argument("--control-every", type=int, default=0, help="K: one recompute control every K returning turns; 0 = off")
    ap.add_argument("--check-ids", action="store_true", help="ask for sglext input/output ids and verify reply reuse exactly")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="wall cap: after this many seconds no conversation starts and running ones stop at their next turn boundary (the gap sleep never crosses the deadline, in-flight requests complete); 0 = none")
    ap.add_argument("--save-text", action="store_true", help="store each reply text in the JSONL (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", default="", help="JSONL, appended (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="send nothing: print each turn's messages and the token budget of the reply the server "
                         "would generate, then exit")
    ap.add_argument("--dry-run-max-chars", type=int, default=0,
                    help="with --dry-run, print at most this many chars of each message; 0 = the whole message")
    args = ap.parse_args()
    if not args.dry_run and not args.output:
        ap.error("--output is required (unless --dry-run)")
    return args


def load_conversations(args: argparse.Namespace) -> tuple:
    """(session ids, conversations) in replay order: empty ones dropped, then --offset, --num-conversations, --max-turns."""
    with open(args.trace) as f:
        data = json.load(f)
    ids = data.get("session_ids")
    if not ids or len(ids) != len(data["conversations"]):
        ids = [None] * len(data["conversations"])
    pairs = [(sid, c) for sid, c in zip(ids, data["conversations"]) if c]
    if args.offset:
        k = args.offset % len(pairs)
        pairs = pairs[k:] + pairs[:k]
    if args.num_conversations:
        pairs = pairs[: args.num_conversations]
    if args.max_turns:
        pairs = [(sid, c[: args.max_turns]) for sid, c in pairs]
    return [sid for sid, _ in pairs], [c for _, c in pairs]


def lcp(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Replayer:
    def __init__(self, args: argparse.Namespace, out) -> None:
        self.args = args
        self.out = out
        self.rng = random.Random(args.seed)
        self.gap_pool = json.load(open(args.gap_sample)) if args.gap_sample else None
        self.sem = asyncio.Semaphore(args.concurrency)
        self.t0 = time.perf_counter()
        self.stats = {"turns": 0, "controls": 0, "errors": 0, "retries": 0, "capped_convs": 0, "ttft_sum": 0.0, "reprefill_sum": 0, "reprefill_n": 0}
        self.controls: list = []

    def write(self, rec: dict) -> None:
        self.out.write(json.dumps(rec) + "\n")
        self.out.flush()

    def next_gap(self, pre_gap: float) -> float:
        g = self.rng.choice(self.gap_pool) if self.gap_pool else pre_gap
        g *= self.args.gap_scale
        if self.args.gap_cap > 0:
            g = min(g, self.args.gap_cap)
        return max(0.0, g)

    def dry_run(self, ids: list, convs: list) -> None:
        """Print what every turn would send and get back, in replay order; no request, no sleep, no JSONL line."""
        a = self.args
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")   # message content can hold anything; never die on the locale
        if getattr(signal, "SIGPIPE", None):
            signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # `| head` ends the dump quietly
        cap = f", capped at {a.gap_cap} s" if a.gap_cap else ""
        if self.gap_pool:
            gap_note = (f"gap per turn = one seeded draw from {a.gap_sample} x {a.gap_scale}{cap}: exactly what a live run at "
                        f"--concurrency 1 (closed loop) sleeps; at higher concurrency or with --arrival-rate the live run assigns "
                        f"the draws to turns differently")
            gap_label = "sampled gap"
        else:
            gap_note = f"gap per turn = pre_gap x {a.gap_scale}{cap} = exactly what a live run sleeps"
            gap_label = "a live run would sleep"
        print(f"DRY RUN: nothing is sent to {a.url}, nothing slept, nothing written; conversations play one after another; "
              f"{gap_note}; live-only flags (--concurrency, --arrival-rate, --control-every, --check-ids, --max-seconds) "
              f"have no effect", flush=True)
        n_msgs = 0
        for conv, (sid, turns) in enumerate(zip(ids, convs)):
            first = turns[0]
            print(f"\n{'=' * 100}\nconv {conv}: session {sid or '?'}  source {first.get('source')}  "
                  f"recorded with {first.get('orig_model')}  turns {len(turns)}  "
                  f"final replay prompt {turns[-1].get('replay_prompt_tokens')} tokens\n{'=' * 100}")
            n_history = 0   # messages the live run would already carry: earlier turns' messages + one reply each
            for turn_idx, turn in enumerate(turns):
                gap = self.next_gap(float(turn.get("pre_gap") or 0.0)) if turn_idx > 0 else 0.0
                msgs = list(turn["messages"])
                merged = int(turn.get("merged_output_length") or 0)
                max_tokens = max(1, int(turn.get("output_length") or 1))   # what conversation() + chat_turn() send
                print(f"\n--- conv {conv} turn {turn_idx} [rid {a.tag or 'replay'}-c{conv}-t{turn_idx}]  "
                      f"trace iteration {turn.get('iteration')}, rows {turn.get('source_rows')}\n"
                      f"    gap: recorded pre_gap {float(turn.get('pre_gap') or 0.0):.3f} s, {gap_label} {gap:.3f} s\n"
                      f"    prompt: {n_history} history + {len(msgs)} new messages = {turn.get('replay_prompt_tokens')} replay tokens "
                      f"(recorded {turn.get('prompt_tokens')}); {turn.get('n_assistant_in_delta')} recorded assistant message(s) "
                      f"dropped from this delta\n"
                      f"    reply: max_tokens {max_tokens}"
                      f"{' (+' + str(merged) + ' tokens of merged-forward calls, not replayed)' if merged else ''}")
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
                      f"{'' if turn_idx == len(turns) - 1 else ', re-fed verbatim as history on the next turn'}")
                n_history += len(msgs) + 1
            print(f"conv {conv} done: {len(turns)} turns (dry run)")
        print(f"\ndry run summary: {len(convs)} conversations, {sum(len(c) for c in convs)} turns, {n_msgs} messages printed, "
              f"0 requests sent{', ' + a.output + ' not written' if a.output else ''}", flush=True)

    async def chat_turn(self, session: aiohttp.ClientSession, messages: list, max_tokens: int, rid: str = "") -> dict:
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
        if self.args.check_ids:
            body["return_input_ids_in_sglext"] = True
            body["return_output_ids_in_sglext"] = True
        res = {"text": "", "ttft": None, "usage": None, "cached_details": None, "input_ids": None, "output_ids": None}
        t_send = time.perf_counter()
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
                for ch in d.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ""
                    if piece:
                        if res["ttft"] is None:
                            res["ttft"] = time.perf_counter() - t_send
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
        res["t_send"] = t_send
        res["latency"] = time.perf_counter() - t_send
        return res

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

    async def conversation(self, session: aiohttp.ClientSession, conv: int, turns: list) -> None:
        t_arrival = time.perf_counter() - self.t0
        async with self.sem:
            if self.args.max_seconds and time.perf_counter() - self.t0 > self.args.max_seconds:
                self.write({"kind": "conv_skipped", "conv": conv, "reason": "max_seconds"})
                return
            t_start = time.perf_counter() - self.t0
            self.write({"kind": "conv_start", "conv": conv, "turns": len(turns), "arrival_s": round(t_arrival, 3),
                        "start_s": round(t_start, 3), "queued_s": round(t_start - t_arrival, 3)})
            history: list = []
            prev = None  # last turn's result, for the reuse check
            for turn_idx, turn in enumerate(turns):
                gap_slept = 0.0
                if turn_idx > 0:
                    gap_slept = self.next_gap(float(turn.get("pre_gap") or 0.0))
                    if self.args.max_seconds:   # wall cap: never sleep past the deadline, then stop at this turn boundary
                        gap_slept = max(0.0, min(gap_slept, self.args.max_seconds - (time.perf_counter() - self.t0)))
                    if gap_slept > 0:
                        await asyncio.sleep(gap_slept)   # measured from the end of the previous reply
                    if self.args.max_seconds and time.perf_counter() - self.t0 >= self.args.max_seconds:
                        self.stats["capped_convs"] += 1
                        self.write({"kind": "conv_end", "conv": conv, "turns_done": turn_idx, "aborted": False, "capped": True})
                        return
                history = history + list(turn["messages"])
                if self.args.control_every and turn_idx > 0 and (turn_idx + conv) % self.args.control_every == 0 \
                        and int(turn.get("prompt_tokens") or 0) > 0:
                    self.controls.append(asyncio.create_task(
                        self.control(session, conv, turn_idx, int(turn["prompt_tokens"]))))
                rec = {"kind": "turn", "conv": conv, "turn": turn_idx, "source": turn.get("source"),
                       "gap_slept": round(gap_slept, 3), "pre_gap": turn.get("pre_gap"),
                       "output_length": turn.get("output_length"),
                       "replay_prompt_tokens": turn.get("replay_prompt_tokens")}
                r, err = None, None
                for attempt in range(2):
                    try:
                        r = await self.chat_turn(session, history, int(turn.get("output_length") or 1),
                                                 rid=f"{self.args.tag or 'replay'}-c{conv}-t{turn_idx}")
                        break
                    except aiohttp.ClientConnectionError as e:
                        # a pooled keep-alive connection the server closed meanwhile (ServerDisconnected / [Errno 104]):
                        # retry once on a fresh connection and say so in the record
                        err = e
                        if attempt == 0:
                            rec["retried"] = f"{type(e).__name__}: {str(e)[:120]}"
                            self.stats["retries"] += 1
                    except Exception as e:  # noqa: BLE001
                        err = e
                        break
                if r is None:
                    rec["error"] = str(err)[:300]
                    self.stats["errors"] += 1
                    self.write(rec)
                    self.write({"kind": "conv_end", "conv": conv, "turns_done": turn_idx, "aborted": True})
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
                if prev is not None and prompt_tokens:
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
                history.append({"role": "assistant", "content": r["text"]})   # verbatim: this is what makes the KV a hit
                prev = {"prompt_tokens": prompt_tokens, "completion": completion,
                        "ids": (r["input_ids"] or []) + (r["output_ids"] or []) if self.args.check_ids else None,
                        "prompt_ids": len(r["input_ids"] or [])}
            self.write({"kind": "conv_end", "conv": conv, "turns_done": len(turns), "aborted": False})
            print(f"conv {conv} done: {len(turns)} turns, {time.perf_counter() - self.t0:.0f} s elapsed", flush=True)

    async def run(self, convs: list) -> None:
        self.write({"kind": "run", "tag": self.args.tag, "t_start": time.time(), "args": vars(self.args),
                    "conversations": len(convs), "turns": sum(len(c) for c in convs)})
        timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if self.args.arrival_rate > 0:
                # open loop: conversations arrive as a Poisson process; --concurrency only caps how many are live
                tasks = []
                for i, c in enumerate(convs):
                    tasks.append(asyncio.create_task(self.conversation(session, i, c)))
                    await asyncio.sleep(self.rng.expovariate(self.args.arrival_rate))
                await asyncio.gather(*tasks)
            else:
                await asyncio.gather(*(self.conversation(session, i, c) for i, c in enumerate(convs)))
            if self.controls:
                await asyncio.gather(*self.controls)
        s = self.stats
        summary = {"kind": "summary", "tag": self.args.tag, "elapsed_s": round(time.perf_counter() - self.t0, 1),
                   "turns": s["turns"], "controls": s["controls"], "errors": s["errors"], "retries": s["retries"],
                   "capped_convs": s["capped_convs"], "capped": s["capped_convs"] > 0,
                   "mean_ttft_s": round(s["ttft_sum"] / s["turns"], 4) if s["turns"] else None,
                   "mean_reply_reprefilled": round(s["reprefill_sum"] / s["reprefill_n"], 1) if s["reprefill_n"] else None}
        self.write(summary)
        print("summary: " + json.dumps(summary), flush=True)


def main() -> int:
    args = parse_args()
    ids, convs = load_conversations(args)
    selection = f"#Conversations: {len(convs)}  turns: {sum(len(c) for c in convs)}"
    if args.dry_run:
        print(selection, flush=True)
        Replayer(args, None).dry_run(ids, convs)
        return 0
    print(f"{selection}  concurrency: {args.concurrency}"
          f"{' (cap), arrival ' + str(args.arrival_rate) + ' conv/s' if args.arrival_rate > 0 else ' (closed loop)'}  "
          f"gap x{args.gap_scale}{' cap ' + str(args.gap_cap) + 's' if args.gap_cap else ''}"
          f"{' sampled from ' + args.gap_sample if args.gap_sample else ''}  thinking: {args.thinking}", flush=True)
    if aiohttp is None:
        sys.exit("aiohttp is not installed: the live replay needs it (only --dry-run runs without it)")
    with open(args.output, "a") as out:
        asyncio.run(Replayer(args, out).run(convs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
