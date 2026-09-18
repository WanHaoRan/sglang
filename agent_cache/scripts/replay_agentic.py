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
"""
import argparse
import asyncio
import json
import random
import sys
import time

import aiohttp

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
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop starting conversations after this many seconds; 0 = none")
    ap.add_argument("--save-text", action="store_true", help="store each reply text in the JSONL (debugging)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--output", required=True, help="JSONL, appended")
    return ap.parse_args()


def load_conversations(args: argparse.Namespace) -> list:
    with open(args.trace) as f:
        data = json.load(f)
    convs = [c for c in data["conversations"] if c]
    if args.offset:
        k = args.offset % len(convs)
        convs = convs[k:] + convs[:k]
    if args.num_conversations:
        convs = convs[: args.num_conversations]
    if args.max_turns:
        convs = [c[: args.max_turns] for c in convs]
    return convs


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
        self.stats = {"turns": 0, "controls": 0, "errors": 0, "ttft_sum": 0.0, "reprefill_sum": 0, "reprefill_n": 0}
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

    async def chat_turn(self, session: aiohttp.ClientSession, messages: list, max_tokens: int) -> dict:
        """One streamed chat completion; returns text, timings, usage and sglext fields."""
        body = {
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
                    if gap_slept > 0:
                        await asyncio.sleep(gap_slept)   # measured from the end of the previous reply
                history = history + list(turn["messages"])
                if self.args.control_every and turn_idx > 0 and (turn_idx + conv) % self.args.control_every == 0 \
                        and int(turn.get("prompt_tokens") or 0) > 0:
                    self.controls.append(asyncio.create_task(
                        self.control(session, conv, turn_idx, int(turn["prompt_tokens"]))))
                rec = {"kind": "turn", "conv": conv, "turn": turn_idx, "source": turn.get("source"),
                       "gap_slept": round(gap_slept, 3), "pre_gap": turn.get("pre_gap"),
                       "output_length": turn.get("output_length"),
                       "replay_prompt_tokens": turn.get("replay_prompt_tokens")}
                try:
                    r = await self.chat_turn(session, history, int(turn.get("output_length") or 1))
                except Exception as e:  # noqa: BLE001
                    rec["error"] = str(e)[:300]
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
                   "turns": s["turns"], "controls": s["controls"], "errors": s["errors"],
                   "mean_ttft_s": round(s["ttft_sum"] / s["turns"], 4) if s["turns"] else None,
                   "mean_reply_reprefilled": round(s["reprefill_sum"] / s["reprefill_n"], 1) if s["reprefill_n"] else None}
        self.write(summary)
        print("summary: " + json.dumps(summary), flush=True)


def main() -> int:
    args = parse_args()
    convs = load_conversations(args)
    print(f"#Conversations: {len(convs)}  turns: {sum(len(c) for c in convs)}  concurrency: {args.concurrency}"
          f"{' (cap), arrival ' + str(args.arrival_rate) + ' conv/s' if args.arrival_rate > 0 else ' (closed loop)'}  "
          f"gap x{args.gap_scale}{' cap ' + str(args.gap_cap) + 's' if args.gap_cap else ''}"
          f"{' sampled from ' + args.gap_sample if args.gap_sample else ''}  thinking: {args.thinking}", flush=True)
    with open(args.output, "a") as out:
        asyncio.run(Replayer(args, out).run(convs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
