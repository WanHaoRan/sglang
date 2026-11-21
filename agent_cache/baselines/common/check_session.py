"""Does a system's own mechanism act? One session's two turns around a wait under GPU pressure. Stdlib + aiohttp.

  python3 check_session.py --backend continuum|tokencake|vllm [--backend-opt KEY=VALUE ...] [--prompt-tokens 20000]
      [--fill 8x16000] [--wait-s 1.0] [--reply-key Bash/ls] [--metrics URL] [--json out.json]

Turn 0 (a prompt of --prompt-tokens) is sent with the fields the backend adds (Continuum's job and tool, TokenCake's
lifecycle metadata). As it decodes, --fill unrelated requests arrive at once and outgrow a small GPU pool, so the
session's prefix is the first thing an LRU cache would drop. When turn 0 completes, the backend's after-reply hook
runs (TokenCake's stall_started), the session waits --wait-s, the before-request hook runs (stall_finished), and turn
1 extends turn 0. The output is turn 1's cached tokens and TTFT. Run it twice, with the system's signals and without
(Continuum: --backend-opt func=off; TokenCake: events=off): the difference is what the mechanism kept. Boot the server
with a small GPU pool (each baseline's README gives the line); --metrics adds the moved series of an endpoint.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))

from backends import BACKENDS, Turn  # noqa: E402
from check_server import ask  # noqa: E402
from program_trace import TokenSynth  # noqa: E402


class Notifier:
    """What a backend's hooks need of the replayer: notify() for side calls, write() for their failures."""

    def __init__(self, http, url: str) -> None:
        self.http, self.url, self.errors = http, url, []

    async def notify(self, path: str, body: dict) -> None:
        async with self.http.post(self.url + path, json=body) as resp:
            text = await resp.text()
            if resp.status != 200:
                self.errors.append(f"{path} HTTP {resp.status}: {text[:200]}")
            print(f"      {path} {json.dumps(body)[:160]} -> {resp.status} {text[:160]}", flush=True)

    def write(self, rec: dict) -> None:
        self.errors.append(json.dumps(rec)[:200])


async def scrape(http, urls: list) -> dict:
    out = {}
    for url in urls:
        async with http.get(url) as resp:
            for line in (await resp.text()).splitlines():
                if line.startswith("#") or " " not in line:
                    continue
                key, _, value = line.rpartition(" ")
                try:
                    out[key] = float(value)
                except ValueError:
                    pass
    return out


async def main_async(args: argparse.Namespace) -> int:
    backend = BACKENDS[args.backend](dict(o.split("=", 1) for o in args.backend_opt))
    synth = TokenSynth(lo=100, hi=30000)
    salt = f"{time.time_ns():x}"
    n_fill, fill_tokens = (int(x) for x in args.fill.split("x"))
    session = f"cs-{salt}-c0"
    turn0 = Turn(rid=f"{session}-t0", session=session, k=0, last=False, reply_key=args.reply_key, next_gap="tool",
                 next_wait_s=args.wait_s, est_wait_s=args.wait_s, role="main")
    turn1 = Turn(rid=f"{session}-t1", session=session, k=1, last=True, reply_key=None, next_gap=None,
                 next_wait_s=None, est_wait_s=None, role="main")
    prompt0 = synth.ids(f"cs|{salt}|p0", args.prompt_tokens)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=1 << 24) as http:
        async with http.get(f"{args.url}/v1/models") as resp:
            model = (await resp.json())["data"][0]["id"]
        rep = Notifier(http, args.url)
        common = dict(url=args.url, backend=backend, model=model)
        print(f"server {args.url}, backend {backend.name} {backend.opts}, turn 0 {args.prompt_tokens} tokens, "
              f"fill {n_fill} x {fill_tokens}, wait {args.wait_s} s", flush=True)
        before = await scrape(http, args.metrics)

        async def filler(i: int):
            t = Turn(rid=f"cs-{salt}-f{i}-t0", session=f"cs-{salt}-f{i}", k=0, last=True, reply_key=None,
                     next_gap=None, next_wait_s=None, est_wait_s=None, role="filler")
            return await ask(http, **common, ids=synth.ids(f"cs|{salt}|f{i}", fill_tokens), out=args.fill_out,
                             rid=t.rid, session=t.session, turn=t)

        first = asyncio.create_task(ask(http, **common, ids=prompt0, out=args.out, rid=turn0.rid, session=session,
                                        turn=turn0))
        await asyncio.sleep(args.fill_after_s)   # turn 0 is admitted first; the fill lands while it decodes
        fills = [asyncio.create_task(filler(i)) for i in range(n_fill)]
        r0 = await first
        t_done = time.perf_counter()
        await backend.after_reply(rep, turn0)
        await asyncio.sleep(max(0.0, args.wait_s - (time.perf_counter() - t_done)))
        await backend.before_request(rep, turn1)
        prompt1 = prompt0 + (r0["ids"] or synth.ids(f"cs|{salt}|r0", args.out)) + synth.ids(f"cs|{salt}|p1", 512)
        r1 = await ask(http, **common, ids=prompt1, out=args.out, rid=turn1.rid, session=session, turn=turn1)
        done = await asyncio.gather(*fills)
        after = await scrape(http, args.metrics)
    moved = {k: round(after[k] - before.get(k, 0.0), 1) for k in after
             if after[k] != before.get(k, 0.0) and any(w in k for w in args.metric_words)}
    result = {"backend": backend.name, "opts": backend.opts, "prompt_tokens": args.prompt_tokens, "fill": args.fill,
              "wait_s": args.wait_s, "turn0": {"cached": r0["cached"], "ttft": round(r0["ttft"], 3)},
              "turn1": {"prompt": r1["prompt_tokens"], "cached": r1["cached"], "ttft": round(r1["ttft"], 3)},
              "fill_cached_mean": round(sum(d["cached"] for d in done) / max(1, len(done)), 1),
              "fill_ttft_max": round(max(d["ttft"] for d in done), 3), "hook_errors": rep.errors, "metrics": moved}
    print(f"turn 1: {r1['cached']} of {r1['prompt_tokens']} prompt tokens cached, TTFT {r1['ttft']:.3f} s "
          f"(turn 0 cold: {r0['ttft']:.3f} s); fill TTFT up to {result['fill_ttft_max']} s", flush=True)
    for k, v in moved.items():
        print(f"      metric {k}: +{v}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--prompt-tokens", type=int, default=20000)
    ap.add_argument("--out", type=int, default=16)
    ap.add_argument("--fill", default="8x16000", help="N x TOKENS unrelated requests sent while turn 0 decodes")
    ap.add_argument("--fill-out", type=int, default=64, help="output tokens of each filler (how long it holds blocks)")
    ap.add_argument("--fill-after-s", type=float, default=0.3, help="delay of the fill after turn 0 is sent")
    ap.add_argument("--wait-s", type=float, default=1.0, help="the session's wait between its two turns")
    ap.add_argument("--reply-key", default="Bash/ls", help="tool key of turn 0's reply")
    ap.add_argument("--metrics", action="append", default=[], metavar="URL")
    ap.add_argument("--metric-words", nargs="*", default=["hit", "cache", "offload", "tokencake", "pin", "load"])
    ap.add_argument("--json", default=None)
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
