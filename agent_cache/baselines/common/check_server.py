"""Correctness checks a server must pass before it is measured. Stdlib + aiohttp; one request at a time.

  python3 check_server.py --url http://127.0.0.1:30000 --backend vllm|openai|continuum|tokencake
      [--prompt-tokens 20000] [--out 16] [--stage NAME:FILL_TOKENS:cached|recomputed ...] [--json out.json]

It sends token-id prompts as the replay does (the same backend class builds the body) and checks, in order:
  cold     a prompt never sent before: nothing cached, and the output has exactly the forced length
  warm     the same prompt again: all of it but the last block is cached, and the first token's top logprobs equal
           the cold run's (a reply computed on a cached prefix must be the reply computed from scratch)
  extend   the prompt, the cold reply and new tokens: the cached prefix covers the first prompt
  stages   for each --stage: FILL_TOKENS of unrelated prompts are sent first (to push the first prompt out of a
           tier), then the first prompt again. "cached" expects it served from a lower tier with the same first-token
           logprobs; "recomputed" expects nothing cached (a system without that tier).
Run it on a server booted with small pools, so a fill takes seconds: each baseline's README gives the boot line and
the stages that match its tiers. Exit status 0 when every check passes.
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
from program_trace import TokenSynth  # noqa: E402

LOGPROB_TOL = 0.15      # bf16 prefill in different chunkings moves a logprob by a few hundredths
TOP = 5
FILL_PROMPT = 16384     # tokens per filler prompt


class Result(dict):
    """One request's outcome: ttft, latency, cached, prompt_tokens, completion_tokens, ids, top (first-token logprobs)."""


async def ask(http, *, url: str, backend, model: str, ids: list, out: int, rid: str, session: str) -> Result:
    turn = Turn(rid=rid, session=session, k=0, last=False, reply_key=None, next_gap=None, next_wait_s=None,
                est_wait_s=None, role="main")
    body = backend.body(model=model, out=out, turn=turn, want_ids=True)
    body["logprobs"] = TOP
    body["prompt"] = ids
    res = Result(ttft=None, ids=[], top=None, usage={})
    t0 = time.perf_counter()
    async with http.post(url + "/v1/completions", json=body) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("error"):
                raise RuntimeError(f"stream error: {json.dumps(d['error'])[:300]}")
            for ch in d.get("choices") or []:
                got = backend.reply_ids(ch)
                if (ch.get("text") or got) and res["ttft"] is None:
                    res["ttft"] = time.perf_counter() - t0
                res["ids"] += got
                tops = (ch.get("logprobs") or {}).get("top_logprobs") or []
                if tops and res["top"] is None:
                    res["top"] = tops[0]
            if d.get("usage"):
                res["usage"] = d["usage"]
    res["latency"] = time.perf_counter() - t0
    u = res["usage"]
    res["prompt_tokens"], res["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
    res["cached"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return res


def same_first_token(a: dict | None, b: dict | None) -> tuple:
    """(ok, largest logprob difference over the tokens both lists hold). The top token must agree unless the two
    leading logprobs are within the tolerance of each other."""
    if not a or not b:
        return False, None
    shared = set(a) & set(b)
    if not shared:
        return False, None
    worst = max(abs(a[t] - b[t]) for t in shared)
    top_a, top_b = max(a, key=a.get), max(b, key=b.get)
    tie = top_a in b and abs(b[top_b] - b[top_a]) <= LOGPROB_TOL
    return worst <= LOGPROB_TOL and (top_a == top_b or tie), worst


class Checker:
    def __init__(self, args: argparse.Namespace, *, http, backend, model: str) -> None:
        self.args, self.http, self.backend, self.model = args, http, backend, model
        self.synth = TokenSynth(lo=100, hi=30000)
        self.salt = args.salt or f"{time.time_ns():x}"   # a new salt per run: nothing sent before is in any tier
        self.rows, self.failed, self._n = [], 0, 0

    async def send(self, name: str, ids: list, *, out: int | None = None) -> Result:
        self._n += 1
        r = await ask(self.http, url=self.args.url, backend=self.backend, model=self.model, ids=ids,
                      out=out or self.args.out, rid=f"chk-{self.salt}-{self._n}", session=f"chk-{self.salt}-{name}")
        return r

    def verdict(self, name: str, ok: bool, text: str, **facts) -> None:
        self.failed += 0 if ok else 1
        self.rows.append({"check": name, "ok": ok, "text": text, **facts})
        print(f"{'ok  ' if ok else 'FAIL'}  {name:10s} {text}", flush=True)

    async def fill(self, tokens: int, tag: str) -> float:
        """Unrelated prompts totalling `tokens`, one at a time, one output token each; seconds taken."""
        t0, sent, n = time.perf_counter(), 0, 0
        while sent < tokens:
            size = min(FILL_PROMPT, tokens - sent)
            await self.send(f"fill-{tag}-{n}", self.synth.ids(f"fill|{self.salt}|{tag}|{n}", size), out=1)
            sent, n = sent + size, n + 1
        return time.perf_counter() - t0

    async def run(self) -> int:
        a = self.args
        prompt = self.synth.ids(f"chk|{self.salt}|A", a.prompt_tokens)
        block = a.block_tokens
        cold = await self.send("A", prompt)
        self.verdict("cold", cold["cached"] == 0 and cold["completion_tokens"] == a.out
                     and cold["prompt_tokens"] == a.prompt_tokens and len(cold["ids"]) == a.out,
                     f"prompt {cold['prompt_tokens']}, cached {cold['cached']} (want 0), output "
                     f"{cold['completion_tokens']} with {len(cold['ids'])} ids (want {a.out}), TTFT {cold['ttft']:.3f} s",
                     ttft=cold["ttft"], cached=cold["cached"])
        warm = await self.send("A", prompt)
        same, worst = same_first_token(cold["top"], warm["top"])
        self.verdict("warm", warm["cached"] >= a.prompt_tokens - block and same,
                     f"cached {warm['cached']} of {a.prompt_tokens} (want all but one block of {block}), TTFT "
                     f"{warm['ttft']:.3f} s against {cold['ttft']:.3f} s cold, first-token logprobs "
                     f"{'equal' if same else 'DIFFER'} (largest difference {worst})",
                     ttft=warm["ttft"], cached=warm["cached"], logprob_diff=worst)
        more = prompt + cold["ids"] + self.synth.ids(f"chk|{self.salt}|B", 2048)
        ext = await self.send("A", more)
        self.verdict("extend", ext["cached"] >= a.prompt_tokens - block,
                     f"prompt {ext['prompt_tokens']}, cached {ext['cached']} (want at least "
                     f"{a.prompt_tokens - block}), TTFT {ext['ttft']:.3f} s", ttft=ext["ttft"], cached=ext["cached"])
        for stage in a.stage:
            name, fill, want = stage.split(":")
            took = await self.fill(int(fill), name)
            back = await self.send("A", prompt)
            same, worst = same_first_token(cold["top"], back["top"])
            if want == "cached":
                ok = back["cached"] >= a.prompt_tokens - block and same
                text = (f"after {int(fill):,} filler tokens ({took:.0f} s): cached {back['cached']} of "
                        f"{a.prompt_tokens} (want all but one block), TTFT {back['ttft']:.3f} s against "
                        f"{cold['ttft']:.3f} s cold, first-token logprobs {'equal' if same else 'DIFFER'} "
                        f"(largest difference {worst})")
            else:
                ok = back["cached"] < block and same
                text = (f"after {int(fill):,} filler tokens ({took:.0f} s): cached {back['cached']} (want 0: "
                        f"recomputed), TTFT {back['ttft']:.3f} s against {cold['ttft']:.3f} s cold, first-token "
                        f"logprobs {'equal' if same else 'DIFFER'} (largest difference {worst})")
            self.verdict(name, ok, text, ttft=back["ttft"], cached=back["cached"], logprob_diff=worst, fill_s=took)
        return self.failed


async def main_async(args: argparse.Namespace) -> int:
    backend = BACKENDS[args.backend](dict(o.split("=", 1) for o in args.backend_opt))
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, read_bufsize=1 << 24) as http:
        async with http.get(f"{args.url}/v1/models") as resp:
            model = (await resp.json())["data"][0]["id"]
        chk = Checker(args, http=http, backend=backend, model=model)
        print(f"server {args.url}, model {model}, backend {backend.name}, prompt {args.prompt_tokens} tokens, "
              f"salt {chk.salt}", flush=True)
        failed = await chk.run()
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"url": args.url, "backend": args.backend, "model": model, "prompt_tokens": args.prompt_tokens,
                       "stages": args.stage, "failed": failed, "checks": chk.rows}, f, indent=1)
    print(f"{'PASS' if not failed else 'FAIL'}: {len(chk.rows) - failed} of {len(chk.rows)} checks passed", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--prompt-tokens", type=int, default=20000)
    ap.add_argument("--out", type=int, default=16, help="forced output length of the checked requests")
    ap.add_argument("--block-tokens", type=int, default=256,
                    help="largest cache granule of the system (a prefix is cached up to a multiple of it)")
    ap.add_argument("--stage", action="append", default=[], metavar="NAME:FILL_TOKENS:cached|recomputed")
    ap.add_argument("--salt", default=None, help="fix the synthetic tokens (default: new ones every run)")
    ap.add_argument("--json", default=None)
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
