"""Correctness checks a server must pass before it is measured. Stdlib + aiohttp; one request at a time.

  python3 check_server.py --url http://127.0.0.1:30000 --backend vllm|openai|continuum|tokencake
      [--prompt-tokens 20000] [--out 16] [--stage NAME:FILL_TOKENS:cached|recomputed ...] [--json out.json]

It sends token-id prompts as the replay does (the same backend class builds the body) and checks, in order:
  cold     a prompt never sent before: nothing cached, and the output has exactly the forced length
  warm     the same prompt again: all of it but the last block is cached, and the first token's top logprobs agree
           with the cold run's (another kernel path, so within WARM_TOL on the likely tokens)
  extend   the prompt, the cold reply and new tokens: the cached prefix covers the first prompt
  stages   for each --stage: FILL_TOKENS of unrelated prompts are sent first (to push the first prompt out of a
           tier), then the first prompt again. "cached" expects it served from a lower tier, and the same first-token
           logprobs as the warm run up to EXACT_TOL: the same computation on KV that must be a bit-exact copy.
           "recomputed" expects nothing cached and the cold run's logprobs (a system without that tier).
  --metrics   series of a server's Prometheus endpoint that the checked request of each stage moved: which tier
              served it (LMCache's lookup_hit_l1/l2 counters, for example)
Run it on a server booted with small pools, so a fill takes seconds: each baseline's README gives the boot line and
the stages that match its tiers. Exit status 0 when every check passes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))

from backends import BACKENDS, Turn  # noqa: E402
from program_trace import TokenSynth  # noqa: E402

# Cold and warm compute the last block on different kernel paths; bf16 moved a likely token's logprob by up to
# 0.13 on vLLM 0.34 (measured 2026-11-21). Arbitrary beyond that.
WARM_TOL = 0.25
# The same computation on a restored copy of the same KV: anything above float noise means the copy differs.
EXACT_TOL = 0.02
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


def same_first_token(a: dict | None, b: dict | None, *, tol: float) -> tuple:
    """(ok, largest logprob difference over the tokens both top lists hold). The top token must agree unless the two
    leading logprobs are within the tolerance of each other. Random-token prompts make the distribution flat (top
    logprob near -4.3), so every listed token counts."""
    if not a or not b:
        return False, None
    shared = set(a) & set(b)
    if not shared:
        return False, None
    worst = round(max(abs(a[t] - b[t]) for t in shared), 4)
    top_a, top_b = max(a, key=a.get), max(b, key=b.get)
    tie = top_a in b and abs(b[top_b] - b[top_a]) <= tol
    return worst <= tol and (top_a == top_b or tie), worst


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
        # a server that streams no token ids (our SGLang's OpenAI route) is checked on the usage count alone
        ids_ok = len(cold["ids"]) in ((a.out,) if self.backend.name != "openai" else (0, a.out))
        self.verdict("cold", cold["cached"] == 0 and cold["completion_tokens"] == a.out
                     and cold["prompt_tokens"] == a.prompt_tokens and ids_ok,
                     f"prompt {cold['prompt_tokens']}, cached {cold['cached']} (want 0), output "
                     f"{cold['completion_tokens']} with {len(cold['ids'])} ids (want {a.out}), TTFT {cold['ttft']:.3f} s",
                     ttft=cold["ttft"], cached=cold["cached"])
        warm = await self.send("A", prompt)
        same, worst = same_first_token(cold["top"], warm["top"], tol=WARM_TOL)
        self.verdict("warm", warm["cached"] >= a.prompt_tokens - block and same,
                     f"cached {warm['cached']} of {a.prompt_tokens} (want all but one block of {block}), TTFT "
                     f"{warm['ttft']:.3f} s against {cold['ttft']:.3f} s cold, first-token logprobs "
                     f"{'equal' if same else 'DIFFER'} (largest difference {worst})",
                     ttft=warm["ttft"], cached=warm["cached"], logprob_diff=worst)
        reply = cold["ids"] or self.synth.ids(f"chk|{self.salt}|R", a.out)
        more = prompt + reply + self.synth.ids(f"chk|{self.salt}|B", 2048)
        ext = await self.send("A", more)
        self.verdict("extend", ext["cached"] >= a.prompt_tokens - block,
                     f"prompt {ext['prompt_tokens']}, cached {ext['cached']} (want at least "
                     f"{a.prompt_tokens - block}), TTFT {ext['ttft']:.3f} s", ttft=ext["ttft"], cached=ext["cached"])
        refs = {warm["cached"]: ("warm", warm)}
        before = await self.metrics()
        for stage in a.stage:
            name, fill, want = stage.split(":")
            took = await self.fill(int(fill), name)
            mid = await self.metrics()
            back = await self.send("A", prompt)
            after = await self.metrics()
            moved = {k: round(after[k] - mid.get(k, 0.0), 1) for k in after if after[k] != mid.get(k, 0.0)}
            if moved:
                print(f"      {name}: metrics moved by the checked request: {moved}", flush=True)
            before = after
            if want == "cached":
                # The same computation needs the same split: a tier's granule can leave more to compute than a GPU
                # hit does, so the exact reference is the earlier result with the same cached count.
                ref_name, ref = refs.get(back["cached"], ("warm", warm))
                tol = EXACT_TOL if ref["cached"] == back["cached"] else WARM_TOL
                same, worst = same_first_token(ref["top"], back["top"], tol=tol)
                refs.setdefault(back["cached"], (name, back))
                ok = back["cached"] >= a.prompt_tokens - block and same
                text = (f"after {int(fill):,} filler tokens ({took:.0f} s): cached {back['cached']} of "
                        f"{a.prompt_tokens} (want all but one block), TTFT {back['ttft']:.3f} s against "
                        f"{warm['ttft']:.3f} s warm and {cold['ttft']:.3f} s cold, first-token logprobs against "
                        f"{ref_name} ({ref['cached']} cached) {'equal' if same else 'DIFFER'} (largest difference "
                        f"{worst}, tolerance {tol})")
            else:
                same, worst = same_first_token(cold["top"], back["top"], tol=EXACT_TOL)
                ok = back["cached"] < block and same
                text = (f"after {int(fill):,} filler tokens ({took:.0f} s): cached {back['cached']} (want 0: "
                        f"recomputed), TTFT {back['ttft']:.3f} s against {cold['ttft']:.3f} s cold, first-token "
                        f"logprobs {'equal' if same else 'DIFFER'} (largest difference {worst})")
            self.verdict(name, ok, text, ttft=back["ttft"], cached=back["cached"], logprob_diff=worst, fill_s=took,
                         metrics_moved=moved)
        return self.failed

    async def metrics(self) -> dict:
        """The values of every --metrics endpoint's series whose name matches --metric-match; {} without one."""
        out = {}
        for url in self.args.metrics:
            async with self.http.get(url) as resp:
                text = await resp.text()
            for line in text.splitlines():
                if line.startswith("#") or not re.search(self.args.metric_match, line.split(" ", 1)[0]):
                    continue
                key, _, value = line.rpartition(" ")
                try:
                    out[key] = float(value)
                except ValueError:
                    continue
        return out


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
    ap.add_argument("--metrics", action="append", default=[], metavar="URL",
                    help="a Prometheus endpoint whose matching series are read around each stage's checked request")
    ap.add_argument("--metric-match", default=r"hit|load|retriev|prefetch",
                    help="regular expression on series names for --metrics")
    ap.add_argument("--json", default=None)
    return asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
