"""A CPU stand-in for the SGLang server, for testing the replay clients without a GPU. Stdlib + aiohttp.

  python3 mock_engine.py --port 30000 [--device-tokens N] [--host-tokens N] [--storage-tokens N] [--time-scale S]

It speaks the wire format replay_programs.py relies on: native /generate (input_ids, sampling_params, rid, kv_hints;
streamed {"text", "output_ids", "meta_info"} chunks with prompt_tokens, completion_tokens, cached_tokens and
cached_tokens_details {device, host, storage}), /v1/completions with a token-id prompt and OpenAI usage, /v1/models,
/health and a /programs/release endpoint (ThunderAgent's and MORI's release call). Unknown /generate fields are
rejected, as the real server's GenerateReqInput does.

The cache is a page-level prefix cache over three tiers with LRU in each: device pages hold the running requests' KV and
whatever idle prefix caches fit beside them; with write-through (the default, as campaign 26's host and SSD tiers) every
new page is also copied to host and storage, so a device eviction only drops the device copy. A request waits until
its prompt and output fit in the device pool beside the running ones, prefills one at a time (chunked prefill is not
modelled), then decodes alongside the others. The time model is deliberately coarse: it orders tiers and queueing
correctly, and its numbers are not the H200's; nothing measured on it belongs in a paper. --time-scale shrinks every
engine delay (0.01: 100x faster) for fast tests.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import sys
import time
from collections import OrderedDict

from aiohttp import web

GENERATE_FIELDS = {"input_ids", "text", "sampling_params", "stream", "rid", "kv_hints", "return_logprob", "priority",
                   "custom_labels", "no_logs", "extra_key", "log_metrics", "routing_key", "conversation_id"}
SAMPLING_FIELDS = {"max_new_tokens", "min_new_tokens", "ignore_eos", "temperature", "top_p", "top_k", "stop",
                   "skip_special_tokens", "no_stop_trim", "stop_token_ids"}


ARGS = web.AppKey("args", argparse.Namespace)


class TieredCache:
    """Page-level prefix cache, three LRU tiers keyed by chained page hashes."""

    def __init__(self, *, page: int, device_pages: int, host_pages: int, storage_pages: int, write_through: bool):
        self.page = page
        self.cap = {"device": device_pages, "host": host_pages, "storage": storage_pages}
        self.tiers = {name: OrderedDict() for name in self.cap}
        self.write_through = write_through
        self.pinned = 0   # device pages held by running requests
        self.evictions = {name: 0 for name in self.cap}

    def page_hashes(self, ids: list) -> list:
        out, h = [], 0
        for i in range(0, len(ids) - self.page + 1, self.page):
            h = hash((h, tuple(ids[i: i + self.page])))
            out.append(h)
        return out

    def match(self, hashes: list) -> dict:
        """Cached prefix pages by tier: device first, then host, then storage, stopping at the first miss."""
        got = {"device": 0, "host": 0, "storage": 0}
        i = 0
        for tier in ("device", "host", "storage"):
            od = self.tiers[tier]
            while i < len(hashes) and hashes[i] in od:
                od.move_to_end(hashes[i])
                got[tier] += 1
                i += 1
        return got

    def fits(self, pages: int) -> bool:
        """Whether a request of `pages` fits once every idle device page is evicted."""
        return pages <= self.cap["device"] - self.pinned

    def pin(self, *, device_hits: list, pages: int) -> None:
        """A request starts: its device hits leave the idle LRU, and `pages` are held until it ends."""
        dev = self.tiers["device"]
        for h in device_hits:
            dev.pop(h, None)
        self.pinned += pages
        self._trim_device()

    def unpin(self, *, pages: int, hashes: list) -> None:
        """A request ends: its pages become idle device pages (MRU), copied down the tiers under write-through."""
        self.pinned -= pages
        dev = self.tiers["device"]
        for h in hashes:
            dev[h] = None
            dev.move_to_end(h)
            if self.write_through:
                self._put("host", h)
                self._put("storage", h)
        self._trim_device()

    def _trim_device(self) -> None:
        dev = self.tiers["device"]
        while dev and len(dev) + self.pinned > self.cap["device"]:
            h, _ = dev.popitem(last=False)
            self.evictions["device"] += 1
            if not self.write_through:
                self._put("host", h)

    def _put(self, tier: str, h: int) -> None:
        if self.cap[tier] <= 0:
            return
        od = self.tiers[tier]
        od[h] = None
        od.move_to_end(h)
        while len(od) > self.cap[tier]:
            old, _ = od.popitem(last=False)
            self.evictions[tier] += 1
            if tier == "host" and not self.write_through:
                self._put("storage", old)


class MockEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        pages = {k: getattr(args, f"{k}_tokens") // args.page for k in ("device", "host", "storage")}
        self.cache = TieredCache(page=args.page, device_pages=pages["device"], host_pages=pages["host"],
                                 storage_pages=pages["storage"], write_through=not args.write_back)
        self.admit = asyncio.Condition()
        self.prefill_lock = asyncio.Lock()
        self.running = 0
        self.stats = {"requests": 0, "prompt_tokens": 0, "cached": {"device": 0, "host": 0, "storage": 0},
                      "releases": 0, "kv_hints": 0, "hint_versions": {}, "rejected": 0, "queued_max": 0}
        self.waiting = 0

    def _sleep(self, seconds: float):
        return asyncio.sleep(max(0.0, seconds) * self.args.time_scale)

    def prefill_seconds(self, *, n: int, cached: dict) -> float:
        c = self.args.page * sum(cached.values())
        uncached = max(0, n - c)
        return (self.args.prefill_fixed + uncached / self.args.prefill_tps + self.args.prefill_quad * (n * n - c * c)
                + self.args.page * cached["host"] / self.args.host_tps
                + self.args.page * cached["storage"] / self.args.storage_tps)

    def output_ids(self, rid: str, n: int) -> list:
        out, ctr = [], 0
        while len(out) < n:
            d = hashlib.blake2b(f"{rid}|{ctr}".encode(), digest_size=64).digest()
            out.extend(100 + x % 29900 for x in struct.unpack("<16I", d))
            ctr += 1
        return out[:n]

    async def run(self, *, ids: list, out: int, rid: str, emit) -> tuple:
        """Admit, prefill, decode; emit(output ids so far, meta) per chunk; returns (final meta_info, output ids)."""
        t0 = time.perf_counter()
        hashes = self.cache.page_hashes(ids)
        need = (len(ids) + out) // self.args.page + 1
        self.waiting += 1
        self.stats["queued_max"] = max(self.stats["queued_max"], self.waiting)
        async with self.admit:
            await self.admit.wait_for(lambda: self.running < self.args.max_running and self.cache.fits(need))
            self.waiting -= 1
            self.running += 1
            cached = self.cache.match(hashes)
            self.cache.pin(device_hits=hashes[: cached["device"]], pages=need)
        gen = None
        try:
            async with self.prefill_lock:
                await self._sleep(self.prefill_seconds(n=len(ids), cached=cached))
            gen = self.output_ids(rid, out)
            meta = {"id": rid, "prompt_tokens": len(ids), "completion_tokens": 0,
                    "cached_tokens": self.args.page * sum(cached.values()),
                    "cached_tokens_details": {k: self.args.page * v for k, v in cached.items()}}
            if self.args.storage_tokens <= 0:
                meta["cached_tokens_details"].pop("storage")
            sent = 0
            while sent < out:
                n = min(self.args.chunk, out - sent)
                await self._sleep(self.args.tpot * n)
                sent += n
                meta["completion_tokens"] = sent
                await emit(gen[:sent], dict(meta))
            meta["e2e_latency"] = time.perf_counter() - t0
            meta["finish_reason"] = {"type": "length", "length": out}
            self.stats["requests"] += 1
            self.stats["prompt_tokens"] += len(ids)
            for k, v in cached.items():
                self.stats["cached"][k] += self.args.page * v
            return meta, gen
        finally:
            async with self.admit:
                self.running -= 1
                self.cache.unpin(pages=need, hashes=self.cache.page_hashes(ids + gen) if gen is not None else hashes)
                self.admit.notify_all()

    def note_hints(self, hints) -> None:
        if not hints:
            return
        if not isinstance(hints, dict) or not isinstance(hints.get("actions"), list):
            raise ValueError("Malformed kv_hints envelope")
        self.stats["kv_hints"] += 1
        for a in hints["actions"]:
            v = str(a.get("action_version"))
            self.stats["hint_versions"][v] = self.stats["hint_versions"].get(v, 0) + 1


ENGINE = web.AppKey("engine", MockEngine)


async def generate(request: web.Request) -> web.StreamResponse:
    eng: MockEngine = request.app[ENGINE]
    body = await request.json()
    bad = set(body) - GENERATE_FIELDS
    sp = body.get("sampling_params") or {}
    bad |= {f"sampling_params.{k}" for k in set(sp) - SAMPLING_FIELDS}
    if bad or not isinstance(body.get("input_ids"), list):
        eng.stats["rejected"] += 1
        return web.json_response({"error": {"message": f"unexpected fields {sorted(bad)} or no input_ids"}}, status=400)
    try:
        eng.note_hints(body.get("kv_hints"))
    except ValueError as e:
        return web.json_response({"error": {"message": str(e)}}, status=400)
    ids, out = body["input_ids"], int(sp.get("max_new_tokens") or 16)
    rid = body.get("rid") or f"mock-{eng.stats['requests']}"
    if not body.get("stream"):
        meta, gen = await eng.run(ids=ids, out=out, rid=rid, emit=_noop)
        return web.json_response({"text": "", "output_ids": gen, "meta_info": meta})
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    async def emit(gen_so_far, meta):
        chunk = {"text": "", "output_ids": gen_so_far, "meta_info": meta}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())

    meta, gen = await eng.run(ids=ids, out=out, rid=rid, emit=emit)
    await resp.write(f"data: {json.dumps({'text': '', 'output_ids': gen, 'meta_info': meta})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def completions(request: web.Request) -> web.StreamResponse:
    eng: MockEngine = request.app[ENGINE]
    body = await request.json()
    ids = body.get("prompt")
    if not isinstance(ids, list) or (ids and not isinstance(ids[0], int)):
        return web.json_response({"error": {"message": "mock: prompt must be a token-id list"}}, status=400)
    out, rid = int(body.get("max_tokens") or 16), body.get("rid") or f"mock-{eng.stats['requests']}"
    want_ids = bool(body.get("return_token_ids"))
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    last = {"n": 0}

    async def emit(gen_so_far, meta):
        delta = gen_so_far[last["n"]:]
        last["n"] = len(gen_so_far)
        choice = {"index": 0, "text": " ".join("t" for _ in delta), "finish_reason": None}
        if want_ids:
            choice["token_ids"] = delta
        await resp.write(f"data: {json.dumps({'object': 'text_completion', 'choices': [choice]})}\n\n".encode())

    meta, gen = await eng.run(ids=ids, out=out, rid=rid, emit=emit)
    usage = {"prompt_tokens": meta["prompt_tokens"], "completion_tokens": meta["completion_tokens"],
             "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
             "prompt_tokens_details": {"cached_tokens": meta["cached_tokens"]}}
    await resp.write(f"data: {json.dumps({'object': 'text_completion', 'choices': [], 'usage': usage})}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def _noop(*_):
    return None


async def release(request: web.Request) -> web.Response:
    request.app[ENGINE].stats["releases"] += 1
    return web.json_response({"released": True})


async def models(request: web.Request) -> web.Response:
    return web.json_response({"object": "list", "data": [{"id": request.app[ARGS].model, "object": "model"}]})


async def health(_: web.Request) -> web.Response:
    return web.Response(text="ok")


async def mock_stats(request: web.Request) -> web.Response:
    eng: MockEngine = request.app[ENGINE]
    c = eng.cache
    return web.json_response({**eng.stats, "running": eng.running, "waiting": eng.waiting,
                              "pages": {k: len(v) for k, v in c.tiers.items()}, "pinned_pages": c.pinned,
                              "evictions": c.evictions})


def build_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(client_max_size=1 << 30)
    app[ARGS] = args
    app.on_startup.append(_start)
    app.add_routes([web.post("/generate", generate), web.post("/v1/completions", completions),
                    web.post("/programs/release", release), web.get("/v1/models", models),
                    web.get("/health", health), web.get("/mock/stats", mock_stats)])
    return app


async def _start(app: web.Application) -> None:
    app[ENGINE] = MockEngine(app[ARGS])


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="mock-model")
    ap.add_argument("--page", type=int, default=16, help="tokens per page")
    ap.add_argument("--device-tokens", type=int, default=668160, help="device KV pool (campaign 26's GPU pool)")
    ap.add_argument("--host-tokens", type=int, default=2_000_000, help="host tier; 0 = none")
    ap.add_argument("--storage-tokens", type=int, default=0, help="storage tier; 0 = none")
    ap.add_argument("--write-back", action="store_true", help="demote on eviction instead of writing through")
    ap.add_argument("--max-running", type=int, default=256, help="most requests admitted at once")
    # Prefill defaults: T(L) = 0.092 + 0.84e-6 L + 3.21e-9 L^2 s, fitted on 4K-65K tokens for Qwen3-30B-A3B on the H200
    # (results/cpu_calib_c44/FINDINGS.md:156); past 65K it is an extrapolation.
    ap.add_argument("--prefill-fixed", type=float, default=0.092, help="seconds per prefill")
    ap.add_argument("--prefill-tps", type=float, default=1.19e6, help="uncached prompt tokens per second (linear)")
    ap.add_argument("--prefill-quad", type=float, default=3.21e-9, help="seconds per token^2 of attention (L^2 - c^2)")
    ap.add_argument("--host-tps", type=float, default=400000.0, help="tokens per second loaded from host (estimate)")
    # 2.11 GB/s shared SSD rate measured on the box over 98,304 bf16 KV bytes per token
    ap.add_argument("--storage-tps", type=float, default=21500.0, help="tokens per second loaded from storage")
    ap.add_argument("--tpot", type=float, default=0.012, help="seconds per output token")
    ap.add_argument("--chunk", type=int, default=16, help="output tokens per streamed chunk")
    ap.add_argument("--time-scale", type=float, default=1.0, help="multiplies every engine delay")
    return ap.parse_args(argv)


def main() -> int:
    args = parse_args()
    web.run_app(build_app(args), host=args.host, port=args.port, print=lambda *_: None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
