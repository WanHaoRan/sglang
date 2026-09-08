"""Background write load: unique prompts at a Poisson rate.

Every prompt is unique, so each request inserts new radix nodes; under
write_through that is len x 144 KiB of D2H plus the same to L3. This is the
knob Exp 2 sweeps.
"""
import argparse
import asyncio
import json
import random
import sys
import time

import aiohttp

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402


async def one(session, base, body, out_tokens, results):
    payload = {
        "text": body,
        "sampling_params": {
            "max_new_tokens": out_tokens,
            "temperature": 0.7,
            "ignore_eos": out_tokens > 1,
        },
    }
    t0 = time.perf_counter()
    try:
        async with session.post(f"{base}/generate", json=payload) as r:
            await r.json()
        results.append({"ok": True, "latency_s": time.perf_counter() - t0})
    except Exception as e:  # a slow server must not kill the generator
        results.append({"ok": False, "latency_s": time.perf_counter() - t0,
                        "err": type(e).__name__})


async def run(a):
    words = hcommon.single_token_words()
    rng = random.Random(a.seed)
    # Pre-build bodies: tokenizing inside the send loop would make the client
    # the bottleneck and distort the arrival process.
    pool = []
    n_words = max(1, a.len - 8)
    for _ in range(a.pool):
        pool.append(" ".join(rng.choice(words) for _ in range(n_words)))

    sem = asyncio.Semaphore(a.max_inflight)
    results = []
    stop = time.time() + a.duration
    idx = 0
    tasks = []

    conn = aiohttp.TCPConnector(limit=a.max_inflight * 2)
    timeout = aiohttp.ClientTimeout(total=a.timeout)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        async def guarded(body):
            async with sem:
                await one(session, a.base, body, a.out, results)

        t_start = time.time()
        while time.time() < stop:
            # The unique marker must be a PREFIX. Appending it left all pool
            # entries sharing their whole prefix, so every len(pool)-th request
            # hit the radix cache instead of inserting new nodes — which voided
            # the plan's "every request inserts new nodes" guarantee and made
            # the generated write volume far smaller than intended.
            body = f"{idx} " + pool[idx % len(pool)]
            idx += 1
            tasks.append(asyncio.create_task(guarded(body)))
            if a.rate > 0:
                await asyncio.sleep(rng.expovariate(a.rate))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - t_start

    ok = sum(1 for r in results if r["ok"])
    achieved = ok / elapsed if elapsed > 0 else 0.0
    gbps = achieved * (a.len + a.out) * hcommon.KV_BYTES_PER_TOKEN / (2 ** 30)
    summary = {
        "requested_rps": a.rate, "achieved_rps": achieved, "sent": len(results),
        "ok": ok, "failed": len(results) - ok, "elapsed_s": elapsed,
        "len": a.len, "out": a.out, "implied_write_gbps": gbps,
    }
    print(json.dumps(summary))
    if a.out_file:
        with open(a.out_file, "w") as f:
            f.write(json.dumps({"summary": summary, "requests": results}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, default=4096)
    ap.add_argument("--out", type=int, default=1)
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--max-inflight", type=int, default=32)
    ap.add_argument("--pool", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--base", default=hcommon.BASE)
    ap.add_argument("--out-file", default=None)
    ap.add_argument("--timeout", type=float, default=600)
    a = ap.parse_args()
    if a.rate <= 0:
        print(json.dumps({"requested_rps": 0, "achieved_rps": 0.0, "sent": 0,
                          "ok": 0, "failed": 0, "elapsed_s": a.duration,
                          "len": a.len, "out": a.out, "implied_write_gbps": 0.0}))
        time.sleep(a.duration)
        return
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
