"""Paired TTFT of two runs on the requests both sent in a window of the client's clock, split by what the
request loaded (cached tokens reported by the server).

    python3 pair_early.py <client.jsonl of A> <client.jsonl of B> [--from S] [--to S]

Requests pair by (play, stream, k). Prints B - A: mean with a 95 % bootstrap interval over plays, the median of the
paired differences, and each run's own p50.
"""
import argparse
import json
import random
import statistics as st


def load(path, lo, hi):
    out = {}
    with open(path) as f:
        for line in f:
            if '"kind":"req"' not in line:
                continue
            r = json.loads(line)
            if r.get("phase") != "profile" or r.get("ttft") is None or not (lo <= r["t_send"] < hi):
                continue
            out[(r["play"], r["stream"], r["k"])] = r
    return out


def report(name, keys, a, b, rng):
    if len(keys) < 20:
        print(f"  {name}: {len(keys)} pairs (too few)")
        return
    d = [b[k]["ttft"] - a[k]["ttft"] for k in keys]
    by_play = {}
    for k, x in zip(keys, d):
        by_play.setdefault(k[0], []).append(x)
    plays = list(by_play)
    boots = []
    for _ in range(2000):
        s = n = 0.0
        for p in rng.choices(plays, k=len(plays)):
            s += sum(by_play[p])
            n += len(by_play[p])
        boots.append(s / n)
    boots.sort()
    ta = sorted(a[k]["ttft"] for k in keys)
    tb = sorted(b[k]["ttft"] for k in keys)
    print(f"  {name}: {len(keys)} pairs; B - A mean {1e3 * st.mean(d):+.1f} ms [{1e3 * boots[50]:+.1f}, {1e3 * boots[1949]:+.1f}], "
          f"median of differences {1e3 * st.median(d):+.1f} ms; p50 A {1e3 * ta[len(ta) // 2]:.0f} ms, B {1e3 * tb[len(tb) // 2]:.0f} ms; "
          f"mean A {st.mean(ta):.3f} s, B {st.mean(tb):.3f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--from", dest="lo", type=float, default=0.0)
    ap.add_argument("--to", dest="hi", type=float, default=1800.0)
    x = ap.parse_args()
    a, b = load(x.a, x.lo, x.hi), load(x.b, x.lo, x.hi)
    keys = sorted(set(a) & set(b))
    rng = random.Random(0)
    print(f"window {x.lo:.0f} to {x.hi:.0f} s: A {len(a)} requests, B {len(b)}, paired {len(keys)}")
    report("all", keys, a, b, rng)
    for name, lo, hi in (("cached under 8,192 tokens in both", 0, 8192), ("cached 8,192 tokens or more in both", 8192, 10 ** 9)):
        sub = [k for k in keys if lo <= (a[k].get("cached_tokens") or 0) < hi and lo <= (b[k].get("cached_tokens") or 0) < hi]
        report(name, sub, a, b, rng)
    fast = [k for k in keys if a[k]["ttft"] < 0.5 and b[k]["ttft"] < 0.5]
    report("first token within 0.5 s in both", fast, a, b, rng)
    fast8 = [k for k in fast if (a[k].get("cached_tokens") or 0) >= 8192 and (b[k].get("cached_tokens") or 0) >= 8192]
    report("the same, cached 8,192 tokens or more", fast8, a, b, rng)


if __name__ == "__main__":
    main()
