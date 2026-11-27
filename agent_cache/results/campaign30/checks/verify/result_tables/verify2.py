"""Second pass: exact bootstrap replication (users resampled, 2,000 draws, one Random(0) stream over all pairs in manifest
order, as compare_arms.py documents), Continuum first/returning by window, rounding edges.
nice -n 10 python3 -I -B verify2.py > verify2.out"""
import collections
import json
import os
import random

RES = "/home/wanhr/sglang/agent_cache/results"
MAN = os.path.join(RES, "campaign30/checks/vs_c34/manifest.txt")
CUT = 6600.0

arms = collections.OrderedDict()
raw = {}
for line in open(MAN):
    if line.strip():
        label, run, prog = line.split()[:3]
        rows = [json.loads(x) for x in open(os.path.join(RES, run, prog, "client.jsonl")) if x.strip()]
        raw[label] = rows
        ok = [r for r in rows if r["kind"] == "req" and r.get("phase") == "profile"
              and not r.get("error") and r.get("ttft") is not None and r["t_send"] < CUT]
        arms[label] = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in ok}

# ----- rounding edges -----
def arrivals(n):
    rng, t, out = random.Random("0:arrivals"), 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(0.1)
    return out


arr = arrivals(2000)
for label in ("ec_exact", "ec_full", "continuum_lmcache", "vllm_lmcache"):
    firsts = [r for r in raw[label] if r["kind"] == "req" and r.get("phase") == "profile" and r["stream"] == 0
              and r["k"] == 0 and r["t_send"] < CUT and r.get("ttft") is not None]
    w = [r["t_send"] - arr[r["play"]] for r in firsts]
    print(f"{label:18s} max slot wait {max(w):.4f} s, mean {sum(w) / len(w):.4f} s")
for label in ("vllm_vanilla", "sglang_vanilla", "continuum_lmcache", "tokencake"):
    rs = [r for r in raw[label] if r["kind"] == "req" and r.get("phase") == "profile" and not r.get("error")
          and r.get("ttft") is not None and r["t_send"] < CUT]
    prompt = sum(r["prompt_tokens"] or 0 for r in rs)
    cached = sum(sum(int(v or 0) for v in (r.get("cached_details") or {}).values()) if r.get("cached_details")
                 else int(r.get("cached_tokens") or 0) for r in rs)
    ideal = sum(r.get("ideal_cached") or 0 for r in rs)
    print(f"{label:18s} delivered {cached / ideal:.5f} recomputed {1 - cached / prompt:.5f} req/min {60 * len(rs) / CUT:.4f}")

# ----- Continuum: first and returning by window, and first requests sent per window per the 'plays' count -----
for label in ("continuum", "continuum_lmcache"):
    rows = raw[label]
    ok = [r for r in rows if r["kind"] == "req" and r.get("phase") == "profile" and r.get("ttft") is not None
          and not r.get("error")]
    lanes = {round(r["t"]): r["plays"] for r in rows if r["kind"] == "lanes"}
    print(f"== {label}")
    for a, b in ((0, 600), (600, 1800), (1800, 3600), (3600, 6600)):
        f = [r["ttft"] for r in ok if a <= r["t_send"] < b and r["stream"] == 0 and r["k"] == 0]
        g = [r["ttft"] for r in ok if a <= r["t_send"] < b and not (r["stream"] == 0 and r["k"] == 0)]
        allw = [r["ttft"] for r in ok if a <= r["t_send"] < b]
        started = lanes.get(b, None), lanes.get(a, 0) if a else 0
        print(f"  {a // 60}-{b // 60} min: first {len(f)} mean {sum(f) / len(f):.2f}; returning {len(g)} mean "
              f"{sum(g) / len(g):.3f}; all {len(allw)} mean {sum(allw) / len(allw):.3f}; sessions started in the window "
              f"(plays count) {started[0] - started[1] if started[0] is not None else None}")

# ----- exact replication of the paired intervals -----
names = list(arms)
rng = random.Random(0)
want = {("ec_full", B) for B in names} | {("ec_exact", B) for B in names} | {(A, "ec_exact") for A in names} \
    | {("hicache", "vllm_lmcache"), ("sglang_vanilla", "vllm_vanilla")}
print("== paired, exact replication")
for i, A in enumerate(names):
    for B in names[i + 1:]:
        a, b = arms[A], arms[B]
        keys = a.keys() & b.keys()
        per = collections.defaultdict(lambda: [0.0, 0])
        for k in keys:
            per[k[0]][0] += b[k] - a[k]
            per[k[0]][1] += 1
        cells = [tuple(per[p]) for p in sorted(per)]
        draws = []
        for _ in range(2000):
            pick = rng.choices(cells, k=len(cells))
            draws.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
        draws.sort()
        if (A, B) in want:
            d = sum(c[0] for c in cells) / len(keys)
            print(f"{A:18s} {B:18s} n {len(keys):6d} users {len(cells):4d} B-A {d:+8.3f} [{draws[50]:+8.2f}, {draws[1949]:+8.2f}]")
