"""Report review (label: numbers): recompute every number the doc "KV Cache Tiering vs Baselines" shows, from the
client records. Read-only; stdlib only. Run from agent_cache/results:
  python3 -I -B campaign30/checks/report_review/numbers/recheck.py
"""
import collections
import json
import math
import os
import sys

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
MANIFEST = os.path.join(RESULTS, "campaign30", "checks", "vs_c34", "manifest.txt")
CUT = 6600.0
DOC_ARMS = ["ec_full", "vllm_lmcache", "hicache", "continuum_lmcache", "tokencake", "continuum", "sglang_vanilla",
            "vllm_vanilla", "ec_exact"]
SLUG = {"ec_full": "ours", "hicache": "hicache", "vllm_lmcache": "vllm-lmcache", "continuum_lmcache": "continuum-lmcache",
        "tokencake": "tokencake", "continuum": "continuum", "sglang_vanilla": "sglang", "vllm_vanilla": "vllm"}


def q_nearest(values, p):
    """compare_arms.py's percentile."""
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


def q_linear(values, p):
    v = sorted(values)
    if not v:
        return float("nan")
    x = p * (len(v) - 1)
    lo = math.floor(x)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (x - lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load(path):
    reqs, errors, plays = [], [], {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] == "req" and r["phase"] == "profile":
                if r.get("error") or r.get("ttft") is None:
                    errors.append(r)
                else:
                    reqs.append(r)
            elif r["kind"] == "play":
                plays[r["play"]] = r
    return reqs, errors, plays


def cache_split(rs):
    prompt = sum(r["prompt_tokens"] or 0 for r in rs)
    cached = 0
    tiers = collections.Counter()
    for r in rs:
        d = r.get("cached_details") or {}
        if d:
            for k in ("device", "host", "storage"):
                tiers[k] += int(d.get(k) or 0)
                cached += int(d.get(k) or 0)
        else:
            tiers["device"] += int(r.get("cached_tokens") or 0)
            cached += int(r.get("cached_tokens") or 0)
    ideal = sum(r.get("ideal_cached") or 0 for r in rs)
    return {"prompt": prompt, "cached": cached, "ideal": ideal, "recomputed": 1 - cached / prompt,
            "delivered": cached / ideal, "new_share": 1 - ideal / prompt,
            "history_recomputed": (ideal - min(cached, ideal)) / ideal}


def main():
    arms = {}
    with open(MANIFEST) as f:
        for line in f:
            label, run, prog = line.split()
            arms[label] = os.path.join(RESULTS, run, prog, "client.jsonl")
    data = {}
    for label, path in arms.items():
        reqs, errors, plays = load(path)
        before = [r for r in reqs if r["t_send"] < CUT]
        data[label] = {"all": reqs, "before": before, "errors": errors, "plays": plays}

    print("== per arm, own requests sent before 6600 s")
    print(f"{'arm':18s} {'n':>6s} {'mean':>8s} {'p50 nr':>8s} {'p50 lin':>8s} {'p90 nr':>8s} {'p90 lin':>8s} {'p99':>8s} "
          f"{'recomp':>7s} {'deliv':>7s} {'1-deliv':>7s} {'new':>6s} | first n mean | returning mean | max ttft")
    for label in arms:
        rs = data[label]["before"]
        t = [r["ttft"] for r in rs]
        c = cache_split(rs)
        first = [r["ttft"] for r in rs if r["stream"] == 0 and r["k"] == 0]
        later = [r["ttft"] for r in rs if not (r["stream"] == 0 and r["k"] == 0)]
        print(f"{label:18s} {len(rs):6d} {mean(t):8.3f} {q_nearest(t, .5):8.3f} {q_linear(t, .5):8.3f} "
              f"{q_nearest(t, .9):8.3f} {q_linear(t, .9):8.3f} {q_nearest(t, .99):8.2f} {c['recomputed']:7.4f} "
              f"{c['delivered']:7.4f} {1 - c['delivered']:7.4f} {c['new_share']:6.4f} | {len(first):4d} {mean(first):8.2f} "
              f"| {mean(later):7.2f} | {max(t):8.1f}")

    print("\n== 5-minute bins by send time (t_send in [5k, 5k+5) min, t_send < 6600 s): n, mean TTFT (2 dp)")
    bins = {}
    for label in arms:
        cells = collections.defaultdict(list)
        for r in data[label]["before"]:
            cells[int(r["t_send"] // 300)].append(r["ttft"])
        bins[label] = {5 * b: (len(v), round(mean(v), 2)) for b, v in sorted(cells.items())}
    for label in arms:
        row = " ".join(f"{m}:{n},{v}" for m, (n, v) in bins[label].items())
        print(f"{label:18s} {row}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bins.json"), "w") as f:
        json.dump({SLUG.get(k, k): v for k, v in bins.items()}, f)

    print("\n== first requests and returning requests by window (minutes), mean TTFT; continuum arms and ours")
    for label in ("ec_full", "continuum", "continuum_lmcache", "vllm_lmcache", "hicache", "tokencake",
                  "sglang_vanilla", "vllm_vanilla"):
        out = []
        for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 110), (10, 110)):
            rs = [r for r in data[label]["before"] if lo * 60 <= r["t_send"] < hi * 60]
            f1 = [r["ttft"] for r in rs if r["stream"] == 0 and r["k"] == 0]
            kk = [r["ttft"] for r in rs if r["k"] > 0]
            out.append(f"{lo}-{hi}: first {len(f1)} {mean(f1):.1f} | k>0 {len(kk)} {mean(kk):.2f}")
        print(f"{label:18s} " + " ; ".join(out))

    print("\n== paired differences B - ec_full on requests both recorded before the cut (no interval)")
    key = lambda r: (r["play"], r["stream"], r["k"])
    a = {key(r): r["ttft"] for r in data["ec_full"]["before"]}
    for label in arms:
        if label == "ec_full":
            continue
        b = {key(r): r["ttft"] for r in data[label]["before"]}
        ks = a.keys() & b.keys()
        diff = sum(b[k] - a[k] for k in ks) / len(ks)
        print(f"{label:18s} n {len(ks):6d} A {mean([a[k] for k in ks]):7.3f} B {mean([b[k] for k in ks]):7.3f} B-A {diff:+8.3f}")

    print("\n== play ids sanity: first-request t_send by play shared across arms?")
    print("ratio requests ec_full / vllm_lmcache:", len(data["ec_full"]["before"]) / len(data["vllm_lmcache"]["before"]))

    print("\n== continuum lower bound variants (mean before cut)")
    for label in ("continuum", "continuum_lmcache"):
        t = [r["ttft"] for r in data[label]["before"]]
        n, s = len(t), sum(t)
        for extra in ((67, 630, 5, 3600), (65, 630, 5, 3600), (72, 630, 0, 0), (21, 630, 0, 0), (19, 630, 0, 0)):
            k1, v1, k2, v2 = extra
            m = (s + k1 * v1 + k2 * v2) / (n + k1 + k2)
            print(f"{label:18s} +{k1} at {v1} s +{k2} at {v2} s: mean {m:.2f} (raise {m - s / n:.2f})")


if __name__ == "__main__":
    sys.exit(main())
