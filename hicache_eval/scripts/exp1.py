"""Experiment 1 - TTFT per tier versus context length.

Phase order is chosen to minimise backup-drain waits: recompute and L1 first
(they need no flush), then L3 (one drain, then a flush per probe), then L2
(the filler pass writes ~80 GB to L3, so nothing that needs a flush may follow
it).
"""
import argparse
import json
import os
import random
import sys
import time

import requests

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402
from probe import probe  # noqa: E402

LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32512]
OUT = os.path.join(os.environ["RESULTS"], os.environ.get("EXP1_OUT", "exp1"))
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(OUT, "ttft_by_tier.csv")
LOG = os.path.join(OUT, "client.jsonl")
BASE = hcommon.BASE


def seed_for(L, rep):
    # exp 1; condition keyed by length so no two conditions share a prompt
    return 1000 * 1 + 10 * L + rep


def emit(rec, tier, L, rep, policy, bg, discards=0):
    rec.update({"tier": tier, "L": L, "rep": rep, "prefetch_policy": policy,
                "background_load": bg, "discards": discards})
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    with open(CSV, "a") as f:
        f.write(",".join(str(x) for x in [
            tier, L, rep, f"{rec['ttft_s']:.6f}", rec.get("cached_device"),
            rec.get("cached_host"), rec.get("cached_storage"), policy, bg,
            discards, rec.get("prompt_tokens")]) + "\n")
    print(json.dumps({k: rec[k] for k in
                      ("tier", "L", "rep", "ttft_s", "cached_device",
                       "cached_host", "cached_storage")}), flush=True)
    return rec


def send_fillers(total_tokens, chunk=8192, seed=90000):
    words = hcommon.single_token_words()
    rng = random.Random(seed)
    sent, n = 0, 0
    while sent < total_tokens:
        body = f"{n} " + " ".join(rng.choice(words) for _ in range(chunk - 8))
        requests.post(f"{BASE}/generate",
                      json={"text": body,
                            "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
                      timeout=900).json()
        sent += chunk
        n += 1
    return n, sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--policy", default="wait_complete")
    ap.add_argument("--bg", default="none")
    ap.add_argument("--bg-rate", type=float, default=0.0,
                    help="if >0, run writeload.py at this rate for the whole run")
    ap.add_argument("--bg-len", type=int, default=1024)
    ap.add_argument("--bg-duration", type=float, default=3600)
    ap.add_argument("--tiers", default="recompute,L1,L3,L2")
    ap.add_argument("--lengths", default=",".join(str(x) for x in LENGTHS))
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")]
    tiers = a.tiers.split(",")

    if not os.path.exists(CSV):
        with open(CSV, "w") as f:
            f.write("tier,L,rep,ttft_s,cached_device,cached_host,cached_storage,"
                    "prefetch_policy,background_load,discards,prompt_tokens\n")

    bodies = {(L, r): hcommon.build_prompt(L, seed_for(L, r))
              for L in lengths for r in range(a.reps)}

    bg_proc = None
    if a.bg_rate > 0:
        import subprocess
        bg_proc = subprocess.Popen(
            ["python3", "/sgl-workspace/sglang/hicache_eval/scripts/writeload.py",
             "--len", str(a.bg_len), "--out", "1", "--rate", str(a.bg_rate),
             "--duration", str(a.bg_duration), "--max-inflight", "32"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(json.dumps({"background_load": f"writeload --len {a.bg_len} "
                          f"--rate {a.bg_rate}"}), flush=True)
        time.sleep(30)  # plan: 30 s warm-up excluded

    # --- recompute + L1: one send is the recompute sample, the immediate
    # --- re-probe is the L1 sample. No flush needed anywhere here.
    if "recompute" in tiers or "L1" in tiers or "L2" in tiers:
        # L2 needs the probes resident before the filler pass, so phase A/B
        # always runs; it only *emits* rows for the tiers actually requested.
        print("\n### phase A/B: populate (emitting only requested tiers)", flush=True)
        for L in lengths:
            for r in range(a.reps):
                b = bodies[(L, r)]
                rec, _ = probe(BASE, b, L, seed_for(L, r), {})
                if "recompute" in tiers:
                    emit(rec, "recompute", L, r, a.policy, a.bg)
                rec, _ = probe(BASE, b, L, seed_for(L, r), {})
                if "L1" in tiers:
                    emit(rec, "L1", L, r, a.policy, a.bg)

    # --- L3: everything above is now backed up; wait once, then flush per probe
    if "L3" in tiers:
        print("\n### phase D: L3 (waiting for backup drain)", flush=True)
        t0 = time.time()
        hcommon.wait_until_flushable(verbose=True)
        print(f"drain took {time.time()-t0:.0f}s", flush=True)
        for L in lengths:
            for r in range(a.reps):
                b = bodies[(L, r)]
                discards = 0
                for attempt in range(3):
                    hcommon.wait_until_flushable()
                    hcommon.drop_page_cache()
                    rec, _ = probe(BASE, b, L, seed_for(L, r), {})
                    if (rec.get("cached_storage") or 0) > 0.9 * (L - 64):
                        emit(rec, "L3", L, r, a.policy, a.bg, discards)
                        break
                    discards += 1
                    print(f"  discard (bad attribution) L={L} rep={r}: "
                          f"dev={rec.get('cached_device')} host={rec.get('cached_host')} "
                          f"stor={rec.get('cached_storage')}", flush=True)

    # --- L2 last: the filler pass writes ~80 GB to L3, so no flush may follow
    if "L2" in tiers:
        print("\n### phase C: L2 (filler pass)", flush=True)
        info = requests.get(f"{BASE}/get_server_info", timeout=30).json()
        maxtot = info["max_total_num_tokens"]
        # The filler budget is squeezed from both sides: it must exceed the
        # device pool to evict the probes from L1, but probes + fillers must
        # stay under the host pool or the probes fall straight through to L3.
        probe_tokens = sum(lengths) * a.reps
        host_total = hcommon.parse_metrics(hcommon.scrape()).get(
            "sglang:hicache_host_total_tokens", 678208)
        lo, hi = int(1.05 * maxtot), int(0.92 * host_total) - probe_tokens
        budget = min(hi, int(1.15 * maxtot)) if hi > lo else lo
        print(json.dumps({"filler_budget": budget, "lo": lo, "hi": hi,
                          "probe_tokens": probe_tokens,
                          "host_total": host_total}), flush=True)
        nf, tf = send_fillers(budget)
        print(json.dumps({"filler_requests": nf, "filler_tokens": tf}), flush=True)
        for L in lengths:
            for r in range(a.reps):
                b = bodies[(L, r)]
                discards = 0
                for attempt in range(3):
                    rec, _ = probe(BASE, b, L, seed_for(L, r), {})
                    if (rec.get("cached_host") or 0) > 0.9 * (L - 64):
                        emit(rec, "L2", L, r, a.policy, a.bg, discards)
                        break
                    discards += 1
                    print(f"  discard L2 L={L} rep={r}: dev={rec.get('cached_device')} "
                          f"host={rec.get('cached_host')}", flush=True)
                    send_fillers(int(0.35 * maxtot), seed=90000 + 977 * (L + r))
    if bg_proc is not None:
        bg_proc.kill()
    print("\nDONE exp1", flush=True)


if __name__ == "__main__":
    main()
