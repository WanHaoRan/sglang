"""Exp 1 background-load variant, L3 (and L2) tiers — plan L207.

The naive approach hangs: the L3 recipe flushes before every probe, but
flush_cache requires the scheduler to be fully idle and a background writeload
never lets it be. So this follows the Exp 2 pattern instead — populate every
probe, drain, flush ONCE while still idle, then start the load and touch each
prompt exactly once, so its only copy is in L3.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402
from probe import probe  # noqa: E402

BASE = hcommon.BASE
R = os.environ["RESULTS"]
CSV = os.path.join(R, "exp1", "ttft_by_tier.csv")
LENGTHS = [2048, 8192, 32512]
REPS = 3
BG = ("--len", "1024", "--out", "1", "--rate", "2")


def emit(rec, tier, L, rep, tag):
    with open(CSV, "a") as f:
        f.write(",".join(str(x) for x in [
            tier, L, rep, f"{rec['ttft_s']:.6f}", rec.get("cached_device"),
            rec.get("cached_host"), rec.get("cached_storage"),
            "wait_complete", tag, 0, rec.get("prompt_tokens")]) + "\n")
    print(json.dumps({"tier": tier, "L": L, "rep": rep, "ttft_s": rec["ttft_s"],
                      "dev": rec.get("cached_device"), "host": rec.get("cached_host"),
                      "stor": rec.get("cached_storage")}), flush=True)


def main():
    seeds = [(L, 7000 + 10 * i + r) for i, L in enumerate(LENGTHS)
             for r in range(REPS)]
    bodies = {s: hcommon.build_prompt(L, s) for L, s in seeds}

    print("### populate all probes into L3", flush=True)
    for L, s in seeds:
        probe(BASE, bodies[(L, s)], L, s, {})
    t0 = time.time()
    hcommon.wait_until_flushable(max_wait_s=2400, verbose=False)
    print(f"  drained in {time.time()-t0:.0f}s", flush=True)
    hcommon.flush_cache()
    print("  flushed L1+L2; probes live only in L3", flush=True)

    print("### start background writeload (rate 2, len 1024)", flush=True)
    bg = subprocess.Popen(
        ["python3", "/sgl-workspace/sglang/hicache_eval/scripts/writeload.py",
         *BG, "--duration", "1800", "--max-inflight", "32"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(30)  # plan: 30 s warm-up excluded
    try:
        for L, s in seeds:
            hcommon.drop_page_cache()
            rec, _ = probe(BASE, bodies[(L, s)], L, s, {})
            emit(rec, "L3", L, s % 10, "rate2")
        # L2 under load: the probes are now device-resident from the L3 reads;
        # the background load itself is the filler that pushes them to host.
        print("### L2 under load (background load acts as the filler)", flush=True)
        time.sleep(90)
        for L, s in seeds:
            rec, _ = probe(BASE, bodies[(L, s)], L, s, {})
            tier = ("L2" if (rec.get("cached_host") or 0) > 0.9 * (L - 64)
                    else "L2_MISS")
            emit(rec, tier, L, s % 10, "rate2")
    finally:
        bg.kill()
    print("DONE exp1 bg L3/L2", flush=True)


if __name__ == "__main__":
    main()
