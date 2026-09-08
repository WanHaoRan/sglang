"""Experiment 0 - sanity and tier attribution.

Proves each tier can be forced deterministically, that per-tier attribution is
reported and correct, that the file backend really stores and serves pages, and
that page-cache eviction actually changes the measurement.
"""
import json
import os
import random
import sys
import time
import urllib.request

import requests

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402
from probe import probe  # noqa: E402

RESULTS = os.environ["RESULTS"]
OUT = os.path.join(RESULTS, os.environ.get("EXP0_OUT", "exp0"))
os.makedirs(OUT, exist_ok=True)
BASE = hcommon.BASE
LOG = os.path.join(OUT, "client.jsonl")

PROBE_LEN = 4096
PROBE_SEED = 1000  # exp0, condition 0, rep 0


def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def server_info():
    with urllib.request.urlopen(BASE + "/get_server_info", timeout=30) as r:
        return json.loads(r.read().decode())


def m():
    return hcommon.parse_metrics(hcommon.scrape())


def send_filler(body):
    requests.post(
        f"{BASE}/generate",
        json={"text": body,
              "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
        timeout=600,
    ).json()


def run_fillers(total_tokens, chunk=8192, seed=90000):
    """Unique prompts until the device pool has certainly turned over."""
    words = hcommon.single_token_words()
    rng = random.Random(seed)
    sent = 0
    n = 0
    while sent < total_tokens:
        body = " ".join(rng.choice(words) for _ in range(chunk - 8))
        send_filler(f"{n} " + body)
        sent += chunk
        n += 1
    return {"filler_requests": n, "filler_tokens": sent}


def step(name, **kw):
    print(f"\n=== {name} ===", flush=True)
    return kw


def main():
    info = server_info()
    max_total = info.get("max_total_num_tokens")
    print(json.dumps({"max_total_num_tokens": max_total}), flush=True)

    body = hcommon.build_prompt(PROBE_LEN, PROBE_SEED)
    results = {"max_total_num_tokens": max_total}

    # --- 1. cold probe = recompute sample -------------------------------
    b0 = m()
    l3_0 = hcommon.l3_stats()
    r, usage = probe(BASE, body, PROBE_LEN, PROBE_SEED, {"step": "1_cold_recompute"})
    log(r)
    results["step1_cold"] = r
    results["usage_shape_example"] = usage
    print(json.dumps(r), flush=True)

    # --- 2. write-through landed in L2 and L3 ---------------------------
    hcommon.wait_backup_drain()
    time.sleep(2)
    a0 = m()
    l3_1 = hcommon.l3_stats()
    results["step2_backup"] = {
        "backup_tokens_delta": a0.get("sglang:hicache_backup_tokens_total", 0)
        - b0.get("sglang:hicache_backup_tokens_total", 0),
        "backup_bytes_delta": a0.get("sglang:hicache_backup_bytes_total", 0)
        - b0.get("sglang:hicache_backup_bytes_total", 0),
        "l3_files_before": l3_0["files"], "l3_files_after": l3_1["files"],
        "l3_bytes_after": l3_1["bytes"],
        "host_used_tokens": a0.get("sglang:hicache_host_used_tokens", 0),
    }
    print(json.dumps(results["step2_backup"]), flush=True)

    # --- 3. immediate re-probe = L1 -------------------------------------
    r, _ = probe(BASE, body, PROBE_LEN, PROBE_SEED, {"step": "3_L1"})
    log(r); results["step3_L1"] = r
    print(json.dumps(r), flush=True)

    # --- 4. fillers turn the device pool over -> L2 ----------------------
    fill = run_fillers(int(1.5 * max_total))
    hcommon.wait_backup_drain()
    r, _ = probe(BASE, body, PROBE_LEN, PROBE_SEED, {"step": "4_L2"})
    log(r); results["step4_L2"] = r; results["step4_fillers"] = fill
    print(json.dumps({**r, **fill}), flush=True)

    # --- 5. flush L1+L2, evict page cache -> L3 -------------------------
    hcommon.flush_cache()
    method = hcommon.drop_page_cache()
    cached_kb = hcommon.cached_meminfo_kb()
    r, _ = probe(BASE, body, PROBE_LEN, PROBE_SEED, {"step": "5_L3_cold_pagecache"})
    log(r); results["step5_L3"] = r
    results["step5_evict"] = {"method": method, "cached_kb_after_evict": cached_kb}
    print(json.dumps(r), flush=True)

    # --- 6. same, but leave the page cache warm -------------------------
    hcommon.flush_cache()
    r, _ = probe(BASE, body, PROBE_LEN, PROBE_SEED, {"step": "6_L3_pagecache_warm"})
    log(r); results["step6_L3_pagecache"] = r
    print(json.dumps(r), flush=True)

    with open(os.path.join(OUT, "exp0_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nWROTE " + os.path.join(OUT, "exp0_results.json"), flush=True)


if __name__ == "__main__":
    main()
