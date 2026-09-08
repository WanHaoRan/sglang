"""A/B one storage backend against the Exp 0 L3 probe.

Writes a 4096-token prompt, waits for the backup to land, flushes L1+L2, drops
the page cache, and probes. The result is directly comparable to Exp 0 step 5
(file backend: 11.58 s).
"""
import json
import os
import sys
import time

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402
from probe import probe  # noqa: E402

BASE = hcommon.BASE
L = int(os.environ.get("AB_LEN", "4096"))
SEED = int(os.environ.get("AB_SEED", "1000"))
TAG = os.environ.get("AB_TAG", "unknown")
STORE = os.environ.get("AB_STORE_DIR", "/var/hicache_l3")


def store_stats():
    import glob
    fs = glob.glob(os.path.join(STORE, "**", "*"), recursive=True)
    fs = [f for f in fs if os.path.isfile(f)]
    return {"files": len(fs), "bytes": sum(os.path.getsize(f) for f in fs)}


def main():
    body = hcommon.build_prompt(L, SEED)
    out = {"backend": TAG, "L": L, "store_dir": STORE}

    m0 = hcommon.parse_metrics(hcommon.scrape())
    t0 = time.time()
    r1, _ = probe(BASE, body, L, SEED, {"step": "cold_recompute"})
    out["cold_recompute_ttft_s"] = r1["ttft_s"]

    drain_t0 = time.time()
    hcommon.wait_until_flushable(max_wait_s=900)
    out["backup_drain_s"] = round(time.time() - drain_t0, 2)
    m1 = hcommon.parse_metrics(hcommon.scrape())
    out["backup_tokens"] = m1.get("sglang:hicache_backup_tokens_total", 0) - \
        m0.get("sglang:hicache_backup_tokens_total", 0)
    out["backup_bytes"] = m1.get("sglang:hicache_backup_bytes_total", 0) - \
        m0.get("sglang:hicache_backup_bytes_total", 0)
    out["store_after_write"] = store_stats()

    r2, _ = probe(BASE, body, L, SEED, {"step": "L1"})
    out["L1_ttft_s"] = r2["ttft_s"]
    out["L1_attr"] = [r2.get("cached_device"), r2.get("cached_host"), r2.get("cached_storage")]

    hcommon.wait_until_flushable(max_wait_s=900)
    hcommon.flush_cache()
    hcommon.drop_page_cache()
    r3, _ = probe(BASE, body, L, SEED, {"step": "L3"})
    out["L3_ttft_s"] = r3["ttft_s"]
    out["L3_attr"] = [r3.get("cached_device"), r3.get("cached_host"), r3.get("cached_storage")]
    cached = r3.get("cached_storage") or 0
    if cached and r3["ttft_s"] > 0:
        out["L3_read_GBps"] = round(
            cached * hcommon.KV_BYTES_PER_TOKEN / r3["ttft_s"] / 2 ** 30, 4)
    if out.get("backup_bytes") and out.get("backup_drain_s"):
        out["L3_write_GBps"] = round(
            out["backup_bytes"] / out["backup_drain_s"] / 2 ** 30, 4)
    out["total_s"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=2))
    d = os.path.join(os.environ["RESULTS"], "backend_ab")
    os.makedirs(d, exist_ok=True)
    json.dump(out, open(os.path.join(d, f"{TAG}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
