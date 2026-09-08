"""Experiment 2 - write-through interference on L3 hit latency.

Sweeps the concurrent backup (write) rate that write_through generates and
measures L3-hit TTFT against it. Controls separate device contention from
host/PCIe/GIL contention.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402
from probe import probe  # noqa: E402

OUT = os.path.join(os.environ["RESULTS"], os.environ.get("EXP2_SUBDIR", "exp2"))
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(OUT, "interference.csv")
BASE = hcommon.BASE
PROBE_L = int(os.environ.get("PROBE_L", 16384))
MODEL_KEY = os.environ.get("MODEL_KEY", "qwen8b")
BACKEND = os.environ.get("HICACHE_BACKEND", "file")
SCRIPTS = "/sgl-workspace/sglang/hicache_eval/scripts"
# Fresh per invocation: guarantees the recompute control has never been sent to
# this store by any earlier stage, so it cannot silently become an L3 hit.
RC_SALT = random.randrange(10 ** 6, 10 ** 7)


def diskstats(dev="vda"):
    """Sectors read/written for one device, straight from /proc/diskstats.

    iostat_sample() measures the interval *before* the probe, so it captures the
    steady write pressure but never the probe's own read burst. Bracketing the
    probe with diskstats gives the read volume that probe actually caused.
    """
    for line in open("/proc/diskstats"):
        f = line.split()
        if len(f) > 13 and f[2] == dev:
            return {"r_sectors": int(f[5]), "w_sectors": int(f[9])}
    return {"r_sectors": 0, "w_sectors": 0}


def iostat_sample(seconds=5):
    """Mean r/w MB/s, awaits and %util for vda over one interval."""
    try:
        o = subprocess.run(["iostat", "-x", "-d", str(seconds), "2", "vda"],
                           capture_output=True, text=True, timeout=seconds * 3).stdout
        rows = [l.split() for l in o.splitlines() if l.startswith("vda")]
        if len(rows) < 2:
            return {}
        r = rows[-1]
        return {"iostat_r_mbps": float(r[2]) / 1024, "iostat_w_mbps": float(r[8]) / 1024,
                "r_await_ms": float(r[5]), "w_await_ms": float(r[11]),
                "util_pct": float(r[-1])}
    except Exception:
        return {}


class WriteLoad:
    """writeload.py in a subprocess for the life of one sweep point."""

    def __init__(self, rate, duration, length=4096):
        self.rate, self.duration, self.length = rate, duration, length
        self.proc = None
        self.summary = {}

    def __enter__(self):
        if self.rate <= 0:
            return self
        self.proc = subprocess.Popen(
            ["python3", os.path.join(SCRIPTS, "writeload.py"),
             "--len", str(self.length), "--out", "1", "--rate", str(self.rate),
             "--duration", str(self.duration), "--max-inflight", "32"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def __exit__(self, *exc):
        if self.proc:
            try:
                # SIGTERM breaks writeload's send loop so it prints its summary;
                # without it communicate() times out and achieved_rps reads 0.
                self.proc.terminate()
                out, _ = self.proc.communicate(timeout=120)
                for line in (out or "").splitlines():
                    if line.strip().startswith("{"):
                        self.summary = json.loads(line)
            except Exception:
                self.proc.kill()
        return False


def prepopulate(bodies, seeds):
    """Send each probe prompt once so it exists in L3, then wait for the drain."""
    for (L, s), b in zip(seeds, bodies):
        probe(BASE, b, L, s, {})
    print("  populated; waiting for backup drain...", flush=True)
    t0 = time.time()
    hcommon.wait_until_flushable(verbose=False)
    print(f"  drain {time.time()-t0:.0f}s; L3={hcommon.l3_stats()}", flush=True)


def run_point(rate, bodies, seeds, reps, control, duration, offset, rc_bodies=None):
    rows = []
    with WriteLoad(rate, duration) as wl:
        time.sleep(30 if rate > 0 else 2)  # warm-up excluded
        b0 = hcommon.parse_metrics(hcommon.scrape())
        t_blk = time.time()
        for rep in range(reps):
            k = offset + rep          # a fresh, never-touched prompt each time
            L, s = seeds[k]
            body = bodies[k]
            io = iostat_sample(5)
            try:
                # No flush here: with writeload running the scheduler is never
                # fully idle, so flush_cache can never succeed. Instead each
                # prompt is probed exactly once, so its only copy is in L3.
                hcommon.drop_page_cache()
                ds0 = diskstats()
                rec, _ = probe(BASE, body, L, s, {})
                ds1 = diskstats()
            except Exception as e:
                print(f"  probe failed at R={rate}: {e}", flush=True)
                continue
            row = {"model": MODEL_KEY, "backend": BACKEND,
                   "requested_rps": rate, "control": control, "rep": rep,
                   "ttft_s": rec["ttft_s"], "cached_storage": rec.get("cached_storage"),
                   "cached_device": rec.get("cached_device"),
                   "cached_host": rec.get("cached_host"),
                   "probe_read_mb": (ds1["r_sectors"] - ds0["r_sectors"]) * 512 / 2**20,
                   "probe_write_mb": (ds1["w_sectors"] - ds0["w_sectors"]) * 512 / 2**20,
                   **io}
            rows.append(row)
            print(json.dumps(row), flush=True)

            # Paired control: a prompt never sent before, so it cannot hit any
            # tier and must recompute. Same load, same instant -- this is the
            # line the L3 hit has to beat.
            if rc_bodies is not None:
                try:
                    ds0 = diskstats()
                    rc, _ = probe(BASE, rc_bodies[k], L, RC_SALT + k, {})
                    ds1 = diskstats()
                    rrow = dict(row)
                    rrow.update({
                        "control": control + "_recompute", "ttft_s": rc["ttft_s"],
                        "rc_seed": RC_SALT + k,
                        "cached_storage": rc.get("cached_storage"),
                        "cached_device": rc.get("cached_device"),
                        "cached_host": rc.get("cached_host"),
                        "probe_read_mb": (ds1["r_sectors"] - ds0["r_sectors"]) * 512 / 2**20,
                        "probe_write_mb": (ds1["w_sectors"] - ds0["w_sectors"]) * 512 / 2**20,
                    })
                    rows.append(rrow)
                    print(json.dumps(rrow), flush=True)
                except Exception as e:
                    print(f"  recompute probe failed at R={rate}: {e}", flush=True)
            if rate > 0:
                time.sleep(10)
        a0 = hcommon.parse_metrics(hcommon.scrape())
        dur = max(1e-9, time.time() - t_blk)
    dbytes = (a0.get("sglang:hicache_backup_bytes_total", 0)
              - b0.get("sglang:hicache_backup_bytes_total", 0))
    for r in rows:
        r["achieved_rps"] = wl.summary.get("achieved_rps", 0.0)
        r["write_gbps_actual"] = dbytes / dur / 2 ** 30
        r["unfulfilled_tokens"] = (
            a0.get("sglang:storage_prefetch_unfulfilled_tokens_total", 0)
            - b0.get("sglang:storage_prefetch_unfulfilled_tokens_total", 0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0,2,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--control", default="baseline")
    ap.add_argument("--duration", type=float, default=420)
    ap.add_argument("--skip-populate", action="store_true")
    a = ap.parse_args()

    rates = [float(x) for x in a.rates.split(",")]
    seeds = [(PROBE_L, 2000 + i) for i in range(a.reps * len(rates))]
    bodies = [hcommon.build_prompt(L, s) for L, s in seeds]
    # Never populated into L3, so every one of these is a guaranteed miss.
    rc_bodies = [hcommon.build_prompt(L, RC_SALT + i) for i, (L, s) in enumerate(seeds)]

    if not os.path.exists(CSV):
        with open(CSV, "w") as f:
            f.write("model,backend,requested_rps,achieved_rps,write_gbps_actual,control,rep,ttft_s,"
                    "cached_storage,cached_device,cached_host,unfulfilled_tokens,"
                    "probe_read_mb,probe_write_mb,"
                    "iostat_r_mbps,iostat_w_mbps,r_await_ms,w_await_ms,util_pct\n")

    if not a.skip_populate:
        print("### pre-populating probe prompts into L3", flush=True)
        prepopulate(bodies, seeds)
        hcommon.flush_cache()      # the only flush; L1+L2 now empty, L3 full
        print("  flushed L1+L2; probes now live only in L3", flush=True)

    for i, rate in enumerate(rates):
        print(f"\n### R={rate} req/s  control={a.control}", flush=True)
        for row in run_point(rate, bodies, seeds, a.reps, a.control, a.duration,
                             offset=i * a.reps, rc_bodies=rc_bodies):
            with open(CSV, "a") as f:
                f.write(",".join(str(row.get(k, "")) for k in [
                    "model", "backend", "requested_rps", "achieved_rps", "write_gbps_actual", "control",
                    "rep", "ttft_s", "cached_storage", "cached_device", "cached_host",
                    "unfulfilled_tokens", "probe_read_mb", "probe_write_mb",
                    "iostat_r_mbps", "iostat_w_mbps",
                    "r_await_ms", "w_await_ms", "util_pct"]) + "\n")
    print("\nDONE exp2", flush=True)


if __name__ == "__main__":
    main()
