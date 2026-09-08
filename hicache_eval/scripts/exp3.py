"""Experiments 3 and 4 - write amplification on a multi-turn reasoning
workload, and the strip-thinking-cache oracle.

One condition = one fresh server on a cold L3. Everything is measured from
metric deltas across the run plus the on-disk L3 size, because the client does
not log per-tier attribution.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402

WORK = "/sgl-workspace/sglang/hicache_eval"
SCRIPTS = f"{WORK}/scripts"
REPO = "/sgl-workspace/sglang"
OUT = os.path.join(os.environ["RESULTS"], "exp3")
os.makedirs(OUT, exist_ok=True)
CSV = os.path.join(OUT, "amplification.csv")
KV = 147456

FIELDS = ["condition", "hicache_size", "write_policy", "extra", "duration_s",
          "written_L3_tokens", "written_L3_bytes", "l3_files", "l3_bytes_on_disk",
          "read_L3_tokens", "prefetched_tokens", "unfulfilled_tokens",
          "dropped_tokens", "backup_dropped_tokens", "generation_tokens",
          "prompt_tokens", "cache_hit_rate", "host_used_tokens",
          "write_amplification_L3", "dead_fraction_L3", "drain_s"]


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True, **kw)


def start(tag, extra_args):
    sh(f"bash {SCRIPTS}/stop_server.sh")
    sh(f"find {hcommon.L3_DIR} -mindepth 1 -delete")
    r = sh(f"source {SCRIPTS}/env.sh && bash {SCRIPTS}/start_server.sh {tag} "
           f"--enable-metrics {extra_args}", executable="/bin/bash")
    if "READY" not in (r.stdout or ""):
        raise RuntimeError(f"server failed to start for {tag}: {r.stdout} {r.stderr}")
    return r.stdout.strip()


def run_condition(name, hicache_size, write_policy, extra, client_args, no_hicache=False):
    tag = f"exp3_{name}"
    cond_out = os.path.join(OUT, name)
    os.makedirs(cond_out, exist_ok=True)

    if no_hicache:
        server_extra = "--reasoning-parser qwen3"
    else:
        server_extra = (
            f"--enable-hierarchical-cache --hicache-size {hicache_size} "
            f"--hicache-write-policy {write_policy} --hicache-io-backend kernel "
            f"--hicache-mem-layout page_first --hicache-storage-backend file "
            f"--hicache-storage-prefetch-policy timeout --radix-eviction-policy lru "
            f"{extra}")
    server_extra += " --default-chat-template-kwargs '{\"enable_thinking\": true}'"

    print(f"\n### condition {name}: {server_extra}", flush=True)
    start(tag, server_extra)

    before = hcommon.parse_metrics(hcommon.scrape(
        os.path.join(cond_out, "metrics_before.txt")))
    t0 = time.time()
    logf = os.path.join(cond_out, "bench.jsonl")
    cmd = (f"cd {REPO}/benchmark/hicache && python3 bench_multiturn.py "
           f"--model-path Qwen/Qwen3-8B --port 30000 --api-format openai "
           f"--disable-random-sample --disable-auto-run --enable-round-barrier "
           f"--ready-queue-policy random --tag {name} --log-file {logf} {client_args}")
    r = sh(cmd)
    dur = time.time() - t0
    open(os.path.join(cond_out, "client_stdout.txt"), "w").write(
        (r.stdout or "") + "\n--- stderr ---\n" + (r.stderr or ""))
    print(f"  client finished in {dur:.0f}s (rc={r.returncode})", flush=True)

    drain_t0 = time.time()
    drain = 0.0
    if not no_hicache:
        try:
            drain = hcommon.wait_until_flushable(max_wait_s=5400, verbose=False)
        except Exception as e:
            print(f"  drain wait aborted: {e}", flush=True)
            drain = time.time() - drain_t0
    after = hcommon.parse_metrics(hcommon.scrape(
        os.path.join(cond_out, "metrics_after.txt")))

    def d(k):
        return after.get(k, 0.0) - before.get(k, 0.0)

    l3 = hcommon.l3_stats()
    wl3_tokens = d("sglang:hicache_backup_tokens_total")
    rl3_tokens = d("sglang:storage_prefetch_hit_tokens_total")
    row = {
        "condition": name, "hicache_size": hicache_size,
        "write_policy": write_policy if not no_hicache else "none",
        "extra": extra.strip() or "-", "duration_s": round(dur, 1),
        "written_L3_tokens": wl3_tokens,
        "written_L3_bytes": d("sglang:hicache_backup_bytes_total"),
        "l3_files": l3["files"], "l3_bytes_on_disk": l3["bytes"],
        "read_L3_tokens": rl3_tokens,
        "prefetched_tokens": d("sglang:prefetched_tokens_total"),
        "unfulfilled_tokens": d("sglang:storage_prefetch_unfulfilled_tokens_total"),
        "dropped_tokens": d("sglang:hicache_dropped_tokens_total"),
        "backup_dropped_tokens": d("sglang:hicache_backup_dropped_tokens_total"),
        "generation_tokens": d("sglang:generation_tokens_total"),
        "prompt_tokens": d("sglang:prompt_tokens_total"),
        "cache_hit_rate": after.get("sglang:cache_hit_rate", 0.0),
        "host_used_tokens": after.get("sglang:hicache_host_used_tokens", 0.0),
        "write_amplification_L3": (wl3_tokens / rl3_tokens) if rl3_tokens else float("inf"),
        "dead_fraction_L3": (1 - rl3_tokens / wl3_tokens) if wl3_tokens else float("nan"),
        "drain_s": round(drain, 1),
    }
    with open(CSV, "a") as f:
        f.write(",".join(str(row.get(k, "")) for k in FIELDS) + "\n")
    json.dump(row, open(os.path.join(cond_out, "summary.json"), "w"), indent=2)
    print(json.dumps(row, indent=2), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="wt100,wt30,wts100,wts30,nohicache")
    ap.add_argument("--client-args", default=(
        "--request-length 1024 --output-length 1024 --num-clients 16 "
        "--num-rounds 4 --max-parallel 8 --request-rate 2"))
    a = ap.parse_args()

    if not os.path.exists(CSV):
        with open(CSV, "w") as f:
            f.write(",".join(FIELDS) + "\n")

    SPECS = {
        "wt100":  dict(hicache_size=100, write_policy="write_through", extra=""),
        "wt30":   dict(hicache_size=30, write_policy="write_through", extra=""),
        "wts100": dict(hicache_size=100, write_policy="write_through_selective", extra=""),
        "wts30":  dict(hicache_size=30, write_policy="write_through_selective", extra=""),
        "nohicache": dict(hicache_size=0, write_policy="none", extra="", no_hicache=True),
        # Experiment 4 - strip-thinking-cache oracle
        "oracle100": dict(hicache_size=100, write_policy="write_through",
                          extra="--strip-thinking-cache"),
        "oracle30": dict(hicache_size=30, write_policy="write_through",
                         extra="--strip-thinking-cache"),
    }
    for name in a.conditions.split(","):
        spec = dict(SPECS[name])
        no_h = spec.pop("no_hicache", False)
        try:
            run_condition(name, client_args=a.client_args, no_hicache=no_h, **spec)
        except Exception as e:
            print(f"CONDITION {name} FAILED: {e}", flush=True)
    print("\nDONE exp3/4", flush=True)


if __name__ == "__main__":
    main()
