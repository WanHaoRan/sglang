"""Parse each unstarted queue line's server flags with its own frozen engine and run the cache validation hooks.
In the container: python3 results/campaign26/checks/validate_queue.py   (from agent_cache/; re-executes itself per engine)."""
import json, os, subprocess, sys

AC = "/sgl-workspace/sglang/agent_cache"
BASE = ["--model-path", "dummy", "--enable-hierarchical-cache", "--hicache-storage-backend", "nixl", "--hicache-storage-prefetch-policy", "wait_complete",
        "--hicache-write-policy", "write_through", "--hicache-storage-next-turn-prefetch", "on", "--hicache-storage-next-turn-reply-key-source", "hint",
        "--hicache-storage-next-turn-read-timing", "just_in_time", "--hicache-storage-next-turn-arrival-gate", "on", "--hicache-storage-next-turn-arrival-hold", "on",
        "--hicache-storage-read-order", "deadline", "--hicache-storage-prefetch-pin-fraction", "0.3", "--hicache-storage-prefetch-retry-poll-interval", "0",
        "--hicache-storage-prefetch-retry-max-attempts", "8", "--radix-eviction-policy", "next_use", "--radix-eviction-policy-config", '{"horizon_s":300}']

if len(sys.argv) > 1:
    from sglang.srt.arg_groups.hicache_hook import handle_hicache, handle_hicache_ratio_default
    from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
    from sglang.srt.server_args import prepare_server_args
    import sglang
    args = prepare_server_args(BASE + json.loads(sys.argv[1]))
    handle_hicache_ratio_default(args)
    handle_hicache(args)
    handle_cache_compatibility(args)
    print("ok", os.path.dirname(os.path.dirname(sglang.__file__)).split("/frozen/")[-1], args.hicache_storage_next_turn_read_timing, args.hicache_storage_next_turn_memory_price,
          args.hicache_storage_next_turn_price_learning, args.hicache_storage_next_turn_soft_pins, args.hicache_storage_next_turn_landing_quantile,
          args.radix_eviction_policy, args.radix_eviction_policy_config)
    sys.exit(0)

done = {line.split()[0] for line in open(f"{AC}/results/campaign26/manifest.txt") if line.strip()}
for line in open(f"{AC}/results/campaign26/queue.txt"):
    f = line.split()
    if not f or f[0].startswith("#") or f[0] in done:
        continue
    env = dict(os.environ, PYTHONPATH=f"{AC}/frozen/{f[8]}/python", CUDA_VISIBLE_DEVICES="7")
    out = subprocess.run([sys.executable, __file__, json.dumps(f[10:])], env=env, capture_output=True, text=True)
    last = (out.stdout.strip().splitlines() or ["?"])[-1] if out.returncode == 0 else "FAILED: " + (out.stderr.strip().splitlines() or ["?"])[-1]
    print(f"{f[0]:14s} users {f[2]:>3s} host {f[3]:>3s} hints {f[5]:5s} {last}")
