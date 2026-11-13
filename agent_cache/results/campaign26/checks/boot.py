"""Campaign 26: what an arm booted with and whether it runs clean. From agent_cache/, on the host:
  nice -n 10 python3 -I -B results/campaign26/checks/boot.py [run folder]
Default run: results/.current_run. Prints the resolved flags the arms differ in, the pools, the banners, the counts of
tracebacks and prefetch events in server.log, and per client folder its hint mode, lines and errors."""
import ast, json, os, re, sys, time

RES = "results"
run = sys.argv[1] if len(sys.argv) > 1 else open(os.path.join(RES, ".current_run")).read().strip()
path = os.path.join(RES, os.path.basename(run.rstrip("/")))
KEYS = ("hicache_storage_next_turn_read_timing", "hicache_storage_next_turn_memory_price", "hicache_storage_next_turn_price_learning",
        "hicache_storage_next_turn_soft_pins", "hicache_storage_next_turn_landing_quantile", "hicache_storage_next_turn_arrival_gate",
        "radix_eviction_policy", "radix_eviction_policy_config")
COUNTS = ("Traceback", "ReqTimeStats", "HICACHE_EVT ntp_arrive", "HICACHE_EVT ntp_plan", "HICACHE_EVT s2h_io")
LINES = ("KV Cache is allocated", "host pool:", "HiCache storage reads:", "HiCache next-turn pricing:", "Engine startup timings")

print(time.strftime("%H:%M:%SZ", time.gmtime()), os.path.basename(path))
for line in open(os.path.join(path, "run.txt")):
    if line.split("=")[0] in ("host_gb", "engine", "started_utc", "server_extra"):
        print("  " + line.strip())
log = open(os.path.join(path, "server.log"), errors="replace").read()
found = re.search(r"server_args=(\{.*\})", log)
if found:
    args = ast.literal_eval(found.group(1))
    for k in KEYS:
        print(f"  {k} = {args.get(k, 'not a flag of this build')!r}")
else:
    print("  server_args not printed yet")
print("  " + ", ".join(f"{k} {log.count(k)}" for k in COUNTS))
for line in log.splitlines():
    if any(k in line for k in LINES):
        print("  " + line[:230])
for d in sorted(d for d in os.listdir(path) if d.startswith("client_")):
    out = os.path.join(path, d, "client.jsonl")
    rows = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
    head = rows[0]["args"] if rows else {}
    print(f"  {d}: hints {head.get('hints')} oracle_hints {head.get('oracle_hints')} point {head.get('hint_point')} concurrency {head.get('concurrency')}; "
          f"lines {len(rows)}, turns {sum(r['kind'] == 'turn' for r in rows)}, errors {sum(bool(r.get('error')) for r in rows)}")
