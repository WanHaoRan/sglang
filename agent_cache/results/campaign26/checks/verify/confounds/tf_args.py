"""Test (f1b): server arguments that differ between runs (server_args.txt is one line holding a dict repr).
  nice -n 19 python3 -I -B tf_args.py"""
import ast, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ARMS = [a for a in C.RUNS]
args = {}
for a in ARMS:
    s = open(os.path.join(C.RES, C.RUNS[a], "server_args.txt")).read()
    body = s[s.index("server_args=") + len("server_args="):].strip()
    try:
        args[a] = ast.literal_eval(body)
        how = "literal"
    except (ValueError, SyntaxError):
        args[a] = dict(re.findall(r"'([a-z_0-9]+)': ((?:'[^']*'|\"[^\"]*\"|\[[^\]]*\]|\{[^}]*\}|[^,}]+))", body))
        how = "regex"
    print(f"{a:14s} {len(args[a])} fields ({how})")
keys = sorted(set().union(*(set(v) for v in args.values())))
print(f"\nfields that differ between runs (of {len(keys)}):")
for k in keys:
    vals = {a: repr(args[a].get(k, "<absent>")) for a in ARMS}
    if len(set(vals.values())) > 1:
        groups = {}
        for a in ARMS:
            groups.setdefault(vals[a], []).append(a)
        print(f"  {k}:")
        for v, g in groups.items():
            print(f"      {v[:160]}  <- {', '.join(g)}")
print("\nfields equal in all runs that bear on capacity and scheduling:")
for k in ("mem_fraction_static", "max_running_requests", "max_total_tokens", "chunked_prefill_size", "max_prefill_tokens", "schedule_policy", "schedule_conservativeness",
          "page_size", "hicache_ratio", "hicache_size", "hicache_write_policy", "hicache_io_backend", "hicache_mem_layout", "hicache_storage_backend", "hicache_storage_prefetch_policy",
          "enable_hierarchical_cache", "radix_eviction_policy", "decode_log_interval", "cuda_graph_max_bs", "attention_backend", "tp_size", "dp_size", "stream_interval"):
    vals = set(repr(args[a].get(k, "<absent>")) for a in ARMS)
    print(f"  {k}: {' / '.join(sorted(vals))[:200]}")
