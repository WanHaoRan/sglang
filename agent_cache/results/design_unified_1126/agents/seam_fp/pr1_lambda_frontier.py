"""PR1 check (CPU, read-only): the price of host room read off the eviction frontier, from run records.

Frontier age: for returning requests, the share whose history had left host memory (an SSD lookup that hit, a read,
or history computed again) by the gap before the request; the gap at which half of the returns find their history
gone is the idle age of the marginal holder (a*), per window of the run.
The price is then lam = k x holding_rate(a*), holding_rate from pr1_index_table.py and
k (seconds of TTFT per token) = (c + kappa) x s_read; s_read (the read's I/O seconds per token) is printed here.
Usage: <venv python> pr1_lambda_frontier.py <label>
"""
import bisect, importlib.util, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib
spec = importlib.util.spec_from_file_location("ntp_policy", "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py")
pol = importlib.util.module_from_spec(spec); sys.modules["ntp_policy"] = pol; spec.loader.exec_module(pol)

label = sys.argv[1]
r = lib.Run(label)
rows = []
for rid, rec in r.requests():
    cl = r.client.get(rid)
    if cl is None or cl.get("wait_s") is None:
        continue
    k = cl.get("k", cl.get("turn", 0))
    if not k:
        continue
    rts = rec["rts"]; adm = rts["entry"] + rts["queue_s"]
    gone = any(e["storage_hit"] >= 256 for e in rec.get("s2h_query", []) if e["t"] <= adm + 0.05)
    ideal, cached = cl.get("ideal_cached"), cl.get("cached_tokens")
    if ideal is not None and cached is not None and ideal - cached >= 256:
        gone = True
    rows.append((r.minute(rts["entry"]), cl["wait_s"], gone, rts["input_len"]))
print(f"== {label}: {len(rows)} returning requests")
edges = [0, 5, 15, 30, 60, 120, 240, 480, 900, 1e9]
for lo_m, hi_m in ((10, 30), (30, 60), (60, 90), (90, 110)):
    w = [x for x in rows if lo_m <= x[0] < hi_m]
    parts = []
    a_star = None
    for i in range(len(edges) - 1):
        b = [x for x in w if edges[i] <= x[1] < edges[i + 1]]
        if len(b) >= 5:
            share = sum(x[2] for x in b) / len(b)
            parts.append(f"{edges[i]:.0f}-{edges[i+1] if edges[i+1] < 1e9 else 'inf'}s:{share:.2f}(n{len(b)})")
            if a_star is None and share >= 0.5:
                a_star = edges[i]
    print(f"   minutes {lo_m}-{hi_m}: share of returns whose history had left memory, by gap: " + " ".join(parts) + f"  | first bin at >= 0.5 starts at {a_star} s")

io = [(e["pages"], e["ms"]) for e in r.kind["s2h_io"] if e["pages"] > 0]
s_read = sum(ms for _, ms in io) / max(1, sum(p for p, _ in io)) / 64 / 1e3
print(f"   read cost: {s_read*1e6:.1f} us/token over {len(io)} reads (whole run, I/O time only)")

# The index itself is tabulated by pr1_index_table.py (on the survival function; over the 64-point support the sup
# jumps with the distance to the next support point and remaining_gaps() is empty once 4 points are left).
