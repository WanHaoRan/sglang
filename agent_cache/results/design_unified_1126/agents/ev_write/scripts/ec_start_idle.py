"""Join-free check of early evictions: the engine's own ec_start lines (expected-cost arms) say how many tokens of an
idle session's committed chain were missing from memory, and how long after the commit. First line per plan only."""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import events, kv, run_paths
for camp, label in (("campaign34", "ec_full"), ("campaign34", "ec_exact"), ("campaign32", "ec_full")):
    _, _, extract = run_paths(camp, label)
    first = {}; chain = {}
    for t, ev, rest in events(extract, {"ec_start", "ntp_plan"}):
        d = kv(rest)
        if ev == "ntp_plan": chain[d["rid"]] = int(d["chain"])
        elif d["rid"] not in first: first[d["rid"]] = (int(d["idle_ms"]) / 1000.0, int(d["tokens"]), t)
    n = len(first)
    rows = [(idle, tok, chain.get(rid, 0)) for rid, (idle, tok, t) in first.items() if t >= 1800]
    print(f"{camp} {label}: {len(chain)} plans, {n} with an ec_start line (the chain had at least 256 tokens out of memory before the turn came); from minute 30: {len(rows)}")
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 30), (30, 120), (120, 1e9)):
        sel = [r for r in rows if lo <= r[0] < hi]
        whole = sum(1 for r in sel if r[2] and r[1] >= 0.9 * r[2])
        print(f"   first seen missing {lo:g}-{hi:g} s after the commit: {len(sel):5d} plans ({len(sel) / max(len(rows), 1):5.1%}), tokens missing then {sum(r[1] for r in sel) / 1e6:6.2f} M "
              f"(median {sorted(r[1] for r in sel)[len(sel) // 2] if sel else 0}), the whole chain in {whole}")
