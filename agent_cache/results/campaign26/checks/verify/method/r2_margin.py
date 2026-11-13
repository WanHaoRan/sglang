"""Does the 300 s margin of the kept-turn rule move a comparison? Shared returning turns and paired differences for
margins of 0, 300, 600 and 900 s, and turns the server saw arrive that the client never recorded, from the cache.
  nice -n 19 python3 -I -B r2_margin.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, mean, LEAD5, SEVEN, ALL11, raw

for names, label, pairs in ((LEAD5, "LEAD5", (("ec_r13", "jit_exact"), ("nofc", "ec_r13"), ("nofc_noend", "ec_r13"), ("nofc_noend", "nofc"))),
                            (SEVEN, "SEVEN", (("jit_p50_ctl", "jit_exact"), ("hintfree_ctl", "jit_exact"), ("nofc_noend", "ec_r13"))),
                            (ALL11, "ALL11", (("base_rep", "baseline"), ("jit_exact", "baseline"), ("nofc_noend", "jit_exact"), ("ec_r13", "nofc_noend")))):
    print(f"== {label}")
    for margin in (0, 300, 600, 900):
        arms = {a: arm(a, margin=margin) for a in names}
        keys = shared(names, arms)
        m = {a: mean(arms[a]["by"][k]["ttft"] for k in keys) for a in names}
        gain = ""
        if "baseline" in m:
            gain = " | share of the gain, nofc_noend: " + " ".join(f"{100 * (m[b] - m['nofc_noend']) / (m[b] - m['jit_exact']):.1f} %" for b in ("baseline", "base_rep"))
        print(f"   margin {margin:3d} s: {len(keys):5d} turns | " + "  ".join(f"{a} - {b} {m[a] - m[b]:+.3f}" for a, b in pairs) + gain)

print("\nrequests the server logged as arrived (prefetch_start with a c-t id, not a staged read's id before arrival) that have no client record, by arm: count, and of them arrived before the kept cutoff")
for a in ALL11 + ["jit_p50_d_cut"]:
    d = arm(a)
    s = d["server"]
    arrived = {}
    for e in s["evts"].get("ntp_arrive", []) or s["evts"].get("prefetch_start", []):
        if e["conv"] is not None:
            arrived.setdefault((e["conv"], e["turn"]), e["t"] - d["t0"])
    src = "ntp_arrive" if s["evts"].get("ntp_arrive") else "prefetch_start"
    missing = sorted(t for k, t in arrived.items() if k not in d["by_all"])
    print(f"   {a:13s} ({src}): {len(missing):3d} without a client record; before the cutoff {sum(t < d['end'] for t in missing)}; client turns with latency over 300 s: {sum(r['latency'] > 300 for r in d['every'])}")
