"""Part 4: the direct cost of a lookup that finds nothing. Queue wait of returning turns with no SSD read, by what
the arrival gate did; light load only (no other ready turn waiting at arrival, pool use < 0.90).
usage: part4_empty_lookup_stall.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

print("Direct cost of a lookup that finds nothing: queue wait of returning turns with no SSD read, by what the arrival gate did.")
print("light load only: arrival with no other ready turn waiting and pool use < 0.90")
for camp, arm, hi in (("c34", "ec_full", 28), ("c34", "jit_exact", 28), ("c34", "ec_exact", 28), ("c33", "ec_full", 110),
                      ("c33", "jit_exact", 110), ("c33", "hicache", 110), ("c34", "hicache", 28)):
    st = qstate.build(camp, arm)
    rows = readers.annotate(st)
    cov, emp1, emp2, busyspan = [], [], [], []
    for r in rows:
        if (r["k"] == 0 or r["arr"] >= hi * 60 or r["n_reads"] > 0 or r["reads"] or r["q_rdy_arr"] > 0 or r["use_arr"] >= 0.90
                or r["ttft"] is None):
            continue
        nl = len(r["lookups"])
        if nl == 0:
            cov.append(r)
        elif all(x[3] == 0 for x in r["lookups"]):
            (emp1 if nl == 1 else emp2).append(r)
            busyspan.append(r["busy_q"])

    def s(xs, f):
        v = [f(r) for r in xs]
        return f"n {len(v):5d} mean {lib.mean(v)*1e3:6.1f} ms p50 {lib.quantile(v, .5)*1e3:5.1f} p90 {lib.quantile(v, .9)*1e3:6.1f}"

    print(f"{camp}:{arm} to minute {hi}:")
    print("   no lookup (gate covered / none):  queue wait " + s(cov, lambda r: r["queue"]) + " | TTFT " + s(cov, lambda r: r["ttft"]))
    print("   one lookup, found nothing:        queue wait " + s(emp1, lambda r: r["queue"]) + " | TTFT " + s(emp1, lambda r: r["ttft"]))
    if emp2:
        print("   2+ lookups, all found nothing:    queue wait " + s(emp2, lambda r: r["queue"]) + " | TTFT " + s(emp2, lambda r: r["ttft"]))
    d = lib.mean(r["queue"] for r in emp1 + emp2) - lib.mean(r["queue"] for r in cov)
    n = len(emp1) + len(emp2)
    print(f"   lookup time (issue to answer, summed per turn): mean {lib.mean(busyspan)*1e3:.1f} ms; extra queue wait {d*1e3:+.1f} ms a turn on {n} turns = {d*n:.1f} s in all "
          f"({d*n/(len(cov)+n)*1e3:.1f} ms per returning turn at light load)")
