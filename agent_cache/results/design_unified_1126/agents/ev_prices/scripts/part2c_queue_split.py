"""Part 2: the queue split into ready turns and turns with a lookup/read under way, per phase and arm (campaign 34).
If passing over a reading turn only reorders admissions, every extra reading turn is offset by one ready turn less.
usage: part2c_queue_split.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

print("Campaign 34: the queue split into ready turns and turns with a lookup/read under way, time averages per phase")
print("arm | minutes | N | X req/min | Q | Qr ready | Q - Qr reading | users pausing (N - LT - LD) | mean TTFT sent | mean stall per request")
tab = {}
for arm in ("ec_full", "jit_exact", "ec_exact", "jit_p50", "hicache"):
    st = qstate.build("c34", arm)
    rows = readers.annotate(st)
    dT, dD = [], []
    for r in rows:
        if r["ttft"] is None or r["latency"] is None:
            continue
        s, ft, e = r["t_send"], r["t_send"] + r["ttft"], r["t_send"] + r["latency"]
        dT += [(s, 1), (ft, -1)]
        dD += [(ft, 1), (e, -1)]
    LT, LD = qstate.Step.from_deltas(dT), qstate.Step.from_deltas(dD)
    for lo, hi in ((28, 60), (60, 85), (85, 110)):
        a, b = lo * 60.0, hi * 60.0
        N, Q, Qr = st["n_live"].mean(a, b), st["tot"].mean(a, b), st["rdy"].mean(a, b)
        fin = [r for r in rows if r["latency"] is not None and a <= r["t_send"] + r["latency"] < b]
        sent = [r for r in rows if a <= r["t_send"] < b and r["ttft"] is not None]
        adm = [r for r in rows if a <= r["adm"] < b]
        X = len(fin) / (b - a) * 60
        lz = N - LT.mean(a, b) - LD.mean(a, b)
        T = lib.mean(r["ttft"] for r in sent)
        S = sum(r["stall"] for r in adm) / len(adm)
        tab[(arm, lo)] = (N, X, Q, Qr, Q - Qr, lz, T, S)
        print(f"{arm:9s} | {lo:3d}-{hi:3d} | {N:6.1f} | {X:6.1f} | {Q:5.1f} | {Qr:5.1f} | {Q-Qr:5.1f} | {lz:5.1f} | {T:6.2f} | {S:5.2f}")
print()
print("differences against ec_full: d(reading), d(ready), d(Q) = share of the extra reading turns NOT offset by a shorter ready queue")
for arm in ("jit_p50", "jit_exact", "ec_exact"):
    for lo in (28, 60, 85):
        x, y = tab[(arm, lo)], tab[("ec_full", lo)]
        dbusy, drdy, dq = x[4] - y[4], x[3] - y[3], x[2] - y[2]
        print(f"{arm:9s} {lo:3d}+: d reading {dbusy:+5.1f}, d ready {drdy:+5.1f}, d Q {dq:+5.1f} ({dq/dbusy if abs(dbusy)>1 else float('nan'):+.2f} of d reading); "
              f"d N {x[0]-y[0]:+.1f}, d pausing {x[5]-y[5]:+.1f}, d X {x[1]-y[1]:+.1f} req/min; d TTFT {x[6]-y[6]:+.2f} s; d stall/req {x[7]-y[7]:+.2f} s")
