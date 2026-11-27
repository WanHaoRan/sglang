#!/usr/bin/env python3
"""Closed-loop identity on the recorded per-minute tables of campaigns 32 and 34 (CPU, read-only).

Reads  results/campaign{32,34}/checks/by_minute_<arm>.out  (columns: minute | in session, in flight |
sent, first requests | returning: mean TTFT s, p50, from SSD | server: running, queued, pool used).

For a window of minutes with N users in session throughout, operational identities (no model):
    X      = requests sent per second                      (throughput; sends = completions in a full loop)
    R      = (in flight) / X                               (Little: mean time a request is in the system)
    Zeff   = (N - in flight) / X                           (Little: mean time a user is outside the system)
    N / X  = R + Zeff                                      (each user is either in the system or outside)
    Dq     = queued / X                                    (mean time in the server's queue)
    Drun   = running / X                                   (mean time in the running set = B / X)
so   R = Dq + Drun (+ client/network) and TTFT = Dq + (admission to first token).

The script prints these per arm for minutes 90-109 and the campaign 32 -> 34 differences, and splits the
longer cycle N * dg into TTFT, running-set residence (B * dg) and the change of Zeff.
"""
import os
import sys

BASE = "/home/wanhr/sglang/agent_cache/results"
ARMS = ["ec_full", "jit_exact", "jit_p50", "ec_exact"]
LO, HI = 90, 109


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = [p.split() for p in line.strip().split("|")]
            if len(parts) != 5 or not parts[0] or not parts[0][0].isdigit():
                continue
            try:
                minute = int(parts[0][0])
                insess, inflight = int(parts[1][0]), int(parts[1][1])
                sent, first = int(parts[2][0]), int(parts[2][1])
                ttft = float(parts[3][0])
                running, queued = int(parts[4][0]), int(parts[4][1])
            except (ValueError, IndexError):
                continue
            rows.append(dict(m=minute, N=insess, F=inflight, sent=sent, first=first, ttft=ttft,
                             B=running, Q=queued))
    return rows


def window(rows, lo, hi):
    w = [r for r in rows if lo <= r["m"] <= hi]
    n = len(w)
    if n == 0:
        return None
    sent = sum(r["sent"] for r in w)
    ret = sum(r["sent"] - r["first"] for r in w)
    X = sent / (60.0 * n)
    N = sum(r["N"] for r in w) / n
    F = sum(r["F"] for r in w) / n
    B = sum(r["B"] for r in w) / n
    Q = sum(r["Q"] for r in w) / n
    # mean TTFT of returning requests, weighted by the number of returning requests sent that minute
    ttft = sum(r["ttft"] * (r["sent"] - r["first"]) for r in w) / max(ret, 1)
    return dict(n=n, X=X, g=1.0 / X, N=N, F=F, B=B, Q=Q, ttft=ttft,
                R=F / X, Zeff=(N - F) / X, Dq=Q / X, Drun=B / X, cycle=N / X)


def main():
    out = []
    P = out.append
    P("Closed-loop identity on the recorded per-minute tables, minutes %d-%d (N = users in session)" % (LO, HI))
    P("X req/s; g = 1/X GPU-s per request; R = in flight/X; Zeff = (N - in flight)/X; Dq = queued/X; Drun = running/X")
    P("")
    hdr = "%-4s %-10s %5s %6s %6s %6s %6s %6s | %7s %7s %7s %7s %7s %7s" % (
        "camp", "arm", "N", "X", "g", "inflt", "queue", "run", "N/X", "R", "Zeff", "Dq", "Drun", "TTFT")
    P(hdr)
    res = {}
    for camp in (32, 34):
        for arm in ARMS:
            path = os.path.join(BASE, "campaign%d" % camp, "checks", "by_minute_%s.out" % arm)
            if not os.path.exists(path):
                continue
            w = window(load(path), LO, HI)
            if not w:
                continue
            res[(camp, arm)] = w
            P("%-4d %-10s %5.0f %6.3f %6.3f %6.1f %6.1f %6.1f | %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f" % (
                camp, arm, w["N"], w["X"], w["g"], w["F"], w["Q"], w["B"], w["cycle"], w["R"], w["Zeff"],
                w["Dq"], w["Drun"], w["ttft"]))
    P("")
    P("Check of the pieces: TTFT against Dq = queued/X, and R against Dq + Drun")
    for k, w in res.items():
        P("  c%d %-10s TTFT %6.1f  Dq %6.1f  (TTFT - Dq = %+5.1f)   R %6.1f  Dq+Drun %6.1f  (diff %+5.1f)" % (
            k[0], k[1], w["ttft"], w["Dq"], w["ttft"] - w["Dq"], w["R"], w["Dq"] + w["Drun"],
            w["R"] - w["Dq"] - w["Drun"]))
    P("")
    P("Campaign 32 minus campaign 34, same arm (the loop's extra GPU time per request, dg):")
    P("%-10s %6s | %8s = %8s + %8s + %8s | %9s %9s %9s | %8s" % (
        "arm", "dg", "N*dg", "dTTFT", "dDrun", "dZeff", "dTTFT/dg", "(N-B)", "N", "B*dg"))
    for arm in ARMS:
        if (32, arm) in res and (34, arm) in res:
            a, b = res[(32, arm)], res[(34, arm)]
            dg = a["g"] - b["g"]
            dcyc = a["cycle"] - b["cycle"]
            dttft = a["ttft"] - b["ttft"]
            # residence beyond TTFT: R - TTFT (running set after first token, client side)
            drun = (a["R"] - a["ttft"]) - (b["R"] - b["ttft"])
            dz = a["Zeff"] - b["Zeff"]
            Bm = 0.5 * (a["B"] + b["B"])
            P("%-10s %6.3f | %8.1f = %8.1f + %8.1f + %8.1f | %9.1f %9.1f %9.0f | %8.1f" % (
                arm, dg, dcyc, dttft, drun, dz, dttft / dg, a["N"] - Bm, a["N"], Bm * dg))
            P("%-10s        shares of N*dg: TTFT %.0f %%, running-set residence %.0f %%, Zeff %.0f %%;"
              "  dTTFT/dg = %.2f N;  (N - B) = %.2f N;  (N - B) - dZeff/dg = %.1f" % (
                  "", 100 * dttft / dcyc, 100 * drun / dcyc, 100 * dz / dcyc, dttft / dg / a["N"],
                  (a["N"] - Bm) / a["N"], a["N"] - Bm - dz / dg))
    P("")
    P("Exact split of the TTFT difference (algebra on the measured pieces):")
    P("  TTFT = (N - B)/X - Zeff - resid + tpre,   resid = R - Dq - Drun,   tpre = TTFT - Dq")
    P("%-10s %9s = %12s %9s %9s %9s" % ("arm", "dTTFT", "d[(N-B)/X]", "- dZeff", "- dresid", "+ dtpre"))
    for arm in ARMS:
        if (32, arm) in res and (34, arm) in res:
            a, b = res[(32, arm)], res[(34, arm)]
            t1 = (a["N"] - a["B"]) / a["X"] - (b["N"] - b["B"]) / b["X"]
            t2 = -(a["Zeff"] - b["Zeff"])
            t3 = -((a["R"] - a["Dq"] - a["Drun"]) - (b["R"] - b["Dq"] - b["Drun"]))
            t4 = (a["ttft"] - a["Dq"]) - (b["ttft"] - b["Dq"])
            P("%-10s %9.1f = %12.1f %9.1f %9.1f %9.1f   (sum %.1f)" % (
                arm, a["ttft"] - b["ttft"], t1, t2, t3, t4, t1 + t2 + t3 + t4))
    P("")
    P("Thinking users (N - in flight) minute by minute, campaign 32 ec_full, minutes 84-109:")
    path = os.path.join(BASE, "campaign32", "checks", "by_minute_ec_full.out")
    rows = [r for r in load(path) if 84 <= r["m"] <= 109]
    P("  " + " ".join("%d" % (r["N"] - r["F"]) for r in rows))
    path = os.path.join(BASE, "campaign34", "checks", "by_minute_ec_full.out")
    rows = [r for r in load(path) if 84 <= r["m"] <= 109]
    P("same, campaign 34 ec_full:")
    P("  " + " ".join("%d" % (r["N"] - r["F"]) for r in rows))
    P("In campaign 32 the thinking population falls from about 50 to about 23 as the loop halves the request")
    P("rate (long pauses that began before the loop run out); in steady state it would be X * Z = about 22.")
    P("")
    P("Reading: N/X = R + Zeff holds by construction; the content is in the split.")
    P("  dR = d(in flight / X) is the part of the longer cycle that users spend in the system;")
    P("  of it, B*dg stays in the running set (after admission), the rest is queue wait = TTFT;")
    P("  dZeff is the change of the time-average number of users outside the system per unit of throughput")
    P("  (a transient of the thinking population in a 20-minute window with pauses of up to 1,200 s).")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "identity_c32_c34.out"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
