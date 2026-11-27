"""F1, the landing quantile: errors of the admission estimate and a replay of the hold under other margins.

  python f1_landing.py <label> [ec]

A. Error of the engine's admission estimate, e = D - predicted, with D the turn's GPU turn (the first time admission
   took a later arrival while this turn waited on its read; its own admission otherwise):
   at arrival (ntp_admit pred_wait_ql, every planned turn), at the first hold (ntp_hold), at the replayed release.
B. The hold replayed with margin m: release when (position + 0.5) / rate - load <= m (never at positions 0-1), the
   line as it was, the read's own time from release to landing held fixed. Per read: stall = (landing + 0.4 s - D)+,
   pin = (D - landing)+ in seconds and token-seconds.
C. The margin the rule q = lambda L / (c + lambda L) picks, with c measured (share of stall seconds with no other ready
   turn waiting) and lambda on a grid, against today's 2 s.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import runlib as R  # noqa: E402

LAG_S = 0.4
WINDOWS = ((0, 30), (30, 70), (70, 90), (90, 110))
QS = (0.05, 0.2, 0.5, 0.8, 0.95, 1.0)
M_GRID = [x * 0.25 for x in range(-48, 81)]     # -12 .. +20 s
LAMBDAS = (0.0, 1e-6, 2e-5, 1e-4)
CS = (1.0, 0.3, 0.05, 0.0)


def qq(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if p >= 1.0:
        return xs[-1]
    pos = p * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def dist(xs):
    if not xs:
        return "n 0"
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return (f"n {len(xs):5d} | p5 {qq(xs, .05):+6.2f} p20 {qq(xs, .2):+6.2f} p50 {qq(xs, .5):+6.2f} p80 {qq(xs, .8):+6.2f} "
            f"p95 {qq(xs, .95):+6.2f} | mean {m:+6.2f} sd {sd:5.2f} | |e| p50 {qq([abs(x) for x in xs], .5):5.2f} "
            f"p90 {qq([abs(x) for x in xs], .9):5.2f}")


def window_of(t):
    m = t / 60.0
    for lo, hi in WINDOWS:
        if lo <= m < hi:
            return f"{lo}-{hi}"
    return None


def pos_bin(p):
    for lo, hi in ((0, 1), (2, 5), (6, 15), (16, 30), (31, 50), (51, 999)):
        if lo <= p <= hi:
            return f"{lo}-{hi if hi < 999 else '+'}"


def main():
    label = sys.argv[1]
    ec = len(sys.argv) > 2 and sys.argv[2] == "ec"
    d = R.load(label)
    Q = R.Queue(d)
    RD = R.Reads(d, Q)
    usage = R.usage_series(d)
    D = R.gpu_turn_refined(Q, RD, usage)
    H = R.hold_replay(d, Q, RD, dt=0.1, expected_cost=ec)
    ready = R.ready_series(Q, RD)
    t0 = d["run_rec"]["t_start"]
    ev = d["evts"]
    admit = R.first_by_rid(ev.get("ntp_admit", []))
    arrive = R.first_by_rid(ev.get("ntp_arrive", []))
    landed = {}
    for r in RD.reads:
        if r["land"] is not None and r["adm"] is not None and r["land"] <= r["adm"] + 0.05:
            landed.setdefault(r["rid"], []).append(r["land"])

    print(f"==== {label} ====")
    # ---------- A1: at arrival ----------
    print("\n== A1. At arrival: e = D - (arrival + pred_wait_ql), planned turns ==")
    rows = []
    for rid, (t_adm, f) in admit.items():
        if rid not in Q.arr or not isinstance(f.get("pred_wait_ql_ms"), int) or rid not in arrive:
            continue
        a = Q.arr[rid]
        pred = f["pred_wait_ql_ms"] / 1e3
        rows.append({"rid": rid, "t": a - t0, "pred": pred, "e_D": D[rid] - (a + pred), "e_adm": Q.adm[rid] - (a + pred),
                     "reader": rid in landed, "pos": arrive[rid][1]["pos"], "usage": R.usage_at(usage, a),
                     "land": max(landed[rid]) if rid in landed else None, "adm": Q.adm[rid], "arr": a})
    for name, sel in (("all planned turns", lambda r: True), ("non-readers", lambda r: not r["reader"]),
                      ("readers", lambda r: r["reader"])):
        print(f"  {name}:")
        for lo, hi in WINDOWS:
            xs = [r["e_D"] for r in rows if sel(r) and lo * 60 <= r["t"] < hi * 60]
            print(f"    minutes {lo:3d}-{hi:<3d}: {dist(xs)}")
        for uname, usel in (("pool < 0.90", lambda r: r["usage"] < 0.90), ("pool >= 0.90", lambda r: r["usage"] >= 0.90)):
            xs = [r["e_D"] for r in rows if sel(r) and usel(r)]
            print(f"    {uname:13s}: {dist(xs)}")
    print("  by queue position at arrival (all planned turns, minutes 30-110):")
    for pb in ("0-1", "2-5", "6-15", "16-30", "31-50", "51-+"):
        xs = [r["e_D"] for r in rows if pos_bin(r["pos"]) == pb and 30 * 60 <= r["t"] < 110 * 60]
        pr = [r["pred"] for r in rows if pos_bin(r["pos"]) == pb and 30 * 60 <= r["t"] < 110 * 60]
        print(f"    pos {pb:6s}: {dist(xs)} | predicted wait p50 {qq(pr, .5):5.1f} s")
    print("  relative error e / predicted wait, positions >= 2, minutes 30-110: " + dist(
        [r["e_D"] / r["pred"] for r in rows if r["pos"] >= 2 and r["pred"] > 0 and 30 * 60 <= r["t"] < 110 * 60]))
    print("  readers, admission against the estimators (minutes 30-110):")
    rd = [r for r in rows if r["reader"] and 30 * 60 <= r["t"] < 110 * 60]
    print("    adm - (arrival + position/rate)                      : " + dist([r["e_adm"] for r in rd]))
    print("    adm - (max(arrival + position/rate, landing) + 0.4 s): " + dist(
        [r["adm"] - (max(r["arr"] + r["pred"], r["land"]) + LAG_S) for r in rd]))
    print("    D   - (arrival + position/rate)                      : " + dist([r["e_D"] for r in rd]))

    # ---------- reads of queued turns ----------
    reads = []
    for r in RD.reads:
        if r["staged"] or r["arr"] is None or r["adm"] is None or r["land"] is None or r["disp"] is None:
            continue
        st = H[r["n"]]
        if not st:
            continue
        rel2 = R.release_time(st, r["pages"], R.HOLD_MARGIN_S)
        if rel2 is None:
            continue
        at = next(s for s in st if s[0] == rel2)
        rate = Q.rate(rel2)
        reads.append({"r": r, "st": st, "rel2": rel2, "pos_rel": at[1],
                      "wait_rel": None if rate is None else (at[1] + 0.5) / rate[0],
                      "load_rel": R.LOAD_FIXED_S + at[4] * (r["pages"] + at[3]), "D": D[r["rid"]],
                      "usage_rel": R.usage_at(usage, rel2), "win": window_of(r["park"] - t0), "tokens": r["pages"] * R.PAGE,
                      "dur": r["land"] - rel2})
    print(f"\n  reads of queued turns replayed: {len(reads)} (of {len(RD.reads)} parked reads)")

    # ---------- A2: at the first hold (logged) ----------
    print("\n== A2. At the first hold (ntp_hold): e = D - (t_hold + wait) ==")
    hrows = [(x, x["r"]["hold"]) for x in reads if "hold" in x["r"]]
    for lo, hi in WINDOWS:
        xs = [x["D"] - (h["t"] + h["wait_ms"] / 1e3) for x, h in hrows if x["win"] == f"{lo}-{hi}"]
        print(f"    minutes {lo:3d}-{hi:<3d}: {dist(xs)}")
    for pb in ("2-5", "6-15", "16-30", "31-50", "51-+"):
        xs = [x["D"] - (h["t"] + h["wait_ms"] / 1e3) for x, h in hrows if pos_bin(h["pos"]) == pb]
        print(f"    pos {pb:6s}: {dist(xs)}")
    print("    hold length at the first decision (s): " + dist([h["hold_ms"] / 1e3 for x, h in hrows]))
    print("    predicted wait at the first decision (s): " + dist([h["wait_ms"] / 1e3 for x, h in hrows]))

    # ---------- A3: at the replayed release ----------
    print("\n== A3. At the replayed release (margin 2 s): e = D - (release + (position + 0.5) / rate) ==")
    timed = [x for x in reads if x["wait_rel"] is not None]
    for x in timed:
        x["e_rel"] = x["D"] - (x["rel2"] + x["wait_rel"])
    for lo, hi in WINDOWS:
        xs = [x["e_rel"] for x in timed if x["win"] == f"{lo}-{hi}"]
        print(f"    minutes {lo:3d}-{hi:<3d}: {dist(xs)}")
        xs = [x["e_rel"] for x in timed if x["win"] == f"{lo}-{hi}" and x["pos_rel"] >= 2]
        print(f"      positions >= 2 at release : {dist(xs)}")
    for uname, usel in (("pool < 0.90", lambda x: x["usage_rel"] < 0.90), ("pool >= 0.90", lambda x: x["usage_rel"] >= 0.90)):
        print(f"    {uname:13s}: {dist([x['e_rel'] for x in timed if usel(x)])}")
    for pb in ("0-1", "2-5", "6-15", "16-30", "31-50", "51-+"):
        xs = [x["e_rel"] for x in timed if pos_bin(x["pos_rel"]) == pb and x["win"] in ("30-70", "70-90", "90-110")]
        print(f"    pos at release {pb:6s} (minutes 30-110): {dist(xs)}")
    print("    predicted wait at release (s), positions >= 2: " + dist([x["wait_rel"] for x in timed if x["pos_rel"] >= 2]))

    # ---------- B: the hold under other margins ----------
    def outcome(x, m):
        rel = R.release_time(x["st"], x["r"]["pages"], m)
        if rel is None:
            # never released by this margin before its turn was admitted: held to the end of what the log shows
            rel = x["st"][-1][0]
        rel = max(rel, x["r"]["park"])
        land = rel + x["dur"]
        stall = max(0.0, land + LAG_S - x["D"])
        pin = max(0.0, x["D"] - land)
        return rel, land, stall, pin

    def table(group, title):
        if len(group) < 20:
            print(f"\n  -- {title}: {len(group)} reads, too few --")
            return None
        tok = sum(x["tokens"] for x in group)
        base = [outcome(x, 2.0) for x in group]
        act_stall = [x["r"]["adm"] - x["D"] for x in group]
        act_pin = [max(0.0, x["D"] - x["r"]["land"]) for x in group]
        print(f"\n  -- {title}: {len(group)} reads, mean {tok / len(group):.0f} tokens; actual stall mean "
              f"{sum(act_stall) / len(group):.2f} s (replay at 2 s: {sum(b[2] for b in base) / len(group):.2f}), actual pin "
              f"mean {sum(act_pin) / len(group):.2f} s (replay {sum(b[3] for b in base) / len(group):.2f}); held under 2 s: "
              f"{sum(1 for x in group if x['rel2'] > x['r']['park'] + 0.15)}; at positions 0-1 at release: "
              f"{sum(1 for x in group if x['pos_rel'] <= 1)} --")
        print("    margin s | released earlier/later than today (reads) | late % | stall s per read | pin s per read | "
              "pin K token-s per read | d stall vs 2 s | d pin K token-s vs 2 s")
        res = {}
        for m in M_GRID:
            o = [outcome(x, m) for x in group]
            res[m] = (sum(v[2] for v in o) / len(group), sum(v[3] for v in o) / len(group),
                      sum(v[3] * x["tokens"] for v, x in zip(o, group)) / len(group),
                      sum(1 for v in o if v[2] > 0) / len(group),
                      sum(1 for v, b in zip(o, base) if v[0] < b[0] - 0.05), sum(1 for v, b in zip(o, base) if v[0] > b[0] + 0.05))
        b = res[2.0]
        for m in (-12.0, -6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0):
            v = res[m]
            print(f"    {m:+6.1f} | {v[4]:5d} / {v[5]:5d} | {100 * v[3]:5.1f} | {v[0]:7.3f} | {v[1]:7.3f} | {v[2] / 1e3:8.1f} | "
                  f"{v[0] - b[0]:+7.3f} | {(v[2] - b[2]) / 1e3:+8.1f}")
        return res

    def quantile_margins(group):
        """Margins that land the read at the q-quantile of D: m_q = -Q_q(e), e the error at the release."""
        e = [x["e_rel"] for x in group if "e_rel" in x and x["pos_rel"] >= 2]
        return {q_: -qq(e, q_) for q_ in QS}, len(e)

    def nearest(res, m):
        key = min(res, key=lambda k: abs(k - m))
        return key, res[key]

    print("\n== B. The hold replayed under other margins (reads of queued turns; own release-to-landing time fixed) ==")
    groups = [(f"minutes {lo}-{hi}", [x for x in reads if x["win"] == f"{lo}-{hi}"]) for lo, hi in WINDOWS]
    groups.append(("pool < 0.90 at release", [x for x in reads if x["usage_rel"] < 0.90]))
    groups.append(("pool >= 0.90 at release", [x for x in reads if x["usage_rel"] >= 0.90]))
    groups.append(("held by today's rule (released after park), all minutes", [x for x in reads if x["rel2"] > x["r"]["park"] + 0.15]))
    summary = {}
    for title, group in groups:
        res = table(group, title)
        if res is None:
            continue
        mq, n_e = quantile_margins(group)
        print(f"    margins from the error quantiles (n {n_e} releases at positions >= 2): " +
              ", ".join(f"q={q_:g}: m={mq[q_]:+.2f} s" for q_ in QS))
        print("    q | margin s | late % | stall s per read | pin K token-s per read | d stall vs today | d pin K token-s vs today")
        b = res[2.0]
        for q_ in QS:
            if math.isnan(mq[q_]):
                continue
            key, v = nearest(res, max(-12.0, min(20.0, mq[q_])))
            print(f"    {q_:4.2f} | {mq[q_]:+6.2f} (grid {key:+.2f}) | {100 * v[3]:5.1f} | {v[0]:7.3f} | {v[2] / 1e3:8.1f} | {v[0] - b[0]:+7.3f} | "
                  f"{(v[2] - b[2]) / 1e3:+8.1f}")
        # the measured lateness weight: share of the stall seconds with no other ready turn waiting
        num = den = 0.0
        for x in group:
            s = x["r"]["adm"] - x["D"]
            if s > 0.05:
                num += s * R.idle_share(ready, x["D"], x["r"]["adm"])
                den += s
        c_meas = num / den if den > 0 else float("nan")
        L = qq([x["tokens"] for x in group], .5)
        print(f"    measured c (share of stall seconds with no other ready turn waiting): {c_meas:.3f}; median read {L:.0f} tokens")
        print("    lambda | c | q = lambda L / (c + lambda L) | margin by the rule | cost-minimizing margin on the grid | "
              "cost today -> at the rule's margin -> at the best margin (s of TTFT per read) | d stall per read (rule - today) | "
              "d pin K token-s per read")
        for lam in LAMBDAS:
            for c in sorted(set(CS + ((round(c_meas, 3),) if not math.isnan(c_meas) else ()))):
                cost = {m: c * v[0] + lam * v[2] for m, v in res.items()}
                best = min(cost, key=lambda k: (cost[k], abs(k - 2.0)))
                if c + lam * L > 0:
                    q_rule = lam * L / (c + lam * L)
                    e = [x["e_rel"] for x in group if "e_rel" in x and x["pos_rel"] >= 2]
                    m_rule = -qq(e, q_rule) if e else float("nan")
                else:
                    q_rule, m_rule = float("nan"), float("nan")
                if math.isnan(m_rule):
                    print(f"    {lam:7.0e} | {c:5.3f} | q undefined (c = 0, lambda = 0): every margin costs the same")
                    continue
                key, v = nearest(res, max(-12.0, min(20.0, m_rule)))
                tag = " (measured c)" if not math.isnan(c_meas) and abs(c - round(c_meas, 3)) < 1e-9 else ""
                print(f"    {lam:7.0e} | {c:5.3f}{tag} | {q_rule:5.3f} | {m_rule:+6.2f} | {best:+6.2f} | {cost[2.0]:7.3f} -> {cost[key]:7.3f} -> "
                      f"{cost[best]:7.3f} | {v[0] - b[0]:+7.3f} | {(v[2] - b[2]) / 1e3:+8.1f}")
        e_all = [x["e_rel"] for x in group if "e_rel" in x and x["pos_rel"] >= 2]
        summary[title] = {"n": len(group), "c": c_meas, "L": L, "res": {f"{m:g}": v for m, v in res.items()},
                          "mq": {f"{k:g}": v for k, v in mq.items()},
                          "e_quantiles": [qq(e_all, i / 100.0) for i in range(101)] if e_all else []}
    with open(os.path.join(C.OUT, f"f1_landing_{label}.json"), "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    main()
