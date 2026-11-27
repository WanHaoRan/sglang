"""Part 2 on the runs: the most a read before arrival could have saved, per regime, from what each turn went through.

  python ceiling.py <label> [<label> ...]

For every returning request of a run: its gap G (what the client slept), whether it waited on its own SSD read, and
its stall S = admission - GPU turn (the part of the read the queue did not hide; 0 for a turn that was not passed over).
  oracle     min(G, S): an exactly timed read started that much earlier lands as the GPU turn comes (read time and
             line as they were).
  c-weighted the same times the share of the stall during which no other ready turn waited (what the sum of TTFT sees).
  rule       the engine's read_start on the entry the client sent for this turn (p10/p50/p90 of the previous request's
             hint), with x = S known, c = 1 and rent = lambda x the read's tokens; scored on the true gap; the pin is
             given back as ec_release does.
Windows by admission time: minutes 0-30, 30-70, 70-110 (campaign 34's regimes), and the whole run.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import fetch_value as FV  # noqa: E402

WINDOWS = ((0, 30), (30, 70), (70, 110), (0, 121))
LAMS = (0.0, 1e-6, 2e-5, 5e-5, 1e-4)   # 5e-5 is the price the campaign 34 arms ran with


def main():
    P = C.policy()
    for label in sys.argv[1:]:
        reqs = json.load(open(os.path.join(C.EXTRACTS, f"{label}.requests.json")))["rows"]
        reads = json.load(open(os.path.join(C.EXTRACTS, f"{label}.reads.json")))["rows"]
        idle = {}
        tokens = {}
        for r in reads:
            if "idle_share_stall" in r:
                idle[r["rid"]] = r["idle_share_stall"]
            tokens[r["rid"]] = max(tokens.get(r["rid"], 0), r["tokens"])
        # the entry sent for each turn: the previous request's hint names this rid as next_rid
        entry = {}
        for r in reqs:
            h = r.get("hint")
            if h and h.get("next_rid") and h.get("p50") is not None and not h.get("ended"):
                entry[h["next_rid"]] = h
        ret = [r for r in reqs if r.get("k") is not None and r["k"] > 0 and r.get("wait_s") is not None]
        print(f"==== {label}: {len(ret)} returning requests, {sum(1 for r in ret if r['reader'])} waited on their own read, "
              f"{sum(1 for r in ret if r['rid'] in entry)} had a table entry ====")
        print("window min | returning | readers % | passed over % | stall s per returning request | oracle min(G, S) | "
              "c-weighted oracle | share of the stall an exact read could hide % | measured c (stall-weighted)")
        for lo, hi in WINDOWS:
            w = [r for r in ret if lo * 60 <= r["adm"] < hi * 60]
            if not w:
                continue
            S = np.array([max(0.0, r["adm"] - r["D"]) if r["reader"] else 0.0 for r in w])
            G = np.array([r["wait_s"] for r in w])
            cw = np.array([idle.get(r["rid"], 0.0) for r in w])
            orc = np.minimum(G, S)
            n = len(w)
            print(f"  {lo:3d}-{hi:<3d} | {n:6d} | {100 * sum(1 for r in w if r['reader']) / n:5.1f} | {100 * (S > 0.05).mean():5.1f} | "
                  f"{S.sum() / n:6.3f} | {orc.sum() / n:6.3f} | {(orc * cw).sum() / n:6.3f} | {100 * orc.sum() / max(1e-9, S.sum()):5.1f} | "
                  f"{(S * cw).sum() / max(1e-9, S.sum()):5.3f}")
        # the rule on the entries the client sent
        have = [r for r in ret if r["reader"] and r["adm"] - r["D"] > 0.05 and r["rid"] in entry]
        if have:
            print("  the engine's read_start with the true x = S, on the entry sent (readers that were passed over and had an entry): "
                  "seconds per returning request of the window")
            print("  window | lambda | readers scored | decisions now/wait/arrival % | own TTFT saved (c = 1) | c-weighted | pin K token-s per "
                  "returning request | net at c = 1 | net at measured c | oracle on the same readers")
            for lo, hi in WINDOWS:
                w_all = [r for r in ret if lo * 60 <= r["adm"] < hi * 60]
                w = [r for r in have if lo * 60 <= r["adm"] < hi * 60]
                if not w or not w_all:
                    continue
                n = len(w_all)
                for lam in LAMS:
                    gain = gain_c = pin_tok = orc = 0.0
                    dec = {"now": 0, "wait": 0, "arrival": 0}
                    for r in w:
                        h = entry[r["rid"]]
                        x = r["adm"] - r["D"]
                        g = np.array([r["wait_s"]])
                        L = tokens.get(r["rid"], 0) or (r.get("ideal_cached") or 0)
                        dist = P.GapDistribution.from_quantiles(p10=h["p10"], p50=h["p50"], p90=h["p90"])
                        sup = dist.support()
                        s = P.read_start(remaining=P.remaining_gaps(support=sup, idle_s=0.0), read_s=x, wait_s=0.0,
                                         rent_per_s=lam * L)
                        release = sup[len(sup) - P.MIN_REMAINING_POINTS] + 5.0
                        st, pin = FV.realized(g, s, x, release=release)
                        dec["arrival" if s is None else ("now" if s == 0.0 else "wait")] += 1
                        gain += x - st
                        gain_c += (x - st) * idle.get(r["rid"], 0.0)
                        pin_tok += pin * L
                        orc += min(r["wait_s"], x)
                    m = len(w)
                    print(f"  {lo:3d}-{hi:<3d} | {lam:7.0e} | {m:5d} | {100 * dec['now'] / m:5.1f}/{100 * dec['wait'] / m:5.1f}/"
                          f"{100 * dec['arrival'] / m:5.1f} | {gain / n:6.3f} | {gain_c / n:6.3f} | {pin_tok / n / 1e3:8.1f} | "
                          f"{(gain - lam * pin_tok) / n:+7.3f} | {(gain_c - lam * pin_tok) / n:+7.3f} | {orc / n:6.3f}")


if __name__ == "__main__":
    main()
