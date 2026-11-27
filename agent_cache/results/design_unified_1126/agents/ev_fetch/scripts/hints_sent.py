"""Part 1 on a run: the shape of the entries the client actually sent (the online table as it stood), per returning
request, for all returning requests and for the ones that waited on their own SSD read.

  python hints_sent.py <label> [<label> ...]
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402


def peak(dist):
    ts = np.exp(np.linspace(math.log(0.01), math.log(1200.0), 600))
    lnS = np.array([math.log(max(dist.survival(float(t)), 1e-300)) for t in ts])
    h = -(lnS[1:] - lnS[:-1]) / (ts[1:] - ts[:-1])
    h = np.where(lnS[1:] >= math.log(0.01), h, -1.0)
    return float(np.sqrt(ts[1:] * ts[:-1])[int(np.argmax(h))])


def main():
    P = C.policy()
    for label in sys.argv[1:]:
        reqs = json.load(open(os.path.join(C.EXTRACTS, f"{label}.requests.json")))["rows"]
        entry = {}
        for r in reqs:
            h = r.get("hint")
            if h and h.get("next_rid"):
                entry[h["next_rid"]] = h
        ret = [r for r in reqs if r.get("k") is not None and r["k"] > 0 and r.get("wait_s") is not None]
        cache = {}
        rows = []
        for r in ret:
            h = entry.get(r["rid"])
            if h is None or h.get("p50") is None or h.get("ended"):
                continue
            k = (h["p10"], h["p50"], h["p90"])
            if k not in cache:
                d = P.GapDistribution.from_quantiles(p10=k[0], p50=k[1], p90=k[2])
                best4 = max(d.chance_within(idle_s=float(s), horizon_s=4.0) for s in np.exp(np.linspace(math.log(0.05), math.log(300.0), 120)))
                cache[k] = (peak(d), d.chance_within(idle_s=0.0, horizon_s=4.0), best4, d.sigma_low, d.sigma_high)
            pk, f0, fmax, sl, sh = cache[k]
            m = h.get("matched") or ""
            level = "*" if m == "*" else ("key" if ("/" in m or m == C.NO_CALL or m == h.get("reply_key")) else "tool")
            rows.append({"reader": r["reader"], "adm": r["adm"], "sharp": h["p90"] / max(h["p10"], 1e-9), "peak": pk,
                         "f0": f0, "fmax": fmax, "level": level, "n": h.get("n"), "gap": r["wait_s"], "p10": h["p10"],
                         "p50": h["p50"], "p90": h["p90"], "nocall": h.get("reply_key") == C.NO_CALL, "sh": sh})
        print(f"==== {label}: {len(ret)} returning requests, {len(rows)} with a distribution entry ====")
        for name, sel in (("all returning requests with an entry", lambda x: True), ("readers", lambda x: x["reader"]),
                          ("readers admitted in minutes 0-30", lambda x: x["reader"] and x["adm"] < 1800),
                          ("readers admitted in minutes 30-70", lambda x: x["reader"] and 1800 <= x["adm"] < 4200),
                          ("readers admitted in minutes 70-110", lambda x: x["reader"] and 4200 <= x["adm"] < 6600)):
            g = [x for x in rows if sel(x)]
            if not g:
                continue
            n = len(g)
            sh = sorted(x["sharp"] for x in g)
            print(f"  {name}: n {n}; entry level key/tool/*: {100 * sum(x['level'] == 'key' for x in g) / n:.0f}/"
                  f"{100 * sum(x['level'] == 'tool' for x in g) / n:.0f}/{100 * sum(x['level'] == '*' for x in g) / n:.0f} %; "
                  f"no-call entries {100 * sum(x['nocall'] for x in g) / n:.1f} %; entry n p50 {C.q([x['n'] for x in g if x['n']], .5)}; "
                  f"true gap p50 {C.q([x['gap'] for x in g], .5):.1f} s")
            print(f"      p90/p10: p10 {C.q(sh, .1):.1f} p50 {C.q(sh, .5):.1f} p90 {C.q(sh, .9):.1f}; under 3: {100 * sum(v < 3 for v in sh) / n:.1f} %, "
                  f"under 5: {100 * sum(v < 5 for v in sh) / n:.1f} %, under 10: {100 * sum(v < 10 for v in sh) / n:.1f} %")
            print(f"      fitted hazard peaks after 1 s: {100 * sum(x['peak'] > 1 for x in g) / n:.1f} %, after 2 s: "
                  f"{100 * sum(x['peak'] > 2 for x in g) / n:.1f} %, after 4 s: {100 * sum(x['peak'] > 4 for x in g) / n:.1f} %, after 6 s: "
                  f"{100 * sum(x['peak'] > 6 for x in g) / n:.1f} %; falling tail (sigma_high >= 1): {100 * sum(x['sh'] >= 1 for x in g) / n:.1f} %")
            print(f"      fitted F(4|s) rises above F(4|0) by more than 0.05: {100 * sum(x['fmax'] > x['f0'] + 0.05 for x in g) / n:.1f} %, "
                  f"by more than 0.15: {100 * sum(x['fmax'] > x['f0'] + 0.15 for x in g) / n:.1f} %")
            ins = sum(1 for x in g if x["p10"] <= x["gap"] <= x["p90"]) / n
            print(f"      true gap inside the entry's p10-p90: {100 * ins:.1f} %; below p10 {100 * sum(x['gap'] < x['p10'] for x in g) / n:.1f} %, "
                  f"above p90 {100 * sum(x['gap'] > x['p90'] for x in g) / n:.1f} %")


if __name__ == "__main__":
    main()
