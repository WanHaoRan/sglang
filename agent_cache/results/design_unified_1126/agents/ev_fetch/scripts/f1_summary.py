"""F1 in one table per arm: by regime, today's 2 s margin against the margin the rule q = lambda L / (c + lambda L)
would pick, in seconds per read, seconds per returning request, and tokens of host memory held on average.

  python f1_summary.py <label> [<label> ...]      (reads out/f1_landing_<label>.json and extracts/<label>.requests.json)
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402

WINDOWS = ((0, 30), (30, 70), (70, 90), (90, 110))
LAMS = (1e-7, 1e-6, 2e-5, 1e-4)
HOST_POOL = 1627648


def nearest(res, m):
    key = min(res, key=lambda k: abs(float(k) - m))
    return float(key), res[key]


def main():
    for label in sys.argv[1:]:
        d = json.load(open(os.path.join(C.OUT, f"f1_landing_{label}.json")))
        reqs = json.load(open(os.path.join(C.EXTRACTS, f"{label}.requests.json")))["rows"]
        ret = [r for r in reqs if r.get("k") is not None and r["k"] > 0]
        print(f"==== {label} ====")
        print("regime | reads | returning requests | today (2 s): late %, stall s/read, pin s/read, pinned tokens on average (% of the host pool) | "
              "measured c | margin at q = 0.05 / 0.2 / 0.5 / 0.8 / 0.95 | release at park: d stall s/read, pinned tokens")
        for lo, hi in WINDOWS:
            g = d.get(f"minutes {lo}-{hi}")
            if g is None:
                continue
            res = g["res"]
            n = g["n"]
            n_ret = sum(1 for r in ret if lo * 60 <= r["adm"] < hi * 60)
            secs = (hi - lo) * 60.0
            b = res["2"]
            far = res["20"]
            mq = g["mq"]
            print(f"  minutes {lo}-{hi} | {n} | {n_ret} | {100 * b[3]:.1f} %, {b[0]:.2f}, {b[1]:.2f}, {b[2] * n / secs / 1e3:.1f} K "
                  f"({100 * b[2] * n / secs / HOST_POOL:.2f} %) | {g['c']:.3f} | " +
                  " / ".join(f"{mq[k]:+.1f}" for k in ("0.05", "0.2", "0.5", "0.8", "0.95")) +
                  f" | {far[0] - b[0]:+.2f}, {far[2] * n / secs / 1e3:.1f} K")
            print("      lambda | q with the measured c | margin | d stall s per read | d mean TTFT of the regime's returning requests: "
                  "readers' own (c = 1), and c-weighted | pinned tokens on average: today -> rule (change, % of the host pool)")
            for lam in LAMS:
                c = max(g["c"], 0.0)
                L = g["L"]
                q = lam * L / (c + lam * L) if c + lam * L > 0 else float("nan")
                # the margin that lands the read at the q-quantile of D: minus the q-quantile of the error at release
                eq = g["e_quantiles"]
                pos = min(100.0, max(0.0, q * 100.0))
                lo_i = int(math.floor(pos))
                hi_i = min(100, lo_i + 1)
                m = -(eq[lo_i] + (eq[hi_i] - eq[lo_i]) * (pos - lo_i))
                key, v = nearest(res, max(-12.0, min(20.0, m)))
                ds = v[0] - b[0]
                print(f"      {lam:7.0e} | {q:5.3f} | {m:+6.2f} | {ds:+6.2f} | {ds * n / max(1, n_ret):+6.3f}, {c * ds * n / max(1, n_ret):+7.4f} | "
                      f"{b[2] * n / secs / 1e3:5.1f} K -> {v[2] * n / secs / 1e3:5.1f} K ({(v[2] - b[2]) * n / secs / 1e3:+5.1f} K, "
                      f"{100 * (v[2] - b[2]) * n / secs / HOST_POOL:+.2f} %)")


if __name__ == "__main__":
    main()
