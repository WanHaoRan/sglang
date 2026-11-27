"""Part 3 check: how much does a long prefill delay the turns that arrive while it runs, at light load?

Natural experiment: a "big" prefill is a request admitted with >= 12,000 new tokens while no ready turn waited.
Its prefill occupies [adm, adm + T_fit(new, prefix)] (per-run idle fit). Every other request that ARRIVES in that
span, has no host/SSD part and < 4,000 new tokens is a "bystander". Its excess = TTFT - T_fit(own new, prefix).
Compared with: the serial prediction (time left of the big prefill at its arrival), the chunk prediction
(min(time left, 0.375 s)), and the excess of the same kind of request arriving when no big prefill runs
(the baseline). Light load only: arrivals before `t_hi` minutes with GPU pool use < 0.90.
usage: part3b_externality_check.py camp:arm:t_hi [...]"""
import bisect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

FIT = json.load(open(f"{lib.OUT}/prefill_fit.json"))


def t_fit(f, n, p):
    return f["a_s"] + f["b_s_per_tok"] * n + f["c_s_per_tok_Mtok"] * n * (p + n / 2) / 1e6


def main():
    tot = {"by": [], "base": []}
    for spec in sys.argv[1:]:
        camp, arm, t_hi = spec.split(":")
        t_hi = float(t_hi) * 60
        f = FIT[f"{camp}:{arm}"]["idle"]
        st = qstate.build(camp, arm)
        rows = readers.annotate(st)
        rows = [r for r in rows if r["ttft"] is not None and r["arr"] < t_hi]
        big = []
        for r in rows:
            new = r["in_len"] - r["cached_srv"]
            cd = r["cd"] or {}
            if new >= 12000 and r["queue"] < 0.05 and not (cd.get("host") or 0) and not (cd.get("storage") or 0):
                big.append((r["adm"], r["adm"] + t_fit(f, new, r["cached_srv"]) - f["a_s"], new, r["rid"]))
        big.sort()
        starts = [b[0] for b in big]
        by, base = [], []
        for r in rows:
            new = r["in_len"] - r["cached_srv"]
            cd = r["cd"] or {}
            if new >= 4000 or (cd.get("host") or 0) or (cd.get("storage") or 0) or r["reads"] or r["use_arr"] >= 0.90:
                continue
            excess = r["ttft"] - t_fit(f, new, r["cached_srv"])
            i = bisect.bisect_right(starts, r["arr"]) - 1
            hit = None
            for k in (i, i - 1):
                if k >= 0 and big[k][0] < r["arr"] < big[k][1] and big[k][3] != r["rid"]:
                    hit = big[k]
                    break
            if hit is None:
                base.append(excess)
            else:
                left = hit[1] - r["arr"]
                by.append((excess, left, min(left, 0.375), hit[2]))
        tot["by"] += by
        tot["base"] += base
        b0 = lib.mean(base)
        print(f"{spec}: big prefills {len(big)} (mean {lib.mean(b[2] for b in big)/1e3:.1f} K new tokens, {lib.mean(b[1]-b[0] for b in big):.2f} s); "
              f"bystanders {len(by)}; baseline requests {len(base)} with excess {b0:+.3f} s (p50 {lib.quantile(base, .5):+.3f}, p90 {lib.quantile(base, .9):+.3f})")
        if by:
            print(f"    bystanders: measured excess over the baseline {lib.mean(x[0] for x in by) - b0:+.3f} s (p50 {lib.quantile([x[0] for x in by], .5) - b0:+.3f}); "
                  f"serial prediction {lib.mean(x[1] for x in by):.3f} s; chunk prediction {lib.mean(x[2] for x in by):.3f} s")
    by, base = tot["by"], tot["base"]
    b0 = lib.mean(base)
    print(f"ALL: bystanders {len(by)}; measured excess over the baseline {lib.mean(x[0] for x in by) - b0:+.3f} s; serial {lib.mean(x[1] for x in by):.3f} s; "
          f"chunk {lib.mean(x[2] for x in by):.3f} s")
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.8), (0.8, 5)):
        b = [x for x in by if lo <= x[1] < hi]
        if b:
            print(f"   time left of the big prefill {lo:.1f}-{hi:.1f} s: n {len(b):4d}; measured excess over the baseline {lib.mean(x[0] for x in b) - b0:+.3f} s "
                  f"(p50 {lib.quantile([x[0] for x in b], .5) - b0:+.3f}); serial {lib.mean(x[1] for x in b):.3f}; chunk {lib.mean(x[2] for x in b):.3f}")


if __name__ == "__main__":
    main()
