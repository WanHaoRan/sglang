"""Tabulate hint_shapes_<pop>.json: classes, top keys, sharpest keys, shares by hazard shape, session ends.

  python hint_shapes_report.py A|B
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402


def f(x, nd=2):
    return C.fmt(x, nd)


def hz_row(r, field):
    return " ".join(f"{(v if v is not None else float('nan')):7.4f}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "     na" for v in r[field])


def main():
    pop = sys.argv[1] if len(sys.argv) > 1 else "A"
    d = json.load(open(os.path.join(C.OUT, f"hint_shapes_{pop}.json")))
    small = json.load(open(os.path.join(C.OUT, f"hint_shapes_{pop}_small.json")))
    recs = d["records"]
    bins = d["bins"]
    print(f"population {pop}: {d['trees']} trees, {d['gaps']} gaps (returning requests), {d['ends_total']} stream ends; "
          f"keys: {d['small_keys']['n_keys_total']} total, {sum(1 for r in recs if r['kind'] == 'key')} with n >= 100 "
          f"(covering {100 * sum(r['n'] for r in recs if r['kind'] == 'key') / d['gaps']:.1f} % of gaps)")
    print("hazard bins (s):", " ".join(f"{a}-{b}" for a, b in bins))
    print()
    print("== classes ==")
    for r in recs:
        if r["kind"] != "class":
            continue
        print(f"{r['key']}: n {r['n']} ({100 * r['share']:.1f} %), ends {r['n_end']} (P(end) {f(r['p_end'], 3)}), "
              f"p10/p50/p90 {f(r['p10'])}/{f(r['p50'])}/{f(r['p90'])} s, p90/p10 {f(r['sharp'], 1)}, "
              f"sigma low/high {f(r['sigma_low'])}/{f(r['sigma_high'])}, capped at 1200 s {100 * r['share_capped']:.2f} %, "
              f"history p50 {r['hist_p50']:.0f} tokens")
        print("   empirical hazard /s      :", hz_row(r, "emp_hz"))
        print("   with ends as no return   :", hz_row(r, "emp_hz_end"))
        print("   fitted (2-piece lognorm) :", hz_row(r, "fit_hz"), f"| fitted hazard peaks at {f(r['fit_peak_s'])} s")
        print(f"   empirical trend 0-10 s: {r['emp_trend_0-10']} (slope {f(r['emp_slope_0-10'])}, z {f(r['emp_z_0-10'], 1)}); "
              f"10-60 s: {r['emp_trend_10-60']} ({f(r['emp_slope_10-60'])}, z {f(r['emp_z_10-60'], 1)}); "
              f"60+ s: {r['emp_trend_60+']} ({f(r['emp_slope_60+'])}, z {f(r['emp_z_60+'], 1)}); "
              f"significant rises between bins: {r['emp_rises']}")
        print("   F(x|s) empirical: " + "; ".join(
            f"x={x:g}: at 0 {f(r[f'F{x:g}_emp0'], 3)}, max {f(r[f'F{x:g}_emp_max'], 3)} at s={r[f'F{x:g}_emp_smax']:g}"
            for x in (1.0, 2.0, 4.0, 6.0)))
        print("   F(x|s) fitted   : " + "; ".join(
            f"x={x:g}: at 0 {f(r[f'F{x:g}_fit0'], 3)}, max {f(r[f'F{x:g}_fit_max'], 3)} at s={f(r[f'F{x:g}_fit_smax'])}"
            for x in (1.0, 2.0, 4.0, 6.0)))
    keys = [r for r in recs if r["kind"] == "key"]
    tot = d["gaps"]
    covered = sum(r["n"] for r in keys)
    print()
    print(f"== keys with n >= 100: {len(keys)} keys, {covered} gaps ({100 * covered / tot:.1f} % of returning requests) ==")
    print("key | n | share % | P(end) | p10 p50 p90 | p90/p10 | sig_lo sig_hi | fit peak s | emp peak bin | trend 0-10/10-60/60+ | "
          "F4(0)->max@s emp | hist p50")
    for r in keys[:60]:
        pb = r["emp_peak_bin"]
        print(f"{r['key']} | {r['n']} | {100 * r['share']:.2f} | {f(r['p_end'], 3)} | {f(r['p10'])} {f(r['p50'])} {f(r['p90'])} | "
              f"{f(r['sharp'], 1)} | {f(r['sigma_low'])} {f(r['sigma_high'])} | {f(r['fit_peak_s'])} | "
              f"{'na' if pb is None else '%d-%d' % tuple(bins[pb])} | {r['emp_trend_0-10']}/{r['emp_trend_10-60']}/{r['emp_trend_60+']} | "
              f"{f(r['F4_emp0'], 2)}->{f(r['F4_emp_max'], 2)}@{r['F4_emp_smax']:g} | {r['hist_p50']:.0f}")

    def share(pred):
        return 100 * sum(r["n"] for r in keys if pred(r)) / covered

    print()
    print("== shares of returning requests (among keys with n >= 100), by shape of their key's hazard ==")
    for thr in (0.5, 1.0, 2.0, 4.0, 6.0, 10.0):
        print(f"  fitted hazard peaks after {thr:g} s: {share(lambda r: r['fit_peak_s'] > thr):.1f} %"
              f"   ({sum(1 for r in keys if r['fit_peak_s'] > thr)} keys)")
    print(f"  fitted hazard peak inside 1-6 s: {share(lambda r: 1.0 < r['fit_peak_s'] <= 6.0):.1f} %")
    for i in range(len(bins)):
        print(f"  empirical hazard highest in bin {bins[i][0]}-{bins[i][1]} s: {share(lambda r: r['emp_peak_bin'] == i):.1f} %"
              f"   ({sum(1 for r in keys if r['emp_peak_bin'] == i)} keys)")
    print(f"  empirical hazard peak in a bin inside 1-6 s (1-2, 2-4, 4-6): {share(lambda r: r['emp_peak_bin'] in (1, 2, 3)):.1f} %")
    print(f"  empirical hazard peak at or beyond 1 s: {share(lambda r: r['emp_peak_bin'] is not None and r['emp_peak_bin'] >= 1):.1f} %")
    print(f"  a significant rise (z > 2) between consecutive bins somewhere in 0-10 s: {share(lambda r: len(r['emp_rises']) > 0):.1f} %"
          f"   ({sum(1 for r in keys if r['emp_rises'])} keys)")
    print(f"  a significant rise whose later bin starts at 1 s or later (1-2 > 2-4, 2-4 > 4-6, 4-6 > 6-10): "
          f"{share(lambda r: any(not x[0].startswith('0-1>') for x in r['emp_rises'])):.1f} %"
          f"   ({sum(1 for r in keys if any(not x[0].startswith('0-1>') for x in r['emp_rises']))} keys)")
    for rng in ("0-10", "10-60", "60+"):
        row = {lab: share(lambda r: r[f"emp_trend_{rng}"] == lab) for lab in ("rising", "flat", "falling", "na")}
        print(f"  empirical trend over {rng} s: " + ", ".join(f"{k} {v:.1f} %" for k, v in row.items()))
    for x in (1.0, 2.0, 4.0, 6.0):
        k0, kmax, ks = f"F{x:g}_emp0", f"F{x:g}_emp_max", f"F{x:g}_emp_smax"
        s_any = share(lambda r: r[ks] > 0 and r[kmax] > r[k0] + 0.05)
        s_big = share(lambda r: r[ks] > 0 and r[kmax] > r[k0] + 0.15)
        s_half = share(lambda r: r[ks] > 0 and r[kmax] >= 0.5 and r[kmax] > r[k0] + 0.05)
        print(f"  F({x:g}|s) empirical rises above F({x:g}|0) by > 0.05: {s_any:.1f} %; by > 0.15: {s_big:.1f} %; "
              f"and reaches >= 0.5: {s_half:.1f} %")
        fk0, fkmax, fks = f"F{x:g}_fit0", f"F{x:g}_fit_max", f"F{x:g}_fit_smax"
        print(f"  F({x:g}|s) fitted rises above F({x:g}|0) by > 0.05: {share(lambda r: r[fkmax] > r[fk0] + 0.05):.1f} %; "
              f"by > 0.15: {share(lambda r: r[fkmax] > r[fk0] + 0.15):.1f} %")
    print()
    print("== sharpest keys (p90/p10), n >= 100 ==")
    for r in sorted(keys, key=lambda r: r["sharp"])[:25]:
        print(f"  {r['key']}: n {r['n']} ({100 * r['share']:.2f} %), p10/p50/p90 {f(r['p10'])}/{f(r['p50'])}/{f(r['p90'])}, "
              f"p90/p10 {f(r['sharp'], 1)}, sigma {f(r['sigma_low'])}/{f(r['sigma_high'])}, fit peak {f(r['fit_peak_s'])} s, "
              f"F4: {f(r['F4_emp0'], 2)}->{f(r['F4_emp_max'], 2)}@{r['F4_emp_smax']:g}, P(end) {f(r['p_end'], 3)}")
    qs = sorted(r["sharp"] for r in keys)
    w = sorted((r["sharp"], r["n"]) for r in keys)
    def wq(p):
        acc = 0
        for s, n in w:
            acc += n
            if acc >= p * covered:
                return s
        return w[-1][0]
    print(f"  p90/p10 over keys (unweighted): min {f(qs[0], 1)}, p10 {f(C.q(qs, .1), 1)}, p50 {f(C.q(qs, .5), 1)}, p90 {f(C.q(qs, .9), 1)}; "
          f"weighted by requests: p10 {f(wq(.1), 1)}, p50 {f(wq(.5), 1)}, p90 {f(wq(.9), 1)}")
    for thr in (3, 5, 10, 20, 50):
        print(f"  keys with p90/p10 < {thr}: {sum(1 for r in keys if r['sharp'] < thr)} keys, {share(lambda r: r['sharp'] < thr):.2f} % of requests")
    sm = sorted(small, key=lambda r: r["sharp"])
    print(f"  small keys (5 <= n < 100): {len(sm)} keys, {sum(r['n'] for r in sm)} gaps; with p90/p10 < 3: "
          f"{sum(1 for r in sm if r['sharp'] < 3)} keys, {sum(r['n'] for r in sm if r['sharp'] < 3)} gaps; < 5: "
          f"{sum(1 for r in sm if r['sharp'] < 5)} keys, {sum(r['n'] for r in sm if r['sharp'] < 5)} gaps")
    print("  sharpest small keys:", [(r["key"], r["n"], round(r["p10"], 2), round(r["p50"], 2), round(r["p90"], 2)) for r in sm[:12]])
    print()
    print("== session ends by key (a stream's last request), n + ends >= 100 ==")
    ke = sorted(keys, key=lambda r: -r["p_end"])
    print("  highest:", [(r["key"], r["n"], r["n_end"], round(r["p_end"], 3)) for r in ke[:15]])
    print("  lowest :", [(r["key"], r["n"], r["n_end"], round(r["p_end"], 3)) for r in ke[-8:]])
    pe = sorted(r["p_end"] for r in keys)
    print(f"  P(end) over keys: p10 {f(C.q(pe, .1), 4)}, p50 {f(C.q(pe, .5), 4)}, p90 {f(C.q(pe, .9), 4)}, max {f(pe[-1], 4)}")


if __name__ == "__main__":
    main()
