"""Tabulate fetch_value_<pop>_<mode>.json on the task's grid: r in {2, 4, 6}, W in {0, 2, 5, 15}, c in {1, 0.3, 0},
lambda in {0, 1e-6, 2e-5, 1e-4}; seconds of TTFT per returning request whose history is out of memory.

  python fetch_value_report.py A|B [censored|replay]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import fetch_value as FV  # noqa: E402


def main():
    pop = sys.argv[1] if len(sys.argv) > 1 else "A"
    mode = sys.argv[2] if len(sys.argv) > 2 else "censored"
    d = json.load(open(os.path.join(C.OUT, f"fetch_value_{pop}_{mode}.json")))
    acc = d["acc"]
    n_all = d["n_all"]

    def cell(a, x, rpt):
        return acc.get(f"{a:g}|{x:g}|{rpt:g}")

    print(f"population {pop}, mode {mode} (capped gaps: "
          f"{'returns after the horizon' if mode == 'censored' else 'a return at 1,200 s, as replayed'}): {n_all} "
          f"returning requests; entries {d['entry_kinds']} covering {d['n_by_kind']} requests")
    print("x = (r - W)+. gain = seconds of the returning request's own TTFT saved against reading at arrival "
          "(stall x); pin = K token-seconds held before the turn is due; net = c x gain - lambda x pin token-seconds "
          "(the draft's priced objective, in seconds of summed TTFT). Per request with gap > a.")
    print("fit = the engine's read_start on the fitted entry; emp = the key's best start chosen on half of its gaps and "
          "scored on the other half; end = the better of read-now and arrival, chosen and scored the same way; "
          "timing = emp net - end net (what a timed start adds to the keep-or-not decision).")
    for a in FV.AGES:
        base = cell(a, 1.0, 0.0)
        if base is None:
            continue
        n = base["n"]
        print(f"\n=== eviction age a = {a:g} s: {int(n)} requests ({100 * n / n_all:.1f} % of returning requests) return later than a ===")
        print("  r  W  x | oracle gain | c  lambda | fit: now/wait/arrival % | fit gain  pin  net | emp gain  pin  net | "
              "end net | timing | read now: gain pin net | jit(p10): gain pin net")
        for r in FV.RS:
            for W in FV.WS:
                x = max(0.0, r - W)
                if x <= 0:
                    print(f"  {r:g} {W:>2g} 0 | 0 (the queue hides the read: nothing is read before arrival)")
                    continue
                for c in FV.CS:
                    for lam in FV.LAMS:
                        if c == 0:
                            if lam == FV.LAMS[0]:
                                v = cell(a, x, 0.0)
                                print(f"  {r:g} {W:>2g} {x:g} | {v['orc_gain'] / n:6.3f} | c=0, any lambda: stall has no weight; "
                                      f"net 0 for arrival, an early read only costs lambda x pin")
                            continue
                        rpt = lam / c
                        v = cell(a, x, rpt)
                        g_f, p_f = v["fit_gain"] / n, v["fit_pin_tok"] / n
                        g_e, p_e = v["cv_emp_gain"] / n, v["cv_emp_pin_tok"] / n
                        g_n, p_n = v["now_gain"] / n, v["now_pin_tok"] / n
                        g_j, p_j = v["jit_gain"] / n, v["jit_pin_tok"] / n
                        net = lambda g, p: c * g - lam * p  # noqa: E731
                        end_net = c * (v["cost_arr"] - v["cv_end_cost"]) / n
                        emp_net = c * (v["cost_arr"] - v["cv_emp_cost"]) / n
                        print(f"  {r:g} {W:>2g} {x:g} | {v['orc_gain'] / n:6.3f} | {c:3.1f} {lam:7.0e} | "
                              f"{100 * v['fit_now'] / n:5.1f}/{100 * v['fit_wait'] / n:5.1f}/{100 * (1 - (v['fit_now'] + v['fit_wait']) / n):5.1f} | "
                              f"{g_f:6.3f} {p_f / 1e3:8.1f} {net(g_f, p_f):+7.3f} | {g_e:6.3f} {p_e / 1e3:8.1f} {emp_net:+7.3f} | "
                              f"{end_net:+7.3f} | {emp_net - end_net:+7.4f} | {g_n:6.3f} {p_n / 1e3:8.1f} {net(g_n, p_n):+7.3f} | "
                              f"{g_j:6.3f} {p_j / 1e3:8.1f} {net(g_j, p_j):+7.3f}")
    print("\n=== Where the optimum sits and what timing adds (c = 1; s per request with gap > a) ===")
    print("  a  x  lambda | in sample: key's optimum now / interior / arrival % of requests | net: best start, best end, "
          "timing | out of sample: now / interior / arrival % | net: best start, best end, timing | fitted rule net | oracle")
    for a in FV.AGES:
        for x in FV.XS:
            for lam in FV.LAMS:
                v = cell(a, x, lam)
                if v is None:
                    continue
                n = v["n"]
                print(f"  {a:>3g} {x:g} {lam:7.0e} | {100 * v['emp_now'] / n:5.1f} / {100 * v['emp_wait'] / n:5.1f} / "
                      f"{100 * (1 - (v['emp_now'] + v['emp_wait']) / n):5.1f} | {(v['cost_arr'] - v['cost_emp']) / n:6.3f} "
                      f"{(v['cost_arr'] - v['cost_end']) / n:6.3f} {(v['cost_end'] - v['cost_emp']) / n:+7.4f} | "
                      f"{100 * v['cv_now'] / n:5.1f} / {100 * v['cv_wait'] / n:5.1f} / {100 * (1 - (v['cv_now'] + v['cv_wait']) / n):5.1f} | "
                      f"{(v['cost_arr'] - v['cv_emp_cost']) / n:+6.3f} {(v['cost_arr'] - v['cv_end_cost']) / n:+6.3f} "
                      f"{(v['cv_end_cost'] - v['cv_emp_cost']) / n:+7.4f} | {(v['cost_arr'] - v['cost_fit']) / n:+7.3f} | {v['orc_gain'] / n:6.3f}")
    pk = d["per_key"]
    print("\n=== Per key (the 30 most frequent), prefix out of memory at the turn's end (a = 0), c = 1 ===")
    print("  start: s after the turn's end at which the read starts (0 = now, i.e. keep; arrival = no early read). "
          "fit = engine's read_start on the entry; emp = best start on the key's own gaps (in sample). gain = s of own TTFT "
          "saved per request of the key; oracle = exact gap.")
    print("  key | n | share % | p10/p50/p90 | L tokens | x=2 lambda=1e-6: fit start, gain; emp start, gain; oracle | "
          "x=2 lambda=2e-5: same | x=4 lambda=1e-6: same | x=4 lambda=2e-5: same")
    top = sorted(pk.items(), key=lambda kv: -kv[1]["n"])[:30]

    def st(v):
        return "arrival" if v is None else f"{v:.1f}"

    for ent, rec in top:
        parts = []
        for x_ in (2.0, 4.0):
            for lam_ in (1e-6, 2e-5):
                cv = rec["cells"].get(f"0|{x_:g}|{lam_:g}")
                if cv is None:
                    parts.append("na")
                    continue
                parts.append(f"{st(cv['s_fit'])}, {cv['gain_fit']:.2f}; {st(cv['s_emp'])}, {cv['gain_emp']:.2f}; {cv['orc']:.2f}")
        print(f"  {ent} | {rec['n']} | {100 * rec['n'] / n_all:.2f} | {rec['p10']:.2f}/{rec['p50']:.2f}/{rec['p90']:.2f} | {rec['L']:.0f} | "
              + " | ".join(parts))
    for (a_sel, x_sel, lam_sel) in ((0.0, 4.0, 2e-5), (0.0, 4.0, 1e-6), (10.0, 4.0, 1e-6)):
        rows = []
        for ent, rec in pk.items():
            for key, cv in rec["cells"].items():
                a, x, rpt = (float(t) for t in key.split("|"))
                if a == a_sel and x == x_sel and abs(rpt - lam_sel) < 1e-12:
                    rows.append((cv["cv_end_cost"] - cv["cv_emp_cost"], ent, rec, cv))
        rows.sort(key=lambda t: -t[0])
        print(f"\n=== Keys where a timed start beats both ends most, out of sample (a = {a_sel:g}, x = {x_sel:g} s, c = 1, lambda = {lam_sel:g}) ===")
        print("  key | n with gap > a | share of returning requests % | p10/p50/p90 | L | start on all gaps (s after a) | timing gain out of sample, "
              "in sample (s per request of the key) | stall saved, pin s (in sample) | read now: gain, pin s | oracle | fitted start, its gain")
        for dlt, ent, rec, cv in rows[:14]:
            print(f"  {ent} | {cv['n']} | {100 * cv['n'] / n_all:.2f} | {rec['p10']:.2f}/{rec['p50']:.2f}/{rec['p90']:.2f} | {rec['L']:.0f} | "
                  f"{'arrival' if cv['s_emp'] is None else '%.2f' % cv['s_emp']} | {dlt:+.3f}, {cv['cost_end'] - cv['cost_emp']:+.3f} | "
                  f"{cv['gain_emp']:.2f}, {cv['pin_emp']:.2f} | {cv['gain_now']:.2f}, {cv['pin_now']:.2f} | {cv['orc']:.2f} | "
                  f"{'arrival' if cv['s_fit'] is None else '%.2f' % cv['s_fit']}, {cv['gain_fit']:.2f}")
        pos = [t for t in rows if t[0] > 0.01]
        print(f"  keys with n >= 100: {len(rows)}; out-of-sample timing gain summed: {sum(t[0] * t[3]['n'] for t in rows) / n_all:+.4f} s per "
              f"returning request; keys with more than 0.01 s: {len(pos)}, {100 * sum(t[3]['n'] for t in pos) / n_all:.2f} % of returning "
              f"requests, {sum(t[0] * t[3]['n'] for t in pos) / n_all:+.4f} s per returning request from them")


if __name__ == "__main__":
    main()
