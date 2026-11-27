"""Part 1 summary: the prices by regime (minute ranges), from the per-minute tables of part1_prices.py.
usage: part1_regimes.py camp:arm [...]"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

REG = [("room free, light load", 0, 8), ("light load, room short", 8, 28), ("GPU saturated, SSD keeps up", 28, 70),
       ("read line filling", 70, 90), ("GPU and SSD saturated", 90, 110)]


def f(v):
    try:
        return float(v)
    except ValueError:
        return float("nan")


def main():
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        rows = list(csv.DictReader(open(f"{lib.OUT}/prices_{camp}_{arm}.csv")))
        print(f"== {spec}")
        print("  regime (minutes)                     |     N   N-B | mu=1+Qr  busy-period mu | c_adm  c_room  c_arr | reads parked, past latest start, read thread busy |"
              " readers %, stall s | host free K, turnover s")
        for name, lo, hi in REG:
            w = rows[lo:hi]

            def m(k):
                xs = [f(r[k]) for r in w if r[k] not in ("", "inf")]
                return sum(xs) / len(xs) if xs else float("nan")

            n_adm = sum(f(r["n_adm"]) for r in w)

            def wa(k):
                return sum(f(r[k]) * f(r["n_adm"]) for r in w if r[k] != "") / n_adm if n_adm else float("nan")

            rd = sum(f(r["readers"]) for r in w)
            stall = sum(f(r["stall"]) * f(r["readers"]) for r in w if r["stall"] != "") / rd if rd else float("nan")
            ev, ho = m("evict_tok_s"), m("host_only")
            tau = ho / ev if ev > 0 and ho == ho else float("nan")
            print(f"  {name:28s} {lo:3d}-{hi:3d} | {m('N'):5.1f} {m('NB'):5.1f} | {m('mu'):6.1f} {m('mu_bp'):12.1f}    | {wa('c_adm'):5.2f} {wa('c_room'):6.2f} {m('c_arr'):6.2f} | "
                  f"{m('parked'):8.2f} {m('late'):8.2f} {m('rd_busy'):8.2f}                      | {100*m('rd_share'):5.1f} {stall:7.2f}   | {m('host_free')/1e3:8.1f} {tau:8.1f}")


if __name__ == "__main__":
    main()
