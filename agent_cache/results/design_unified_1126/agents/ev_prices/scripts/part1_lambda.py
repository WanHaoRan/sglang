"""Part 1, lambda: the draft's host price as the index of the marginal holder (DESIGN.md sections 4.1, 6).

  lambda(t) = (K / L) x h(a*),  a* = the idle age at which a prefix is given up (host turnover),
  K / L = c x (exposed read stall per token read) for a prefix the SSD holds        (K_read, kappa term left out)

Inputs per window of minutes (from part1_prices.per_minute and readers.annotate):
  tau_host   host-only tokens / host eviction rate (census in campaign 34; else pool - mirror estimate)
  a50        idle age at which half of the returning requests needed an SSD read (logistic-free: the age bin
             where the share reading crosses 0.5), from the requests of the window
  h(a)       hazard of return at idle age a, from the run's own gaps: returns with gap in [a, 1.25a) over the
             idle time at risk in that bin; 'live' leaves ended sessions out, 'all' keeps them at risk until
             the run's end (they hold memory until evicted)
  stall/tok  sum of readers' stall over the tokens their reads loaded
Compared with the engine's logged price (ec_price lines).
usage: part1_lambda.py camp:arm [...]"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import part1_prices  # noqa: E402

EDGES = [0.5 * 1.25 ** i for i in range(40)]  # 0.5 s .. ~3,000 s


def stream_ends(camp):
    """Last index of every stream that ended in the campaign's ec_full run (its client logs the end hint);
    streams are the same in every arm (same trees)."""
    ends = {}
    for r in lib.requests(camp, "ec_full"):
        h = r.get("hint") or {}
        if ("true_gap_s" in h and h["true_gap_s"] is None) or h.get("reply_key") == "_end":
            ends[(r["tree"], r["stream"])] = r["k"]
    return ends


def gaps(rows, ends, t_end=7200.0):
    """Idle periods per stream: (gap, returned?) with gap = next arrival - this finish (server clock)."""
    streams = {}
    for r in rows:
        streams.setdefault((r["play"], r["stream"]), []).append(r)
    ret, ended, cens = [], [], []
    for rs in streams.values():
        rs.sort(key=lambda r: r["k"])
        for a, b in zip(rs, rs[1:]):
            if a["fin"] is not None and b["arr"] is not None:
                ret.append((max(0.0, b["arr"] - a["fin"]), b["gap"]))
        last = rs[-1]
        if last["fin"] is None:
            continue
        is_end = ends.get((last["tree"], last["stream"])) == last["k"]
        (ended if is_end else cens).append(max(0.0, t_end - last["fin"]))
    return ret, ended, cens


def hazard(ret, ended, cens, include_ended):
    """Piecewise-constant hazard on EDGES."""
    out = []
    risk_pop = [g for g, _ in ret] + cens + (ended if include_ended else [])
    for lo, hi in zip(EDGES, EDGES[1:]):
        events = sum(1 for g, _ in ret if lo <= g < hi)
        risk = sum(max(0.0, min(g, hi) - lo) for g in risk_pop if g > lo)
        out.append((lo, hi, events, risk, events / risk if risk > 0 else float("nan")))
    return out


def h_at(hz, a):
    for lo, hi, ev, risk, h in hz:
        if lo <= a < hi:
            return h
    return float("nan")


def a50_of(rows, lo_s, hi_s):
    """Share of returning requests whose history came (at least half) from the SSD or was lost, by idle age."""
    streams = {}
    for r in rows:
        streams.setdefault((r["play"], r["stream"]), {})[r["k"]] = r
    bins = {}
    for rs in streams.values():
        for k, r in rs.items():
            p = rs.get(k - 1)
            if p is None or p["fin"] is None or r["arr"] is None or not lo_s <= r["arr"] < hi_s or r["ideal"] < 2048:
                continue
            age = max(0.01, r["arr"] - p["fin"])
            cd = r["cd"] or {}
            out_of_mem = ((cd.get("storage") or 0) + max(0, r["ideal"] - r["cached"])) >= 0.5 * r["ideal"]
            b = int(math.floor(math.log(age / 0.5) / math.log(2))) if age >= 0.5 else -1
            n, s = bins.get(b, (0, 0))
            bins[b] = (n + 1, s + (1 if out_of_mem else 0))
    series = [(0.5 * 2 ** b, n, s / n) for b, (n, s) in sorted(bins.items()) if b >= 0 and n >= 5]
    a50 = float("inf")
    for (a0, n0, s0), (a1, n1, s1) in zip(series, series[1:]):
        if s0 < 0.5 <= s1:
            a50 = a0 * 2 ** 0.5 * (a1 / a0) ** ((0.5 - s0) / (s1 - s0))  # geometric interpolation between bin centres
            break
    return a50, series


def main():
    wins = [(0, 8), (8, 18), (18, 28), (28, 40), (40, 60), (60, 75), (75, 90), (90, 110)]
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        out, st, rows = part1_prices.per_minute(camp, arm)
        ret, ended, cens = gaps(rows, stream_ends(camp))
        hz_live = hazard(ret, ended, cens, False)
        hz_all = hazard(ret, ended, cens, True)
        print(f"== {spec}: {len(ret)} returns, {len(ended)} ended sessions, {len(cens)} sessions cut by the run's end")
        print("  hazard of return by idle age (1/s): age s | live sessions | with ended sessions at risk | returns in bin")
        for (lo, hi, ev, risk, h), (_, _, _, _, h2) in zip(hz_live, hz_all):
            if lo in (EDGES[2], EDGES[5], EDGES[8], EDGES[11], EDGES[14], EDGES[17], EDGES[19], EDGES[21], EDGES[23], EDGES[25],
                      EDGES[27], EDGES[29], EDGES[31], EDGES[33]):
                print(f"     {lo:7.1f}-{hi:7.1f} | {h:.5f} | {h2:.5f} | {ev}")
        price = [o["ec_price"] for o in out if o["ec_price"] == o["ec_price"]]
        print(f"  engine price (ec_price lines): {('min %.3g max %.3g' % (min(price), max(price))) if price else 'none logged'}")
        print("  minutes | c_adm | tau_host s | a50 s | h_live(tau) h_all(tau) 1/s | readers stall/tok s | K/L = c x stall/tok |"
              " lambda (live) | lambda (all) | lambda at c=1 (live)")
        for lo, hi in wins:
            g = lambda k: part1_prices.window(out, lo, hi, k)  # noqa: E731
            n_adm = sum(o["n_adm"] for o in out[lo:hi])
            c = (sum(o["c_adm"] * o["n_adm"] for o in out[lo:hi] if o["c_adm"] == o["c_adm"]) / n_adm) if n_adm else float("nan")
            ev, ho = g("evict_tok_s"), g("host_only")
            if ho != ho:
                ho = lib.POOL_HOST - lib.POOL_GPU * g("use")
            tau = ho / ev if ev and ev > 0 else float("inf")
            a50, _ = a50_of(rows, lo * 60.0, hi * 60.0)
            w = [r for r in rows if r["k"] > 0 and r["n_reads"] > 0 and lo * 60 <= r["adm"] < hi * 60]
            tok = sum(r["read_tok"] for r in w)
            spt = sum(r["stall"] for r in w) / tok if tok else float("nan")
            if tau == float("inf"):
                print(f"  {lo:3d}-{hi:3d} | {c:5.2f} | no host eviction: lambda = 0 (room free)")
                continue
            hl, ha = h_at(hz_live, tau), h_at(hz_all, tau)
            print(f"  {lo:3d}-{hi:3d} | {c:5.2f} | {tau:8.1f} | {a50:6.1f} | {hl:.5f} {ha:.5f} | {len(w):5d} {spt:.3e} | {c*spt:.3e} | "
                  f"{c*spt*hl:.2e} | {c*spt*ha:.2e} | {spt*hl:.2e}")


if __name__ == "__main__":
    main()
