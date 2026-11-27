"""Part 2: the value of reading before arrival, per key and weighted by key frequency (trace level).

  python fetch_value.py A|B [censored|replay]

A returning request whose history is on the SSD only. x = (r - W)+ is the part of the read the queue does not hide;
a read started s seconds after the prefix left memory lands at s + r, the turn is due at G + W. Cost of the draft
(section 3.1, kappa = 0): c x stall + lambda L x pin seconds, with recourse (a turn that arrives before s reads at
arrival and stalls x). The cost depends on r and W through x only, and on c and lambda through lambda L / c.

Policies, each evaluated on the key's true gaps:
  fit     the engine's read_start on the 64-point support of the two-piece lognormal through the entry's p10/p50/p90
          (next_turn_policy.py, imported by path), conditioned on the idle age as remaining_gaps does;
  emp     the best start for the key's own empirical gaps: in sample (an upper bound that overfits the last gaps of
          a tail) and out of sample (chosen on one random half of the key's gaps, scored on the other, both ways);
  ends    the better of the two ends only: read now (the same as keeping the prefix) or at arrival; in and out of sample;
  jit     just-in-time on the entry's p10 (the jit_p50 arm: start at p10 - x - 1 s, no conditioning on idle time);
  oracle  the exact gap: the read lands at the due time, gain min(G, x), no pin.
Eviction age a: the prefix left memory a seconds after the turn ended, so only requests with G > a need a read and
the earliest start is a (0 = the prefix is out of memory when the turn ends).
The replay caps a gap at 1,200 s, which puts a point mass there (0.6 % of gaps, 7 % of the human ones) that a start at
1,200 - x would hit exactly. Mode "censored" (default) treats a capped gap as a return after the horizon: no early
read saves its stall, a read made early is pinned to the horizon, the oracle still gains x. Mode "replay" keeps the
mass as the client plays it. The fitted rule's pin is given back as the engine does (ec_release: 5 s after fewer
than 4 support points remain); a turn that comes later is counted as reading again at arrival.
Entries as the client's table resolves them: the key's own entry from 5 gaps, else its tool's, else "*".
"""
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402

XS = (1.0, 2.0, 4.0, 6.0)
AGES = (0.0, 2.0, 10.0, 60.0, 300.0)
RS = (2.0, 4.0, 6.0)
WS = (0.0, 2.0, 5.0, 15.0)
CS = (1.0, 0.3, 0.0)
LAMS = (0.0, 1e-6, 2e-5, 1e-4)
JIT_MARGIN_S = 1.0
SEED = 20261126


def rents():
    """Distinct lambda / c over the grid with c > 0."""
    return sorted({lam / c for lam in LAMS for c in CS if c > 0})


def best_start(g, x, rent, n_c=0, horizon=0.0):
    """read_start on an ascending numpy array of remaining gaps, plus n_c requests that return after `horizon`
    (they stall x whatever is done and pin an early read to the horizon): (start or None, cost at it, cost now,
    cost at arrival); costs in seconds per request (stall + rent x pin seconds)."""
    n = len(g) + n_c
    if n == 0 or x <= 0:
        return None, x, x, x
    pre = np.concatenate(([0.0], np.cumsum(g)))
    total = pre[-1]
    cand = np.unique(np.concatenate(([0.0], np.maximum(0.0, g - x))))
    arrived = np.searchsorted(g, cand, side="right")
    late = np.searchsorted(g, cand + x, side="right")
    stall = arrived * x + (late - arrived) * (cand + x) - (pre[late] - pre[arrived]) + n_c * x
    pinned = (total - pre[late]) - (len(g) - late) * (cand + x) + n_c * np.maximum(0.0, horizon - cand - x)
    cost = (stall + rent * pinned) / n
    i = int(np.argmin(cost))
    now = float(cost[0])
    if cost[i] < x:
        return float(cand[i]), float(cost[i]), now, x
    return None, x, now, x


def realized(g, s, x, n_c=0, horizon=0.0, release=None):
    """(mean stall s, mean pin s) on the true remaining gaps g (and n_c returns after the horizon) for a read
    started at s (None: at arrival). `release`: the pin is given back then; a turn that comes after it reads again
    at arrival (stall x)."""
    n = len(g) + n_c
    if s is None or n == 0:
        return x, 0.0
    before = g <= s
    stall = np.where(before, x, np.maximum(0.0, s + x - g))
    pin = np.where(before, 0.0, np.maximum(0.0, g - s - x))
    pin_c = max(0.0, horizon - s - x)
    if release is not None:
        gone = (~before) & (g > release)
        stall = np.where(gone, x, stall)
        pin = np.where(gone, max(0.0, release - s - x), pin)
        pin_c = max(0.0, min(horizon, release) - s - x)
    return float((stall.sum() + n_c * x) / n), float((pin.sum() + n_c * pin_c) / n)


def entry_groups(trees):
    """Requests grouped by the table entry their reply key resolves to: {entry: {src gaps, gaps, keep tokens}}."""
    by_key = collections.defaultdict(list)
    by_tool = collections.defaultdict(list)
    all_gaps = []
    for (t, s, k, key, gap, kind, keep, prompt, system) in C.iter_gaps(trees):
        by_key[key].append((gap, keep))
        by_tool[C.tool_of(key)].append(gap)
        all_gaps.append(gap)
    groups = {}
    for key, v in by_key.items():
        tool = C.tool_of(key)
        if len(v) >= C.MIN_POOLED:
            ent, src = key, [g for g, _ in v]
        elif len(by_tool[tool]) >= C.MIN_POOLED:
            ent, src = "tool:" + tool, by_tool[tool]
        else:
            ent, src = "*", all_gaps
        grp = groups.setdefault(ent, {"src": src, "gaps": [], "keep": []})
        grp["gaps"].extend(g for g, _ in v)
        grp["keep"].extend(kp for _, kp in v)
    return groups, len(all_gaps)


def main():
    pop = sys.argv[1] if len(sys.argv) > 1 else "A"
    mode = sys.argv[2] if len(sys.argv) > 2 else "censored"
    censor = mode == "censored"
    P = C.policy()
    trees = C.load_trace(C.MAX_CONTEXT if pop == "A" else None)
    groups, n_all = entry_groups(trees)
    rent_list = rents()
    rng = np.random.default_rng(SEED)
    # accumulators: (age, x, rent_per_token) -> sums over requests with gap > age
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    per_key = {}
    for ent in sorted(groups):
        grp = groups[ent]
        gaps = np.asarray(grp["gaps"], float)
        fold = rng.integers(0, 2, size=len(gaps))
        L = float(np.median(grp["keep"]))
        p10, p50, p90 = C.entry_quantiles(grp["src"])
        dist = P.GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
        support = dist.support()
        rec = {"n": len(gaps), "L": L, "p10": p10, "p50": p50, "p90": p90, "cells": {}}
        # the engine gives a staged pin back 5 s after fewer than 4 support points remain (NTP_RELEASE_GRACE_S)
        release_age = support[len(support) - P.MIN_REMAINING_POINTS] + 5.0 if len(support) > 1 else None
        for a in AGES:
            m = gaps > a
            n = int(m.sum())
            if n == 0:
                continue
            rem, fd = gaps[m] - a, fold[m]
            horizon = C.GAP_CAP_S - a

            def part(sel):
                r = np.sort(rem[sel])
                nc = int((r >= horizon - 1e-6).sum()) if censor else 0
                return (r[: len(r) - nc] if nc else r), nc

            g, n_c = part(np.ones(n, bool))
            halves = [(part(fd != f), part(fd == f)) for f in (0, 1)]
            remaining = P.remaining_gaps(support=support, idle_s=a)
            release = None if release_age is None else max(0.0, release_age - a)
            for x in XS:
                orc = float((np.minimum(g, x).sum() + n_c * x) / n)
                s_jit = max(0.0, p10 - x - JIT_MARGIN_S - a)
                # the jit arm stages once and never conditions on idle time: past its start time it reads at once;
                # the plan (and its pin) expires max(60 s, p50) after the predicted arrival
                jit_release = max(0.0, p50 + max(60.0, p50) - a)
                st_j, pin_j = realized(g, s_jit, x, n_c, horizon, release=jit_release)
                st_n, pin_n = realized(g, 0.0, x, n_c, horizon)
                for rpt in rent_list:
                    rent = rpt * L
                    s_fit = P.read_start(remaining=remaining, read_s=x, wait_s=0.0, rent_per_s=rent) if remaining else None
                    st_f, pin_f = realized(g, s_fit, x, n_c, horizon, release=release)
                    s_emp, cost_emp, cost_now, cost_arr = best_start(g, x, rent, n_c, horizon)
                    st_e, pin_e = realized(g, s_emp, x, n_c, horizon)
                    A = acc[(a, x, rpt)]
                    # out of sample: choose on one half, score on the other
                    cv_cost = cv_end = 0.0
                    for (tr, n_tr), (te, n_te) in halves:
                        n_te_all = len(te) + n_te
                        if n_te_all == 0:
                            continue
                        if len(tr) + n_tr == 0:
                            s_tr, keep_now = None, False
                        else:
                            s_tr, _, c_now_tr, c_arr_tr = best_start(tr, x, rent, n_tr, horizon)
                            keep_now = c_now_tr < c_arr_tr
                        st, pin = realized(te, s_tr, x, n_te, horizon)
                        A["cv_emp_cost"] += n_te_all * (st + rent * pin)
                        A["cv_emp_gain"] += n_te_all * (x - st)
                        A["cv_emp_pin_tok"] += n_te_all * pin * L
                        A["cv_now"] += n_te_all * (s_tr == 0.0)
                        A["cv_wait"] += n_te_all * (s_tr is not None and s_tr > 0.0)
                        st2, pin2 = realized(te, 0.0 if keep_now else None, x, n_te, horizon)
                        A["cv_end_cost"] += n_te_all * (st2 + rent * pin2)
                        cv_cost += n_te_all * (st + rent * pin)
                        cv_end += n_te_all * (st2 + rent * pin2)
                    A["n"] += n
                    A["orc_gain"] += n * orc
                    A["fit_gain"] += n * (x - st_f); A["fit_pin_tok"] += n * pin_f * L
                    A["fit_now"] += n * (s_fit == 0.0); A["fit_wait"] += n * (s_fit is not None and s_fit > 0.0)
                    A["emp_gain"] += n * (x - st_e); A["emp_pin_tok"] += n * pin_e * L
                    A["emp_now"] += n * (s_emp == 0.0); A["emp_wait"] += n * (s_emp is not None and s_emp > 0.0)
                    A["now_gain"] += n * (x - st_n); A["now_pin_tok"] += n * pin_n * L
                    A["jit_gain"] += n * (x - st_j); A["jit_pin_tok"] += n * pin_j * L
                    # priced costs in units of stall seconds (the objective divided by c): stall + (lambda L / c) pin
                    A["cost_arr"] += n * x
                    A["cost_fit"] += n * (st_f + rent * pin_f)
                    A["cost_emp"] += n * cost_emp
                    A["cost_end"] += n * min(cost_now, cost_arr)
                    A["cost_now"] += n * cost_now
                    A["cost_jit"] += n * (st_j + rent * pin_j)
                    if a in (0.0, 10.0) and n >= 100:
                        rec["cells"][f"{a:g}|{x:g}|{rpt:g}"] = {
                            "n": n, "s_fit": s_fit, "s_emp": s_emp, "gain_fit": x - st_f, "gain_emp": x - st_e,
                            "gain_now": x - st_n, "orc": orc, "pin_emp": pin_e, "pin_fit": pin_f, "pin_now": pin_n,
                            "cost_emp": cost_emp, "cost_end": min(cost_now, cost_arr), "cost_fit": st_f + rent * pin_f,
                            "cv_emp_cost": cv_cost / n, "cv_end_cost": cv_end / n}
        if len(gaps) >= 100:
            per_key[ent] = rec
    out = {"pop": pop, "mode": mode, "n_all": n_all,
           "acc": {f"{a:g}|{x:g}|{rpt:g}": dict(v) for (a, x, rpt), v in acc.items()},
           "per_key": per_key, "entries": len(groups),
           "entry_kinds": {"own": sum(1 for e in groups if not e.startswith("tool:") and e != "*"),
                           "tool": sum(1 for e in groups if e.startswith("tool:")), "any": int("*" in groups)},
           "n_by_kind": {"own": sum(len(g["gaps"]) for e, g in groups.items() if not e.startswith("tool:") and e != "*"),
                         "tool": sum(len(g["gaps"]) for e, g in groups.items() if e.startswith("tool:")),
                         "any": len(groups["*"]["gaps"]) if "*" in groups else 0}}
    name = f"fetch_value_{pop}_{mode}.json"
    with open(os.path.join(C.OUT, name), "w") as f:
        json.dump(out, f)
    print("wrote", name, "entries", len(groups), out["entry_kinds"], out["n_by_kind"])


if __name__ == "__main__":
    main()
