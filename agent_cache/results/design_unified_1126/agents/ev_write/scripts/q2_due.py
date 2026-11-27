"""Q2: when did each written page's host copy go, and how long before that was its SSD write done (the slack)?

No log line names an evicted node, so a page's exit from memory is bracketed by what later lookups saw. Every
prefetch_start gives the resident prefix of its key (GPU plus host) at that instant: the request's own prompt after
arrival; the session's committed chain for a read staged before arrival. ec_start (expected-cost arms) and ntp_due
case 1 give the same for an idle session's chain. For a page written at t_done:
  L = the last instant it was seen in memory before it was first seen missing (t_done if never seen: a page with a
      pending write is host-locked);
  U = the first instant it was seen missing (inf if never).
The page left memory in (L, U], so the write's slack is in (L - t_done, U - t_done].

  python3 -I -B q2_due.py <campaign> <label> [<label> ...]
"""
from __future__ import annotations

import bisect
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import BASE, PAGE, load_trees  # noqa: E402
from model import load, prev_req  # noqa: E402
from pages import build_pages  # noqa: E402

EDGES = [0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 3600]


def q(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(p * len(ys)))]


def snapshots(m, p) -> list:
    """Per request index: [(t, resident pages of that request's chain)], time-ordered."""
    snaps = [[] for _ in p.rs]
    n_own = n_staged = n_ec = n_due = 0
    for rid, lst in m.ps_all.items():
        r = m.reqs.get(rid)
        st = m.reqstat.get(rid)
        chain = m.plan_chain.get(rid)
        for t_ps, tok in lst:
            placed = False
            if r is not None and st is not None and t_ps >= st["entry"] - 0.005:
                start = r.P - 1 - tok
                if start >= 0 and start % PAGE == 0:
                    snaps[p.idx[rid]].append((t_ps, start // PAGE))
                    n_own += 1
                    placed = True
            if not placed and chain is not None:
                start = chain - tok
                y = prev_req(m, rid)
                if y is not None and start >= 0 and start % PAGE == 0:
                    snaps[p.idx[y.rid]].append((t_ps, start // PAGE))
                    n_staged += 1
    for rid, lst in m.ec_start.items():
        chain = m.plan_chain.get(rid)
        y = prev_req(m, rid)
        if chain is None or y is None:
            continue
        for t, _dec, missing, _idle, _r, _w in lst:
            if 0 <= missing <= chain:
                snaps[p.idx[y.rid]].append((t, (chain - missing) // PAGE))
                n_ec += 1
    for rid, lst in m.due.items():
        chain = m.plan_chain.get(rid)
        y = prev_req(m, rid)
        if chain is None or y is None:
            continue
        for t, case, l3_pages, _l, _w in lst:
            if case == 1 and 0 <= l3_pages <= chain // PAGE:
                snaps[p.idx[y.rid]].append((t, chain // PAGE - l3_pages))
                n_due += 1
    for s in snaps:
        s.sort()
    return snaps, {"own": n_own, "staged": n_staged, "ec_start": n_ec, "ntp_due": n_due}


def turnbull(intervals: dict, edges: list, iters: int = 400) -> list:
    """NPMLE of an interval-censored time on the bins of `edges` (bin b = (edges[b-1], edges[b]], last = beyond).
    intervals: {(lo_bin, hi_bin): weight}, the event lies in bins lo_bin..hi_bin inclusive."""
    nb = len(edges) + 1
    pr = [1.0 / nb] * nb
    tot = sum(intervals.values())
    for _ in range(iters):
        new = [0.0] * nb
        for (lo, hi), w in intervals.items():
            s = sum(pr[lo:hi + 1])
            if s <= 0:
                continue
            for b in range(lo, hi + 1):
                new[b] += w * pr[b] / s
        pr = [x / tot for x in new]
    return pr


def analyse(campaign: str, label: str, trees) -> dict:
    m = load(campaign, label, trees)
    p = build_pages(m)
    snaps, n_snaps = snapshots(m, p)
    last_rids = {lst[-1].rid for lst in m.sessions.values() if lst[-1].last}
    res = {"campaign": campaign, "label": label, "snapshots": n_snaps}
    # per first write of a prompt page
    lo_s, hi_s, wts = [], [], []
    before_done = 0
    before_done_tok = 0
    seen_any = 0
    n_first = 0
    cur_status = collections.defaultdict(lambda: [0, 0])     # age bin of the first observation -> [pages, missing]
    intervals = collections.Counter()
    by_part = collections.defaultdict(lambda: collections.Counter())
    done_lag = []
    for kid, ws in enumerate(p.writes):
        if not ws:
            continue
        t_sub, t_done, op = ws[0]
        if t_done is None:
            continue
        n_first += 1
        writer = p.idx[m.ops[op]["rid"]]
        j = p.kj[kid]
        obs = []
        for u in p.uses[kid]:
            for t, res_pages in snaps[u]:
                if t > t_sub:
                    obs.append((t, j < res_pages))
        if len(ws) > 1:
            t_cut = ws[1][0]     # a second submit means the page was computed again: it had left before that
            obs = [o for o in obs if o[0] < t_cut]
            obs.append((t_cut, False))
        obs.sort()
        L, U = t_done, math.inf
        for t, resident in obs:
            if resident:
                L = max(L, t)
            else:
                U = t
                break
        if U < t_done:
            before_done += 1
            before_done_tok += PAGE
        if obs:
            seen_any += 1
            a = obs[0][0] - t_done
            b = bisect.bisect_left(EDGES, a)
            cur_status[b][0] += 1
            cur_status[b][1] += 0 if obs[0][1] else 1
        lo, hi = max(0.0, L - t_done), U - t_done
        lo_s.append(lo)
        hi_s.append(hi)
        lb = bisect.bisect_left(EDGES, lo) if lo > 0 else 0
        hb = bisect.bisect_left(EDGES, hi) if hi < math.inf else len(EDGES)
        if obs:
            intervals[(min(lb, hb), hb)] += 1
        kind = "last" if m.ops[op]["rid"] in last_rids else "live"
        c = by_part[kind]
        c["pages"] += 1
        c["seen"] += 1 if obs else 0
        for x in (5, 30, 60, 120, 300):
            c[f"hi<{x}"] += 1 if hi < x else 0
            c[f"lo<{x}"] += 1 if lo < x else 0
        c["never_missing"] += 1 if hi == math.inf else 0
        done_lag.append(t_done - t_sub)
    n = len(lo_s)
    print(f"\n=== {campaign} {label}: {n} prompt pages first written ({n * PAGE / 1e6:.2f} M tokens); snapshots of "
          f"resident prefixes used: {n_snaps}; pages with at least one later observation: {seen_any} "
          f"({seen_any / n:.1%})")
    print(f"write submit to done per page: p50 {q(done_lag, .5):.2f} s, p90 {q(done_lag, .9):.2f}, p99 "
          f"{q(done_lag, .99):.2f}, max {max(done_lag):.2f}")
    print(f"pages seen missing from memory before their write was done: {before_done} ({before_done_tok} tokens)")
    print("slack of the write = time from write done to the page leaving memory, bracketed (share of pages):")
    print(f"{'kind':<6} {'pages':>8} {'observed':>9} {'never seen missing':>19} | "
          + " ".join(f"{'<' + str(x) + ' s: sure..at most':>24}" for x in (5, 30, 60, 120, 300)))
    for kind in ("live", "last"):
        c = by_part[kind]
        if not c["pages"]:
            continue
        print(f"{kind:<6} {c['pages']:8d} {c['seen'] / c['pages']:9.1%} {c['never_missing'] / c['pages']:19.1%} | "
              + " ".join(f"{c[f'hi<{x}'] / c['pages']:11.2%}..{c[f'lo<{x}'] / c['pages']:<11.2%}"
                         for x in (5, 30, 60, 120, 300)))
    res["bracket"] = {k: dict(v) for k, v in by_part.items()}
    print("first later observation of a page, by its age since the write was done (current status):")
    print(f"{'age (s)':<14} {'pages':>8} {'missing':>8} {'share':>7}")
    cs = {}
    for b in sorted(cur_status):
        lo_e = 0 if b == 0 else EDGES[b - 1]
        hi_e = EDGES[b] if b < len(EDGES) else math.inf
        np_, nm = cur_status[b]
        print(f"{lo_e:>5}-{hi_e:<8} {np_:8d} {nm:8d} {nm / max(np_, 1):7.2%}")
        cs[f"{lo_e}-{hi_e}"] = (np_, nm)
    res["current_status"] = cs
    pr = turnbull(intervals, EDGES)
    cum = 0.0
    print("NPMLE (Turnbull) of the slack over pages with an observation: P(slack <= x)")
    tb = {}
    for b, e in enumerate(EDGES):
        cum += pr[b]
        tb[e] = cum
    print("  " + "  ".join(f"<={e:g}s {tb[e]:.3f}" for e in EDGES) + f"  later/never {pr[-1]:.3f}")
    res["turnbull"] = tb
    res["before_done"] = before_done
    # the dirty set against room, from the census
    cz = [c for c in m.census if c["t"] > 0]
    if cz:
        rows = []
        for c in cz:
            room = c["free"] + c["evict_leaf"] + c["evict_casc"]
            rows.append((c["t"], c["pending_write"], c["h_write"], c["free"], c["evict_leaf"], c["evict_casc"], room,
                         c.get("wq", 0)))
        print("dirty set against room, census lines by 30-minute window (K tokens; median, p90, max):")
        print(f"{'window':<9} {'lines':>5} {'pending_write':>22} {'h_write (only writes hold)':>28} {'free':>18} "
              f"{'evict_leaf':>18} {'free+leaf+casc':>22} {'pending/room max':>17} {'pending/(free+leaf) p50,p90,max':>32}")
        cen = {}
        for w in range(0, 5):
            sel = [r for r in rows if w * 1800 <= r[0] < (w + 1) * 1800]
            if not sel:
                continue
            f = lambda k: f"{q([r[k] for r in sel], .5) / 1e3:6.0f} {q([r[k] for r in sel], .9) / 1e3:6.0f} {max(r[k] for r in sel) / 1e3:6.0f}"  # noqa: E731
            ratio = [r[1] / max(r[6], 1) for r in sel]
            ratio2 = [r[1] / max(r[3] + r[4], 1) for r in sel]
            print(f"{w * 30:3d}-{(w + 1) * 30:<5d} {len(sel):5d} {f(1):>22} {f(2):>28} {f(3):>18} {f(4):>18} {f(6):>22} "
                  f"{max(ratio):17.3f} {q(ratio2, .5):10.2f} {q(ratio2, .9):10.2f} {max(ratio2):10.2f}")
            cen[w * 30] = {"pending_p50": q([r[1] for r in sel], .5), "pending_p90": q([r[1] for r in sel], .9),
                           "pending_max": max(r[1] for r in sel), "room_p50": q([r[6] for r in sel], .5),
                           "room_min": min(r[6] for r in sel), "ratio_max": max(ratio),
                           "freeleaf_p50": q([r[3] + r[4] for r in sel], .5),
                           "freeleaf_min": min(r[3] + r[4] for r in sel)}
        res["census"] = cen
    # pending writes at every submit (finer than the census)
    pend = [(o["t_sub"], o["pending"]) for o in m.ops.values()]
    pend.sort()
    print("pending SSD writes at each submit (K tokens), by 30-minute window: p50, p90, p99, max; share of submits "
          "at or over 50 K")
    for w in range(0, 5):
        sel = [x[1] for x in pend if w * 1800 <= x[0] < (w + 1) * 1800]
        if sel:
            print(f"  {w * 30:3d}-{(w + 1) * 30:<4d} {q(sel, .5) / 1e3:7.1f} {q(sel, .9) / 1e3:7.1f} "
                  f"{q(sel, .99) / 1e3:7.1f} {max(sel) / 1e3:7.1f}   {sum(1 for x in sel if x >= 50000) / len(sel):6.1%}")
    return res


def main() -> None:
    campaign, labels = sys.argv[1], sys.argv[2:]
    trees = load_trees()
    path = f"{BASE}/out/q2_{campaign.replace('campaign', 'c')}.json"
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    for label in labels:
        old[label] = analyse(campaign, label, trees)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
