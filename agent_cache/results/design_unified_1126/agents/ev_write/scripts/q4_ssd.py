"""Q4: what the SSD holds over time (pages of ended sessions against live ones), a replay of the logged cleaner
passes on page write times, and what an ended-sessions-first cleaner would have left.

Model of the store: a page is on the SSD from the ack of its write; a later submit of the same key (a rewrite, or
fix 1's touch of a stored page) moves its file time to that ack. A cleaner pass (the 'NIXL L3 cleanup' line: N
groups deleted, a group is one 64-token page of two files) deletes the N pages with the oldest file time among those
stored when its scan started (line time minus the logged seconds).
Check of the model: every lookup's first missing page must be a page never written, one a replayed pass deleted, or
one whose write was not acked yet.

A page's state at time t:
  ended        every session that has used it so far has ended (its last request finished), and no later request holds it
  ended_reused ... has ended, but a session arriving later holds it (a shared head)
  live_needed  a session in progress holds it, and a later request (of any session) holds it again
  live_dropped a session in progress holds it, but no later request does (dropped from the prompt)
  out_ended / out_live   output nodes (never part of a later prompt), by their session's state

  python3 -I -B q4_ssd.py <campaign> <label> [<label> ...]
"""
from __future__ import annotations

import bisect
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import BASE, PAGE, load_trees  # noqa: E402
from model import load  # noqa: E402
from pages import build_pages  # noqa: E402

CLEAN = re.compile(r"deleted (\d+) groups / (\d+) files \(([\d.]+) GiB\) in ([\d.]+)s, initial_hot=\{[^:]+: '([\d.]+)%'\} "
                   r"final=\{[^:]+: '([\d.]+)%'\}")
CATS = ["ended", "ended_reused", "live_needed", "live_dropped", "out_ended", "out_live"]


def analyse(campaign: str, label: str, trees) -> dict:
    m = load(campaign, label, trees)
    p = build_pages(m)
    rs = p.rs
    # sessions: arrival of the first request, end (finish of the final request, if it was the stream's last)
    sess_start, sess_end = {}, {}
    for inst, lst in m.sessions.items():
        i0 = p.idx[lst[0].rid]
        sess_start[inst] = p.t_arr[i0]
        sess_end[inst] = p.t_fin[p.idx[lst[-1].rid]] if lst[-1].last else float("inf")
    n_keys = len(p.kj)
    # per key: sessions that use it (arrival of their first request holding it), and all use arrival times
    key_sessions = [None] * n_keys
    for kid in range(n_keys):
        d = {}
        for u in p.uses[kid]:
            inst = rs[u].inst
            if inst not in d:
                d[inst] = p.t_arr[u]
        key_sessions[kid] = d

    def state(kid: int, t: float) -> str:
        used_by = [inst for inst, t0 in key_sessions[kid].items() if t0 <= t]
        live = any(sess_end[inst] > t for inst in used_by)
        us = p.uses[kid]
        # uses are in arrival order
        lo, hi = 0, len(us)
        while lo < hi:
            mid = (lo + hi) // 2
            if p.t_arr[us[mid]] > t:
                hi = mid
            else:
                lo = mid + 1
        later = lo < len(us)
        if live:
            return "live_needed" if later else "live_dropped"
        return "ended_reused" if later else "ended"

    # timeline of store events
    ev = []
    for op in m.order:
        o = m.ops[op]
        if o["t_done"] is None or o["part"] is None:
            continue
        ev.append((o["t_done"], 0, op))
    passes = []
    for t, rest in m.cleans:
        g = CLEAN.search(rest)
        if g:
            passes.append({"t": t, "groups": int(g[1]), "gib": float(g[3]), "secs": float(g[4]),
                           "pct0": float(g[5]), "pct1": float(g[6])})
    for i, ps in enumerate(passes):
        ev.append((ps["t"] - ps["secs"], 1, i))
    marks = [600.0 * k for k in range(1, 13)]
    for t in marks:
        ev.append((t, 2, t))
    ev.sort()
    stored: dict = {}       # kid -> file time
    out_stored: dict = {}   # op -> (file time, pages)
    deleted_at: dict = {}   # kid -> [(t_deleted)]
    deleted_state: dict = {}
    res = {"campaign": campaign, "label": label, "passes": [], "marks": []}

    def tally(t: float) -> dict:
        c = collections.Counter()
        for kid in stored:
            c[state(kid, t)] += 1
        for op, (_ft, n) in out_stored.items():
            inst = m.reqs[m.ops[op]["rid"]].inst
            c["out_live" if sess_end[inst] > t else "out_ended"] += n
        return c

    print(f"\n=== {campaign} {label}: {len(passes)} cleaner passes")
    print("what the store holds (M tokens), every 10 minutes and at each pass:")
    print(f"{'t (s)':>8} " + " ".join(f"{c:>13}" for c in CATS) + f" {'all':>8} {'ended share':>12}")

    def show(t: float, c: dict, tag: str = "") -> None:
        tot = sum(c.values())
        dead = c["ended"] + c["out_ended"]
        print(f"{t:8.0f} " + " ".join(f"{c[x] * PAGE / 1e6:13.2f}" for x in CATS)
              + f" {tot * PAGE / 1e6:8.2f} {dead / max(tot, 1):12.1%} {tag}")

    for t, kind, x in ev:
        if kind == 0:
            o = m.ops[x]
            if o["part"] == "out":
                out_stored[x] = (t, o["tokens"] // PAGE)
            else:
                a = p.kids_of[p.idx[o["rid"]]]
                for j in range(o["p0"], min(o["p0"] + o["tokens"] // PAGE, len(a))):
                    stored[a[j]] = t
        elif kind == 2:
            c = tally(t)
            show(t, c)
            res["marks"].append({"t": t, **{k: c[k] * PAGE for k in CATS}})
        else:
            ps = passes[x]
            c0 = tally(t)
            n_stored = sum(c0.values())
            # today's cleaner: the oldest file times first
            items = [(ft, 0, kid, 1) for kid, ft in stored.items()] + \
                    [(ft, 1, op, n) for op, (ft, n) in out_stored.items()]
            items.sort()
            need = ps["groups"]
            today = collections.Counter()
            victims_p, victims_o = [], []
            for ft, typ, key, n in items:
                if need <= 0:
                    break
                take = min(n, need)
                need -= take
                if typ == 0:
                    today[state(key, t)] += 1
                    victims_p.append(key)
                else:
                    inst = m.reqs[m.ops[key]["rid"]].inst
                    today["out_live" if sess_end[inst] > t else "out_ended"] += take
                    victims_o.append((key, take, n))
            cutoff_age = t - items[min(len(items) - 1, max(0, len(victims_p) + len(victims_o) - 1))][0]
            # ended first: pages of ended sessions and their outputs, oldest first; then by age
            dead_pool = c0["ended"] + c0["ended_reused"] + c0["out_ended"]
            ended_first_live = max(0, ps["groups"] - dead_pool)
            # how the ended-first pass splits its N deletions
            e_items = [(ft, typ, key, n) for ft, typ, key, n in items
                       if (typ == 0 and state(key, t) in ("ended", "ended_reused"))
                       or (typ == 1 and sess_end[m.reqs[m.ops[key]["rid"]].inst] <= t)]
            need = ps["groups"]
            ef = collections.Counter()
            for ft, typ, key, n in e_items:
                if need <= 0:
                    break
                take = min(n, need)
                need -= take
                ef[state(key, t) if typ == 0 else "out_ended"] += take
            # ended first, and within them the deepest chain positions first (tails before heads)
            t_items = sorted(((p.kj[key] if typ == 0 else 10**9, ft, typ, key, n) for ft, typ, key, n in e_items),
                             key=lambda x: (-x[0], x[1]))
            need = ps["groups"]
            tf = collections.Counter()
            for _j, ft, typ, key, n in t_items:
                if need <= 0:
                    break
                take = min(n, need)
                need -= take
                tf[state(key, t) if typ == 0 else "out_ended"] += take
            live_total = c0["live_needed"] + c0["live_dropped"]
            row = {"t": ps["t"], "scan_t": t, "groups": ps["groups"], "pct0": ps["pct0"], "stored_pages": n_stored,
                   "stored": {k: c0[k] * PAGE for k in CATS}, "today": {k: today[k] * PAGE for k in CATS},
                   "ended_first": {k: ef[k] * PAGE for k in CATS}, "ended_first_live_pages": ended_first_live,
                   "oldest_kept_age_s": cutoff_age}
            res["passes"].append(row)
            show(t, c0, f"<- pass {x + 1}: log says {ps['pct0']}% full, deletes {ps['groups']} pages "
                        f"({ps['groups'] * PAGE / 1e6:.2f} M tokens); the model holds {n_stored} pages")
            print(f"{'':8} today's order (oldest file time): " + ", ".join(
                f"{k} {today[k] * PAGE / 1e6:.2f}" for k in CATS if today[k]) +
                  f"; youngest page deleted was written {cutoff_age / 60:.1f} min before the pass")
            print(f"{'':8} ended sessions first: " + ", ".join(f"{k} {ef[k] * PAGE / 1e6:.2f}" for k in CATS if ef[k])
                  + f"; pages of sessions in progress it must take: {ended_first_live} "
                    f"(they hold {live_total * PAGE / 1e6:.2f} M; today's pass took "
                    f"{(today['live_needed'] + today['live_dropped']) * PAGE / 1e6:.2f} M of it, "
                    f"{today['live_needed'] * PAGE / 1e6:.2f} M needed again)")
            print(f"{'':8} ended sessions first, tails before heads: " + ", ".join(
                f"{k} {tf[k] * PAGE / 1e6:.2f}" for k in CATS if tf[k]) + f"; shared heads a later session uses: "
                  f"{tf['ended_reused'] * PAGE / 1e6:.3f} M (oldest-first order: {ef['ended_reused'] * PAGE / 1e6:.3f} M, "
                  f"today's pass: {today['ended_reused'] * PAGE / 1e6:.3f} M)")
            row["ended_tails_first"] = {k: tf[k] * PAGE for k in CATS}
            for kid in victims_p:
                del stored[kid]
                deleted_at.setdefault(kid, []).append(t)
                deleted_state[(kid, t)] = state(kid, t)
            for op, take, n in victims_o:
                if take >= n:
                    del out_stored[op]
                else:
                    out_stored[op] = (out_stored[op][0], n - take)
    done_t = {}
    for kid, ws in enumerate(p.writes):
        if ws:
            done_t[kid] = [w[1] for w in ws if w[1] is not None]
    # check of the model: the first missing page of every lookup
    chk = collections.Counter()
    first_pass = passes[0]["t"] - passes[0]["secs"] if passes else float("inf")
    for q in m.lookups:
        if q["start"] is None or q["staged"]:
            continue
        i = p.idx.get(q["rid"])
        if i is None:
            continue
        a = p.kids_of[i]
        first_missing = (q["start"] + q["hit"]) // PAGE
        end = (q["start"] + q["tokens_req"]) // PAGE
        if first_missing >= end:
            continue     # everything looked up was found
        if first_missing >= len(a):
            continue
        kid = a[first_missing]
        when = "after" if q["t"] >= first_pass else "before"
        ds = done_t.get(kid)
        if not ds or min(ds) > q["t"]:
            chk[(when, "never written or write not acked yet")] += 1
            continue
        # stored by the model at q.t?  last ack before q.t against later deletions
        last_ack = max(d for d in ds if d <= q["t"])
        dels = [d for d in deleted_at.get(kid, []) if last_ack <= d <= q["t"]]
        if dels:
            chk[(when, "a replayed cleaner pass deleted it")] += 1
        else:
            chk[(when, "the model says it is stored")] += 1
    # what the replayed deletions cost: requests that computed history again because their first uncached page
    # had been deleted (and not written again since)
    cost = collections.Counter()
    cost_n = collections.Counter()
    for i, r in enumerate(rs):
        a = p.kids_of[i]
        cp = p.cached_pages[i]
        ideal_pages = min(r.ideal // PAGE, len(a))
        if cp >= ideal_pages:
            continue
        kid = a[cp]
        t = p.t_arr[i]
        ds = done_t.get(kid) or []
        acks = [d for d in ds if d <= t]
        lost = (ideal_pages - cp) * PAGE
        if not acks:
            cost["history never on the SSD (not written or not acked)"] += lost
            cost_n["history never on the SSD (not written or not acked)"] += 1
            continue
        last_ack = max(acks)
        dels = [d for d in deleted_at.get(kid, []) if last_ack <= d <= p.t_adm[i]]
        if dels:
            st_ = deleted_state[(kid, dels[-1])]
            cost[f"first uncached page deleted by a pass; it was {st_} then"] += lost
            cost_n[f"first uncached page deleted by a pass; it was {st_} then"] += 1
        else:
            cost["first uncached page stored (lookup cap, eviction while queued, or not looked up)"] += lost
            cost_n["first uncached page stored (lookup cap, eviction while queued, or not looked up)"] += 1
    print("history computed again (ideal_cached beyond the cached prefix), by what happened to its first page:")
    for k, v in sorted(cost.items(), key=lambda kv_: -kv_[1]):
        print(f"  {k}: {cost_n[k]} requests, {v / 1e6:.3f} M tokens")
    res["recompute_by_cause"] = {k: (cost_n[k], v) for k, v in cost.items()}
    print("check: first missing page of each lookup made after arrival (lookups with something missing):")
    for k, v in sorted(chk.items()):
        print(f"  {k[0]} the first pass: {k[1]}: {v}")
    res["check"] = {f"{k[0]}|{k[1]}": v for k, v in chk.items()}
    return res


def main() -> None:
    campaign, labels = sys.argv[1], sys.argv[2:]
    trees = load_trees()
    path = f"{BASE}/out/q4_{campaign.replace('campaign', 'c')}.json"
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
