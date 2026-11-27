"""Per-page tables on top of model.py: for every prompt page key, its SSD writes, its SSD reads and its uses.

A use is a later request whose prompt holds the page. It was served from memory (GPU or host), from an SSD read made
for that request, or computed again (the page lies past the request's cached prefix).
"""
from __future__ import annotations

import os
import sys
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE  # noqa: E402
from model import prev_req  # noqa: E402


class Pages:
    pass


def build_pages(m) -> Pages:
    p = Pages()
    reqstat = m.reqstat
    arr_of = lambda r: reqstat[r.rid]["entry"] if r.rid in reqstat else r.t_send  # noqa: E731
    rs = sorted(m.reqs.values(), key=arr_of)
    p.rs = rs
    p.idx = {r.rid: i for i, r in enumerate(rs)}
    p.t_arr = [arr_of(r) for r in rs]
    p.t_adm = [(reqstat[r.rid]["entry"] + reqstat[r.rid]["queue_ms"] / 1000.0) if r.rid in reqstat else r.t_send
               for r in rs]
    p.t_fin = [reqstat[r.rid]["t"] if r.rid in reqstat else (r.t_done or r.t_send) for r in rs]
    p.cached_pages = [(reqstat[r.rid]["cached"] if r.rid in reqstat else r.cached) // PAGE for r in rs]
    key_id: dict = {}
    kj: list = []
    uses: list = []
    kids_of: list = []       # request index -> array of key ids of its prompt pages
    for i, r in enumerate(rs):
        a = array("I")
        for j, key in enumerate(r.pages):
            kid = key_id.get(key)
            if kid is None:
                kid = key_id[key] = len(kj)
                kj.append(j)
                uses.append(array("I"))
            uses[kid].append(i)
            a.append(kid)
        kids_of.append(a)
    p.key_id, p.kj, p.uses, p.kids_of = key_id, kj, uses, kids_of
    n = len(kj)
    # writes: (t_sub, t_done, op) per key, in submit order
    writes = [None] * n
    for op in m.order:
        o = m.ops[op]
        if o["part"] != "prompt":
            continue
        i = p.idx[o["rid"]]
        a = kids_of[i]
        for j in range(o["p0"], o["p0"] + o["tokens"] // PAGE):
            if j >= len(a):
                break
            kid = a[j]
            if writes[kid] is None:
                writes[kid] = []
            writes[kid].append((o["t_sub"], o["t_done"], op))
    p.writes = writes
    # reads: per key (t0, t1, read index); per read the keys it covered and its dead output pages
    reads = [None] * n
    read_for: dict = {}      # rid -> set of key ids read from the SSD for that request
    p.read_info = []
    for ri, rd in enumerate(m.reads):
        info = {"kids": (), "out_pages": 0, "unplaced": False, "owner": None, "beyond": 0}
        p.read_info.append(info)
        if rd["start"] is None:
            info["unplaced"] = True
            continue
        s, npg = rd["start"] // PAGE, rd["pages"]
        if rd["staged"]:
            y = prev_req(m, rd["rid"])
            if y is None:
                info["unplaced"] = True
                continue
            a = kids_of[p.idx[y.rid]]
            info["owner"] = y.rid
        else:
            if rd["rid"] not in p.idx:
                info["unplaced"] = True
                continue
            a = kids_of[p.idx[rd["rid"]]]
            info["owner"] = rd["rid"]
        hi = min(s + npg, len(a))
        kids = a[s:hi] if s < len(a) else array("I")
        info["kids"] = kids
        beyond = npg - len(kids)
        if rd["staged"]:
            info["out_pages"] = beyond     # the previous turn's output node
        else:
            info["beyond"] = beyond
        got = read_for.setdefault(rd["rid"], set())
        for kid in kids:
            if reads[kid] is None:
                reads[kid] = []
            reads[kid].append((rd["t0"], rd["t1"], ri))
            got.add(kid)
    p.reads, p.read_for = reads, read_for
    return p


def served(p: Pages, i: int, kid: int) -> str:
    """How request i got page kid: 'mem', 'ssd' or 'rec' (computed again)."""
    if p.kj[kid] >= p.cached_pages[i]:
        return "rec"
    got = p.read_for.get(p.rs[i].rid)
    return "ssd" if got is not None and kid in got else "mem"
