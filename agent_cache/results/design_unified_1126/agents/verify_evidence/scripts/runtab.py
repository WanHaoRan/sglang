"""Per-run tables rebuilt from the logs with my own rules (verify_evidence).

build(camp, arm) -> dict
  reqs:  rid -> dict(entry, adm, queue, fin, k, play, stream, t_send, ttft, latency, prompt, out, cached, ideal,
                     dev, host, sto, wait_s, key, hint, reads=[indices into reads])
  reads: list of dict(rid, park, deadline, hold (first ntp_hold: t,pos,pages,wait,load,hold) or None,
                      disp, waited, io_end, io_ms, pages, pin, pin_tokens, staged)
  timeline: lists T, U, M, NR, NQ  (time; last pool usage; earliest entry among ready waiting turns or inf;
                      number of ready waiting turns; number of waiting turns) after each event
  stall: rid -> dict(D, stall, kind) for readers (kind: 'pass' first pass-over, 'idle' no ready turn ahead and
                      pool room, 'none')
A waiting turn is 'ready' when no read of its own is parked or in flight.
"""
import bisect
import heapq
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv

INF = float("inf")


def _reads(ev):
    reads = []
    open_by_rid = {}
    seq = []
    for name in ("srs_park", "srs_dispatch", "s2h_io", "srs_pin", "ntp_hold", "srs_pin_skipped"):
        for i, (t, r) in enumerate(ev.get(name, [])):
            seq.append((t, {"srs_park": 0, "ntp_hold": 1, "srs_dispatch": 2, "s2h_io": 3, "srs_pin": 4, "srs_pin_skipped": 4}[name], i, name, r))
    seq.sort(key=lambda x: (x[0], x[1], x[2]))
    for t, _, _, name, r in seq:
        d = kv(r)
        rid = d.get("rid")
        if rid is None or rid == "io":
            continue
        lst = open_by_rid.setdefault(rid, [])
        if name == "srs_park":
            rec = dict(rid=rid, park=t, deadline=d.get("deadline_ms"), hold=None, disp=None, waited=None, io_end=None,
                       io_ms=None, pages=int(d["pages"]), pin=None, pin_tokens=None, refused=0)
            reads.append(rec)
            lst.append(rec)
        elif name == "ntp_hold":
            for rec in lst:
                if rec["disp"] is None and rec["hold"] is None:
                    rec["hold"] = dict(t=t, pos=int(d["pos"]), pages=int(d["pages"]), wait=float(d["wait_ms"]) / 1e3,
                                       load=float(d["load_ms"]) / 1e3, hold=float(d["hold_ms"]) / 1e3)
                    break
        elif name == "srs_dispatch":
            for rec in lst:
                if rec["disp"] is None:
                    if d.get("launched") == "1":
                        rec["disp"] = t
                        rec["waited"] = float(d["waited_ms"]) / 1e3
                    else:
                        rec["refused"] += 1
                        rec["refused_at"] = t
                        rec["disp"] = t
                        rec["dead"] = True
                    break
        elif name == "s2h_io":
            for rec in lst:
                if rec["disp"] is not None and rec["io_end"] is None and not rec.get("dead"):
                    rec["io_end"] = t
                    rec["io_ms"] = float(d["ms"]) / 1e3
                    break
        else:
            for rec in reversed(lst):
                if rec["io_end"] is not None and rec["pin"] is None:
                    rec["pin"] = t
                    rec["pin_tokens"] = int(d.get("tokens", 0))
                    break
    return reads


def build(camp, arm, refresh=False):
    p = f"{vlib.EXT}/runtab_c{camp}_{arm}.pkl"
    if os.path.exists(p) and not refresh:
        with open(p, "rb") as f:
            return pickle.load(f)
    ev = vlib.load(camp, arm)
    rs = vlib.req_stats(ev, camp, arm)
    cl = vlib.client(camp, arm)
    reqs = {}
    for q in cl["reqs"]:
        r = rs.get(q["rid"])
        if r is None:
            continue
        cd = q.get("cached_details") or {}
        reqs[q["rid"]] = dict(entry=r["entry"], adm=r["adm"], queue=r["queue"], fin=r["fin"], fwd=r["fwd"], k=q["k"],
                              play=q["play"], stream=q["stream"], inst=q["inst"], t_send=q["t_send"], ttft=q.get("ttft"),
                              latency=q.get("latency"), prompt=q.get("prompt_tokens") or 0,
                              out=q.get("completion_tokens") or 0, cached=q.get("cached_tokens") or 0,
                              ideal=q.get("ideal_cached") or 0, dev=cd.get("device", 0), host=cd.get("host", 0),
                              sto=cd.get("storage", 0), wait_s=q.get("wait_s"), key=q.get("key"), hint=q.get("hint"),
                              tree=q["tree"], gap=q.get("gap"), reads=[])
    reads = _reads(ev)
    for i, rd in enumerate(reads):
        q = reqs.get(rd["rid"])
        rd["staged"] = bool(q is not None and rd["park"] < q["entry"] - 1e-3)
        if q is not None:
            q["reads"].append(i)
    # ---- sweep: ready-queue timeline
    evs = []
    for rid, q in reqs.items():
        evs.append((q["entry"], 0, rid, "in"))
        evs.append((q["adm"], 3, rid, "out"))
        for i in q["reads"]:
            rd = reads[i]
            if rd.get("dead"):
                continue
            end = rd["pin"] if rd["pin"] is not None else (rd["io_end"] if rd["io_end"] is not None else q["adm"])
            a = max(rd["park"], q["entry"])
            b = min(end, q["adm"])
            if b > a:
                evs.append((a, 1, rid, "rd+"))
                evs.append((b, 2, rid, "rd-"))
    for t, kind, running, queue, usage, _, _ in vlib.batches(ev):
        evs.append((t, 4, usage, "batch"))
    evs.sort(key=lambda x: (x[0], x[1]))
    waiting = {}
    heap = []
    U = 0.0
    T, Us, M, NR, NQ = [], [], [], [], []
    nready = 0
    for t, _, x, kind in evs:
        if kind == "in":
            waiting[x] = 0
            nready += 1
            heapq.heappush(heap, (reqs[x]["entry"], x))
        elif kind == "out":
            if waiting.pop(x, 1) == 0:
                nready -= 1
        elif kind == "rd+":
            if x in waiting:
                if waiting[x] == 0:
                    nready -= 1
                waiting[x] += 1
        elif kind == "rd-":
            if x in waiting:
                waiting[x] -= 1
                if waiting[x] == 0:
                    nready += 1
                    heapq.heappush(heap, (reqs[x]["entry"], x))
        else:
            U = x
        while heap and (heap[0][1] not in waiting or waiting[heap[0][1]] != 0):
            heapq.heappop(heap)
        T.append(t)
        Us.append(U)
        M.append(heap[0][0] if heap else INF)
        NR.append(nready)
        NQ.append(len(waiting))
    # ---- stall per reader
    adms = sorted((q["adm"], q["entry"], rid) for rid, q in reqs.items())
    adm_t = [a[0] for a in adms]
    stall = {}
    for rid, q in reqs.items():
        spans = []
        for i in q["reads"]:
            rd = reads[i]
            if rd.get("dead"):
                continue
            end = rd["pin"] if rd["pin"] is not None else (rd["io_end"] if rd["io_end"] is not None else q["adm"])
            a = max(rd["park"], q["entry"])
            b = min(end, q["adm"])
            if b > a:
                spans.append((a, b))
        if not spans:
            continue
        e, adm = q["entry"], q["adm"]
        first = min(s[0] for s in spans)
        last = max(s[1] for s in spans)
        # (a) first admission of a later arrival while this turn waits with a read of its own under way
        Da = INF
        j = bisect.bisect_right(adm_t, first)
        while j < len(adms) and adms[j][0] < adm:
            if adms[j][1] > e:
                Da = adms[j][0]
                break
            j += 1
        # (b) first moment no earlier-arrived ready turn waits and the pool has room
        Db = INF
        j = bisect.bisect_left(T, e)
        while j < len(T) and T[j] < adm:
            if Us[j] < 0.90 and M[j] >= e:
                Db = T[j]
                break
            j += 1
        D = min(Da, Db, adm)
        kind = "none" if D == adm else ("pass" if Da <= Db else "idle")
        # share of [D, adm] with no ready turn waiting at all
        free = 0.0
        if adm > D:
            j = bisect.bisect_left(T, D)
            prev = D
            state = (NR[j - 1] == 0) if j > 0 else True
            while j < len(T) and T[j] < adm:
                if state:
                    free += T[j] - prev
                prev = T[j]
                state = NR[j] == 0
                j += 1
            if state:
                free += adm - prev
        stall[rid] = dict(D=D, stall=adm - D, kind=kind, first=first, last=last, free=free)
    out = dict(reqs=reqs, reads=reads, T=T, U=Us, M=M, NR=NR, NQ=NQ, stall=stall, plays=cl["plays"], run=cl["run"],
               lanes=cl["lanes"])
    with open(p, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return out


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        c, a = spec.split(":")
        tab = build(int(c), a, refresh=True)
        print(spec, "reqs", len(tab["reqs"]), "reads", len(tab["reads"]), "readers", len(tab["stall"]),
              "timeline", len(tab["T"]), flush=True)
