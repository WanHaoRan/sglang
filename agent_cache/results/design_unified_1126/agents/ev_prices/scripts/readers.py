"""Per-request GPU-side timing against the read line.

annotate(st) adds to every request with a server record:
  q_rdy_arr   other ready turns waiting just before it arrived
  q_tot_arr   queue length just before it arrived
  use_arr     GPU pool use on the last batch line at or before its arrival
  run_arr     #running-req on that line
  po          first admission of a LATER arrival while this request still waited (pass-over), or None
  due         when the GPU could have admitted it had its history been resident:
                min(po, first time >= arrival at which no EARLIER arrival was ready and waiting and the pool
                had room for it), capped at its own admission
  stall       admission - due (>= 0): the wait its own lookups/reads added beyond the GPU's turn
  read_tok    tokens its reads put in host memory before admission (sum of loaded)
  n_reads     reads that landed before admission
  first_t0    first lookup-with-hit issue time; last_done: last landing before admission
  left_rdy    ready turns still waiting just after its admission (0.05 s later)
Pool room test: (1 - use) x 668,096 >= prompt + output + 3,341 (half a logged usage step).
"""
import heapq
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

ROOM_MARGIN = 3341


def annotate(st):
    rows = [r for r in st["rows"] if r["arr"] is not None]
    rows.sort(key=lambda r: (r["arr"], r["adm"]))
    n = len(rows)
    suf = [float("inf")] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = min(suf[i + 1], rows[i]["adm"])
    usage, running, rdy, tot = st["usage"], st["running"], st["rdy"], st["tot"]
    uts, uvs = usage.ts, usage.vs
    import bisect
    heap = []  # (adm, idx) of requests in the queue
    for i, r in enumerate(rows):
        a, d = r["arr"], r["adm"]
        while heap and heap[0][0] <= a:
            heapq.heappop(heap)
        ahead = [rows[j] for _, j in heap]
        heapq.heappush(heap, (d, i))
        r["q_rdy_arr"] = rdy.before(a)
        r["q_tot_arr"] = tot.before(a)
        r["use_arr"] = usage.at(a)
        r["run_arr"] = running.at(a)
        r["left_rdy"] = rdy.at(d + 0.05)
        po = suf[i + 1]
        r["po"] = po if po < d - 1e-6 else None
        landed = [rd for rd in r["reads"] if rd.get("done") is not None and rd["done"] <= d + 0.05]
        r["read_tok"] = sum(rd.get("loaded") or 0 for rd in landed)
        r["n_reads"] = sum(1 for rd in landed if (rd.get("loaded") or 0) > 0)
        r["first_t0"] = min((rd["t0"] for rd in landed), default=None)
        r["last_done"] = max((rd["done"] for rd in landed), default=None)
        if r["busy_q"] <= 0.02:
            # never had a lookup/read of its own under way in the queue for long: nothing to attribute
            r["due"], r["stall"] = d, 0.0
            continue
        need = (r["in_len"] or 0) + (r["out_len"] or 0) + ROOM_MARGIN
        # spans in [a, d] in which some earlier arrival is ready and waiting
        blocks = []
        for o in ahead:
            for s, e in o["ready_spans"]:
                s2, e2 = max(s, a), min(e, d)
                if e2 > s2:
                    blocks.append((s2, e2))
        blocks.sort()
        free, t = [], a
        for s, e in blocks:
            if s > t:
                free.append((t, s))
            t = max(t, e)
        if d > t:
            free.append((t, d))
        due = None
        for s, e in free:
            # pool room at s (last batch line at or before s), then at each batch line inside the span
            k = bisect.bisect_right(uts, s) - 1
            if k >= 0 and (1 - uvs[k]) * lib.POOL_GPU >= need:
                due = s
                break
            k += 1
            while k < len(uts) and uts[k] < e:
                if (1 - uvs[k]) * lib.POOL_GPU >= need:
                    due = uts[k]
                    break
                k += 1
            if due is not None:
                break
        if r["po"] is not None:
            due = r["po"] if due is None else min(due, r["po"])
        if due is None or due > d:
            due = d
        r["due"] = due
        r["stall"] = d - due
    return rows


if __name__ == "__main__":
    import qstate
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        st = qstate.build(camp, arm)
        rows = annotate(st)
        rd = [r for r in rows if r["k"] > 0 and r["n_reads"] > 0]
        po_nonbusy = sum(1 for r in rows if r["po"] is not None and r["busy_q"] <= 0.02)
        print(f"{spec}: requests {len(rows)}, returning with a landed SSD read {len(rd)}; passed over though never "
              f"busy {po_nonbusy}; requests passed over {sum(1 for r in rows if r['po'] is not None)}")
        for lo, hi in ((0, 1800), (1800, 3600), (3600, 5400), (5400, 6600)):
            w = [r for r in rd if lo <= r["arr"] < hi]
            if not w:
                continue
            print(f"   {lo//60:3d}-{hi//60:3d} min: readers {len(w):4d}, queue wait mean {lib.mean(r['queue'] for r in w):6.2f} s, "
                  f"stall mean {lib.mean(r['stall'] for r in w):6.2f} s (p50 {lib.quantile([r['stall'] for r in w], .5):5.2f}), "
                  f"tokens read mean {lib.mean(r['read_tok'] for r in w)/1e3:5.1f} K, passed over {sum(1 for r in w if r['po'] is not None)}, "
                  f"arrived to an empty ready queue {sum(1 for r in w if r['q_rdy_arr'] == 0)}")
