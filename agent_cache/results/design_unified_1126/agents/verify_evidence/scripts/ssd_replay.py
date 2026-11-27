"""My own page-level replay of the SSD tier of a campaign 34 arm (open loop, recorded timeline).

Pages: 64-token pages of each prompt, keyed by the chained block id of the page's last block (chains.py); a page
that reaches into a prompt's unique tail, and every output page, is unique to its request.
Writes:  at the request's first token (t_send + ttft) the prompt pages from cached_tokens//64 on are written;
         a page the store already holds is touched (fix 1).  Output pages are written at the request's end.
Passes:  the logged cleaner passes; the victims are the N oldest file times as of the scan's start, where N is
         the logged group count; they are unlinked just before the logged end of the pass.
Lookups: every logged s2h_query, placed on the chain with the prefetch_start before it (start = input_len - 1 -
         tokens); the store answers with the run of pages present from the start page.
Reads:   a lookup that found pages is followed by the read the log shows for that rid (same page count); a page
         missing when the read's I/O starts fails its 128-page batch and every later batch.
Loss:    per request, pages a store that deletes nothing delivers minus pages this store delivers.
"""
import bisect
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chains
import runtab
import vlib
from vlib import kv

CL = re.compile(r"deleted (\d+) groups / (\d+) files \(([0-9.]+) GiB\) in ([0-9.]+)s")


class Data:
    def __init__(self, camp, arm):
        self.camp, self.arm = camp, arm
        tab = runtab.build(camp, arm)
        ev = vlib.load(camp, arm)
        ch = chains.load_chains(camp, arm)
        rs = vlib.req_stats(ev, camp, arm)
        self.reqs = tab["reqs"]
        ids = {}
        self.keys = {}      # rid -> list of int page ids (prompt pages)
        self.first_pages = {}   # session -> number of pages of its first prompt
        self.sess = {}
        self.out_pages = {}
        self.npage_ids = 0
        self.last_rid = {}
        self.ended_at = {}
        for rid, q in self.reqs.items():
            c = ch[rid]
            ks = []
            for p, k in enumerate(chains.page_keys(c)):
                if k is None:
                    k = ("u", rid, p)
                i = ids.get(k)
                if i is None:
                    i = ids[k] = len(ids)
                ks.append(i)
            self.keys[rid] = ks
            s = rid.rsplit("-t", 1)[0]
            self.sess[rid] = s
            if q["k"] == 0:
                self.first_pages[s] = len(ks)
            n_out = (c["n_tokens"] + q["out"]) // 64 - c["n_tokens"] // 64
            op = []
            for j in range(n_out):
                k = ("o", rid, j)
                i = ids[k] = len(ids)
                op.append(i)
            self.out_pages[rid] = op
            if c["last"]:
                self.ended_at[s] = q["t_send"] + q["latency"]
        self.n_ids = len(ids)
        self.is_out = [False] * self.n_ids
        for rid, op in self.out_pages.items():
            for i in op:
                self.is_out[i] = True
        # passes
        self.passes = []
        for t, r in ev.get("cleanup", []):
            m = CL.search(r)
            self.passes.append((t - float(m.group(4)), t, int(m.group(1))))
        # lookups, placed on the chain
        last = {}
        seq = sorted([(t, 0, r) for t, r in ev["prefetch_start"]] + [(t, 1, r) for t, r in ev["s2h_query"]],
                     key=lambda x: (x[0], x[1]))
        self.lookups = []   # (t, rid, start_page, req_pages, logged_hit_pages, key list)
        self.n_staged_lookups = 0
        for t, k, r in seq:
            d = kv(r)
            rid = d["rid"]
            if k == 0:
                last[rid] = int(d["tokens"])
            else:
                if rid not in self.reqs or rid not in last or rid not in rs:
                    continue
                q = self.reqs[rid]
                if t < q["entry"] - 1e-3:
                    # staged before arrival: the planned chain is the previous prompt and its output
                    prev = "%s-t%d" % (self.sess[rid], q["k"] - 1)
                    if prev not in self.keys:
                        continue
                    kl = self.keys[prev] + self.out_pages[prev]
                    start = len(kl) * 64 - last[rid]
                    self.n_staged_lookups += 1
                else:
                    kl = self.keys[rid]
                    start = rs[rid]["input"] - 1 - last[rid]
                self.lookups.append((t, rid, max(start, 0) // 64, int(d["tokens_req"]) // 64,
                                     int(d["storage_hit"]) // 64, kl))
        # reads per rid in order: (io_start, pages)
        self.reads = {}
        for rd in tab["reads"]:
            if rd["io_end"] is None or rd.get("dead"):
                continue
            self.reads.setdefault(rd["rid"], []).append((rd["io_end"] - rd["io_ms"], rd["pages"]))
        # failed reads logged
        self.failed = set()
        for t, r in ev.get("other", []):
            m = re.match(r"Prefetch operation (\S+) failed to retrieve page", r)
            if m:
                self.failed.add(m.group(1))


def replay(D, policy="today", delete=True, seed=0, passes=None, refresh_period=0.0, base=None):
    # base: the result of the no-deletion replay.  When given, a request that this store fails (pages the
    # no-deletion store delivers and this one does not) recomputes from the first such page and writes its
    # prompt again from there, as the engine does, so one hole is not counted again at every later turn.
    """-> dict(hits=[...per lookup pages found], got=rid->set(page idx), sizes=[(t, pages before, deleted)],
               deleted_live=..., store_at_pass=[...])"""
    rng = random.Random(seed)
    reqs = D.reqs
    mt = {}            # page id -> file time
    tie = {}           # page id -> scan-order tiebreak (random: directory order is unrelated to time)
    evs = []
    for rid, q in reqs.items():
        if q["ttft"] is None:
            continue
        evs.append((q["t_send"] + q["ttft"], 1, "wp", rid))
        evs.append((q["t_send"] + q["latency"], 2, "wo", rid))
    for i, lk in enumerate(D.lookups):
        evs.append((lk[0], 4, "lk", i))
    for s, t in D.ended_at.items():
        evs.append((t, 3, "end", s))
    pss = list(D.passes) if passes is None else passes
    cap = None
    if isinstance(passes, dict):
        # volume model: every 30 s, a pass starts when the store is at or over hi x cap pages and deletes down
        # to lo x cap (in units of 50 groups); the scan takes scan_s seconds, the unlinks come at its end
        cap = passes
        pss = []
        t = 30.0
        while t < 7230.0:
            evs.append((t, 0, "tick", None))
            t += 30.0
    elif delete:
        for i, (t0, t1, n) in enumerate(pss):
            evs.append((t0, 0, "scan", i))
            evs.append((max(t1 - 0.8, t0), 0, "unlink", i))
    evs.sort(key=lambda x: (x[0], x[1]))
    # reads: schedule a read check for each lookup that finds pages (assigned in order per rid)
    read_ptr = {}
    pending_reads = []   # filled as lookups hit: (io_start, rid, start_page, hit_pages, lookup idx)
    hits = [0] * len(D.lookups)
    got = {}
    victims = {}
    sizes = []
    last_refresh = {}
    sess_pages = {}    # session -> set of page ids in its chains so far (prompt and output)
    extra = []         # dynamically scheduled events (reads), processed in time order with the main list
    import heapq
    heap = []
    idx = 0
    n_evs = len(evs)
    log_deleted = []
    heap2 = []
    busy_until = -1.0
    failed = set()
    deliv = {}
    gen = {}
    li_gen = {}

    def touch(i, t):
        if i in mt:
            mt[i] = t

    while idx < n_evs or heap:
        if heap and (idx >= n_evs or heap[0][0] <= evs[idx][0]):
            t, _, rid, sp, hp, li = heapq.heappop(heap)
            ks = D.lookups[li][5]
            miss = None
            for j in range(sp, min(sp + hp, len(ks))):
                if ks[j] not in mt:
                    miss = j - sp
                    break
            delivered = hp if miss is None else (miss // 128) * 128
            g = got.setdefault(rid, {})
            for j in range(sp, min(sp + delivered, len(ks))):
                deliv[ks[j]] = deliv.get(ks[j], 0) + 1
            if gen.get(rid, 0) == li_gen.get(li, 0):
                for j in range(sp, min(sp + delivered, len(ks))):
                    g[j] = ks[j]
            if miss is not None:
                failed.add(rid)
            continue
        t, _, kind, x = evs[idx]
        idx += 1
        while heap2 and heap2[0][0] <= t:
            ut, ux = heapq.heappop(heap2)
            vs = victims.pop(ux, [])
            for i in vs:
                if "recheck" in policy and mt.get(i, 0.0) > pss[ux][0]:
                    continue
                mt.pop(i, None)
            log_deleted.append((ut, vs))
        if kind == "wp":
            rid = x
            q = reqs[rid]
            ks = D.keys[rid]
            s = D.sess[rid]
            sp = sess_pages.setdefault(s, set())
            sp.update(ks)
            c0 = q["cached"] // 64
            if base is not None:
                need = base["got"].get(rid)
                if need:
                    g = got.get(rid, {})
                    lostpos = [j for j, i in need.items() if j < len(ks) and ks[j] == i and g.get(j) != i]
                    if lostpos:
                        c0 = min(c0, min(lostpos))
            for j in range(c0, len(ks)):
                i = ks[j]
                if i not in mt:
                    tie[i] = rng.random()
                mt[i] = t
        elif kind == "wo":
            rid = x
            s = D.sess[rid]
            sp = sess_pages.setdefault(s, set())
            for i in D.out_pages[rid]:
                mt[i] = t
                tie[i] = rng.random()
                sp.add(i)
            if "refresh" in policy:
                if refresh_period <= 0 or t - last_refresh.get(s, -1e18) >= refresh_period:
                    last_refresh[s] = t
                    for i in D.keys[rid]:
                        touch(i, t)
        elif kind == "end":
            s = x
            if "b_chain" in policy:
                for i in sess_pages.get(s, ()):
                    if i in mt:
                        mt[i] = 0.0
            elif "b_tail" in policy:
                # pages past the session's first prompt (by position in its chains) and its outputs
                keep = D.first_set.get(s, set())
                for i in sess_pages.get(s, ()):
                    if i in mt and i not in keep:
                        mt[i] = 0.0
        elif kind == "lk":
            lt, rid, sp, rp, logged, ks = D.lookups[x]
            h = 0
            for j in range(sp, min(sp + rp, len(ks))):
                if ks[j] in mt:
                    h += 1
                else:
                    break
            hits[x] = h
            # a new lookup from page sp means nothing at or past sp is in memory any more
            g = got.get(rid)
            if g:
                for j in [j for j in g if j >= sp]:
                    del g[j]
            gen[rid] = gen.get(rid, 0) + 1
            li_gen[x] = gen[rid]
            if h > 0:
                if "touch_on_read" in policy:
                    for j in range(sp, sp + h):
                        mt[ks[j]] = t
                # the read that follows: the next logged read of this rid with the logged page count
                rl = D.reads.get(rid, [])
                p = read_ptr.get(rid, 0)
                io_t = None
                while p < len(rl):
                    if rl[p][0] >= lt - 1e-3:
                        io_t = rl[p][0]
                        p += 1
                        break
                    p += 1
                read_ptr[rid] = p
                if io_t is None:
                    io_t = t + 1.0
                heapq.heappush(heap, (io_t, x, rid, sp, h, x))
        elif kind == "tick":
            if busy_until > t or len(mt) < cap["hi"] * cap["pages"]:
                continue
            want = len(mt) - int(cap["lo"] * cap["pages"]) + 1
            n = ((want + 49) // 50) * 50
            pss.append((t, t + cap.get("scan_s", 11.0), n))
            x = len(pss) - 1
            busy_until = t + cap.get("scan_s", 11.0)
            order = sorted(mt, key=lambda i: (mt[i], tie[i]))
            victims[x] = order[:n]
            sizes.append((t, len(mt), n))
            heapq.heappush(heap2, (t + cap.get("scan_s", 11.0) - 0.8, x))
        elif kind == "scan":
            t0, t1, n = pss[x]
            if policy.startswith("oracle"):
                order = sorted(mt, key=lambda i: (-D.next_use(i, t), tie[i]))
            else:
                order = sorted(mt, key=lambda i: (mt[i], tie[i]))
            victims[x] = order[:n]
            sizes.append((t0, len(mt), n))
        elif kind == "unlink":
            vs = victims.pop(x, [])
            cnt = 0
            for i in vs:
                if "recheck" in policy and mt.get(i, 0.0) > pss[x][0]:
                    continue
                if mt.pop(i, None) is not None:
                    cnt += 1
            log_deleted.append((t, vs))
    return dict(hits=hits, got=got, sizes=sizes, deleted=log_deleted, final=len(mt), failed=failed, deliv=deliv, passes=pss)


def _keyset(self, rid):
    ks = self._ksets.get(rid)
    if ks is None:
        ks = self._ksets[rid] = set(self.keys[rid])
    return ks


Data._ksets = None
Data.keyset = _keyset


def first_sets(D):
    D._ksets = {}
    """session -> set of page ids of its first prompt (request k = 0)"""
    out = {}
    for rid, q in D.reqs.items():
        if q["k"] == 0:
            out[D.sess[rid]] = set(D.keys[rid])
    D.first_set = out
    return out


def uses_index(D):
    """page id -> sorted send times of the requests whose prompt holds it (for the oracle and the liveness tests)"""
    use = {}
    for rid, q in D.reqs.items():
        t = q["t_send"]
        for i in D.keys[rid]:
            use.setdefault(i, []).append(t)
    for v in use.values():
        v.sort()
    D.use = use

    def next_use(i, t):
        v = use.get(i)
        if not v:
            return float("inf")
        j = bisect.bisect_right(v, t)
        return v[j] if j < len(v) else float("inf")
    D.next_use = next_use
    return use
