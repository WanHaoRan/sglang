"""Page-level replay of the SSD tier of a campaign 34 arm (label ev_ssd). Stdlib only, CPU only, read-only on records.

What is replayed (open loop, on the recorded timeline):
  * page chains per request from the trace's block ids (c34lib.TreeChains), shared prefixes included;
  * writes: at a request's prefill end, its prompt pages past what the caches served are written (or, if the store
    holds them, touched: fix 1); at its end, its output pages. Checked against the h2s_submit stream (check_writes.py);
  * lookups: each logged s2h_query (start page, pages asked) is answered from the simulated store: the run of
    present pages from the start page (a lookup stops at the first missing page);
  * reads: a landed read needs every page of a 128-page batch at read time; a missing file fails that batch and
    all later ones (cache_controller._page_transfer / hicache_nixl._batch_xfer);
  * the cleaner: either the logged passes (time, groups deleted) or a volume model (a pass at the first 30 s tick
    with the store at or over the high mark, deleting in the policy's order down to the low mark).
A request's outcome under a counterfactual store follows the recorded sequence of its lookups: the in-memory prefix
is the recorded one, what the SSD adds is recomputed from the simulated store (see Req.replay_lookup).
"""
import sys, bisect, collections, heapq, json, math
from array import array

sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from c34lib import *  # noqa: E402,F401

READ_BATCH = 128          # hicache_storage.STORAGE_BATCH_SIZE
THRESH_PAGES = 4          # prefetch_threshold 256 tokens (l3_extra.json)
W_DELAY_P = 0.3           # client prefill end -> SSD write done, median (check_wtime.py)
W_DELAY_O = 0.15          # client request end -> SSD write done, median
GAMMA = 0.043e-3          # s per recomputed token (FA3), task constant
N_USERS = 128


class Lookup:
    __slots__ = ("t", "m", "n", "hit", "staged", "chain", "useful", "read_t", "read_done", "completed", "failed",
                 "parked")

    def __init__(self, t, m, n, hit, staged, chain, useful):
        self.t, self.m, self.n, self.hit, self.staged, self.chain, self.useful = t, m, n, hit, staged, chain, useful
        self.read_t = None      # actual read start (s2h_io time - ms)
        self.read_done = None
        self.completed = None   # pages the actual read landed
        self.failed = False
        self.parked = False


class Req:
    __slots__ = ("rid", "conv", "k", "t_send", "t_pre", "t_end", "P", "C", "c_act", "H", "q", "pages", "out0",
                 "n_out", "ended", "lookups", "idx", "first_shared", "storage_act", "prev", "t_admit", "gap", "wait_s",
                 "hint_ended")


def load_arm(arm):
    """Requests (with page chains and lookups) of one arm, plus the logged cleaner passes."""
    header, trees = load_trees()
    t_start, recs, plays, args = load_client(arm)
    ev = load_events(arm)
    inter = Interner()
    chains = {}
    reqs = []
    for rec in recs:
        if rec.get("ttft") is None or rec.get("prompt_tokens") is None:
            continue
        tc = chains.get(rec["play"])
        if tc is None:
            tc = chains[rec["play"]] = TreeChains(trees[rec["tree"]], inter)
        r = Req()
        r.rid = rec["rid"]
        r.conv = rec["inst"]
        r.k = rec["k"]
        r.t_send = rec["t_send"]
        r.t_pre = rec["t_send"] + rec["ttft"]
        r.t_end = rec["t_send"] + rec["latency"]
        r.P, r.C = rec["prompt_tokens"], rec["completion_tokens"]
        r.c_act = (rec["cached_tokens"] or 0) // PAGE
        r.H = (rec["ideal_cached"] or 0) // PAGE
        r.pages = tc.pages(rec["stream"], rec["k"])[0]
        r.q = r.P // PAGE
        assert r.q == len(r.pages), (r.rid, r.q, len(r.pages))
        r.n_out = max(0, (r.P + r.C - 1) // PAGE - r.q)
        # the end signal: the stream's last request in the trace (dist hints carry it as hint.ended; with exact
        # hints the client sends no next-turn hint after the last request, which the engine reads as the end)
        r.ended = rec["k"] + 1 >= len(trees[rec["tree"]]["streams"][rec["stream"]]["requests"])
        r.hint_ended = bool((rec.get("hint") or {}).get("ended"))
        r.lookups = []
        r.storage_act = ((rec.get("cached_details") or {}).get("storage") or 0) // PAGE
        r.gap = rec.get("gap")
        r.wait_s = rec.get("wait_s")
        r.t_admit = None
        reqs.append(r)
    n_prompt_pages = len(inter)
    nxt = n_prompt_pages
    for r in reqs:
        r.out0 = nxt
        nxt += r.n_out
    n_ids = nxt
    reqs.sort(key=lambda r: r.t_send)
    byrid = {}
    last_of_conv = {}
    for i, r in enumerate(reqs):
        r.idx = i
        byrid[r.rid] = r
    for r in sorted(reqs, key=lambda r: (r.conv, r.k)):
        r.prev = last_of_conv.get(r.conv)
        last_of_conv[r.conv] = r
    # server-side admission time
    for d in ev.get("req_stats", []):
        r = byrid.get(d["rid"])
        if r is not None:
            r.t_admit = d["entry"] + d["queue_ms"] / 1000.0 - t_start
    entry = {d["rid"]: d["entry"] - t_start for d in ev.get("req_stats", [])}
    # lookups, with their parks / reads / results paired in log order per rid
    per = collections.defaultdict(list)
    for name in ("s2h_query", "srs_park", "s2h_io", "prefetch_ok", "read_fail"):
        for d in ev.get(name, []):
            per[d["rid"]].append((d["t"] - t_start, name, d))
    n_unk = 0
    for rid, lst in per.items():
        r = byrid.get(rid)
        if r is None:
            n_unk += sum(1 for x in lst if x[1] == "s2h_query")
            continue
        lst.sort(key=lambda x: x[0])
        cur = None
        e = entry.get(rid, r.t_send)
        for t, name, d in lst:
            if name == "s2h_query":
                n = d["tokens_req"] // PAGE
                ql = (r.P - 1) // PAGE
                staged = t < e - 0.001 or n > ql
                if staged and r.prev is not None:
                    p = r.prev
                    chain = array("I", p.pages)
                    chain.extend(range(p.out0, p.out0 + p.n_out))
                    qp = len(chain)
                    m = qp - n
                    useful = 0
                    for a, b in zip(p.pages, r.pages):
                        if a != b:
                            break
                        useful += 1
                    if m < 0:      # plan longer than what we rebuilt: fall back to the request's own chain
                        chain, m, useful, staged = r.pages, max(0, ql - n), r.q, False
                else:
                    staged = False
                    chain, m, useful = r.pages, ql - n, r.q
                    if m < 0:
                        m = 0
                cur = Lookup(t, m, n, d["storage_hit"] // PAGE, staged, chain, useful)
                r.lookups.append(cur)
            elif cur is None:
                continue
            elif name == "srs_park":
                cur.parked = True
            elif name == "s2h_io":
                cur.read_done = t
                cur.read_t = t - d["ms"] / 1000.0
            elif name == "prefetch_ok":
                cur.completed = d["completed"] // PAGE
            elif name == "read_fail":
                cur.failed = True
    passes = [(d["t"] - t_start, d["secs"], d["groups"], d["pct0"], d["pct1"]) for d in ev.get("cleanup", [])]
    cleaner_start = None
    with open(evt_path(arm)) as f:
        for line in f:
            if "HiCacheL3Cleaner started" in line:
                cleaner_start = epoch(line) - t_start
                break
    # typical lookup -> read start delay by 10-minute window (for reads only the model makes)
    dl = collections.defaultdict(list)
    for r in reqs:
        for lk in r.lookups:
            if lk.read_t is not None:
                dl[int(lk.t // 600)].append(lk.read_t - lk.t)
    delay = {k: sorted(v)[len(v) // 2] for k, v in dl.items()}
    return dict(arm=arm, t_start=t_start, reqs=reqs, byrid=byrid, n_prompt_pages=n_prompt_pages, n_ids=n_ids,
                passes=passes, cleaner_start=cleaner_start, read_delay=delay, n_unknown_lookups=n_unk, plays=plays,
                ev=ev)


# ---------------------------------------------------------------------------------------------------------------
def page_sharing(data):
    """Number of distinct sessions whose prompts hold each page (bytearray, capped at 255)."""
    n = data["n_ids"]
    cnt = bytearray(n)
    last = array("i", [-1]) * n
    for r in sorted(data["reqs"], key=lambda r: r.conv):
        c = r.conv
        for p in r.pages:
            if last[p] != c:
                last[p] = c
                if cnt[p] < 255:
                    cnt[p] += 1
    return cnt


def page_shared_since(data):
    """For each page, the send time of the first request by a second session whose prompt holds it (inf if none):
    what a per-page 'used by more than one session' bit kept by an index of stored pages would know at time t."""
    n = data["n_ids"]
    first = array("i", [-1]) * n
    since = array("d", [math.inf]) * n
    for r in data["reqs"]:            # send order
        c = r.conv
        t = r.t_send
        for p in r.pages:
            f = first[p]
            if f < 0:
                first[p] = c
            elif f != c and since[p] == math.inf:
                since[p] = t
    return since


def page_user_times(data):
    """page -> sorted send times at which a new session first held it (pages with >= 2 sessions only): the use
    count an index of stored head pages would keep."""
    n = data["n_ids"]
    users = {}
    first = array("i", [-1]) * n
    for r in data["reqs"]:            # send order
        c = r.conv
        t = r.t_send
        for p in r.pages:
            f = first[p]
            if f < 0:
                first[p] = c
            elif f != c:
                u = users.get(p)
                if u is None:
                    users[p] = ({f, c}, [t])
                elif c not in u[0]:
                    u[0].add(c)
                    u[1].append(t)
    return {p: v[1] for p, v in users.items()}


def first_requests(data):
    out = {}
    for r in data["reqs"]:
        if r.conv not in out or r.k < out[r.conv].k:
            out[r.conv] = r
    return out


class Sim:
    """One replay. policy: dict with
         order: 'mtime' (oldest file time first) | 'oracle_use' | 'oracle_need'
         end_zero: None | 'chain' | 'own' | 'index'      (b: what the end signal zeroes)
         refresh: None | 'end'                            (c: a request refreshes its chain's stored pages at its end)
         refresh_limit: seconds between refreshes of one session (0 = every request); a request that ends inside
                        the limit gets a deferred refresh when the limit expires
         head_last: True (head touched last: tails older than heads) | False (head touched first)
         touch_on_hit: True = a lookup touches the pages it finds (protects a parked read)
       passes: 'logged' | 'volume' | 'none';  cap/hi/lo: volume model (pages, fractions)."""

    def __init__(self, data, policy, passes="logged", cap=390100, hi=0.90, lo=0.80, del_frac=0.93, pass_secs=11.6,
                 tick=30.0, t_stop=None):
        self.d = data
        self.policy = dict(order="mtime", end_zero=None, refresh=None, refresh_limit=0.0, head_last=True,
                           touch_on_hit=False, recheck=False, repair=False)
        self.policy.update(policy)
        self.passes_mode = passes
        self.cap, self.hi, self.lo = cap, hi, lo
        self.del_frac, self.pass_secs, self.tick = del_frac, pass_secs, tick
        n = data["n_ids"]
        self.mt = array("d", [-1.0]) * n        # file time of each stored page; -1 = not stored
        self.writer = array("i", [-1]) * n      # session that last wrote the page
        self.first_writer = array("i", [-1]) * n
        self.ever = bytearray(n)                # ever written
        self.del_pass = array("i", [-1]) * n    # pass that deleted the page (while it stays deleted)
        self.del_mt = array("d", [-1.0]) * n    # its file time when deleted
        self.stored = 0
        self.t_stop = t_stop
        self.pass_log = []
        self.touches = []                       # (t, pages touched by a refresh or a lookup)
        self.zeroed = 0
        self.repairs = 0
        self.conv_written = collections.defaultdict(list)   # session -> pages it wrote (b: own)
        self.conv_last_refresh = {}
        self.conv_deferred = set()
        self.conv_ended = {}                    # session -> end time
        self.live_chain = {}                    # session -> latest request
        self.out = {}                           # rid -> (cached pages, first loss or None)
        self.pending = {}                       # rid -> state while the request is between send and prefill end
        self.lookup_cmp = []                    # (r, lookup, predicted hit) for validation
        self.readfail_pred = []
        self.lookup_losses = []                 # (t, 'zero' | 'part') for every lookup a deleted page cut short
        self.scan = None

    # ----- store -----
    def write(self, pid, t, conv):
        if self.mt[pid] < 0:
            self.stored += 1
            self.del_pass[pid] = -1
        self.mt[pid] = t
        self.writer[pid] = conv
        if not self.ever[pid]:
            self.ever[pid] = 1
            self.first_writer[pid] = conv

    def run_len(self, chain, start, limit):
        """Pages present from chain[start], up to chain[limit-1]."""
        mt = self.mt
        j = start
        while j < limit and mt[chain[j]] >= 0:
            j += 1
        return j - start

    def run_len_ever(self, chain, start, limit):
        ever = self.ever
        j = start
        while j < limit and ever[chain[j]]:
            j += 1
        return j - start

    # ----- event loop -----
    def run(self):
        d = self.d
        reqs = d["reqs"]
        heap = []
        seq = 0

        def push(t, kind, a=None, b=None):
            nonlocal seq
            seq += 1
            heapq.heappush(heap, (t, seq, kind, a, b))

        for r in reqs:
            push(r.t_send, "send", r)
            for i, lk in enumerate(r.lookups):
                push(lk.t, "lookup", r, i)
            push(max(r.t_pre, r.t_send + 1e-6), "final", r)
        if self.passes_mode == "logged":
            for k, (t_end, secs, groups, p0, p1) in enumerate(d["passes"]):
                push(t_end - secs, "pass_snap", k, groups)
        elif self.passes_mode == "volume":
            push(d["cleaner_start"], "tick")
        self.t_last = max(r.t_end for r in reqs) + 120.0     # no cleaner tick after the run
        self.push = push
        while heap:
            t, _, kind, a, b = heapq.heappop(heap)
            if self.t_stop is not None and t > self.t_stop:
                break
            getattr(self, "ev_" + kind)(t, a, b)
        return self

    # ----- request side -----
    @staticmethod
    def _new_state():
        return dict(a_cur=None, c_cur=None, reads=0, final_due=False, first_loss=None, staged_seen=False, useful=None)

    def ev_send(self, t, r, _):
        if r.rid not in self.pending:
            self.pending[r.rid] = self._new_state()
        self.live_chain[r.conv] = r

    def loss_info(self, kind, r, lk, start, hit, pid, t, **kw):
        info = dict(kind=kind, start=start, hit=hit, pid=pid, pos=start + hit, del_pass=self.del_pass[pid],
                    del_mt=self.del_mt[pid], writer=self.writer[pid], first_writer=self.first_writer[pid], t=t,
                    staged=lk.staged, writer_end=self.conv_ended.get(self.writer[pid]))
        info.update(kw)
        return info

    def ev_lookup(self, t, r, i):
        lk = r.lookups[i]
        st = self.pending.get(r.rid)
        if st is None:      # staged lookup before the send
            st = self.pending[r.rid] = self._new_state()
        if lk.staged:
            st["staged_seen"] = True
        elif st["staged_seen"] and st["a_cur"] is not None:
            # leave the planned chain's coordinates: only its common prefix with the prompt counts
            u = st["useful"] if st["useful"] is not None else r.q
            st["a_cur"] = min(st["a_cur"], u)
            st["c_cur"] = min(st["c_cur"], u)
            st["staged_seen"] = False
        if st["a_cur"] is None:
            st["a_cur"] = st["c_cur"] = lk.m
        else:
            if lk.m < st["a_cur"]:
                st["c_cur"] = min(st["c_cur"], lk.m)      # the recorded prefix shrank (evicted again)
            elif lk.m > st["a_cur"]:
                st["c_cur"] = max(st["c_cur"], lk.m)      # memory gained pages (another request computed them)
            st["a_cur"] = lk.m
        st["useful"] = lk.useful
        end = min(lk.m + lk.n, len(lk.chain))
        start = min(st["c_cur"], end)
        hit = self.run_len(lk.chain, start, end)
        hit_ever = self.run_len_ever(lk.chain, start, end)
        if start == lk.m:
            self.lookup_cmp.append((r, lk, hit))
        if hit < hit_ever:
            self.lookup_losses.append((t, "zero" if hit == 0 else "part"))
        if hit < hit_ever and st["first_loss"] is None:
            pid = lk.chain[start + hit]
            kind = "zero_head" if (start == 0 and hit == 0) else "zero_mid" if hit == 0 else "partial"
            st["first_loss"] = self.loss_info(kind, r, lk, start, hit, pid, t, hit_ever=hit_ever)
        if hit and self.policy["touch_on_hit"]:
            mt = self.mt
            ch = lk.chain
            for j in range(start, start + hit):
                mt[ch[j]] = t + (start + hit - j) * 1e-6
            self.touches.append((t, hit))
        # the read
        if hit >= THRESH_PAGES:
            if lk.read_t is not None and lk.read_t > t:
                due = lk.read_t
            else:
                due = t + self.d["read_delay"].get(int(t // 600), 1.0)
            st["reads"] += 1
            st["c_cur"] = start + hit                      # taken as landed until the read is resolved
            self.push(due, "read", r, (i, start, hit))
        # the recorded world
        if lk.completed is not None:
            st["a_cur"] = lk.m + lk.completed
        elif lk.hit >= THRESH_PAGES and lk.parked:
            st["a_cur"] = lk.m + lk.hit

    def ev_read(self, t, r, info):
        i, start, hit = info
        lk = r.lookups[i]
        st = self.pending.get(r.rid)
        if st is None:
            return
        mt = self.mt
        ch = lk.chain
        landed = 0
        for b0 in range(start, start + hit, READ_BATCH):
            b1 = min(b0 + READ_BATCH, start + hit)
            if all(mt[ch[j]] >= 0 for j in range(b0, b1)):
                landed += b1 - b0
            else:
                break
        st["reads"] -= 1
        if landed < hit:
            st["c_cur"] = min(st["c_cur"], start + landed)
            self.readfail_pred.append((r, lk, start, hit, landed, t))
            if st["first_loss"] is None:
                j = next(j for j in range(start + landed, start + hit) if mt[ch[j]] < 0)
                st["first_loss"] = self.loss_info("read_in_flight", r, lk, start, hit, ch[j], t, landed=landed)
        if st["final_due"] and st["reads"] == 0:
            self.finalize(max(t, r.t_pre), r)

    def ev_final(self, t, r, _):
        st = self.pending.get(r.rid)
        if st is None:
            return
        if st["reads"] > 0:
            st["final_due"] = True      # a read the model added is still out: the request waits for it
            return
        self.finalize(t, r)

    def finalize(self, t, r):
        st = self.pending.pop(r.rid)
        if st["a_cur"] is None:
            c = r.c_act
        else:
            a_cur, c_cur = st["a_cur"], st["c_cur"]
            if st["staged_seen"]:
                u = st["useful"] if st["useful"] is not None else r.q
                a_cur, c_cur = min(a_cur, u), min(c_cur, u)
            if r.c_act < a_cur:
                c = min(c_cur, r.c_act)
            elif r.c_act > a_cur:
                c = max(c_cur, r.c_act)
            else:
                c = c_cur
        c = max(0, min(c, r.q))
        self.out[r.rid] = (c, st["first_loss"])
        self.push(t + W_DELAY_P, "wprompt", r, c)
        self.push(max(r.t_end, t) + W_DELAY_O, "wout", r)

    def ev_wprompt(self, t, r, c):
        pages = r.pages
        for j in range(c, r.q):
            self.write(pages[j], t + (j - c) * 0.003, r.conv)
        self.conv_written[r.conv].extend(pages[c:r.q])

    def ev_wout(self, t, r, _):
        own = self.conv_written[r.conv]
        for pid in range(r.out0, r.out0 + r.n_out):
            self.write(pid, t, r.conv)
            own.append(pid)
        if r.ended:
            self.push(t + 0.05, "end", r)
        if self.policy["refresh"] == "end":
            self.push(t + 0.35, "refresh", r)

    def ev_refresh(self, t, r, deferred):
        """(c) touch the stored pages of the request's chain; with head_last the head gets the newest time."""
        conv = r.conv
        if conv in self.conv_ended and self.policy["end_zero"]:
            return
        if deferred:
            self.conv_deferred.discard(conv)
            r = self.live_chain.get(conv, r)
        lim = self.policy["refresh_limit"]
        last = self.conv_last_refresh.get(conv)
        if lim and last is not None and t - last < lim - 1e-9:
            if conv not in self.conv_deferred:
                self.conv_deferred.add(conv)
                self.push(last + lim, "refresh", r, True)
            return
        self.conv_last_refresh[conv] = t
        mt = self.mt
        n = 0
        q = r.q
        pages = r.pages
        if self.policy.get("repair"):
            # a touch that finds no file tells the engine the page is gone; memory holds it now, so write it again
            for j in range(q):
                p = pages[j]
                if mt[p] < 0:
                    self.write(p, t, conv)
                    self.repairs += 1
        if self.policy["head_last"]:
            for j in range(q):
                p = pages[j]
                if mt[p] >= 0:
                    mt[p] = t + (q - j) * 1e-6
                    n += 1
        else:
            for j in range(q):
                p = pages[j]
                if mt[p] >= 0:
                    mt[p] = t + j * 1e-6
                    n += 1
        self.touches.append((t, n))

    def ev_end(self, t, r, _):
        """(b) the end signal: the session's pages get file time zero."""
        self.conv_ended[r.conv] = t
        mode = self.policy["end_zero"]
        if self.live_chain.get(r.conv) is r:
            del self.live_chain[r.conv]
        if not mode:
            return
        mt = self.mt
        if mode == "chain":
            targets = list(r.pages) + self.conv_written[r.conv]
        elif mode == "own":
            targets = self.conv_written[r.conv]
        elif mode == "index":
            # an index of stored pages by session: zero only pages no other open session's latest chain holds
            held = set()
            for conv, lr in self.live_chain.items():
                held.update(lr.pages)
            targets = [p for p in (list(r.pages) + self.conv_written[r.conv]) if p not in held]
        elif mode == "tail":
            # no index: spare the session's first prompt (where prefixes shared across sessions live)
            r0 = self.d["first_req"][r.conv]
            keep = set(r0.pages)
            targets = [p for p in (list(r.pages[r0.q:]) + self.conv_written[r.conv]) if p not in keep]
        elif mode in ("tier", "tier_index", "tier_shared", "tier_count"):
            # three classes by file time: the ended session's pages past its first prompt get time zero; its first
            # prompt (a possible head for sessions not yet here) is demoted below every open session's pages, the
            # head newest; a later use refreshes it (c). "tier" needs no index and demotes only the first-prompt
            # pages the session wrote itself; "tier_index" demotes every first-prompt page that no other open
            # session's latest chain holds (reference counts over open sessions' chains).
            r0 = self.d["first_req"][r.conv]
            keep = set(r0.pages)
            targets = [p for p in (list(r.pages[r0.q:]) + self.conv_written[r.conv]) if p not in keep]
            if mode == "tier":
                own = set(self.conv_written[r.conv])
                demote = [(j, p) for j, p in enumerate(r0.pages) if p in own]
            else:
                held = set()
                for conv, lr in self.live_chain.items():
                    held.update(lr.pages)
                targets = [p for p in targets if p not in held]
                demote = [(j, p) for j, p in enumerate(r0.pages) if p not in held]
                if mode == "tier_shared":
                    # a 'used by a second session' bit per page: a head others have used keeps its time
                    since = self.d["shared_since"]
                    demote = [(j, p) for j, p in demote if since[p] > t]
                elif mode == "tier_count":
                    # a use count per stored head page: among demoted heads, the least shared go first
                    ut = self.d["user_times"]
                    q0 = r0.q
                    for j, p in demote:
                        if mt[p] > 0:
                            cnt = 1 + bisect.bisect_right(ut.get(p, ()), t)
                            mt[p] = 1e-3 * min(cnt, 8) + (t + (q0 - j) * 1e-6) * 1e-9
                    demote = []
            q0 = r0.q
            for j, p in demote:
                if mt[p] > 0:
                    mt[p] = 1e-3 + (t + (q0 - j) * 1e-6) * 1e-9
        elif mode == "priv":
            # an index with a 'used by a second session' bit: zero only pages no other session has used so far
            since = self.d["shared_since"]
            npp = self.d["n_prompt_pages"]
            targets = [p for p in (list(r.pages) + self.conv_written[r.conv]) if p >= npp or since[p] > t]
        else:
            raise ValueError(mode)
        z = 0
        for pid in targets:
            if mt[pid] > 0:
                mt[pid] = 0.0
                z += 1
        self.zeroed += z

    # ----- cleaner -----
    def ev_tick(self, t, _a, _b):
        if t > self.t_last:
            return
        if self.stored >= self.hi * self.cap:
            self.push(t, "pass_snap", len(self.pass_log), None)
            self.push(t + self.pass_secs + self.tick, "tick")
        else:
            self.push(t + self.tick, "tick")

    def ev_pass_snap(self, t, k, groups):
        secs = self.d["passes"][k][1] if self.passes_mode == "logged" else self.pass_secs
        # the scan reads the file times over most of the pass: take them at mid-scan; unlinks come at the end
        self.push(t + 0.5 * self.del_frac * secs, "pass_scan", k, None)
        self.push(t + self.del_frac * secs, "pass_del", k, (groups, t))

    def ev_pass_scan(self, t, k, _):
        self.scan = array("d", self.mt)

    def next_use_table(self, t):
        """Next time each page is in the prompt of a request not yet prefilled (requests in progress count as now)."""
        nu = array("d", [math.inf]) * self.d["n_ids"]
        for r in self.d["reqs"]:
            if r.t_pre <= t:
                continue
            u = max(r.t_send, t)
            pages = r.pages
            for j in range(r.q - 1, -1, -1):
                p = pages[j]
                if nu[p] <= u:
                    break
                nu[p] = u
        return nu

    def next_need_table(self, t):
        """Next time a recorded lookup or read (after t, or still in progress at t) asks the SSD for each page."""
        nn = array("d", [math.inf]) * self.d["n_ids"]
        for r in self.d["reqs"]:
            if r.t_pre <= t:
                continue
            for lk in r.lookups:
                t_last = lk.read_done if lk.read_done is not None else lk.t
                if t_last < t:
                    continue
                u = max(lk.t, t)
                end = min(lk.m + lk.n, len(lk.chain), max(lk.useful, lk.m))
                ch = lk.chain
                for j in range(lk.m, end):
                    p = ch[j]
                    if nn[p] > u:
                        nn[p] = u
        return nn

    def ev_pass_del(self, t, k, info):
        groups, t_snap = info
        mt = self.mt
        scan = self.scan
        n_ids = self.d["n_ids"]
        cand = [p for p in range(n_ids) if scan[p] >= 0 and mt[p] >= 0]
        if self.policy.get("recheck"):
            # the cleaner stats a file again before the unlink and skips one whose time moved since the scan
            cand = [p for p in cand if mt[p] <= scan[p]]
        order = self.policy["order"]
        nu = self.next_use_table(t)
        nn = self.next_need_table(t)
        seed = self.policy.get("tie_seed", 0)
        tie = lambda p: ((p + seed) * 2654435761) & 0xFFFFFFFF    # files with equal times go in directory-scan order
        if order == "mtime":
            cand.sort(key=lambda p: (scan[p], tie(p)))
        elif order == "oracle_use":
            cand.sort(key=lambda p: (-nu[p], scan[p]))
        elif order == "oracle_need":
            cand.sort(key=lambda p: (-nn[p], -nu[p], scan[p]))
        else:
            raise ValueError(order)
        if groups is None:
            want = self.stored - int(self.lo * self.cap) + 1
            groups = max(0, min(len(cand), ((want + 49) // 50) * 50))
        victims = cand[:groups]
        n_zero = sum(1 for p in cand if scan[p] == 0.0)
        n_demoted = sum(1 for p in cand if 0.0 < scan[p] < 0.01)
        live_use = live_need = zero_t = shared = 0
        sh = self.d.get("sharing")
        for p in victims:
            if nu[p] != math.inf:
                live_use += 1
                if sh is not None and sh[p] >= 2:
                    shared += 1
            if nn[p] != math.inf:
                live_need += 1
            if scan[p] == 0.0:
                zero_t += 1
            self.del_mt[p] = scan[p]
            mt[p] = -1.0
            self.del_pass[p] = k
        self.stored -= len(victims)
        newest = scan[victims[-1]] if victims else None
        self.pass_log.append(dict(k=k, t=t, t_snap=t_snap, deleted=len(victims), live_use=live_use, live_shared=shared,
                                  live_need=live_need, zero_time=zero_t, stored_after=self.stored, cand=len(cand),
                                  newest_deleted=newest, stored_zero_time=n_zero, stored_demoted=n_demoted))
        self.scan = None


# ---------------------------------------------------------------------------------------------------------------
def summarize(sim, ref_out, lo_t, hi_t):
    """Losses against a reference outcome (a replay in which nothing is deleted), for requests sent in [lo_t, hi_t)."""
    reqs = sim.d["reqs"]
    n = lost_req = 0
    tok_ref = tok = 0
    by_kind = collections.Counter()
    tok_kind = collections.Counter()
    first_n = first_tok = 0
    for r in reqs:
        if not (lo_t <= r.t_send < hi_t) or r.rid not in sim.out:
            continue
        n += 1
        c, fl = sim.out[r.rid]
        c0 = ref_out[r.rid][0] if r.rid in ref_out else c
        h = min(r.H, r.q)
        rec = max(0, h - c) * PAGE
        rec0 = max(0, h - c0) * PAGE
        tok += rec
        tok_ref += rec0
        if rec - rec0 >= 64:
            lost_req += 1
            kind = fl["kind"] if fl else "other"
            by_kind[kind] += 1
            tok_kind[kind] += rec - rec0
            if r.k == 0:
                first_n += 1
                first_tok += rec - rec0
    extra = tok - tok_ref
    ttft = N_USERS * GAMMA * extra / max(1, n)
    return dict(requests=n, recomputed_tokens=tok, recomputed_ref=tok_ref, extra_tokens=extra, lost_requests=lost_req,
                by_kind=dict(by_kind), tok_kind=dict(tok_kind), ttft_s=ttft, lost_first_requests=first_n,
                extra_tokens_first_requests=first_tok)
