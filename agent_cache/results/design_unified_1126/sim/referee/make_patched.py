"""Referee: build kvsim_patched.py from ../kvsim.py by exact text replacement (each must match once), so that the
patch is reproducible and the original file is never touched. Run: python -I -B make_patched.py

The defects fixed (see REFEREE_NOTES.md for the evidence of each):
  D1  the clock goes backwards during the grace period (a send scheduled at the deadline after the deadline)
  D2  an orphaned (unreachable, still locked) host node that is inserted again is counted once but held twice, or
      held once and never returned to the evictable set; a node with two writes pending is 'clean' after the first
  D3  the GPU pool is exceeded when a reply's pages are inserted and nothing can be evicted
  D4  the cleaner's order pops one stored node per pass that it does not delete; no later pass can find it
  D5  a read whose pages the cleaner deleted between the lookup and the I/O is served as if they were there
  D6  mirror off: host pages on the admitted request's own path can be evicted by the GPU eviction of the same
      admission and are then loaded from nowhere; host_mirror goes negative
  D7  the infinite-cache ideal is taken at the end of prefill from pages that were prefilled, not at the send from
      pages whose requests completed (the client's rule)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "kvsim.py")
DST = os.path.join(HERE, "kvsim_patched.py")

EDITS = [
    # the patched copy lives one folder down: the modules are next to the original
    ("path", '''sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kvsim_trace as kt  # noqa: E402''',
     '''sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # referee copy: modules are one up
import kvsim_trace as kt  # noqa: E402'''),
    # D2: orphan tokens as a count, writes pending per node
    ("D2 state", '''        self.orphan = [False] * n     # host pages under a deleted node: unreachable, held until their lock goes
''', '''        self.orphan = [False] * n     # host pages under a deleted node: unreachable, held until their lock goes
        self.orphan_tok = [0] * n     # [referee D2] tokens those copies hold (a node can be orphaned, inserted again and orphaned again)
        self.wpend = [0] * n          # [referee D2] SSD writes pending per node
        self.seen = [False] * n       # [referee D7] pages of requests whose reply is complete (the client's infinite cache)
'''),
    ("D2 hlock_dec", '''        if self.hlock[n] == 0:
            if self.orphan[n]:
                self.orphan[n] = False
                self.host_used -= self.tok[n]
                self.cnt["evict_host_tokens"] += self.tok[n]
            else:
                self._hc(n)
''', '''        if self.hlock[n] == 0:
            if self.orphan[n]:
                self.orphan[n] = False
                t = self.orphan_tok[n]              # [referee D2] every orphaned copy, once
                self.orphan_tok[n] = 0
                self.host_used -= t
                self.cnt["evict_host_tokens"] += t
                if self.in_host[n]:
                    self._hc(n)                     # [referee D2] the copy inserted since is evictable again
            else:
                self._hc(n)
'''),
    ("D2 orphan", '''        self.hcand.discard(c)
        self.in_host[c] = False
        self.orphan[c] = True
        self.mup[c] = False
''', '''        self.hcand.discard(c)
        self.in_host[c] = False
        self.orphan[c] = True
        self.orphan_tok[c] += self.tok[c]           # [referee D2]
        self.mup[c] = False
'''),
    ("D2 backup", '''        if self.pol.should_write(n, r):
            self.dirty[n] = True
            self._hlock_inc(n)
''', '''        if self.pol.should_write(n, r):
            self.dirty[n] = True
            self.wpend[n] += 1                      # [referee D2]
            self._hlock_inc(n)
'''),
    ("D2 write_ack", '''        self.dirty[n] = False
        self.unwritten[n] = failed
        self.pending_write -= t
''', '''        self.wpend[n] -= 1                          # [referee D2] dirty until its last pending write is acknowledged
        self.dirty[n] = self.wpend[n] > 0
        self.unwritten[n] = failed
        self.pending_write -= t
'''),
    # D1
    ("D1", '''        self.at(min(t, self.cfg.duration), self._send, r)
''', '''        self.at(max(self.now, min(t, self.cfg.duration)), self._send, r)   # [referee D1] never before now
'''),
    # D3
    ("D3", '''        if tw is not None:
            t = self.tok[tw]
            if self.C_G - self.gpu_used < t:
                self.evict_gpu(t)
            self.in_gpu[tw] = True
''', '''        if tw is not None and self.C_G - self.gpu_used < self.tok[tw] and not self.evict_gpu(self.tok[tw]):
            # [referee D3] no room and nothing evictable: the reply's pages are not cached (the pool is never exceeded)
            self.cnt["twig_dropped"] = self.cnt.get("twig_dropped", 0) + 1
            tw = None
            r.twig, r.twig_pages = None, 0
        if tw is not None:
            t = self.tok[tw]
            self.in_gpu[tw] = True
'''),
    # D4
    ("D4", '''        for n in self.pol.cleaner_order(target):
            if self.ssd_used <= target:
                break
            if not self.ssd[n]:
                continue
''', '''        order = iter(self.pol.cleaner_order(target))
        while self.ssd_used > target:               # [referee D4] test before taking the next node off the order
            n = next(order, None)
            if n is None:
                break
            if not self.ssd[n]:
                continue
'''),
    # D5
    ("D5 slot", '''                 "t_land", "pinned", "seq", "io_start", "rid")
''', '''                 "t_land", "pinned", "seq", "io_start", "rid", "lost_from")
'''),
    ("D5 init", '''        self.deadline = None
        self.pinned = False
''', '''        self.deadline = None
        self.pinned = False
        self.lost_from = None         # [referee D5] index of the first node whose SSD copy was gone when the I/O began
'''),
    ("D5 start", '''            self.r_cur, self.r_rem = rd, float(rd.tokens)
            rd.io_start = self.now
''', '''            self.r_cur, self.r_rem = rd, float(rd.tokens)
            rd.io_start = self.now
            rd.lost_from = next((i for i, n in enumerate(rd.nodes) if not self.ssd[n]), None)   # [referee D5]
'''),
    ("D5 land", '''        landed = []
        for n in rd.nodes:
            if self.in_host[n] or self.in_gpu[n]:
                if not self.in_host[n] and not self.cfg.mirror:
                    self.in_host[n] = True      # GPU-only under exclusive tiering: keep the copy that was read
                    landed.append(n)
                    continue
''', '''        landed = []
        for i_node, n in enumerate(rd.nodes):
            if rd.lost_from is not None and i_node >= rd.lost_from:
                # [referee D5] the cleaner deleted this page (or one before it) before the I/O began: nothing was read
                self.host_used -= self.tok[n]
                self.cnt["read_deleted_tokens"] = self.cnt.get("read_deleted_tokens", 0) + self.tok[n]
                continue
            if self.in_host[n] or self.in_gpu[n]:
                if not self.in_host[n] and not self.cfg.mirror:
                    self.in_host[n] = True      # GPU-only under exclusive tiering: keep the copy that was read
                    self.host_mirror += self.tok[n]     # [referee D6] it is a host copy of a GPU page now
                    landed.append(n)
                    continue
'''),
    # D6
    ("D6", '''        now = self.now
        for i in range(ig):
            self._glock_inc(path[i])
        if not self.evict_gpu(need):
            for i in range(ig):
                self._glock_dec(path[i])
            return 2
''', '''        now = self.now
        for i in range(ig):
            self._glock_inc(path[i])
        if not self.cfg.mirror:
            # [referee D6] mirror off: a GPU eviction copies pages to host and may evict host pages to make room;
            # the host pages this admission is about to load must not be among them
            for i in range(ig, it):
                self._hlock_inc(path[i])
        fits = self.evict_gpu(need)
        if not self.cfg.mirror:
            for i in range(ig, it):
                self._hlock_dec(path[i])
        if not fits:
            for i in range(ig):
                self._glock_dec(path[i])
            return 2
'''),
    # D7
    ("D7 send", '''        r.t_send = self.now
        r.hint = self.hinter.at_send(r, self.now)
''', '''        r.t_send = self.now
        # [referee D7] the client's ideal_cached: at the send, the leading pages of the prompt whose requests have
        # completed, and in 16-token blocks what it keeps of its base once that has completed
        ideal = 0
        for n in r.path:
            if not self.seen[n]:
                break
            ideal += self.tok[n]
        if r.base is not None and r.play.streams[r.base[0]][r.base[1]].t_done is not None:
            ideal = max(ideal, min(r.keep_tokens, r.prompt))
        r.ideal_at_send = ideal
        r.hint = self.hinter.at_send(r, self.now)
'''),
    ("D7 prefill", '''        r.ideal_seen = ideal_pages
''', '''        r.ideal_seen = r.ideal_at_send              # [referee D7]
'''),
    ("D7 done", '''    def _client_done(self, r) -> None:
        r.t_done = self.now
''', '''    def _client_done(self, r) -> None:
        r.t_done = self.now
        seen = self.seen                            # [referee D7]
        for n in r.path:
            seen[n] = True
'''),
]


def main() -> None:
    with open(SRC) as f:
        text = f.read()
    for name, old, new in EDITS:
        if text.count(old) != 1:
            raise SystemExit(f"edit {name}: the old text occurs {text.count(old)} times in kvsim.py (expected 1)")
        text = text.replace(old, new)
    header = ('"""REFEREE COPY of kvsim.py with the defects D1-D7 of referee/REFEREE_NOTES.md patched (built by '
              'referee/make_patched.py;\nthe original is untouched). Every changed line carries a [referee Dn] mark.\n\n')
    assert text.startswith('"""kvsim:')
    text = header + text[3:]
    with open(DST, "w") as f:
        f.write(text)
    print(f"wrote {DST}: {len(EDITS)} edits")


if __name__ == "__main__":
    main()
