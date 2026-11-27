"""Workload for kvsim: the program trace reduced to pages, the replay client's schedule and the gap hints.

Standard library plus numpy. Three parts:

1. build_cache(): one streaming pass over a program trace (scripts/program_trace.py format, "blocks" content). Every
   tree whose largest request (prompt + output) fits MAX_CONTEXT is kept, as replay_programs.load_trees does, and
   reduced to what a cache simulator needs: per request the block counts, the recorded wait, the output length, the
   hint key, the dependencies, and the ids of the 64-token pages it adds. The result is a pickle next to this file.

   Pages. The trace's blocks are 16 tokens and block ids are chained (equal ids at a position mean equal prefixes).
   The server caches whole 64-token pages, so page j of a prompt is identified by the id of block 4j+3. A prompt of n
   blocks plus a tail of unique tokens (under one block) holds n // 4 shareable pages; the page that starts in the
   leftover blocks and runs into the tail and the generated output is unique to the request (its "twig").

2. Workload: the replay client's deterministic part for one seed (replay_programs.py run_arrivals): the tree order
   (random.Random(f"{seed}:order").shuffle), the Poisson arrival times (random.Random(f"{seed}:arrivals")), and for
   the plays that can arrive before the deadline a static radix trie over their pages (nodes are runs of pages cut at
   every point where two prompts diverge), with each request's path through it.

3. Hints: the online gap table of gap_hints.OnlineGapTable (per agent system, empirical ln-gap quantiles per key with
   the key -> tool -> "*" fallback, at least MIN_POOLED samples) and exact hints.
"""
from __future__ import annotations

import bisect
import gzip
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRACE = "/home/wanhr/sglang/agent_cache/data/converted/freeinference_16w.programs.jsonl.gz"
PAGE = 64
BLOCKS_PER_PAGE = 4          # 16-token blocks
GAP_KINDS = {None: 0, "tool": 1, "user": 2, "spawn": 3, "join": 4}
GAP_NAMES = {v: k for k, v in GAP_KINDS.items()}

# gap_hints.py constants (kept equal; the client is not imported so that this module has no path dependencies).
MIN_POOLED = 5
ANY_KEY = "*"
END_KEY = "_end"
NO_CALL_KEY = "_no_call"
KEY_SEP = "/"
GAP_FLOOR_S = 0.01
TAUS = (0.1, 0.5, 0.9)


# ===== Trace cache =====

def cache_path(trace: str, max_context: int) -> str:
    name = os.path.basename(trace).split(".")[0]
    return os.path.join(HERE, f"trace_cache_{name}_ctx{max_context}.pkl")


def _peak_tokens(tree: dict, block_tokens: int) -> int:
    peak = 0
    for st in tree["streams"]:
        for r in st["requests"]:
            n = (r["keep"] + len(r["new"])) * block_tokens + int(r.get("tail") or 0)
            peak = max(peak, n + r["out"])
    return peak


def _reduce_tree(tree: dict, block_tokens: int) -> dict:
    """One kept tree as arrays. new_pages[s][k] holds the ids of the pages request (s, k) adds after the pages it
    keeps from its base: the ids of blocks 4j+3 for keep // 4 <= j < n_blocks // 4."""
    streams = []
    for st in tree["streams"]:
        reqs = st["requests"]
        n = len(reqs)
        keep = np.zeros(n, np.int32)
        nblocks = np.zeros(n, np.int32)
        tail = np.zeros(n, np.int32)
        out = np.zeros(n, np.int32)
        wait = np.zeros(n, np.float64)
        t_rec = np.zeros(n, np.float64)
        gap = np.zeros(n, np.int8)
        base_s = np.full(n, -1, np.int32)
        base_k = np.full(n, -1, np.int32)
        keys, deps, new_pages = [], [], []
        for k, r in enumerate(reqs):
            kp = int(r["keep"]) if r.get("base") is not None else 0
            new = r["new"]
            nb = kp + len(new)
            keep[k], nblocks[k] = kp, nb
            tail[k] = int(r.get("tail") or 0)
            out[k] = int(r["out"])
            wait[k] = float(r.get("wait") or 0.0)
            t_rec[k] = float(r.get("t") or 0.0)
            gap[k] = GAP_KINDS.get(r.get("gap"), 0)
            if r.get("base") is not None:
                base_s[k], base_k[k] = r["base"]
            keys.append(r.get("key"))
            deps.append([tuple(d) for d in r["deps"]])
            j0, j1 = kp // BLOCKS_PER_PAGE, nb // BLOCKS_PER_PAGE
            new_pages.append(np.array([new[4 * j + 3 - kp] for j in range(j0, j1)], np.int64))
        streams.append({"role": st.get("role"), "keep": keep, "nblocks": nblocks, "tail": tail, "out": out,
                        "wait": wait, "t": t_rec, "gap": gap, "base_s": base_s, "base_k": base_k, "keys": keys,
                        "deps": deps, "new_pages": new_pages})
    return {"id": tree["tree"], "system": tuple(tree.get("system") or ()), "scope": str(tree.get("scope") or tree["tree"]),
            "harness": tree.get("harness"), "streams": streams}


def build_cache(trace: str = DEFAULT_TRACE, max_context: int = 65536, out: str | None = None,
                verbose: bool = True) -> str:
    """Stream the trace once and write the kept trees (file order) as a pickle; returns its path."""
    out = out or cache_path(trace, max_context)
    t0 = time.time()
    kept, dropped, n_req = [], 0, 0
    with gzip.open(trace, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        if header.get("content") != "blocks":
            raise ValueError(f"{trace}: content {header.get('content')}; kvsim reads 'blocks' traces only")
        B = int(header["block_tokens"])
        if B * BLOCKS_PER_PAGE != PAGE:
            raise ValueError(f"block_tokens {B}: expected {PAGE // BLOCKS_PER_PAGE}")
        for i, line in enumerate(f):
            if not line.strip():
                continue
            tree = json.loads(line)
            if max_context and _peak_tokens(tree, B) > max_context:
                dropped += 1
                continue
            red = _reduce_tree(tree, B)
            n_req += sum(len(s["out"]) for s in red["streams"])
            kept.append(red)
            if verbose and (i + 1) % 1000 == 0:
                print(f"  {i + 1} trees read, {len(kept)} kept, {time.time() - t0:.0f} s", flush=True)
    meta = {"trace": trace, "max_context": max_context, "block_tokens": B, "trees": len(kept), "dropped": dropped,
            "requests": n_req, "built_s": round(time.time() - t0, 1)}
    with open(out + ".tmp", "wb") as f:
        pickle.dump({"meta": meta, "trees": kept}, f, protocol=5)
    os.replace(out + ".tmp", out)
    if verbose:
        print(f"wrote {out}: {meta}", flush=True)
    return out


_CACHE: dict = {}


def load_cache(trace: str = DEFAULT_TRACE, max_context: int = 65536) -> dict:
    path = cache_path(trace, max_context)
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing: build it once with "
                                    f"'kvsim_trace.py --build --max-context {max_context}' (a few minutes)")
        with open(path, "rb") as f:
            _CACHE[path] = pickle.load(f)
    return _CACHE[path]


# ===== The client's schedule and the page trie of one run =====

class Req:
    """One request of one play. The static fields are set by Workload; the simulator adds its own."""

    def __init__(self):
        self.play = self.s = self.k = self.uid = None
        self.prompt = self.pages = self.out = self.wait = self.gap = self.key = self.deps = None
        self.path = ()
        self.twig = None
        self.twig_pages = 0
        self.keep_tokens = 0
        self.base = None
        self.is_last = False
        self.true_next_gap = None


class Play:
    __slots__ = ("pid", "tree_idx", "tree", "arrival", "t_start", "t_end", "streams", "n_left", "system", "scope",
                 "harness", "done_count")


class Workload:
    """The deterministic part of one replay: who arrives when, what each request holds, and the page trie.

    Node arrays (index 0 is the root, with no pages): node_pages, node_parent, node_children (lists). A request's
    `path` is the tuple of node ids from the root's child to the node its prompt ends in; `twig` is the node of its
    unique tail pages (None when the tail is under a page).
    """

    def __init__(self, *, trace: str = DEFAULT_TRACE, max_context: int = 65536, seed: int = 0, rate: float = 0.1,
                 duration: float = 7200.0, gap_scale: float = 1.0, gap_cap: float | None = 1200.0,
                 hint_no_call: bool = True, max_plays: int | None = None):
        cache = load_cache(trace, max_context)
        self.meta = cache["meta"]
        trees = cache["trees"]
        self.seed, self.rate, self.duration = seed, rate, duration
        self.gap_scale, self.gap_cap, self.hint_no_call = gap_scale, gap_cap, hint_no_call
        order = list(range(len(trees)))
        random.Random(f"{seed}:order").shuffle(order)
        rng = random.Random(f"{seed}:arrivals")
        self.plays: list[Play] = []
        t = 0.0
        for idx in order:
            if t >= duration or (max_plays is not None and len(self.plays) >= max_plays):
                break
            p = Play()
            p.pid, p.tree_idx, p.tree, p.arrival = len(self.plays), idx, trees[idx], t
            p.system, p.scope, p.harness = trees[idx]["system"], trees[idx]["scope"], trees[idx]["harness"]
            p.t_start = p.t_end = None
            self.plays.append(p)
            t += rng.expovariate(rate)
        self._build_trie()

    def scaled_wait(self, w: float) -> float:
        w = float(w) * self.gap_scale
        return w if self.gap_cap is None else min(w, self.gap_cap)

    # ----- trie -----

    def _build_trie(self) -> None:
        """Raw segments first (one per request that adds pages, attached at a page of an earlier segment), then
        cut at every attach point so that every path is made of whole nodes."""
        scope_ids: dict = {}
        page_at: dict = {}            # scope << 44 | page id -> (raw segment, offset)
        seg_parent = [(-1, 0)]        # raw segment -> (parent segment, pages of the parent before the attach point)
        seg_len = [0]
        cuts: list = [set()]          # raw segment -> offsets (pages from its start) where it is cut
        req_end = []                  # per request: (raw segment, pages of it used) of the prompt's last page
        uid = 0
        for p in self.plays:
            sc = scope_ids.setdefault(p.scope, len(scope_ids)) << 44
            tree = p.tree
            chains: dict = {}
            p.streams = []
            pending = [(s, k) for s, st in enumerate(tree["streams"]) for k in range(len(st["out"]))]
            reqs_by_key: dict = {}
            while pending:
                rest = []
                for s, k in pending:
                    st = tree["streams"][s]
                    bs, bk = int(st["base_s"][k]), int(st["base_k"][k])
                    if bs >= 0 and (bs, bk) not in chains:
                        rest.append((s, k))
                        continue
                    keep = int(st["keep"][k])
                    np_new = st["new_pages"][k]
                    j0 = keep // BLOCKS_PER_PAGE
                    base_chain = chains[(bs, bk)] if bs >= 0 else None
                    if base_chain is not None and j0 > len(base_chain):
                        j0 = len(base_chain)      # cannot happen for a well-formed trace; guard
                    chain = np_new if base_chain is None or j0 == 0 else np.concatenate((base_chain[:j0], np_new))
                    chains[(s, k)] = chain
                    # pages of np_new that an earlier request already created (chained ids: a prefix of them)
                    lo, hi = 0, len(np_new)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if (sc | int(np_new[mid - 1])) in page_at:
                            lo = mid
                        else:
                            hi = mid - 1
                    shared = lo
                    depth = j0 + shared           # pages of the chain that exist already
                    if depth > 0:
                        at = page_at[sc | int(chain[depth - 1])]
                        attach = (at[0], at[1] + 1)
                    else:
                        attach = (0, 0)
                    n_add = len(np_new) - shared
                    if n_add > 0:
                        seg = len(seg_len)
                        seg_parent.append(attach)
                        seg_len.append(n_add)
                        cuts.append(set())
                        if attach[0] > 0:
                            cuts[attach[0]].add(attach[1])
                        for i in range(n_add):
                            page_at[sc | int(np_new[shared + i])] = (seg, i)
                        end = (seg, n_add)
                    else:
                        end = attach
                    if end[0] > 0:
                        cuts[end[0]].add(end[1])
                    r = Req()
                    r.play, r.s, r.k, r.uid = p, s, k, uid
                    uid += 1
                    nb = int(st["nblocks"][k])
                    r.prompt = nb * 16 + int(st["tail"][k])
                    r.pages = nb // BLOCKS_PER_PAGE
                    r.out = int(st["out"][k])
                    r.wait = self.scaled_wait(st["wait"][k])
                    r.gap = int(st["gap"][k])
                    r.key = st["keys"][k]
                    r.deps = st["deps"][k]
                    r.keep_tokens = keep * 16
                    r.base = (bs, bk) if bs >= 0 else None
                    r.twig_pages = (r.prompt + r.out) // PAGE - r.pages
                    reqs_by_key[(s, k)] = r
                    req_end.append(end)
                if len(rest) == len(pending):
                    raise ValueError(f"tree {tree['id']}: unresolvable base references")
                pending = rest
            for s, st in enumerate(tree["streams"]):
                n = len(st["out"])
                lst = [reqs_by_key[(s, k)] for k in range(n)]
                for k, r in enumerate(lst):
                    r.is_last = k + 1 >= n
                    r.true_next_gap = None if r.is_last else lst[k + 1].wait
                p.streams.append(lst)
        # cut raw segments into nodes
        n_seg = len(seg_len)
        seg_first = [0] * n_seg       # node id of the first piece
        seg_bounds: list = [None] * n_seg
        node_pages, node_parent = [0], [-1]
        node_of_end: dict = {}        # (segment, end offset) -> node whose last page that is
        for seg in range(1, n_seg):
            bounds = sorted(c for c in cuts[seg] if 0 < c < seg_len[seg]) + [seg_len[seg]]
            seg_bounds[seg] = bounds
            prev_off = 0
            pseg, poff = seg_parent[seg]
            parent = 0 if pseg <= 0 else node_of_end[(pseg, poff)]
            seg_first[seg] = len(node_pages)
            for b in bounds:
                node = len(node_pages)
                node_pages.append(b - prev_off)
                node_parent.append(parent)
                node_of_end[(seg, b)] = node
                parent, prev_off = node, b
        # twigs and paths
        all_reqs = [r for p in self.plays for st in p.streams for r in st]
        all_reqs.sort(key=lambda r: r.uid)
        n_shared = len(node_pages)
        for r, end in zip(all_reqs, req_end):
            leaf = 0 if end[0] <= 0 else node_of_end[end]
            path = []
            n = leaf
            while n > 0:
                path.append(n)
                n = node_parent[n]
            path.reverse()
            r.path = tuple(path)
            if r.twig_pages > 0:
                r.twig = len(node_pages)
                node_pages.append(r.twig_pages)
                node_parent.append(leaf)
        self.node_pages = np.array(node_pages, np.int64)
        self.node_tokens = self.node_pages * PAGE
        self.node_parent = np.array(node_parent, np.int64)
        self.n_nodes = len(node_pages)
        self.n_shared_nodes = n_shared
        self.requests = all_reqs
        # sanity: a path's pages add up to the prompt's shareable pages
        for r in all_reqs[:: max(1, len(all_reqs) // 200)]:
            if sum(node_pages[n] for n in r.path) != r.pages:
                raise AssertionError(f"path pages {sum(node_pages[n] for n in r.path)} != {r.pages} for request "
                                     f"{r.play.pid}/{r.s}/{r.k}")

    def stats(self) -> dict:
        reqs = self.requests
        return {"plays": len(self.plays), "requests": len(reqs), "nodes": self.n_nodes,
                "prompt_mean": float(np.mean([r.prompt for r in reqs])), "out_mean": float(np.mean([r.out for r in reqs]))}


# ===== Hints =====

def key_levels(key):
    if key is None:
        return (ANY_KEY,)
    tool = key.split(KEY_SEP, 1)[0]
    if tool == key:
        return (tool, ANY_KEY)
    return (key, tool, ANY_KEY)


def _quantile(sorted_xs, p):
    pos = p * (len(sorted_xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def _sig(x: float) -> float:
    return float(f"{x:.6g}")


class OnlineGapTable:
    """gap_hints.OnlineGapTable for one agent system: ln-gaps per key level, kept sorted."""

    def __init__(self):
        self._xs: dict = {}

    def observe(self, key, gap_s: float) -> None:
        x = math.log(max(gap_s, GAP_FLOOR_S))
        for k in key_levels(key):
            bisect.insort(self._xs.setdefault(k, []), x)

    def pick(self, key):
        """(p10, p50, p90, n, matched key) of the entry the engine picks for `key`, or None (gap_hints.pick_entry:
        the key, else its tool, else "*"; an entry exists from MIN_POOLED samples on)."""
        cands = (key, key.split(KEY_SEP, 1)[0], ANY_KEY) if key is not None else (ANY_KEY,)
        for k in cands:
            xs = self._xs.get(k)
            if xs is not None and len(xs) >= MIN_POOLED:
                q = [_sig(math.exp(_quantile(xs, p))) for p in TAUS]
                return q[0], q[1], q[2], len(xs), k
        return None


class Hinter:
    """What the client sends with each request, per hint mode: 'none', 'program' (the session's next request id
    only), 'dist' (the online table's entry for the recorded reply's key and the end of a stream) or 'exact' (the
    next gap; nothing for a stream's last request or when the next send would fall past the deadline, as
    replay_programs.hints_for does)."""

    def __init__(self, mode: str, *, hint_no_call: bool, deadline: float):
        if mode not in ("none", "dist", "exact", "program"):
            raise ValueError(f"hint mode {mode}")
        self.mode, self.hint_no_call, self.deadline = mode, hint_no_call, deadline
        self.tables: dict = {}
        self.stats = {"sent": 0, "entry": 0, "end": 0, "none": 0}

    def reply_key(self, r: Req):
        return r.key or (NO_CALL_KEY if self.hint_no_call else None)

    def at_send(self, r: Req, now: float):
        """Called when request r is sent. Returns None or a dict: {'end': True} | {'exact': gap} |
        {'p10','p50','p90','n','key'}."""
        if self.mode == "none":
            return None
        self.stats["sent"] += 1
        if self.mode == "program":
            return {"program": True}       # the next request's id only: no gap, no end (the program-ID interface)
        if self.mode == "exact":
            if r.is_last:
                self.stats["none"] += 1
                return None
            gap = r.true_next_gap
            if now + gap >= self.deadline:
                self.stats["none"] += 1
                return None
            return {"exact": gap}
        table = self.tables.get(r.play.system)
        if table is None:
            table = self.tables[r.play.system] = OnlineGapTable()
        if r.k > 0:
            prev = r.play.streams[r.s][r.k - 1]
            table.observe(self.reply_key(prev), r.wait)
        if r.is_last:
            self.stats["end"] += 1
            return {"end": True}
        got = table.pick(self.reply_key(r))
        if got is None:
            self.stats["none"] += 1
            return None
        self.stats["entry"] += 1
        return {"p10": got[0], "p50": got[1], "p90": got[2], "n": got[3], "key": got[4]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="kvsim workload: build the trace cache, or print one seed's workload")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--max-context", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.build:
        build_cache(args.trace, args.max_context)
        return 0
    t0 = time.time()
    w = Workload(trace=args.trace, max_context=args.max_context, seed=args.seed)
    print(w.meta)
    print(w.stats(), f"built in {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
