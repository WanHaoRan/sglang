"""Shared loaders for the ev_write checks (stdlib only).

Run: the client's records joined with the trace's block ids, so every prompt page has a key that is equal across
sessions exactly when the engine's chained page hash is: (scope, id of the page's last 16-token block). Block ids are
chained in the trace (equal ids at a position mean equal prefixes), the replay maps (scope, block) to fixed tokens,
and a prompt's whole pages never hold its unique tail (tail < 16 tokens, a page is 4 blocks).
"""
from __future__ import annotations

import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RESULTS = "/home/wanhr/sglang/agent_cache/results"
PAGE = 64
BLOCK = 16
RID = re.compile(r"-c(\d+)-t(\d+)$")


def manifest(campaign: str) -> dict:
    out = {}
    with open(f"{RESULTS}/{campaign}/manifest.txt") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                out[p[0]] = (p[1], p[2])
    return out


def run_paths(campaign: str, label: str) -> tuple:
    run, prog = manifest(campaign)[label]
    tag = campaign.replace("campaign", "c")
    return f"{RESULTS}/{run}", prog, f"{BASE}/extracts/{tag}_{label}.tsv"


def kv(rest: str) -> dict:
    return dict(x.split("=", 1) for x in rest.split() if "=" in x)


def events(path: str, want=None):
    """(t, event, rest) of an extract, optionally only the events in `want`."""
    with open(path) as f:
        for line in f:
            if line[0] == "#":
                continue
            t, ev, rest = line.rstrip("\n").split("\t", 2)
            if want is None or ev in want:
                yield float(t), ev, rest


def load_trees(path: str = f"{BASE}/extracts/trees_c32_c34.jsonl.gz") -> dict:
    trees = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        f.readline()
        for line in f:
            t = json.loads(line)
            trees[t["tree"]] = t
    return trees


class Blocks:
    """Block lists of one tree's requests, materialized on demand (base/keep/new)."""

    def __init__(self, tree: dict):
        self.tree = tree
        self.scope = str(tree.get("scope") or tree["tree"])
        self._b: dict = {}

    def blocks(self, s: int, k: int) -> list:
        got = self._b.get((s, k))
        if got is None:
            r = self.tree["streams"][s]["requests"][k]
            base = r.get("base")
            got = list(r["new"]) if base is None else self.blocks(*base)[: r["keep"]] + list(r["new"])
            self._b[(s, k)] = got
        return got


class Req:
    __slots__ = ("rid", "inst", "k", "play", "tree", "stream", "role", "t_send", "ttft", "latency", "P", "out",
                 "cached", "dev", "host", "stor", "ideal", "gap", "wait_s", "key", "true_gap", "reply_key", "last",
                 "pages", "n_stream", "t_first", "t_done", "error", "oracle_gap")


class Run:
    """One arm: requests by rid, sessions (stream instances) in order, plays, and page keys per request."""

    def __init__(self, campaign: str, label: str, trees: dict | None = None):
        self.campaign, self.label = campaign, label
        self.run_dir, self.prog, self.extract = run_paths(campaign, label)
        self.reqs: dict = {}
        self.sessions: dict = {}     # inst -> [Req] by k
        self.plays: dict = {}        # play -> record
        self.lanes: list = []
        trees = trees if trees is not None else load_trees()
        blocks: dict = {}
        self.scope_of: dict = {}
        with open(f"{self.run_dir}/{self.prog}/client.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                kind = rec.get("kind")
                if kind == "run":
                    self.t_start = rec["t_start"]
                    self.args = rec["args"]
                elif kind == "play":
                    self.plays[rec["play"]] = rec
                elif kind == "lanes":
                    self.lanes.append(rec)
                elif kind == "summary":
                    self.summary = rec
                elif kind == "req":
                    r = Req()
                    r.rid, r.inst, r.k, r.play, r.tree, r.stream = rec["rid"], rec["inst"], rec["k"], rec["play"], rec["tree"], rec["stream"]
                    r.role = rec.get("role")
                    r.t_send, r.ttft, r.latency = rec["t_send"], rec.get("ttft"), rec.get("latency")
                    r.P, r.out = rec.get("prompt_tokens") or rec.get("prompt_sent") or 0, rec.get("completion_tokens") or 0
                    r.cached = rec.get("cached_tokens") or 0
                    d = rec.get("cached_details") or {}
                    r.dev, r.host, r.stor = d.get("device", 0) or 0, d.get("host", 0) or 0, d.get("storage", 0) or 0
                    r.ideal = rec.get("ideal_cached") or 0
                    r.gap, r.wait_s, r.key = rec.get("gap"), rec.get("wait_s"), rec.get("key")
                    h = rec.get("hint") or {}
                    r.true_gap, r.reply_key = h.get("true_gap_s"), h.get("reply_key")
                    r.oracle_gap = rec.get("oracle_gap_s")
                    r.error = rec.get("error")
                    r.t_first = None if r.ttft is None else r.t_send + r.ttft
                    r.t_done = None if r.latency is None else r.t_send + r.latency
                    tree = trees[r.tree]
                    bl = blocks.get(r.tree)
                    if bl is None:
                        bl = blocks[r.tree] = Blocks(tree)
                    b = bl.blocks(r.stream, r.k)
                    n_pages = len(b) // 4
                    scope = bl.scope
                    r.pages = [(scope, b[4 * j + 3]) for j in range(n_pages)]
                    r.n_stream = len(tree["streams"][r.stream]["requests"])
                    r.last = r.k + 1 >= r.n_stream
                    self.reqs[r.rid] = r
                    self.sessions.setdefault(r.inst, []).append(r)
        for lst in self.sessions.values():
            lst.sort(key=lambda x: x.k)

    def sess_of(self, rid: str):
        m = RID.search(rid)
        return (int(m[1]), int(m[2])) if m else (None, None)
