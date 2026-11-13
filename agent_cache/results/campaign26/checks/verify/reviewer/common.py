"""Shared loader for the reviewer scripts (reads cache_<label>.json written by parse_cache.py)."""
import json, os, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
CUT7 = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"]
EPISODES = ((5500, 7000), (8750, 11000))
MARGIN = 300
# turn row: conv, turn, t_send, ttft, latency, prompt, completion, cached, rec, gap_slept, dev, host, storage, p10, p50, p90, n, matched, true_gap, reply_key, source
F = {n: i for i, n in enumerate("conv turn t_send ttft latency prompt completion cached rec gap_slept dev host storage p10 p50 p90 n matched true_gap reply_key source".split())}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def mean(v):
    return st.fmean(v) if v else float("nan")


class Run:
    def __init__(self, label):
        d = json.load(open(os.path.join(HERE, f"cache_{label}.json")))
        self.label = label
        self.t0 = d["t0"]
        self.raw = d
        every = sorted((r for r in d["turns"] if r[F["ttft"]] is not None), key=lambda r: r[F["t_send"]])
        self.every = every
        self.end = every[-1][F["t_send"]] - MARGIN
        self.kept = [r for r in every if r[F["t_send"]] < self.end]
        self.by = {(r[0], r[1]): r for r in self.kept}
        self.queue = {tuple(int(x) for x in k.split("-")): v for k, v in d["queue"].items()}
        self.pages = {}
        self.reads = {}
        self.io = {}
        for c, t, tend, pages, ms in d["io"]:
            k = (c, t)
            self.pages[k] = self.pages.get(k, 0) + pages
            self.reads[k] = self.reads.get(k, 0) + 1
            self.io.setdefault(k, []).append((tend, pages, ms))
        self.staged = {tuple(int(x) for x in k.split("-")): v for k, v in d["staged"].items()}
        self.refused = {tuple(int(x) for x in k.split("-")): v for k, v in d["refused"].items()}
        self.arrive = {tuple(int(x) for x in k.split("-")): v for k, v in d["arrive"].items()}

    def ttft(self, k):
        return self.by[k][F["ttft"]]


def order_rank():
    d = json.load(open(os.path.join(HERE, "cache_ORDER.json")))
    every = sorted((r for r in d["turns"] if r[F["ttft"]] is not None), key=lambda r: r[F["t_send"]])
    return {(r[0], r[1]): i for i, r in enumerate(every)}


def shared(runs, labels, rank):
    keys = set.intersection(*(set(k for k in runs[a].by if k[1] > 0) for a in labels))
    return sorted(keys, key=lambda k: rank.get(k, 1 << 30))


def in_episode(rank, k):
    p = rank.get(k, -1)
    return any(lo <= p < hi for lo, hi in EPISODES)


def block_boot(diffs_by_key, rank, block=500, draws=2000, seed=0):
    """diffs_by_key: list of (key, diff). Returns mean, p5, p95 of a block bootstrap over `block` consecutive positions."""
    blocks = {}
    for k, d in diffs_by_key:
        s = blocks.setdefault(rank.get(k, 1 << 30) // block, [0.0, 0])
        s[0] += d
        s[1] += 1
    cells = list(blocks.values())
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        pick = rng.choices(cells, k=len(cells))
        out.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    return sum(c[0] for c in cells) / sum(c[1] for c in cells), pct(out, .05), pct(out, .95), blocks


def conv_boot(diffs_by_key, draws=2000, seed=0):
    cl = {}
    for k, d in diffs_by_key:
        s = cl.setdefault(k[0], [0.0, 0])
        s[0] += d
        s[1] += 1
    cells = list(cl.values())
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        pick = rng.choices(cells, k=len(cells))
        out.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    return pct(out, .05), pct(out, .95), len(cells)
