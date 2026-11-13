"""Shared helpers for the verification scripts: an independent loader on top of cache/*.pkl (see parse_cache.py).
Nothing here imports or executes the campaign's scripts."""
import collections, os, pickle, random, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
MARGIN = 300
STRETCHES = ((0, 3000), (3000, 5500), (5500, 7000), (7000, 8750), (8750, 11000))
EPISODES = ((5500, 7000), (8750, 11000))
ALL11 = ["baseline", "base_rep", "hintfree", "jit_p50", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend"]
LEAD5 = ["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"]
SEVEN = LEAD5 + ["jit_p50_ctl", "hintfree_ctl"]
CUT7 = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"]
CTL9 = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "nofc_noend", "ec_r13", "jit_exact"]
_raw = {}


def raw(label):
    if label not in _raw:
        _raw[label] = pickle.load(open(os.path.join(CACHE, label + ".pkl"), "rb"))
    return _raw[label]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def mean(v):
    v = list(v)
    return sum(v) / len(v) if v else float("nan")


def rec(r):
    return max(r.get("reply_reprefilled") or 0, 0)


_arm = {}


def arm(label, margin=MARGIN, cut_at=None):
    """Kept turns of a run by the stated rule, plus per-key server facts. cut_at (s) keeps t_send < cut_at instead."""
    key = (label, margin, cut_at)
    if key in _arm:
        return _arm[key]
    d = raw(label)
    c = d["client"]
    t0 = c["run"]["t_start"]
    every = sorted((r for r in c["turns"] if r["ttft"] is not None), key=lambda r: r["t_send"])
    end = every[-1]["t_send"] - margin if cut_at is None else cut_at
    turns = [r for r in every if r["t_send"] < end]
    out = dict(label=label, t0=t0, every=every, end=end, turns=turns, by={(r["conv"], r["turn"]): r for r in turns},
               by_all={(r["conv"], r["turn"]): r for r in every})
    if "server" in d:
        s = d["server"]
        queue, entry, fwd = {}, {}, {}
        for r in s["rts"]:
            if r["conv"] is not None:
                k = (r["conv"], r["turn"])
                queue[k] = r["queue_ms"] / 1000
                entry[k] = r["entry_time"]
                fwd[k] = r["forward_ms"] / 1000
        pages, reads, staged, refused = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
        ios = collections.defaultdict(list)
        for e in s["evts"].get("s2h_io", []):
            k = (e["conv"], e["turn"])
            pages[k] += e["pages"]
            reads[k] += 1
            ios[k].append((e["t"], e["ms"] / 1000, e["pages"]))
        for e in s["evts"].get("ntp_stage", []):
            staged[(e["conv"], e["turn"])] += 1
        for e in s["evts"].get("srs_dispatch", []):
            if e["launched"] == 0:
                refused[(e["conv"], e["turn"])] += 1
        out.update(queue=queue, entry=entry, fwd=fwd, pages=pages, reads=reads, staged=staged, refused=refused, ios=ios, server=s)
    _arm[key] = out
    return out


_rank = None


def rank():
    global _rank
    if _rank is None:
        c = raw("ORDER")["client"]
        order = sorted((r for r in c["turns"] if r["ttft"] is not None), key=lambda r: r["t_send"])
        _rank = {(r["conv"], r["turn"]): i for i, r in enumerate(order)}
    return _rank


def shared(names, arms=None):
    """Returning-turn keys kept in every named arm, in the order run's order (absent keys last)."""
    arms = arms or {a: arm(a) for a in names}
    rk = rank()
    return sorted(set.intersection(*(set(k for k in arms[a]["by"] if k[1] > 0) for a in names)), key=lambda k: (rk.get(k, 1 << 30), k))


def in_episode(k):
    p = rank().get(k, -1)
    return any(lo <= p < hi for lo, hi in EPISODES)


def boot(diffs_by_block, draws=2000, seed=0, q=(.05, .95)):
    """Block bootstrap as pairs_ec.py states it: blocks drawn with replacement, ratio of summed differences to summed counts."""
    cells = list(diffs_by_block.values())
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        pick = rng.choices(cells, k=len(cells))
        out.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    return sum(c[0] for c in cells) / sum(c[1] for c in cells), pct(out, q[0]), pct(out, q[1])


def blocks_by_rank(vals, block=500, absent="one"):
    """vals: {key: difference}. Blocks of `block` consecutive order-run positions; absent keys in one block (the script's
    behaviour), or absent='own' to chain them after the order run's last position in a given own order."""
    rk = rank()
    cells = {}
    for k, v in vals.items():
        b = rk.get(k, 1 << 30) // block
        s = cells.setdefault(b, [0.0, 0])
        s[0] += v
        s[1] += 1
    return cells
