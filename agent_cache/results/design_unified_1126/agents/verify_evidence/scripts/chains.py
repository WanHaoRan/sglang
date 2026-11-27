"""Block chains of the requests a run played, rebuilt from the trace (my own code).

load_chains(camp, arm) -> dict rid -> dict(blocks=[...], tail=int, scope=str, n_tokens=int, last=bool, nstream=int)
Page p of a prompt (64 tokens = 4 blocks of 16) is identified by (scope, blocks[4p+3]): block ids are chained, so
equal ids at a position mean equal prefixes up to it.  A page that reaches into the unique tail has no shared key.
"""
import gzip, json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

_TREES = None
def trees():
    global _TREES
    if _TREES is None:
        _TREES = {}
        with gzip.open(vlib.EXT + "/trees_played.jsonl.gz", "rt") as f:
            for line in f:
                d = json.loads(line)
                _TREES[d["tree"]] = d
    return _TREES

def tree_blocks(tree):
    out = {}
    def get(s, k):
        got = out.get((s, k))
        if got is None:
            r = tree["streams"][s]["requests"][k]
            base = r.get("base")
            got = list(r["new"]) if base is None else get(base[0], base[1])[: r["keep"]] + list(r["new"])
            out[(s, k)] = got
        return got
    return get

def load_chains(camp, arm):
    p = f"{vlib.EXT}/chains_c{camp}_{arm}.pkl"
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    T = trees()
    cl = vlib.client(camp, arm)
    getters = {}
    out = {}
    for q in cl["reqs"]:
        tree = T[q["tree"]]
        g = getters.get(q["tree"])
        if g is None:
            g = getters[q["tree"]] = tree_blocks(tree)
        s, k = q["stream"], q["k"]
        r = tree["streams"][s]["requests"][k]
        blocks = g(s, k)
        tail = int(r.get("tail") or 0)
        out[q["rid"]] = dict(blocks=blocks, tail=tail, scope=str(tree.get("scope") or tree["tree"]),
                             n_tokens=len(blocks) * 16 + tail, last=(k == len(tree["streams"][s]["requests"]) - 1),
                             nstream=len(tree["streams"][s]["requests"]))
    with open(p, "wb") as f:
        pickle.dump(out, f, protocol=4)
    return out

def page_keys(ch):
    """keys of the full 64-token prompt pages; a page reaching into the tail gets None (unique)."""
    nb = len(ch["blocks"])
    npages = ch["n_tokens"] // 64
    sc = ch["scope"]
    b = ch["blocks"]
    return [(sc, b[4 * p + 3]) if 4 * p + 3 < nb else None for p in range(npages)]

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        c, a = spec.split(":")
        camp = int(c)
        ch = load_chains(camp, a)
        cl = vlib.client(camp, a)
        bad_n = 0
        # ideal_cached: the longest prefix of blocks all completed earlier in the run in the same scope
        evs = []
        for q in cl["reqs"]:
            evs.append((q["t_send"], 0, q["rid"]))
            evs.append((q["t_send"] + q["latency"], 1, q["rid"]))
        evs.sort()
        seen = {}
        byrid = {q["rid"]: q for q in cl["reqs"]}
        ok = 0; bad = 0; diffs = []
        for t, kind, rid in evs:
            c_ = ch[rid]; q = byrid[rid]
            s = seen.setdefault(c_["scope"], set())
            if kind == 0:
                if c_["n_tokens"] != q["prompt_sent"]: bad_n += 1
                b = c_["blocks"]; lo, hi = 0, len(b)
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if b[mid - 1] in s: lo = mid
                    else: hi = mid - 1
                if lo * 16 == q["ideal_cached"]: ok += 1
                else: bad += 1; diffs.append(lo * 16 - q["ideal_cached"])
            else:
                s.update(c_["blocks"])
        print(f"c{camp} {a}: requests {len(cl['reqs'])}; prompt length mismatches {bad_n}; ideal_cached reproduced for {ok}, not for {bad} {sorted(diffs)[:5]}{sorted(diffs)[-5:]}")
