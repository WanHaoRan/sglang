"""Check the page-chain reconstruction against the client's own ideal_cached (tokens an unbounded cache holds).
Usage: python3 -I -B check_ideal.py <arm>"""
import sys, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from c34lib import *
arm = sys.argv[1]
header, trees = load_trees()
t_start, reqs, plays, args = load_client(arm)
print(arm, "requests", len(reqs), "plays", len(plays), "block_tokens", header["block_tokens"], "cache_bust", args["cache_bust"])
inter = Interner()
chains = {}
# events: (time, kind, idx): send -> evaluate ideal; completion -> add blocks to seen (the client adds at completion,
# in file order)
seen_blocks = collections.defaultdict(set)   # scope -> block ids (the client's rule)
seen_pages = set()
ev = []
for i, r in enumerate(reqs):
    ev.append((r["t_send"], 0, i))
    if "error" not in r:
        ev.append((r["t_send"] + (r.get("latency") or 0.0), 1, i))
ev.sort()
bad_tok = bad_pg = n = 0
tail_ge16 = 0
ex = []
for t, kind, i in ev:
    r = reqs[i]
    tc = chains.get(r["play"])
    if tc is None:
        tc = chains[r["play"]] = TreeChains(trees[r["tree"]], inter)
    blocks = tc.blocks(r["stream"], r["k"])
    if kind == 1:
        seen_blocks[tc.scope].update(blocks)
        seen_pages.update(tc.pages(r["stream"], r["k"])[0])
        continue
    n += 1
    tr = trees[r["tree"]]["streams"][r["stream"]]["requests"][r["k"]]
    tail = int(tr.get("tail") or 0)
    if tail >= 16:
        tail_ge16 += 1
    assert r["prompt_sent"] == len(blocks) * 16 + tail, (r["rid"], r["prompt_sent"], len(blocks), tail)
    # the client's rule on block ids
    sb = seen_blocks[tc.scope]
    lo, hi = 0, len(blocks)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if blocks[mid - 1] in sb:
            lo = mid
        else:
            hi = mid - 1
    if lo * 16 != r["ideal_cached"]:
        bad_tok += 1
        if len(ex) < 5:
            ex.append((r["rid"], lo * 16, r["ideal_cached"]))
    # the page-hash rule: leading pages already completed by anyone
    ids = tc.pages(r["stream"], r["k"])[0]
    c = 0
    for p in ids:
        if p in seen_pages:
            c += 1
        else:
            break
    if c != r["ideal_cached"] // 64:
        bad_pg += 1
        if len(ex) < 10:
            ex.append(("pg", r["rid"], c * 64, r["ideal_cached"]))
print(f"requests checked {n}; ideal_cached by the client's block rule differs on {bad_tok}; by the page-hash chain "
      f"(pages) differs on {bad_pg}; tails >= 16 tokens: {tail_ge16}; distinct prompt pages {len(inter)}")
print("examples", ex)
