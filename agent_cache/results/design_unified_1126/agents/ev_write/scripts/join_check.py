"""For requests whose commit names their prompt node: is the op the join gave them that node's, and if not, was the
node's op submitted at the same instant (a swap between two requests of one prefill batch, harmless for timing)?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
camp, label = sys.argv[1], sys.argv[2]
m = load(camp, label, load_trees())
by_node = {o["node"]: o for o in m.ops.values()}
tail = {}
for rid, st in m.reqstat.items():
    for t, node, _ in m.reg.get(rid, []):
        if abs(t - st["t"]) < 0.05: tail[rid] = node
n = right = same_t = other = 0; worst = []
for rid, run_ops in m.prompt_ops.items():
    r = m.reqs[rid]
    if rid not in tail or tail[rid] not in by_node or (r.P + r.out) // PAGE != r.P // PAGE: continue
    n += 1
    got = m.ops[run_ops[-1]]
    true = by_node[tail[rid]]
    if got["node"] == true["node"]: right += 1
    elif abs(got["t_sub"] - true["t_sub"]) < 0.002 and got["tokens"] == true["tokens"]: same_t += 1
    else:
        other += 1; worst.append((rid, got["node"], got["t_sub"], got["tokens"], true["node"], true["t_sub"], true["tokens"], true["rid"], true["part"]))
print(f"{camp} {label}: scored {n}; right node {right}; another node of the same size submitted in the same millisecond {same_t}; other {other}")
for w in worst[:10]: print("  ", w)
