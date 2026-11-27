"""Do the h2s_submit lines before each d2h_done line sum to the tokens of the d2h_submit lines in FIFO order?"""
import sys, os, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import events, kv, run_paths
camp, label = sys.argv[1], sys.argv[2]
_, _, extract = run_paths(camp, label)
subs = collections.deque(); pend = []; ok = bad = 0; examples = []
items = []
n_noops = 0
for t, ev, rest in events(extract, {"d2h_submit", "d2h_done", "h2s_submit", "PREFILL", "REQSTAT"}):
    d = kv(rest)
    if ev in ("PREFILL", "REQSTAT"):
        items.append((t, ev, d))
    elif ev == "d2h_submit":
        subs.append((t, int(d["nodes"]), int(d["tokens"]), [x for x in items if x[0] >= t - 0.005]))
        items = []
    elif ev == "h2s_submit":
        pend.append((t, int(d["op"]), int(d["node"]), int(d["tokens"])))
    elif ev == "d2h_done":
        k, T = int(d["nodes"]), int(d["tokens"])
        if not subs:
            bad += 1; pend = []; continue
        s = subs.popleft()
        tok = sum(x[3] for x in pend)
        if s[1] == k and s[2] == T and tok == T and len(pend) >= 1:
            ok += 1
        else:
            bad += 1
            if len(examples) < 8: examples.append((t, "done", k, T, "sub", s[:3], "ops", len(pend), tok))
        pend = []
print(f"{camp} {label}: transfers whose ops sum to the copy's tokens in FIFO order: {ok}; not: {bad}")
for e in examples: print("  ", e)
