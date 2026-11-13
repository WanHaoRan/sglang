"""Test (a), detail: at the same turn, is the same set of sessions in progress in every arm, and at the same turn index?
For every 10th shared turn of LEAD5: the sessions in progress at its send time in each arm (sessions cut by the run's end
stay in progress), their overlap with jit_exact's set (Jaccard), and for the sessions in both sets the difference in the
turn index they have reached. Also: are sessions admitted in the same order?
  nice -n 19 python3 -I -B ta_sets.py"""
import bisect, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
D = {a: C.load(a) for a in C.LEAD5}
B = {a: C.by_key(C.kept(D[a])) for a in C.LEAD5}
keys = C.shared_keys(B, C.LEAD5, rank)
S = {}
for a in C.LEAD5:
    conv = {}
    for t in sorted(D[a]["turns"], key=lambda t: t["t_send"]):
        conv.setdefault(t["conv"], []).append(t)
    info = {}
    for c, v in conv.items():
        ce = D[a]["conv_end"].get(str(c))
        end = float("inf") if (ce is None or ce.get("capped")) else v[-1]["t_send"] + (v[-1]["latency"] or 0)
        info[c] = ([t["t_send"] for t in v], [t["turn"] for t in v], end)
    S[a] = info
print("admission order: sessions sorted by first send; position-by-position agreement with jit_exact over the sessions both started")
ref_order = sorted(S["jit_exact"], key=lambda c: S["jit_exact"][c][0][0])
for a in C.LEAD5[1:]:
    o = sorted(S[a], key=lambda c: S[a][c][0][0])
    n = min(len(o), len(ref_order))
    print(f"  {a:11s} same session at the same rank for {sum(1 for i in range(n) if o[i] == ref_order[i])} of {n}; first-send time difference to jit_exact (s): mean {C.mean(S[a][c][0][0] - S['jit_exact'][c][0][0] for c in o if c in S['jit_exact']):+.1f}, "
          f"max abs {max(abs(S[a][c][0][0] - S['jit_exact'][c][0][0]) for c in o if c in S['jit_exact']):.1f}")


def state(a, t):
    out = {}
    for c, (sends, idx, end) in S[a].items():
        if sends[0] <= t < end:
            out[c] = idx[bisect.bisect_right(sends, t) - 1]
    return out


sample = keys[::10]
print(f"\n{len(sample)} sampled shared turns: per arm against jit_exact: mean Jaccard overlap of the in-progress sets, share of turns with identical sets, mean sessions only in one set | "
      "sessions in both: mean and p5..p95 of the turn-index difference (arm - jit_exact)")
ep = set(k for k in keys if C.in_episode(rank, k))
for a in C.LEAD5[1:]:
    jac, same, only, dif = [], 0, [], []
    for k in sample:
        x = state(a, B[a][k]["t_send"])
        y = state("jit_exact", B["jit_exact"][k]["t_send"])
        inter = set(x) & set(y)
        union = set(x) | set(y)
        jac.append(len(inter) / len(union))
        same += set(x) == set(y)
        only.append(len(union) - len(inter))
        dif += [x[c] - y[c] for c in inter]
    print(f"  {a:11s} Jaccard {C.mean(jac):.4f}, identical sets {100 * same / len(sample):5.1f} %, {C.mean(only):.2f} sessions differ | turn index {C.mean(dif):+.3f} [{C.pct(dif, .05):+d}..{C.pct(dif, .95):+d}], "
          f"same index for {100 * sum(1 for v in dif if v == 0) / len(dif):.1f} % of session-moments")
