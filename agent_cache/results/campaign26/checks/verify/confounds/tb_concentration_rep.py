"""Test (b), calibration with the repeat of rules 1 and 3: the concentration statistics of tb_concentration.py for the
pair of runs of the SAME arm (ec_r13_rep - ec_r13) and for the repeat against every other LEAD5 arm, on the six runs'
shared returning turns. What these statistics return for two runs with no policy difference is the scale to read them by.
  nice -n 19 python3 -I -B tb_concentration_rep.py"""
import math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rank = C.order_rank()
SIX = C.LEAD5 + ["ec_r13_rep"]
B = {a: C.by_key(C.kept(C.load(a))) for a in SIX}
keys = C.shared_keys(B, SIX, rank)
ep = [k for k in keys if C.in_episode(rank, k)]
calm = [k for k in keys if not C.in_episode(rank, k)]


def analyse(a, b, ks, label, draws=2000):
    """Difference b - a per turn; same statistics and the same random seed as tb_concentration.py."""
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks}
    n = len(ks)
    total = sum(d.values())
    top = sorted(ks, key=lambda k: -abs(d[k]))[:max(1, n // 100)]
    top_sum = sum(d[k] for k in top)
    sess = {}
    for k in ks:
        s = sess.setdefault(k[0], [0.0, 0])
        s[0] += d[k]
        s[1] += 1
    sign = 1 if total >= 0 else -1
    by_abs = sorted(sess, key=lambda c: -abs(sess[c][0]))[:5]
    by_dir = sorted(sess, key=lambda c: -sign * sess[c][0])[:5]
    drop = lambda cs: (total - sum(sess[c][0] for c in cs)) / (n - sum(sess[c][1] for c in cs))
    loo = [(total - s[0]) / (n - s[1]) for s in sess.values()]
    lower = sum(1 for s in sess.values() if s[0] > 0)
    higher = sum(1 for s in sess.values() if s[0] < 0)
    order = sorted(sess, key=lambda c: -sign * sess[c][0])
    run_sum, run_n, flips = total, n, None
    for i, c in enumerate(order, 1):
        run_sum -= sess[c][0]
        run_n -= sess[c][1]
        if run_n <= 0 or sign * run_sum <= 0:
            flips = i
            break
    rng = random.Random(0)
    cells = list(sess.values())
    bs = []
    for _ in range(draws):
        pick = rng.choices(cells, k=len(cells))
        bs.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    print(f"  {label:9s} {b:11s} - {a:11s} n {n:5d} mean {total / n:+7.3f} | top 1 % ({len(top)} turns) carry {100 * top_sum / total:+7.1f} %, without them {(total - top_sum) / (n - len(top)):+7.3f} | "
          f"5 sessions by |sum| carry {100 * sum(sess[c][0] for c in by_abs) / total:+7.1f} %, without {drop(by_abs):+7.3f}; 5 in the direction carry {100 * sum(sess[c][0] for c in by_dir) / total:+7.1f} %, without {drop(by_dir):+7.3f} | "
          f"leave-one-session-out {min(loo):+7.3f} .. {max(loo):+7.3f} | sessions {len(sess)}: {a} lower in {lower}, higher in {higher} (p {L.p2(lower, lower + higher):.3g}) | sign flips after dropping {flips} sessions | "
          f"session bootstrap [{C.pct(bs, .05):+6.3f}, {C.pct(bs, .95):+6.3f}]")


print(f"{len(keys)} returning turns shared by the six runs ({len(set(k[0] for k in keys))} sessions); difference = second-named minus first-named")
for title, ks in (("all", keys), ("episodes", ep), ("elsewhere", calm)):
    print(f"\n{title}: the same arm twice, then the repeat against the other arms")
    analyse("ec_r13", "ec_r13_rep", ks, title)
    for a in ("jit_exact", "ec_r1", "nofc", "nofc_noend"):
        analyse(a, "ec_r13_rep", ks, title)
