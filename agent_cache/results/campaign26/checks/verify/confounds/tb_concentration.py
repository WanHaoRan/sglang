"""Test (b): concentration of the paired TTFT differences.
For each pair of arms on a shared set: the share of the paired mean difference carried by the top 1 % of turns by
absolute difference and by the 5 sessions with the largest summed difference, whether the sign survives dropping them,
the leave-one-session-out range, a sign test over sessions, how many top sessions must go before the sign flips, and a
bootstrap over sessions (5 %, 95 %).
  nice -n 19 python3 -I -B tb_concentration.py"""
import itertools, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
D = {a: C.load(a) for a in C.RUNS if a != "order"}
B = {a: C.by_key(C.kept(D[a])) for a in D}


def binom_two_sided(k, n):
    """Two-sided exact binomial p-value for k successes of n at p = 0.5."""
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyse(a, b, keys, label, draws=2000):
    """Difference b - a per turn."""
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in keys}
    n = len(keys)
    total = sum(d.values())
    m = total / n
    top = sorted(keys, key=lambda k: -abs(d[k]))[:max(1, n // 100)]
    top_sum = sum(d[k] for k in top)
    rest1 = (total - top_sum) / (n - len(top))
    sess = {}
    for k in keys:
        s = sess.setdefault(k[0], [0.0, 0, 0.0, 0.0])
        s[0] += d[k]
        s[1] += 1
        s[2] += B[a][k]["ttft"]
        s[3] += B[b][k]["ttft"]
    sign = 1 if total >= 0 else -1
    by_abs = sorted(sess, key=lambda c: -abs(sess[c][0]))[:5]
    by_dir = sorted(sess, key=lambda c: -sign * sess[c][0])[:5]
    drop = lambda cs: (total - sum(sess[c][0] for c in cs)) / (n - sum(sess[c][1] for c in cs))
    loo = [(total - s[0]) / (n - s[1]) for s in sess.values()]
    a_lower = sum(1 for s in sess.values() if s[2] < s[3])
    a_higher = sum(1 for s in sess.values() if s[2] > s[3])
    # how many sessions, taken in order of their contribution in the direction of the difference, before the sign flips
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
    # turn-level sign counts
    pos = sum(1 for v in d.values() if v > 0)
    neg = sum(1 for v in d.values() if v < 0)
    med = C.pct(list(d.values()), .5)
    print(f"  {label:9s} {b:13s} - {a:13s} n {n:5d} mean {m:+7.3f} | top 1 % ({len(top)} turns) carry {100 * top_sum / total:+7.1f} %, without them {rest1:+7.3f} | "
          f"5 sessions by |sum| carry {100 * sum(sess[c][0] for c in by_abs) / total:+7.1f} %, without {drop(by_abs):+7.3f}; 5 in the direction carry {100 * sum(sess[c][0] for c in by_dir) / total:+7.1f} %, without {drop(by_dir):+7.3f} | "
          f"leave-one-session-out {min(loo):+7.3f} .. {max(loo):+7.3f} | sessions {len(sess)}: {a} lower in {a_lower}, higher in {a_higher} (p {binom_two_sided(a_lower, a_lower + a_higher):.3g}) | "
          f"sign flips after dropping {flips} sessions | session bootstrap [{C.pct(bs, .05):+6.3f}, {C.pct(bs, .95):+6.3f}] | turns: {b} slower {pos}, faster {neg}, median diff {med:+.4f}")
    return m


keys = C.shared_keys(B, C.LEAD5, rank)
ep = [k for k in keys if C.in_episode(rank, k)]
calm = [k for k in keys if not C.in_episode(rank, k)]
print(f"LEAD5: {len(keys)} shared returning turns, {len(set(k[0] for k in keys))} sessions; difference = later-named minus earlier-named arm")
for a, b in itertools.combinations(C.LEAD5, 2):
    analyse(a, b, keys, "all")
print("\nLEAD5, the two load episodes only")
for a, b in itertools.combinations(C.LEAD5, 2):
    analyse(a, b, ep, "episodes")
print("\nLEAD5, elsewhere only")
for a, b in itertools.combinations(C.LEAD5, 2):
    analyse(a, b, calm, "elsewhere")

k2 = C.shared_keys(B, ["nofc", "nofc_noend"], rank)
print(f"\nR2 on the two rivals' own shared turns ({len(k2)})")
analyse("nofc", "nofc_noend", k2, "all")

k11 = C.shared_keys(B, C.ALL11, rank)
print(f"\nR3 pairs on ALL11's shared turns ({len(k11)}, {len(set(k[0] for k in k11))} sessions)")
for a, b in (("jit_exact", "nofc_noend"), ("nofc_noend", "baseline"), ("nofc_noend", "base_rep"), ("nofc_noend", "hintfree"), ("nofc_noend", "hintfree_ctl"), ("nofc_noend", "jit_p50_ctl"),
             ("baseline", "base_rep"), ("hintfree", "hintfree_ctl"), ("jit_p50_ctl", "jit_p50"), ("jit_exact", "ec_r13"), ("ec_r13", "nofc_noend")):
    analyse(a, b, k11, "all")

kc = C.shared_keys(B, C.CUT7, rank)
print(f"\nR4 pairs on CUT7's shared turns ({len(kc)}, {len(set(k[0] for k in kc))} sessions)")
for a, b in (("jit_p50_ctl", "jit_p50_d_cut"), ("jit_p50_d_cut", "jit_p50"), ("jit_p50_ctl", "jit_p50"), ("nofc_noend", "jit_p50_d_cut"), ("ec_r13", "jit_p50_d_cut"), ("jit_exact", "jit_p50_d_cut"),
             ("nofc_noend", "jit_p50_ctl"), ("ec_r13", "jit_p50_ctl")):
    analyse(a, b, kc, "all")
