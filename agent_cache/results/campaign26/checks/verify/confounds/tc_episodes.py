"""Test (c): episode alignment.
Mean TTFT by 250-position bin (positions of the order run) for each LEAD5 arm on the shared returning turns; the shared
turns absent from the order run are binned by their consensus order (mean send time over the five arms). Then: where each
arm's own slow bins are, what the fixed windows (5500-6999, 8750-10999) leave outside, and R1's episode means redone with
windows chosen from the data (bins whose mean over the five arms exceeds a threshold).
  nice -n 19 python3 -I -B tc_episodes.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NPOS = len(rank)
D = {a: C.load(a) for a in C.LEAD5}
B = {a: C.by_key(C.kept(D[a])) for a in C.LEAD5}
keys = C.shared_keys(B, C.LEAD5, rank)
present = [k for k in keys if k in rank]
absent = sorted((k for k in keys if k not in rank), key=lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))
print(f"LEAD5 shared returning turns {len(keys)}; in the order run {len(present)} (positions 0-{NPOS - 1}); absent {len(absent)}")
print(f"absent turns: mean send minute per arm " + " ".join(f"{a} {C.mean(B[a][k]['t_send'] for k in absent) / 60:.1f}" for a in C.LEAD5)
      + f"; earliest send minute {min(B[a][k]['t_send'] for a in C.LEAD5 for k in absent) / 60:.1f}; sessions {len(set(k[0] for k in absent))}")
last_pos_minute = {a: C.pct([B[a][k]["t_send"] for k in present if rank[k] >= NPOS - 200], .5) / 60 for a in C.LEAD5}
print("median send minute of the shared turns in the order run's last 200 positions: " + " ".join(f"{a} {last_pos_minute[a]:.1f}" for a in C.LEAD5))
frac_late = {a: sum(1 for k in absent if B[a][k]["t_send"] / 60 > last_pos_minute[a]) for a in C.LEAD5}
print("absent turns sent after that minute: " + " ".join(f"{a} {frac_late[a]}" for a in C.LEAD5))


def bins_order(width):
    """[(label, keys)] : order-run position bins, then the absent turns in bins of the same number of turns by consensus order."""
    out = []
    for lo in range(0, NPOS, width):
        ks = [k for k in present if lo <= rank[k] < lo + width]
        if ks:
            out.append((f"{lo:5d}", lo, ks))
    for i in range(0, len(absent), width):
        out.append((f"abs{i // width:02d}", None, absent[i:i + width]))
    return out


def fixed(lo):
    return lo is not None and any(a <= lo < b for a, b in C.EPISODES)


print("\n1. Mean TTFT by 250-position bin of the order run (absNN = turns absent from the order run, by consensus order): turns | per arm | mean of the five | F = inside the fixed windows, D = five-arm mean over 1 s")
print("  bin     turns | " + " ".join(f"{a:>10s}" for a in C.LEAD5) + " |  mean5 | minute(jit_exact)")
bins = bins_order(250)
rows = []
for label, lo, ks in bins:
    m = {a: C.mean(B[a][k]["ttft"] for k in ks) for a in C.LEAD5}
    m5 = C.mean(m.values())
    rows.append((label, lo, ks, m, m5))
    print(f"  {label}  {len(ks):5d} | " + " ".join(f"{m[a]:10.3f}" for a in C.LEAD5) + f" | {m5:6.3f} | {C.mean(B['jit_exact'][k]['t_send'] for k in ks) / 60:6.1f} {'F' if fixed(lo) else ' '}{'D' if m5 > 1 else ' '}")

print("\n2. Each arm's own slow bins (own bin mean over 1 s): bin labels")
for a in C.LEAD5:
    own = [r[0].strip() for r in rows if r[3][a] > 1]
    print(f"  {a:11s} {len(own):2d} bins: {' '.join(own)}")
allslow = [set(r[0] for r in rows if r[3][a] > 1) for a in C.LEAD5]
print(f"  bins slow in all five: {len(set.intersection(*allslow))}; slow in at least one: {len(set.union(*allslow))}; "
      f"slow in at least one and outside the fixed windows: {sorted(x.strip() for x in set.union(*allslow) if not fixed(next(r[1] for r in rows if r[0] == x)))}")
print("  peak bin per arm in each fixed window and its mean: " + "; ".join(
    f"{a} " + ", ".join(max(((r[3][a], r[0].strip()) for r in rows if r[1] is not None and lo <= r[1] < hi), default=(0, "-"))[1]
                        + f" ({max((r[3][a] for r in rows if r[1] is not None and lo <= r[1] < hi), default=0):.2f})" for lo, hi in C.EPISODES) for a in C.LEAD5))
print("  centre of the excess (TTFT over 0.4 s, summed) along the bin index, per arm, in each fixed window widened by 3 bins each side:")
for lo, hi in C.EPISODES:
    cells = []
    for a in C.LEAD5:
        num = den = 0.0
        for r in rows:
            if r[1] is not None and lo - 750 <= r[1] < hi + 750:
                ex = sum(max(B[a][k]["ttft"] - 0.4, 0) for k in r[2])
                num += ex * (r[1] + 125)
                den += ex
        cells.append(f"{a} {num / den:7.0f}")
    print(f"    window {lo}-{hi}: " + "  ".join(cells))

print("\n3. What the fixed windows leave outside: mean TTFT per arm in the bins next to them and in the absent turns")
for title, sel in (("positions 5000-5499", lambda lo: lo is not None and 5000 <= lo < 5500), ("positions 7000-7499", lambda lo: lo is not None and 7000 <= lo < 7500),
                   ("positions 8250-8749", lambda lo: lo is not None and 8250 <= lo < 8750), ("positions 7500-8249", lambda lo: lo is not None and 7500 <= lo < 8250),
                   ("absent from the order run", lambda lo: lo is None), ("positions 0-4999", lambda lo: lo is not None and lo < 5000)):
    ks = [k for r in rows if sel(r[1]) for k in r[2]]
    print(f"  {title:28s} {len(ks):5d} turns: " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in ks):6.3f}" for a in C.LEAD5))
print("  share of each arm's summed TTFT excess over 0.4 s that falls outside the fixed windows:")
for a in C.LEAD5:
    tot = sum(max(B[a][k]["ttft"] - 0.4, 0) for k in keys)
    out = sum(max(B[a][k]["ttft"] - 0.4, 0) for k in keys if not C.in_episode(rank, k))
    ab = sum(max(B[a][k]["ttft"] - 0.4, 0) for k in absent)
    print(f"    {a:11s} outside {100 * out / tot:5.1f} %  (of it in the absent turns {100 * ab / tot:5.1f} %); turns over 10 s: inside {sum(1 for k in keys if C.in_episode(rank, k) and B[a][k]['ttft'] > 10)}, outside {sum(1 for k in keys if not C.in_episode(rank, k) and B[a][k]['ttft'] > 10)}")


def redo(width, thr, by_consensus=False):
    if by_consensus:
        cons = sorted(keys, key=lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))
        bb = [(f"c{i}", i, cons[i:i + width]) for i in range(0, len(cons), width)]
    else:
        bb = bins_order(width)
    inside, labels = [], []
    for label, lo, ks in bb:
        if C.mean(C.mean(B[a][k]["ttft"] for k in ks) for a in C.LEAD5) > thr:
            inside += ks
            labels.append(label.strip())
    ins = set(inside)
    outside = [k for k in keys if k not in ins]
    mi = {a: C.mean(B[a][k]["ttft"] for k in inside) for a in C.LEAD5}
    mo = {a: C.mean(B[a][k]["ttft"] for k in outside) for a in C.LEAD5}
    kind = "consensus order" if by_consensus else "order-run positions"
    print(f"  bins of {width} by {kind}, five-arm mean over {thr} s: {len(labels)} bins, {len(inside)} turns inside, {len(outside)} outside")
    print(f"      bins: {' '.join(labels)}")
    print("      inside:  " + "  ".join(f"{a} {mi[a]:6.3f}" for a in C.LEAD5) + f"   | ec_r13 ahead of nofc {mi['nofc'] - mi['ec_r13']:+.3f}, of nofc_noend {mi['nofc_noend'] - mi['ec_r13']:+.3f}, of ec_r1 {mi['ec_r1'] - mi['ec_r13']:+.3f}; behind jit_exact {mi['ec_r13'] - mi['jit_exact']:+.3f}")
    print("      outside: " + "  ".join(f"{a} {mo[a]:6.3f}" for a in C.LEAD5) + f"   | range {max(mo.values()) - min(mo.values()):.3f} s")
    return mi, mo


print("\n4. R1's episode means redone with windows chosen from the data")
ep = [k for k in keys if C.in_episode(rank, k)]
calm = [k for k in keys if not C.in_episode(rank, k)]
print(f"  fixed windows: {len(ep)} turns inside, {len(calm)} outside")
print("      inside:  " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in ep):6.3f}" for a in C.LEAD5))
mo = {a: C.mean(B[a][k]['ttft'] for k in calm) for a in C.LEAD5}
print("      outside: " + "  ".join(f"{a} {mo[a]:6.3f}" for a in C.LEAD5) + f"   | range {max(mo.values()) - min(mo.values()):.3f} s")
calm2 = [k for k in calm if k in rank]
mo = {a: C.mean(B[a][k]['ttft'] for k in calm2) for a in C.LEAD5}
print(f"  fixed windows, outside without the absent turns ({len(calm2)} turns): " + "  ".join(f"{a} {mo[a]:6.3f}" for a in C.LEAD5) + f"   | range {max(mo.values()) - min(mo.values()):.3f} s")
for width in (250, 100, 500):
    for thr in (1.0, 0.75, 1.5):
        redo(width, thr)
for width in (250, 100):
    redo(width, 1.0, by_consensus=True)

print("\n5. Per-arm windows: each arm judged on its own slow bins (own 250-bin mean over 1 s) against the fixed window")
for a in C.LEAD5:
    own = [k for r in rows if r[3][a] > 1 for k in r[2]]
    rest = [k for r in rows if r[3][a] <= 1 for k in r[2]]
    print(f"  {a:11s} own slow bins: {len(own):5d} turns mean {C.mean(B[a][k]['ttft'] for k in own):6.3f}; other bins {len(rest):5d} turns mean {C.mean(B[a][k]['ttft'] for k in rest):6.3f}")
