"""Test (f12): smaller alternatives.
A. A cache left over from an earlier run: cached and storage tokens on the first turns each run served.
B. The host as a confound for the two runs that finished after the task's list was written (other work ran on the host
   during minutes 4-33 of jit_p50_d): client send slack, client-to-server entry time and a placebo (turns with no queue,
   no SSD read, at most 64 recomputed tokens in every arm), by run and by 10-minute window.
C. R1's order under other weightings: every session weighing the same; the share of block-bootstrap draws (250-position
   bins) in which each order holds.
D. R3 with first turns included and on each run's own kept turns.
E. R2: the tie in the mean, turn by turn: without the single 250-position bin where the two rivals differ most, by
   quantile, and what one bin moved between the two runs of rules 1 and 3.
  nice -n 19 python3 -I -B tg_misc.py"""
import itertools, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rank = C.order_rank()
NPOS = len(rank)
SIX = C.LEAD5 + ["ec_r13_rep"]
NAMES = sorted(set(C.ALL11 + SIX + ["jit_p50_d", "jit_p50_d_cut"]))
D = {a: C.load(a) for a in NAMES}
B = {a: C.by_key(C.kept(D[a])) for a in NAMES}
READ = {a: L.read_pages(D[a]) for a in NAMES}

print("A. first turns each run served: the first request of the run (cached, storage tokens), then the first ten sessions' first turns: cached tokens > 0, storage tokens > 0, and the first s2h read (s after the start)")
for a in sorted(NAMES, key=lambda a: D[a]["t0"]):
    turns = sorted(D[a]["turns"], key=lambda t: t["t_send"])
    f = turns[0]
    first10 = [t for t in turns if t["turn"] == 0][:10]
    print(f"  {a:14s} first request conv {f['conv']} turn {f['turn']} prompt {f['pt']} cached {f['cached']} storage {f['sto']} host {f['host']} | first ten first turns: cached>0 {sum(1 for t in first10 if (t['cached'] or 0) > 0)}, "
          f"storage>0 {sum(1 for t in first10 if (t['sto'] or 0) > 0)}, mean cached {C.mean(t['cached'] or 0 for t in first10):.0f} | first SSD read at {min((x[0] - D[a]['t0'] for x in D[a]['s2h']), default=float('nan')):.0f} s")

print("\nB. the client and host as instruments, per run: send slack ms (mean, p99, max) | server entry - client send ms (median, p90, p99) on returning turns | the same on turns with under 20K prompt tokens (median)")


def slack(d):
    conv = {}
    for t in sorted(d["turns"], key=lambda t: t["t_send"]):
        conv.setdefault(t["conv"], []).append(t)
    out = []
    for v in conv.values():
        for p, t in zip(v, v[1:]):
            if p["latency"] is not None and t["turn"] == p["turn"] + 1:
                out.append((t["t_send"], 1000 * (t["t_send"] - (p["t_send"] + p["latency"] + (t["gap"] or 0)))))
    return out


def entry(d):
    out = []
    for t in d["turns"]:
        r = d["req"].get(f"{t['conv']}-{t['turn']}")
        if r and t["turn"] > 0:
            out.append((t["t_send"], 1000 * (r[2] - (d["t0"] + t["t_send"])), t["pt"] or 0))
    return out


HOSTRUNS = ["jit_exact", "jit_p50_ctl", "ec_r13", "ec_r1", "nofc", "nofc_noend", "jit_p50_d_cut", "jit_p50_d", "ec_r13_rep"]
SL = {a: slack(D[a]) for a in HOSTRUNS}
EN = {a: entry(D[a]) for a in HOSTRUNS}
for a in HOSTRUNS:
    s = [x[1] for x in SL[a]]
    e = [x[1] for x in EN[a]]
    small = [x[1] for x in EN[a] if x[2] < 20000]
    print(f"  {a:14s} {C.mean(s):5.2f} {C.pct(s, .99):5.2f} {max(s):6.1f} | {C.pct(e, .5):6.1f} {C.pct(e, .9):7.1f} {C.pct(e, .99):7.1f} | {C.pct(small, .5):6.1f} ({len(small)})")
print("  by 10-minute window of each run's clock: median server entry - client send ms, and send slack p99 ms (other work ran on the host in minutes 4-33 of jit_p50_d: check scripts and plotting, 20:49-21:18Z)")
for a in ("jit_p50_ctl", "jit_p50_d_cut", "jit_p50_d", "ec_r13", "ec_r13_rep"):
    cells = []
    for m in range(0, 120, 10):
        e = [x[1] for x in EN[a] if m * 60 <= x[0] < (m + 10) * 60]
        s = [x[1] for x in SL[a] if m * 60 <= x[0] < (m + 10) * 60]
        cells.append(f"{C.pct(e, .5):5.1f}/{C.pct(s, .99):4.1f}" if e else "     -    ")
    print(f"    {a:14s} " + " ".join(cells))
keys = C.shared_keys(B, SIX, rank)


def clean(a, k):
    r = D[a]["req"].get(f"{k[0]}-{k[1]}")
    return r is not None and r[0] < 10 and READ[a].get(k, 0) == 0 and C.rec(B[a][k]) <= 64


ks = [k for k in keys if all(clean(a, k) for a in SIX)]
print(f"  placebo: {len(ks)} of the six runs' {len(keys)} shared returning turns had no queue wait (under 10 ms), no SSD read and at most 64 recomputed tokens in every one of the six: mean TTFT, median, paired mean difference to ec_r13")
for a in SIX:
    v = [B[a][k]["ttft"] for k in ks]
    print(f"    {a:11s} {C.mean(v):.4f} {C.pct(v, .5):.4f} {C.mean(B[a][k]['ttft'] - B['ec_r13'][k]['ttft'] for k in ks):+.4f}")
J = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_p50_d"]
kj = C.shared_keys(B, J, rank)
ksj = [k for k in kj if all(clean(a, k) for a in J)]
print(f"  placebo, the four JIT p50 runs: {len(ksj)} of {len(kj)} shared turns clean in all four: " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in ksj):.4f}" for a in J)
      + "; of them sent in minutes 4-33 of jit_p50_d: " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in ksj if 240 <= B['jit_p50_d'][k]['t_send'] < 1980):.4f}" for a in J)
      + f" ({sum(1 for k in ksj if 240 <= B['jit_p50_d'][k]['t_send'] < 1980)} turns)")

print("\nC. R1's order under other weightings, LEAD5's shared returning turns")
k5 = C.shared_keys(B, C.LEAD5, rank)
sess = {}
for k in k5:
    sess.setdefault(k[0], []).append(k)
print(f"  {len(k5)} turns, {len(sess)} sessions; turns per session: median {L.med([len(v) for v in sess.values()]):.0f}, max {max(len(v) for v in sess.values())}")
print("  every turn the same weight:    " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in k5):.3f}" for a in C.LEAD5))
print("  every session the same weight: " + "  ".join(f"{a} {C.mean(C.mean(B[a][k]['ttft'] for k in v) for v in sess.values()):.3f}" for a in C.LEAD5))
print("  median of the session means:   " + "  ".join(f"{a} {L.med([C.mean(B[a][k]['ttft'] for k in v) for v in sess.values()]):.3f}" for a in C.LEAD5))
bins = [b[2] for b in L.pos_bins(k5, rank, NPOS, 250, lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))]
cells = [[sum(B[a][k]["ttft"] for k in b) for a in C.LEAD5] + [len(b)] for b in bins]
rng = random.Random(0)
orders = {}
rank_count = {a: [0] * 5 for a in C.LEAD5}
N = 4000
for _ in range(N):
    pick = rng.choices(cells, k=len(cells))
    n = sum(c[5] for c in pick)
    m = [sum(c[i] for c in pick) / n for i in range(5)]
    o = tuple(sorted(range(5), key=lambda i: m[i]))
    orders[o] = orders.get(o, 0) + 1
    for pos, i in enumerate(o):
        rank_count[C.LEAD5[i]][pos] += 1
print(f"  block bootstrap over {len(bins)} bins of 250 positions (the turns' scatter inside these five runs; it holds nothing of what a second run moves), {N} draws:")
for o, c in sorted(orders.items(), key=lambda x: -x[1])[:5]:
    print(f"    {100 * c / N:5.1f} %  " + " < ".join(C.LEAD5[i] for i in o))
print("    share of draws at each rank (1 = lowest mean TTFT): " + "; ".join(f"{a} " + "/".join(f"{100 * c / N:.0f}" for c in rank_count[a]) for a in C.LEAD5))
k6 = keys
bins6 = [b[2] for b in L.pos_bins(k6, rank, NPOS, 250, lambda k: C.mean(B[a][k]["t_send"] for a in SIX))]
cells6 = [[sum(B[a][k]["ttft"] for k in b) for a in SIX] + [len(b)] for b in bins6]
rc = {a: [0] * 6 for a in SIX}
rng = random.Random(0)
for _ in range(N):
    pick = rng.choices(cells6, k=len(cells6))
    n = sum(c[6] for c in pick)
    m = [sum(c[i] for c in pick) / n for i in range(6)]
    for pos, i in enumerate(sorted(range(6), key=lambda i: m[i])):
        rc[SIX[i]][pos] += 1
print("    with the repeat as a sixth run, share of draws at each rank: " + "; ".join(f"{a} " + "/".join(f"{100 * c / N:.0f}" for c in rc[a]) for a in SIX))

print("\nD. R3 with first turns included, and on each run's own kept turns")
k11 = C.shared_keys(B, C.ALL11, rank)
all11 = sorted(set.intersection(*(set(B[a]) for a in C.ALL11)))
for title, f in (("shared returning turns", lambda a: C.mean(B[a][k]["ttft"] for k in k11)), ("shared turns, first turns included", lambda a: C.mean(B[a][k]["ttft"] for k in all11)),
                 ("own kept returning turns", lambda a: C.mean(t["ttft"] for k, t in B[a].items() if k[1] > 0)), ("own kept turns, all", lambda a: C.mean(t["ttft"] for t in B[a].values()))):
    m = {a: f(a) for a in C.ALL11}
    n = len(k11) if title.startswith("shared returning") else (len(all11) if title.startswith("shared turns") else 0)
    print(f"  {title:34s} {n if n else '':>5} | baseline {m['baseline']:.3f} base_rep {m['base_rep']:.3f} hintfree {m['hintfree']:.3f} hintfree_ctl {m['hintfree_ctl']:.3f} nofc_noend {m['nofc_noend']:.3f} jit_exact {m['jit_exact']:.3f} | "
          f"share of the way: nofc_noend {100 * (m['baseline'] - m['nofc_noend']) / (m['baseline'] - m['jit_exact']):.1f} % / {100 * (m['base_rep'] - m['nofc_noend']) / (m['base_rep'] - m['jit_exact']):.1f} %, "
          f"hint-free {100 * (m['baseline'] - m['hintfree']) / (m['baseline'] - m['jit_exact']):.1f}-{100 * (m['base_rep'] - m['hintfree']) / (m['base_rep'] - m['jit_exact']):.1f} %, "
          f"its repeat {100 * (m['baseline'] - m['hintfree_ctl']) / (m['baseline'] - m['jit_exact']):.1f}-{100 * (m['base_rep'] - m['hintfree_ctl']) / (m['base_rep'] - m['jit_exact']):.1f} %")
print("  where ALL11's shared turns end in each run (last send among them, minute) and the positions they cover: " + "  ".join(f"{a} {max(B[a][k]['t_send'] for k in k11) / 60:.0f}" for a in C.ALL11)
      + f"; positions up to {max(rank.get(k, -1) for k in k11)}; turns at positions 8750 or later: {sum(1 for k in k11 if rank.get(k, -1) >= 8750)}")

print("\nE. R2 turn by turn: nofc_noend against nofc on LEAD5's shared turns")
d = {k: B["nofc_noend"][k]["ttft"] - B["nofc"][k]["ttft"] for k in k5}
tot = sum(d.values())
lab = L.pos_bins(k5, rank, NPOS, 250, lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))
per = sorted(((sum(d[k] for k in ks_), label, len(ks_)) for label, lo, ks_ in lab), key=lambda x: -abs(x[0]))
print(f"  mean {tot / len(k5):+.3f} s; the three bins with the largest summed difference: " + ", ".join(f"{l} ({n} turns) {s / n:+.3f} s per turn" for s, l, n in per[:3]))
s0, l0, n0 = per[0]
print(f"  without bin {l0} ({n0} turns, {100 * n0 / len(k5):.1f} % of turns): {(tot - s0) / (len(k5) - n0):+.3f} s; without the two fixed windows: {C.mean(d[k] for k in k5 if not C.in_episode(rank, k)):+.3f} s")
kb = [k for label, lo, ks_ in lab if label == l0 for k in ks_]
print(f"  bin {l0}, mean TTFT per run: " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in kb if k in B[a]):.2f}" for a in SIX) + "  (the two runs of rules 1 and 3 are "
      f"{abs(C.mean(B['ec_r13'][k]['ttft'] for k in kb if k in B['ec_r13_rep']) - C.mean(B['ec_r13_rep'][k]['ttft'] for k in kb if k in B['ec_r13_rep'])):.2f} s apart there)")
print("  quantiles of TTFT, nofc / nofc_noend: " + "  ".join(f"p{int(q * 100)} {C.pct([B['nofc'][k]['ttft'] for k in k5], q):.3f}/{C.pct([B['nofc_noend'][k]['ttft'] for k in k5], q):.3f}" for q in (.5, .75, .9, .95, .99)))
print("  turns over 1, 2, 5, 10, 20 s, nofc / nofc_noend: " + "  ".join(f">{x} s {sum(1 for k in k5 if B['nofc'][k]['ttft'] > x)}/{sum(1 for k in k5 if B['nofc_noend'][k]['ttft'] > x)}" for x in (1, 2, 5, 10, 20)))
for cap in (2, 5, 10):
    print(f"  mean with TTFT capped at {cap} s: nofc {C.mean(min(B['nofc'][k]['ttft'], cap) for k in k5):.3f}, nofc_noend {C.mean(min(B['nofc_noend'][k]['ttft'], cap) for k in k5):.3f} ({C.mean(min(B['nofc_noend'][k]['ttft'], cap) - min(B['nofc'][k]['ttft'], cap) for k in k5):+.3f})")
