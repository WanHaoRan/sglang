"""Test (f2): further alternatives on the readings.
A. First turns (turn 0), which every reading leaves out: per arm on the first turns LEAD5 shares; all-turn means.
B. The rest of the response: latency and decode time (latency - ttft) on the shared returning turns.
C. Concentration in time: the share of each pairwise difference carried by single 250-position bins (one congestion event).
D. What a repeat of the same arm moves inside the episodes, in seconds and in % of the episode mean, against the gaps
   between the LEAD5 arms there.
E. R2: turns reading from SSD, nofc_noend against nofc: by stretch, by session, and what a repeat moves.
F. R3: the share with other statistics and other reference arms.
  nice -n 19 python3 -I -B tf_more.py"""
import itertools, math, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
NPOS = len(rank)
ARMS = [a for a in C.RUNS if a != "order"]
D = {a: C.load(a) for a in ARMS}
B = {a: C.by_key(C.kept(D[a])) for a in ARMS}
READ = {a: {} for a in ARMS}
for a in ARMS:
    for t, c, n, pages, tok, ms in D[a]["s2h"]:
        READ[a][(c, n)] = READ[a].get((c, n), 0) + pages
keys = C.shared_keys(B, C.LEAD5, rank)
ep = [k for k in keys if C.in_episode(rank, k)]

print("A. first turns (turn 0) kept in all five LEAD5 arms")
first = sorted(set.intersection(*(set(k for k in B[a] if k[1] == 0) for a in C.LEAD5)))
print(f"  {len(first)} first turns: per arm mean TTFT, median, p90, max, turns over 5 s | mean server queue wait s | mean cached tokens")
for a in C.LEAD5:
    v = [B[a][k]["ttft"] for k in first]
    q = [D[a]["req"].get(f"{k[0]}-0", [0])[0] / 1000 for k in first]
    print(f"    {a:11s} {C.mean(v):6.3f} {C.pct(v, .5):6.3f} {C.pct(v, .9):6.2f} {max(v):6.1f} {sum(1 for x in v if x > 5):4d} | {C.mean(q):6.3f} | {C.mean(B[a][k]['cached'] or 0 for k in first):7.0f}")
print("  paired first-turn difference against nofc: mean, without the 5 largest |differences|, arm slower / faster")
for a in ("jit_exact", "ec_r13", "ec_r1", "nofc_noend"):
    d = sorted((B[a][k]["ttft"] - B["nofc"][k]["ttft"] for k in first), key=abs)
    print(f"    {a:11s} {C.mean(d):+6.3f}  {C.mean(d[:-5]):+6.3f}  {sum(1 for x in d if x > 0)}/{sum(1 for x in d if x < 0)}")
allk = sorted(set.intersection(*(set(B[a]) for a in C.LEAD5)))
print(f"  all shared turns, first turns included ({len(allk)}): " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in allk):.3f}" for a in C.LEAD5))
print(f"  each arm's own kept returning turns (not shared): " + "  ".join(f"{a} {C.mean(t['ttft'] for k, t in B[a].items() if k[1] > 0):.3f} ({sum(1 for k in B[a] if k[1] > 0)})" for a in C.LEAD5))

print("\nB. the rest of the response on the 11198 shared returning turns: mean latency s, mean decode time (latency - ttft) s, decode ms per completion token (median); in the episodes")
for a in C.LEAD5:
    dec = [B[a][k]["latency"] - B[a][k]["ttft"] for k in keys]
    per = [1000 * (B[a][k]["latency"] - B[a][k]["ttft"]) / B[a][k]["ct"] for k in keys if B[a][k]["ct"] >= 20]
    dece = [B[a][k]["latency"] - B[a][k]["ttft"] for k in ep]
    print(f"    {a:11s} latency {C.mean(B[a][k]['latency'] for k in keys):6.3f} | decode {C.mean(dec):6.3f} | {C.pct(per, .5):6.2f} ms/token | episodes: latency {C.mean(B[a][k]['latency'] for k in ep):6.3f} decode {C.mean(dece):6.3f}")
print("  paired against jit_exact: ttft difference, decode-time difference, latency difference (s per turn)")
for a in C.LEAD5[1:]:
    print(f"    {a:11s} ttft {C.mean(B[a][k]['ttft'] - B['jit_exact'][k]['ttft'] for k in keys):+6.3f}  decode {C.mean((B[a][k]['latency'] - B[a][k]['ttft']) - (B['jit_exact'][k]['latency'] - B['jit_exact'][k]['ttft']) for k in keys):+6.3f}  "
          f"latency {C.mean(B[a][k]['latency'] - B['jit_exact'][k]['latency'] for k in keys):+6.3f}")

print("\nC. concentration in time: 250-position bins of the order run (absent turns in 250-turn bins by consensus order); for each pair the three bins carrying most of the summed difference")
present = [k for k in keys if k in rank]
absent = sorted((k for k in keys if k not in rank), key=lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))
bins = [(f"{lo}", [k for k in present if lo <= rank[k] < lo + 250]) for lo in range(0, NPOS, 250)] + [(f"abs{i // 250:02d}", absent[i:i + 250]) for i in range(0, len(absent), 250)]
bins = [b for b in bins if b[1]]
for a, b in itertools.combinations(C.LEAD5, 2):
    total = sum(B[b][k]["ttft"] - B[a][k]["ttft"] for k in keys)
    per = sorted(((sum(B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks), label, len(ks)) for label, ks in bins), key=lambda x: -abs(x[0]))
    top = per[:3]
    two = sum(x[0] for x in per if x[1] in ("8750", "9000"))
    n2 = sum(x[2] for x in per if x[1] in ("8750", "9000"))
    rest = (total - two) / (len(keys) - n2)
    print(f"  {b:10s} - {a:10s} mean {total / len(keys):+6.3f} | top bins: " + ", ".join(f"{x[1]} {100 * x[0] / total:+6.1f} %" for x in top)
          + f" | bins 8750+9000 ({n2} turns, {100 * n2 / len(keys):.1f} % of turns) carry {100 * two / total:+6.1f} %; without them {rest:+6.3f}")
print("  minutes the bins 8750 and 9000 span per arm (first to last send of their shared turns): " + "  ".join(
    f"{a} {min(B[a][k]['t_send'] for l, ks in bins if l in ('8750', '9000') for k in ks) / 60:.1f}-{max(B[a][k]['t_send'] for l, ks in bins if l in ('8750', '9000') for k in ks) / 60:.1f}" for a in C.LEAD5))

print("\nD. what a repeat of the same arm moves inside the episodes (positions 5500-6999 and 8750-10999), on the turns the two runs share")
for name, a, b in (("original HiCache", "baseline", "base_rep"), ("hint-free", "hintfree", "hintfree_ctl"), ("JIT p50", "jit_p50", "jit_p50_ctl")):
    ks = C.shared_keys(B, [a, b], rank)
    for label, lo, hi in (("ep1", 5500, 7000), ("ep2", 8750, 11000), ("both", None, None)):
        sel = [k for k in ks if (C.in_episode(rank, k) if lo is None else lo <= rank.get(k, -1) < hi)]
        if len(sel) < 50:
            print(f"  {name:16s} {label}: {len(sel)} shared turns, too few")
            continue
        ma, mb = C.mean(B[a][k]["ttft"] for k in sel), C.mean(B[b][k]["ttft"] for k in sel)
        print(f"  {name:16s} {label}: {len(sel):5d} turns  {a} {ma:6.3f}  {b} {mb:6.3f}  apart {abs(ma - mb):6.3f} s = {100 * abs(ma - mb) / min(ma, mb):5.1f} % of the lower")
    calm = [k for k in ks if not C.in_episode(rank, k)]
    ma, mb = C.mean(B[a][k]["ttft"] for k in calm), C.mean(B[b][k]["ttft"] for k in calm)
    print(f"  {name:16s} elsewhere: {len(calm):5d} turns  {ma:6.3f}  {mb:6.3f}  apart {abs(ma - mb):6.3f} s = {100 * abs(ma - mb) / min(ma, mb):5.1f} %")
print("  gaps between LEAD5 arms in the same windows, on their 11198 shared turns:")
for label, lo, hi in (("ep1", 5500, 7000), ("ep2", 8750, 11000), ("both", None, None)):
    sel = [k for k in keys if (C.in_episode(rank, k) if lo is None else lo <= rank.get(k, -1) < hi)]
    m = {a: C.mean(B[a][k]["ttft"] for k in sel) for a in C.LEAD5}
    print(f"    {label}: " + "  ".join(f"{a} {m[a]:.3f}" for a in C.LEAD5) + " | " + "; ".join(
        f"{b}-{a} {m[b] - m[a]:+.3f} ({100 * (m[b] - m[a]) / min(m[a], m[b]):+.0f} %)" for a, b in (("jit_exact", "ec_r13"), ("ec_r13", "ec_r1"), ("ec_r13", "nofc"), ("ec_r13", "nofc_noend"), ("nofc", "nofc_noend"))))

print("\nE. R2: turns reading from SSD (s2h pages > 0), nofc_noend against nofc")
k2 = C.shared_keys(B, ["nofc", "nofc_noend"], rank)
for title, ks in (("their 11295 shared turns", k2), ("LEAD5's 11198", keys)):
    rn = [k for k in ks if READ["nofc_noend"].get(k, 0) > 0]
    rf = [k for k in ks if READ["nofc"].get(k, 0) > 0]
    both = len(set(rn) & set(rf))
    print(f"  {title}: nofc_noend {len(rn)} ({100 * len(rn) / len(ks):.1f} %), nofc {len(rf)} ({100 * len(rf) / len(ks):.1f} %): {100 * (len(rn) / len(rf) - 1):+.1f} %, {100 * (len(rn) - len(rf)) / len(ks):+.1f} points; "
          f"read in both {both}, only nofc_noend {len(rn) - both}, only nofc {len(rf) - both}; pages {sum(READ['nofc_noend'].get(k, 0) for k in ks)} against {sum(READ['nofc'].get(k, 0) for k in ks)} "
          f"({100 * (sum(READ['nofc_noend'].get(k, 0) for k in ks) / sum(READ['nofc'].get(k, 0) for k in ks) - 1):+.1f} %)")
print("  by stretch of the order run (reading turns nofc_noend / nofc):")
for label, sel in (("0-2999", lambda p: 0 <= p < 3000), ("3000-5499", lambda p: 3000 <= p < 5500), ("5500-6999", lambda p: 5500 <= p < 7000), ("7000-8749", lambda p: 7000 <= p < 8750),
                   ("8750-10999", lambda p: 8750 <= p < 11000), ("absent", lambda p: p < 0)):
    ks = [k for k in k2 if sel(rank.get(k, -1))]
    rn = sum(1 for k in ks if READ["nofc_noend"].get(k, 0) > 0)
    rf = sum(1 for k in ks if READ["nofc"].get(k, 0) > 0)
    print(f"    {label:11s} {len(ks):5d} turns: {rn:4d} / {rf:4d}  ({100 * (rn / rf - 1) if rf else float('nan'):+6.1f} %)")
sess = {}
for k in k2:
    s = sess.setdefault(k[0], [0, 0])
    s[0] += READ["nofc_noend"].get(k, 0) > 0
    s[1] += READ["nofc"].get(k, 0) > 0
more = sum(1 for s in sess.values() if s[0] > s[1])
less = sum(1 for s in sess.values() if s[0] < s[1])
top5 = sorted(sess.values(), key=lambda s: -(s[0] - s[1]))[:5]
tot = sum(s[0] - s[1] for s in sess.values())
print(f"  sessions {len(sess)}: more reading turns in nofc_noend {more}, fewer {less}, equal {len(sess) - more - less}; the 5 sessions with the largest surplus carry {sum(s[0] - s[1] for s in top5)} of the {tot} extra reading turns; "
      f"without them {100 * ((sum(s[0] for s in sess.values()) - sum(s[0] for s in top5)) / (sum(s[1] for s in sess.values()) - sum(s[1] for s in top5)) - 1):+.1f} %")
print("  what a repeat of the same arm moves, reading turns on the turns the two runs share:")
for a, b in (("hintfree", "hintfree_ctl"), ("jit_p50", "jit_p50_ctl"), ("jit_p50_ctl", "jit_p50_d_cut"), ("baseline", "base_rep")):
    ks = C.shared_keys(B, [a, b], rank)
    ra = sum(1 for k in ks if READ[a].get(k, 0) > 0)
    rb = sum(1 for k in ks if READ[b].get(k, 0) > 0)
    print(f"    {a} {ra} against {b} {rb} on {len(ks)} turns: {100 * (rb / ra - 1):+.1f} %")
print("  the other arms that read at or near arrival, on LEAD5's turns: " + "  ".join(f"{a} {sum(1 for k in keys if READ[a].get(k, 0) > 0)}" for a in C.LEAD5))

print("\nF. R3 on ALL11's shared turns: share of the way to jit_exact that nofc_noend covers, by statistic and by reference arm")
k11 = C.shared_keys(B, C.ALL11, rank)
stat = {"mean": C.mean, "median": lambda v: C.pct(list(v), .5), "p90": lambda v: C.pct(list(v), .9), "p99": lambda v: C.pct(list(v), .99),
        "mean capped at 10 s": lambda v: C.mean(min(x, 10) for x in v), "mean capped at 30 s": lambda v: C.mean(min(x, 30) for x in v)}
for name, f in stat.items():
    m = {a: f([B[a][k]["ttft"] for k in k11]) for a in C.ALL11}
    print(f"  {name:20s} " + "  ".join(f"from {r} ({m[r]:.3f}) {100 * (m[r] - m['nofc_noend']) / (m[r] - m['jit_exact']):5.1f} %" for r in ("baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl"))
          + f" | nofc_noend {m['nofc_noend']:.3f} jit_exact {m['jit_exact']:.3f}")
calm = [k for k in k11 if not C.in_episode(rank, k)]
ep11 = [k for k in k11 if C.in_episode(rank, k)]
for name, ks in (("episode 5500-6999", ep11), ("elsewhere", calm)):
    m = {a: C.mean(B[a][k]["ttft"] for k in ks) for a in C.ALL11}
    print(f"  mean, {name:18s} ({len(ks)} turns) " + "  ".join(f"from {r} ({m[r]:.3f}) {100 * (m[r] - m['nofc_noend']) / (m[r] - m['jit_exact']):5.1f} %" for r in ("baseline", "base_rep", "hintfree", "hintfree_ctl"))
          + f" | nofc_noend {m['nofc_noend']:.3f} jit_exact {m['jit_exact']:.3f}")
