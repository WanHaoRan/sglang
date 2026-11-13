"""Test (f10): what a second run of the same arm moves among the leading arms.
ec_r13_rep (rules 1 and 3 again, finished 2026-11-12 00:48Z, outside the task's list of runs) against ec_r13 and the
other LEAD5 arms: the same definitions as R1 (kept turns, shared returning turns, positions, fixed windows).
  nice -n 19 python3 -I -B tg_repeat.py"""
import itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rank = C.order_rank()
NPOS = len(rank)
SIX = C.LEAD5 + ["ec_r13_rep"]
NAMES = sorted(set(SIX + C.ALL11 + ["jit_p50_d", "jit_p50_d_cut"]))
D = {a: C.load(a) for a in NAMES}
B = {a: C.by_key(C.kept(D[a])) for a in NAMES}
READ = {a: L.read_pages(D[a]) for a in NAMES}
REP, REF = "ec_r13_rep", "ec_r13"

print("0. the two runs of rules 1 and 3")
units = L.gpu_units()
for a in (REF, REP):
    d = D[a]
    print(f"  {a:11s} {d['run']}  GPU unit(s) in launcher.log {units.get(a)}  turns {len(d['turns'])} errors {sum(1 for t in d['turns'] if t['err'])} null ttft {sum(1 for t in d['turns'] if t['ttft'] is None)} "
          f"kept {len(B[a])} last send {max(t['t_send'] for t in d['turns']) / 60:.1f} min")
sa, sb = L.server_args(REF), L.server_args(REP)
dif = sorted(k for k in set(sa) | set(sb) if sa.get(k, "<absent>") != sb.get(k, "<absent>"))
print(f"  server_args fields that differ ({len(dif)} of {len(set(sa) | set(sb))}): " + "; ".join(f"{k}: {str(sa.get(k))[-60:]} | {str(sb.get(k))[-60:]}" for k in dif))
ca, cb = D[REF]["args"], D[REP]["args"]
print("  client args that differ: " + ("; ".join(f"{k}: {ca.get(k)} | {cb.get(k)}" for k in sorted(set(ca) | set(cb)) if ca.get(k) != cb.get(k) and k != "output") or "none besides the output path"))


def windows(keys):
    ep1 = [k for k in keys if 5500 <= rank.get(k, -1) < 7000]
    ep2 = [k for k in keys if 8750 <= rank.get(k, -1) < 11000]
    calm = [k for k in keys if not C.in_episode(rank, k)]
    return (("all", keys), ("episodes", ep1 + ep2), ("ep1 5500-6999", ep1), ("ep2 8750-10999", ep2), ("elsewhere", calm),
            ("event 5750-6749", [k for k in keys if 5750 <= rank.get(k, -1) < 6750]), ("event 8750-9249", [k for k in keys if 8750 <= rank.get(k, -1) < 9250]))


keys = C.shared_keys(B, SIX, rank)
W = windows(keys)
print(f"\n1. the six runs on the {len(keys)} returning turns they share ({len(set(k[0] for k in keys))} sessions; LEAD5 alone shares {len(C.shared_keys(B, C.LEAD5, rank))})")
print("  arm          mean  median   p90    p99  >10s | " + " | ".join(f"{n} ({len(ks)})" for n, ks in W[1:]) + " | reading turns  pages | recomputed/turn | mean queue wait s")
for a in SIX:
    v = [B[a][k]["ttft"] for k in keys]
    q = C.mean(D[a]["req"].get(f"{k[0]}-{k[1]}", [0])[0] / 1000 for k in keys)
    print(f"  {a:11s} {C.mean(v):5.3f} {C.pct(v, .5):6.3f} {C.pct(v, .9):6.2f} {C.pct(v, .99):6.1f} {sum(x > 10 for x in v):5d} | "
          + " | ".join(f"{C.mean(B[a][k]['ttft'] for k in ks):6.3f}" for n, ks in W[1:])
          + f" | {sum(1 for k in keys if READ[a].get(k, 0) > 0):5d} {sum(READ[a].get(k, 0) for k in keys):7d} | {C.mean(C.rec(B[a][k]) for k in keys):5.0f} | {q:5.3f}")
m = {a: C.mean(B[a][k]["ttft"] for k in keys) for a in SIX}
print("  order by mean: " + " < ".join(f"{a} {m[a]:.3f}" for a in sorted(m, key=m.get)))
me = {a: C.mean(B[a][k]["ttft"] for k in W[1][1]) for a in SIX}
print("  order in the fixed episode windows: " + " < ".join(f"{a} {me[a]:.3f}" for a in sorted(me, key=me.get)))

bins250 = [b[2] for b in L.pos_bins(keys, rank, NPOS, 250, lambda k: C.mean(B[a][k]["t_send"] for a in SIX))]
bins500 = [b[2] for b in L.pos_bins(keys, rank, NPOS, 500, lambda k: C.mean(B[a][k]["t_send"] for a in SIX))]


def paired(a, b, ks, label):
    """b - a per turn on ks."""
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks}
    n = len(ks)
    tot = sum(d.values())
    kset = set(ks)
    b250 = [[k for k in bn if k in kset] for bn in bins250]
    b500 = [[k for k in bn if k in kset] for bn in bins500]
    lo250, hi250, opp = L.block_boot(d, b250)
    lo500, hi500, _ = L.block_boot(d, b500)
    slo, shi = L.session_boot(d)
    sess = {}
    for k in ks:
        s = sess.setdefault(k[0], [0.0, 0])
        s[0] += d[k]
        s[1] += 1
    loo = [(tot - s[0]) / (n - s[1]) for s in sess.values()]
    al = sum(1 for s in sess.values() if s[0] > 0)
    ah = sum(1 for s in sess.values() if s[0] < 0)
    bl = sum(1 for bn in b250 if bn and sum(d[k] for k in bn) > 0)
    bh = sum(1 for bn in b250 if bn and sum(d[k] for k in bn) < 0)
    top = sorted(ks, key=lambda k: -abs(d[k]))[:max(1, n // 100)]
    ts = sum(d[k] for k in top)
    print(f"  {label:16s} {b:11s} - {a:11s} n {n:5d} mean {tot / n:+6.3f} | block bootstrap 250 [{lo250:+6.3f}, {hi250:+6.3f}] 500 [{lo500:+6.3f}, {hi500:+6.3f}] opposite sign in {100 * opp:4.1f} % of draws | "
          f"session bootstrap [{slo:+6.3f}, {shi:+6.3f}] | leave-one-session-out {min(loo):+6.3f}..{max(loo):+6.3f} | sessions {a} lower {al} higher {ah} (p {L.p2(al, al + ah):.3g}) | "
          f"250-bins {a} lower {bl} higher {bh} (p {L.p2(bl, bl + bh):.3f}) | top 1 % ({len(top)} turns) carry {100 * ts / tot if tot else 0:+6.1f} %, without them {(tot - ts) / (n - len(top)):+6.3f} | "
          f"median diff {C.pct(list(d.values()), .5):+.4f}")
    return tot / n


print(f"\n2. the same arm run twice: {REP} - {REF} (same flags, same engine build, another GPU unit, 15.8 h later)")
own = C.shared_keys(B, [REF, REP], rank)
paired(REF, REP, own, f"own shared")
for n, ks in W:
    paired(REF, REP, ks, n)
print("  by 250-position bin (order run; absNN = shared turns absent from it): turns | ec_r13, ec_r13_rep, rep - ec_r13 | the other four")
for label, lo, ks in L.pos_bins(keys, rank, NPOS, 250, lambda k: C.mean(B[a][k]["t_send"] for a in SIX)):
    f = lambda a: C.mean(B[a][k]["ttft"] for k in ks)
    if max(f(a) for a in SIX) > 0.75:
        print(f"    {label:>6s} {len(ks):4d} | {f(REF):6.3f} {f(REP):6.3f} {f(REP) - f(REF):+6.3f} | " + " ".join(f"{a} {f(a):6.3f}" for a in ("jit_exact", "ec_r1", "nofc", "nofc_noend")))
d = {k: B[REP][k]["ttft"] - B[REF][k]["ttft"] for k in keys}
tot = sum(d.values())
ev = [k for k in keys if 8750 <= rank.get(k, -1) < 9250]
print(f"  share of the summed difference rep - ec_r13 carried by positions 8750-9249 ({len(ev)} turns, {100 * len(ev) / len(keys):.1f} %): {100 * sum(d[k] for k in ev) / tot:+.1f} %; without them {(tot - sum(d[k] for k in ev)) / (len(keys) - len(ev)):+.3f} s; "
      f"inside the fixed windows {100 * sum(d[k] for k in keys if C.in_episode(rank, k)) / tot:+.1f} %")

print("\n3. every pair of the six on the shared turns: row minus column, s per turn [5 %, 95 % block bootstrap over 250-position bins]; then the fixed episode windows; then elsewhere")
for n, ks in (W[0], W[1], W[4]):
    kset = set(ks)
    bb = [[k for k in bn if k in kset] for bn in bins250]
    print(f"  {n} ({len(ks)} turns)")
    for b in SIX[1:] + [SIX[0]]:
        cells = []
        for a in SIX:
            if a == b:
                continue
            dd = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks}
            lo, hi, _ = L.block_boot(dd, bb)
            cells.append(f"-{a} {C.mean(dd.values()):+6.3f} [{lo:+5.2f},{hi:+5.2f}]")
        print(f"    {b:11s} " + "  ".join(cells))

print("\n4. the campaign's rule for repeats (DECISIONS.md 15:20Z): a difference between two arms counts as shown when every pair of their runs has the same sign")
print("   and the smallest of those differences is larger than the largest move between two runs of either arm. Runs per arm here: rules 1+3 two, the others one.")
runs = {"jit_exact": ["jit_exact"], "rules 1+3": [REF, REP], "ec_r1": ["ec_r1"], "nofc": ["nofc"], "nofc_noend": ["nofc_noend"]}
for n, ks in (W[0], W[1], W[4]):
    move = abs(C.mean(B[REP][k]["ttft"] - B[REF][k]["ttft"] for k in ks))
    print(f"  {n} ({len(ks)} turns): the one same-arm move available among these arms, rules 1+3: {move:.3f} s")
    for x, y in itertools.combinations(runs, 2):
        ds = [C.mean(B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks) for a in runs[x] for b in runs[y]]
        same = all(v > 0 for v in ds) or all(v < 0 for v in ds)
        print(f"    {y:10s} - {x:10s} {' '.join(f'{v:+6.3f}' for v in ds)} | one sign: {'yes' if same else 'no'}; smallest |difference| {min(abs(v) for v in ds):.3f} "
              f"{'>' if min(abs(v) for v in ds) > move else '<='} the rules-1+3 move {move:.3f} -> {'shown' if same and min(abs(v) for v in ds) > move else 'not separated'}"
              + ("" if "rules 1+3" in (x, y) else " (neither arm has a repeat; the rules-1+3 move stands in)"))

print("\n5. is the repeat's deficit a heavier load at the same turns? (sessions cut by the run's end kept in progress) mean sessions in progress | live context K | in flight | send minute | mean TTFT")
LS = {a: L.load_state(D[a], True) for a in SIX}
LT = {a: L.load_state(D[a], False) for a in SIX}
for n, ks in W[:5]:
    print(f"  {n} ({len(ks)} turns)")
    for a in SIX:
        print(f"    {a:11s} {C.mean(LS[a][k][0] for k in ks):6.2f} | {C.mean(LS[a][k][1] for k in ks) / 1000:7.1f} (task definition {C.mean(LT[a][k][1] for k in ks) / 1000:7.1f}) | {C.mean(LS[a][k][2] for k in ks):6.2f} | "
              f"{C.mean(B[a][k]['t_send'] for k in ks) / 60:6.2f} | {C.mean(B[a][k]['ttft'] for k in ks):6.3f}")
    r = [100.0 * (LS[REP][k][1] - LS[REF][k][1]) / LS[REF][k][1] for k in ks]
    dt = [(B[REP][k]["t_send"] - B[REF][k]["t_send"]) / 60 for k in ks]
    print(f"    rep - ec_r13 turn by turn: live context {C.mean(LS[REP][k][1] - LS[REF][k][1] for k in ks) / 1000:+.1f}K ({100 * C.mean(LS[REP][k][1] - LS[REF][k][1] for k in ks) / C.mean(LS[REF][k][1] for k in ks):+.2f} %), "
          f"p5 {C.pct(r, .05):+.2f} % p50 {C.pct(r, .5):+.2f} % p95 {C.pct(r, .95):+.2f} %; send minute {C.mean(dt):+.2f} [p10 {C.pct(dt, .1):+.2f}, p90 {C.pct(dt, .9):+.2f}]")
# regression of the paired TTFT difference on the paired live-context difference
ks = keys
y = [B[REP][k]["ttft"] - B[REF][k]["ttft"] for k in ks]
x = [(LS[REP][k][1] - LS[REF][k][1]) / 10000 for k in ks]
mx, my = C.mean(x), C.mean(y)
slope = sum((u - mx) * (v - my) for u, v in zip(x, y)) / sum((u - mx) ** 2 for u in x)
print(f"  regression of the paired TTFT difference on the paired live-context difference: raw {my:+.3f} = intercept {my - slope * mx:+.3f} + context {slope * mx:+.3f} ({slope:+.3f} s per 10K tokens x {mx:+.2f})")

print("\n6. hardware of the two GPU units: decode ms per step, first 60 minutes, Decode lines with no queue: median (n) by running requests; then the same turns of the first 8 minutes, paired")
HW = ["jit_exact", "jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_p50_d_cut", "jit_p50_d", "ec_r13_rep"]
seven = HW[:7]
tab = {a: [[s[1] for s in L.decode_steps(D[a], 0, 3600, lo, hi)] for lo, hi in L.BUCKETS] for a in HW}
ref = [L.med([L.med(tab[a][i]) for a in seven]) for i in range(3)]
for a in HW:
    u = units.get(a) or units.get(a.replace("_cut", "")) or [("2026-11-01", "campaign 23 unit")]
    unit = u[0][1] if a != "jit_p50_d" else u[-1][1]
    print(f"  {a:14s} {unit:16s} " + " | ".join(f"{lo}-{hi}: {L.med(tab[a][i]):6.2f} ({len(tab[a][i]):4d}) {100 * (L.med(tab[a][i]) / ref[i] - 1):+5.1f} %" for i, (lo, hi) in enumerate(L.BUCKETS)))
B8 = {a: {(t["conv"], t["turn"]): t for t in D[a]["turns"] if t["ttft"] is not None and t["t_send"] + (t["latency"] or 0) < 480} for a in HW}
k8 = [k for k in set.intersection(*(set(v) for v in B8.values())) if len(set((B8[a][k]["pt"], B8[a][k]["ct"]) for a in HW)) == 1 and B8[HW[0]][k]["ct"] >= 20]
tot8 = {a: sum(B8[a][k]["latency"] - B8[a][k]["ttft"] for k in k8) for a in HW}
r8 = L.med([tot8[a] for a in seven])
print(f"  the same {len(k8)} turns (identical token counts, 20 or more completion tokens) finished within 480 s in all ten runs: total decode seconds and ratio to the median of the task's seven; median ms per completion token")
for a in HW:
    print(f"    {a:14s} {tot8[a]:8.1f} {100 * (tot8[a] / r8 - 1):+5.2f} % | {L.med([1000 * (B8[a][k]['latency'] - B8[a][k]['ttft']) / B8[a][k]['ct'] for k in k8]):6.2f}")
print("  whole run: first turns with no queue wait (under 50 ms) and over 2K uncomputed tokens, ms per token of (ttft - queue), median (n) | SSD ms per page in reads of 500 pages or more, median (n)")
for a in HW:
    fresh = []
    for t in D[a]["turns"]:
        if t["turn"] == 0 and t["ttft"] is not None:
            r = D[a]["req"].get(f"{t['conv']}-0")
            unc = (t["pt"] or 0) - (t["cached"] or 0)
            if r and r[0] < 50 and unc > 2000:
                fresh.append((t["ttft"] - r[0] / 1000) / unc * 1000)
    big = [x[5] / x[3] for x in D[a]["s2h"] if x[3] >= 500]
    print(f"    {a:14s} prefill {L.med(fresh):7.4f} ({len(fresh):3d}) | SSD {L.med(big):6.3f} ({len(big):4d})")

print("\n7. the mechanisms in the two runs of rules 1 and 3, over each run's kept window: staged reads, dispatches, refused room, s2h reads, pages, read seconds, evicted tokens GPU / host (M), Decode lines with a queue")
for a in (REF, REP):
    d = D[a]
    t0 = d["t0"]
    end = max(t["t_send"] for t in B[a].values())
    inw = lambda t: 0 <= t - t0 < end
    io = [x for x in d["s2h"] if inw(x[0])]
    dec = [x for x in d["dec"] if inw(x[0])]
    print(f"  {a:11s} staged {sum(1 for x in d['stage'] if inw(x[0])):5d} | dispatches {sum(1 for x in d['disp'] if inw(x[0])):5d} refused {sum(1 for x in d['disp'] if inw(x[0]) and x[5] == 0):5d} | "
          f"reads {len(io):5d} pages {sum(x[3] for x in io):7d} seconds {sum(x[5] for x in io) / 1000:7.1f} | evicted {sum(x[1] for x in d['evict_device'] if inw(x[0])) / 1e6:6.1f} / {sum(x[1] for x in d['evict_host'] if inw(x[0])) / 1e6:6.1f} | "
          f"queue on {100 * C.mean(1 if x[5] > 0 else 0 for x in dec):5.1f} % of {len(dec)} Decode lines, mean running {C.mean(x[1] for x in dec):5.2f}")

print("\n8. what the repeat moves in the other readings")
rr, rp = sum(1 for k in own if READ[REF].get(k, 0) > 0), sum(1 for k in own if READ[REP].get(k, 0) > 0)
print(f"  R2 scale: turns reading from SSD on the pair's {len(own)} shared turns: ec_r13 {rr}, ec_r13_rep {rp} ({100 * (rp / rr - 1):+.1f} %); pages {sum(READ[REF].get(k, 0) for k in own)} against {sum(READ[REP].get(k, 0) for k in own)} "
      f"({100 * (sum(READ[REP].get(k, 0) for k in own) / sum(READ[REF].get(k, 0) for k in own) - 1):+.1f} %)")
rn, rf = sum(1 for k in keys if READ["nofc_noend"].get(k, 0) > 0), sum(1 for k in keys if READ["nofc"].get(k, 0) > 0)
print(f"           on the six-run shared turns: nofc_noend {rn} against nofc {rf} ({100 * (rn / rf - 1):+.1f} %); TTFT nofc_noend - nofc {m['nofc_noend'] - m['nofc']:+.3f} s against the same-arm move {abs(m[REP] - m[REF]):.3f} s")
k12 = C.shared_keys(B, C.ALL11 + [REP], rank)
m12 = {a: C.mean(B[a][k]["ttft"] for k in k12) for a in C.ALL11 + [REP]}
print(f"  R3 scale: ALL11 plus the repeat share {len(k12)} returning turns: " + "  ".join(f"{a} {m12[a]:.3f}" for a in ("baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend")))
for base in ("baseline", "base_rep"):
    print(f"    share of the way from {base} to jit_exact: " + "  ".join(f"{a} {100 * (m12[base] - m12[a]) / (m12[base] - m12['jit_exact']):.1f} %" for a in ("ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend")))
print(f"    the repeat moves rules 1+3 by {m12[REP] - m12[REF]:+.3f} s here; the same move applied to nofc_noend ({m12['nofc_noend']:.3f} s) gives a share of "
      + " / ".join(f"{100 * (m12[b] - (m12['nofc_noend'] + s * abs(m12[REP] - m12[REF]))) / (m12[b] - m12['jit_exact']):.1f} %" for b in ("baseline", "base_rep") for s in (-1, 1)) + " (baseline -, +; base_rep -, +)")
