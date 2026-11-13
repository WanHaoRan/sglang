"""Test (f11): R4 with the complete run of JIT p50 on the new engine build.
jit_p50_d (engine_ec_20261111d, relaunched after the outage on GPU-a7198925, finished 2026-11-11 22:46Z, outside the
task's list of runs) against jit_p50_ctl and jit_p50 (old build) over the whole 115 minutes, against jit_p50_d_cut (the
same build before the outage, another GPU unit) over the first 74.7, and the leading arms against all of them.
  nice -n 19 python3 -I -B tg_r4full.py"""
import itertools, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rank = C.order_rank()
NPOS = len(rank)
JIT3 = ["jit_p50", "jit_p50_ctl", "jit_p50_d"]
JIT4 = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_p50_d"]
F7 = ["jit_p50", "jit_p50_ctl", "jit_p50_d", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"]
LEADERS = ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
NAMES = sorted(set(JIT4 + F7 + LEADERS + C.CUT7))
D = {a: C.load(a) for a in NAMES}
B = {a: C.by_key(C.kept(D[a])) for a in NAMES}
READ = {a: L.read_pages(D[a]) for a in NAMES}
units = L.gpu_units()

print("0. the JIT p50 runs")
for a in JIT4:
    d = D[a]
    eng = dict(line.strip().split("=", 1) for line in open(os.path.join(L.run_dir(a), "run.txt")) if "=" in line).get("engine", "").split("/frozen/")[-1].split("/")[0]
    key = "jit_p50_d" if a.startswith("jit_p50_d") else a
    u = units.get(key)
    unit = "campaign 23 unit" if not u else (u[0][1] if a != "jit_p50_d" else u[-1][1])
    print(f"  {a:14s} {d['run'][:15]} engine {eng:22s} {unit:16s} turns {len(d['turns'])} errors {sum(1 for t in d['turns'] if t['err'])} kept {len(B[a])} to minute {max(t['t_send'] for t in B[a].values()) / 60:.1f}")
for a, b in (("jit_p50_d_cut", "jit_p50_d"), ("jit_p50_ctl", "jit_p50_d")):
    sa, sb = L.server_args(a), L.server_args(b)
    dif = sorted(k for k in set(sa) | set(sb) if sa.get(k, "<absent>") != sb.get(k, "<absent>"))
    print(f"  server_args {a} against {b}: {len(dif)} fields differ: " + "; ".join(f"{k}={str(sa.get(k, '<absent>'))[-28:]}|{str(sb.get(k, '<absent>'))[-28:]}" for k in dif))


def split(keys):
    ep1 = [k for k in keys if 5500 <= rank.get(k, -1) < 7000]
    ep2 = [k for k in keys if 8750 <= rank.get(k, -1) < 11000]
    return (("all", keys), ("episodes", ep1 + ep2), ("ep1", ep1), ("ep2", ep2), ("elsewhere", [k for k in keys if not C.in_episode(rank, k)]))


def table(names, title):
    keys = C.shared_keys(B, names, rank)
    S = split(keys)
    print(f"\n{title}: {len(keys)} shared returning turns ({len(set(k[0] for k in keys))} sessions); " + ", ".join(f"{n} {len(ks)}" for n, ks in S[1:]))
    print("  arm            mean  median   p90    p99  >10s | episodes   ep1    ep2  elsewhere | reading turns   pages | recomputed/turn | queue wait s")
    for a in names:
        v = [B[a][k]["ttft"] for k in keys]
        q = C.mean(D[a]["req"].get(f"{k[0]}-{k[1]}", [0])[0] / 1000 for k in keys)
        print(f"  {a:14s} {C.mean(v):5.3f} {C.pct(v, .5):6.3f} {C.pct(v, .9):6.2f} {C.pct(v, .99):6.1f} {sum(x > 10 for x in v):5d} | " + " ".join(f"{C.mean(B[a][k]['ttft'] for k in ks):6.3f}" for n, ks in S[1:])
              + f" | {sum(1 for k in keys if READ[a].get(k, 0) > 0):5d} {sum(READ[a].get(k, 0) for k in keys):8d} | {C.mean(C.rec(B[a][k]) for k in keys):5.0f} | {q:5.3f}")
    return keys


def paired(a, b, ks, label, send):
    d = {k: B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks}
    n = len(ks)
    tot = sum(d.values())
    b250 = [x[2] for x in L.pos_bins(ks, rank, NPOS, 250, send)]
    lo, hi, opp = L.block_boot(d, b250)
    slo, shi = L.session_boot(d)
    sess = {}
    for k in ks:
        sess[k[0]] = sess.get(k[0], 0.0) + d[k]
    al = sum(1 for v in sess.values() if v > 0)
    ah = sum(1 for v in sess.values() if v < 0)
    bl = sum(1 for bn in b250 if sum(d[k] for k in bn) > 0)
    bh = sum(1 for bn in b250 if sum(d[k] for k in bn) < 0)
    top = sorted(ks, key=lambda k: -abs(d[k]))[:max(1, n // 100)]
    ts = sum(d[k] for k in top)
    print(f"  {label:10s} {b:14s} - {a:14s} n {n:5d} mean {tot / n:+6.3f} | block bootstrap 250 [{lo:+6.3f}, {hi:+6.3f}], opposite sign {100 * opp:4.1f} % | session bootstrap [{slo:+6.3f}, {shi:+6.3f}] | "
          f"sessions {a} lower {al} higher {ah} (p {L.p2(al, al + ah):.3g}) | bins lower {bl} higher {bh} | top 1 % carry {100 * ts / tot if tot else 0:+6.1f} %, without {(tot - ts) / (n - len(top)):+6.3f} | median {C.pct(list(d.values()), .5):+.4f}")
    return tot / n


k7 = table(F7, "1. R4 over the whole run: the seven of CUT7 with the complete jit_p50_d in place of the cut one")
m7 = {a: C.mean(B[a][k]["ttft"] for k in k7) for a in F7}
print(f"  jit_p50_d - jit_p50_ctl {m7['jit_p50_d'] - m7['jit_p50_ctl']:+.3f} s; jit_p50_d - jit_p50 {m7['jit_p50_d'] - m7['jit_p50']:+.3f}; jit_p50_ctl - jit_p50 {m7['jit_p50_ctl'] - m7['jit_p50']:+.3f}; "
      f"smallest lead of nofc_noend / ec_r13 / jit_exact over any of the three JIT p50 runs: " + " / ".join(f"{min(m7[j] - m7[n] for j in JIT3):+.3f}" for n in ("nofc_noend", "ec_r13", "jit_exact")))

print("\n2. the three complete JIT p50 runs against each other, on the turns the three share and by window")
k3 = C.shared_keys(B, JIT3, rank)
send3 = lambda k: C.mean(B[a][k]["t_send"] for a in JIT3)
print(f"  {len(k3)} shared returning turns: " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in k3):.3f}" for a in JIT3))
for a, b in (("jit_p50_ctl", "jit_p50_d"), ("jit_p50", "jit_p50_d"), ("jit_p50", "jit_p50_ctl")):
    for n, ks in split(k3):
        paired(a, b, ks, n, send3)
print("  by 250-position bin where any of the three is over 1 s: turns | jit_p50 (c23), jit_p50_ctl, jit_p50_d | d - ctl")
for label, lo, ks in L.pos_bins(k3, rank, NPOS, 250, send3):
    f = lambda a: C.mean(B[a][k]["ttft"] for k in ks)
    if max(f(a) for a in JIT3) > 1.0:
        print(f"    {label:>6s} {len(ks):4d} | {f('jit_p50'):6.3f} {f('jit_p50_ctl'):6.3f} {f('jit_p50_d'):6.3f} | {f('jit_p50_d') - f('jit_p50_ctl'):+6.3f}")
print("  by 10-minute window of jit_p50_ctl's send time: turns | jit_p50, jit_p50_ctl, jit_p50_d | d - ctl | c23 - ctl")
for mnt in range(0, 120, 10):
    ks = [k for k in k3 if mnt * 60 <= B["jit_p50_ctl"][k]["t_send"] < (mnt + 10) * 60]
    if ks:
        f = lambda a: C.mean(B[a][k]["ttft"] for k in ks)
        print(f"    {mnt:3d}-{mnt + 10:3d} {len(ks):5d} | {f('jit_p50'):6.3f} {f('jit_p50_ctl'):6.3f} {f('jit_p50_d'):6.3f} | {f('jit_p50_d') - f('jit_p50_ctl'):+6.3f} | {f('jit_p50') - f('jit_p50_ctl'):+6.3f}")

print("\n3. the four JIT p50 runs on the turns all four share (the cut run's 74.7 minutes): same build on two GPU units, old build on two units")
k4 = C.shared_keys(B, JIT4, rank)
send4 = lambda k: C.mean(B[a][k]["t_send"] for a in JIT4)
S4 = split(k4)
print(f"  {len(k4)} shared returning turns; " + ", ".join(f"{n} {len(ks)}" for n, ks in S4[1:]))
for a in JIT4:
    print(f"    {a:14s} " + "  ".join(f"{n} {C.mean(B[a][k]['ttft'] for k in ks) if ks else float('nan'):6.3f}" for n, ks in S4) + f" | reading turns {sum(1 for k in k4 if READ[a].get(k, 0) > 0)} | recomputed {C.mean(C.rec(B[a][k]) for k in k4):.0f}")
for a, b in itertools.combinations(JIT4, 2):
    paired(a, b, k4, "all", send4)
ds = {(a, b): C.mean(B[b][k]["ttft"] - B[a][k]["ttft"] for k in k4) for a, b in itertools.combinations(JIT4, 2)}
same_build = [abs(ds[("jit_p50", "jit_p50_ctl")]), abs(ds[("jit_p50_d_cut", "jit_p50_d")])]
cross = [abs(ds[k]) for k in ds if k not in (("jit_p50", "jit_p50_ctl"), ("jit_p50_d_cut", "jit_p50_d"))]
print(f"  |difference| between two runs of one build: old {same_build[0]:.3f}, new {same_build[1]:.3f}; between an old-build and a new-build run: " + " ".join(f"{x:.3f}" for x in cross)
      + f" (mean {C.mean(cross):.3f}); mean of the two builds' runs: old {C.mean(C.mean(B[a][k]['ttft'] for k in k4) for a in ('jit_p50', 'jit_p50_ctl')):.3f}, new {C.mean(C.mean(B[a][k]['ttft'] for k in k4) for a in ('jit_p50_d_cut', 'jit_p50_d')):.3f}")

print("\n4. the leading arms against every complete JIT p50 run")
NINE = JIT3 + LEADERS
k9 = table(NINE, "   the three complete JIT p50 runs and the six leading runs")
m9 = {a: C.mean(B[a][k]["ttft"] for k in k9) for a in NINE}
move = max(abs(m9[a] - m9[b]) for a, b in itertools.combinations(JIT3, 2))
print(f"  largest move between two JIT p50 runs here: {move:.3f} s (old build twice {abs(m9['jit_p50'] - m9['jit_p50_ctl']):.3f}, new against old on campaign 26's first unit {abs(m9['jit_p50_d'] - m9['jit_p50_ctl']):.3f})")
for n in LEADERS:
    leads = [m9[j] - m9[n] for j in JIT3]
    print(f"    {n:11s} lead over jit_p50 / jit_p50_ctl / jit_p50_d: " + " ".join(f"{x:+6.3f}" for x in leads) + f" | smallest {min(leads):+.3f} = {min(leads) / move:.1f} x the largest JIT p50 move")
S9 = split(k9)
for nme, ks in S9[1:]:
    print(f"    {nme:9s} ({len(ks)} turns): " + "  ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in ks):.3f}" for a in NINE))

print("\n5. hardware and mechanisms of the three complete JIT p50 runs over each run's kept window")
print("   decode ms per step, first 60 minutes, no queue: median (n) by running requests | first-turn prefill ms per token | SSD ms per page (reads of 500 pages or more) | staged, refused | s2h reads, pages | evicted GPU / host (M)")
for a in JIT4:
    d = D[a]
    t0 = d["t0"]
    end = max(t["t_send"] for t in B[a].values())
    inw = lambda t: 0 <= t - t0 < end
    cells = [f"{lo}-{hi} {L.med([s[1] for s in L.decode_steps(d, 0, 3600, lo, hi)]):6.2f} ({len(L.decode_steps(d, 0, 3600, lo, hi)):3d})" for lo, hi in L.BUCKETS]
    fresh = []
    for t in d["turns"]:
        if t["turn"] == 0 and t["ttft"] is not None:
            r = d["req"].get(f"{t['conv']}-0")
            unc = (t["pt"] or 0) - (t["cached"] or 0)
            if r and r[0] < 50 and unc > 2000:
                fresh.append((t["ttft"] - r[0] / 1000) / unc * 1000)
    io = [x for x in d["s2h"] if inw(x[0])]
    big = [x[5] / x[3] for x in io if x[3] >= 500]
    print(f"  {a:14s} " + " ".join(cells) + f" | {L.med(fresh):.4f} ({len(fresh)}) | {L.med(big):5.2f} ({len(big)}) | {sum(1 for x in d['stage'] if inw(x[0]))} {sum(1 for x in d['disp'] if inw(x[0]) and x[5] == 0)} | "
          f"{len(io)} {sum(x[3] for x in io)} | {sum(x[1] for x in d['evict_device'] if inw(x[0])) / 1e6:.1f} / {sum(x[1] for x in d['evict_host'] if inw(x[0])) / 1e6:.1f}")
print("   the same in the first 74.7 minutes of each run's clock (the cut run's kept window): decode by bucket | staged, refused | reads, pages")
for a in JIT4:
    d = D[a]
    t0 = d["t0"]
    inw = lambda t: 0 <= t - t0 < 4482
    cells = [f"{lo}-{hi} {L.med([s[1] for s in L.decode_steps(d, 0, 4482, lo, hi)]):6.2f} ({len(L.decode_steps(d, 0, 4482, lo, hi)):3d})" for lo, hi in L.BUCKETS]
    io = [x for x in d["s2h"] if inw(x[0])]
    print(f"  {a:14s} " + " ".join(cells) + f" | {sum(1 for x in d['stage'] if inw(x[0]))} {sum(1 for x in d['disp'] if inw(x[0]) and x[5] == 0)} | {len(io)} {sum(x[3] for x in io)}")
