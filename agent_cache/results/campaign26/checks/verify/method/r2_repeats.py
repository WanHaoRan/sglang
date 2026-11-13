"""The write-up's rule for repeats, applied to the two runs that landed after its last entry (ec_r13_rep, jit_p50_d):
pairs_ec.py's statistics recomputed from the cache (same kept-turn rule, shared returning turns, positions, block bootstrap).
  nice -n 19 python3 -I -B r2_repeats.py"""
import itertools, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, rank, in_episode, blocks_by_rank, boot, mean, pct, rec

rk = rank()


def report(names, groups, label):
    keys = shared(names)
    ep = [k for k in keys if in_episode(k)]
    calm = [k for k in keys if not in_episode(k)]
    T = {a: arm(a)["by"] for a in names}
    print(f"\n== {label}: {len(keys)} returning turns shared by {', '.join(names)} ({len(ep)} in the episode positions, {len(calm)} elsewhere, {sum(k not in rk for k in keys)} absent from the order run)")
    for a in names:
        v = [T[a][k]["ttft"] for k in keys]
        d = arm(a)
        print(f"   {a:13s} mean {mean(v):.3f} median {st.median(v):.2f} p90 {pct(v, .9):.2f} p99 {pct(v, .99):.1f} over10 {sum(x > 10 for x in v):4d} | episodes {mean(T[a][k]['ttft'] for k in ep):.3f} elsewhere {mean(T[a][k]['ttft'] for k in calm):.3f} | "
              f"reading {sum(d['pages'][k] > 0 for k in keys):5d} pages {sum(d['pages'][k] for k in keys):7d} recomputed {mean(rec(T[a][k]) for k in keys):6.1f} | own kept turns {len(d['turns'])}, up to minute {d['end'] / 60:.1f}")

    def diff(a, b, ks):
        return boot(blocks_by_rank({k: T[a][k]["ttft"] - T[b][k]["ttft"] for k in ks}))

    grp = {r: g for g, runs in groups.items() for r in runs}
    lab = lambda a: grp.get(a, a)
    print("   the same arm run again (later minus earlier): all [5 %, 95 %] | episodes | elsewhere")
    moves = {}
    for g, runs in groups.items():
        for a, b in itertools.combinations([r for r in runs if r in names], 2):
            d, lo, hi = diff(b, a, keys)
            moves.setdefault(g, []).append(abs(d))
            print(f"     {g:14s} {b:13s} - {a:13s} {d:+.3f} [{lo:+.2f},{hi:+.2f}] | {diff(b, a, ep)[0]:+.3f} | {diff(b, a, calm)[0]:+.3f}")
    print("   between arms, every pair of their runs: smallest, largest difference (pairs) | largest move between two runs of either arm | by the write-up's rule")
    for x, y in itertools.combinations(dict.fromkeys(lab(a) for a in names), 2):
        d = [diff(b, a, keys)[0] for a in names for b in names if lab(a) == x and lab(b) == y]
        mv = max(moves.get(x, [float("nan")]) + moves.get(y, [float("nan")]), key=lambda v: (v == v, v))
        known = [v for v in moves.get(x, []) + moves.get(y, [])]
        same_sign = min(d) > 0 or max(d) < 0
        verdict = "no repeat on either side" if not known else ("shown" if same_sign and min(abs(v) for v in d) > max(known) else "not separated")
        print(f"     {y:14s} - {x:14s} {min(d):+.3f} {max(d):+.3f} ({len(d)}) | {max(known) if known else float('nan'):.3f} | {verdict}")
    return keys


G = {"rules 1 and 3": ("ec_r13", "ec_r13_rep"), "JIT p50": ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_p50_d"), "hint-free": ("hintfree", "hintfree_ctl"), "original HiCache": ("baseline", "base_rep")}
report(["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"], G, "leading arms with the repeat of rules 1 and 3")
report(["jit_p50", "jit_p50_ctl", "jit_p50_d", "hintfree_ctl", "nofc_noend", "ec_r13", "ec_r13_rep", "jit_exact"], G, "JIT p50's three full runs with the leading arms")
report(["jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_p50_d"], G, "the four JIT p50 runs on the cut run's turns")

print("\nhardware by GPU unit, whole kept window: decode ms per step with no queue by running requests (median, n), first-turn prefill ms per token, SSD ms per page in reads of 500+ pages")
for a in ("jit_p50_ctl", "jit_p50_d", "ec_r13", "ec_r13_rep", "jit_exact"):
    d = arm(a)
    dec = [(n, 1000 * n / thr) for t, n, _, _, thr, q in d["server"]["dec"] if 0 <= t - d["t0"] < d["end"] and q == 0 and thr > 0]
    cells = " ".join(f"{lo}-{hi} {st.median([ms for n, ms in dec if lo <= n <= hi]):5.2f} ({sum(lo <= n <= hi for n, _ in dec):4d})" for lo, hi in ((8, 15), (16, 23), (24, 31)))
    first = [1000 * (r["ttft"] - d["queue"].get((r["conv"], r["turn"]), 0.0)) / (r["prompt_tokens"] - (r["cached_tokens"] or 0)) for r in d["turns"]
             if r["turn"] == 0 and d["queue"].get((r["conv"], r["turn"]), 0.0) < 0.05 and (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0) > 2000]
    ssd = [1000 * ms / p for v in d["ios"].values() for t, ms, p in v if p >= 500]
    print(f"   {a:12s} decode {cells} | prefill {st.median(first):.3f} ({len(first)}) | SSD {st.median(ssd):.2f} ({len(ssd)})")
