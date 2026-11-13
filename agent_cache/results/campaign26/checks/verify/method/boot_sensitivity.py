"""How much the block-bootstrap interval of pairs_ec.py depends on choices the write-up does not state: the seed, the
block length and offset, what is done with keys absent from the order run, and the number of blocks a subset is left with.
Also the placebo the write-up names: the same arm run twice.
  nice -n 19 python3 -I -B boot_sensitivity.py"""
import collections, os, random, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

rk = rank()


def cells_for(vals, block=500, offset=0, absent="one", own=None):
    """absent: 'one' = the script (one block for every key the order run lacks); 'own' = chained after the order run in the
    given own order, in blocks of `block`."""
    cells = {}
    n_order = len(rk)
    extra = {}
    if absent == "own":
        missing = sorted((k for k in vals if k not in rk), key=lambda k: own[k])
        extra = {k: n_order + i for i, k in enumerate(missing)}
    for k, v in vals.items():
        p = rk.get(k)
        if p is None:
            p = extra.get(k, 1 << 30)
        s = cells.setdefault((p + offset) // block, [0.0, 0])
        s[0] += v
        s[1] += 1
    return cells


def show(label, vals, own):
    d0, lo0, hi0 = boot(cells_for(vals))
    print(f"  {label}: {len(vals)} turns, difference {d0:+.3f}; as published [{lo0:+.3f},{hi0:+.3f}] with {len(cells_for(vals))} blocks")
    los, his = zip(*[boot(cells_for(vals), seed=s)[1:] for s in range(20)])
    print(f"     seeds 0-19: lower {min(los):+.3f}..{max(los):+.3f}, upper {min(his):+.3f}..{max(his):+.3f}")
    row = []
    for block in (250, 500, 1000, 1500):
        for off in (0, block // 2):
            c = cells_for(vals, block=block, offset=off, absent="own", own=own)
            _, lo, hi = boot(c, draws=4000)
            row.append(f"{block}/{off}: [{lo:+.3f},{hi:+.3f}] ({len(c)})")
    print("     block/offset, absent keys chained in own order: " + "  ".join(row))
    # sessions as blocks
    c = {}
    for k, v in vals.items():
        s = c.setdefault(k[0], [0.0, 0])
        s[0] += v
        s[1] += 1
    _, lo, hi = boot(c, draws=4000)
    print(f"     sessions as blocks ({len(c)}): [{lo:+.3f},{hi:+.3f}]")
    # 2.5 % / 97.5 %
    _, lo, hi = boot(cells_for(vals), draws=4000, q=(.025, .975))
    print(f"     published blocks, 2.5 % to 97.5 %: [{lo:+.3f},{hi:+.3f}]")
    # leave-one-block-out: how much one block carries
    cc = cells_for(vals)
    tot, n = sum(x[0] for x in cc.values()), sum(x[1] for x in cc.values())
    loo = sorted(((tot - x[0]) / (n - x[1]), b, x[1], x[0] / n) for b, x in cc.items())
    print(f"     leave one block out: difference from {loo[0][0]:+.3f} (without block {loo[0][1]}, {loo[0][2]} turns, which carries {loo[0][3]:+.3f}) to {loo[-1][0]:+.3f} (without block {loo[-1][1]}); "
          f"the three largest blocks carry {sum(sorted((abs(x[0]) for x in cc.values()), reverse=True)[:3]) / sum(abs(x[0]) for x in cc.values()) * 100:.0f} % of the summed |block difference|")


k5 = shared(LEAD5)
own5 = {(r["conv"], r["turn"]): i for i, r in enumerate(arm("jit_exact")["turns"])}
print("[1] LEAD5, all shared returning turns")
for a, b in (("ec_r13", "jit_exact"), ("ec_r1", "ec_r13"), ("nofc", "ec_r13"), ("nofc_noend", "ec_r13"), ("nofc_noend", "nofc"), ("nofc_noend", "ec_r1")):
    show(f"{a} - {b}", {k: arm(a)["by"][k]["ttft"] - arm(b)["by"][k]["ttft"] for k in k5}, own5)

print("\n[2] LEAD5, the 'other positions' matrix: with the absent keys in one block (published) and without them at all")
calm = [k for k in k5 if not in_episode(k)]
calm_present = [k for k in calm if k in rk]
for a, b in (("ec_r13", "jit_exact"), ("ec_r1", "jit_exact"), ("nofc", "jit_exact"), ("nofc_noend", "jit_exact"), ("nofc_noend", "nofc")):
    v = {k: arm(a)["by"][k]["ttft"] - arm(b)["by"][k]["ttft"] for k in calm}
    vp = {k: v[k] for k in calm_present}
    va = {k: v[k] for k in calm if k not in rk}
    d, lo, hi = boot(cells_for(v))
    dp, lop, hip = boot(cells_for(vp))
    do, loo_, hio = boot(cells_for(v, absent="own", own=own5))
    print(f"  {a} - {b}: published {d:+.3f} [{lo:+.3f},{hi:+.3f}] ({len(v)} turns) | absent chained in blocks of 500 [{loo_:+.3f},{hio:+.3f}] | positions the order run has only: {dp:+.3f} [{lop:+.3f},{hip:+.3f}] ({len(vp)}) | absent keys alone: {mean(va.values()):+.3f} ({len(va)})")

print("\n[3] subsets left with three blocks (the episode rows of the 6,715- and 6,579-turn sets): every outcome the resampling can give")
for names, a, b in ((CTL9, "jit_p50_ctl", "jit_p50"), (CTL9, "hintfree_ctl", "hintfree"), (CTL9, "base_rep", "baseline"), (CUT7, "jit_p50_d_cut", "jit_p50_ctl"), (CUT7, "nofc_noend", "jit_p50_ctl")):
    ks = [k for k in shared(names) if in_episode(k)]
    c = cells_for({k: arm(a)["by"][k]["ttft"] - arm(b)["by"][k]["ttft"] for k in ks})
    d, lo, hi = boot(c)
    per = {blk: x[0] / x[1] for blk, x in sorted(c.items())}
    print(f"  {a} - {b}: {len(ks)} turns in {len(c)} blocks; per-block mean difference " + ", ".join(f"block {blk} ({c[blk][1]} turns) {v:+.2f}" for blk, v in per.items())
          + f"; published [{lo:+.2f},{hi:+.2f}]; with 3 blocks a draw takes one of 10 values, so the 5 % and 95 % points are block combinations, not a tail estimate")

print("\n[4] placebo: the same arm run twice (later minus earlier), on each pair's own shared returning turns and on the 6,715")
for a, b in (("baseline", "base_rep"), ("hintfree", "hintfree_ctl"), ("jit_p50", "jit_p50_ctl")):
    for label, ks in ((f"own shared set", shared([a, b])), ("the 6,715", shared(CTL9))):
        v = {k: arm(b)["by"][k]["ttft"] - arm(a)["by"][k]["ttft"] for k in ks}
        own = {(r["conv"], r["turn"]): i for i, r in enumerate(arm(a)["turns"])}
        d, lo, hi = boot(cells_for(v))
        c2 = cells_for(v, block=1500, absent="own", own=own)
        _, lo2, hi2 = boot(c2)
        cs = {}
        for k, x in v.items():
            s = cs.setdefault(k[0], [0.0, 0])
            s[0] += x
            s[1] += 1
        _, lo3, hi3 = boot(cs)
        print(f"  {b} - {a} on {label} ({len(ks)}): {d:+.3f} published-style [{lo:+.2f},{hi:+.2f}] ({len(cells_for(v))} blocks) | blocks of 1500 [{lo2:+.2f},{hi2:+.2f}] ({len(c2)}) | sessions [{lo3:+.2f},{hi3:+.2f}] ({len(cs)})"
              f" | absent keys {sum(k not in rk for k in ks)}")
