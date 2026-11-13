"""Outside-reading point 12 of DECISIONS.md ('one block of 500 turns recomputed 183K tokens against 15-18K'): recomputed
tokens summed per block of 500 order-run positions, leading arms, on their shared returning turns, from the cache.
  nice -n 19 python3 -I -B r2_recompute_blocks.py"""
import collections, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, shared, rank, rec

rk = rank()
for names in (["jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend"], ["jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]):
    keys = shared(names)
    blocks = collections.defaultdict(lambda: collections.Counter())
    n = collections.Counter()
    for k in keys:
        b = rk.get(k, 1 << 30) // 500
        n[b] += 1
        for a in names:
            blocks[b][a] += rec(arm(a)["by"][k])
    print(f"\n{len(keys)} turns shared by {', '.join(names)}: recomputed K tokens per block of 500 positions (blocks where some arm is over 100K, and the totals)")
    for b in sorted(blocks):
        if max(blocks[b].values()) > 100000:
            print(f"   block {b if b < 100000 else 'absent'} ({n[b]} turns): " + "  ".join(f"{a} {blocks[b][a] / 1000:.0f}K" for a in names))
    print("   whole set: " + "  ".join(f"{a} {sum(blocks[b][a] for b in blocks) / 1e6:.2f}M ({sum(blocks[b][a] for b in blocks) / len(keys):.0f}/turn)" for a in names))
    worst = max(blocks, key=lambda b: max(blocks[b][a] for a in names if a.startswith("ec_r13")) - min(blocks[b][a] for a in names if not a.startswith("ec_r13")))
    print(f"   block where rules 1 and 3 stand out most: {worst} ({n[worst]} turns): " + "  ".join(f"{a} {blocks[worst][a] / 1000:.0f}K" for a in names))
