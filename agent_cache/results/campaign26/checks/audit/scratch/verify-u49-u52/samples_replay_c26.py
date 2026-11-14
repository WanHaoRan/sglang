"""Read-only check for audit item U50: the eviction replay's "samples so far" score on campaign 26's own timelines,
with the end signalled and never signalled. Same model and policies as
frozen/ec_policy_20261111/evict_replay/evict_replay.py (its `noend` mode), run on a campaign 26 run instead of campaign 23's.
From agent_cache/:  nice -n 10 python3 -I -B <this file> [label run_folder measured_lru_share]
Prints only; writes nothing."""
import collections
import os
import sys

SRC = os.path.join("frozen", "ec_policy_20261111", "evict_replay", "evict_replay.py")
source = open(SRC).read()
ns = {"__name__": "evict_replay_model"}
exec(compile(source[:source.index('\nif len(sys.argv) > 1 and sys.argv[1] == "noend":')], "evict_replay.py", "exec"), ns)
load, simulate, share, Dist = ns["load"], ns["simulate"], ns["share"], ns["Dist"]

label = sys.argv[1] if len(sys.argv) > 1 else "nofc_noend_rep"
run = sys.argv[2] if len(sys.argv) > 2 else "20261112_174138_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"
target = float(sys.argv[3]) if len(sys.argv) > 3 else 0.170

turns = load(run)
by = {(r["conv"], r["turn"]): r for r in turns}
gaps = collections.defaultdict(list)
for r in turns:
    nxt = by.get((r["conv"], r["turn"] + 1))
    if nxt:
        gaps[r["source"]].append(nxt["t_send"] - r["t_send"] - r["latency"])
pooled = {}
for src, xs in gaps.items():
    xs.sort()
    q = lambda p: max(xs[min(len(xs) - 1, int(p * len(xs)))], 0.01)
    pooled[src] = Dist(q(0.1), q(0.5), q(0.9))

lo, hi = 200_000, 3_000_000
for _ in range(18):
    mid = (lo + hi) // 2
    if share(simulate(turns, mid, ns["lru"])) > target:
        lo = mid
    else:
        hi = mid
cap = hi
ended = sum(1 for r in turns if (r.get("dist_hint") or {}).get("ended"))
print(f"{label}: {len(turns)} turns, pool M = {cap:,} tokens fitted so that LRU reads on {100 * target:.1f} % of returning turns; "
      f"{ended} session ends signalled in the run")
print("  share of returning turns reading from SSD: end signalled | never signalled")
rows = [("LRU", ns["lru"]), ("LRU, ended sessions first", ns["lru_end"]), ("next use from the p50 (today)", ns["next_use"]()),
        ("return chance H=30, three quantiles", ns["return_chance"](30, ns["chance_own"])),
        ("return chance H=30, samples so far", ns["empirical"](30)),
        ("exact next use", ns["exact"])]
for name, prio in rows:
    a = simulate(turns, cap, prio, pooled_of=pooled)
    b = simulate(turns, cap, prio, pooled_of=pooled, signal_ends=False)
    print(f"  {name:38s} {100 * share(a):5.1f} % | {100 * share(b):5.1f} %   (K tokens read per returning turn {a['read_tokens'] / a['returning'] / 1000:.2f} | {b['read_tokens'] / b['returning'] / 1000:.2f})")
