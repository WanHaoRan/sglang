"""Host-pool sweep on recorded timelines: the share of returning turns that read from SSD as the host pool grows.
From agent_cache/:  nice -n 10 python3 -I -B results/campaign26/checks/limits/pool_sweep.py | tee results/campaign26/checks/limits/pool_sweep.out
Uses the one-pool model of frozen/ec_policy_20261111/evict_replay/evict_replay.py, with its limits: no pins, no GPU and
host split, the run's own timeline, so it gives read shares and not TTFT. The model's pool M is fitted once, on a 70 GB
run, so that LRU gives that run's measured share. Another host size moves M in one of two ways, since the fit does not
say which: by the same ratio as the host pool, or by the same number of tokens. The 100 GB cut run is the check."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
source = open(os.path.join(HERE, "..", "..", "..", "..", "frozen", "ec_policy_20261111", "evict_replay", "evict_replay.py")).read()
ns = {"__name__": "evict_replay_model"}
exec(compile(source[:source.index('\nif len(sys.argv) > 1 and sys.argv[1] == "noend":')], "evict_replay.py", "exec"), ns)
load, simulate, share = ns["load"], ns["simulate"], ns["share"]
POLICIES = (("LRU", ns["lru"]), ("LRU, ended sessions first", ns["lru_end"]), ("next use from the p50", ns["next_use"]()),
            ("return chance within 30 s", ns["return_chance"](30, ns["chance_own"])), ("exact next use", ns["exact"]))

TOKENS_PER_GB = 10172.8  # host pool tokens per GB: 1,017,280 at 100 GB, 1,627,648 at 160 GB
FIT = ("nofc_noend_rep", "20261112_174138_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70", 70, 0.170)
CHECK = ("nofc_noend_h100_cut", "20261113_060157_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H100", 100, 56 / 8030)
SIZES = (70, 75, 80, 85, 90, 95, 100, 110, 120, 136, 160)


def fit(turns, target):
    """The pool at which LRU reads on the target share of returning turns."""
    lo, hi = 200_000, 3_000_000
    for _ in range(18):
        mid = (lo + hi) // 2
        if share(simulate(turns, mid, ns["lru"])) > target:
            lo = mid
        else:
            hi = mid
    return hi


def pools(m70, gb):
    """M at a host pool of gb GB: scaled by ratio, and moved by the tokens added."""
    return int(m70 * gb / 70), int(m70 + (gb - 70) * TOKENS_PER_GB)


name, run, gb, target = FIT
turns = load(run)
m70 = fit(turns, target)
print(f"fit on {name} ({gb} GB, measured {100 * target:.1f} % of returning turns reading from SSD): {len(turns)} turns, M = {m70:,} tokens "
      f"(the host pool is {int(gb * TOKENS_PER_GB):,})")

cname, crun, cgb, cshare = CHECK
cturns = load(crun)
by_ratio, by_tokens = pools(m70, cgb)
print(f"\ncheck on {cname}'s own timeline ({cgb} GB, {len(cturns)} turns, measured {100 * cshare:.1f} %): LRU share with M by ratio ({by_ratio:,}) "
      f"{100 * share(simulate(cturns, by_ratio, ns['lru'])):.1f} %, by tokens ({by_tokens:,}) {100 * share(simulate(cturns, by_tokens, ns['lru'])):.1f} %")
# The same window of the 70 GB run, so the two timelines can be told apart from the two pool sizes.
t_cut = max(r["t_send"] for r in cturns)
head = [r for r in turns if r["t_send"] <= t_cut]
print(f"  the 70 GB run's first {t_cut / 60:.0f} min ({len(head)} turns) at those pools: {100 * share(simulate(head, by_ratio, ns['lru'])):.1f} %, "
      f"{100 * share(simulate(head, by_tokens, ns['lru'])):.1f} %; at its own M {100 * share(simulate(head, m70, ns['lru'])):.1f} %")

for how, pick in (("M scaled by the host pool's ratio", 0), ("M moved by the tokens added", 1)):
    print(f"\n{name}'s timeline, {how}: share of returning turns reading from SSD (K tokens read per returning turn)")
    print(f"  {'host GB':>8s} {'M':>10s} | " + " | ".join(f"{label:>26s}" for label, _ in POLICIES))
    for size in SIZES:
        m = pools(m70, size)[pick]
        cells = []
        for _, prio in POLICIES:
            o = simulate(turns, m, prio)
            cells.append(f"{100 * share(o):17.1f} % ({o['read_tokens'] / o['returning'] / 1000:5.2f})")
        print(f"  {size:8d} {m:10,d} | " + " | ".join(cells))
