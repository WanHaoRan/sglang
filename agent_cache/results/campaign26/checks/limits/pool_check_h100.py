"""The pool sweep's model against the three full 100 GB runs: each run's own timeline, replayed at the pool the 70 GB fit
gives for 100 GB, under every eviction order; the order nearest to what the arm ran is marked with its measured share.
From agent_cache/:  nice -n 10 python3 -I -B results/campaign26/checks/limits/pool_check_h100.py | tee results/campaign26/checks/limits/pool_check_h100.out
Reuses pool_sweep.py's model and fit (M fitted on nofc_noend_rep so that LRU gives its 17.0 %); same limits: one pool,
no pins, read shares and not TTFT. Measured shares are compare_ec_h100.out's, on each arm's own returning turns."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
source = open(os.path.join(HERE, "pool_sweep.py")).read()
ns = {"__name__": "pool_sweep_model", "__file__": os.path.join(HERE, "pool_sweep.py")}
exec(compile(source[:source.index("\nname, run, gb, target = FIT")], "pool_sweep.py", "exec"), ns)
load, simulate, share, fit, pools, POLICIES, FIT = (ns[k] for k in ("load", "simulate", "share", "fit", "pools", "POLICIES", "FIT"))
# label, run, the eviction order nearest to the arm's, measured share of returning turns reading from SSD
RUNS = (("nofc_noend_h100", "20261113_192822_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H100", "LRU", 0.009),
        ("jit_p50_h100", "20261113_213339_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H100", "next use from the p50", 0.010),
        ("jit_exact_h100", "20261113_233556_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H100", "exact next use", 0.006))

m70 = fit(load(FIT[1]), FIT[3])
by_ratio, by_tokens = pools(m70, 100)
print(f"M fitted at 70 GB: {m70:,}; at 100 GB by ratio {by_ratio:,}, by tokens {by_tokens:,}")
print("model's share of returning turns reading from SSD on each run's own timeline, by ratio / by tokens; * is the order nearest to the arm's")
for label, run, own, measured in RUNS:
    turns = load(run)
    cells = []
    for name, prio in POLICIES:
        a, b = (100 * share(simulate(turns, m, prio)) for m in (by_ratio, by_tokens))
        cells.append(f"{name}{'*' if name == own else ''} {a:.1f} / {b:.1f} %")
    print(f"  {label} ({len(turns)} turns, measured {100 * measured:.1f} %): " + " | ".join(cells))
