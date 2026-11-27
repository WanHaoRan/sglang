"""Closed-loop identity at saturation: N = X (TTFT + decode + Z), X = 1/g.
Inputs: minutes 90-109 means from campaign{32,34}/checks/by_minute_ec_full.out (computed with awk),
recompute per returning request from campaign34/checks/fix_minutes_table_c32_ec_full.out (13.83 K)
and ..._ec_full.out (0.36 K), FA3 recompute 0.042-0.044 ms a token (campaign32/DECISIONS.md)."""
N = 128
sent = {"c32_ec_full": 58.9, "c34_ec_full": 129.4, "c32_jit_exact": 59.1, "c34_jit_exact": 133.4}
ttft = {"c32_ec_full": 80.2, "c34_ec_full": 28.2, "c32_jit_exact": 80.0, "c34_jit_exact": 27.0}
rec_k = {"c32_ec_full": 13.83, "c34_ec_full": 0.36, "c32_jit_exact": 13.71, "c34_jit_exact": 0.37}
for a in ("ec_full", "jit_exact"):
    g32, g34 = 60 / sent["c32_" + a], 60 / sent["c34_" + a]
    dg = g32 - g34
    rec = [(rec_k["c32_" + a] - rec_k["c34_" + a]) * 1e3 * c for c in (0.042e-3, 0.044e-3)]
    cyc32, cyc34 = N * g32, N * g34
    print(f"{a}: g per request {g34:.3f} -> {g32:.3f} s (+{dg:.3f}); recompute GPU per request +{rec[0]:.3f}..{rec[1]:.3f} s")
    print(f"   cycle N*g {cyc34:.1f} -> {cyc32:.1f} s (+{cyc32-cyc34:.1f}); TTFT {ttft['c34_'+a]:.1f} -> {ttft['c32_'+a]:.1f} s "
          f"(+{ttft['c32_'+a]-ttft['c34_'+a]:.1f}, {100*(ttft['c32_'+a]-ttft['c34_'+a])/(cyc32-cyc34):.0f} % of the cycle change)")
# cleaner's recompute in c34 minutes 90-110 (evidence facts: 1.14-1.31 M a run in 22 min)
for tok in (1.14e6, 1.31e6):
    gpu = tok * 0.043e-3
    reqs = 129.4 * 22
    print(f"cleaner recompute {tok/1e6:.2f} M -> {gpu:.0f} GPU-s over {reqs:.0f} requests = {gpu/reqs*1000:.1f} ms a request "
          f"-> N*dg = {N*gpu/reqs:.2f} s of cycle (about 3/4 in TTFT)")
