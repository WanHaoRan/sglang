# Campaign 9 — campaign 7 at GAP=1 (the trace's native gaps)

Launched automatically by `../chain_c9.sh` when campaign 8 printed ALL DONE (marker `../.c9_launched`, driver log
`../compare_driver_c9.log`). Everything identical to campaigns 7 and 8 (`../compare_20260922_023714/`, `../compare_20260922_165655/`)
except **`GAP=1`**; `CLIENT_TIMEOUT=9000` enforced; the server logs per-request `ReqTimeStats` (queue_duration /
forward_duration) as campaign 8's last arm did, so the queue wait is measured for every turn.

## Why

The third point of the gap-scale series (x70 / x10 / x1). At native gaps (p50 0.70 s, p90 1.22 s; 14 % of gaps >= the
1 s L3 read of a 20K-token restore, `../compare_20260922_023714/gap_distribution.png`) the closed loop of 128 sessions
offers ~20 turns/s, far above any arm's capacity, so every arm runs at server capacity from the first minute. The
question is whether anything about the tiers survives that regime, and how the queue / prefill / decode split
(`../analysis_gapscale/delay_components.py`) moves against x10 and x70.

## Expected, so a wrong number is recognisable

- All four arms saturate immediately (queue depth ~60-100 of 128) and hit the 9,000 s cap with ~4,650 turns each.
- The x10 finding should repeat: the arrival-time L2 hit expires during the queue wait (turnover ~ queue wait) and the
  SSD arms restore almost nothing from L3 (campaign 8: 6-10 storage hits per arm), so hbm_host ~ three_tier_to ~
  three_tier_wc, with hbm_lru behind them only by its device-hit share.
- Mean TTFT ~ 110-120 s for the three host-pool arms, a little more for hbm_lru; recompute p50 ~ 210 s.
- If an SSD arm shows materially more storage hits than at x10, the shorter gaps kept the L2 chain alive between
  turns (fewer competing insertions per wait), which would be worth understanding before the joint analysis.

## 13:10Z: campaign complete (ALL DONE 13:08:34Z); artefacts generated

Arms 1-3 finished on their own (client wall 8,628 / 8,687 / 8,679 s: at native gaps the tail has no long idle stretches);
hbm_lru hit the 9,000 s cap with 4,561 turns (29 conversations cut at a turn boundary). Returning-turn means (compare.csv):
hbm_host 130.5 s, three_tier_to 131.2 s, three_tier_wc 131.4 s, hbm_lru 150.0 s; storage hits 0 / 3 / 2 / 0 (15 restores
in all in each SSD arm counting warm-up); cold returns 59 / 59 / 59 / 82 %; every turn has a measured queue duration
(ReqTimeStats): mean 108.9-109.5 s for the host-pool arms, 122.1 s for hbm_lru, forward 55-68 s. The x10 finding holds at
native gaps: the arrival-time L2 hit expires in the queue, the SSD tier is never reached, all host-pool arms coincide.
Figures: compare.png, timeline_<arm>.png; cross-scale set in ../analysis_gapscale/ (crossscale.csv/png,
delay_components.png, turn_components_*.csv). The in-container iostat logger was stopped by chain_c9.sh at ALL DONE.
