# Campaign 12 — campaigns 7/8/9 (x70 / x10 / x1) with Poisson arrivals and varied conversation lengths

Requested 2026-10-01: rerun the x70, x10 and x1 evaluations with the same settings and the same total volume as
campaigns 7-9 (`../compare_20260922_023714`, `../compare_20260922_165655`, `../compare_20260923_031116`), except that
(1) conversations arrive as a Poisson process (as campaign 10, `../compare_20260923_235624`), and (2) conversations end
at different times (varied lengths), so that users come and go and there is a high-concurrency period in which every
conversation is active.

Driver: `../../scripts/run_campaign12.sh` (resumable; per-scale manifests `manifest_x70.txt`, `manifest_x10.txt`,
`manifest_x1.txt` here; driver logs `driver.log`, `driver_x<scale>_*.log`).

## Same as campaigns 7-9

Trace `lmcache_agentic_trace_262k.json`, the same 128 conversations (`OFFSET=32 NCONV=128`), gap scales x70 / x10 / x1
with `GAP_CAP=1200`, `SEED=0`, `K=0`, `CHECK_IDS=0`, level `NAT160` (L1 natural, L2 160 GB
write-through, L3 nixl on `/mnt/ssd`, cleaner 70/60, L3 wiped before every three-tier arm), Qwen3-30B-A3B-Instruct-2507
bf16 KV, `CTX=131072`, stock template, arms `hbm_host three_tier_to three_tier_wc hbm_lru` in that order, `C=128` (no
client-side cap).

## What changes

| | Campaigns 7-9 | Campaign 12 |
|---|---|---|
| Start | all 128 conversations at t = 0 (closed loop) | Poisson arrivals (`--arrival-rate`), rate per scale below |
| Length | `TURNS=40` cap: 85 conversations at 40 turns, 43 at their recorded length (18-39) | each conversation replays min(recorded turns, a seeded uniform draw in [30, 50]) (`--turns-range 30:50 --turns-seed 120`, `TURNS=50`) |
| Lengths | p10 25, p50 40, max 40 | 18-50 turns: p10 25, p25 32, p50 37, p75 42, p90 46 |
| Volume | 4,676 turns, 112.2 M prompt tokens, 0.99 M output tokens | 4,671 turns, 112.4 M prompt tokens, 0.99 M output tokens |

**Wall cap per arm** (counted from client start, so it includes the arrival window): x70 12,600 s (campaign 7's cap never
took effect — `timeout --foreground` — and its slowest arm ran 11,429 s), x10 and x1 9,736 s (campaigns 8/9's
9,000 s per conversation plus the seed-0 arrival window, 147.11 / rate s).

The length draws use their own RNG (seed 120 was picked from 200 tried as the one whose totals match campaigns 7-9 best),
so the arrival times (`--seed 0`) are untouched and every arm replays the same arrivals and lengths.

**Also different: the engine.** Campaigns 7-9 ran the pre-merge fork (`81124a061`, paced retries off). This campaign runs
HEAD `47ceeb6d26` (engine = `ace8ac79a7`, with upstream #39283 and retry defaults 8/8), the engine of campaign 11.
Campaign 11 (x10, all-at-once start, same engine) showed the engine change matters only for the SSD arms, so x10
comparisons should use campaign 11 as the burst baseline; x70 and x1 have no burst baseline on this engine.

## Arrival rates: why they differ per scale

The peak should have all 128 conversations active, so no conversation may finish before the last one arrives. How long a
conversation lasts without load scales with the gap: a 35-turn conversation takes ~35 min at x70, ~8 min at x10 and ~2.5
min at x1. Using the trace's own gaps and a 2-4 s light-load latency per turn, the rates at which no conversation can
finish before the last arrival even with no queueing are ≥ 0.2/s at x70 and ≥ 0.75/s at x10. At x1 the no-load bound
would need ≥ 1.5/s, but load slows conversations down long before that.

Pilot (`../20261001_205003_three_tier_wc_NAT160`, x1, `three_tier_wc`, 0.5/s, 8 min wall cap): the server was unloaded for
2.5 min (~4 turns/s, TTFT ~0.3 s) and saturated from ~65 live conversations (TTFT 2 → 26 s); peak 127 live; one
conversation (conv 17, 18 turns) finished at 3.9 min, before the last arrival at 4.9 min. No errors; 4,671 planned turns.

First choice (launch 21:16Z): the slowest rate per scale that keeps every conversation active at the peak — x70 0.2/s,
x10 0.75/s, x1 1.0/s (12.3 / 3.3 / 2.5 min windows).

**Changed at ~21:40Z (user: "no need to strictly overlap, a high concurrency works"): 0.2/s at every scale**, so all three
scales share one arrival pattern (128 arrivals over 12.3 min with seed 0) and a longer arrival phase. x70 was already at
0.2/s and kept running; the first driver was stopped and `relaunch_c12.sh` restarts it with 0.2/s for x10 and x1 once
x70's `run_compare.sh` ends. Expected peaks (latency model fitted to the pilot; saturation past ~65 live conversations
should push x10 and x1 higher): x70 128 (guaranteed), x10 ~110-128, x1 ~80-110. 0.1/s was rejected: x1 would peak near
26 and never saturate.

| Scale | Rate | Arrival window (seed 0) | Wall cap |
|---|---|---|---|
| x70 | 0.2 /s | 12.3 min | 12,600 s |
| x10 | 0.2 /s | 12.3 min | 9,736 s |
| x1 | 0.2 /s | 12.3 min | 9,736 s |

## Running it

- Launched with `ARRIVAL_X70=0.2 ARRIVAL_X10=0.75 ARRIVAL_X1=1.0 bash run_campaign12.sh ../results/campaign12`, then
  relaunched for x10/x1 with 0.2/s by `relaunch_c12.sh` (to resume after a reboot use 0.2 for all three); progress in `driver.log` (`SCALE START/END`) and `driver_x<scale>_*.log` (`STAGE ...`).
- Do not `git pull`, and do not edit `python/` or `agent_cache/scripts/` while it runs: every arm boots and replays from
  the working tree (`scripts_at_launch_*.txt` records the uncommitted script changes at launch).
- Client records now carry `end_s` on `conv_end` and `arrival_s` on `conv_skipped`, so live conversations over time are
  exact (arrival_s ≤ t < end_s).
- SSD telemetry: `iostat_vdc_x<scale>.log` here.
- Analysis per scale into its own directory, e.g. `python3 scripts/timeline.py --manifest results/campaign12/manifest_x70.txt
  --out results/campaign12/x70` (the tools write fixed file names next to `--out`).
- After a host reboot: `sudo docker start sglang_hicache`, then relaunch with `ARRIVAL_X70=0.2 ARRIVAL_X10=0.2 ARRIVAL_X1=0.2`; finished arms are kept and the
  interrupted arm restarts.
- Expected ~24-30 h: per arm ~1-2.5 h at x10/x1 (the cap binds for the slowest arms, as in campaigns 8/9) and up to
  ~3.5 h for x70 hbm_lru (campaign 7: 3.2 h), plus 4-7 min per boot.

## 2026-10-02: engine edits during the run, outage, frozen engine

- **Engine per arm.** The user's oracle step 2 (`2abe005bb2`, committed 01:12Z: chat `kv_hints` transport + the
  `--hicache-storage-next-turn-prefetch` flag, default `off`) and an uncommitted, uncalled step-3 helper
  (`resident_prefix_len`) landed in the working tree while the campaign ran. Arms booted before 01:12Z (x70 `hbm_host`,
  `three_tier_to`) ran engine `47ceeb6d26`; later arms ran `2abe005bb2` (their `server.log` lists
  `'hicache_storage_next_turn_prefetch': 'off'`). The campaign client sends no hints, so the flag is off and nothing
  calls the helper: no behaviour difference is expected between these arms.
- **Outage.** The host went down at ~08:33Z (last log lines) and came back at ~14:29Z. x10 `three_tier_to`
  (`../20261002_075232_three_tier_to_NAT160`, ~40 min in) was cut off and is not used; x10 `hbm_host` had finished.
- **Frozen engine from 14:33Z.** To keep later edits out of the remaining arms, the driver was relaunched with
  `ENGINE_PYTHONPATH=agent_cache/frozen/engine_2abe005bb2/python` (`git archive 2abe005bb2 python`, ignored via
  `.git/info/exclude`); `start_server.sh` puts it on `PYTHONPATH`, prints `engine: ...` at boot and records it in
  `run.txt`. Remaining: x10 `three_tier_to`, `three_tier_wc`, `hbm_lru`, then all four x1 arms (~14-16 h).
- To resume after another reboot: `sudo docker start sglang_hicache`, then the launch command with
  `ARRIVAL_X70=0.2 ARRIVAL_X10=0.2 ARRIVAL_X1=0.2 ENGINE_PYTHONPATH=/sgl-workspace/sglang/agent_cache/frozen/engine_2abe005bb2/python`.

## 2026-10-03: done

- `CAMPAIGN DONE 2026-10-03T02:28:15Z`: all 12 arms, 0 errors. x10 `hbm_lru` hit its wall cap with 2 conversations
  unfinished; every other arm finished all 4,671 turns. The GPU is free again, and the do-not-edit rule above no longer
  applies.
- Results, analysis and caveats: `README.md`. New analysis scripts: `scripts/vs_baseline.py` (whole-run statistics and
  same-turn differences against a burst baseline), `scripts/requery_share.py` (SSD reads by the lookup that triggered
  them) and `scripts/c12_summary.py` (`summary.png`).
- Correction to "Arrival rates" above: the no-load durations (35 / 8 / 2.5 min for a 35-turn conversation) were rough.
  From the replayed gaps, the median conversation (37 turns) needs about 32 / 6-7 / 2-3 min at x70 / x10 / x1 with 2-4 s
  per turn. The chosen rates are unaffected.
