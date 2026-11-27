# Campaigns 35 to 43: the unified four-action design against the current one

Ordered 2026-11-26 by the author ("refine the design part and verify the effectiveness ... then implement the
strategy and compare with current one"). Design: `results/design_unified_1126/DESIGN.md`; proofs: `PROOFS.md` there;
slides: "Four Actions, Three Prices". This file is the record for all nine campaign folders; each folder's
`README.txt` has its own log of events.

## Question

Does the unified design (Fetch, Retain, Preempt and Price on every tier, with the queue wait and the hint's
distribution in the model) beat the current design (`ec_full`: the four rules on the gap table) on the same users, and
which part does the work: the cache rules (Fetch and Price by measured prices, Retain on the SSD), or the load hold
(Fetch on the top tier: the copy from host memory runs beside decoding)?

## Settings and folders

FreeInference on FA3, 0.1 users/s, recorded waits x1, seed 0, one H200 (`GPU-b1c6de25`), every server bound to
127.0.0.1, the four write-path fixes of campaign 34 on in every arm. The launcher takes switches per campaign folder,
so an arm with another switch set runs in a sibling folder with the same users; requests pair by user, stream and
index across the folders of one setting.

| Setting | Baseline (`ec_full`) | Hold only (`ec_full_load`) | Cache rules (`unified`) | Both (`unified_load`) |
|---|---|---|---|---|
| A: 128 users, 160 GB host, 2 h (campaign 34's) | campaign 35 | campaign 39 | campaign 35 | campaign 40 (45: fixed price; 44: + pass-over) |
| B: A with the SSD cleaner's marks at 50/40 % | campaign 38 | | campaign 38 | campaign 41 |
| C: 128 users, 112 GB host, 2 h | campaign 37 | | | campaign 42 |
| D: 64 users, 88 GB host, 1 h | campaign 36 | | | campaign 43 |

Campaign 44 adds one arm at setting A: everything of campaign 40 plus the admission pass-over (Preempt on GPU room:
behind a turn refused for room, up to 8 later turns that fit are admitted, at most 8 times per refused turn;
`SGLANG_PREFILL_PASS_OVER_LOOKAHEAD=8`). It runs after the six arms above (`campaign44/after_chain.sh`).
Campaign 45 (added 2026-11-27 after campaign 40) is campaign 40's arm with the price of host room fixed at the
current design's 5e-5 instead of measured (`..._MEASURED_PRICES` and `..._HOLD_QUANTILE` off, same engine copy): the
measured price was 1e-8 to 7e-7 and staged reads early (finding 3 below). It ran first, from 16:13Z, while campaign
44's pre-flight waited to be repeated without the one test its engine copy is meant to fail
(`campaign44/README.txt`).

Switch sets (all off by default):

- Hold: `SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=16`.
- Cache rules: `SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES`, `..._LATENESS_WEIGHT`, `..._READ_LINE`, `..._HOLD_QUANTILE`,
  `..._FREE_EMPTY_LOOKUPS`, `SGLANG_HICACHE_STORAGE_TRACK_STORED`, `..._SKIP_ENDED_WRITES`,
  `SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S=300`, `SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES`,
  `SGLANG_LOG_HICACHE_PRICES_INTERVAL=10`; on the queue line price learning off and
  `--radix-eviction-policy-config {"horizon_s":30,"queued_order":"farthest"}`.
- Read-or-recompute by price (`..._RECOMPUTE_BY_PRICE`, `..._RECOMPUTE_WHEN_STARVED`) is built and stays off: it loses
  in every simulated setting (`results/design_unified_1126/sim/POLICY_RESULTS.md`).

## Baselines

Requests sent before 6,600 s (3,000 s in setting D); pairs are B - A on the requests both runs sent, with a 95 %
interval over users (`compare_arms.out` in the folders named).

| Run | Mean TTFT | p50 / p90 / p99 | Requests | Against | Paired difference |
|---|---|---|---|---|---|
| campaign 35 `ec_full` | 10.97 s | 8.21 / 23.66 / 53.36 | 15,560 | campaign 34 `ec_full` (another GPU unit, 3 days earlier) | +0.01 s [-0.09, +0.11] (`campaign35/checks/vs_c34`) |
| campaign 38 `ec_full` (cleaner from minute 50) | 11.56 s | 8.48 / 25.06 / 57.48 | 15,370 | campaign 35 `ec_full` | +0.77 s [+0.61, +0.92] (`campaign38/checks/vs_c35`) |
| campaign 37 `ec_full` (112 GB) | 12.82 s | 0.26 / 63.34 / 112.97 | 15,455 | campaign 35 `ec_full` | +2.26 s [+1.30, +3.25] (`campaign37/checks/vs_c35`) |
| campaign 36 `ec_full` (64 users, 88 GB, 1 h) | 0.68 s | 0.10 / 1.59 / 10.64 | 7,002 | | |

1. Two runs of the current design at setting A differ by 0.01 s: a paired difference of 0.1 s or more is a result.
2. The cleaner's cost is measured directly: the same design with the cleaner from minute 50 instead of 88 is 0.77 s
   slower (minutes 60 to 110: 22.34 s against 20.80 s; prompt tokens recomputed 7.0 % against 6.2 %).
3. A tight host pool costs 2.3 s and splits the requests: half get their first token within 0.3 s, the readers wait
   far longer (p90 63 s).
4. At 64 users the GPU pool fills only now and then: 0.68 s, with a p99 of 10.6 s.

## Expected (simulator and replay, before the runs)

On the calibrated and refereed simulator, on these users (`sim/POLICY_RESULTS.md`): the hold up to -3.1 s at A (an
upper bound: the simulator gives the copy's GPU time back entirely and charges half of the blocked time the logs
show); the cache rules -0.3 s at A, -0.8 s at B, -1.0 s at C, -0.2 s at D; both up to -3.4, -3.8, -4.1 and -0.2 s.

## Results

Paired differences are B - A on the requests both runs sent before the cut, with a 95 % interval over users
(`results/compare_unified/<setting>/compare_arms.out`; figures `timeline.png`, `ttft_cdf.png` there).

| Setting | Arm | Mean TTFT | p50 / p90 / p99 | Against the current design | Requests paired |
|---|---|---|---|---|---|
| A | `ec_full` (campaign 35) | 10.97 s | 8.21 / 23.66 / 53.36 | | |
| A | `ec_full_load` (campaign 39): the hold alone | 9.36 s on the pairs (9.68 s on its own 15,944 requests) | 6.95 / 20.60 / 50.29 | -1.61 s [-1.79, -1.43] | 15,560 |
| A | `unified` (campaign 35): the cache rules alone | 10.58 s | 8.33 / 23.13 / 53.49 | -0.38 s [-0.51, -0.25] | 15,548 |
| A | `unified_load` (campaign 40): everything | 9.18 s on the pairs (9.52 s on its own 16,107) | 7.17 / 20.55 / 49.91 | -1.79 s [-2.00, -1.59]; against the hold alone -0.22 s [-0.32, -0.11] | 15,560 |
| A | `unified_load_po` (campaign 44): everything + admission pass-over | 9.10 s (9.51 s on its own 16,130) | 6.13 / 21.70 / 55.07 | -1.87 s [-2.13, -1.61]; against campaign 40 -0.02 s [-0.22, +0.17] | 15,560 |
| A | `unified_load_fixedprice` (campaign 45): everything, fixed price of host room | 9.19 s (9.57 s on its own 16,095) | 7.21 / 21.01 / 48.20 | -1.79 s [-1.99, -1.59]; against campaign 40 +0.04 s [-0.04, +0.13] | 15,560 |
| B | `ec_full` (campaign 38) | 11.56 s | 8.48 / 25.06 / 57.48 | | |
| B | `unified` (campaign 38): the cache rules alone | 10.56 s | 8.46 / 23.33 / 55.60 | -0.99 s [-1.18, -0.81] | 15,366 |
| B | `unified_load` (campaign 41): everything | 9.07 s on the pairs (9.66 s on its own 16,040) | 7.20 / 21.01 / 50.07 | -2.49 s [-2.76, -2.21] | 15,370 |
| C | `ec_full` (campaign 37) | 12.80 s | 0.26 / 63.34 / 112.97 | | |
| C | `unified_load` (campaign 42): everything | 9.41 s on the pairs (10.04 s on its own 16,270) | 2.99 / 36.72 / 78.10 | -3.39 s [-4.09, -2.74] | 15,449 |
| D | `ec_full` (campaign 36) | 0.68 s | 0.10 / 1.59 / 10.64 | | |
| D | `unified_load` (campaign 43): everything | 0.54 s | 0.10 / 1.40 / 7.26 | -0.14 s [-0.20, -0.08] | 6,998 |

1. The load hold alone is worth 1.61 s of 10.97 s (15 %) at setting A, nothing at light load (-0.008 s in the first
   28 minutes) and 2.5 to 3.5 s between minutes 50 and 90. On the GPU's own log the pass no longer waits on the copy
   (slope of interval length on copy time 0.71 -> 0.00) and the GPU runs 4.8 % more decode steps a second; the
   closed loop's accounting turns that into -1.34 s against -1.44 s measured (`campaign39/README.txt`,
   `campaign39/checks/vs_c35/`). The simulator's free-copy bound was -2.6 s: the copy's cost is its excess over the
   pass's own compute (Proposition 23), not the whole copy.
2. The whole design at setting A: -1.79 s (16 %). The cache rules add -0.22 s to the hold, as the simulator said
   (-0.28 s), and nearly all of it once the SSD cleaner deletes (minutes 88-110: -1.16 s; recomputed history 0.32 %
   of prompt tokens against 1.27 %). Before that they read more from the SSD and cost +0.16 s in minutes 50-70. The
   two parts add up: -1.61 and -0.22 against -1.79 together (`campaign40/README.txt`).
3. The measured price of host room is too low once the GPU is saturated: 1e-8 to 7e-7 per token-second against the
   fixed 5e-5. It prices a displaced history by the stall of reading it back at c (0.01 to 0.07 under load) and not
   by the recompute a displacement can cause (about mu x 2e-5 = 1e-3 per token). So staging before arrival looked
   free: campaign 40's reads landed 6.3 s before admission (1.6 s with the hold alone), pins were held 9.2 s on
   average (1.6 s), and 7 staged copies were held until the 600 s safety release; one traced turn
   (c102-t16) staged 54K tokens 4 minutes after its commit and arrived 13 minutes later. Campaign 45 runs the same
   arm with the fixed 5e-5: the early staging goes (no pin at the safety release, reads land 2.1 s ahead on
   average against 14.0 s, 5.6 against 9.0 M tokens read in minutes 50-70), and mean TTFT does not move (+0.04 s
   [-0.04, +0.13] against campaign 40), minutes 50 to 70 included. So the measured price wastes SSD reads and host
   room without costing TTFT at this setting, and the +0.16 s of minutes 50 to 70 (cache rules on top of the hold)
   has another cause; no arm separates the candidates. A fix in the design (pricing a victim by min(read,
   recompute) with the chance the read fails, and a drop time for staged copies, Theorem 3 b) is not built.
4. Setting B (the cleaner deletes from minute 50 instead of 88): everything -2.49 s (22 %), -3.2 to -4.8 s a window
   once the cleaner runs; recomputed history 0.20 to 0.51 % of prompt tokens against 1.37 to 1.75 %. Less the hold's
   -1.61 s at A, about -0.9 s is the cache rules', the simulator's -0.93 s (campaign 38's `unified` arm measures it
   alone). (`campaign41/README.txt`)
5. Setting C (112 GB of host memory): everything -3.39 s (26 %), the largest gain, and the tail most of all (p90
   63 -> 37 s, p99 113 -> 78 s). The median rises from 0.26 to 2.99 s and 44 % of paired requests are faster: the
   current design served 43 % of the history from the GPU and kept readers waiting behind those requests; the
   unified design serves readers sooner, at 5 % more throughput. The paper's metric, mean TTFT, falls; a reader of
   medians would see a regression (`campaign42/README.txt`, `compare_unified/C_112g/ttft_cdf.png`).
6. Setting D (64 users, 88 GB, 1 h): everything -0.14 s (21 %), as simulated; p99 10.6 -> 7.3 s. With the four
   settings measured the whole design is faster in every one: -1.79, -2.49, -3.39 and -0.14 s
   (`campaign43/README.txt`).
7. The cache rules alone at A: -0.38 s, gaining in every window (-0.14 to -1.03 s), the most once the cleaner
   deletes. With the hold they add -0.22 s, and cost +0.16 s in minutes 50 to 70 where alone they gain -0.38 s: the
   parts are slightly sub-additive (-1.61 and -0.38 apart, -1.79 together), and the mid-run cost of finding 3 shows
   only with the hold (`campaign35/README.txt`). The simulator: -0.22 s (five seeds), -0.28 s (these users).
8. The cache rules alone at B: -0.99 s, the simulator's -0.93 s, all of it from minute 50 when the cleaner starts
   (-1.0 to -2.3 s a window). The hold adds -1.45 s to them. So the Retain rules on the SSD are worth about 1 s when
   the cleaner deletes for an hour and 0.4 s when it deletes for 20 minutes; the hold is worth 1.5 to 1.6 s at
   saturation in both (`campaign38/README.txt`).
9. The measured price against the fixed one (campaign 45): a tie, 9.19 against 9.18 s (+0.04 s [-0.04, +0.13]).
   The Price action's measured lambda changes where reads land, not the mean (finding 3; `campaign45/README.txt`).
10. The admission pass-over (campaign 44) ties on the mean (-0.02 s [-0.22, +0.17] against campaign 40) and moves the
    distribution: the median 7.17 -> 6.13 s, p99 49.9 -> 55.1 s. Small turns going ahead of a refused large one is
    shortest-job-first in effect; the 2 % of the GPU pool it recovers does not show in the mean. It stays off by
    default (`campaign44/README.txt`).

## Engine copies

- `frozen/engine_fa3fix_20261122`: the baseline arms.
- `frozen/engine_loadhold_20261126`: the baseline plus the load hold only; campaign 39's `ec_full_load` runs on it
  (pre-flight record in `campaign39/README.txt`: tier content exact under held batches, 678 held batches in a
  ten-minute smoke with no error).
- `frozen/engine_unified_po_20261127`: campaign 44's engine, the whole build plus the pass-over and the fixes of the
  review of the scheduler side (`results/design_unified_1126/impl/REVIEW_SCHED.md`); `README.txt` there says what
  differs in served behaviour from `engine_unified_20261126`.
- `frozen/engine_unified_20261126`: the whole build, every switch off by default (`README.txt` there: how it was
  merged and checked). Its first boot stopped at argument checking (the eviction config key `queued_order` was not
  accepted under `return_chance`); fixed and frozen again at 02:14Z on 2026-11-27, before any arm. GPU pre-flight:
  `campaign40/preflight.log` (tier check with every switch on, quiet and under load; two ten-minute smokes, switches
  off and on).

Found after both copies were frozen (03:23Z on 2026-11-27): on `engine_loadhold_20261126` and
`engine_unified_20261126` four cases of two existing unit test files that build a `Scheduler` without `__init__` fail
(the hold's attribute was set in `__init__` only). A server always runs `__init__`, so the arms are unaffected; the
working tree has class-level defaults and passes them.

## Checks around the comparison

- `results/compare_unified/update.sh`: paired tables per setting and the figures (`timeline.png`, `ttft_cdf.png`).
  The launcher's own figure check (`arm_checks.sh` -> `c28_users/plot_arms.py`) reports FAIL for the new labels
  (`ec_full_load`, `unified`, `unified_load`) because that figure set does not know them; it is not a run fault.
- `campaign39/checks/hold_gpu_time.py`: what a prefill batch costs the running turns against its copy time, per run
  (the hold's mechanism, measured on the GPU's own log).
- `campaign39/checks/pair_early.py`: paired TTFT in a window, split by what the request loaded (the hold's cost for
  the held turn at light load).
- `results/design_unified_1126/main/checks/pass_model.py`, `prop23_check.py`: Proposition 23 (the pass meets the
  copy) against the logs and by simulation of the layer recursion.
