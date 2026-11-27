# Referee notes on kvsim (2026-11-26)

Reviewed: `../kvsim.py` (md5 650ab9d2...), `../kvsim_policy.py` (md5 a51bc5a7..., 1,438 lines), `../kvsim_trace.py`,
`../kvsim_compare.py`, `../CALIBRATION.md`. Nothing outside this folder was written. CPU only, nice 19, 4 processes
at most (once 5 for about 3 minutes, when an invariant check overlapped the sensitivity matrix).

`../kvsim_policy.py` was changed by someone else at 20:10 UTC during the review (+21 lines: the switch `F4B`, not
part of `all`). The five anchored arms give the same seed-0 numbers before and after (10.62 / 31.58 / 11.11 / 10.58 /
10.60 s). Line numbers below for `kvsim_policy.py` are the reviewed version's; `kvsim.py` did not change.

## 1. Verdict

- The simulator reproduces what the builder claims for campaign 34 (fixes on), campaign 28 (Triton) and, roughly,
  campaign 32 (fixes off). I re-derived every measured number from the client records and server logs
  (`ref_measured.py`) and re-ran every anchored arm on seed 0 and on eight fresh seeds (5 to 12).
- Seven defects in the code, all small; a patched copy is `kvsim_patched.py` (built by `make_patched.py`, diff in
  `kvsim_patched.diff`). With it every invariant holds in every mode, and the anchors are the same or slightly
  better. Use `kvsim_patched.py` (through `kvsim_compare_patched.py`) from here on.
- Two claims in `CALIBRATION.md` are wrong: pending writes are not "about half" of the measured ones but 2 to 4
  times them (the comparison used the census field `h_write` instead of `pending_write`); and the GPU-speed
  sensitivity of `hicache` is not smooth (a GPU 0.5 or 1 % faster gives 19.7 s, not about 31 s).
- What it can decide: effects of 2 s or more that hold on five or more seeds, and the sign of 0.2 to 0.7 s
  effects of read timing when the sign holds on every seed. What it cannot: differences under about 0.5 s as
  statements about the GPU, anything that runs through refused reads or the write backlog with the fixes on, the
  timing of the loop with the fixes off, and settings with no GPU anchor (section 7).

## 2. Files here

| file | what it is |
|---|---|
| `ref_measured.py` -> `measured.json`, `measured.out` | the measured side: campaigns 34 and 32, client records and server logs, streamed |
| `ref_anchors.py`, `reflib.py` -> `anchors_table.md`, `anchors_table_patched.md` | every anchor, measured against seed 0 and fresh seeds, with verdicts (255 rows for campaign 34, 225 for campaign 32, 16 paired rows) |
| `ref_workload_check.py` | the simulated client against all 11 recorded runs |
| `ref_gpu_work.py` | GPU work per window (decode share, non-decode time per new token) |
| `ref_invariants.py` -> `invariants/` | invariant checks by subclassing `Sim`; `inv_*` = original, `invp_*` = patched |
| `ref_cleaner_check.py` | defect D4 shown directly |
| `make_patched.py` -> `kvsim_patched.py`, `kvsim_patched.diff` | the patched copy (17 exact text edits, each marked `[referee Dn]`) |
| `kvsim_compare_patched.py`, `ref_run.py` | `kvsim_compare.py` and a plain matrix runner on the patched copy |
| `ref_sens.py` -> `sens_report.md`, `runs/sens/` | each fitted constant x 0.8 and x 1.2: 660 runs |
| `ref_noise.py` -> `noise_seed0.md` | the floor under a paired difference (six replicas with 5 ms send jitter) |
| `policy_patched.out` | four single switches on the patched copy, the settings of `../POLICY_RESULTS.md` |
| `kvsim_only_d2.py`, `kvsim_no_d2.py` | scratch variants for one ablation (section 4) |
| `runs/` | every run (551 MB); `logs/` the run logs |

## 3. Anchors (original simulator; seed 0 = the campaigns' users)

Workload: all 154,415 recorded requests of the 11 runs of campaigns 34 and 32 match the simulated client (prompt,
output, wait, gap kind, key, tree; every arrival time). Replayed in the recorded send order, the hint table gives the
recorded p50 on 61,518 of 61,523 hinted requests.

Campaign 34, reproduced (measured -> simulated):
- Mean TTFT 31.88 / 11.47 / 10.92 / 10.96 / 10.66 s -> 31.58 / 11.11 / 10.58 / 10.62 / 10.60 (`hicache`, `jit_p50`,
  `jit_exact`, `ec_full`, `ec_exact`). Half-hour cells: 19 of 20 within 10 % (16 within 5 %); the 20th is -10.5 %
  (`jit_exact`, minutes 0-30, 0.40 -> 0.36 s). The builder's "all 20" is off by that one cell.
- Requests a minute 141.5-142.2 -> 142.3-142.6 (`hicache` 90.6 -> 90.9); minutes 90-109: 128.3-132.6 -> 132.2-134.6
  (`hicache` 55.3 -> 55.9).
- Running requests, median: 22 / 20-21 / 20 in minutes 30-60 / 60-90 / 90-110 -> 22 / 20-21 / 20; queued 60-61 -> 59-60
  (`hicache` 83 -> 83); pool use 0.97 -> 0.973-0.980.
- The cap of 128 users: first reached at minute 74-75 -> 72.7-74.5, held from minute 85-86 -> 85.0-85.2 (`hicache`
  53 -> 52); requests in flight in minutes 90-110: 80 -> 79 (`hicache` 104 -> 103). Not in the builder's table.
- GPU work (`ref_gpu_work.py` against `calib_gpu_fit.py` on the logs): decode share 0.760 / 0.767 / 0.760 -> 0.757 /
  0.766 / 0.768 (`ec_full`); 0.475 / 0.328 / 0.325 -> 0.476 / 0.326 / 0.330 (`hicache`, which was not in the fit);
  non-decode time per new token 0.061-0.065 -> 0.061-0.063 ms and 0.038-0.042 -> 0.038-0.041 ms.
- Decode time per output token on the client's clock (TPOT, which the builder did not check): 30.4-31.4 -> 30.6-31.0
  ms in minutes 30-110 (hinted arms); `hicache` 49.7 / 69.6 / 70.9 -> 50.6 / 69.2 / 71.2 ms.
- Tier shares within 0.01; history computed again 0.39-0.53 % -> 0.40-0.61 %, `hicache` 28.9 -> 29.1 %; `hicache`
  lookups 39,963 -> 40,504, requests at the 8-lookup cap 3,789 -> 3,874.
- Who waits (minutes 90-110): readers 26-32 % -> 27-34 %, their TTFT 46.7-69.6 -> 46.0-66.6 s, the others 13.1-19.0
  -> 12.0-16.9 s.
- Pinned host tokens (census `pinned`): `jit_p50` median 153 K, p90 319 K -> 172 K, 310 K; the others median 0, p90
  55-61 K -> 0-2 K, 57-70 K. (The builder compared with 136 K, which is the staged part only.)
- Read thread busy 0.09-0.55 / 0.56-0.97 / 1.00 -> 0.12-0.54 / 0.56-0.95 / 0.99-1.00; `hicache` 0.34-0.38 ->
  0.38-0.44. Write thread busy in minutes 90-110: 0.73-0.74 -> 0.66-0.67.
- First cleaner pass 5,252-5,301 s -> 5,190 s (the capacity was fitted to it); none for `hicache` in both; passes at
  intervals of 600-700 s in both, 2.5 M tokens each.
- Reads staged before arrival 1,208 / 99 / 1 / 39 -> 1,216 / 103 / 1 / 46; SSD reads and tokens within 7 %.
- Paired `jit_p50` -> `ec_full` -0.62 [-1.02, -0.22] -> -0.59 [-1.02, -0.19]; fresh seeds -0.20 to -0.47, the same
  sign in all nine seeds. `hicache` -> `ec_exact` -26.96 -> -26.63; fresh seeds -5.0 to -33.9.
- Light load (minutes 0-30): mean 0.39-0.41 -> 0.36-0.40 s, p50 0.086-0.094 -> 0.083-0.094 s, GPU share 0.68-0.76
  -> 0.70-0.77 in the same order across arms.

Campaign 34, not reproduced:
- Pending SSD writes (census `pending_write`, the counter the hold reads): p90 12-17 K -> 33-41 K, p75 3-6 K ->
  11-22 K, max 51-57 K -> 78-93 K. `CALIBRATION.md` says "about half (33 K against 65 to 94 K)"; 65-94 K is
  `h_write`, host-only tokens held by pending writes with their ancestors, another quantity. The write hold fires
  51-80 times against 28-30 for the same reason.
- Refused reads before the cut with the fixes on: 0 in every arm -> 46 for `jit_p50` (0-58 over the fresh seeds), 0
  for the others on seed 0 (0-17 over the fresh seeds); 147 copies of new pages to host refused for `jit_p50`
  against 0 in the census.
- `ec_exact`'s lead: `ec_full` -> `ec_exact` -0.40 [-0.51, -0.29] -> +0.01; -0.12 to +0.06 over the fresh seeds. This
  one cannot count against the simulator: `ec_exact` ran on another GPU unit, the units differ by up to 0.4-0.7 %
  (`campaign34/checks/gpu_unit.out`), and a GPU 0.5 % faster under `ec_exact` alone gives -0.48 s here.
- SSD writes of `hicache`: 29.4 M tokens written (65.3 M skipped as stored) -> 19.9 M: the engine wrote 9.5 M
  tokens of recomputed pages again that the simulator skips.
- Free host room 16-18 K -> 11-13 K; cleaner passes before the cut 2 / 3 / 2 / 3 -> 3.

Campaign 32 (fixes off), roughly:
- Mean TTFT 33.52 / 17.97 / 14.86 / 14.64 / 14.95 -> 32.08 / 15.35 / 13.45 / 13.48 / 13.58 (one run; five jitter
  replicas: 31.2-32.0 / 14.1-18.7 / 13.35-13.60 / 13.33-13.74 / 13.50-13.62).
- Loop onset minute 42 (67) / 80 / 84 / 79 -> 78 / 88 / 88 / 87 (`jit_p50` 42-89 over replicas; 40-105 over the
  fresh seeds; a loop in 8 of 8 fresh seeds for the two `jit` arms and 7 of 8 for the two `ec` arms).
- Not reproduced: p90 TTFT of `jit_exact`, `ec_full`, `ec_exact` 50.1-51.2 -> 28.2-30.9 s; history computed again
  in minutes 60-90: 4.9-8.0 -> 2.5-2.8 M tokens; `hicache` SSD share 0.045 -> 0.068; the order among the hinted
  arms (`ec_full` -> `jit_exact` +1.26 -> -0.04); the fixes' gain -3.7 to -6.5 s -> -2.9 to -4.2 s (`hicache` -1.6 ->
  -0.5); first cleaner pass 5,584-6,798 s -> 5,190-5,730 s.

Campaign 28 (Triton), reproduced on my rerun: 115.7 / 49.4 / 49.8 / 48.3 / 48.6 -> 121.3 / 50.2 / 49.6 / 49.8 / 50.0
s; requests a minute 41.3 -> 39.3 and 76.9-78.9 -> 76.9-77.2; by window within 10 %; SSD share 0.41-0.44 -> 0.46-0.47.

The absolute agreement on seed 0 is in-sample for the GPU: the decode step and the prefill terms were fitted to
campaign 34's own decode-log intervals, and in a closed loop a throughput matched to 0.5 % fixes mean TTFT to a
few per cent. The out-of-sample evidence is `hicache` (GPU work, lookups, the storm, bin by bin), the mechanism
counters (staged reads, pins, capped lookups), campaign 28 and, weakly, campaign 32.

## 4. Code review and invariants

`ref_invariants.py` checks, after every event: host, GPU and SSD use within their pools, no negative counter, users
in session <= cap, no event scheduled in the past; at every 10 s sample, by a full scan: host tokens by class
(mirror + host-only + reads in flight + orphans + free = pool), GPU tokens (nodes + running requests' private
tokens), pending writes, pins, SSD tokens, child counts, the two candidate sets, prefix-closed residency on GPU and
in memory, every running request's pages on the GPU and locked, queue in arrival order; at every send: sent once,
after every dependency's reply plus the recorded gap; at every commit: the engine's plan carries the sent hint (the
table's p50 in the table arms, the true gap only in the exact arms); per device job: its rate.

Passed everywhere: the closed loop, the cap, causality of the hints, arrival order of admissions, tier tokens adding
up, device rates (reads 15.6 K or 24.0 K tokens/s, writes 5.8 K or 18.8 K), GPU time never double-booked.

Defects (line numbers in `../kvsim.py`):

| | where | what | effect measured |
|---|---|---|---|
| D1 | 327 `_stream_next` | a send is scheduled at min(t, deadline) after the deadline has passed: the clock steps back by up to 30 s | 23-72 events a run, all in the grace period after 7,200 s; no effect before the cut |
| D2 | 437-449 `_hlock_dec`, 557-572 `_orphan`, 874-878 `backup`, 940 `_write_ack` | a host node that is unreachable but still locked (an "orphan") and is computed again is held twice and freed once, or freed without returning to the evictable set; a node with two writes pending counts as clean after the first | fixes off only, in 6 of the 12 runs checked: host tokens leak (up to 5.8 K by the end of a run), up to 12 evictable leaves missing, pending-write counter off by up to 9.5 K. Not seen with the fixes on |
| D3 | 1246-1252 `_finish` | the reply's pages are inserted on the GPU even when nothing could be evicted for them | GPU pool exceeded 7-197 times a run, by up to 8,073 tokens (1.2 %) |
| D4 | 960-962 `_cleaner` with 977-983 `ssd_oldest` | the generator pops a node before the loop tests the target; the first node not deleted leaves the cleaner's order for good | 4 passes -> 4 stored nodes (2.6-5.4 K tokens) no later pass can find (`ref_cleaner_check.py`) |
| D5 | 761-767 `_read_next`, 785-803 `_read_landed` | a read parked or queued while the cleaner deletes its pages is served as if they were there | 0-14 reads a run (up to 129 K tokens) served from deleted pages; the engine fails such reads (9-12 an arm) |
| D6 | 1123-1128 `_admit` with 521-529 `_gpu_drop`; 794-797 `_read_landed` | mirror off: the GPU eviction of an admission copies pages to host and may evict the host pages the same admission is about to load, which are then loaded from nowhere; `host_mirror` goes negative | `jit_p50`, mirror off: memory set not prefix-closed in 401 of 723 samples (45 nodes), child counts wrong; `host_mirror` negative in every mirror-off run (to -14 K) |
| D7 | 1191-1201, 1223 `_prefill_done` | the infinite-cache ideal is taken at the end of prefill from pages that were prefilled; the client takes it at the send from requests that completed | no row with cached > ideal, where the recorded clients have 3-6 a run; history computed again a few per cent high |

Not patched (model limits, or by design):
- The write path. One FIFO of nodes at a fixed rate; no merged writes. Result: the pending-write distribution above.
- The read thread alone runs at 24.0 K tokens/s, above the device's 21.4 K (`Config`, lines 94-96; `_retime`, 752).
  "Device work never exceeds its rate" holds only while both threads run. It matches the read thread's measured
  per-page time (2.5-2.8 ms uncontended), so it is a choice, not an error.
- One read share (0.73) for both campaigns. In campaign 32's loop (minutes 90-110) the write thread delivered 10.7 K
  tokens/s at 100 % busy while the read thread was 76-80 % busy; the simulator's split gives it about 8.6 K at
  that duty. See section 5: the loop's existence turns on this constant.
- `Policy.session_ended` (`kvsim_policy.py` 424-431) reads `r.is_last` in the exact-hint mode, where the engine
  sees only "no hint" (also sent when the next send would fall past the deadline). Used by R1 only; no effect
  inside a run.
- Free actions: R4's touches of SSD copies cost no device or CPU time; the cleaner is instant.

The patched copy fixes D1-D7 and nothing else. With it (`invariants/invp_*.out`): no violation in any arm, with the
fixes on or off, mirror on or off, seeds 0 and 5, and `unified` with every switch; a few rows with cached > ideal, as
in the records; with the mirror off, a lock outlives its node 5-18 samples a run (one node, harmless).

Anchors with the patched copy (`anchors_table_patched.md`): campaign 34 mean TTFT 31.57 / 11.22 / 10.57 / 10.63 /
10.57 s; `jit_p50` -> `ec_full` -0.70 (measured -0.62); `jit_p50` refusals 21 (46); history computed again closer in
7 rows; 217 rows reproduced against 209. Campaign 32, five jitter replicas: medians 32.08 / 15.15 / 13.42 / 13.70 /
13.51 s against the original's 31.79 / 14.85 / 13.55 / 13.66 / 13.57, onsets 80 / 88 / 87 / 87 against 84 / 88 / 88 /
87: the same within the spread of the replicas (one un-jittered run of `jit_exact` gave 11.87 s, an outlier of the
tipping point). `hicache` without fixes is closer (minutes 90-110: 104.6 s against 95.7; measured 111.8). Triton:
121.2 / 50.1 / 50.2 / 50.0 / 49.8 s. Mirror off: 16.69 s inline and 10.43 s asynchronous for `ec_full`, as the
original.

A policy conclusion that changes with the patch (`policy_patched.out`, seeds 0-2, against `../POLICY_RESULTS.md`):
P2 on the 112 GB pool, -0.40 s "the same sign in every seed" -> -0.01 s, range -0.62 to +0.82. R4 keeps its numbers
(-0.23 and -0.86 s; -0.08 s at 112 GB against -0.17), F3 too (+0.37, +0.51, +1.12 s).

## 5. Sensitivity: each fitted constant x 0.8 and x 1.2 (`sens_report.md`)

16 constants, 32 settings, five arms, seeds 0, 5 and 7 with the fixes on and seed 0 with the fixes off (660 runs).

| conclusion | fixes on: seeds 0 / 5 / 7 | fixes off: seed 0 |
|---|---|---|
| `hicache` behind `ec_full` (paired) | 32 / 32 / 32 of 32 (+0.2 to +30.5 s) | 32 of 32 |
| `jit_p50` behind `ec_full`, point estimate | 32 / 32 / 32 (+0.09 to +2.74 s) | 31 of 32 (-0.1 to +13.4 s) |
| ... with the interval over users clear of 0 | 30 / 3 / 32 | 29 |
| `jit_exact` within 0.2 s of `ec_full` | 32 / 29 / 29 (worst 0.59) | 10 (-7.9 to +4.1 s) |
| `ec_exact` within 0.2 s of `ec_full` | 32 / 32 / 30 (worst 0.27) | 6 (-6.4 to +4.3 s) |
| no hinted arm loops with the fixes on | 31 / 31 / 31 | |
| every hinted arm loops with the fixes off | | 30 of 32 |

- The one setting that breaks "the fixes end the loop" is the read share x 1.2 (0.876: the write thread gets 2.65 K
  tokens/s under read load, below the 4.3 K it must write): a loop with the fixes on, in all three seeds (onset
  minutes 38-83). The one that breaks "the loop appears without the fixes" is the read share x 0.8 (0.584): no
  refused read at all, the run equals the fixed one. So the loop is there or not as (1 - share) x 21.4 K is under
  or over the write demand, and the fitted 0.73 leaves a margin of 9 % to the edge. Campaign 32's own loop phase
  gives a lower share than its early phase (section 4), so the constant is not one number.
- The level is not robust: the decode term per token of context x 0.8 / x 1.2 moves `ec_full` on seed 0 from 10.6
  s to 3.9 / 16.8 s; each of the other GPU terms moves it by 0.4 to 1.6 s; the device and SSD terms by 0.15 s or
  less with the fixes on, the read share x 1.2 apart.
- With the fixes off the order among `ec_full`, `jit_exact` and `ec_exact` is whatever the tipping minute makes it.

GPU speed (all GPU times x 0.99 ... x 1.01, seed 0): hinted arms move smoothly, 0.6 s per 1 % (`ec_full` 9.96 /
10.32 / 10.62 / 10.90 / 11.17). `hicache` does not: 19.4 / 19.7 / 31.6 / 31.6 / 32.8 s. At x 0.99 and x 0.995 its
storm starts at minute 60 instead of 40 (29.1-30.6 s again at x 0.975-0.985, 28.5 at x 0.97). The builder's check at
x 0.98 and x 1.02 missed it. Both measured `hicache` runs stormed at minute 40, and the simulator follows the
measured run within 4 % in each 10-minute bin from minute 40; but one seed's `hicache` level can move by 12 s on half a per cent
of GPU speed.

## 6. The floor under a difference (`noise_seed0.md`)

Six replicas of seed 0 with up to 5 ms of send jitter, fixes on. Two replicas of the same arm, paired request by
request, differ by (mean / largest): `hicache` 0.29 / 0.68 s, `jit_p50` 0.11 / 0.27, `jit_exact` 0.04 / 0.08,
`ec_full` 0.06 / 0.12, `ec_exact` 0.10 / 0.28. Over all 36 replica pairs: `jit_p50` -> `ec_full` -0.66 s [-0.87,
-0.48], the same sign in all; `ec_full` -> `jit_exact` -0.01 [-0.11, +0.09]; `ec_full` -> `ec_exact` -0.08 [-0.30,
+0.09]; `hicache` -> `ec_full` -26.6 [-26.8, -26.3].

Across users (nine seeds): mean TTFT of a hinted arm 5.2 to 16.8 s, `hicache` 8.6 to 45.8 s; `hicache` behind
`ec_full` by 5.0 to 33.9 s; `jit_p50` behind by 0.20 to 0.59 s (interval over users clear of 0 in about half of the seeds).

Against the GPU: the measured runs differ in GPU speed by 0.3-0.4 % on one unit and up to 0.7 % between units,
which is 0.2 to 0.4 s of mean TTFT at saturation. No simulated difference under about 0.5 s can be checked against
one GPU run per arm.

## 7. What it can be trusted for

Trusted, at the anchored setting (FreeInference, FA3 or Triton costs, 128 users, 160 GB, 2 h, fixes on):
- Sign and rough size of effects that change GPU work per request by 5 % or more (history computed again, a lookup
  budget, read-or-recompute rules): differences of 2 s or more with one sign on five or more seeds.
- The sign of 0.2 to 0.7 s effects of read timing that act through pinned memory and the read line (`jit_p50`
  against `ec_full`), when the sign holds on every seed and over jitter replicas; the size to a factor of two.
- Throughput, queue, running requests, cap-binding time, who waits, GPU work per token, TPOT.
- That a loop of refused reads and rewrites exists without the fixes and not with them, as long as the write
  thread's share under read load stays above the write demand.

Not trusted:
- Differences under 0.5 s as statements about the GPU; under 0.3 s on one seed even inside the simulator (0.7 s when
  `hicache` is one side).
- Absolute TTFT away from seed 0 at the anchored setting (GPU terms good to 2-3 % give 10-15 %); the size of
  `hicache`'s gap on one seed.
- Anything that acts through refused reads, pending writes or the write hold with the fixes on (P1, P2, R2, R3,
  which fix does what): the simulator's write backlog is 2 to 4 times the measured one and it refuses reads the
  engine did not.
- The fixes-off regime beyond "a loop exists": onset minute, order of the hinted arms, size of the fixes' gain.
- SSD-tier rules (R1, R4, R5) beyond their sign: whole nodes, an instant cleaner, free touches, and three passes in
  the last 22 minutes of a run.
- Mirror off (no measurement; the original breaks residency, use the patched copy), `nohint` and every `unified`
  switch (no anchor by construction).
- Settings with no GPU anchor: other host sizes or user caps (112 GB, 64 users / 88 GB), the cleaner at 50 / 40 %,
  light load with a short room (campaign 26's regime), other traces (TraceLab is not read). Campaigns 35 to 38 can
  become anchors; until then results there are extrapolation.
