# Campaign 34: campaign 32 on the fixed engine, three arms, to verify the FA3 engine fixes

Ordered 2026-11-22 by the author ("prototype the engine fixes while campaign 33 runs, after campaign 33 finishes,
verify the fixes"). The question: do the four fixes of `OPEN_ITEMS.md` §7 end the loop campaign 32 found (reads
refused host memory, history computed again and written to the SSD again, a write backlog that holds host memory),
and what do they change in the numbers?

## Setting

Campaign 32's (FreeInference on FA3; `settings.env` is campaign 32's plus the switches), so requests pair with
campaign 32's arms by user, stream and index. Same GPU unit (`GPU-0f61782e`).

| | campaign 32 | campaign 34 |
|---|---|---|
| engine | `frozen/engine_queuedfix_20261122` | `frozen/engine_fa3fix_20261122`: the same plus the fixes and the census (`diff -rq`: only the files of the fixes, `environ.py` and the new `host_census.py`) |
| switches (`settings.env`) | none | the five below |
| server bind | `0.0.0.0` | `127.0.0.1` (queue lines, set 13 s before the first launch; `results/campaign33/DECISIONS.md`, "Outside requests") |

| Switch | Fix |
|---|---|
| `SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES=1` | 1. a page whose files the SSD store holds is touched, not written again |
| `SGLANG_HICACHE_STORAGE_WRITE_COALESCE_PAGES=128` | 2. queued writes merge into transfers of up to 128 pages |
| `SGLANG_HICACHE_STORAGE_READ_HOLD_WRITE_TOKENS=50000` | 3. timed reads wait while pending writes hold 50,000 host tokens |
| `SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY=1` | 4. a read refused host room is retried without spending the turn's lookups |
| `SGLANG_LOG_HICACHE_HOST_CENSUS_INTERVAL=10` | a `host_census` line every 10 s: what holds host memory, by reason (logging only) |

Each switch is off by default; every arm ran with all five on (the launcher's `LAUNCH` lines; the first arm's server
environment was checked in `/proc`).

## Arms (`queue.txt`, in this order)

| Label | Server arm | Hints the client sends | Campaign 32's loop onset |
|---|---|---|---|
| `jit_p50` | `three_tier_wc_oracle_v2_nu_owned`: just-in-time reads, arrival gate and hold | the gap table's p50 | minute 42 (67) |
| `ec_full` | the same with the four rules | the gap table (p10, p50, p90) and the end of a session | minute 84 (84) |
| `jit_exact` | as `jit_p50` | the exact gap | minute 80 (80) |

Onset: the first of 3 (in brackets 5) minutes in a row with 5 or more reads refused host room
(`checks/fix_minutes.py`, which reproduces campaign 32's minute 67 for `jit_p50`).

## Run

`chain_fix.sh` (`chain.log`) started when campaign 33 ended (07:23:42Z, 2026-11-23). On the frozen copy every unit
test passed (`unit_tests.log`: read scheduler 9, read dispatch 28, next-turn prefetch 77, cleaner 14, oracle flags 10,
prefetch lifecycle 8, `next_use` 20, census 7, NIXL storage 21, radix cache L3 43 on the GPU). The tier check (the
`hicache` arm on small pools with the fixes on) passed 5 of 5: prompts served from host and from SSD gave the warm
run's first-token logprobs, with 2 merged write groups in it
(`results/20261123_072712_three_tier_wc_custom_L1100000_H30/check_fixes.log`). Then the three arms:

| arm | client start | done | requests | errors | SSD cleaner passes (run time) | SSD at the end |
|---|---|---|---|---|---|---|
| `jit_p50` | 07:34:56Z | 09:35:50Z | 17,049 | 0 | 5,301, 6,032 and 6,644 s | 90 % |
| `ec_full` | 09:40:31Z | 11:41:21Z | 17,095 | 0 | 5,275, 5,977, 6,589 and 7,200 s | 81 % |
| `jit_exact` | 11:46:07Z | 13:46:55Z | 17,114 | 0 | 5,272, 5,973, 6,614 and 7,226 s | 81 % |

- Every per-arm check passed (`checks/arm_checks_<label>.log`). `jit_p50`'s server log holds the known traceback at
  the server's stop (`prefetch_io_aux_func`, `OPEN_ITEMS.md` §6), after the client's summary, as campaign 32's
  `jit_p50` and `jit_exact` did.
- Each server listened on 127.0.0.1 only (`server_args.txt`), and every request line in the three server logs comes
  from 127.0.0.1: no outside request reached this campaign.
- The first cleaner pass comes at 5,272 to 5,301 s in every arm, earlier than in campaign 32 (5,584 to 6,798 s): the
  fixed arms serve more requests, so more new pages. Before the 6,600 s cut there are two, three and two passes
  (campaign 32: none, one, one).
- The slowest recorded request took 702 to 727 s (campaign 32: 552 to 611 s), so a few requests sent shortly before
  6,600 s may be missing. Counted to 5,200 s, before any cleaner pass in all six runs, the slowest took 548 to 552 s
  and none should be (Results).

## Results

### The loop does not start

Per window of minutes, from the server logs (`checks/fix_minutes.out`; minute by minute in
`checks/fix_minutes_table_<arm>.out`, and `_c32_<arm>.out` for campaign 32):

| campaign 32 → 34, minutes 30-60 / 60-90 / 90-110 | `jit_p50` | `ec_full` | `jit_exact` |
|---|---|---|---|
| loop onset | minute 42 → none | 84 → none | 80 → none |
| reads refused host room | 68 / 1,182 / 1,164 → 0 / 0 / 0 | 0 / 282 / 1,121 → 0 / 0 / 0 | 0 / 529 / 1,182 → 0 / 0 / 0 |
| history computed again, M tokens | 1.04 / 16.24 / 16.43 → 0.25 / 0.59 / 0.93 | 0.11 / 4.92 / 15.71 → 0.11 / 0.41 / 0.89 | 0.11 / 7.61 / 15.67 → 0.11 / 0.60 / 0.96 |
| pending SSD writes at their peak, K tokens | 166 / 625 / 675 → 60 / 65 / 79 | 62 / 551 / 610 → 59 / 72 / 65 | 59 / 523 / 515 → 70 / 58 / 87 |
| write submit to done, median of the minutes | 1.5 / 28.4 / 41.6 s → 0.3 / 0.6 / 0.6 s | 0.1 / 4.0 / 35.3 → 0.1 / 0.4 / 0.5 | 0.1 / 2.5 / 34.0 → 0.1 / 0.3 / 0.6 |
| tokens submitted for SSD writes, M | 9.1 / 16.7 / 12.9 → 8.4 / 7.8 / 5.3 | 8.3 / 10.2 / 12.8 → 8.3 / 7.7 / 5.2 | 8.3 / 11.6 / 12.7 → 8.3 / 7.7 / 5.4 |
| tokens read from the SSD, M | 16.7 / 15.9 / 8.2 → 14.9 / 25.5 / 17.6 | 7.8 / 18.0 / 8.6 → 7.4 / 18.8 / 17.4 | 3.0 / 14.0 / 8.7 → 3.0 / 16.3 / 17.5 |

- In the whole run no read was refused host room while the client sent requests; the only refusals are 3 in
  `jit_exact` at 7,221 s, after the 7,200 s deadline. In minutes 60 to 110 the history computed again falls 15 to
  21 times.
- What each fix did, in the same windows:
  - Fix 1 skipped 0 to 2 % of the pages to write in minutes 30-60 and 11 to 12 % in 90-110. In campaign 32's loop 74
    to 93 % of the writes were pages the store held; without the loop there are few such pages.
  - Fix 2 merged writes into 1,011 to 1,651 groups a window, and writes are done in 0.1 to 0.6 s.
  - Fix 3's hold switched on 4 to 11 times a window.
  - Fix 4 never acted (0 gate `retry`): before the deadline no read was refused or cut for room (`ntp_room`; 6 in
    `jit_exact` at 7,218 to 7,221 s), so there was nothing to retry.
  - All four ran together, so the run does not say which one kept the first refusals away. In campaign 32 they
    came while pending writes held up to 166 K tokens (`jit_p50`, minutes 30-60); here pending writes peak at 60 K in
    those minutes.
- The census, between 3,600 and 6,600 s (299 lines an arm): the 1.63 M-token host pool holds the GPU mirror (median
  0.65 M tokens; host eviction does not take it) and host-only pages (0.90 to 0.92 M), most of which eviction can take
  once their subtree goes (`evict_casc` median 0.60 to 0.84 M). Free room is small (median 16 to 18 K), but locks are
  too: pending writes hold 0 to 7 K at the median (p90 65 to 94 K), pins 0 to 25 K, reads in flight 0. `jit_p50`'s
  early reads hold 136 K (p90 280 K); the other arms' hold none at the median.
- History computed again rises in every arm from minute 87 or 88, the minute of the first cleaner pass: from 3 to
  4 K tokens a minute before to up to 290 K in a minute after, until the end. The cleaner deletes the oldest pages, so a returning
  user whose history it deleted computes it again (inferred from the timing; not traced request by request).

### Against campaign 32

Each arm against its campaign 32 run, requests sent before 6,600 s (`checks/vs_c32/compare_arms.out`):

| campaign 32 → 34 | `jit_p50` | `ec_full` | `jit_exact` |
|---|---|---|---|
| users with a request | 522 → 617 | 557 → 620 | 550 → 622 |
| requests (a minute) | 12,591 (114.5) → 15,608 (141.9) | 13,752 (125.0) → 15,562 (141.5) | 13,547 (123.2) → 15,601 (141.8) |
| output tokens a second | 540 → 666 | 590 → 664 | 581 → 665 |
| prompt tokens from GPU, host, SSD; recomputed | 23, 45, 15 %; 17 % → 19, 60, 15 %; 6 % | 22, 53, 13 %; 12 % → 19, 61, 14 %; 6 % | 24, 53, 10 %; 13 % → 20, 62, 11 %; 6 % |
| history delivered, of the ideal | 0.886 → 0.995 | 0.937 → 0.996 | 0.927 → 0.995 |
| TTFT mean, p50, p90 (each run's own requests) | 17.97, 6.47, 63.89 → 11.47, 6.38, 22.41 s | 14.64, 6.51, 51.24 → 10.96, 8.34, 24.01 s | 14.86, 6.12, 50.52 → 10.92, 8.24, 22.26 s |
| paired, 34 - 32 on the requests both sent (interval over users, 95 %) | -9.52 s [-11.05, -8.17], 12,591 requests | -5.79 s [-6.68, -4.91], 13,752 | -6.18 s [-7.25, -5.23], 13,547 |
| the same, counted to 5,200 s (`compare_arms_cut5200.out`) | -4.79 s [-5.64, -4.02] | -0.81 s [-0.94, -0.70] | -1.23 s [-1.46, -1.02] |
| wait for a slot: mean, max | 91.7, 1,348 → 70.3, 579 s | 59.4, 1,097 → 63.0, 538 s | 58.4, 1,178 → 65.1, 550 s |

- The fixed arms serve 13 to 24 % more requests before the cut and let 63 to 95 more users in, and the three are
  within 0.3 % of each other (141.5 to 141.9 requests a minute).
- The gain sits where campaign 32's loop ran. By send time, minutes 60 to 110: mean TTFT 49.94, 34.34 and 36.20 s →
  21.77, 20.82 and 20.71 s, on 77, 36 and 43 % more requests (3,830, 4,945 and 4,724 → 6,789, 6,739 and 6,772),
  and the recomputed share of prompt tokens 38.0, 21.1 and 24.0 % → 6.2, 6.2 and 6.3 %. Minutes 0 to 30 are the same
  in both campaigns (0.14 to 0.55 s); in minutes 30 to 60 the fixed runs are 0.05 to 0.90 s faster.
- Counted to 5,200 s the gain is smaller for `ec_full` and `jit_exact`, whose campaign 32 runs had been in the loop
  for only 3 and 7 minutes by then; `jit_p50`'s had been in it since minute 42.
- The fixed runs' own p50 is higher in `ec_full` and `jit_exact` (6.51 → 8.34, 6.12 → 8.24 s) because they serve many
  more requests late in the run, when the box is loaded. On the 12,589 requests all six runs sent: means 17.96, 9.36
  and 10.21 s → 8.44, 7.42 and 7.41 s; p50 6.47, 5.51 and 5.29 → 4.70, 5.25 and 5.39 s.
- The box is still saturated late. 80 to 88 requests are in flight at 6,000 to 6,600 s (campaign 32: 95 to 105)
  against at most 64 running, and over all returning requests sent before minute 110 the server's queue is 5.50 to
  7.13 s of each arm's mean (`figs.log`, `results/compare_c34_designs/`).

### The arms against each other

Paired, requests both sent before the cut (`checks/vs_c32/compare_arms.out` and `compare_arms_cut5200.out`, one
bootstrap per file; campaign 34's own `compare_arms.out` gives the same differences):

| A → B | campaign 32 | campaign 34 | campaign 32 to 5,200 s | campaign 34 to 5,200 s |
|---|---|---|---|---|
| `jit_p50` → `ec_full` | -8.60 s [-9.93, -7.30] | -0.62 s [-1.02, -0.22] | -4.93 s [-5.75, -4.18] | -0.68 s [-0.98, -0.38] |
| `jit_p50` → `jit_exact` | -7.75 s [-9.04, -6.61] | -0.67 s [-1.07, -0.28] | -5.17 s [-5.97, -4.40] | -0.68 s [-0.98, -0.39] |
| `ec_full` → `jit_exact` | +1.26 s [+0.92, +1.63] | -0.06 s [-0.19, +0.06] | +0.59 s [+0.41, +0.77] | +0.00 s [-0.04, +0.05] |

- With the loop gone the arms are within 0.7 s. The four rules on p50 hints tie with exact hints, and the
  just-in-time engine on p50 hints is 0.6 to 0.7 s behind them. Campaign 32's 8.6 and 7.75 s were the loop's timing,
  and its 1.26 s between `ec_full` and `jit_exact`, unresolved there, is gone too.
- Where `jit_p50` loses: more of its returning requests read from the SSD in minutes 30 to 60 (13.6 % against 8.2
  and 3.0 %), and its early reads hold host memory (census, above). By harness it is ahead on `eval:swe-bench` (8.32
  against 9.72 and 9.51 s on the requests every arm sent) and behind on all the others together (14.05 against 11.64
  and 11.71 s; `compare_arms.out`). Not looked into further.

## Reading

- The four fixes keep the loop from starting in all three arms: no read is refused host memory, pending writes stay
  under 90 K tokens and the median write is done in under a second, and the history computed again in minutes 60 to
  110 falls 15 to 21 times. On the same users the fixed engine is 5.8 to 9.5 s faster and serves 13 to 24 % more requests.
- Campaign 32's spread between the policies was mostly when each arm entered the loop. On the fixed engine at this
  setting the four rules on p50 hints equal exact hints, and just-in-time reads on p50 hints cost 0.6 to 0.7 s.
- What remains late in the run is the server's queue (the box is past saturation at 128 users) and, from minute 88,
  history the SSD cleaner deleted.
- One run per arm, all four fixes at once: the run shows the four together work, not which one is needed. Fix 4 had
  nothing to do.

## Open after campaign 34

- The switches are still off by default in `python/`. Whether to turn them on, and which, is the author's call. Runs
  with fix 2 alone and fix 3 alone would say which keeps the first refusals away. Fix 4 has only its unit tests.
- The SSD cleaner deletes the oldest pages even when live sessions still need them; from the first pass on they are
  computed again. A larger volume, or a cleaner that spares pages of open sessions, are options (not evaluated).
- Defect 1 is still in the engine (`OPEN_ITEMS.md` §1): a window held 0 to 17 lookups refused at the cap, and 26 to
  216 reads whose pages were evicted again before admission with no re-read left (`ntp_reevict capped=1`).
- TraceLab (campaign 33) did not saturate on FA3 and showed no loop, so it is not rerun on the fixed engine.
- `scripts/start_server.sh` and the baselines' launch scripts still bind `0.0.0.0` (`OPEN_ITEMS.md` §0, item 8).
