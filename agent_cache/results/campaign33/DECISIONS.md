# Campaign 33: campaign 29 on FlashAttention 3, its six arms for the full 2 h

Ordered 2026-11-22 with campaign 32 by the author ("yes, queue the FA3 reruns after tonight's runs") after the
20-minute FA3 check (`baselines/README.md`, "FA3"). The question: do the policies matter on FA3 on TraceLab, which on
Triton saturated the box from minute 6 (campaign 29)? Campaign 32 is the same on FreeInference and ran first.

## Setting

Campaign 29's (`settings.env` is a copy; `results/campaign29/DECISIONS.md`, "Setting") with campaign 32's two changes:

| | campaign 29 | campaign 33 |
|---|---|---|
| attention | Triton (`scripts/start_server.sh` pins it) | FA3 (`--attention-backend fa3`; `--max-total-tokens 668096` keeps the GPU pool) |
| engine | `frozen/engine_ec_20261111d` | `frozen/engine_queuedfix_20261122`: copy e plus the `next_use` fix of defect 2 (commit `94c3dd0c85`); defect 1 is still in it (`OPEN_ITEMS.md` §1) |

Every arm's server log shows FA3, bf16 KV, the GPU pool of 668,096 tokens, the 160 GB host pool, at most 64 requests
running, and the intended eviction: `lru` (`hicache`), `return_chance` (`ec_full`, `ec_exact`), `next_use` (the
others). The users are campaign 29's: play numbers name the same session in all twelve arms
(`checks/vs_c29/compare_arms.out`).

## Arms (`queue.txt`, in this order)

| Label | Server arm | Hints the client sends |
|---|---|---|
| `hicache` | `three_tier_wc`: original HiCache | none |
| `ec_full` | `three_tier_wc_oracle_v2_nu_owned` with the four rules | the gap table (p10, p50, p90) and the end of a session |
| `ec_exact` | `ec_full`'s server, the oracle arm | the exact gap |
| `jit_p50` | `three_tier_wc_oracle_v2_nu_owned`: just-in-time reads, arrival gate and hold | the gap table's p50 |
| `jit_exact` | the same | the exact gap |
| `oracle_v2` | `three_tier_wc_oracle_v2_nu`: the old oracle | the exact gap |

## Run

Launched 18:46:09Z on 2026-11-22, the second campaign 32's last arm ended, by the same launcher process (GPU unit
`GPU-0f61782e`). About 2 h 6 min an arm; the last arm was done at 07:23:42Z on 2026-11-23.

| arm | client start | done | requests | errors | SSD cleaner passes (run time) | SSD at the end |
|---|---|---|---|---|---|---|
| `hicache` | 18:51:14Z | 20:52:04Z | 7,314 | 0 | five, 4,653 to 6,948 s | 86 % |
| `ec_full` | 20:57:42Z | 22:58:33Z | 7,314 | 0 | five, 4,654 to 6,888 s | 87 % |
| `ec_exact` | 23:04:27Z | 01:05:16Z | 7,314 | 0 | five, 4,644 to 6,909 s | 87 % |
| `jit_p50` | 01:10:51Z | 03:11:39Z | 7,314 | 0 | five, 4,647 to 6,881 s | 87 % |
| `jit_exact` | 03:17:15Z | 05:18:02Z | 7,313 | 0 | five, 4,651 to 6,995 s | 85 % |
| `oracle_v2` | 05:22:56Z | 07:23:42Z | 7,314 | 0 | five, 4,656 to 6,904 s | 87 % |

- Every per-arm check passed (`checks/arm_checks_<label>.log`); no server log holds a traceback.
- Each cleaner pass deleted 231 to 250 GiB of the oldest pages (90 % of the volume down to 80 %), at nearly the same
  run times in every arm: the arms write alike.
- The slowest recorded request took 339 to 345 s (long outputs), so none sent before 6,600 s is missing.
- An outside address stopped `oracle_v2`'s server from admitting requests for 44 s at 409 s ("Outside requests").

## Results

Requests sent before 6,600 s (`compare_arms.out`):

| | `hicache` | `ec_full` | `ec_exact` | `jit_p50` | `jit_exact` | `oracle_v2` |
|---|---|---|---|---|---|---|
| users with a request; sessions ended | 674; 635 | 674; 635 | 674; 635 | 674; 635 | 674; 635 | 674; 635 |
| requests (a minute) | 6,781 (61.6) | 6,778 (61.6) | 6,776 (61.6) | 6,782 (61.7) | 6,775 (61.6) | 6,776 (61.6) |
| TTFT mean, p50, p90 | 0.66, 0.19, 1.66 s | 0.63, 0.18, 1.58 s | 0.64, 0.18, 1.57 s | 0.64, 0.18, 1.60 s | 0.63, 0.18, 1.54 s | 0.77, 0.19, 1.69 s |
| prompt tokens from GPU, host, SSD; recomputed | 71, 15, 1 %; 13 % | 72, 15, 0 %; 13 % | 73, 14, 0 %; 13 % | 71, 14, 1 %; 13 % | 73, 13, 1 %; 13 % | 72, 14, 1 %; 13 % |
| wait for a slot | 0 | 0 | 0 | 0 | 0 | 0 |
| TTFT mean on the 6,775 requests every arm sent | 0.66 s | 0.63 s | 0.64 s | 0.64 s | 0.63 s | 0.77 s |

| sent in minutes | `hicache` | `ec_full` | `ec_exact` | `jit_p50` | `jit_exact` | `oracle_v2` |
|---|---|---|---|---|---|---|
| 0-10 | 0.37 s | 0.36 s | 0.37 s | 0.36 s | 0.36 s | 1.83 s |
| 10-30 | 0.38 s | 0.34 s | 0.35 s | 0.36 s | 0.35 s | 0.39 s |
| 30-60 | 1.38 s | 1.34 s | 1.37 s | 1.34 s | 1.34 s | 1.41 s |
| 60-110 | 0.30 s | 0.28 s | 0.28 s | 0.29 s | 0.29 s | 0.29 s |

Pairs, B - A on the requests both sent (95 % interval over users):

| A | B | requests | B - A |
|---|---|---|---|
| `hicache` | `ec_full` | 6,778 | -0.03 s [-0.05, -0.01] |
| `ec_full` | `jit_p50` | 6,778 | +0.01 s [-0.01, +0.03] |
| `jit_p50` | `jit_exact` | 6,775 | -0.00 s [-0.02, +0.01] |
| `ec_full` | `ec_exact` | 6,776 | +0.01 s [-0.00, +0.02] |
| `ec_full` | `jit_exact` | 6,775 | +0.01 s [-0.01, +0.02] |
| `jit_exact` | `oracle_v2` | 6,775 | +0.13 s [+0.09, +0.18]; +0.03 s [+0.01, +0.05] without the requests sent at 400 to 460 s (`checks/stall_window.out`) |

- The box keeps up for the whole run. At most 46 users are in session (the cap of 128 never binds) and at most 18
  requests run; nobody waits for a slot, all 729 users who arrive in 2 h are admitted, and every arm sends all 7,314
  requests (`jit_exact` 7,313). A short queue (6 or 7 at most) forms in the same 10 minutes in every arm (30 to 35,
  46, 47, 49 and 54; `hicache` also 41), when the GPU pool reaches 0.96 to 0.98; returning requests then take 1.1 to
  5.6 s, which is the 30-60 row (`checks/by_minute_<label>.out`).
- The five arms other than the old oracle are within 0.03 s of each other, and every interval between them lies
  within 0.05 s of zero.
- Most of the old oracle's 0.13 s is one stall caused from outside ("Outside requests"): the 35 requests it sent at
  400 to 460 s took 703 s more than on `jit_exact`. Without the requests sent in that minute it trails `jit_exact`
  by 0.030 s [+0.013, +0.047] and `ec_full` by 0.035 s, and ties `hicache`: +0.005 s [-0.015, +0.027]
  (`checks/stall_window.py`, `.out`; pairs by stream and turn, interval over streams).
- History is almost never computed again: 0.07 to 0.08 % of it in the four gate arms, with no request refused a read
  at the lookup cap (`checks/cap_share_<label>.out`; defect 1 does not show), and 0.1 % in `hicache`, where the 267
  requests sent before 6,600 s that reached the 8-lookup cap at arrival still got their history
  (`checks/storm_cause_hicache_all.out`). The 13 % computed is new text and first requests.
- The SSD is written, seldom read: 0 to 2 % of returning requests read from it, the read thread is 1 to 3 % busy
  (`checks/read_thread.out`), and reads stay that rare after the cleaner's first pass at minute 77 (0.2 to 2.3 % of
  returning requests a minute in minutes 78 to 110).
- The rules decided almost nothing (`checks/rules_activity_ec_full.out`): the price stayed at 5e-05 through 1,437
  updates, no release or bump; 66 "wait" and 76 "arrival" decisions, nearly all in minutes 30 to 40. Reads started
  before arrival: none in `ec_full`, 22 to 86 in the other hinted arms (`checks/read_starts.out`).
- Where the TTFT goes, over the 6,101 returning requests all six arms sent before minute 110
  (`results/compare_c33_designs/ec_wait_split.csv`): 0.60 to 0.63 s in five arms, of which 0.23 to 0.24 s after
  admission, 0.36 to 0.38 s in the queue and at most 0.02 s waiting for reads; the old oracle 0.73 s, 0.32 after
  admission and 0.41 in the queue.
- Figures: `results/compare_c33_designs/` (the launcher's `figs.sh`, redrawn after each arm).

## Against campaign 29 (Triton)

Each arm against its Triton run, the same users (`checks/vs_c29/compare_arms.out`):

| arm | Triton: TTFT mean (requests before 6,600 s) | FA3 | on the requests both sent |
|---|---|---|---|
| `hicache` | 289.2 s (2,081) | 0.66 s (6,781) | -288.4 s [-311.9, -265.7] on 2,081 |
| `ec_full` | 150.7 s (3,723) | 0.63 s (6,778) | -149.9 s [-156.5, -143.2] on 3,723 |
| `ec_exact` | 148.9 s (3,763) | 0.64 s (6,776) | -148.0 s [-154.6, -140.9] on 3,763 |
| `jit_p50` | 164.2 s (3,451) | 0.64 s (6,782) | -163.3 s [-171.4, -155.0] on 3,451 |
| `jit_exact` | 167.4 s (3,390) | 0.63 s (6,775) | -166.5 s [-174.0, -158.0] on 3,390 |
| `oracle_v2` | 227.4 s (2,556) | 0.77 s (6,776) | -226.3 s [-242.1, -209.5] on 2,556 |

- On Triton the GPU pool filled at minute 6 and the 128-user cap bound from minute 30 or 31, with 18 or 19 requests
  running and 102 to 105 queued to the end; the arms served 18.9 to 34.2 requests a minute and admitted 292 to 458
  users before the cut. FA3 serves every request of every user who arrives: 61.6 a minute, 445 output tokens a
  second against 132 to 251, and 674 users.
- Campaign 29's order (`ec_full` 15.6 s ahead of `jit_p50`, the old oracle 78 to 97 s behind the gate arms,
  `hicache` 98 to 175 s behind every hinted arm) came from the queue Triton's prefill built and the lookup cap's draw
  within it. On FA3 there is no lasting queue and no request refused at the cap.

## Outside requests

- `scripts/start_server.sh` starts the server on `0.0.0.0:30000` with no API key, and the container uses the host's
  network, so port 30000 is reachable from the internet. 104 requests from outside addresses reached the servers of
  26 of the 209 runs under `results/` since 2026-09-18 (`checks/outside_requests.txt`). Most are scans (`GET /`,
  login pages, `/server_info`, `/get_model_info`, `/metrics`); two `POST /v1/chat/completions` were answered (run
  `20261112_025130_three_tier_wc_oracle_v2_nu_owned_NAT160`); one address sent `POST /update_weights_from_disk` to
  campaign 32's `ec_exact` and this campaign's `oracle_v2`.
- A weight update waits for a lock that every running request holds until its last token, and no new request takes
  the lock while an update waits (`TokenizerManager.update_weights_from_disk`, `model_update_lock`,
  `utils/aio_rwlock.py`). So each call drained the server; the updates then failed (400) and serving resumed.
- `oracle_v2`: 05:29:45 to 05:30:29Z, 409 to 454 s into its run, running requests went from 11 to 0 and nothing was
  admitted for 44 s.
- Campaign 32's `ec_exact`: three calls at 11:15:35 to 11:16:58Z (2,776 to 2,859 s into its run, with 13 to 23
  requests queued) stopped admission for about 26, 20 and 20 s. Its requests sent in minute 46 took 14.6 s against
  7.7 and 8.6 s in `jit_exact` and `ec_full`, and from minute 50 it is with the others again: 960 requests sent at
  2,716 to 3,099 s took 1,007 s more than in `ec_full`, 0.08 s of its mean. Without them `ec_full` to `ec_exact`
  stays +1.47 s [+1.08, +1.91], and `jit_exact`'s 0.16 s lead over `ec_exact` falls to 0.07 s [-0.10, +0.24]
  (`results/campaign32/checks/stall_window.out`). Campaign 32's reading stands; its record has a note.
- The other arms of campaigns 28 to 33 saw only `GET` requests.
- Campaign 34 binds `127.0.0.1` (`--host 127.0.0.1` on its queue lines, set before its first arm; its first server
  listens on 127.0.0.1:30000 only). `start_server.sh` and the baselines' `start_server.sh` files (vLLM,
  `--host 0.0.0.0`) are unchanged.

## Reading

- On FA3, TraceLab at 0.1 users a second with a 160 GB host pool runs under the box's capacity: the GPU and host
  pools hold the live histories, the SSD is written and seldom read, and the six arms tie within 0.03 s apart from one
  stall caused from outside. Nothing here tests the policies.
- Campaign 29's spread (148 to 288 s) and its order came from Triton's prefill. With campaign 32, where FA3 saturates
  FreeInference from minute 28 and the arms separate by the refused-read loop, FreeInference is the trace that loads
  FA3 in this setting; TraceLab would need more users a second or a smaller host pool.
- One run per arm.

## Open after campaign 33

- Bind the server to localhost or require a key, in `scripts/start_server.sh` and the baselines' `start_server.sh`
  files (pinned while campaign 34 runs; the author's call).
- If TraceLab should test the policies on FA3: a higher arrival rate or a smaller host pool. Not run.
- `checks/gpu_work.out` converts tokens to GPU seconds at Triton's cost and overstates FA3's GPU time; campaign 32's
  record has FA3's costs.
