# Campaign 32: campaign 28 on FlashAttention 3, its six arms for the full 2 h

Ordered 2026-11-22 by the author ("yes, queue the FA3 reruns after tonight's runs") after the 20-minute FA3 check, in
which every arm kept up and the policies tied (`baselines/README.md`, "FA3"). The question: do the policies matter on
the attention backend the box deploys, once a 2 h run saturates? Campaign 33 is the same on TraceLab (campaign 29's
setting) and runs after this one.

## Setting

Campaign 28's (`settings.env` is a copy; `results/campaign28/DECISIONS.md`, "Setting") with two changes:

| | campaign 28 | campaign 32 |
|---|---|---|
| attention | Triton (`scripts/start_server.sh` pins it) | FA3 (`--attention-backend fa3`; `--max-total-tokens 668096` keeps the GPU pool) |
| engine | `frozen/engine_ec_20261111d` | `frozen/engine_queuedfix_20261122`: copy e plus the `next_use` fix of defect 2 (commit `94c3dd0c85`); defect 1 is still in it (`OPEN_ITEMS.md` §1) |

Every arm's server log shows FA3, bf16 KV, the GPU pool of 668,096 tokens, the 160 GB host pool, at most 64 requests
running, and the intended eviction: `lru` (`hicache`), `return_chance` (`ec_full`, `ec_exact`), `next_use` (the
others). The users are campaign 28's: play numbers name the same session in all twelve arms
(`checks/vs_c28/compare_arms.out`).

## Arms (`queue.txt`, in this order)

| Label | Server arm | Hints the client sends |
|---|---|---|
| `hicache` | `three_tier_wc`: original HiCache | none |
| `ec_full` | `three_tier_wc_oracle_v2_nu_owned` with the four rules | the gap table (p10, p50, p90) and the end of a session |
| `ec_exact` | `ec_full`'s server, the oracle arm | the exact gap |
| `jit_p50` | `three_tier_wc_oracle_v2_nu_owned`: just-in-time reads, arrival gate and hold | the gap table's p50 |
| `jit_exact` | the same | the exact gap |
| `oracle_v2` | `three_tier_wc_oracle_v2_nu`: the old oracle | the exact gap |

The stock baseline and the two rules arms ran first, the three `next_use` arms after the fix check
(`results/qfix_check/`), which passed.

## Run

Launched 06:15:07Z on 2026-11-22 (GPU unit `GPU-0f61782e`) by `chain_fa3.sh` once tonight's chain had ended; one
launcher process runs campaigns 32 and 33 (`run_campaigns.sh`). About 2 h 5 min an arm; the last arm was done at
18:46:09Z and campaign 33 launched the same second.

| arm | client start | done | requests | errors | SSD cleaner passes (run time) | SSD at the end |
|---|---|---|---|---|---|---|
| `hicache` | 06:19:21Z | 08:20:11Z | 10,077 | 0 | none | 75 % |
| `ec_full` | 08:24:05Z | 10:24:52Z | 14,285 | 0 | 5,584 s and 7,047 s | 82 % |
| `ec_exact` | 10:29:19Z | 12:30:09Z | 14,090 | 0 | 5,814 s | 90 % |
| `jit_p50` | 12:34:30Z | 14:35:20Z | 13,094 | 0 | 6,798 s | 84 % |
| `jit_exact` | 14:40:02Z | 16:40:53Z | 14,101 | 0 | 5,779 s | 90 % |
| `oracle_v2` | 16:45:19Z | 18:46:09Z | 13,225 | 0 | 6,828 s | 83 % |

- Every per-arm check passed (`checks/arm_checks_<label>.log`). The server logs of `jit_p50` and `jit_exact` hold
  the known traceback at the server's stop (`prefetch_io_aux_func`), after the client's summary.
- The SSD cleaner (at 90 % of the volume it deletes the oldest pages, about 235 GiB, down to 80 %) ran before the
  6,600 s cut in `ec_full`, `ec_exact` and `jit_exact` (minutes 93 to 97), after it in `jit_p50` and `oracle_v2`,
  never in `hicache`, which wrote the most but mostly over pages the store already held ("`hicache` on FA3" below).
  Counted to 5,500 s, before any pass, the arms keep their order (Results).
- The slowest recorded request took 696 s in `hicache` and 552 to 611 s in the others, so a few requests sent shortly
  before 6,600 s may be missing; at the 5,500 s cut none should be.

## Results

Requests sent before 6,600 s (`compare_arms.out`):

| | `hicache` | `ec_full` | `ec_exact` | `jit_p50` | `jit_exact` | `oracle_v2` |
|---|---|---|---|---|---|---|
| users with a request; sessions ended | 433; 306 | 557; 429 | 547; 419 | 522; 394 | 550; 422 | 525; 397 |
| requests (a minute) | 9,653 (87.8) | 13,752 (125.0) | 13,532 (123.0) | 12,591 (114.5) | 13,547 (123.2) | 12,742 (115.8) |
| TTFT mean, p50, p90 | 33.5, 5.0, 99.8 s | 14.6, 6.5, 51.2 s | 15.0, 6.2, 50.1 s | 18.0, 6.5, 63.9 s | 14.9, 6.1, 50.5 s | 17.2, 5.7, 61.7 s |
| prompt tokens from GPU, host, SSD; recomputed | 31, 29, 5 %; 36 % | 22, 53, 13 %; 12 % | 24, 53, 11 %; 13 % | 23, 45, 15 %; 17 % | 24, 53, 10 %; 13 % | 25, 42, 16 %; 16 % |
| wait for a slot, mean | 237 s | 59 s | 53 s | 92 s | 58 s | 74 s |
| TTFT mean on the 9,653 requests every arm sent | 33.5 s | 5.26 s | 5.23 s | 7.42 s | 5.10 s | 6.04 s |

| sent in minutes | `hicache` | `ec_full` | `ec_exact` | `jit_p50` | `jit_exact` | `oracle_v2` |
|---|---|---|---|---|---|---|
| 0-10 | 0.14 s | 0.15 s | 0.15 s | 0.15 s | 0.15 s | 0.15 s |
| 10-30 | 0.51 s | 0.55 s | 0.48 s | 0.50 s | 0.48 s | 0.49 s |
| 30-60 | 29.6 s | 6.4 s | 6.4 s | 7.3 s | 6.2 s | 6.6 s |
| 60-110 | 93.8 s | 34.3 s | 36.3 s | 49.9 s | 36.2 s | 47.4 s |

Pairs, B - A on the requests both sent (95 % interval over users); the last column counts to 5,500 s
(`checks/compare_arms_cut5500.out`):

| A | B | requests | B - A | to 5,500 s |
|---|---|---|---|---|
| `hicache` | `ec_full` | 9,653 | -28.3 s [-32.3, -24.4] | -21.6 s [-25.0, -18.6] |
| `ec_full` | `jit_p50` | 12,589 | +8.60 s [+7.25, +10.01] | +5.95 s [+5.02, +7.01] |
| `jit_p50` | `jit_exact` | 12,590 | -7.75 s [-9.01, -6.57] | -6.05 s [-7.02, -5.15] |
| `jit_exact` | `oracle_v2` | 12,740 | +6.31 s [+5.24, +7.43] | +4.79 s [+4.02, +5.65] |
| `jit_p50` | `oracle_v2` | 12,584 | -1.71 s [-2.09, -1.36] | -1.86 s [-2.24, -1.50] |
| `ec_full` | `jit_exact` | 13,543 | +1.26 s [+0.93, +1.62] | +0.98 s [+0.71, +1.29] |
| `ec_full` | `ec_exact` | 13,530 | +1.43 s [+1.06, +1.82] | +1.24 s [+0.94, +1.56] |
| `ec_exact` | `jit_exact` | 13,505 | -0.16 s [-0.33, -0.02] | -0.27 s [-0.41, -0.15] |

- Three groups: `ec_full`, `ec_exact` and `jit_exact` within 1.4 s; `oracle_v2` and `jit_p50` 6 to 9 s behind them;
  `hicache` 28 s behind everything.
- Course of the run (`checks/by_minute_<label>.out`): every arm keeps up for 25 minutes (0.1 to 0.5 s). The GPU pool
  is full from minute 28 and a queue of 25 forms at minute 40 in every arm. `hicache` storms from minute 40 and
  reaches the cap of 128 users at minute 52; the hinted arms reach it at minutes 71 to 74. From minute 90 every arm
  holds about 20 requests running and 76 to 85 queued, and sends 47 (`hicache`) or 59 to 63 a minute.
- The first 20 minutes repeat the 20-minute FA3 check (`results/fa3_smoke/`, engine copy d, five of these arms)
  within 0.01 s in each of the five (`checks/vs_fa3_smoke/compare_arms.out`; 0.20 to 0.23 s). No repeat of the
  saturated part exists, so the run-to-run spread there is not measured.
- Where the wait goes, over the 9,158 returning requests all six arms sent before minute 110 (`ec_wait_split.csv`):
  `hicache` 33.18 s = 23.32 queued for other reasons + 8.82 waiting for its SSD read and then the GPU + 0.49 the read
  alone + 0.55 after admission; `ec_full` 5.18 = 4.17 + 0.77 + 0.06 + 0.18; `ec_exact` 5.15 = 4.15 + 0.62 + 0.02 +
  0.36; `jit_exact` 5.02 = 4.30 + 0.53 + 0.01 + 0.18; `jit_p50` 7.24 = 4.52 + 2.35 + 0.16 + 0.22; `oracle_v2` 5.98 =
  4.42 + 1.26 + 0.09 + 0.20.

## Against campaign 28 (Triton)

Each arm against its Triton run, the same users (`checks/vs_c28/compare_arms.out`):

| arm | Triton: TTFT mean (requests before 6,600 s) | FA3 | on the requests both sent |
|---|---|---|---|
| `hicache` | 115.7 s (4,540) | 33.5 s (9,653) | 115.69 against 3.10 s on 4,540: -112.6 s [-123.7, -101.5] |
| `ec_full` | 48.3 s (8,678) | 14.6 s (13,752) | -44.1 s [-46.2, -42.0] on 8,678 |
| `ec_exact` | 48.6 s (8,644) | 15.0 s (13,532) | -44.5 s [-46.5, -42.4] on 8,644 |
| `jit_p50` | 49.4 s (8,563) | 18.0 s (12,591) | -44.6 s [-46.7, -42.4] on 8,563 |
| `jit_exact` | 49.8 s (8,464) | 14.9 s (13,547) | -45.9 s [-48.1, -43.5] on 8,464 |
| `oracle_v2` | 47.7 s (8,731) | 17.2 s (12,742) | -43.2 s [-45.0, -41.2] on 8,731 |

- FA3 serves 46 to 113 % more requests and admits 433 to 557 users before the cut against 248 to 402. It saturates
  too, later: the GPU pool fills at minute 28 against 9, a queue of 25 forms at minute 40 against 13 to 14, and the
  cap of 128 users binds at minutes 71 to 74 (`hicache` 52) against 28 to 30 (`checks/by_minute_<label>.out` of both
  campaigns).
- The order changes. On Triton the five hinted arms were within 3.0 s of each other (`oracle_v2` first, 3.0 s ahead
  of `jit_exact`; `jit_p50` 0.9 s ahead of `jit_exact`; the rules 1.7 s ahead of `jit_p50`). On FA3 the rules lead
  `jit_p50` by 8.6 s, exact hints lead p50 hints by 7.75 s on the just-in-time engine, and the old oracle trails
  `jit_exact` by 6.3 s. Campaign 28's engine also had defect 2, which this one does not.

## Defect 1 on FA3

Returning requests sent before 6,600 s (`checks/cap_share_<label>.out`, `results/campaign28/checks/` for Triton):

| | `ec_full` | `ec_exact` | `jit_p50` | `jit_exact` |
|---|---|---|---|---|
| refused a read at the cap, FA3 | 136 of 13,171 | 157 of 12,961 | 272 of 12,046 | 151 of 12,974 |
| history computed again, FA3 | 6.34 % | 7.26 % | 11.40 % | 7.27 % |
| of it, by the refused requests, FA3 | 20.2 % | 21.3 % | 19.6 % | 19.0 % |
| refused, Triton | 33 of 8,257 | 33 of 8,223 | 93 of 8,145 | 97 of 8,048 |
| history computed again, Triton | 0.45 % | 0.51 % | 1.21 % | 1.37 % |
| of it, by the refused requests, Triton | 77.0 % | 79.8 % | 91.4 % | 92.6 % |

- On FA3 the hinted arms compute five to fourteen times more of the history again than on Triton, and the requests
  refused at the cap carry a fifth of it; on Triton they carried three quarters or more. Requests capped without a
  host-room refusal (next section) carry 9.8 to 14.0 % of FA3's recomputed history, against 76 to 92 % on Triton.
- On FA3 refused reads spend the same allowance: re-reads that hit the cap number 1,206 to 1,967 per arm in minutes
  60 to 110, against 304 to 592 on Triton (`checks/mechanisms_findings.txt`, cross_backend "defects").

## Why the arms separate on FA3

Analysed 2026-11-22 as four questions, each answered by one analyst agent and checked by an adversarial reviewer who
recomputed the numbers from the raw logs with their own code. Claims, verdicts and corrections:
`checks/mechanisms_findings.txt`; their scripts and outputs: `checks/mechanisms.tgz` (folders per question). The
numbers below are the reviewed ones; one run per arm throughout.

**What breaks: SSD reads refused host memory, then a loop through the SSD.**

- A read is sent to the SSD only with host pages to land in. When the host allocation fails after evicting what it
  can (`_launch_storage_hit`; log `srs_dispatch launched=0`), the read is refused, or cut to the room there is. No
  request is admitted before its read lands: in every arm, no launched read landed after its request's admission.
- Refused and cut reads carry 82 to 95 % of the history the hinted arms computed again in minutes 60 to 110; the
  requests with a refused read carry 77 to 89 % of all history computed again before 6,600 s. Triton refused none of
  its 4,046 to 4,896 dispatches per hinted arm.
- A refused read is retried, and the retries spend defect 1's allowance: 93 to 99 % of the refused requests end at
  the cap (`MAX_REFETCHES = 2`) and are admitted without their history a median 6 to 7.5 s later (`oracle_v2` 35 s).
- Every page computed is written through to the SSD by one write thread, with no check for pages the store already
  holds, and a page waiting for its write keeps a host lock (`write_backup_storage`; a locked node or one with
  children is not evictable, `_is_host_leaf`). FA3 serves about twice Triton's returning requests in minutes 30 to 60
  (146 to 147 a minute against 75 to 79), at about 1.9 K tokens written each, so it must write 4.2 to 5.3 K tokens
  a second. Once reads take the SSD (reads and writes together reach about 2.1 GB/s on this disk), the write thread
  manages 3.8 to 7.0 K. Triton needed about 2 to 3 K and its write thread stayed 40 to 63 % busy.
- The loop: the first refusals come with under 160 K tokens of writes pending and 11 to 18 s write latency; the
  history they recompute is written again (writes go from 2.6-3.5 to 5.1-7.2 M tokens per 10 minutes), the pending
  writes grow to 240 to 460 K tokens (p50) with 28 to 44 s latency, and more reads are refused. From then on the SSD
  runs at 99 % with reads and writes about 1 GB/s each. At the first refusals the logged host locks (pins, reads in
  flight, pending writes) come to 0.2 to 0.5 M of the 0.98 M tokens of host room beyond the GPU mirror; what holds
  the rest is not in the logs (inferred: host-only ancestors of locked nodes, the anchor locks of parked lookups).
- End state, minutes 90 to 110: 57 to 63 returning requests a minute with 46 to 51 % of their history computed
  again, the GPU busy throughout and 62 to 65 % of its time outside decode. Campaign 28 served 61 to 65 a minute with
  0 to 3 % computed again. With every FA3 arm in the loop (5,400 to 6,600 s) FA3 serves 58.9 to 64.6 a minute,
  Triton 62.5 to 67.2.
- Ruled out: the SSD cleaner (each arm's first pass comes 7 to 45 minutes after its loop started), the device
  slowing down (the same ceiling with and without cleaner passes; uncontended reads at 19 to 25 K tokens a second on
  both backends), another load on the SSD (the device's reads equal the engine's landed tokens), defect 2 (fixed).

**Why `jit_p50` and `oracle_v2` enter the loop 8 to 17 minutes earlier.** Sustained refusals (5 or more a minute for
5 minutes) start at minute 67 (`jit_p50`), 71 (`oracle_v2`), 79 (`ec_exact`), 80 (`jit_exact`) and 84 (`ec_full`).
The load is the same until then: queues within 4 of each other in minutes 56 to 66, sends within 5 %.

- Both read before the request arrives, on a box whose queue no longer hides it. In minutes 30 to 70, 57 %
  (`jit_p50`) and 64 % (`oracle_v2`) of their reads start before arrival (FA3 `jit_exact` 8 %, Triton `jit_p50` 7 %).
  With 6 s queues those reads land early and stay pinned: 208 and 267 K tokens of host memory on average in minutes
  30 to 60, against 7 to 14 K in the other arms; early-read pins held 27 and 32 s on average in minutes 30 to 65,
  where Triton's 40 to 68 s read line landed `oracle_v2`'s near admission (5.4 s).
- Many are wasted: 41 % of `jit_p50`'s early reads were given back when the plan expired; 30 % of `oracle_v2`'s were
  read again and 18 % never used. In minutes 30 to 70 `jit_p50` and `oracle_v2` read 29.8 and 31.3 M tokens from the
  SSD (the others 7.0 to 14.8 M) and evict 42 M tokens each from host (the others 18 to 26 M); their SSD is at 95 %
  or more from minute 41 and 39 (the others 75 to 79) and their write thread is 82 to 92 % busy in minutes 40 to 50
  (the others 29 to 38 %). Their first refusals come at minutes 32.4 and 31.5, in bursts that stop when the load
  eases; the other arms have none before minute 76.7.
- Read volume alone does not order the onsets: `ec_full` read twice `jit_exact`'s volume and entered the loop last.
  And `oracle_v2`'s pins had drained by its sustained onset (25 to 127 K in minutes 65 to 71); there the loop runs
  through histories pushed to the SSD earlier (the reviewer's reading).
- `jit_p50` on p50 hints: a plan expires at the predicted time plus max(60 s, gap) and registers its session as
  ended, evicted first; 533 plans expired and 519 of those turns came anyway, a median 111 s later. `next_use`'s
  overdue rank hardly acts (5 requests). Paired over minutes 30 to 65, `jit_p50` used 9.3 M more SSD tokens than
  `ec_full`, which gets the same hints: 4.0 M on turns it had staged, 4.6 M on regular turns and only 0.8 M on turns
  after an expired plan. Of the requests that arrived within 10 s of their hint, 463 of `jit_p50`'s read from the
  SSD (10.2 M tokens) against 60 of `ec_full`'s (1.5 M). What separates the two is what the server does with the
  hints, not the hints.
- Defect 1's empty lookups are 5 to 7 % of `jit_p50`'s extra recompute; `oracle_v2` has no gate.

**What the four rules did (`ec_full` against `jit_p50`, the same hints).**

- Read timing by expected cost acted: 2 reads before arrival against `jit_p50`'s 804. Its 1,907 "wait" decisions
  planned starts a median 982 s out, and 1,831 of them were decided again as "arrival" before then; "arrival" is
  62 % of its decisions (Triton: 2,354 "arrival", 30 "wait").
- The price never moved (5e-05 in all 1,445 updates: nothing staged, no harm). In `ec_exact` it fell to its 1e-07
  floor within 2.5 minutes of its first staged reads, with no harm charged.
- Soft pins: no bump or release in `ec_full` (2 releases in `ec_exact`). The "refused room" column of
  `checks/rules_activity_ec_full.out` is the engine's refused and cut reads, logged only in the expected-cost arms,
  not a rule's action.
- Eviction by chance of return ranks every eviction, but these arms cannot separate it from the read timing. With
  exact hints held fixed, `return_chance` (`ec_exact`) against `next_use` (`jit_exact`): 5.0 against 3.2 % of
  histories not resident at admission in minutes 30 to 65, and 0.16 s slower.
- The 8.6 s is the loop's timing: 8.31 s of it comes from requests `ec_full` sent in minutes 65 to 90, between
  `jit_p50`'s runaway at minute 68 and `ec_full`'s at 86. It cannot be credited to one rule.

**`ec_full` against the exact-hint arms (1.3 to 1.4 s): not resolved.** Also timing: all three had refusal bursts at
minutes 76 to 77; the exact-hint arms ran away at minutes 80 to 81 and `ec_full`, in and out until then, at 86 (the
minutes 80 to 85 alone are worth 0.9 to 1.0 s). No mechanism favours `ec_full` before that: it read more from the SSD
(11.3 against 4.5 and 6.8 M tokens in minutes 30 to 65), had more histories missing at admission (9.0 against 3.2
and 5.0 %) and registered 392 false session ends from hints with no table entry. One tipping event is shared by all
users of a run, so the interval over users does not cover it, and `ec_exact` and `ec_full` on the same server differ
by the same 1.4 s.

**Why the arms tied on Triton.** The box was GPU-bound from minute 13 with 60 s queues. Reads waited 40 to 68 s in
the read line and still landed 1.3 to 1.5 s before admission; 90 to 100 % of the just-in-time arms' reads were timed
from the queue position; writes ran at 0.17 to 0.30 GB/s with almost nothing pending; no read was refused. Only
defect 1 separated the arms: 0.2 to 1.8 % of the history computed again, 76 to 92 % of it by the capped requests.

**Costs on FA3.** Fitted from the decode-log intervals of both campaigns: computing a 20 to 30 K history again under
load costs 0.042 to 0.044 ms a token on FA3 against 0.13 on Triton (3.0 to 3.1 times); all prefill 0.047 against
0.148 to 0.157 ms a token; a decode step with 22 running about 23.5 against 38 ms. `checks/gpu_work.out` converts at
Triton's 0.11 ms a token and overstates FA3's GPU time 2.5 times (its 159 % for `hicache`); only its token columns
hold here. Reading 20 K tokens from an uncontended SSD takes 0.8 to 1.1 s on both backends, against 0.86 s to
compute them again on FA3 and 2.6 s on Triton; in the loop a read is 0.9 to 1.7 times the recompute. These runs do
not test which serves more, since in the loop the GPU and the SSD are both saturated.

**`hicache` on FA3.** Its own storm (the 8-lookup cap at arrival, 3,191 requests) starts at minute 39 against 22 on
Triton, and the user cap binds at minute 52 to 53 against 28. FA3 computes the storm's recompute faster: 74 to 113 s
mean TTFT in minutes 50 to 110 against 209 to 254 s. It writes 1.2 to 1.6 GB/s to the SSD, 84.4 M tokens (8.30 TB)
to a 2.46 TB volume in 2 h with no cleaner pass, so at least 73 % rewrote stored pages; writes fall 31 to 75 s behind
with 0.9 to 1.2 M tokens pending, and 21 to 45 % of what it reads from the SSD is delivered.

## Reading

- FA3 serves 46 to 60 % more requests in the hinted arms (113 % more in `hicache`) and saturates later, but the
  saturated box meets a new limit: the SSD write path and host memory, not the GPU's prefill. Every arm ends in the
  same loop at about Triton's request rate.
- The policies decide when an arm enters the loop. Reading before arrival on a short queue (the old oracle's 120 s
  lead, the just-in-time engine on p50 hints) fills host memory and the SSD and brings the loop 8 to 17 minutes
  earlier; the four rules on p50 hints do not read early and do as well as exact hints. That is the 6 to 9 s between
  the groups. Inside the loop (5,400 to 6,600 s) the early arms serve slightly more, so a longer run would likely
  shrink the gaps (inferred).
- The loop is an engine property (writes of pages the store already holds, one write thread, host locks held by
  pending writes, room refusals spending the lookup allowance), so a fixed engine may change these numbers and the
  order of the late arms. One run per arm; the saturated part has no repeat.

## Open after campaign 32

- Engine (`OPEN_ITEMS.md` §7): skip SSD writes of pages the store holds; more write-back concurrency, or writes
  before reads; refused reads should not spend `MAX_REFETCHES` (defect 1's fix A would cover them, inferred); a
  periodic log of host memory by lock reason, to name what fills it at the first refusals.
- Runs that would separate the rules: just-in-time timing with `return_chance` eviction, and expected-cost timing
  with `next_use`; and a repeat of `ec_full` and one exact-hint arm, to measure the spread of the minute the loop
  starts.
- `results/campaign28/checks/gpu_work.py` converts tokens to GPU seconds at Triton's cost; campaigns 32 and 33 need
  FA3's.
