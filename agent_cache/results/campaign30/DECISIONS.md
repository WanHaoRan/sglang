# Campaign 30: the baselines at campaign 34's setting

Ordered 2026-11-23 by the author ("schedule the run for all the baselines the same evaluation as in campaign 34,
namely sglang_vanilla, vllm_vanilla, vllm_lmcache, tokencake, continumm, continumm+lmcache"). Continuum runs twice:
its README's plain command is GPU only, and LMCache is its optional "With CPU offload", to which this setup adds
LMCache's local disk tier. The question: how the baseline systems do on campaign 34's users, against campaign 34's
arms on the fixed engine.

## Setting

`settings.env` is campaign 34's without the engine's four fix switches and its host-census logging: FreeInference
sessions as users, 0.1 users/s, at most 128 in session, recorded waits x1 (capped at 1,200 s), 2 h, seed 0; users with
a request over 65,536 tokens of prompt and output are left out. The users and their requests are campaign 34's, so
requests pair by user, stream and index (`checks/vs_c34/`, which holds campaign 34's five arms and these six).

| Label | System | Tiers |
|---|---|---|
| `sglang_vanilla` | this engine (`frozen/engine_fa3fix_20261122`, campaign 34's copy), arm `hbm_lru`: the radix cache on the GPU only, FA3 | GPU |
| `vllm_vanilla` | vLLM 0.33.0, prefix caching | GPU |
| `vllm_lmcache` | vLLM 0.33.0 + LMCache 0.5.7.post1 (server mode) | GPU; CPU 149.01 GiB (= 160 GB); disk, LRU, capped at 1,800 GiB (1,933 GB) |
| `tokencake` | TokenCake (fork `1901ebf`, vLLM 0.22.0), mode `offload-agent` | GPU; CPU 120 GiB (vLLM 0.22 pins the tier as one tensor, and torch rounds a pinned request up to a power of two: 149 GiB would ask for 256 GiB, more than the host's 196 GiB; left as is by the author's decision, `baselines/tokencake/README.md`); no disk tier |
| `continuum` | Continuum (fork `316a587`, vLLM 0.10.2), `--scheduling-policy continuum`, LMCache off | GPU |
| `continuum_lmcache` | the same with LMCache 0.3.6 in-process | GPU; CPU 149.01 GiB; disk, capped at 1,800 GiB |

Every server has the GPU pool of campaign 34 (668,096 tokens, from each boot log) and 64 running requests; the
vLLM-based servers batch 8,192 tokens a step (prefill and decode together), SGLang prefills in chunks of 8,192 with
decode in separate steps. Versions, patches and checks: `baselines/README.md` and each system's README. Every server
bound 127.0.0.1 (queue lines; `chain.log` records the bound address of each), and every request line in the six server
logs came from 127.0.0.1.

## Run

`chain_baselines.sh` (`chain.log`) started `baselines/common/run_baselines.sh campaign30` when campaign 34 ended.

| arm | GPU unit | client start | done | requests | errors | LMCache disk at the end |
|---|---|---|---|---|---|---|
| `sglang_vanilla` | GPU-36746535 | 03:47:43Z | 05:48:19Z | 7,140 | 0 | |
| `vllm_vanilla` | GPU-42ffbe53 | 14:26:47Z | 16:27:19Z | 7,169 | 0 | |
| `vllm_lmcache` | GPU-42ffbe53 | 16:30:53Z | 18:31:53Z | 11,688 | 0 | 1.4 TiB (SSD 61 % used) |
| `tokencake` | GPU-42ffbe53 | 18:35:54Z | 20:36:40Z | 7,854 | 0 | |
| `continuum` | GPU-42ffbe53 | 20:39:45Z | 22:40:18Z | 5,570 | 1 | |
| `continuum_lmcache` | GPU-42ffbe53 | 22:44:40Z | 00:45:36Z (11-25) | 10,832 | 0 | 1.7 TiB (SSD 72 % used) |

- The host was preempted twice during the campaign: in `sglang_vanilla`'s first attempt (02:45Z) and in
  `vllm_vanilla`'s (06:17Z, down until 14:14Z). Each arm restarted from its start; the interrupted attempts were
  removed at the author's request (`README.txt`).
- GPU units: campaign 34's `jit_p50`, `ec_full` and `jit_exact` ran on GPU-0f61782e; a host restart during campaign 34
  moved the box to GPU-36746535, where its `hicache` and `ec_exact` and this campaign's `sglang_vanilla` ran; the second
  restart here moved it to GPU-42ffbe53, where the five vLLM-based arms ran. On identical requests of minutes 0 to 10
  the medians agree: cold prefill and decode time per token within 0.7 % between the first two units on SGLang
  (`../campaign34/checks/gpu_unit.out`), within 0.8 % (cold prefill) and 1.4 % (decode) among the three on vLLM
  0.33.0 (`checks/gpu_unit_vllm.out`). On GPU-42ffbe53 the mean cold prefill is 13 to 16 % higher than on the other
  two, from its slowest quarter. `nvidia-smi` on it showed a 700 W cap and SM clocks of 1,965 to 1,980 MHz with
  software power-cap dips (1,575 MHz at 692 to 694 W), no thermal or hardware slowdown and 0 uncorrected ECC errors;
  with one run per unit, batching variance cannot be told from a slightly slower, power-capped card.
- The client gives up on a request after 3,600 s without data (`sock_read`, `scripts/replay_programs.py:710`) and
  retries it once unless the 7,200 s deadline has passed. In `continuum` five first requests waited that long:
  `continuum-c228-t0` (sent at 3,600 s) timed out after the deadline and is the arm's one error; `c224-t0` to `c227-t0`
  (sent at 3,530 to 3,599 s) were retried at 7,130 to 7,199 s and are recorded with the retry's send time and TTFT (0.5
  to 60 s). The timeout also caps every recorded TTFT (the longest in `continuum` is 3,555 s).
- Two server logs hold a traceback, neither during the replay: `vllm_vanilla`'s at its stop (`EngineDeadError`, after
  the client's summary) and `tokencake`'s at boot (`start_server.sh`'s own probe of `/v1/tokencake/events`, which
  expects a 400).
- TokenCake answered 219 of the client's 15,163 stall notifications (`/v1/tokencake/events`) with HTTP 404 and the
  rest with 200; its lifecycle counter marks 219 requests "unknown". Not looked into.
- Requests still running when the client stops (7,200 s plus 30 s) are cancelled and leave no record. Of the requests
  sent before the 6,600 s cut, that drops none in any arm but Continuum's: 67 in `continuum` and 21 in
  `continuum_lmcache`, 61 and 20 of them users' first requests. (`checks/missing_at_cut.out` prints 1 for `hicache` and
  2 for `tokencake`: requests sent in the 0.1 s between 6,600 s and the client's in-flight count, all recorded; for
  `continuum` it prints 72, which adds the five timed-out first requests. `checks/continuum_first.out`.) Counted at
  630 s each, with the five timed-out first requests at 3,600 s, they would raise `continuum`'s mean by at least 10.1
  s and `continuum_lmcache`'s by at least 1.2 s.
- Every number in this record was checked against the raw files by three independent readers (`checks/verify/`).

## Results

Each arm's own requests sent before 6,600 s (`checks/vs_c34/compare_arms.out`; users with an answered request). For
the vLLM-based systems the client sees only how many prompt tokens were cached, not from which tier, so the table gives
the recomputed share. "Recomputed" is compare_arms.py's share of all prompt tokens computed, which includes each
request's new text (about 6 % of prompt tokens); the share of the history computed again is 1 minus "history
delivered" (ours 0.4 %, the GPU-only baselines 62 to 80 %; `checks/report_review/clarity/stats.out`).

| arm | users | requests (a minute) | TTFT mean, p50, p90, p99 (s) | first request, returning (s) | recomputed | history delivered, of the ideal | wait for a slot: mean, max (s) |
|---|---|---|---|---|---|---|---|
| `ec_exact` (campaign 34) | 621 | 15,643 (142.2) | 10.66, 7.87, 22.60, 54.94 | 10.70, 10.66 | 6 % | 0.996 | 58.6, 517 |
| `ec_full` (campaign 34) | 620 | 15,562 (141.5) | 10.96, 8.34, 24.01, 53.07 | 11.04, 10.96 | 6 % | 0.996 | 63.0, 538 |
| `hicache` (campaign 34) | 448 | 9,967 (90.6) | 31.88, 5.95, 85.61, 110.46 | 31.94, 31.88 | 34 % | 0.711 | 286.3, 2,236 |
| `vllm_lmcache` | 481 | 11,103 (100.9) | 31.37, 9.02, 87.94, 114.72 | 21.52, 31.82 | 7 % | 0.995 | 254.8, 1,746 |
| `continuum_lmcache` | 435 | 10,082 (91.7) | 37.90, 1.37, 13.09, 1,298 | 738.32, 6.31 | 6 % | 0.997 | about 448, 2,160 (*) |
| `tokencake` | 356 | 7,501 (68.2) | 57.94, 17.03, 174.30, 254.89 | 28.90, 59.39 | 49 % | 0.551 | 628.3, 3,135 |
| `continuum` | 222 | 5,278 (48.0) | 65.84, 1.41, 5.30, 3,018 | 1,480.54, 3.72 | 64 % | 0.381 | about 740, 3,780 (*) |
| `sglang_vanilla` | 338 | 6,762 (61.5) | 68.61, 94.28, 114.61, 134.53 | 57.12, 69.21 | 81 % | 0.202 | 784.6, 3,306 |
| `vllm_vanilla` | 339 | 6,775 (61.6) | 68.73, 93.50, 113.79, 132.46 | 57.06, 69.35 | 82 % | 0.195 | 792.4, 3,351 |

(*) Over every user let in before the cut (455 and 288; `checks/verify/result_tables/slot_wait_all.out`).
`compare_arms.py` counts only users with an answered first request and prints 381.7, 2,023 and 200.1, 1,798; in every
other arm the two are the same. Campaign 34's `jit_exact` and `jit_p50` (10.92 and 11.47 s on 15,601 and 15,608
requests) are in the same file.

On the 4,844 requests every one of the eleven arms sent before the cut and got answered (from the 222 users GPU-only
Continuum answered; 83 % of them sent in the first 30 minutes by `ec_exact`'s clock, later in the slow arms):

| arm | TTFT mean, p50, p90 (s) |
|---|---|
| `ec_exact` | 1.29, 0.10, 4.82 |
| `jit_exact` | 1.30, 0.10, 4.91 |
| `ec_full` | 1.35, 0.11, 4.95 |
| `jit_p50` | 1.42, 0.11, 5.15 |
| `hicache` | 3.32, 0.11, 5.66 |
| `vllm_lmcache` | 3.74, 0.21, 8.98 |
| `continuum_lmcache` | 16.16, 0.66, 9.11 |
| `tokencake` | 42.82, 4.31, 157.62 |
| `sglang_vanilla` | 53.04, 49.53, 109.47 |
| `vllm_vanilla` | 53.50, 52.72, 108.90 |
| `continuum` | 71.51, 1.29, 5.48 |

Each system against the four rules, on the requests both sent before the cut (B - A, interval over users, 95 %):

| B | requests | - `ec_full` | - `ec_exact` |
|---|---|---|---|
| `hicache` | 9,967 | +26.71 s [+23.52, +30.28] | +26.96 s [+23.52, +30.54] |
| `vllm_lmcache` | 11,103 | +25.24 s [+22.55, +27.99] | +25.52 s [+22.81, +28.41] |
| `continuum_lmcache` | 10,081 / 10,082 | +32.95 s [+28.53, +37.98] | +33.22 s [+29.01, +38.38] |
| `tokencake` | 7,501 | +54.80 s [+49.18, +60.40] | +54.94 s [+49.62, +60.45] |
| `continuum` | 5,278 | +64.21 s [+53.00, +78.12] | +64.29 s [+52.92, +78.68] |
| `sglang_vanilla` | 6,762 | +65.56 s [+60.30, +70.87] | +65.68 s [+60.02, +70.85] |
| `vllm_vanilla` | 6,775 | +65.68 s [+60.49, +70.63] | +65.79 s [+60.51, +70.90] |

By send time, requests and mean TTFT (s):

| minutes | `ec_exact` | `ec_full` | `hicache` | `vllm_lmcache` | `continuum_lmcache` | `tokencake` | `continuum` | `sglang_vanilla` | `vllm_vanilla` |
|---|---|---|---|---|---|---|---|---|---|
| 0-10 | 1,027, 0.15 | 1,025, 0.14 | 1,025, 0.16 | 1,024, 0.18 | 945, 0.16 | 927, 0.27 | 868, 0.35 | 994, 0.19 | 982, 0.26 |
| 10-30 | 3,177, 0.47 | 3,173, 0.50 | 3,177, 0.50 | 3,151, 0.76 | 2,364, 14.91 | 2,423, 8.76 | 1,112, 169.28 | 1,654, 23.92 | 1,639, 24.97 |
| 30-60 | 4,644, 5.91 | 4,625, 6.18 | 2,911, 30.53 | 3,200, 31.66 | 2,758, 58.15 | 1,783, 89.36 | 1,345, 93.83 | 1,667, 93.79 | 1,688, 93.22 |
| 60-110 | 6,795, 20.27 | 6,739, 20.82 | 2,854, 79.60 | 3,728, 65.58 | 4,015, 46.40 | 2,368, 107.19 | 1,953, 16.78 | 2,447, 109.45 | 2,466, 108.32 |

The Continuum numbers above are the client's records. With its four retried first requests placed at their original
sends, `continuum` has 5,282 requests at 68.53 s, minutes 30 to 60 read 1,349 at 104.28 s, and `continuum` - `ec_full`
is +66.90 s (`checks/verify/result_tables/retried_effect.out`); the 67 cancelled requests are still missing (Run).

- With the GPU pool alone, this engine's GPU-only arm (`sglang_vanilla`) and stock vLLM tie: 68.61 and 68.73 s on
  6,762 and 6,775 requests (+0.04 s [-0.20, +0.25] on the same requests, run on two different units), 81 and 82 %
  recomputed. Both fall behind in minutes 10 to 30 (24 and 25 s) and stay near 93 to 109 s after.
- vLLM + LMCache keeps the history (7 % recomputed, 0.995 of the ideal delivered, as campaign 34's rule and
  timed-read arms: 6 %, 0.995 to 0.996) and is the best baseline by mean TTFT and requests served, but its TTFT climbs
  from 0.76 s in minutes 10 to 30 to 31.66 and 65.58 s later: 25.2 s behind `ec_full` on the same requests, and 4.73 s
  ahead of `hicache` on `hicache`'s 9,967 (interval 3.49 to 6.02 s). Continuum + LMCache answers 64 % of the 9,711
  requests both sent faster (lower p50 and p90) but is 13.18 s [+8.76, +18.34] slower on average, because it starves
  first requests.
- Continuum's queue serves streams already started before any newly started one, whether a new user's first request
  or a new subagent stream: the order its README describes (pinned jobs first, then jobs by first arrival; a job is
  one stream, `vllm/v1/core/sched/request_queue.py:240-273` in the fork), and its preemption takes the newest job
  (3,537 preemptions on the GPU only). After minute 10 a returning request (index above 0) has a mean TTFT of 3.6 to
  4.7 s (GPU only) and 3.9 to 6.6 s (with LMCache); a first request 1,552 to 3,412 s and 209 to 1,211 s, capped by the
  client's 3,600 s timeout; with LMCache, new subagent streams 985 to 1,365 s (`checks/continuum_first.out`,
  `checks/verify/claims/split.out`). So its p50 is the lowest of all (1.41 and 1.37 s) while its mean is high: 65.84 s,
  third of eleven and at least 75.9 s with the missing requests; 37.90 s with LMCache, second of the baselines.
  GPU-only Continuum answers the fewest users before the cut (222 of the 288 it let in; with LMCache 435 of 455), and
  its mean in minutes 60 to 110 is only 16.78 s because just 7 of the 68 first requests sent then were answered
  before the end (1 timed out, 60 were still waiting).
- TokenCake (CPU 120 GiB, no disk) recomputes 49 % and serves 7,501 requests at 57.94 s, between the LMCache systems
  and the GPU-only ones by mean, though its p90 (174.30 s) is the highest of all arms. Its CPU tier served 33 % of the
  prompt tokens (57.8 M of 177.0 M). Its benefit test kept 5,030 stalled requests and declined most others (73,406
  backoff, 67,007 no waiting demand), never for lack of CPU room (0 `cpu_capacity`), and the tier turned over about 34
  times (4.37 TB written into it), so the run does not say whether a 160 GB tier would deliver more
  (`metrics_end_vllm.txt` in its run folder).
- Users of the baselines wait longer for a session slot: 255 to 792 s on average over every user each let in before
  the cut (GPU-only Continuum about 740 s), against 59 to 63 s for the four rules (0.0 to 0.1 s on each baseline's own
  users). The 140 sessions that ended before the cut in every arm last longer too (mean 566 to 2,152 s against 541 to
  543 s).

## Reading

- On the same users and GPU pool, with 160 GB of host memory and the SSD offered (TokenCake takes 120 GiB and no
  disk; three baselines use the GPU only; the baselines ran on another H200 unit, Run), the four rules on the gap
  table (`ec_full`) are 25 s ahead of the best baseline (vLLM + LMCache), 33 s ahead of Continuum + LMCache, 55 s ahead
  of TokenCake and 64 to 66 s ahead of the GPU-only systems (more for Continuum, whose gaps are lower bounds), and
  serve 40 % more requests than the best baseline (15,562 against 11,103). On exact hints (`ec_exact`), which no
  baseline gets, each gap is 0.1 to 0.3 s larger.
- The baselines group by tier: the two with a disk tier deliver nearly all the history (0.995, 0.997) and lead;
  TokenCake (CPU only) delivers 0.551 and comes next; the GPU-only three deliver a fifth to two fifths and trail.
  Within a group the order does not follow the history: GPU-only Continuum delivers 0.381 against about 0.20, yet is
  17 s slower than the other two on the same requests (+17.02 s [+6.22, +30.83] against `sglang_vanilla`, +16.74 s
  [+5.63, +29.75] against `vllm_vanilla`), and Continuum + LMCache is 13 s behind vLLM + LMCache at the same delivery.
  The engines differ (vLLM 0.10.2, 0.22.0, 0.33.0), so tiers alone do not explain the order.
- One run per arm. The vLLM-based arms ran on a third H200 unit with the same median speed but a 13 to 16 % higher
  mean cold prefill (Run). TokenCake's CPU tier is 120 GiB, not 160 GB, and it has no disk. Continuum's means are lower
  bounds (Run).

## Open after campaign 30

- Why vLLM + LMCache, which delivers 0.995 of the history, is 25 s behind the four rules (the GPU unit is one
  candidate, Run).
- TokenCake's 219 stall notifications answered 404 (219 lifecycles marked "unknown").
- TraceLab (campaign 31, prepared) is not run.
