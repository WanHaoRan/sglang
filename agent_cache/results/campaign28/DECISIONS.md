# Campaign 28: the campaign 26 arms on FreeInference, 16 weeks, sessions as arriving users

Ordered 2026-11-19 02:45Z: replay the public traces under the campaigns' own schedule, every recorded session one
arriving user, and compare original HiCache, the old oracle v2, JIT exact, JIT p50 and the four rules on the gap
distributions. Campaign 29 is the same setup on TraceLab and runs after this one. Background and the test runs: `results/campaign27/DECISIONS.md`.

## Setting (`settings.env`, the same for every arm)

| | |
|---|---|
| trace | `data/converted/freeinference_16w.programs.jsonl.gz (12,002 sessions of 16 weeks, 1,222,199 requests; 3,787 sessions under the cap)` |
| users | arrive as a Poisson process at 0.1 a second, at most 128 in session (idle ones count), seed 0 |
| waits | recorded, x1, each capped at 1,200 s |
| sessions | whole sessions, those with a request over 65,536 tokens left out; no cut by turns |
| length | the client stops sending after 7,200 s; requests in flight get 30 s |
| server | `Qwen/Qwen3-30B-A3B-Instruct-2507`, context 262,144, bf16 KV, GPU pool 668,096 tokens, host pool 160 GB (1,627,648 tokens), SSD store wiped before every arm, frozen engine `engine_ec_20261111d` |
| client | `scripts/replay_programs.py --recipe none`: synthetic token ids of the recorded lengths on `/generate`, outputs of the recorded length |

Launcher `frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign28 campaign29` (one process for both
campaigns), watcher `c28_users/watch.sh`, queue `queue.txt`, finished arms in `manifest.txt`, progress in `launcher.log`.
Relaunch after a shutdown: `: > results/.current_run`, `docker start sglang_hicache`, the `docker exec -d` line at the top
of the launcher, then the watcher from `agent_cache/`. Finished arms are skipped; an interrupted arm reruns from its start.

## Arms (`queue.txt`, in this order)

| Label | Server arm | Hints the client sends |
|---|---|---|
| `hicache` | `three_tier_wc`: original HiCache | none |
| `oracle_v2` | `three_tier_wc_oracle_v2_nu`: the old oracle, next-use eviction | the exact gap to the stream's next request |
| `jit_exact` | `three_tier_wc_oracle_v2_nu_owned`: just-in-time reads, arrival gate and hold | the exact gap |
| `jit_p50` | the same | the gap table of the session's harness and model, and the recorded reply's key |
| `ec_full` | the same with the four rules (campaign 26's `ec_full` flags) | the gap table |
| `ec_exact` | `ec_full`'s server, the oracle arm (added 2026-11-22, Results) | the exact gap |

Choices the order left open, as launched:

- **Sessions under 65,536 tokens**, as in every test run: with the model's whole window this box decodes at about
  110 ms a token and serves under two requests a minute (campaign 27).
- **The 1,200 s gap cap** of the earlier campaigns is kept; no cut at 50 turns (the new client has none).
- **A reply with no tool call has the key `_no_call`** in the two table arms (`--hint-no-call`, added to the client on
  2026-11-19 with a test; off by default). Those replies are the human waits; without the key they fall into the
  pooled entry. The earlier client did the same (`scripts/reply_keys.py`). Asked of the author; changeable by
  editing two queue lines before the arms start.
- **Tables stay per harness and model**, learned online from the run's own gaps.
- **Prompt tokens are uniform random ids** (no token pool).
- **The SSD store's cleaner is at 90 and 80 % of the volume**, not the 70 and 60 % of the earlier campaigns. These
  arms write an estimated 1.4 to 2.2 TB in 2 h; at 70 % (1.72 TB) the cleaner, which deletes the oldest pages first
  whether or not a live session needs them, would fire late in some arms and not in others. Campaign 26 never
  reached its mark. Each arm's DONE line gives the cleaner's passes and the SSD's fill.
- **An arm is done when its client wrote a summary with errors on at most 1 % of its requests**; the launcher stops
  otherwise. The trace and the scripts are pinned by `inputs.sha256` at the first launch; an arm refuses to start on
  changed inputs.
- **The client reads lines up to 32 MiB** (`read_bufsize`): `/generate` streams every output id so far in each line,
  and outputs reach 21K tokens here, past aiohttp's default limit.
- **Who the users are** (`checks/arrival_set.out`, the 729 users that arrive in 2 h, the same in every arm): 83 agent
  systems; SWE-bench evaluation agents are 189 users and 46 % of the requests (tool waits of 0.8 s at the median, 0.3 %
  over 60 s), the other 13 harnesses are people's sessions (9 to 21 % of their waits over 60 s). The one week tested
  in campaign 27 had none of the evaluation sessions. Results are to be read by harness as well.
- **Expected load** (`results/campaign27/checks/our_setup_cap128.out`, from one week's sessions): 106 users in session
  and 626 output tokens a second asked, about 156 % of what this box sustains; with the evaluation agents it will be
  higher. A saturated run, as campaign 26's 128-user setting was. Told to the author at launch.

A pre-launch review by three reader agents (`results/campaign28/checks/prelaunch_review.json`) found the launcher's
missing start guard, the done test, the cleaner mark and the line limit, all fixed before launch; it also notes that a
table arm's request with no entry to pick is treated by the engine as a session's end (3 to 5 % of such requests in a
compressed dry run, mostly in the first half hour), that the exact gap is wrong when a parent waits for a subagent
(0.04 % of hints), and that requests still in flight 30 s after the deadline leave no record, so arms are compared on
requests sent before 6,900 s. A rehearsal of all five arms, three minutes each, ran through the launcher first
(`results/campaign28r/`).

## Results

**Launched 04:26:39Z on 2026-11-19 (GPU unit `GPU-dcd42ca3`), `hicache` first.** Both campaigns run from one launcher
process, about 2 h 10 min an arm: campaign 28 to about 15:10Z, campaign 29 to about 01:50Z on 2026-11-20. `hicache`
booted with its settings (GPU pool 668,096 tokens, host pool 1,627,648, cleaner at 90 and 80 %, no prefetcher; ready
after 286 s with cold weights; client from 04:32:40Z after 65 s of trace loading). First four minutes: 16 users in
session, TTFT 0.06 to 0.43 s on returning requests, no queue.

**`hicache` done 06:33:34Z: saturated, as expected; mean TTFT 115.7 s.** 4,650 requests, no errors, no cleaner pass,
SSD 38 % full at the end. Table: `compare_arms.out` (requests sent before 6,600 s); per minute:
`checks/by_minute_hicache.out`.

| | |
|---|---|
| users with a request before 6,600 s | 248 (729 arrive in 2 h); 120 sessions ended |
| requests before 6,600 s | 4,540 (41.3 a minute); 193 output tokens/s |
| TTFT mean, p50, p90, p99 | 115.7, 128.4, 245.7, 272.4 s |
| prompt tokens from GPU, host, SSD; recomputed | 12 %, 26 %, 16 %; 45 % (delivered / ideal 0.60) |
| wait for a slot | mean 585 s, max 4,152 s |

| sent in minutes | requests | TTFT mean | returning requests reading SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 721 | 0.71 s | 0 % | 11 % |
| 10-30 | 1,648 | 33.3 s | 8 % | 20 % |
| 30-60 | 863 | 196.5 s | 27 % | 71 % |
| 60-110 | 1,308 | 229.6 s | 31 % | 69 % |

- Course of the run: clean up to about 47 users in session (minute 8: 24 running, GPU pool 0.8 full, TTFT 0.3 s). The
  GPU pool fills at minute 9 with 26 requests running; the queue passes 25 at minute 13, SSD reads start at minute 23,
  the cap of 128 binds from minute 28. From there to the end the server holds about 23 requests running and 85 to 105
  queued, sends drop from about 100 a minute to 20 to 30, and 70 % of the prompt tokens are computed again although
  92 % were in the cache of an unbounded server.
- Counting rule, changed from the head's 6,900 s: requests sent before 6,600 s. With TTFT near 250 s a request sent
  after 6,900 s rarely finishes inside the 30 s grace, and 113 users had a request in flight when the client stopped.
  The slowest recorded request took 3,326 s, so a few sent before 6,600 s are still missing; arms are also compared
  on the requests they share (`compare_arms.py`, pairs by user, stream and index).
- `oracle_v2` launched 06:33:34Z on the same GPU unit: ready after 180 s, pools as above, prefetcher on, read order
  `deadline`, pin fraction 0.3, write-through, eviction `next_use`; client from 06:36:55Z. Its first three minutes
  repeat `hicache`'s arrivals and TTFT.
- Where `hicache`'s TTFT goes, over its 4,174 returning requests sent before 6,600 s (mean 113.5 s): 81.0 s in the
  queue for reasons other than the request's own read, 30.1 s waiting for its SSD read and then for the GPU, 0.8 s for
  the read alone, 1.6 s after admission (`results/compare_c28_designs/ec_wait_split.csv`).
- Tools for both campaigns, in `frozen/oracle_scripts_staging/c28_users/`: `compare_arms.py` (the table above, pairs
  over users, harnesses, hint coverage; host, stdlib) and `figs.sh <campaign>` (the figure set of
  `results/compare_c26_designs`, into `results/compare_c28_designs/` and `compare_c29_designs/`: per-arm timelines,
  `compare.png`, `ttft_over_time_60s.png` with the users waiting for a slot, `pricing_vs_jit.png`,
  `ec_wait_split.png`). Both read the campaign's manifest, so they are rerun as arms land.
- Why `hicache` recomputes (`checks/storm_cause_hicache.out`, returning requests sent in minutes 60 to 110): 553 of
  1,253 got nothing from the cache, 305 a part and 395 everything, 381 of those from SSD. Of the 553, 491 had reached the
  engine's cap of 8 SSD lookups at arrival: their history was still in host memory then, so the lookups asked only for
  the new tokens and found none; it was evicted from host memory during the 4-minute queue wait, and at admission the
  engine looks no further ("admitted without further L3 lookups"). Example `hicache-c86-t24`: 32,129 of 33,075 tokens
  matched at arrival, 249 s queued, `cached_input_len=0`, 23 s of prefill.

**`oracle_v2` done 08:38:49Z: no recompute storm, mean TTFT 47.7 s, still saturated.** 9,357 requests, no errors, no
cleaner pass, SSD 72 % full at the end. Per minute: `checks/by_minute_oracle_v2.out`.

| requests sent before 6,600 s | `hicache` | `oracle_v2` |
|---|---|---|
| users with a request; sessions ended | 248; 120 | 402; 274 |
| requests (a minute); output tokens/s | 4,540 (41.3); 193 | 8,731 (79.4); 379 |
| TTFT mean, p50, p90, p99 | 115.7, 128.4, 245.7, 272.4 s | 47.7, 52.4, 78.9, 96.6 s |
| prompt tokens from GPU, host, SSD; recomputed | 12, 26, 16 %; 45 % | 7, 39, 47 %; 7 % |
| delivered / ideal | 0.60 | 0.997 |
| slot wait over the 248 users both admitted | mean 585 s | mean 111 s |
| length of the 120 sessions both ended | mean 1,871 s | mean 996 s |

| sent in minutes | requests | TTFT mean | returning requests reading SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 723 | 0.72 s | 0 % | 11 % |
| 10-30 | 1,958 | 23.3 s | 28 % | 9 % |
| 30-60 | 2,472 | 59.6 s | 64 % | 6 % |
| 60-110 | 3,578 | 62.5 s | 65 % | 6 % |

- On the 4,540 requests both arms sent before 6,600 s (248 users): 115.7 against 34.4 s, a difference of -81.3 s
  (95 % interval over users -89.7 to -73.4), `oracle_v2` faster on 77 %. All harnesses but `eval:swe-bench`: 125.4
  against 48.9 s on each arm's own requests.
- Course of the run: the same first 20 minutes as `hicache` (GPU pool full at minute 9, 24 queued at minute 13),
  with SSD reads from minute 19. The cap of 128 binds from minute 30; from there to minute 118 a mean of 21 requests
  run (17 to 26), 77 queue (63 to 89) and 75 are sent a minute (43 to 117); `hicache` had 23 running, 96 queued and
  27 sent a minute. The GPU pool stays 0.97 full in both: the box is bound by what fits and decodes on the GPU, and
  this arm keeps that room for decoding.
- No storm (`checks/storm_cause_oracle_v2.out`, minutes 60 to 110): 0.2 % of the history an unbounded cache holds
  was computed again, against 66.6 %; no request reached the SSD lookup cap; 2,226 of 3,436 returning requests were
  served from SSD.
- Where the TTFT goes, over the 4,174 returning requests both arms sent (34.2 s): 16.6 s in the queue for other
  reasons, 16.6 s waiting for the request's SSD read and then for the GPU, 0.5 s for the read alone, 0.5 s after
  admission (`results/compare_c28_designs/ec_wait_split.csv`).
- Reading: with the cap bound, the 128 users in session cycle through the queue, so TTFT is about the queue length
  over the completion rate (75 / 1.2 a second, about 62 s; `hicache` 95 / 0.4, about 237 s). At this setting TTFT
  and requests served measure the same thing: how much GPU time goes to decoding and not to computing prompts again.
- `jit_exact` launched 08:38:50Z on the same GPU unit: pools as above, read timing `just_in_time`, arrival gate and
  hold on, prefetch on, read order `deadline`, pin fraction 0.3, write-through, eviction `next_use`, price learning
  and soft pins off.
- `compare_arms.py` now also prints the slot wait over the users every arm admitted.

**`jit_exact` done 10:44:14Z: ties the old oracle, 3 s behind (49.8 against 47.7 s).** 9,030 requests, no errors, no
cleaner pass, SSD 70 % full at the end. Per minute: `checks/by_minute_jit_exact.out`.

| requests sent before 6,600 s | `oracle_v2` | `jit_exact` |
|---|---|---|
| users with a request; sessions ended | 402; 274 | 396; 268 |
| requests (a minute); output tokens/s | 8,731 (79.4); 379 | 8,464 (76.9); 366 |
| TTFT mean, p50, p90, p99 | 47.7, 52.4, 78.9, 96.6 s | 49.8, 53.9, 85.9, 115.1 s |
| prompt tokens from GPU, host, SSD; recomputed | 7, 39, 47 %; 7 % | 7, 43, 41 %; 8 % |
| delivered / ideal | 0.997 | 0.986 |
| slot wait over the 248 users all three admitted | mean 111 s | mean 111 s |

| sent in minutes | requests | TTFT mean | returning requests reading SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 724 | 0.68 s | 0 % | 11 % |
| 10-30 | 1,992 | 22.7 s | 7 % | 9 % |
| 30-60 | 2,340 | 63.0 s | 60 % | 8 % |
| 60-110 | 3,408 | 67.0 s | 63 % | 8 % |

- Pairs: against `oracle_v2` on the 8,464 requests both sent (396 users), 46.8 against 49.8 s, +3.0 s (95 % interval
  over users +2.6 to +3.5), `jit_exact` faster on 45 %; against `hicache` -80.6 s (-89.2 to -72.4). On the 4,540
  requests all three sent: 115.7, 34.4 and 35.1 s. One run per arm: the interval covers the users drawn, not the
  spread between runs.
- Course of the run: as `oracle_v2`. From minute 30 to 118 a mean of 22 requests run, 78 queue and 71 are sent a
  minute (75 for `oracle_v2`). It reads from SSD later: 7 % of the returning requests of minutes 10 to 30 against
  28 %, the same share from minute 30.
- Where the TTFT goes, over the 4,174 returning requests all three sent (34.8 s): 17.0 s in the queue for other
  reasons, 16.4 s waiting for the request's SSD read and then for the GPU, 0.8 s for the read alone, 0.5 s after
  admission; `oracle_v2` 16.6, 16.6, 0.5 and 0.5 s.
- What it computes again, and why (`checks/storm_cause_jit_exact.out`, `checks/missed_requests_jit_exact.out`): 97
  of its 8,048 returning requests sent before 6,600 s were refused a read at the engine's cap of three SSD lookups a
  request (`MAX_REFETCHES = 2` in `next_turn_prefetch.py`); they carry 93 % of the 2.56 M history tokens computed
  again (1.4 % of the history; 2.0 % in minutes 60 to 110 against 0.2 % for `oracle_v2`). Example
  `jit_exact-c61-t25`: 15,936 of 16,192 tokens matched at arrival; the lookups for the 256-token tail found nothing
  on SSD, once at arrival and twice a minute later; the whole history then left host memory in the last 17 s of a
  79 s queue wait, the gate answered `capped`, and the prompt took 23 s to compute. The cap counts lookups that
  found nothing; counting only reads that moved pages would close this. An engine change, so not for this campaign.
  Added from campaign 29's `jit_exact` entry: that tail is the previous reply, which the replay sends back as new
  ids (or the part of the previous prompt the next one drops), so it is in no tier; on TraceLab the cap takes 333
  of 2,924 returning requests and 11.8 % of the history.
  Added from campaign 29's `ec_full` entry (`checks/cap_path.out` here and there): "twice a minute later" is the
  `next_use` eviction ranking a turn queued for over 60 s as overdue and evicting its session's tail
  (`next_use_eviction.py`, `overdue_grace_s`); the plan then spends its two re-reads on lookups that find nothing.
  Here the queue wait is 54 s, so few requests reach that mark (4 in each `next_use` arm); on TraceLab 291 and 252
  do. `return_chance`, the rules arm's eviction, ranks a queued turn last, which is why `ec_full` is refused less on
  both traces. Campaign 29's record ends with both campaigns side by side.
- `jit_p50` launched 10:44:15Z with `--hint-no-call` (no word from the author on that key before the launch).
- `jit_p50`'s boot stalled on the weights: 7 GB of three shards had left the page cache during the last run, and
  the loader's page faults read them from the root disk at 9 MB/s (4 KB reads, read-around off), about 13 minutes
  of a 40-minute boot limit. Reading the three files with `cat` took 19 s and the load finished at once (10:50:50Z,
  boot about 5 minutes late). From here `c28_users/warm_weights.sh` runs on the host (log `results/c28_warm.log`):
  it reads the weight files once at each `LAUNCH` line. It changes boot time only; the client starts after the
  server is ready.

**`jit_p50` done 12:54:03Z: ties `jit_exact` (49.4 against 49.8 s); the table's hints cost nothing here.** 9,102
requests, no errors, no cleaner pass, SSD 70 % full at the end. Per minute: `checks/by_minute_jit_p50.out`.

| requests sent before 6,600 s | `oracle_v2` | `jit_exact` | `jit_p50` |
|---|---|---|---|
| users with a request; sessions ended | 402; 274 | 396; 268 | 398; 270 |
| requests (a minute); output tokens/s | 8,731 (79.4); 379 | 8,464 (76.9); 366 | 8,563 (77.8); 370 |
| TTFT mean, p50, p90, p99 | 47.7, 52.4, 78.9, 96.6 s | 49.8, 53.9, 85.9, 115.1 s | 49.4, 52.7, 86.2, 111.9 s |
| prompt tokens from GPU, host, SSD; recomputed | 7, 39, 47 %; 7 % | 7, 43, 41 %; 8 % | 7, 44, 42 %; 8 % |
| delivered / ideal | 0.997 | 0.986 | 0.988 |
| slot wait over the 248 users all four admitted | mean 111 s | mean 111 s | mean 113 s |

| sent in minutes | requests | TTFT mean | returning requests reading SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 724 | 0.74 s | 0 % | 11 % |
| 10-30 | 1,988 | 22.9 s | 12 % | 9 % |
| 30-60 | 2,329 | 63.5 s | 61 % | 8 % |
| 60-110 | 3,522 | 65.0 s | 60 % | 7 % |

- Pairs: against `jit_exact` on the 8,447 requests both sent (396 users), 49.7 against 48.8 s, -0.9 s (95 % interval
  over users -1.3 to -0.4), `jit_p50` faster on 50 %; against `oracle_v2` on 8,554 requests, +2.3 s (+1.8 to +2.8);
  against `hicache` -79.9 s (-88.0 to -72.1). On the 4,540 requests all four sent: 115.7, 34.4, 35.1 and 35.8 s.
  One run per arm, and the spread between two runs of one arm is not measured at this setting, so the 0.9 s between
  the two just-in-time arms is not a finding.
- Hints (`compare_arms.out`, `checks/hint_quality_jit_p50.out`): 8,563 requests name a next request; 284 of them
  say the session ends, 8,279 hint a wait. Of those 95.7 % come from a table entry (by level: tool and first word
  3,915, tool 2,793, the harness's all-replies row 1,215) and 8.8 % are for replies with no tool call. The true wait
  falls inside the hint's p10 to p90 for 68.5 %; p50 over true wait has median 0.98, p10 0.10, p90 3.29.
- The table knows the tool waits and not the human ones. 83 % of the waits are under 10 s, and there the hinted p50
  is 0.8 to 1.2 times the true wait at the median. The 535 waits of a minute or more (6.5 %) get a hint of 6 to 20 s
  (median) against 103 to 956 s, short by a mean of 99 s (60 to 300 s) and 777 s (over 300 s). Replies with no tool
  call are the human turns: true wait median 65 s against 1.1 s after a tool call; their `_no_call` row says 26 s
  (median), short for 69 %. By harness: SWE-bench evaluation agents 99 % with an entry and 73 % inside p10 to p90,
  the others 80 to 97 % and 55 to 74 %.
- The result does not move with these errors (`checks/hint_effect_jit_exact_jit_p50.out`, the pair split by the
  wait before the request): after waits of 60 to 300 s `jit_p50` is +0.9 s (-0.6 to +2.3) on 61 s, after 300 s or
  more +1.5 s (-1.1 to +3.9) on 68 s, after 10 to 60 s +1.4 s (+0.4 to +2.4); after waits under 1 s it is 2.5 s
  faster (-3.2 to -1.7), where the two arms' hints differ least (hinted p50 0.7 s against a true 0.6 s), so I take
  about 2 s as what two runs differ by. My reading: a request waits about a minute in the queue, so its read has time to land either way.
- The same split for `oracle_v2` against `jit_exact` (`checks/hint_effect_oracle_v2_jit_exact.out`): just-in-time is
  behind after every wait, +2.5 s under 1 s rising to +6.4 s (+3.8 to +9.0) after 300 s or more.
- Course of the run: as `jit_exact`. Minutes 30 to 118: a mean of 22 requests run (18-26), 78 queue (63-91) and 71
  are sent a minute.
- Where the TTFT goes, over the 4,174 returning requests all four sent (35.4 s): 15.8 s in the queue for other
  reasons, 18.2 s waiting for the request's SSD read and then for the GPU, 1.0 s for the read alone, 0.6 s after
  admission.
- The read cap again (`checks/cap_share_jit_p50.out`, `checks/storm_cause_jit_p50.out`): 93 of 8,145 returning
  requests sent before 6,600 s were refused a read; they carry 91 % of the 2.30 M history tokens computed again
  (1.2 % of the history). `checks/cap_share.py` prints this for an arm; for `jit_exact` it gives the 97 requests and
  93 % above.
- The watcher's alert just before the DONE line was the server's stop, not the run: at 12:53:51Z, 5 s after the
  client's summary, the SSD-read thread touched the host pool after its release (`AttributeError: 'NoneType' object
  has no attribute 'data_ptr'` in `prefetch_io_aux_func`). `oracle_v2` and `jit_exact` log the same one traceback at
  their stop; `hicache` none. No effect on the records. The watcher now alerts on a traceback only while the client
  runs.
- `ec_full` launched 12:54:04Z, same GPU unit. The weight helper read 21.7 GB back from disk in 77 s (that much had
  left the page cache; at 9 MB/s it would have taken the whole 40-minute boot limit); READY after 137 s. Pools as
  above, cleaner 90/80, read timing `expected_cost`, memory price 5e-5, price learning `harm`, soft pins on, eviction
  `return_chance` with a 30 s horizon; client `--hints dist --hint-no-call`, sending from 12:56:45Z. The server logs
  `ec_price` events from the first minute.

**`ec_full` done 14:58:39Z: ties the other three hinted arms (48.3 s; old oracle 47.7, just-in-time 49.4 and 49.8);
the rules had nothing to decide.** 9,330 requests, no errors, no cleaner pass, SSD 72 % full at the end, no
traceback in the server log (the three arms before it log one at their stop). Per minute:
`checks/by_minute_ec_full.out`. All five arms are done.

| requests sent before 6,600 s | `hicache` | `oracle_v2` | `jit_exact` | `jit_p50` | `ec_full` |
|---|---|---|---|---|---|
| users with a request; sessions ended | 248; 120 | 402; 274 | 396; 268 | 398; 270 | 401; 273 |
| requests (a minute); output tokens/s | 4,540 (41.3); 193 | 8,731 (79.4); 379 | 8,464 (76.9); 366 | 8,563 (77.8); 370 | 8,678 (78.9); 377 |
| TTFT mean, p50, p90, p99 (s) | 115.7, 128.4, 245.7, 272.4 | 47.7, 52.4, 78.9, 96.6 | 49.8, 53.9, 85.9, 115.1 | 49.4, 52.7, 86.2, 111.9 | 48.3, 51.7, 80.2, 103.1 |
| prompt tokens from GPU, host, SSD; recomputed | 12, 26, 16 %; 45 % | 7, 39, 47 %; 7 % | 7, 43, 41 %; 8 % | 7, 44, 42 %; 8 % | 7, 42, 43 %; 7 % |
| delivered / ideal | 0.599 | 0.997 | 0.986 | 0.988 | 0.996 |
| history computed again (`checks/cap_share_*.out`, `checks/hint_effect_*.out`) | 40 % | 0.3 % | 1.4 % | 1.2 % | 0.4 % |
| slot wait over the 248 users all five admitted | mean 585 s | 111 s | 111 s | 113 s | 106 s |
| whole run: requests; users admitted of 729 | 4,650; 256 | 9,357; 424 | 9,030; 411 | 9,102; 414 | 9,330; 425 |

| `ec_full`, sent in minutes | requests | TTFT mean | returning requests reading SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 724 | 0.74 s | 0 % | 11 % |
| 10-30 | 2,004 | 22.8 s | 14 % | 9 % |
| 30-60 | 2,439 | 59.7 s | 62 % | 6 % |
| 60-110 | 3,511 | 64.7 s | 63 % | 7 % |

- Pairs for `ec_full` (`compare_arms.out`): against `oracle_v2` on 8,656 requests (401 users) +0.7 s (95 % interval
  over users +0.4 to +1.1); against `jit_exact` on 8,463 requests -2.6 s (-3.0 to -2.2); against `jit_p50` on 8,558
  requests -1.7 s (-2.1 to -1.3); against `hicache` -81.6 s (-90.4 to -73.7). On the 4,540 requests all five sent:
  115.7, 34.4, 35.1, 35.8 and 34.1 s.
- My reading: the four hinted arms tie. They span 2.1 s of TTFT (4 %) and 3 % in requests served, one run per arm,
  and the spread between two runs is not measured here. Their order follows the share of history each computes
  again: 0.3, 0.4, 1.2 and 1.4 % for 47.7, 48.3, 49.4 and 49.8 s. The gain is the one over `hicache`: about 80 s of
  TTFT and 1.9 times the requests.
- What the rules did (`checks/rules_activity_ec_full.out`): nothing that needs a price.
  - Rule 1, read timing by expected cost: 2,354 decisions for hinted requests whose history had left host memory,
    every one "leave the read to the arrival". The engine's estimate of the queue wait (median 58 s) was over its
    estimate of the read (median 5.9 s) each time, by 5.3 s at the least, so a read started at arrival lands before
    admission and an earlier one only holds room. 30 more decisions, "wait", came in the seconds after the client
    stopped and the queue emptied.
  - So no read was staged before its request arrived: no early-read pin, no release, and no bump (rule 2, soft pins).
  - Rule 4, the learned price: 1,401 updates, all at the start value 5e-5. No read was refused room and no harm was
    charged; reads in flight or pinned held about 90 K of the 980 K tokens of host room beyond the GPU pool's mirror.
  - Rule 3, eviction by chance of return, is the one rule that could act. The split of prompt tokens over GPU, host
    and SSD is the same as `jit_p50` within 2 points; this arm cannot tell its effect from the others'.
- Why the arms tie (`checks/read_starts.out`, the server's `ntp_due` lines): the three arms on the just-in-time
  engine start their reads while the request waits in the queue, timed from its queue position: 4,083 of 4,104
  reads in `jit_exact`, 3,494 of 3,873 in `jit_p50`, 4,322 of 4,322 in `ec_full`. A hint started the read before
  arrival for 21, 379 and 0 requests. `oracle_v2` starts 2,322 of 4,781 before arrival and lands in the same place.
  With about a minute of queue and a read of about 6 s, when the read starts is not what limits a request; the GPU
  is (22 requests run, 77 queue, in every arm). `jit_p50` also lost 405 plans that expired before their request
  came (60 s or more past the hinted time); `jit_exact` 49, `ec_full` 23 (its plans last until the request comes,
  the session ends or 1,800 s pass).
- The read cap (`checks/cap_share_ec_full.out`, `checks/storm_cause_ec_full.out`): 33 requests of 8,257 refused a
  read, against 97 and 93 in the two just-in-time arms; 0.45 % of the history computed again (860 K tokens), 77 % of
  it by those 33. This arm rereads more (3,059 against 2,741 and 2,138) and is capped less; which flag does that
  (eviction by chance of return, or no staged reads using up lookups) this arm alone does not say.
- The pair split by the wait before the request (`checks/hint_effect_*_ec_full.out`): against `jit_p50` it is 1.1 to
  3.5 s faster after every wait; against `oracle_v2` it is -0.7 s after waits under 1 s and +1.0 to +4.2 s after
  longer ones (+4.2 s, +2.2 to +6.3, after 300 s or more). Hints as in `jit_p50`: 8,391 of a wait, 287 of a
  session's end, 95.7 % from a table entry, true wait inside p10 to p90 for 68.6 % (`checks/hint_quality_ec_full.out`).
- Course of the run, minutes 30 to 118: a mean of 21 requests run (18-26), 77 queue (64-89) and 74 are sent a minute.
  Where the TTFT goes over the 4,174 returning requests all five sent (33.8 s): 15.5 s in the queue for other
  reasons, 17.1 s waiting for the request's SSD read and then for the GPU, 0.7 s for the read alone, 0.5 s after
  admission.
- Figures with five arms: `results/compare_c28_designs/` (`pricing_vs_jit.png`, `ec_wait_split.png`, timelines).
- What this setting cannot show: whether a hint is needed at all (every hinted arm reads in the queue, as a no-hint
  arm on the same engine would), and what the price and soft pins do (no early read to price or bump). A sixth arm,
  the just-in-time engine with no hints, answers the first; a setting below saturation is needed for the second.
- Campaign 29 took over at 14:58:39Z (`results/campaign29/DECISIONS.md`).

**Why `oracle_v2` is ahead of `jit_exact` by 3.0 s (asked by the author): the just-in-time engine computes nine
times more history again, and on a GPU-bound box that time comes out of every queued request's wait.**
`checks/gpu_work.out` (requests sent in minutes 30 to 110; GPU time at the engine's measured prefill cost,
`RECOMPUTE_S_PER_TOKEN` = 0.11 ms a token):

| sent in minutes 30 to 110 | `oracle_v2` | `ec_full` | `jit_p50` | `jit_exact` |
|---|---|---|---|---|
| history computed again | 0.27 M tokens (0.18 %) | 0.81 M (0.56 %) | 2.20 M (1.55 %) | 2.46 M (1.77 %) |
| GPU time for it; share of the 4,800 s | 30 s; 0.6 % | 89 s; 1.9 % | 242 s; 5.0 % | 271 s; 5.6 % |
| output tokens a second | 376 | 368 | 359 | 354 |
| requests a minute | 75.6 | 74.4 | 73.1 | 71.8 |
| time per output token, mean | 57.1 ms | 58.8 ms | 59.3 ms | 60.6 ms |
| TTFT mean | 61.3 s | 62.6 s | 64.4 s | 65.4 s |

- The chain: `jit_exact` gives 5.0 points more of the GPU to history it had cached, serves 5.0 % fewer requests a
  minute, and with the queue as long (78 against 77; the cap of 128 users fixes it) a request waits 6.7 % longer.
  The slow requests themselves are at most 0.3 s of the 3.0 s (97 requests, 73 s against 50 s); the rest is every
  other request queuing behind the extra prefill.
- The gap exists only once the run saturates. Minutes 0 to 30: `jit_exact` computes less history again than
  `oracle_v2` (0.11 M against 0.27 M tokens), serves 90.5 against 89.4 requests a minute and has the lower TTFT in
  minutes 10 to 30 (22.7 against 23.3 s). So the read timing is not the cost.
- The cost is the arrival gate's cap, which `oracle_v2` runs without (server flags that differ: arrival gate and
  arrival hold on, read timing `just_in_time` for `horizon`, reply key from the hint). 93 % of `jit_exact`'s history
  computed again belongs to the 97 requests the gate refused a read (`checks/cap_share_jit_exact.out`). In minutes
  60 to 110, 57 of its requests missed history, all at the cap; `oracle_v2` had 10 (`checks/storm_cause_*.out`).
  Both arms lose a queued request's pages again (`ntp_reevict`: 2,317 and 1,550). Of the requests that reached the
  prefetcher's refetch limit that way, `oracle_v2` still served 574 of 581 their whole history (0.21 M tokens
  computed again), reading once more with no gate to refuse it; `jit_exact` served 800 of 888, and the other 88
  computed 2.38 M tokens again.
- The four hinted arms sit on one line: GPU share for history 0.6, 1.9, 5.0, 5.6 % against 376, 368, 359 and 354
  output tokens a second. So the order of the four arms is this cap, not their hints or rules.
- The fix stays the one above (count only lookups that moved pages); an engine change, the author's call.

**Turns per session (asked by the author; `checks/turn_counts.out`, a turn is one request to the model).**

| sessions | count | requests | mean | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| whole converted trace (16 weeks) | 12,002 | 1,222,199 | 101.8 | 8 | 44 | 225 | 945 | 5,040 |
| kept under 65,536 tokens | 3,787 | 107,107 | 28.3 | 5 | 19 | 66 | 117 | 615 |
| the 729 users that arrive in 2 h | 729 | 21,258 | 29.2 | 6 | 20 | 66 | 117 | 152 |
| of them, SWE-bench evaluation agents | 189 | 9,864 | 52.2 | 15 | 47 | 95 | 140 | 152 |
| of them, the other harnesses | 540 | 11,394 | 21.1 | 5 | 16 | 45 | 80 | 117 |

- The 65,536-token limit is what shortens the sessions: it leaves out 68 % of them and 91 % of the requests, the long
  ones (the whole trace has 46 % of its sessions over 50 turns, the kept set 16 %).
- No cut at 50 turns was applied: 44 % of the arriving users' requests sit in sessions of over 50 turns (73 % for
  the SWE-bench agents, 20 % for the others).
- What an arm replayed before 6,600 s: `oracle_v2` 402 users and 8,731 requests, 21.7 a user (p50 16, p90 47, max
  121); 276 sessions sent whole and 126 cut by the end of the run, 74 % of those users' turns; a request's turn
  number has p50 13, p90 45, max 120, and 8 % of the requests are at turn 50 or later. The other three hinted arms
  are within 1 % of that; `hicache` 248 users, 4,540 requests, 18.3 a user, 125 whole and 123 cut.
- 13 of the 729 users have a subagent stream (298 of their requests are off the main stream).

**GPU memory or GPU computation (asked by the author)? Both bind, and computation sets the rate: a decode step's
time follows the context tokens the running batch holds, not how many requests share it.**
`checks/decode_scale.out`: the four hinted arms, the server's decode lines with no prefill batch inside (1,682
intervals of 40 steps), so the rate is decode alone.

| running requests | context in decode | ms per step | output tokens/s, decode alone |
|---|---|---|---|
| 1 | 7 K | 7.6 | 132 |
| 7-9 | 145 K | 20.3 | 357 |
| 10-13 | 258 K | 29.0 | 389 |
| 14-17 | 644 K | 37.9 | 439 |
| 18-20 | 638 K | 38.4 | 504 |
| 21-23 | 638 K | 37.1 | 588 |
| 24-26 | 644 K | 37.0 | 669 |
| 27 or more | 653 K | 35.9 | 755 |

- With the pool full a step takes 36 to 38 ms whether 14 or 27 requests are in it. Fit: 11.2 ms + 0.18 ms a running
  request + 0.036 ms a thousand context tokens (R2 0.92); at 22 requests and 635 K tokens the context is 60 % of the
  step, the fixed part 30 %, the requests 10 %.
- At a full pool decode takes 62 % of the time and prefill the other 38 % (51.8 M tokens, 0.158 ms a token; the
  0.11 ms above is first-turn prefill, which a whole history computed again is).
- So the pool decides how many requests run (22 of these 26 K-token ones; the server allows 64), and the work per
  step decides the rate. By the fit, twice the pool with the same requests: 44 running, 64 ms a step, 683 against
  582 output tokens a second in decode (+17 %), about +10 % in requests a minute with prefill counted. The same pool
  with half the context a request: +81 % in decode. Both are extrapolations of the fit.
- A live sample in campaign 29's `hicache` at 16:23Z (another arm, in its storm): GPU utilization 100 % in 12 of 12
  one-second samples. To repeat in a hinted arm.

**How to leave the saturated region (asked by the author): either a lower arrival rate or a lower user cap does it,
at about 0.034 users/s or 33 users for 80 % of the GPU; and host memory has to shrink with it, or nothing reaches
the SSD.** `checks/load_model.out`: a model from the trace (the 729 users: 29.2 turns a user, 25.0 s mean wait
before a turn, 26.6 K prompt tokens of which 1.58 K new, 281 output tokens, 31.2 K tokens held a session) and the
measured GPU costs (`checks/decode_scale.out`), with the measured capacity of 1.25 requests/s. Not a run: it leaves
out queueing from bursts, and its own capacity (1.44 requests/s) is 15 % over the measured one.

| share of the GPU's capacity | requests/s | users/s | users in session | running | session KV; host GB to hold it all |
|---|---|---|---|---|---|
| 60 % | 0.75 | 0.026 | 24 | 4 | 0.74 M tokens; 67 GB |
| 70 % | 0.88 | 0.030 | 28 | 6 | 0.88 M; 80 GB |
| 80 % | 1.00 | 0.034 | 33 | 7 | 1.04 M; 95 GB |
| 90 % | 1.12 | 0.039 | 39 | 10 | 1.22 M; 111 GB |
| 95 % | 1.19 | 0.041 | 42 | 12 | 1.32 M; 121 GB |

- Check against the runs: they were clean up to about 47 users in session (minute 8), which the model puts at 95 to
  100 % of capacity. So a cap in the mid-40s, my earlier suggestion, is the edge and not below it.
- The two knobs give the same mean load (users in session = arrival rate x session length, about 970 s here). They
  differ in what adapts. With the cap binding, a slower arm sends less (each user waits for its reply), arms serve
  different users, and an overload cannot show. With arrivals below capacity and the cap left as a guard, every arm
  gets the same users and requests, and an arm that storms piles users up to the cap, which is a result.
- At these loads 160 GB holds every session (95 GB at 80 %), so no request would read from SSD and no hint would
  have a read to time, as campaign 26's 100 GB arms showed. For the SSD to matter the host pool has to be under the
  sessions' KV: about 70 GB at 80 % (712 K tokens for 1.04 M held), campaign 26's own pairing of 44 users and 70 GB.
- My proposal, the author's call: 0.034 users/s, cap 128 kept as a guard, 70 GB, x1 gaps, 2 h (about 245 users and
  6,500 requests an arm), the five arms plus the no-hint arm on the just-in-time engine.

**Discussion with the author: a setting that separates the four hinted designs (no decision yet).** The four differ
in when an evicted history is read back and how long the read holds host room. That shows only where the GPU has
headroom (so a queue does not cover the read) and the idle sessions' KV is past the host pool (so there are reads,
and room to compete for). Model numbers at 80 % of the GPU (`checks/load_model*.out`, `checks/think_times.out`):

| workload (FreeInference) | users/s | users in session | session KV | users in 2 h | session KV past a 100 GB host pool |
|---|---|---|---|---|---|
| 65,536 limit, every harness, waits capped at 1,200 s (campaign 28's trace) | 0.034 | 33 | 95 GB | 245 | 0 % |
| the same without the SWE-bench agents | 0.032 | 37 | 112 GB | 233 | 11 % |
| without them, waits capped at 3,600 s | 0.032 | 54 | 165 GB | 233 | 39 % |
| 131,072 limit, without them, 3,600 s | 0.0086 | 20 | 103 GB | 62 | 3 % |
| 262,144 limit, without them, 3,600 s | 0.0032 | 11 | 101 GB | 23 | 1 % |

- A longer context limit does not put more on the SSD on this box: sessions get larger and the GPU carries as many
  fewer, so the KV alive stays near 1.1 M tokens, and a 2 h run sees 23 to 62 users.
- Idle time does: the SWE-bench agents wait 2.3 s a turn (0.3 % of waits over 60 s), the other harnesses 44.6 s
  capped at 1,200 s (12 % over 60 s), 68.4 s capped at 3,600 s. The sessions hold about (1 + wait / reply time)
  times the KV of the running requests.
- My proposal: FreeInference without the SWE-bench agents, 65,536 limit, waits capped at 3,600 s, x1, 0.032 users/s,
  cap 128 as a guard, 100 GB of host memory (room for about 10 idle sessions beyond the GPU pool's mirror), the five
  arms and the no-hint arm, after a 30-minute trial. To report: TTFT after waits over 60 s beside the mean, p99,
  host room held by early reads, SSD bytes read and not used.
- What to expect: 88 % of turns follow a wait under a minute and stay in the GPU pool, so the mean moves by tenths
  of a second (campaign 26: 0.14 to 0.21 s for exact hints); the turns after a human wait move by the read time,
  about 3 to 4 s for 30 K tokens. A read from this SSD (7 to 10 K tokens/s) is no faster than this model's prefill
  (9 K tokens/s), so here the tiers save GPU time and reads done ahead, not a slow recompute; a dense model would
  make a miss several times dearer. (Corrected below: the read itself runs at 20 K tokens/s, twice the prefill; the
  7 to 10 K were the engine's estimates, which include the wait for the read thread.)

**The storage read thread is near its limit in every hinted arm (found 2026-11-19 while reading campaign 29's old
oracle).** `checks/read_thread.out` (each read's pages and I/O time, the server's `s2h_io` lines) and
`checks/read_waits.py`. One thread reads SSD pages into the host pool, one read at a time.

| arm | reads | tokens read | read thread busy, minutes 30 to 110 | read speed | a read parked before dispatch, minutes 50 to 110 |
|---|---|---|---|---|---|
| `hicache` | 1,019 | 21.8 M | 19 % | 18.8 K tokens/s | not parked (no deadline order) |
| `oracle_v2` | 4,860 | 113.3 M | 96 % | 19.6 K | 51 to 66 s (medians by ten minutes) |
| `jit_exact` | 4,032 | 93.8 M | 83 % | 20.0 K | 52 to 67 s |
| `jit_p50` | 4,305 | 101.7 M | 91 % | 19.7 K | 61 to 66 s |
| `ec_full` | 4,325 | 101.4 M | 87 % | 20.0 K | 53 to 64 s |

- The thread reads 19.6 to 20.0 K tokens a second (1.9 GB/s): a 30 K-token history in 1.5 s, against 3.3 s to
  compute it. This corrects the read speed in the discussion entry above.
- It is a second resource at its limit beside the GPU, which the entry on memory and computation did not see. The
  GPU still sets the rate: the pool is full of running requests, and `jit_exact` leaves 17 % of the thread idle
  while serving 5 % fewer requests than the old oracle. For the old oracle at 96 % the two cannot be told apart, and
  the extrapolation above (twice the pool, +10 % requests a minute) would need 10 % more reads from a full thread.
- It explains the tie of the four hinted arms better than their timing rules do. Every arm's reads wait in one
  deadline-ordered line for this thread, so a read lands when its turn comes, whatever start time the arm chose.
  The old oracle has no start time for a queued request's read, and its reads still landed 2.6 s before admission
  (`checks/reread_fate_oracle_v2.out`: 4,787 reads pinned, 68 found no room, a pin held 2.6 s). On TraceLab the
  line is short and the same engine loses half of its reads (campaign 29's record).
- The old oracle read 113 M tokens for the requests the just-in-time arms served with 94 to 102 M: 11 to 21 % more
  SSD reading for the same work.
- For a follow-up setting: history served from SSD is capped near 20 K tokens a second. A setting meant to compare
  read timing should keep this thread well under full, or the line for the thread does the timing. The proposal
  above passes by the model: at 80 % of the GPU its requests carry 0.68 x 30.2 K = 20.5 K history tokens a second,
  and with at most 39 % of the session KV past a 100 GB host pool the reads are under 8 K tokens a second, 40 % of
  the thread.

**Discussion with the author: start sessions with a long context, or let the trace progress as now (asked
2026-11-19; no decision yet)?** `checks/start_state.py`: the sessions replayed on paper from an empty system, the
client's own loader and arrival draw, no queueing (a load under the GPU's capacity), 12 h so the settled level
shows. Outputs `checks/start_state.out`, `checks/start_state_proposal.out`, `results/campaign29/checks/start_state.out`.

| setting, below saturation | settles at | session KV at 80 %, 90 % of that | sessions over 1 h | first prompt, context at the end (medians) |
|---|---|---|---|---|
| FreeInference, every harness, waits capped at 1,200 s, 0.034 users/s | 28 users, 0.91 M tokens (90 GB) | minute 21, minute 29 | 3 % of sessions, 23 % of the time in session | 11.5 K, 35.8 K |
| the proposal: no SWE-bench agents, waits capped at 3,600 s, 0.032 users/s | 40 users, 1.34 M tokens (132 GB) | minute 45, minute 54 | 12 %, 55 % | 17.5 K, 37.0 K |
| TraceLab, 0.044 users/s | 17 users, 0.57 M tokens (56 GB) | minute 16, minute 16 | 1 %, 14 % | 15.7 K, 35.8 K |

- A session is long from its first request: half of its final context at the median (0.48 to 0.50; 0.37 with the
  SWE-bench agents), then 1.5 to 2 K tokens a turn. A start at a uniform instant of the session would give 30 to
  34 K tokens against 14 to 18 K at turn 1, a gap the trace closes by about turn 15.
- What takes long is the population, not the context. In the proposal the KV past a 100 GB host pool, the only
  state that puts anything on the SSD, is 0 at minute 30, 0.06 M tokens at minute 45, 0.20 M at minute 60 and
  0.23 M from minute 90: the sessions that idle for long are the ones that reach the SSD, and they are 12 % of
  sessions and 55 % of the time in session. A 2 h arm from empty would spend its first hour filling.
- The saturated campaigns do not have this problem: overload fills the system, the user cap binds at minute 28 to
  31, and the records already count minutes 30 to 110 apart.
- TraceLab below saturation never passes the host pool (56 GB of session KV): it reaches the SSD only overloaded,
  whatever the start.
- The client's own mid-session start (`--start-window` with `--warmup primer`, the AgentX recipe) exists for lanes
  only, a closed loop with the weakness of a bound user cap. Its primer computes every initial history cold and at
  once, leaves the tiers in the order of priming and not as an arm's policy would, and meets empty hint tables with
  a full system; its sessions are drawn uniformly, so the long ones are fewer than in a running system. The first
  minutes after it still have to be left out.
- My recommendation: keep the trace progressing from turn 1 under Poisson arrivals as the workload, and count from
  the minute the population has settled. For the follow-up below saturation either run 3 h arms and count hours 2
  and 3, or add a warm start to the arrival mode (the settled population seeded mid-session at t = 0, drawn in
  proportion to session length, primed, then arrivals from turn 1 as now), which makes a 75-minute arm worth a 2 h
  one. The second is a client change, to make after campaign 29 (its inputs are pinned until then).
- Limits: one arrival draw; the settled levels move by about a fifth between hours (heavy-tailed session lengths);
  main streams only.

**Open after both campaigns (2026-11-20).** The two engine defects (the lookup cap, the `next_use` overdue rank of
queued turns), the decisions that wait for the author (the fix, the follow-up setting, the start state) and the
steps to pick the work up are in `agent_cache/OPEN_ITEMS.md`.

**The oracle arm `ec_exact` (2026-11-22).** Ordered by the author with the `next_use` fix: the oracle is the four
rules given each true next-request time as their distribution (a point), not just-in-time on exact hints, which stays
as the arm with exact hints and no rules, so the arms form {just-in-time, rules} x {p50, exact}. Queue line: `ec_full`'s
flags with `oracle` hints, on `ec_full`'s copy d (the fix does not touch the rules' eviction), so the pair differs by
the hint alone; the server arguments match `ec_full`'s but for the run path and the random seed. Started by
`results/qfix_check/chain.sh` after the FA3 batch: launched 01:27:48Z, DONE 03:31:56Z, 9,228 requests, 0 errors, no
traceback; 8,357 exact hints of the 8,644 requests sent before 6,600 s.

| A | `ec_exact` - A, requests both sent before 6,600 s | interval |
|---|---|---|
| `hicache` | -80.86 s | [-88.90, -72.65] |
| `oracle_v2` | +1.15 s | [+0.78, +1.56] |
| `jit_exact` | -2.02 s | [-2.45, -1.59] |
| `jit_p50` | -1.21 s | [-1.58, -0.83] |
| `ec_full` | +0.50 s | [+0.16, +0.82] |

- Exact hints add nothing to the rules here: `ec_exact` is 0.5 s (1 %) behind `ec_full`, one run each, and behind it
  in every wait bucket but the waits over 300 s (`checks/hint_effect_ec_full_ec_exact.out`: +0.20 to +1.14 s).
- The rules decide nothing with exact hints either (`checks/rules_activity_ec_exact.out`): the price stays at
  5e-05, no room is refused, no release or bump; 2,406 of 2,415 read starts are at arrival, the queue wait estimate
  (median 59.6 s) covering the read (5.8 s). A saturated queue leaves the timing nothing to decide, as for `ec_full`.
- Figures redrawn with six series (`results/compare_c28_designs/`, the oracle dashed in the rules' colour). Over the
  4,174 returning requests every arm sent before minute 110: HiCache 113.50, old oracle 34.20, JIT p50 35.44, JIT
  exact 34.78, the rules 33.84, the oracle 34.60 s.
