# Campaign 29: the campaign 26 arms on TraceLab v0.0.2, sessions as arriving users

Ordered 2026-11-19 02:45Z: replay the public traces under the campaigns' own schedule, every recorded session one
arriving user, and compare original HiCache, the old oracle v2, JIT exact, JIT p50 and the four rules on the gap
distributions. Campaign 28 is the same setup on FreeInference and runs before this one. Background and the test runs: `results/campaign27/DECISIONS.md`.

## Setting (`settings.env`, the same for every arm)

| | |
|---|---|
| trace | `data/converted/tracelab_v0.0.2.programs.jsonl.gz (8,058 sessions, 665,453 requests; 3,679 sessions under the cap)` |
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
- **Who the users are** (`checks/arrival_set.out`, the 729 users that arrive in 2 h, the same in every arm): 24 agent
  systems, Claude Code 541 users and 72 % of the requests, Codex 188 and 28 %; tool waits of 0.1 to 0.4 s at the
  median, 5 to 7 % of waits over 60 s.
- **Expected load** (`results/campaign27/checks/our_setup_cap128.out`): 46 users in session and 461 output tokens a
  second asked, about 115 % of what this box sustains; the 25-minute probe at 44 users and x10 gaps already saturated
  (campaign 27). A saturated run is expected. Told to the author at launch.

A pre-launch review by three reader agents (`results/campaign28/checks/prelaunch_review.json`) found the launcher's
missing start guard, the done test, the cleaner mark and the line limit, all fixed before launch; it also notes that a
table arm's request with no entry to pick is treated by the engine as a session's end (3 to 5 % of such requests in a
compressed dry run, mostly in the first half hour), that the exact gap is wrong when a parent waits for a subagent
(0.04 % of hints), and that requests still in flight 30 s after the deadline leave no record, so arms are compared on
requests sent before 6,600 s (6,900 s at launch; moved when campaign 28's first arm ran with TTFT near 250 s, see its
record). A rehearsal of all five arms, three minutes each, ran through the launcher first
(`results/campaign28r/`).

## Results

Times are UTC. Requests are counted when sent before 6,600 s; pairs across arms by (user, stream, request number);
intervals are 95 %, by bootstrap over users (2,000 draws, seed 0); one run per arm. Tools as campaign 28:
`c28_users/compare_arms.py results/campaign29 > results/campaign29/compare_arms.out`, `c28_users/figs.sh campaign29`
(figures in `results/compare_c29_designs/`), and the checks in `results/campaign28/checks/` (each prints its usage).

- `hicache` launched 14:58:39Z on 2026-11-19, straight after campaign 28's last arm, same GPU unit (GPU-dcd42ca3).
  The weight helper read the weights back in 76 s; READY after 140 s. GPU pool 668,096 tokens, host pool 1,627,648
  tokens (160 GB), cleaner 90/80, next-turn prefetch off, 8 lookups a request. Client: 3,679 sessions kept, 4,379
  left out over 65,536 tokens; `--hints none`, 0.1 users/s, at most 128 in session, 7,200 s, gaps x1 capped at
  1,200 s; clock from 15:02:46Z. First two minutes: 9 users in session, 44 requests, returning TTFT 0.3 to 0.6 s, no
  errors.

**`hicache` done 17:03:35Z: a harder storm than on FreeInference: mean TTFT 289 s, 2,081 requests, 76 % of prompt
tokens computed.** 2,090 requests in all, no errors, no cleaner pass, SSD 49 % full at the end, no traceback.
Per minute: `checks/by_minute_hicache.out`.

| requests sent before 6,600 s | `hicache` |
|---|---|
| users with a request; sessions ended | 292; 164 |
| requests (a minute); output tokens/s | 2,081 (18.9); 132 |
| TTFT mean, p50, p90, p99 | 289.2, 377.9, 465.7, 534.8 s |
| first requests; returning | 226.7 s; 299.4 s |
| prompt tokens from GPU, host, SSD; recomputed | 8, 11, 4 %; 76 % |
| delivered / ideal | 0.293 |
| slot wait | mean 600 s |
| by harness | Claude Code 1,561 requests, 284.5 s; Codex 520, 303.4 s |

| sent in minutes | requests | TTFT mean | returning requests reading SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 333 | 8.4 s | 0 % | 22 % |
| 10-30 | 536 | 152.8 s | 4 % | 65 % |
| 30-60 | 502 | 387.0 s | 5 % | 96 % |
| 60-110 | 710 | 454.8 s | 7 % | 93 % |

- Course: clean for six minutes. The GPU pool fills at minute 6 with 31 users in session (FreeInference: about 47,
  minute 8) and requests queue from there; the user cap binds by minute 30. Minutes 30 to 118: a mean of 19 requests
  run, 105 queue and 15 are sent a minute.
- The storm's cause is the one of campaign 28 (`checks/storm_cause_hicache.out`): of the 642 returning requests sent
  in minutes 60 to 110, 595 got nothing from the cache, 576 of them after using up their 8 SSD lookups at arrival;
  92 % of the history was computed again. 1,816 requests of the run reached the lookup cap.
- GPU time (`checks/gpu_work.out`, minutes 30 to 110): 29.3 M tokens of history computed again, 3,220 s at 0.11 ms a
  token, 67 % of the window; 100 output tokens a second; 172 ms an output token.
- Where the TTFT goes, over the 1,680 returning requests (286.1 s): 264.7 s in the queue for other reasons, 17.9 s
  waiting for the request's SSD read and then the GPU, 0.2 s for the read alone, 3.3 s after admission.
- Figures: `results/compare_c29_designs/` (`c28_users/figs.sh campaign29`, its first run on this campaign).
- `oracle_v2` launched 17:03:35Z, same GPU unit. The weight helper read the weights back in 137 s (more had left
  the page cache during the storm arm); READY after 205 s. Pools as above, cleaner 90/80, next-turn prefetch on, read
  order `deadline` with pin fraction 0.3, write-through, eviction `next_use`; client `--hints oracle`.

**`oracle_v2` done 19:09:28Z: 98 s under original HiCache on the requests both sent, and still half of the prompt
tokens computed again: mean TTFT 227 s, 2,556 requests.** 2,611 requests in all, no errors, no cleaner pass, SSD 58 %
full at the end, no traceback. Per minute: `checks/by_minute_oracle_v2.out`.

| requests sent before 6,600 s | `hicache` | `oracle_v2` |
|---|---|---|
| users with a request; sessions ended | 292; 164 | 334; 206 |
| requests (a minute); output tokens/s | 2,081 (18.9); 132 | 2,556 (23.2); 173 |
| TTFT mean, p50, p90, p99 | 289.2, 377.9, 465.7, 534.8 s | 227.4, 245.9, 395.2, 429.0 s |
| first requests; returning | 226.7 s; 299.4 s | 177.4 s; 235.0 s |
| prompt tokens from GPU, host, SSD; recomputed | 8, 11, 4 %; 76 % | 7, 17, 28 %; 49 % |
| delivered / ideal | 0.293 | 0.620 |
| slot wait | mean 600 s | mean 537 s (221 s over the 292 users both admitted) |
| by harness | Claude Code 1,561 requests, 284.5 s; Codex 520, 303.4 s | Claude Code 1,899, 222.2 s; Codex 657, 242.5 s |

On the 2,081 requests of 292 users both arms sent: 289.2 to 190.8 s, -98.4 s [-106.2, -90.3], 87.6 % of them faster.

| sent in minutes | requests | TTFT mean | returning requests served from SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 332 | 8.4 s | 0 % | 22 % |
| 10-30 | 667 | 101.6 s | 53 % | 32 % |
| 30-60 | 727 | 267.3 s | 55 % | 57 % |
| 60-110 | 830 | 381.2 s | 45 % | 63 % |

- Course: the first six minutes are those of `hicache` (same arrivals; the pool fills at minute 6 with 31 users).
  SSD reads serve from minute 15, the user cap binds at minute 31. Minutes 30 to 110: a mean of 19 requests run,
  104 queue and 19.5 are sent a minute (`hicache` 15.2).
- **Why half is still computed again: the reads come back minutes before admission and the host pool does not keep
  them** (`checks/reread_fate_oracle_v2.out`, `checks/read_waits_oracle_v2.out`, `checks/missed_requests_oracle_v2.out`).
  Not the `hicache` storm: no request reached the 8-lookup cap at arrival (`checks/storm_cause_oracle_v2.out`).
  - A request waits about 260 s in the queue (median). The engine finds its history gone from GPU and host a median
    of 9 to 10 s after it arrives and asks for the read at once: under the deadline order a queued request's read has
    no start time (`_check_queued` skips the trigger test), only a place in the read thread's line.
  - The read thread is not backed up here (parked 3.4 s at the median, read 2.6 s), so the read lands a median of
    137 s before admission (p90 327 s).
  - A landed read stays only under a pin, and pins are capped at 0.3 of the host pool (488,294 tokens, about 14
    requests of 35K). A pin is held to admission, 143 s at the median: 960 reads got a pin, 2,295 found no room.
  - A read with no pin is gone again a median of 1.0 s after it lands (p90 5.5 s; 1,389 times). The engine re-reads
    twice and then stops (`ntp_reevict ... capped=1`).
  - Minutes 30 to 110, 1,392 returning requests: 696 got nothing, 686 got everything. Of the 696, 686 had a read
    land, none kept a pin and all had their re-reads stopped; of the 686 served, 608 held a pin.
- The same engine on FreeInference (campaign 28, `results/campaign28/checks/reread_fate_oracle_v2.out`): 4,787 reads
  pinned, 68 found no room, a pin held 2.6 s at the median, reads landing 2.6 s before admission. There the read
  thread was 96 % busy and a read stayed parked 54 s at the median in deadline order, so the thread's own backlog
  delayed each read to just before its admission. Here the thread is 81 % busy in minutes 30 to 110
  (`checks/read_thread.out`) and each request stops asking after two re-reads, so nothing delays the reads.
- GPU time (`checks/gpu_work.out`, minutes 30 to 110): 22.3 M tokens of history computed again (54 %), 2,453 s, 51 %
  of the window (`hicache` 67 %); 141 output tokens a second; 129 ms an output token. At a full pool the GPU spends a
  third of its time in decode and two thirds in prefill (`checks/decode_scale.out`; the step time follows campaign
  28's law, 37 to 39 ms at a full pool).
- Reads by start (`checks/read_starts.out`): 2,376 plans, 33 reads before arrival, 1,882 in the queue. Tool waits
  are 0.1 to 0.4 s at the median, so the history is still in memory when the hint comes and leaves while the
  request queues.
- Where the TTFT goes, over the 1,680 returning requests both arms sent (185.5 s): 96.4 s in the queue for other
  reasons, 86.2 s waiting for the request's SSD read and then the GPU, 0.7 s for the read alone, 2.2 s after
  admission.
- What the next arm tests: the just-in-time engine holds a queued request's read until it would land at admission
  (`read_not_before`, margin 2 s) and gates lookups at arrival. This arm's loss is the case the hold was built for;
  on FreeInference the read thread's backlog hid the difference between the two engines.
- Figures: `results/compare_c29_designs/`, redrawn with two arms.
- `jit_exact` launched 19:09:28Z, same GPU unit. Weights read back in 110 s; READY after 182 s. Pools as above,
  cleaner 90/80, next-turn prefetch on, read order `deadline`, pin fraction 0.3, write-through, eviction `next_use`
  (horizon 300 s), arrival gate on, arrival hold on, read timing `just_in_time`, reply key source `hint`; client
  `--hints oracle`, clock from about 19:14:20Z. First minute: 4 users, 23 requests, returning TTFT 0.28 s, no errors.
- GPU utilization in `jit_exact` at saturation (`checks/gpu_util_jit_exact.out`, 19:56Z, minute 42 of the client
  clock, 128 users in session, about 20 running and 100 queued): 100 % in 30 of 30 one-second samples, 354 to
  544 W, 124.6 GB of GPU memory in use. The sample campaign 28's record asked to repeat in a hinted arm.
- `jit_exact` at 2,520 s: 1,263 requests against 1,201 for `oracle_v2` and 955 for `hicache` at the same time;
  returning TTFT 160 to 200 s in minutes 31 to 39 (`oracle_v2` about 237 s), 61 to 94 % of returning requests
  served from SSD (`oracle_v2` 40 to 66 %). Early; the arm ends about 21:15Z.

**`jit_exact` done 21:15:03Z: 78 s under the old oracle on the requests both sent and a third more requests served:
mean TTFT 167 s, 3,390 requests, 26 % of prompt tokens computed.** 3,598 requests in all, no errors, no cleaner pass,
SSD 77 % full at the end (the store is wiped at each arm's start), one traceback at the stop (the read thread's
`data_ptr` one of campaign 28, at 21:14:50Z, after the client ended). Per minute: `checks/by_minute_jit_exact.out`.

| requests sent before 6,600 s | `hicache` | `oracle_v2` | `jit_exact` |
|---|---|---|---|
| users with a request; sessions ended | 292; 164 | 334; 206 | 424; 296 |
| requests (a minute); output tokens/s | 2,081 (18.9); 132 | 2,556 (23.2); 173 | 3,390 (30.8); 231 |
| TTFT mean, p50, p90, p99 | 289.2, 377.9, 465.7, 534.8 s | 227.4, 245.9, 395.2, 429.0 s | 167.4, 188.7, 252.7, 293.0 s |
| first requests; returning | 226.7 s; 299.4 s | 177.4 s; 235.0 s | 135.2 s; 172.0 s |
| prompt tokens from GPU, host, SSD; recomputed | 8, 11, 4 %; 76 % | 7, 17, 28 %; 49 % | 5, 16, 53 %; 26 % |
| delivered / ideal | 0.293 | 0.620 | 0.882 |
| slot wait over the 292 users all admitted | mean 600 s | 221 s | 114 s |
| session length of those users, mean | 1,607 s | 1,046 s | 878 s |
| by harness | Claude Code 1,561 requests, 284.5 s; Codex 520, 303.4 s | Claude Code 1,899, 222.2 s; Codex 657, 242.5 s | Claude Code 2,506, 165.2 s; Codex 884, 173.6 s |

Pairs on the requests both arms sent: `oracle_v2` to `jit_exact`, 2,556 requests of 334 users, 227.4 to 149.8 s,
-77.7 s [-84.7, -70.4], 77.5 % of them faster; `hicache` to `jit_exact`, 2,081 requests of 292 users, 289.2 to
132.3 s, -156.9 s [-169.3, -144.0], 88.4 % faster. On the 2,081 requests all three sent: 289.2, 190.8 and 132.3 s.

| sent in minutes | requests | TTFT mean | returning requests served from SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 333 | 8.4 s | 0 % | 22 % |
| 10-30 | 643 | 106.9 s | 48 % | 33 % |
| 30-60 | 1,026 | 180.1 s | 87 % | 26 % |
| 60-110 | 1,388 | 224.3 s | 86 % | 24 % |

- Course: the first six minutes are those of the other arms. SSD reads serve from minute 15, the user cap binds at
  minute 31. Minutes 30 to 110: a mean of 18 requests run, 103 queue and 30.2 are sent a minute (`oracle_v2` 19.5,
  `hicache` 15.2). The queue is as long in every arm (the user cap sets it), so the TTFT follows the service rate:
  the gain is 55 % more requests a minute than the old oracle, seen through the same queue.
- **The hold removes the old oracle's loss** (`checks/reread_fate_jit_exact.out`, `checks/read_waits_jit_exact.out`).
  - A read is parked with a start time, 150 s at the median, and lands 3.1 s before admission (p90 8.6 s); the old
    oracle's landed 137 s before (p90 327 s).
  - A pin is held 3.0 s at the median against 143 s. All 2,314 reads got a pin and none found the budget full: it
    is 9 % used at the median and 39 % at most, against 96 % at the median for the old oracle (`srs_pin pinned=`).
  - One read a request (1.00) against 1.5 to 2.1; 60.0 M tokens read from SSD against 83.2 M, for a third more
    requests. The read thread is 45 % busy in minutes 30 to 110 against 81 % (`checks/read_thread.out`).
  - Minutes 30 to 110, 2,148 returning requests: 1,877 got everything, each from one pinned read; 271 got nothing.
- **What is still computed again is the engine's three-lookup cap, spent on text the next prompt does not share**
  (`checks/cap_share_jit_exact.out`, `checks/storm_cause_jit_exact.out`, `checks/missed_requests_jit_exact.out`).
  - 333 of the 2,924 returning requests sent before 6,600 s were refused a read at the cap. They carry 99.4 % of
    the 10.17 M history tokens computed again (11.8 % of the history). Campaign 28 had 97 such requests of 8,048.
  - What the lookups are for. A turn's chain is the previous prompt plus the previous reply. The replay sends the
    reply back as new synthetic ids, so the tree matches at most through the previous prompt and the reply's length
    is left over (c216: prompt 39,389 + reply 250, chain 39,616, matched 39,360); a prompt that keeps only part of
    the previous one leaves more (c104-t1: chain 32,192, matched 20,608). A leftover of 256 tokens or more passes
    the gate's threshold and the gate asks the SSD for text that is in no tier. Of the 355 capped requests of the
    run, 233 have a leftover within 128 tokens of the previous reply's length.
  - One lookup goes at arrival and two more about a minute into the queue wait (the third a median of 62 s after
    arrival), each finding nothing: all 355 spent every lookup this way. When the history then leaves memory (a
    median of 163 s after arrival) the gate answers `capped`. `jit_exact-c216-t7`: three lookups in the first
    second, history gone at 82 s, admitted at 212 s, 43,411 tokens computed in 21 s. A request whose history
    leaves before the minute check keeps a lookup, is read back and served (`jit_exact-c188-t12`).
  - So it needs a leftover of 256 tokens or more and a queue wait well over a minute: rare on FreeInference (queue
    wait 53 s), common here (190 s). My reading, not run: a client that sends real text matches through the reply,
    which would spare the two thirds whose leftover is the reply; a prompt that drops part of its history would
    still trigger it. Counting only lookups that find pages would close both: an engine change, not for this
    campaign. `jit_p50` and `ec_full` carry the same cap; compare their capped counts before reading a difference
    between the three (campaign 28's `ec_full` was capped a third as often).
  - The old oracle spends re-reads on the same leftover (`ntp_reevict` with only a tail missing, 1,493 times). Its
    loss is the pin budget all the same: a read with no pin is gone a second after it lands, so more re-reads
    would not keep it until an admission two minutes away.
- GPU time (`checks/gpu_work.out`, minutes 30 to 110): 8.4 M tokens of history computed again (12.8 %), 923 s, 19 %
  of the window (`oracle_v2` 51 %, `hicache` 67 %); 224 output tokens a second (141, 100); 80 ms an output token
  (129, 172). At a full pool decode takes 40 % of the GPU's time and prefill 60 % (`checks/decode_scale.out`).
- This box on TraceLab under the 128-user cap serves 30 requests a minute with a fifth of the GPU's time still
  lost at the cap; the paper model gave 35 a minute at a full pool (`checks/load_model.out`).
- Reads by start (`checks/read_starts.out`): 3,254 plans, 3 reads before arrival, 2,777 in the queue.
- By the wait before the request (`checks/hint_effect_oracle_v2_jit_exact.out`, 2,222 returning requests both
  sent): -80.7 s [-88.6, -73.5], between -74 and -96 s in every wait class; history computed again 38.0 % to 11.5 %.
- Where the TTFT goes, over the 1,680 returning requests all three arms sent (130.8 s): 26.3 s in the queue before
  the read is asked, 101.7 s from there to admission (the read is held parked for most of the queue wait, so this
  is queue time too), 1.4 s for the read alone, 1.3 s after admission.
- Hints: exact, 3,082 of 3,390 requests.
- Figures: `results/compare_c29_designs/`, redrawn with three arms. Its last panel counts requests for which a read
  was issued, so the old oracle shows near 100 % there against 45 to 55 % served from SSD in the table above.
- `jit_p50` launched 21:15:03Z, same GPU unit. Weights read back in 129 s; READY after 193 s. Pools and flags as
  `jit_exact` (arrival gate on, arrival hold on, read timing `just_in_time`, reply key source `hint`, soft pins and
  price learning off); client `--hints dist --hint-no-call`, clock from about 21:18:40Z.

**`jit_p50` done 23:20:39Z: the same as `jit_exact`: mean TTFT 164 s, 3,451 requests, 25 % of prompt tokens
computed. Table hints lose nothing to exact ones at this load.** 3,683 requests in all, no errors, no cleaner pass,
SSD 79 % full at the end, one traceback at the stop (the `data_ptr` one, at 23:20:28Z, after the client's last
record at 23:20:20Z). Per minute: `checks/by_minute_jit_p50.out`.

| requests sent before 6,600 s | `hicache` | `oracle_v2` | `jit_exact` | `jit_p50` |
|---|---|---|---|---|
| users with a request; sessions ended | 292; 164 | 334; 206 | 424; 296 | 427; 299 |
| requests (a minute); output tokens/s | 2,081 (18.9); 132 | 2,556 (23.2); 173 | 3,390 (30.8); 231 | 3,451 (31.4); 234 |
| TTFT mean, p50, p90, p99 | 289.2, 377.9, 465.7, 534.8 s | 227.4, 245.9, 395.2, 429.0 s | 167.4, 188.7, 252.7, 293.0 s | 164.2, 174.8, 252.9, 285.5 s |
| first requests; returning | 226.7 s; 299.4 s | 177.4 s; 235.0 s | 135.2 s; 172.0 s | 128.6 s; 169.2 s |
| prompt tokens from GPU, host, SSD; recomputed | 8, 11, 4 %; 76 % | 7, 17, 28 %; 49 % | 5, 16, 53 %; 26 % | 5, 17, 53 %; 25 % |
| delivered / ideal | 0.293 | 0.620 | 0.882 | 0.892 |
| slot wait over the 292 users all admitted | mean 600 s | 221 s | 114 s | 77 s |
| session length of those users, mean | 1,607 s | 1,046 s | 878 s | 819 s |
| TTFT mean on the 2,081 requests all four sent | 289.2 s | 190.8 s | 132.3 s | 123.8 s |
| by harness | Claude Code 1,561 requests, 284.5 s; Codex 520, 303.4 s | Claude Code 1,899, 222.2 s; Codex 657, 242.5 s | Claude Code 2,506, 165.2 s; Codex 884, 173.6 s | Claude Code 2,549, 161.5 s; Codex 902, 171.9 s |

Pairs on the requests both arms sent: `jit_exact` to `jit_p50`, 3,387 requests of 424 users, 167.4 to 163.7 s,
-3.7 s [-4.7, -2.7], 55.8 % of them faster; `oracle_v2` to `jit_p50`, 2,556 requests of 334 users, 227.4 to 144.7 s,
-82.7 s [-90.0, -75.5], 86.2 % faster; `hicache` to `jit_p50`, 2,081 requests of 292 users, 289.2 to 123.8 s,
-165.4 s [-178.3, -151.9], 88.8 % faster. (The script draws every pair from one seeded stream, so an earlier
pair's interval moves by a few tenths of a second when a pair is added: `oracle_v2` to `jit_exact` now prints
[-85.1, -70.2].)

| sent in minutes | requests | TTFT mean | returning requests served from SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 332 | 8.3 s | 0 % | 22 % |
| 10-30 | 685 (`jit_exact` 643) | 94.5 s (106.9) | 54 % (48) | 28 % (33) |
| 30-60 | 1,060 (1,026) | 177.3 s (180.1) | 88 % (87) | 25 % (26) |
| 60-110 | 1,374 (1,388) | 226.5 s (224.3) | 86 % (86) | 25 % (24) |

- **I read the 3.7 s as a tie, not as table hints beating exact ones.**
  - The two arms are the same through minute 17 and from minute 31 on. Minutes 30 to 110: 18 requests run and 102
    queue (18 and 103), 30.4 are sent a minute (30.2), 12.80 % of the history is computed again in both, 224
    output tokens a second in both (`checks/gpu_work.out`).
  - The gap opens in minutes 18 to 30, as the queue forms: returning TTFT peaks at 222 s in minute 26 for
    `jit_exact` and at 188 s in minute 25 for `jit_p50`.
  - It is the three-lookup cap's draw in that window. Sent in minutes 10 to 30: 62 requests capped and 1.76 M
    history tokens computed again in `jit_exact`, 45 and 1.03 M in `jit_p50`. After minute 30: 271 and 8.35 M
    against 268 and 8.40 M. The 0.73 M tokens are about 90 s of prefill that the queue of the time waits behind.
  - Which request is capped is not a property of the trace: of the 355 and 332 capped in the two runs, 111 are
    the same request. It depends on when the history leaves memory during that request's queue wait.
  - The interval is over the users of one run each. A run's users share one queue, so it understates the spread
    between runs; a rerun of either arm would tell.
- The cap, whole run (`checks/cap_share_jit_p50.out`, `checks/storm_cause_jit_p50_all.out`): 313 of the 2,982
  returning requests sent before 6,600 s were refused a read (`jit_exact` 333 of 2,924). They carry 99.3 % of the
  9.50 M history tokens computed again, 10.8 % of the history (11.8 %). So the two arms carry the same handicap
  and the comparison between them is fair.
- The hold works as in `jit_exact` (`checks/reread_fate_jit_p50.out`, `checks/read_waits_jit_p50.out`): a read is
  parked 154 s at the median and lands 3.1 s before admission (p90 8.5 s); a pin is held 2.9 s at the median; all
  2,415 reads got a pin, none found the budget full (11 % used at the median, 45 % at most); 1.00 read a request;
  62.3 M tokens read; the read thread is 46 % busy in minutes 30 to 110 (`checks/read_thread.out`).
- **Why a hint's precision buys nothing here** (`checks/hint_quality_jit_p50.out`, `checks/read_starts.out`).
  - The table is right at the median and far off for long waits: hinted p50 over true wait 1.02 at the median,
    96.5 % of hints find an entry, 63 % of true waits fall inside p10 to p90; waits of 60 to 300 s are hinted at
    8 s (true median 115 s) and waits of 300 s or more at 19 s (556 s).
  - A request waits 185 s in the queue at the median, so the server knows of it long before it can run and times
    the read from its queue position: 2,702 reads start in the queue, 19 before arrival (`jit_exact` 2,777 and 3).
  - 111 plans timed out before their request came (none with exact hints). The 102 of those requests sent before
    6,600 s (wait before them 176 s at the median) were read back all the same: 99.9 % of their history delivered.
  - By the wait before the request (`checks/hint_effect_jit_exact_jit_p50.out`, 2,963 returning requests both
    sent): -3.3 s [-4.3, -2.2]; -3.5, -1.3, -4.1, -4.0 and -8.9 s by class, no class where exact hints win.
  - The two clients tell the server the same about a session's end: the table client sends an end key (313
    hints), the exact client sends no hint after the last request, which the engine also reads as the end
    (`next_turn_prefetch.py`, "The session ended (or the client sends no hints): evict it first").
  - My reading, not run: at this load a just-in-time arm with no hints would land close to both (the sixth arm in
    the open list); hints get a test only where a request can run before its read is done, as in campaign 26.
- GPU time (`checks/gpu_work.out`, minutes 30 to 110): 8.45 M tokens of history computed again, 930 s, 19 % of the
  window; 82.6 ms an output token (`jit_exact` 80.0).
- Where the TTFT goes, over the 1,680 returning requests all four arms sent (123.1 s; `jit_exact` 130.8): 22.8 s in
  the queue before the read is asked, 97.8 s from there to admission, 1.1 s for the read alone, 1.4 s after admission.
- Hints: table, 3,138 of a wait and 313 of a session's end; 6.0 % of replies make no tool call (`--hint-no-call`).
- Figures: `results/compare_c29_designs/`, redrawn with four arms; the two just-in-time lines lie on each other.
- `ec_full` launched 23:20:39Z, same GPU unit. Weights read back in 107 s; READY after 166 s. Pools as the other
  arms (668,096 and 1,627,648 tokens); arrival gate and hold on, read timing `expected_cost`, memory price 5e-05,
  price learning `harm`, soft pins on, eviction `return_chance` with a 30 s horizon (the boot check's "want
  next_use" line is its generic text); client `--hints dist --hint-no-call`, clock from about 23:26:00Z. Ends about
  01:26Z. First two minutes as the other arms (23 and 12 requests, returning TTFT 0.3 s), `ec_price` events logged.

**`ec_full` done 01:25:51Z (2026-11-20), campaign 29 complete: the rules arm leads, 15.6 s under `jit_p50` on the
requests both sent: mean TTFT 151 s, 3,723 requests, 21 % of prompt tokens computed. The lead is the eviction order
acting through the lookup cap; the price, the read timing and the soft pins decided nothing.** 4,016 requests in
all, no errors, no cleaner pass, no traceback, SSD 85 % full at the end (5 points under the cleaner's mark). Per
minute: `checks/by_minute_ec_full.out`.

| requests sent before 6,600 s | `hicache` | `oracle_v2` | `jit_exact` | `jit_p50` | `ec_full` |
|---|---|---|---|---|---|
| users with a request; sessions ended | 292; 164 | 334; 206 | 424; 296 | 427; 299 | 453; 325 |
| requests (a minute) | 2,081 (18.9) | 2,556 (23.2) | 3,390 (30.8) | 3,451 (31.4) | 3,723 (33.8) |
| output tokens/s | 132 | 173 | 231 | 234 | 249 |
| TTFT mean | 289.2 s | 227.4 s | 167.4 s | 164.2 s | 150.7 s |
| TTFT p50, p90, p99 | 377.9, 465.7, 534.8 | 245.9, 395.2, 429.0 | 188.7, 252.7, 293.0 | 174.8, 252.9, 285.5 | 167.8, 227.1, 265.2 |
| first requests; returning | 226.7; 299.4 s | 177.4; 235.0 s | 135.2; 172.0 s | 128.6; 169.2 s | 112.4; 156.1 s |
| prompt tokens from GPU, host, SSD | 8, 11, 4 % | 7, 17, 28 % | 5, 16, 53 % | 5, 17, 53 % | 4, 16, 60 % |
| prompt tokens recomputed | 76 % | 49 % | 26 % | 25 % | 21 % |
| of them new text; history lost | 18; 58 % | 17; 32 % | 16; 10 % | 16; 9 % | 16; 5 % |
| delivered / ideal | 0.293 | 0.620 | 0.882 | 0.892 | 0.943 |
| refused a read at the cap (returning requests) | no gate | no gate | 333 of 2,924 | 313 of 2,982 | 175 of 3,228 |
| slot wait over the 292 users all admitted | mean 600 s | 221 s | 114 s | 77 s | 50 s |
| session length of those users, mean | 1,607 s | 1,046 s | 878 s | 819 s | 768 s |
| TTFT mean on the 2,081 requests all five sent | 289.2 s | 190.8 s | 132.3 s | 123.8 s | 113.9 s |
| Claude Code: requests, TTFT | 1,561, 284.5 s | 1,899, 222.2 s | 2,506, 165.2 s | 2,549, 161.5 s | 2,753, 148.6 s |
| Codex: requests, TTFT | 520, 303.4 s | 657, 242.5 s | 884, 173.6 s | 902, 171.9 s | 970, 156.9 s |

Pairs on the requests both arms sent (final run of `compare_arms.py`, `compare_arms.out`):

| A to B | requests, users | A, B mean | B - A [95 %] | B faster |
|---|---|---|---|---|
| `hicache` to `oracle_v2` | 2,081, 292 | 289.2, 190.8 s | -98.4 [-106.2, -90.3] | 87.6 % |
| `hicache` to `jit_exact` | 2,081, 292 | 289.2, 132.3 s | -156.9 [-169.3, -144.0] | 88.4 % |
| `hicache` to `jit_p50` | 2,081, 292 | 289.2, 123.8 s | -165.4 [-178.3, -151.9] | 88.8 % |
| `hicache` to `ec_full` | 2,081, 292 | 289.2, 113.9 s | -175.3 [-189.6, -160.7] | 88.8 % |
| `oracle_v2` to `jit_exact` | 2,556, 334 | 227.4, 149.8 s | -77.7 [-85.1, -70.4] | 77.5 % |
| `oracle_v2` to `jit_p50` | 2,556, 334 | 227.4, 144.7 s | -82.7 [-89.4, -75.7] | 86.2 % |
| `oracle_v2` to `ec_full` | 2,556, 334 | 227.4, 131.1 s | -96.3 [-104.6, -88.3] | 86.7 % |
| `jit_exact` to `jit_p50` | 3,387, 424 | 167.4, 163.7 s | -3.7 [-4.8, -2.6] | 55.8 % |
| `jit_exact` to `ec_full` | 3,390, 424 | 167.4, 148.1 s | -19.3 [-20.5, -18.0] | 72.3 % |
| `jit_p50` to `ec_full` | 3,451, 427 | 164.2, 148.6 s | -15.6 [-17.0, -14.3] | 68.9 % |

| sent in minutes | requests | TTFT mean | returning requests served from SSD | prompt tokens recomputed |
|---|---|---|---|---|
| 0-10 | 333 | 8.4 s | 0 % | 22 % |
| 10-30 | 706 (`jit_p50` 685) | 89.5 s (94.5) | 52 % (54) | 24 % (28) |
| 30-60 | 1,106 (1,060) | 165.8 s (177.3) | 90 % (88) | 21 % (25) |
| 60-110 | 1,578 (1,374) | 197.6 s (226.5) | 91 % (86) | 19 % (25) |

- Course: as the other arms to minute 10. Minutes 30 to 110: 18 requests run, 102 queue, 33.5 are sent a minute
  (`jit_p50` 30.4, `jit_exact` 30.2). Same queue, a tenth more service: the gain holds to the end of the run.
- **The whole lead is history not computed again** (`checks/gpu_work.out`, minutes 30 to 110): 5.08 M tokens
  (6.9 %) against 8.45 M (12.8 %), 559 s of GPU time against 930 s, 11.6 % of the window against 19.4 %. Moving 7.7
  points of GPU time from computing history again to new work gives +9.7 % requests; the arm sends +10.2 %.
- **And the history computed again is the cap** (`checks/cap_share_ec_full.out`): 175 of 3,228 returning requests
  refused a read, against 313 of 2,982 and 333 of 2,924; they carry 98.7 % of the 5.46 M history tokens computed
  again (5.7 % of the history against 10.8 and 11.8 %).
- **The four rules decided nothing, as on FreeInference** (`checks/rules_activity_ec_full.out`): the price stayed
  at 5e-05 through 1,374 updates, no room was refused, no read started early; 765 of 774 timing decisions were
  "leave the read to the arrival" (queue wait estimate 168 s at the median against a 5.6 s read) and 9 "wait" at the
  run's end; no release, no bump.
- **Why it is refused half as often: `return_chance` ranks a queued turn last, `next_use` ranks it as overdue
  after 60 s** (`checks/cap_path.out`, `mem_cache/next_use_eviction.py`).
  - At arrival the prefetcher registers the session's tail with next use = now. `NextUseStrategy` puts a
    registration more than `overdue_grace_s` = 60 s in the past in `TIER_OVERDUE` ("the turn is not coming"),
    evicted right after ended sessions. The grace was set to the plan's minimum expiry, for queue waits under a
    minute. `ReturnChanceStrategy` puts a session registered with a time in `RETURN_TIER_QUEUED`, evicted last.
  - So in the two `next_use` arms a turn still queued at 60 s loses its tail (`jit_p50-c58-t11`: `ntp_due` 61.0 s
    after arrival, `shrink=2496`). The plan sees pages missing and asks twice more; each lookup asks the SSD for
    the new prompt's own text beyond its match (`tokens_req=3008 storage_hit=0`), and the allowance is gone at
    62 s. The history leaves memory at 186 s, the gate answers `capped`, the request is admitted at 187 s.
  - Requests whose allowance went on empty lookups 30 s or more before admission: 369 in `jit_exact` and 329 in
    `jit_p50`, 291 and 252 of them marked 55 to 70 s after arrival (275 and 243 then refused). `ec_full`: 236, of
    them 29 in that window (14 refused). It has more in the first 5 s (137 against 59 and 62; 129 refused), where
    the leftover is already 256 tokens or more at arrival.
  - On FreeInference the queue wait is 54 s, under the grace, so that window is nearly empty (4, 4 and 1 requests,
    `results/campaign28/checks/cap_path.out`); its `ec_full` was refused less (26 of 84 at risk against 46 of 53
    and 48 of 52) because its history stayed in host memory to admission more often.
- So what the arm shows is one part of rule 3: an eviction order that knows a turn is queued. Its size here is
  inflated by the cap: a lost history costs a recompute of about 30,000 tokens where a read would cost about 2 s.
  My reading, not run: with the cap counting only lookups that find pages the three arms on the just-in-time
  engine would be within a few seconds of each other on this trace.
- Two engine defects to fix before a follow-up (engine changes, not made): the cap counts lookups that find
  nothing (`MAX_REFETCHES`, `prefetch_gate`); `next_use` ranks a turn that waits over 60 s in the queue as overdue.
  The old oracle arm runs `next_use` too, so the second one adds to its loss here (a landed read's session is
  ranked overdue, which fits its unpinned reads being gone a second after landing).
- Reads (`checks/reread_fate_ec_full.out`, `checks/read_waits_ec_full.out`, `checks/read_thread.out`): parked 130 s
  at the median, landing 2.7 s before admission (p90 8.3 s), pin held 2.6 s; all 2,725 reads got a pin; 1.00 read
  a request; 75.2 M tokens read (`jit_p50` 62.3 M); read thread 59 % busy in minutes 30 to 110 (46 %).
- By the wait before the request (`checks/hint_effect_jit_p50_ec_full.out`, 3,024 returning requests both sent):
  -15.5 s [-16.9, -14.2]; -15.1, -17.2, -15.6, -14.6 and -19.8 s by class. The same gain after every wait: a
  faster queue, not a better-timed read.
- Where the TTFT goes, over the 1,680 returning requests all five arms sent (113.6 s): 19.5 s in the queue before
  the read is asked, 91.4 s from there to admission, 1.6 s for the read alone, 1.1 s after admission.
- Hints: table, 3,387 of a wait and 336 of a session's end; 96.7 % with an entry; 64 % of true waits inside p10 to
  p90; waits of 60 to 300 s hinted at 10 s, of 300 s or more at 19 s.
- SSD: 85 % at the end against 77 to 79 % for the two other arms on this engine, since more requests wrote more KV.
  A faster arm or a longer run at this setting reaches the cleaner's 90 % mark; a follow-up on TraceLab should
  plan for it.
- Figures: `results/compare_c29_designs/`, five arms.
- The launcher is idle after the last arm ("queues done, waiting for more lines or END"); the server is stopped.

## Both campaigns side by side

Same setting (0.1 users/s, cap 128, 160 GB, x1, 2 h, one run per arm), requests sent before 6,600 s.

| | FreeInference (campaign 28) | TraceLab (campaign 29) |
|---|---|---|
| prompt of a returning request, mean; new text a turn | 24,500; 1,180 tokens | 32,400; 3,190 tokens |
| requests a user in the window | 21 | 8 |
| running, queued (hinted arms, minutes 30-110) | 22, 77 | 18, 102 |
| queue wait, just-in-time arms, median | 54 s | 171 to 193 s |
| requests a minute: `hicache`, `oracle_v2`, `jit_exact`, `jit_p50`, `ec_full` | 41.3, 79.4, 76.9, 77.8, 78.9 | 18.9, 23.2, 30.8, 31.4, 33.8 |
| TTFT mean, same order | 115.7, 47.7, 49.8, 49.4, 48.3 s | 289.2, 227.4, 167.4, 164.2, 150.7 s |
| prompt tokens recomputed | 45, 7, 8, 8, 7 % | 76, 49, 26, 25, 21 % |
| of them new text | 7 to 8 % | 16 to 18 % |
| history lost, of the history | 40, 0.3, 1.4, 1.2, 0.4 % | 71, 38, 11.8, 10.8, 5.7 % |
| refused at the cap, the three gate arms | 97, 93, 33 | 333, 313, 175 |
| `hicache` to `oracle_v2` | -81.3 s [-89.7, -73.4] | -98.4 s [-106.2, -90.3] |
| `oracle_v2` to `jit_exact` | +3.0 s [+2.6, +3.5] | -77.7 s [-85.1, -70.4] |
| `jit_exact` to `jit_p50` | -0.9 s [-1.4, -0.4] | -3.7 s [-4.8, -2.6] |
| `jit_p50` to `ec_full` | -1.7 s [-2.1, -1.3] | -15.6 s [-17.0, -14.3] |

What the two say together:

1. Every next-turn design beats original HiCache at saturation, on both traces: 80 s on FreeInference, 98 to 175 s
   on TraceLab. Original HiCache spends its lookups at arrival and loses the history during the queue wait.
2. The hold is what long queues need. With a 54 s queue the old oracle and just-in-time tie; with a queue of three
   minutes the old oracle's reads land two minutes early and find no room to stay pinned, and just-in-time is 78 s
   ahead with a third more requests served.
3. Exact and table hints tie on both traces. At saturation a request is queued long before it can run, so the
   server times its read from the queue and a hint's precision has nothing to add.
4. The rules arm ties `jit_p50` on FreeInference and leads by 16 s on TraceLab. On both the price, the timing rule
   and the soft pins made no decision; the lead is the eviction order's handling of queued turns, through the cap.
5. Neither campaign tests what a hint or a price is worth: both run saturated (the GPU pool is full with about 20
   requests running and 77 to 102 queued). That needs a setting under saturation with idle KV past the host pool
   ("Discussion with the author" in campaign 28's record).
6. Found on the way, in the engine: the three-lookup cap that counts empty lookups, and the `next_use` overdue
   rank of queued turns. Both inflate the differences between the hinted arms on TraceLab.

One run per arm; intervals are over users within a run and leave out the spread between runs.
