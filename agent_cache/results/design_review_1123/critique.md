## verdicts
- [plausible] (user design note) The gain comes from both pre-arrival reads and the queue-delay-based read schedule, so both timings belong in the fetch decision: Both effects were measured, but at very different sizes. At 44 users / 70 GB on Triton (campaign 26), the no-hint queue-aware engine reached 0.71-0.72 s against HiCache's 5.47-6.62 s, and exact hints added only 0.14-0.21 s (agent_cache/results/campaign26/DECISIONS.md:338-346, 632-633). On FA3 with the fixes, ec_full staged 1 read before arrival and tied jit_exact, which staged 99 (-0.06 s [-0.19, +0.06], campaign34/DECISIONS.md:147). Reads timed on p50 hints cost 0.6-0.9 s (campaign26/DECISIONS.md:895-937; campaign34/DECISIONS.md:143-155). One rule can carry both timings, with the pre-arrival case as its light-load corner.
- [contradicted] (user design note) Model the queue wait as prefill work ahead, because position x rate lags a growing queue (sections 2 and 5): Binned by its own prediction, the engine's position x rate (ntp_admit pred_wait_ql_ms) is unbiased for turns without their own read. The median error is -0.5 to +0.2 s up to 20 s predicted in all three campaign 34 arms, and -0.2 s overall in campaign 32 ec_full through the loop. The errors are elsewhere. Readers wait a median 8.4-27.3 s longer than predicted (the read line). Non-readers wait 7-20 s less at 20-60 s predicted, because the readers ahead of them are passed over. For readers, max(arrival + position x rate, own last landing) gives a median error of +0.4-0.5 s (|err| p90 3.0-4.0 s), against a p90 of 24-52 s for position x rate alone. A prefill sum models neither effect, and admission is bound by the KV pool at 19-23 running requests. Files: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err_c34.out, /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err_c32_ecfull.out, /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/max_structure.out.
- [contradicted] (user design note) Regime table: campaign 32 minutes 30-70 is the 'W < r' regime: W was 5.8-7.1 s against 0.8-1.1 s to read 20K tokens uncontended (agent_cache/results/campaign32/checks/mechanisms_findings.txt:178; campaign32/DECISIONS.md:253-254). At ec_full's 'wait' decisions the engine's own estimates were about equal (3.81 against 4.48 s).
- [unsupported] (user design note) 'Nov 20 trace study' (human turns 76-95 % of resume read time), '974 of 2,065 and 1,209 of 2,139', '+0.003 s' for the index: A grep of agent_cache/ (data/ and frozen/ excluded) and of the memory folder finds these numbers only in the design note itself. The records give 979 of 2,009 and 1,144 of 2,032 (campaign25 compare_h_rep.out); 'about half' still holds. Rule 3 alone is +0.004 s [-0.11, +0.12] (campaign26/DECISIONS.md:967-969).
- [contradicted] (user design note) Campaign 32's first refused reads came while pending writes held up to 166K tokens: 166K is jit_p50's window peak for minutes 30-60 (agent_cache/results/campaign34/DECISIONS.md:79). At the first refusals of the early-reading arms, pins plus reads in flight held 207-393K, against 33-68K of pending writes (campaign32/checks/mechanisms_findings.txt, early-onset entries).
- [contradicted] (user design note) Class 4 (staged reads) is evicted before class 5 (queued), and 'a page cannot leave host memory until it is on the SSD': As a description of the engine, both are wrong. A read short of room first evicts every evictable leaf, queued-tier prefixes of other requests included, and only then bumps staged pins (python/sglang/srt/mem_cache/unified_radix_cache.py:3039-3047). _drain_backup releases the write lock at the ack without checking completed_tokens (3511-3515). A refused GPU->host copy never reaches the SSD, and the tree never learns of the cleaner's deletions. Class 5 has never been run as a policy.
- [supported] (scheduling) Admission passes over a turn whose read is under way, so a late read only reorders admissions while ready work is queued (lateness costs no GPU capacity in campaign 34): The skip is python/sglang/srt/managers/scheduler.py:4006-4012. I re-ran pool_idle.awk on c34 ec_full and jit_p50. From minute 30 the queue was non-empty on 93-100 % of batch lines; on those lines pool use was below 0.90 only 0.4-1.3 % of the time, with 2.5-3.1 % unused on average. Token usage is (pool - free - evictable)/pool (managers/scheduler_components/pool_stats_observer.py:221-224). Caveat: in minutes 0-30, 50-53 % of the lines with a queue had the pool under 0.90, so lateness had a cost there. File: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/pool_idle_c34.out.
- [contradicted] (scheduling) W should count only ready requests ahead; near admission this cuts the p90 error from 9-15 s to 2-5 s: The horizon metric conditions on the time left to admission. Predicting zero scores a p90 |err| of 2.0, 5.0 and 10.0 s at the 2, 5 and 10 s horizons; the ready-only count scores 1.8, 4.4 and 9.4 s (c34 ec_full, minutes 30-110). At arrival, where releases are decided, the ready-only count under-predicts: a median of +1.3 s for non-readers and +12.1 s for readers. Its |err| p90 is 10.5 and 29.1 s in the 5-10 and 10-20 s prediction bins, against 4.2 and 6.9 s for position x rate. The proposal's own toy agrees: the hold with this W (holdr) gave 87.8 against 46.3 s with tight memory and 28.8 against 27.2 s with ample memory. File: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/w_est2_zero_ecfull.out.
- [supported] (scheduling) Once the read thread saturates, the pre-arrival read time is 4-6x short (planned 8.3 s against 30-52 s actual): I re-ran staged_reads.py on c34 jit_p50. Planned r had a p50 of 8.3 s; stage to landing took 30.3 s (minutes 60-90) and 52.5 s (90+), and reads landed a median 15.9 and 56.1 s after the turn arrived. The planner's backlog counts only dispatched pages (python/sglang/srt/mem_cache/unified_radix_cache.py:2334-2337). File: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/staged_reads_c34_jitp50.out.
- [supported] (scheduling) Turns with their own SSD read waited a median 6.2, 11.8 and 20.6 s longer than their queue position predicted: Against the engine's logged prediction the medians are +8.4 s (ec_full), +12.7 s (jit_exact) and +27.3 s (jit_p50) in minutes 30-110. Admission follows max(GPU turn, own landing) + 0.4 s (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err_c34.out, /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/max_structure.out).
- [plausible] (scheduling) Toy results: release rules tie when memory is ample (27.2-27.4 s); pull K=2 beats the hold by 16 s when it is tight; light load ranks exact early reads 0.12 s, hold 0.26 s, p10 reads 1.65 s: These are toy results only.
- The 16 s is pull2 against the hold (30.05 vs 46.34 s). The proposed unified rule (uni_ec) gave 35.32 s in the current code. An earlier code version (set2, no longer reproducible from the file) gave 91.42 s.
- The toy locks queued histories, and a refusal recomputes the whole history. The engine evicts queued-tier prefixes before refusing (python/sglang/srt/mem_cache/unified_radix_cache.py:3039-3047).
- There is no GPU mirror and no write path.
- ec_dist matches pre_exact to every printed digit in both saturated regimes. Its 1 s gap to the hold is likely the tick's second dispatch() call, not early reads (inferred from the code). The seed sd is 2.1 s.
- The toy's 2.33 req/s is compared with c34's whole-run 2.36 req/s. The late phase, whose 27 s TTFT the toy matches, ran at 2.16-2.22 req/s (campaign34/checks/by_minute_*.out, minutes 90-109).
- [supported] (scheduling) A prefill work-ahead sum is the wrong W model; tool gaps need no projection of W: Admission is bound by the KV pool (19-23 running at 0.97 use), and decode takes 61-75 % of GPU time (agent_cache/results/campaign34/checks/decode_scale.out). Position x rate is already unbiased for non-readers, as in the design-note verdict above. I did not re-check the drift figure (0.9 s over 1 s).
- [unsupported] (scheduling) Replace the time hold with release by ready position (pull, small K): Nothing in the engine logs supports it. Under saturation the hold barely binds: ec_full's first holds had a median of 3.8 s, and the last landing came a median 0.8 s before admission (agent_cache/results/campaign34/checks/read_waits_ec_full.out). Pull moves more waiting onto readers, who already wait 27-55 s longer in minutes 90-110. Its gain needs tight memory, and the engine's queued-tier eviction makes tight memory less harsh than in the toy. Keep it as a simulator arm.
- [supported] (scheduling) Generated outputs are never re-sent, so about 20 % of new KV is dead: The storage designer measured 15 %: 54 of 16,400 next prompts reuse the previous output, and outputs are 4.78 M of the 31.6-31.9 M tokens submitted per run (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-storage/out/client_writes_ec_full.out). Prompts are rebuilt from block ids (agent_cache/scripts/replay_programs.py:7-9). This is a replay artifact; real harnesses re-send outputs.
- [supported] (economics) Closed-loop identity: campaign 32's loop added 0.555-0.565 s of GPU per request, recompute explains 0.56-0.59 s, and 73 % of the longer cycle landed in TTFT: I recomputed the inputs from campaign{32,34}/checks/by_minute_{ec_full,jit_exact}.out for minutes 90-109: 128 users in session in all four runs, 58.9/59.1 against 129.4/133.4 requests sent a minute, TTFT 80.2/80.0 against 28.2/27.0 s. N x g = R + Z is Little's law. The substantive check is that Delta g matches recompute GPU time, in one window and two arms. The 27 % that did not land in TTFT is not shown to be decode.
- [plausible] (economics) Once the queue persists, one GPU-second costs about N = 128 TTFT-seconds, not 1 + Q = 77: The toy's 'exactly N' is the median of 40 injections; the means range from -7,204 to +3,322 (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/mu_check_d001.out). The campaigns give 52.0/0.555 = 94 TTFT-s per GPU-s (0.73 N), against 1 + Q = 61-78. Both are much larger than 1, so read-or-recompute and protection decisions come out the same.
- [supported] (economics) Readers (24-28 %) waited 45.6-52.2 s against 18.2-21.7 s and were admitted 0.6-0.9 s after their last landing, so the read line timed them: This holds for ec_full and jit_exact (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/read_vs_queue_c34.out). jit_p50 differs: its readers waited 68-69 s and were admitted 20-26 s after the last landing on average. Its early landings then waited for their GPU turn, which still fits max(GPU turn, landing).
- [contradicted] (economics) kappa (SSD-time price) is the number of requests bound by the read line, while an exposed read costs only its owner: This conflicts with the proposal's own identity and pass-over argument. After the cap, with the pool 97 % used, read-line delay permutes admissions and leaves X, and hence mean TTFT, unchanged. In c34, jit_p50's readers waited 68-69 s against ec_full's 46-48 s, while its non-readers waited 10-16 s against 14-22 s (read_vs_queue_c34.out). kappa has mean-TTFT weight only when the read line starves the GPU.
- [contradicted] (economics) W must be projected to the arrival time, because today's estimate lags a growing queue (19.8 s short for 30-60 s waits): That figure bins the error by the measured wait, which selects on the outcome. Binned by the prediction, non-readers are unbiased up to 20 s and over-predicted at 20-60 s. In c34 jit_exact, 832 of the 900 requests with 30-60 s waits were readers, and their extra wait is the read line. In campaign 32 ec_full's loop, the non-readers' median error was -0.2 s (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err_c34.out, /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err_c32_ecfull.out). Projecting W over a tool gap (p50 1.0 s) changes little.
- [supported] (economics) Choose the start by an argmin with recourse over priced cost, not by the threshold F(x|t) >= q: read_start already minimises cost (python/sglang/srt/mem_cache/next_turn_policy.py:165-201). On the tool entry the threshold says read now, which costs 1.81 s against 1.10 s for reading at arrival. Both FreeInference entries have a falling hazard: sigma_high 1.79 and 1.86 (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/facts-fetch-timing/sim/check_read_start.py).
- [unsupported] (economics) Exclusive tiering should gain more than the 0.6-0.7 s that separates campaign 34's arms: Under the proposal's own identity, mean TTFT after the cap moves only with X, and c34 recomputes about 0.5 % of history. A gain is likelier before the cap and where host room is small: at campaign 26's 70 GB, the room beyond the GPU copy is 44K tokens (agent_cache/results/campaign26/DECISIONS.md:873), and 30 GB more bought 0.52 s (1201-1206).
- [unsupported] (economics) Computed lambda of about 2e-5 late in campaign 34, 1e-6 early and 4e-5 in campaign 32's loop: The proposal marks these as inferred and shows no computation. The engine's harm price never moved on FA3: it stayed at 5e-5 in c32/c34 ec_full and sat at the 1e-7 floor in c32 ec_exact (fetch fact-finder).
- [plausible] (economics) The cleaner's recompute costs about 1.6-1.9 s of TTFT through the identity: The arithmetic holds: 1.14-1.31 M tokens at 0.043 ms over 2,847 requests (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/identity_check.out). It is not observed, though. The late TTFT rise starts at minute 85-86, before the first pass, and head-missing requests were no slower than their minute's mean (evidence fact-finder; campaign34/checks/by_minute_*.out).
- [supported] (economics) Late in campaign 34 the mandatory flows fill the device (4.3K tokens/s of writes plus 17.7K of reads, against 21.4K): iostat samples at >= 98 % utilisation total 2.10-2.11 GB/s (agent_cache/results/campaign34/iostat_vdc.log, per the calibration fact-finder). The read thread ran at 17.4-18.0K tokens/s while 100 % busy (campaign34/checks/read_thread.out).
- [plausible] (economics) Ration by delay, not refusal; campaign 32's loop came from unpriced rewrites plus rationing by refusal: It fits campaign 32 (74-93 % rewrites, refusals) against campaign 34 (no refusals). But all four fixes changed at once and fix 4 never acted (agent_cache/results/campaign34/DECISIONS.md:89-97), so no mechanism is shown to be necessary.
- [contradicted] (economics) Reading a 25K history is close to a tie with recomputing it at mu = 1 + Q: This counts the owner's 28 s stall at weight 1, which the proposal's own pass-over argument redistributes at saturation. Reading wins by far more, consistent with campaign 34's arms, which never chose to recompute.
- [supported] (storage) Pending writes do not fill host memory; the write-through mirror (0.64-0.65 M) equals the running requests' KV from minute 30: mirror_vs_running_ec_full.out gives a mirror of 637-656K against 647-653K of running requests' KV. The census shows pending writes at a median 0-7K (agent_cache/results/campaign34/DECISIONS.md:98-102). Host eviction skips the mirror (python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:2145-2146), and duplicate reclaim exists only under write_back (1748-1752).
- [supported] (storage) From minute 70-80, queued histories (0.9 M, rising to 1.6-1.7 M) exceed the 0.90-0.92 M of host-only room: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-storage/out/mirror_vs_running_ec_full.out: 930K in minutes 70-80 and 1,628-1,646K in 90-110, against 902-923K of host-only room.
- [plausible] (storage) Open-loop replay: the mirror is 56-61 % of SSD read demand, and oracle next use with ends saves 22-25 %: The replay gives ec_full 63.3 -> 24.6 M and jit_p50 56.6 -> 24.7 M, with oracle orders at 49.5 and 42.2 M (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-storage/out/host_room_replay_ec_full.out). Write-through LRU reproduces ec_full's 63.4 M delivered but falls 16 % short of jit_p50's 67.3 M. The replay is open-loop and session-level, and gives no TTFT.
- [supported] (storage) Readers waited 46-69 s against 12-19 s; jit_p50's early reads raised SSD volume to 82.5 M against 54-64 M and saturated the read line 20 minutes earlier: From read_vs_noread_*.out. Tokens read in minutes 30-110 were 58.0 M against 43.6 and 36.8 M (agent_cache/results/campaign34/DECISIONS.md:83). The read thread was 100 % busy from minute 70, against 90 for the other arms (campaign34/checks/read_thread.out).
- [supported] (storage) Keep eager write-through: 56-67 % of prompt positions are read back from SSD, 4.2-4.5 times each: client_writes_*.out gives 61.7 % and 66.8 % of positions, read 4.50x and 4.44x. Campaign 32's dirty backlog (515-675K tokens pending) shows what deferring writes risks (agent_cache/results/campaign34/DECISIONS.md:79).
- [supported] (storage) The cleaner's losses hit shared prefix heads that an older session wrote once and later sessions used from memory; touch-on-read would not protect them: All 48 zero-head requests after the first pass had a shared head. In my control, page-0 history lookups after the pass that found something had a shared head 65-66 % of the time (279/420, 174/269, 297/458); against that base rate, 48 of 48 has p of about 1e-9. Deletion by oldest mtime (python/sglang/srt/mem_cache/storage/nixl/nixl_cleaner.py:181-184, 270) and a lookup that stops at the first missing chained key (hicache_nixl.py:658-664) give the mechanism. File: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/head_control_c34.out.
- [plausible] (storage) 46 of the 48 sessions were younger than the window the passes deleted: head_missing.py tests only session age < first pass - 75 min, a first write after minute 13, which suits pass 1. Passes 2-4 delete younger pages, and the p90 session ages are 45-73 min.
- [plausible] (storage) Backward latest-start scheduling over all reads should replace the per-read hold: It is sound when the read thread is underloaded, but expected value is small. At saturation the latest starts are already past, as the proposal itself says. Below saturation the hold already lands reads a median 0.8 s before admission. Untested.
- [supported] (storage) After the cap binds, storage changes mostly move waiting between readers and non-readers: The pool is at least 0.97 used whenever requests are queued (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/pool_idle_c34.out). Campaign 34's arms tie on the mean (10.92-11.47 s) while readers' waits differ by about 20 s between arms.
- [supported] (all three proposals) Merge Preempt and Retain into one ranking, and expect little TTFT from the ranking itself: There were 0 ec_bump events in campaigns 32 and 34 (evict fact-finder). Campaign 26 had 462 bumps with no TTFT gain (agent_cache/results/campaign26/DECISIONS.md:854-870). Wherever eviction order was isolated, its effect was at most 0.1 s (campaign26/DECISIONS.md:965-989; campaign32/DECISIONS.md:59-64).
- [contradicted] (paper draft) eq. fetch: start when F(x|t) >= q, because F rises as the session stays idle: The engine minimises cost over the support (python/sglang/srt/mem_cache/next_turn_policy.py:165-201), and both of the trace's gap entries have falling hazards.
- [contradicted] (paper draft) A read started at arrival completes before the head of the queue, and r = bL/beta: In campaign 34, reads due at once sat parked a median 30 s, and late in the run readers waited 27-55 s longer than non-readers. Most of r is the read line, not the transfer.
- [contradicted] (paper draft) Storage holds every prefix cache (design.tex:99): From minute 88 the cleaner deletes about 2.5 M tokens per pass. After it, 65 requests found nothing and 18 found only part of their history, and 31 reads failed on deleted files (evidence fact-finder, campaign 34 server logs).

## synthesis
Repo paths are relative to /home/wanhr/sglang.

FETCH RULE
Treat each needed history as one read job. One function times it at both seams: _check_pending before arrival, and read_not_before / read_deadline after arrival.

Deadline and landing:
- The job's deadline D is when admission would take the turn if its history were resident. Its landing C comes from the read line.
- The structure is measured. For readers in campaign 34, max(arrival + position x rate, own last landing) predicts the wait to a median +0.4-0.5 s (|err| p90 3.0-4.0 s). Position x rate alone gives +8 to +27 s (p90 24-52 s). In campaign 32 ec_full: +0.4 s (p90 8.3 s) against +2.4 s (p90 14.7 s).

Estimators:
- Keep position x rate on the GPU side. Count a reader ahead only if its projected landing comes before the turn's slot; pass-over is why non-readers are over-predicted by 7-20 s at 20-60 s.
- Project C from the parked and in-flight pages in deadline order, at the read thread's measured ms/page, plus the write-hold state. Planned r was 8.3 s against 30-52 s actual in c34 jit_p50.
- Do not build the design note's prefill work-ahead W or the ready-only count; neither captures these two effects.

Choosing the release time:
- Use the argmin with recourse that read_start already computes, over c_L x E[lateness] + lambda x L x E[earliness].
- c_L is the chance the GPU has no other ready work when the turn reaches the front. It is about 1 at light load and about 0 once the KV pool is full with ready work queued, which holds in c34 from minute 30. EC's GPU-bound filter is the binary version.
- Before arrival, feed in the gap support conditioned on idle time. After arrival, G = 0.
- Under saturation this makes no early reads. At light load it reads early only with room that evicts nothing but ended sessions (or clean duplicates).
- Never use the threshold form: the trace's gap entries have falling hazards.

Pre-arrival reads are this rule's light-load corner. Exact hints added 0.14-0.21 s in campaign 26; realistic hints showed nothing; p50 timing cost 0.6-0.9 s. Lambda stays a fixed knob or comes from measured slack; harm learning never moved on FA3. Pull release and backward latest starts stay simulator arms.

EVICTION AND WRITES
The largest host lever on record is room, not order. Under write-through, host memory always mirrors the GPU pool: 0.65 M of 1.63 M tokens at 160 GB, and 668K of 712K at campaign 26's 70 GB, which left 44K of room. There, 30 GB more memory bought 0.52 s, more than any hint. Decide the mirror first, by CPU replay and then a GPU smoke. The options are dropping mirror copies whose SSD write has acked, or the existing write_back mode with the host-anchor rule relaxed.

Then rank every host holder in one list (idle histories, staged pins, queued histories beyond a protected head):
- Order by slack: a low quantile of T minus restore time.
- Restore is a read when an acked SSD copy exists. Otherwise it is a recompute priced at mu, which is much larger than 1 under saturation, so only copies stay.
- Ended sessions go first. Never evict pinned, in-flight, dirty or near-head pages.
- Expect at most 0.1 s from the ranking itself.
- Mark a page clean only when the ack covers the whole op.

Writes:
- Keep eager write-through to the SSD: 56-67 % of prompt positions are read back, 4.4-4.5 times each.
- Keep fixes 1 and 2.
- Read the end signal at insert and skip ended sessions' last turn.
- Skip never-reused output KV only behind a learned per-harness flag, since this is a replay artifact.
- Turn fix 3 into a write-against-read priority tied to the room the next due reads need.

With the fixes, pending writes hold a median 0-7K tokens. The write path is a stability component (campaign 32 -> 34: -5.8 to -9.5 s), not a consumer of room; its real host cost is the write-through mirror.

SSD tier:
- Never delete a page with stored descendants of a live session.
- Delete ended sessions' chains first.
- Refresh shared heads when memory matches them. All 48 zero-head losses had a shared head, against 65-66 % of page-0 lookups that hit, so touch-on-read alone does not fix them.

PAPER
Design in five parts:
1. T = G + W, with two coupled queues and the max(GPU turn, landing) structure.
2. Read release (deadline order, pins until admission, the hold) as the main mechanism, with the argmin as its rule and pre-arrival reads as its light-load case.
3. Host memory manager: mirror accounting, one ranking, ended sessions.
4. I/O path and stability: ration by delay, not refusal; the write fixes as instances, crediting the upstream precedent (Mooncake skips rewrites) as the note does.
5. SSD tier.

Answers to the design note's open questions:
- GPU time is priced at much more than 1 under saturation (about N once the user cap binds), with no learned second price. The Lagrangian goes to the appendix, as a decision rule, not a bound.
- Queued histories beyond the head may yield, last.

Evaluation:
- State the served model: Qwen3-30B-A3B, bf16 KV.
- Explain that after the 128-user cap binds, mean TTFT follows GPU throughput (Little's law; campaign 32 against 34, minutes 90-109).
- Report readers' TTFT, p90, requests per minute and slot wait, and mark the cap minute.
- Regimes: light load (campaign 26); GPU-saturated (campaigns 28/29, campaign 34 minutes 30-70); both saturated (campaign 34 minutes 90-110); host cliff (campaign 32).

Drop:
- the threshold rule;
- 'a read started at arrival completes before the head';
- 'storage holds every prefix';
- 'recency is a weak proxy';
- retention-index gains;
- the multiplicative price;
- Triton's b/(2 rho beta) constants;
- the unsourced numbers.

## simulator_must_reproduce
- Workload and metric exactly as replayed:
- 729 Poisson users at 0.1/s, a 128-user cap whose slot waits TTFT excludes, recorded gaps at x1 capped at 1,200 s, and requests sent before 6,600 s.
- FreeInference trees with block-level shared prefixes: 17 % of prompt tokens are held by earlier sessions, and first requests find 24 %.
- Outputs forced to the recorded length and never re-sent: 54 of 16,400 prompts reuse them (agent_cache/scripts/replay_programs.py; calibration fact-finder).
- Campaign 34 ramp:
- The KV pool is at least 0.97 used whenever requests queue from minute ~30, with 19.7-20.1 running. The pool binds, not max_running 64.
- 128 users are in session from minute 84-85.
- Minutes 90-109: 129-133 requests sent a minute and ~60 queued.
- Mean TTFT 10.9-11.5 s over the run and 27-28 s in minutes 90-109.
Sources: agent_cache/results/campaign34/checks/by_minute_*.out, compare_arms.out.
- Campaign 34 two-queue behaviour:
- Readers are 24-28 % of returning requests in minutes 90-110. They wait 46-52 s (ec_full, jit_exact) or 68-69 s (jit_p50), against 12-22 s for the others.
- The read thread is 100 % busy from minute 70 (jit_p50) and from minute 90 (the others). Reads due at once sit parked a median 30 s. Last landing to admission has a p50 of 0.8 s.
- A reader's admission is max(arrival + position x rate, own landing) + 0.4 s, and position x rate is unbiased for non-readers below 20 s (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/max_structure.out, /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err_c34.out).
- A simulator that misses this structure will mis-rank release rules.
- Campaign 34 host memory and write path:
- Mirror 0.65 M, host-only 0.90-0.92 M, free 16-18K.
- Pending writes at a median of 0-7K (p90 65-94K); jit_p50's staged pins at a median of 136K.
- Write demand 4.0-4.8K tokens/s, with writes slowing from 3-4 to 9-12 ms/page as reads load the disk. Reads plus writes come to about 2.1 GB/s.
- Zero refusals; fix 3's hold switches on 28-30 times.
- Campaign 32 with the fixes off:
- The loop must appear, in the recorded order: hicache's storm at minute 39-42, then sustained refusals for jit_p50 at 67, oracle_v2 71, ec_exact 79, jit_exact 80 and ec_full 84.
- Pending writes reach 515-675K, and 74-93 % of writes are rewrites.
- Mean TTFT 33.5 / 14.6 / 15.0 / 18.0 / 14.9 / 17.2 s; 87.8-125.0 requests a minute.
- Minutes 90-109: 58.9-59.1 requests sent a minute, at 80 s TTFT.
- If the onsets need a fitted factor for the 0.25-0.5 M of host room that no logged lock explains, every stability result inherits that factor.
- Campaign 26 with Triton costs (44 users, x10 gaps, 70 GB):
- HiCache 5.3-6.6 s; the pins-only no-hint arm about 1.5-1.7 s; the no-hint queue-aware arm 0.71-0.72 s.
- Exact hints 0.14-0.21 s ahead of no hints; JIT p50 0.64-0.77 s behind.
- About half of unpinned landed reads lost before admission.
- 44K of room beyond the mirror.
- At 100 GB all arms tie at 0.41 s with under 1 % of turns reading; reads halve for every 5 GB above 70 GB.
- Campaign 28 with Triton costs:
- Hinted arms within 3 s of each other (47.7-49.8 s); hicache 115.7 s.
- Reads parked 40-68 s yet landed 1.3-1.5 s before admission.
- 'arrival' chosen in all 2,354 decisions.
- If page-level SSD chains are modelled:
- First cleaner pass at about minute 88.
- Zero-head losses only in shared-head sessions: 48/48, against 65 % among page-0 lookups that hit.
- 1.1-1.3 M tokens recomputed per arm in the 22 minutes after the pass.
- Tolerances:
- Every arm difference above keeps its ranking and sign.
- Cap-binding and loop-onset minutes within about +/-5 min; rates and waits within about +/-15 %.
- Seed spread reported.
The saturated FA3 phase has no measured run-to-run spread, so simulated sub-second differences decide nothing.

## experiments_ranked
- 1. Two-queue projection of admission and landing, offline on the campaign 34 and campaign 32 logs; no simulator needed.
- Predict each request's admission at arrival and each read's landing at park time. Admission: position x rate, counting a reader ahead only if its projected landing comes before the slot. Landing: parked plus in-flight pages in deadline order at the logged read rate, plus the write-hold state.
- Score against three alternatives: plain position x rate, the ready-only count, and the design note's work-ahead sum. Scorers: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/admit_err.py and /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/reviewer/max_structure.py.
Decides the W and r inputs of the unified fetch rule, and whether the work-ahead estimator is built (expected: no). The cheapest and most decision-relevant experiment.
- 2. Host-room replay of the write-through mirror.
- Extend /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-storage/scripts/host_room_replay2.py to campaign 26's 70 GB and 100 GB runs and to campaigns 32 and 34.
- Validate on the 100 GB result (under 1 % of turns reading) and on campaign 34's 63.4 M tokens delivered.
- Compare write-through with exclusive tiering, and with mirror copies dropped once their SSD write acks.
Decides whether mirror reclaim is the next engine project, ahead of any timing rule, and whether a write_back GPU smoke is worth queuing.
- 3. Calibrated closed-loop discrete-event simulation, validated on the must-reproduce list.
- Model: both queues; page-level host accounting including the mirror and pins; one read thread and one write thread on a shared 2.1 GB/s device; refusals; the 3-lookup cap; the cleaner optional.
- Arms:
  - the hold against the unified c_L rule against pull K;
  - early reads with exact, p50 and distribution hints at light load;
  - fixes 1, 2 and 3 each alone;
  - a write-versus-read priority against the 50K/25K hold;
  - write-through against exclusive tiering at 70, 100 and 160 GB.
Decides the fetch-rule form, which switches become defaults, and which two or three GPU runs to queue.
- 4. SSD cleaner replay with page-level chains and shared heads.
- It must first reproduce the 48 zero-head and 18 partial losses after the first pass.
- Run at the run's volume and at 6-12 M tokens.
- Orders: today's oldest mtime; touch on read; touch on memory match; leaf-first; ended sessions first.
Decides the SSD-tier subsection and whether to build an in-memory index of stored pages.
- 5. Write-value and dead-tail accounting.
- The share of written tokens never read back, by class: ended sessions' last turns, never-reused outputs, live pages.
- Whether staged reads and the 26-216 ntp_reevict events per window target the generated-output tail of plan.key.
Decides write skipping, reading the end signal at insert, and whether the replay client should send generated outputs.
- 6. Early-read ceiling per regime: min(G, r - W)+ per reading request, with exact gaps, in campaigns 26, 32 and 34.
Decides how much paper space the pre-arrival case gets and which setting must exercise it.
- 7. Closed-loop identity per minute in campaigns 28, 32 and 34: TTFT against N/X - Z - decode, before and after the cap binds.
Decides the evaluation-methodology paragraph and the metric set.
- 8. Fetch-rule form on the gap tables, with W and r as distributions (extend /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/facts-fetch-timing/sim/check_read_start.py).
Settles only the paper's equation; the engine already minimises cost. Lowest marginal value.

## gpu_only
- Run-to-run spread in the saturated FA3 phase. No repeat exists, so campaign 34's 0.6-0.7 s arm differences cannot be interpreted without one.
- Which write fix keeps the loop away (fix 2 alone, fix 3 alone). The device's read/write split (80/20 in campaign 34, 50/50 in campaign 32) and NIXL's behaviour under it are hardware facts.
- Mirror reclaim in the real engine. The engine accepts write_back with storage in cache mode; python/sglang/srt/arg_groups/hicache_hook.py:353 rejects it only for buffer_only. Two problems remain:
- GPU eviction then runs a D2H backup inline (python/sglang/srt/mem_cache/unified_radix_cache.py:983-990).
- An SSD lookup needs a host-backed or root anchor (python/sglang/srt/managers/scheduler.py:3160-3167).
Scheduler stalls and blocked lookups need a GPU run after the anchor rule is relaxed.
- FA3 prefill cost above 30K tokens, where only one request was measured. It sets rho and the read/recompute crossover the paper states.
- How the unified rule interacts with the lookup cap (defect 1), room retry, plan expiry and the 600 s pin release in the real engine.
- Whether hints matter at all on FA3: a no-hint arm (arrival timing, gate, hold, queued-last) against ec_full and jit_exact on the fixed engine.
- Early reads where W < r and host room is ample. Never tested; it needs a lighter FA3 setting, with fewer users or a larger pool.
- Whether chain-aware cleaning removes the post-pass losses in a longer run or on a smaller volume. This depends on filesystem and NIXL timing.
- The hicache and ec_exact reruns now on the GPU, to be added as fixed-engine targets for the simulator.

## unresolved
- Where the design note's 'Nov 20 trace study' (76-95 %) and the counts '974 of 2,065' and '1,209 of 2,139' come from: no record holds them.
- What held 0.25-0.5 M tokens of host room at campaign 32's first refusals. No census existed before campaign 34, so a simulator may need a fitted factor.
- Why jit_p50 trails by 0.6 s in campaign 34 with no refusals. Its pins hold 136K at the median, and throughput per 10 minutes moves by about 5 % between arms.
- Whether mu is N or 1 + Q. Both are much larger than 1 at saturation; the choice matters only near the knee, where the toy's means are noise.
- Whether the objective should weigh readers' tails. At saturation, pass-over moves 27-55 s of waiting onto readers without changing the mean.
- How representative the write-path results are. The replay never re-sends generated outputs, so 15 % of writes are dead and plan keys carry a dead tail.
- Which mechanisms beyond shared heads cause the 18 partial-hit and 17 full-hit-but-lost requests after the cleaner pass.
- Whether the max(GPU turn, landing) admission structure also holds at light load (campaign 26, where c_L is about 1). It was checked only on campaigns 32 and 34.
- Whether the toy's finding that pull K=2 beats the hold by 16 s survives the engine's behaviour of evicting queued-tier prefixes instead of refusing reads.