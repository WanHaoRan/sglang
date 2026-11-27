## position
I agree with both ideas, but I would change the unit they are measured in. The records say the gain is GPU time not spent on recompute, so GPU time should be the currency, and fetch, eviction and writes should be stated as comparisons of three marginal costs: GPU time (mu), SSD time (kappa) and host token-seconds (lambda).

At saturation the replay is a closed loop, so mean TTFT ≈ N·g − Z − decode, where g is GPU time per request. In minutes 90-110:
- campaign 32's loop added 0.555-0.565 s of GPU per request;
- the measured recompute accounts for 0.56-0.59 s of it (13.7-13.8 K tokens at 0.042-0.044 ms);
- 73 % of the 71-72 s longer cycle landed in TTFT.
Sources: /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/identity_check.out, built from agent_cache/results/campaign{32,34}/checks/by_minute_ec_full.out and campaign34/checks/fix_minutes_table_c32_ec_full.out.

So once the queue persists, one GPU-second costs about N ≈ 128 TTFT-seconds, not 1 + Q ≈ 77. In a closed-network toy (one first-come-first-served server) the cost was exactly N for a 0.01 s injection at 51-128 users, and 120 against 1 + Q = 77 for a 1 s injection at 128 users (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/mu_check_d001.out and mu_check.out). An exposed read, by contrast, costs only its own request, because admission passes over it.

(A) T = G + W is the right deadline, but the start time needs a third term, the storage queue V, and an argmin rather than a threshold. In campaign 34's minutes 90-110:
- the 24-28 % of returning requests that read from the SSD waited 45.6-52.2 s, against 18.2-21.7 s for the rest;
- they were admitted 0.6-0.9 s after their last read landed (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/read_vs_queue_c34.out).
So the read line, not the GPU queue, timed them, and the hold did not bind. Reads before arrival are the light-load corner of that rule, not half of the gain: exact hints added 0.14-0.21 s at 44 users (campaign 26), and reads timed on p50 hints did harm.

(B) Preempt and Retain become one ranking of every host holder by value per token-second. Losing a clean page costs a re-read (kappa); losing the only copy costs a recompute (mu). Writing becomes its own action: write, defer or skip, with a priority against reads on the shared 2.1 GB/s disk. The SSD cleaner is eviction on the SSD tier.

On the theory:
- The per-session separation survives with mu and kappa as Pigouvian prices. It holds as a decision rule, but not as a lower bound in the closed loop.
- lambda should be computed as the value of the marginal host holder, not learned.
- Speculative holds should be capped by measured slack rather than priced, because host memory's cost has a cliff (Weitzman's prices against quantities argument).

What both ideas miss is capacity. The write-through GPU mirror holds 0.65 M of 1.63 M host tokens, and the disk ran at its limit late in campaign 34. Neither is a timing question.

## fetch_rule
STATEMENT
For a history that is not resident:
- Land it at the q-quantile of its admission time D = A + W, with q = lambda·L/(1 + lambda·L). Start the read r̂ seconds before that.
- Before arrival, choose the start time by minimising priced cost over the support of the gap conditioned on idle time. This is an argmin, as read_start already does (python/sglang/srt/mem_cache/next_turn_policy.py:165-201), not the paper's threshold F(x|t) ≥ q.
- Before arrival, spend only measured host slack.

Priced cost of dispatching at time s:
- E[(s + r(s) − D)^+], the own stall, at weight 1. Under wait_complete, admission passes over a reading request. In campaign 34, jit_p50's non-readers waited less than ec_full's in minutes 70-110: 10.3-16.0 s against 14.3-21.7 s (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/read_vs_queue_c34.out).
- lambda·L·E[(D − s − r(s))^+], the host rent.
- kappa(s)·t·L, the disk time. It is zero when the disk would otherwise sit idle.
- kappa·t·L·P(read never used).
A refusal costs mu·c_rec·L, so a read that could cause one is ruled out by the slack budget rather than priced.

HOW G AND W ENTER
- G is the hint's gap distribution conditioned on idle time: 64 support points, or a single point for exact hints.
- W must be projected to the arrival time, not read now. It is the work ahead, minus GPU service until A, plus other sessions' expected arrivals taken from their G.
- W's spread should be calibrated on the ntp_admit lines. Today's queue-length estimate errs by +0.3 to +0.9 s for waits of 0.5-30 s, but is 19.8 s short for waits of 30-60 s (campaign 34 jit_exact server.log, calibration fact-finder). It lags a growing queue.
- After arrival G = 0, and the same rule times the hold.

HOW r AND THE STORAGE QUEUE ENTER
- r(s) = V(s) + own transfer + lags.
- V = (pages due before D that are parked or in flight, plus the expected write-hold time) / read rate.
- The planner before arrival and the hold after it use the same r.
- kappa is the number of requests whose admission is bound by the read line. Late in campaign 34, 12-31 reads were parked.
- Once V > W, dispatch order is the decision: earliest deadline first while the deadlines are feasible, value per disk-second once they are not.

READ OR RECOMPUTE
Read iff kappa·t_r + stall/U + lambda·hold < mu·c_rec, per token.
- On an idle FA3 GPU (mu = 1), recompute wins for short histories: 0.62 s against 0.8-1.1 s per 20 K tokens (agent_cache/results/campaign32/checks/mechanisms_findings.txt:174).
- With one request queued, reading wins.
- In campaign 34's late minutes (inferred): a 25 K-token read costs its owner about 28 s, plus about 1.4 s × kappa to others, against 1.07 s × mu for recompute. That is close to a tie at mu = 1 + Q ≈ 60, and about a 2× win for reading at mu ≈ 0.73 N. Splitting a history between the read and recompute is worth simulating.

WHAT IT REDUCES TO
- Light load (mu ≈ 1, kappa ≈ 0, W < r; campaign 26): reads before arrival by argmin. With exact hints, it lands at D minus a margin, which is today's JIT exact. On falling-hazard entries (FreeInference tool and user, sigma_high 1.79-1.86), it reads at arrival.
- GPU saturated, disk keeping up (W ≥ r; campaigns 28/29, campaign 34 minutes 30-70): no early reads. ec_full logged 1 'now' among 6,185 decisions in campaign 34. What remains is the hold, with V inside r and a margin set by the price.
- Both saturated (V > W; campaign 34 minutes 90-110): the hold never binds. Dispatch order and bytes read decide the outcome, and early reads only reorder a saturated disk.
- Host at the cliff (campaign 32's loop): the speculative budget is 0, and writes go before timed reads.

NEW AGAINST THE ENGINE
1. r counts parked reads and the write hold. Today:
   - the r before arrival counts only dispatched pages (python/sglang/srt/mem_cache/unified_radix_cache.py:2334-2337);
   - the hold counts the parked reads ahead of it in that pass;
   - neither counts the write hold;
   - in campaign 34, reads that were due at once still waited a median 30 s parked.
2. W is a projected distribution. Today it is a point value at the current time, and the expected-cost path counts the queue only while the GPU is the bottleneck (python/sglang/srt/mem_cache/next_turn_prefetch.py:698-712).
3. q follows lambda·L. Today the margins are fixed: 1 s (next_turn_prefetch.py:125) and 2 s (next_turn_prefetch.py:128), and jit_p50 lands at the entry's p10.
4. lambda is computed, not learned.
5. The speculative budget comes from the census, not from 0.3 of the host pool.
6. When the read line cannot meet its deadlines, it orders by value per disk-second.
The single seam is trigger_time (python/sglang/srt/mem_cache/next_turn_estimators.py:147), with read_not_before and read_deadline (next_turn_prefetch.py:1448 and 1388).

EVIDENCE
- Early reads add at most 0.14-0.21 s, and only at light load with exact hints (agent_cache/results/campaign26/DECISIONS.md:835-876).
- Reads timed on p50 hints cost 0.64-0.77 s in campaign 26 and 0.62-0.67 s in campaign 34 (agent_cache/results/campaign34/DECISIONS.md:145-152). In campaign 32 they started the loop 8-17 minutes early.
- On FA3, ec_full (1 early read) ties jit_exact (99 early reads).
- The threshold form would read the FreeInference tool entry at once: 1.81 s of cost against 1.10 s for reading at arrival (fetch fact-finder's CPU check).

## eviction_and_writes
ONE RANKED LIST (Retain and Preempt)
Rank every host holder by ν = (chance it is needed per second) × (priced loss if it is not resident) / tokens, and evict the lowest first.
- A page with an intact SSD copy loses κ·t_read plus its own stall, (r − W)^+.
- A page that is the only copy loses μ·c_rec. Only copies arise from a write that failed (the lock is still released at the ack, python/sglang/srt/mem_cache/unified_radix_cache.py:3511-3515), from an SSD copy the cleaner deleted, or from a GPU→host copy that was refused.

Order at saturation:
1. Ended sessions (ν = 0).
2. The GPU mirror. Its ν is about 0, but under write_through it cannot be evicted (python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py:2145-2146). It holds 0.65 M of 1.63 M host tokens (agent_cache/results/campaign34/DECISIONS.md:98-102), the largest misallocation.
3. Idle clean prefixes, by hazard × (κ·t_r + stall).
4. Staged pins, by P(return within a re-read) × (κ·t_r + stall). This is the paper's F_j(r_j|t) when κ = 0. When the disk is busy, a preempted read means a full re-read, so preempt the pins least likely to return at all.
5. Queued prefixes far from the head, only when their re-read can finish before admission. This is a hypothesis; keep queued-last while the lookup cap and refusals exist.
6. Arrived pins and only copies: never.

The engine does the reverse at the boundary: it evicts other queued requests' unpinned prefixes before releasing staged pins (unified_radix_cache.py:3039-3047). Expect little TTFT from refining the ranking:
- forecast ranking within idle sessions moved TTFT by 0.1 s at most (agent_cache/results/campaign26/DECISIONS.md:965-989);
- preemption never fired on FA3 (0 ec_bump in campaigns 32 and 34), and in campaign 26 it bumped 462 reads with no gain (campaign26/DECISIONS.md:854-870).
What matters is the boundary between holders whose loss costs μ (recompute) and those whose loss costs κ (a re-read), and pins and queued-last already approximate it.

CLEAN AND DIRTY
A pending write host-locks its page from submit to ack (unified_radix_cache.py:2055-2060), so dirty pages are outside the eviction heap, and the write order decides which pages become evictable.
- Write the next victims (lowest ν) first.
- Defer pages with high ν; they stay resident anyway.
- Mark a page as on the SSD only when the ack covers it.

THE WRITE ACTION
Write page p iff P(it is needed from the SSD after its host copy goes) × (μ·c_rec − κ·t_r − stall) ≥ κ·t_w + λ·tokens·latency + σ, where σ is the price of SSD capacity.
Zero-value writes should never happen:
- Rewrites of stored pages: 74-93 % of writes in campaign 32's loop (campaign34/DECISIONS.md:89-90).
- Ended sessions' last turns: the end signal cut SSD pages by 14-17 % in campaign 26. Today it is read only at commit (python/sglang/srt/mem_cache/next_turn_prefetch.py:513-532).

WRITES AGAINST READS
The disk is one device of about 2.1 GB/s, roughly 21.4 K tokens/s at 98,304 B a token. In its saturated minutes, campaign 34 split it 1.72 GB/s reads against 0.39 GB/s writes; campaign 32's loop split it 1.10 against 1.00 (iostat_vdc.log of both campaigns).
- Dispatch the transfer with the highest value per disk-second.
- A read is worth κ: the requests bound by the read line.
- A write is worth λ × the tokens it frees when host memory is at the boundary.
- Fix 3 (python/sglang/srt/mem_cache/storage_read_scheduler.py:147-150, 170-181) is the threshold version of this rule, and it points the right way.

SSD TIER
The cleaner is eviction on the SSD. σ is the value of its marginal victim.
- Today it deletes the oldest-mtime groups first (python/sglang/srt/mem_cache/storage/nixl/nixl_cleaner.py:183, 270). Only writes refresh mtime, and with chained keys a deleted head cuts off the whole chain.
- After the first cleaner pass, campaign 34 had 65 requests whose lookup found nothing, 18 that found only part of their history, and 31 reads that failed on deleted files. That is 1.1-1.3 M recomputed tokens per arm in 22 minutes (evidence fact-finder).
- By the closed-loop identity that is 17-20 ms of GPU time per request, about 2.2-2.5 s of cycle time and about 1.6-1.9 s of TTFT (inferred; /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/identity_check.out).
- Priced order: ended sessions first, then the tails of the coldest live chains (deleting page j loses pages j through the end), never a live head. Refreshing mtime on reads is the minimal fix.

PRICE λ
Compute it rather than learn it: λ = the smoothed maximum ν among the holders evicted in the last interval.
- With a need rate of about 1/60 s (host turnover is about a minute), late campaign 34 gives about 2e-5, dominated by the disk; early campaign 34 gives about 1e-6; campaign 32's loop gives about 4e-5 (all inferred).
- Today's 'harm' rule cannot move. It charges only refusals up to the staged pins seen (next_turn_prefetch.py:1345-1360). It uses Triton's 1.1e-4 s a token (python/sglang/srt/mem_cache/next_turn_policy.py:44), 2.5× FA3's cost. In campaigns 32 and 34 it stayed at 5e-5 (campaign32/DECISIONS.md:225-226).
- Use λ for ranking and for comparing writes with reads. Govern speculative holds with budgets.

STABILITY
Campaign 32's loop had two causes in price terms.
- A missing price: rewrites were bought at zero.
- Rationing by refusal: a refused read is recomputed, and the recompute is written again, so demand grew with the shortage.
Campaign 34 rationed by delay instead. Readers absorbed the shortage as their own wait (+26-33 s late), nothing fed back into write demand, and no read was refused.

A priced design is stable if:
1. zero-value flows are priced out;
2. a scarce resource is rationed by an action that does not create demand for it: delay a read rather than refuse it;
3. speculative holds come only from measured slack. The fixed pin budget of 0.3 of the host pool did not stop jit_p50's 208-267 K early pins in campaign 32 (agent_cache/results/campaign32/DECISIONS.md:197-200);
4. the mandatory flows fit the device. Late in campaign 34: about 4.3 K tokens/s of writes plus 17.7 K of reads, against 21.4 K.

Damping it needs:
- smooth μ and κ over the dispatch-to-landing horizon, and λ over host turnover;
- hysteresis bands. The 50 K/25 K hold switched on 28-30 times a run, for a median of 2.0-2.8 s;
- a rate limit on speculative starts, since one change in a global price flips every plan at once;
- caps: μ ≤ users in session, and λ ≤ μ·c_rec divided by the shortest useful hold.

Prices cannot create capacity. When mandatory flows exceed it, the cheapest rationing order is delay (weight 1), then skipping low-value writes, then recompute (weight μ).

## paper_structure
Organise §3 around one accounting identity and three prices, and present today's four rules as the cases they reduce to.

3.1 Where TTFT goes. TTFT_i ≈ W_i + S_i + prefill. At saturation the replay is a closed loop, so mean TTFT ≈ N·g − Z − decode. A policy therefore acts through three things:
- GPU work per request (recompute);
- the read line (V);
- the request's own stall.
Lead with campaign 32 against 34 in minutes 90-110: 0.56 s of GPU time per request, 13.8 K recomputed tokens, 52 of 71 s in TTFT.

3.2 Three prices from three queues, in one table of observable, estimator and range seen:
- μ, GPU time: about 1 below saturation and about the number of users in session once the queue persists;
- κ, SSD time: the number of requests bound by the read line;
- λ, host token-seconds: the value of the marginal holder, computed rather than learned.
σ, SSD capacity, is optional.

3.3 Actions as priced comparisons, in one table:
- read or recompute;
- when to dispatch (deadline D = A + W, read time from V);
- hold;
- evict (one ranked list that includes staged reads and the GPU mirror);
- write (write, defer or skip, and its priority against reads);
- clean (the SSD tier).

3.4 Regimes, each with the campaign that shows it:
- light load (campaign 26);
- GPU saturated, disk keeping up (campaigns 28/29, campaign 34 minutes 30-70);
- both saturated (campaign 34 minutes 90-110);
- host cliff (campaign 32).

3.5 Stability. Ration by delay, not refusal. Use slack budgets and hysteresis for speculative holds. The four write-path fixes are instances of these rules: fix 1 prices out rewrites; fix 3 rations reads by delay while writes back up; fix 4 retries a refused read without spending the turn's lookups; fix 2's role is not settled, since campaign 34 ran all four at once.

3.6 What hints buy:
- G, for light-load timing;
- arrival forecasts, for W and μ;
- the end signal, for eviction, writes and cleaning.
Information levels stay fallbacks, not separate arms.

Appendix:
- the separable Lagrangian with μ and κ as subgradients of convex queueing costs. First-come-first-served waiting is convex in work by Lindley's recursion, so a weak bound holds for fixed arrival times. In the closed loop it fails: the toy's paired effect is negative near the knee;
- μ = N in a saturated closed network;
- the newsvendor landing quantile;
- argmin against threshold for falling hazards;
- prices against quantities (Weitzman);
- the engine mapping: switches and line numbers.

Drop or move:
- The D(λ) lower bound as a headline. It has no queue term, and the queue is 97-98 % of TTFT.
- The multiplicative λ update and the 'harm' learning. Neither moved in campaigns 32 or 34.
- A gain from the retention index: 0.1 s at most, and never built as written.
- 'Recency is a weak proxy': no record supports it.
- The b/(2ρβ) crossover as the reason to read. On FA3, recompute costs 0.042-0.044 ms a token against about 0.055 ms of disk to read it; reading wins because μ > κ.
- 'A read started at arrival completes before the head of the queue': false once V > W.
- eq. fetch as the rule: keep it only as the rising-hazard case.
- Preempt as its own section.
- Numbers without a record behind them: the Nov 20 trace study, '974 of 2,065', '+0.003 s'.
- State the served model: Qwen3-30B-A3B with bf16 KV, not Qwen3-32B-FP8.

## experiments
- **Closed-loop identity (offline)**: Q: Does mean TTFT ≈ N·g − Z − decode hold minute by minute once the GPU is saturated? Does the change in GPU time per request (g) equal the measured recompute GPU time, in every arm and on both backends?
  - method: Offline, from agent_cache/results/campaign{28,32,34}/checks/by_minute_<arm>.out, the fix_minutes tables and decode_scale.out. Per minute: g = 60 / requests sent; recompute GPU time = history computed again × cost per token (0.043 ms on FA3, 0.13 ms on Triton); N = users in session. Regress TTFT on N·g. Minutes 90-110 of ec_full and jit_exact are already done (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/identity_check.py).
  - expected: A slope of 0.7-1.0 × N after the 128-user cap binds, and near 0 before the GPU pool fills. Recompute explains the change in g within ±15 %. Already seen: g rose by 0.555-0.565 s against 0.56-0.59 s of recompute.
  - decides: Whether the paper states GPU time as its currency, with mu ≈ N at saturation, and builds §3.1 on this identity instead of on per-session queue terms.
- **Estimator for mu (CPU discrete-event simulation)**: Q: Which observable predicts the TTFT cost of one more GPU-second: 1 + Q, the number of users in session, or 1 + Q plus the arrivals expected (from hints) before the queue drains?
  - method: A calibrated closed-loop simulation: the 729 users of campaign 34, FA3 prefill and decode fits, a batch bound by the KV pool at 19-23 requests, one read thread and one write thread on a 21.4 K tokens/s device. Inject 0.05-0.5 GPU-seconds at random times under common random numbers. Compare the change in summed TTFT and in throughput with each estimator, by phase: before minute 28, minutes 28-85, after the cap binds.
  - expected: About 1 before the GPU pool fills and about N after the cap binds. 1 + Q is 3-8× low in minutes 30-70. The toy gave exactly N for a 0.01 s injection at 51-128 users, and 120 against 1 + Q = 77 for a 1 s injection (/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/mu_check_d001.out).
  - decides: Which mu estimator the engine uses. Also whether a priced early-read rule would have rejected jit_p50's early pins in campaign 32's minutes 30-60; it does only if mu is not 1 + Q.
- **GPU line against read line (offline)**: Q: How much of late TTFT is set by the SSD read line, and how large is kappa?
  - method: Extend /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/economics/read_vs_queue.py to all campaign 34 arms and to the hicache and ec_exact reruns once they finish. Per minute: the waits of requests with and without an SSD read; the wait after the last landing; parked reads whose landing estimate falls after their admission estimate (from srs_park, srs_dispatch and ntp_admit); the device split from iostat.
  - expected: Requests that read from the SSD wait +26-33 s longer (ec_full, jit_exact) and +27-55 s longer (jit_p50) from minutes 70-90 on. kappa is 12-31, and reads take about 80 % of the device.
  - decides: Whether fetch work should target V first (the storage queue inside r, dispatch order, fewer bytes read) rather than G, and whether the paper needs a 'both saturated' regime.
- **Host room: GPU mirror against exclusive tiering (CPU discrete-event simulation)**: Q: What are the 0.65 M mirror tokens worth at campaign 34's setting, compared with any timing rule?
  - method: The simulation with usable host room of 0.98 M tokens (write_through), 1.3 M and 1.63 M (exclusive), each running today's rules. Add a no-hint queue-aware arm, as campaign 26's nofc_noend.
  - expected: Fewer SSD reads, and the read line saturates later. The gain should exceed the 0.6-0.7 s that separates the arms in campaign 34; in campaign 26, 30 GB more host memory was worth more than any hint (agent_cache/results/campaign26/DECISIONS.md:1096-1118).
  - decides: Whether exclusive tiering is the next engine project, ahead of timing rules. It is a change against two engine constraints: an anchor without a host copy blocks SSD lookups, and GPU eviction deletes nodes that have no host copy. Also whether eviction class 2 stays in the paper.
- **Value of SSD writes (offline)**: Q: What share of the tokens written to the SSD is ever read back, split by the session's state at write time?
  - method: Per conversation, from campaign 34's server logs: tokens in h2s_submit and h2s_done against later s2h_io tokens of the same conversation, joined with the client's session ends. Split by session ended, live and later served from host, and live and later read from the SSD.
  - expected: A large share is never read from the SSD; ended sessions alone are about 15 %, given campaign 26's 14-17 % cut in SSD pages. Reads exceed writes overall: 14.9-25.5 M read against 5.3-8.4 M written per window (agent_cache/results/campaign34/DECISIONS.md, fix table).
  - decides: Whether a write action (skip, defer, order by ν) is worth building, and whether the end signal must be read when the turn is inserted instead of at commit.
- **Loop stability and damping (CPU discrete-event simulation)**: Q: Which mechanisms keep campaign 32's loop from starting, and can priced rules replace the fixed thresholds without oscillating?
  - method: The simulation with host locks: pending writes, pins, reads in flight, plus an ancestor-cascade factor fitted so that campaign 32's onsets come at minutes 67/71/79/80/84 and campaign 34 shows none. Arms: no fixes; fix 1, 2 and 3 each alone; all four; early reads gated by a computed lambda(mu, kappa), with and without a slack budget and hysteresis; a priced write-against-read dispatcher in place of the 50 K/25 K hold; load steps to provoke herding.
  - expected: Fix 1 removes the rewrite feedback. Fix 3 turns refusals into reader delay. Priced rules are stable only with a slack budget and with smoothing at least as long as the write latency. Prices taken instantaneously cause bursts of early reads whenever mu falls.
  - decides: Which switches become defaults, whether lambda is used for anything beyond ranking, and the damping constants: smoothing, hysteresis band and budget.
- **Form of the fetch rule (CPU, no queue)**: Q: On the trace's gap entries, how do three rules compare: the threshold F(x|t) ≥ q, the argmin of priced cost, and landing at a fixed quantile?
  - method: Extend the fetch fact-finder's check_read_start.py. Per entry (tool, user, '*'): lambda from 1e-6 to 1e-4; W as a distribution built from campaign 34's ntp_admit errors; r including V. Compute expected stall + rent + wasted reads under each rule.
  - expected: On falling-hazard entries (sigma_high 1.79-1.86) the threshold fires right after the turn ends and pins memory through the tail. The argmin chooses reading at arrival unless W < r and the entry is tight.
  - decides: The paper's fetch equation (the argmin, with the threshold kept as the rising-hazard case), and a landing quantile in place of the 1 s and 2 s margins.
- **Filling idle disk time, and dispatch order (CPU discrete-event simulation)**: Q: Should held reads dispatch early when the disk would otherwise sit idle? When parked work cannot meet its deadlines, should the read line order by value per disk-second instead of by deadline?
  - method: Simulation arms: today's hold with earliest-deadline-first order; work-conserving dispatch within a slack budget; pre-arrival reads into idle disk time with exact against p50 hints; earliest-deadline-first against value order once parked work exceeds capacity; splitting one history between a read and a recompute when V − W is large.
  - expected: Gains only while the disk has idle time (campaign 34 minutes 30-90, read thread 19-86 % busy). Zero-sum once the disk is saturated. Harm with p50 hints.
  - decides: Whether the unified fetch rule credits idle disk time and drops earliest-deadline-first order, or stays strict just-in-time.
- **SSD cleaner order (CPU replay)**: Q: Which cleaner order removes campaign 34's losses after the first cleaner pass?
  - method: Replay campaign 34's writes and reads as session-level chains, with the cleaner at its 90 %/80 % marks. Orders: oldest mtime (today), mtime refreshed on reads, ended sessions first, tails first, and the last two together.
  - expected: Most of the 65 requests whose lookup found nothing, the 18 that found only part of their history, and the 31 reads that failed on deleted files disappear with ended-sessions-first plus tails-first.
  - decides: The SSD-tier eviction rule, and whether refreshing mtime on reads is enough.

## risks
- At saturation, mu ≈ N makes the priced design reduce to 'never recompute', which campaign 34's simple arms already do: they tie within 0.7 s. The novelty has to come from accounting for capacity (host room, the read line, writes). Otherwise the design reads as relabelled heuristics.
- Prices taken instantaneously can herd. Every plan sees the same change in mu or lambda at once, so speculative reads would start in bursts. Without slack budgets and hysteresis, a priced design could bring back campaign 32's early onset (inferred).
- A calibrated simulation may not reproduce campaign 32's onset. At the first refusals, 0.25-0.5 M tokens of host room were held by no logged lock (agent_cache/results/campaign32/DECISIONS.md:179-184). Any stability conclusion then rests on a fitted cascade factor.
- mu is not 1 + Q. Both the toy and the campaign 32/34 identity give about N. An engine that uses queue length underprices GPU time by about 1.6× at the 128-user cap, and by more near the knee, which is exactly where early reads are decided.
- The replay's metric, mean TTFT of requests sent before 6,600 s, mixes in throughput. Campaign 34's p50 rose because it served more requests late in the run. A price on summed TTFT can trade away throughput, so report paired TTFT alongside requests per minute.
- Each arm ran once, and no run-to-run spread is measured at saturation. Campaign 34's 0.6-0.7 s differences and the loop-onset minutes may be noise.
- Constants must be the served model's. Campaigns 28-34 ran Qwen3-30B-A3B with bf16 KV (98,304 B a token), not Qwen3-32B-FP8. The engine's harm price still uses Triton's 1.1e-4 s a token (python/sglang/srt/mem_cache/next_turn_policy.py:44), 2.5× FA3's cost.
- Exclusive tiering and reordering writes both break engine assumptions: an anchor without a host copy blocks SSD lookups, and assume_stored relies on prefixes being written first. The largest lever is also the riskiest change.
- The 'both saturated' regime may belong to this box: one read thread, NIXL POSIX, a 2.1 GB/s disk, bf16 KV. With more disk bandwidth or FP8 KV it could disappear, and the read-line rules with it.
- The toy behind the mu claim is a single first-come-first-served server. Continuous batching splits the cost between TTFT (73 % in campaign 32 against 34) and decode, and its knee is not sharp.