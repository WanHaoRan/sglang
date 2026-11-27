## position
I agree that T = G + W should key every decision, and that the write path and the SSD tier belong in the design. On the fixed engine, though, the storage costs sit somewhere other than where the proposal puts them.

- **Pending writes are not what fills host memory.** In campaign 34 they hold a median 0-7 K tokens. The write-through mirror holds 0.64-0.65 M of the 1.63 M-token pool, and from minute 30 it equals the KV of the requests running on the GPU (0.65 M), so nothing ever reads it from host while it is there.
- **The queue outgrows host memory, and the read line saturates.** From minute 70-80, queued requests' histories (0.9 M, rising to 1.6-1.7 M) exceed the 0.90-0.92 M of host-only room. The excess cycles through the SSD and the single read line saturates. In minutes 90-110, returning requests with a read waited 46-69 s; the others waited 12-19 s.
- **Host accounting should come before forecast-based eviction.** An open-loop replay of the three campaign 34 timelines puts the mirror at 56-61 % of SSD read demand. A perfect next-use order with session ends saves 22-25 %.
- **For (A):** the read start is not a per-read decision. Reads share one device of about 2.1 GB/s with writes, so r must come from the I/O queue (parked reads, reads in flight, the write share). Reads should be scheduled backward from their deadlines. Reads before arrival should run only when the device has slack: jit_p50's early reads raised SSD volume to 82.5 M tokens, against 54-64 M for the other arms, and saturated the read line 20 minutes earlier.
- **For (B):** I would merge Preempt and Retain into one order keyed on slack, T minus (read-line delay + r), over explicit page states: pinned, dirty, clean, clean duplicate. Keep eager write-through, since pages are immutable and each page read back is read 4.2-4.5 times. Stop writing dead pages, and replace the fixed 50 K hold with a clean-reserve rule.
- **Missing from both ideas: the SSD tier's own eviction.** The cleaner deletes the oldest-written pages first. Those are the shared prefix heads that live sessions keep using from memory. All 48 head-missing lookups after the first pass came from sessions with such a head; 73 % of all sessions have one. At that pass, finished sessions held 16.2 M tokens of written KV and live ones 4.7 M.
- **Metrics:** once the 128-user cap binds, the GPU pool stays 96-98 % full. Storage changes that do not cut recompute then mostly move waiting between requests that read and requests that don't (inferred). The paper should report the TTFT of requests that read, p90 and SSD volume, as well as the mean.

## fetch_rule
**Statement.** Each read i is a job with a size (p_i missing pages) and a deadline d_i, set at a low quantile of its need time T_i = G_i + W_i; after arrival T_i is nearly a point.

- The I/O scheduler holds every known job: arrived reads, plus candidates before arrival as tentative jobs.
- It computes latest starts backward in deadline order, over the read rate left after writes (beta_r = beta - w):
  - s_i = min(d_i, s_next) - p_i * b / beta_r, where s_next is the start of the next job by deadline.
  - Every job due by d_i can then finish by d_i.
- A read is dispatched at its latest start, which keeps host memory free. It goes earlier only when the device would otherwise sit idle.
- Arrived reads always qualify. A read before arrival qualifies only if both hold:
  1. It fits in device slack: dispatching it now pushes no arrived read past its deadline.
  2. Its room comes from free pages or evict-first classes (ended sessions, clean duplicates), and its expected saved stall, P(T_i < D_r + r_i | idle time) * r_i, beats the eviction cost of the marginal victim.

**How G and W enter.**
- T = G + W.
- G comes from the hint distribution conditioned on idle time, and is 0 after arrival.
- W before arrival must be projected to the predicted arrival: queued GPU work plus expected arrivals. Today the engine uses the wait a request arriving now would see.
- Deadlines use a low quantile of T, so latest starts stay safe when the estimator lags. In campaign 34 jit_exact, the position x rate estimate was a median 19.8 s short for 30-60 s waits (ntp_admit lines; sim-calibration finding).

**How r and the storage queue enter.** r_i = D_r(d_i) + p_i * b / beta_r.
- D_r is the device time of jobs ahead of i in deadline order: parked reads, reads in flight, and 'now' reads expected before d_i.
- beta_r excludes the write rate. At saturation in campaign 34, writes took about 4.3 K tokens/s, about 0.42 GB/s of about 2.1 GB/s (campaign34/checks/fix_minutes.out, read_thread.out). While the write hold is on, beta_r = 0 for timed reads.
- Today the estimate before arrival counts only pages in flight (unified_radix_cache.py:2334-2337). The hold counts the parked pages ahead in the current pass (next_turn_prefetch.py:1448-1481). Neither counts writes or 'now' reads still to come.
- In campaign 34, reads due at once waited a median 30 s parked.

**What it reduces to, by regime.**
1. **W much larger than r, device has slack** (TraceLab on Triton): latest start = T - r - margin, which is today's hold. Reads before arrival never qualify.
2. **Device saturated** (campaign 28; campaign 34 from minute 70 for jit_p50 and minute 90 for the others): latest starts are already past, so reads run back to back in deadline order and holds never bind. Nothing before arrival qualifies. The lever is read demand (host room), not timing.
3. **W < r with device slack** (campaign 26 at 44 users): arrived reads start at once. Reads before arrival go on hint quantiles, gated by room. This is the only regime with a measured gain: 0.14-0.21 s with exact hints (campaign26/DECISIONS.md:835-876).
4. **Idle GPU, busy device:** recompute short histories. On an idle FA3 GPU, 20 K tokens recompute in about 0.62 s against a 0.8-1.1 s read (campaign32/checks/mechanisms_findings.txt:174).

**What is new against the engine today.**
- One r from the I/O scheduler, used by the planner, the hold and refetch.
- Latest starts computed backward over all jobs, replacing the per-read hold.
- The write rate inside r.
- Reads before arrival gated by device slack and evict-first room, instead of lambda (expected_cost) or early_s (just_in_time).
- W projected to the arrival time.

**Evidence.**
- Campaign 34, minutes 90-110: returning requests with an SSD read waited 45.6-68.7 s; the others waited 11.7-19.4 s. Their read stall is nearly their whole wait (45.0 of 45.6 s for ec_full), so the read line, not the GPU, set their admission (out/read_vs_noread_*.out).
- The read thread was 100 % busy from minute 70 (jit_p50) and minute 90 (ec_full, jit_exact), with reads parked 37-67 s (campaign34/checks/read_thread.out).
- jit_p50's 1,208 early reads raised SSD volume to 82.5 M tokens, against 64.1 M (ec_full) and 54.4 M (jit_exact), and held 95-241 K staged host tokens (out/census_windows_jit_p50.out).
- ec_full had W >= r in 3,929 of its 3,973 'arrival' decisions.
- Campaign 29's hold: -77.7 s, confounded with the arrival gate and the 120 s lead (campaign29/DECISIONS.md:210-232).

Scratch outputs are in /tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/designer-storage/out/.

## eviction_and_writes
**Page states (OS page-cache view).**
- Pinned: landed reads until admission (budget 0.3 x pool), anchors of lookups and of parked or in-flight reads, and load-back sources.
- Dirty: an SSD write is pending; the page is host-locked from submit to ack (unified_radix_cache.py:2027-2061, 3504-3524).
- Clean: host-only and already on the SSD.
- Clean duplicate: a host copy of KV that is still on the GPU.

Campaign 34 census, minutes 30-110 (out/census_windows_*.out; campaign34/DECISIONS.md:98-102):
- clean duplicates 0.64-0.65 M tokens;
- clean host-only 0.87-0.98 M;
- dirty median 0-7 K (p90 65-94 K);
- pins 0-54 K;
- jit_p50's staged reads 95-241 K;
- parked-read anchors up to 107 K.

The clean duplicates equal the GPU tokens of running requests from minute 30 (0.64-0.65 M against 0.65 M; out/mirror_vs_running_*.out). Host eviction never takes them, because the host-leaf test requires node.evicted (unified_tree_core.py:2138-2158).

**One eviction order, by slack.** Evict the page group with the largest slack, T - (D_r + r): the least chance of being needed before it could be brought back. D_r is the read line's current delay. Never evict negative slack, pinned or dirty pages.

In class terms:
1. Ended sessions.
2. Clean duplicates of running requests. They are recreated by a D2H copy at finish, and only for sessions whose next T is short.
3. Idle prefixes, largest T first. This is where the forecast acts.
4. Staged reads, ordered by F(D_r + r). This is the draft's Preempt.
5. Queued requests farthest from admission, only while the read line can bring them back in time.
6. Never: the head of the queue.

Retain and Preempt become one rule.

The replay (out/host_room_replay_*.out) is open-loop, holds the recorded timeline fixed, and works at session level. Its write-through + LRU arm comes within about 20 % of the recorded SSD volumes.
- Without the mirror, SSD read demand falls from 63.3/62.7/56.6 M to 24.6/24.6/24.7 M tokens. That is the same as adding 0.65 M tokens of host memory.
- A perfect next-use order with session ends saves 22-25 % with the mirror, and a further 40-43 % without it. Forecasts matter more once host room is not wasted.
- Ordering the queued class farthest-first moves volume by under 1 %. The model cannot see timing.

Engine seams:
- Duplicate reclaim exists only under write_back (unified_tree_core.py:1739-1765).
- write_back blocks the scheduler on each D2H copy at GPU eviction (unified_radix_cache.py:983-990, 3745-3758).
- Under write_through, a GPU-evicted node with no host copy is deleted (unified_tree_core.py:1630-1640).
- An SSD lookup needs a host-backed anchor (scheduler.py:3160-3167).

Preempt has no evidence on FA3: 0 bumps in campaigns 32 and 34, and 462 in campaign 26 with no TTFT gain shown. It should not be a headline rule.

**Write path.** Keep eager write-through to the SSD:
- KV pages are immutable and append-only.
- 56-67 % of prompt positions are read back from the SSD at least once, each 4.2-4.5 times (out/client_writes_*.out).
- Eviction must stay instant. Campaign 32's loop came from dirty pages: 240-675 K tokens pending (campaign32/DECISIONS.md:163-191).

Changes:
- **(a) Skip dead pages.** In this replay the next prompt carries the recorded reply, not the generated tokens. Only 54 of about 16,400 next prompts reuse the previous output, so 4.78 M of the 31.6-31.9 M tokens submitted per run (15 %) are output KV that is never read (h2s_submit sums). The plan key includes this dead tail too (unified_radix_cache.py:1235-1241). Make output skipping a per-harness flag learned from the next turn's match. Also skip the last turn of ended sessions.
- **(b) Order queued writes by eviction rank instead of FIFO,** so the eviction frontier becomes clean first. This matters only when writes fall behind; the median write takes 0.1-0.6 s in campaign 34.
- **(c) Set the on-SSD state only when the ack covers the whole write.** Today a failed write releases the lock as if it had succeeded (unified_radix_cache.py:3511-3515).
- **(d) Keep fixes 1 and 2.**

**Writes against reads.**
- At saturation, reads ran at 17.4-18.0 K tokens/s and writes at about 4.3 K tokens/s on a device that tops out near 2.1 GB/s. Writes are mandatory at that rate, so priority can only smooth bursts; reducing volume is the real lever.
- I/O classes, highest first:
  1. 'now' reads;
  2. deadline reads, by latest start;
  3. frontier writes, when the clean reserve is short;
  4. reads before arrival, only in device slack;
  5. other writes.
- Replace fix 3's fixed 50 K hold with a dirty limit in the style of Linux dirty_ratio. Fix 3 switched on 28-30 times a run, for 57-81 s in total. The new rule holds timed reads only when free plus clean evictable room outside protected classes falls below the allocation expected over the write drain time (pending write tokens divided by the write rate).
- At this setting the new rule would change nothing. It matters for smaller hosts or faster GPUs.

**SSD tier.**
- How it fails today:
  - The cleaner deletes the page groups with the oldest mtime, from 90 % down to 80 % full (nixl_cleaner.py:168-212).
  - mtime changes only on writes and on fix 1's touch.
  - A lookup stops at the first missing page (hicache_nixl.py:658-664).
- Shared prefix heads are written once by the first session and then served to later sessions from GPU or host, so they are the oldest files.
- The evidence (out/head_missing_*.out, out/ssd_dead_live_*.out):
  - All 48 head-missing requests after the first pass (17/16/15 per arm) came from sessions whose first prompt shared a prefix with an older session; the shared part was a median 20-32 % of that prompt. Across all sessions the rate is 73 %, so 48 of 48 by chance has odds of about 3e-7.
  - 46 of the 48 sessions were younger than the window the passes deleted.
  - At the first pass, finished conversations held 16.2-16.3 M tokens of written KV and live ones 4.7 M. Each pass deletes about 2.5 M.
- Proposal:
  - Keep an in-memory index of stored pages: key, parent key, last use, and the live sessions holding the page. That is about 400 K pages, tens of MB.
  - Delete leaf-first: never a page with a stored descendant.
  - Delete ended or expired sessions first, then by last use or largest T.
  - Never delete pages with a parked or in-flight read. 31 'No such file' read failures came within 3 minutes of a pass.
- Touching files on read does not protect shared heads, because they are used from memory, not read from the SSD.

## paper_structure
1. **Problem statement.** TTFT ≈ W + S + recompute, and W responds to recompute (keep section 2 of the design note). State the resources as they really are:
   - host room R depends on the mirror policy, and under write-through the mirror is the running set's KV;
   - one SSD of about 2.1 GB/s is shared by reads and writes;
   - the SSD tier has finite capacity and evicts on its own.
2. **Design: one memory-hierarchy manager with four components,** each with its rule and evidence.
   - (a) Time-to-use estimator: T = G + W, and r from the I/O queue.
   - (b) Host memory manager: page states and one slack order that covers retain, preempt and queued protection, with the mirror reclaimed.
   - (c) I/O scheduler: deadline reads with backward latest starts, reads before arrival only in device slack, and the write path (skip stored, merge, skip dead pages, clean-reserve hold).
   - (d) SSD tier: leaf-first, session-aware cleaning.
   - One figure: a session's prefix moving through the tiers on a timeline, with its page state.
   - One table: component, rule, engine seam, evidence.
   - lambda becomes one sentence: the marginal victim's eviction cost.
3. **Appendix.** Move there:
   - the per-session cost model and the newsvendor threshold;
   - the retention index and the dual bound, if kept;
   - the FA3 cost fits;
   - the replay and simulation models.
4. **Drop or rewrite these claims.**
   - "Storage holds every prefix cache" (design.tex:99). It is false once the volume fills (campaign 34: first pass at minute 88).
   - r_i = bL_i/beta. The read line dominates: reads were parked 17-67 s in campaign 34.
   - "Beyond b/(2 rho beta) reading is cheaper, and agentic contexts easily pass it." On FA3 the crossover is about 33-37 K tokens, above the 24.9 K mean history (inferred), and it depends on load.
   - "A read holds L_i tokens from its start." Parked reads hold only an anchor; pages are allocated at dispatch.
   - The price update as a main section: lambda never moved on FA3 (campaigns 32 and 34).
   - "Recency is a weak proxy": no TTFT gain from forecast-based eviction was shown in any campaign.
   - Preempt as a separate rule.
   - Any claim that reads before arrival carry the gain. They are shown only in campaign 26 at 44 users with exact hints, 0.14-0.21 s.
5. **Add these.**
   - The mirror and queue-room finding.
   - The write path as a design component. Campaign 34 against campaign 32 is -5.8 to -9.5 s, the largest measured effect on FA3.
   - The chain-aware SSD tier.
6. **Evaluation.** Report:
   - SSD read volume and read-line use;
   - TTFT of requests that read, and p90;
   - throughput;
   - the host census over time.
   Mark where the 128-user cap binds (minute 84-85). After it the box is a closed loop with a full GPU, and the mean is pinned by GPU throughput.

## experiments
- **E1 Host room: the mirror against eviction order**: Q: How much of the SSD read demand at 128 users comes from the write-through mirror, and how much can eviction order win back?
  - method: CPU, offline. A first version is done: scratch scripts/host_room_replay2.py, outputs in out/host_room_replay_*.out. It replays the three campaign 34 timelines open-loop. Arrival and admission come from entry_time and queue_duration in server.log; finish and history (ideal_cached) come from client.jsonl. Host pool is 1.61 M tokens, with session-level units. Arms: mirror counted (write-through) or not (exclusive); LRU with queued last, queued farthest first, or oracle next use with session ends; plus write-through with 0.65 M more host memory as a control. Next steps: page-level prefixes with shared heads and leaf-only eviction; campaign 32's six runs and the running hicache rerun; and a read line of 17.5-21.5 K tokens/s that turns demand into lateness.
  - expected: First version, per run: write-through with LRU needs 56.6-63.3 M tokens of SSD reads, within about 20 % of the recorded 53.0-67.3 M. Exclusive needs 24.6-24.7 M (-56 to -61 %), the same as adding 0.65 M of host memory. Oracle next use with ends saves 22-25 %, and both together save 74-78 %. Under exclusive tiering, late-run demand is 7.5-7.9 K tokens/s, below the read line's 17.5 K.
  - decides: Whether reclaiming the mirror comes first among engine changes, and whether forecast-based eviction gets main-text space or becomes a tie-break.
- **E2 Closed-loop effect on TTFT**: Q: With the GPU full and the 128-user cap binding, does cutting SSD reads move mean TTFT, or only the TTFT of requests that read and the p90?
  - method: CPU discrete-event model. A closed-loop client replays the 729 users with recorded waits (gap cap 1,200 s, cap 128). The GPU is bound by its 668 K-token pool, with FA3 costs from the campaign 32/34 fits: prefill 0.0374 ms/token plus 27-35 ms per batch plus a prefix term; decode 3.5 + 0.47 x running + 0.016 x context-K ms. The host model is E1's. One read line shares 2.1 GB/s with writes, and admission is wait_complete. Arms: write-through against exclusive. Validate on campaign 34's per-minute request counts, queue lengths, and the waits of requests with and without a read.
  - expected: After minute 85, mean TTFT barely changes because the GPU is the bottleneck. The wait of requests with a read falls from 46-69 s toward the 12-19 s of requests without one, and p90 falls. Before the cap binds (minutes 60-85), mean TTFT falls by the share of stall that reads cause.
  - decides: Which metrics the paper reports for storage changes, and whether a GPU run of exclusive tiering justifies the engine work: an async D2H at GPU eviction or at finish, and SSD lookups from GPU-only anchors.
- **E3 Write volume and dirty memory by write policy**: Q: How much SSD write bandwidth do dead pages and eager writes cost, and is delayed write-through safe?
  - method: CPU replay of page chains per conversation, from client.jsonl plus the h2s_submit lines (node, tokens, time) of the three campaign 34 logs. Arms: eager write-through, validated against the 31.6-31.9 M tokens submitted per run; skip outputs that are never re-sent; skip the last turn of ended sessions; delayed write-through, writing a page when it nears E1's eviction frontier. Track the dirty tokens over time against a clean-reserve rule.
  - expected: Outputs are 15 % of writes in this replay. Ended sessions' last turns are a few percent. Delayed writes save more, but during bursts they need dirty memory well above the 50 K hold.
  - decides: Keep eager write-through. Build dead-page skipping as a per-harness learned flag. Drop write-back from the plan unless the dirty set stays bounded.
- **E4 SSD cleaning order**: Q: Which cleaning order stops live sessions from losing their SSD chains?
  - method: CPU replay of stored pages. Page keys come from token chains, shared heads included, written in h2s_submit order; lookups come from s2h_query. Simulate the volume as run (about 25 M tokens) and at 12 M and 6 M, to emulate a long-running service. Orders: oldest write first (today's; it must reproduce the 48 zero-head and 18 partial losses after the first pass); oldest write plus touch on read; leaf-first by last use; leaf-first with ended sessions first.
  - expected: Touch on read leaves shared heads exposed, because they are used from memory and never read. Leaf-first removes the zero-head losses. With ended sessions first, no live page is deleted at 25 M (16.2 M dead against 4.7 M live at the first pass), and most losses disappear at 6-12 M.
  - decides: Whether to build an in-memory index of stored pages for the cleaner, and whether the SSD tier gets its own design subsection.
- **E5 Device scheduling at saturation**: Q: With reads and writes on one 2.1 GB/s device, how much do I/O priority and backward scheduling change how late reads land?
  - method: CPU discrete-event model of the read line (deadline-ordered parked queue, 512-page window, FIFO read thread) and the write thread, sharing 2.1 GB/s. Feed it campaign 34's minutes 60-120: reads from srs_park (time, pages, deadline) and writes from h2s_submit. Arms: as run (FIFO writes, 50 K hold); writes yield to deadline reads under a clean-reserve limit; outputs not written; reads before arrival only in device slack.
  - expected: Priority helps only during bursts. Saturation persists, because writes are mandatory at about 4.3 K tokens/s. Skipping outputs adds a few percent of read rate. Reads land on time mainly when read demand falls (E1).
  - decides: Whether an I/O priority layer belongs in the design, or a clean-reserve hold is enough.
- **E6 Read-time estimator**: Q: How far is the engine's r from how long a read actually takes, and does a backward estimate in deadline order close the gap?
  - method: Offline over the campaign 34 logs. For each read, compare the load estimates at the plan and at the hold (ntp_hold, ec_start) with the time from park to landing (srs_park, srs_dispatch, s2h_io). Recompute r at park time from the parked pages ahead in deadline order, the pages in flight, the write-hold state and the measured read rate.
  - expected: Today's r misses the parked time, a median 17-30 s late in the run. The backward estimate lands within a few seconds, outside write-hold episodes.
  - decides: The r term of the unified fetch rule, and whether latest-start times replace the per-read hold.

## risks
- Exclusive tiering means changing the tree core. Under write_through, a GPU-evicted node with no host copy is deleted (unified_tree_core.py:1630-1640). The existing write_back path blocks the scheduler on each D2H copy (unified_radix_cache.py:983-990, 3745-3758). SSD lookups need a host-backed anchor (scheduler.py:3160-3167). Copying back at finish costs about 6 GB/s of D2H at about 2.3 requests/s (inferred). Scheduler stalls could cancel the room gained.
- The host-room replay is open-loop and session-level. It holds W fixed and ignores shared prefixes, partial eviction, pins and read timing. The -56 to -61 % is a first-order estimate, not a prediction of TTFT.
- Once the user cap binds, the box is a closed loop with a 96-98 % full GPU pool. Changes that do not cut recompute may leave mean TTFT flat and move only the tail (inferred). The paper has to choose its metrics with that in mind.
- Dead output KV is an artifact of the replay: next prompts carry recorded replies, so only 54 of about 16,400 reuse the generated output. Real harnesses usually re-send outputs, so the 15 % write saving may not transfer. The artifact also inflates writes in every arm, and the plan key carries the dead tail. That tail may explain some ntp_reevict events; not traced.
- Delayed write-through or write-back risks campaign 32's loop, where dirty pages blocked eviction and pending writes reached 240-675 K tokens. A hard dirty limit tied to the clean reserve is required.
- SSD-tier effects appear only after the volume fills (minute 88 in a 2 h run). Testing the cleaner order needs lowered watermarks or longer runs, which change the regime. The page index must be rebuilt after a restart.
- There is one run per arm and no measured spread in the saturated part. Campaign 34's arms differ by at most 0.7 s.
- Backward scheduling relies on deadlines, and W predicted from position x rate lags a growing queue (a median 19.8 s short for 30-60 s waits). Latest starts can land late unless a conservative quantile is used.
- Once dispatched, reads go to one FIFO read thread, and 'now' reads bypass the 512-page window. Any priority scheme can reorder only the parked queue.
- The brief names Qwen3-32B-FP8. Campaigns 28-34 actually ran Qwen3-30B-A3B-Instruct-2507 with bf16 KV, 98,304 B per token. Any b/beta figure in the paper must use the latter.