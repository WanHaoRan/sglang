# Four actions, three prices: the unified design for the three-tier KV cache (v2, 2026-11-26; measured 2026-11-27)

Status: design after the checks of 26 November (math, literature, logs, engine code maps). It replaces draft v1
(`DESIGN_v1_draft.md`). Proofs: `PROOFS.md`. Checks: `main/results/*.json` (one file per checker) and `agents/`.
Engine work and GPU comparison: section 11.

The author's brief (2026-11-26): keep the paper's four actions (Fetch, Preempt, Retain, Price); bring in the queueing
delay, the hint (a distribution of the tool call's duration) and the SSD write; let the four actions govern read timing
and order, eviction, and write timing and order across GPU memory, host memory and SSD; make the design provably
optimal under a stated model; consider every gain, however small.

## 0. The design in one page

A session's prefix cache (the KV of its context) is next needed on the GPU at T = A + W: the arrival A of its next
request, which the agent knows as a distribution (the hint), plus the wait W in the GPU queue, which the engine knows.
Every movement of KV between tiers is a job with a due time derived from T: a read (SSD to host) is due at T, a write
(host to SSD) when the page's host copy is given up, a backup (GPU to host) when the GPU gives the page up.
Three shared resources carry the jobs, and each has a price in seconds of TTFT:

| Resource | Price | What one unit costs | Measured (campaign 34) |
|---|---|---|---|
| GPU time | mu, per GPU-second | every waiting turn waits too | 1.0-1.3 at light load; N - B = 109 at saturation (measured 91-102) |
| Host memory | lambda, per token-second | the best use the room is kept from | 0 with free room; 2e-7 at light load; about 0 at saturation |
| SSD time | kappa, per device-second | the transfers waiting behind | about 0 in every regime measured |

and one weight: c, what a second of one turn's stall costs the sum (1 when the turn waits alone, 0 when other ready
turns take the GPU meanwhile; measured 0.94-0.98, then 0.06, then 0.00).

| Action | GPU | Host memory | SSD |
|---|---|---|---|
| **Fetch**: when a copy moves up | start the copy from host one copy time before the prefill pass | start the read so it lands at a quantile of T; before arrival only on a sharp hint | serve reads in due-time order; read or recompute by price |
| **Retain**: which copies stay | evict by the same index | keep while the index nu = (K0/L)/(tau* + W - rho) is above lambda | write = keeping the copy one tier down; cleaner = the same index on the SSD |
| **Preempt**: who yields now | pass over a turn whose read or load has not landed; decode meanwhile | lowest index yields: ended, idle, staged, queued with slack; never the head | a late write before a late read iff (lambda - nu_clean) L_read > c |
| **Price**: what a unit costs | mu and c from the GPU queue | lambda = index of the marginal holder | kappa from the device's line |

Three results organise everything else:

1. **Each action is the exact optimum of a priced single-session problem** (sections 4 to 6): a quantile rule for
   Fetch, an index rule for Retain, a two-lives rule for the write. Belady, LRU, the newsvendor quantile and
   earliest-due-date come out as special cases.
2. **The queue enters through one identity** (section 2): with the users in a closed loop and the GPU never idle,
   mean TTFT = (N - B) g - Z': it moves only with the GPU-seconds per request g. So at saturation a cache policy can
   change the mean only through recompute and GPU idleness; read timing only permutes who waits.
3. **A floor follows** (section 9): no policy with the same GPU does better than (N - B) g_min - Z'. On campaign 34
   the current design is 0.45 s (4 %) above it, and two thirds of that is one cause, the SSD cleaner deleting live
   sessions' pages.
4. **The same rule holds one tier up** (section 4.5): g itself contains the tier moves. The copy from host memory to
   the GPU is a Fetch that today lands inside the prefill pass and blocks the GPU for 40 to 53 ms per admitted
   request, 8 to 12 % of GPU time at saturation, three times the recompute the cache policies fight over. Started
   first and passed over like an SSD read, it costs the turn the same wait and the GPU nothing.

## 1. Model

### 1.1 Sessions and hints
- N sessions are in progress. A session alternates a turn (one request: the whole conversation so far) and a gap
  (the agent runs the tool the reply asked for, or a person answers). The turn ends at e; the gap is G; the next
  request arrives at A = e + G.
- The hint: with each request the agent framework sends a table. For each kind of tool call k (the tool and the
  command's first word; `_no_call` for a reply with no tool call) it holds the p10, p50, p90 of past gaps and
  whether the session is over. The engine reads k off its own reply and fits a distribution Phi_k (a two-piece
  lognormal through the three quantiles), with an atom at infinity for a session that may not return.
- At idle age a = t - e the engine uses R = G - a given G > a (the time left), F(x | a) = P(R <= x) and the return
  rate h(a) = phi_k(a) / (1 - Phi_k(a)). tau(x | a) = E[min(R, x)].
- A hint is **sharp** where h rises with a, and **heavy-tailed** where it falls. A lognormal's rate rises to a
  peak and falls after it; the peak is past the median only if p90/p50 < 2.78 (sigma < 0.80).

### 1.2 Tiers and transfers
- GPU memory: C_G tokens of KV. Host memory: C_H tokens. SSD: C_S tokens and one device of beta tokens per second,
  shared by reads and writes.
- A page (64 tokens of KV, 6 MiB) may have a copy in any subset of the tiers. A session's prefix is a chain of
  pages; a lookup on the SSD stops at the first missing page; chains share heads (first prompts).
- Transfers: read (SSD to host: the device's line V, then L / beta, then small lags), write (host to SSD), backup
  (GPU to host), load (host to GPU, at admission), recompute (GPU time gamma(L) for L tokens).
- A host page is pinned (a landed read waiting for its turn), dirty (no complete SSD copy), clean, or a mirror
  (the GPU still holds the same page).

### 1.3 What a returning request waits for
TTFT = W + S + gamma (Delta + U): the queue wait W until the GPU would take the turn if its KV were in memory; the
stall S = (l - D)^+ if its read lands at l after that moment D = A + W; and prefill of the Delta new tokens plus U
tokens of history that were lost and not read back.

### 1.4 The joint problem
Choose at every instant which transfers to start and which copies to drop, to minimise the expected sum of TTFT,
subject to: host tokens <= C_H, device work <= beta, SSD tokens <= C_S, GPU pool <= C_G, and admission in arrival
order among turns whose prefix is in memory (a turn whose read has not landed is passed over).

## 2. The queue in the model

### 2.1 The price of a GPU-second
The GPU is a pool-bound batch: at most B turns run at once; prefill is serial and pauses decoding; a decode step
gives each running turn one token. A user sends its next request only after the reply and the gap, so the loop is
closed and, for every policy, N / X = R + Z (R: mean time from a request to the end of its reply; X: requests
completed per second; Z: mean gap). With the pool always full and never short of ready turns, X = 1 / g with g the
GPU-seconds per request, the time from admission to the last token is B / X, and
mean TTFT = (N - B) g - Z + p (p: the request's own prefill). One more GPU-second per request raises the response
time by N and the TTFT by N - B.
So mu = N - B at saturation: 109 in campaign 34 (128 users, 19 running), measured 91 to 94 over minutes 90-110 and
99 to 102 in the last ten minutes (the rest is pauses that had not settled). Below saturation a job of g seconds
delays its own turn, the ready turns waiting and the turns that arrive during it: mu = 1 + ready + rate x g / 2,
measured 1.7 to 2.8 for a history-sized recompute at light load. In an open queue the same price is
1 + (Q + rate x g) / (1 - rho). The pool fills at about N = B + Z / g = 68 users in session on this box: below about
50 users a GPU-second costs 1 to a few, above about 72 it costs N - B. The engine's constant counted
"1 + turns waiting", three times too low at saturation.

### 2.2 The weight of a second of stall
Admission passes over a turn whose read has not landed. If another ready turn takes the slot, the sum of waits does
not change (exactly for equal GPU work, on average for exchangeable work); if the slot stays empty, the stalled turn
loses its own second and the GPU loses a little throughput. In the closed loop the exact statement is: delta mean
TTFT = (N - B) x (GPU time lost per request). So a stall has no cost of its own: it costs mu times the GPU time it
wastes. We write c for the resulting weight per second of stall: for this batch GPU, the chance that the slot
stays empty. (For a single work queue the weight is different: 1.0 to 1.6 up to 94 % utilisation.)
Measured as the share of admissions that leave no ready turn waiting, c is 0.94-0.98 until the GPU saturates
(minute 28), 0.06 in minutes 28-70 and 0.00 after. Two cautions. Before the user cap binds the loop is not closed,
and one arm's extra stall (jit_p50, +3.2 s per request) came with +0.8 to +1.3 s of TTFT, an effective weight of
0.1 to 0.35 on one run per arm. And the mean hides who waits: with reads 40 s late a batch model keeps the mean at
31.2 s while readers wait 47.8 s and the others 25.7 s. The engine therefore keeps a floor under c.

### 2.3 What matters when
| Regime (campaign 34 minutes) | c | mu | What a cache policy can still move |
|---|---|---|---|
| Room free (0-8) | 0.98 | 1.0 | nothing: everything is in memory |
| Light load, room short (8-28) | 0.94 | 1.3-2.8 | each reader's own stall: 0.5-2.7 % of returning requests read |
| GPU saturated, SSD keeps up (28-70) | 0.06 | 71 | recompute only; stalls are absorbed |
| Read line filling (70-90) | 0.00 | 102 | recompute; who waits (readers' tail) |
| GPU and SSD saturated (90-110) | 0.00 | 109 | recompute; 18-20 reads late at any time, 27-31 s of stall per reader |

## 3. The priced problem
Relax the capacity constraints to hold on average, with multipliers lambda (host token-seconds), kappa (device
seconds), sigma (SSD token-seconds); take mu and c from section 2. The objective separates by session:

    J_i = E[ c S_i + mu gamma U_i + lambda * integral m_i dt + kappa * (device seconds of i) + sigma * integral s_i dt ].

Each session minimises its own J_i; sections 4 to 6 solve that problem action by action, and section 7 says where the
prices come from. In the open model (sessions arrive independently, mu and c given) the sum of the sessions' optimal
J_i minus the priced capacities is a lower bound on every policy that meets the capacities with the same hints
(weak duality). In the saturated closed loop that bound fails (more GPU work for one user can lower the others'
response times); the bound there is the floor of section 9, item 9.

One restore cost is used throughout. K0 = kappa L / beta + c E[(r - W)^+] is the price-free cost of restoring a
clean prefix by a read (r = V + L / beta + lags); K_rec = mu gamma(L) is the cost of recomputing one that has no
copy. Restore by the cheaper. On FA3 the request's own times tie near 44-55K tokens (an uncontended read takes
0.05 s + 0.044 ms a token; a cold prefill 0.03 s + 0.012 ms a token + 1.4 us a token per thousand tokens of
position), so at mu = 1 recompute wins below that length, at mu = 2 below 17-21K, at mu = 5 below 3K, and at
saturation the threshold is a few pages: the engine's fixed 256-token read threshold is the priced threshold at
mu = 128 and c = 1.

## 4. Fetch: when a copy moves up

### 4.1 After arrival: land at a quantile of the admission time
The turn has arrived; only D is uncertain. A read that lands at l costs c (l - D)^+ + lambda L (D - l)^+. The
minimum is at P(D <= l*) = q, q = lambda L / (c + lambda L): the newsvendor quantile. The read is released at
l* - r. The engine's fixed 2 s margin is the 20th percentile of the measured error of its estimate of D
(campaign 34, four arms: +1.6 to +2.1 s) and within 1 to 4 % of the best cost for lambda / c near 2e-5.
At saturation both lambda and c go to zero and q is a ratio of noise: q is clamped to [0.05, 0.5].

### 4.2 Before arrival: the hint's shape decides
Let x = (r - W)^+ be the part of the read the queue does not hide. If x = 0 nothing is read before arrival. For a
read started at s with recourse (a turn that arrives first starts its own read):

    dJ/ds = P(A > s) [ c F(x | s) - lambda L (1 - F(x | s)) ].

- Sharp hint (rate rising over the next x seconds): J is quasi-convex; start when F(x | s) >= q. This is the
  paper's threshold.
- Heavy-tailed hint (rate falling): J is quasi-concave; the best start is an end point, "now" or "at arrival".
  Reading now is the same as not evicting, so on such a hint there is no timing decision before arrival, only
  Retain's keep-or-drop. If the session may have ended, the stall saved counts only finite returns.
- The value of any read before arrival is at most c (r - W)^+ per returning request: the queue makes read timing
  worthless once W >= r, whatever the hint.
On FreeInference the rate falls beyond 2 s for the keys of 94 % of requests (sharp keys within a read time:
web_search, sleep, ps; 1 to 3 %). A timed start adds 0.000 to 0.002 s per returning request. Where gaps are sharp
(campaign 26: x10 gaps, p90/p10 median 3.7) the ceiling was 0.23 to 0.29 s and exact hints took 0.15 to 0.22 s.

### 4.3 Order on the device
Reads share one device. If some order lands every read by its due time, earliest-due-time order does (Jackson's
rule; with releases, within one page-group transfer of the best maximum lateness, about 1.5 s). Once some reads
must be late no simple order is optimal for the stall sum (weighted tardiness is NP-hard); the rule is then to
serve by cost rate over max(size, time to due): it is Smith's ratio when all are late, and 1.24 times the optimum
on mixed instances against 2.28 for a hard switch. With c = 0 the order decides only who waits; earliest due time
then also minimises the longest reader stall.

### 4.4 Read or recompute
Restore by the cheaper of K0 and K_rec (section 3). Two regimes follow. With a queue (c near 0, mu large) every
span of a page or more is read. With no ready turn waiting, a recompute uses GPU time nobody else is waiting for,
but it also delays the turns that arrive during it: on FreeInference 0.7 to 2.2 turns arrive during a history-sized
recompute and lose more than the request gains, so the rule would fire only for short spans (under 16-24K tokens).
When the SSD is the bottleneck and no queued turn is ready, the GPU idles while reads wait in line: then
recomputing the first turn whose read has not been dispatched is free GPU time (the restore mix: read at the
device's rate, recompute the overflow).

Verdict: off. The gain is bounded by 3 to 19 TTFT-seconds per two-hour run (0.001 s per request) and the simulator
shows a loss in every setting: +0.12 to +0.59 s for the rule restricted to moments with no ready turn waiting and
c at 0.8 or more, +1.2 to +18.8 s for the first draft's rule (section 7's caution: rent charged on the device's
line, and mu taken as 1 + ready turns). The engine's fixed 256-token threshold is the priced value at saturation
and stays. The switches exist (`..._RECOMPUTE_BY_PRICE`, `..._RECOMPUTE_WHEN_STARVED`) and are not part of the
unified arm.

### 4.5 Fetch on the top tier: the load from host memory
A turn admitted with its prefix in host memory copies it to the GPU (2.5 us a token: 57 to 66 ms for the 22.6 to
26.2K tokens that 95 to 98 % of admitted requests load at saturation). The engine starts the copy when the prefill
batch is built and the prefill forward waits on it layer by layer; nothing decodes meanwhile. Measured on campaign
34: a decode interval that holds one small prefill batch is longer than a pure one by the copy's own time (slope
0.88 to 0.96), 40 to 53 ms blocked per admitted request, 9.7 to 12.1 % of GPU wall time (regressions on intervals
give 5 to 8 %). In minutes 90 to 110 that is 123 to 137 s of blocked GPU against 43 GPU-seconds of recomputed
history. Through (N - B) g: 2.8 to 3.2 s of the 10.96 s run mean (inferred; 1.9 to 4.2 s on the simulator).

It is the Fetch problem of 4.1 one tier up, with one difference: a late landing does not stall one turn at weight
c, it stalls the GPU at weight mu, because the forward waits. The two actions that fix it are the ones the SSD tier
already has. Fetch: release the copy one copy time before the pass that needs it. Preempt: pass over a turn whose
load has not landed and run the next decode step. The turn then waits about as long as before (the copy time) and
the GPU decodes beside the copy. In the engine: a prefill batch whose load is 8,192 tokens or more is held while
decode steps run, and prefilled once the copy's last layer has landed, after a bound of steps, or when nothing is
left to decode (`SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS`; `scheduler_components/prefill_load_hold.py`).
One rule covers every tier boundary: a transfer that is not complete never blocks the resource above it.

Measured (campaign 39 against campaign 35, the current design with and without the hold, same users, 2 h): mean
TTFT 9.36 s against 10.97 s, -1.61 s [-1.79, -1.43] on 15,560 paired requests; nothing at light load (-0.008 s in
the first 28 minutes), -2.5 to -3.5 s in minutes 50 to 90. On the GPU's own log the slope of a decode interval's
length on its prefill batch's copy time falls from 0.71 +- 0.05 to 0.00 +- 0.04, the GPU runs 4.8 % more decode
steps a second, and 7,059 of 7,060 held batches are released with the copy landed. The closed loop's accounting
(TTFT = requests in flight / throughput - decode time, per ten-minute window) gives -1.34 s against -1.44 s
measured on the same windows. The inference above (2.8 to 3.2 s) and the simulator's free copy (-2.6 s) both
charged the whole copy; the pass's own compute hides part of it, as Proposition 23 says.

The best start of the pass is known exactly (Proposition 23). The forward takes the layers in order and waits for
one that has not landed, so a pass of p seconds of compute started at s beside a copy that ends at r finishes at
max(s + p, max(r, p)): the engine's default (s = 0) waits (r - p)^+, and the start s* = (r - p)^+ finishes at the
same moment without waiting. An earlier start wastes GPU time and a later one delays the turn, whatever the prices:
on this tier Fetch needs no price because the landing time is known, where on the SSD tier it is a quantile. On
campaign 35 a batch costs the running turns max(copy, 24 ms + 40 us x new tokens), the copy outlasts the pass in
80 % of batches, and the pass waits 38 ms a batch. The rule built starts the pass at the first decode-step boundary
after the copy has landed: it wastes no GPU time and delays the held turn by its pass plus half a step (about
37 ms; measured +8 ms [+5, +11] on the quick requests of campaign 39's first 20 minutes, where few batches are
held). Starting at the first boundary where the copy's remaining time is at most the pass (the meeting rule) would
keep the saving and leave half a step of delay (about 12 ms): 26 ms per held turn, 0.01 to 0.03 s of mean TTFT,
below what a two-hour pair resolves; derived, not built.

The two tiers have one structure. A transfer is exposed for the time its cover does not hide: (r - W)^+ of an SSD
read behind the queue wait W (Theorem 1), (r - p)^+ of a host-to-GPU copy behind the pass's own compute p. The
exposed time costs the price of what it blocks: c for one turn on the SSD tier, mu for the GPU on the top tier.
Fetch starts the transfer so that the exposure is smallest at the price of holding the copy early; Preempt gives
the blocked resource to other work during the exposure.

## 5. Retain: which copies each tier keeps

### 5.1 The holding index
A clean prefix of L tokens is in host memory at idle age a. Keeping it costs lambda L per second; if it is dropped
and the session returns, K is paid. With "not back yet" as the only news:

    keep while nu(a) > lambda,   nu(a) = (K / L) sup_h F(h | a) / tau(h | a),

and drop at the first age where it fails. nu is the Gittins index of a job of unknown size (the gap) and the
Whittle index of the session; geometrically it is the slope of the lower convex hull of the curve
(E[min(G, t)], P(G > t)). Special cases: exact hint, nu = K / (L R0) (Belady); rate falling, nu = (K / L) h(a)
(LRU under one law); ended session, nu = 0; two modes, the index looks across the dead stretch (a rule that reads
only the current rate costs 1.8 times more on a 1 s / 40 s gap).

With the queue, one formula covers idle prefixes, staged reads and queued turns:

    nu = (K0 / L) / (tau* + W - rho)^+ ,

tau* = inf_h tau / F the expected holding time per return (0 for a turn that has arrived, R0 for an exact hint,
1 / h for a falling rate, infinite for an ended session), W the queue wait, rho the time the restored copy would
itself sit in memory before admission. It ranks holders by time to next use on the GPU net of the restore's own
hold. Two corollaries. A long queue flattens the value of hints: for the tool entry tau* is 2.6 s at age 0 and
11.9 s at age 10 s, a ratio of 4.6 with no queue and 1.41 with W = 20 s; this is the measured tie of the hinted
arms. And among queued turns the latest arrival has the lowest index: under FCFS it is admitted last. (The engine
ranked queued turns by recency of use, which takes the history of the turn nearest admission.)

Under a hard capacity "evict in increasing nu" is exact for exact hints with equal sizes and costs, and for one
falling-rate law with equal sizes (LRU is then optimal among policies that do not see the future); otherwise it is
a ranking whose gap to the priced bound shrinks with the number of sessions (+24 % at 32, +6 % at 128, +2 to 3 %
at 512 to 2,048).

### 5.2 The write is retention one tier down
A dirty page costs K_rec if its host copy is lost; a clean one K0. A write (device time w = kappa L / beta) buys
the difference. Its optimal policy has two lives: (W) hold, write, drop; or (N) hold, drop unwritten. W is chosen
iff V(K_rec) - V(K0 + w) >= w P(ended), V(K) the optimal holding cost with restore cost K. Consequences:
- An ended session's page is never written.
- At a constant device price a write is never better early than late: the write point is the eviction frontier.
  With an idle device it is free, so eager writing is optimal; and at saturation eager wins as insurance as soon
  as a forced eviction of the only copy has a 0.4 % chance.
- A write that is late costs L (lambda - nu_clean) per second, zero at its due time: writes go lowest clean index
  first (the size cancels), which for one falling-rate law is the order of a FIFO queue; a late write goes before
  a late read of L_read tokens iff (lambda - nu_clean) L_read > c. The engine's fixed hold (timed reads wait while
  pending writes hold 50K tokens) is the case nu_clean = 0.
- A page whose key the SSD already holds is clean at birth and is not written again; a page counts as clean only
  on a full acknowledgement. Without the first, recompute feeds rewrites, rewrites hold room and room refusals
  feed recompute (campaign 32's loop).
On this workload 57 to 74 % of written pages are ever needed (model) against 56 to 67 % read back (measured); a
deferred write would skip the rest, about 1.1 to 1.8K of 4.3K tokens/s, and the device is idle enough that eager
writing costs nothing.

### 5.3 One tier up and one tier down
- GPU to host. The backup is due when the GPU gives the page up; made at the end of prefill, 0.65M of 1.63M host
  tokens mirror the GPU. The mirror's index is (cost of a late backup) / (L x time to GPU eviction): the eager
  backup is optimal if the late copy would block the scheduler (about 6.6 TTFT-seconds per request at saturation),
  and wasteful if it could run aside.
- SSD capacity. The cleaner is eviction on the SSD: the same index with K = K_rec and the chain structure. A
  lookup stops at the first missing page, so a chain is worth what its head is worth, and heads are shared: 94.5 %
  of the pages two or more sessions hold lie in a first prompt, and 60 % of the shared tokens new sessions found
  (1.38 of 2.30M) were held only by sessions already over. The index therefore gives three classes, deleted in
  this order: (1) an ended session's pages past its first prompt, worth nothing; (2) ended sessions' first
  prompts, worth their reuse by sessions not yet arrived; (3) open sessions' chains by last use, each refreshed
  as a whole. The engine's cleaner deletes the oldest-written file, which is a head: 7,038 live pages in three
  passes, 6,793 of them shared.
  A page-level replay of campaign 34 (it reproduces all 9,103 lookups, the 10 failed reads and every request's
  cached tokens) gives, in seconds of mean TTFT in minutes 90 to 110 (ec_full / jit_exact / jit_p50):
  | Order | Run's volume | Half | Quarter |
  |---|---|---|---|
  | today: oldest written first | 1.91 / 1.70 / 1.63 | 2.7 to 3.7 | 8.0 to 8.8 |
  | ended sessions' whole chains first | 3.6 to 4.0 | 2.3 to 2.4 | 3.3 to 4.3 |
  | chains refreshed on use (per request, or once per session per 300 s) | 0.07 to 0.13 | 0.29 to 0.32 | 2.4 to 2.7 |
  | refresh + class 1 | 0.00 to 0.03 | 0.05 to 0.07 | 2.1 to 2.3 |
  | refresh + classes 1 and 2 | 0.00 to 0.03 | 0.05 to 0.06 | 0.74 to 0.98 |
  | the same with reference counts over open chains, a touch on lookup hits, a re-check before the unlink | 0.00 | 0.04 | 0.40 to 0.46 |
  | oracle (farthest next use, never a page with a live descendant) | 0 | 0 | 0 |
  The oracle loses nothing down to a quarter of the volume (open sessions' latest chains are 3.9M of 22.6M stored
  tokens), so capacity is not the constraint: every loss is a failure to tell live pages from dead ones. Retiring
  an ended session's whole chain is worse than today's order. A refresh costs 216 pages (431 file touches) a
  second at one per session per 300 s; touching only what a lookup finds (283 pages a second) does as well at the
  run's volume and falls behind below it. One deleted file fails its 128-page read batch and every later batch of
  that read (10 reads lost up to 51K tokens each), which the re-check before the unlink removes. An in-memory
  index of stored pages adds nothing for the cleaner (3 % or less). The replay is open loop: it cannot show how
  less recompute changes the queue. Scripts and outputs: `agents/ev_ssd/`.

## 6. Preempt: who yields when a resource is full now
- Host room. Freeing m tokens at least cost takes holders in increasing index (exact for the divisible problem,
  within one holder otherwise): ended sessions; idle clean prefixes; staged reads of sessions not back; queued
  turns, largest slack first; never a turn whose re-read could not land before its admission, a dirty page or a
  read in flight. If nothing can yield the read waits; a refusal turns into a recompute at mu.
- The device. Section 4.3 for reads; section 5.2 for a write against a read.
- The GPU. A turn whose read has not landed is passed over, so the GPU never waits for the SSD while ready work
  exists. This is what makes c small under load.

## 7. Price: where the three prices come from
- mu: N - B once ready turns always wait (the prefetcher sees N - B as its plans plus the queued turns without
  one); 1 + ready + rate x g / 2 for a job of g seconds otherwise.
- c: the decayed share of admissions that leave no ready turn waiting, with a floor.
- kappa: c times the reads past their latest start, plus lambda times the tokens pending writes hold.
- lambda: the index of the marginal holder, read off the eviction frontier, zero while room is free. It is the
  multiplier of the room constraint in the relaxed problem (the index at which the holders' total demand meets the
  room), so it is computed, not learned from refusals. Measured: 2e-7 at light load, under 1e-8 once the GPU
  saturates; the engine's fixed 5e-5 was 240 to 640 times the light-load value, which is why rule 1 staged one
  read in a two-hour run.
- sigma: the index of the cleaner's marginal victim; zero until the volume fills.
Of the four, only mu is large where the queues are long.

GPU room has a price too, and it is the dearest room: in the pool-bound model an empty slot costs about 2
TTFT-seconds per slot-second (one for the turn that could run, one for the throughput lost), about 6e-5 per
token-second for a 33K-token slot, some 50 times the host prices measured. Two consequences, neither built yet:
admission should not stop at a ready turn that does not fit while a later ready turn does (2 % of the pool idles
behind such a head; 0.3 s on the simulator), and the GPU's own retention of a finished turn's KV is priced at this
rate (about break-even here: the next admission follows a finish within 0.03 s).

lambda must be read off the price-free form of the index (5.1): K0 there holds the restore's stall and device time
only, and the restore's own rent enters through rho. An index whose K contains lambda L r feeds itself when its
frontier is taken as lambda: the estimate grows without bound once (read time) x (return rate of the marginal
holder) reaches 1, and in the simulator it sat at the engine's ceiling of 1e-3 from minute 60. The same care
applies to "read or recompute" (4.4): a parked read holds no host memory, so its rent is lambda L times the hold
after landing, not times the read time with the device's line in it; with the line in the rent the rule chooses
recompute at saturation (simulated: 3,403 recomputes by choice, mean TTFT 13.3 s against 10.6 s).

What the GPU run showed (campaign 40, everything on, 128 users, 160 GB). Once saturated the engine measured mu 62 to
107, c 0.01 to 0.07 and lambda 1e-8 to 7e-7. The lambda that came out is too low, and the reason is the restore
cost in the index: a displaced history is priced by the stall of reading it back, c x read time, and at saturation
c is small because other ready turns fill the GPU. What a displacement also does, now and then, is turn a later
read into a recompute (refused room, a read that cannot land in time), which costs mu x 2e-5 s, about 1e-3 per
token: a 0.1 % chance of it outweighs the read term. With lambda near zero, staging before arrival looked free:
reads landed 6.3 s before admission (1.6 s in the current design with the hold), staged pins were held 9.2 s on
average, and seven copies were held until the engine's 600 s safety release because the hazard of a long gap
falls and no drop time was built (Theorem 3 b). The same arm with the fixed price (campaign 45) stages nothing
early (reads land 2.1 s ahead on average, against 14.0 s) and has the same mean TTFT (+0.04 s [-0.04, +0.13]): at
this setting the measured lambda wastes SSD reads and host room, not time, because host room is not what binds
once the GPU is saturated. Two corrections follow from the model: K0 = min(read stall, recompute) weighted by the
chance the read fails, and a drop time for staged copies; neither is built.

## 8. Regimes the rules reduce to
| Regime | Fetch | Retain and write | Seen in |
|---|---|---|---|
| Light load, sharp hints, room short | early reads timed by F(x | s) >= q | index order matters | campaign 26 |
| Light load, heavy-tailed hints | read at arrival; short spans recomputed | keep by return rate | campaign 34 minutes 8-28, campaign 33 |
| GPU saturated, SSD keeps up | no early reads; land at admission | never lose the only copy | campaign 34 minutes 28-70 |
| GPU and SSD saturated | due-time order; readers wait | fewer reads: order among queued turns | campaign 34 minutes 90-110 |
| SSD the bottleneck, GPU starved | recompute the overflow | as above | not yet run (campaign 37 probes it) |
| Host at the cliff | no speculative holds | writes before timed reads; wait, do not refuse | campaign 32 |
| SSD volume full | | chain-aware cleaning | campaign 34 from minute 88; campaign 38 |

## 9. What is proved
Statements and proofs: `PROOFS.md`. Part 0 there holds the final list after the referee, 21 theorems
and two propositions (of 39 results sent to the referee 33 stand as derived, 5 gained a condition, 1 stays open;
Proposition 23 was added afterwards). "Exact"
means for the priced single-session problem under the stated assumptions.
1. Fetch before arrival (Theorems 1 to 4, 6): the cost J_F of a start time and its derivative; the threshold
   "start when F(x | s) reaches q" is optimal for every price iff F(x | s) never falls (a rising hazard), and needs
   a finite mean gap; under a falling hazard the optimum is an end point, and a staged copy needs a drop time; with
   an uncertain queue wait the same derivative holds with the chance of being admitted before the read lands; a
   read before arrival is worth at most c (r - W)^+ against a copy that never left.
2. Fetch after arrival (Theorem 5): land at the q-quantile of the admission time, q = lambda L / (c + lambda L).
3. Retain (Theorems 7 to 9): keep iff nu(a) > lambda; first passage, dynamic consistency, indexability; nu is the
   Gittins index and a hull slope; Belady and LRU as special cases; one price-free formula for idle prefixes, staged
   reads and queued turns.
4. Many sessions (Theorem 10): exact for the priced problem; under the hard capacity the index order is a heuristic
   whose gap to the priced bound shrinks as sessions grow (cited for one constraint and identical types; 1 % at
   512 sessions, 28 % at 32 in our numerics). This is the open result.
5. Write (Theorems 11, 12, 19; Proposition 22): the two lives of a dirty page; never better early than late at a
   constant device price; a session that ended is never written; due date and order of writes; the smallest model
   of the write loop of campaign 32 and the two rules that remove it; the share of writes ever needed.
6. Device order (Theorem 13): earliest due time meets every due time if any order does; beyond that minimising the
   stall sum is NP-hard.
7. Read or recompute (Theorem 14): the exact rule and its crossover (built, off: it loses on the simulator).
8. The queue (Theorems 15, 16): the weights of a GPU-second and of a stall-second in an open queue; in the closed
   loop mean TTFT = N / X - Z - (running turns) / X + (admission to first token), and with the pool always full
   mean TTFT = (N - B) g - Z + p, so a stall changes the mean only through GPU time lost, and every policy that
   keeps the system saturated has mean TTFT >= (N - B) g0 - Z + p0 (tight). A running system's distance from the
   floor is (N - B) x (recompute seconds + loss from empty slots) per request: 0.45 s over campaign 34's run.
9. Preempt (Theorem 17): the greedy order solves the divisible room problem.
10. Price (Theorem 18): lambda is the marginal holder's index (complementary slackness); weak duality gives a
    lower bound for any prices in the open model.
11. The tiers above and below (Theorems 20, 21): SSD capacity with chained pages (tail before head; the index of a
    shared head); the backup from GPU to host is a job due at GPU eviction.
12. Fetch on the top tier (Proposition 23): a pass started at s beside a copy ending at r finishes at
    max(s + p, max(r, p)); s* = (r - p)^+ is optimal for every price; the rule built is within its pass plus a
    step of it and wastes no GPU time.
Not proved: optimality of the index order under the hard capacity in general; the exact weight of a stall when the
loop is not closed.

## 10. What the logs say (campaign 34, the current design, 15,562 requests)
| Lever | Action | At stake | Verdict |
|---|---|---|---|
| Prefill pass waits on the host-to-GPU copy | Fetch / Preempt (GPU) | 40-53 ms blocked per admitted request, 8-12 % of GPU time; 2.8-3.2 s of the run mean (inferred), 1.9-4.2 s simulated | build (largest by far) |
| Admission stops at a ready turn that does not fit | Preempt (GPU room) | 2 % of the pool idle; 0.3 s simulated | later |
| The cleaner's scan runs in the scheduler's process | Retain (SSD) | decode interval +17 to +39 % for 11 s per pass; 0.1-0.3 s while passes run | later |
| SSD cleaner deletes live sessions' pages | Retain (SSD) | 1.19M tokens recomputed = 0.33 s per request; 1.6-1.9 s in minutes 90-110; 2.7-3.7 s at half the SSD volume, 8.0-8.8 s at a quarter (page replay) | build (largest) |
| Sessions wrongly registered as ended (410 of 981) | Retain | their history is first out for a median 2.3 s; next turn has no plan | build |
| Queued turns lose resident history (111-171 per half hour late) | Retain / Preempt | 2.2-4.0M tokens read late per half hour; readers' tail | build (order among queued turns) |
| Lookups that find nothing | Fetch | 63-75 % of lookups; 0.007-0.16 s per request in recompute; 19-28 ms per turn at light load | build |
| Landing quantile for held reads | Fetch | at most 0.013 s; today's 2 s is the 20th percentile | build for completeness |
| Lateness weight and read line in the start rule | Fetch | 0 here; needed together or the rule reads early at saturation | build together |
| Read or recompute at an idle GPU | Fetch | 0.2-1.4 % of returning requests; +3 to +19 TTFT-seconds per run at best | build, off unless the GPU starves |
| Skip ended sessions' last write | Retain | 4.7 % of writes; no TTFT | build with the cleaner change |
| Write order, reserve-based hold, stored-first | Preempt | no target at this setting | later |
| Page rounding (under one page a request) | none | 380K tokens; not a policy's to move | none |
What is left to recompute in the best arms is 1.5 to 1.9M tokens (0.40 to 0.53 s per request), 63 to 71 % of it
from the cleaner. Everything else at this setting is below the run-to-run spread.

### What the simulator says
`sim/` is a discrete-event simulator of the queue, the three tiers and the device, calibrated on campaigns 34, 32
and 28 (mean TTFT of five arms within 1 to 3 %, 19 of 20 half-hour cells within 10 %, no constant fitted per arm;
`sim/CALIBRATION.md`) and refereed (`sim/referee/`: seven small defects patched; it can carry the sign and rough
size of effects of 0.5 s or more that hold in every seed). Each rule alone and the sets, paired with `ec_full` on
the same users, mean over five seeds on the patched copy, in seconds of mean TTFT (`sim/POLICY_RESULTS.md`); A to D
are the settings of campaigns 35, 38, 37 and 36; a star marks a sign that holds in every seed:

| Rule | A | B | C | D |
|---|---|---|---|---|
| a copy from host that costs no GPU time (upper bound of 4.5) | -2.61* | -2.38* | -1.50* | -0.07* |
| SSD keeps pages by index (cleaner order) | -0.25* | -0.94* | -0.14* | 0.00 |
| no write for an ended session's last turn | -0.03 | -0.03 | -0.17 | -0.07 |
| landing quantile; lookups that find nothing free | +0.03; +0.02 | +0.02; +0.04 | -0.04; +0.03 | 0.00; 0.00 |
| read line without the stall weight; with it | +0.26; +0.04 | +0.33; +0.03 | +1.02*; +0.40* | -0.01; +0.01 |
| read or recompute (restricted form) | +0.22* | +0.16* | +0.40 | +0.10* |
| host eviction by the full index | +0.13 | +0.06 | +0.33 | +0.03 |
| the first build (Fetch and Price by measured prices, Retain on the SSD) | -0.22* | -0.93* | -0.50* | -0.14 |
| the first build with the free copy | -2.84* | -3.52* | -3.04* | -0.18* |

Four readings. The largest term is GPU time that no cache policy touches: the pass waiting on the copy. Among the
cache rules the gain is what the SSD keeps. The Fetch and Price rules gain nothing alone, but only the build as a
whole is faster in all four settings. And the engine's host eviction (return chance at one 30 s horizon) is
already the index at one horizon: the full index changes nothing measurable. Nothing is claimed for the rules that
act through refused reads and pending writes (room order, write order, reads against writes): the simulator does
not model that path well enough.

### What the GPU measured
Two-hour runs on one H200, the same users and the same requests in every arm of a setting; paired differences
with the current design (`ec_full`), seconds of mean TTFT over the requests both sent before the last ten minutes,
95 % interval over users (`results/campaign35/DECISIONS.md`, `results/compare_unified/`). Two runs of the current
design differ by 0.01 s [-0.09, +0.11].

| Setting | Current design | Hold only | Cache rules only | Everything |
|---|---|---|---|---|
| A: 128 users, 160 GB, 2 h | 10.97 s | -1.61 [-1.79, -1.43] | -0.38 [-0.51, -0.25] | -1.79 [-2.00, -1.59] |
| B: A with the cleaner from minute 50 | 11.56 s | | -0.99 [-1.18, -0.81] | -2.49 [-2.76, -2.21] |
| C: 128 users, 112 GB | 12.80 s | | | -3.39 [-4.09, -2.74] |
| D: 64 users, 88 GB, 1 h | 0.68 s | | | -0.14 [-0.20, -0.08] |

Two more arms at A, each against everything: the price of host room fixed instead of measured, +0.04 s [-0.04,
+0.13], and the admission pass-over (Preempt on GPU room), -0.02 s [-0.22, +0.17] with the median 7.17 -> 6.13 s and
p99 49.9 -> 55.1 s. At C the mean falls most (26 %) while the median rises from 0.26 to 2.99 s: the current design
served 43 % of the history from the GPU and kept readers waiting behind those requests.

At A the hold gives 4.8 % more decode steps a second and 1.61 s; the cache rules alone 0.38 s, and on top of the
hold 0.22 s [0.11, 0.32] (the parts are slightly sub-additive),
1.16 s once the cleaner deletes and -0.16 s before, where the measured price of host room was too low (section 7).
The simulator had said -2.6 s for a free copy (an upper bound: the pass's own compute hides part of the copy) and
-0.28 s for the cache rules on these users.

## 11. What is built and how it is compared
Switches (all off by default; `python/sglang/srt/environ.py`), by action:
- Fetch on the GPU tier: `SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS` (4.5; most decode steps a batch is held, 0 off).
- Price: `SGLANG_LOG_HICACHE_PRICES_INTERVAL` (the prices line), `SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES`.
- Fetch: `..._NEXT_TURN_LATENESS_WEIGHT`, `..._NEXT_TURN_READ_LINE`, `..._NEXT_TURN_HOLD_QUANTILE`,
  `..._NEXT_TURN_FREE_EMPTY_LOOKUPS`; built and left off: `..._NEXT_TURN_RECOMPUTE_BY_PRICE`,
  `..._NEXT_TURN_RECOMPUTE_WHEN_STARVED` (4.4).
- Retain: `--radix-eviction-policy-config {"queued_order":"farthest"}`, `SGLANG_HICACHE_STORAGE_TRACK_STORED`,
  `SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES`, `SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S`,
  `SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES`.
Comparison, the current design (`ec_full`) against the unified one on the same GPU unit, same users. The launcher
takes its switches per campaign folder, so the arms of one setting run in sibling folders (record:
`results/campaign35/DECISIONS.md`; paired tables and figures: `results/compare_unified/`):
| Setting | Current design | Hold only | Cache rules only | Everything | What it tests |
|---|---|---|---|---|---|
| A: 128 users, 160 GB, 2 h (campaign 34's) | campaign 35 | campaign 39 | campaign 35 | campaign 40 | saturation; which part does the work (simulated: hold up to -3.1 s, cache rules -0.3 s) |
| B: A with the cleaner from minute 50 | campaign 38 | | campaign 38 | campaign 41 | the cleaner change over an hour (simulated -0.8 s; page replay: about 3 s late in the run) |
| C: 128 users, 112 GB | campaign 37 | | | campaign 42 | a tight host pool: SSD-bound readers (simulated -1.0 s) |
| D: 64 users, 88 GB, 1 h | campaign 36 | | | campaign 43 | no regression at the knee (simulated -0.2 s) |

The current design run twice in campaign 34's setting (campaigns 34 and 35: another GPU unit, three days and a host
restart apart) differs by +0.01 s [-0.09, +0.11] on 15,521 paired requests, so 0.1 s is what a pair can resolve.

## Appendix A. Symbols
i session; L tokens of its prefix; Delta new tokens; U tokens lost; e turn end; G gap; A arrival; a idle age; R
time left; Phi_k gap distribution of hint key k; F(x | a), h(a), tau(x | a) as in 1.1; W queue wait; D = A + W;
T time of next use; r read time; V wait in the device's line; x = (r - W)^+; l landing time; S stall; q landing
quantile; gamma GPU seconds per token; beta device tokens per second; C_G, C_H, C_S capacities; N users; B turns
running past their first token; X throughput; Z mean gap; g GPU-seconds per request; mu, c, lambda, kappa, sigma
prices; K, K0, K_rec restore costs; nu holding index; tau* holding time per return; rho the restore's own hold.

## Appendix B. Measured constants (campaign 34, FA3, Qwen3-30B-A3B, 16-bit KV)
KV 98,304 bytes a token; GPU pool 668,096 tokens; host pool 1,627,648 tokens (mirror 0.65M, host-only 0.90-0.97M);
SSD 2.0-2.1 GB/s reads and writes together; uncontended read 0.05 s + 0.044 ms a token; cold prefill 0.03 s +
0.012 ms a token + 1.4 us a token per thousand tokens of position (20K: 0.56 s; 30K: 1.03 s), 0.043 ms a token
under load; load host to GPU 0.03-0.04 s for 20-29K tokens; requests 15,562 in 110 minutes, 141.5 a minute at
saturation; prompts 25.5K tokens with 1.1K new; gaps as replayed: tool 91.5 %, p10/p50/p90 0.32/1.02/8.1 s; a
person 8.5 %, 2.7/68/722 s, 22 % of those never return.
