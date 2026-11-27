# Four actions, three prices: a unified design for the three-tier KV cache (draft v1, 2026-11-26)

Status: working draft for checking (math, logs, engine seams). Nothing here is implemented yet.
Inputs: the paper draft's design section (`inputs/paper_sections/design.tex`: Fetch, Preempt, Retain, Price on host
memory only), the deck of 6 November (`inputs/deck1_pricing_the_room_1106.txt`), the revised deck of 23 November
(`inputs/deck2_revised_1123.txt`: five components around T = G + W), the review `../design_review_1123/` and the
records of campaigns 23 to 35.

The author's brief (2026-11-26): keep the four-action framework; bring in the queueing delay, the hint (a distribution
of the tool call's duration) and the write; let the four actions govern read timing and order, eviction, and write
timing and order across GPU memory, host memory and SSD; make the design provably optimal under a stated model;
consider every gain, however small.

## 0. The idea in one paragraph

A session's prefix cache (the KV of its context) is needed on the GPU again at time T = A + W: the session's next
arrival A (known to the agent only as a distribution, the hint) plus the wait W in the GPU queue (known to the
engine). Every movement of KV between tiers is a job with a due date. A read (SSD to host) is due at T. A write (host
to SSD) is due when its page's host copy is given up. A copy from GPU to host is due when the GPU gives the page up.
Three shared resources carry these jobs: GPU time, host memory and the one SSD. Each has a queue, and the length of
each queue is a price: mu for a GPU-second, lambda for a host token-second, kappa for an SSD-second. With the prices
known, each session's best move is a one-line comparison, and those comparisons are the four actions:

- **Fetch** decides when each copy moves up (when a read starts, and in which order the SSD serves reads).
- **Retain** decides which copies each tier keeps, and so when the copy one tier down must exist (eviction order on
  GPU and host, the SSD write and its due date, what the SSD's own cleaner deletes).
- **Preempt** decides who yields when a resource is full now (host room; the SSD between a read and a write; the
  GPU between a turn that is ready and one whose read has not landed).
- **Price** measures mu, lambda, kappa and the lateness weight c from the three queues.

Each action is the exact optimum of a priced single-session problem (sections 3 to 6), the prices are the multipliers
of the relaxed joint problem (section 7), and the classical policies come out as special cases: Belady, LRU,
earliest-deadline-first, Smith's ratio rule, the newsvendor quantile.

## 1. Model

### 1.1 Sessions and hints
- N sessions. A session alternates a turn (one request) and a gap (the agent runs the tool call in the reply, or a
  person types). Turn of session i ends at e_i; the gap is G_i; the next request arrives at A_i = e_i + G_i.
- Hint: with each request the agent framework sends, for the key k of the tool call (tool plus the command's first
  word; `_no_call` for a reply with no tool call), a distribution Phi_k of the gap, and the end of a session
  (G = infinity). At idle age a = t - e_i the engine uses the conditional law
  F_i(x | a) = P(G <= a + x | G > a) = (Phi_k(a + x) - Phi_k(a)) / (1 - Phi_k(a)),
  and the hazard h_i(a) = phi_k(a) / (1 - Phi_k(a)).
- R = A - t given A > t is the remaining gap.

### 1.2 Tiers, copies and jobs
- GPU memory: KV pool of C_G tokens. Host memory: C_H tokens. SSD: C_S tokens of capacity and one device of beta
  tokens per second, shared by reads and writes.
- A page (64 tokens of KV) may have a copy on any subset of {GPU, host, SSD}. A session's prefix is a chain of pages;
  chains share heads (a common system prompt).
- Jobs: read (SSD to host, device time L / beta for L tokens, plus the wait V in the device's line), write (host to
  SSD, same device), backup (GPU to host copy), load (host to GPU copy, about 0.4 s, done at admission), recompute
  (GPU time gamma per token; on FA3 gamma is about 0.037 to 0.044 ms and nearly constant in the context length).
- Page states on the host: pinned (a landed read waiting for its turn; a read in flight), dirty (no complete SSD
  copy yet: the page is the only copy below the GPU), clean (an SSD copy exists), mirror (the GPU still holds the
  same page).

### 1.3 What a returning request costs
A request arrives at A, would be admitted by the GPU at D = A + W if its prefix were resident, waits S = (l - D)^+
more if its read lands at l > D (stall), and prefills Delta new tokens plus U prefix tokens that were lost and not
read back:

    TTFT = W + S + gamma (Delta + U) + const.

T = D is the time of next use of the prefix on the GPU.

### 1.4 The joint problem
Choose, at every instant, which jobs to start and which copies to drop, to minimize E[sum over requests of TTFT]
subject to: host tokens held <= C_H at every t; device work <= beta; SSD tokens <= C_S; GPU pool <= C_G, admission in
arrival order among turns whose prefix is resident (a turn whose read has not landed is passed over).

### 1.5 Where the queue enters: two weights
The queue wait of a turn is the GPU work of the turns served before it. Two facts follow (measured in campaigns 32
and 34, see `../design_review_1123/critique.md`):

- (mu) One more second of GPU work at time t delays every turn that waits behind it. Summed TTFT rises by mu(t)
  seconds, with mu = 1 + (turns delayed). In an open queue mu is about 1 + Q (queue length). In the closed loop of the
  replay (N users, each sends its next turn after the reply and its pause), once the GPU is never idle,
  mean TTFT = N g - Z - decode, with g the GPU seconds per request and Z the mean pause, so mu = N.
  Measured: 0.73 N (94 TTFT-seconds per GPU-second of recompute, minutes 90 to 109).
- (c) One more second of stall of turn i delays only turn i, and the GPU serves another ready turn meanwhile. If
  another turn is ready, that turn is admitted earlier by about the same amount and the sum does not change. So a
  second of stall costs c(t) = P(no other ready turn when turn i is due) seconds of summed TTFT: about 1 at light
  load, about 0 once ready work always waits.

So, to first order in the policy's effect,

    sum of TTFT = (terms no cache policy changes) + sum over requests of [ c S + mu gamma U ].

### 1.6 The priced problem
Relax the three capacity constraints to hold on average, with multipliers lambda (seconds of TTFT per host
token-second), kappa (per second of device time) and sigma (per SSD token-second). The objective separates:

    sum_i J_i,   J_i = E[ c S_i + mu gamma U_i + lambda * integral of m_i(t) dt + kappa * (device seconds of i) + sigma * integral of s_i(t) dt ],

m_i = host tokens held for session i, s_i = SSD tokens. Each session minimizes its own J_i. Sections 3 to 6 solve
that problem action by action; section 7 gives the prices.

## 2. One restore cost
Write K for the priced cost of not having a prefix of L tokens in memory when its turn is due:

- K_read = c * E[(r - W)^+] + kappa * L / beta + lambda L r, if a complete SSD copy exists (r = V + L / beta + lags:
  the read time including the device's line);
- K_rec = mu gamma L, if not (the turn recomputes it and every turn behind waits).

Read-or-recompute: restore by the cheaper one. On an idle FA3 GPU (c = 1, mu = 1, kappa = 0) recompute wins for any
length for which gamma L < lags + L / beta, i.e. always here (gamma = 0.037 to 0.044 ms against 1 / beta = 0.047 ms
plus 0.9 s of lags); with one turn waiting behind (mu >= 2) the read wins for long prefixes; at saturation
(mu about N, c about 0) the read always wins. The engine today always reads.

## 3. Fetch: when a copy moves up

### 3.1 The rule
For a prefix on the SSD only, choose the start s of its read to minimize

    J_F(s) = c * E[(l - T)^+] + lambda L * E[(T - l)^+] + kappa (L / beta) * P(the read is made),   l = min(s, A) + r.

A read that lands late costs stall at weight c; a read that lands early holds L tokens of host memory until
admission at price lambda. A turn that arrives before s starts its own read (recourse), so a wrong forecast costs
memory, never more stall than reading at arrival.

### 3.2 After arrival (G = 0): land at a quantile of the admission time
The only uncertainty left is D. Minimizing c E[(l - D)^+] + lambda L E[(D - l)^+] over l gives

    P(D <= l*) = q,   q = lambda L / (c + lambda L).

The read lands at the q-quantile of the admission time: early when host memory is cheap (q near 0), exactly at the
last moment when the GPU is saturated or memory is dear (q near 1). The read is released at l* - r. The engine's hold
uses a fixed 2 s margin; the rule replaces it with the q-quantile of the measured error of the admission estimate.
(Estimate of D: queue position over admission rate for turns with no read; for a turn that reads, the later of that
and its own landing, plus 0.4 s. Measured in campaign 34: median error 0.4 to 0.5 s.)

### 3.3 Before arrival: what the hint decides
Let x = (r - W)^+ be the part of the read the queue does not hide. If x = 0, nothing is read before arrival (3.2
applies). If x > 0 and s is before the arrival,

    dJ_F/ds = c P(s < A <= s + x) - lambda L P(A > s + x) = P(A > s) [ c F(x | s) - lambda L (1 - F(x | s)) ],

so J_F falls while F(x | s) < q and rises while F(x | s) > q. Three cases:

- (IFR) If the gap's hazard rises with idle time (a sharp hint: a test run of 20 +- 2 s), F(x | s) rises in s, J_F
  is quasi-convex, and the optimum is the threshold rule of the paper: start when F(x | s) >= q.
- (DFR) If the hazard falls with idle time (a heavy tail: both gap entries of the FreeInference trace, tool waits
  with p10/p50/p90 0.41/1.69/16.8 s and human waits 3.1/96/1,040 s), F(x | s) falls in s, J_F is quasi-concave, and
  the optimum is at an end: read now (at the turn's end) or wait for the arrival. Read now iff
  c E[min(R, x)] >= lambda L E[(R - x)^+] + kappa (L / beta) P(G = infinity).
  Reading at the turn's end is the same as not evicting, so on such keys Fetch before arrival is empty and the
  decision is Retain's (section 4).
- (General, e.g. two modes) minimize J_F over the support of the conditional law (the engine's `read_start` does
  this over 64 points).

This is why reads before arrival paid only with exact hints (0.14 to 0.21 s, campaign 26), why reads timed on a p50
hurt (0.6 to 0.9 s), and why the expected-cost arm staged 1 read in campaign 34.

### 3.4 Order on the device
Reads share one device. A read is a job with release time (its start s*), size and due date (T, at the quantile
above). Serving the released jobs in earliest-due-date order minimizes the maximum lateness (Jackson's rule; with
release times and preemption at page-group boundaries, Horn's theorem), so if any order lands every read before its
turn is due, this one does. When the device cannot meet the due dates, lateness is rationed by cost per device-second:
c_j per second late over size.

### 3.5 What is new against the engine
c in the cost (today 1); the landing quantile in place of the 2 s and 1 s margins; r that counts parked reads and the
write share before arrival as well; read-or-recompute by price (section 2); a lookup that finds nothing does not
spend the turn's lookup budget (OPEN_ITEMS.md section 1).

## 4. Retain: which copies each tier keeps

### 4.1 Host memory: an index
A session idle for a holds L tokens in host memory. Keeping them costs lambda L per second; dropping them costs K
when the session returns. Keeping for some horizon beats dropping now iff

    nu(a) = (K / L) * sup over h > 0 of F(h | a) / tau(h | a)  >=  lambda,    tau(h | a) = E[min(R, h)].

nu is the Gittins index of the session's holding problem (the best ratio of expected cost avoided to expected
token-seconds spent), so "keep while nu >= lambda, evict in increasing nu" is the exact optimum of the priced
single-session problem, and the problem is indexable (the set of ages at which keeping is optimal shrinks as lambda
grows). Special cases:
- exact hint, return in R0 seconds: nu = K / (L R0). Equal K / L: Belady's farthest next use.
- falling hazard (DFR): the sup is at h -> 0 and nu = (K / L) h(a). One distribution for all sessions: LRU.
- rising hazard (IFR): the sup is at h -> infinity, nu = (K / L) P(R < infinity) / E[R]: the longest idle is kept.
- ended session (hint G = infinity): nu = 0, first out.
- K = K_read for a clean page and K_rec for a dirty one: at saturation the only copy of a page is worth mu gamma L
  and is kept; a copy the SSD also holds is worth a read.
- a queued turn at position k: T is nearly known (now + W_k): nu = K / (L W_k): farthest from admission first, and
  only while its re-read fits before admission.
- the mirror (a host copy of a page the GPU still holds for a running turn): nothing reads it while the GPU holds
  the page, so its nu is that of the copy being needed after the GPU gives the page up.

The same index orders GPU eviction (K = the load time when a host copy exists).

### 4.2 The write is retention one tier down
A dirty page has restore cost K_rec, a clean one K_read. The write buys the difference, and only if the host copy
goes before the page's next use:

    value of writing page p now = V_dirty(a) - V_clean(a) - kappa L_p / beta - sigma (...),

where V_x(a) is the session's optimal priced cost with restore cost K_x. Consequences:
- A page whose session has ended is never worth writing, and its SSD chain is first out of the SSD.
- A page that stays in memory until its next use is not worth writing yet. Its write becomes due at
  d_w = the first time its clean index nu_clean falls below lambda (the time its host copy would be given up).
  Before d_w the write costs only device time; after it every second costs about lambda L_p (room held only because
  the page is dirty), or a refusal if no clean page is left to give up.
- So writes are jobs with due dates, like reads: write in the order in which pages will be evicted (lowest clean
  index first), and no later than d_w. With an idle device, write at once (it is free); with a busy device, order all
  transfers by due date (section 5.2).
- A page already on the SSD (same chained key) is clean at birth: a recomputed page is not written again.
- A page counts as clean only when the SSD has acknowledged the whole write.

On this workload eager writing is right for live sessions: 56 to 67 % of prompt positions are read back from the SSD,
4.2 to 4.5 times each (campaign 34).

### 4.3 The same rule one tier up and one tier down
- GPU to host: the backup is due when the GPU gives the page up. Write-through makes it at the end of prefill, so
  the mirror holds 0.65 M of 1.63 M host tokens for nothing (campaign 34); dropping mirror copies once their SSD write
  is acknowledged, or backing up at GPU eviction, pays lambda only when it buys something. (Engine constraints: an SSD
  lookup needs a host-backed anchor; a GPU-evicted node with no host copy is deleted.)
- SSD capacity: the cleaner is eviction on the SSD, by the same index with K = K_rec for the whole chain below a
  page: ended sessions' chains first, a chain's tail before its head (a lookup stops at the first missing page), a
  head shared by live sessions last. Today's cleaner deletes the oldest-written page first, which is the shared
  head.

## 5. Preempt: who yields when a resource is full now

### 5.1 Host room
A turn that has arrived needs m tokens for its read and none are free. Displacing holder j costs C_j (the rise in its
priced cost). Freeing m tokens at least cost is a knapsack; taking holders in increasing C_j / L_j is optimal for the
divisible problem and within one holder otherwise. In index terms the order is: ended sessions; clean idle prefixes
by nu; staged reads of sessions not yet back, by (K_read F_j(r_j | t) + kappa p_j P(return)) / L_j; queued turns far
from the head, by c E[(t + r_j - T_j)^+] + kappa p_j, only while the re-read fits; never dirty pages, reads in
flight, or turns near the head. If nothing can yield, the read waits (delay); it is not refused, because a refusal
turns into a recompute at mu and, without the born-clean rule, into more writes.

### 5.2 The device: reads against writes
Reads are due at T and cost c per second late; writes are due at d_w and cost lambda L_p per second late. Serve the
device in earliest-due-date order over both kinds; if the due dates cannot all be met, serve the largest cost per
device-second first. With free room (lambda = 0) writes never go before a due read; when dirty pages hold room that
due reads need, writes go first. The engine's fixed rule (hold timed reads while pending writes hold 50 K tokens,
resume at 25 K) is one threshold of this comparison.

### 5.3 The GPU
A turn whose read has not landed is passed over and the next ready turn is admitted (already so): the GPU never
waits for the SSD while ready work exists. This is what makes c small under load.

## 6. Price: three queues, three prices
- mu(t), GPU time: 1 + the number of ready turns waiting, at most the users in session.
- c(t), lateness: the share of recent admissions made with no other ready turn waiting.
- kappa(t), device time: the summed lateness cost rate of the transfers waiting behind (c_j of reads past their
  latest start, lambda L of writes past their due date).
- lambda(t), host room: the index nu of the marginal holder (the highest index among the pages given up lately); 0
  while room is free. This is the multiplier of the room constraint in the relaxed problem (the index of the
  marginal item in a knapsack), so it is read off the eviction frontier, not learned from refusals. (The engine's
  learned price never moved on FA3.)
- sigma(t), SSD capacity: the index of the cleaner's marginal victim; 0 until the volume fills.

## 7. What can be proved
1. (Fetch) Section 3: the derivative of J_F; the threshold rule is optimal iff F(x | s) crosses q once from below
   (in particular for IFR gaps); for DFR gaps the optimum is at an end; after arrival the optimum lands at the
   q-quantile of the admission time.
2. (Retain) Section 4.1: nu is the Gittins index of the holding problem; keeping iff nu >= lambda is optimal for the
   priced session; indexability; the reductions to Belady and LRU.
3. (Write) Section 4.2: the value of a write and its due date; writing in eviction order.
4. (Order) Sections 3.4 and 5.2: earliest due date minimizes the maximum lateness on the device, over reads and
   writes together.
5. (Preempt) Section 5.1: the greedy order by cost per token freed solves the divisible problem.
6. (Price) Section 6: at the optimum of the relaxed problem lambda is the index of the marginal holder, and
   complementary slackness holds (lambda = 0 while room is free).
7. (Bound) For any prices, the sum of the sessions' optimal J_i minus the priced capacities is a lower bound on the
   priced cost of every policy that meets the capacities with the same hints (weak duality), in the open model where
   mu and c are given.
8. (Saturation) In the closed loop with the GPU never idle, mean TTFT = N g - Z - decode: two policies with the same
   GPU work per request have the same mean TTFT, whatever their read timing. The optimum then minimizes recompute.
9. (Stability) With born-clean pages and delay in place of refusal, the dirty set is bounded whenever the write rate
   exceeds the rate of new tokens; without them recompute feeds writes, writes hold room, and room refusals feed
   recompute (campaign 32's loop).
10. (Many sessions) Ranking by the index is asymptotically optimal for the relaxed problem as N grows (index-policy
    theory for restless bandits); cited, not re-proved.

## 8. Regimes the rules reduce to
| Regime | Prices | Fetch | Retain / write | Seen in |
|---|---|---|---|---|
| Light load, room short | c = 1, mu = 1, lambda > 0 | early reads only on sharp hints; recompute short prefixes | index order matters | campaign 26 |
| GPU saturated, SSD keeps up | c ~ 0, mu ~ N, kappa ~ 0 | no early reads; land at admission | never lose the only copy | campaigns 28, 34 (minutes 30 to 70) |
| GPU and SSD saturated | c ~ 0, kappa > 0 | reads in due-date order | fewer reads: room, order | campaign 34 (minutes 90 to 110) |
| Host at the cliff | lambda large | no speculative holds | writes before timed reads | campaign 32 |

## 9. Candidate engine changes (to be confirmed by the checks)
Fetch: landing quantile for the hold (F1); c in the expected-cost start (F2); read time that counts the line before
arrival (F3); read-or-recompute by price at light load (F4); lookups that find nothing are free (F5).
Retain: skip the SSD write of an ended session's last turn and mark its chain first out (R1); write queue in eviction
order (R2); clean only on a full acknowledgement (R3); SSD-tier order through file times: ended first, tails before
heads, heads refreshed when memory serves them (R4); restore-cost-aware order (only copies stay) (R5).
Preempt: staged reads yield before queued turns; queued turns by slack (P1); reads against writes by due date and
clean reserve instead of the 50 K / 25 K thresholds (P2).
Price: mu, c, kappa, lambda measured and logged every few seconds; FA3's recompute cost in place of Triton's (PR1).
