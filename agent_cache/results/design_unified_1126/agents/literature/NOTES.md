# Literature check of "Four actions, three prices" (DESIGN.md, draft v1, 2026-11-26)

Agent label: literature. Written 2026-11-26. Files in this folder:

- `references.bib`: 99 BibTeX entries, each checked against a page opened today.
- `verify_log.txt`: what was opened for each source and what the page showed.
- `quotes.txt`: the short verbatim passages these notes rely on, with the URL of each.
- `checks/`: four CPU scripts and their outputs (brute force, quadrature, simulation).
- `tools/`: the two helpers used to read PDFs saved by the web-fetch tool.

Method. Every request went through the web search and fetch tools, with DOIs, paper titles and standard terms
only; nothing about this project, machine or author was sent, and no request left this machine. Bibliographic
data comes from the Crossref record of each DOI, from publisher or arXiv pages, or from the bibliography of
Lenstra and Shmoys (arXiv:2001.06005). Statements come from abstracts and, where a full text was reachable, from
the text itself. Where I could not open the text, the statement is marked "from memory" and, where possible,
checked numerically. Section 5 lists what could not be opened.

Symbols used below, as in the draft. G: gap between a turn's end and the next request. a: idle age. R: remaining
gap, G − a given G > a. F(h | a) = P(R ≤ h). h(a): hazard of the gap at age a. L: prefix length in tokens.
K: priced cost of not having the prefix in memory when its turn is due. λ: price of a host token-second.
κ: price of a device-second. μ: TTFT-seconds caused by one GPU-second. c: TTFT-seconds caused by one second of
stall. W: wait in the GPU queue. r: read time including the device's line. x = (r − W)^+.
q = λL / (c + λL). For device jobs: p_j size in device-seconds, d_j due date, r_j release time, w_j cost per
second late, C_j completion time, lateness C_j − d_j, tardiness max(0, C_j − d_j).

## 1. The points that change the draft

### 1.1 Order on the device (sections 3.4, 5.2, 7.4): three conditions are missing

What the sources say (Lenstra and Shmoys, chapters 3, 4 and 8; quotes in `quotes.txt`):

- Jackson 1955: with every job available from the start, earliest-due-date order minimizes the maximum lateness,
  and "all due dates can be met if and only if they are met in any EDD schedule" (Theorem 3.1, Corollary 3.2).
- Horn 1974: with release times, the same holds for the preemptive rule (at every release and completion, run
  the released job due first), provided interruption is free (Theorem 3.5).
- Without preemption and with release times the problem is strongly NP-hard, and the rule fails.
- Smith 1956: largest w_j / p_j first minimizes the sum of w_j C_j (Theorem 4.1). The sum of weighted tardiness
  reduces to it only "if d_j = 0 for all j", that is, when every job is already late.
- The sum of weighted tardiness is NP-hard in the strong sense; the unweighted sum is NP-hard (Du and Leung 1990).

What that means for the draft:

1. "With release times and preemption at page-group boundaries, Horn's theorem" is not Horn's theorem. With
   switches only between pieces of at most g seconds, earliest-due-date is within g of the best maximum lateness,
   not equal to it. Proof: move each release to the next switch point of the schedule; the schedule is then the
   preemptive rule for the moved releases, hence optimal for them (Horn), and moving releases by at most g costs
   at most g. Check: 9,000 random instances, largest excess 0.99 g to 1.00 g, never above g
   (`checks/sched_checks.out`, part d). Counterexample without preemption: A (released 0, size 4, due 10) and
   B (released 1, size 1, due 2.5). The rule starts A and B lands 2.5 late; waiting one second for B lands both on
   time. Corrected sentence: "if some order lands every read at least one page-group time before its due date,
   earliest-due-date order with switches at page-group boundaries lands every read on time."
2. The theorem is about the maximum lateness; the paper's cost is the sum of c × stall, a weighted tardiness.
   Earliest-due-date is optimal for that sum only when no read has to be late (the sum is then zero).
   Counterexample: sizes 10, 1, 1, 1 with due dates 10, 10.5, 10.6, 10.7: earliest-due-date gives 4.2 s of total
   tardiness, the best order 3.0 s (part g).
3. "If the due dates cannot all be met, serve the largest cost per device-second first" is Smith's rule, optimal
   only when every waiting transfer is already past due. Switching to it at the first infeasibility is costly when
   only a few transfers must be late. Toy check, 7 jobs, all orders tried (`checks/sched_checks_wtard.out`), mean
   excess over the optimum in weighted seconds, with the optimum in brackets:

   | due dates up to | earliest due date | ratio rule | draft's switch | slack-discounted ratio (ATC) |
   |---|---|---|---|---|
   | 2.0 × total work (2.9) | 0.93 | 9.11 | 6.93 | 0.50 |
   | 1.0 × (13.8) | 8.45 | 10.76 | 10.76 | 2.00 |
   | 0.6 × (32.9) | 24.40 | 5.13 | 5.13 | 1.34 |
   | 0.25 × (55.6) | 34.87 | 0.79 | 0.79 | 0.45 |
   | all late at the start (78.0) | 37.30 | 0 | 0 | 0 |

   No simple order is optimal in between (the problem is strongly NP-hard), so item 4 of section 7 can claim
   optimality only in the two end cases. A single index that discounts the ratio by the remaining slack
   (Vepsalainen and Morton 1987) stays within 18 % of the optimum in every row of this toy. Van Mieghem (1995,
   2003) gives the queueing version: with convex delay costs, serve the class with the largest marginal delay cost
   times service rate, asymptotically optimal in heavy traffic. A stall cost that starts at the due date is convex
   and nondecreasing; I did not check whether his technical conditions admit its kink (his 2003 paper treats due
   dates through a sequence of convex costs). The toy is static and uniform; it is evidence about the switch, not
   about the engine.
4. Under the draft's own weights every late write has the same cost per device-second (λ L_p over L_p / β, that
   is λβ), and a late read j has c_j β / L_j. So in overload the rule reduces to: a late read goes before late
   writes if and only if c_j / L_j > λ, the same as q_j < 1/2.
5. Horn assumes interruption is free. A read stopped halfway keeps its pages pinned in host memory, so preemption
   has a price in λ that the theorem does not see.

### 1.2 "nu is the Gittins index" (sections 4.1, 7.2): right for one problem, and that problem must be named

What is right. The draft's ν(a) = (K / L) sup_h F(h | a) / τ(h | a) is exactly the Gittins index used for
scheduling jobs of unknown size S, with the gap in the role of the size and the idle age in the role of the
service received: index(x) = sup over y > x of P(S ≤ y | S > x) / E[min(S, y) − x | S > x]. This is Definition
IV.2 of Scully and Harchol-Balter 2021 with holding cost 1; their Example V.2 writes it with the weight 1/S. The
form goes back to Gittins's work on scheduling (from memory). Their "Gittins game" is the draft's priced
single-session problem: serve until a give-up set and pay a penalty on giving up. The optimal give-up set is the
set of states whose index is at most the scaled price, and "The sets Y*(r) are monotonic in r" (Definition VI.3,
Lemma VI.6). That is the draft's "keep while ν ≥ λ" and its shrinking keep-set. Whittle 1980 is the retirement
form of the same statement.

Numeric confirmation (`checks/index_checks.out`). For six gap laws (two with falling hazard, two with rising
hazard, a lognormal, a two-moded mix): with a falling hazard the supremum is at h → 0 and ν = (K / L) h(a); with a
rising hazard it is at h → ∞ and ν = (K / L) / E[R]; in the lull of the two-moded law ν stays 14 times above
the hazard. A dynamic program for the stopping problem agrees with "keep iff ν(a) ≥ λ" at every grid age (at most
2 of 2,864 to 19,998 ages differ, all within 2 % of the threshold).

What needs a condition.

1. The index solves the problem in which a dropped prefix stays out until its turn arrives. If the engine may
   read it back before the arrival (Fetch), dropping is not final and the comparison has a third plan. Example by
   quadrature: gap Normal(20 s, 2 s), L = 25,500 tokens, r = 2.09 s, no queue, c = 1, κ = 0. With λL = 0.05
   (TTFT-seconds per second of holding the prefix): keep through the gap 1.00 s; drop and read at arrival 2.20 s;
   drop and read back at 14.6 s 0.31 s. The index rule says keep. The timed read wins in 7 of the 8 price pairs
   tried. So for sharp (rising-hazard) keys the cost of dropping is the value of the best Fetch plan from that
   age, which changes with age; the constant-K index is exact for falling-hazard keys, where section 3.3 shows
   Fetch before arrival is empty. The same applies to "exact hint: ν = K / (L R0)": with an exact hint and a free
   device the session's optimum is to drop and read back just in time whenever R0 > r.
2. "Evict in increasing ν" is not part of the single-session optimum. It is a ranking rule for the coupled
   problem. Gittins's theorem (the index order is optimal) needs one arm served at a time and arms that stand
   still when not served; here many prefixes are kept at once and a dropped session keeps aging. With several
   servers even the scheduling version is only known to be near-optimal (Scully, Grosof and Harchol-Balter 2020).
3. "Indexable" is Whittle's 1988 word for restless bandits (the passive set grows with the subsidy). In Whittle's
   own relaxation an arm may switch freely, and then the index of this problem is the hazard (K / L) h(a) for every
   gap law: with free re-entry, keeping for the next instant pays if and only if K h(a) ≥ λ L. That equals ν only
   for falling hazards. So there are three objects: the hazard index (free re-entry; it is the bound of Panigrahy
   et al.), the stopping index ν (no re-entry), and the real problem (re-entry at the price of a read).

Corrected statement: "For the priced single-session problem in which an evicted prefix is not read back before
its turn, keeping is optimal exactly while ν(a) ≥ λ; ν is the Gittins index of that stopping problem, and the set
of ages at which keeping is optimal shrinks as λ grows. For falling-hazard gaps ν = (K / L) h(a) and the
restriction costs nothing. Ranking sessions by ν under a hard capacity is a heuristic; see 1.3."

### 1.3 "Asymptotically optimal as N grows" (section 7, item 10): the cited theorems do not cover this model

- Whittle 1988 relaxes "exactly m of n active" to "m on average", defines the index from the multiplier, and
  conjectures the index policy is nearly optimal for m and n large in constant ratio.
- Weber and Weiss 1990 prove it "if the differential equation describing the fluid approximation to the index
  policy has a globally stable equilibrium point", give a counterexample otherwise, and (1991 addendum) prove
  the condition for three states. Identical arms, long-run average reward.
- Verloop 2016 extends to several classes, arrivals and several actions, again "under a global attractor
  property and a technical condition"; indexability is not required.
- Brown and Smith 2020, finite horizon, one cardinality constraint: the gap of the optimal Lagrangian index
  policy to the Lagrangian bound grows like the square root of N, "whereas the gaps for Whittle index policy
  grow linearly with N", and "a Lagrangian index policy that breaks ties randomly may also exhibit linear growth".
- Hodge and Glazebrook 2015 (several activation levels under one resource constraint) and Graczová and Jacko 2014
  (a knapsack constraint with items of different sizes) are the closest to unequal prefix lengths.
- Brown and Zhang 2022: with shared resources and shared random signals, "competing state-of-the-art approaches
  used in the literature on weakly coupled dynamic programs in general fail to provide asymptotic optimality".

The draft's model has three coupled resources, unequal sizes, and a GPU queue that every session sees (a shared
signal) and, in the closed loop, changes. No theorem above covers it as stated. Also, "asymptotically optimal for
the relaxed problem" is the wrong way round: the index policy is optimal for the relaxed problem by construction;
the theorems say its cost in the hard-capacity problem approaches the relaxed bound. Suggested wording: "For one
capacity, finitely many session types in fixed proportions and capacity growing with N, ranking by a Lagrangian
index approaches the relaxed bound under the conditions of [Weber-Weiss, Verloop] (long-run average) or
[Brown-Smith] (finite horizon, with tie-breaking). Our model adds a second and third shared resource and a shared
queue; we do not claim the result for it."

### 1.4 The GPU price μ (sections 1.5, 6, 7 item 8)

- Closed loop. The identity is the response time law, R = N / X − Z (Lazowska et al. 1984, chapter 3, derived
  as Little's law on the whole system: N = X (R + Z)). With the GPU never idle, X = 1 / g by the utilization
  law, and R is TTFT plus decode. So "mean TTFT = N g − Z − decode" is correct, and so is item 8, provided the GPU
  is never idle under both policies and N is the same. Cite the response time law; Little 1961 is the root.
  μ = N then holds exactly in the long run: after one extra δ of GPU work every later completion moves by δ, so
  each of the N users has exactly one request delayed by δ.
- Open queue. "μ is about 1 + Q" understates. One more unit of work delays the Q turns waiting and every turn
  that arrives before the queue next empties. For Poisson arrivals at one first-come-first-served server with
  utilization ρ, the expected number delayed is (Q + ρ) / (1 − ρ), so μ = (1 + Q) / (1 − ρ). Simulation
  (`checks/misc_checks.out`, part 4): at ρ = 0.8 and Q = 5, 28.4 turns delayed (prediction 29.0), not 5; averaged,
  μ = 20.6 against 1 + E[Q] = 4.2. The literature's name for the part on turns already present is the "tangible
  externality" (Haviv and Ritov 1998); the draft's 1 + Q is that part.
- The lateness weight c. "The sum does not change" when another ready turn is admitted instead is Kleinrock's
  conservation law (1965): the sum of ρ_p W_p is the same for every work-conserving, non-preemptive order. It is
  the plain sum of waits only if turns need equal GPU work; otherwise passing over a long turn for a short one
  changes the plain sum.
- Naming. In queueing, "the c-mu rule" means cost rate times service rate. The draft's c and mu are two other
  things. A reader from that field will misread "c ... mu"; say so once or rename.

### 1.5 The host price λ (sections 1.6, 6, 7 item 6)

- Two different multipliers are mixed. Section 1.6 relaxes capacity "on average"; its multiplier is one number
  for the horizon, zero only if the capacity is slack on average (Whittle 1988; Dehghan et al. 2019, whose
  Lagrangian has one multiplier for the capacity; Ferragut et al. 2016, Theorem 2: the timers satisfy
  "hazard at the timer = constant", the constant being that multiplier). Section 6 uses a spot price, "0 while
  room is free", which is the multiplier of the capacity at each instant. Both are legitimate, but the
  single-session rules of sections 3 and 4 are derived for a constant price. With a spot price that is zero until
  the room is full, q = 0 and every early read looks free up to the instant of the cliff.
- That is the case Weitzman 1974 analyses. His coefficient is Δ = σ² B'' / (2 C''²) + σ² / (2 C''); "the sign
  of Δ simply equals the sign of C'' + B''", and "if benefits are almost kinked at the optimum level of output
  ... the quantity mode scores a lot of points". Host memory's cost is kinked at capacity. The review of 23
  November drew the right conclusion (cap speculative holds by measured slack); the draft drops it. Keep a
  quantity limit for speculative holds next to the price (the clean reserve of P2 is one).
- "Read off the eviction frontier" has support: for a cache evicting in a fixed order the frontier is nearly
  deterministic when many items share the room (Che et al. 2002 and Fricker et al. 2012: hit probability
  1 − exp(−q(n) t_C) with one characteristic time t_C), and Dehghan et al. relate that characteristic time to
  the multiplier. With about 38 prefixes of 25.5 K tokens in 1 M tokens of room the frontier is noisy; smooth it
  over a window comparable to the holding times.
- "Within one holder" (section 5.1) is an additive bound: greedy by cost per token costs at most the divisible
  optimum plus the last holder taken (Dantzig's rule for the divisible knapsack; 3,000 instances, no violation).
  In ratio it can be about 2: need 10 K tokens, B holds 9 K at cost 9.0, A holds 10 K at cost 10.1; greedy takes
  both (19.1), the optimum takes A (10.1). Mean ratio in the random instances 1.13, largest 2.36.

### 1.6 Belady, LRU and the hazard-rate bound (section 4.1)

- Belady 1966: evicting the page whose next use is farthest minimizes page faults, for pages of equal size and
  equal miss cost, fetched on demand. With unequal sizes offline caching is strongly NP-complete (Chrobak et al.
  2012). So "equal K / L: Belady" is right as the order the index gives, but the optimality needs equal L and
  equal K. Under the time-averaged relaxation K / (L R0) is exactly the divisible-knapsack ratio, which is the
  cleaner thing to say.
- Panigrahy et al. (2021, 2022): keeping at every instant the objects with the largest hazard (or hazard over
  size) gives at least as many expected hits as any policy that cannot see the future, when the request
  processes are conditionally independent. They "call it a rule, not a policy": the rule may put an object back
  with no request, and for unequal sizes it "does not meet the cache size constraint". Two consequences. First,
  the bound holds under the hard capacity at every instant, not the relaxed one. Second, where the rule is
  realizable it is optimal: with one falling-hazard gap law shared by all sessions and equal sizes the rule keeps
  the most recently used, which is LRU, so LRU is optimal among non-anticipative policies in their model. (This is
  my reading of their Lemma 1; they do not state it in this form.) That is a provable statement for Retain in a
  stated model, and stronger than the relaxed one. With a weight K_i per hit the same proof gives the order
  K_i h_i / L_i, which is the draft's ν for falling hazards.
- Ferragut et al. 2016 is the closest precedent for the draft's split: under rising hazards "the optimal TTL
  caching policy ... is to statically store the C most popular contents" (all or nothing); under falling hazards
  the timers equalize the hazards. Cite it for both special cases of section 4.1.

### 1.7 Prior systems already contain parts of the four actions (section 3 of these notes lists them)

- TripleKV (arXiv 2611.25112, 18 November 2026) reads from SSD into host memory at
  t_issue = t_end + q − R − ε: predicted tool end plus the live queue delay minus the read time minus a fixed
  margin. That is T = A + W with a per-tool median in place of the hint's law and a constant in place of the
  quantile. It also removes host copies of GPU-resident blocks (the draft's mirror) and protects prefetched
  blocks by class. Same model family (Qwen3-30B-A3B). The paper must cite it and state the difference: a
  distribution and a price decide whether a timed read pays at all (section 3.3), where TripleKV always times on
  the median.
- CachedAttention (USENIX ATC 2024) already fetches from disk for jobs in a "look-ahead prefetching window" of the
  queue and evicts "the item located at the tail of the look-ahead eviction window" first. That is the draft's
  "queued turn farthest from admission first".
- Continuum sets a time-to-live by τ* = argmax P(τ, f) × Benefit − Cost(τ), with the memory charged for the whole
  τ. The draft charges it until the session returns. The two first-order conditions differ: K × density(h) = λL
  against K × hazard(h) = λL. For a lognormal gap with median 1.69 s and p90 16.8 s the full-charge TTL is 2 to
  35 times shorter and costs 5 % to 280 % more under the draft's accounting (`checks/index_checks.out`, part 4).
  Continuum pins GPU memory, so this compares formulas, not systems.
- SARIA (arXiv 2610.31169) persists a block to SSD only when eviction and reuse counts say it will be needed
  beyond its stay in memory, and paces memory reads while SSD transfers run, on TraceLab sessions. It is the
  closest precedent for section 4.2 and for R1 and R2.

### 1.8 Smaller points

- Newsvendor (section 3.2): correct. The minimizer of c E[(l − D)^+] + λL E[(D − l)^+] satisfies
  P(D ≤ l*) = λL / (c + λL); checked by quadrature to four digits (`checks/misc_checks.out`, part 1). Cite Arrow,
  Harris and Marschak 1951; for timing a release against a due date, Yano 1987 and Baker and Scudder 1990.
- Derivative of J_F (section 3.3): correct (finite differences agree to 2e-4 or better); falling hazard and the
  lognormal give the optimum at s = 0, rising hazard an interior threshold. One correction: with mass at infinity
  (session ended) the left side of the read-now test is c E[min(R, x); R < ∞], and E[(R − x)^+] is infinite unless
  the hold is capped.
- Threshold on a return probability set by load has a precedent in network prefetching: Jiang and Kleinrock 1998
  prefetch files whose access probability exceeds a threshold computed from load, capacity and cost (statement
  from index records; text not opened).
- Weak duality (section 7, item 7): correct as stated ("in the open model where mu and c are given"). Sources:
  Everett 1963; Adelman and Mersereau 2008; Brown and Smith 2020, Proposition 1. Brown and Zhang 2023: the
  Lagrangian bound and the tighter linear-programming bound are "very close and often identical".
- Hardness (deck 1, commented line in design.tex): Papadimitriou and Tsitsiklis 1999 show the restless bandit
  problem "is complete for polynomial space". "Our problem generalizes restless bandits" needs one line showing a
  restless bandit instance is an instance of ours; otherwise write "belongs to a class for which".
- "The length of each queue is a price" (section 0): true for μ and κ as externality prices (Naor 1969; Haviv and
  Ritov 1998) and for multipliers updated from excess demand (Kelly et al. 1998; Low and Lapsley 1999). Host
  memory has no queue; its price is the marginal holder's index. Reword the slogan.
- Write path (sections 4.2, 5.2): no theorem is cited and none is needed. Precedents to name: clean-first
  eviction (Park et al. 2006), the kernel's two dirty thresholds (background writeback, then the writer itself
  writes) and the deadline scheduler, which gives each request an expiry and "controls how many times we give
  preference to reads over writes". The draft's "never dirty pages" is clean-first with the window set to the
  whole list; clean-first lowers the hit ratio, which is why the original keeps a window.

## 2. Source by source

Verdict words: correct; reword (how); not used (the draft does not cite it; what it would support).
"Opened" names the page I read; DOIs are in `references.bib`. In the last column, plain section numbers are the
draft's; "notes 1.x" points to section 1 of this file.

### 2.1 Scheduling

| Source | Statement | Opened | Draft's use and verdict |
|---|---|---|---|
| Jackson 1955 | All jobs available: earliest-due-date order minimizes maximum lateness; meets every due date if any order does. | arxiv.org/abs/2001.06005 (Thm 3.1, Cor 3.2; bibliography entry) | 3.4, 7.4. Reword: the criterion is maximum lateness, not the stall sum; optimal for the stall sum only when no read need be late. |
| Horn 1974 | Release times and free preemption: the preemptive earliest-due-date rule minimizes maximum lateness. | api.crossref.org/works/10.1002/nav.3800210113; arXiv 2001.06005 Thm 3.5 and notes | 3.4. Reword: needs free preemption; at page-group boundaries the guarantee is "within one page-group time". |
| Smith 1956 | Largest w/p first minimizes the sum of w C. | Crossref 10.1002/nav.3800030106; arXiv 2001.06005 Thm 4.1 | 0, 3.4, 5.2. Reword: optimal for the stall sum only when every waiting transfer is already late; it is a device rule, not a special case of the priced session. |
| Cox and Smith 1961; Buyukkoc et al. 1985 | Linear delay costs: serve the largest cost rate × service rate. | rand.org/pubs/papers/P4525.html (attribution); Crossref 10.2307/1427064 | Not used by name. "Largest cost per device-second" is this rule. Note the clash of letters with the draft's c and mu. |
| Van Mieghem 1995 | Convex delay costs: serve the class with the largest marginal cost × service rate; asymptotically optimal in heavy traffic. | Crossref 10.1214/aoap/1177004706 (abstract) | Not used. Apt for 5.2 in overload; heavy traffic and convexity are its conditions. |
| Van Mieghem 2003 | Due dates turned into convex delay costs; two dynamic priority rules asymptotically optimal in heavy traffic. | Crossref 10.1287/opre.51.1.113.12793 (abstract) | Not used. The queueing counterpart of 3.4. |
| Lawler 1977; Lenstra, Rinnooy Kan, Brucker 1977 | Sum of weighted tardiness on one machine is NP-hard in the strong sense; with release times and no preemption so is maximum lateness. | Crossref records; statement in arXiv 2001.06005 ch. 3 and ch. 8 | Not cited. The draft's implied optimal order in overload does not exist in general. Attribution of the weighted-tardiness result to these two papers is from memory. |
| Du and Leung 1990 | Sum of tardiness on one machine is NP-hard. | Crossref 10.1287/moor.15.3.483 (abstract) | Same. |
| Vepsalainen and Morton 1987 | Dispatching rules for weighted tardiness (apparent tardiness cost). | Crossref 10.1287/mnsc.33.8.1035 | Not used. Suggested single index in place of the switch. |
| Kleinrock 1965 | Sum of ρ_p W_p is the same for every work-conserving non-preemptive discipline. | Crossref 10.1002/nav.3800120206 (abstract) | Not used. The exact form of "the sum does not change" in 1.5 (c). |

### 2.2 Index theory

| Source | Statement | Opened | Draft's use and verdict |
|---|---|---|---|
| Gittins 1979 | The index of a bandit process (best ratio of expected reward to expected time over stopping times) solves the multi-armed bandit. | Crossref 10.1111/j.2517-6161.1979.tb01068.x (abstract); en.wikipedia.org/wiki/Gittins_index (pointer) | 4.1, 7.2. Reword as in notes 1.2: correct for the stopping problem with no re-read; the ranking is not covered. |
| Whittle 1980 | With the option of retiring for a reward M, the index is the M at which one is indifferent; proves the index rule. | Crossref 10.1111/j.2517-6161.1980.tb01111.x (abstract) | Supports "keep while ν ≥ λ" with a constant drop cost. |
| Aalto, Ayesta, Righter 2009; Scully and Harchol-Balter 2021 | Gittins index of a job of unknown size; falling hazard gives the hazard, rising hazard the inverse mean remaining size; the single-job game has give-up sets monotone in the penalty. | cs.cmu.edu/~harchol/Papers/WIOPT21.pdf (text); research.aalto.fi record | More apt than Gittins 1979 for the formula; cite one of them. The hazard cases of Aalto et al. are from memory, confirmed numerically. |
| Whittle 1988 | Relax "m of n active" to an average; the multiplier defines an index "which reduces to the Gittins index when projects not being operated are static"; conjecture for large n. | Crossref 10.2307/3214163 (abstract) | 1.6. Correct as the relaxation move. "Indexable" in 4.1: reword (notes 1.2, point 3). |
| Weber and Weiss 1990, 1991 | Conjecture true under a globally stable fluid equilibrium; false in general; true for three states. | Crossref 10.2307/3214547, 10.2307/1427757 (abstracts) | 7 item 10. Reword as in notes 1.3. |
| Verloop 2016 | Priority policies asymptotically optimal under a global attractor property, several classes, arrivals; indexability not required. | Crossref 10.1214/15-AAP1137; arxiv.org/abs/1609.00563 | Same. |
| Brown and Smith 2020 | Finite horizon: optimal Lagrangian index policy within order √N of the bound; Whittle index and random tie-breaking can be linear. | author copy (text); Crossref 10.1287/mnsc.2019.3342 | Same. The most apt for a 2-hour run; tie-breaking matters. |
| Hodge and Glazebrook 2015; Graczová and Jacko 2014 | Index heuristics under one resource constraint with several activation levels; knapsack with unequal sizes, where an index-knapsack heuristic beats the plain index order. | Crossref records (abstracts) | Not used. Closest to unequal prefix lengths; supports 5.1's knapsack reading. |
| Scully, Grosof, Harchol-Balter 2020 | With k servers the Gittins order is within an additive term of the one-server optimum (optimal in heavy traffic); not known to be optimal. | Crossref 10.1145/3428328 (abstract) | Not used. Analogy for "many kept at once". |

### 2.3 Relaxations, bounds, hardness

| Source | Statement | Opened | Draft's use and verdict |
|---|---|---|---|
| Adelman and Mersereau 2008 | Lagrangian and linear-programming relaxations of dynamic programs coupled only through constraints on actions; comparison of the two bounds. | Crossref 10.1287/opre.1070.0445 (abstract) | 7 item 7 (deck 1). Correct. |
| Brown and Zhang 2022, 2023 | Shared resources and signals: standard relaxation policies can fail to be asymptotically optimal; the two bounds are very close. | Crossref 10.1287/opre.2021.2181, 10.1287/opre.2022.2287 (abstracts) | Not used. The GPU queue is a shared signal: limits item 10. |
| Papadimitriou and Tsitsiklis 1999 | Restless bandits are complete for polynomial space; queueing network control complete for exponential time. | Crossref 10.1287/moor.24.2.293 (abstract) | Deck 1, design.tex comment. Reword: show the reduction or say "a class for which". |
| Everett 1963 | Lagrange multipliers for allocation among independent activities; any solution found is a true solution. | Crossref 10.1287/opre.11.3.399 (abstract) | design.tex comment. Correct. |
| Dantzig 1957 | Linear-programming methods for discrete problems (the ratio rule for the divisible knapsack). | Crossref 10.1287/opre.5.2.266 (abstract) | 5.1, 6. Correct for the divisible problem; "within one holder" is additive. Content of the ratio rule from memory; checked numerically. |

### 2.4 Caching and prefetching

| Source | Statement | Opened | Draft's use and verdict |
|---|---|---|---|
| Belady 1966; Mattson et al. 1970 | Evict the page used farthest in the future: fewest faults for equal pages. | Crossref 10.1147/sj.52.0078, 10.1147/sj.92.0078; en.wikipedia.org/wiki/Cache_replacement_policies (pointer) | 0, 4.1. Reword: needs equal sizes and equal miss cost (notes 1.6). Texts not opened. |
| Chrobak et al. 2012 | Caching with pages of several sizes is strongly NP-complete in all four standard variants. | Crossref 10.1007/s00453-011-9502-9; Semantic Scholar abstract | Not used. Why "Belady" stops at equal sizes. |
| Cao, Felten, Karlin, Li 1995 | Four rules for one disk and a known reference string; rule 3: "Never discard block A to prefetch block B when A will be referenced before B." | homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf (text) | design.tex Preempt; 5.1. Correct. The rules are stated for equal blocks, one disk, exact knowledge. |
| Kimbrel et al. 1996; Albers, Garg, Leonardi 2000 | "A block prefetched too early reduces the effectiveness of the cache"; aggressive prefetching when stalls limit, conservative otherwise; stall time minimized by a linear program. | pdl.cmu.edu OSDI96 abstract; Crossref 10.1145/355541.355542 (abstract) | Not used. Source of the word "stall" and of the two regimes of section 8. |
| Patterson et al. 1995 | Applications disclose future accesses; a cost-benefit analysis in one unit allocates buffers. | pdl.cmu.edu SOSP15 abstract | Not used. The classical "hint plus one currency" design; cite in related work. |
| Panigrahy et al. 2021, 2022 | Hazard-rate (over size) rule bounds every non-anticipative policy's hits; a rule, not a policy. | arxiv.org/abs/2106.06457 (text); Crossref 10.1145/3453953.3453985, 10.1145/3547332 | Deck 1 for the rate form of Retain. Correct; use it for the hard-capacity statement in notes 1.6. Note the correct DOI of the short paper ends in 985. |
| Ferragut, Rodríguez, Paganini 2016 | Rising hazards: keep the most popular forever. Falling hazards: timers that equalize hazards at one multiplier. | grupo-mate.github.io/assets/pdf/sigmetrics16.pdf (text) | Deck 1. Correct, and it covers both special cases of 4.1. |
| Dehghan et al. 2019 | Utilities of hit probabilities maximized under a capacity; one multiplier; dual algorithm; characteristic time related to the multiplier. | arxiv.org/abs/1601.06838 (text); Crossref 10.1109/TNET.2019.2913677 | Deck 1 for Price. Correct. The draft now computes the price instead of learning it; same dual reading. |
| Che, Tung, Wang 2002; Fricker et al. 2012 | LRU behaves like a timer cache with one characteristic time. | arxiv.org/abs/1202.3974 (text); Crossref 10.1109/JSAC.2002.801752 | Not used. Supports reading λ off the eviction frontier when many prefixes share the room. |
| Shahrad et al. 2020 | Per application: pre-warm window from the 5th percentile of idle times, keep-alive from the 99th; no unloading when the head percentile rounds to 0; fixed keep-alive when the histogram is not representative. | usenix.org/conference/atc20/presentation/shahrad; PDF (text) | Deck 1. Correct. It also shows the draft's split: concentrated idle times get a timed pre-warm, spread ones get retention only. |
| Jiang and Kleinrock 1998 | Prefetch what exceeds a probability threshold set from load, capacity and cost. | Crossref 10.1109/49.669044; index records by search | Not used. Precedent for F ≥ q with q set by load. Text not opened. |

### 2.5 Inventory, prices, queues

| Source | Statement | Opened | Draft's use and verdict |
|---|---|---|---|
| Newsvendor: Arrow, Harris, Marschak 1951 (Edgeworth 1888) | Optimal stock at the quantile underage / (underage + overage). | en.wikipedia.org/wiki/Newsvendor_model (pointer); Crossref 10.2307/1906813 | 0, 3.2. Correct. |
| Yano 1987; Baker and Scudder 1990 | Planned lead times against tardiness and holding costs; earliness and tardiness scheduling. | Crossref 10.1287/mnsc.33.1.95, 10.1287/opre.38.1.22 (abstracts) | Not used. Closer than the newsvendor to "when to release a job with a due date". |
| Little 1961 | L = λ W under stationarity. | Crossref 10.1287/opre.9.3.383 (abstract) | critique.md, 1.5. Correct. |
| Denning and Buzen 1978; Lazowska et al. 1984 | Response time law R = N / X − Z; utilization law U = X S. | homes.cs.washington.edu/~lazowska/qsp/Images/Chap_03.pdf (text); Crossref 10.1145/356733.356735 | 1.5, 7 item 8. Correct; cite this law by name. |
| Naor 1969; Haviv and Ritov 1998 | A customer's cost on others; the part on those present while it is served. | Crossref 10.2307/1909200, 10.1287/mnsc.44.6.850 (abstract) | 1.5 (mu). Reword the open-queue value to (1 + Q) / (1 − ρ). |
| Kelly et al. 1998; Low and Lapsley 1999 | Shadow prices of shared links; "network links adjust bandwidth prices to coordinate the sources' decisions". | Crossref 10.1057/palgrave.jors.2600523; authors.library.caltech.edu/8579 (abstract) | design.tex comment; section 0. Correct for multipliers moved by excess demand. |
| Weitzman 1974 | Prices beat quantities iff costs are more curved than benefits; kinked benefits favour quantities. | ens.psl.eu copy (text); Crossref 10.2307/2296698 | design_economics.md. Correct there; missing from the draft (notes 1.5). |

### 2.6 Flash and writeback

| Source | Statement | Opened | Draft's use and verdict |
|---|---|---|---|
| Park et al. 2006 (CFLRU) | Evict a clean page from the clean-first region before any dirty page. | Crossref 10.1145/1176760.1176789; arxiv.org/pdf/2307.11866 (survey, text) | 5.1 "never dirty pages", R5. Correct as precedent; the motive there is write cost, here restore cost. Paper text not opened. |
| Linux writeback | Background flusher starts at one dirty threshold; at a higher one the writing process writes; data older than an expiry is eligible. | docs.kernel.org/admin-guide/sysctl/vm.html | 5.2, P2 (the 50 K / 25 K rule). Correct as precedent for two thresholds and for delay instead of refusal. |
| Linux deadline scheduler | Each request gets an expiry; reads preferred a set number of times before writes are served. | docs.kernel.org/block/deadline-iosched.html | Not used. The closest existing mechanism to "due dates over reads and writes". |

## 3. Closest systems (positioning only)

One sentence on what each does, then how the four-action framing differs.

1. InferCept (ICML 2024). For a request paused by a tool call it chooses to preserve, discard or swap its KV by
   least memory-time waste, with the pause estimated from the time already elapsed. Difference: GPU and host
   memory only, no SSD, no read timed before the return, no price shared across resources.
2. Continuum (arXiv 2511.02230; ICLR 2026 workshop poster). Pins a tool-calling request's KV in GPU memory for a
   time-to-live chosen from the tool's past durations, with program-level first-come-first-served. Difference: it
   is Retain on the GPU tier with the memory charged for the whole TTL; no Fetch, no write path, no price.
3. TokenCake (arXiv 2510.18586; EuroSys '27 per arXiv). Offloads an agent's KV to host memory during a function
   call and uploads it back near the predicted end, with reserved GPU memory for critical agents. Difference: a
   point estimate (developer hint blended with a moving average), two tiers, no cost model.
4. TripleKV (arXiv 2611.25112). Three tiers; prefetches SSD to host memory at tool-end estimate plus queue delay
   minus read time minus a margin, protects prefetched blocks by class, keeps no host copy of GPU-resident blocks.
   Difference: median and fixed margin instead of a law and a priced quantile; no test of whether a timed read
   pays (falling hazards); first-in-first-out queues with read and write priority threads instead of due dates;
   no optimality statement. Closest work; must be cited and compared.
5. SARIA (arXiv 2610.31169). Writes a block to SSD only when counts show reuse beyond its stay in memory,
   promotes SSD hits to memory, and paces memory reads while SSD transfers run. Difference: no tool-call hints
   and no read before arrival; its write admission is by thresholds, the draft's by a due date.
6. MORI (arXiv 2606.00866). Ranks programs by idleness (share of recent time in tool calls) and enforces
   admission at GPU and host tiers. Difference: Retain order without duration hints; no SSD, no Fetch.
7. CachedAttention (USENIX ATC 2024). Hierarchical KV cache for conversations; fetches from disk for jobs in a
   look-ahead window of the queue and evicts from the queue's tail. Difference: acts only after arrival, with
   windows sized by capacity instead of a landing time; no hints about the gap.
8. KVFlow (arXiv 2507.07400). Evicts by steps-to-execution in an agent step graph and prefetches host to GPU for
   agents scheduled next. Difference: workflow structure instead of time; two tiers.
9. LMCache (arXiv 2510.09665). A KV cache layer that moves KV across GPU, host, storage and network for vLLM and
   SGLang, with a control API. Difference: mechanism and interface; the policy is left to the user.
10. Mooncake (FAST 2025). KVCache-centric disaggregated serving with a global cache over DRAM and SSD of the
    cluster and a cache-aware scheduler. Difference: cluster-level placement and routing, not per-session timing.
11. Pensieve (EuroSys 2025). Keeps conversation state across requests in a GPU and host cache. Difference: two
    tiers, recency-based.
12. Strata (arXiv 2508.18572) and SGLang HiCache (blog, September 2025). GPU-assisted I/O and cache-aware
    scheduling on SGLang; write-through, selective and write-back policies; best-effort, wait-complete and
    timeout prefetch. Difference: the engine and switches this work builds on; no hint-driven policy.
13. HCache (EuroSys 2025 per arXiv). Restores state from saved activations, mixing compute and I/O. Difference: a
    third restore path next to read and recompute; orthogonal to the actions.
14. Autellix (arXiv 2502.13965). Schedules LLM calls by their program's history. Difference: GPU order only.
15. SYMPHONY (arXiv 2412.16434). Uses multi-turn hints to migrate KV off the critical path across a cluster.
    Difference: load balancing across servers.
16. EfficientAgent (arXiv 2609.33762). Sizes the host tier by the reuse working set and stops writing refills
    that are evicted before being read. Difference: capacity planning and a write filter; no timing.
17. MORTIS (arXiv 2611.08638) and PBKV (arXiv 2605.06472). Predict reuse from program structure or past
    workflows to guide retention, eviction and prefetch. Difference: what will be reused, not when.
18. "Ask the Tool, Don't Guess" (arXiv 2609.18849). Tools report progress while they run, which is more accurate
    than any estimate fixed before the call. Difference: a better hint source; it would feed the conditional law
    F(· | a) of the draft.
19. vLLM RFC 68079 (open issue, 17 November 2026). A host-tier eviction policy fed by per-request advice
    (expected return, expiry, done). Difference: an interface proposal for the same hints, advisory only.
20. Further: KVCache in the wild (USENIX ATC 2025; eviction from per-category reuse patterns), CacheGen
    (SIGCOMM 2024; KV compression and streaming), hot-cold HBM and flash tiering (IEEE CAL 2026), PACE
    (arXiv 2611.10240; token-level eviction inside one context, not tiering).

## 4. Numeric checks

All on CPU under `nice -n 10`; no GPU, no container, no file outside this folder. Part f of
`checks/sched_checks.out` is an earlier, coarser run of the comparison in `checks/sched_checks_wtard.out`; use the
latter.

| Script and output | What it checks | Result |
|---|---|---|
| `checks/sched_checks.py` → `.out` | Jackson, Horn, Smith by brute force; no preemption; preemption at boundaries; total tardiness. | Jackson 0 failures in 3,000; Horn 0 in 2,000; Smith 0 in 3,000; non-idling rule without preemption worse in 37 % of instances; boundary preemption never more than g above the optimum; total tardiness counterexample 4.2 s against 3.0 s. |
| `checks/sched_checks_wtard.py` → `.out` | Weighted tardiness: five rules against the optimum over all orders, five tightness regimes. | Table in 1.1. |
| `checks/index_checks.py` → `.out` | ν against hazard and mean remaining gap for six laws; dynamic program against the index rule; three plans for a sharp key; two TTL formulas. | 1.2 and 1.7. |
| `checks/misc_checks.py` → `.out` | Newsvendor quantile; derivative of J_F; greedy knapsack; open-queue externality. | 1.4, 1.5, 1.8. |

## 5. What could not be opened, and what that leaves

- Project Euclid pages render by script; van Mieghem 1995 was read only as the abstract in its Crossref record,
  summarized by the fetch tool (it would not quote it in full).
- Aalto, Ayesta, Righter 2009: record page only (the open copy returned 403). Its two hazard results are from
  memory; the numeric check in `checks/index_checks.out` confirms the mathematics, not the attribution.
- Jackson 1955 is an unpublished report; Cox and Smith 1961 a book: both cited through secondary sources.
- Belady 1966, Mattson et al. 1970, Dantzig 1957, Denning and Buzen 1978, Gittins 1979, Whittle 1980 and 1988,
  Park et al. 2006, Jiang and Kleinrock 1998: bibliographic data verified; statements from abstracts, secondary
  sources or memory as marked.
- The attribution of strong NP-hardness of weighted tardiness to Lawler 1977 and to Lenstra, Rinnooy Kan and
  Brucker 1977 is from memory; the fact itself is in Lenstra and Shmoys, chapter 8.
- The 2026 arXiv papers were read as abstracts, plus the HTML text of TripleKV, SARIA, MORI, Continuum,
  TokenCake and InferCept through the fetch tool's summaries; details beyond those listed are unverified.
- The fetch tool summarizes pages with a small model. Where it paraphrased, I say "abstract" and quote nothing.
  One of its paraphrases was wrong (it called weighted tardiness weakly NP-hard from its own memory); the
  statement used here is from the book's text.
