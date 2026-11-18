# Caching theory for stochastic request processes: memory prices, hazard-rate keep rules and eviction indices

Verified 2026-11-17. arXiv records were opened as arxiv.org abstract pages, arxiv.org search listings or arXiv PDFs; the export.arxiv.org API returned HTTP 429 throughout. Venue versions and classic papers were checked against Crossref records, publisher or USENIX pages, or author-hosted PDFs. Each paper is tagged with the design component it parallels:

- **RETAIN**: keep while μ(t)·r ≥ q, with q = λL/(1+λL).
- **INDEX**: ν = (b/β)(1+λL)·max_H F(H|t)/τ(H).
- **PRICE**: Lagrangian λ with multiplicative dual ascent.
- **BOUND**: the weak-duality lower bound.
- **FETCH**: the critical-ratio read start.

## Q0. Which papers come closest to our retention rule, eviction index and memory price? (ranked, verified list)

### Takeaway
A hazard-rate keep rule whose threshold is the Lagrange multiplier of the memory constraint is established prior art, but only for unit-size items, hit-rate objectives and free, instant fetches.

- **Ferragut–Rodríguez–Paganini (SIGMETRICS 2016)** prove that, for decreasing hazard rates, the optimal TTL sets each timer where the hazard rate equals the memory multiplier.
- **Ferragut–Carrasco–Paganini (Allerton 2024) and Carrasco–Ferragut–Paganini (arXiv 2412.00279; Stochastic Systems 2026)** extend this to "store an item iff its current hazard rate exceeds a threshold". It covers both caching (decreasing hazard) and pre-fetching (increasing hazard). For renewal arrivals, ranking by current hazard rate is the optimal causal policy.
- **The index:** our max-over-horizons index is the M/G/1 Gittins index for jobs of unknown size. Its closest caching precedent is Bilardi–Versaci's Least Profit Rate policy (TCS 2013), which also treats occupancy as a cost and decomposes the problem by item.

We found no paper that combines all of the following:

- a TTFT (latency) cost;
- a reload time that grows with size, plus a charge for the memory the reload holds while in flight;
- per-session conditional forecasts;
- a Gittins-type horizon maximum, motivated by the fact that refetching is costly;
- an online multiplicative price driven by refused or preempted demand.

### Cited Findings

#### Ranking at a glance (closest first; full entries in A and B below)

| # | Paper (first author, year, venue) | ID / DOI | Parallels | Prior-art flag |
|---|---|---|---|---|
| 1 | Ferragut et al., 2024, Allerton: "Caching or Pre-fetching? The Role of Hazard Rates" | not on arXiv; [10.1109/allerton63246.2024.10735277](https://doi.org/10.1109/allerton63246.2024.10735277) (B1) | RETAIN, FETCH, PRICE | **Yes**: store iff hazard rate > θ (θ = Lagrange multiplier), for caching and pre-fetching |
| 2 | Carrasco et al., 2024/2026, Stochastic Systems | [arXiv:2412.00279](https://arxiv.org/abs/2412.00279) (A1) | RETAIN, INDEX, PRICE, averages-vs-instants | **Yes**: rank by current hazard rate; fixed-threshold limit; soft average-occupancy constraint |
| 3 | Ferragut et al., 2016, SIGMETRICS (+ Queueing Systems journal version) | not on arXiv; [10.1145/2896377.2901459](https://doi.org/10.1145/2896377.2901459) (B2) | RETAIN, PRICE | **Yes**: decreasing hazard → η(T\*) = p\* (hazard = memory multiplier) |
| 4 | Bilardi & Versaci, 2011/2013, TCS | [arXiv:1109.6643](https://arxiv.org/abs/1109.6643) (A2) | INDEX, PRICE, BOUND | **Yes**: occupancy as cost, per-item decomposition, max-ratio "profit rate" index |
| 5 | Panigrahy et al., 2020–2022, IFIP Performance / TOMPECS | [arXiv:2106.06457](https://arxiv.org/abs/2106.06457) (A3) | INDEX, BOUND | **Yes** for hazard-rate ranking (HR, HR/size) |
| 6 | Beckmann & Sanchez, 2017, HPCA (EVA) | not on arXiv; [10.1109/hpca.2017.43](https://doi.org/10.1109/hpca.2017.43) (B4) | INDEX, PRICE | **Yes** for charging candidates an opportunity cost per unit space-time |
| 7 | Scully & Harchol-Balter, 2021, WiOpt (Gittins in M/G/1) | [arXiv:2111.10703](https://arxiv.org/abs/2111.10703) (A4) | INDEX | Same functional form as our index (scheduling, not caching) |
| 8 | Dehghan et al., 2016, INFOCOM / 2019 ToN | [arXiv:1601.06838](https://arxiv.org/abs/1601.06838) (A5) | PRICE | **Yes** for an online dual price on cache capacity (occupancy − capacity) |
| 9 | Panigrahy et al., 2017 (stationary requests) | [arXiv:1712.07307](https://arxiv.org/abs/1712.07307) (A6) | PRICE, RETAIN | Partial: convexity under non-increasing hazard; dual implementation of LRU |
| 10 | Beckmann et al., 2018, NSDI (LHD) | not on arXiv; [USENIX](https://www.usenix.org/conference/nsdi18/presentation/beckmann) (B5) | INDEX | Hits per space-time at a single horizon |
| 11 | Carra et al., 2018/2020 ToN (+ INFOCOM 2019) | [arXiv:1802.04696](https://arxiv.org/abs/1802.04696) (A7) | PRICE (exogenous), online adaptation | Partial: storage cost per unit time + miss cost |
| 12 | Lund et al., 1994 FOCS / 1999 JCSS (distributional paging) | not on arXiv; [10.1006/jcss.1997.1498](https://doi.org/10.1006/jcss.1997.1498) (B6) | INDEX (distributional forecasts) | **Yes** for eviction from a probabilistic model of future requests (pairwise "requested first" probabilities; median rule) |
| 13 | Pabbaraju & Vakilian, 2025 (Markov paging) | [arXiv:2502.05511](https://arxiv.org/abs/2502.05511) (A8) | INDEX | Improved bounds for #12 |
| 14 | Hu et al., 2022, CoNEXT (Raven) | not on arXiv; [10.1145/3555050.3569134](https://doi.org/10.1145/3555050.3569134) (B7) | INDEX (learned distributions) | Learned next-arrival distributions in eviction |
| 15 | Bertsimas & Niño-Mora, 2000, Oper. Res. | not on arXiv; [10.1287/opre.48.1.80.12444](https://doi.org/10.1287/opre.48.1.80.12444) (B8) | BOUND, INDEX from duals | Generic restless-bandit precedent |
| 16 | Whittle-index caching (Pavamana & Singh; Koley & Singh; Xiong et al.) | arXiv 2304.12227, 2410.18627, 2404.12468, 2202.13187 (A9) | PRICE (subsidy), INDEX | Restless-bandit caching formulations |
| 17 | Elsayed et al., 2024, IFIP Networking | [arXiv:2405.04402](https://arxiv.org/abs/2405.04402) (A10) | FETCH-adjacent (random fetch delay) | TTL utility with fetch delays |
| 18 | Basu et al., 2017/2018 ToN | [arXiv:1704.04448](https://arxiv.org/abs/1704.04448) (A11) | online adaptation | Adaptive TTL toward targets, no price |
| 19 | Yan et al., 2021 CoNEXT (LHR); Torabi et al., 2024 (HR-Cache) | not on arXiv (B9); [arXiv:2402.02795](https://arxiv.org/abs/2402.02795) (A12) | INDEX (learned HR ordering) | Learning systems that imitate the HR bound |
| 20 | Atre et al., 2020, SIGCOMM (delayed hits) | not on arXiv; [10.1145/3387514.3405883](https://doi.org/10.1145/3387514.3405883) (B10) | latency cost, in-flight fetch | Latency-aware caching |
| 21 | Five-minute rule (Gray & Putzolu 1987; Zhang et al. 2025/ISCA 2026) | B12; [arXiv:2511.03944](https://arxiv.org/abs/2511.03944) (A13) | PRICE (memory vs access) | Economic break-even residency |
| 22 | Size- and cost-aware caching (GDS, Landlord, Irani, FOO, AdaptSize, Hyperbolic, access-time-aware) | A14, A15, B13 | miss cost by size | Size-aware but not priced/forecast-based |
| 23 | Learning-augmented paging with point predictions; learned Belady (LRB, Parrot, Hawkeye, Mockingjay) | A16, A17, B14 | INDEX inputs | Point predictions / Belady imitation |
| 24 | Foundations (Gittins, Whittle, Weber–Weiss, Everett, Kelly et al., Belady, Aho–Denning–Ullman, Sevcik, Markov paging) | B15–B17 | all | Already cited or background |

#### A. Papers on arXiv (closest first)

**A1. "Optimal local storage policy based on stochastic intensities and its large scale behavior"**
- *Bibliography:* Carrasco, M., Ferragut, A. and Paganini, F. [arXiv:2412.00279](https://arxiv.org/abs/2412.00279), submitted 29 Nov 2024. Journal version: "Optimal Local Storage Policy Based on Stochastic Intensities and Its Large-Scale Behavior", *Stochastic Systems* 16:211–240 (2026), [DOI 10.1287/stsy.2024.0093](https://doi.org/10.1287/stsy.2024.0093).
- *Result.* The paper characterizes the optimal causal (non-anticipative) memory policy for independent stationary request processes. At every instant it stores the C items with the largest stochastic intensities (Theorem 1). For renewal arrivals, that means ranking items by their current interval hazard rate and storing the top C (Corollary 1). The authors note that "This proposal was already presented in Panigrahy et al. (2022)" ([preprint PDF](https://aferragu.github.io/assets/pdf/stochsys24_arxiv.pdf)).
  - In a large-catalog limit, the random threshold converges to a deterministic constant ([arXiv abstract](https://arxiv.org/abs/2412.00279)).
  - Section 6 studies fixed-threshold policies C_θ(t) = {i : λ_i(t) > θ}. These satisfy only "the soft constraint where the average memory occupation is C", and the section connects them to timer policies when hazard rates are monotone ([preprint PDF](https://aferragu.github.io/assets/pdf/stochsys24_arxiv.pdf)).
  - A causal policy is any predictable set-valued process of at most C items (Definition 4). An item can therefore be in memory at any time without a request, which means fetching is implicitly free and instantaneous ([preprint PDF](https://aferragu.github.io/assets/pdf/stochsys24_arxiv.pdf)).
- *Relation to our design.* Parallels RETAIN, INDEX and PRICE, plus our "from averages to instants" discussion.
  - **Shares:** the decision variable is the conditional arrival intensity at the current idle age. A threshold enforces an average-occupancy constraint, and a top-C ranking enforces the hard one, matching our split between the price and the eviction index.
  - **Differs:**
    - hit probability rather than TTFT;
    - equal-size items with unit miss cost;
    - no read time, bandwidth or in-flight memory;
    - the threshold is the limit of a quantile, not a price learned online;
    - free re-insertion means a pure current-hazard ranking is optimal even for non-monotone hazards, whereas our costly re-reads motivate the Gittins max over H.
  - **PRIOR-ART FLAG:** hazard-rate keep rule and hazard-rate eviction ranking under a soft (priced) occupancy constraint.

**A2. "Optimal Eviction Policies for Stochastic Address Traces"**
- *Bibliography:* Bilardi, G. and Versaci, F. [arXiv:1109.6643](https://arxiv.org/abs/1109.6643) (v1 Sept 2011). *Theoretical Computer Science* 514:36–60 (2013), [DOI 10.1016/j.tcs.2013.01.016](https://doi.org/10.1016/j.tcs.2013.01.016).
- *Result.* Besides the fixed-capacity problem, the paper studies "a relaxed version ... in which buffer occupancy can vary and its average is constrained". "Viewing occupancy as a cost rather than as a constraint", it builds the optimal eviction policy by composing solutions for the individual items ([arXiv PDF](https://arxiv.org/pdf/1109.6643)).
  - Under the LRU stack model, the optimal policy for any target average occupancy is also optimal at fixed capacity. It is described by priorities, the Least Profit Rate (LPR) policy.
  - An item at stack depth i has profit rate ξ(i) = max_j s̄(i, j), where s̄(i, j) is the average of the stack-distance probabilities from depth i to j. The paper calls this "the best achievable ratio between expected hits and the expected occupancy for that item". LPR evicts argmin ξ.
  - Its K-L policies include LRU and MRU as special cases. The ratio of LPR's miss rate to OPT's (Belady's) is O(log C), and O(1) under mild conditions on the model.
- *Relation to our design.* Parallels PRICE (occupancy priced and decoupled per item), INDEX (max over horizons of hits per unit occupancy) and BOUND (the relaxed average-occupancy optimum).
  - **Shares:** the relaxation, the per-item decomposition, and an index that maximizes a hit-per-occupancy ratio over how long or deep to keep the item. This is the discrete analog of max_H F(H|t)/τ(H).
  - **Differs:**
    - a single shared stack-distance distribution, not per-item arrival forecasts;
    - unit size and unit miss cost;
    - discrete reference time;
    - no fetch latency;
    - no online price.
  - **PRIOR-ART FLAG:** priced-occupancy formulation plus a max-ratio eviction index in caching.

**A3. "A New Upper Bound on Cache Hit Probability for Non-anticipative Caching Policies"**
- *Bibliography:* Panigrahy, N. K., Nain, P., Neglia, G. and Towsley, D. [arXiv:2106.06457](https://arxiv.org/abs/2106.06457) (June 2021; comment "IFIP WG 7.3 Performance").
  - Conference version: *ACM SIGMETRICS PER* 48:138–143 (2021), [DOI 10.1145/3453953.3453985](https://doi.org/10.1145/3453953.3453985).
  - Extended version: *ACM TOMPECS* 7:1–24 (2022), [DOI 10.1145/3547332](https://doi.org/10.1145/3547332).
- *Result.* The hazard rate is defined as the conditional intensity given the request history. Rule HR-E caches, at every instant, the B objects with the largest hazard rates. Theorem 1 shows that its stationary hit probability is at least that of any non-anticipative policy ([arXiv PDF](https://arxiv.org/pdf/2106.06457)).
  - For variable sizes, byte hits reduce to a fractional knapsack ordered by λ_i(t) alone, because size cancels. Object hits reduce to a 0-1 knapsack whose relaxation orders by λ_i(t)/s_i (HR-VC).
  - The authors stress that "we do not provide new caching policies ... but instead we provide various upper bounds".
  - In simulations the HR bound is tighter than Belady's offline bound, and the two are close when inter-request times have increasing hazard (IHR) ([arXiv PDF](https://arxiv.org/pdf/2106.06457); [abstract](https://arxiv.org/abs/2106.06457)).
- *Relation to our design.* Parallels INDEX and BOUND.
  - **Shares:** ranking by current conditional hazard, normalized by size, using distributional knowledge only. It bounds every non-anticipative policy, like our weak-duality bound.
  - **Differs:**
    - hit probability, not TTFT;
    - the bound is a sample-path argument that assumes free insertion, not a Lagrangian bound;
    - no memory price;
    - no reload time or in-flight memory.
  - Our per-token reload cost (b/β) is the byte-hit case, where size cancels. Our extra (1+λL) factor has no counterpart.
  - **PRIOR-ART FLAG:** hazard-rate (and HR/size) eviction ranking.

**A4. "The Gittins Policy in the M/G/1 Queue"**
- *Bibliography:* Scully, Z. and Harchol-Balter, M. [arXiv:2111.10703](https://arxiv.org/abs/2111.10703) (Nov 2021; "Originally published at WiOpt 2021").
- *Result.* The paper gives the first fully general proof of Gittins's optimality in the M/G/1 ([abstract](https://arxiv.org/abs/2111.10703)).
  - For unknown service times, with attained service x as the state and holding cost 1, it states the index as "a formula well-known in the literature": index(x) = sup_{y>x} P[S ≤ y | S > x] / E[min{S, y} − x | S > x].
  - The rank is 1/index. With known service times, rank equals remaining service time and Gittins reduces to SRPT ([arXiv PDF](https://arxiv.org/pdf/2111.10703)).
- *Relation to our design.* Parallels INDEX.
  - **Shares:** our ν/((b/β)(1+λL)) is exactly this index. Take S as the idle gap, x as the current idle age a, and y = a + H. The numerator is then F(H|t) and the denominator is τ(H).
  - Our "exact forecasts → Belady" reduction mirrors "known sizes → SRPT".
  - **Differs:** it is a scheduling result (serve the job with the largest index to minimize response time), with no memory, price or size-weighted cost. We use the index to keep caches with the largest expected reload cost avoided per token-second.

**A5. "A Utility Optimization Approach to Network Cache Design"**
- *Bibliography:* Dehghan, M., Massoulié, L., Towsley, D., Menasché, D. and Tay, Y. C. [arXiv:1601.06838](https://arxiv.org/abs/1601.06838) (Jan 2016; IEEE INFOCOM 2016, [DOI 10.1109/infocom.2016.7524445](https://doi.org/10.1109/infocom.2016.7524445)). Journal version: *IEEE/ACM ToN* 27:1013–1027 (2019), [DOI 10.1109/tnet.2019.2913677](https://doi.org/10.1109/tnet.2019.2913677).
- *Result.* Cache utility maximization over hit probabilities with TTL caches under Poisson arrivals. LRU's and FIFO's characteristic time is tied to the Lagrange multiplier α of the capacity constraint ([arXiv PDF](https://arxiv.org/pdf/1601.06838)).
  - Online dual algorithm: t_i = −(1/λ_i)·log(1 − U_i′⁻¹(α)) and α ← max{0, α + γ(B_curr − B)}, where B_curr is the current number of cached items. LRU and FIFO are emulated with t_i = 1/α without precomputing the characteristic time.
  - The framework supports "revenue based models where content publishers react to prices set by (cache) service providers" ([arXiv PDF](https://arxiv.org/pdf/1601.06838)).
- *Relation to our design.* Parallels PRICE.
  - **Shares:** the capacity multiplier decouples contents and is updated online by a subgradient step on occupancy minus capacity, the additive counterpart of our λ ← λe^{ηg}.
  - **Differs:**
    - Poisson arrivals (constant hazard, so timers do not depend on age);
    - hit-probability utilities;
    - unit size;
    - additive step;
    - excess is measured as occupancy above B, whereas our memory never exceeds R, so excess is counted as refused or preempted demand.
  - **PRIOR-ART FLAG:** online dual price on cache capacity.

**A6. "Network Cache Design under Stationary Requests: Exact Analysis and Poisson Approximation"**
- *Bibliography:* Panigrahy, N. K., Li, J., Towsley, D. and Hollot, C. V. [arXiv:1712.07307](https://arxiv.org/abs/1712.07307) (Dec 2017).
- *Result.* Utility maximization for stationary and ergodic request processes. The problem "is non-convex in general but we reformulate it as a convex optimization problem when the inter-request time (irt) distribution has a non-increasing hazard rate function". The paper also gives a "reverse engineering based dual implementation of LRU under stationary arrivals" and decentralized algorithms whose stability it proves with a discrete-time Lyapunov technique ([abstract](https://arxiv.org/abs/1712.07307)).
- *Relation to our design.* Parallels PRICE and RETAIN.
  - **Shares:** Lagrangian duality with renewal inter-arrivals, where tractability hinges on decreasing hazard.
  - **Differs:** hit-rate or hit-probability utilities, unit size, no fetch latency, no forecasts per session.

**A7. "Elastic Provisioning of Cloud Caches: a Cost-aware TTL Approach"**
- *Bibliography:* Carra, D., Neglia, G. and Michiardi, P. [arXiv:1802.04696](https://arxiv.org/abs/1802.04696) (Feb 2018). *IEEE/ACM ToN* 28:1283–1296 (2020), [DOI 10.1109/tnet.2020.2980105](https://doi.org/10.1109/tnet.2020.2980105). See also B11.
- *Result.* Total cost = storage cost + miss cost. Storage is c_s per cache instance per billing epoch, set by the cloud provider's pricing. Each object o has its own miss cost m_o ([arXiv PDF](https://arxiv.org/pdf/1802.04696)).
  - A virtual TTL cache's TTL "is adapted through stochastic approximation iterations", and its virtual size sets the number of instances.
  - The abstract reports a 17% cost reduction versus a static configuration on Akamai traces ([abstract](https://arxiv.org/abs/1802.04696)).
- *Relation to our design.* Parallels PRICE.
  - **Shares:** occupancy is priced per unit time and traded against miss costs, with online adaptation.
  - **Differs:**
    - an exogenous monetary price, not a shadow price;
    - one global TTL, not per-item hazard thresholds;
    - elastic capacity rather than a fixed R;
    - no forecasts.

**A8. "New and Improved Bounds for Markov Paging"**
- *Bibliography:* Pabbaraju, C. and Vakilian, A. [arXiv:2502.05511](https://arxiv.org/abs/2502.05511) (Feb 2025; v2 "Amended the guarantee for the median algorithm"). The venue is not confirmed on the arXiv page.
- *Result.* In Markov paging, computing the optimal online algorithm takes time exponential in the cache size. Lund, Phillips and Reingold's dominating-distribution algorithm was 4-competitive against the optimal online algorithm; this paper shows it is 2-competitive and proves a 1.5907 lower bound ([abstract](https://arxiv.org/abs/2502.05511)).
  - The algorithm evicts from a distribution μ such that, for every cached q, E_{p∼μ}[Pr(p requested before q)] ≤ 1/2.
  - The "median algorithm" evicts the cached page with the largest median time to its next request. It is 4-competitive under independent inter-arrival times; an earlier version wrongly claimed this held in the general Markov model ([arXiv HTML v2](https://arxiv.org/html/2502.05511v2)).
- *Relation to our design.* Parallels INDEX (eviction from distributional forecasts).
  - **Shares:** eviction uses full next-request distributions. The median rule is exactly the "point estimate taken from a distribution" that our design argues against for fetch timing.
  - **Differs:** fault counting with unit sizes, no memory-time cost, no price, and competitive analysis against the optimal online algorithm.

**A9. Whittle-index (restless-bandit) caching formulations**
- *Papers:*
  - Pavamana K J and Singh, C., "Caching Contents with Varying Popularity using Restless Bandits", [arXiv:2304.12227](https://arxiv.org/abs/2304.12227) (2023; overlaps arXiv:2212.03291).
  - Koley, A. and Singh, C., "Dynamic Content Caching with Waiting Costs via Restless Multi-Armed Bandits", [arXiv:2410.18627](https://arxiv.org/abs/2410.18627) (2024, revised 2026).
  - Koley, A. and Singh, C., "Fresh Caching of Dynamic Contents using Restless Multi-armed Bandits", [arXiv:2404.12468](https://arxiv.org/abs/2404.12468) (2024).
  - Xiong, G., Wang, S., Li, J. and Singh, R., "Whittle Index based Q-Learning for Wireless Edge Caching with Linear Function Approximation", [arXiv:2202.13187](https://arxiv.org/abs/2202.13187) (2022).
- *Results:*
  - Pavamana & Singh pose proactive caching under time-varying popularity, with caching and missing costs, as a discounted-cost restless bandit. They give indexability conditions and a Whittle index policy ([abstract](https://arxiv.org/abs/2304.12227)).
  - Koley & Singh (2410.18627) model serve-stale, fetch or wait decisions, with age-of-version, fetching and per-unit-time waiting costs, under a per-epoch capacity constraint. They characterize the Whittle indices explicitly ([abstract](https://arxiv.org/abs/2410.18627)).
  - Koley & Singh (2404.12468) handle partially observed age-of-version via a semi-Markov reformulation ([abstract](https://arxiv.org/abs/2404.12468)).
  - Xiong et al. prove threshold-type optimal policies and indexability for latency-minimizing edge caching, and learn Whittle indices ([abstract](https://arxiv.org/abs/2202.13187)).
- *Relation to our design.* Parallels PRICE (the Whittle subsidy is a price per cache slot) and INDEX.
  - **Shares:** the Lagrangian relaxation of a per-instant capacity constraint and indices from per-arm subproblems.
  - **Differs:**
    - states are popularity or freshness, not the idle age of a renewal gap;
    - unit sizes;
    - no per-session forecasts;
    - none uses the Gittins completion-rate form on time-to-next-request.

**A10. "Utility-driven Optimization of TTL Cache Hierarchies under Network Delays"**
- *Bibliography:* Elsayed, K. S., Geyer, F. and Rizk, A. [arXiv:2405.04402](https://arxiv.org/abs/2405.04402) (May 2024). IFIP Networking 2024, pp. 249–257, [DOI 10.23919/ifipnetworking62109.2024.10619843](https://doi.org/10.23919/ifipnetworking62109.2024.10619843). Journal follow-up in *Computer Networks* 280:112180 (2026), [DOI 10.1016/j.comnet.2026.112180](https://doi.org/10.1016/j.comnet.2026.112180).
- *Result.* Per-object TTL utility maximization in which "upon a miss the object requires a random time to be fetched from a parent cache". It uses the exact hit probability under random delays, solves iteratively, and uses machine learning to estimate optimal TTLs for large hierarchies ([abstract](https://arxiv.org/abs/2405.04402)).
- *Relation to our design.* Adjacent to FETCH and RETAIN.
  - **Shares:** fetch latency enters the TTL optimization, as r does for us.
  - **Differs:** hit-probability utility, no memory price learned online, no per-session forecasts, no read-ahead decision.

**A11. "Adaptive TTL-Based Caching for Content Delivery"**
- *Bibliography:* Basu, S., Sundarrajan, A., Ghaderi, J., Shakkottai, S. and Sitaraman, R. [arXiv:1704.04448](https://arxiv.org/abs/1704.04448) (Apr 2017).
  - SIGMETRICS 2017 two-page abstract, pp. 45–46: [DOI 10.1145/3078505.3078560](https://doi.org/10.1145/3078505.3078560) (also *PER* 45, [DOI 10.1145/3143314.3078560](https://doi.org/10.1145/3143314.3078560)).
  - Journal version: *IEEE/ACM ToN* 26:1063–1077 (2018), [DOI 10.1109/tnet.2018.2818468](https://doi.org/10.1109/tnet.2018.2818468).
- *Result.* d-TTL "dynamically adapts a TTL parameter using a stochastic approximation approach" and converges to a feasible target hit rate under bursty, non-stationary traffic. f-TTL uses two caches to meet targets for both hit rate and expected cache size ([abstract](https://arxiv.org/abs/1704.04448)).
- *Relation to our design.* Parallels the online adaptation loop.
  - **Shares:** a scalar control is adjusted from observed error.
  - **Differs:** it tracks targets (no Lagrangian or price), uses one TTL rather than per-item hazard, and has no size-dependent cost or forecasts.
  - Not prior art for a hazard keep rule.

**A12. "A Learning-Based Caching Mechanism for Edge Content Delivery" (HR-Cache)**
- *Bibliography:* Torabi, H., Khazaei, H. and Litoiu, M. [arXiv:2402.02795](https://arxiv.org/abs/2402.02795) (Feb 2024).
- *Result.* HR-Cache is "grounded in the principles of Hazard Rate (HR) ordering". It labels requests as cache-friendly or cache-averse from HR-ordering decisions, learns a lightweight predictor, and reports 2.2–14.6% more WAN traffic savings than LRU ([abstract](https://arxiv.org/abs/2402.02795)).
- *Relation to our design.* Parallels INDEX.
  - **Shares:** shows that hazard-rate ordering is a practical target to learn.
  - **Differs:** byte-hit objective, binary classification, no price or forecasts.

**A13. "Five-Minute Rule 40 Years Later: A First-Principles Revisit for Modern Memory Hierarchy"**
- *Bibliography:* Zhang, T. et al. [arXiv:2511.03944](https://arxiv.org/abs/2511.03944) (Nov 2025; journal-ref "International Symposium on Computer Architecture (ISCA), 2026").
- *Result.* Revisits Gray and Putzolu's storage-memory economics rule. Adding host costs and SSD performance and cost models, it finds that "for modern AI platforms, especially GPU-centric hosts paired with ultra-high-IOPS SSDs ... the DRAM↔flash caching threshold collapses from minutes to a few seconds" ([abstract](https://arxiv.org/abs/2511.03944)).
- *Relation to our design.* Parallels PRICE and RETAIN in deterministic, economic form.
  - **Shares:** a break-even residency set by the cost of memory versus the cost of a storage access.
  - **Differs:** money rather than TTFT, and an average access rate rather than a conditional hazard.
  - Useful as motivation that host-DRAM retention of KV may pay for only seconds.

**A14. "Practical Bounds on Optimal Caching with Variable Object Sizes"**
- *Bibliography:* Berger, D. S., Beckmann, N. and Harchol-Balter, M. [arXiv:1711.03709](https://arxiv.org/abs/1711.03709). POMACS 2(2), Art. 32 (2018), [DOI 10.1145/3224427](https://doi.org/10.1145/3224427).
- *Result.* With variable sizes, computing OPT is NP-hard. FOO models caching as min-cost flow to obtain upper and lower bounds on OPT, which are tight as the number of objects grows under independence assumptions. PFOO scales to full traces and shows that current systems are far from OPT ([abstract](https://arxiv.org/abs/1711.03709)).
- *Relation to our design.* Parallels BOUND, as an offline reference.
  - **Differs:** a clairvoyant, offline miss-ratio bound, whereas ours is a non-anticipative relaxation bound in TTFT given forecasts.

**A15. "On-Line File Caching" (Landlord)**
- *Bibliography:* Young, N. E. [arXiv:cs/0205033](https://arxiv.org/abs/cs/0205033). SODA 1998; *Algorithmica* 33:371–383 (2002), [DOI 10.1007/s00453-001-0124-5](https://doi.org/10.1007/s00453-001-0124-5).
- *Result.* Files have arbitrary sizes and retrieval costs. Landlord, "a natural generalization of LRU", has an optimal guarantee among deterministic online algorithms, plus a "loosely competitive" analysis ([abstract](https://arxiv.org/abs/cs/0205033)).
- *Relation to our design.* Adjacent to miss cost by size.
  - **Shares:** size- and cost-aware eviction.
  - **Differs:** adversarial (worst-case) analysis, with no distributions or price.

**A16. Learning-augmented paging with point predictions**
- *Papers:*
  - Lykouris, T. and Vassilvitskii, S., "Competitive caching with machine learned advice", [arXiv:1802.05399](https://arxiv.org/abs/1802.05399) (ICML 2018; *JACM* 68 (2021), [DOI 10.1145/3447579](https://doi.org/10.1145/3447579)).
  - Bansal, N., Coester, C., Kumar, R., Purohit, M. and Vee, E., "Learning-Augmented Weighted Paging", [arXiv:2011.09076](https://arxiv.org/abs/2011.09076) (SODA 2022, [DOI 10.1137/1.9781611977073.4](https://doi.org/10.1137/1.9781611977073.4)).
  - Jiang, Z., Panigrahi, D. and Sun, K., "Online Algorithms for Weighted Paging with Predictions", [arXiv:2006.09509](https://arxiv.org/abs/2006.09509) (ICALP 2020; *TALG* 18 (2022), [DOI 10.1145/3548774](https://doi.org/10.1145/3548774)).
  - Antoniadis, A. et al., "Paging with Succinct Predictions", [arXiv:2210.02775](https://arxiv.org/abs/2210.02775) (2022).
- *Results:*
  - Lykouris & Vassilvitskii treat the oracle as a black box and modify Marker. The competitive ratio falls with oracle error and is capped at O(log k) ([abstract](https://arxiv.org/abs/1802.05399)).
  - Bansal et al. extend that model, in which "predictions ... about the next arrival of each page" are given, to weighted paging, with ℓ-competitive deterministic and O(log ℓ)-competitive randomized algorithms for perfect predictions ([abstract](https://arxiv.org/abs/2011.09076)).
  - Jiang et al. show that in weighted paging, knowing the next request of every page is not enough; adding lookahead (their SPRP model) gives a 2-competitive algorithm ([abstract](https://arxiv.org/abs/2006.09509)).
  - Antoniadis et al. list prior prediction types: reoccurrence time, optimal cache state, requests until reuse, and relative order. They study 1-bit predictions ([abstract](https://arxiv.org/abs/2210.02775)).
- *Relation to our design.* INDEX inputs.
  - **Shares:** eviction driven by next-arrival information.
  - **Differs:** point or bit predictions, not distributions; worst-case analysis; no memory-time price.
  - Our exact-forecast limit (Belady) is the perfect-prediction case.

**A17. Learned approximations of Belady**
- *Papers:*
  - Liu, E. Z. et al., "An Imitation Learning Approach for Cache Replacement", [arXiv:2006.16239](https://arxiv.org/abs/2006.16239) (ICML 2020).
  - Wang, P. and Liu, Y., "Optimizing Replacement Policies for Content Delivery Network Caching: Beyond Belady to Attain A Seemingly Unattainable Byte Miss Ratio", [arXiv:2212.13671](https://arxiv.org/abs/2212.13671) (2022).
- *Results:*
  - Parrot trains a policy conditioned only on past accesses to imitate Belady's decisions ([abstract](https://arxiv.org/abs/2006.16239)).
  - Wang & Liu show that approximating Belady optimizes object miss ratio but not byte miss ratio with variable sizes, and propose LRU-BaSE ([abstract](https://arxiv.org/abs/2212.13671)).
- *Relation to our design.* INDEX.
  - **Differs:** these learn Belady's clairvoyant decisions, whereas we compute a Gittins-type index from explicit conditional distributions with a cost and price.

**A18. Computing and characterizing Gittins indices**
- *Papers:*
  - Scully, Z., Harchol-Balter, M. and Scheller-Wolf, A., "SOAP: One Clean Analysis of All Age-Based Scheduling Policies", [arXiv:1712.00790](https://arxiv.org/abs/1712.00790) (POMACS 2(1) 2018, [DOI 10.1145/3179419](https://doi.org/10.1145/3179419)).
  - Gast, N., Gaujal, B. and Obrecht, A., "Scheduling jobs with unknown size distribution in a M/G/1 queue: the shifted empirical Gittins", [arXiv:2606.24703](https://arxiv.org/abs/2606.24703) (June 2026).
  - Aalto, S., "Characterization of the Gittins index for sequential multistage jobs", [arXiv:2103.10646](https://arxiv.org/abs/2103.10646) (2021).
- *Results:*
  - SOAP analyzes any age-based priority policy, Gittins included ([abstract](https://arxiv.org/abs/1712.00790)).
  - Gast et al. compute indices from n samples of the job-size distribution so that the index policy is asymptotically optimal as n grows ([abstract](https://arxiv.org/abs/2606.24703)).
  - Aalto notes that the Gittins index is "thoroughly characterized for ordinary jobs whose state is described by the attained service" and treats stages with monotone hazard rates ([abstract](https://arxiv.org/abs/2103.10646)).
- *Relation to our design.* INDEX.
  - Our forecasts are empirical tables of past gaps, so computing ν from samples is the problem Gast et al. study. Their shifted-empirical construction is a candidate estimator.

**A19. Passing mentions, out of scope (LLM serving; other researchers cover these)**
- "Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live", Li, H. et al., [arXiv:2511.02230](https://arxiv.org/abs/2511.02230) (title and listing only).
- "Queuing for Inference: Dynamic Memory Footprints and Costly Preemption", Ni, J. and Tripathi, V., [arXiv:2610.33923](https://arxiv.org/abs/2610.33923) (Oct 2026). It reports that Gittins "can incur substantial preemption overhead" when preemption must load or unload cache state ([abstract](https://arxiv.org/abs/2610.33923)).
- "A TTL-based Approach for Content Placement in Edge Networks", Panigrahy, N. K. et al., [arXiv:1711.03941](https://arxiv.org/abs/1711.03941). Distributed TTL utility optimization over edge caches ([abstract](https://arxiv.org/abs/1711.03941)).

#### B. Not on arXiv (verified via Crossref, publisher, USENIX or author PDF; closest first)

**B1. "Caching or Pre-fetching? The Role of Hazard Rates"** (not on arXiv)
- *Bibliography:* Ferragut, A., Carrasco, M. and Paganini, F. 60th Allerton Conference, 2024, [DOI 10.1109/allerton63246.2024.10735277](https://doi.org/10.1109/allerton63246.2024.10735277); [author PDF](https://aferragu.github.io/assets/pdf/allerton24.pdf).
- *Result.* The paper restates the 2016 TTL result (Theorem 1): for decreasing hazard there is a hazard-rate threshold θ\*, with η(T\*) = θ\* for interior timers. It then introduces Timer-based Pre-fetching for increasing hazard. Using a Lagrangian with multiplier θ ≥ 0, Theorem 2 shows the optimal pre-fetching policy is also a threshold policy: "an item is stored in the local memory if and only if its current hazard rate is greater than the threshold"; items "are pre-fetched after a time T\* since the last request, when their hazard rates reach the threshold" ([author PDF](https://aferragu.github.io/assets/pdf/allerton24.pdf)).
  - Theorem 3: for decreasing hazard, pre-fetching degenerates to the static policy. "Caching makes more sense for DHR and pre-fetching for IHR".
  - Fig. 4 shows both as "threshold policies for the hazard rates".
  - In the large-N limit the threshold is a fixed θ\* (Theorem 4) ([author PDF](https://aferragu.github.io/assets/pdf/allerton24.pdf)).
- *Relation to our design.* Parallels RETAIN, FETCH and PRICE.
  - **Shares:** one multiplier on average memory occupancy gives both the keep rule (keep while hazard ≥ θ) and the load rule (load once hazard reaches θ).
  - **Differs:**
    - equal-size items;
    - hit rate;
    - the pre-fetch is instantaneous at T\*, with no read duration;
    - no newsvendor criterion over the read's own duration and no stall cost;
    - the threshold is solved offline rather than learned.
  - **PRIOR-ART FLAG (strongest):** hazard-threshold rules, with the threshold equal to the memory multiplier, for both keeping and pre-loading.

**B2. "Optimizing TTL Caches under Heavy-Tailed Demands"** (not on arXiv)
- *Bibliography:* Ferragut, A., Rodríguez, I. and Paganini, F. ACM SIGMETRICS 2016, pp. 101–112, [DOI 10.1145/2896377.2901459](https://doi.org/10.1145/2896377.2901459); [author PDF](https://grupo-mate.github.io/assets/pdf/sigmetrics16.pdf).
  - Journal version: "Optimal timer-based caching policies for general arrival processes", *Queueing Systems* 88:207–241 (online 2017; cited by the authors as 2018), [DOI 10.1007/s11134-017-9540-3](https://doi.org/10.1007/s11134-017-9540-3).
- *Result.* The paper maximizes hit rate Σλ_n F_n(T_n) subject to average occupancy Σ F̂_n(T_n) ≤ C over per-content timers ([author PDF](https://grupo-mate.github.io/assets/pdf/sigmetrics16.pdf)).
  - Under increasing hazard, the static most-popular policy is optimal (Theorem 1).
  - Under decreasing hazard, the problem is convex. With multiplier p for the memory constraint, ∂L/∂u_n = η_n(T_n) − p. Hence:
    - η_n(0) ≤ p\* ⇒ never cache;
    - η_n(∞) > p\* ⇒ cache forever;
    - otherwise η_n(T\*_n) = p\* (Theorem 2).
  - Remark 1 interprets η_n(T_n) as "the marginal utility derived from increasing the timer", equalized across contents.
  - The paper gives explicit Pareto/Zipf solutions, a fluid limit, an LRU comparison, and a line-network extension.
- *Relation to our design.* Parallels RETAIN and PRICE.
  - **Shares:** our rule μ(t)·r(1+λL) ≥ λL has the same marginal structure (hazard × value of a hit ≥ memory cost rate). With unit hit value and unit size, their p plays the role of our q/r.
  - **Differs:**
    - hit rate, not TTFT;
    - no size or reload time;
    - stationary renewal per content with timer reset, rather than one conditional gap per session;
    - requires decreasing hazard for convexity;
    - multiplier from KKT and fluid analysis, not online.
  - **PRIOR-ART FLAG:** keep while the hazard rate is at least the price (decreasing hazard).

**B3. "Timer-based pre-fetching for increasing hazard rates"** (not on arXiv)
- *Bibliography:* Ferragut, A., Carrasco, M. and Paganini, F. *ACM SIGMETRICS PER* 52(2):9–11 (2024; MAMA workshop), [DOI 10.1145/3695411.3695416](https://doi.org/10.1145/3695411.3695416); [author PDF](https://aferragu.github.io/assets/pdf/mama24_caching.pdf).
- *Result.* Defines "a dual of the TTL policy, dubbed Timer-based Pre-fetching", suited to regular (increasing-hazard) traffic, and shows how to compute optimal timers for a large class of request processes. It points to Panigrahy et al. for "a related result for replacement-based policies" ([author PDF](https://aferragu.github.io/assets/pdf/mama24_caching.pdf)).
- *Relation to our design.* Parallels FETCH.
  - **Differs:** instantaneous loads, unit sizes, hit rate.

**B4. "Maximizing Cache Performance Under Uncertainty" (EVA)** (not on arXiv)
- *Bibliography:* Beckmann, N. and Sanchez, D. IEEE HPCA 2017, pp. 109–120, [DOI 10.1109/hpca.2017.43](https://doi.org/10.1109/hpca.2017.43); [author PDF](https://people.csail.mit.edu/sanchez/papers/2017.eva.hpca.pdf). Earlier tech report: [MIT-CSAIL-TR-2015-034](https://dspace.mit.edu/handle/1721.1/100465).
- *Result.* EVA views "time spent in the cache as forgone hits, i.e., as the opportunity cost of retaining lines". It "charges candidates at a rate of a single line's average hit rate (i.e., the cache's hit rate divided by its size)" ([author PDF](https://people.csail.mit.edu/sanchez/papers/2017.eva.hpca.pdf)).
  - EVA = Expected hits − (Cache hit rate / Cache size) × Expected time. It is computed from a candidate's age by conditional probability.
  - The authors justify it via Markov decision process (MDP) theory, citing the bias criterion ([abstract](https://doi.org/10.1109/hpca.2017.43); [author PDF](https://people.csail.mit.edu/sanchez/papers/2017.eva.hpca.pdf)).
- *Relation to our design.* Parallels INDEX and PRICE.
  - **Shares:** each candidate is charged a price per unit of space-time, and expected benefit is net of that charge, conditioned on age.
  - **Differs:**
    - the price is the cache's own average hit rate per line (self-consistent), not a dual variable driven by demand versus capacity;
    - hits, not latency;
    - unit lines (processor caches);
    - a fixed lifetime horizon rather than a max over horizons.
  - **PRIOR-ART FLAG:** priced occupancy in an eviction rank.

**B5. "LHD: Improving Cache Hit Rate by Maximizing Hit Density"** (not on arXiv)
- *Bibliography:* Beckmann, N., Chen, H. and Cidon, A. USENIX NSDI 2018, [USENIX page](https://www.usenix.org/conference/nsdi18/presentation/beckmann); [PDF](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-beckmann.pdf).
- *Result.* LHD ranks by "Hit density = Hit probability / (Object size × Expected time in cache)", "in units of hits per byte-access". It uses conditional probability on object age and evicts the least hit density. RankCache (memcached-based) needs "on average 8x less [space] than LRU" to match hit rate ([PDF](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-beckmann.pdf); [abstract](https://www.usenix.org/conference/nsdi18/presentation/beckmann)).
- *Relation to our design.* Parallels INDEX.
  - **Shares:** hits per unit of space-time.
  - **Differs:** LHD evaluates the ratio at the policy's own lifetime (one horizon), whereas we take the max over H. It also counts hits rather than cost-weighted reloads and has no price.

**B6. "IP over connection-oriented networks and distributional paging"** (not on arXiv)
- *Bibliography:* Lund, C., Phillips, S. and Reingold, N. FOCS 1994, pp. 424–434, [DOI 10.1109/sfcs.1994.365674](https://doi.org/10.1109/sfcs.1994.365674). Journal version: "Paging Against a Distribution and IP Networking", *JCSS* 58:222–231 (1999), [DOI 10.1006/jcss.1997.1498](https://doi.org/10.1006/jcss.1997.1498).
- *Result (via A8, which restates it).* The algorithm knows a probabilistic model of future requests; A8 analyzes the Markov-chain case and does not restate a general definition of "distributional paging". In that setting the dominating-distribution algorithm, which uses the pairwise probabilities that one page is requested before another, is 4-competitive against the optimal online algorithm. The median algorithm, which evicts the largest median next-request time, has a guarantee under independent inter-arrival times ([arXiv HTML of A8](https://arxiv.org/html/2502.05511v2); [A8 abstract](https://arxiv.org/abs/2502.05511)).
- *Relation to our design.* Parallels INDEX.
  - The earliest verified use of next-request distributions in eviction, and the median rule is an explicit point-estimate reduction.
  - **Differs:** unit costs and sizes, no memory price.
  - The original text was not read (secondary account only).

**B7. "Raven: Belady-Guided, Predictive (Deep) Learning for In-Memory and Content Caching"** (not on arXiv)
- *Bibliography:* Hu, X., Ramadan, E., Ye, W., Tian, F. and Zhang, Z.-L. ACM CoNEXT 2022, pp. 72–90, [DOI 10.1145/3555050.3569134](https://doi.org/10.1145/3555050.3569134); [NSF PAR PDF](https://par.nsf.gov/servlets/purl/10424505).
- *Result.* A mixture density network learns "distributions of object next-arrival times". Two sampling techniques estimate "the probability that an object's 'residual time' to the next request given the current time is the farthest in the future than all other objects", and Raven "selects the object with the largest estimated probability for eviction" ([PDF](https://par.nsf.gov/servlets/purl/10424505)).
  - Its size-aware variant draws on the Belady extension that evicts "the highest cost = object size × next-use distance".
  - It critiques LHR as estimating hazard rates under a Poisson assumption ([PDF](https://par.nsf.gov/servlets/purl/10424505)).
- *Relation to our design.* Parallels INDEX.
  - **Shares:** explicit conditional next-arrival distributions given the current residual time.
  - **Differs:**
    - uses them to imitate Belady through an order statistic, not a cost-per-space-time index;
    - no price;
    - no reload cost by size beyond the Belady-size heuristic.

**B8. "Restless Bandits, Linear Programming Relaxations, and a Primal-Dual Index Heuristic"** (not on arXiv)
- *Bibliography:* Bertsimas, D. and Niño-Mora, J. *Operations Research* 48(1):80–90 (2000), [DOI 10.1287/opre.48.1.80.12444](https://doi.org/10.1287/opre.48.1.80.12444).
- *Result.* A hierarchy of linear programming (LP) relaxations bounds the PSPACE-hard restless bandit. The paper also proposes "a priority-index heuristic scheduling policy from the solution to the first-order relaxation, where the indices are defined in terms of optimal dual variables" ([abstract via Semantic Scholar record of the DOI](https://doi.org/10.1287/opre.48.1.80.12444)).
- *Relation to our design.* Parallels BOUND and INDEX.
  - **Shares:** relaxation bounds, plus indices derived from duals of the relaxation.
  - **Differs:** generic, with no caching model.

**B9. "Learning from optimal caching for content delivery" (LHR)** (not on arXiv)
- *Bibliography:* Yan, G., Li, J. and Towsley, D. ACM CoNEXT 2021, pp. 344–358, [DOI 10.1145/3485983.3494855](https://doi.org/10.1145/3485983.3494855).
- *Result.* "We first propose HRO to compute the upper bound on optimal caching in an online manner, and then leverage HRO to inform future content admission and eviction" (LHR). It uses a threshold-based admission model with a simple eviction rule ([abstract](https://doi.org/10.1145/3485983.3494855)). Raven describes LHR as estimating hazard rates assuming Poisson requests ([Raven PDF](https://par.nsf.gov/servlets/purl/10424505)).
- *Relation to our design.* Parallels INDEX.
  - **Shares:** a learning system built on the hazard-rate (non-anticipative) optimum rather than on Belady.

**B10. "Caching with Delayed Hits"** (not on arXiv)
- *Bibliography:* Atre, N., Sherry, J., Wang, W. and Berger, D. S. ACM SIGCOMM 2020, pp. 495–513, [DOI 10.1145/3387514.3405883](https://doi.org/10.1145/3387514.3405883).
- *Result.* When requests queue behind an outstanding miss ("delayed hits"), hit-rate-optimal policies including Belady fail to minimize latency.
  - The offline algorithm "belatedly" reduces average latency by up to 45% versus Belady.
  - The online MAD heuristic uses aggregate delay and gains up to 40% ([abstract](https://doi.org/10.1145/3387514.3405883)).
- *Relation to our design.*
  - **Shares:** a latency objective, with fetches in flight while requests arrive (our stall S).
  - **Differs:** no forecasts and no memory price.

**B11. "TTL-based Cloud Caches"** (not on arXiv)
- *Bibliography:* Carra, D., Neglia, G. and Michiardi, P. IEEE INFOCOM 2019, pp. 685–693, [DOI 10.1109/infocom.2019.8737546](https://doi.org/10.1109/infocom.2019.8737546).
- *Result.* A dynamic TTL algorithm minimizes storage-plus-miss cost for elastic cloud caches with constant per-request overhead ([abstract via Semantic Scholar](https://doi.org/10.1109/infocom.2019.8737546)).
- *Relation to our design.* Same as A7 (PRICE, exogenous).

**B12. Five-minute rule** (not on arXiv)
- *Bibliography:*
  - Gray, J. and Putzolu, F., "The 5 minute rule for trading memory for disc accesses and the 10 byte rule for trading memory for CPU time", SIGMOD 1987, pp. 395–398, [DOI 10.1145/38713.38755](https://doi.org/10.1145/38713.38755).
  - Gray, J. and Graefe, G., "The five-minute rule ten years later, and other computer storage rules of thumb", *SIGMOD Record* 26:63–68 (1997), [DOI 10.1145/271074.271094](https://doi.org/10.1145/271074.271094).
  - Graefe, G., "The five-minute rule twenty years later, and how flash memory changes the rules", DaMoN 2007, [DOI 10.1145/1363189.1363198](https://doi.org/10.1145/1363189.1363198).
- *Relation to our design.* Classic price-of-memory versus price-of-access keep rule (PRICE, RETAIN). See A13 for the 2026 GPU/SSD revisit.

**B13. Size- and cost-aware caching classics and systems** (not on arXiv)
- *Bibliography:*
  - Cao, P. and Irani, S., "Cost-Aware WWW Proxy Caching Algorithms", USENIX USITS 1997, [USENIX page](https://www.usenix.org/legacy/publications/library/proceedings/usits97/cao.html).
  - Young, N. E., "The k-server dual and loose competitiveness for paging" (GreedyDual), *Algorithmica* 11:525–541 (1994), [DOI 10.1007/bf01189992](https://doi.org/10.1007/bf01189992).
  - Irani, S., "Page replacement with multi-size pages and applications to Web caching", STOC 1997, [DOI 10.1145/258533.258666](https://doi.org/10.1145/258533.258666); *Algorithmica* 33:384–409 (2002), [DOI 10.1007/s00453-001-0125-4](https://doi.org/10.1007/s00453-001-0125-4).
  - Bansal, N., Buchbinder, N. and Naor, J., "A primal-dual randomized algorithm for weighted paging", FOCS 2007, [DOI 10.1109/focs.2007.43](https://doi.org/10.1109/focs.2007.43); *JACM* 59 (2012), [DOI 10.1145/2339123.2339126](https://doi.org/10.1145/2339123.2339126).
  - Berger, D. S., Sitaraman, R. K. and Harchol-Balter, M., "AdaptSize: Orchestrating the Hot Object Memory Cache in a Content Delivery Network", NSDI 2017, [USENIX page](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/berger).
  - Blankstein, A., Sen, S. and Freedman, M. J., "Hyperbolic Caching: Flexible Caching for Web Applications", USENIX ATC 2017, [USENIX page](https://www.usenix.org/conference/atc17/technical-sessions/presentation/blankstein).
  - Neglia, G. et al., "Access-time aware cache algorithms", ITC 28 (2016), pp. 148–156, [DOI 10.1109/itc-28.2016.127](https://doi.org/10.1109/itc-28.2016.127). The EURECOM listing names a journal version in ACM TOMPECS 2017, which was not checked in Crossref ([EURECOM page](https://www.eurecom.fr/en/publication/5393)).
- *Results:*
  - GreedyDual-Size "incorporates locality with cost and size concerns in a simple and non-parameterized fashion" ([USENIX](https://www.usenix.org/legacy/publications/library/proceedings/usits97/cao.html)).
  - AdaptSize is a size-aware admission policy tuned by a Markov cache model ([USENIX](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/berger)).
  - Hyperbolic caching decays item priorities at variable rates. It supports item cost, expiration and "cost classes", and evicts by random sampling ([USENIX](https://www.usenix.org/conference/atc17/technical-sessions/presentation/blankstein)).
  - Access-time-aware caching exploits the memory/disk access-time gap. Per its project pages (seen through a search summary, not the paper text), the policy is optimal under the independent reference model (IRM) and extends "to minimize any retrieval cost, as long as costs add over cache misses" ([EURECOM](https://www.eurecom.fr/en/publication/5393); [Akamai-hosted PDF](https://www.akamai.com/site/en/documents/research-paper/access-time-aware-cache-algorithms.pdf)).
- *Relation to our design.* Miss cost by size and cost, without forecasts or price.

**B14. Learned Belady and learning-augmented paging (venue-only verification)**
- *Bibliography:*
  - Song, Z., Berger, D. S., Li, K. and Lloyd, W., "Learning Relaxed Belady for Content Distribution Network Caching", NSDI 2020, [USENIX page](https://www.usenix.org/conference/nsdi20/presentation/song).
  - Jain, A. and Lin, C., "Back to the Future: Leveraging Belady's Algorithm for Improved Cache Replacement", ISCA 2016, pp. 78–89, [DOI 10.1109/isca.2016.17](https://doi.org/10.1109/isca.2016.17).
  - Shah, I., Jain, A. and Lin, C., "Effective Mimicry of Belady's MIN Policy", HPCA 2022, pp. 558–572, [DOI 10.1109/hpca53966.2022.00048](https://doi.org/10.1109/hpca53966.2022.00048).
  - Rohatgi, D., "Near-Optimal Bounds for Online Caching with Machine Learned Advice", SODA 2020, pp. 1834–1845, [DOI 10.1137/1.9781611975994.112](https://doi.org/10.1137/1.9781611975994.112). Its arXiv version was not checked.
- *Results:*
  - LRB "uses machine learning to approximate the Belady MIN algorithm" via "the Relaxed Belady algorithm, the Belady boundary, and the good decision ratio". It reduces WAN traffic by 5–24% versus a production design ([USENIX](https://www.usenix.org/conference/nsdi20/presentation/song)).
  - Hawkeye learns whether lines are "Cache Friendly" or "Cache Averse" under MIN ([Mockingjay abstract](https://doi.org/10.1109/hpca53966.2022.00048)).
  - Mockingjay predicts each line's reuse distance and "evicts lines based on their predicted time of reuse" ([abstract](https://doi.org/10.1109/hpca53966.2022.00048)).
- *Relation to our design.* INDEX inputs from point predictions or labels.

**B15. Index-policy foundations** (not on arXiv)
- *Bibliography:*
  - Gittins, J. C., "Bandit Processes and Dynamic Allocation Indices", *JRSS-B* 41:148–164 (1979), [DOI 10.1111/j.2517-6161.1979.tb01068.x](https://doi.org/10.1111/j.2517-6161.1979.tb01068.x).
  - Whittle, P., "Restless bandits: activity allocation in a changing world", *J. Appl. Probab.* 25:287–298 (1988), [DOI 10.2307/3214163](https://doi.org/10.2307/3214163).
  - Weber, R. and Weiss, G., "On an index policy for restless bandits", *J. Appl. Probab.* 27:637–648 (1990), [DOI 10.2307/3214547](https://doi.org/10.2307/3214547).
  - Gittins, J., Glazebrook, K. and Weber, R., *Multi-Armed Bandit Allocation Indices* (Wiley 2011), [DOI 10.1002/9780470980033](https://doi.org/10.1002/9780470980033). Its ch. 6 is titled "Restless Bandits and Lagrangian Relaxation", [DOI 10.1002/9780470980033.ch6](https://doi.org/10.1002/9780470980033.ch6).
  - Sevcik, K. C., "Scheduling for Minimum Total Loss Using Service Time Distributions", *JACM* 21:66–75 (1974), [DOI 10.1145/321796.321803](https://doi.org/10.1145/321796.321803). Per A4, Sevcik proved index results in special cases and conjectured optimality more generally ([A4 PDF](https://arxiv.org/pdf/2111.10703)).
  - Aalto, S., Ayesta, U. and Righter, R., "On the Gittins index in the M/G/1 queue", *Queueing Systems* 63:437–458 (2009), [DOI 10.1007/s11134-009-9141-x](https://doi.org/10.1007/s11134-009-9141-x).
  - Aalto, S., Ayesta, U. and Righter, R., "Properties of the Gittins index with application to optimal scheduling", *PEIS* 25:269–288 (2011), [DOI 10.1017/s0269964811000015](https://doi.org/10.1017/s0269964811000015).
  - Aalto, S. and Ayesta, U., "Optimal scheduling of jobs with a DHR tail in the M/G/1 queue", ValueTools 2008, [DOI 10.4108/icst.valuetools2008.4335](https://doi.org/10.4108/icst.valuetools2008.4335).
- *Relation to our design.* INDEX, PRICE and BOUND foundations. Content beyond titles was not re-read here.

**B16. Pricing foundations already cited in the design** (not on arXiv)
- Everett, H., "Generalized Lagrange Multiplier Method for Solving Problems of Optimum Allocation of Resources", *Oper. Res.* 11:399–417 (1963), [DOI 10.1287/opre.11.3.399](https://doi.org/10.1287/opre.11.3.399).
- Kelly, F., Maulloo, A. and Tan, D., "Rate control for communication networks: shadow prices, proportional fairness and stability", *JORS* 49:237–252 (1998), [DOI 10.1057/palgrave.jors.2600523](https://doi.org/10.1057/palgrave.jors.2600523). Dehghan et al. cite it (A5).

**B17. Classic replacement under stochastic models** (not on arXiv)
- *Bibliography:*
  - Belady, L. A., "A study of replacement algorithms for a virtual-storage computer", *IBM Syst. J.* 5:78–101 (1966), [DOI 10.1147/sj.52.0078](https://doi.org/10.1147/sj.52.0078).
  - Aho, A., Denning, P. and Ullman, J., "Principles of Optimal Page Replacement", *JACM* 18:80–93 (1971), [DOI 10.1145/321623.321632](https://doi.org/10.1145/321623.321632).
  - O'Neil, E., O'Neil, P. and Weikum, G., "The LRU-K page replacement algorithm for database disk buffering", SIGMOD 1993, [DOI 10.1145/170035.170081](https://doi.org/10.1145/170035.170081). Follow-up "An optimality proof of the LRU-K page replacement algorithm", *JACM* 46:92–112 (1999), [DOI 10.1145/300515.300518](https://doi.org/10.1145/300515.300518).
  - Takagi, M. and Hiraki, K., "Inter-reference gap distribution replacement", ICS 2004, [DOI 10.1145/1006209.1006213](https://doi.org/10.1145/1006209.1006213).
  - Karlin, A., Phillips, S. and Raghavan, P., "Markov Paging", FOCS 1992, [DOI 10.1109/sfcs.1992.267771](https://doi.org/10.1109/sfcs.1992.267771); *SIAM J. Comput.* 30:906–922 (2000), [DOI 10.1137/s0097539794268042](https://doi.org/10.1137/s0097539794268042).
  - Avrachenkov, K. and Borkar, V., "Whittle Index Policy for Crawling Ephemeral Content", CDC 2015, [DOI 10.1109/cdc.2015.7403283](https://doi.org/10.1109/cdc.2015.7403283); *IEEE TCNS* 5:446–455 (2018), [DOI 10.1109/tcns.2016.2619066](https://doi.org/10.1109/tcns.2016.2619066).
  - Niño-Mora, J., "A Dynamic Page-Refresh Index Policy for Web Crawlers", ASMTA 2014, [DOI 10.1007/978-3-319-08219-6_4](https://doi.org/10.1007/978-3-319-08219-6_4).
- *Results:*
  - Per EVA's background section, Belady's MIN evicts "the candidate that is referenced furthest in the future".
  - Aho et al. model replacement under the IRM as an MDP and "show that the optimal policy is to evict the page with the lowest reference probability, i.e., LFU".
  - Inter-reference gap distribution replacement (IGDR) "computes a statistical cost function from sampled reuse distance histograms" ([EVA PDF](https://people.csail.mit.edu/sanchez/papers/2017.eva.hpca.pdf)).
- *Relation to our design.* Background for the INDEX special cases and for freshness indices.

### Inferences
- **What may and may not be claimed as new.**
  - Do not claim as new: "keep while hazard ≥ price" (B2, B1, A1); a priced-occupancy relaxation that decouples items (A2, B2, A5); hazard-rate eviction ranking (A3, A1); the Gittins ratio form (A4, and A2 in caching).
  - Plausibly new, as no source found does it:
    - TTFT-denominated per-session costs with a reload cost proportional to size, plus a charge for memory held during the reload (the (1+λL) factor);
    - conditional forecasts from tool-duration tables, with Gittins horizons justified by costly refetch;
    - one price driving fetch, preempt and retain together;
    - a multiplicative online price whose demand counts refused and preempted reads.
- **Mapping to the closest prior result.** Ferragut et al.'s condition η(T\*) = p\* is our μ(t)·r(1+λL) = λL with the hit value normalized to 1 and size dropped. Their multiplier p corresponds to our q/r.
- **Why the Gittins max is needed.** In A1 and A3, an item evicted during a lull can return at no cost, so the current hazard alone is optimal. When re-reading costs r seconds and memory, keeping through a lull can be worth more, which is the case for max_H (A2's max over depth plays the same role).

### Gaps
- The Queueing Systems journal version of B2 and the TOMPECS version of A3 were not read; they may contain extensions (e.g., sizes or costs) that sharpen the comparison.
- B6 (Lund et al.) was read only through A8's restatement.
- **Unverified leads** (not verified; do not cite without checking):
  - Fofack, Nain, Neglia and Towsley on TTL cache networks (2012/2014).
  - Jelenković on LRU under heavy-tailed or renewal requests.
  - Gopalan et al., "Caching with expiration times" (SODA 2002).
  - Kolobov et al. on freshness crawling with Lagrangian rates (2019).
  - Tripathi and Modiano on Whittle indices for age of information.
  - Wei (APPROX 2020), Im et al. (ICML 2022) and Chłędowski et al. (ICML 2021) on learning-augmented caching.
  - Glider (MICRO 2019), HALP (NSDI 2023), LeCaR, CACHEUS, 3L-Cache.
  - Ma and Towsley on TTL contracts with linear pricing (INFOCOM 2015).
  - Scully et al. on Gittins in the M/G/k.
  - A search snippet attributing to Versaci's thesis that LRU is optimal for non-increasing stack-distance distributions.
  - The ICALP 2025 venue for A8.
- **Records verified, content not retrieved:**
  - Neglia, Carra and Michiardi, "Cache Policies for Linear Utility Maximization" (INFOCOM 2017, [DOI 10.1109/infocom.2017.8057192](https://doi.org/10.1109/infocom.2017.8057192); ToN 26:302–313 (2018), [DOI 10.1109/tnet.2017.2783623](https://doi.org/10.1109/tnet.2017.2783623)).
  - Berger et al., "Maximizing Cache Hit Ratios by Variance Reduction" (PER 43:57–59, 2015, [DOI 10.1145/2825236.2825259](https://doi.org/10.1145/2825236.2825259)).
  - Martina, Garetto and Leonardi, "A unified approach to the performance analysis of caching systems" (INFOCOM 2014, [DOI 10.1109/infocom.2014.6848145](https://doi.org/10.1109/infocom.2014.6848145); TOMPECS 2016, [DOI 10.1145/2896380](https://doi.org/10.1145/2896380)).
  - Berger, Gland, Singla and Ciucu, "Exact analysis of TTL cache networks" (Perf. Eval. 79:2–23, 2014, [DOI 10.1016/j.peva.2014.07.001](https://doi.org/10.1016/j.peva.2014.07.001)).
  - Che, Tung and Wang, "Hierarchical Web caching systems: modeling, design and experimental results" (JSAC 2002, [DOI 10.1109/jsac.2002.801752](https://doi.org/10.1109/jsac.2002.801752)).
  - Jung, Berger and Balakrishnan, "Modeling TTL-based Internet caches" (INFOCOM 2003, [DOI 10.1109/infcom.2003.1208693](https://doi.org/10.1109/infcom.2003.1208693)).
  - Koutsoupias and Papadimitriou, "Beyond Competitive Analysis" (SICOMP 30:300–317, 2000, [DOI 10.1137/s0097539796299540](https://doi.org/10.1137/s0097539796299540)).
  - Franaszek and Wagner, "Some Distribution-Free Aspects of Paging Algorithm Performance" (JACM 21:31–39, 1974, [DOI 10.1145/321796.321800](https://doi.org/10.1145/321796.321800)).

## Q1. Utility-maximizing TTL caches: which formulations price occupancy through Lagrangian duality and adapt the price online, and do any derive "keep while hazard ≥ price"?

### Takeaway
Yes. Four formulations price occupancy through a Lagrange multiplier:

- **Dehghan et al.:** Poisson arrivals, with an online dual price on capacity.
- **Ferragut–Rodríguez–Paganini:** renewal arrivals with a memory multiplier. Under decreasing hazard the optimal timer stops where the hazard equals the multiplier, i.e., keep while hazard ≥ price.
- **Panigrahy–Li–Towsley–Hollot:** stationary requests; convex under non-increasing hazard; dual implementation of LRU.
- **Ferragut–Carrasco–Paganini:** the same threshold for pre-fetching under increasing hazard.

Basu et al. adapt a TTL online toward targets without a price. Carra et al. price storage exogenously.

The hazard keep rule is therefore prior art in the unit-size, hit-rate, no-fetch-latency form. Our rule is its cost- and size-weighted version, with an in-flight memory term and an online multiplicative price.

### Cited Findings
- **Dehghan et al. (A5), Poisson TTL caches.**
  - The capacity multiplier α ties to LRU and FIFO characteristic times, with timers t_i = 1/α for LRU emulation ([arXiv PDF](https://arxiv.org/pdf/1601.06838)).
  - Online dual update: α ← max{0, α + γ(B_curr − B)} ([arXiv PDF](https://arxiv.org/pdf/1601.06838)).
  - Prices "set by (cache) service providers" ([arXiv PDF](https://arxiv.org/pdf/1601.06838)).
- **Ferragut, Rodríguez and Paganini (B2).**
  - Lagrangian L(u, p) = Σ λ_n F_n(F̂_n⁻¹(u_n)) − p(Σ u_n − C).
  - Under decreasing hazard, η_n(T\*_n) = p\* for interior timers, with corner cases η_n(0) ≤ p\* ⇒ T\* = 0 and η_n(∞) > p\* ⇒ T\* = ∞.
  - Under increasing hazard, the static policy is optimal ([author PDF](https://grupo-mate.github.io/assets/pdf/sigmetrics16.pdf)).
- **Ferragut, Carrasco and Paganini (B1).** Under increasing hazard, with multiplier θ: "an item is stored ... if and only if its current hazard rate is greater than the threshold". Items are pre-fetched when their hazard reaches θ ([author PDF](https://aferragu.github.io/assets/pdf/allerton24.pdf)).
- **Carrasco et al. (A1).** Fixed-threshold policies C_θ(t) = {i : λ_i(t) > θ} satisfy only a soft average-occupancy constraint, and the optimal causal threshold converges to a constant ([preprint PDF](https://aferragu.github.io/assets/pdf/stochsys24_arxiv.pdf)).
- **Panigrahy, Li, Towsley and Hollot (A6).** Convexity under a non-increasing hazard rate, a "reverse engineering based dual implementation of LRU under stationary arrivals", and decentralized algorithms ([abstract](https://arxiv.org/abs/1712.07307)).
- **Basu et al. (A11).** d-TTL adapts the TTL by stochastic approximation to a target hit rate; f-TTL meets hit-rate and cache-size targets ([abstract](https://arxiv.org/abs/1704.04448)).
- **Carra et al. (A7, B11).** Storage cost per instance per epoch plus per-object miss cost m_o, with the TTL adapted by stochastic approximation ([arXiv PDF](https://arxiv.org/pdf/1802.04696)).
- **Elsayed et al. (A10).** Per-object TTL utility with random fetch delays ([abstract](https://arxiv.org/abs/2405.04402)).

### Inferences
- **How close is the closest rule?** Ferragut et al.'s optimality condition is our retention rule in units of hits.
  - Our rule: μ(t)·[r(1+λL)] ≥ λL. The left side is the hazard times the TTFT saved per return (stall r, plus memory λLr the reload would hold). The right side is the memory charge rate.
  - Theirs: μ(t)·1 ≥ p.
  - Both are first-order (marginal) conditions on the timer. They coincide in form whenever the hazard is non-increasing.
- **Where we go beyond it:**
  - multi-modal hazards, handled by the Gittins max (their convex program needs decreasing hazard);
  - a price learned online;
  - a TTFT objective that couples to the fetch rule through the same q.
- **Closest precedent for online dual ascent.** Dehghan et al.'s additive update is the closest. Our multiplicative step and the use of refused or preempted demand, needed because occupancy can never exceed R, appear not to be in the TTL literature found.

### Gaps
- No TTL paper found uses a multiplicative price update or counts refused demand.
- The ToN 2019 version of Dehghan et al. and the Queueing Systems version of Ferragut et al. were not read for extensions such as renewal or size.
- The content of Neglia et al. on linear utility was not retrieved.

## Q2. Hazard-rate eviction and upper bounds for non-anticipative policies: what do they say, and how do they relate to Belady's MIN and LRU?

### Takeaway
For independent stationary request processes, keeping the C items with the largest current conditional intensity is the optimal causal policy (A1). The same ordering gives an upper bound on the hit probability of every non-anticipative policy (A3). For variable sizes, the ordering is HR/size for object hits and HR alone for byte hits (A3). Empirically, A3 finds the bound tighter than Belady's offline bound, and close to it under increasing hazard. Poisson arrivals give the static (LFU-like) policy (A1, B2). Our index coincides with cost-weighted hazard ranking under decreasing hazard and departs from it only when refetch is costly.

### Cited Findings
- **A3 (HR bound).**
  - HR-E caches the B objects with the largest hazard rate (conditional intensity given history), and h_{HR-E} ≥ max over non-anticipative policies (Theorem 1). With sizes, the byte-hit fractional knapsack orders by λ_i(t); object hits (relaxed 0-1 knapsack) order by λ_i(t)/s_i ([arXiv PDF](https://arxiv.org/pdf/2106.06457)).
  - The authors provide "upper bounds", not new policies ([arXiv PDF](https://arxiv.org/pdf/2106.06457)).
  - The HR bound is tighter than Belady's bound, and the two are close under increasing hazard ([arXiv PDF](https://arxiv.org/pdf/2106.06457)).
- **A1 (optimal causal policy).**
  - Theorem 1: the policy maximizes stationary hit rate and hit probability.
  - Corollary 1 (renewal): rank by current interval hazard rate and store the top C.
  - Poisson case: the static policy that stores the highest-intensity items, i.e., the IRM ([preprint PDF](https://aferragu.github.io/assets/pdf/stochsys24_arxiv.pdf)).
- **B2 (TTL view).** Under increasing hazard the static policy is optimal; under decreasing hazard, timer policies outperform it ([author PDF](https://grupo-mate.github.io/assets/pdf/sigmetrics16.pdf)).
- **Dehghan et al. (A5).** LRU and FIFO correspond to TTL caches with a common characteristic time T ([arXiv PDF](https://arxiv.org/pdf/1601.06838)).
- **Learning systems built on the HR rule:**
  - LHR uses HRO, an online upper bound on optimal caching, to drive admission and eviction (B9; [abstract](https://doi.org/10.1145/3485983.3494855)).
  - HR-Cache learns from HR-ordering decisions (A12; [abstract](https://arxiv.org/abs/2402.02795)).
- **Aho, Denning and Ullman (B17).** Under the IRM, evicting the lowest reference probability (LFU) is optimal ([EVA PDF summary](https://people.csail.mit.edu/sanchez/papers/2017.eva.hpca.pdf)).

### Inferences
- **LRU as a special case.** With one shared distribution that has a decreasing hazard, hazard ranking equals ranking by idle age, i.e., LRU. This follows directly from Corollary 1 of A1 and matches the design's "pooled distribution → LRU" claim.
- **Belady as a special case.** With exact forecasts, both A3's HR (a point mass) and our index degenerate to farthest-next-use, Belady.
- **Two different "bounds".** A3's bound is an upper bound on hits obtained by a pathwise argument that allows free insertion. Ours is a Lagrangian lower bound on cost. They are complementary references, and both limit non-anticipative policies given the same distributional information.

### Gaps
- The TOMPECS 2022 version of A3 was not read; it may add policies, not only bounds.
- No paper found studies hazard-rate eviction with a fetch latency or a size-proportional reload cost in a latency objective.

## Q3. Index policies for caching, paging and freshness: Gittins, Whittle, LHD, EVA, GreedyDual-Size. Is max_H F(H|t)/τ(H) known?

### Takeaway
The form is known.

- **Scheduling:** it is exactly the M/G/1 Gittins index for unknown job sizes, with state = attained service (A4): sup over y of P[S ≤ y | S > x] / E[min{S, y} − x | S > x].
- **Caching:** the closest precedent is Bilardi–Versaci's Least Profit Rate, ξ(i) = max_j s̄(i, j), "the best achievable ratio between expected hits and the expected occupancy" (A2).
- **LHD** evaluates hits per space-time at one horizon (B5).
- **EVA** subtracts a price per unit of space-time, the cache's average hit rate per line (B4).
- **Whittle-index caching papers** (A9) use the Lagrangian subsidy on popularity or freshness states.
- **Bertsimas–Niño-Mora** (B8) derive indices from duals of LP relaxations.
- **GreedyDual-Size and Landlord** handle cost and size in worst-case models (B13, A15).

We found no caching paper that applies the continuous-time Gittins completion-rate index to per-item next-request distributions with a size- and price-weighted reload cost.

### Cited Findings
- **Gittins index (A4).** Defined as the maximal holding-cost decrease rate over stopping sets. The unknown-size formula is "well-known in the literature"; rank = 1/index; known sizes give SRPT ([arXiv PDF](https://arxiv.org/pdf/2111.10703)).
- **LPR (A2).** LPR evicts argmin_i max_j s̄(i, j), and it is optimal under the LRU stack model at fixed capacity. The priorities come from the convex hull of the Pareto points of the average-occupancy problem ([arXiv PDF](https://arxiv.org/pdf/1109.6643)).
- **LHD (B5).** Hit density = Hit probability / (Object size × Expected time in cache) ([PDF](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-beckmann.pdf)).
- **EVA (B4).** EVA = Expected hits − (Cache hit rate / Cache size) × Expected time, justified by MDP theory ([author PDF](https://people.csail.mit.edu/sanchez/papers/2017.eva.hpca.pdf)).
- **Whittle-index caching (A9).**
  - Pavamana & Singh: indexability conditions plus a Whittle policy ([abstract](https://arxiv.org/abs/2304.12227)).
  - Koley & Singh: Whittle indices characterized explicitly for fetch, serve-stale or wait with waiting costs ([abstract](https://arxiv.org/abs/2410.18627)).
  - Xiong et al.: threshold-type optimal policies, indexability, and learned Whittle indices ([abstract](https://arxiv.org/abs/2202.13187)).
- **Freshness / crawling indices.** Avrachenkov & Borkar's Whittle crawling policy and Niño-Mora's page-refresh index policy (B17; records only).
- **Bertsimas–Niño-Mora (B8).** Indices "defined in terms of optimal dual variables" ([DOI record](https://doi.org/10.1287/opre.48.1.80.12444)).
- **Cost and size.** GreedyDual-Size combines locality with cost and size ([USENIX](https://www.usenix.org/legacy/publications/library/proceedings/usits97/cao.html)). Landlord generalizes LRU to arbitrary sizes and retrieval costs ([abstract](https://arxiv.org/abs/cs/0205033)).
- **Computing indices from samples.** Gittins indices computed from n samples are asymptotically optimal (A18; [abstract](https://arxiv.org/abs/2606.24703)).

### Inferences
- **Translating our index into Gittins terms:** ν/((b/β)(1+λL)) = index_Gittins(a), with the job's "service" taken to be the idle gap and "completion" taken to be the request arrival.
  - Under decreasing hazard, the ratio is a hazard-weighted average over [a, a+H] that falls with H. The sup is therefore attained as H → 0, and ν equals the hazard (consistent with the design).
  - Under increasing hazard, the sup is attained as H → ∞, and ν becomes the inverse of the mean residual idle time. Ranking is then by expected time to return, a "Belady in expectation" (derivation, not sourced).
- **LHD versus our index.** LHD equals our ratio at a single horizon (the policy's own eviction age), with size in the denominator. Ours takes the Gittins sup and weights hits by the per-token reload cost.
- **EVA versus our price.** EVA's price is the cache's average hit rate per line. Ours is the dual variable that equates demand and capacity.
- **Where to cite which.** Cite A4/A2/B15 for the index form and B4/B5 for the systems analogs. Whittle-caching papers (A9) support the relaxation-plus-index approach but use different state models.

### Gaps
- Searches for "Gittins index caching" on arXiv returned only an LLM-queueing paper (A19).
- The Aalto–Ayesta–Righter results on the Gittins index under monotone hazards were verified by title only.

## Q4. Learning-augmented caching: probabilistic or distributional predictions of the next request versus point predictions; learned approximations of Belady

### Takeaway
Two lines use distributional next-request information:

- **Distributional paging theory** (Lund–Phillips–Reingold 1994; Pabbaraju–Vakilian 2025) evicts using pairwise probabilities of who is requested first. Its median rule is an explicit point-estimate reduction.
- **Raven (2022)** learns per-object next-arrival distributions with a mixture density network and evicts the object most likely to be requested last.

Most learning-augmented theory instead uses point predictions of each page's next arrival, or single bits (A16). Most systems imitate Belady through binary or multiclass labels or regression (LRB, Parrot, Hawkeye, Mockingjay). LHR and HR-Cache imitate the hazard-rate bound. None prices memory-time or converts the distribution into a cost-per-space-time index.

### Cited Findings
- **Distributional paging (A8, B6).** The dominating-distribution algorithm evicts from μ with E_{p∼μ}[Pr(p before q)] ≤ 1/2. It is 4-competitive against the optimal online algorithm (Lund et al.), improved to 2 (Pabbaraju & Vakilian), with a 1.5907 lower bound. The median algorithm evicts the largest median next-request time and is 4-competitive under independent inter-arrival times ([abstract](https://arxiv.org/abs/2502.05511); [HTML](https://arxiv.org/html/2502.05511v2)).
- **Raven (B7).** Learns "distributions of object next-arrival times" with a mixture density network and evicts by the estimated probability that the residual time is the farthest ([PDF](https://par.nsf.gov/servlets/purl/10424505)).
- **Point predictions (A16).**
  - Lykouris & Vassilvitskii modify Marker with a black-box oracle ([abstract](https://arxiv.org/abs/1802.05399)).
  - Bansal et al. use "predictions ... about the next arrival of each page" for weighted paging ([abstract](https://arxiv.org/abs/2011.09076)).
  - Jiang et al. show that next-request knowledge alone is insufficient for weighted paging ([abstract](https://arxiv.org/abs/2006.09509)).
  - Antoniadis et al. catalogue prediction types and study 1-bit predictions ([abstract](https://arxiv.org/abs/2210.02775)).
- **Learned Belady.**
  - LRB introduces Relaxed Belady and the Belady boundary ([USENIX](https://www.usenix.org/conference/nsdi20/presentation/song)).
  - Parrot imitates Belady ([abstract](https://arxiv.org/abs/2006.16239)).
  - Hawkeye learns friendly or averse labels; Mockingjay predicts reuse distance and evicts by predicted reuse time ([abstract](https://doi.org/10.1109/hpca53966.2022.00048)).
  - Wang & Liu show that Belady imitation is not byte-miss-optimal ([abstract](https://arxiv.org/abs/2212.13671)).
- **Hazard-rate learners.** LHR (B9; [abstract](https://doi.org/10.1145/3485983.3494855)) and HR-Cache (A12; [abstract](https://arxiv.org/abs/2402.02795)).

### Inferences
- **How predictions enter each family:**
  - Belady-style learners use a predicted next-use time or label to approximate farthest-in-future eviction.
  - Raven and the distributional-paging algorithms use the distribution to estimate who is requested last.
  - Ours uses it to compute expected cost avoided per token-second.
- **Contrast with Raven.** Raven's order-statistic rule ignores how long an item will occupy memory before its hit. Ours charges memory-time, which matters when sizes and the price vary.
- **Supporting precedent for the design's argument.** The median algorithm (A8, B6) is a published example of collapsing a distribution to a point estimate. It supports the argument that a fixed quantile ignores size and price.

### Gaps
- Wei (APPROX 2020), Im et al. (ICML 2022) and Chłędowski et al. (ICML 2021) were not verified; arXiv search was throttled.
- No arXiv paper explicitly on "paging with distributional predictions" (beyond Markov or distributional paging) was found. arXiv searches for "paging distributional predictions" and "caching probabilistic predictions" returned only irrelevant hits.

## Q5. Variable-size objects whose miss cost depends on size

### Takeaway
- **Offline:** optimal caching with variable sizes is NP-hard, and FOO/PFOO give flow-based bounds (A14).
- **Online worst case:** Landlord is optimal among deterministic algorithms for arbitrary sizes and retrieval costs (A15); GreedyDual-Size is the practical heuristic (B13).
- **Stochastic bounds:** order by HR/size for object hits, and by HR alone when value is proportional to size (byte hits) (A3).
- **Latency-aware work:** delayed hits (B10), access-time-aware caching (B13) and TTL under random fetch delays (A10) model retrieval time. None combines it with forecasts or a memory price.

Our reload cost r ∝ L cancels per token, as in byte hits. The in-flight memory term λLr does not cancel, so the index favors longer prefixes by (1+λL) at equal hazard. Recomputation cost ρU(2L−U) is superlinear in position. We found no prior model with either feature.

### Cited Findings
- **FOO (A14).** OPT with variable sizes is NP-hard; min-cost-flow bounds are tight asymptotically under independence ([abstract](https://arxiv.org/abs/1711.03709)).
- **Landlord (A15).** Files with non-negative sizes and retrieval costs; optimal deterministic guarantee ([abstract](https://arxiv.org/abs/cs/0205033)).
- **GreedyDual-Size (B13).** Cost- and size-aware ([USENIX](https://www.usenix.org/legacy/publications/library/proceedings/usits97/cao.html)).
- **Primal-dual weighted paging (B13).** Bansal, Buchbinder and Naor, [DOI 10.1145/2339123.2339126](https://doi.org/10.1145/2339123.2339126) (record only).
- **HR with sizes (A3).** Byte hits: ratio s_i·λ_i(t)/s_i = λ_i(t). Object hits: λ_i(t)/s_i ([arXiv PDF](https://arxiv.org/pdf/2106.06457)).
- **LHD (B5).** Divides by object size ([PDF](https://www.usenix.org/system/files/conference/nsdi18/nsdi18-beckmann.pdf)).
- **Raven (B7).** Its size-aware variant draws on evicting the largest size × next-use distance ([PDF](https://par.nsf.gov/servlets/purl/10424505)).
- **Wang & Liu (A17).** Object-miss-optimal and byte-miss-optimal policies differ ([abstract](https://arxiv.org/abs/2212.13671)).
- **Size-aware admission and cost classes.** AdaptSize admission ([USENIX](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/berger)); Hyperbolic caching cost classes ([USENIX](https://www.usenix.org/conference/atc17/technical-sessions/presentation/blankstein)).
- **Latency-aware caching.**
  - Delayed hits: latency-optimal offline "belatedly" versus Belady ([abstract](https://doi.org/10.1145/3387514.3405883)).
  - Access-time-aware caching, extensible to additive retrieval costs ([EURECOM](https://www.eurecom.fr/en/publication/5393)).
  - TTL under random fetch delays ([abstract](https://arxiv.org/abs/2405.04402)).

### Inferences
- **The byte-hit analogy.** In A3's byte-hit case, size cancels and ranking is by hazard alone. Our per-token reload cost (b/β) is the same structure. The additional (1+λL) factor arises only because a reload holds L tokens of memory for r seconds, a coupling of reload time and memory price that no surveyed model has.
- **Recomputation cost.** It grows superlinearly with the lost span (ρU(2L−U)), unlike every linear cost model found. Treating recomputation as a failure to avoid, as the design does, sidesteps the need for a superlinear-cost caching theory.

### Gaps
- Irani's multi-size paging results (B13) were not read in detail.
- No paper with a superlinear (attention-like) miss cost in object size was found.
