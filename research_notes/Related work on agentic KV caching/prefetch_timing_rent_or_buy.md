# Prefetch timing and keep-or-release under a forecast distribution: verified related work

How entries were verified (2026-11-17):
- **arXiv entries.** Each was checked on its arxiv.org/abs page. The first-submission month comes from the arXiv ID; the venue comes from the arXiv journal-ref or comments field unless stated otherwise.
- **Classic entries ("not on arXiv").** Each was checked against its Crossref record, which is metadata deposited by the publisher. Abstracts came from OpenAlex, Semantic Scholar, publisher pages, or the authors' PDFs.
- **Rate limits.** The arXiv API returned HTTP 429 early in the session. arxiv.org/abs pages, a few arxiv.org/search queries, OpenAlex, Crossref and web search covered the rest (see Gaps).
- **Scope.** TTL caching, hazard-rate eviction, LLM serving systems and concrete serverless keep-alive systems are covered by other researchers. They appear here only in passing.
- **Mapping notes.** Notes that map a paper onto our design are my own analysis and are labelled "(inference)".

## Q0. Which verified papers match or anticipate our fetch rule and keep-or-release structure? (Ranked, closest first)

### Takeaway
- **Retention rule.** The closest prior art is Narayana & Kash (UAI 2023). Their Theorem 2 places the optimal keep-alive windows where c_p − c_cs·λ(x) changes sign, with λ(x) the hazard rate. Our rule μ(t)·r(1+λL) ≥ λL is the same condition with c_p = λL and c_cs = r(1+λL).
- **Pre-warm.** In the same paper, the pre-warm start (for an increasing hazard) uses the same threshold. This anticipates the threshold that our fetch and retention rules share, but only in the zero-read-time limit.
- **Fetch rule.** The rule is the classic earliness–tardiness newsvendor:
  - **OR precedents.** It appears in appointment start times (Weiss 1990) and in planned lead times (Ould-Louly & Dolgui 2002, a "generalized newsboy" model).
  - **Systems precedent.** It appears as a fixed-quantile chance constraint in RobustScaler (ICDE 2022). That paper creates each instance at the α-quantile of (arrival − startup time).
  - **Gap.** No verified paper prices early completion by a size-dependent, learned shadow price, conditions on idle time, or orders preemption by the probability of timely use.

### Cited Findings

#### Ranked list, part A: on arXiv (closest first)

**A1. Narayana & Kash.** *Keep-Alive Caching for the Hawkes process.* Sushirdeep Narayana et al. arXiv 2309.03521 (Sep 2023), cs.GT. UAI 2023, PMLR 216:1499–1509. https://arxiv.org/abs/2309.03521. **FLAG: prior art for a hazard-rate keep rule, and for a hazard-rate pre-warm rule when loading is instantaneous.**
- **Model.**
  - The cache has no fixed size. An item costs c_p per unit time while it is cached, and an arrival outside a cached window costs c_cs, a cold start.
  - A policy is a set of keep-alive windows within each inter-arrival time, optionally preceded by a pre-warming delay τ_pw.
  - The paper notes that the cost formula assumes moving items in and out of the cache is free.
  - — [arXiv PDF](https://arxiv.org/pdf/2309.03521)
- **Results.**
  - Lemma 1 gives expected cost = c_cs + ∫π(x)·g(x)dx, with g(x) = c_p(1−F(x|H)) − c_cs·f(x|H).
  - Theorem 2: the window endpoints are 0, ∞, or points where c_p − c_cs·λ(x|H) changes sign, with λ = f/(1−F) the hazard rate.
  - Corollary 2.1, weakly decreasing hazard: a single window starting at 0, ending where λ(τ) = c_p/c_cs. Poisson and Hawkes arrivals fall in this case.
  - Corollary 2.2, weakly increasing hazard: a pre-warm delay τ_pw where λ(τ_pw) = c_p/c_cs, then keep the item forever.
  - Single-peaked hazard: both a pre-warm and a keep-alive window. The authors say this is the policy form used by Shahrad et al. (2020).
  - — [arXiv PDF](https://arxiv.org/pdf/2309.03521); [PMLR](https://proceedings.mlr.press/v216/narayana23a.html)
- **Relation to our design.**
  - **Parallels:** retention, and fetch timing in the zero-read-time limit.
  - **Shares (inference):**
    - Our retention rule (design eq. "retain") is exactly their Theorem 2 with c_p = λL_i and c_cs = r_i(1+λL_i).
    - Their single-window corollary for a decreasing hazard matches our observation that "for a non-increasing return rate, the maximum is attained as H → 0".
    - Their multi-window result for non-monotone hazards is the issue that our max-over-H retention index and our grid search over start times handle.
    - Like us, they condition on the time since the last event and replan after each arrival.
  - **Differs:**
    - No capacity constraint: the cache is infinite, so there is no shared price λ, dual ascent or preemption.
    - Loading is instantaneous, so there is no read time. Their pre-warm therefore compares the hazard rate with c_p/c_cs, while ours compares F(r−Ŵ|t), the probability of arrival within the read window, with q. A late arrival costs them the full c_cs, not a partial stall.
    - c_cs is uniform across applications rather than size-dependent.
    - Their arrival model is Poisson or Hawkes, not empirical gap distributions per tool key. Their analysis is expected-cost, like ours.
  - **Positioning (inference):** our retention rule should be presented as an instance of this characterization. Novelty must rest on the coupling price, the read-duration fetch rule, queue awareness and preemption.

**A2. RobustScaler.** *RobustScaler: QoS-Aware Autoscaling for Complex Workloads.* Huajie Qian et al. arXiv 2204.07197 (Apr 2022), cs.DC. ICDE 2022. https://arxiv.org/abs/2204.07197. **FLAG: prior art for timing a preparation action at a quantile of (arrival − preparation lead time), with a fixed quantile rather than a cost ratio.**
- **Setting.** Proactive autoscaling for applications that need one new instance per query, such as a container registry or FaaS. Each instance has a random pending (startup) time τ_i and serves one query. — [arXiv](https://arxiv.org/abs/2204.07197)
- **Result.**
  - For query i with arrival time ξ_i, the paper minimizes E[(ξ_i − τ_i − x_i)^+], the instance's idle time before the query, subject to a hitting probability P(ξ_i > x_i + τ_i) ≥ 1−α.
  - The optimal creation time is x_i* = the α-quantile of (ξ_i − τ_i).
  - An expected-response-time-constrained variant has similar structure.
  - Arrivals follow a non-homogeneous Poisson process fitted with ADMM.
  - — [arXiv PDF](https://arxiv.org/pdf/2204.07197)
- **Relation to our design.**
  - **Parallels:** fetch timing.
  - **Shares (inference):** the decomposition matches ours: start time, preparation lead time, uncertain arrival, and idle holding versus lateness. Their α corresponds to our q.
  - **Differs:**
    - α is a fixed QoS target for every query. This is the "one fixed q whatever its size" special case that our Discussion section argues against.
    - There is no size-dependent holding cost, no shared-capacity price and no conditioning on elapsed idle time.
    - It plans creation times for a batch of upcoming queries from a fitted arrival process.
    - Inference: if their chance constraint is Lagrangified, α becomes a cost ratio, which is our q.

**A3. Cui & Dinitz.** *Ski Rental with Distributional Predictions of Unknown Quality.* Qiming Cui et al. arXiv 2602.21104 (Feb 2026), cs.LG. https://arxiv.org/abs/2602.21104. **Partial flag: the marginal-cost identity is a discrete hazard comparison.**
- **Result.**
  - The prediction is a distribution p̂ over the number of ski days.
  - The algorithm's expected cost is at most OPT_p + O(min(max(η,1)·√b, b·log b)), where OPT_p is the optimal policy for the true distribution p, b is the buy cost and η is the Earth Mover's distance between p and p̂.
  - Consistency is O(√b) and robustness O(b log b), without knowing η. Lower bounds show the trade-off is essentially tight.
  - — [arXiv](https://arxiv.org/abs/2602.21104)
- **Policy structure.**
  - Policies are thresholds A_K ("buy after K days").
  - The paper uses c(A_{t+1}) − c(A_t) = Q_t − b·p_{t+1}, where Q_t is the tail mass. Delaying the purchase therefore pays exactly when b·p_{t+1} > Q_t, a discrete hazard-rate comparison, though the paper does not call it that.
  - The benchmark is the expected cost of the optimal policy for p, not the hindsight optimum. The authors argue the hindsight benchmark forces Ω(b) loss.
  - — [arXiv PDF](https://arxiv.org/pdf/2602.21104)
- **Relation to our design.**
  - **Parallels:** retention (rent-or-buy with a forecast distribution), and the lower bound.
  - **Shares:** distributional advice and an expected-cost benchmark. Inference: their OPT_p plays the role of our weak-duality bound, which is also measured against policies with the same forecasts, not against a clairvoyant.
  - **Differs:** a single decision with unit rent; no capacity coupling, preparation time or size dependence. Its contribution is robustness to a wrong forecast, which our design lacks.

**A4. Kim & Fan.** *Robust and Consistent Ski Rental with Distributional Advice.* Jihwan Kim et al. arXiv 2603.29233 (Mar 2026), cs.LG. https://arxiv.org/abs/2603.29233.
- **Result.**
  - With a perfect distributional prediction, the paper derives an O(n) algorithm for the optimal threshold buy day t* = argmin_t f(t), using the survival function S(t).
  - It gives conditions under which the expected competitive ratio matches the classic randomized bound.
  - For imperfect predictions, a "Clamp Policy" restricts the threshold to a safe range set by a tunable parameter, and is shown to be both robust and consistent.
  - A water-filling algorithm characterizes randomized stopping distributions.
  - Experiments use Gaussian, geometric and bi-modal distributions.
  - — [arXiv](https://arxiv.org/abs/2603.29233); [arXiv PDF](https://arxiv.org/pdf/2603.29233)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Shares:** a full predicted distribution, and multi-modal test cases (our tool gaps are multi-modal).
  - **Differs:** the optimum is found by direct minimization rather than a stated hazard rule. Single decision, unit costs, no capacity, no read time.
  - **Inference:** the clamp is a cheap safeguard we could add to our fetch and retain thresholds.

**A5. Kang et al.** *Learning-Augmented Ski Rental with Discrete Distributions: A Bayesian Approach.* Bosun Kang et al. arXiv 2512.07313 (Dec 2025), cs.LG. OpenAlex lists an AAAI 2026 version, DOI 10.1609/aaai.v40i43.40991. https://arxiv.org/abs/2512.07313
- **Result.**
  - A discrete Bayesian framework keeps exact posterior distributions over the time horizon and can incorporate expert priors.
  - It gives prior-dependent competitive guarantees that interpolate between the worst-case and fully-informed settings.
  - It extends to multiple predictions and contextual information.
  - — [arXiv](https://arxiv.org/abs/2512.07313)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Shares (inference):** updating a posterior as days pass is the same operation as our conditioning of F on idle time t.
  - **Differs:** ski rental only, no capacity or preparation time.

**A6. Sun et al.** *Online Algorithms with Uncertainty-Quantified Predictions.* Bo Sun et al. arXiv 2310.11558 (Oct 2023; latest version Jun 2024), cs.LG. Venue not verified. https://arxiv.org/abs/2310.11558
- **Result.**
  - The paper studies ski rental and online search when predictions come with uncertainty quantification (UQ): the likelihood that the truth lies in a range.
  - It shows that non-trivial changes to algorithm design are needed to exploit UQ.
  - It proposes online learning to exploit more general UQ forms across instances.
  - — [arXiv](https://arxiv.org/abs/2310.11558)
- **Relation to our design.**
  - **Parallels:** retention and fetch.
  - **Shares:** partial distributional information.
  - **Differs:** competitive-ratio analysis, single decision, interval-type UQ rather than a full conditional distribution.

**A7. Shen et al.** *Algorithms with Calibrated Machine Learning Predictions.* Judy Hanwen Shen et al. arXiv 2502.02861 (Feb 2025), stat.ML. ICML 2025 (per arXiv comments). https://arxiv.org/abs/2502.02861
- **Result.**
  - For ski rental, calibrated advice yields near-optimal prediction-dependent performance.
  - In high-variance settings, calibrated advice guides decisions better than alternative UQ methods.
  - — [arXiv](https://arxiv.org/abs/2502.02861)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Inference:** this supports requiring our per-key gap tables to be calibrated. Our "heavier tail beyond the 90th percentile" adjustment is a crude recalibration.
  - **Differs:** a single decision, no timing or capacity.

**A8. Kodialam.** *Optimal Algorithms for Ski Rental with Soft Machine-Learned Predictions.* Rohan Kodialam. arXiv 1903.00092 (Mar 2019), cs.LG. https://arxiv.org/abs/1903.00092
- **Result.**
  - A black-box predictor supplies the probability that there will be at most a threshold number of ski days.
  - The paper derives optimal randomized algorithms that minimize the worst-case expected competitive ratio given that prediction.
  - — [arXiv](https://arxiv.org/abs/1903.00092)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Shares:** probabilistic rather than point advice.
  - **Differs:** a single probability rather than a distribution, worst-case expected competitive ratio, no timing.

**A9. Antoniadis et al.** *Learning-Augmented Dynamic Power Management with Multiple States via New Ski Rental Bounds.* Antonios Antoniadis et al. arXiv 2110.13116 (Oct 2021), cs.DS. https://arxiv.org/abs/2110.13116
- **Result.**
  - A learning-augmented algorithm for multi-state dynamic power management uses predicted idle-period lengths.
  - It is near-optimal with accurate predictions, degrades gracefully, and keeps a worst case nearly equal to the best classical online algorithm.
  - A key ingredient is a new learning-augmented ski-rental algorithm whose dependence on the prediction error is tight.
  - — [arXiv](https://arxiv.org/abs/2110.13116)
- **Relation to our design.**
  - **Parallels:** retention across multiple tiers (inference: GPU, host and SSD resemble power states).
  - **Differs:** point predictions, competitive analysis, no capacity coupling, no prefetch before arrival.

**A10. Shin et al. (two papers).** Yongho Shin et al.:
- *Improved Learning-Augmented Algorithms for the Multi-Option Ski Rental Problem via Best-Possible Competitive Analysis.* arXiv 2302.06832 (Feb 2023), cs.DS. OpenAlex lists an ACM TALG version, DOI 10.1145/3763239. https://arxiv.org/abs/2302.06832
- *On Optimal Consistency-Robustness Trade-Off for Learning-Augmented Multi-Option Ski Rental.* arXiv 2312.02547 (Dec 2023). https://arxiv.org/abs/2312.02547
- **Result.**
  - The first paper gives the first randomized learning-augmented algorithm for multi-option ski rental, built on a provably best-possible randomized competitive algorithm.
  - The second gives a deterministic algorithm matching the known lower bound, and the first nontrivial randomized consistency-robustness lower bound.
  - — [arXiv 2302.06832](https://arxiv.org/abs/2302.06832); [arXiv 2312.02547](https://arxiv.org/abs/2312.02547)
- **Relation to our design.**
  - **Parallels:** multi-tier retention (inference).
  - **Differs:** point predictions, worst-case analysis.

**A11. Narayana & Kash (follow-up).** *Custom Keep-Alive Cache Policies.* Sushirdeep Narayana et al. arXiv 2603.03091 (Mar 2026), cs.GT. EAI GameNets 2026 (per comments). https://arxiv.org/abs/2603.03091
- **Result.**
  - Customers bid their own cold-start (miss) cost.
  - The cache allocation policy learns online from a mixture of fixed-allocation experts. It is asymptotically efficient and monotonically non-increasing in the bid.
  - Two payment schemes are analyzed: Myerson-style and externality-based.
  - — [arXiv](https://arxiv.org/abs/2603.03091)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Shares:** heterogeneous miss costs. Inference: this is the analogue of our size-dependent r_i(1+λL_i).
  - **Differs:** a mechanism-design focus, no read time.

**A12. Dinitz et al.** *Controlling Tail Risk in Online Ski-Rental.* Michael Dinitz et al. arXiv 2308.05067 (Aug 2023), cs.DS. Per its OpenAlex DOI, 10.1137/1.9781611977912.147, a SIAM proceedings version, listed as SODA 2024. https://arxiv.org/abs/2308.05067
- **Result.**
  - The e/(e−1)-competitive randomized algorithm has a more than 37% chance of a ratio above 2.
  - Bounding that tail risk makes the optimal buy probability non-monotone, discontinuous and arbitrarily large.
  - — [arXiv](https://arxiv.org/abs/2308.05067)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Inference:** our rules optimize mean TTFT. If tail TTFT were the objective, the clean threshold structure may not survive.

**A13. Li et al.** *Prediction-Specific Design of Learning-Augmented Algorithms.* Sizhe Li et al. arXiv 2510.14887 (Oct 2025), cs.DS. OpenAlex lists a POMACS / SIGMETRICS 2026 version, DOI 10.1145/3788100. https://arxiv.org/abs/2510.14887
- **Result.**
  - "Strongly-optimal" algorithms are Pareto-optimal in the prediction-specific robustness/consistency trade-off, not only in the worst case.
  - The paper gives explicit algorithms for deterministic and randomized ski rental and one-max search, with a dynamic power management case study.
  - — [arXiv](https://arxiv.org/abs/2510.14887)
- **Relation to our design.** Retention, with point predictions.

**A14. Angelopoulos et al.** *Decision-Theoretic Approaches for Improved Learning-Augmented Algorithms.* Spyros Angelopoulos et al. arXiv 2501.17701 (Jan 2025), cs.DS. https://arxiv.org/abs/2501.17701
- **Result.**
  - Introduces deterministic (distance-based) and stochastic (performance–risk) metrics to choose among algorithms over the full range of prediction error.
  - Applies them to ski rental, one-max search and contract scheduling.
  - — [arXiv](https://arxiv.org/abs/2501.17701)
- **Relation to our design.** An evaluation lens for retention under imperfect forecasts.

**A15. Distributional-advice results outside ski rental.** These are context for Q5.
- **Dinitz et al.** *Binary Search with Distributional Predictions*, arXiv 2411.16030 (Nov 2024). Query complexity O(H(p) + log η), where η is the Earth Mover's distance between the predicted and true distributions. Using the classical algorithm with any single point prediction can do poorly. — [arXiv](https://arxiv.org/abs/2411.16030)
- **Angelopoulos et al.** *Contract Scheduling with Distributional and Multiple Advice*, arXiv 2404.12485, IJCAI 2024. Schedules are optimal when the prediction is accurate and keep the best worst case when it is adversarial. — [arXiv](https://arxiv.org/abs/2404.12485)
- **Angelopoulos & Simon.** *Learning-Augmented Online Bidding in Stochastic Settings*, arXiv 2510.25582. Pareto-optimal consistency/robustness under distributional predictions. — [arXiv](https://arxiv.org/abs/2510.25582)
- **Canonne et al.** *With a Little Help From My Friends: Exploiting Probability Distribution Advice in Algorithm Design*, arXiv 2505.04949. Online metric matching that is optimal under perfect advice and degrades smoothly. — [arXiv](https://arxiv.org/abs/2505.04949)
- **Dinitz & Dong.** *Tree Search With Distributional Predictions*, arXiv 2609.32260 (Sep 2026). Expected queries O(H(p) + k·log η), with a matching lower bound. — [arXiv](https://arxiv.org/abs/2609.32260)
- **Relation to our design:** consistency and robustness templates for acting on a possibly wrong forecast distribution. None involves timing or capacity.

**A16. Point-prediction learning-augmented baselines.** Used for context on the consistency/robustness vocabulary.
- **Wei & Zhang.** *Optimal Robustness-Consistency Trade-offs for Learning-Augmented Online Algorithms*, arXiv 2010.11443, NeurIPS 2020. Gives the first non-trivial lower bounds and optimal trade-offs for ski rental and non-clairvoyant scheduling. — [arXiv](https://arxiv.org/abs/2010.11443)
- **Bamas et al.** *The Primal-Dual method for Learning Augmented Algorithms*, arXiv 2010.11632, NeurIPS 2020. — [arXiv](https://arxiv.org/abs/2010.11632)
- **Anand et al.** *A Regression Approach to Learning-Augmented Online Algorithms*, arXiv 2205.08717. Builds online-optimization benchmarks into the regression loss and gives sample-complexity bounds; covers generalized ski rental. — [arXiv](https://arxiv.org/abs/2205.08717)
- **Wang et al.** *Online Algorithms for Multi-shop Ski Rental with Machine Learned Advice*, arXiv 2002.05808, NeurIPS 2020. — [arXiv](https://arxiv.org/abs/2002.05808)
- **Lykouris & Vassilvitskii.** *Competitive caching with machine learned advice*, arXiv 1802.05399 (preliminary versions at ICML 2018 and SysML 2018). Uses predicted next arrivals for eviction; naive trust can be very poor; competitive ratio capped at O(log k). — [arXiv](https://arxiv.org/abs/1802.05399)
- **Mitzenmacher & Vassilvitskii.** *Algorithms with Predictions* (survey), arXiv 2006.09123. — [arXiv](https://arxiv.org/abs/2006.09123)
- **Zhao et al.** *Learning-Augmented Algorithms: Guarantees, Construction Mechanisms, and System-Level Implications* (survey), arXiv 2609.04787 (Sep 2026). — [arXiv](https://arxiv.org/abs/2609.04787)

**A17. Newsvendor and critical-fractile structure in systems or provisioning on arXiv.** Mentioned briefly.
- **Cheng et al.** *Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty*, arXiv 2607.16892 (Jul 2026), cs.NI.
  - A Wasserstein distributionally robust model of KV reservation under uncertain output length. The authors report "a critical fractile structure that automatically adapts reservation quantiles to different preemption and memory cost regimes."
  - — [arXiv](https://arxiv.org/abs/2607.16892)
  - **Relation:** critical-ratio sizing (how much to reserve), not timing (when to start). LLM serving, so another researcher's scope.
- **An et al.** *The Nonstationary Newsvendor with (and without) Predictions*, arXiv 2305.07993, math.OC.
  - Sequential newsvendor decisions under unknown nonstationary demand, motivated by cloud provisioning and staffing.
  - Order-optimal regret without knowing the level of nonstationarity, and a prediction-aware policy that does not need the predictions' accuracy.
  - — [arXiv](https://arxiv.org/abs/2305.07993)
  - **Relation:** learning a quantile under drift (inference: compare with our recalibration of Φ_k and λ). It decides a quantity, not a time.
- **Zhou et al.** *Dynamic Cloud Service-Capacity Deployment with Costly Response Readiness*, arXiv 2607.20961, math.OC.
  - A dynamic program that decides capacity deployment separately from whether to preserve a costly "response readiness" channel, under predictive beliefs.
  - The authors note that "zero deployment may represent Standby rather than Exit."
  - — [arXiv](https://arxiv.org/abs/2607.20961)
  - **Relation:** a keep-or-release-readiness decision (an analogue of retention), at the scale of capacity planning.
- **Seiringer et al.** *Release Date Optimization in MRP Using Clearing Functions*, arXiv 2503.01862.
  - Computes planned lead times per production order via clearing functions; there is no newsvendor structure.
  - — [arXiv](https://arxiv.org/abs/2503.01862)

**A18. Systems that time a prefetch, mentioned in passing.**
- **Zapridou & Ailamaki.** *Low-Latency Stateful Stream Processing through Timely and Accurate Prefetching*, arXiv 2603.19890, ICDE 2026.
  - Upstream operators extract future access keys and stage the corresponding state in memory before tuples arrive.
  - "Timestamp-Aware Caching" manages prefetched and previously accessed entries jointly.
  - — [arXiv](https://arxiv.org/abs/2603.19890)
  - **Relation:** fetch timing with deterministic hints, no distribution.
- **Zheng et al.** *Efficient Serving for Dynamic Agent Workflows with Prediction-based KV-Cache Management*, arXiv 2605.06472 (May 2026).
  - Predicts future agent invocations, keeps high-potential KV entries in GPU memory, and uses predictions "conservatively during both cache eviction and prefetching."
  - — [arXiv](https://arxiv.org/abs/2605.06472)
  - **Relation:** the nearest neighbour in our domain (LLM serving), which the LLM-serving researcher should cover. It does not state a timing rule.
- **Shahrad et al.** *Serverless in the Wild: Characterizing and Optimizing the Serverless Workload at a Large Cloud Provider*, arXiv 2003.03423, USENIX ATC 2020.
  - Per Narayana & Kash, its policy uses a pre-warm window plus a keep-alive window.
  - — [arXiv](https://arxiv.org/abs/2003.03423); [N&K PDF](https://arxiv.org/pdf/2309.03521)
  - **Relation:** a concrete serverless system, which another researcher covers.
- **Tangential.**
  - Cardas et al., *Ski Rental with Uniform Stochastic Signals*, arXiv 2610.26214: a stochastic signal at a time drawn from a known distribution scaled to [0,t]; best competitive ratios ≈1.478 (deterministic) and ≈1.438 (randomized). — [arXiv](https://arxiv.org/abs/2610.26214)
  - Kesselheim et al., *Online Algorithms via Minimax and Posterior Matching*, arXiv 2608.01616, FOCS 2026: Bayesian design via a "posterior matching" rule, with ski rental among the applications. — [arXiv](https://arxiv.org/abs/2608.01616)

#### Ranked list, part B: not on arXiv (classic; records verified via Crossref and the publisher, abstracts via OpenAlex, publisher or author PDFs)

**B1. Probability-based power-down: Augustine, Irani & Swamy; Irani, Shukla & Gupta; Karlin, Manasse, McGeoch & Owicki.** **FLAG: prior art for picking the expected-cost-minimizing keep threshold from a known idle-time distribution.** I did not find it stated as a hazard rule in these papers.
- **Papers.**
  - Augustine, Irani & Swamy: *Optimal Power-Down Strategies*. John Augustine et al., FOCS 2004, pp. 530–539, DOI 10.1109/FOCS.2004.50; SIAM J. Comput. 37(5):1499–1516 (2008), DOI 10.1137/05063787X. — [Crossref/DOI](https://doi.org/10.1137/05063787X); [author PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf)
  - Irani, Shukla & Gupta: *Online strategies for dynamic power management in systems with multiple power-saving states*. Sandy Irani et al., ACM TECS 2(3):325–346 (2003), DOI 10.1145/860176.860180. Also *Competitive analysis of dynamic power management strategies for systems with multiple power saving states*, DATE 2002, pp. 117–123, DOI 10.1109/DATE.2002.998258. — [DOI](https://doi.org/10.1145/860176.860180)
  - Karlin, Manasse, McGeoch & Owicki (KMMO): *Competitive randomized algorithms for nonuniform problems*. A. R. Karlin et al., Algorithmica 11(6):542–571 (1994), DOI 10.1007/BF01189993; SODA 1990, pp. 301–309, per the reference list of Augustine, Irani & Swamy. — [DOI](https://doi.org/10.1007/BF01189993)
- **Results.**
  - Augustine, Irani & Swamy note that the two-state case is a continuous ski rental. The deterministic "stay up until the energy spent equals the power-up cost" rule is 2-competitive, and randomization gives e/(e−1); both results are credited to KMMO.
  - "If the idle period is generated by a known probability distribution, then the algorithm that chooses T so as to minimize the expected cost is always within a factor of e/(e−1) of optimal," and some distribution makes this tight. They call these "probability-based algorithms." — [author PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf)
  - For multiple sleep states, they give an algorithm that takes the system and the idle-length distribution, in practice a histogram, and outputs the optimal power-down strategy. — [author PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf)
  - Irani, Shukla & Gupta's TECS paper shows that a simple "online learning" scheme, which drives "probability-based" strategies, improves on the competitive ratio of deterministic strategies. — [OpenAlex abstract via DOI](https://doi.org/10.1145/860176.860180)
  - Augustine, Irani & Swamy attribute two results to the DATE 2002 paper (their reference [7]): with additive transition costs, the generalized deterministic algorithm is 2-competitive and the probability-based algorithm is e/(e−1)-competitive. They cite the TECS paper (their [8]) for the experimental gains from estimating the idle-length distribution from recent history. — [Augustine et al. PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf)
- **Relation to our design.**
  - **Parallels:** retention, and the lower-bound benchmark.
  - **Shares:**
    - An expected-cost threshold computed from a known distribution of idle time.
    - A multi-state version: inference, GPU/host/SSD tiers resemble power states with different holding rates and wake-up costs.
    - Learning the distribution from recent history.
  - **Differs:**
    - No capacity coupling or price.
    - The wake-up cost does not depend on size.
    - No wake-up before the request arrives (no prefetch).
    - The benchmark is the clairvoyant offline optimum (a factor of e/(e−1)), whereas our weak-duality bound compares against policies with the same forecasts.

**B2. Weiss.** *Models for Determining Estimated Start Times and Case Orderings In Hospital Operating Rooms.* Elliott N. Weiss, IIE Transactions 22(2):143–150 (1990), DOI 10.1080/07408179008964166. **FLAG: prior art in OR for critical-ratio timing.**
- **Result.**
  - The operating room is modeled as a D/GI/1 queue whose arrival times are decisions.
  - Estimated start times balance idle cost (the start is later than the room is free) against surgeon waiting (the start is before the previous case ends).
  - "The estimated starting times ... depend upon a single critical number that is easily determined from the relevant costs."
  - — [DOI / OpenAlex abstract](https://doi.org/10.1080/07408179008964166)
- **Relation to our design.**
  - **Parallels:** fetch timing.
  - **Shares (inference):** linear earliness and lateness costs give a quantile rule. Weiss: P(previous case ends by S) = w/(w+i). Ours: P(late) = λL/(1+λL).
  - **Differs:** the uncertain quantity is the previous job's duration, not the request's arrival. Costs do not depend on size. There is no conditioning on elapsed time and no shared price.

**B3. Planned lead times and safety lead times (OR).** **FLAG: prior art in OR for critical-ratio timing.**
- **Papers.**
  - M.-A. Ould-Louly et al., *Generalized newsboy model to compute the optimal planned lead times in assembly systems*, IJPR 40(17):4401–4414 (2002), DOI 10.1080/00207540210158825. The title itself states the newsboy structure for planned lead times. A search-result summary of the HAL record describes random component lead times, component holding cost against product backlogging cost, and a generalization of the discrete newsboy model. I could not open the HAL page (bot wall). — [DOI](https://doi.org/10.1080/00207540210158825); [HAL record](https://hal-emse.ccsd.cnrs.fr/emse-00704620)
  - A. Dolgui et al., *A model for supply planning under lead time uncertainty*, IJPE 78(2):145–152 (2002), DOI 10.1016/S0925-5273(00)00180-8. Optimal planned lead times minimizing expected backlogging plus holding cost; the content comes from a secondary index. — [DOI](https://doi.org/10.1016/S0925-5273(00)00180-8); [IAOR index](https://iaorifors.com/paper/38559)
  - Candace Arai Yano, *Setting Planned Leadtimes in Serial Production Systems with Tardiness Costs*, Management Science 33(1):95–106 (1987), DOI 10.1287/mnsc.33.1.95. Optimal planned lead times under stochastic processing times, minimizing holding plus tardiness costs for a given due date. — [DOI](https://doi.org/10.1287/mnsc.33.1.95)
  - Wallace J. Hopp et al., *Setting safety leadtimes for purchased components in assembly systems*, IIE Transactions 25(2):2–11 (1993), DOI 10.1080/07408179308964272. Minimizes carrying plus tardiness cost, or carrying cost under a service constraint. — [DOI](https://doi.org/10.1080/07408179308964272)
  - Mohsen Elhafsi, *Optimal leadtimes planning in serial production systems with earliness and tardiness costs*, IIE Transactions 34(3):233–243 (2002), DOI 10.1080/07408170208928865. — [DOI](https://doi.org/10.1080/07408170208928865)
- **Relation to our design.**
  - **Parallels:** fetch timing ("choosing a lead time").
  - **Shares (inference):**
    - A release time trades early-holding cost h against lateness cost b, so the optimal P(late) = h/(h+b), which is our q with h = λL and b = 1.
    - Items with a higher holding cost accept a higher lateness risk. This is the OR counterpart of our "larger reads accept higher stall risk," derived from the standard fractile formula rather than quoted from these papers.
  - **Differs:** random supply time and a fixed due date, where we have a deterministic read time and a random demand time. The plans are static, with no conditioning on elapsed time, no capacity price and no preemption.

**B4. Cao, Felten, Karlin & Li.** *A study of integrated prefetching and caching strategies.* Pei Cao et al., ACM SIGMETRICS 1995, pp. 188–197, DOI 10.1145/223587.223608. Companion: *Implementation and performance of integrated application-controlled file caching, prefetching, and disk scheduling*, ACM TOCS 14(4):311–343 (1996), DOI 10.1145/235543.235544.
- **Results.**
  - The four rules, quoted from the paper:
    1. Optimal Prefetching: "every prefetch should bring into the cache the next block in the reference stream that is not in the cache."
    2. Optimal Replacement: "discard the block whose next reference is furthest in the future."
    3. Do No Harm: "never discard block A to prefetch block B when A will be referenced before B."
    4. First Opportunity: perform each prefetch at the first opportunity.
  - The conservative strategy is within a factor of 2 of optimal, and the aggressive strategy significantly less than 2. In trace-driven simulation, running time drops by up to 50%.
  - The TOCS paper combines LRU-SP kernel allocation with "controlled-aggressive" per-process prefetching. Running time drops by 3–49% (average 26%) for single-process and 5–76% (average 32%) for multiprocess workloads.
  - — [author PDF](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf); [DOI](https://doi.org/10.1145/223587.223608); [TOCS DOI](https://doi.org/10.1145/235543.235544)
- **Relation to our design.**
  - **Parallels:** preemption, fetch timing and replacement.
  - **Shares:**
    - Do No Harm is the principle behind our preemption. Inference: our F_j(r_j|t) order generalizes "referenced later" to "less likely to be needed within r_j."
    - First Opportunity is the λ → 0 limit of our fetch rule, where every read starts when the gap begins.
    - Rule 2 (Belady) is the exact-forecast limit of our retention index.
  - **Differs:**
    - The reference stream is known and deterministic.
    - Blocks have unit size and a fixed fetch time F.
    - Occupancy is not priced except through displacement, so "first opportunity" is optimal there and is not optimal for us.
    - The guarantees are worst-case approximation factors.

**B5. Patterson et al. (TIP) and Tomkins et al. (TIPTOE).** *Informed prefetching and caching.* R. H. Patterson et al., SOSP 1995, pp. 79–95, DOI 10.1145/224056.224064. Follow-up: *Informed multi-process prefetching and caching*, Andrew Tomkins et al., SIGMETRICS 1997, pp. 100–114, DOI 10.1145/258612.258680.
- **Results.**
  - Applications disclose hints of future accesses. Cost-benefit analysis allocates buffers among prefetching hinted blocks, caching hinted blocks and LRU caching.
  - Benefits and costs are expressed in a "common currency": the change in I/O service time per buffer-access.
  - The "prefetch horizon" P(T_CPU) = T_disk/(T_CPU + T_hit + T_driver) is the prefetch depth beyond which stall is zero, so prefetching deeper brings no benefit.
  - A hinted block can be ejected and re-prefetched later when the savings elsewhere exceed "the cost of ejecting the block."
  - TIPTOE adds adaptivity to dynamic disk load.
  - — [author PDF](https://www.pdl.cmu.edu/PDL-FTP/TIP/SOSP15.pdf); [DOI](https://doi.org/10.1145/224056.224064); [TIPTOE DOI](https://doi.org/10.1145/258612.258680)
- **Relation to our design.**
  - **Parallels:** fetch timing, preemption and the price.
  - **Shares (inference):**
    - Our λ, in seconds of TTFT per token-second, is a common currency in TIP's sense.
    - The prefetch horizon is the deterministic limit of our rule ("start x_i seconds before arrival").
    - Ejecting a hinted block and re-prefetching it later is our "preempt a staged read, then replan."
    - TIPTOE's load adaptivity parallels our queue-aware Ŵ and adaptive λ.
  - **Differs:** hints give the exact order of accesses, not a distribution of arrival times. Costs are greedy, local, unit-size estimates rather than derived from a Lagrangian.

**B6. Jiang & Kleinrock.** *An adaptive network prefetch scheme.* Z. Jiang et al., IEEE JSAC 16(3):358–368 (1998), DOI 10.1109/49.669044. **FLAG: prior art for a probability threshold that rises with resource cost and load.** It decides which items to fetch, not when.
- **Result.**
  - A prediction module estimates each file's access probability.
  - A threshold module computes a per-server prefetch threshold "dynamically based on system load, capacity, and the cost of time and system resources."
  - Prefetching files whose probability is at least the threshold always lowers average cost.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1109/49.669044)
- **Relation to our design.**
  - **Parallels:** fetch (whether) and the price.
  - **Shares:** a probability threshold set by the price of resources, like q, which rises with λ.
  - **Differs:** a one-shot choice of which items to fetch. There is no read-duration window, no conditioning on idle time and no learned dual price.

**B7. Krishnan, Long & Vitter.** *Adaptive Disk Spindown via Optimal Rent-to-Buy in Probabilistic Environments.* P. Krishnan et al., Algorithmica 23(1):31–56 (1999), DOI 10.1007/PL00009249. A search result lists an earlier ICML 1995 version, which I did not open.
- **Result.**
  - The paper makes a sequence of rent-to-buy decisions when use times are i.i.d. from an unknown distribution.
  - The algorithm uses O(√t) time and space. Its expected cost for the t-th use converges to optimal at rate O(√(log t / t)) for any bounded distribution; an O(1)-space variant almost converges.
  - Motivations include disk spin-down, thread blocking, and "virtual circuit holding times in IP-over-ATM networks" (a connection timeout).
  - Trace simulations beat the nonadaptive 2-competitive algorithm.
  - — [Nokia Bell Labs abstract page](https://www.nokia.com/bell-labs/publications-and-media/publications/adaptive-disk-spindown-via-optimal-rent-to-buy-in-probabilistic-environments/); [DOI](https://doi.org/10.1007/PL00009249)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Shares:** learning the distribution and then the optimal threshold from past durations. Inference: this is our per-key Φ_k learned from past gaps.
  - **Differs:** a single resource with unit rent and fixed buy cost; no capacity coupling or prefetch.

**B8. Simunic et al. and Benini et al.**
- **Papers.** *Event-driven power management*, T. Simunic et al., IEEE TCAD 20(7):840–857 (2001), DOI 10.1109/43.931003. *Policy optimization for dynamic power management*, L. Benini et al., IEEE TCAD 18(6):813–833 (1999), DOI 10.1109/43.766730.
- **Result.**
  - Earlier stochastic-control power management (Benini et al.) is optimal only when the system can be modeled with exponential distributions, and the authors measure a large mismatch for real request arrivals.
  - Two new models handle general distributions: renewal theory, and a time-indexed semi-Markov decision process. Both give globally optimal policies, verified on hard disks and WLAN.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1109/43.931003); [Benini DOI](https://doi.org/10.1109/43.766730)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Shares (inference):** non-exponential idle times force decisions indexed by elapsed time. This is the same reason we condition F on idle time.
  - **Differs:** an energy objective, a single device, no capacity coupling.

**B9. Vellanki & Chervenak.** *A cost-benefit scheme for high performance predictive prefetching.* Vivekanand Vellanki et al., SC 1999, DOI 10.1145/331532.331582.
- **Result.**
  - A probability tree predicts access probabilities.
  - The scheme chooses which blocks to prefetch by probability of access, and decides "whether to prefetch a particular block at a given time using a cost-benefit analysis."
  - Combined with one-block lookahead, it cuts file-cache miss rates by up to 36% on workloads that gain nothing from sequential prefetching.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1145/331532.331582)
- **Relation to our design.** This is TIP's cost-benefit extended to probabilistic hints (the follow-up the key question asks about). It trades stall against buffer use, but over access probability rather than an arrival-time distribution.

**B10. Higgins et al. (IMP).** *Informed mobile prefetching.* Brett D. Higgins et al., MobiSys 2012, pp. 155–168, DOI 10.1145/2307636.2307651.
- **Result.**
  - The abstract (via OpenAlex) frames prefetching as weighing hidden latency against energy and cellular data costs.
  - The details below come from a search-tool summary of the paper text. ACM DL returned HTTP 403 and the PDF was not retrievable, so treat them as lower confidence.
    - IMP uses cost-benefit analysis to decide when to prefetch.
    - "Goal-directed adaptation" minimizes response time while meeting budgets for battery lifetime and cellular data.
    - IMP is inspired by TIP.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1145/2307636.2307651); [paper copy listed in search](https://typeset.io/pdf/informed-mobile-prefetching-17wqdetih3.pdf)
- **Relation to our design.**
  - **Parallels:** the price.
  - **Shares (inference):** adapting a resource price so that consumption meets a budget is the systems ancestor of our dual ascent on λ.
  - **Differs:** no arrival-time distribution and no stated theory.

**B11. Follow-ups in integrated prefetching and caching theory.**
- **Kimbrel & Karlin.** *Near-Optimal Parallel Prefetching and Caching*, SIAM J. Comput. 29(4):1051–1082 (2000), DOI 10.1137/S0097539797326976; FOCS 1996, DOI 10.1109/SFCS.1996.548513.
  - Extending aggressive to D disks is suboptimal by a factor near D in the worst case. The new "reverse aggressive" algorithm is near-optimal.
  - — [DOI](https://doi.org/10.1137/S0097539797326976)
- **Albers, Garg & Leonardi.** *Minimizing stall time in single and parallel disk systems*, J. ACM 47(6):969–986 (2000), DOI 10.1145/355541.355542; STOC 1998, DOI 10.1145/276698.276858.
  - The optimal single-disk schedule is computable in polynomial time via linear programs.
  - Approximations for stall time with parallel disks use a few extra cache blocks.
  - — [DOI](https://doi.org/10.1145/355541.355542)
- **Albers & Büttner.** *Integrated prefetching and caching in single and parallel disk systems*, SPAA 2003, pp. 109–117, DOI 10.1145/777412.777431; Information and Computation 198(1):24–39 (2005), DOI 10.1016/j.ic.2005.01.003.
  - Shows the original analysis of Aggressive was too pessimistic, with a tight refined bound.
  - Gives the first polynomial-time minimum-stall schedule for D disks, using at most 3(D−1) extra cache locations, built on LP plus "synchronized schedules."
  - — [DOI](https://doi.org/10.1145/777412.777431)
- **Relation to our design.**
  - **Parallels:** stall-time objective; inference, LP-based optimality and lower bounds parallel our weak-duality lower bound.
  - **Differs:** offline and deterministic sequences, unit blocks, no arrival uncertainty.

**B12. Spinning, spin-down and rent-or-buy with distributional information.**
- **Karlin, Li, Manasse & Owicki.** *Empirical studies of competitive spinning for a shared-memory multiprocessor*, SOSP 1991, pp. 41–55, DOI 10.1145/121132.286599 (the Crossref record spells it "competitve").
  - Seven spin-then-block strategies are evaluated on measured lock-waiting-time distributions. Adaptive strategies beat non-adaptive ones, and always-block performs poorly.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1145/121132.286599)
- **Helmbold et al.** *A dynamic disk spin-down technique for mobile computing*, MobiCom 1996, pp. 130–142, DOI 10.1145/236387.236423.
  - An online-learning timeout uses as little as 88% of the energy of the best fixed timeout chosen in retrospect.
  - — [DOI](https://doi.org/10.1145/236387.236423)
- **Fujiwara & Iwama.** *Average-Case Competitive Analyses for Ski-Rental Problems*, Algorithmica 42(1):95–107 (2005), DOI 10.1007/s00453-004-1142-x (ISAAC 2002).
  - Under an exponential distribution f(t)=λe^{−λt}: rent forever if 1/λ ≤ s; otherwise buy after about s²λ rentals. Here s is the ratio of the buy cost to the rent cost.
  - The authors note this differs from both worst-case and traditional average-cost analysis.
  - — [DOI / Semantic Scholar abstract](https://doi.org/10.1007/s00453-004-1142-x)
- **Khanafer et al.** *The constrained Ski-Rental problem and its application to online cloud cost optimization*, INFOCOM 2013, pp. 1492–1500, DOI 10.1109/INFCOM.2013.6566944.
  - When the first or second moment of the duration distribution is known, a new class of randomized algorithms achieves the best worst-case expected competitive ratio.
  - It is applied to the choice between paying for compute and paying for cache storage in a cloud file system.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1109/INFCOM.2013.6566944)
- **Relation to our design.**
  - **Parallels:** retention.
  - **Inference:** Fujiwara & Iwama's "rent forever if 1/λ ≤ s" is the constant-hazard case of the hazard rule (λ·s ≥ 1, i.e., hazard × buy cost ≥ rent rate).

**B13. Learning-augmented ski rental with point or expert predictions (no arXiv version found in my searches).**
- **Purohit, Svitkina & Kumar.** *Improving Online Algorithms via ML Predictions*, NeurIPS 2018.
  - Ski rental and non-clairvoyant scheduling with predictions. The algorithms are "oblivious to the performance of the predictor, improve with better predictions, but do not degrade much if the predictions are poor."
  - — [NeurIPS page](https://papers.nips.cc/paper_files/paper/2018/hash/73a427badebe0e32caa2e1fc7530b7f3-Abstract.html)
- **Gollapudi & Panigrahi.** *Online Algorithms for Rent-Or-Buy with Expert Advice*, ICML 2019, PMLR 97:2319–2327.
  - Multiple experts, with matching lower bounds.
  - — [PMLR](https://proceedings.mlr.press/v97/gollapudi19a.html)
- **Anand, Ge & Panigrahi.** *Customizing ML Predictions for Online Algorithms*, ICML 2020, PMLR 119:303–313.
  - Building optimization benchmarks into ML loss functions improves rent-or-buy while keeping a worst-case guarantee.
  - — [PMLR](https://proceedings.mlr.press/v119/anand20a.html)
- **Diakonikolas et al.** *Learning Online Algorithms with Distributional Advice*, ICML 2021, PMLR 139:2687–2696.
  - Learns from i.i.d. samples of the instance distribution, for ski rental and prophet inequalities.
  - For general distributions, beating the worst-case competitive ratio with any finite sample size is information-theoretically impossible. For log-concave distributions, poly(1/ε) samples give a (1+ε)-competitive ratio.
  - — [PMLR](https://proceedings.mlr.press/v139/diakonikolas21a.html)
- **Mahdian, Nazerzadeh & Saberi.** *Online Optimization with Uncertain Information*, ACM TALG 8(1) (2012), DOI 10.1145/2071379.2071381.
  - An early framework that uses additional information while keeping a reasonable competitive ratio if the information is wrong.
  - — [DOI](https://doi.org/10.1145/2071379.2071381)

**B14. Preparation actions and probabilistic prefetch, mentioned in passing.**
- **Cohen & Kaplan.** *Prefetching the means for document transfer: a new approach for reducing Web latency*, INFOCOM 2000, DOI 10.1109/INFCOM.2000.832260; Computer Networks 39(4):437–455 (2002), DOI 10.1016/S1389-1286(02)00184-6.
  - Pre-resolving DNS names, pre-connecting TCP, and "pre-warming" by sending a dummy HEAD request. These beat document prefetching in latency improvement per unit of bandwidth.
  - — [DOI / OpenAlex abstract](https://doi.org/10.1109/INFCOM.2000.832260)
- **Vitter & Krishnan.** *Optimal prefetching via data compression*, J. ACM 43(5):771–793 (1996), DOI 10.1145/234752.234753.
  - Prefetchers derived from data compression are optimal in the limit in page-fault rate for Markov sources.
  - — [DOI](https://doi.org/10.1145/234752.234753)
- **Parate et al.** *Practical prediction and prefetch for faster access to applications on mobile phones*, UbiComp 2013, pp. 275–284, DOI 10.1145/2493432.2493490.
  - The APPM predictor estimates which app will be used next and when. The system prefetches on screen unlock.
  - — [DOI](https://doi.org/10.1145/2493432.2493490)
- **Relation:** preparation actions driven by predictions. None states a timing rule derived from costs.

### Inferences
- **Novelty, retention.** The hazard-rate keep rule should not be claimed as new. It is Narayana & Kash's Theorem 2, and it is the classic first-order condition of expected-cost ski rental: Cui & Dinitz's identity in discrete time, and Fujiwara & Iwama's exponential case.
- **Novelty, fetch.** The critical-ratio timing should be credited to the earliness–tardiness newsvendor in OR (Weiss; planned lead times). RobustScaler is its fixed-quantile systems instance.
- **What appears new**, among the papers I could verify:
  - (a) Earliness is priced by a size-dependent shadow price λL that is learned by dual ascent under a shared capacity constraint.
  - (b) The fetch rule uses a read of non-zero duration with partial stalls, measured net of queueing delay (x = r − Ŵ), and conditioned on idle time.
  - (c) Fetch and retention share the threshold q. Narayana & Kash have a shared threshold only with zero read time.
  - (d) Staged reads are preempted in the order of F_j(r_j|t), a probabilistic version of "do no harm."
  - (e) The weak-duality bound is measured against policies with the same forecasts.
- **Limits of the design.**
  - With exact forecasts, our design reduces to TIP's prefetch horizon and Cao's Belady replacement.
  - With λ → 0, it reduces to Cao's "first opportunity."
  - These limits are good anchors for the related-work narrative.
- **Benchmark.** The benchmarks differ in kind. Competitive analysis compares against a clairvoyant offline optimum, and a known-distribution threshold is within e/(e−1) of it. Our D(λ) and Cui & Dinitz's OPT_p are Bayesian benchmarks, with the same forecasts. A paper should say which benchmark it uses.

### Gaps
- **No hazard rule in the classic rent-or-buy texts.** I could not verify that KMMO, Augustine–Irani–Swamy or Irani–Shukla–Gupta state the known-distribution optimum explicitly as a hazard-rate rule. Springer pages returned a bot challenge, and the PDF I read says only "choose T to minimize expected cost." The explicit hazard statement I found is Narayana & Kash (2023).
- **Unverified leads.** For each of these, the record or link is known but the content or venue claim is unverified:
  - **Baker & Trietsch.** *Safe scheduling: Setting due dates in single-machine problems*, EJOR 196(1):69–77 (2009), DOI 10.1016/j.ejor.2008.02.009. The record is verified via Crossref. A critical-fractile due-date result is likely but unverified. The same holds for Baker, *Setting optimal due dates in a basic safe-scheduling model*, Computers & OR 41:109–114 (2014), DOI 10.1016/j.cor.2013.07.022.
  - **Xu, Xu & Li.** *On the on-line rent-or-buy problem in probabilistic environments*, J. Global Optim. 38(1):1–20, DOI 10.1007/s10898-006-9079-z. The record is verified; the content is not (Springer bot wall).
  - **Cohen, Kaplan & Zwick.** *Connection caching* (STOC 1999, DOI 10.1145/301250.301416) and *Connection caching: model and algorithms* (JCSS 67(1):92–126, 2003, DOI 10.1016/S0022-0000(03)00041-2). Records verified; content not. These are relevant to connection keep-alive.
  - **Lu & De Micheli.** *Comparing system level power management policies*, IEEE Design & Test 18(2):10–19 (2001), DOI 10.1109/54.914592. Record only.
  - **Kimbrel et al.** *A trace-driven comparison of algorithms for parallel prefetching and caching*, OSDI 1996. The USENIX page appeared in search results but was not opened.
  - **INFOCOM 2022 serverless-edge paper.** A search summary said it maps keep-alive to ski rental (PDF on fangmingliu.github.io). Title and authors are unverified, and serverless keep-alive is another researcher's scope.
  - **"PLEA" probability-based DPM by Ma & Gupta.** Seen only in a search summary.
  - **Virtual-circuit holding-time papers.** Cited inside the Krishnan–Long–Vitter abstract; not verified.
  - **Shahrad et al.'s histogram percentiles.** The exact pre-warm and keep-alive percentiles (from memory) are not verified.
  - **MagicScaler** (VLDB 2023, DOI 10.14778/3611540.3611566), uncertainty-aware predictive autoscaling. OpenAlex record only.
  - **Unverified venues.** Kim & Fan at ICML 2026 (a search result listed icml.cc/virtual/2026/poster/66404, not opened). Antoniadis et al. at NeurIPS 2021. Dinitz et al., binary search, at NeurIPS 2024. Sun et al. at ICML 2024.

## Q1. Integrated prefetching and caching: the classic rules, follow-ups, and arXiv-era extensions to probabilistic hints or stall-versus-occupancy trade-offs

### Takeaway
- **The classic theory.**
  - It assumes a known, deterministic reference stream and unit-size blocks (Cao et al.; Kimbrel & Karlin; Albers et al.).
  - Buffer occupancy costs nothing except by displacing other blocks, so "first opportunity" is optimal.
  - TIP adds a cost-benefit "common currency" and a prefetch horizon.
- **Probabilistic extensions predate arXiv.** Vellanki & Chervenak, Jiang & Kleinrock and IMP use access probabilities or budgets, but not distributions of arrival time.
- **No arXiv paper found.** None extends the Cao/Kimbrel/Albers theory to distributional timing hints. The nearest arXiv-era works are RobustScaler (quantile pre-start) and Narayana & Kash (pre-warm and keep-alive by hazard rate).

### Cited Findings
- **Cao et al. (1995).** The four rules are quoted in B4. The conservative strategy is within 2× of optimal, and the aggressive strategy is significantly less than 2×. — [author PDF](https://homes.cs.washington.edu/~karlin/papers/sigmetrics.pdf); [DOI](https://doi.org/10.1145/223587.223608)
- **TIP.** Cost-benefit analysis in a "common currency" of change in I/O service time per buffer-access. The prefetch horizon is P(T_CPU) = T_disk/(T_CPU+T_hit+T_driver). A block is ejected when the benefit elsewhere exceeds the ejection cost. — [author PDF](https://www.pdl.cmu.edu/PDL-FTP/TIP/SOSP15.pdf)
- **TIPTOE.** Cost-benefit allocation across processes, adaptive to disk load. It delivered 7% lower execution time than LRU-SP on average. — [DOI](https://doi.org/10.1145/258612.258680)
- **Kimbrel & Karlin.** Reverse aggressive is near-optimal for parallel disks; aggressive is off by a factor near D. — [DOI](https://doi.org/10.1137/S0097539797326976)
- **Albers, Garg & Leonardi.** The optimal single-disk schedule is computable in polynomial time via LP; stall time is "important and harder to approximate." — [DOI](https://doi.org/10.1145/355541.355542)
- **Albers & Büttner.** The first polynomial-time minimum-stall schedule for D disks, with 3(D−1) extra cache locations. — [DOI](https://doi.org/10.1145/777412.777431)
- **Vellanki & Chervenak.** Probability-of-access prediction with cost-benefit decisions on whether to prefetch a block "at a given time." — [DOI](https://doi.org/10.1145/331532.331582)
- **Jiang & Kleinrock.** Prefetch any item whose access probability is at least a threshold computed from load, capacity and the cost of time and resources. — [DOI](https://doi.org/10.1109/49.669044)
- **IMP.** Cost-benefit prefetching under energy and data budgets (lower-confidence details; see B10). — [DOI](https://doi.org/10.1145/2307636.2307651)
- **arXiv-era, closest.**
  - RobustScaler: x_i* is the α-quantile of (ξ_i − τ_i). — [arXiv PDF](https://arxiv.org/pdf/2204.07197)
  - Narayana & Kash: pre-warm and keep-alive windows set by the hazard rate. — [arXiv PDF](https://arxiv.org/pdf/2309.03521)
  - Keyed prefetching in stream processing stages state before tuples arrive. — [arXiv](https://arxiv.org/abs/2603.19890)
- **Learning-augmented caching uses point predictions of the next arrival for eviction**, with no prefetching or stall model. — [arXiv 1802.05399](https://arxiv.org/abs/1802.05399)

### Inferences
- **Mapping to our mechanisms.**
  - Our preemption order is a probabilistic "do no harm": an arrived request always outranks speculative staged reads, and among staged reads the one least likely to be needed within its re-read time yields first.
  - TIP's ejection with later re-prefetch is the deterministic precursor of "preempt, then let the fetch rule replan."
- **Why we depart from "first opportunity."** Once occupancy carries a price (λL per second), the stall avoided by starting earlier must be weighed against the held memory. In Cao's model the occupancy price is implicitly zero, hence "first opportunity."
- **Framing for the paper.** "Integrated prefetching and caching with a priced buffer and distributional arrival times" appears to be open in the theory literature I could reach.

### Gaps
- **Incomplete arXiv coverage.** The arXiv API returned HTTP 429, and the arxiv.org/search queries "prefetching and caching" and "prefetching competitive ratio predictions" also hit 429. OpenAlex and web search found no arXiv extension of Cao/Kimbrel/Albers to probabilistic timing hints, but that coverage is not exhaustive. — [arXiv search, the query that was not completed](https://arxiv.org/search/?query=%22prefetching+and+caching%22&searchtype=all)
- **Kimbrel et al. (OSDI 1996) not checked.** The trace-driven comparison was not opened (see Q0 Gaps).

## Q2. Newsvendor and critical-fractile formulations for timing a prefetch or pre-provisioning action, choosing a lead time, or sizing a buffer

### Takeaway
- **OR.** Critical-ratio timing is classical. Appointment start times are set by "a single critical number" (Weiss 1990), and planned lead times follow a "generalized newsboy model" (Ould-Louly & Dolgui 2002; Yano 1987; Hopp & Spearman 1993).
- **Systems on arXiv.**
  - The closest timing result is RobustScaler's α-quantile instance creation. It is chance-constrained with a fixed α, not cost-ratio derived.
  - Critical-fractile buffer sizing appears in Cheng et al. (2026), for KV reservation.
- **Negative finding.** arxiv.org searches for "newsvendor prefetch," "newsvendor cache" and "newsvendor serverless" returned no results.

### Cited Findings
- **Weiss (1990).** Estimated start times balance operating-room idle time against surgeon waiting time and "depend upon a single critical number" set by the costs. The model is a D/GI/1 queue with controllable arrival times. — [DOI](https://doi.org/10.1080/07408179008964166)
- **Planned lead times.**
  - Ould-Louly & Dolgui (2002): *Generalized newsboy model to compute the optimal planned lead times in assembly systems*. — [DOI](https://doi.org/10.1080/00207540210158825)
  - Yano (1987): holding plus tardiness costs under stochastic lead times. — [DOI](https://doi.org/10.1287/mnsc.33.1.95)
  - Hopp & Spearman (1993): carrying plus tardiness costs, or a service-constrained version. — [DOI](https://doi.org/10.1080/07408179308964272)
  - Elhafsi (2002): earliness plus tardiness costs over N stages. — [DOI](https://doi.org/10.1080/07408170208928865)
- **RobustScaler.** Minimizes E[(ξ−τ−x)^+] subject to P(ξ > x+τ) ≥ 1−α, giving x* as the α-quantile of (ξ−τ). — [arXiv PDF](https://arxiv.org/pdf/2204.07197)
- **Cheng et al. (2026).** Distributionally robust KV reservation with "a critical fractile structure that automatically adapts reservation quantiles to different preemption and memory cost regimes." — [arXiv](https://arxiv.org/abs/2607.16892)
- **An et al.** A nonstationary newsvendor with and without predictions, motivated by cloud provisioning, with order-optimal regret. — [arXiv](https://arxiv.org/abs/2305.07993)
- **Zero-result searches** (arxiv.org/search, 2026-11-17, all fields):
  - [newsvendor prefetch](https://arxiv.org/search/?query=newsvendor+prefetch&searchtype=all)
  - [newsvendor cache](https://arxiv.org/search/?query=newsvendor+cache&searchtype=all)
  - [newsvendor serverless](https://arxiv.org/search/?query=newsvendor+serverless&searchtype=all)
  - The query ["critical fractile"](https://arxiv.org/search/?query=%22critical+fractile%22&searchtype=all) returned 4 results, none about timing.
  - ["planned lead time"](https://arxiv.org/search/?query=%22planned+lead+time%22&searchtype=all) returned 1 result, the clearing-function MRP paper above.

### Inferences
- **Derivation.** Our fetch rule is the earliness–tardiness newsvendor.
  - Completion time ℓ = s + r and need time D = A + W.
  - Expected cost E[(ℓ−D)^+] + λL·(r + E[(D−ℓ)^+]) is minimized where P(D < ℓ) = λL/(1+λL).
  - Weiss and the planned-lead-time models solve the same first-order condition with constant per-unit costs.
- **What we add.**
  - Earliness cost proportional to read size, at a learned shadow price.
  - Conditioning on elapsed idle time (the papers above plan statically).
  - A net lead time that subtracts the predicted queueing delay.
  - Coupling of many "orders" through one capacity.
- **RobustScaler as a foil.** It is the natural foil for "timing every read to a fixed quantile," and our design's Discussion already argues against a fixed quantile.
- **The OR analogue of size-dependent risk.** "Larger reads accept more stall risk yet start earlier" corresponds in OR to "a higher holding cost lowers the service level, while a longer lead time moves the release earlier." I derived this from the fractile formula; I did not find it quoted.

### Gaps
- **No newsvendor timing for prefetching on arXiv.** I found no arXiv paper that applies a newsvendor or critical-ratio rule to time a prefetch or pre-warm under an arrival-time distribution with size-dependent holding cost. arXiv search coverage was partial (HTTP 429 after about 11 queries).
- **Not checked.** The Baker & Trietsch "safe scheduling" critical-fractile due-date result is unverified (see Q0 Gaps).
- **Not found.** No "just-in-time provisioning from quantile forecasts" paper with an explicit cost-ratio quantile turned up, beyond RobustScaler's chance constraint.

## Q3. Ski rental with predictions and known distributions: what is the structure of the optimal policy, and is it a hazard-rate threshold?

### Takeaway
- **Known distribution.** The optimal deterministic policy is the threshold that minimizes expected cost.
- **Hazard structure.** At an interior optimum, its first-order condition compares hazard × buy cost with the rent rate.
  - This is explicit in Narayana & Kash's Theorem 2, in continuous time and with history-conditioned hazards.
  - It is implicit in Cui & Dinitz's marginal identity, in discrete time.
- **Monotone versus non-monotone hazards.** With a monotone hazard the optimum is a single crossing. Otherwise, optimal windows sit at sign changes and must be compared globally.
- **Benchmark gap.** The known-distribution threshold is within e/(e−1) of the offline optimum, and this is tight. This is credited to KMMO by Augustine, Irani & Swamy.
- **Learning-augmented work.** Most uses point predictions with consistency/robustness guarantees. Distributional versions appeared in 2025–2026.

### Cited Findings
- **Narayana & Kash.** Hazard-rate characterization (Theorem 2), single-window and pre-warm corollaries. — [arXiv PDF](https://arxiv.org/pdf/2309.03521)
- **Cui & Dinitz.** c(A_{t+1}) − c(A_t) = Q_t − b·p_{t+1}. Benchmark OPT_p, and additive loss O(min(max(η,1)√b, b log b)). — [arXiv PDF](https://arxiv.org/pdf/2602.21104)
- **Kim & Fan.** O(n) computation of the optimal threshold t* from the predicted distribution, and a Clamp Policy for robustness. — [arXiv](https://arxiv.org/abs/2603.29233)
- **Kang et al.** A Bayesian posterior over the horizon, with prior-dependent guarantees. — [arXiv](https://arxiv.org/abs/2512.07313)
- **Augustine, Irani & Swamy, citing KMMO.** The deterministic rule is 2-competitive and randomization gives e/(e−1). A known-distribution threshold is within e/(e−1) of optimal, and this is tight. — [author PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf)
- **Fujiwara & Iwama.** Exponential case: rent forever if 1/λ ≤ s, otherwise buy after about s²λ. — [DOI](https://doi.org/10.1007/s00453-004-1142-x)
- **Khanafer et al.** Moment-constrained ski rental. — [DOI](https://doi.org/10.1109/INFCOM.2013.6566944)
- **Krishnan, Long & Vitter.** Learn the optimal threshold from i.i.d. use times, converging as O(√(log t/t)). — [Bell Labs abstract](https://www.nokia.com/bell-labs/publications-and-media/publications/adaptive-disk-spindown-via-optimal-rent-to-buy-in-probabilistic-environments/)
- **Point and expert predictions.**
  - Purohit et al. 2018 — [NeurIPS](https://papers.nips.cc/paper_files/paper/2018/hash/73a427badebe0e32caa2e1fc7530b7f3-Abstract.html)
  - Gollapudi & Panigrahi 2019 — [PMLR](https://proceedings.mlr.press/v97/gollapudi19a.html)
  - Anand et al. 2020 — [PMLR](https://proceedings.mlr.press/v119/anand20a.html)
  - Wei & Zhang 2020, optimal trade-offs — [arXiv](https://arxiv.org/abs/2010.11443)
- **Multi-option and multi-shop ski rental.**
  - Shin et al. — [arXiv](https://arxiv.org/abs/2302.06832); [arXiv](https://arxiv.org/abs/2312.02547)
  - Wang et al. — [arXiv](https://arxiv.org/abs/2002.05808)
- **Tail risk.** Constraining it changes the optimal randomized structure. — [arXiv](https://arxiv.org/abs/2308.05067)

### Inferences
- **Our retention rule is this hazard condition.** With a rent of λL per second and a "buy" cost of r(1+λL), the condition is μ(t)·r(1+λL) ≥ λL.
- **Why we use a max-over-horizons index.** It is the device for non-monotone hazards, which Narayana & Kash's multi-window theorem addresses.
- **No paper with the full structure.** I found none on rent-or-buy with a known distribution that includes a non-zero preparation time before arrival, or many sessions coupled by a capacity price. This is where our design adds structure.

### Gaps
- **Classic full texts unreachable.** KMMO's full text (Springer bot wall) could not be checked for an explicit hazard statement.
- **Not checked.** Xu, Xu & Li (2007) is unverified (see Q0 Gaps).

## Q4. Dynamic power management, disk spin-down, and connection or keep-alive timeouts under a known idle-time distribution

### Takeaway
- **Bounds.** With two states, power-down is continuous ski rental: 2-competitive deterministic, e/(e−1) randomized.
- **Probability-based policies.** These choose the threshold that minimizes expected cost under a known (learned) idle-time distribution, and are within e/(e−1) of optimal.
- **Multiple states.** Augustine, Irani & Swamy compute the optimal multi-state strategy from a distribution.
- **Non-exponential idle times.** These require policies indexed by elapsed time (Simunic et al.).
- **arXiv follow-ups.** Learning-augmented multi-state power management (Antoniadis et al. 2021) and prediction-specific designs (Li et al. 2025).
- **Timeouts.** Krishnan–Long–Vitter explicitly target virtual-circuit (connection) holding times. Narayana & Kash give the hazard-rate theory for serverless keep-alive.

### Cited Findings
- **Augustine, Irani & Swamy.** Probability-based algorithms, the e/(e−1) bound with tightness, and an optimal strategy from a histogram distribution. — [author PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf); [DOI](https://doi.org/10.1137/05063787X)
- **Irani, Shukla & Gupta.**
  - TECS 2003: probability-based online power management with "online learning." — [DOI](https://doi.org/10.1145/860176.860180)
  - DATE 2002: with additive transition costs, the probability-based algorithm is e/(e−1)-competitive, as summarized by Augustine, Irani & Swamy. — [DATE DOI](https://doi.org/10.1109/DATE.2002.998258); [author PDF](https://math.uwaterloo.ca/~cswamy/papers/dpm-journ.pdf)
- **Simunic et al.** Renewal-theory and time-indexed semi-Markov models give globally optimal policies for general (non-exponential) distributions. — [DOI](https://doi.org/10.1109/43.931003)
- **Helmbold et al.** An online-learned spin-down timeout. — [DOI](https://doi.org/10.1145/236387.236423)
- **Karlin, Li, Manasse & Owicki.** Spinning with measured waiting-time distributions; adaptive strategies win. — [DOI](https://doi.org/10.1145/121132.286599)
- **Krishnan, Long & Vitter.** Spin-down, thread blocking, and virtual-circuit holding times. — [Bell Labs abstract](https://www.nokia.com/bell-labs/publications-and-media/publications/adaptive-disk-spindown-via-optimal-rent-to-buy-in-probabilistic-environments/)
- **Antoniadis et al.** Learning-augmented multi-state power management. — [arXiv](https://arxiv.org/abs/2110.13116)
- **Li et al.** Strongly-optimal prediction-specific ski rental, with a power-management case study. — [arXiv](https://arxiv.org/abs/2510.14887)
- **Narayana & Kash (2023, 2026).** Keep-alive theory and market design for serverless keep-alive. — [arXiv](https://arxiv.org/abs/2309.03521); [arXiv](https://arxiv.org/abs/2603.03091)

### Inferences
- **Same conditioning as ours.** Power-management work conditions on elapsed idle time for the same reason we condition F on t: tool gaps are not memoryless.
- **Multi-tier extension.** The multi-state results suggest extending retention from host versus SSD to GPU/host/SSD with per-tier holding rates and wake-up costs.
- **Missing pieces.** None of these works has pre-arrival wake-up with a duration, or capacity coupling between devices.

### Gaps
- **Connection-timeout theory unverified.** Cohen, Kaplan & Zwick's connection caching and the virtual-circuit papers (see Q0 Gaps) could not be checked.
- **Partial arXiv search.** The arXiv-specific searches for "spin-down," "power management timeout" and "idle time hazard rate" were blocked by HTTP 429. OpenAlex and web search found no further arXiv follow-up.

## Q5. Distributional advice: which online algorithms take a full distribution, and which consistency and robustness results matter when the forecast may be wrong?

### Takeaway
- **Algorithms that take a full distribution:**
  - Diakonikolas et al. (ICML 2021), from samples.
  - Dinitz et al. (binary search, 2024).
  - Angelopoulos et al. (contract scheduling, IJCAI 2024; bidding, 2025).
  - Canonne et al. (matching, 2025).
  - Cui & Dinitz and Kim & Fan (ski rental, 2026).
  - Kang et al. (Bayesian ski rental).
  - Dinitz & Dong (tree search, 2026).
  - Related uncertainty forms: Sun et al. (UQ intervals) and Shen et al. (calibration).
- **Robustness results that matter for us:**
  - Additive loss that grows with the Wasserstein error and needs no knowledge of it (Cui & Dinitz).
  - Clamping the threshold to a safe range (Kim & Fan).
  - The impossibility of beating the worst case from finite samples for general distributions, with poly(1/ε) samples sufficient for log-concave distributions (Diakonikolas et al.).
  - Calibration helps under high variance (Shen et al.).

### Cited Findings
- **Cui & Dinitz.** Expected cost is at most OPT_p + O(min(max(η,1)√b, b log b)), with η the Earth Mover's distance; consistency O(√b), robustness O(b log b); η need not be known; the bounds are essentially tight. — [arXiv](https://arxiv.org/abs/2602.21104)
- **Kim & Fan.** The Clamp Policy is robust and consistent; a water-filling algorithm handles the randomized case. — [arXiv](https://arxiv.org/abs/2603.29233)
- **Diakonikolas et al.** Impossible for general distributions with finite samples; poly(1/ε) samples give a (1+ε) ratio for log-concave distributions. — [PMLR](https://proceedings.mlr.press/v139/diakonikolas21a.html)
- **Dinitz et al.** Binary search in O(H(p) + log η) queries, essentially optimal; any single point prediction can do poorly. — [arXiv](https://arxiv.org/abs/2411.16030)
- **Angelopoulos et al.** Optimal schedules when the prediction is accurate, with the best worst case when it is adversarial, plus robustness evidence in the distributional setting. — [arXiv](https://arxiv.org/abs/2404.12485)
- **Angelopoulos & Simon.** Pareto-optimal bidding under distributional predictions. — [arXiv](https://arxiv.org/abs/2510.25582)
- **Canonne et al.** Matching with distributional advice. — [arXiv](https://arxiv.org/abs/2505.04949)
- **Dinitz & Dong.** Tree search with distributional predictions. — [arXiv](https://arxiv.org/abs/2609.32260)
- **Sun et al.** UQ-augmented predictions for ski rental and online search. — [arXiv](https://arxiv.org/abs/2310.11558)
- **Shen et al.** Calibrated predictions for ski rental. — [arXiv](https://arxiv.org/abs/2502.02861)
- **Kodialam.** Soft (probabilistic) predictions. — [arXiv](https://arxiv.org/abs/1903.00092)
- **Kang et al.** Bayesian discrete ski rental. — [arXiv](https://arxiv.org/abs/2512.07313)
- **Khanafer et al.** Moment information. — [DOI](https://doi.org/10.1109/INFCOM.2013.6566944)
- **Mahdian et al.** An early framework for guarantees when side information is wrong. — [DOI](https://doi.org/10.1145/2071379.2071381)
- **Decision-theoretic metrics for imperfect oracles.** — [arXiv](https://arxiv.org/abs/2501.17701)

### Inferences
- **Our design trusts the forecast.** It acts on F(x|t) as if exact. Its only safeguard is a heuristic: a key whose requests repeatedly arrive beyond the 90th percentile receives a heavier tail.
- **Candidate safeguards from this literature:**
  - A clamp on start times and keep horizons (Kim & Fan).
  - A delayed-commit rule with bounded additive regret (Cui & Dinitz).
  - Calibration checks on the per-key tables (Shen et al.).
- **Caution about learning gap distributions.** Diakonikolas et al.'s impossibility for general distributions applies to learning gap distributions from samples. Agentic tool gaps are multi-modal and so not log-concave.
- **What none of these results covers.** Capacity coupling (a shared λ) and preparation times. Robustness of the coupled system to a miscalibrated forecast is an open question that our evaluation could address empirically.

### Gaps
- **Distributional advice in our setting.** I found no distributional-advice result for prefetch timing with a preparation duration, or for problems with a shared capacity or dual price.
- **Venues not verified.** Several venues for 2023–2026 arXiv papers are listed in Q0 Gaps.
