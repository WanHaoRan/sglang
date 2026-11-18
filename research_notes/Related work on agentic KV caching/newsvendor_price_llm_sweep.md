# Newsvendor, hazard-rate and price rules for KV-cache memory in LLM serving (2024 to Nov 2026): three flagged papers verified, plus a targeted sweep

Scope and method (2026-11-17).
- **Design shorthand.**
  - (b) start an SSD read once P(request arrives before the read completes | idle t) ≥ q = λL/(1+λL);
  - (c) keep an idle prefix cache in host memory while hazard × read time ≥ q, and evict by a Gittins-type index;
  - (d) λ prices host memory in TTFT seconds per token-second, is learned by multiplicative dual ascent on demand minus capacity, and gives a weak-duality lower bound.
- **How the three papers were read.** Full texts were downloaded from arxiv.org (HTML for 2607.16892 and 2608.30830, the PDF for 2610.28637) and converted to text locally. Quotes below are copied from those texts. Formulas are re-typed from the LaTeX alt-text in plain notation.
- **Sweep.** 30 arXiv listing searches (searchtype=all, which covers metadata and abstracts, not full text) and 5 general web searches. Each hit was checked against the arXiv IDs already in this folder (302 IDs).
- "Inference" marks my own reasoning, not the paper's.

## Objective: does any LLM-serving paper anticipate claims (b), (c) or (d)?

### Takeaway
- **(b), partly.** The critical-ratio algebra is classical, and an LLM-serving paper says so (Cheng et al., 2607.16892, for KV *reservation size*). One new find, Hermes (2506.14851, ACM TACO 2026), *times* a KV-cache prewarm with a probability threshold of the same form. Its threshold is a fixed knob (default 0.5), not a priced, size-dependent ratio.
- **(c), no.** No LLM-serving paper found uses a hazard-rate retention rule.
- **(d), partly.** One new find (2607.03948) learns *shadow prices on KV-cache memory online* by projected dual subgradient steps (capacity share minus demand at the current price). It does this for routing new requests, with GPU decode memory, additive steps and no bound. Adaptive KV Retention uses a configured, load-indexed mean-field price. KVLearn uses a formula-based per-byte-second cost and a PI-controlled admission threshold.
- **Net.** The novelty statements for (b) and (d) must be narrowed (see Inferences).

### Cited Findings
- **Cheng et al., reservation size.** The paper derives r* = L_in + F⁻¹(ρ/(ρ+1)) for per-class KV *reservation size* and calls it "the classical newsvendor critical ratio" (Prop. IV.2, Remark 3) — [arXiv HTML 2607.16892v1](https://arxiv.org/html/2607.16892v1)
- **Hermes, prewarm timing.** "Given the knob K, if p_s<K, then Hermes does not trigger prewarming; otherwise, it triggers prewarming at the time such that p_e=K", with p_e = p_s·P(t_c > t_s + t_p) (Sec. 3.4, Eq. (3)). The paper says this "also applies to the cache resources in LLM backends" — [arXiv HTML 2506.14851v1](https://arxiv.org/html/2506.14851v1)
- **Online LP routing, KV-memory price.** A dual variable is "the shadow price of resource type u on device g at time k", including "KV-cache memory (when u=mem)". It is updated online by warm-started projected subgradient steps (Sec. 3.2–3.3, Eqs. (6)–(8)) — [arXiv HTML 2607.03948v1](https://arxiv.org/html/2607.03948v1)
- **Adaptive KV Retention, host-slot price.** The price on host slots is α₂(λ) = max(0, λ − λ_crit)·β₃/C₂, with λ "the configured offered rate for the condition" (Sec. 3.3 Eq. (3); App. D) — [arXiv HTML 2608.30830v1](https://arxiv.org/html/2608.30830v1)
- **KVLearn, memory cost.** U(b, Δt) = γ·|b|·Δt with γ = R̄·λ/M_S. A PI controller adjusts the admission threshold θ, not γ (Sec. 3.2, 4.3, 4.4 Eq. (11)) — [arXiv PDF 2610.28637](https://arxiv.org/pdf/2610.28637)

### Inferences
- **Suggested wording, (b).**
  - Do not claim the critical ratio or "start a load when the probability of arriving before it completes crosses a threshold" as new.
  - Claim instead that the threshold is *derived* from priced earliness/tardiness costs, so q = λL/(1+λL) varies with prefix size and a learned memory price. It is applied to a per-tool idle-gap distribution conditioned on elapsed idle time, to time SSD→host reads of idle agent prefix caches.
  - Cite Cheng et al. for the critical-fractile structure and Hermes for fixed-threshold prewarm timing in LLM serving.
- **Suggested wording, (d).**
  - Do not claim "first online-learned shadow price on KV memory in LLM serving"; 2607.03948 has one.
  - Defensible claim: the first price on *host* memory held by *idle* prefix caches, in TTFT seconds per token-second, learned by *multiplicative* dual ascent whose demand includes refused and preempted reads, with a reported weak-duality lower bound.
  - None of 2607.03948, 2608.30830 or 2610.28637 has all of these.
- **Suggested wording, (c).** The LLM-serving literature found here has no hazard-rate keep rule. Hazard-rate keep rules are prior art in caching theory and serverless keep-alive (Ferragut et al., Carrasco et al., Narayana & Kash, already covered in this folder). So (c) should be credited to that theory, and its novelty placed in the read-time factor, the size- and price-dependent q, and the LLM tiering setting.

### Gaps
- Hermes' and 2607.03948's own related-work sections were not traced further.
- Full-text-only uses of these terms in papers whose abstracts omit them could be missed. arXiv listing search does not index full text, and web search is incomplete.

## arXiv 2607.16892: what does its critical fractile govern?

### Takeaway
- **Decision.** The fractile sets **how many KV tokens to reserve per request class at admission**, when output length is unknown. It is a quantity decision, not prefetch timing or retention.
- **Costs.** Underage is c_p per token of overflow (preemption and recomputation). Overage is c_w per token of over-reservation (lost serving capacity).
- **Derivation.** The ratio comes from a piecewise-linear cost model. ρ = c_p/c_w is operator-set, and no price is learned online.
- **Relation to our fetch rule.** It is the same newsvendor algebra applied to a different decision. It has no read time, no conditioning, no size-dependent price and no learning.

### Cited Findings
- **Record.** "Robust KV Cache Management for LLM Serving under Output Token Length Uncertainty", Jiaming Cheng, Duong The Do, Duong Tung Nguyen. v1 submitted 18 Jul 2026, cs.NI. Comments field: "10 figures, 10 pages"; no venue stated — [arXiv abs](https://arxiv.org/abs/2607.16892)
- **Decision variable (Sec. III-B).** "Reservation r_i ∈ [L_in^i, L_in^i + L_max^i]: KV cache tokens reserved per class-i request at admission, where L_max^i is the maximum output length." — [HTML](https://arxiv.org/html/2607.16892v1)
- **Underage cost (Sec. III-C, Eq. (2)).** C_preempt = c_p·(L_in^i + ξ_i − r_i)⁺, "which implies that larger under-reservation leads to more severe recomputation overhead." — [HTML](https://arxiv.org/html/2607.16892v1)
- **Overage cost (Sec. III-C, Eq. (3)).** "This opportunity cost is modeled as:" C_waste = c_w·(r_i − L_in^i − ξ_i)⁺, "reflecting the loss of serving capacity due to over-reservation." — [HTML](https://arxiv.org/html/2607.16892v1)
- **Ratio (Sec. III-C).** "The relative importance of these effects is governed by the cost ratio ρ=c_p/c_w, which encodes the asymmetry between under- and over-reservation and reflects the system's tolerance for preemption versus memory inefficiency." — [HTML](https://arxiv.org/html/2607.16892v1)
- **Rule (Sec. IV-B, Prop. IV.2, Eq. (9)).** "Consider the reservation subproblem for class i with fixed routing. Let ρ=c_p/c_w denote the cost ratio. The optimal reservation r_i^* satisfies: r_i^* = L_in^i + F_i⁻¹(ρ/(ρ+1)) (9) where F_i is the cumulative distribution function of output length ξ_i under the worst-case distribution in P_ε^i." — [HTML](https://arxiv.org/html/2607.16892v1)
- **Proof (Eqs. (10a)–(10b)).** ∂E[Q]/∂η = −c_p·Pr(ξ>η) + c_w·Pr(ξ≤η) = (c_p+c_w)F(η) − c_p. "Setting to zero yields F(η*)=c_p/(c_p+c_w)=ρ/(ρ+1)." — [HTML](https://arxiv.org/html/2607.16892v1)
- **Self-attribution (Remark 3).** "This is the classical newsvendor critical ratio, and its distributionally robust form is likewise known in operations research [14, 17]; we treat both as building blocks and locate our contribution in coupling the reservation to routing, configuration, and the shared-memory capacity constraint (6e)." The introduction says: "The single-request reservation reduces to the classical newsvendor problem [15, 16]". — [HTML](https://arxiv.org/html/2607.16892v1)
- **Costs are configured, not learned (Sec. V-A).**
  - "Default knobs. Two operator-set parameters drive BCD-DRO: the cost ratio ρ=c_p/c_w and the Wasserstein radius ε."
  - Defaults: "c_p/c_w=10". "c_p and c_w are per token (a preempted or wasted KV slot)."
  - Table IV sweeps ρ ∈ {2, 10, 100}. At ρ = 10, q* ≈ 91%.
  - — [HTML](https://arxiv.org/html/2607.16892v1)
- **The only "dual" is the Wasserstein multiplier (Remark 1, App. A).** "The dual variable γ_i acts as a transportation cost penalty: a large γ_i keeps the worst-case distribution close to empirical data". It comes from the Kantorovich-duality reformulation, not from a memory price — [HTML](https://arxiv.org/html/2607.16892v1)
- **Capacity enters as a hard constraint.**
  - Memory constraint (6e) is a Little's-law budget: "This is the key constraint linking reservation to throughput."
  - The paper states the reservation "depends only on ρ". At its operating point "neither the SLO penalty nor the memory-headroom margin is the binding constraint" (Sec. V-B, "Other cost parameters").
  - — [HTML](https://arxiv.org/html/2607.16892v1)
- **"Online" means re-solving with fixed costs.**
  - "The online phase is O(1) per request: simply sample the routing decision and apply the precomputed reservation" (Sec. IV-C).
  - "Rolling-horizon adaptation periodically re-optimizes using recent observations" (Sec. V-C).
  - The conclusion lists "online adaptation using streaming workload statistics" as future work.
  - Evaluation is simulated ("We simulate J=8 A100-80GB GPUs").
  - — [HTML](https://arxiv.org/html/2607.16892v1)

### Inferences
- **Exact correspondence to our fetch rule.**
  - Our rule minimizes E[1·(stall seconds) + λL·(early-staging seconds)] over the read start time. Its first-order condition is P(arrival before completion | t) = λL/(1+λL). So the on-time probability at the optimum is 1/(1+λL).
  - That equals Cheng's ρ/(ρ+1) with ρ = 1/(λL): c_p ↔ 1 TTFT-second per stall second, c_w ↔ λL per staging second. The algebra is identical.
- **What differs.**
  - Decision type: a quantity (tokens) vs a start time.
  - Read time: none in Cheng, vs our read time r inside the probability.
  - Distribution: per-class output length, unconditional and worst-case, vs a per-tool idle gap conditioned on elapsed idle time.
  - Overage cost: a constant per-token c_w set by the operator, vs a size-dependent λL whose λ is learned by dual ascent.
- **Where a price would appear.** Prop. IV.2 holds when (6e) is slack. If (6e) binds, its LP multiplier would add to c_w, giving a price-adjusted fractile. The paper neither derives nor reports this, which is exactly the coupling our λ makes explicit.
- **Correction to the existing entry** (prefetch_timing_rent_or_buy.md, A17). It is accurate ("critical-ratio sizing ... not timing"). Add three points:
  - the paper itself attributes the ratio to the classical newsvendor;
  - ρ is operator-set;
  - the capacity constraint does not enter the fractile.

### Gaps
- References [14, 17] (DRO newsvendor) and [15, 16] (newsvendor) were not opened.
- The appendix's full LP2 was not traced to confirm (6e) never binds in the reported runs.

## arXiv 2608.30830 (Adaptive KV Retention): price formula, expiry derivation, nature of the price

### Takeaway
- **Price.** The host-memory price is a **mean-field congestion price per suspended-context slot**: α₂(λ) = max(0, λ − λ_crit)·β₃/C₂ with λ_crit = C₂/W_ref. It is overflow recomputation cost amortized over slots.
- **Expiry.** t₂(λ) = (β₃ − β₂)/α₂(λ), and ∞ at or below λ_crit.
- **Not a multiplier, not learned.** It is not derived as a Lagrange multiplier, and it is computed from a **configured** offered load, not updated online.
- **No per-session forecasts.** The paper explicitly avoids per-request wait prediction and hazard rates. The TTL is one-shot and size-independent.

### Cited Findings
- **Record.** "Adaptive KV Retention for LLM Agents at Human-Approval Timescales", Minseo Choi, Ananya Joshi. v1 submitted 31 Aug 2026, cs.OS. No Comments/venue field on the abs page — [arXiv abs](https://arxiv.org/abs/2608.30830)
- **Cost model (Sec. 3.1, Eq. (1)).** "For state j∈{1=GPU HBM, 2=host DRAM, 3=discard}, let α_j denote its holding cost per second and β_j its one-time resume cost. If a request remains in state j for ℓ seconds, its cost is C_j(ℓ)=α_jℓ+β_j." Costs are in GPU-seconds ("GPU opportunity cost, measured in GPU-seconds") — [HTML](https://arxiv.org/html/2608.30830v1)
- **Uncongested vs congested (Sec. 3.1).**
  - "When host DRAM has spare capacity, α₂=α₃=0".
  - "Once host DRAM becomes scarce, retaining a host copy imposes an opportunity cost on other suspended contexts. We capture this with a nonzero, load-dependent holding price α₂(λ), where λ is the offered rate of suspended requests."
  - — [HTML](https://arxiv.org/html/2608.30830v1)
- **HBM break-evens (Sec. 3.1, Eq. (2)).** t₁ = β₂/α₁, t* = β₃/α₁, "where t₁ is the HBM–DRAM break-even and t* is the HBM–discard break-even." Sec. 3.4: "t₁≈1 s is short enough that we offload at suspension." — [HTML](https://arxiv.org/html/2608.30830v1)
- **Price derivation (Sec. 3.3), verbatim.** "Let C₂ denote the number of suspended contexts that fit in host DRAM and W_ref the reference residence scale. By Little's law, host saturation occurs at λ_crit=C₂/W_ref. Above this threshold, we use a mean-field congestion price. Excess arrivals occur at rate λ−λ_crit under the reference normalization, and each overflow incurs recomputation cost β₃. Amortizing this cost across the C₂ occupied host slots gives" — [HTML](https://arxiv.org/html/2608.30830v1)
- **Formula (Eq. (3)).** α₂(λ) = max(0, λ − λ_crit)·β₃/C₂; t₂(λ) = ∞ if λ ≤ λ_crit, else (β₃ − β₂)/α₂(λ). "This yields a load-indexed host expiration policy, cpu_ttl(λ)... Because the congestion price is a mean-field approximation rather than an exact occupancy model, we evaluate the resulting policy directly with finite-capacity replay" — [HTML](https://arxiv.org/html/2608.30830v1)
- **Configured, not online (App. D; Sec. 5 Limitations).**
  - "In our experiments, λ is the configured offered rate for the condition."
  - "Our experiments use the configured offered load λ, so a production deployment would require online load estimation and periodic recalibration under changing approval-wait distributions."
  - Sec. 3.4: "At runtime the controller requires only the offered load λ to evaluate t₂(λ)."
  - — [HTML](https://arxiv.org/html/2608.30830v1)
- **Branch chosen offline (Sec. 3.4).** "We replay the same calibration waits under both modes, compute their total GPU opportunity cost, and freeze the lower-cost branch for the held-out run." The crossover in their setup is at P(W ≤ t*) = 0.39 (Sec. 4.3) — [HTML](https://arxiv.org/html/2608.30830v1)
- **No per-session forecasts.**
  - "The TTL decision is one-shot... the request is not subsequently re-scored from elapsed wait" (App. D).
  - The policy works "without per-request wait prediction" (Sec. 5).
  - The paper's only age argument is informal (Sec. 1): "If we were to model this regime using memoryless waits, such as an exponential distribution, the elapsed time is uninformative and expiration only adds churn. For heterogeneous short- and long-waiting populations, age can help separate contexts likely to resume soon from those likely to remain suspended."
  - — [HTML](https://arxiv.org/html/2608.30830v1)
- **Calibrated constants (App. B).** α₁ = 0.0176 GPU-s/s, β₂ = 0.02 GPU-s, β₃ = 1.91 GPU-s. C₂ = 425 slots in the cross-platform replay (App. C) — [HTML](https://arxiv.org/html/2608.30830v1)

### Inferences
- **The expiry is a break-even.** t₂ solves α₂·t₂ + β₂ = β₃: the DRAM holding-plus-restore cost equals the discard cost, since α₃ = 0. This parallels Eq. (2); the paper calls t₂ the "DRAM–discard threshold" without writing out the break-even.
- **Units, and why it is not a multiplier.**
  - α₂ has units of GPU-seconds per slot-second and is per *context*, so it does not scale with prefix size.
  - It is a fluid, Little's-law estimate of the marginal cost of a slot that uses the reference residence W_ref, not the actual wait distribution.
  - It is not a dual variable of a capacity constraint, and nothing is learned.
  - Compare our λ: per token-second, in TTFT seconds, learned by multiplicative dual ascent on demand minus capacity, with a weak-duality bound.
- **Hazard argument stated in words only.** The memoryless remark is an informal statement of the hazard argument (constant hazard ⇒ age-based expiry is useless; mixtures ⇒ decreasing hazard). The paper never computes a hazard or a per-session rule.
- **Existing entry is accurate** (agentic_tool_call_kv.md, Rank 5). The "about 1.1 s" HBM→DRAM break-even matches β₂/α₁ = 0.02/0.0176 ≈ 1.14 s from App. B; the main text rounds it to "≈ 1 s". Add two clarifications:
  - α₂ is per context slot, not per token;
  - the expiry is a one-shot break-even TTL from a mean-field price, not a multiplier.

### Gaps
- The NeurIPS 2026 listing for this paper noted in the earlier entry remains unverified. The arXiv abs page shows no venue.

## arXiv 2610.28637 (KVLearn): exact score, nature of the memory cost, duality

### Takeaway
- **Score.** CARS weights recompute and transfer costs by a predicted horizon-reuse probability and subtracts a byte-time cost U = γ·|b|·Δt_est.
- **Internal inconsistency.** The typeset Eq. (10), and Eq. (4), read P̂·R − T − U. Algorithm 1, Eq. (5) and the prose read P̂·[R − T] − U.
- **γ is not the controller's output.** It is a formula from running averages (R̄·λ/M_S). The PI controller outputs an additive, size-independent admission threshold θ that tracks utilization 0.85.
- **No duality.** There is no Lagrangian, multiplier or duality bound. The only upper bound is an empirical oracle.

### Cited Findings
- **Record.**
  - "To Keep or Not to Keep: Learning KV Cache Retention in Disaggregated LLM Serving Systems", Dong Liu, Yanxuan Yu, Eric Jiang, Shu Wang, Ying Nian Wu. Subjects cs.LG and cs.DC.
  - Comments: "Accepted to ACM SYSTOR 2026"; the page header reads "SYSTOR '26, September 8–9, 2026, Virtual Event, Israel".
  - The abs page says "[Submitted on 10 Aug 2026]", v1 only, although the ID is 2610.
  - — [arXiv abs](https://arxiv.org/abs/2610.28637); [PDF](https://arxiv.org/pdf/2610.28637)
- **Memory cost (Sec. 3.2).** "Storage cost U(b, Δt): opportunity cost of occupying |b| bytes of pool capacity for duration Δt. We model this as U(b, Δt) = γ·|b|·Δt, where γ is a scalar reflecting how many additional requests could have been served had that memory been freed." — [PDF](https://arxiv.org/pdf/2610.28637)
- **Value (Sec. 3.3, Eq. (3)).** V(b) = P_H(b)·C_miss(b) − C_hit(b) − C_admit(b) − C_storage(b), with "P_H(b) = P(reuse within H)" over "a finite lookahead horizon H steps" — [PDF](https://arxiv.org/pdf/2610.28637)
- **Instantiation (Sec. 3.3, Eqs. (4)–(5)).**
  - With C_miss = R, C_hit = T, C_admit ≈ 0 and C_storage = U: "V(b) = P_H(b)·R(b) − T(b) − U(b, Δt). (4)".
  - "The optimal keep decision is KEEP(b) ⟺ V(b) > 0, which gives a per-block admission threshold P*_H(b) = U(b, Δt) / (R(b) − T(b)). (5)".
  - — [PDF](https://arxiv.org/pdf/2610.28637)
- **Score (Sec. 4.3, Eq. (10)), as typeset.** "CARS(b) = P̂(b)·R(b) − T(b) − U(b, Δt_est). (10)" — [PDF](https://arxiv.org/pdf/2610.28637)
- **Score (Algorithm 1, line 3).** "CARS(b) ← P̂(b)·[R(b) − T(b)] − U(b, Δt_est)". The prose agrees with the bracketed form: "The factor R(b) − T(b) is the net benefit per cache hit". Admission: "A block is admitted if CARS(b) > θ... At θ = 0 with perfect P̂(b), this recovers the optimal threshold in Eq. (5)." — [PDF](https://arxiv.org/pdf/2610.28637)
- **γ and Δt (Sec. 4.3).**
  - "γ is derived from the pool capacity target: at ρ* = 0.85, we set γ = R̄·λ/M_S, where R̄ is the running average recompute cost, λ is the current block arrival rate (blocks/s), and M_S is pool capacity in bytes."
  - "Δt_est is estimated as the exponentially weighted average inter-eviction interval for same-modality blocks in a 30-second sliding window; ATC (§4.4) keeps γΔt_est consistent with the admission threshold via the pool-pressure feedback loop."
  - — [PDF](https://arxiv.org/pdf/2610.28637)
- **Controller (Sec. 4.4, Eq. (11)).**
  - "We use a lightweight PI controller: e(t) = ρ_pool(t) − ρ*, θ(t+1) = θ(t) + K_P·e(t) + K_I·Σ_{j≤t} e(j)".
  - "ρ* is the target utilization (default 0.85). The controller raises θ when the pool is too full... We additionally clamp θ ∈ [−C_max, C_max] with C_max = max_b |CARS(b)|".
  - Hit-rate bias: "Δθ_hr = K_hr(h_floor − h(t))".
  - Table 6 gives (ρ*, K_P, K_I) = (0.85, 2×10⁻², 5×10⁻⁴).
  - — [PDF](https://arxiv.org/pdf/2610.28637)
- **Eviction (Sec. 4.3).** "blocks leave in ascending CARS order—removing the lowest-expected-value block first" — [PDF](https://arxiv.org/pdf/2610.28637)
- **Only an empirical bound (Sec. 5).** The Oracle "admits a block iff it will actually be reused within horizon H=1000 requests; provides an upper bound." No Lagrangian, dual or weak-duality statement appears in the text (my keyword search of the full text found none) — [PDF](https://arxiv.org/pdf/2610.28637)

### Inferences
- **The two score forms give different thresholds.**
  - Eq. (3) with C_hit = T unweighted by P_H yields the typeset Eq. (4)/(10), whose threshold would be (T + U)/R.
  - The paper's Eq. (5), Algorithm 1 and prose instead use P̂·[R − T] − U, with threshold U/(R − T).
  - The bracketed form is the one implemented and analyzed (Prop. 1 uses Eq. (5)).
- **Is the memory cost a learned price?**
  - γ is not the controller's output. It is a mean-field estimate recomputed from running averages: recompute cost × block arrival rate / capacity. It has the same shape as Adaptive KV Retention's α₂ but without the λ_crit offset. Despite the wording "derived from the pool capacity target", ρ* does not appear in the formula.
  - θ is the learned quantity. It is an additive, per-block offset in latency units, not a per-byte-second price. Its update θ ← θ + K_P(ρ_pool − ρ*) + ... is algebraically like an additive dual-ascent step on a utilization constraint ρ_pool ≤ ρ*, but the paper does not interpret it as a multiplier.
  - The constraint is a utilization target, not demand vs capacity, and there is no bound.
- **Horizon probability, not a hazard.** P_H is a reuse probability within H requests. There is no time-to-reuse distribution, hazard or read-time factor.
- **Corrections to the existing entry** (multi_tier_kv_cache.md, R5 and Q4):
  - (1) Quote both forms. The note gives the Algorithm-1 form as the definition; Eq. (10) as typeset differs.
  - (2) "γ is a fixed heuristic" should read: "γ is a formula-based estimate, R̄·λ/M_S, updated from running averages. It is neither fixed nor the controller output. The PI controller adjusts θ."
  - (3) Date: "Submitted on 10 Aug 2026" under a 2610 ID. Inference: arXiv assigns IDs at announcement, so this suggests an October announcement.

### Gaps
- Appendix A (proof of Prop. 1) and Sec. 5.7 (cost-model sensitivity) were not read in full.
- Whether the released code uses the bracketed or unbracketed score was not checked.

## Targeted sweep: new LLM-serving papers using newsvendor, price/dual or hazard/TTL rules

### Takeaway
- **Two new, relevant papers.**
  - **Hermes** (2506.14851, ACM TACO 2026): a fixed-probability-threshold rule for *when* to prewarm, including KV cache, from a completion-time distribution. Relevant to (b).
  - **Online LP routing** (2607.03948, Jul 2026): shadow prices on per-device KV-cache memory learned online by projected dual subgradient steps. Relevant to (d).
- **Checked and not matching.** Six further new hits do not use the target rules on KV memory: HAServe, LYREO, AsymCache, SARA, the ICLR 2026 eviction/routing paper, and three compute-side "shadow price" papers.
- **No hazard-rate, survival-analysis, ski-rental or newsvendor-timing rule** for KV memory turned up in any 2024–2026 LLM-serving abstract.

### Cited Findings

New paper 1: Hermes (relevant to (b)).
- **Record.**
  - "Efficient Serving of LLM Applications with Probabilistic Demand Modeling", Yifei Liu, Zuo Gan, Zhenghao Gan, et al. (11 authors). Submitted 17 Jun 2025, cs.DC.
  - Journal-ref: "ACM Transactions on Architecture and Code Optimization 23, 2, Article (June 2026), 1-25".
  - Not in this folder's notes (grep for the ID, "Hermes" and "PDGraph" found nothing).
  - — [arXiv abs](https://arxiv.org/abs/2506.14851)
- **Rule (Sec. 3.4, Eq. (3)).**
  - "We let p_s be the probability of selecting the target unit, t_c be the actual completion time of the current unit, t_s be the moment to start prewarming, and t_p be the duration of the prewarming operation. Then the probability that prewarming is effective (i.e., when the target downstream request finally arrives, its backend is already well-warmed) can be expressed by p_e = p_s * P(t_c > t_s + t_p). (3)"
  - "Given the knob K, if p_s<K, then Hermes does not trigger prewarming; otherwise, it triggers prewarming at the time such that p_e=K. Such a knob K can be deemed as a kind of service level agreement".
  - — [HTML](https://arxiv.org/html/2506.14851v1)
- **Why the trade-off exists (Sec. 3.4).** "a functional unit may have multiple downstream units, and it is possible that the prewarmed backends are never used; on the other hand, the current unit may complete much later than expected, meaning that the prewarmed backend would wait idly for a long time." — [HTML](https://arxiv.org/html/2506.14851v1)
- **Applies to KV cache.** "The above prewarming technique applies to the non-LLM backends like docker and DNNs, and also applies to the cache resources in LLM backends... preloading the desired KV cache or LoRA is also necessary". In Sec. 5.4, the KV-cache hit ratio at 8/16/32 GB is compared against LRU and EPWQ ("up to 1.11× (0.33×)"). The default is K = 0.5 — [HTML](https://arxiv.org/html/2506.14851v1)
- **Fixed knob, conditioned distribution.**
  - K is a fixed knob; Fig. 14's caption says "A lower K value indicates more aggressive prewarming".
  - The demand estimate is refreshed at runtime: the scheduler "keep[s] refreshing the demand estimation (at per-bucket granularity) based on the latest execution status".
  - Scheduling uses a Gittins index "based on its executed time so far and its size distribution".
  - — [HTML](https://arxiv.org/html/2506.14851v1)

New paper 2: online LP routing with KV-cache shadow prices (relevant to (d)).
- **Record.** "Online Linear Programming for Multi-Objective Routing in LLM Serving", Zixi Chen, Yinyu Ye, Zijie Zhou. v1 submitted 4 Jul 2026; cs.AI, cs.LG, math.OC; no venue stated. Not in this folder's notes — [arXiv abs](https://arxiv.org/abs/2607.03948)
- **Constraint (Sec. 2).** "KV-cache capacity: at any period, the total KV-cache usage of all active requests on a decode device cannot exceed a budget M; otherwise, the system must swap requests out and stall compute." — [HTML](https://arxiv.org/html/2607.03948v1)
- **Price (Sec. 3.2).** "the dual variable p_i represents the shadow price of resource type u on device g at time k: it quantifies the marginal value of one additional unit of batch capacity (when u=bat) or KV-cache memory (when u=mem). High shadow prices indicate anticipated congestion on that resource". — [HTML](https://arxiv.org/html/2607.03948v1)
- **Decision rule (Sec. 3.2).**
  - Margin: "Δ_j(g) := r_{j,g,k} − a_{(j,g,k)}ᵀp".
  - "Given the price vector p_k learned from history, a natural bid-price rule is to accept a request j if its best margin... is positive."
  - In practice: "(i) compute bid price margins for all candidates, (ii) rank candidates by margin, and (iii) greedily admit the highest-margin actions subject to the residual constraints".
  - — [HTML](https://arxiv.org/html/2607.03948v1)
- **Online price learning (Sec. 3.2–3.3, Eqs. (6)–(8)).**
  - SAA dual: f_k(p) = d_kᵀp + (1/|Hist|)·Σ_{(r*,a*)∈Hist} (r* − a*ᵀp)⁺, with d_k := b^(k−1)/max{1, n̂_remain(k)}.
  - Update: "p^(s+1) ← Π_{ℝ₊}(p^(s) − η_k·ĝ_k(p^(s)))" with ĝ_k(p) := d_k − (1/|B_k|)·Σ_{(r,a)∈B_k} 1{r − aᵀp > 0}·a.
  - "we initialize p^(0) ← p_{k−1} and perform a small fixed number K of projected subgradient steps".
  - — [HTML](https://arxiv.org/html/2607.03948v1)
- **Scope (Sec. 5–6).**
  - "In decode-heavy scenarios with long output sequences, memory prices dominate".
  - "Our validation relies on simulation using the Vidur framework".
  - The full text contains no weak-duality or lower/upper-bound result (keyword search for "bound", "weak", "hindsight" found only reference titles and SLO bounds).
  - — [HTML](https://arxiv.org/html/2607.03948v1)

Checked, not a match (all new to this folder).
- **HAServe** (2605.08581; Xingyu Qu et al.; v2 13 Oct 2026; "25 pages, 9 figures, Preprint").
  - Chooses "the feasible joint admission and KV-retention candidate with the lowest predicted latency cost".
  - Its "prefix survival" means a cached prefix lasting until use ("how long a cached prefix must survive before use"), not survival analysis.
  - The full text has no price, multiplier or hazard.
  - — [abs](https://arxiv.org/abs/2605.08581); [HTML v2](https://arxiv.org/html/2605.08581v2)
- **LYREO** (2609.17193; Zhen Li et al.; 15 Sep 2026; edge request scheduling).
  - Defines workload as "normalized KV cache memory-time consumption".
  - Its Lyapunov "virtual queue for the workload deviation" tracks Y_s(t) = (η_s(t) − η̄(t))² above a tolerance ε, i.e. load *imbalance*.
  - GPU memory capacity is handled by a fixed penalty: "Υ(t)=Σ_s[M_s^pk(t)−M_s]⁺ is a penalty term enforcing the GPU memory constraint, and ϱ>0 is the corresponding penalty coefficient".
  - — [abs](https://arxiv.org/abs/2609.17193); [HTML](https://arxiv.org/html/2609.17193v1)
- **AsymCache** (2606.02964; Chunan Shi et al.; 1 Jun 2026).
  - Evicts by "argmin_B E(B,t) = argmin_B f_B(t)·ΔT_B", with f_B a piecewise-exponential recency weight.
  - Its online "coefficient λ" rescales f_B to shift "the effective turning point". It is not a price or multiplier.
  - — [abs](https://arxiv.org/abs/2606.02964); [HTML](https://arxiv.org/html/2606.02964v1)
- **Randomization Boosts KV Caching, Learning Balances Query Load** (2601.18999; Fangzhou Wu et al.; ICLR 2026).
  - Abstract: "provably competitive randomized KV cache eviction with learning-based methods to adaptively route queries". No price or hazard is mentioned (abstract only).
  - — [abs](https://arxiv.org/abs/2601.18999)
- **SARA** (2609.26763; Shicong Liu et al.; 22 Sep 2026).
  - "maximizes goodput under a deployment cost constraint and a series of quantile-based SLO constraints" using queueing models. This is resource provisioning, not KV timing or retention (abstract only).
  - — [abs](https://arxiv.org/abs/2609.26763)
- **"Shadow price" papers on non-KV resources** (abstract only):
  - 2608.23986: "the dual of the capacity constraint gives the marginal cost of one more query by class and time" (compute capacity) — [abs](https://arxiv.org/abs/2608.23986).
  - 2609.00710: "evaluates products using resource shadow prices" for LLM API pricing and admission — [abs](https://arxiv.org/abs/2609.00710).
  - 2606.03092: shadow price of reasoning budgets — [listing](https://arxiv.org/search/?query=%22shadow+price%22+LLM&searchtype=all&abstracts=show&order=-announced_date_first&size=50).

Sweep log (arXiv listing search, searchtype=all, 2026-11-17).
- **Zero results:**
  - newsvendor inference serving;
  - "dual ascent" LLM serving; "Lagrange multiplier" KV cache; "dual decomposition" LLM inference; primal-dual KV cache; "Lagrangian relaxation" LLM serving; "dual variable" KV cache;
  - "hazard rate" LLM; "ski rental" LLM; "opportunity cost" KV cache;
  - "virtual queue" LLM serving; "chance constraint" KV cache; "critical ratio" cache; "memory price" LLM serving.
  - Example URL: [critical ratio cache](https://arxiv.org/search/?query=%22critical+ratio%22+cache&searchtype=all&abstracts=show&order=-announced_date_first&size=50)
- **Results with no new relevant paper:**
  - [newsvendor LLM](https://arxiv.org/search/?query=newsvendor+LLM&searchtype=all&abstracts=show&order=-announced_date_first&size=50): 3, all LLM-as-decision-maker papers.
  - ["critical fractile"](https://arxiv.org/search/?query=%22critical+fractile%22&searchtype=all&abstracts=show&order=-announced_date_first&size=50): 4; only 2607.16892 is LLM serving.
  - "time-to-live" KV cache: 1, Continuum (already covered).
  - rent-or-buy KV cache: 1, unrelated.
  - "survival analysis" LLM serving: 3, clinical and measurement papers.
  - "congestion price" LLM: 4, transport.
  - "bid price" LLM: 3, unrelated.
  - survival KV cache: 35, mostly "surviving tokens"; HAServe was checked.
  - Lagrangian KV cache: 2, compression bit allocation and routing.
- **Searches that produced the new hits:**
  - ["shadow price" cache](https://arxiv.org/search/?query=%22shadow+price%22+cache&searchtype=all&abstracts=show&order=-announced_date_first&size=50) → 2607.03948.
  - [Gittins LLM](https://arxiv.org/search/?query=Gittins+LLM&searchtype=all&abstracts=show&order=-announced_date_first&size=50) → 2506.14851.
  - [Lyapunov KV cache](https://arxiv.org/search/?query=Lyapunov+KV+cache&searchtype=all&abstracts=show&order=-announced_date_first&size=50) → 2609.17193.
- **General web searches** ("KV cache" with "shadow price", "ski rental", "hazard rate", "newsvendor" and "Lagrange multiplier") returned only covered IDs plus AsymCache and 2601.18999, both checked above.

### Inferences
- **Hermes maps exactly onto our trigger** (with p_s = 1). It starts at the t_s where P(t_c ≤ t_s + t_p) = 1 − K, i.e. when the probability that the request arrives before the warm-up completes reaches 1 − K. That is our trigger with q = 1 − K.
  - Default K = 0.5 means a fixed q = 0.5 for every load.
  - K is an SLA knob, not derived from costs. It does not depend on size or memory price, and there is no capacity coupling.
  - The prewarm is a model, LoRA or KV load before a downstream call in an application DAG, not an SSD read of an idle agent prefix cache.
  - This is the strongest LLM-serving precedent for the *form* of (b), closer than the TempoKV/TokenCake lead-time rules cited in multi_tier_kv_cache.md.
- **Hermes' K semantics are ambiguous.**
  - Under Eq. (3), p_e is non-increasing in t_s. If t_s is an absolute start time, a lower K starts later.
  - Yet the paper says a lower K is "more aggressive" (more prewarming, more waste). That is consistent only with the p_s < K gate admitting more branches, or with a different reference point for t_s.
  - Cite the rule verbatim and do not characterize its monotonicity.
- **2607.03948 is the closest LLM-serving precedent for (d).**
  - Its subgradient is "capacity share minus the resource use of candidates that clear the current price", so prices rise when demand exceeds capacity. This is dual (sub)gradient pricing of KV memory, learned online.
  - Differences from ours:
    - decision: routing/admission of *new* requests to decode workers, not retention or fetch timing for *idle* sessions;
    - memory: GPU decode KV, not host DRAM, with no SSD tier;
    - prices: indexed by (device, period), in SLO-reward units, not one host price in TTFT seconds per token-second;
    - update: additive projected mini-batch steps on an SAA dual over past candidates, not multiplicative steps on observed demand (with refused/preempted reads) minus capacity;
    - bound: no weak-duality lower bound reported;
    - evaluation: simulation only.
- **Corrections to existing "no match" claims.**
  - **(d):** "no serving system ... learns it by dual ascent on demand minus capacity" (multi_tier_kv_cache.md Q4) and "Price on host memory learned by dual ascent: no match found" should be narrowed. Online dual pricing of KV memory exists (2607.03948), though not for host memory, idle sessions, multiplicative updates or bounds.
  - **(b):** "(b) the newsvendor fetch threshold: no prior art found" (agentic_tool_call_kv.md) should note that Hermes times KV prewarm with a fixed probability threshold of the same form, not derived from a newsvendor cost ratio.

### Gaps
- Abstract-only items (2601.18999, 2609.26763, 2608.23986, 2609.00710, 2606.03092) were not read in full. Their non-match is judged from abstracts.
- Hermes' TACO version may differ from arXiv v1; only v1 was read.
- 2607.03948 states no venue. Whether a later version adds regret or duality bounds is unknown (v1 only).
- arXiv listing search does not index full text, so papers that use these rules only in their body could be missed. General web search is an incomplete backstop.
- Non-arXiv venues (OSDI/SOSP/NSDI/EuroSys papers without an arXiv version) were not searched.
- **Unverified leads:** none new. The only open lead is the earlier entry's NeurIPS 2026 page for 2608.30830 (not opened).
