math_retain: checks of DESIGN.md section 4 (Retain, the write, the tiers above and below) and
results 2 and 3 of section 7. CPU only; nothing here touches the engine, the GPU or /mnt/ssd.

Run any script with
  CUDA_VISIBLE_DEVICES= nice -n 10 /home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B <script>
(numpy, scipy). Each script prints to stdout; the matching .out file is the output of 2026-11-26.

hold_index.py     The holding problem as optimal stopping. Exact checks on 12 discrete laws x 37 prices:
                  keep is optimal iff nu >= lam; the first age with nu <= lam is the earliest optimal drop
                  age; backward induction gives the same value and stopping set; drop sets are nested in
                  lam; index by the lower convex hull of (E[min(G,t)], P(G>t)) equals the O(n^2) definition.
                  Monte Carlo of the rule's cost. Closed forms: Weibull, Lomax, lognormal (hazard peak and
                  tangent point), exact hint, sharp hint with an end probability.
hard_capacity.py  What a hard capacity costs: two counterexamples; offline brute force (index order = Belady,
                  optimal for equal sizes and costs, not otherwise); LRU / pooled index / class-aware index /
                  clairvoyant order for 8..2048 sessions against the priced (Lagrangian) lower bound.
queued_staged.py  One index for idle prefixes, staged reads and queued turns:
                  nu = (K0/L)/(g + w - rho); brute-force check and numbers.
refetch.py        Retain when the dropped prefix can be read back before the return (K replaced by the
                  optimal Fetch cost): equal to the simple index for falling hazards, far below it for
                  sharp hints; the keep set stays an interval in the price.
write_model.py    The write: three-state backward induction against the "two lives" theorem; value of a
                  write as an integral of pay probabilities; cost of missing the due date; life of a page
                  over (lam, mu, kappa); idle/busy device; many gaps (lazy against eager, share ever written).
write_order.py    Order of writes on one device: EDD for equal-size pages with one convex lateness shape
                  (permutation brute force), Smith's ratio for overdue pages, crossing-index counterexample,
                  stochastic check with returns that cancel a write.
ssd_chain.py      SSD capacity with a first-miss lookup: tail-first deletion (all subsets), the case of a
                  resident head, the tree index theta*rho(v), the shared head's marginal value.
extra_checks.py   Rising hazard plus an end probability; the two-piece lognormal entries; the restore's own
                  hold in the index; share of pages ever due (Monte Carlo); write against read priority.
numbers_c34.py    Arithmetic on the campaign 34 constants: restore costs per token, the price read off the
                  eviction frontier, keep times, the mirror's rent, share of pages ever due, a fluid balance.
