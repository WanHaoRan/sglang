# Proofs for "Four actions, three prices" (2026-11-26)

Statements, proofs and numerical checks behind `DESIGN.md`. Three independent derivations were made on 26 November
(Fetch; Retain and the write; order, Preempt, Price and the queue), each against the first draft of the design
(`DESIGN_v1_draft.md`, whose section numbers the "checks" lines refer to), each with its own numerics; a referee then
re-derived every result (part D) and wrote the final statements (part 0), which are the ones to cite. Scripts and outputs: `agents/math_fetch/`, `agents/math_retain/`,
`agents/math_order_price/`, `agents/verify_math/`. Notation is defined inside each statement; `DESIGN.md` appendix A
lists the symbols the design uses.

Status of a result: "proved"; "proved, with the assumptions stated"; "the first draft's claim was false; the
corrected statement is given"; "open".

Where each part is used: A in `DESIGN.md` sections 3 and 4; B in section 5; C in sections 2, 4.3, 6, 7 and 9.


## 0. Final statements (after the referee)

Twenty-one theorems and two propositions, each self-contained (Proposition 23 was added on 2026-11-27, after the
referee's pass; it is elementary and checked by `main/checks/prop23_check.py` and against the logs by
`main/checks/pass_model.py`). They replace the statements of parts A to C where the
two differ; the proofs are in parts A to C under the result names given in part D.

**Theorem 1 (Fetch: the cost of a start time).** A session's turn has ended; R is the remaining time until its next
request, S(t) = P(R > t), F = 1 - S, and M(t) = E[(R - t)^+] is the mean overshoot beyond t. Its prefix of L tokens is
on the SSD only. A read needs r seconds from release until its pages are usable; a turn whose prefix is resident waits
W seconds for the GPU; r and W are known constants and x = r - W > 0 is the part of the read the queue does not hide.
Policy 'start s': release the read at time s if the request has not arrived, otherwise at the arrival; the landed copy
stays until the turn is admitted. A second of stall (turn due, read not landed) costs c >= 0, a second of holding the
landed L tokens costs lambda L >= 0, and each read costs a fixed kappa L / beta. If E[R] is finite, the expected cost
is J_F(s) = c x - c M(s) + (c + lambda L) M(s + x) + kappa L / beta, with J_F(arrival) = c x + kappa L / beta; J_F is
continuous and its right derivative is S(s) [ (c + lambda L) F(x|s) - lambda L ], where F(x|s) = P(R <= s + x | R > s)
is the chance that a read started at s stalls its turn. So J_F falls while F(x|s) < q and rises while F(x|s) > q, with
q = lambda L / (c + lambda L): q is the stall risk the read accepts, not its complement. If E[R] is infinite (the
session may never return, or the tail is heavier than 1/t) and lambda > 0, J_F is infinite at every finite start: a
copy staged before arrival then needs a drop time (Theorem 3).

**Theorem 2 (When the threshold rule is optimal).** Setting of Theorem 1 with E[R] finite. Let tau be the first time
at which F(x|s) >= q (tau is the arrival if this never happens). (a) A start just after which F(x|s) < q is never
optimal. (b) 'Start at tau' is optimal if and only if the integral from tau to s of S(u) (F(x|u) - q) du is >= 0 for
every s > tau, the arrival included. A single crossing of q from below is sufficient; it is not necessary. (c) If
F(x|s) is continuous in s, the rule is optimal for every price q in (0, 1) if and only if F(x|s) never falls as s
grows; this holds for every x exactly when the gap has a rising hazard (-ln S convex), and a chance of never returning
rules it out.

**Theorem 3 (Falling hazard; a read with a drop time).** Setting of Theorem 1. (a) If F(x|s) never rises in s (true
when the hazard of R never rises) and E[R] is finite, J_F has no minimum strictly between now and the arrival: read
now if c E[min(R, x)] >= lambda L E[(R - x)^+], that is E[min(R, x)] / E[R] >= q, and otherwise wait for the arrival;
with prices fixed, once reading now stops winning as the session ages it never wins again. (b) Any law, a chance of
never returning allowed. Let the landed copy be dropped at time theta >= s + r if the session has not returned (a
later return reads at arrival), and let K_io = kappa L / beta + lambda L r_m be the fixed cost of one read, r_m being
the seconds the read holds its pages before landing. The expected gain over reading at arrival is G(s, theta) = c *
integral_s^{s+x} [S(v) - S(theta)] dv - lambda L * [ integral_{s+x}^{theta} S(v) dv - W S(theta) ] - K_io S(theta).
Its derivative in s is minus the derivative of Theorem 1; its derivative in theta is S(theta) [K_net h(theta) - lambda
L], with h the hazard of R and K_net = c x + K_io - lambda L W. A read before arrival pays if and only if G(s, theta)
> 0 for some s and theta.

**Theorem 4 (Uncertain queue wait).** Setting of Theorem 1 with W random and independent of R, r constant. Let Phi(s)
= P(R + W <= s + r | R > s): given that the session is still away at s, the chance that its turn would be admitted
before a read released now lands. (a) If nothing about W is learned when the request arrives, dJ/ds = S(s) [ (c +
lambda L) Phi(s) - lambda L ]; if P(W <= r) < q nothing is read before arrival, and a read released after arrival is
delayed so that it lands at the q-quantile of the admission time. (b) If W is observed at arrival, a read released
then lands exactly at admission, and dJ/ds = S(s) [ (c + lambda L) Phi(s) - lambda L (1 + h(s) E[(W - r)^+]) ], h
being the hazard of R: a read before arrival needs Phi(s) >= q (1 + h(s) E[(W - r)^+]). In both cases P(W <= r) < q is
sufficient for 'no read before arrival' and not necessary.

**Theorem 5 (After arrival: where the read lands).** The turn has arrived; D is the time at which it would be admitted
if its prefix were resident, with distribution function F_D as known now; the read time r is known; lateness costs c
per second and earliness lambda L per second, both linear. The cost c E[(l - D)^+] + lambda L E[(D - l)^+] is convex
in the landing time l and is smallest at l* = max(now + r, Q_D(q)), Q_D(q) being the q-quantile of D and q = lambda L
/ (c + lambda L); when the constraint does not bind, P(D <= l*) = q, so q is the chance that the turn is due before
its read lands. If the read time is uncertain, the best release time is max(now, the q-quantile of D - r), whatever
the dependence between D and r. For c = 0 every landing after the latest possible D costs the same, so the rule needs
a floor on c. If a read cannot be re-timed once released and the estimate of D keeps changing, the first time at which
P(D <= t + r | what is known at t) >= q is never later than the best release time, and is the best one if that
probability then stays above q.

**Theorem 6 (A read before arrival against a copy that never left).** Symbols of Theorems 1 and 3; any law; drop time
theta. Measured against 'no copy from age s on, read at arrival', keeping a copy from s to theta gains G_keep(s,
theta) = integral_s^theta S(u) [K_net h(u) - lambda L] du, and G_keep(s, theta) - G(s, theta) = S(s) { (c + lambda L)
E[(x - R_s)^+] + kappa L / beta - lambda L (r - r_m) } for every theta, R_s being the remaining gap of a session still
away at s. Hence: (i) not evicting and reading back at the turn's end are not the same; they differ by this margin.
(ii) If the margin is >= 0, which holds when a read is charged rent for its whole duration (r_m = r) or when kappa /
beta >= lambda (r - r_m), a read before arrival pays only where keeping a copy would pay; with a hazard that never
rises and unchanged prices, a copy that Retain dropped is then never worth reading back before arrival. (iii) If reads
wait in the device's line without holding pages and device time is cheap, the margin can be negative and a read
released before arrival can pay even on such a key. (iv) After a fall of the price or a forced eviction, a missing
copy that Retain would keep is read back at once.

**Theorem 7 (Retain: the holding index).** A prefix of L tokens of an idle session is in host memory. G is the gap (G
= infinity: the session never returns), S(t) = P(G > t), I(t) = integral_0^t S. At idle age a let F(h|a) = P(G <= a +
h | G > a) and tau(h|a) = E[min(G - a, h) | G > a]. Holding costs lambda L per second; if the prefix is dropped before
the return, the return pays K >= 0, where K depends neither on lambda nor on the drop age; the only information over
time is that the session is not back; a dropped prefix is not read back before the return. Let nu(a) = (K / L) * sup
over h > 0 of F(h|a) / tau(h|a). Then (i) dropping at age a is optimal if and only if nu(a) <= lambda, and keeping is
strictly better if and only if nu(a) > lambda; (ii) the first age at which nu <= lambda is the earliest optimal drop
age, and the same test is optimal at every later age however that age was reached; (iii) the set of ages at which
dropping is optimal grows with lambda, so nu is the Gittins index of this stopping problem and the Whittle index of
the session; (iv) (L/K) nu(a) is minus the slope of the first edge, from the point of age a, of the lower convex hull
of the curve t -> (I(t), S(t)), t >= a, so the index at every age of an n-point law costs O(n).

**Theorem 8 (Special cases of the index).** Setting of Theorem 7, eta the hazard of G. Exact hint (return in R0
seconds): nu = K / (L R0), and K (1 - p) / (L R0) if with probability p the session has ended instead. Hazard never
rising from age a on (a chance of never returning allowed): nu(a) = (K/L) eta(a+), falling with age; with one law for
all sessions and equal K/L the eviction order is least recently used. Hazard never falling (this forces P(G =
infinity) = 0): nu(a) = (K/L) / E[G - a | G > a], rising with age, and the priced decision is made once: keep to the
return if and only if K >= lambda L E[G]. Sharp hint that may never be honoured (G = infinity with probability p,
otherwise a law whose density does not fall on a support ending at b): nu(a) = (K/L) P(return | not back at a) /
E[min(G, b) - a | G > a]. Ended session: nu = 0. Always nu(a) >= (K/L) eta(a+). Lognormal gap (ln G normal with
standard deviation s): the hazard rises up to median * exp(s z*), z* solving phi(z) / (1 - Phi(z)) - z = s, and falls
after; it is still rising at the median if and only if s < 0.798, i.e. p90/p50 < 2.78; past the peak nu(a) = (K/L)
eta(a), and before it nu(a) = (K/L) eta(b), b being the point past the peak with eta(b) (I(b) - I(a)) = S(a) - S(b). A
rule that keeps while (K/L) x hazard >= lambda therefore evicts every lognormal-hinted prefix at its turn's end; the
index does not.

**Theorem 9 (One index for idle prefixes, staged reads and queued turns).** A clean prefix of L tokens is in host
memory. Its session returns after a remaining gap R (R = 0 if the turn has arrived) and is admitted W seconds later, W
independent of R with mean w. If the copy is dropped, the turn restores it at its arrival at a cost K0 that does not
contain lambda (K0 = kappa L / beta + c E[(r - W)^+]), and the restored copy occupies L tokens for rho seconds before
admission. Let g(a) = inf over h of tau(h|a) / F(h|a), the expected holding time per return over the best horizon: 0
for an arrived turn, R0 for an exact hint, 1/hazard past the hazard's peak, E[R] for a rising hazard, infinite for an
ended session. Keeping (to some horizon, and through the queue once the turn arrives) beats dropping now if and only
if lambda < nu(a) = (K0 / L) / (g(a) + w - rho), with nu infinite when g + w <= rho. For a queued turn nu = K0 / (L (W
- rho)); for an idle prefix with no queue and a hazard eta past its peak nu = (K0/L) eta / (1 - rho eta). All host
holders are ranked by this one number; staged reads do not yield before queued turns as a class.

**Theorem 10 (Many sessions).** Sessions i = 1..N with gaps independent across sessions given the hints, restore costs
K_i and sizes L_i not affected by the policy, no read-back before the return. (a) Relaxed problem (held tokens <= C on
time average): there is a price lambda* >= 0 such that 'every session drops its prefix at the first age its index
reaches lambda*', with a coin flip between the two tie-breaking versions for at most one class of sessions, is
optimal; lambda* = 0 if the unpriced demand fits. (b) For every lambda >= 0, sum_i v_i(lambda) - lambda C, v_i being
session i's optimal priced cost rate, is a lower bound on the restore cost of every policy that respects C at every
instant and has the same information. (c) Under the hard capacity, 'evict the lowest index when full' is exactly
optimal for exact hints with equal sizes and equal costs (Belady's rule) and is not optimal in general: with one slot,
A (cost 1, back in 10 s) and B (cost 1.05, back in 11 s), the index evicts B and pays 1.05 where evicting A pays 1.
How close it comes to the bound of (b) is an empirical matter: in an equal-size simulation with the trace's two gap
entries and room for a quarter of the sessions it is 28 % above the bound at 32 sessions, 3.5 % at 128 and within 1 %
at 512, while LRU stays 29 to 34 % above.

**Theorem 11 (The write: two lives of a dirty page).** One idle gap, constant prices, a write that is instantaneous
and costs w, no forced eviction. A page whose only copy below the GPU is in host memory (dirty) costs K_d at the
return if it was dropped; a page with a complete SSD copy (clean) costs K_c <= K_d. With S, I and p_inf = P(G =
infinity) as in Theorem 7: (a) writing strictly before the page is dropped is never better than writing at the moment
it is dropped; for any horizon h within the clean page's holding time, 'hold h, then write if the session is not back'
costs exactly w F(h|a) less than 'write now'. (b) The optimal life is one of two: hold until the first age at which
the index with restore cost K_c + w falls to lambda, then write and drop; or hold until the first age at which the
index with restore cost K_d falls to lambda, then drop unwritten. The first is chosen if and only if V0(K_d) - V0(K_c
+ w) >= w p_inf, where V0(K) = min over s of [lambda L I(s) + K (S(s) - p_inf)]. (c) V0(K_d) - V0(K_c) equals the
integral over K from K_c to K_d of the probability that a page of restore cost K, under its own optimal rule, is
dropped before a return. (d) A page of an ended session is never written. Over several gaps (the page stays dirty
across returns until it is written), (a) still holds and the test of (b) is sufficient for writing but not necessary,
because a write also serves the later gaps.

**Theorem 12 (Due date of a write and order of writes).** Setting of Theorem 11. Let d_w be the first idle age at
which the page's clean index falls to lambda. Holding the page beyond d_w only because it is dirty costs, per second
and given that the session is not back, L (lambda - (K_c / L) eta(age)), which is 0 at d_w and tends to lambda L only
as the hazard dies: the due date is soft, and 'lambda L per second late' is an upper bound. For writes of equal size
whose lateness is charged by one convex non-decreasing function, serving in every slot the released write with the
earliest due date minimises the total lateness cost; with a concave shape or page-specific shapes it does not. When
every waiting write is overdue with constant cost rates L_j (lambda - nu_j), the optimal order is lowest clean index
first, whatever the sizes.

**Theorem 13 (Order on the device).** One device; transfer j (a read or a write) has a release time, a size p_j in
device-seconds and a due date d_j; C_j is its completion time. (a) With all transfers released, serving in order of
due date minimises the maximum lateness max_j (C_j - d_j). (b) With release times and preemption at any instant,
always serving the released transfer with the earliest due date minimises the maximum lateness, whose optimum is L* =
max over pairs (r, d) of [r + (work of the transfers released at or after r and due by d) - d]; hence if any schedule
meets every due date, this one does. (c) If transfers can be interrupted only between pieces of at most p_max
device-seconds, the same rule has maximum lateness below L* + p_max and need not be optimal. (d) A write that must
finish before a given read can start is handled by giving the write the due date min(its own, the read's due date
minus the read's size). (e) These are statements about the maximum lateness. For the weighted sum of lateness, sum_j
w_j (C_j - d_j)^+, earliest due date is optimal when it meets every due date, largest w_j / p_j first is optimal when
every waiting transfer is already late, and in between neither is optimal and the problem is NP-hard.

**Theorem 14 (Read or recompute).** A turn arrives whose prefix of L tokens is not in memory. If it is read back, the
pages are usable V + lags + L / beta seconds after arrival (V the wait in the device's line, beta the device rate in
tokens per second); the turn would get the GPU W seconds after arrival; a second of stall costs c and a device-second
kappa. If it is recomputed, it takes gamma GPU-seconds per token, each costing mu seconds of summed TTFT (mu counts
the turn itself). With lambda = 0, read if and only if mu gamma L > c (V + lags + L / beta - W)^+ + kappa L / beta.
With D0 = V + lags - W: if D0 >= 0, read if and only if mu gamma > (c + kappa) / beta and L > c D0 / (mu gamma - (c +
kappa) / beta). If D0 < 0: read at every length when mu gamma >= (c + kappa) / beta; recompute at every length when mu
gamma <= kappa / beta; otherwise read if and only if L < c (-D0) / ((c + kappa) / beta - mu gamma). With the constants
measured on FA3 (gamma beta = 0.79 to 0.94, lags 0.9 s, V = W = 0, kappa = 0, c = 1): recompute at every length at mu
= 1; read above 22 to 33 K tokens at mu = 2 and above 5.2 to 6.5 K at mu = 5.

**Theorem 15 (Queue weights in an open first-come-first-served queue).** One server, fixed arrival times, turn j needs
s_j seconds of GPU. (a) Every wait is a convex, non-decreasing, piecewise-linear function of the work added to an
earlier turn. Almost surely the summed wait has slope B_k in the work of turn k, B_k being the number of turns served
after k in its busy period, and sum_k (1 + B_k) x (extra prefill of turn k) is a lower bound on the rise of summed
TTFT for any amounts of extra work. (b) With Poisson arrivals of rate alpha and load rho < 1, E[B_k | Q turns waiting,
x seconds of the current service left] = (Q + alpha x) / (1 - rho): a GPU-second costs 1 + (Q + alpha x) / (1 - rho)
seconds of summed TTFT, not 1 + Q. (c) Pass-over. If a turn that is due is not ready for S seconds and ready turns are
served meanwhile without the server idling, the summed wait changes by (the summed work of the turns that went ahead)
- (their number) x (the stalled turn's work), which is zero for equal work. If no turn is ready, each idle second
costs 1 + B. For short stalls in the M/G/1 queue the expected cost of a stall-second is 1 / E[exp(-alpha s)] (1 + rho
for exponential work, exp(rho) for equal work), which is at least 1; so 'the chance that no other turn is ready' is
not the weight of a stall-second there.

**Theorem 16 (Closed loop).** N users stay in session; each alternates a pause and one request. Over a window with
completion rate X, mean time in the system R and Z = (time-average number of pausing users) / X, N / X = R + Z
whatever the server. (a) One server that is never idle: adding delta to the work of the request in service, with delta
small enough not to change the order of service, delays exactly one request of every user, so summed response time
rises by N delta. (b) Pool-bound batch server (at most B turns run; prefill is serial and pauses decoding; a decode
step with n running turns takes tau(n) and gives each a token): while the pool is always full of ready turns, 1 / X =
P + tau(B) n_out / B (P prefill seconds and n_out output tokens per request) and mean TTFT = (N - B) / X - Z + t_pre
(t_pre = admission to first token). One more second of prefill raises summed response time by N and summed TTFT by N
minus the number of turns decoding at that moment. (c) Hence two policies under which the pool stays full, the prefill
seconds per request are equal and the same requests complete have the same mean TTFT, whatever their read timing and
device order; and mean TTFT >= (N - B) g0 - Z + p0 for every policy that keeps the system saturated, g0 being the GPU
time per request with nothing recomputed and a full pool and p0 the prefill of a request's own new tokens. (d) With
requests of unequal work no bound of the form base + mu x work holds for finite amounts in a closed loop: in a
three-user example 0.2 s more work lowers summed response time by 8.4 s.

**Theorem 17 (Preempt: who yields room).** A read needs m tokens; holder j has L_j tokens and displacing it raises its
priced cost by C_j >= 0; costs add up across holders at fixed prices. (a) If holders can be displaced in part at
proportional cost, taking them in increasing C_j / L_j is optimal. (b) Taking whole holders in that order until m is
covered costs less than the optimum plus the cost of the last holder taken and frees less than that holder's size
beyond m; its ratio to the optimum is unbounded. (c) Walking the same order, recording the set 'holders taken so far
plus this one' whenever a holder would complete the cover and skipping that holder, and returning the cheapest
recorded set, costs at most twice the optimum.

**Theorem 18 (The price of room, and a bound).** (a) Filling the room C_H in decreasing index solves 'maximise sum_j
nu_j L_j x_j subject to sum_j L_j x_j <= C_H, 0 <= x_j <= 1', and the multiplier of the room constraint is the index
of the holder at which the room runs out, 0 if everything fits. (b) In the relaxed problem of Theorem 10 the price
lambda* is the index of the marginal holders and lambda* x (demand - capacity) = 0. (c) Open model: arrivals, queue
waits and the weights c and mu are not affected by the policy; sessions are independent given the hints; every held
token is charged to one session; the per-session problems are solved with no device line. Then for any prices lambda,
kappa, sigma >= 0 over a horizon T, the sum of the sessions' optimal priced costs minus (lambda C_H + kappa + sigma
C_S) T is a lower bound on E sum over requests of [c x stall + mu gamma x recomputed tokens] for every policy that
meets the host, device and SSD capacities with the same hints. It is not a bound in the closed loop (Theorem 16 d).

**Theorem 19 (Stability of the write path, minimal model).** Let d(t) be the dirty tokens in host memory; a share p(d)
of reads, non-decreasing in d, is refused for lack of clean room and recomputed; a request then takes g0 + gamma L p
GPU-seconds and the saturated loop completes 1 / (g0 + gamma L p) requests per second; each request creates D_w new
tokens to write; writes drain at w tokens per second. Without further rules d' = (D_w + L p(d)) / (g0 + gamma L p(d))
- w, and when g0 > gamma D_w the inflow rises with d: for D_w / g0 < w < (D_w + L p_max) / (g0 + gamma L p_max) there
are two stable states, d = 0 and the room's limit, separated by the tipping point at which p = (w g0 - D_w) / (L (1 -
w gamma)). If a recomputed page that the SSD already holds complete is not written again, or if a read that finds no
room waits instead of being refused, the inflow is at most D_w / g0 and d stays bounded by the burstiness of new
tokens whenever the rate guaranteed to writes exceeds D_w / g0. Either rule alone is enough in this model.

**Theorem 20 (SSD capacity with chained pages).** A lookup starts just below the session's deepest resident page and
reads stored pages down the chain until the first missing one. (a) For lookups that start above the pages concerned,
deleting a later page of a chain instead of an earlier one never retrieves less; among all ways to delete m pages of a
chain read from its root, deleting the last m is the unique optimum. (b) For a tree of page groups read from the root,
with rho(v) the summed lookup rate of the sessions whose chain passes through group v and theta the value per token
read instead of recomputed, the optimal stored set at an SSD price sigma per token-second is {v : theta rho(v) >
sigma}; these sets are nested in sigma, a tail never outranks its head, and an ended session contributes nothing. (c)
If a shared head is resident on some lookups, its SSD copy is worth theta x the sum over sessions of (rate of lookups
that find the head not resident) x (head length + the session's stored chain below it); it is worth nothing if the
head is always resident, and (a) then fails for it.

**Theorem 21 (Backup from GPU to host).** A page is on the GPU until the GPU gives it up at time d_G; every use of the
page before d_G is served by the GPU copy. If making the host copy later costs nothing else (the copy does not delay
the scheduler), then for lambda > 0 the best time for the copy is the latest feasible one, d_G minus the copy time: an
earlier copy only pays rent. An existing mirror copy has index (cost of making the copy at d_G instead) / (L x (d_G -
now - copy time)): 0 when the late copy is free, and (mu x copy time) / (time to GPU eviction) per token when the late
copy blocks the scheduler, in which case the early copy is optimal whenever lambda x (time to GPU eviction) < mu /
(copy rate).

**Proposition 22 (Share of writes ever needed).** If after each turn a session ends with probability e, and otherwise
the following gap outlasts the host stay of its pages with probability x, independently from gap to gap, then a page
is given up by the host while its session is alive with probability q = (1 - e) x / (x + e - x e). With e = 1/29.2 and
x = 0.046 to 0.103 (a host stay of 30 to 60 s) q = 0.56 to 0.74, so a write deferred to its due date would skip a
share 1 - q of the write volume. At a constant device price and with no forced eviction deferring is weakly optimal
(Theorem 11 a); writing at once is optimal whenever the device is idle.

**Proposition 23 (Fetch on the top tier: the pass meets the copy).** A prefill batch is built at time 0 while other
turns decode. Its history in host memory reaches the GPU by a copy that lands the n layers in order, layer i (from 0)
at (i + 1) r / n, and that takes no GPU compute (measured: a copy is as fast beside decode steps as alone, and so are
the steps). Its forward pass needs p seconds of GPU compute, takes the layers in order, p / n each, and waits for a
layer that has not landed. The GPU runs one pass or one decode step at a time. (a) A pass started at s ends at
end(s) = max(s + p, max(r, p) + min(r, p) / n) and waits end(s) - s - p of that time. (b) Started at 0, as the
engine does by default, it holds the GPU for max(r, p) + min(r, p) / n and waits (r - p)^+ + min(r, p) / n of it; no
running turn gets a token meanwhile. (c) Let s* = (r - p)^+ + min(r, p) / n. A pass started at s* ends when the pass
started at 0 ends and never waits. An earlier start gives the batch nothing and wastes s* - s of GPU time; a later
start delays the batch by s - s* and frees no GPU time. So s* is optimal whatever the weight of a GPU-second and
whatever the weight of the batch's own delay: on this tier Fetch needs no price, because the landing time is known
(the quantile of Theorem 5 is a point). (d) If passes start only at decode-step boundaries, delta apart: the rule
"first boundary at which the copy's remaining time is at most p" (the meeting rule) waits at most min(r, p) / n and
delays the batch by less than delta; the rule "first boundary after the copy has landed" (the rule built) never waits
and delays the batch by less than p + delta. (e) Against a start at 0 the rule built is better if and only if
mu [(r - p)^+ + min(r, p) / n] > w d, with mu the value of a GPU-second (Theorems 15 and 16), w <= 1 the weight of
the batch's own first token and d < p + delta its delay: at saturation (mu about 100) for every copy longer than
its pass, at light load (mu 1 to 3, w = 1) only for a copy about twice its pass or longer.
Proof. Let a_i be the time layer i's compute starts: a_0 = max(s, r / n), a_i = max(a_{i-1} + p / n, (i + 1) r / n).
Unrolling, the end a_{n-1} + p / n is the maximum of s + p and, over i, of (i + 1) r / n + (n - i) p / n, which is
linear in i and so largest at i = n - 1 when r >= p (r + p / n) and at i = 0 otherwise (p + r / n): (a). (b) is (a)
at s = 0. (c): end(s) = end(0) exactly for s <= s*, and for s >= s* end(s) = s + p, so the wait is zero; the two
monotonicities are read off end(s). (d) and (e) follow by putting the two boundaries into (a).
Numbers. Campaign 35, minutes 28 to 110, 985 decode intervals holding one prefill batch: a batch costs the running
turns 1.6 ms + max(copy, 24 ms + 40 us x new tokens) (median error 17.9 ms; the additive fit 12 ms + 22 us x new +
0.75 x copy has 20.5 ms); the copy outlasts the pass in 80 % of batches and the pass waits 37.9 ms a batch on
average (copy 103 ms, compute 74 ms on average). With delta = 24 ms and a copy of 60 ms against a pass of 26 ms:
start at 0 waits 34.5 ms; the rule built waits 0 and delays the batch 37.5 ms; the meeting rule waits 0 and delays
it 11.5 ms.

## A. Fetch

Summary of the check.

The Fetch mathematics of sections 3.1 to 3.3 holds in its core, but four statements are false as drafted and several
need conditions. The derivative of J_F is right for x > 0, constant r and W and a copy held until admission; J_F has
the closed form c x - c M(s) + (c + lambda L) M(s + x), with M(t) = E[(gap - t)^+]. For a random W independent of the
gap, F(x|s) becomes P(A + W <= s + r | A > s), and nothing is read before arrival iff P(W <= r) < q. Single crossing
is sufficient for the threshold rule, not necessary, so the 'iff' in section 7 result 1 is false; the rule is right at
every price iff F(x|s) never falls, which IFR gives. The DFR comparison is right for a proper law held until
admission, and wrong with a mass at infinity: it credits c x P(G = infinity) of stall that is never incurred, and its
rent term is infinite. It needs a drop time; with one, reading at the turn's end on the tool entry at lambda = 2e-5
and c = 1 gains 0.09 to 1.22 s where the draft's formula says arrival. 'Reading at the turn's end is the same as not
evicting' is false: keeping beats fetching by exactly K_net S(s) - (c + lambda L) [M(s) - M(s + x)]. Fetch before
arrival is empty on a falling-hazard key only for a copy that Retain itself dropped at unchanged prices; after a price
fall a read at once is worth 1.6 to 2.0 s per session on the tool entry. Neither trace entry has a falling hazard
everywhere (the tool model rises for 0.79 s, 24.6 % of gaps; the human model jumps up 1.44 times at its median; the
1,200 s cap is an atom), and 28.9 % of the 16,118 per-key entries the replay sent still have a rising hazard at their
median. On those hints, rules that hold a staged read until admission lose 0.8 to 2.5 priced seconds per turn at
lambda >= 2e-5, because 4.7 % of gaps exceed the model's p99; the same read with a drop time gains 0.3 to 0.7 s. In
section 1.5 the open-queue price is (1 + Q)/(1 - rho), not 1 + Q; mu = N holds for TTFT plus decode; c is P(no other
ready turn) times the cost of an empty slot-second. In section 2 recompute wins at mu = 1 and the read wins above 22
to 33 K tokens at mu = 2 with c = 1, but K_read must let the queue wait hide the lags and charge rent only while pages
are held. 


### R1. J_F with recourse, constant W: closed form and derivative

Status: proved, with the assumptions stated.

**Statement.**

Notation for all results. Time runs from now, when the session has been idle for a seconds. R is the remaining gap
(arrival A = now + R); S(t) = P(R > t), F = 1 - S, h the hazard, p_inf = P(R = infinity) (session ended). M(t) = E[(R
- t)^+] = integral_t^inf S(v) dv. F(x|s) = P(R <= s + x | R > s). r = seconds from releasing a read to its pages being
usable (device line V + L/beta + lags). W = queue wait of the turn if its prefix is resident; D = A + W is when it is
due (T in the draft). x = r - W. c = seconds of summed TTFT per second of stall; lambda = price of host memory; L
tokens; lamL = lambda L; q = lamL/(c + lamL); kappa = price of a device-second; beta = device rate. Policy s: release
the read at s if the turn has not arrived, else at arrival; 'arrival' is s = infinity.

Claim. Let r and W be constants with x > 0, the landed copy held until admission, costs linear, p_inf = 0. Then
E[stall] = integral_s^{s+x} F(v) dv, E[early hold] = M(s + x), and
  J_F(s) = c x - c M(s) + (c + lamL) M(s + x) + kappa L/beta,   J_F(arrival) = c x + kappa L/beta.
J_F is continuous; its right derivative at every s with S(s) > 0 is
  dJ_F/ds = c [F(s+x) - F(s)] - lamL S(s+x) = S(s) [c F(x|s) - lamL (1 - F(x|s))],
and this is the two-sided derivative wherever the law has no atom at s or s + x. If p_inf > 0 and lambda > 0, J_F(s)
is infinite for every finite s unless the copy is dropped at some time theta; with a drop time the derivative is
unchanged for s + r < theta.

**Proof.**

Fix s. If R <= s the read starts at arrival and lands at A + r = D + x: stall x, no early hold. If R > s it starts at
s, lands at s + r, and D = R + W, so landing - D = x - (R - s): stall (x - (R - s))^+, early hold (R - s - x)^+. So
E[stall] = x F(s) + integral over u in (s, s+x] of (s + x - u) dF(u). Write s + x - u = integral_u^{s+x} dv and swap
the integrals: the second term is integral_s^{s+x} (F(v) - F(s)) dv. Hence E[stall] = integral_s^{s+x} F(v) dv = x -
(M(s) - M(s+x)). E[early hold] = E[(R - (s+x))^+] = M(s+x). The read is made in every case, so the device term is
constant. M is convex with right derivative -S, so dJ_F/ds = c S(s) - (c + lamL) S(s+x) = c [S(s) - S(s+x)] - lamL
S(s+x); dividing by S(s) gives the draft's form. An atom at s or s + x makes the left and right derivatives differ,
but M, and so J_F, is continuous. With p_inf > 0, M is infinite. With a drop at theta the early hold becomes
integral_{s+x}^{theta} S(v) dv - W S(theta) (result R4), whose derivative in s is again -S(s+x).

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/check_derivative.py part A: Monte Carlo
of the policy itself (2,000,000 gaps per law) against the closed form in 108 cells (6 laws: lognormal, the tool entry,
Weibull k = 2 and 0.6, exponential, two atoms; 3 price sets; 6 starts): worst z-score 2.48. Finite-difference
derivative against the formula: within 5.6e-6 on the lognormal laws, 2.4e-4 at worst (Weibull k = 0.6). M closed form
against quadrature: equal to 6 decimals (8.530160 at t = 0 for the tool entry). Atoms at 3 s and 12 s: J continuous,
right derivative jumps from -0.05 to -0.35 at s = 3 and to +0.70 at s = 10 = 12 - x, the minimum.

**What it checks in the first draft.**

Section 3.3, the displayed derivative, and section 7 result 1 ('the derivative of J_F'). The formula is right. Added
conditions: constant r and W, x > 0, copy held until admission, proper law (or a drop time), and at an atom of the
hint (a point hint) it is the right derivative. New: the closed form in M.


### R2. J_F for a random queue wait independent of the gap

Status: proved.

**Statement.**

Let W be random, independent of R, r constant. Write Phi(s) = P(A + W <= s + r | A > s) = E_W[F((r - W)^+ | s)]: given
the session is still away at s, the chance that its turn would be admitted before a read released now lands.
(B) If a read released at arrival is not re-timed: J(s) = kbar - c M(s) + (c + lamL) E_W[M(s + (r - W)^+)], with kbar
= c E(r - W)^+ + lamL E(W - r)^+, and dJ/ds = S(s) [(c + lamL) Phi(s) - lamL]. The draft's rule holds with F(x|s)
replaced by Phi(s).
(C) If a read released at arrival is held to its best landing (R6), at cost kstar <= kbar: J_C(s) = J(s) - (kbar -
kstar) F(s).
(B) and (C) have the same best start: if P(W <= r) >= q then kstar = kbar; if P(W <= r) < q then Phi(s) <= P(W <= r) <
q for all s and nothing is read before arrival. Phi is nondecreasing in s for an IFR gap and nonincreasing for a DFR
gap, so R3 and R4 carry over.

**Proof.**

Condition on W = w and put x_w = r - w. For x_w > 0, R1 gives J_w(s) = c x_w - c M(s) + (c + lamL) M(s + x_w). For x_w
<= 0: if R <= s the read at arrival lands w - r before admission (hold -x_w, no stall); if R > s the hold is R - s -
x_w; so J_w(s) = lamL (-x_w) + lamL M(s). Both cases read J_w(s) = c x_w^+ + lamL (-x_w)^+ - c M(s) + (c + lamL) M(s +
x_w^+). Average over W: the formula in (B). Differentiate: dJ/ds = c S(s) - (c + lamL) E_W[S(s + (r - W)^+)]. On {R >
s}, R - s > (r - W)^+ iff R + W > s + r, so E_W[S(s + (r - W)^+)] = P(R > s, R + W > s + r) = S(s) (1 - Phi(s)), which
gives the derivative. (C): arrivals before s cost kstar instead of kbar, a difference (kbar - kstar) F(s). The
after-arrival cost c E(d + r - W)^+ + lamL E(W - d - r)^+ is convex in the delay d >= 0 with slope (c + lamL)(P(W <=
r) - q) at d = 0: if P(W <= r) >= q the best delay is 0 and kstar = kbar. If P(W <= r) < q: on {R > s} the event R - s
+ W <= r implies W < r, so by independence Phi(s) <= P(W <= r) < q, dJ/ds < 0 everywhere, and the extra term of (C)
only lowers the derivative further: arrival is best in both. Monotonicity: Phi is an average over x of F(x|s), each
monotone in s in the stated direction.

**Numerical check.**

check_derivative.py parts B and C (2,000,000 samples; gap lognormal, median 20 s, sigma 0.15; r = 4 s). Case P(W <= r)
= 0.969 >= q = 0.333: formula against Monte Carlo at 6 starts, worst z 1.29; derivative by finite difference, by the
formula and with Phi estimated by Monte Carlo agree to 1e-3; best start 16.80 s where Phi = q; cost 1.591 against
2.539 at arrival. Case P(W <= r) = 0.278 < q = 0.600: kbar 3.811, kstar 2.925 at a delay of 2.27 s (the q-quantile of
W minus r); sup of Phi = 0.134; arrival is best under both (B) and (C); worst z 1.74.

**What it checks in the first draft.**

Section 3.3, 'Let x = (r - W)^+ be the part of the read the queue does not hide. If x = 0, nothing is read before
arrival'. The draft treats W as known. With an uncertain W the test quantity is P(A + W <= s + r | A > s), and 'x = 0'
becomes 'P(W <= r) < q'.


### R3. When the threshold rule 'start when F(x|s) >= q' is optimal

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

Let g(s) = (c + lamL)(F(x|s) - q) where S(s) > 0; by R1, J_F(s2) - J_F(s1) = integral_{s1}^{s2} S(u) g(u) du.
(i) Necessary, any law: a start s with F(x|u) < q on [s, s + e) is never optimal. The best start lies where F(x|s) >=
q, or is the arrival.
(ii) Sufficient: if F(x|s) < q before some s0 and >= q after it, s0 is optimal.
(iii) Exact: with tau the first time F(x|s) >= q, the rule is optimal iff I(s) = integral_tau^s S(u)(F(x|u) - q) du >=
0 for every s > tau, the arrival included.
(iv) All prices: for continuous F(x|.), the rule is optimal for every q in (0,1) iff F(x|s) is nondecreasing in s.
(v) IFR (the cumulative hazard -ln S is convex) is the same as 'F(x|s) nondecreasing in s for every x > 0', so IFR
implies (iv). With a density, d/ds F(x|s) = (1 - F(x|s)) (h(s+x) - h(s)): for a given x what matters is the hazard's
change over a lag x. A mass at infinity destroys IFR (F(x|s) then falls to 0), and the first crossing must be compared
with the arrival.

**Proof.**

(i) dJ_F/ds = S g < 0 on [s, s+e), so a slightly later start is strictly cheaper. (ii) J_F is nonincreasing before s0
and nondecreasing after. (iii) Before tau, g < 0 and J_F is nonincreasing, so no earlier start beats tau; for s > tau,
J_F(s) - J_F(tau) = (c + lamL) I(s). (iv) If F(x|.) is nondecreasing it crosses every q once from below: apply (ii).
Conversely take s1 < s2 with F(x|s1) > F(x|s2) and S(s2) > 0. Let m be the maximum of F(x|.) on [0, s2], first
attained at u_m < s2. m < 1: if m = 1 then S(u_m + x) = 0 and F(x|u) = 1 for every later u with S(u) > 0,
contradicting F(x|s2) < m. Take q = m: then tau = u_m, and on (u_m, s2] F(x|u) <= q with strict inequality near s2, so
I(s2) < 0 and by (iii) the rule fails at this price. (v) F(x|s) = 1 - exp(-(Lam(s+x) - Lam(s))) with Lam = -ln S; Lam
is convex iff its increments over a fixed lag are nondecreasing; differentiating gives the hazard form.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/threshold_counterexamples.py. Two sharp
modes (10 +- 0.5 s and 14 +- 0.5 s), x = 3 s, c = 1. At q = 0.30, F(x|s) crosses q up at 7.13 s, down at 10.19 s, up
at 10.62 s; the first crossing is still the global minimum (J = 0.9952, min of I = 0): three crossings, rule optimal.
Same law at q = 0.45: crossings 7.64 up, 9.55 down, 10.91 up; rule cost 1.7162, minimum 1.6768 at 10.91 s, min of I =
-0.0217: not optimal. A tool that fails in 1 s (30 %) or runs 40 s (70 %), q = 0.20: the rule starts at the turn's
end, cost 7.08; the minimum is 1.60 at 34.8 s; reading at arrival costs 3.00. improvements.py I2: for 189 lognormal
cases min(J(first up-crossing), J(arrival)) equals the global minimum to 2e-6.

**What it checks in the first draft.**

Section 7 result 1: 'the threshold rule is optimal iff F(x|s) crosses q once from below (in particular for IFR gaps)'.
'If' is true; 'only if' is false. Corrected: optimal iff the survival-weighted integral of F(x|u) - q after the first
crossing never goes negative; optimal at every price iff F(x|s) never falls. Section 3.3 '(IFR) ... J_F is
quasi-convex ... the threshold rule' is right for a proper law.


### R4. Falling hazard: the optimum is at an end; exact comparison with a mass at infinity, a drop time and the device term

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

(a) If F(x|s) is nonincreasing in s from now on (true when the hazard is), J_F rises up to the last time F(x|s) >= q
and falls after: the minimum is 'now' or 'arrival', never between. If the hazard is nonincreasing from age a0, 'read
now beats arrival' holds on an initial interval of ages [a0, a*] and never again at the same prices.
(b) Proper law, copy held until admission: read now iff c E[min(R, x)] >= lamL E[(R - x)^+], i.e. E[min(R,x)]/E[R] >=
q. The device term cancels.
(c) Any law, with a drop time: the landed copy is dropped if the session has not arrived by theta >= s + r, and a
later return reads at arrival. Let K_io = kappa L/beta + lamL r_m be the fixed cost of one read (device time plus the
rent of its pages for the r_m seconds it holds them before landing). The gain over reading at arrival is
  G(s, theta) = c integral_s^{s+x} [S(v) - S(theta)] dv - lamL [integral_{s+x}^{theta} S(v) dv - W S(theta)] - K_io
  S(theta),
where S includes the mass at infinity. Read now iff G(0, theta) > 0 for some theta. dG/ds = -dJ_F/ds; dG/dtheta =
S(theta) [K_net h(theta) - lamL] with K_net = c x + K_io - lamL W.
(d) Special cases. p_inf = 0, theta = infinity: (b). p_inf > 0, theta = infinity: G = -infinity unless lambda = 0, and
then read now iff c E[min(R, x); R < infinity] >= kappa (L/beta) p_inf.

**Proof.**

(a) dJ_F/ds = S(s) g(s) with g nonincreasing: nonnegative, then nonpositive. For the ages: let I(a) = integral_a^inf
S(u) g(u) du, which is J(arrival) minus J(start at age a) up to the factor S(a). I'(a) = -S(a) g(a): I falls while g >
0 and rises after, toward I(inf) = 0, so I <= 0 once g <= 0 and {I >= 0} is an initial interval. (b) J_F(0) - J_F(inf)
= -c (M(0) - M(x)) + lamL M(x), with M(0) - M(x) = E[min(R,x)], M(x) = E[(R-x)^+], M(0) = E[R]. (c) Three events. R <=
s: same as the arrival policy. s < R <= theta: stall (x - (R - s))^+, early hold (R - s - x)^+, one read. R > theta:
the read at s is wasted, its copy is held theta - s - r seconds, and if R < infinity the turn reads at arrival (stall
x, a second read). Against arrival (stall x and one read per returning session): the stall saved is min(R - s, x) on
{s < R <= theta}, with mean integral_s^{s+x} [S(v) - S(theta)] dv; the extra rent is E[(R - s - x)^+; s < R <= theta]
+ (theta - s - r) S(theta) = integral_{s+x}^{theta} S(v) dv - W S(theta) (use r = x + W); the extra reads are 1{R >
theta}. Differentiate for the two derivatives. (d) Substitute.

**Numerical check.**

check_derivative.py part D (tool entry mixed with p_inf = 0.15; r = 3, W = 1, c = 1, lamL = 0.5, K_io = 0.4): G(s,
theta) against a Monte Carlo of the policy at 7 pairs, fresh 2,000,000-sample draw each, worst z 1.96 (e.g. G(0, 3) =
+0.3598 formula, +0.3602 simulated). The draft's c E[min(R,x)] = 1.5009 against the stall actually saved 1.2009
(difference c x p_inf = 0.30). jf_grid.py: in all 160 cells with x > 0 and c > 0 on the two entries the minimum of J_F
is at the turn's end or at the arrival side (the cap's atom), never between. On the tool entry at lambda = 2e-5, c =
1, the draft's test says arrival (J(now) - J(arrival) = +2.09 s at x = 2, +0.13 s at x = 6); with a drop time and the
in-flight rent charged, reading at once and dropping after 3.6 to 12.8 s gains 0.09 (x = 2), 0.42 (3), 0.71 (4), 0.98
(5), 1.22 s (6).

**What it checks in the first draft.**

Section 3.3 (DFR): 'Read now iff c E[min(R, x)] >= lambda L E[(R - x)^+] + kappa (L/beta) P(G = infinity)', and
section 7 result 1 'for DFR gaps the optimum is at an end'. The end-point statement is right. The formula is right for
a proper law, where its last term is zero. With a mass at infinity it is wrong twice: E[min(R, x)] counts c x P(G =
infinity) for sessions that never return and would cost nothing at arrival, and E[(R - x)^+] is infinite. The
wasted-read probability is P(G > theta), which equals P(G = infinity) only if the copy is never dropped. J_F needs a
drop time.


### R5. Fetch against Retain: exact identity, hysteresis, and when Fetch before arrival is empty

Status: proved, with the assumptions stated.

**Statement.**

Two policies, both measured against 'no copy from age s on, read at arrival'. keep(s, theta): a copy is in host memory
at age s and stays until theta. fetch(s, theta): the copy is on the SSD; its read is released at s and the landed copy
stays until theta >= s + r. With K_io and K_net as in R4:
  G_keep(s, theta) = integral_s^theta S(u) [K_net h(u) - lamL] du = K_net [S(s) - S(theta)] - lamL [M(s) - M(theta)],
and for every law and every theta
  G_keep(s, theta) - G_fetch(s, theta) = K_net S(s) - (c + lamL) [M(s) - M(s + x)]
      = S(s) { (c + lamL) E[(x - R_s)^+] + kappa L/beta - lamL (r - r_m) },
R_s being the remaining gap of a session still away at s.
(1) Not evicting beats reading at the turn's end by this amount; the two are not the same.
(2) If a read holds its pages for its whole duration (r_m = r), or kappa/beta >= lambda (r - r_m), the margin is >= 0:
a read before arrival pays only where keeping a copy would pay. Any law.
(3) A read released at s pays iff the best G_keep(s, theta) over theta >= s + r exceeds the margin. Eviction and
read-back thresholds differ by the margin.
(4) Falling hazard from the current age: Retain keeps a copy until the age u* where K_net h(u*) = lamL and never
after. With c, lambda, kappa, W, r unchanged, a copy Retain dropped is never worth reading back before arrival. Fetch
before arrival is non-empty on such a key only when a copy is missing although Retain would keep it: a price fell
after the eviction (room freed, queue drained, device idle), or Preempt or a hard-capacity eviction took it. Then the
read starts at once.

**Proof.**

G_keep: if s < R <= theta the kept copy saves the stall and the read (c x + K_io) and pays rent from s to admission,
lamL (R - s + W); if R > theta it pays lamL (theta - s) and then behaves as the baseline. The mean is (c x + K_io -
lamL W)[S(s) - S(theta)] - lamL E[min(R, theta) - s; R > s], and E[min(R, theta) - s; R > s] = integral_s^theta S =
M(s) - M(theta). Writing S(s) - S(theta) = integral_s^theta S h gives the hazard form. Subtract G_fetch of R4(c),
written with M as c [M(s) - M(s+x) - x S(theta)] - lamL [M(s+x) - M(theta) - W S(theta)] - K_io S(theta): every term
in S(theta) and M(theta) cancels, leaving K_net S(s) - (c + lamL)[M(s) - M(s+x)]. Use M(s) - M(s+x) = S(s) E[min(R_s,
x)], K_io = kappa L/beta + lamL r_m and W = r - x: per survivor, c x + kappa L/beta + lamL (r_m - W) - (c + lamL)
E[min(R_s, x)] = (c + lamL) E[(x - R_s)^+] + kappa L/beta - lamL (r - r_m). (1) is s = 0. (2): G_fetch <= G_keep, so
if no theta makes G_keep positive none makes G_fetch positive. (3) is the identity rearranged. (4): with a
nonincreasing hazard, K_net h(u) - lamL changes sign once, from + to -, at u*; for s >= u* every G_keep(s, theta) <=
0, hence every G_fetch(s, theta) <= 0.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/fetch_vs_retain.py. Identity on 4,000
random cells: max difference 4.9e-15; fetch never beat keep in a cell with K_io >= lamL r. G_keep against Monte Carlo:
worst z 2.03. With K_io < lamL r (a 6 s read parked 3.9 s holding no pages) fetch beats keep (-0.219 against -0.276).
Tool entry, uncapped, r = 3 s, W = 0, c = 1, L = 25 K. lambda = 2e-5: Retain keeps a copy until 6.8 s idle; a missing
copy is worth reading back only in the first 0.8 s (0.31 s at age 0, 0.01 s at 0.8 s) and that equals keep value minus
margin to 4 decimals (2.065 - 1.758 = 0.307); from 1.2 to 6.8 s a copy is worth keeping (1.38 down to 0.02 s) but not
worth a read. lambda = 1e-6: Retain keeps until 205 s; a missing copy is worth 1.6 to 2.05 s per session at any age up
to 80 s. lambda = 1e-4: keep 2.0 s, never read back. Price fall from 1e-4 to 1e-6 after the drop: read at once, worth
1.70, 2.02, 2.01, 1.60, 0.51 s at ages 1, 3, 10, 30, 100 s.

**What it checks in the first draft.**

Section 3.3 (DFR): 'Reading at the turn's end is the same as not evicting, so on such keys Fetch before arrival is
empty and the decision is Retain's'. The first half is false: keeping is strictly better, by the stall of arrivals
during the flight plus one read's device time. The conclusion holds under added conditions: a read is charged for its
pages over its whole duration (or device time is priced), prices and W are unchanged since the drop, and Retain made
the drop. It fails after a price fall or a forced eviction. Also bears on section 2: the K that Retain needs is K_net
= c x + K_io - lambda L W, because a kept copy also sits in memory through the queue wait; equivalently the index's
denominator should run to admission, E[min(R, h)] + W F(h|a).


### R6. After arrival: where the read should land

Status: proved, with the assumptions stated.

**Statement.**

With r constant and F_D the distribution of the due time D as seen at the decision, the cost c E[(l - D)^+] + lamL
E[(D - l)^+] is convex in the landing l with right derivative (c + lamL)(F_D(l) - q). A read cannot land before now +
r, so the best landing is l* = max(now + r, Q_D(q)), Q_D(q) the smallest l with F_D(l) >= q.
(a) Uncertain read time: the best release time is max(now, Q_{D - r}(q)), the q-quantile of the slack D - r. Only the
law of D - r matters; D and r may be dependent.
(b) c = 0 gives q = 1: every landing at or after the latest possible D costs the same, including never. The rule needs
a floor on c.
(c) If being late at all costs a fixed e seconds more (the wait for the next opening after the landing), the
first-order condition is F_D(l) + (1 - q) e f_D(l) = q: an earlier landing.
(d) A read that cannot be re-timed once released, with an estimate of D that keeps changing: 'release at the first
time P(D <= t + r | what is known at t) >= q' is never later than the best rule; it is the best rule if that
probability, once above q, stays above q; otherwise it can be too early.
(e) D's law is the same for every landing because a turn whose read is in flight is passed over: its GPU turn is set
by the turns ahead of it. That fails without pass-over (then a late read delays every turn behind and its lateness
costs more than 1 per second), and it fails across turns: a release that lengthens the device line of a turn ahead
makes that turn late, it is passed over, and this turn's D moves earlier.

**Proof.**

Convexity and the derivative are the newsvendor computation: d/dl of c E(l - D)^+ is c F_D(l), of lamL E(D - l)^+ is
-lamL (1 - F_D(l)). The derivative is nondecreasing and changes sign at Q_D(q); on the feasible set l >= now + r the
minimiser is the larger of the two. (a) With release time t the cost is c E(t - (D - r))^+ + lamL E((D - r) - t)^+:
the same problem with D replaced by D - r. (b) Set c = 0. (c) Add c e P(D < l) to the cost and differentiate: (c +
lamL) F_D(l) - lamL + c e f_D(l) = 0; divide by c + lamL. (d) If a rule releases at a time t where p_t = P(D <= t + r
| known at t) < q, releasing dt later changes the expected cost by (c + lamL)(p_t - q) dt < 0, so that rule is not
optimal; hence the best release is at or after the first time p_t >= q. If p_t then stays >= q, every further delay
costs, so the first crossing is optimal (the monotone case of optimal stopping). (e) By the definition of D in section
1.3 and the admission rule of 1.4.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/landing_check.py (200,000 samples; D = 10
s plus a Laplace error with median +0.45 s and scale 1.5 s). Grid-best landing equals max(now + r, Q_D(q)) within 0.01
s in six price cases (5.92, 9.84, 11.28, 10.88, 13.02, 3.00 s) and equals now + r when the quantile is earlier. Random
read time (mean 3 s, sd 1.5 s): grid-best release 1.78, 6.64, 8.76 s against Q_{D-r}(q) = 1.77, 6.64, 8.76 s; 'Q_D(q)
minus the mean read time' is off by +1.13, +0.20, -0.47 s. Fixed late charge 0.5 s: grid 9.54, 11.14, 10.68 s equal
the roots of the condition; the plain quantiles are 9.84, 11.29, 10.88 s. Under a Laplace error fitted to campaign
34's readers (median +0.45 s, |error| p90 3.5 s; the shape is my assumption) the fixed 2 s margin is the rule's answer
at q = 0.097, i.e. lambda = 4.3e-6 at L = 25 K, c = 1; the rule would land 4.05 s before the estimate at lambda =
1e-6, 0.15 s before at 2e-5, and 2.09 s after it at c = 0.1, lambda = 2e-5. Moving estimate: first-crossing release
costs 1.5 where waiting for the news costs 0.5.

**What it checks in the first draft.**

Section 3.2: 'P(D <= l*) = q ... The read is released at l* - r ... replaces [the 2 s margin] with the q-quantile of
the measured error of the admission estimate', and section 7 result 1 'after arrival the optimum lands at the
q-quantile of the admission time'. Right when the read time is known and Q_D(q) >= now + r. Changes: take the maximum
with now + r; use the error of the slack (admission estimate minus read-time estimate), since the read time was the
larger error at saturation (planned 8.3 s against 30 to 52 s); c = 0 is ill-posed. Section 1.3's S = (l - D)^+ omits
the wait for the next opening after a late landing.


### R7. The hazard of the trace's two gap entries

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

In the engine's model (two-piece lognormal through p10/p50/p90) neither entry has a hazard that falls everywhere.
Tool waits (0.41/1.69/16.8 s; sigma below the median 1.105, above 1.792): the hazard rises from 0 to 0.478 /s at 0.79
s idle, the 24.6th percentile, then falls; at the median it drops by the factor sigma_low/sigma_high = 0.617 (0.427 to
0.263 /s) because the two-piece density is discontinuous there; then 0.136 at 5 s, 0.058 at 16.8 s, 0.022 at 60 s.
Human waits (3.1/96/1,040 s; sigmas 2.679 and 1.859): peak 0.056 /s at 0.076 s (0.4th percentile), then falling, with
a jump UP by 1.441 at the median (0.0031 to 0.0045 /s at 96 s); the 1,200 s cap carries 8.7 % of the mass as an atom,
where the hazard is infinite.
For Fetch what matters is F(x|s). Tool entry: it rises only for s below 0.59 s (x = 0.5), 0.45 s (x = 1), 0.16 s (x =
2), 0.09 s (x = 6). Human entry: it also rises on [96 - x, 96] and toward the cap.
General fact: a lognormal hazard peaks at median times exp(sigma z*), where phi(z*)/(1 - Phi(z*)) - z* = sigma; the
peak sits at the median exactly when sigma = 0.798, i.e. p90/p50 = 2.78. Below that the hazard is still rising at the
median, above it is already falling.

**Proof.**

Below the median the survival function is that of lognormal(median, sigma_low), above it that of lognormal(median,
sigma_high), and both equal 1/2 at the median; so the hazard is the hazard of the first below the median and of the
second above. A lognormal hazard is h(t) = phi(z)/(sigma t (1 - Phi(z))), z = ln(t/median)/sigma. Setting d ln h/dt =
0 gives phi(z)/(1 - Phi(z)) = sigma + z; the left side minus z falls from +infinity to 0, so the root z* is unique,
and the hazard rises before it and falls after. For sigma_high = 1.792 the peak would be at 0.081 s, below the median,
so above the median the hazard only falls; for sigma_low = 1.105, z* = -0.687 and the peak is at 0.79 s. The density
on each side is phi(z)/(sigma t), so at the median it jumps by sigma_low/sigma_high. The sign of dF(x|s)/ds is that of
h(s+x) - h(s) (R3).

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/hazard_trace.py: hazard on a
400,001-point grid; rising only on [0, 0.79] s (tool) and [0, 0.076] s plus the jump at 96 s (human). Model p99 109 s
(tool) and 7,255 s (human) against the trace's 213 s and 29,366 s: the real tails are heavier. Replayed gaps (campaign
34 ec_full client log, life-table hazard): tool, n = 15,106: 0.08, 0.44, 0.46, 0.30 /s in the bins up to 0.45 s, peak
1.18 /s in (0.45, 0.65] s (31.5 % of gaps at or below), then falling in every bin to 0.005 /s at 256 to 362 s. Human,
n = 1,294: a bump of 0.046 /s at 1.8 to 3.7 s, 0.0045 /s at 5.3 to 7.5 s, a plateau of 0.009 to 0.011 /s from 15 to 63
s, falling after; 80 gaps at the cap.

**What it checks in the first draft.**

Section 3.3: '(DFR) If the hazard falls with idle time (a heavy tail: both gap entries of the FreeInference trace
...), F(x|s) falls in s, J_F is quasi-concave'. As modelled, the tool entry's hazard rises for the first 0.79 s (a
quarter of its gaps), the human entry's jumps up at its median, and the cap is an atom. For x >= 2 s the conclusion
(optimum at an end) still holds numerically (R8). The claim also does not describe the per-key entries the engine uses
(R9).


### R8. J_F on the two entries: best start and gain over reading at arrival

Status: proved.

**Statement.**

L = 25,000 tokens; r = 2 to 6 s; W = 0, 2, 5 s (only x = r - W matters; nothing is read for x <= 0); c = 1, 0.3, 0;
lambda = 0, 1e-6, 2e-5, 1e-4 (rent 0, 0.025, 0.5, 2.5 s per second held). Engine's law, capped at 1,200 s; copy held
until admission.
Tool entry. lambda = 0: start at once; gain c E[min(G, x)] = 0.86, 1.41, 1.83, 2.17, 2.46, 2.72 s for x = 1..6 at c =
1. lambda = 1e-6: start at once; gain 0.67, 1.24, 1.66, 2.02, 2.32, 2.58 s (c = 1) and 0.07, 0.25, 0.38, 0.50, 0.59,
0.67 s (c = 0.3). lambda = 2e-5 and 1e-4: arrival (the only gain is the cap's atom, at most 0.0008 s); reading at once
would cost 0.13 to 2.9 s more than arrival at 2e-5 and 12 to 18 s more at 1e-4. c = 0: arrival for every lambda > 0.
Exact boundary (uncapped): a read held until admission pays iff lambda L/c < E[min(G,x)]/E[(G-x)^+] = 0.112, 0.199,
0.273, 0.341, 0.468 for x = 1, 2, 3, 4, 6, i.e. lambda < 4.5e-6 to 1.9e-5 at c = 1.
Human entry. lambda = 0: start at once (1.92 s at x = 2, c = 1). Any lambda > 0: the only gain is the cap, start at
1,200 - x, gain 0.087 c x (0.17 s at x = 2, 0.52 s at x = 6); reading at once costs 4.7 to 660 s more than arrival.
The threshold rule starts within 1 s of the turn's end in 58 of the tool entry's 80 cells and in 18 of them that costs
more than reading at arrival (worst 17.4 against 5.0 s); 16 of 36 for the human entry.
With a drop time and the read's in-flight rent charged: tool entry, lambda = 2e-5, c = 1, read at once and drop at 3.6
to 12.8 s, gain 0.09 to 1.22 s (x = 2..6); human entry, lambda = 1e-6, drop at 22 to 63 s, gain 0.29 to 1.10 s (x =
3..6).

**Proof.**

Computation of J_F(s) = c x - c M(s) + (c + lamL) M(s + x) (R1) on a grid of 8,000 starts, with M in closed form for
the two-piece lognormal: M(t) = m exp(sigma^2/2) Phi(sigma - z) - t (1 - Phi(z)) on each side of the median, capped by
subtracting M(1,200). The boundary is R4(b). The cap gain is c x P(G >= 1,200): a read started at 1,200 - x lands
exactly on the atom and pays no rent. The drop column maximises G(s, theta) of R4(c) with K_io = lamL min(r, 0.9 +
L/beta).

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/jf_grid.py, table in jf_grid.out and
jf_grid.csv (360 rows). Later idle ages (r = 3 s, c = 1, tool entry): at lambda = 1e-6 'now' wins by 1.57 to 2.00 s at
every age up to 30 s and loses from 300 s; at 2e-5 arrival wins at every age. The engine's own 64-point read_start on
the same cells returns 'the last support point minus x' wherever the continuous law says arrival (R13).
improvements.py I3 gives the boundary values 0.1008 to 0.3188 for sup of 1 - M(s+x)/M(s).

**What it checks in the first draft.**

Section 3.3, the DFR case applied to the two entries ('the optimum is at an end: read now or wait for the arrival').
Confirmed in all 160 cells with x > 0 and c > 0. What the draft does not say: with free or cheap memory (lambda <=
1e-6) 'now' wins by 0.7 to 2.7 s on the tool entry, so the end chosen depends on the price; the human entry's only
pre-arrival read is the replay's cap; and with a drop time 'now' also wins at lambda = 2e-5.


### R9. The start rules on the hints the replay actually sent

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

Campaign 34 ec_full's client log holds 16,118 turns with a p10/p50/p90 entry and the true next gap.
Shapes. 28.9 % of the entries have sigma_high < 0.798, so their hazard is still rising at the median (30.4 % of
tool-call entries, 10.7 % of no-call entries); 22.3 % have sigma_high < 0.5; 6.7 % below 0.2.
Calibration of the two-piece lognormal. 13.6 % of true gaps fall below the entry's p10, 72.5 % inside [p10, p90], 13.9
% above the p90 and 4.7 % above the model's p99 (1 % if calibrated); mean true gap 21.9 s against 16.4 s for the
model; E[(gap - 2 s)^+] 20.7 s against 16.1 s.
Realized priced cost per turn (stall plus rent plus the read's in-flight rent; a prefix assumed on the SSD at the
turn's end; L = 25 K, W = 0, c = 1; 4,000 sampled turns). x = 2 s, lambda = 1e-6: arrival 2.05, exact hint 0.80,
argmin held to admission 1.14, threshold 1.29, land at p50 1.42, land at p10 1.35, engine re-run every 0.5 s 1.15,
read now with a drop time 1.07. x = 2 s, lambda = 2e-5: arrival 3.00, exact 1.75, argmin 3.86, threshold 6.11, p50
11.30, p10 12.46, engine 3.83, read now with a drop time 2.66. x = 4 s, lambda = 2e-5: 6.00, 4.17, 6.81, 9.03, 13.56,
14.64, 6.86, 5.33. x = 4 s, lambda = 1e-4: 14.00, 12.17, 16.53, 21.07, 58.5, 64.4, 16.57, 14.13. A copy that never
left memory and is kept until its entry's best drop time costs 0.30, 1.56, 2.12, 7.58 in the same four settings.
So at lambda >= 2e-5 every rule that holds a staged read until admission is worse than reading at arrival, although
each expects a gain on its own model (argmin expects +0.38 s and realizes -0.86 s at x = 2 s, lambda = 2e-5); the same
early read with a drop time gains 0.34 to 0.67 s at 2e-5; keeping beats every fetch rule, the exact hint included,
from 2e-5 up.

**Proof.**

Each rule's start is computed from the entry alone and then charged on the true gap g: reading at arrival costs c x +
K_io; a read planned at s costs the same if g <= s, else c (x - (g - s))^+ + lamL (g - s - x)^+ + K_io; with a drop at
theta it costs lamL (theta - r) + 2 K_io + c x if g > theta; a kept copy costs lamL g if g <= theta, else lamL theta +
c x + K_io. K_io = lamL r. The argmin and threshold starts use the entry's two-piece lognormal capped at 1,200 s; the
engine column calls the engine's own remaining_gaps and read_start every 0.5 s of idle time until the true arrival.
These are priced costs under assumed prices, not TTFT measurements.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/hints_replay.py (reads only the client
log of 20261123_093604_...; 28 s of CPU). Reads started before arrival out of 4,000 at x = 2 s, lambda = 2e-5: argmin
2,192, threshold 3,578, engine 2,448, read-now-with-drop 3,308. Decile shares of the true gaps under each entry's own
law: 13.6, 11.2, 7.5, 6.5, 9.0, 13.5, 7.9, 7.6, 9.2, 13.9 %.

**What it checks in the first draft.**

Section 3.3, last paragraph: 'This is why reads before arrival paid only with exact hints ..., why reads timed on a
p50 hurt ..., and why the expected-cost arm staged 1 read in campaign 34', and section 3.1: 'a wrong forecast costs
memory, never more stall than reading at arrival'. The falling-hazard argument does not describe the per-key entries
(29 % rise at the median). The third clause has the wrong cause: the fetch fact-finder recorded W >= r in 3,929 of
ec_full's 3,973 'arrival' decisions, so x = 0; its 2,211 'wait' decisions are the 64-point artifact of R13. The p50
clause is supported in priced terms (8.3 and 7.6 s worse than arrival at lambda = 2e-5). 3.1's sentence is true, but
the memory cost is what turns the argmin's expected gain into a loss; it needs the drop time.


### R10. Read or recompute (section 2): K_read, K_rec, the exact rule and the crossover

Status: proved, with the assumptions stated.

**Statement.**

K_rec = mu gamma L is right, and the turn's own extra prefill is already in it: it is the '1' of mu = 1 + (turns
delayed). It must not be added again.
K_read needs three changes. (1) The stall is c (V + lags + L/beta - W)^+, so a queue wait hides the fixed lags first.
(2) The rent is lambda L r_m, with r_m the seconds the read holds its pages (from dispatch, not while parked in the
device line): lambda L (lags + L/beta), quadratic in L; 1.03 s at L = 25 K and lambda = 2e-5. (3) If W can exceed r
the first term is the after-arrival optimum of R6, not c E[(r - W)^+] alone.
Exact rule, linear recompute, lambda = 0: read iff mu gamma L > c (V + lags + L/beta - W)^+ + kappa L/beta. With D0 =
V + lags - W:
  D0 >= 0: read iff L > L_x = c D0 / (mu gamma - (c + kappa)/beta) when mu gamma > (c + kappa)/beta; otherwise
  recompute at every length.
  D0 < 0: read at every length when mu gamma >= (c + kappa)/beta; otherwise read iff L < c (-D0) / ((c + kappa)/beta -
  mu gamma).
Per token the read is cheaper once mu > mu_x = (c + kappa)/(gamma beta).
Numbers (beta = 21.4 K tokens/s, lags 0.9 s, gamma = 0.037 to 0.044 ms, kappa = 0, lambda = 0, no device line). gamma
beta = 0.79 to 0.94; mu_x = 1.26 to 1.06 at c = 1. For L = 25 K: K_rec = 0.92 to 1.10 s (mu = 1), 1.85 to 2.20 (mu =
2), 4.62 to 5.50 (mu = 5), 118 to 141 (mu = 128); K_read = 2.07 s (c = 1, W = 0), 0.62 (c = 0.3, W = 0), 0.07 (c = 1,
W = 2 s), 0 (W = 5 s or c = 0). Crossover at W = 0, c = 1: mu = 1 recompute at every length; mu = 2 read above 33.0 K
(gamma 0.037) or 21.8 K (0.044); mu = 5 above 6.5 K or 5.2 K; mu = 128 above 192 or 161 tokens. At c = 0.3: 11.7 K or
9.0 K (mu = 1), 4.5 K or 3.7 K (mu = 2), 1.6 K or 1.3 K (mu = 5), 57 or 48 tokens (mu = 128). With W >= 2 s, or c = 0:
read at every length up to 65,536 for every mu >= 1. At saturation (c = 0) recompute would win only if kappa > mu
gamma beta - lambda beta r_m, i.e. 74 to 120 TTFT-seconds per device-second at mu = 94 to 128.
mu and c move together: mu >= 2 with c = 1 means the turn behind is itself waiting on a read; if it is ready, c is
near 0 and the read wins at every length. So recompute pays only into GPU time that would otherwise be idle.

**Proof.**

TTFT = W + S + gamma (Delta + U) (section 1.3). Recompute: the turn waits its GPU turn W, then prefills L more tokens;
its own first token is gamma L later and so is every turn behind: mu gamma L with mu = 1 + turns delayed. Recompute
cannot start before the GPU turn, so W hides none of it. Read: released at arrival, lands at V + lags + L/beta; the
stall beyond the GPU turn is the positive part of that minus W, weighted c; it uses L/beta device-seconds and holds L
tokens while in flight. Subtract: K_rec - K_read = L (mu gamma - (c + kappa)/beta) - c D0 where the read sticks out of
the queue wait, and L (mu gamma - kappa/beta) where it does not; solving for the sign gives the four cases. With the
rent the difference gains -lambda L (lags + L/beta) and the crossover is the root of a quadratic.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/read_or_recompute.py: all crossover
tables. With 31 ms per 8,192-token chunk and the prefix term 1.25e-10 s/token^2: mu = 1 recompute up to 65,536; mu = 2
read above 21.4 K; mu = 5 above 5.2 K; mu = 128 read at every length. Restore times at mu = 1: read 2.07 s against
recompute 0.93 to 1.14 s at 25 K; 3.96 against 2.4 to 3.2 s at 65.5 K.

**What it checks in the first draft.**

Section 2: K_read = c E[(r - W)^+] + kappa L/beta + lambda L r; K_rec = mu gamma L; 'recompute wins for any length for
which gamma L < lags + L/beta, i.e. always here; with one turn waiting behind (mu >= 2) the read wins for long
prefixes; at saturation the read always wins'. All three verdicts hold. Refinements: 'long' at mu = 2 means above 22
to 33 K, about the mean prompt; with the rent at lambda = 2e-5 recompute wins at every length up to 65 K at mu = 2, c
= 1; 'any length' holds up to the 65,536 limit on the loaded fit (crossing at 110 K) but an idle-box fit crosses near
50 K (inferred, one request above 30 K); lambda L r should count only the time pages are held.


### R11. The two weights of section 1.5

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

(mu, open queue) In a first-come-first-served queue with utilisation rho, one more second of GPU work at a service
start with Q turns waiting raises summed waiting by (1 + Q)/(1 - rho) seconds: the turn itself, the Q waiting, and
every turn that arrives before the busy period ends. '1 + Q' leaves out the last group.
(mu, closed loop) If the extra work d shifts every user's timeline by d (GPU never idle), then sum of extra TTFT = N d
- (sum of extra decode time). So mu = N is exact for TTFT plus decode; for TTFT alone it is N minus the number of
running requests whose decode the extra prefill stalls.
(c) With identical requests, a stall changes the sum by exactly 0 while another ready turn is waiting to take the
slot. When none is, the empty slot delays the stalled turn and every turn served later in the same busy period. So the
mean cost of a stall-second is P(no other ready turn) times m, where m is the summed TTFT per empty slot-second; m = 1
only when the GPU has spare capacity.

**Proof.**

Open queue: extra work d at a service start raises the unfinished work by d until the busy period ends; under first
come first served every turn served in that period after the injection waits d longer (to first order in d). With
deterministic service tau the work present is (Q + 1) tau, the busy period lasts (Q + 1) tau/(1 - rho) more in
expectation, and rho (Q + 1)/(1 - rho) turns arrive in it; total 1 + Q + rho (Q + 1)/(1 - rho) = (1 + Q)/(1 - rho).
Closed loop: for one user let A_n, F_n, E_n be the arrival, first token and end of its n-th request, A_{n+1} = E_n +
pause. Then the change in TTFT_n is dF_n - dA_n = dF_n - dF_{n-1} - d(decode_{n-1}); summing over n telescopes to the
eventual shift of the user's timeline, d, minus the user's total change of decode time; sum over the N users. c: the
admission instants are fixed by the GPU's work; while a ready turn fills each one, delaying one turn's readiness only
permutes who takes which instant, and with identical requests the sum of admission times is unchanged. If an instant
is left empty for I seconds, every turn served later in the busy period, the stalled one included, is I later.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/queue_prices.py (toys with identical
requests, not the engine). Q1, M/D/1: at rho = 0.8 the cost per injected second is 4.90, 9.86, 14.87 for Q = 0, 1, 2
against (1 + Q)/(1 - rho) = 5, 10, 15 and 1 + Q = 1, 2, 3; over all injections 13.19 against a mean 1 + Q of 2.62; rho
= 0.9: 51.9 against 5.1. Q2, one server with pass-over, stall of 0.2 service times: true c = 1.31, 1.69, 2.11, 2.03,
2.49 at rho = 0.3, 0.6, 0.8, 0.9, 0.95 against P(no other ready turn) = 0.94, 0.73, 0.45, 0.25, 0.13; c given a ready
turn waiting = 0.000 at every rho; c given none = 1.4 to 19. Q3, closed loop (N = 128, 20 slots, a prefill pass runs
alone, decode steps advance all running requests): extra summed TTFT per injected second 109.00, extra summed decode
19.00, total 128.00 = N, at four injection times; the identity N/X - Z - decode gives 22.40 s against a mean TTFT of
22.18 s. For one of about 20 slots I did not simulate m; by the campaign 32 and 34 decode-step fits an empty slot
costs 0.7 to 0.8 % of token throughput, which puts m near 1 to 2 at N = 128 (inferred).

**What it checks in the first draft.**

Section 1.5: 'In an open queue mu is about 1 + Q'; 'mean TTFT = N g - Z - decode ... so mu = N'; 'a second of stall
costs c(t) = P(no other ready turn when turn i is due)'. The first is short by the factor 1/(1 - rho). The second is
right for TTFT plus decode; for TTFT alone it is less (the toy gives 0.85 N; the campaigns measured 0.73 N). The third
is right at both ends (about 1 with spare capacity, exactly 0 while ready work always waits) but is not a probability
in between: it needs the factor m. For Fetch nothing changes in form: q = lambda L/(c + lambda L) holds with the true
c, which can exceed 1.


### R12. Order on the device (section 3.4)

Status: proved, with the assumptions stated.

**Statement.**

With every read released, earliest due date minimises the maximum lateness. With release times, preemptive
earliest-due-date is a lower bound on the maximum lateness of every order; when reads cannot be interrupted,
earliest-due-date among the released reads can miss a schedule that meets every due date, by at most the largest piece
that cannot be interrupted. When every due date has already passed, serving in decreasing c_j / size_j minimises the
sum of c_j times lateness. In between, where some reads can still be on time, the cost is weighted tardiness and
neither order is optimal in general.

**Proof.**

The first is Jackson's exchange argument: swapping two adjacent jobs that are out of due-date order does not raise the
larger of their latenesses. The second is Horn's theorem for preemptive scheduling with release times; a
non-preemptible piece delays a more urgent arrival by at most its own remaining length. The third is Smith's rule:
with all jobs late the cost is the sum of c_j C_j minus a constant, and an adjacent swap toward decreasing c_j/p_j
lowers it. The fourth is by counterexample (enumeration below); minimising total weighted tardiness on one machine is
strongly NP-hard.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/order_check.py: 300 random instances of 6
reads, all 720 orders enumerated. Earliest due date missed the minimum maximum lateness 0 times with all reads
released. With release times preemptive EDD never exceeded the best non-preemptive order; the largest gap was 1.22 s,
below the largest read; non-preemptive EDD was not the best order in 197 instances. With all due dates passed the
c/size order missed the minimum 0 times. With mixed due dates EDD was not optimal in 216 instances and c/size in 275
(worst excess 10.3 and 5.4 s).

**What it checks in the first draft.**

Section 3.4: 'Serving the released jobs in earliest-due-date order minimizes the maximum lateness ... so if any order
lands every read before its turn is due, this one does. When the device cannot meet the due dates, lateness is
rationed by cost per device-second'. True with preemption. The engine's dispatched reads go to one FIFO thread within
a 512-page window, so the guarantee holds up to one window (about 1.8 s at 3.5 ms a page). The ratio rule is exact
only when every waiting read is already late.


### R13. The engine's 64-point read_start is not the minimiser of J_F in the tail

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

For x = read_s - wait_s > 0 and a non-empty support, read_start never returns None ('arrival'): starting at the last
support point minus x always costs at most stall_at_arrival times (count - 1)/count, 1/64 less than arrival at age 0.
On a continuous falling-hazard tail the true minimiser is the arrival (R4), so the engine reports a far 'wait' where
the law says 'arrival'. The plan is harmless only because remaining_gaps returns nothing once fewer than 4 points are
left: for the tool entry the last points are 29.8, 39.7, 59.5, 128.7 s, so the plan dies at 29.8 s idle and the start
at 125.7 s is never reached. It is not harmless at the cap: 6 of the human entry's 64 points sit at 1,200 s, so a
session idle that long is staged at 1,197 s. That read is right for the replay, whose client caps gaps at 1,200 s, and
an artifact for a real trace.

**Proof.**

In read_start the candidate start s_last = last gap minus lead has every other support point arriving before it or
inside its window, each with stall at most stall_at_arrival, and the last point with no stall and no rent; so its cost
is at most stall_at_arrival (count - 1)/count, strictly below the initial best_cost. Hence best_start is set at least
once. remaining_gaps returns () when len(support) - first < MIN_REMAINING_POINTS = 4.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_fetch/engine_artifact.py (loads the engine file
read-only by path): 16,209 random entries with x > 0: read_start returned None 0 times and returned 'last support
point minus x' in 63.0 %. Tool entry: never staged before the plan dies. Human entry: staged at 1,197.0 s. In
jf_grid.out the engine's start is 122.7 to 127.7 s in every tool cell where the continuous law says arrival, at a true
cost 0.5 to 3.4 s above arrival if it were ever executed. The stall_weight argument added to read_start in the working
tree today gives the same start as dividing the rent by c: 0 differences in 17,366 random cases.

**What it checks in the first draft.**

Section 3.3 (General): 'minimize J_F over the support of the conditional law (the engine's read_start does this over
64 points)'. The 64-point support turns every tail into an atom at its last point, so its minimiser differs from J_F's
wherever the continuous answer is 'arrival'. This also explains campaign 34 ec_full's 2,211 'wait' decisions planned a
median 985 s out and its single read at 1,198 s.


### Further results proposed by this check

- State Fetch on the stop-loss function. J_F(s) = c x - c M(s) + (c + lambda L) M(s + x) with M(t) = E[(gap - t)^+];
the whole pre-arrival problem depends on the hint only through M, and the now-or-arrival test needs two numbers,
E[min(G, x)] and E[(G - x)^+]. M is closed-form for the two-piece lognormal (m exp(sigma^2/2) Phi(sigma - z) - t (1 -
Phi(z)) on each side of the median), Weibull (incomplete gamma) and exponential. Consequence for the hint format:
three quantiles do not pin down E[(G - x)^+]; on the replayed hints the model understates it by 22 % (16.1 against
20.7 s) and 4.7 % of gaps exceed the model's p99. Send a capped mean or a p99 with each entry. Checked in
check_derivative.py and hints_replay.py.

- Make the staged copy Retain's, not Fetch's: 'Fetch = flight + keep'. G_fetch(s, theta) = G_keep(s, theta) - [K_net
S(s) - (c + lambda L)(M(s) - M(s + x))], with dG/ds the Fetch condition F(x|s) = q and dG/dtheta the Retain condition
K_net h(theta) = lambda L. One function gives both actions, removes the infinite rent of an ended session, and yields
a hysteresis band of (c + lambda L) E[(x - R_s)^+] + kappa L/beta per session between 'worth keeping' and 'worth
reading back'. On the 16,118 real hints, holding until admission loses 0.8 to 2.5 priced seconds per turn at lambda >=
2e-5 where the same read with a drop time gains 0.3 to 0.7 s. Checked in fetch_vs_retain.py and hints_replay.py.

- A life-cycle statement for the paper: a host copy should exist on the ages where Retain's index is above the price;
Retain acts at down-crossings, Fetch at up-crossings, a read time early. Falling hazard: keep then drop, no Fetch.
Rising hazard or a sharp hint: drop at the turn's end and read back when F(x|s) = q. Example (test run of about 20 s,
sigma 0.1, L = 25 K, lambda = 2e-5, x = 3 s): keeping all along costs 10.05 s, reading at arrival 3.00 s, dropping and
reading back at 16.2 s costs 1.06 s. For any unimodal hazard (lognormal included) the optimum is the first up-crossing
or the arrival, nothing else: 189 cases agree to 2e-6. Checked in improvements.py I2 and I6.

- A one-number test of an entry's regime: a lognormal hazard still rises at the median iff sigma < 0.798, i.e. p90/p50
< 2.78; the quantile at which it peaks solves phi(z)/(1 - Phi(z)) - z = sigma (99.7 % at sigma 0.3, 87 % at 0.5, 32 %
at 1.0, 4.5 % at 1.79). 28.9 % of the replayed per-key entries are on the rising side, so the paper should report both
regimes per key, not one verdict for the trace. Checked in hazard_trace.py and hints_replay.py.

- Closed form for a Weibull gap with shape k > 1 and scale eta: the best start solves (s + x)^k - s^k = eta^k ln(1 +
lambda L/c); for k = 2, s* = eta^2 ln(1 + lambda L/c)/(2x) - x/2; for an exponential gap with mean m the rule is all
or nothing, read now iff x >= m ln(1 + lambda L/c). Agrees with a grid search to 0.004 s (improvements.py I1).

- Value of a hint against its sharpness. V = c x - min J_F. For a sharp hint with standard deviation d, V is about c x
- (c + lambda L) d phi(z_q), z_q the normal q-quantile: each second of spread costs (c + lambda L) phi(z_q) seconds,
0.545 at q = 1/3. Exactly (lognormal, median 30 s, x = 3 s, q = 1/3): V/(c x) = 0.97, 0.73, 0.49, 0.16, 0.005 at d/x =
0.05, 0.5, 1, 2, 3.2, and 0 beyond; at q = 0.714 it is 0.41 at d/x = 0.5 and 0 from d/x = 2. Exact test that any read
before arrival (held to admission) pays: sup over s of 1 - M(s + x)/M(s) > q, i.e. some start has E[min(R_s,
x)]/E[R_s] > q; it matched every row. Checked in improvements.py I3.

- Point forecast against distribution, exactly. A point forecast used as 'land at this time' is optimal iff it equals
l* = s* + x, and l* sits at the quantile level q + (1 - q) F(s*) of the gap, which moves with lambda, L, c, r and W;
so no single number is right at two prices unless the gap is deterministic, and the recourse-aware landing is never
earlier than the plain q-quantile. Example (median 30 s, sigma 0.3, x = 3 s, rents 0.025 to 10 s per second): always
landing at the p10 costs 0.07, 0.02, 2.8, 25, 109 s more than the optimum; at the p50 1.0, 0.4, 0.5, 9.3, 42 s; at the
p90 2.2, 1.4, 0.003, 1.4, 6.8 s. Checked in improvements.py I4.

- Landing quantiles on a shared device. For a fixed order the summed cost is convex in the start times, and reads that
run back to back land as a block with sum over the block of (c_j + lambda L_j)(F_j(C_j) - q_j) = 0: some members land
before their own quantile, some after. In a three-read example the members land at levels 0.09, 0.41, 0.75 against own
q of 0.33, 0.20, 0.63, at cost 1.69 against 2.05 when each is released at its own quantile and the line pushes the
later ones (17 % less). For the order when not every due date can be met, an index that falls from c_j/size_j as slack
grows (apparent tardiness cost) bridges earliest-due-date and the ratio rule, which are the two limits of R12. Checked
in improvements.py I5 and order_check.py.

- Hybrid restore: recompute the head [0, a) on the GPU while the SSD reads the tail [a, L). Priced cost mu gamma a + c
(V + lags + (L - a)/beta - W - gamma a)^+ + kappa (L - a)/beta. The best split is where the two finish together, a* =
(V + lags - W + L/beta)/(gamma + 1/beta), exactly when kappa/beta <= mu gamma < c (gamma + 1/beta) + kappa/beta, i.e.
mu < 2.06 to 2.26 at c = 1, kappa = 0; above that read everything, and if the lags alone exceed gamma L recompute
everything. With no lags the restore rate is the sum of the two rates, 46.4 K tokens/s against 25 K and 21.4 K. At mu
= 1: with 0.9 s of lags it saves 0.01 to 0.10 s at 25 K, 0.36 to 0.56 s at 46.5 K, 0.67 to 0.96 s at 65.5 K; with 0.1
s of lags (a lightly loaded box) 0.37 to 0.49 s at 25 K. Limits: the split the other way has no interior optimum (the
tail's recompute must wait for the head's read); it is an after-arrival action, since the engine holds no tokens of an
SSD-only prefix; and the lookup needs a host-backed anchor, which the engine-seam check must clear. Rule matched a
6,001-point search in 3,000 random cases (read_or_recompute.py).

- A refill rule after a price fall. When room frees or the queue drains, read back at once the SSD-only prefixes whose
keep value exceeds the margin of R5, largest value per token first; for a falling-hazard key that is 'un-evict in
reverse eviction order'. Tool entry, lambda falling from 1e-4 to 1e-6: 1.7 to 2.0 s per session at ages 1 to 30 s, 0.5
s at 100 s. This is the one place where pre-arrival Fetch is non-empty on heavy-tailed keys, and it is triggered by a
price change, not by idle time passing (fetch_vs_retain.py).

- Write section 1.5's weights so they survive a referee: mu = (1 + Q)/(1 - rho) in an open queue; in the closed loop
sum of extra TTFT = N d - (extra decode time), which also explains part of the measured 0.73 N (the toy gives 0.85 N
when a prefill pass stalls the 19 other running requests); c = P(slot left empty) times the cost of an empty
slot-second, exactly 0 while a ready turn waits. And give c a floor in the landing rule, since c = 0 makes every late
landing optimal (queue_prices.py, landing_check.py).

- Engine items found on the way. (a) read_start's 64-point support makes 'last support point minus x' always beat
arrival; compare against a continuous tail (the closed-form M) or drop the last point from the candidates. (b) The
release rule should use the measured error of the slack (admission estimate minus read-time estimate); under an
assumed Laplace error the fixed 2 s margin equals the rule at lambda = 4.3e-6, L = 25 K, c = 1, and is 2 to 6 s too
early once c <= 0.1 at lambda = 2e-5. (c) J_F's device term should be kappa at the read's own start time, so an early
read into idle device time is credited the difference from the price at arrival.


## B. Retain and the write

Summary of the check.

Section 4.1's core holds. With 'not yet back' as the only information, Retain is a one-dimensional stopping problem:
keeping is optimal iff nu(a) = (K/L) sup_h F(h|a)/tau(h|a) >= lambda, the first age with nu <= lambda is the earliest
optimal drop age, the same test applies at every later age, and the drop set grows with lambda. nu is the Gittins
index of the single holding problem and, because every session ages whether held or not, the Whittle index of the
session (exact checks on 12 laws x 37 prices, plus Monte Carlo). Three things in 4.1 must change. (1) K must not
contain lambda (K_read does): the price-free index is nu = (K0/L)/(g + W - rho), with g = inf_h tau/F, W the queue
wait and rho the restore's own hold; it ranks idle prefixes, staged reads and queued turns in one list and does not
put staged reads before queued turns as a class. (2) The rising-hazard formula gives 0 whenever there is an end
probability (E[R] is infinite, and such a law cannot have a rising hazard), and the falling-hazard formula holds only
past the hazard's peak (0.79 s for the tool entry; a myopic hazard rule would evict every lognormal-hinted prefix at
its turn's end). (3) 'Evict in increasing nu' is exact only for the priced problem, as a common threshold; under the
hard capacity it is exact for exact hints with equal sizes and costs (Belady) and otherwise a heuristic: 2x the
optimum in a two-session example, +0.6 to +2.3 % in offline brute force with unequal sizes or costs, and against the
priced bound +24 % at 32 sessions, +6 % at 128, +2 to 3 % at 512 to 2,048. For the write (4.2, result 7.3): V_dirty -
V_clean - kappa L/beta is the value of writing at some time, not of writing now; at a constant device price writing
later weakly dominates writing now, a dirty page has exactly two candidate lives (hold, write, drop; or hold, drop
unwritten), and writing at once is optimal when the device is idle or when a forced eviction of the only copy has more
than about 0.4 % chance at saturation prices. The draft's cost of a late write, lambda L_p per second, is an upper
bound: the rate is L_p (lambda - nu_clean), zero at the due date (1 s late costs 3 to 17 % of lambda L_p); due-date
order is optimal for equal-size pages with one convex lateness shape, overdue pages go lowest clean index first with
the size cancelling, and the two orders coincide only when index curves do not cross. Eager writing is right on this
workload because device time is nearly free at both ends and as insurance, not because 56 to 67 % of positions are
read back: that share is the fraction of writes ever needed, which the model reproduces (57 to 74 % for a 30 to 60 s
host stay), and a deferred write would skip the rest. Tail-before-head deletion on the SSD is proved when lookups
start above the pages concerned and fails when a head stays resident; a shared head's index is a sum over the sessions
below it. The mirror's index is (cost of a late backup)/(L x time to GPU eviction): zero if the late copy is
asynchronous, but about 6.6 TTFT-seconds per request if it blocks the scheduler at saturation, in which case the eager
backup is the optimum. Prices (kappa, c, lambda) were not measured here; every number with them is arithmetic on the
brief's constants.


### R1 Holding problem: keep iff nu(a) >= lambda, first passage, dynamic consistency, indexability

Status: proved.

**Statement.**

Symbols. G in (0, inf] is the gap of one idle period (G = inf: the session never returns). S(t) = P(G > t), p_inf =
P(G = inf), I(t) = integral_0^t S(u) du = E[min(G, t)]. At idle age a with S(a) > 0: R = G - a given G > a, F(h|a) =
(S(a) - S(a+h))/S(a) = P(R <= h), tau(h|a) = (I(a+h) - I(a))/S(a) = E[min(R, h)]. The prefix has L tokens; holding it
costs lambda*L per second (lambda in TTFT-seconds per token-second); if it is dropped before the return and the
session returns, K is paid (K >= 0, independent of lambda and of the drop age). The only information over time is 'not
yet back'. nu(a) = (K/L) sup_{h>0} F(h|a)/tau(h|a).
Claims. (i) A policy is a drop age s in [a, inf]. Seen from age 0 its cost is C(s) = lambda*L*I(s) + K*(S(s) - p_inf);
seen from age a it is (C(s) - lambda*L*I(a))/S(a): every age solves min over s >= a of the same function C. (ii) C(a)
- C(a+h) = S(a)*[K*F(h|a) - lambda*L*tau(h|a)]. (iii) Dropping at age a is optimal iff nu(a) <= lambda; keeping is
strictly better iff nu(a) > lambda. (iv) s*(a) = min{s >= a : nu(s) <= lambda} exists and is the earliest optimal drop
age. (v) Dynamic consistency: at any later age a', reached for whatever reason, the optimal continuation is again
'keep iff nu(a') > lambda'. (vi) Indexability: the drop set D(lambda) = {a : nu(a) <= lambda} grows with lambda; D(0)
is the set of ages with no chance of return.

**Proof.**

Step 1 (policies). The controller observes only 1{G <= t}. A stopping time of that filtration, even with an
independent randomisation U, equals on {G > t} a function s(U) of U alone; after the return nothing is decided. So its
cost is E_U[C(s(U))] >= inf_s C(s): deterministic drop ages suffice.
Step 2 (cost). Dropping at s pays rent lambda*L*min(G, s) and pays K iff s < G < inf. Taking expectations, C(s) =
lambda*L*I(s) + K*(S(s) - p_inf). Given G > a and s >= a the cost is [lambda*L*(I(s) - I(a)) + K*(S(s) - p_inf)]/S(a)
= (C(s) - lambda*L*I(a))/S(a), an increasing affine function of C(s).
Step 3 (difference). C(a) - C(a+h) = K*(S(a) - S(a+h)) - lambda*L*(I(a+h) - I(a)) = S(a)*[K*F(h|a) -
lambda*L*tau(h|a)].
Step 4 (test). Dropping at a is optimal iff C(a) <= C(a+h) for all h > 0, iff K*F(h|a) <= lambda*L*tau(h|a) for all h,
iff nu(a) <= lambda. If nu(a) > lambda some h has K*F > lambda*L*tau, so keeping until a+h is strictly better.
Step 5 (first passage). S is right-continuous and non-increasing, hence lower semicontinuous; I is continuous; put
C(inf) = lim C(s). So C is lower semicontinuous on the compact interval [a, inf] and attains its minimum; let s1 be
the smallest minimiser. C(s1) <= C(u) for u >= s1, so nu(s1) <= lambda by Step 4. For a <= s < s1, C(s) > C(s1), so
nu(s) > lambda by Step 4. Hence s1 = min{s >= a : nu(s) <= lambda}.
Step 6 (consistency). By Step 2 the problem at any age a' is min over s >= a' of the same C, so the rule at a' does
not depend on how a' was reached (Bellman's principle for a deterministic path).
Step 7 (indexability). D(lambda) is defined by nu(a) <= lambda, which is monotone in lambda; nu(a) = 0 iff F(h|a) = 0
for all h.
Ties. At nu(a) = lambda dropping is optimal; keeping until a+h is equally good iff the sup is attained at h. 'Drop at
the first age with nu < lambda' is optimal whenever nu falls through lambda without a flat stretch (the falling-hazard
case); 'first age with nu <= lambda' is always optimal.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_retain/hold_index.py (output hold_index.out).
T1: 12 discrete laws (lognormal, the two trace entries as two-piece lognormals, Weibull k = 0.5 and 3, uniform, point,
bimodal, trimodal, with and without an end mass, one random law) x 37 prices from 1e-8 to 1e-2, 2,400 grid ages each.
Violations of 'drop is optimal iff nu <= lambda': 0. Violations of 'the first age with nu <= lambda attains min C': 0.
Backward induction gives the same value (largest absolute difference 4.3e-9) and the same stopping set (0
differences). Drop sets nested in lambda: 0 violations. T1b Monte Carlo, 400,000 gaps, K = 2 s, L = 25,000: tool entry
at lambda = 5e-6: the rule drops at 15.25 s, cost 0.7294 (simulated 0.7309 +- 0.0018), equal to the brute-force best
age; dropping at once costs 1.983. Ages with survival below 1e-4 were left out of the comparison (double precision).

**What it checks in the first draft.**

4.1 'Keeping for some horizon beats dropping now iff nu(a) = (K/L) sup_h F(h|a)/tau(h|a) >= lambda', 'the problem is
indexable', and 7.2 'keeping iff nu >= lambda is optimal for the priced session; indexability'. They hold. Two
qualifications: at nu = lambda dropping is optimal and keeping is as good only if the sup is attained; and K must be
free of lambda (the draft's K_read contains lambda L r; see R7).


### R2 nu is the Gittins index of the holding problem and the Whittle index of the session

Status: proved.

**Statement.**

Let pi(a) = P(G < inf | G > a) and V(a) the optimal priced cost from age a. Then u(a) = K*pi(a) - V(a) = sup over drop
ages s >= a of E[K*1{G <= s} - lambda*L*(min(G, s) - a) | G > a]: a process that earns K when it completes (the
session returns while the copy is held), is charged lambda*L per second while it runs, and can be retired for 0. The
Gittins index of that process (Gittins 1979; the retirement formulation of Whittle 1980: the smallest charge at which
retiring is optimal; the fair charge of Weber 1992) is sup_s E[reward]/E[run time] = K*sup_h F(h|a)/tau(h|a) per
second, that is nu(a) per token-second. It is the same formula as the index of a job with attained service a in
single-server scheduling (Sevcik 1974; Aalto, Ayesta and Righter 2009). Because idle age advances for every session,
held or not, and many sessions are held at once, the many-session problem is a restless bandit: nu is the Whittle
index (Whittle 1988) of the session's project, and R1 (vi) is Whittle indexability. The Gittins index theorem
therefore does not give optimality of the ranking under a hard capacity (see R6).

**Proof.**

K*pi(a) = K*P(G <= s | G > a) + P(G > s | G > a)*K*pi(s). The policy 'drop at s' costs lambda*L*E[min(G, s) - a | G >
a] + P(G > s | G > a)*K*pi(s). Subtracting, K*pi(a) - cost(s) = E[K*1{G <= s} - lambda*L*(min(G, s) - a) | G > a]. Its
sup over s is u(a) >= 0, and by R1 u(a) = 0 iff lambda >= nu(a): the charge at which continuing and retiring are
equally good is lambda = nu(a), which is the definition of the index in the retirement formulation and of the fair
charge sup_s E[K*1{G <= s}]/E[L*(min(G, s) - a)]. Restlessness: the state of a session that is not held (its idle age
and hint) keeps moving; Gittins' theorem needs frozen passive arms and one active arm at a time, neither of which
holds here. Whittle's index is defined through exactly the priced one-project problem of R1, and indexability is R1
(vi).

**Numerical check.**

Same runs as R1: the backward induction of hold_index.py is the retirement formulation in discrete time and returns
the same stopping set as the index on all 12 laws.

**What it checks in the first draft.**

4.1 'nu is the Gittins index of the session's holding problem (the best ratio of expected cost avoided to expected
token-seconds spent)' and 7.2. The formula and the name are right for the single problem. The sentence that follows in
the draft ('so ... evict in increasing nu is the exact optimum') does not follow from it: for many sessions the right
name is Whittle index, and the supporting theorems are those of restless bandits (R6).


### R3 Geometric form: the index is the slope of a convex hull

Status: proved.

**Statement.**

Plot the curve Gamma = {P_t = (I(t), S(t)) : t >= 0}: expected token-seconds held per token against the probability of
still being out. Its slope is minus the hazard. (i) (L/K)*nu(a) is minus the slope of the first edge at P_a of the
lower convex hull of {P_t : t >= a} (with the limit point (I(inf), p_inf) when I(inf) is finite). (ii) For price
lambda the optimal drop age is where a line of slope -lambda*L/K supports that hull. (iii) Falling hazard after a iff
Gamma is convex there iff nu = (K/L)*hazard; rising hazard iff concave iff the hull is the chord to the end point.
(iv) The lower hull of Gamma is the session's efficient frontier (memory-time per gap against probability of paying
K); stretches where the hazard rises are never stopped in. (v) The whole index curve of a law on n points costs O(n)
(one monotone-chain pass from the right).

**Proof.**

C(s) = lambda*L*x + K*y - K*p_inf with (x, y) = P_s is linear with positive weights, so minimising C over s >= a is
minimising a linear functional over the curve; minimisers lie on its lower-left convex hull, which gives (ii) and (iv)
(mixtures of drop ages give convex combinations). By definition (L/K)*nu(a) = max over s > a of (y_a - y_s)/(x_s -
x_a), the steepest descending chord from P_a; P_a is the leftmost point of {P_t : t >= a}, and the first edge of a
lower hull from its leftmost point has the smallest slope among all chords from it: (i). dS/dI = -f/S = -hazard, so
convexity of Gamma is a falling hazard: (iii). (v) is the standard monotone-chain hull, adding points from right to
left.

**Numerical check.**

hold_index.py: index_hull (O(n)) against the O(n^2) definition on the 12 laws: largest relative difference 6.9e-11.
hard_capacity.py priced_bound uses the same curves for the many-session bound.

**What it checks in the first draft.**

Not in the draft. It restates R1 and gives 4.1's special cases in one picture; it also gives section 6 a computation:
summing the sessions' hull demand curves gives the price that clears a given room.


### R4 Special cases and closed forms (exact hint, falling hazard, rising hazard, ended; Weibull, Lomax, lognormal)

Status: proved, with the assumptions stated.

**Statement.**

eta(t) = f(t)/S(t) is the hazard. (a) Exact hint, return in R0 seconds: nu = K/(L*R0); if the session has ended
instead with probability p: nu = K*(1-p)/(L*R0). (b) eta non-increasing on [a, inf) (an end mass is allowed): nu(a) =
(K/L)*eta(a+); nu falls with age, the drop set is [a*, inf) (a time-to-live); with one pooled law and equal K/L the
order is least recently used. (c) eta non-decreasing on [a, inf) (this forces p_inf = 0): nu(a) = (K/L)/E[R | a],
rising with age; the priced problem is decided once, at age 0: keep to the return iff K >= lambda*L*E[G]. (d) Ended
session: nu = 0. (e) Any law: nu(a) >= (K/L)*eta(a+), and at an interior best horizon h*, nu(a) = (K/L)*eta(a + h*).
(f) Closed forms. Weibull S = exp(-(t/theta)^k): k <= 1: nu = (K/L)*(k/theta)*(a/theta)^(k-1); k >= 1: nu =
(K/L)/m(a), m(a) = exp(x)*(theta/k)*Gamma(1/k, x), x = (a/theta)^k (upper incomplete gamma). Lomax S = (1 +
t/theta)^(-alpha): nu = (K/L)*alpha/(theta + a), time-to-live a* = alpha*K/(lambda*L) - theta. Lognormal ln G ~ N(m,
s^2), z = (ln t - m)/s: eta(t) = phi(z)/(s*t*Phibar(z)), I(t) = t*Phibar(z) + exp(m + s^2/2)*Phi(z - s); the hazard
rises up to t* = exp(m + s*z*), z* the root of phi(z)/Phibar(z) = z + s, and falls after; nu(a) = (K/L)*eta(a) for a
>= t*; for a < t*, nu(a) = (K/L)*eta(b*) with b* > t* the root of eta(b)*(I(b) - I(a)) = S(a) - S(b).

**Proof.**

F(h|a)/tau(h|a) = integral_a^{a+h} eta(u) S(u) du / integral_a^{a+h} S(u) du: the S-weighted average of the hazard
over [a, a+h] (law with a density on (a, inf) apart from the mass at infinity). (b) If eta is non-increasing the
average is non-increasing in h: the sup is the limit h -> 0, eta(a+). (c) If eta is non-decreasing the average is
non-decreasing in h: the sup is the limit h -> inf, (S(a) - p_inf)/(I(inf) - I(a)). Since integral_0^inf eta = -ln
p_inf, p_inf > 0 would make a non-decreasing eta integrable, hence zero; so p_inf = 0 and the limit is 1/E[R | a]. A
rising hazard implies a falling mean residual life, so nu rises. C'(s) = S(s)*(lambda*L - K*eta(s)) changes sign at
most once, from + to -, so C is quasi-concave and its minimum is at s = 0 or s = inf: K against lambda*L*E[G]. (a)
F(h|a) = 1{h >= R0}, tau = min(h, R0): the ratio is 0 for h < R0 and 1/R0 from R0 on. With an end probability p, for h
>= R0: F = 1 - p, tau = (1-p)*R0 + p*h, largest at h = R0. (d) F = 0. (e) the derivative of F/tau in h vanishes iff
f_a(h)*tau(h) = F(h)*S_a(h), i.e. eta(a+h) = F(h)/tau(h); the lower bound is the limit h -> 0. (f) Weibull: the hazard
(k/theta)(t/theta)^(k-1) is monotone; the mean residual life follows from u = (t/theta)^k. Lomax: hazard alpha/(theta
+ t). Lognormal: d ln eta/dt = (phi(z)/Phibar(z) - z - s)/(s*t); the inverse Mills ratio minus z falls from inf to 0,
so there is one sign change, at z*. I(t) = t*S(t) + integral_0^t u f(u) du, and the partial mean of a lognormal is
exp(m + s^2/2)*Phi(z - s). By R3 the curve is concave before t* and convex after, so from a < t* the hull is the
tangent from P_a to the convex part; tangency is the stated equation.

**Numerical check.**

hold_index.py T2, T3 (quadrature plus a scalar search over the horizon). Weibull k = 0.5: index = hazard to 6 digits
at a = 0.5, 5, 50 (0.316228, 0.100000, 0.031623), best horizon at the lower bound. Lomax: 0.500000, 0.187500, 0.028302
= alpha/(theta + a). Weibull k = 3: 0.055992, 0.108133, 0.294414 = 1/mean residual life. Lognormal closed form against
quadrature, 4 laws x 3 to 4 ages: agreement to 6 or 7 digits; hazard peak at median x 0.0486 for sigma = 1.79, x 1.76
for sigma = 0.5. extra_checks.py b: hazard peaks and the ages where index and hazard differ for the two trace entries.
T1b: the myopic hazard rule costs 1.98 against 0.73 for the index rule on the tool entry (K = 2 s, L = 25,000, lambda
= 5e-6).

**What it checks in the first draft.**

4.1 special cases and 7.2 'the reductions to Belady and LRU'. Exact hint, ended session and the Belady and LRU orders
hold as orders (equal K/L). The falling-hazard bullet needs 'past the hazard's peak': a lognormal hazard starts at 0,
so nu = (K/L) h(a) fails at the turn's end. For the tool entry (two-piece lognormal 0.41/1.69/16.8) the hazard rises
until 0.79 s and the index at age 0 is 0.395 K/L per second while the hazard is 0; a rule that keeps while (K/L) x
hazard >= lambda (the paper's eq. retain) evicts at the turn's end. For the human entry the two-piece form makes the
hazard jump up by 1.44 at the median (sigma_low 2.68 > sigma_high 1.86), so the index exceeds the hazard for ages 74
to 96 s: an artifact of the form. The rising-hazard bullet is treated in R5.


### R5 Rising hazard with an end probability: the draft's formula is wrong

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

If P(R < inf) < 1 then E[R] is infinite, so the draft's nu = (K/L)*P(R < inf)/E[R] is 0 for every such law; and no law
with P(G = inf) > 0 has a hazard that rises forever. Correct statements. (i) Proper law, rising hazard: nu(a) =
(K/L)/E[R | a]. (ii) Sharp hint that may never be honoured: G = inf with probability p, otherwise G0 whose density
does not fall on its support, which ends at age b: nu(a) = (K/L)*P(R < inf | a)/E[min(R, b - a) | a]. (iii) In
general, with pi(a) = P(the session is alive | idle for a) and F0, tau0 those of the proper part: nu(a) = (K/L)*sup_h
pi(a)*F0(h|a)/(pi(a)*tau0(h|a) + (1 - pi(a))*h), with the sup at a finite horizon (possibly h -> 0).

**Proof.**

(i) is R4 (c). Impossibility: integral_0^inf eta(t) dt = -ln p_inf is finite when p_inf > 0, and a non-decreasing
non-negative integrable function on [0, inf) is zero. (iii) Only live sessions return: F(h|a) = pi(a)*F0(h|a); a dead
session is held for the whole horizon: tau(h|a) = pi(a)*tau0(h|a) + (1 - pi(a))*h. As h -> inf the numerator is
bounded and the denominator grows linearly, so the ratio tends to 0 and the sup is at a finite h. (ii) The mixture's
hazard is (1-p)*f0/(p + (1-p)*S0); on the support the numerator does not fall and the denominator does not rise, so
the hazard is non-decreasing up to b and 0 after. The S-weighted average hazard over [a, a+h] (R4) therefore rises up
to h = b - a and falls after (only zero-hazard time is added), so the sup is at h = b - a, where F = P(R < inf | a)
and tau = E[min(R, b - a) | a].
Counterexample to the draft: G uniform on [15, 25] s with probability 0.9, never with probability 0.1, age 0. True
index: 0.9/(0.1*25 + 0.9*20) = 0.0439 K/L per second. Draft: 0.9/inf = 0.

**Numerical check.**

hold_index.py T2: U[15,25] with P(ended) = 0.1 and 0.3: index by quadrature 0.043902 and 0.032558 (best horizon 25.00
s), the same on a 200,001-point grid, the same from formula (ii); the draft's formula gives 0. extra_checks.py a:
Weibull k = 1.5, theta = 20 (rising hazard, falling density) with P(ended) = 0.05: sup at h = 43.5 s, value 0.04794;
(1-p)/E[R | returns] would give 0.05262, so even that repair overstates when the density falls.

**What it checks in the first draft.**

4.1 'rising hazard (IFR): the sup is at h -> infinity, nu = (K/L) P(R < infinity)/E[R]: the longest idle is kept'.
With P(R < inf) = 1 it is right (and 'the longest idle is kept' holds for one pooled law). With an end probability the
formula gives 0. Replacement: 'rising hazard (a sharp hint): nu = (K/L)/E[R]; with a chance that the session has
ended, nu = (K/L) P(return)/E[min(R, time to the end of the hint's support)]'.


### R6 Many sessions: exact for the priced problem, a heuristic under the hard capacity; the size term

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

Sessions i = 1..N, gaps independent across sessions given the hints, capacity C tokens. Relaxed problem: minimise the
long-run restore cost subject to the time-average of held tokens <= C. (i) There is lambda* >= 0 such that 'every
session drops its prefix at the first age its index reaches lambda*' (mixing the two tie-breaking versions if needed)
is optimal for the relaxed problem; at every instant the prefixes held are those whose index has stayed above lambda*
since their turn ended. For every lambda >= 0, sum_i v_i(lambda) - lambda*C (v_i = session i's optimal priced cost
rate) is a lower bound on the cost of every non-anticipative policy that respects C at every instant. (ii) Under the
hard capacity, 'when full, evict the lowest index' is not optimal in general. Counterexample A (exact hints, one slot,
equal sizes): K_A = 1 back in 10 s, K_B = 1.05 back in 11 s; nu_A = 0.100 > nu_B = 0.095, the index evicts B and pays
1.05; evicting A pays 1. Counterexample B (equal sizes and costs, one slot, no early re-fetch): A returns at 1 s with
probability 1/2, else never; B returns at 10 s; nu_A = 0.5 K/L > nu_B = 0.1 K/L, the index keeps A and pays K; keeping
B pays K/2. (iii) It is exactly optimal for exact hints with equal sizes and equal costs (Belady's rule; Mattson et
al. 1970). (iv) Size. The relaxed problem needs no size term beyond K/L. Under the hard capacity the greedy order
solves the divisible knapsack and is within one prefix of the integral one. K/L itself depends on L when a read has an
exposed fixed lag: K_read/L = c*(V + lags - W)^+/L + (c + kappa)/beta (V the wait in the device's line), so long
prefixes go first and one whole prefix goes before the tails of two; with K = mu*gamma*L, or at c = 0, there is no
size effect.

**Proof.**

(i) For any policy p respecting the average constraint and any lambda >= 0: cost(p) >= cost(p) + lambda*(mem(p) - C)
>= min over policies of [cost + lambda*mem] - lambda*C = sum_i v_i(lambda) - lambda*C. The minimum separates because
the priced objective is a sum over sessions and the other sessions' histories are independent of session i's gap, so
they cannot lower session i's priced cost; within a session each gap is a fresh problem solved by R1. v_i is concave
in lambda with slope m_i(lambda), the memory used by the index rule (the smallest- and largest-memory optimal rules
differ only at ties). Take lambda* with sum m_i(lambda*+) <= C <= sum m_i(lambda*-) (lambda* = 0 if sum m_i(0) <= C)
and mix the two tie-breaking rules to meet C: cost = sum v_i(lambda*) - lambda* C, equality in the bound, hence
optimal. A policy that respects C at every instant respects it on average, so the bound applies to it. (ii) A: one of
the two must go; costs are 1.05 and 1. B: enumeration; the index charges B for 10 s of room at one price although the
slot is empty and worthless after 1 s. (iii) Classical. (iv) In the priced problem L multiplies the rent and enters
the restore cost only through K, so the comparison of R1 is in K/L; nothing else depends on L. Fractional knapsack:
greedy by ratio is optimal for divisible items. Fixed lag: freeing m tokens from one prefix costs P(return)*[c*(V +
lags + m/beta - W)^+ + kappa*m/beta]; from two prefixes, m/2 each, the fixed part is paid twice.

**Numerical check.**

hard_capacity.py (output hard_capacity.out). Offline brute force, exact hints, 6 items, 14 requests, 300 random
sequences each: equal sizes and costs: index order suboptimal in 0; costs in {1, 3}: 22 sequences, worst ratio 1.22,
total cost +0.86 %; sizes in {1, 2} with cost = size: 63, worst 1.20, +2.3 %; sizes in {1, 2}, equal costs: 13, worst
1.17, +0.64 %. Simulation, 300,000 returns per cell after a 3,600 s warm-up, gaps tool 0.41/1.69/16.8 s (92 %) and
human 3.1/96/1,040 s (8 %), capped at 1,200 s, class known at the turn's end, equal sizes. Capacity for 25 % of the
sessions: priced bound on the miss ratio 0.0734 (0.0936 for a policy that does not know the class); class-aware index
0.1542, 0.0911, 0.0775, 0.0749, 0.0754 at N = 8, 32, 128, 512, 2,048 (standard error 0.0005 to 0.0007); LRU 0.1648,
0.1089, 0.0960, 0.0954, 0.0954; clairvoyant farthest-next-return 0.1150 down to 0.0445. Capacity for 50 %: bound
0.0325 (0.0344 pooled); index 0.0407, 0.0339, 0.0329, 0.0326, 0.0323; LRU 0.0443 down to 0.0342.

**What it checks in the first draft.**

4.1 'so keep while nu >= lambda, evict in increasing nu is the exact optimum of the priced single-session problem',
7.2, and 7.10 'Ranking by the index is asymptotically optimal for the relaxed problem as N grows'. 'Keep while nu >=
lambda' is exact for the single session; an eviction order is not a single-session object. For the relaxed
many-session problem the common threshold is exactly optimal at every N. Ranking is the hard-capacity rule, and it is
for the hard-capacity problem that it is asymptotically optimal: for one pooled falling-hazard law this is the
time-to-live approximation of LRU (Che et al. 2002; Fricker, Robert and Roberts 2012; Jiang, Nain and Towsley 2018);
in general it needs the global-attractor condition of Weber and Weiss (1990), not verified here. Whether LRU is
exactly optimal at finite N for one pooled falling-hazard law was not settled.


### R7 Price-free index with the queue: one formula for idle prefixes, staged reads and queued turns

Status: proved, with the assumptions stated.

**Statement.**

A clean prefix of L tokens is in host memory. Its session returns after a remaining gap R (law F(.|a); R = 0 if the
turn has arrived) and is admitted W seconds after it returns (W independent of R, mean w): the prefix is next needed
on the GPU at T = R + W. If the copy is dropped, the turn restores it at its arrival; this costs K0, free of lambda
(K0 = kappa*L/beta + c*E[(r - W)^+], plus p_fail*mu*gamma*L if the re-read can fail; r = wait in the device's line +
L/beta + lags), and the restored copy occupies L tokens for rho seconds before admission (the re-read's own hold: from
dispatch to admission). Let g(a) = inf_h tau(h|a)/F(h|a), the expected holding time per return over the best horizon:
g = 0 for an arrived turn, R0 for an exact hint, 1/hazard for falling hazards, E[R] for rising hazards, inf for an
ended session. Claim: keeping (to some horizon, and through the queue once the turn arrives) beats dropping now iff
lambda < nu(a) = (K0/L)/(g(a) + w - rho)^+, with nu = inf when g + w <= rho. Special cases. Queued turn: nu_q =
[kappa*L/beta + c*E[(r - W)^+]]/(L*(W - rho)^+). Idle prefix, no queue, falling hazard: nu = (K0/L)*eta/(1 - rho*eta)
(the paper's eq. retain is this with K0 = rho = r). For equal K0/L, w and rho: nu_staged <= nu_queued, with equality
only if g = 0.

**Proof.**

The policy 'keep until horizon h; if the turn arrives by then keep until admission; otherwise drop and restore at
arrival' costs lambda*L*[tau(h|a) + F(h|a)*w] + (P(R < inf | a) - F(h|a))*(K0 + lambda*L*rho). Dropping now costs P(R
< inf | a)*(K0 + lambda*L*rho). The difference (drop now minus keep to h) is F(h|a)*K0 - lambda*L*[tau(h|a) +
F(h|a)*(w - rho)]. It is positive for some h iff lambda < (K0/L)*sup_h F/(tau + F*(w - rho)) = (K0/L)/(inf_h tau/F + w
- rho), since x -> 1/(x + w - rho) is decreasing. This is R1 for the cost function with restore cost K0 + lambda*L*rho
and holding time extended by the queue wait, so first passage, consistency and indexability carry over (the drop set
{a : nu(a) <= lambda} is monotone in lambda). After arrival g = 0 and the remaining wait only shrinks, so nu_q rises
with time: a turn kept at arrival stays kept at a constant price. 'Fits before admission': if r <= W almost surely the
numerator is the device price alone; otherwise the stall term enters; if W <= rho nothing is freed and nu = inf.

**Numerical check.**

queued_staged.py (output queued_staged.out): 4,000 random cases (5 laws; L in 2,000 to 60,000; K0 in 0.05 to 5 s; w in
0 to 40 s; rho <= min(w, 3 s); lambda at 0.5, 0.9, 1.1 and 2 times the index): keeping is better iff lambda < nu in
all 4,000. extra_checks.py c: restore's own hold with no queue, 2,000 random (K0, rho, lambda) on the tool law (g =
2.556 s): 0 violations of nu = (K0/L)/(g - rho)^+. Numbers at L = 25,000, rho = 1 s: nu_q = 4.7e-5, 1.2e-5, 5.2e-6,
2.5e-6, 1.2e-6 x kappa for W = 2, 5, 10, 20, 40 s; g on the tool law is 2.6, 4.2, 11.9, 45.5, 169 s at idle ages 0, 2,
10, 60, 300 s.

**What it checks in the first draft.**

4.1 'a queued turn at position k: T is nearly known (now + W_k): nu = K/(L W_k): farthest from admission first, and
only while its re-read fits before admission', with section 2's K_read as K. Differences. (1) K_read contains lambda L
r, so 'nu >= lambda' has lambda on both sides; the price-free form moves the read's own hold into the denominator (W_k
- rho instead of W_k). Section 6's 'lambda = the index of the marginal holder' is well defined only with this form.
(2) A parked read holds no pages (the engine allocates at dispatch), so rho is the time from dispatch to admission,
not the whole r. (3) 'Farthest from admission first' holds for equal K0/L. (4) The same formula ranks staged reads (g
> 0). 5.1's class order 'staged reads of sessions not yet back, then queued turns' and its scores per token (not per
token-second) do not follow from it: at kappa = 1 a queued turn 40 s from admission has nu = 1.2e-6 and yields before
a staged read idle 2 s on the tool law with w = 5 s (nu = 5.7e-6). 5.1's staged-read score K_read F_j(r_j | t)
presumes a re-staged read; for falling hazards the re-read is at arrival and the stall is (r - W)^+ times P(return).
(5) At c = 0 the 'fits' condition is not needed for the sum; it protects the turn's own wait.


### R8 Retain when the dropped prefix can be fetched back early (K replaced by the optimal Fetch cost)

Status: proved, with the assumptions stated.

**Statement.**

Let Phi(a) be the optimal priced cost of a session of idle age a whose prefix is on the SSD only (the value of section
3's Fetch problem, with the read at arrival as recourse), and K' = K0 + lambda*L*rho the cost of a read made at
arrival. With early re-fetch the holding problem is V(a) = min(Phi(a), inf_h [lambda*L*tau(h|a) + (1 -
F(h|a))*Phi(a+h)]): keeping until a+h beats dropping iff Phi(a) - (1 - F(h|a))*Phi(a+h) > lambda*L*tau(h|a). (i) If
the optimal Fetch from every age is 'read at arrival' (falling hazard, or a read hidden by the queue), then Phi(a) =
K'*P(return | a) and the test is the index of R7 exactly. (ii) Exact hint R0 whose re-read fits: Phi = kappa*L/beta +
lambda*L*rho, and the index is (kappa/beta)/(R0 + W - rho): the order is still farthest next use first, but only
device time is at stake; with a free device every prefix whose re-read fits has index 0. (iii) Always Phi(a) - (1 -
F(h|a))*Phi(a+h) <= K'*F(h|a): early re-fetch can only lower the value of keeping, so the index with re-fetch is at
most the index with K paid at arrival.

**Proof.**

Structure: after a drop the best continuation is the Fetch problem, so the terminal cost of stopping at age s is
Phi(s) instead of K*P(return | s); the rest of R1's argument is unchanged. (i) Phi(a) - (1 - F(h|a))*Phi(a+h) =
K'*[P(R < inf | a) - P(h < R < inf | a)] = K'*F(h|a). (ii) Keeping costs lambda*L*(R0 + W); dropping and reading back
just in time costs kappa*L/beta + lambda*L*rho; equate. (iii) From age a the policy 'do not read; read at arrival if
it comes within h; at a+h follow the optimal Fetch policy' is admissible and costs F(h|a)*K' + (1 - F(h|a))*Phi(a+h)
(nothing is held while waiting); Phi(a) is at most that. Monotonicity of the keep set in lambda (indexability with
re-fetch) is not proved in general.

**Numerical check.**

refetch.py (output refetch.out): brute force over all (keep horizon, read start) pairs on a 0.25 s grid, L = 25,000,
read 2 s, 5 laws x 3 price settings. Falling-hazard laws (tool entry, lognormal median 5 s sigma 1.5): generalised
index / simple index = 1.000 in all 6 cases. Sharp hints (uniform [40, 50], lognormal median 40 s sigma 0.1, point 40
s) with stall weight 1: free device: generalised index 0 (never keep) against 1.9e-6 to 2.1e-6 for the simple index;
device cost 0.5 s per read: ratio 0.20 to 0.26, close to kdev/(L*(E[R] - r)) (exact for the point hint: 5.263e-7).
With stall weight 0 the two indices agree in all cases. The keep set was an interval in the price in all 15 cases.

**What it checks in the first draft.**

4.1 uses one K 'when the session returns' and 3.3 says 'Reading at the turn's end is the same as not evicting, so on
such keys Fetch before arrival is empty and the decision is Retain's'. Confirmed for falling hazards: the index with K
paid at arrival is exact. For sharp hints 4.1's index overstates the value of keeping: the exact-hint bullet 'nu =
K/(L R0)' is right only with K = the device price of one read (plus stall if the re-read does not fit), so with a free
device a far prefix should be evicted and read back just in time.


### R9 Value of a write; at a constant device price a write is never better early than late

Status: proved, with the assumptions stated.

**Statement.**

Per page of L tokens in host memory at idle age a. Dirty (no complete SSD copy): losing the host copy costs K_d =
mu*gamma*L at the return. Clean: K_c = min(K_read, K_d) (restore by the cheaper). A write costs w = kappa*L/beta (plus
SSD rent sigma) and is instantaneous here. V0(K; a) = min_s [lambda*L*tau(s-a|a) + K*P(s-a < R < inf | a)] is the
optimal cost of one gap with restore cost K and no write. (i) Value of a write against never writing in this gap:
D0(a) = V0(K_d; a) - V0(K_c; a) - w = integral_{K_c}^{K_d} P_pay(K; a) dK - w, where P_pay(K; a) is the probability
that a page with restore cost K is dropped before the return and the session returns, under its own optimal rule.
Hence (K_d - K_c)*P_pay(K_d) <= V0(K_d) - V0(K_c) <= (K_d - K_c)*P_pay(K_c). (ii) D0 > 0 is necessary for the write to
be worth making, not sufficient for making it now: for any h such that a clean page would still be held at a+h, 'hold
h, then write if the session is not back' costs exactly w*F(h|a) less than 'write now'. With many gaps (the page stays
dirty across returns until written or until the session ends) the saving is w*F(h|a)*(1 - q'), q' the probability that
the page must be written later anyway; it is still >= 0.

**Proof.**

(i) V0(K; a) is a minimum over s of functions linear in K, hence concave in K, with derivative P(s*(K) - a < R < inf |
a) = P_pay(K; a) wherever the minimiser is unique (envelope theorem); integrate from K_c to K_d. A costlier page is
kept longer, so P_pay is non-increasing in K, which gives the two bounds. (ii) Let s_c be the clean drop age and a + h
<= s_c. Write now: w + V_c(a) = w + lambda*L*tau(h|a) + (1 - F(h|a))*V_c(a+h), because holding until a+h is optimal
for the clean page. Hold h, then write: lambda*L*tau(h|a) + (1 - F(h|a))*(w + V_c(a+h)). The difference is w*F(h|a).
With many gaps a return before the write postpones the write with probability q' instead of cancelling it: the saving
is w*F*(1 - q').

**Numerical check.**

write_model.py A (output write_model.out). Identity: tool-like lognormal with P(ended) = 0.10, lambda = 2e-5, K_c/L =
7e-5, K_d/L = 4e-3: V0(K_d) - V0(K_c) = 2.281441e-4; integral of P_pay over K (501-point trapezoid) = 2.281384e-4;
bounds 6.87e-5 and 2.11e-3. Three-state backward induction (hold / write / drop) on 6 laws x 60 random (K_c, K_d, w,
lambda): 'write, then keep holding' strictly optimal in 0 of 360 cases.

**What it checks in the first draft.**

4.2 'A dirty page has restore cost K_rec, a clean one K_read. The write buys the difference, and only if the host copy
goes before the page's next use: value of writing page p now = V_dirty(a) - V_clean(a) - kappa L_p/beta - sigma(...)',
and 7.3 'the value of a write'. The sentence is exact (the integral of pay probabilities). The formula is the value of
writing at some time. As the value of writing NOW it is wrong: if V_dirty excludes a later write it credits the write
made now with a gain a later, cheaper write would also get; if V_dirty includes the later write the expression is <= 0
always and 0 exactly when writing now is optimal. Also K_clean should be min(K_read, K_rec): on an idle FA3 GPU (mu =
1) recompute is the cheaper restore, both values coincide and the write is worth nothing at current prices.


### R10 The two lives of a dirty page; when skipping a write is optimal

Status: proved, with the assumptions stated.

**Statement.**

Model of R9, constant prices. pi(s) = P(G < inf | G > s). From the dirty state the optimal policy is one of two lives:
(W) hold until a_w, the first age at which the index with restore cost K_c + w falls to lambda, then write and drop at
once; (N) hold until a_d, the first age at which the index with restore cost K_d falls to lambda, then drop unwritten.
With C_W(s) = lambda*L*I(s) + (K_c + w)*(S(s) - p_inf) + w*p_inf and C_N(s) = lambda*L*I(s) + K_d*(S(s) - p_inf):
V_dirty = min(min_s C_W, min_s C_N), and W is chosen iff V0(K_d) - V0(K_c + w) >= w*p_inf, with V0(K) = min_s
[lambda*L*I(s) + K*(S(s) - p_inf)] (from age a: condition on G > a and replace p_inf by 1 - pi(a)). With theta =
w/(K_d - K_c): pi(a_w) >= theta is necessary for W and pi(a_d) >= theta is sufficient. Consequences. (1) Ended session
(pi = 0): never write; drop at once. (2) No end mass: write iff K_d > K_c + w, per token mu*gamma > K_c/L +
kappa/beta. (3) A page that returns before a_w is not written in this gap. (4) Dependence: a larger mu raises K_d
(towards W); a larger kappa raises w and K_c (towards N, and a later a_w); a larger lambda makes pages leave younger,
when pi is higher (towards W); a law with a long dead tail lowers pi at the drop age (towards N). (5) With a write of
duration l_w the write starts at a_w - l_w and is paid with probability S(a_w - l_w) instead of S(a_w).

**Proof.**

A policy is a pair (u, s): optional write age u <= s and drop age s. If u < s the cost is lambda*L*I(s) + w*S(u) +
K_c*(S(s) - p_inf) >= lambda*L*I(s) + w*S(s) + K_c*(S(s) - p_inf), the cost of writing at s: writing strictly before
the drop is never better. Without a write the cost is C_N(s). With the write at the drop it is lambda*L*I(s) + (K_c +
w)*S(s) - K_c*p_inf = C_W(s). Minimising the lower envelope min(C_W, C_N) over s gives min(min C_W, min C_N). C_W is
the holding cost of R1 with restore cost K_c + w plus the constant w*p_inf; C_N is the holding cost with K_d; R1 gives
a_w and a_d and the criterion. Thresholds: at a common age s, C_W(s) <= C_N(s) iff w*S(s) <= (K_d - K_c)*(S(s) -
p_inf) iff pi(s) >= theta. If W is optimal, C_W(a_w) <= C_N(a_d) <= C_N(a_w), so pi(a_w) >= theta. If pi(a_d) >=
theta, min C_W <= C_W(a_d) <= C_N(a_d). (2): p_inf = 0 and V0 is non-decreasing in K.

**Numerical check.**

write_model.py A: backward induction against min(min C_W, min C_N) on 360 random cases: 0 value mismatches (worst
relative error 5.8e-14), 0 cases with the wrong life. write_model.py C, tool law with P(ended) = 0.05, K_c/L = w/L =
kappa/beta, K_d/L = mu*gamma: lambda = 1e-6: mu = 1: recompute cheaper than a read, never write; mu = 2: hold 42.7 s,
drop unwritten; mu = 20 or 94: hold 45.1 s, write, drop (P(return) at the write 0.39); mu = 94, kappa = 30: hold 280.7
s, drop unwritten (P(return) 0.04). lambda = 1e-7, mu = 20: hold 376 s, drop unwritten (P(return) 0.02); mu = 94:
write at 152 s. lambda = 1e-4, mu = 94, kappa = 30: write at 17.9 s. A clean page at lambda = 1e-6 is kept 27.6 s: the
lazy dirty page stays longer (45.1 s) because a return also saves the write.

**What it checks in the first draft.**

4.2 'A page whose session has ended is never worth writing' (holds: pi = 0), 'A page that stays in memory until its
next use is not worth writing yet' (holds: consequence 3 and R9 ii), and 4.1 'at saturation the only copy of a page is
worth mu gamma L and is kept; a copy the SSD also holds is worth a read'. The last needs a change: the only copy is
not kept unconditionally; its index is K_d/K_c times the clean one (86/kappa at saturation) and when it falls to
lambda the page is written and dropped, or dropped unwritten if pi < theta. 5.1's 'never dirty pages' contradicts 4.2
for ended sessions: their dirty pages are first out.


### R11 Due date of a write and the cost of missing it

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

Let d_w be the first idle age at which the page's clean index falls to lambda (the age at which a clean copy would be
given up). If the page is still dirty and held at age d_w + x, the extra cost of the delay, per second and given the
session is not back, is L*(lambda - (K_c/L)*eta(d_w + x)); for falling hazards this is L*(lambda - nu_clean(d_w + x)).
It is 0 at the due date, rises with the lateness and tends to lambda*L only as the hazard dies. The cost of being x
late is integral_0^x [S(d_w + y)/S(d_w)]*L*(lambda - (K_c/L)*eta(d_w + y)) dy, which is o(x): the due date is soft.
Under lazy writing in one gap the page is held to a_w >= d_w (R10); with many gaps the due date lies in [d_w, a_w] and
equals d_w when a return only postpones the write.

**Proof.**

C_c(s) = lambda*L*I(s) + K_c*(S(s) - p_inf) has derivative S(s)*(lambda*L - K_c*eta(s)); holding from d_w to d_w + x
instead of dropping at d_w costs C_c(d_w + x) - C_c(d_w), which divided by S(d_w) is the stated integral. For falling
hazards nu_clean = (K_c/L)*eta (R4 b) and nu_clean(d_w) = lambda, so the integrand starts at 0. a_w >= d_w because
indices are proportional to the restore cost and K_c + w >= K_c. If no clean page can be given up and a read needs the
room, the cost is that read's delay (weight c) or a refusal (mu*gamma per token), which is what lambda measures on
average.

**Numerical check.**

write_model.py B, tool law, K_c/L = 7e-5. lambda = 5e-6: due at 12.7 s of idle; 1, 5, 20, 60, 300 s late costs 0.026,
0.095, 0.168, 0.150, 0.059 of lambda x lateness per token, and the rate at those times is 0.05, 0.22, 0.51, 0.74, 0.92
of lambda. lambda = 2e-5: due at 1.69 s; 1, 5, 20 s late costs 0.174, 0.256, 0.205 of lambda x lateness; rate 0.30,
0.61, 0.83 of lambda. extra_checks.py e: write-before-read comparison in four price settings.

**What it checks in the first draft.**

4.2 'Its write becomes due at d_w = the first time its clean index nu_clean falls below lambda ... Before d_w the
write costs only device time; after it every second costs about lambda L_p (room held only because the page is dirty),
or a refusal if no clean page is left to give up', and 5.2 'writes are due at d_w and cost lambda L_p per second
late'. The definition and the first half hold. The rate lambda L_p is an upper bound reached only for a dead page: the
rate is L_p (lambda - nu_clean). So the due date is soft, the Smith ratio of a late write is beta*(lambda - nu_clean)
and not beta*lambda, and a late write goes before a late read of L_read tokens iff (lambda - nu_clean)*L_read > c.


### R12 Order of writes on one device

Status: proved, with the assumptions stated.

**Statement.**

(i) Model M1: one device; writes of equal size (one page or one page group, one time slot each, preemption only
between pages); page j exists from r_j and has due date d_j; finishing it at C_j costs f(C_j - d_j), with ONE function
f for all pages, non-decreasing, convex, zero for x <= 0. Then serving in every slot the released page with the
earliest due date (EDD) minimises sum_j f(C_j - d_j), whether or not the due dates can be met. (ii) Feasibility: if
some schedule meets every due date, EDD does (Jackson 1955; with release times and preemption, Horn 1974), at zero
lateness cost. (iii) Model M3: all pages overdue, sizes L_j, constant cost rates L_j*(lambda - nu_j): the optimal
order is by Smith's ratio beta*(lambda - nu_j), lowest clean index first; the size cancels. (iv) 'Lowest clean index
first' is the due-date order iff index curves do not cross. With one pooled falling-hazard law and equal K/L both are
'oldest turn end first', the order of a FIFO write queue, and f is convex (its slope L*(lambda - nu_clean) rises). (v)
Outside these models (page-specific lateness shapes; unequal sizes with due dates) no order rule is optimal: weighted
tardiness on one machine is NP-hard (Lenstra, Rinnooy Kan and Brucker 1977).

**Proof.**

(i) Exchange. Idling cannot help since f is non-decreasing. Take an optimal non-idling schedule that serves j in the
slot ending at t1 while a released page i with d_i < d_j is served in the slot ending at t2 > t1. Swap them (both are
released by then). New cost minus old cost = [f(t2 - d_j) - f(t1 - d_j)] - [f(t2 - d_i) - f(t1 - d_i)]. Both brackets
are increments of f over an interval of length t2 - t1; the second starts further right (t1 - d_i > t1 - d_j), so by
convexity it is at least the first: the swap does not raise the cost. Repeating gives EDD among released pages. (ii)
Classical. (iii) Sum of w_j*C_j with w_j = L_j*(lambda - nu_j) and processing time p_j = L_j/beta: Smith's rule orders
by w_j/p_j = beta*(lambda - nu_j). (iv) d_j is the first time nu_j(t) reaches lambda; if the curves never cross, the
order of the current indices is the order of the first passages. Under a pooled law nu_j(t) = nu(t - e_j) with nu
decreasing, so both orders are by turn end e_j, and f'(x) = L*(lambda - nu(a* + x)) is non-decreasing. Counterexample
with crossing curves: A has index 5 lambda now and is due in 1 s; B has index 2 lambda and is due in 10 s; each write
takes 1 s. Lowest index first finishes B at 1 s and A at 2 s (A is 1 s late); EDD finishes A at 1 s and B at 2 s
(nothing late).

**Numerical check.**

write_order.py (output write_order.out), brute force over all 720 priority orders of 6 unit pages with random release
times and due dates, 400 instances each: one convex shape (x^+)^2: EDD suboptimal in 0; one linear shape x^+: 0; one
concave shape 1 - exp(-x^+/2): 7, worst ratio 1.37; page-specific slopes w_j in {1, 5}: 11, worst ratio 5.0. M3 (sizes
64 to 8,192 tokens, 300 instances, all 720 orders): lowest-index-first suboptimal in 0. Stochastic check where a
return cancels the write (3 pages, falling hazard, exact dynamic program, 300 instances): oldest-first suboptimal in
0; this case is numerical only.

**What it checks in the first draft.**

4.2 'So writes are jobs with due dates, like reads: write in the order in which pages will be evicted (lowest clean
index first), and no later than d_w. With an idle device, write at once (it is free); with a busy device, order all
transfers by due date' and 7.3 'writing in eviction order'. Holds under M1 (cost) or as feasibility; among overdue
pages it is the lowest-index order (M3). The parenthesis equates two orders that differ when index curves cross: the
rule is due-date order. Under LRU-like eviction the engine's FIFO queue is already this order, so candidate change R2
changes nothing unless eviction becomes class-aware. 'It is free' needs sigma = 0 and a write that cannot delay a read
arriving during it (one 128-page batch is about 0.38 s of device time).


### R13 When eager writing is optimal; this workload

Status: proved, with the assumptions stated.

**Statement.**

Let kappa_now be the device price when the page is born, kappa_due the expected price when its write would be due, q
the probability that the write is ever needed (the host copy is given up while the session is alive), eps the
probability that the page is forced out of memory before a deferred write completes, pi the chance of return then,
sigma = 0. To first order (ignoring the extra rent of the lazy page), writing at birth is optimal iff kappa_now*L/beta
<= q*kappa_due*L/beta + eps*pi*(K_d - K_c). (i) Idle device (kappa_now = 0): eager is weakly optimal. (ii) Constant
price, no forced evictions: lazy is weakly optimal and saves the share 1 - q of the write volume. (iii) At saturation
K_d - K_c = (mu*gamma - kappa/beta)*L, so with kappa_now = kappa_due = kappa eager wins as soon as eps*pi >= kappa*(1
- q)/(mu*gamma*beta - kappa); mu*gamma*beta = 86 at mu = 93, so the threshold is 0.004 for kappa = 1 and 1 - q = 0.35,
and 0.046 for kappa = 10. (iv) For a host time-to-live of d seconds, q = (1 - e)*x/(x + e - x*e) over all pages (x/(x
+ e - x*e) for pages of a turn that is not the last), x = P(gap > d) per gap, e = probability that a turn is the
session's last. (v) With a price that switches between idle and busy, the optimum writes whenever the device is idle
and, while it is busy, waits until the due date forces the write.

**Proof.**

Eager pays kappa_now*L/beta now. Lazy pays kappa_due*L/beta with probability q, and with probability eps the only copy
is lost before the write, which costs K_d - K_c more at the return (probability pi). Compare. (i), (ii), (iii) are
substitutions; (ii) is R9 (ii). (iv) After each turn the session ends with probability e; otherwise a gap follows,
longer than d with probability x (the page is then due), else the session returns and the argument repeats: P = (1 -
e)*[x + (1 - x)*P]. (v) is the backward induction with the device state added: in the idle state the write is free, so
delaying it gains nothing (R9 ii with w = 0); in the busy state waiting has the option value of an idle moment before
the due date.

**Numerical check.**

write_model.py D (tool law with P(ended) = 0.05, lambda = 5e-6, K_c/L = 4.7e-5, K_d/L = 4e-3): born with the device
idle: the optimum writes at once and equals eager; born busy, busy spells of 2, 5, 20 s (busy 20, 60, 90 % of the
time): waiting saves 1.00, 0.99, 0.89 of one write; with 20 s spells the write starts at age 21.95 s even if busy.
write_model.py E (many gaps, tool 92 % / human 8 %, e = 1/29.2, fixed point over continuation values): lambda = 2e-6,
mu = 94, kappa = 1: lazy writes at 32.8 s into a tool gap and 1.1 s into a human gap, 80 % of pages are ever written,
eager costs 0.188 of a write more per page; where the clean keep time is 0 (lambda = 2e-5, kappa = 1) lazy and eager
coincide. extra_checks.py d: Monte Carlo of (iv), 200,000 pages: x = 0.046, 0.067, 0.103 give 0.564, 0.652, 0.744
against 0.565, 0.654, 0.744 from the formula. numbers_c34.py 5: for a host stay of 30 to 60 s, x = 0.046 to 0.103
(as-replayed and trace-wide gap laws), so 57 to 74 % of pages are ever due; measured 56 to 67 % read back.

**What it checks in the first draft.**

4.2 'On this workload eager writing is right for live sessions: 56 to 67 % of prompt positions are read back from the
SSD, 4.2 to 4.5 times each'. The conclusion stands; the reason does not. The read-back share is q, the fraction of
writes ever needed (a deferred write would skip the rest, about a third); '4.2 to 4.5 times each' shows the write is
worth making (one write replaces that many recomputes by reads), not that it should be early. Eager is right because
device time is nearly free at both ends (idle device at light load; at saturation c is near 0 and therefore kappa, the
lateness cost of waiting transfers, is near 0) and because losing an only copy costs 86/kappa writes, so a 0.4 %
chance of a forced eviction is enough. At light load on FA3 a write has no value at current prices (recompute at mu =
1 is cheaper than a read); it is an investment for later load.


### R14 Born clean, clean only on a full acknowledgement, and revocation

Status: proved, with the assumptions stated.

**Statement.**

(i) Born clean: if every page of the chain from the resident anchor down to page p is on the SSD and complete, p's
restore cost is already K_c, so the value of writing it again is 0 (R9 with K_d replaced by K_c) while the write costs
kappa*L_p/beta and holds room while it waits: skip it. (ii) The condition is reachability, not the existence of p's
own key: with a lookup that stops at the first missing page, a stored page below a deleted ancestor cannot be read, so
it is not clean; the missing ancestors must be written, after which the stored descendants are clean again without
being rewritten. (iii) Clean is revocable: if the cleaner deletes p's SSD copy or an ancestor's while the host copy
exists, p is an only copy again (restore cost K_d). (iv) Counting a page as clean before the whole write is
acknowledged prices an only copy at K_c instead of K_d: its index is understated by K_d/K_c (86/kappa at saturation)
and the expected loss per page is (K_d - K_c)*P_pay.

**Proof.**

Keys are chained: key_j = H(key_{j-1}, tokens of page j), so equal keys mean equal token prefixes, and in a causal
model the KV of a position depends only on the tokens up to it; a stored page with the same key can therefore replace
the recomputed one. (i) is R9 (i) with equal restore costs: the integral over an empty range is 0. (ii) and (iii)
follow from the lookup semantics (hicache_nixl stops at the first missing key) and the definition of clean as 'restore
cost K_c'. (iv) is R1 with the wrong K: the page is dropped as soon as (K_c/L)*sup F/tau <= lambda although its true
index is K_d/K_c times larger; by R9 (i) the loss is the integral of P_pay between the two costs, at least (K_d -
K_c)*P_pay(K_d).

**Numerical check.**

No separate computation: definitions plus R9. The reachability facts are checks S1 and S2 of
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_retain/ssd_chain.py; the factor 86/kappa is
numbers_c34.py item 1 (mu*gamma*beta with mu = 93, gamma = 0.043 ms, beta = 21,400 tokens/s).

**What it checks in the first draft.**

4.2 'A page already on the SSD (same chained key) is clean at birth: a recomputed page is not written again' and 'A
page counts as clean only when the SSD has acknowledged the whole write'. Both hold. Add: clean means the whole chain
above the page is stored (reachable), and a page stops being clean when the cleaner deletes its copy or an ancestor's;
today the tree never learns of deletions, so a host copy can be evicted as clean after its SSD copy is gone.


### R15 One tier up: the backup GPU to host is a job due at GPU eviction (the mirror)

Status: proved, with the assumptions stated.

**Statement.**

A page is on the GPU until time d_G, when the GPU gives it up (the engine chooses d_G). A host copy made at b <= d_G
costs lambda*L*(d_G - b) of rent before d_G and changes no other cost, because every use of the page before d_G is
served by the GPU copy. Hence: (i) for lambda > 0 and a copy cost that does not depend on b, the optimal backup time
is the latest feasible one, d_G - L/beta_GH (beta_GH = GPU-to-host copy rate); (ii) the index of an existing mirror
copy is nu_m = (K_b/L)/(d_G - now - L/beta_GH)^+, where K_b is the priced cost of making the copy at d_G instead;
(iii) K_b = 0 if the late copy does not delay the scheduler: then nu_m = 0 and the mirror is first out together with
ended sessions; K_b = mu*L/beta_GH if the copy blocks the scheduler: then the eager backup is optimal whenever
lambda*(d_G - now) < mu/beta_GH; (iv) GPU eviction itself is R1 with the GPU's price of memory and K = the load time
when a host copy exists (K_b when none does).

**Proof.**

Take any policy and any sample path. Compare it with the same policy in which the host copy of the page is made at d_G
- L/beta_GH instead of b. Until d_G every read of the page uses the GPU copy in both; from d_G on the host copy exists
in both. TTFT is identical on the path; rent is lower by lambda*L*(d_G - L/beta_GH - b). This gives (i). (ii):
dropping the mirror now and copying again at d_G saves lambda*L*(d_G - now - L/beta_GH) of rent and costs K_b; equate.
(iii): substitution. (iv): the GPU tier has the same structure (rent on GPU tokens, a restore cost at the next use).

**Numerical check.**

numbers_c34.py item 4 (arithmetic on the brief's constants; nothing measured here): mirror 0.65 M of 1.63 M tokens;
without it the host-only room grows from 0.92 M by a factor 1.71; the mirror's rent is 0.15 s per request at lambda =
5e-7, 0.59 s at 2e-6 (2.2 requests/s); a copy of 24,900 tokens takes about 70 ms at the one measured rate (2,304
tokens in 6.5 ms), 0.15 of one copy stream at 2.2 requests/s; if it blocked the scheduler it would cost mu*L/beta_GH =
6.6 TTFT-seconds per request, i.e. nu_m = 2.6e-4/(seconds to GPU eviction) per token against lambda of order 1e-6.

**What it checks in the first draft.**

4.1 'the mirror ...: nothing reads it while the GPU holds the page, so its nu is that of the copy being needed after
the GPU gives the page up', 4.3 'GPU to host: the backup is due when the GPU gives the page up. Write-through makes it
at the end of prefill, so the mirror holds 0.65 M of 1.63 M host tokens for nothing ... dropping mirror copies once
their SSD write is acknowledged, or backing up at GPU eviction, pays lambda only when it buys something', and 'The
same index orders GPU eviction'. Holds in the priced model if the late backup is asynchronous. The bullet should state
the index, (cost of a late backup)/(L x time to GPU eviction). 'For nothing' is false if the backup at GPU eviction
runs inline, as the existing write_back path does: at mu = 93 one blocked copy of a 24.9 K history costs 6.6
TTFT-seconds, more than its rent over an 8 s decode for any lambda below 3e-5. Dropping the mirror after the SSD
acknowledgement and copying again at the turn's end makes two copies per turn; one backup at the turn's end, issued
like today's end-of-prefill copies, makes one.


### R16 One tier down: SSD capacity with chained pages (tail before head; the index of a shared head)

Status: proved, with the assumptions stated.

**Statement.**

Lookup: it starts just below the session's resident anchor (the deepest page still in GPU or host memory) and reads
stored pages down the chain until the first missing one. (i) One chain, pages i < j, lookups with no resident anchor
inside [i, j): if page i is deleted and page j kept, deleting j and keeping i instead retrieves at least as much on
every lookup. For lookups from the root, among all ways to delete m pages of a chain of n the tail is the unique
optimum and loses exactly m pages. (ii) With anchors between i and j the exchange can fail: deleting page j from a
full chain loses P(anchor above j)*(n - j + 1) pages per lookup, which is smallest at the head when the head is
usually resident. (iii) Tree of page groups v (size len_v), root-anchored lookups, session s with lookup rate rho_s,
theta = mu*gamma - (c + kappa)/beta the value per token read instead of recomputed: for every SSD price sigma per
token-second the optimal stored set is {v : theta*rho(v) > sigma}, rho(v) = sum of rho_s over the sessions whose chain
passes through v. The sets are nested in sigma, so deleting in increasing theta*rho(v) is optimal for every capacity
equal to a threshold set's size (and for the divisible problem in between); rho does not rise down a chain, so a tail
goes no later than its head; an ended session contributes 0. With time-varying return laws rho_s is replaced by the
session's hazard of a cold return, and the hazards of independent sessions add. (iv) Shared head h that is sometimes
resident: with rho_s^cold the rate of lookups of s that find h not resident and n_s the stored tokens of s below h,
the SSD copy of h is worth theta*sum_s rho_s^cold*(len_h + n_s) per second, an index theta*sum_s rho_s^cold*(1 +
n_s/len_h) per token; it is 0 if h is never cold.

**Proof.**

(i) A lookup whose anchor is at or below j needs neither page. A lookup whose anchor is above i stops, when i is
deleted, at or before page i - 1; after the swap the pages before i are unchanged and i is present, so it reads at
least as far. Uniqueness: a deleted set D with m pages has min D <= n - m + 1; a root lookup loses n - min D + 1 >= m
pages, with equality iff D is the last m pages. (ii) A lookup with its anchor above j, on a chain complete except for
j, loses pages j to n. (iii) For root-anchored lookups a page below a missing ancestor is never read, so the stored
set may be taken ancestor-closed without loss; then value(S) - sigma*size(S) = sum over v in S of len_v*(theta*rho(v)
- sigma). Every lookup through a child passes through its parent, so rho does not rise down a chain and {theta*rho >
sigma} is ancestor-closed; it maximises the sum term by term. A set that maximises value - sigma*size is optimal among
all sets of at most its size (Everett 1963). (iv) Lookups of s start at the root at rate rho_s^cold (they need h and
then the chain) and below h otherwise (they do not use h's SSD copy); the difference of values with and without h on
the SSD is the stated sum.

**Numerical check.**

ssd_chain.py (output ssd_chain.out). S1: chains of 2 to 12 pages, every deletion count, all subsets: the tail is the
unique optimum, 0 violations. S2: chain of 8 pages, one page to delete: lookups from the root: loss 8, 7, ..., 1 by
deleted page (delete the tail); anchor uniform: 1.00 at both ends, up to 2.50 in the middle; head usually resident
(root lookups 2 %): deleting page 4 loses 0.10 against 1.00 for the tail. Exchange under the stated condition: 7,450
random checks, 0 violations. S3: 200 random trees of 4 to 10 page groups, all subsets: the threshold set is optimal
for its price and for its size in every case, sets nested, no child with a higher rate than its parent. S4: marginal
value of a shared head, 500 random cases, 0 mismatches; 5 sessions with 20 K chains below a 6 K head: head index /
tail index = 21.7, 6.5, 1.08, 0 when the head is cold on 100 %, 30 %, 5 %, 0 % of lookups.

**What it checks in the first draft.**

4.3 'SSD capacity: the cleaner is eviction on the SSD, by the same index with K = K_rec for the whole chain below a
page: ended sessions' chains first, a chain's tail before its head (a lookup stops at the first missing page), a head
shared by live sessions last'. Proved for lookups that start above the pages concerned. Changes: the head's index is a
sum of rates over the sessions below it, with each session's stake the head plus its stored chain, not one K_rec; 'a
shared head last' is wrong for a head that is always resident, whose SSD copy is worth nothing: the cleaner may delete
it before any tail, provided the cache then treats the host copy as an only copy (R14 iii); 'ended sessions' chains
first' applies to the part no live session shares.


### Further results proposed by this check

- Hull theorem as the paper's Retain statement. Plot each hint law as the curve (E[min(G,t)], P(G>t)). Its lower
convex hull is the session's efficient frontier of memory-time against restore probability; the index is the hull's
slope (R3). Sketch of use: (a) one O(n) pass over the engine's 64 support points gives nu at every age; (b) the drop
age at price lambda is a tangency, so each session has a demand curve x_i(lambda) = I_i(s_i(lambda)); (c) the price
that clears a room R solves sum_i rate_i L_i x_i(lambda) = R, a closed form for section 6 in place of a learned price;
(d) Belady, LRU and 'keep to the return or not at all' are the point, convex and concave cases of the same picture.

- One index for every host holder: nu = (K0/L)/(g + W - rho)^+, K0 the price-free restore cost, g = inf_h tau/F the
Gittins holding time (0 for a queued turn, R0 for an exact hint, 1/hazard for falling hazards, infinite for an ended
session), W the queue wait, rho the restore's own hold (R7, verified on 4,000 random cases). The mirror fits too (g =
time to GPU eviction, K0 = cost of a late backup, R15). This would replace 4.1's separate bullets and 5.1's class
order by one ranking 'time to next GPU use net of the restore's own hold'. Corollary worth stating: a long queue
flattens the value of hints. On the tool law g is 2.6 s at age 0 and 11.9 s at age 10 s; with no queue the index
differs by 4.6x between those sessions, with W = 20 s by (11.9 + 20)/(2.6 + 20) = 1.41x. Only human-gap sessions (g of
100 s and more) stay clearly separated, which matches the measured ties of the hinted arms under saturation.

- A provable ceiling on what hints can buy for retention. The priced bound of R6 (weak duality over the hull curves)
is a lower bound on the miss ratio of every non-anticipative policy with a given information level. In the equal-size
toy with the trace's two gap entries and room for 25 % of the sessions: no class information 0.0936 (LRU reaches
0.0954 at 512 to 2,048 sessions, the time-to-live approximation), class known 0.0734 (the index policy reaches 0.0749
to 0.0754), exact gaps 0.0445 (clairvoyant order). At 50 %: 0.0344, 0.0325, 0.0151. Sketch: compute the same three
bounds from the replayed gaps of a campaign and report measured SSD-read shares against them; the paper can then say
how much of the gap between LRU and the oracle a distribution hint can close at all (about 22 % of misses at a quarter
of the sessions held, 6 % at half).

- The write's life in two theorems (R9, R10): (1) at a constant device price a write is never better early than late,
so the write-back point is the eviction frontier; (2) a dirty page has exactly two candidate lives, chosen by
V0(K_rec) - V0(K_read + w) >= w P(ended), with the pointwise threshold P(return) > w/(K_rec - K_read). Add the closed
form for the share of pages ever due under a host stay d: q = (1 - e) x/(x + e - x e), x = P(gap > d), e = P(a turn is
the last). It gives 57 to 74 % for 30 to 60 s against the measured 56 to 67 % read back, and predicts that a deferred
write would skip 26 to 43 % of the write volume (about 1.1 to 1.8 K of 4.3 K tokens/s). An offline count of written
pages never read back, by class, would test this prediction on the campaign 34 logs.

- A copy-down rule that explains both the eager SSD write and the wasted mirror. For a copy from tier X to the tier
below: earliness costs (price of the lower tier's space) x tokens per second plus the chance the copy turns out
unneeded times the transfer price; lateness costs the upper tier's blocked room or the loss of the only copy. With a
known due date the copy is made as late as possible when the lower tier's space or the transfer has a price, and at
once when both are free. SSD: space free until the volume fills and device nearly free, so eager is optimal. Host:
space priced at lambda, so the end-of-prefill backup is the one place the engine pays rent for nothing (0.65 M
tokens), unless the late copy would block the scheduler (R15). Sketch of proof: the sample-path argument of R15 plus
R9 (ii); with an uncertain due date it becomes the same quantile rule as Fetch (land the copy at the
lambda-versus-lateness critical ratio of the due date).

- Write priority is the eviction order with the size cancelling. The cost rate of a late write is L_p (lambda -
nu_clean) and its device time is L_p/beta, so its Smith ratio is beta (lambda - nu_clean): pages are written lowest
clean index first whatever their size, and a late write goes before a late read of L_read tokens iff (lambda -
nu_clean) L_read > c (R11, R12). This turns 5.2's comparison into one inequality and shows that the 50 K / 25 K hold
is the special case nu_clean = 0. For the cost-optimality of due-date order the exact model is equal-size pages with
one convex lateness shape (exchange proof); with returns that cancel a write the oldest-first order was optimal in all
300 small instances, and a coupling proof (older page: higher cost rate and lower chance to leave by itself) looks
reachable.

- SSD index and the hot-head exception (R16). With root-anchored lookups the index of a page group is theta times the
summed cold-return hazard of the sessions below it, so tails-before-heads and ended-sessions-first are corollaries of
one monotone quantity, and the cleaner needs only per-page session counts and hazards. A head that is always resident
has index 0 on the SSD. If the cleaner reports deletions and the tree marks the host copy as an only copy again, the
SSD copy of a hot head can be deleted first and rewritten once when the head cools (cost kappa len_h/beta), which is
the opposite of 'refresh heads when memory serves them' (R4 in section 9) and cheaper when the volume is tight. A
replay of campaign 34's 48 head-missing requests under both orders would decide between them.

- Closed-form index for lognormal hints, and what it says about the hint law. The hazard of a lognormal peaks at exp(m
+ s z*), z* the root of phi(z)/Phibar(z) = z + s (median x 0.049 for s = 1.79, x 1.76 for s = 0.5); past the peak the
index is the hazard, before it the hazard at a tangent point with a one-line equation (R4, agreement with quadrature
to 6 or 7 digits). Two consequences for the engine's two-piece form: no lognormal hint has a falling hazard at age 0,
so any myopic keep rule evicts at the turn's end (cost 1.98 against 0.73 in the tool-entry example); and the density
jumps at the median by sigma_low/sigma_high, upward by 1.44 for the human entry, which makes the index bulge between
74 and 96 s. A single-sigma fit or a smooth splice would remove the artifact. For Pareto-like tails the keep time is
linear in the restore cost per token: a* = alpha K/(lambda L) - theta.

- Calibrating lambda from the eviction frontier. For one pooled falling-hazard law, lambda = (K0/L) x hazard(T_C)/(1 -
rho x hazard(T_C)), T_C the idle age of the pages being evicted. With the trace's gap mix the pooled hazard is 0.022/s
at 30 s and 0.0099/s at 60 s, so lambda is about 9e-7 to 4e-7 TTFT-seconds per token-second at light load (K/L =
gamma) and about the same times kappa for clean pages at saturation; an only copy at mu = 93 has index 9e-5 to 4e-5 at
those ages. Campaign 26's measured value of room (0.52 s per turn for 305 K tokens at 1.1 to 1.5 turns/s) is about
2e-6 on Triton, the same order. The engine's 5e-5 is two orders higher: under the index rule it would keep no clean 25
K prefix even at the turn's end. These are inferred values (numbers_c34.py), to be checked against logged eviction
ages.

- Size lemma with an exposed fixed read lag. K_read(l) = c (V + lags + l/beta - W)^+ + kappa l/beta for the evicted
tail of l tokens is concave in l with a fixed part, so (a) the per-token index of a whole prefix falls with its length
(at light load a 2 K prefix has 6x the per-token index of a 25 K one: 5.0e-4 against 8.3e-5), (b) freeing m tokens
from one prefix is cheaper than m/2 from each of two by exactly one fixed lag times P(return), (c) at c = 0, or when
recompute is the cheaper restore (idle FA3 GPU), the index is size-neutral. This is the GreedyDual-Size effect derived
from the cost model instead of assumed, and it says leaf-first eviction should finish one session's chain before
starting another's when stalls are exposed.

- Open item worth a proof or a counterexample: indexability with early re-fetch (the keep set was an interval in the
price in all 15 cases of refetch.py, R8), and finite-N optimality of LRU under one pooled falling-hazard law with
equal sizes and costs (the simulation shows only convergence to the priced bound). Both would let section 7 claim more
than the relaxed-problem optimum.


## C. Order, Preempt, Price, the queue and the floor

Summary of the check.

Holds, with stated conditions: earliest due date minimises the maximum lateness on the device for reads and writes
together (to within one page group, about 1.5 s); the ratio greedy solves the divisible room problem; the relaxed
problem's room price is the index of the marginal holder with complementary slackness; weak duality holds in the open
model; the saturation identity holds; and born-clean or delay alone bounds the dirty set once writes are guaranteed
more than the new-token rate (4.1 K tokens/s). Fails as drafted: (1) in an open queue mu is 1 + (Q + lam x)/(1 - rho),
not 1 + Q (91 against 9 at 90 % load). (2) In the saturated loop mu is N for response time but N - B for TTFT (108,
not 128); the measured 0.73 N is that minus a transient of the pausing users, and the recorded tables of campaigns 32
and 34 close to within 1 s. (3) c = P(no other ready turn) is false for the single work queue section 1.5 describes:
there a stall-second costs 1.0 to 1.6 s up to about 94 % utilisation. It is about right for the engine's pool-bound
batch, where an empty slot costs about the turn's own second. (4) The Lagrangian bound is not a bound in the closed
loop: in a three-user example 0.2 s more GPU work lowers summed response time by 8.4 s. (5) The device fallback is
optimal only when every transfer is already late; as written it costs 2.3 times the optimum on mixed instances. (6)
Result 10 is not covered by the cited theorems for the joint design. Must change: state the model as a pool-bound
batch in a closed loop, with mu = N - B and c = P(the slot stays empty). For the saturated regime use the floor mean
TTFT >= (N - B) g0 - Z + own prefill, attained exactly when nothing is recomputed and no slot is left empty; campaign
34's arms sit about 1.6 s above it. Keep the Lagrangian as the decision rule and as a bound for the open, light-load
model only, read lambda off the lowest index still held, and on the device use earliest due date while feasible and
the index w / max(p, d - t) otherwise. 


### 1. The weight mu in an open queue (section 1.5, first half; section 6)

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

Model: one GPU server that serves one turn at a time, first come first served, without preemption; arrival times a_1 <
a_2 < ... are fixed (open queue); turn j needs s_j seconds of GPU work; W_j is its queue wait. Add delta >= 0 seconds
of work to turn k. (i) Each W_j is a convex, non-decreasing, piecewise-linear function of delta with slopes 0 or 1.
(ii) The derivative of sum_j W_j at delta = 0+ is B_k, the number of turns served after k in k's busy period (the
maximal stretch in which the server never idles). If the extra work comes before k's own first token, summed TTFT
rises at rate mu_k = 1 + B_k. (iii) For a finite injection the rise is at least delta * B_k. (iv) If arrivals are
Poisson with rate lam, work is i.i.d. with mean s_bar and rho = lam * s_bar < 1, then given Q turns waiting behind the
turn in service and x seconds of its service left when the work is added, E[B_k | Q, x] = (Q + lam * x) / (1 - rho).
Averaged over the turns of an M/M/1 queue E[B_k] = rho / (1 - rho)^2, while the mean queue length at a service start
is rho^2 / (1 - rho).

**Proof.**

Step 1 (Lindley). W_{j+1} = max(0, W_j + s_j - (a_{j+1} - a_j)). Unrolling gives W_j = max over i <= j of
[sum_{l=i}^{j-1} s_l - (a_j - a_i)], the term i = j being 0. Step 2. With s_k replaced by s_k + delta, term i has
slope 1 in delta if i <= k < j and slope 0 otherwise. A maximum of affine functions is convex and piecewise linear;
here it is non-decreasing with slopes 0 or 1. That is (i). Step 3. The right-derivative of a maximum of affine
functions is the largest slope among the terms that attain the maximum. For j > k a term with i <= k attains it
exactly when the server has not idled between the start of k and the arrival of j, that is when j is in k's busy
period. So W_j has right-derivative 1 for those j and 0 for the others; the sum is B_k. Turn k's own first token adds
1. That is (ii). Step 4. By convexity W_j(delta) - W_j(0) >= delta * (right-derivative at 0); summing over j gives
(iii). Step 5 (M/G/1). The turns served after k in its busy period are the Q waiting ones, the arrivals during the x
seconds left (Poisson with mean lam * x), the arrivals during the service of each of those, and so on. Each
first-generation turn heads a sub-busy period of mean size 1 + rho + rho^2 + ... = 1/(1 - rho), because a turn has on
average rho arrivals during its own service. Hence E[B_k | Q, x] = (Q + lam x)/(1 - rho). For M/M/1 at a service start
E[Q] = rho^2/(1 - rho) and x = s with mean s_bar, so E[B_k] = (rho^2/(1 - rho) + rho)/(1 - rho) = rho/(1 - rho)^2.
QED.

**Numerical check.**

/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/open_queue_mu.py, output
open_queue_mu.out (numpy). (1) Finite differences on one M/M/1 path (rho 0.8, 200 turns k, delta 1e-7): max |slope -
B_k| = 2.9e-5, B_k from 0 to 131. (2) Finite injections of 0.1 / 1 / 5 s: mean rise per second 11.2 / 12.2 / 18.1
against mean B_k 11.1; the bound of (iii) violated 0 times. (3) 2,000,000 turns per load, rho = 0.3 / 0.5 / 0.7 / 0.8
/ 0.9 / 0.95: simulated E[B_k] 0.61 / 2.02 / 7.83 / 20.2 / 88.6 / 369 against rho/(1-rho)^2 = 0.61 / 2.00 / 7.78 /
20.0 / 90.0 / 380; simulated E[Q] 0.13 / 0.51 / 1.64 / 3.22 / 8.08 / 18.0. (4) Lognormal work, rho 0.8: mean B_k by
bins of the predictor (Q + lam s)/(1 - rho): 5.2 / 25.6 / 52.3 / 75.3 / 112 / 185 against predicted 5.2 / 25.7 / 52.2
/ 75.4 / 110 / 189; regression slope through the origin 1.001.

**What it checks in the first draft.**

Section 1.5: 'In an open queue mu is about 1 + Q (queue length)'. Section 6: 'mu(t), GPU time: 1 + the number of ready
turns waiting'. Both understate. The turns delayed are all later turns of the busy period, including those that have
not arrived yet: mu = 1 + (Q + lam x)/(1 - rho) in expectation, the queue length divided by the slack 1 - rho. At rho
= 0.9 the mean is 91 against 1 + Q = 9. 1 + Q is a lower bound, exact only when no turn arrives before the queue
drains.


### 2. The weight mu in the closed loop: N for response time, N - B for TTFT (section 1.5)

Status: proved, with the assumptions stated.

**Statement.**

Model: N users stay in session; each alternates a pause (given by the trace) and one request; a request is in the
system from its arrival to its last token. Over a window of length T let X = requests completed / T, R = mean time in
the system, Z = (time-average number of pausing users) / X. (a) Accounting, with no assumption on the server: N / X =
R + Z. (b) Single server (one request at a time, never idle on the window). Add delta seconds to the work of the
request in service at time t0, delta small enough that the order of service does not change. Then exactly one request
of every user is delayed by delta and no other response time changes: the sum of response times rises by N * delta. If
every request gets eps more work, mean R rises by N * eps. (c) Below saturation, in steady state, mu := d(mean
R)/d(work per request) = N * (1 - a), where a in [0, 1] is the share of an added GPU-second that displaces idle time.
(d) Pool-bound batch (the engine): at most B turns run; prefill is serial and pauses decoding; a decode step with n
running turns takes tau(n) seconds and gives each of them one token. If the pool is always full and never short of
ready turns, then GPU time per request is g = 1/X = P + tau(B) * n_out / B (P = mean prefill seconds per request
including recompute, n_out = mean output tokens), the mean time between admission and last token is B / X, and mean
TTFT = (N - B)/X - Z + t_pre (t_pre = mean time from admission to first token). One more second of prefill work at
time t0 raises the sum of response times by N and the sum of TTFT by N - B_dec, where B_dec is the number of running
turns already past their first token.

**Proof.**

(a) At every instant each of the N users is either pausing or has a request in the system, so N * T = integral of
(number pausing) + integral of (number in the system). The second integral is the summed time in system of the
requests of the window, up to the requests cut by its two ends. Divide by X * T. (b) The server is never idle, so the
m-th service completion after t0 happens at (completion of the request in service at t0) + (sum of the next m works).
The order is unchanged, so every completion after t0 moves by exactly delta, and so does every later service start (a
start is the previous completion). Take a user u. If u has a request in the system at t0, that request completes delta
later and its arrival does not move: +delta. If u is pausing at t0, its next arrival does not move (the pause began
before t0) and its start moves by delta: +delta. After that delayed request u's next arrival = completion + pause
moves by delta, and so do its start and completion: no change, and the same for all its later requests. So each user
loses delta once: N * delta. The order is unchanged whenever delta is smaller than the smallest distance between a
moved arrival and an unmoved one, which is positive with probability one for continuous pause laws. The sustained
statement is (a) with X = 1/g: R = N * g - Z. (c) In general 1/X = g + i with i the GPU idle time per request, so
dR/dg = N * (1 + di/dg) = N * (1 - a), a = -di/dg. (d) In T seconds the GPU spends X*T*P on prefill and the rest on
decode steps; (T - X*T*P)/tau(B) steps give B tokens each and must supply X*T*n_out tokens, so X * (P + tau(B) n_out /
B) = 1. Little's law on the running set (always B turns) gives B/X between admission and completion; on the queue it
gives (N - B)/X - Z of queue wait; TTFT = queue wait + t_pre. Single injection: all GPU events after t0 move by delta
(the GPU is never idle and admissions happen at completions). The B_dec turns past their first token finish delta
later: response +delta, TTFT unchanged. Every other user (being prefilled, admitted, queued or pausing) gets its next
first token delta later, and after its delayed request its whole timeline is shifted, as in (b). Total: N for
response, N - B_dec for TTFT. QED.

**Numerical check.**

exp_mu.py (uses netsim.py), output exp_mu.out; identity_c32_c34.py, output identity_c32_c34.out; all in
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/. Single server, saturated,
injection of 1e-6 s: slope of the summed response time 8.0000 / 32.000 / 128.000 for N = 8 / 32 / 128 on 8 paths each.
Finite injections (N = 32, 300 paths): mean 31.75 (se 2.98) at 1 s, 32.08 (se 0.64) at 5 s, median 32.00 at every
size; at 0.01 s the mean is -8.9 (se 36) because one change of service order moves the sum by seconds, so single small
injections cannot measure mu. Below saturation, M/M/1//N closed form (pause 20 s, work 0.46 s, knee at 44.5 users): mu
= 1.4 / 6.7 / 20 / 49 / 63 / 128 at N = 8 / 32 / 44 / 56 / 64 / 128 against 1 + Q = 1.0 / 2.2 / 5.2 / 13 / 21 / 85;
simulator against closed form at N = 32 / 44 / 64: 5.9 / 20.3 / 64.6 against 6.8 / 20.6 / 63.2. Batch model with FA3
constants (N = 128, B = 20, step 3.5 + 0.99 n ms, 40,000 s): g = 0.460 s (formula 0.463), mean TTFT 31.20 s against (N
- B)/X - Z + t_pre = 31.25. With 0.555 s more prefill per request: TTFT 31.2 -> 90.6 s, d(response)/dg = 126,
d(TTFT)/dg = 107 (N - B = 108), d(response - TTFT)/dg = 18.9 (B = 20). One injection of 1e-6 s on 6 paths: response
slope 127.99 to 128.01, TTFT slope 109.00 with 19 turns past their first token. Recorded tables, campaign 32 minus 34,
minutes 90-109, ec_full: dg = 0.555 s; N dg = 71.1 s = 52.4 (TTFT) + 9.8 (running set; B dg = 10.9) + 8.8 (pausing
users); exact split of dTTFT: 52.4 = 60.4 [(N - B)/X] - 8.8 [pausing] - 0.7 + 1.5 [admission to first token];
jit_exact, ec_exact and jit_p50 close the same way.

**What it checks in the first draft.**

Section 1.5: 'mean TTFT = N g - Z - decode ... so mu = N. Measured: 0.73 N (94 TTFT-seconds per GPU-second)'. The
identity is right, and 'mu = N' holds for response time. For TTFT the term 'decode' (first token to completion) is not
a constant: under continuous batching it is about B * g, so mu for TTFT is N - B, about 108 = 0.85 N at N = 128 and B
= 20. The measured 0.73 N is that minus a transient: in campaign 32's minutes 90-109 the number of pausing users was
still falling from about 50 to 23, because long pauses that began before the loop were running out. Also: 'The queue
wait of a turn is the GPU work of the turns served before it' describes a single server; in the engine admission is
bound by the pool and the wait is queue position over the admission rate. Section 6's estimator '1 + ready turns
waiting' gives about 46 to 61 where the weight is 108.


### 3. The lateness weight c under pass-over (sections 1.5, 5.3, 6)

Status: the first draft's claim was false; the corrected statement is given.

**Statement.**

Model A (one server, the work queue of section 1.5): turn i arrives at a_i and is ready (its read has landed) at l_i
>= a_i; whenever the server is free it starts the earliest-arrived ready turn. D_i = the time i would start if it were
ready; stall S = l_i - D_i > 0; s_j = GPU work of turn j. (L1) If the server is never idle from D_i until i starts,
and J = (j_1..j_m) are the turns started in between (the jumpers), the summed wait of all turns changes by exactly
sum_{j in J} s_j - m * s_i. That is 0 for equal work, 0 in expectation when the jumpers' work is i.i.d. with the same
mean as s_i, and about (1 - s_i / s_bar) per second of stall otherwise (s_bar = mean work of the jumpers). (L2) If no
other turn is ready during the stall, the summed wait rises by S * (1 + B_i) for small S, B_i = later turns in i's
busy period: an idle GPU-second is priced like a second of work (result 1). (L3) In a stable open M/M/1 queue with
exchangeable work the expected cost per second of stall is 1 + rho for short stalls and tends to 1 for long ones. (L4)
Closed loop: late reads change the mean TTFT only through throughput; with the same work per request, change of mean
TTFT = N * (change of GPU idle time per request); it is 0 when the GPU never idles, whatever the differences in work
between turns (over a finite window a policy that delays heavier turns completes more requests, a composition effect).
Model B (pool-bound batch of result 2(d)): (L5) a stalled turn costs its own second while the slot it would take stays
empty, and nothing in the mean while another ready turn takes it. An empty slot also loses a / (B * tau(B))
GPU-seconds per second (a = fixed part of the decode step), worth N * a / (B tau(B)) seconds of summed response time
in the saturated loop: 0.96 at N = 128, B = 20, a = 3.5 ms, tau = 23.3 ms.

**Proof.**

(L1) Without the stall the order from D_i on is i, j_1, .., j_m, rest; with it the order is j_1, .., j_m, i, rest,
with no idle time. Turn i starts sum_J s_j later. Each jumper starts s_i earlier (a jumper that arrives during the
stall finds the server busy in both schedules). After i's service both schedules have done the same work without
idling, so every later start is the same. Sum: sum_J s_j - m s_i. m is the first index at which the jumpers'
cumulative work reaches S, a stopping time, so by Wald E[sum_J s_j] = E[m] * s_bar, and E[m * s_i] = E[m] E[s_i] when
s_i is independent of the jumpers: the expectation is E[m] (s_bar - E s_i). (L2) i starts S later; this is result 1
with S more seconds of work on i before its start. (L3) Short stall: with probability P0 = P(no turn waits behind i at
D_i) the server idles for S, and each idle second delays i and the later turns of its busy period, 1 + rho/(1 - rho) =
1/(1 - rho) turns in expectation (result 1 with Q = 0). Otherwise a waiting turn j is started, i waits for all of s_j
and j gains s_i: zero in expectation. The turns behind i at D_i are the arrivals during its wait; in M/M/1 the wait is
0 with probability 1 - rho and exponential with rate (1 - rho)/s_bar otherwise, so P0 = (1 - rho) + rho (1 - rho) = 1
- rho^2 and the cost is (1 - rho^2)/(1 - rho) = 1 + rho per second. Long stall: i loses S plus a bounded residual
wait; for the others, i's work is removed at D_i and inserted again S later into a queue that has forgotten the
removal; each insertion delays a number of turns whose mean is finite and does not depend on S, so the others' change
is bounded and the cost is S + O(1). (L4) Result 2(a): mean R = N/X - Z and 1/X = work per request + idle per request.
(L5) The turn's own delay is immediate. A fixed set of requests needs sum n_out tokens = (number of steps) * B when
the pool is full; k steps with one slot empty waste k slot-steps, that is k/B extra steps, each costing the fixed part
a (the per-turn part of a step is spent only when the turn decodes). One slot-second empty is 1/tau(B) steps, so the
loss is a/(B tau(B)) GPU-seconds, priced at N by result 2. QED.

**Numerical check.**

exp_c.py (uses netsim.py), output
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/exp_c.out. L1 (60 turns queued, the
first not ready for S = 0.5 / 3 / 10 / 25.5 s): equal work 0.00 at every S; stalled 0.1 s against 1 s turns +0.90 /
+2.70 / +9.00 / +23.40; stalled 5 s -4 / -12 / -40 / -104; formula exact (max error 6e-8 over 2,000 random draws; mean
cost per stall-second +0.025, se 0.026). L2: 200 draws, |slope - (1 + B_i)| = 0.000, slopes 1 to 36. Open M/M/1, stall
2 s, 1,500 draws a load: cost per stall-second 1.10 (se 0.02) / 1.25 (0.05) / 1.59 (0.16) / 1.31 (0.28) / 1.65 (0.59)
at rho 0.2 / 0.5 / 0.8 / 0.9 / 0.95, while P(no other turn waiting) = 0.96 / 0.75 / 0.37 / 0.19 / 0.12 (theory 1 -
rho^2) and (idle time caused)/(1 - rho) = 1.09 / 1.25 / 1.47 / 1.55 / 2.02. Closed loop, one server (knee at 44.5
users), 25 % readers landing Exp(4 s) late, at N = 16 / 32 / 40 / 44 / 48 / 56 / 64 (utilisation 0.36 / 0.69 / 0.84 /
0.90 / 0.94 / 0.989 / 0.999): cost per second of exposed stall 1.04 / 1.10 / 1.09 / 1.08 / 1.00 / 0.65 / 0.22; share
of stall time with the GPU idle 0.51 / 0.22 / 0.13 / 0.10 / 0.07 / 0.02 / 0.005; section 6's estimator 0.91 / 0.60 /
0.39 / 0.29 / 0.19 / 0.05 / 0.006; N x (idle added per turn) equals the TTFT change to 0.01 s in every row. N = 96
with readers carrying 3 times the work: mean TTFT 46.117 -> 46.116 s. Batch model, N = 128, B = 20: lags of mean 0 /
10 / 40 s give mean TTFT 31.20 / 31.22 / 31.15 s with readers at 31.2 / 31.6 / 47.8 s and the others at 31.2 / 31.1 /
25.7 s; lag 150 s leaves 6.6 of 20 slots empty: 38.65 s. Batch model across load (lag 10 s), N = 30 / 50 / 60 / 64 /
68 / 72 / 80: cost per stall-second 1.00 / 0.98 / 0.74 / 0.56 / 0.32 / 0.18 / 0.02; share of stall-seconds with the
slot empty 1 / 1 / 0.92 / 0.60 / 0.29 / 0.13 / 0.02; section 6's estimator 1.00 / 0.99 / 0.82 / 0.64 / 0.42 / 0.21 /
0.04.

**What it checks in the first draft.**

Sections 1.5 (c), 5.3 and 6: 'a second of stall costs c(t) = P(no other ready turn when turn i is due): about 1 at
light load, about 0 once ready work always waits', estimated as 'the share of recent admissions made with no other
ready turn waiting'. The two ends are right. In between it is false for the work queue that 1.5 describes: the cost is
P(the GPU idles during the stall) times the price of an idle GPU-second, and the product stays at or above 1 at every
stable open load and up to about 94 % utilisation in the closed loop (measured 1.0 to 1.1 where the draft's two
estimators give 0.07 to 0.6). 'That turn is admitted earlier by about the same amount' needs (1) ready work behind the
stalled turn for the whole stall, not one ready turn when it is due, and (2) in the open model, jumpers with the same
mean work as the stalled turn. Counterexamples: a stalled 0.1 s turn in front of 1 s turns costs 0.9 per stall-second
although ready work always waits; a stalled 5 s turn gains 4 per stall-second; a lone stalled turn costs 1 + B_i, up
to 36 in the test. For the engine's pool-bound batch the draft's c is about right (within 0.1 in simulation), because
an empty slot costs about the turn's own second; there the precise form is c * S = E[(stall - time for which ready
turns keep the slot filled)+]. It is a statement about the mean: the stalled turn itself always loses its stall.


### 4. Order on the device: Jackson, Horn, and what happens when due dates cannot be met (sections 3.4, 5.2; result 4)

Status: proved, with the assumptions stated.

**Statement.**

The device is one machine of beta tokens per second. A transfer j (read or write) has a release time r_j, a size p_j
(device-seconds) and a due date d_j (a read: its turn's admission time T_j; a write: the time d_w at which its host
copy would be given up). C_j = completion, lateness C_j - d_j, tardiness (C_j - d_j)+. (a) Jackson 1955: all jobs
available, no preemption: serving in non-decreasing due date minimises the maximum lateness. (b) Horn 1974: with
release times and preemption at any instant, always serving the released unfinished job with the earliest due date
minimises the maximum lateness; the optimum is L* = max over pairs (r, d) of [r + work of the jobs released at or
after r and due by d - d]. Corollary (feasibility): if any schedule meets every due date, this one does. (c) If a
transfer can be interrupted only between pieces of at most p_max device-seconds, earliest due date over released
pieces has maximum lateness below L* + p_max, and it is not always optimal. (d) (a) to (c) hold for reads alone and
for reads and writes together: they use only release times, sizes and due dates, and dividing the device's rate
between a read stream and a write stream cannot beat serving one job at a time. With free room a write has no due date
and goes after every due read. (e) These are statements about the maximum lateness in seconds. The design's objective
on the device is the weighted tardiness sum_j w_j (C_j - d_j)+, w_j = c_j for a read and lambda * L_p for a write.
Earliest due date minimises it exactly when it meets every due date (value 0). When every waiting job is already late
the objective is sum w_j C_j minus a constant and Smith's ratio rule (largest w_j / p_j first) is optimal. In between
the problem is NP-hard (strongly so with weights) and neither rule, nor the two combined, is optimal. Stochastic
counterparts: with linear waiting costs the c-mu rule (priority to the largest cost rate over mean size) minimises the
long-run average cost of a multi-class M/G/1 queue among non-idling, non-anticipating policies (Cox and Smith 1961);
with convex increasing delay costs C_k the generalised c-mu rule (serve the class with the largest C_k'(age of its
oldest job) over mean size) is optimal in the heavy-traffic limit (van Mieghem 1995). For the tardiness cost the
derivative is 0 before the due date and w_k after it, so that rule says 'among late transfers, largest w/p' and is
silent among the others; the kink is outside the theorem's smoothness assumptions, so this is guidance, not a proof.
(f) With a precedence 'this write must finish before that read can start' (the read needs the room the write cleans),
earliest due date on the modified dates d'_w = min(d_w, T_j - p_j) minimises the maximum lateness among the orders
that respect the precedence (Lawler 1973). (g) In the saturated loop a late read costs nothing in the mean while ready
turns wait (result 3), so every order that keeps the GPU supplied ties on the mean, and earliest due date is then
optimal for the second criterion, the longest stall of a reader.

**Proof.**

(a) Take an optimal order with adjacent jobs i before j and d_i > d_j, starting at t. After swapping, j completes
earlier, and i completes at t + p_i + p_j, the old completion of j, with lateness t + p_i + p_j - d_i < t + p_i + p_j
- d_j, the old lateness of j. The maximum does not rise; repeat until sorted. (b) Lower bound: the jobs released at or
after r and due by d cannot start before r, so the last of them completes at or after r + their work. Attainment: let
j* have the largest lateness under the rule, completing at C, and let t1 be the start of the last stretch ending at C
in which only jobs due by d_{j*} are served without idling. Just before t1 the machine idles or serves a job due later
than d_{j*}; in both cases no job served in [t1, C] was released before t1, or the rule would have served it. So these
jobs are released at or after t1, due by d_{j*}, and need at least C - t1 of work: the bound gives L* >= C - d_{j*}.
(c) Same argument: the later-due piece that runs just before t1 started at t0 > t1 - p_max, when none of the jobs
served in [t1, C] was released, so in any schedule their last completion is after t0 + (C - t1) > C - p_max. (d) Any
division of the device's rate is a limit of preemptive one-at-a-time schedules with the same completion times, so (b)
covers it; the engine's device delivered about 2.1 GB/s whatever the read/write split. (e) Maximum lateness at most 0
means all tardiness is 0. If every d_j is in the past, (C_j - d_j)+ = C_j - d_j; for adjacent i before j starting at t
the cost w_i (t + p_i) + w_j (t + p_i + p_j) is at most the swapped cost iff w_i / p_i >= w_j / p_j (Smith 1956).
Hardness: Lawler 1977 and Lenstra, Rinnooy Kan and Brucker 1977 (weighted), Du and Leung 1990 (unweighted).
Counterexamples in between, weights 1 unless given: jobs (work, due) = (10, 9), (1, 9.5): earliest due date costs 2.5,
the other order 2.0. Jobs (work, due, weight) = (10, 10, 1), (1, 100, 5), (5, 12, 1): Smith's order costs 6, the
optimum 3. (f) The modified dates are no later than the originals, so the maximum lateness can only grow. Conversely,
if the write's modified lateness exceeds its original one, then d'_w = T_j - p_j, and the read starts after the write,
so C_j - T_j >= C_w + p_j - T_j = C_w - d'_w: the read is at least as late. So on precedence-feasible orders the two
maxima are equal. Sorting by d' puts the write before its read (d'_w < T_j) and by (a) minimises the maximum for the
modified dates. (g) Result 3 (L1, L4) and (b). QED.

**Numerical check.**

device_order.py, output
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/device_order.out (exhaustive search
and a dynamic programme over subsets). Jackson: 300 instances of 2-7 jobs against all orders, 0 misses. Horn: 2,000
instances with release dates, preemptive earliest due date equals L* (max difference 4e-15). Pieces of 1.5 s: above L*
in 545 of 3,000 instances, largest excess 1.499 s, never above p_max. No preemption: worse than the best order in 102
of 400 instances, excess at most 0.85 of the longest job. Weighted tardiness, 8 jobs, 1,500 instances a regime, cost
over the optimum's cost. All due dates feasible: EDD and the composites optimal, Smith's ratio costs 3.5 s against 0.
Mixed: EDD 1.92, Smith 2.59, 'EDD else Smith' 2.28, 'late jobs by Smith, rest EDD' 2.14, index w/max(p, d - t) 1.29,
'EDD if feasible else that index' 1.24, ATC 1.29. Heavy overload: 2.31 / 1.10 / 1.10 / 1.55 / 1.02 / 1.02 / 1.05.
Every job late: EDD 1.35, all others 1.00. Precedence: modified due dates reach the optimum in 400 of 400 instances.

**What it checks in the first draft.**

Section 3.4: 'Serving the released jobs in earliest-due-date order minimizes the maximum lateness (Jackson's rule;
with release times and preemption at page-group boundaries, Horn's theorem), so if any order lands every read before
its turn is due, this one does'; result 4: 'over reads and writes together'; section 5.2: 'if the due dates cannot all
be met, serve the largest cost per device-second first'. Differences. (1) With preemption only at page-group
boundaries Horn's theorem holds up to p_max, about 1.5 s for a 512-page read window and 0.4 to 1.5 s for a 128-page
write batch: feasibility is guaranteed only if the due dates can be met with p_max to spare. (2) The theorem is about
the maximum lateness in seconds, not the paper's sum of weighted lateness; a second of read lateness (weight c) and of
write lateness (weight lambda L_p) are not the same cost. (3) The fallback is optimal only when every waiting transfer
is already late. Read literally (switch the whole queue to cost per device-second once one due date is missed) it
costs 2.3 times the optimum on mixed instances; 'earliest due date while feasible, otherwise the single index w_j /
max(p_j, d_j - t)' costs 1.24 times. So the rule is defensible as a two-regime rule, optimal for the maximum lateness,
for weighted tardiness when the due dates are feasible or all missed, and lexicographically at saturation (g). (4)
'Writes go first when dirty pages hold room that due reads need' is the precedence case (f); it can replace the 50 K /
25 K thresholds.


### 5. Preempt on host room: the covering knapsack and the greedy bound (section 5.1; result 5)

Status: proved, with the assumptions stated.

**Statement.**

A read needs m tokens of host room. Holder j has L_j tokens and displacing it costs C_j >= 0 (the rise of its priced
cost at the current prices). Problem: minimise sum_j C_j x_j subject to sum_j L_j x_j >= m, x_j in {0, 1}. (a)
Divisible version (0 <= x_j <= 1, cost proportional to the part taken): taking holders in increasing C_j / L_j, the
last one (the critical holder k) in part, is optimal; its value Z_LP is a lower bound on the 0-1 optimum OPT. (b) 0-1
version: taking whole holders in that order until m is covered costs less than Z_LP + C_k, hence less than OPT plus
the cost of one holder, and frees less than L_k tokens beyond m. (c) The ratio to OPT is unbounded. (d) A variant with
ratio at most 2: walk the same order; a holder that would complete the cover is recorded as a candidate (the holders
taken so far plus this one) and skipped; return the cheapest candidate. Assumptions behind the knapsack itself: costs
add up across holders (fixed prices: displacing several holders does not move kappa or lambda) and, for (a), the cost
of a partial eviction is proportional to the tokens taken.

**Proof.**

(a) Exchange: if a solution takes part of a holder with a larger ratio while a holder with a smaller ratio is not
fully taken, moving a token of room from the first to the second keeps the room and does not raise the cost; repeating
reaches the greedy solution. (b) Greedy takes holders 1..k whole: cost = Z_LP + (1 - x_k) C_k < Z_LP + C_k <= OPT +
C_k; the surplus is (1 - x_k) L_k < L_k. (c) m = 1; holder A has 1 token and cost 1; holder B has M tokens and cost
0.9 M. B has the smaller ratio and is taken first: cost 0.9 M against 1. (d) Call a holder big if the walk skipped it
and small if it took it. The small holders never cover m, so an optimal set contains a big holder; let k be the first
big holder in the order with C_k <= OPT, and S the small holders before it. Big holders before k cost more than OPT,
so they are not in the optimal set, which therefore uses, among the holders before k, only members of S. The optimal
set covers m and S does not, so the tokens of the optimal set outside S exceed the tokens of S outside the optimal
set, and they have ratios at least as large; hence cost(optimal set outside S) >= cost(S outside optimal set), so
cost(S) <= OPT. The candidate S plus k costs at most 2 OPT. QED.

**Numerical check.**

knapsack.py, output /home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/knapsack.out.
Divisible: ratio greedy equals scipy's LP optimum (HiGHS) on 300 instances, max difference 6e-14. 0-1: 3,000 instances
of 2-12 holders against exhaustive search: the additive bound never violated, largest (greedy - optimum)/(last
holder's cost) 0.86, largest ratio 4.2; worst-case family: ratio 9 / 90 / 900 for M = 10 / 100 / 1,000. Skip variant:
largest ratio 1.70 on those instances and 1.80 on 20,000 small integer instances. Engine-sized instances (60 holders
around 25 K tokens in 64-token pages, need 25 K): greedy / optimum mean 1.39, p90 2.1, max 5.2; skip variant mean
1.05, max 1.63; exact optimum by a dynamic programme over pages (about 2e4 steps).

**What it checks in the first draft.**

Section 5.1 and result 5: 'Freeing m tokens at least cost is a knapsack; taking holders in increasing C_j / L_j is
optimal for the divisible problem and within one holder otherwise.' True as stated, with three additions. (1) 'Within
one holder' is an additive bound, and one holder is about the size of the whole need here (25 K tokens against 25 K):
on engine-sized instances whole-holder greedy costs 39 % more than the optimum on average and frees 87 % more room
than needed; the worst-case ratio is unbounded. (2) The order written in 5.1 (ended sessions; clean idle prefixes by
nu; staged reads; queued turns) is a fixed order of classes, not a ratio order; it is the knapsack order only if the
classes' cost-per-token ranges do not overlap. The entries also mix units: nu is a cost per token-second, the
staged-read score is a cost per token, and the queued-turn score is not divided by L_j. One list by C_j / L_j is
needed, with C_j / L_j = sup over horizons h of [(K_j / L_j) F_j(h | a) - lambda * tau_j(h | a)], the surplus per
token of keeping j at the current price (positive exactly when nu_j > lambda). (3) Divisibility needs a cost
proportional to the tokens taken. At light load a partly evicted prefix still pays the read's fixed lag (about 0.9 s,
against 1.2 s to transfer 25 K tokens), so the cost has a fixed charge and a few whole holders are cheaper than many
partial ones.


### 6. Price: the relaxed problem, the marginal holder, complementary slackness, and the frontier estimate (section 6; result 6)

Status: proved, with the assumptions stated.

**Statement.**

Relaxed joint problem (open model). Session i controls the host tokens it holds m_i(t), its use of the device u_i(t)
in {0, 1} and its SSD tokens s_i(t). Its cost is Phi_i = E sum over its requests of [c S + mu gamma U] (S = stall
seconds, U = recomputed tokens, gamma = GPU seconds per token), with the weights c and mu, the arrivals and the queue
waits given. Hard constraints at every t: sum_i m_i <= C_H, sum_i u_i <= 1, sum_i s_i <= C_S. Relaxed constraints: the
same on average over [0, T]. Lagrangian with lambda, kappa, sigma >= 0: sum_i Phi_i + lambda (E int sum_i m_i - C_H T)
+ kappa (E int sum_i u_i - T) + sigma (E int sum_i s_i - C_S T) = sum_i J_i - (lambda C_H + kappa + sigma C_S) T, with
J_i as in section 1.6. Theorem (room price). Assume that for every price lambda >= 0 an optimal policy of session i
for J_i holds its prefix in state x exactly when its index nu_i(x) >= lambda, and that 'nu > lambda' is optimal too
(the holding problem is indexable with index nu). Let M+(lambda) and M-(lambda) be the total expected host tokens
demanded under 'hold iff nu > lambda' and 'hold iff nu >= lambda', and lambda* = inf{lambda >= 0 : M+(lambda) <= C_H}.
(i) If M+(0) <= C_H then lambda* = 0 and the unpriced policies are optimal for the relaxed problem. (ii) Otherwise
M+(lambda*) <= C_H <= M-(lambda*), and the policy that holds when nu > lambda*, drops when nu < lambda*, and holds
with a probability theta when nu = lambda* (theta chosen so that demand equals C_H) is optimal for the relaxed
problem, with multiplier lambda*. So lambda* is the index of the holders at the margin and lambda* (demand - C_H) = 0.
(iii) Online estimate. Evicting the lowest index, and only when room is needed, fills the room in decreasing index;
that is the primal optimum of the instant problem 'maximise sum_j nu_j L_j x_j subject to sum_j L_j x_j <= C_H, 0 <=
x_j <= 1', and its dual variable is the index at which the room runs out: lambda(t) = the lowest index still held, 0
while room is free. Primal and dual come from the same sort, with no step size and no iteration. Dual ascent (lambda
<- lambda * exp(eta (d - C)/C)) instead iterates on the price with excess demand as its gradient; a hard capacity
censors the excess at 0 unless refusals and forced evictions are counted, and it takes many intervals to converge.
(iv) sigma: the same theorem with SSD tokens, C_S and the index of a stored chain (tail before head, because a lookup
stops at the first missing page). (v) kappa and mu are prices of a different kind: kappa(t) = the summed lateness cost
rate of the transfers that complete late after t in the device's busy period; it is what one more unit of device
capacity at t would save.

**Proof.**

Decoupling: for fixed multipliers each term of the Lagrangian depends on one session's own controls (weights and
arrivals are given), so it is minimised session by session. Sufficiency (Everett 1963): let pi* be the policy of (ii).
Both rules minimise J_i at price lambda*, and so does a mixture decided by a coin flipped at each return, whose demand
moves linearly from M+(lambda*) to M-(lambda*). M+ is non-increasing and right-continuous and M- is its left limit, so
M+(lambda*) <= C_H <= M-(lambda*) and a theta with demand exactly C_H exists. For any policy pi that meets the relaxed
constraint: cost(pi) >= cost(pi) + lambda* (demand(pi) - C_H) >= cost(pi*) + lambda* (demand(pi*) - C_H) = cost(pi*).
The first step uses demand(pi) <= C_H, the second that pi* minimises the Lagrangian, the third demand(pi*) = C_H (or
lambda* = 0 in case (i)). Holders with nu above lambda* are kept and those below dropped, so lambda* is the index at
the margin. Instant problem: its dual is 'minimise lambda C_H + sum_j L_j (nu_j - lambda)+ over lambda >= 0', whose
optimum is the index of the critical holder in the fill by decreasing index (0 if everything fits). The two agree in a
stationary system with many holders: the frontier is the empirical quantile of the holders' indices at which
cumulative tokens reach C_H, and lambda* is the same quantile of the expected distribution. Device price: one more
unit of work at t in front of the transfers behind moves the completion of each later transfer of the busy period by
one unit, so sum_j w_j (C_j - d_j)+ rises by the weights of those that complete late; one more unit of capacity is the
mirror image. QED.

**Numerical check.**

price_dual.py -> price_dual.out; knapsack.py part K5 -> knapsack.out; kappa_check.py -> kappa_check.out; all in
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/. Toy: 1,000 sessions in three hint
classes (tool gaps p10/p50/p90 1/2/18 s, person 9/96/1,096 s, test run 20 +- 2 s), room 30 % of all prefixes, hold or
drop only. Single session: the index rule equals the best drop age at 60 prices (excess 0.0) and its drop age never
rises with the price. The dual function is concave with maximum at lambda* = 4.627e-6 s per token-second; demand 6.759
M tokens just below lambda*, 6.639 M just above, room 6.750 M; lambda* equals the index of the tool class at idle age
19 s to 5e-13; with room for everything lambda* = 0. Frontier estimators in the hard-constrained run, in units of
lambda*: lowest index still held 1.00 (time average and median); highest index evicted per second 0.99; token-weighted
mean of evicted indices 0.88; 'highest given up lately' 1.02 / 1.07 / 1.12 / 1.19 for windows of 5 / 30 / 120 / 600 s.
Dual ascent from 0.01 or 100 lambda*: within a factor 1.5 after 589 to 1,109 s (eta 0.05), 114 to 164 s (0.3), 44 to
49 s (1.0, then about 15 % jitter). Instant LP (HiGHS, 300 instances): the multiplier of the room constraint equals
the marginal holder's index to 9e-16 and is 0 in the 74 instances where everything fits. Device: exact price against
finite differences on 1,920 cases, error below 2e-4; draft's estimator over exact = 0.03 / 0.06 / 0.08 / 0.14 at
device loads 0.6 / 0.8 / 0.9 / 0.95.

**What it checks in the first draft.**

Section 6 and result 6: 'lambda(t): the index nu of the marginal holder (the highest index among the pages given up
lately); 0 while room is free. This is the multiplier of the room constraint in the relaxed problem ... read off the
eviction frontier, not learned from refusals'; 'sigma(t): the index of the cleaner's marginal victim'; 'kappa(t): the
summed lateness cost rate of the transfers waiting behind'. What holds: the theorem, given indexability of the holding
problem (result 2, not checked here beyond the toy) and a relaxation on average; lambda* is a constant, lambda(t) is
the multiplier of the instant problem. What differs. (1) 'The highest index among the pages given up lately' is biased
upward with the length of 'lately' (2 % at 5 s, 7 % at 30 s, 19 % at 600 s in the toy); the time-average of the lowest
index still held is unbiased (1.00 lambda*). (2) lambda sits inside the index through K_read (the term lambda L r), so
the frontier solves lambda = (a + lambda r) h, that is lambda = a h / (1 - r h); harmless while r h is small. (3) The
frontier prices a marginal token; a 25 K-token read is 1.5 to 3 % of the room and its cost is the knapsack cost of
result 5, not lambda times m. (4) A price that is too low is harmless while eviction is by index and only on need
(cost +0.5 % at 0.1 lambda*); a price that is too high wastes room (+38 % at 2 lambda*): never drop a holder for the
price while room is free. (5) kappa as drafted counts only the transfers waiting now and already late; the exact price
also counts transfers that arrive later in the busy period and those that become late. In the test the draft's
estimator is 3 to 14 % of the exact value. (6) Room prices (lambda, sigma) are indices of marginal holders; server
prices (mu, kappa) are cost rates of the work behind in the busy period. Section 0's 'the length of each queue is a
price' holds for neither kind without the factor 1/(1 - load).


### 7. The bound: weak duality in the open model, and why it fails in the closed loop (sections 1.6, 7; result 7)

Status: proved, with the assumptions stated.

**Statement.**

For every lambda, kappa, sigma >= 0 let D = sum_i J_i* - (lambda C_H + kappa + sigma C_S) T, where J_i* is the
smallest value of J_i over the policies of session i alone (alone, a session has no line at the device and no limit on
room, and pays the prices). Then for every policy pi that meets the three hard constraints with the same information,
the cost sum_i E[c S_i + mu gamma U_i] under pi is at least D. Assumptions: (A1) open model: arrivals, queue waits and
the weights c and mu do not depend on the policy; (A2) sessions' gaps are independent of each other given the hints
(or J_i* is taken over policies that may use everything the joint policy knows); (A3) every held token is charged to
one session (no shared pages, or shared pages charged in parts adding up to at most one); (A4) J_i* is computed with
the line wait V = 0. In the open first-come-first-served queue without pass-over, taking mu_k = 1 + B_k on the path
with no recompute makes base + sum_k mu_k gamma U_k a true lower bound on summed TTFT, so D extends to summed TTFT
there. In the closed loop the statement is false as a bound on summed TTFT.

**Proof.**

Step 1. A policy that meets the hard constraints at every instant meets them on average, so each penalty term is at
most 0 and sum_i Phi_i(pi) >= Lagrangian(pi). Step 2. Lagrangian(pi) = sum_i J_i(pi) - (lambda C_H + kappa + sigma
C_S) T, where J_i(pi) uses the holdings, device use and SSD use that pi gives session i. Step 3. Seen from session i
the other sessions' states are outside randomness; by A1 and A2 they carry no information about i's future beyond its
own hint, and by A4 any schedule that pi gives i, waiting in line included, is also available to i alone. So J_i(pi)
>= J_i*. Step 4. Combine: sum_i Phi_i(pi) >= D. Open first-come-first-served extension: by result 1 the sum of waits
is convex in the vector of works, so it is at least its linearisation at the no-recompute path, whose slopes are the
B_k of that path and do not depend on the policy. Closed loop, counterexample: three users and one
first-come-first-served server. A needs 5 s and then, 0.9 s after its reply, 10 s; B needs 1 s twice (arriving at 0.1
s, then 0.001 s after its reply); C sends one 1 s request at 5.95 s. The five requests have a summed response time of
44.05 s. With 0.2 s more work on A's first request it is 35.65 s: A returns 0.2 s later and C's short request now goes
before A's long one. More work, lower cost; so no bound of the form base + mu * work holds there, and D added to the
base is not a lower bound. QED.

**Numerical check.**

price_dual.py -> price_dual.out; closed_counterexample.py -> closed_counterexample.out; both in
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/. Toy (hold or drop, hard room,
three seeds a row): bound 41.559e-3 s of read per session-second; the index policy is 5.5 % (se 0.3) / 2.4 % (0.2) /
0.45 % (0.19) / 0.17 % (0.16) / 0.04 % (0.09) above it at N = 30 / 100 / 300 / 1,000 / 3,000; LRU 43 to 51 % above; a
policy that knows the gaps exactly 13 to 25 % below. Counterexample: 44.049 s -> 35.649 s for +0.2 s of work, where
the first-order rule mu = N = 3 predicts +0.6 s.

**What it checks in the first draft.**

Result 7: 'For any prices, the sum of the sessions' optimal J_i minus the priced capacities is a lower bound on the
priced cost of every policy that meets the capacities with the same hints (weak duality), in the open model where mu
and c are given.' Holds with four additions. (1) It bounds the unpriced cost sum E[c S + mu gamma U], not a 'priced
cost'. (2) The per-session problems must be solved without the device's line: section 2 puts the measured line wait V
inside r, which is right for the decision rule and wrong for the bound. (3) Shared prefix heads must not be charged in
full to every sharer (17 % of prompt tokens are held by earlier sessions). (4) 'The same hints' is essential: a policy
that knows each gap exactly is 13 to 25 % below the bound in the toy. With constant prices the device term is nearly
empty: the average device constraint is slack unless the device is busy all the time, so the bound ignores the read
line, which is most of r in the records. In the closed loop of the replay, which is section 1.5's own setting for mu =
N, it bounds nothing; the usable statement there is the floor of result 8.


### 8. Saturation: the identity, the tie between policies, and a tight floor (result 8; section 8)

Status: proved, with the assumptions stated.

**Statement.**

Setting: N users in session throughout a window; pauses given by the trace; the pool-bound batch GPU of result 2(d).
(a) For any policy, mean TTFT = N/X - Z - D_run + t_pre exactly, with X the completion rate, Z = (time-average number
of pausing users)/X, D_run = (time-average number of running turns)/X, t_pre the mean time from admission to first
token. (b) If under two policies the pool is always full while turns wait (B running), the prefill seconds per request
P are the same (new tokens, recompute and per-batch overhead), and the same requests complete (same mean pause and
output length), then X and the mean TTFT are the same, whatever the timing of reads, the order on the device and the
turns that wait. (c) Floor: for every policy under which the system stays saturated, mean TTFT >= (N - B) g0 - Z + p0,
where g0 = P0 + tau(B) n_out / B is the GPU time per request with no recompute and a full pool and p0 is the mean
prefill time of a request's own new tokens. A policy reaches the floor (up to prefill queueing after admission) when
nothing is recomputed and no slot is left empty while a turn waits; its distance from the floor is (N - B) times its
wasted GPU seconds per request, recompute plus the loss from empty slots. (d) What matters where, with FA3 constants:
the pool fills at about N_sat = B + Z / g0 = 20 + 22/0.46 = 68 users in session. Below about 50 users every second of
exposed stall costs about one second and a GPU-second costs 1 to a few: Fetch timing and Retain order act directly.
Above about 72 users only wasted GPU seconds move the mean: Retain, Preempt and the write path matter through
recompute (never lose the only copy, never refuse a read), and Fetch and the device order matter only for keeping some
ready turn available and for who waits. Between the two both act.

**Proof.**

(a) Result 2(a), with the system split into queue and running set, and TTFT = queue wait + t_pre. (b) By result 2(d)
1/X = P + tau(B) n_out / B depends only on P, B and n_out; insert in (a) with D_run = B/X. (c) Recompute adds gamma U
to P. An empty slot lowers the tokens per step from B to B - 1 at step time tau(B - 1), and (B - 1)/tau(B - 1) <
B/tau(B) because tau has a positive fixed part. Both raise 1/X above g0. With fewer than B running, N - (number
running) is larger as well, so (N - running)/X >= (N - B) g0. t_pre is at least the turn's own prefill, mean p0.
Insert in (a). (d) The pool is full when the pausing users, X * Z at most Z/g0, plus B fall short of N; the regimes
then follow from results 2 and 3. QED.

**Numerical check.**

exp_mu.py -> exp_mu.out, exp_c.py -> exp_c.out, identity_c32_c34.py -> identity_c32_c34.out, in
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/. Batch model, N = 128, B = 20:
mean TTFT 31.20 s against (N - B)/X - Z + t_pre = 31.25. Read lags of mean 0 / 10 / 40 s on 25 % of turns: mean TTFT
31.20 / 31.22 / 31.15 s, throughput 2.176 / 2.177 / 2.175 per second, readers 31.2 / 31.6 / 47.8 s, others 31.2 / 31.1
/ 25.7 s. Readers with 3 times the prefill work: 38.63 s on time, 38.59 s late. Starvation (lag 150 s, 6.6 slots
empty): 38.65 s. Extra prefill of 0.05 and 0.555 s per request: TTFT +5.6 and +59.4 s, that is 110 and 107 per second
of g (N - B = 108). Saturation point in the simulation (pause about 18.5 s): 95 % of slots used at 64 users, 99.6 % at
72; campaign 34's first sustained queue came with 76 users in session. Records: campaign 34's arms recompute about 350
tokens per request in minutes 90-110, about 0.015 s, so they sit about 108 x 0.015 = 1.6 s above the floor of their
own g0 (27 to 28 s measured).

**What it checks in the first draft.**

Result 8 and section 1.5: 'In the closed loop with the GPU never idle, mean TTFT = N g - Z - decode: two policies with
the same GPU work per request have the same mean TTFT, whatever their read timing. The optimum then minimizes
recompute.' Holds with these conditions: N constant (the 128-user cap binding); the mean taken over the requests
completed in a window long against the longest pause (in the 20-minute windows of the records the pausing population
had not settled, which is 8.8 to 10.9 s of the difference between campaigns 32 and 34); 'never idle' read as 'the pool
is full of ready turns'; 'GPU work per request' read as prefill seconds per request including per-batch overhead
(0.062 against 0.048 ms a token in campaigns 34 and 32), with the decode term B g moving with g; the same requests
completing under both policies. 'Minimizes recompute' should read 'minimizes wasted GPU seconds per request: recompute
and slots left empty'. It is a statement about the mean only: read timing moves waiting between readers and the others
(47.8 against 25.7 s in the simulation, 46-69 against 12-22 s in campaign 34).


### 9. Stability: the smallest model of campaign 32's loop and the two rules that remove it (result 9)

Status: proved, with the assumptions stated.

**Statement.**

Model: d(t) = dirty tokens in host memory (pages whose SSD write is pending; they cannot be evicted). p(d) = share of
reads refused for lack of clean room, non-decreasing in d, 0 for d <= d0, at most p_max. A refused read is recomputed:
GPU time per request g(p) = g0 + gamma L p (L = history tokens, gamma = seconds per token), request rate x(p) = 1/g(p)
in the saturated loop. Each request writes Dw new tokens; without the born-clean rule the recomputed L p tokens are
written again. Writes drain at w tokens per second while d > 0. So d' = F(d) - w, F(d) = x(p(d)) (Dw + L p(d)). (a)
Without the two rules F increases with p whenever g0 > gamma Dw (0.46 s against 0.08 s here). With a_max := Dw / g0
and a_loop := (Dw + L p_max)/(g0 + gamma L p_max): if w < a_max the dirty set grows from 0; if a_max < w < a_loop the
state d = 0 is stable, there is an unstable point d_u with p(d_u) = p* = (w g0 - Dw)/(L (1 - w gamma)), and above d_u
the dirty set runs away to the room's limit (two stable states); if w > a_loop only d = 0 remains. Near d_u the excess
grows like exp(t / T_e), 1/T_e = p'(d_u) L (g0 - gamma Dw) x(p*)^2. (b) Born clean (a recomputed page that the SSD
holds complete is not written again): F(d) = Dw x(p(d)) <= a_max for every d. If the writes are guaranteed w_min >
a_max, then d(t) <= sup_s [A(s, t) - w_min (t - s)], at most the burst allowance of the new-token process A, for all t
and whatever the refusals. (c) Delay instead of refusal (a read that finds no room waits): p = 0, nothing is
recomputed for lack of room, the same bound holds and the request rate stays 1/g0; it needs waiting reads not to hold
room that only their own progress would free. So each rule alone removes the runaway in this model; born-clean leaves
the recompute of refused reads, delay removes it.

**Proof.**

(a) dF/dp = L (g0 - gamma Dw)/(g0 + gamma L p)^2 > 0, and p is non-decreasing in d, so F is non-decreasing in d with
F(0) = a_max. d' = F(d) - w changes sign from negative to positive where F(d_u) = w, that is Dw + L p = w (g0 + gamma
L p), p* = (w g0 - Dw)/(L (1 - w gamma)). Such a point exists in the range of p exactly when a_max < w < a_loop. It is
unstable because F is increasing there; d = 0 is stable because F(0) < w. Linearising d' at d_u gives the rate. (b)
With born-clean every token enters the dirty set at most once, when first produced. Requests complete at rate at most
1/g0, since recompute only slows them, so new tokens arrive at rate at most a_max plus bursts. A work-conserving
server of rate at least w_min > a_max has backlog sup_s [A(s, t) - w_min (t - s)], which the burst allowance bounds.
(c) The same with p = 0. QED.

**Numerical check.**

stability.py, output /home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/stability.out.
Constants from the records (g0 0.46 s, 0.043 ms/token, L 25 K, Dw 1.9 K) and a refusal ramp fitted to two records of
campaign 32 (first refusals near 100-160 K pending, half the history recomputed near 500-650 K: d0 100 K, d1 600 K,
p_max 0.55; this ramp is an assumption). a_max = 4.13 K tokens/s; a_loop = 14.9 K; in the loop 0.95 requests/s
(campaign 32: 0.98) and 88 % of writes are rewrites (campaign 32: 74-93 %). Tipping point d_u = 101 / 130 / 155 / 281
/ 436 K tokens at w = 4.2 / 5.5 / 6.5 / 10.2 / 13 K, e-folding 20 to 71 s; only the clean state above 14.9 K. Episode
(w = 8 K, then 3.4 K for six minutes, then 10.2 K): the plain system reaches the room limit (700 K) and is still there
20 minutes later; born clean 169 K then 0; delay 263 K then 0. Stochastic, 400 two-hour runs a row: without the rules
the loop appears in 100 / 87 / 32 / 4 / 0 % of runs at w / a_max = 1.04 / 1.11 / 1.21 / 1.33 / 1.57; 0 % with
born-clean; with delay 12 % reach 400 K at 1.04 and none above. Early pins (refusals start 100 K sooner): tipping
point 22 to 66 K instead of 119 to 155 K, and the loop in 100 % of runs within 184 to 660 s.

**What it checks in the first draft.**

Result 9: 'With born-clean pages and delay in place of refusal, the dirty set is bounded whenever the write rate
exceeds the rate of new tokens; without them recompute feeds writes, writes hold room, and room refusals feed
recompute.' Holds; either rule alone is enough in the minimal model. Conditions to add. (1) The write rate is the rate
guaranteed to writes while reads load the device (5.5 to 7.4 K tokens/s in campaign 34, against 17 to 21 K alone), not
the device's rate. (2) The new-token rate is taken at full request speed, a_max = Dw / g0 = 4.1 K tokens/s (measured
4.0 to 4.8 K), plus pages the SSD cleaner deleted, which are new again (about 0.9 K tokens/s late in campaign 34). (3)
A page counts as clean only on a full acknowledgement. (4) Waiting reads must not deadlock on room held by other
waiting reads (parked reads' anchors held up to 107 K tokens in campaign 34). (5) 'Bounded' means bounded by bursts:
at w / a_max = 1.04 the delayed system still reached 400 K dirty tokens in 12 % of two-hour runs. The model also shows
why campaign 32 did not recover: between a_max and a_loop (4.1 and 14.9 K tokens/s) the loop is a second stable state,
and the loop's own write rate of about 10 K lies inside that range.


### 10. Many sessions: what index-policy theory gives and what it does not (result 10)

Status: open.

**Statement.**

What the cited theory gives (from memory; to be checked against the papers). Weber and Weiss (1990): N identical
restless bandits with finitely many states, a fixed share of them active at a time; if the bandits are indexable and
the differential equation of the index policy's fluid limit has a single globally attracting fixed point, the index
policy's reward per bandit converges to the value of the relaxed problem as N grows. They give counterexamples with
four states in which the fluid limit cycles and the index policy is not asymptotically optimal. Verloop (2016): the
same for several classes of bandits that may arrive and depart, for a set of priority policies built from the relaxed
linear programme (Whittle's index policy among them when indexable), again under a global-attractor condition. Brown
and Smith (2020): finite horizon, non-identical and non-stationary items; the gap between a Lagrangian index policy
with careful tie-breaking and the Lagrangian bound is of order sqrt(N) while the value is of order N; no indexability
or attractor condition, but one resource, two actions per item and independent items. What these support here: in the
open model restricted to Retain (hold or drop, restore cost K given, sessions independent given their hints, finitely
many hint classes, room C_H growing in proportion to N, each prefix small against the room), 'evict the lowest index
when room is needed' has a cost per session that converges to the relaxed bound of result 7, at rate 1/sqrt(N) or
faster, provided the fluid limit has a global attractor.

**Proof.**

No new proof. The restricted problem is a multi-class restless bandit with a knapsack constraint: each session is an
arm with state (class, idle age, held or not); holding is the active action and uses L tokens; a dropped arm stays
passive until its return. It is indexable because the single-arm problem is an optimal stopping problem, in which the
set of ages where holding is optimal shrinks as the price rises (checked numerically in the toy). The relaxed problem
and its bound are results 6 and 7. The theorems then apply after two steps that are not done here: idle age is
continuous and the gap laws are not exponential, so the arm must be approximated by finitely many phases; and the
sizes L_i differ, so the constraint is a knapsack, which is covered when the largest L_i over C_H tends to 0. The
global-attractor condition has to be assumed or checked for the fluid model.

**Numerical check.**

price_dual.py, output
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/math_order_price/price_dual.out (Retain-only toy,
three hint classes, room 30 % of all prefixes, three seeds a row). Gap of the index policy to the relaxed bound: 5.5 %
(se 0.3) at N = 30, 2.4 % (0.2) at 100, 0.45 % (0.19) at 300, 0.17 % (0.16) at 1,000, 0.04 % (0.09) at 3,000; log-log
slope -1.1 over N = 30 to 300, faster than the general -0.5. LRU stays 43 to 51 % above the bound at every N.

**What it checks in the first draft.**

Result 10: 'Ranking by the index is asymptotically optimal for the relaxed problem as N grows (index-policy theory for
restless bandits); cited, not re-proved.' Three corrections. (1) Wording: the index policy is asymptotically optimal
for the hard-constrained problem; the relaxed problem supplies the bound it approaches. (2) Scope: the theory covers
one resource, two actions per arm and arms that evolve independently. The design has three resources and several
actions per session (hold, read, write, recompute), sessions coupled through the GPU queue and the device line (W, V,
c and mu depend on everyone's actions, and in the closed loop the return times do too), and shared prefix heads. For
that problem nothing is proved. The nearest results are for LP-based or fluid policies that are re-solved over time in
weakly coupled problems with independent arms (Brown and Zhang 2022; Gast, Gaujal and Yan 2023), not for a static
ranking by separate indices. (3) Conditions: the global-attractor condition is not a formality here, because campaign
32's loop is a second attracting state of the real system (result 9), so it holds at best with the born-clean and
delay rules. N is 128 with about 40 to 60 prefixes in the host room (one prefix is 1.5 to 7 % of it), where a
1/sqrt(N) term is 10 to 15 %. The runs are two-hour transients, so the finite-horizon statement of Brown and Smith is
the relevant one.


### Further results proposed by this check

- A provable optimum for the saturated regime: the floor. For every policy in the saturated closed loop, mean TTFT >=
(N - B) g0 - Z + own prefill, reached when nothing is recomputed and no slot is left empty (result 8c). It is tight,
it holds exactly where the Lagrangian bound fails, and the gap of a running system is observable: (N - B) x
(recomputed tokens x gamma + empty slot-seconds x a/(B tau)) per request, about 1.6 s for campaign 34. Sketch:
Little's law on the queue and on the running set, plus throughput falling with recompute and with empty slots.

- One currency instead of two weights. Count lost GPU-seconds l = recompute seconds + loss from empty slots; then
summed TTFT = base + (N - B) x l, exactly linear at saturation, not only to first order. c becomes a derived quantity:
P(the slot stays empty) for the batch, (price of an idle GPU-second) x P(the GPU idles) for a single server. Fetch's
lateness term and Retain's recompute term are then comparable by construction. Sketch: results 2(d) and 3(L4, L5).

- A system-level due date for reads. Under pass-over a read is needed for the mean only when the GPU would run out of
ready turns: T_starve = D + (ready turns behind) / (admission rate). Cost of landing at l: a fairness term on (l - D)+
for the turn itself and a mean term E[(l - T_starve)+]. Sketch: lemma L1 of result 3 while the ready backlog lasts, L2
after it. This makes c a quantity computed from the queue the engine already has, and explains why the hold never
binds at saturation.

- Estimators that match the proofs. mu = users in session - running turns once the pool has been full for one cycle
time, otherwise 1 + Q/(1 - load) capped at that value. c = slot-seconds left empty while a turn waits on its read,
over the stall-seconds (within 0.1 of the true value in simulation; section 6's estimator is too, for the batch).
kappa = (late cost rates waiting + arrival rate x late cost rate x residual)/(1 - device load). lambda = time-average
of the lowest index still held.

- Device rule as one index plus a feasibility test: earliest due date while that schedule meets every due date,
otherwise serve by w_j / max(p_j, d_j - t) (weighted modified due date; it reduces to Smith's ratio when all are
late). Writes that free room for a read inherit d' = min(d_w, T_j - p_j). On random instances this costs 1.24 times
the optimum in the mixed regime against 2.28 for the rule as drafted; the precedence part replaces the 50 K / 25 K
thresholds and is optimal for the maximum lateness (Lawler 1973).

- Preempt: use the skip variant of the greedy (factor 2 guaranteed; 5 % above the optimum on engine-sized instances
against 39 % for the plain greedy), or the exact dynamic programme over pages (about 2e4 steps per event), and rank
all classes in one list by surplus per token at the current price.

- Tighter bound with time-varying prices. Replace the constants lambda and kappa by one multiplier per period (the
dual of the constraint at each period, as in Brown and Smith). The bound stays valid for any non-negative path fixed
in advance, the best path solves a convex problem, and the device term stops being empty. For the device alone, the
conservation law of a work-conserving server gives a closed-form floor for linear costs (the c-mu value).

- Stability as a design rule: no rationing action may create work for the rationed resource. In the model, if the
write inflow does not increase with the dirty set (born clean, or delay), the clean state is the only fixed point and
attracts every start; otherwise there are two stable states between a_max and a_loop. Two usable numbers: the clean
reserve must keep dirty tokens below d_u(w) = d0 + (d1 - d0) p*(w)/p_max, and write priority alone removes the loop
when the uncontended write rate exceeds a_loop (14.9 K tokens/s here; the write thread alone does 17 to 21 K).

- State the regimes in users in session. With FA3 constants the pool fills at N_sat = B + Z/g0 = about 68 users
(campaign 34's first sustained queue came at 76). Below about 50 users c = 1 and mu is 1 to a few; above about 72, c =
0 and mu = N - B. A light-load evaluation setting on this box should stay below 50 users in session; the M/M/1//N
table in exp_mu.out gives mu across the knee.

- Add a second criterion for who waits. At saturation the mean is blind to read timing, so Fetch and the device order
have no optimum under the sum alone. With the longest reader stall as the second criterion, earliest due date is
lexicographically optimal (result 4g), and the paper can report readers' TTFT as the quantity that rule minimises.

- Make result 10 a theorem for the restricted problem: discretise idle age into phases, check the global attractor of
the index policy's fluid equation numerically, and cite Verloop for several classes with a knapsack constraint; or use
Brown and Smith's finite-horizon bound, which needs neither indexability nor the attractor and gives an explicit
sqrt(N) gap for the two-hour transient.


## D. Referee

Each of the 39 results of parts A to C was derived again from its assumptions and checked on new numerics
(`agents/verify_math/`): 33 confirmed as stated, 5 need an added condition, 1 left open (many sessions).

| Result | Verdict |
|---|---|
| math_fetch R1. J_F with recourse, constant W: closed form and derivative | needs a condition |
| math_fetch R2. J_F for a random queue wait independent of the gap | needs a condition |
| math_fetch R3. When the threshold rule 'start when F(x|s) >= q' is optimal | confirmed |
| math_fetch R4. Falling hazard: the optimum is at an end; exact comparison with a mass at infinity, a drop time and the device term | confirmed |
| math_fetch R5. Fetch against Retain: exact identity, hysteresis, and when Fetch before arrival is empty | needs a condition |
| math_fetch R6. After arrival: where the read should land | confirmed |
| math_fetch R7. The hazard of the trace's two gap entries | confirmed |
| math_fetch R8. J_F on the two entries: best start and gain over reading at arrival | confirmed |
| math_fetch R9. The start rules on the hints the replay actually sent | confirmed |
| math_fetch R10. Read or recompute (section 2): K_read, K_rec, the exact rule and the crossover | needs a condition |
| math_fetch R11. The two weights of section 1.5 | confirmed |
| math_fetch R12. Order on the device (section 3.4) | confirmed |
| math_fetch R13. The engine's 64-point read_start is not the minimiser of J_F in the tail | confirmed |
| math_retain R1. Holding problem: keep iff nu(a) >= lambda, first passage, dynamic consistency, indexability | confirmed |
| math_retain R2. nu is the Gittins index of the holding problem and the Whittle index of the session | confirmed |
| math_retain R3. Geometric form: the index is the slope of a convex hull | confirmed |
| math_retain R4. Special cases and closed forms (exact hint, falling hazard, rising hazard, ended; Weibull, Lomax, lognormal) | confirmed |
| math_retain R5. Rising hazard with an end probability: the draft's formula is wrong | confirmed |
| math_retain R6. Many sessions: exact for the priced problem, a heuristic under the hard capacity; the size term | confirmed |
| math_retain R7. Price-free index with the queue: one formula for idle prefixes, staged reads and queued turns | confirmed |
| math_retain R8. Retain when the dropped prefix can be fetched back early (K replaced by the optimal Fetch cost) | confirmed |
| math_retain R9. Value of a write; at a constant device price a write is never better early than late | confirmed |
| math_retain R10. The two lives of a dirty page; when skipping a write is optimal | needs a condition |
| math_retain R11. Due date of a write and the cost of missing it | confirmed |
| math_retain R12. Order of writes on one device | confirmed |
| math_retain R13. When eager writing is optimal; this workload | confirmed |
| math_retain R14. Born clean, clean only on a full acknowledgement, and revocation | confirmed |
| math_retain R15. One tier up: the backup GPU to host is a job due at GPU eviction (the mirror) | confirmed |
| math_retain R16. One tier down: SSD capacity with chained pages (tail before head; the index of a shared head) | confirmed |
| math_order_price 1. The weight mu in an open queue | confirmed |
| math_order_price 2. The weight mu in the closed loop: N for response time, N - B for TTFT | confirmed |
| math_order_price 3. The lateness weight c under pass-over | confirmed |
| math_order_price 4. Order on the device: Jackson, Horn, and what happens when due dates cannot be met | confirmed |
| math_order_price 5. Preempt on host room: the covering knapsack and the greedy bound | confirmed |
| math_order_price 6. Price: the relaxed problem, the marginal holder, complementary slackness, and the frontier estimate | confirmed |
| math_order_price 7. The bound: weak duality in the open model, and why it fails in the closed loop | confirmed |
| math_order_price 8. Saturation: the identity, the tie between policies, and a tight floor | confirmed |
| math_order_price 9. Stability: the smallest model of campaign 32's loop and the two rules that remove it | confirmed |
| math_order_price 10. Many sessions: what index-policy theory gives and what it does not | left open |

### Conditions added and the open result

**math_fetch R1. J_F with recourse, constant W: closed form and derivative** (needs a condition).

Check: Re-derived from the two events: if the gap R <= s the read starts at arrival (stall x, no hold); if R > s it
lands at s + r, so stall = (x - (R - s))^+ and hold = (R - s - x)^+. Swapping the integrals gives E[stall] =
integral_s^{s+x} F and E[hold] = M(s + x); with M' = -S the slope is c[F(s+x) - F(s)] - lambda L S(s+x) = S(s)[c
F(x|s) - lambda L (1 - F(x|s))]. The sign is right: the slope is positive exactly when the chance of an arrival within
x exceeds q. Own simulation of the policy
(/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/verify_math/vf_fetch.py, block r1): 750 cells
(gamma, Lomax alpha 3.5, Weibull k 0.6, three atoms, capped tool entry, uniform; c and lambda L including 0; x from 1
ms to 1,000 s; 5 starts), 1.5 M gaps per law, z-scores with the exact standard error (E[cost^2] by quadrature): worst
|z| 3.65, and 3.13, 2.03, 2.14 in three reruns with other seeds, no cell repeating. Slope formula against finite
differences: within 6.6e-3 (worst at s = 0 on Weibull k 0.6, where the density is infinite); against event counts
worst |z| 3.48. Atoms: J continuous, left and right slopes -0.05 / -0.35 at s = 2 and -0.35 / +0.40 at s = 6 = 9 - x;
the formula gives the right slopes. Counter-case to 'p_inf = 0 is enough': Lomax alpha 0.8 (proper law, infinite
mean): E[min((R - x)^+, cut)] = 45, 153, 426, 1,069 for cut = 1e3, 1e5, 1e7, 1e9, so the hold and J_F are infinite at
every finite start. Not academic: the quantiles quoted for the trace (tool p90 16.8 s to p99 213 s; human 1,040 s to
29,366 s) give tail exponents 0.91 and 0.69, below 1; the replayed tool gaps (p90 8.1 s, p99 82 s) give 1.0.

Statement to use: Add 'the remaining gap has a finite mean' to the conditions (p_inf = 0 is not enough). With it:
J_F(s) = c x - c M(s) + (c + lambda L) M(s + x) + kappa L/beta, J_F(arrival) = c x + kappa L/beta, J_F continuous,
right derivative S(s)[(c + lambda L) F(x|s) - lambda L] wherever S(s) > 0 (two-sided where the law has no atom at s or
s + x). Without a finite mean (a chance of never returning, or a tail heavier than 1/t) and with lambda > 0, J_F is
infinite at every finite start unless the landed copy has a drop time; with a drop time theta the slope in s is
unchanged for s + r < theta.

**math_fetch R2. J_F for a random queue wait independent of the gap** (needs a condition).

Check: Re-derived (B): conditioning on W = w gives c x_w^+ + lambda L (-x_w)^+ - c M(s) + (c + lambda L) M(s + x_w^+);
on {R > s}, R > s + (r - W)^+ is the same event as R + W > s + r, which gives dJ/ds = S(s)[(c + lambda L) Phi(s) -
lambda L]. Simulation (2 M samples, Weibull k = 3 gap, gamma waits, r = 4 s; vf_fetch.py block r2): formula against
simulated cost worst |z| 1.74; best start 28.5 s where Phi = q; with P(W <= r) = 0.143 < q = 0.6 arrival is best and
the best delay after arrival is 4.35 s = Q_W(q) - r. Hidden assumption found: case (C) uses the same law of W after
the arrival as before it. If the wait is observed at arrival (the engine's estimate has a median error of 0.4 to 0.5
s), a read released at arrival lands exactly at admission, arrival costs c E(r - W)^+ only, and J(s) = J_B(s) - lambda
L E[(W - r)^+] F(s). Example: W = 0 or 30 s (half each), gap near 25 s, r = 4, c = 1, lambda L = 0.25: (B) and (C)
both start at 23.75 s (cost 5.06 against 5.25 at arrival); with W seen at arrival, arrival costs 2.00 and is the
optimum, the early start costs about 3.2. In the gamma-wait case the first-order root moves from 28.5 to 28.75 s. The
summary's 'nothing is read before arrival iff P(W <= r) < q' is false in the 'only if' direction: W = 0, r = 4,
exponential gap of mean 200 s, q = 0.5 has P(W <= r) = 1 and F(x|s) = 0.0198 < q for all s.

Statement to use: (B) stands. If nothing about W is learned at arrival, dJ/ds = S(s)[(c + lambda L) Phi(s) - lambda L]
and re-timing after arrival does not move the best start. If W is observed at arrival, dJ/ds = S(s)[(c + lambda L)
Phi(s) - lambda L (1 + h(s) E[(W - r)^+])], h the hazard of the gap: start only when Phi(s) >= q (1 + h(s) E[(W -
r)^+]). In both cases P(W <= r) < q is sufficient, not necessary, for 'no read before arrival'. Assumptions: W
independent of the gap, r constant, linear costs.

**math_fetch R5. Fetch against Retain: exact identity, hysteresis, and when Fetch before arrival is empty** (needs a condition).

Check: Re-derived both gains; every term in S(theta) and M(theta) cancels and the margin is K_net S(s) - (c + lambda
L)[M(s) - M(s+x)] = S(s){(c + lambda L) E[(x - R_s)^+] + kappa L/beta - lambda L (r - r_m)}. Numerically (vf_fetch.py
block r5): both forms agree to 5.8e-15 on 1,500 random cells, two laws with a mass at infinity included; G_keep
simulated -0.8558 +- 0.0017 against -0.8565. The tool-entry numbers reproduce: at lambda = 2e-5 Retain keeps to 6.8 s,
a missing copy is worth 0.307 s at age 0 and 0.011 s at 0.8 s, keep value 2.065; at 1e-6 keep to 204.9 s and a missing
copy is worth 1.67 to 2.02 s up to 30 s and 0.51 s at 100 s. Counterexample to item (4) as worded: exponential gap of
mean 12 s (constant hazard), r = 10 s of which the pages are held 1 s, W = 0, c = 1, lambda L = 1, kappa = 0: K_net x
hazard = 0.917 < lambda L, so Retain drops at once, yet a read released at once and dropped at 10 s gains +2.00
(simulated +2.01) while keeping gains -0.57. The margin is negative in 541 of the 1,500 random cells.

Statement to use: Items (1) to (3) stand, with 'keeping beats reading back' meaning 'by the margin, which is >= 0 only
under (2)'. Item (4) needs the condition of item (2): with a hazard that never rises and unchanged c, lambda, kappa,
W, r, a copy that Retain dropped is never worth reading back before arrival provided a read is charged rent for its
whole duration (r_m = r) or kappa/beta >= lambda (r - r_m). If reads wait in the device's line without holding pages
and device time is unpriced (the long read line of campaign 34), a read released before arrival can pay on such a key.

**math_fetch R10. Read or recompute (section 2): K_read, K_rec, the exact rule and the crossover** (needs a condition).

Check: Re-derived K_rec - K_read on both sides of the point where the read stops hiding in the queue wait. Compared
the stated rule with a direct comparison on every length 64..65,536 for 192 parameter cells (gamma 0.037 / 0.044 ms,
mu 1, 2, 5, 128, c 1 / 0.3, W 0, 2, 5 s, kappa 0, 0.5, 2, 5;
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/verify_math/vo_sched.py block read_or_recompute):
the rule as written is wrong in 40 cells, all with D0 < 0 and mu gamma <= kappa/beta; with the missing branch it is
right in all 192. Counterexample: c = 1, kappa = 2, mu = 1, gamma = 0.04 ms, W = 5 s, lags 0.9 s: the rule says 'read
below 40.9 K', but at 20 K the read costs 1.87 s of device time alone against 0.80 s to recompute, and recompute wins
at every length. Everything else reproduces: gamma beta = 0.792 to 0.942; mu_x = 1.26 to 1.06; K_rec = 0.92 to 1.10 s
and K_read = 2.07 s at 25 K; crossovers at W = 0, c = 1: never at mu = 1, 33.0 / 21.8 K at mu = 2, 6.51 / 5.19 K at mu
= 5, 192 / 161 tokens at mu = 128; at c = 0.3: 11.75 / 9.01 K, 4.50 / 3.65 K, 1.58 / 1.31 K, 57 / 48 tokens; with the
in-flight rent at lambda = 2e-5 the read is cheaper on 0 of 1,024 lengths at mu = 2; 74 to 121 at c = 0.

Statement to use: Exact rule, lambda = 0, D0 = V + lags - W: if D0 >= 0, read iff mu gamma > (c + kappa)/beta and L >
c D0 / (mu gamma - (c + kappa)/beta). If D0 < 0: read at every length when mu gamma >= (c + kappa)/beta; recompute at
every length when mu gamma <= kappa/beta (while the read hides in the wait only device time counts); otherwise read
iff L < c (-D0) / ((c + kappa)/beta - mu gamma). The three changes to K_read (the wait hides the lags first; rent only
while pages are held; the after-arrival optimum when W can exceed r) stand.

**math_retain R10. The two lives of a dirty page; when skipping a write is optimal** (needs a condition).

Check: Single gap: backward induction against min(min C_W, min C_N) on 400 random cases: 0 mismatches; the test
V0(K_d) - V0(K_c + w) >= w p_inf agrees with the direct comparison in all; the necessary and the sufficient threshold
on P(return) never violated. Hand check of one of their numbers: tool law with P(ended) = 0.05, K_c/L = w/L = 1/beta,
lambda = 1e-6: the index with K_c + w reaches lambda at 45.1 s, where P(return) = 0.39. Several gaps (my probe,
vr_write.py block multi_gap: each turn is the last with probability 0.02, 0.1 or 0.3, the page stays dirty across
returns until written; fixed point over the continuation values, 300 random cases): 'write, then keep holding' is
still never strictly optimal, but the single-gap test picks a different life in 39 cases, and in all 39 the
several-gap optimum writes where the single-gap test says 'drop unwritten' (never the other way), because a write also
serves the later gaps.

Statement to use: For one gap the statement stands as written. Over several gaps: a page is still written only at the
moment it is dropped (or never), but the test 'write iff V0(K_d) - V0(K_c + w) >= w p_inf' is then sufficient for
writing and not necessary; the exact test compares the two lives with the continuation values of a clean and of a
dirty page at the next gap. Consequences (1) ended session: never write, and (3) a page that returns before its write
age is not written in this gap, stand.

**math_order_price 10. Many sessions: what index-policy theory gives and what it does not** (left open).

Check: The scope claim is right by inspection: the cited theorems cover one resource, two actions per arm and arms
that evolve independently, and the joint design has three resources, several actions and sessions coupled through the
GPU queue and the device line. For the Retain-only problem my simulation shows the index policy reaching the relaxed
bound (0.0934, 0.0755, 0.0730 against 0.0730 at 32, 128, 512 sessions; 0.0340, 0.0323, 0.0322 against 0.0324 with room
for half). The literature statements match my memory, and two quick look-ups confirm the main ones: Weber and Weiss
prove asymptotic optimality when the fluid model has a globally stable equilibrium and give a counterexample
([abstract](https://www.dpmms.cam.ac.uk/~rrw1/abstracts/w90c.html)), their addendum shows three-state bandits always
satisfy the condition ([addendum](https://www.dpmms.cam.ac.uk/~rrw1/abstracts/w90b.html)), Verloop extends this to
several classes with arrivals under the same kind of condition ([Verloop 2016](https://arxiv.org/abs/1609.00563v1)),
and Brown and Smith obtain asymptotic optimality of Lagrangian index policies in a finite horizon with tie-breaking
playing an essential role ([Brown and Smith 2020](https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.2019.3342)). The
snippets did not show the 'four states' of the counterexample or the square-root-N rate, and nobody has checked the
global-attractor condition for this model.

Statement to use: For the paper: 'In the open model restricted to Retain (hold or drop, restore cost given, sessions
independent given their hints), evicting the lowest index when room is needed has a cost per session that approaches
the relaxed bound as the number of sessions grows; this is observed in simulation and follows from restless-bandit
theory if the fluid model of the index policy has a global attractor, which is not verified. Nothing is proved for the
joint design.' Read the four papers before citing the rate or the four-state counterexample.


### What the three derivations missed

1. What is known about the queue wait at arrival changes the pre-arrival rule. math_fetch R2 lets the same law of W
hold before and after the arrival; the engine observes W at arrival (median error 0.4 to 0.5 s). Then a read released
at arrival lands exactly at admission, waiting has an option value, and the start condition is Phi(s) >= q (1 + h(s)
E[(W - r)^+]) instead of Phi(s) >= q. Example: W = 0 or 30 s, r = 4 s, lambda L = 0.25, c = 1: the stated rule starts
at 23.75 s (cost 5.06); the true optimum is the arrival (cost 2.00). Output:
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/verify_math/vf_fetch.out, block R2.

2. A finite mean gap is a real condition, not a formality. J_F with a copy held to admission is infinite not only when
a session may never return but for any tail heavier than 1/t. The quantiles quoted for the trace give tail exponents
of 0.91 (tool, p90 16.8 s to p99 213 s) and 0.69 (human, 1,040 s to 29,366 s), and the replayed tool gaps give 1.0. So
in the replay the 1,200 s cap, not the hint, decides the rent of every staged read; the drop time of Theorem 3 belongs
in the Fetch rule itself.

3. 'Fetch before arrival is empty on falling-hazard keys' needs the read to pay rent for its whole duration or a
priced device. When reads wait in the device's line without holding pages (pages are allocated at dispatch) and kappa
is near 0, the margin of the Fetch-against-keep identity is negative and releasing a read early, to join the line,
pays even though Retain drops the copy (my counterexample gains +2.0 s where keeping gains -0.57 s). This is the
regime of campaign 34's read line (planned read time 8.3 s against 30 to 52 s actual). None of the three connected the
identity to that regime.

4. The landing rule and the pass-over result contradict each other unless c is allowed to depend on the length of the
stall. math_fetch R6 minimises a cost linear in lateness with a constant c; math_order_price 3 shows the mean cost of
lateness is zero while ready turns keep the slot filled and about one second per second after that. The cost is
therefore convex in lateness, the fixed-quantile rule is exact only for the linear model, and for the mean the due
date of a read is the time at which the GPU would run out of ready turns, not D. No theorem in the three reports
states the landing optimum under that cost.

5. Shared pages in host memory have no index in the draft or in the reports. Theorem 7 is per session, but 17 % of
prompt tokens are held by earlier sessions and first requests find 24 %. A shared head is needed at the first return
of any sharer and stays needed afterwards, so its holding problem is not a single stopping problem: for hazards past
their peaks its myopic index is (K/L) times the sum of the sharers' hazards, each at its own idle age, and that sum
jumps up whenever a sharer returns. math_retain R16 treats only the SSD copy of a shared head; math_order_price 7 only
excludes shared pages from the bound.

6. The write-or-not test is a single-gap result. With several gaps the structure survives (write only at the moment of
dropping) but the test under-writes: in 39 of 300 random cases the several-gap optimum writes where the single-gap
test says 'drop unwritten', never the reverse. Output:
/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/verify_math/vr_write.out, block 'R10 probe'.

7. The read-or-recompute rule has a missing branch: when the read hides entirely in the queue wait, only device time
counts, so if mu gamma <= kappa/beta recompute wins at every length. The rule as written by math_fetch says 'read
below a threshold' there (wrong in 40 of 192 parameter cells once kappa is 2 or 5). It matters exactly where the draft
wants read-or-recompute by price: light GPU load with a congested device.

8. The GPU pool has its own price and it is not small. From the pool-bound batch model, an empty slot costs about one
second of the waiting turn plus N a / (B tau(B)) = 0.96 s of lost throughput per second, i.e. about 2 TTFT-seconds per
slot-second, or about 6e-5 per token-second for a slot of 33 K tokens: some 50 times the host prices the reports infer
(around 1e-6). The draft's line 'the same index orders GPU eviction' and math_retain R15 (iv) leave that price
unstated. This figure is my arithmetic on their constants, not a measurement.

9. The index policy does reach the priced bound in simulation. math_retain reports a residual +2 to 3 % at 512 to
2,048 sessions; my independent simulation gives 0.0730 against a bound of 0.0730 at 512 sessions with room for a
quarter (0.0322 against 0.0324 with room for half), and the index of the evicted prefix equals lambda* to within 1 to
9 % from 128 sessions up. Their residual is an implementation effect; it should not be quoted as a property of ranking
by the index.

10. A closed form for the stall weight in an open queue was within reach and nobody wrote it: for short stalls in
M/G/1 the cost of a stall-second is 1 / E[exp(-alpha s)] (alpha the arrival rate, s the work of a turn), which gives 1
+ rho for exponential work and exp(rho) for equal work, always at least 1, against the draft's P(no other ready turn)
= (1 - rho) / E[exp(-alpha s)], always at most 1. Checked by simulation: 1.34, 1.78, 2.25 against 1.35, 1.82, 2.23 for
equal work at loads 0.3, 0.6, 0.8.

11. The sample behind math_fetch R9 overstates the losses. On all 16,118 hints the cost-minimising start held to
admission loses 0.33 to 2.42 s per turn at lambda >= 2e-5, not 0.8 to 2.5 s, and the model's E[(gap - 2 s)^+] is 15.2
s, not 16.1 s. The directions are unchanged, but the paper should quote full-population numbers.

