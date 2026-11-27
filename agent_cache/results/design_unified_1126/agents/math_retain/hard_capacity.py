"""What "evict in increasing nu" loses under a hard capacity (DESIGN.md 4.1, 7.2, 7.10).

1. Two analytic counterexamples (enumerated).
2. Offline brute force on small request sequences with exact hints: the index order is Belady's
   farthest next use and is optimal for equal sizes and equal costs; it is not with unequal
   costs or unequal sizes.
3. Many sessions, stochastic gaps: LRU and the index policy under a hard capacity against the
   Lagrangian (priced, TTL) lower bound, for N = 8 ... 2048 at a fixed capacity share.

Run: python -I -B hard_capacity.py
"""

from __future__ import annotations

import functools
import itertools
import math
import os
import sys

import numpy as np

# -I leaves the script's folder off sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hold_index import (  # noqa: E402
    held_time,
    index_hull,
    surv_two_piece_lognormal,
)

RNG = np.random.default_rng(1126)


# ----------------------------------------------------------------------------- 1. analytic counterexamples
def counterexample_unequal_cost() -> None:
    # one slot, two prefixes of equal size, exact hints: A back in 10 s (K=1), B back in 11 s (K=1.05)
    nu_a, nu_b = 1.0 / 10.0, 1.05 / 11.0
    index_evicts = "B" if nu_b < nu_a else "A"
    cost_index = 1.05 if index_evicts == "B" else 1.0
    cost_best = min(1.0, 1.05)
    print(f"  A: K=1, back in 10 s -> nu={nu_a:.4f};  B: K=1.05, back in 11 s -> nu={nu_b:.4f}")
    print(f"  index evicts {index_evicts}: cost {cost_index:.2f};  best: evict A, cost {cost_best:.2f}  (ratio {cost_index/cost_best:.3f})")


def counterexample_equal_cost() -> None:
    # one slot, equal sizes, equal K=1.  A: back at 1 s with prob 1/2, else never.  B: back at 10 s surely.
    # no early re-fetch.  index: nu_A = sup_h F/tau = 0.5/1 = 0.5 ; nu_B = 1/10.
    nu_a, nu_b = 0.5 / 1.0, 1.0 / 10.0
    cost_keep_a = 1.0  # B evicted, returns surely -> pays 1 ; A held, never pays
    cost_keep_b = 0.5  # A evicted, returns with prob 1/2 -> pays 0.5 ; B held
    print(f"  nu_A={nu_a:.2f} > nu_B={nu_b:.2f}: index keeps A and evicts B: expected cost {cost_keep_a:.2f}")
    print(f"  keeping B instead costs {cost_keep_b:.2f}: the index policy pays {cost_keep_a/cost_keep_b:.1f}x the optimum")
    print("  reason: after 1 s the slot is empty and worth nothing; the index charges B for 10 s of room at one price")


# ----------------------------------------------------------------------------- 2. offline brute force
def offline_optimum(seq: tuple[int, ...], sizes: tuple[int, ...], costs: tuple[float, ...], cap: int) -> float:
    """Min total miss cost. At a request the item is loaded (after paying on a miss); then any
    items, the new one included, may be dropped until the sizes fit."""
    n_items = len(sizes)

    @functools.lru_cache(maxsize=None)
    def best(t: int, cache: frozenset) -> float:
        if t == len(seq):
            return 0.0
        item = seq[t]
        pay = 0.0 if item in cache else costs[item]
        full = cache | {item}
        out = math.inf
        members = sorted(full)
        # enumerate subsets that fit; prune dominated ones lightly by requiring maximal fits
        for r in range(len(members) + 1):
            for drop in itertools.combinations(members, r):
                keep = full - set(drop)
                if sum(sizes[i] for i in keep) > cap:
                    continue
                out = min(out, best(t + 1, frozenset(keep)))
            if out < math.inf and r >= 1:
                # dropping more than needed never helps when a superset fits (monotone), stop early
                break
        return pay + out

    return best(0, frozenset())


def index_policy_cost(seq, sizes, costs, cap) -> float:
    """Exact hints: nu_i = K_i / (L_i * time to next use); evict lowest nu until the sizes fit."""
    cache: set[int] = set()
    total = 0.0
    for t, item in enumerate(seq):
        if item not in cache:
            total += costs[item]
        cache.add(item)
        while sum(sizes[i] for i in cache) > cap:

            def nu(i: int) -> float:
                nxt = next((u - t for u in range(t + 1, len(seq)) if seq[u] == i), math.inf)
                return 0.0 if nxt == math.inf else costs[i] / (sizes[i] * nxt)

            victim = min(cache, key=lambda i: (nu(i), i))
            cache.remove(victim)
    return total


def offline_study(label: str, n_items: int, seq_len: int, cap: int, trials: int, sizes_fn, costs_fn) -> None:
    worse = 0
    worst = 1.0
    tot_idx = tot_opt = 0.0
    for _ in range(trials):
        seq = tuple(int(x) for x in RNG.integers(0, n_items, seq_len))
        sizes = sizes_fn()
        costs = costs_fn(sizes)
        opt = offline_optimum(seq, sizes, costs, cap)
        idx = index_policy_cost(seq, sizes, costs, cap)
        tot_idx += idx
        tot_opt += opt
        if idx > opt + 1e-9:
            worse += 1
            worst = max(worst, idx / opt if opt > 0 else math.inf)
    print(
        f"  {label:58s} suboptimal in {worse:4d}/{trials} sequences; worst ratio {worst:.3f}; "
        f"total cost index/optimum = {tot_idx/tot_opt:.4f}"
    )


# ----------------------------------------------------------------------------- 3. many sessions
class GapClass:
    """A gap law on a grid, capped at cap_s (an atom at the cap, as the replay client does)."""

    def __init__(self, p10: float, p50: float, p90: float, cap_s: float, dt: float):
        n = int(round(cap_s / dt))
        t = np.linspace(0.0, cap_s, n + 1)
        surv = surv_two_piece_lognormal(t, p10, p50, p90)
        surv[-1] = 0.0  # everything still out at the cap returns at the cap
        self.dt, self.t, self.surv = dt, t, surv
        self.held = held_time(surv, dt)
        self.mean = float(self.held[-1])
        self.nu = index_hull(surv, self.held, 1.0)

    def sample(self, size: int) -> np.ndarray:
        u = RNG.random(size)
        j = np.searchsorted(-self.surv, -u, side="left")
        return np.maximum(j, 1) * self.dt


def priced_bound(classes: list["GapClass"], probs: list[float], rho: float) -> tuple[float, float]:
    """Weak-duality bound on the miss ratio of any non-anticipative policy that holds at most a
    share rho of the sessions:  max over price >= 0 of  sum_j p_j min_a [S_j(a) + price*I_j(a)/mean_gap] - price*rho.
    The inner minimum runs over every grid age, so the value is exact for the grid laws."""
    mean_gap = sum(p * c.mean for p, c in zip(probs, classes))

    def dual(price: float) -> float:
        return sum(p * float(np.min(c.surv + price * c.held / mean_gap)) for p, c in zip(probs, classes)) - price * rho

    prices = np.geomspace(1e-6, 1e4, 6001)
    vals = np.array([dual(x) for x in prices])
    i = int(np.argmax(vals))
    lo, hi = prices[max(i - 1, 0)], prices[min(i + 1, len(prices) - 1)]
    # the dual is concave: golden-section refine
    for _ in range(80):
        a = lo + 0.381966 * (hi - lo)
        b = lo + 0.618034 * (hi - lo)
        if dual(a) < dual(b):
            lo = a
        else:
            hi = b
    price = 0.5 * (lo + hi)
    return max(dual(price), float(vals[i])), price


def simulate(classes: list["GapClass"], probs: list[float], n_sessions: int, cap_slots: int, policy: str, n_events: int, warm_s: float):
    """Each session: gap, return (miss if not held), immediately idle again with a new gap class.
    Starts each session at a random point of a first gap, then warms up for warm_s seconds."""
    n_cls = len(classes)
    cls = RNG.choice(n_cls, size=n_sessions, p=probs)
    first = np.empty(n_sessions)
    for c in range(n_cls):
        sel = cls == c
        first[sel] = classes[c].sample(int(sel.sum()))
    phase = RNG.random(n_sessions)
    last_end = -phase * first
    next_ret = (1.0 - phase) * first
    held = np.zeros(n_sessions, dtype=bool)
    held[np.argsort(next_ret)[:cap_slots]] = True
    pooled = pooled_index(classes, probs) if policy == "index_pooled" else None
    hits = misses = 0
    dt = classes[0].dt
    nmax = len(classes[0].nu) - 1
    while hits + misses < n_events:
        i = int(np.argmin(next_ret))
        now = next_ret[i]
        if now > warm_s:
            if held[i]:
                hits += 1
            else:
                misses += 1
        held[i] = True
        last_end[i] = now
        cls[i] = RNG.choice(n_cls, p=probs)
        next_ret[i] = now + classes[cls[i]].sample(1)[0]
        if held.sum() > cap_slots:
            ages = now - last_end
            if policy == "lru":
                score = -ages
            elif policy == "index":
                k = np.minimum((ages / dt).astype(int), nmax)
                score = np.empty(n_sessions)
                for c in range(n_cls):
                    sel = cls == c
                    score[sel] = classes[c].nu[k[sel]]
            elif policy == "index_pooled":
                k = np.minimum((ages / dt).astype(int), nmax)
                score = pooled[k]
            elif policy == "oracle":  # Belady with the true next return (clairvoyant)
                score = -next_ret
            else:
                raise ValueError(policy)
            score = np.where(held, score, np.inf)
            held[int(np.argmin(score))] = False
    ratio = misses / (hits + misses)
    return ratio, math.sqrt(ratio * (1 - ratio) / (hits + misses))


class PooledClass:
    """The mixture law seen by a policy that does not know the class."""

    def __init__(self, classes: list["GapClass"], probs: list[float]):
        self.dt = classes[0].dt
        self.surv = sum(p * c.surv for p, c in zip(probs, classes))
        self.held = held_time(self.surv, self.dt)
        self.mean = float(self.held[-1])


def pooled_index(classes: list[GapClass], probs: list[float]) -> np.ndarray:
    surv = sum(p * c.surv for p, c in zip(probs, classes))
    held = held_time(surv, classes[0].dt)
    return index_hull(surv, held, 1.0)


def main() -> None:
    print("=" * 100)
    print("1a. Hard capacity, exact hints, unequal restore costs (one slot):")
    counterexample_unequal_cost()
    print("1b. Hard capacity, equal sizes and costs, a slot that falls idle (one slot, no early re-fetch):")
    counterexample_equal_cost()

    print("=" * 100)
    print("2.  Offline brute force, exact hints (6 items, 14 requests, capacity 3 size units, 300 random sequences each)")
    offline_study("equal sizes, equal costs (index order = Belady)", 6, 14, 3, 300, lambda: (1,) * 6, lambda s: (1.0,) * 6)
    offline_study(
        "equal sizes, costs in {1, 3} (weighted caching)", 6, 14, 3, 300,
        lambda: (1,) * 6, lambda s: tuple(float(x) for x in RNG.choice([1.0, 3.0], 6)),
    )
    offline_study(
        "sizes in {1, 2}, cost = size (K/L equal)", 6, 14, 4, 300,
        lambda: tuple(int(x) for x in RNG.choice([1, 2], 6)), lambda s: tuple(float(x) for x in s),
    )
    offline_study(
        "sizes in {1, 2}, costs equal", 6, 14, 4, 300,
        lambda: tuple(int(x) for x in RNG.choice([1, 2], 6)), lambda s: (1.0,) * 6,
    )

    print("=" * 100)
    print("3.  Many sessions, hard capacity vs the priced (Lagrangian) lower bound.")
    print("    Gap classes: tool 0.41/1.69/16.8 s (92 %), human 3.1/96/1040 s (8 %), both capped at 1,200 s; class known at the turn's end.")
    dt = 0.1
    tool = GapClass(0.41, 1.69, 16.8, 1200.0, dt)
    human = GapClass(3.1, 96.0, 1040.0, 1200.0, dt)
    classes, probs = [tool, human], [0.92, 0.08]
    mean_gap = sum(p * c.mean for p, c in zip(probs, classes))
    print(f"    mean gap: tool {tool.mean:.2f} s, human {human.mean:.1f} s, mixed {mean_gap:.2f} s")
    pooled_cls = PooledClass(classes, probs)
    for rho in (0.25, 0.5):
        bound, price = priced_bound(classes, probs, rho)
        bound_pooled, _ = priced_bound([pooled_cls], [1.0], rho)
        print(
            f"    capacity share rho = {rho}: priced lower bound on the miss ratio = {bound:.4f} (class known), "
            f"{bound_pooled:.4f} (one pooled law, no class); price per share-second {price:.4g}"
        )
        for n_sessions in (8, 32, 128, 512, 2048):
            cap_slots = max(1, int(round(rho * n_sessions)))
            row = []
            for policy in ("lru", "index_pooled", "index", "oracle"):
                m, se = simulate(classes, probs, n_sessions, cap_slots, policy, 300_000, 3600.0)
                row.append(f"{policy} {m:.4f}+-{se:.4f}")
            print(f"      N={n_sessions:5d} slots={cap_slots:5d}: " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
