"""SSD capacity with chained pages: checks of DESIGN.md 4.3 (the cleaner).

Lookup semantics (hicache_nixl: chained keys): a lookup for a session starts just below its
resident anchor (the deepest page still in GPU or host memory) and walks down the chain; it
stops at the first page missing from the SSD. Pages after a missing page are unreachable.

S1. One chain, every lookup starts at the root (nothing resident): for every number of pages to
    delete, deleting the tail is optimal and the optimum is unique (brute force over subsets).
S2. Lookups that start below a resident head: tail-first is no longer always optimal; the
    exchange holds for a pair i < j iff no lookup starts in [i, j).
S3. A tree of pages (shared heads), root-anchored lookups: for every SSD price sigma the best
    set is {v : theta * rho(v) > sigma}, rho(v) = summed lookup rate of the sessions below v;
    the sets are nested, tails go before heads, and a head's index is the sum over its sessions.
S4. A shared head that is sometimes resident: its marginal value is
    sum_i rho_i^cold * (len(head) + stored chain of i), and 0 if it is always resident.

Run: python -I -B ssd_chain.py
"""

from __future__ import annotations

import itertools
import math

import numpy as np

RNG = np.random.default_rng(11)


def retrieved_chain(present: tuple[bool, ...], anchor: int) -> int:
    """Pages read back by a lookup that starts after `anchor` resident pages (0 = root)."""
    k = anchor
    while k < len(present) and present[k]:
        k += 1
    return k - anchor


def s1_root_anchored() -> None:
    bad = 0
    for n in range(2, 13):
        for m in range(1, n):
            best = -1
            arg = []
            for deleted in itertools.combinations(range(n), m):
                present = tuple(i not in deleted for i in range(n))
                got = retrieved_chain(present, 0)
                if got > best:
                    best, arg = got, [deleted]
                elif got == best:
                    arg.append(deleted)
            tail = tuple(range(n - m, n))
            if best != n - m or arg != [tail]:
                bad += 1
    print(f"  S1 chains of 2..12 pages, all deletion counts: tail deletion is the unique optimum; violations {bad}")


def s2_resident_heads() -> None:
    # chain of n pages; lookups start at anchor g with probability p_g (g pages resident), g < n
    n = 8
    print("  S2 chain of 8 pages, one page to delete; expected pages lost per lookup = P(g < j) * (n - j + 1):")
    for label, probs in (
        ("all lookups from the root (g = 0)", [1.0] + [0.0] * 7),
        ("anchor uniform on 0..7", [1 / 8] * 8),
        ("head hot: g = 0 w.p. 0.02, else uniform on 4..7", [0.02, 0, 0, 0] + [0.245] * 4),
    ):
        losses = []
        for j in range(1, n + 1):  # delete page j (1-based)
            present = tuple(i != j - 1 for i in range(n))
            loss = sum(p * ((n - g) - retrieved_chain(present, g)) for g, p in enumerate(probs))
            formula = sum(p for g, p in enumerate(probs) if g < j) * (n - j + 1)
            assert abs(loss - formula) < 1e-12
            losses.append(loss)
        best = int(np.argmin(losses)) + 1
        print(f"     {label:48s} loss by deleted page 1..8: " + " ".join(f"{x:.2f}" for x in losses) + f"  -> delete page {best}")
    # exchange condition: for i < j, swapping 'delete i' for 'delete j' never hurts iff no anchor in [i, j)
    viol = 0
    checks = 0
    for _ in range(3000):
        n = int(RNG.integers(3, 9))
        deleted = set(int(x) for x in RNG.choice(n, size=int(RNG.integers(1, n)), replace=False))
        kept = [x for x in range(n) if x not in deleted]
        if not kept:
            continue
        i = int(RNG.choice(sorted(deleted)))
        later = [x for x in kept if x > i]
        if not later:
            continue
        j = int(RNG.choice(later))
        anchors = [g for g in range(n) if not (i < g <= j)]  # g pages resident; anchor inside [i, j) in 1-based pages
        present_a = tuple(x not in deleted for x in range(n))
        swapped = (deleted - {i}) | {j}
        present_b = tuple(x not in swapped for x in range(n))
        for g in anchors:
            checks += 1
            if retrieved_chain(present_b, g) < retrieved_chain(present_a, g):
                viol += 1
    print(f"  S2 exchange (delete the later page j instead of the earlier i), lookups with no anchor between them: {checks} checks, violations {viol}")


def s3_tree() -> None:
    """Random trees; node v has size len_v; each leaf-to-root path belongs to one session with
    lookup rate rho_s (root-anchored). value(S) = theta * sum over sessions of rho_s * tokens read."""
    theta = 1.0
    bad_lagr = bad_nested = bad_formula = 0
    trials = 200
    for _ in range(trials):
        n = int(RNG.integers(4, 11))
        parent = [-1] + [int(RNG.integers(0, v)) for v in range(1, n)]
        size = RNG.integers(1, 6, n).astype(float)
        children = [[] for _ in range(n)]
        for v in range(1, n):
            children[parent[v]].append(v)
        # one session ends at every node with probability 0.6 (always at leaves)
        sessions = [(v, float(RNG.uniform(0.1, 2.0))) for v in range(n) if not children[v] or RNG.random() < 0.6]

        def path(v: int) -> list[int]:
            out = []
            while v != -1:
                out.append(v)
                v = parent[v]
            return out[::-1]

        rho = np.zeros(n)
        for v, r in sessions:
            for u in path(v):
                rho[u] += r

        def value(mask: int) -> float:
            tot = 0.0
            for v, r in sessions:
                got = 0.0
                for u in path(v):
                    if not (mask >> u) & 1:
                        break
                    got += size[u]
                tot += r * got
            return theta * tot

        vals = np.array([value(mask) for mask in range(1 << n)])
        sizes = np.array([sum(size[u] for u in range(n) if (mask >> u) & 1) for mask in range(1 << n)])
        prev_set = None
        for sigma in sorted(set(np.round(np.concatenate((rho * theta * 0.999, rho * theta * 1.001, [0.0, 1e9])), 9))):
            claim = sum(1 << u for u in range(n) if theta * rho[u] > sigma)
            best = float(np.max(vals - sigma * sizes))
            if vals[claim] - sigma * sizes[claim] < best - 1e-9:
                bad_lagr += 1
            if prev_set is not None and (claim & ~prev_set):
                bad_nested += 1  # sets must shrink as sigma grows
            prev_set = claim
            # exact for the hard capacity equal to the claimed set's size
            if vals[claim] < float(np.max(vals[sizes <= sizes[claim] + 1e-9])) - 1e-9:
                bad_lagr += 1
        # rho is non-increasing down every path, so a child is deleted no later than its parent
        for v in range(1, n):
            if rho[v] > rho[parent[v]] + 1e-12:
                bad_formula += 1
    print(
        f"  S3 {trials} random trees (4..10 page groups): threshold set not optimal for its price or its size: {bad_lagr}; "
        f"not nested: {bad_nested}; a child with a higher rate than its parent: {bad_formula}"
    )


def s4_shared_head() -> None:
    """Head h (len_h tokens) shared by m sessions with private chains n_i. A lookup of session i
    starts at the root when the head is cold (rate rho_i^A) and below the head when it is
    resident (rate rho_i^B). Brute force over 'head on SSD or not' with all private chains stored."""
    bad = 0
    for _ in range(500):
        m = int(RNG.integers(1, 6))
        len_h = float(RNG.integers(1, 20))
        chain = RNG.integers(1, 40, m).astype(float)
        rho_a = RNG.uniform(0.0, 1.0, m)
        rho_b = RNG.uniform(0.0, 1.0, m)

        def value(head_on_ssd: bool) -> float:
            tot = 0.0
            for i in range(m):
                tot += rho_b[i] * chain[i]  # head resident: the private chain is read whatever the SSD holds of the head
                if head_on_ssd:
                    tot += rho_a[i] * (len_h + chain[i])
            return tot

        marginal = value(True) - value(False)
        formula = float(np.sum(rho_a * (len_h + chain)))
        if abs(marginal - formula) > 1e-9:
            bad += 1
    print(f"  S4 marginal value of a shared head's SSD copy = sum_i rho_i^cold (len_h + chain_i): mismatches {bad}/500")
    # numbers: head 6,000 tokens (about a quarter of a 25,000-token first prompt), 5 sessions with 20,000-token chains
    len_h, chain_len, m = 6_000.0, 20_000.0, 5
    for cold_share in (1.0, 0.3, 0.05, 0.0):
        rate = 1.0 / 600.0  # one SSD lookup per session per 10 minutes
        head = m * rate * cold_share * (len_h + chain_len) / len_h
        tail = rate
        print(
            f"     5 sessions below a 6K head, 20K private chains, cold share {cold_share:4.2f}: head index / tail index = {head / tail:6.2f}"
        )


def main() -> None:
    print("=" * 100)
    print("SSD tier with a first-miss lookup")
    s1_root_anchored()
    s2_resident_heads()
    s3_tree()
    s4_shared_head()


if __name__ == "__main__":
    main()
