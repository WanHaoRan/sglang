"""Skeptic check for the finding "forward_entry_time is stamped at build, so the load hold is
booked as prefill time and the meter's prefill-cost fit learns it".

Part 1 (code path): with measured prices on, can the fit's harm_s move any price? It goes to
PriceController.charge; measured prices force price_learning off, i.e. rule "fixed".
Part 2 (magnitude): the real PrefillCostModel fed synthetic turns shaped like campaign 34/40
(Appendix B of DESIGN.md), with and without the hold's extra time on host-loaded turns.
CPU only, no server.
"""

import math
import os
import random

os.environ["SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES"] = "1"

from sglang.srt.mem_cache.next_turn_policy import PriceController
from sglang.srt.mem_cache.next_turn_prices import PrefillCostModel
from sglang.srt.mem_cache.next_turn_switches import Switches

# ---- Part 1 -------------------------------------------------------------------------------
for learning in ("harm", "on"):
    try:
        Switches.read(
            acting=True,
            gated=True,
            holding=True,
            parking=True,
            staging_by_cost=True,
            price_learning=learning,
            priced_eviction=True,
        )
        print(f"part1: measured prices with price_learning={learning}: accepted (!)")
    except ValueError as exc:
        print(f"part1: measured prices with price_learning={learning}: refused: {exc}")

ctl = PriceController(price=5e-5, rule="fixed")
ctl.observe(now=0.0, idle_tokens=0, staged_tokens=100_000)
ctl.charge(1000.0)  # an absurd harm
ctl.observe(now=10.0, idle_tokens=0, staged_tokens=100_000)
update = ctl.step(room_tokens=1_000_000)
print(f"part1: rule fixed after charge(1000 s): price={ctl.price:g} logged harm_s={update.harm_s:g}")

# ---- Part 2 -------------------------------------------------------------------------------
A, B = 12.8e-6, 0.70e-9  # campaigns 33/34 at an empty queue (PrefillCostModel docstring)
STEP = 0.024  # decode step, s (prefill_load_hold.py)
COPY = 2.5e-6  # host-to-GPU, s per token (prefill_load_hold.py)
MAX_STEPS = 16  # campaign 40's SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS
MIN_LOAD = 8192


def pass_s(new, total):
    cached = total - new
    return A * new + B * (total * total - cached * cached)


def sample_turns(rng, n, host_share):
    """(new, total, host_loaded) of admitted turns: first turns, device hits, host loads."""
    turns = []
    for _ in range(n):
        kind = rng.random()
        if kind < 0.15:  # first turn, cold
            total = rng.randint(2_000, 30_000)
            turns.append((total, total, 0))
        else:
            total = int(min(64_000, max(4_000, rng.gauss(25_500, 8_000))))
            new = int(min(total - 1, max(64, rng.expovariate(1 / 1_100))))
            host = kind >= 1.0 - host_share  # copied from host memory, else on the GPU
            turns.append((new, total, total - new if host else 0))
    return turns


def measured_s(rng, new, total, loaded, hold):
    p = pass_s(new, total)
    lag = rng.uniform(0.0, STEP)  # overlap: previous batch's tail before the pass starts
    r = COPY * loaded
    if hold and loaded >= MIN_LOAD:
        steps = min(MAX_STEPS, math.ceil(r / STEP))
        # Released at the first step boundary after the copy landed; then the pass alone.
        return steps * STEP + max(0.0, r - steps * STEP) + p + lag
    # The forward waits on the copy layer by layer: max(r, p).
    return max(r, p) + lag


def fit(turns, rng, hold):
    model = PrefillCostModel(seed_a=1.1e-4)
    held = 0
    for new, total, loaded in turns:
        held += hold and loaded >= MIN_LOAD
        model.observe(
            new_tokens=new,
            total_tokens=total,
            seconds=measured_s(rng, new, total, loaded, hold),
        )
    return model, held


shapes = {
    "harm: recompute 20K lost (new=total=20K)": (20_000, 20_000),
    "quote: 16K missing on 25.5K (new=17.1K)": (17_100, 25_500),
    "starved: 12K span from 8K (total 20K)": (12_000, 20_000),
}
for host_share in (0.2, 0.45):
    turns = sample_turns(random.Random(1), 4_000, host_share)
    for hold in (False, True):
        model, held = fit(turns, random.Random(2), hold)
        a, b = model.coefficients()
        print(
            f"part2: host_share={host_share} hold={int(hold)} "
            f"held_samples={held}/{len(turns)} a={a * 1e6:.2f} us b={b * 1e9:.3f} ns"
        )
        for name, (new, total) in shapes.items():
            fitted = model.seconds(new_tokens=new, total_tokens=total)
            print(
                f"    {name}: fit {fitted * 1e3:.0f} ms"
                f" vs pass {pass_s(new, total) * 1e3:.0f} ms"
            )
