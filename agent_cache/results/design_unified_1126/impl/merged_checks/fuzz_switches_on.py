"""Random scenarios through the merged prefetcher with the unified arm's switches on: no exception, and the new
lines appear. It reuses the events and the fake caches of A/checks/test_switches_off_equivalence.py; here only the
merged module runs (there is nothing to compare with), on the arm's shape (priced cache, expected-cost timing, gate
and hold on, return_chance eviction with queued_order farthest, price learning off) and on random valid shapes.

CPU only. In the container, GPU hidden, from this folder:
  docker exec -e PYTHONPATH=/sgl-workspace/sglang/python:/sgl-workspace/sglang/test/registered/unit/mem_cache:\
/sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/A/checks \
      -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
      'cd /sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/merged_checks &&
       nice -n 10 python3 -m pytest -q -p no:cacheprovider -s fuzz_switches_on.py'
"""
import collections
import contextlib
import functools
import random
from types import SimpleNamespace

import test_switches_off_equivalence as harness

from sglang.srt.environ import envs

ARM = dict(
    cache="priced", mode="on", timing="expected_cost", gate="on", hold="on", soft_pins="on", learning="off",
    landing_quantile=None, price=5e-5, policy="return_chance", score="entry", room_retry=True,
)
SWITCHES = dict(
    SGLANG_LOG_HICACHE_PRICES_INTERVAL=10.0,
    SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES=True,
    SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT=True,
    SGLANG_HICACHE_NEXT_TURN_READ_LINE=True,
    SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE=True,
    SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS=True,
)


@contextlib.contextmanager
def switches_on():
    with contextlib.ExitStack() as stack:
        for name, value in SWITCHES.items():
            stack.enter_context(getattr(envs, name).override(value))
        yield


class Side(harness.Side):
    """The harness's side with the request fields the Retain half reads at a commit."""

    def apply(self, event, now):
        if event[0] == "commit":
            # The prompt as the engine's Req carries it; its length is all the code reads.
            self.req(event[1]).origin_input_ids = [0] * event[3]
        super().apply(event, now)


def run(config, events, *, explicit_ends, retire_ended):
    clock = SimpleNamespace(now=0.0)
    module = harness.changed
    fixtures = harness.fixtures
    original, priced = module.NextTurnPrefetcher, fixtures.PricedCache
    module.NextTurnPrefetcher = functools.partial(original, explicit_ends=explicit_ends, retire_ended=retire_ended)
    # The fake that also answers what the measured prices and the read line ask of the cache.
    fixtures.PricedCache = fixtures.MeasuredCache
    try:
        with switches_on(), envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(config["room_retry"]):
            side = Side(module, config, lambda: clock.now, prices_interval_s=10.0)
    finally:
        module.NextTurnPrefetcher, fixtures.PricedCache = original, priced
    # What the retire asks of the cache, as the Retain half's own tests stub it.
    retired = side.cache.retired = []
    side.cache.get_last_hash_value = lambda node_id: f"key-{node_id}"
    side.cache.retire_storage_chain = lambda tail, *, keep_tokens, own_leaf_keys: retired.append(
        (tail, keep_tokens, set(own_leaf_keys))
    )
    strategy = side.cache.tree_core.eviction_strategy
    if strategy is not None:
        strategy.queued_order = "farthest"
    try:
        for dt, event in events:
            clock.now += dt
            side.apply(event, clock.now)
        kinds = collections.Counter(line.split()[1] for line in side.messages.lines if line.startswith("HICACHE_EVT "))
        kinds["retire calls"] = len(retired)
        return kinds
    finally:
        side.close()


def test_the_unified_arms_shape_runs_240_random_scenarios_with_every_switch_on():
    rng = random.Random(20261127)
    kinds = collections.Counter()
    for scenario in range(240):
        events = harness.random_events(rng, 600)
        kinds += run(ARM, events, explicit_ends=True, retire_ended=scenario % 2 == 0)
    print("lines by kind, the arm's shape:", dict(sorted(kinds.items())))
    assert kinds["prices"] > 1000
    assert kinds["ntp_end"] > 0


def test_random_valid_shapes_run_with_every_switch_on():
    rng = random.Random(20261128)
    kinds = collections.Counter()
    ran = refused = 0
    for scenario in range(240):
        # The priced cache only: the simpler fakes do not model the read line.
        config = dict(harness.random_config(rng), cache="priced", learning="off")
        config["timing"] = rng.choice(("horizon", "just_in_time", "expected_cost", "arrival"))
        events = harness.random_events(rng, 400)
        try:
            kinds += run(config, events, explicit_ends=rng.random() < 0.7, retire_ended=rng.random() < 0.5)
            ran += 1
        except ValueError as error:
            # A shape the switches refuse at startup (next_turn_switches.py), not a failure of a run.
            assert "SGLANG_HICACHE_NEXT_TURN" in str(error), error
            refused += 1
    print(f"random shapes: {ran} ran, {refused} refused at startup; lines by kind:", dict(sorted(kinds.items())))
    assert ran >= 60
