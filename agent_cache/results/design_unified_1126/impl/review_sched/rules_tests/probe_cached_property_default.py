"""Probe: a functools.cached_property default gives each bare Scheduler its own disabled
collaborator, and an init_* assignment still replaces it. Subclass only; CPU only."""

import functools

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    PrefillLoadHold,
)


class LazyDefaultScheduler(Scheduler):
    @functools.cached_property
    def prefill_load_hold(self) -> PrefillLoadHold:
        return PrefillLoadHold(max_steps=0)


a = LazyDefaultScheduler.__new__(LazyDefaultScheduler)
b = LazyDefaultScheduler.__new__(LazyDefaultScheduler)
print("distinct per bare instance:", a.prefill_load_hold is not b.prefill_load_hold)
a.prefill_load_hold.max_steps = 2
a.prefill_load_hold.hold("batch-of-a")
print("b unaffected:", b.prefill_load_hold.held is None and b.prefill_load_hold.max_steps == 0)
c = LazyDefaultScheduler.__new__(LazyDefaultScheduler)
c.prefill_load_hold = PrefillLoadHold(max_steps=16)  # what init_prefill_load_hold does
print("assignment wins:", c.prefill_load_hold.max_steps == 16)
