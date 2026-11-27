"""Probe: are the Scheduler class-level defaults shared, and what does a bare-scheduler
test that tunes one of them leak? CPU only; imports scheduler.py like the unit tests."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    AdmissionPassOver,
)
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    PrefillLoadHold,
)

a = Scheduler.__new__(Scheduler)
b = Scheduler.__new__(Scheduler)
print("hold shared across bare schedulers:", a.prefill_load_hold is b.prefill_load_hold)
print("pass_over shared across bare schedulers:", a.admission_pass_over is b.admission_pass_over)
print(
    "msgspec copies the {} default per instance:",
    AdmissionPassOver(lookahead=0).passes is not AdmissionPassOver(lookahead=0).passes,
)

# What a future wiring test might write on a bare scheduler.
a.prefill_load_hold.max_steps = 2
a.prefill_load_hold.hold("batch-of-test-a")
print("b sees a's held batch:", b.prefill_load_hold.held)
print("class attr now:", Scheduler.prefill_load_hold)

# Restore, then show that the lookahead=0 default stays unchanged by the walk protocol.
Scheduler.prefill_load_hold.max_steps = 0
Scheduler.prefill_load_hold.held = None
p = Scheduler.admission_pass_over
before = (p.lookahead, dict(p.passes), p.blocked_rid, p.tried, p.admitted_behind)
p.begin()
p.admitted("r1")
p.refused("r2")
p.end()
after = (p.lookahead, dict(p.passes), p.blocked_rid, p.tried, p.admitted_behind)
print("lookahead=0 default unchanged by begin/admitted/refused/end:", before == after)
