"""Swap both Scheduler class-level defaults for functools.cached_property in this process
only, then run the bare-Scheduler test files that reach the wired methods."""

import functools
import sys
import unittest

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    AdmissionPassOver,
)
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    PrefillLoadHold,
)

for name, make in (
    ("prefill_load_hold", lambda self: PrefillLoadHold(max_steps=0)),
    ("admission_pass_over", lambda self: AdmissionPassOver(lookahead=0)),
):
    prop = functools.cached_property(make)
    prop.__set_name__(Scheduler, name)
    setattr(Scheduler, name, prop)

FILES = [
    "test/registered/unit/managers/test_scheduler_chunked_req_gate.py",
    "test/registered/unit/managers/test_scheduler_hicache_events.py",
    "test/registered/unit/managers/test_scheduler_flush_cache_after_retract.py",
    "test/registered/unit/disaggregation/test_decode_queue_cleanup.py",
    "test/registered/unit/managers/test_scheduler_prefill_decode_interval.py",
]
loader = unittest.TestLoader()
suite = unittest.TestSuite()
for path in FILES:
    directory, module = path.rsplit("/", 1)
    sys.path.insert(0, directory)
    suite.addTests(loader.loadTestsFromName(module[: -len(".py")]))
result = unittest.TextTestRunner(verbosity=0).run(suite)
print(f"ran={result.testsRun} failures={len(result.failures)} errors={len(result.errors)}")
print("class attr is a cached_property:", type(Scheduler.__dict__["prefill_load_hold"]).__name__)
