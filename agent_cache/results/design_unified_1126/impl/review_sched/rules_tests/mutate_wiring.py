"""Mutation check for the wiring probes: each mutation rewrites one Scheduler method's
source in this process only (never on disk) and runs the probe tests; a guarded
mutation must fail at least one probe."""

import inspect
import sys
import textwrap
import unittest

sys.path.insert(0, "agent_cache/results/design_unified_1126/impl/review_sched/rules_tests")

import probe_hold_wiring_test  # noqa: E402
import probe_pass_over_wiring_test  # noqa: E402

import sglang.srt.managers.scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

MUTATIONS = {
    "refused() consulted for any refusal, not only NO_TOKEN": (
        "_get_new_batch_prefill_raw",
        "res == AddReqResult.NO_TOKEN\n                    and not added\n",
        "not added\n",
    ),
    "post-walk batch_is_full dropped": (
        "_get_new_batch_prefill_raw",
        "if pass_over.end() and not running_batch.batch_is_full:",
        "if pass_over.end() and False:",
    ),
    "admitted() only on CONTINUE": (
        "_get_new_batch_prefill_raw",
        "if added:\n                pass_over.admitted(req.rid)",
        "if added and res == AddReqResult.CONTINUE:\n                pass_over.admitted(req.rid)",
    ),
    "hold ignores a chunked prefill the batch just started": (
        "_hold_prefill_batch_for_load",
        "chunked=self.chunked_req is not None or self.is_mixed_chunk",
        "chunked=self.is_mixed_chunk",
    ),
    "hold ignores hicache_consumer_index": (
        "_hold_prefill_batch_for_load",
        "if new_batch is None or new_batch.hicache_consumer_index < 0:",
        "if new_batch is None:",
    ),
    "is_mixed_chunk dropped from the chunked predicate (expect only the documenting probe to fail)": (
        "_hold_prefill_batch_for_load",
        "chunked=self.chunked_req is not None or self.is_mixed_chunk",
        "chunked=self.chunked_req is not None",
    ),
    "admission runs while a batch is held": (
        "get_next_batch_to_run",
        "\n        elif self.prefill_load_hold.held is not None:\n            new_batch = self._take_held_prefill_batch(running_batch)",
        "",
    ),
}


def _run_probes():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        [
            loader.loadTestsFromModule(probe_hold_wiring_test),
            loader.loadTestsFromModule(probe_pass_over_wiring_test),
        ]
    )
    result = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w")).run(suite)
    failed = [test.id().rsplit(".", 1)[-1] for test, _ in result.failures + result.errors]
    return failed, result.testsRun


baseline_failed, ran = _run_probes()
print(f"baseline: {len(baseline_failed)} of {ran} probes fail")
for name, (method, old, new) in MUTATIONS.items():
    original = getattr(Scheduler, method)
    source = inspect.getsource(original)
    assert source.count(old) == 1, (name, source.count(old))
    namespace = {}
    exec(compile(textwrap.dedent(source.replace(old, new)), f"<mutant {name}>", "exec"),
         scheduler_module.__dict__, namespace)
    setattr(Scheduler, method, namespace[method])
    try:
        failed, _ = _run_probes()
    finally:
        setattr(Scheduler, method, original)
    print(f"{'CAUGHT' if failed else 'MISSED'}: {name} -> {failed}")
