"""Feasibility probe (not for the tree): the admission pass-over's wiring inside
Scheduler._get_new_batch_prefill_raw on a bare Scheduler, with a scripted PrefillAdder."""

import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    AdmissionPassOver,
)

FIT, REFUSED = "fit", "refused"


class _ScriptedAdder:
    """add_one_req per rid: (admitted?, verdict for the next candidate)."""

    script: dict = {}

    def __init__(self, *args, **kwargs):
        self.can_run_list, self.preempt_list = [], []
        self.new_chunked_req, self.rem_chunk_tokens = None, None
        self.offered = []
        _ScriptedAdder.last = self

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.offered.append(req.rid)
        admitted, verdict = self.script[req.rid]
        if admitted:
            self.can_run_list.append(req)
        return verdict


def _req(rid):
    req = MagicMock()
    req.rid = rid
    req.beam_group = None
    req.kv = SimpleNamespace(
        holds_mamba=False, mamba_cow_src_index=None, mamba_needs_clear=False
    )
    return req


def _bare_scheduler(*, lookahead, rids):
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = s.is_hybrid_swa = s.enable_lora = False
    s.enable_hicache_storage = s.is_mixed_chunk = False
    s.enable_unified_cache_external_linker = False
    s.enable_hierarchical_cache = True
    s.enable_priority_scheduling = False
    s.enable_overlap = True
    s.chunked_req = s.min_free_slots_delayer = s.dynamic_chunk_sizer = None
    s.dllm_config = s.model_config = s.token_to_kv_pool_allocator = None
    s.truncation_align_size = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.get_num_allocatable_reqs = MagicMock(return_value=100)
    s.policy = MagicMock()
    s.processed_tokens_counter = MagicMock()
    s.chunked_prefill_size, s.max_prefill_tokens, s.page_size = 8192, 16384, 1
    s.tp_worker = MagicMock()
    s.tp_worker.model_runner.prefill_aware_swa = False
    s.tree_cache = MagicMock()
    s.tree_cache.buffer_pipeline = None
    s.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs, s.max_running_requests = 0, 64
    s.req_to_token_pool = SimpleNamespace()
    s.load_inquirer = MagicMock()
    s.spec_algorithm = MagicMock()
    s.waiting_queue = [_req(rid) for rid in rids]
    s.admission_pass_over = AdmissionPassOver(lookahead=lookahead)
    running = MagicMock()
    running.batch_is_full = False
    running.reqs = []
    running.is_empty.return_value = True
    return s, running


def _walk(lookahead, script):
    s, running = _bare_scheduler(lookahead=lookahead, rids=list(script))
    _ScriptedAdder.script = {
        rid: (kind == FIT, verdict) for rid, (kind, verdict) in script.items()
    }
    with mock.patch.object(scheduler_module, "PrefillAdder", _ScriptedAdder), \
            mock.patch.object(scheduler_module, "get_schedule",
                              lambda: SimpleNamespace(prefill_max_requests=None)), \
            mock.patch.object(scheduler_module, "PrefillStats", MagicMock()), \
            mock.patch.object(scheduler_module, "set_time_batch", lambda *a, **k: None), \
            mock.patch.object(scheduler_module.ScheduleBatch, "init_new", MagicMock()):
        Scheduler._get_new_batch_prefill_raw(
            s, prefill_delayer_single_pass=None, running_batch=running
        )
    return (
        _ScriptedAdder.last.offered,
        [r.rid for r in s.waiting_queue],
        running.batch_is_full,
        dict(s.admission_pass_over.passes),
    )


C, N, O = AddReqResult.CONTINUE, AddReqResult.NO_TOKEN, AddReqResult.OTHER


class TestPassOverWiring(CustomTestCase):
    def test_later_requests_are_admitted_behind_one_refused_for_room(self):
        script = {"a": (FIT, C), "big": (REFUSED, N), "b": (FIT, C), "c": (FIT, C)}
        self.assertEqual(
            _walk(4, script),
            (["a", "big", "b", "c"], ["big"], True, {"big": 2}),
        )
        self.assertEqual(
            _walk(0, script), (["a", "big"], ["big", "b", "c"], True, {})
        )

    def test_the_walk_stops_at_a_refusal_that_is_not_for_room(self):
        script = {"big": (REFUSED, N), "x": (REFUSED, O), "y": (FIT, C)}
        offered, waiting, _, _ = _walk(4, script)
        self.assertEqual((offered, waiting), (["big", "x"], ["big", "x", "y"]))

    def test_an_admitted_request_that_exhausts_the_budget_stops_the_walk(self):
        script = {"big": (REFUSED, N), "a": (FIT, N), "b": (FIT, C)}
        self.assertEqual(
            _walk(4, script), (["big", "a"], ["big", "b"], True, {"big": 1})
        )


if __name__ == "__main__":
    unittest.main()
