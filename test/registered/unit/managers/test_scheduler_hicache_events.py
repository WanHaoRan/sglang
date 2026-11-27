"""HiCache progress must not depend on whether a prefill batch is admitted."""

import unittest
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, Mock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    enter_scope,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as scheduler_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import NextBatchPlan
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.admission_pass_over import (
    AdmissionPassOver,
)
from sglang.srt.managers.scheduler_components.prefill_load_hold import (
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSchedulerHiCacheEvents(unittest.TestCase):
    def setUp(self):
        self.calls = Mock()
        self.scheduler = s = Scheduler.__new__(Scheduler)
        s.scheduler_stage_metrics = None
        s.enable_hierarchical_cache = True
        s.enable_unified_cache_external_linker = False
        s.enable_hicache_storage = True
        s.tree_cache = SimpleNamespace(check_hicache_events=self.calls.drain)
        s._process_storage_prefetch_retries = self.calls.retry
        s.next_turn_prefetch = None
        s.process_pending_chunked_abort = Mock()
        s.process_prefill_chunk = Mock()
        s.dp_attn_adapter = Mock()
        s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.return_value = None
        self.running_batch = SimpleNamespace(
            batch_is_full=False, is_prefill_only=False, is_empty=lambda: True
        )

    def test_feature_gates(self):
        for hierarchical, flexkv, linker, storage in (
            (False, False, False, False),
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (True, False, False, True),
        ):
            with (
                self.subTest(
                    hierarchical=hierarchical,
                    flexkv=flexkv,
                    linker=linker,
                    storage=storage,
                ),
                patch(
                    "sglang.srt.managers.scheduler.get_memory",
                    return_value=SimpleNamespace(enable_flexkv=flexkv),
                ),
            ):
                self.calls.reset_mock()
                s = self.scheduler
                s.enable_hierarchical_cache = hierarchical
                s.enable_unified_cache_external_linker = linker
                s.enable_hicache_storage = storage
                s._process_hicache_events()
                expected = [call.drain()] if hierarchical or flexkv or linker else []
                if storage:
                    expected.append(call.retry())
                self.assertEqual(self.calls.mock_calls, expected)

    def test_next_turn_refetch_reissues_then_reports_after_the_retries(self):
        """The next-turn prefetcher ticks after the paced retries (so it sees
        their reads); each request it returns gets its own prefetch re-issued,
        then whether a read started is reported back."""
        s = self.scheduler
        req = SimpleNamespace(cache_request_handle="handle")
        s.waiting_queue = [req]
        s.next_turn_prefetch = self.calls.ntp
        self.calls.ntp.tick.return_value = [req]
        s._prefetch_kvcache = self.calls.prefetch
        s.tree_cache.has_ongoing_prefetch = lambda handle: handle == "handle"
        s._process_hicache_events()
        self.assertEqual(
            self.calls.mock_calls,
            [
                call.drain(),
                call.retry(),
                call.ntp.tick(waiting_queue=[req], now=ANY),
                call.prefetch(req),
                call.ntp.on_refetch_result(req=req, issued=True),
            ],
        )

    def test_pd_prefill_drains_before_admission_even_with_empty_queue(self):
        s = self.scheduler
        s.resolve_waiting_queue_bootstrap = Mock()
        s.get_new_batch_prefill = self.calls.admit
        s.get_new_batch_prefill.return_value = NextBatchPlan(
            batch_to_run=None, running_batch=self.running_batch
        )
        for waiting_queue in ([], [SimpleNamespace(rid="pending_l3")]):
            with self.subTest(waiting=bool(waiting_queue)):
                s.waiting_queue = waiting_queue
                self.calls.reset_mock()
                for _ in range(2):
                    plan = s.get_next_disagg_prefill_batch_to_run(
                        running_batch=self.running_batch, last_batch=None
                    )
                    self.assertIsNone(plan.batch_to_run)
                self.assertEqual(
                    self.calls.mock_calls,
                    [call.drain(), call.retry(), call.admit(self.running_batch)] * 2,
                )

    def test_unified_drains_when_prefill_is_deferred(self):
        s = self.scheduler
        s.enable_fpm = False
        s._abort_on_waiting_timeout = Mock()
        s._abort_on_running_timeout = Mock()
        s.dllm_config = None
        s.chunked_req = None
        s.enable_hisparse = False
        s.require_mlp_sync = False
        s._should_defer_prefill = self.calls.defer
        s._should_defer_prefill.return_value = True
        s.get_new_batch_prefill = Mock()
        s.dp_attn_adapter.maybe_convert_decode_to_extend.return_value = None
        s._arm_prefill_decode_interval = Mock()
        s.ngram_embedding_manager = Mock()
        s.ngram_embedding_manager.prepare_for_forward.return_value = None

        plan = s.get_next_batch_to_run(self.running_batch, None)

        self.assertIsNone(plan.batch_to_run)
        s.get_new_batch_prefill.assert_not_called()
        self.assertEqual(
            self.calls.mock_calls, [call.drain(), call.retry(), call.defer()]
        )

    def test_pp_prefill_drains_before_admission(self):
        s = self.scheduler
        s.init_pp_loop_state = Mock()
        s.pp_loop_size = 1
        enter_scope(self, published_topology(pp_size=2))
        s.pp_group = SimpleNamespace(is_last_rank=True)
        s.running_mbs = [self.running_batch]
        s.last_mbs = [None]
        s.ingest_requests = Mock(return_value=[])
        s._pp_pd_get_bootstrapped_ids = Mock(return_value=[])
        s._pp_pd_get_prefill_transferred_ids = Mock(return_value=[])
        s._pp_commit_comm_work = Mock()
        s.get_new_batch_prefill = self.calls.admit
        # Stop the infinite loop at admission; no PP transport or GPU is needed.
        s.get_new_batch_prefill.side_effect = StopIteration

        with self.assertRaises(StopIteration):
            s.event_loop_pp_disagg_prefill()

        self.assertEqual(
            self.calls.mock_calls,
            [call.drain(), call.retry(), call.admit(self.running_batch)],
        )


def _hold_scheduler(*, prefill_batch, load_done):
    """A bare Scheduler whose admission returns prefill_batch beside a decoding batch."""
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.dllm_config = s.dllm_manager = None
    s.enable_hisparse = s.enable_fpm = s.enable_hicache_storage = False
    s.enable_hierarchical_cache = True
    s.enable_unified_cache_external_linker = False
    s.is_mixed_chunk = False
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.chunked_req = s._pending_chunked_abort_req = None
    s.prefill_decode_interval = s._prefill_decode_interval_remaining = 0
    running = MagicMock()
    running.is_empty.return_value = False
    running.is_prefill_only = False
    running.batch_is_full = False
    running.reqs = []
    running.forward_mode.is_extend.return_value = False
    s.running_batch = running
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=prefill_batch, running_batch=running)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda b, **_: b
    s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda b: b
    s.ngram_embedding_manager = MagicMock()
    s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda b, **_: b
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    s.tree_cache = MagicMock()
    s.tree_cache.is_load_back_event_done.side_effect = lambda i: load_done[0]
    s.prefill_load_hold = PrefillLoadHold(max_steps=4)
    return s


def _loading_batch(*, load_tokens=MIN_LOAD_TOKENS, consumer_index=1, req_loads=0):
    batch = MagicMock()
    batch.hicache_consumer_index = consumer_index
    batch.hicache_load_tokens = load_tokens
    batch.reqs = [SimpleNamespace(host_loaded_length=req_loads)]
    batch.forward_mode.is_extend.return_value = True
    return batch


def _next_batch(s):
    return Scheduler.get_next_batch_to_run(
        s, running_batch=s.running_batch, last_batch=None
    ).batch_to_run


class TestLoadHoldWiring(unittest.TestCase):
    def test_a_held_batch_decodes_beside_its_load_and_nothing_is_admitted_meanwhile(
        self,
    ):
        batch, load_done = _loading_batch(), [False]
        s = _hold_scheduler(prefill_batch=batch, load_done=load_done)
        self.assertIs(_next_batch(s), s.running_batch)
        self.assertIs(_next_batch(s), s.running_batch)
        self.assertEqual(s.get_new_batch_prefill.call_count, 1)
        load_done[0] = True
        self.assertIs(_next_batch(s), batch)
        s.tree_cache.is_load_back_event_done.assert_called_with(1)
        self.assertIsNone(s.prefill_load_hold.held)

    def test_a_batch_that_starts_a_chunked_prefill_is_not_held(self):
        """The next pass would cache the chunk as computed before it ran."""
        batch = _loading_batch()
        s = _hold_scheduler(prefill_batch=batch, load_done=[False])
        plan = s.get_new_batch_prefill.return_value

        def admit_and_chunk(running_batch):
            s.chunked_req = MagicMock()
            return plan

        s.get_new_batch_prefill = MagicMock(side_effect=admit_and_chunk)
        self.assertIs(_next_batch(s), batch)
        self.assertIsNone(s.prefill_load_hold.held)

    def test_mixed_chunk_holds_only_a_batch_that_was_left_unmixed(self):
        """A mixed batch carries the running requests and leaves nothing to decode."""
        for mixed in (True, False):
            with self.subTest(mixed=mixed):
                batch = _loading_batch()
                s = _hold_scheduler(prefill_batch=batch, load_done=[False])
                s.is_mixed_chunk = True
                if mixed:
                    empty = MagicMock()
                    empty.is_empty.return_value = True
                    empty.is_prefill_only = False
                    empty.reqs = []
                    s.get_new_batch_prefill = MagicMock(
                        return_value=NextBatchPlan(
                            batch_to_run=batch, running_batch=empty
                        )
                    )
                self.assertIs(_next_batch(s) is batch, mixed)

    def test_only_the_batch_own_copy_decides_a_hold(self):
        """A request admitted earlier keeps host_loaded_length from an older load (a
        chunked prefill's last chunk); a batch whose own copy is short is not held."""
        for batch in (
            _loading_batch(consumer_index=-1),
            _loading_batch(load_tokens=2048, req_loads=30000),
        ):
            s = _hold_scheduler(prefill_batch=batch, load_done=[False])
            self.assertIs(_next_batch(s), batch)
            self.assertIsNone(s.prefill_load_hold.held)


FIT, REFUSED = "fit", "refused"


class _ScriptedAdder:
    """add_one_req per rid: (admitted?, the verdict for the next candidate)."""

    script: dict = {}

    def __init__(self, *args, **kwargs):
        self.can_run_list, self.preempt_list = [], []
        self.new_chunked_req, self.rem_chunk_tokens = None, None
        self.host_load_tokens = 0
        self.offered = []
        _ScriptedAdder.last = self

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        self.offered.append(req.rid)
        admitted, verdict = self.script[req.rid]
        if admitted:
            self.can_run_list.append(req)
        return verdict


def _waiting_req(rid):
    req = MagicMock()
    req.rid = rid
    req.beam_group = None
    req.kv = SimpleNamespace(
        holds_mamba=False, mamba_cow_src_index=None, mamba_needs_clear=False
    )
    return req


def _walk(lookahead, script, *, hierarchical=True, ongoing=()):
    """Runs one admission walk over the script's rids; returns (offered, left waiting,
    batch_is_full, pass counts, rids whose read progress was checked)."""
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = s.is_hybrid_swa = s.enable_lora = False
    s.enable_priority_scheduling = s.is_mixed_chunk = False
    s.enable_hicache_storage = bool(ongoing)
    s.enable_unified_cache_external_linker = False
    s.enable_hierarchical_cache = hierarchical
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
    s.tree_cache.has_ongoing_prefetch.side_effect = lambda h: h in ongoing
    s.tree_cache.check_prefetch_progress.side_effect = lambda h: h not in ongoing
    s.tree_cache.pop_prefetch_loaded_span.return_value = (0, 0)
    s._prefetch_after_device_hit_loss = MagicMock(return_value=False)
    s.new_token_ratio_tracker = SimpleNamespace(current=0.5)
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs, s.max_running_requests = 0, 64
    s.req_to_token_pool = SimpleNamespace()
    s.load_inquirer = MagicMock()
    s.spec_algorithm = MagicMock()
    s.waiting_queue = [_waiting_req(rid) for rid in script]
    for req in s.waiting_queue:
        req.cache_request_handle = req.rid
    s.admission_pass_over = AdmissionPassOver(lookahead=lookahead)
    running = MagicMock()
    running.batch_is_full = False
    running.reqs = []
    running.is_empty.return_value = True
    _ScriptedAdder.script = {
        rid: (kind == FIT, verdict) for rid, (kind, verdict) in script.items()
    }
    with (
        patch.object(scheduler_module, "PrefillAdder", _ScriptedAdder),
        patch.object(
            scheduler_module,
            "get_schedule",
            lambda: SimpleNamespace(prefill_max_requests=None),
        ),
        patch.object(scheduler_module, "PrefillStats", MagicMock()),
        patch.object(scheduler_module, "set_time_batch", lambda *a, **k: None),
        patch.object(scheduler_module.ScheduleBatch, "init_new", MagicMock()),
    ):
        Scheduler._get_new_batch_prefill_raw(
            s, prefill_delayer_single_pass=None, running_batch=running
        )
    checked = [c.args[0] for c in s.tree_cache.check_prefetch_progress.call_args_list]
    return (
        _ScriptedAdder.last.offered,
        [r.rid for r in s.waiting_queue],
        running.batch_is_full,
        dict(s.admission_pass_over.passes),
        checked,
    )


C, N, O = AddReqResult.CONTINUE, AddReqResult.NO_TOKEN, AddReqResult.OTHER


class TestAdmissionPassOverWiring(unittest.TestCase):
    def test_later_requests_are_admitted_behind_one_refused_for_room(self):
        script = {"a": (FIT, C), "big": (REFUSED, N), "b": (FIT, C), "c": (FIT, C)}
        self.assertEqual(
            _walk(4, script)[:4], (["a", "big", "b", "c"], ["big"], True, {"big": 2})
        )
        # Lookahead 0 is the walk without the pass-over: it stops at the refusal.
        self.assertEqual(
            _walk(0, script)[:4], (["a", "big"], ["big", "b", "c"], True, {})
        )

    def test_the_walk_stops_at_a_refusal_that_is_not_for_room(self):
        script = {"big": (REFUSED, N), "x": (REFUSED, O), "y": (FIT, C)}
        offered, waiting, _, _, _ = _walk(4, script)
        self.assertEqual((offered, waiting), (["big", "x"], ["big", "x", "y"]))

    def test_an_admitted_request_that_exhausts_the_budget_stops_the_walk(self):
        script = {"big": (REFUSED, N), "a": (FIT, N), "b": (FIT, C)}
        self.assertEqual(
            _walk(4, script)[:4], (["big", "a"], ["big", "b"], True, {"big": 1})
        )

    def test_without_the_hierarchical_cache_a_refusal_marks_the_batch_full(self):
        """As the walk without the pass-over does, whatever was admitted behind it."""
        script = {"big": (REFUSED, N), "small": (REFUSED, N)}
        for hierarchical, full in ((False, True), (True, False)):
            with self.subTest(hierarchical=hierarchical):
                self.assertIs(_walk(4, script, hierarchical=hierarchical)[2], full)

    def test_a_read_in_flight_behind_a_refusal_is_not_checked(self):
        """Checking a read's progress can end it early under the timeout and best_effort
        prefetch policies; behind a refused request the walk leaves it to land."""
        script = {"big": (REFUSED, N), "reading": (FIT, C), "c": (FIT, C)}
        offered, waiting, _, _, checked = _walk(4, script, ongoing={"reading"})
        self.assertEqual(checked, ["big", "c"])
        self.assertEqual((offered, waiting), (["big", "c"], ["big", "reading"]))


if __name__ == "__main__":
    unittest.main()
