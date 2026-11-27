"""Skeptic check: does a held prefill batch make the paused-engine flush fail where it
would otherwise succeed? Runs the real Scheduler.pause_generation / is_fully_idle /
flush_cache / _hold_prefill_batch_for_load / _take_held_prefill_batch and the real
UnifiedRadixCache.loading_check / is_load_back_event_done / _count_ready_acks against
fake CUDA events that complete in H2D-stream order (last-layer event, then the ack's
finish event, l2_transfer.py:117-118). CPU only."""

import itertools
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    UnifiedRadixCache,
    _OngoingLoadBack,
)

P_NODE = 7
SLOT = 0


class Clock:
    def __init__(self):
        self.t = 0.0


class FakeEvent:
    def __init__(self, clock, done_at):
        self.clock, self.done_at = clock, done_at

    def query(self):
        return self.clock.t >= self.done_at

    def synchronize(self):
        self.clock.t = max(self.clock.t, self.done_at)


class FakeReq:
    def __init__(self, rid, host_loaded_length=0):
        self.rid = rid
        self.host_loaded_length = host_loaded_length

    def finished(self):
        return False


class FakeBatch:
    def __init__(self, reqs, *, extend, consumer_index=-1):
        self.reqs = list(reqs)
        self.forward_mode = SimpleNamespace(is_extend=lambda: extend)
        self.hicache_consumer_index = consumer_index
        self.is_prefill_only = False
        self.batch_is_full = False

    def is_empty(self):
        return len(self.reqs) == 0


def make_tree_cache(clock):
    tc = UnifiedRadixCache.__new__(UnifiedRadixCache)
    tc.cache_controller = SimpleNamespace(
        ack_load_queue=[],
        ack_write_queue=[],
        layer_done_counter=SimpleNamespace(
            events=[SimpleNamespace(finish_event=FakeEvent(clock, 0.0)) for _ in range(3)]
        ),
    )
    tc.pp_rank = 0
    tc.tree_core = SimpleNamespace(
        write_back_duplicate_reclaim_digest=0, finish_load_back=lambda node: None
    )
    tc._all_reduce = lambda tensor, op: None
    tc.buffer_pipeline = None
    tc.ongoing_load_back = {}
    tc.ongoing_write_through = {}
    tc.ongoing_prefetch = {}
    tc.ongoing_backup = {}
    # The fork runs an SSD tier (enable_storage True); False here only skips the
    # get_memory() lookup in is_fully_idle. More terms could only make idle False more often.
    tc.enable_storage = False
    tc.dec_lock_ref = lambda node, params: None
    tc.dec_host_lock_ref = lambda node, params: None
    tc.metrics_collector = None
    tc.reset = MagicMock()
    return tc


def admit_p(tc, clock, *, load_passes, ack_gap):
    """What admission does for P: init_load_back registers the entry
    (unified_radix_cache.py:2101), ready_to_load_host_cache starts the copy and queues
    its ack (cache_controller.py:982-1017)."""
    layer_done = clock.t + load_passes
    tc.ongoing_load_back[P_NODE] = _OngoingLoadBack(P_NODE, None, None)
    tc.cache_controller.layer_done_counter.events[SLOT] = SimpleNamespace(
        finish_event=FakeEvent(clock, layer_done)
    )
    tc.cache_controller.ack_load_queue.append(
        SimpleNamespace(
            finish_event=FakeEvent(clock, layer_done + ack_gap),
            node_ids=[P_NODE],
            num_tokens=MIN_LOAD_TOKENS,
            num_bytes=0,
            timing_enabled=False,
            num_tokens_by_pool={},
        )
    )
    return FakeBatch(
        [FakeReq("P", host_loaded_length=MIN_LOAD_TOKENS)],
        extend=True,
        consumer_index=SLOT,
    )


def make_scheduler(tc, *, max_steps):
    s = Scheduler.__new__(Scheduler)
    s._engine_paused = False
    s.enable_overlap = True
    s.result_queue = deque()
    s.last_batch = None
    s.cur_batch_for_debug = None
    s.chunked_req = None
    s.is_mixed_chunk = False
    s.disaggregation_mode = DisaggregationMode.NULL
    s.running_batch = FakeBatch([FakeReq("R1"), FakeReq("R2")], extend=False)
    s.waiting_queue = []
    s.dllm_manager = MagicMock()
    s.dllm_manager.any_staging_reqs.return_value = False
    s._pp_microbatches_drained = MagicMock(return_value=True)
    s.grammar_manager = MagicMock()
    s.grammar_manager.grammar_queue = []
    s.enable_hierarchical_cache = True
    s.enable_hisparse = False
    s.tree_cache = tc
    s.req_to_token_pool = MagicMock()
    s.token_to_kv_pool_allocator = MagicMock()
    s.draft_worker = None
    s.metrics_reporter = MagicMock()
    s.kv_events_publisher = MagicMock()
    s.hisparse_coordinator = None
    s.process_batch_result = MagicMock()
    s._add_request_to_queue = lambda req: s.waiting_queue.append(req)
    s.prefill_load_hold = PrefillLoadHold(max_steps=max_steps)
    return s


def run_pass(s, tc, clock, *, admit=None):
    """One overlap-loop pass after ingest: check_hicache_events' load half, the
    get_next_batch_to_run branch, launch, pop_and_process of the previous batch."""
    clock.t += 0.1
    tc.loading_check()  # what check_hicache_events does for loads (finish_count=None path)
    clock.t += 0.05
    if s.prefill_load_hold.held is not None:
        new_batch = s._take_held_prefill_batch(s.running_batch)
    elif admit is not None:
        new_batch = s._hold_prefill_batch_for_load(admit(), s.running_batch)
    else:
        new_batch = None
    batch = new_batch if new_batch is not None else s.running_batch
    s.result_queue.append((batch, None))
    if s.last_batch is not None:
        s.result_queue.popleft()
    s.last_batch = batch
    clock.t = float(int(clock.t) + 1)  # next ingest boundary
    return batch


def pause_retract(s):
    with patch("sglang.srt.managers.scheduler.retract_all") as retract_all:
        s.pause_generation(SimpleNamespace(mode="retract"))
    return retract_all


def weight_update_flush(s):
    updater = SimpleNamespace(flush_cache=s.flush_cache)
    SchedulerWeightUpdaterManager.flush_cache_after_weight_update(
        updater, SimpleNamespace(flush_cache=True, torch_empty_cache=False)
    )


class TestHoldPauseRetract(unittest.TestCase):
    def _held_then_paused(self, max_steps=8):
        clock = Clock()
        tc = make_tree_cache(clock)
        s = make_scheduler(tc, max_steps=max_steps)
        run_pass(
            s, tc, clock, admit=lambda: admit_p(tc, clock, load_passes=3.0, ack_gap=0.0)
        )
        self.assertIsNotNone(s.prefill_load_hold.held)  # P held, R decoded
        pause_retract(s)
        return s, tc, clock

    def test_finding_mechanism_held_batch_survives_retract_and_flush_asserts(self):
        s, tc, _ = self._held_then_paused()
        self.assertIsNotNone(s.prefill_load_hold.held)
        self.assertNotIn("P", [r.rid for r in s.waiting_queue])
        with self.assertRaises(AssertionError):
            weight_update_flush(s)

    def test_clearing_the_hold_alone_does_not_make_the_flush_succeed(self):
        s, tc, _ = self._held_then_paused()
        held = s.prefill_load_hold.held
        s.waiting_queue += [r for r in held.reqs if r not in s.waiting_queue]
        s.prefill_load_hold.held = None  # first half of the proposed fix
        self.assertIn(P_NODE, tc.ongoing_load_back)
        self.assertFalse(s.is_fully_idle(ignore_waiting=True))
        with self.assertRaises(AssertionError):
            weight_update_flush(s)

    def test_with_the_hold_off_the_same_pause_also_asserts(self):
        clock = Clock()
        tc = make_tree_cache(clock)
        s = make_scheduler(tc, max_steps=0)
        run_pass(
            s, tc, clock, admit=lambda: admit_p(tc, clock, load_passes=3.0, ack_gap=0.0)
        )
        self.assertIsNone(s.prefill_load_hold.held)
        self.assertTrue(s.last_batch.forward_mode.is_extend())  # P ran
        retract_all = pause_retract(s)
        retracted = [r.rid for r in retract_all.call_args.kwargs["reqs"]]
        self.assertIn("P", retracted)  # P retracted as the extend last_batch
        self.assertIn(P_NODE, tc.ongoing_load_back)
        with self.assertRaises(AssertionError):
            weight_update_flush(s)

    def test_with_the_hold_off_the_flush_fails_until_a_check_after_the_load(self):
        # Pause one pass later than P's admission, then two: both fail while the
        # load has not been acked by a non-paused pass.
        for extra_passes in (0, 1, 2):
            with self.subTest(extra_passes=extra_passes):
                clock = Clock()
                tc = make_tree_cache(clock)
                s = make_scheduler(tc, max_steps=0)
                run_pass(
                    s,
                    tc,
                    clock,
                    admit=lambda: admit_p(tc, clock, load_passes=3.0, ack_gap=0.0),
                )
                for _ in range(extra_passes):
                    run_pass(s, tc, clock)
                pause_retract(s)
                self.assertFalse(s.flush_cache(empty_cache=False))

    def test_draining_the_ack_is_the_load_bearing_part_with_or_without_a_hold(self):
        for max_steps in (8, 0):
            with self.subTest(max_steps=max_steps):
                clock = Clock()
                tc = make_tree_cache(clock)
                s = make_scheduler(tc, max_steps=max_steps)
                run_pass(
                    s,
                    tc,
                    clock,
                    admit=lambda: admit_p(tc, clock, load_passes=3.0, ack_gap=0.0),
                )
                pause_retract(s)
                if (held := s.prefill_load_hold.held) is not None:
                    s.waiting_queue += held.reqs
                    s.prefill_load_hold.held = None
                tc.cache_controller.layer_done_counter.events[
                    SLOT
                ].finish_event.synchronize()
                tc.cache_controller.ack_load_queue[0].finish_event.synchronize()
                tc.loading_check()
                self.assertTrue(s.flush_cache(empty_cache=False))

    def test_invariant_held_implies_unacked_load_at_every_ingest_boundary(self):
        """Over load lengths, layer-to-ack gaps and step bounds: whenever the hold is
        set at an ingest boundary (where pause_generation runs), P's load-back entry is
        still in ongoing_load_back, so the held term never decides is_fully_idle."""
        checked = 0
        for load_passes, ack_gap, max_steps in itertools.product(
            (0.01, 0.12, 0.16, 0.5, 0.99, 1.0, 1.12, 1.16, 2.5, 7.3, 40.0),
            (0.0, 0.001, 0.04, 0.3, 1.2),
            (1, 2, 4, 8, 64),
        ):
            clock = Clock()
            tc = make_tree_cache(clock)
            s = make_scheduler(tc, max_steps=max_steps)
            run_pass(
                s,
                tc,
                clock,
                admit=lambda: admit_p(
                    tc, clock, load_passes=load_passes, ack_gap=ack_gap
                ),
            )
            for _ in range(80):
                if s.prefill_load_hold.held is not None:
                    checked += 1
                    self.assertIn(
                        P_NODE,
                        tc.ongoing_load_back,
                        (load_passes, ack_gap, max_steps, clock.t),
                    )
                    self.assertFalse(s.is_fully_idle(ignore_waiting=True))
                    # Same state with the held term removed: still not idle.
                    held, s.prefill_load_hold.held = s.prefill_load_hold.held, None
                    self.assertFalse(s.is_fully_idle(ignore_waiting=True))
                    s.prefill_load_hold.held = held
                run_pass(s, tc, clock)
        self.assertGreater(checked, 100)
        print(f"\nheld-at-ingest states checked: {checked}")


if __name__ == "__main__":
    unittest.main()
