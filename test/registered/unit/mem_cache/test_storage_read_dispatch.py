"""Cache-mode storage hits under the deadline read order: park without host
memory, dispatch by deadline into the read window, allocate at dispatch, pin the
landed span until admission (agent_cache/ORACLE_V2_PLAN.md 4.2-4.3). Timed reads
wait while pending SSD writes lock host memory, which the write thread drains in
merged transfers (agent_cache/OPEN_ITEMS.md section 7). No GPU."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import logging
import unittest
from array import array
from collections import defaultdict
from contextlib import ExitStack, nullcontext
from queue import Queue
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import (
    STORAGE_READ_RATE_MIN_SAMPLES,
    HiCacheController,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchSubmission,
    StorageOperation,
)
from sglang.srt.mem_cache.next_turn_estimators import LAG_MIN_SAMPLES, ReadPathLags
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.storage_read_scheduler import StorageReadScheduler
from sglang.srt.mem_cache.storage_retention import (
    RetentionSwitches,
    resolve_retention_switches,
)
from sglang.srt.mem_cache.unified_cache.storage_attachment import StorageAttachment
from sglang.srt.mem_cache.unified_cache.unified_tree_core import StorageBackupSpec
from sglang.srt.mem_cache.unified_radix_cache import (
    PREFETCH_PIN_MAX_AGE_S,
    UnifiedRadixCache,
    _OngoingPrefetch,
    _OngoingWriteThrough,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

PAGE = 2
HOST_POOL_TOKENS = 1000
LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


class _HostPool:
    """Records every allocation; short of pages only when `available` says so."""

    def __init__(self, available=10**6):
        self.allocs = []
        self.size = HOST_POOL_TOKENS
        self.available = available

    def alloc(self, n):
        if n > self.available:
            return None
        self.allocs.append(n)
        return torch.arange(n)

    def available_size(self):
        return self.available


class _LockedTreeCore:
    """Only the host lock accounting the pin uses."""

    def __init__(self):
        self.locks = defaultdict(int)
        self.eviction_strategy = None
        self.page_size = PAGE

    def inc_host_lock_ref(self, node_id):
        self.locks[node_id] += 1
        return SimpleNamespace(to_dec_params=lambda: ("dec", node_id))

    def dec_host_lock_ref(self, node_id, params):
        assert params == ("dec", node_id)
        self.locks[node_id] -= 1


def cache_mode_fixture(
    *,
    order="deadline",
    window_pages=10**6,
    budget_tokens=10**9,
    pin_fraction=0.0,
    write_hold_tokens=0,
):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.disable = False
    cache.host_memory_mode = "cache"
    # page_size is a property over tree_core; the core goes first.
    cache.tree_core = _LockedTreeCore()
    cache.prefetch_threshold = PAGE
    cache.prefetch_stop_policy = "wait_complete"
    cache.enable_storage_metrics = False
    cache.storage_metrics_collector = None
    cache.buffer_pipeline = None
    cache.storage_prefetch_retries = StoragePrefetchRetries()
    cache._prefetch_outcome_stats = defaultdict(int)
    cache._storage_prefetch_hit_remaining_by_reqid = {}
    cache._record_storage_prefetch_hit = Mock()
    cache._invalidate_absent_from_hit_query = Mock()
    cache._account_prefetch_outcome = Mock()
    cache._resolve_storage_prefetch_tokens = Mock()
    cache._finish_storage_prefetch = Mock()
    cache._alloc_prefetch_aux_staging = Mock(return_value=True)
    cache._log_storage_prefetch_deferred = Mock()
    cache.evict_host = Mock()
    cache.ongoing_prefetch = {}
    cache.cache_controller = SimpleNamespace(
        prefetch_hit_queue=Queue(),
        prefetch_buffer=Queue(),
        ack_prefetch_queue=Queue(),
        ack_backup_queue=Queue(),
        host_mem_release_queue=Queue(),
        extra_host_mem_release_queues={},
        prefetch_tokens_occupied=0,
        mem_pool_host=_HostPool(),
        append_host_mem_release=Mock(),
        prefetch_rate_limited=lambda: False,
    )
    cache.deadlines = {}
    cache.read_deadline_fn = cache.deadlines.get
    cache.prefetch_gate_fn = None
    cache.read_not_before_fn = None
    cache.soft_pin_victims_fn = None
    cache.prefetch_landed_fn = None
    cache.room_refusals = None
    cache.linker = None
    cache.prefetch_loaded_tokens_by_reqid = {}
    cache.prefetch_loaded_storage_start_by_reqid = {}
    cache.read_scheduler = (
        None
        if order == "fifo"
        else StorageReadScheduler(
            window_pages=window_pages,
            budget_tokens=budget_tokens,
            page_size=PAGE,
            write_hold_tokens=write_hold_tokens,
        )
    )
    cache.pinned_prefetch = {}
    cache.pinned_prefetch_tokens = 0
    cache.prefetch_pin_budget_tokens = int(pin_fraction * HOST_POOL_TOKENS)
    cache.ongoing_backup = {}
    cache.pending_backup_tokens = 0
    cache.prefetch_refused_fn = None
    cache.host_census = None
    return cache


def hit(cache, rid, tokens):
    """A storage hit of `tokens` for rid, as the controller publishes it."""
    handle = CacheRequestHandle(rid, 0)
    operation = SimpleNamespace(
        request_id=rid,
        handle=handle,
        storage_hit_count=tokens,
        hash_value=[f"{rid}-{p}" for p in range(tokens // PAGE)],
        storage_start=0,
        stats_requested_tokens=tokens,
        host_indices=None,
        start_time=-1.0,
        is_terminated=lambda: False,
    )
    cache.ongoing_prefetch[handle] = _OngoingPrefetch(
        0, RadixKey(array("q", range(tokens))), None, operation, None, {}
    )
    cache.cache_controller.prefetch_hit_queue.put(operation)
    return handle, operation


def drain(cache, n, *, write_acks=0):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=n,
        n_ack_prefetch=0,
        n_backup=write_acks,
        n_release=0,
        extra_release_counts={},
        log_metrics=False,
    )


def launched(cache):
    """rids handed to the read thread, in order."""
    queue = cache.cache_controller.prefetch_buffer
    rids = []
    while not queue.empty():
        rids.append(queue.get().request_id)
    return rids


def finish_read(cache, handle):
    """The read landed and was taken: it no longer counts as in flight."""
    del cache.ongoing_prefetch[handle]


class TestParkAndDispatch(CustomTestCase):
    def test_fifo_order_launches_at_the_drain_as_before(self):
        """With the flag off the drain is the pre-v2 path: the bounce is allocated
        at the hit and the read thread gets the op at once, nothing is parked."""
        cache = cache_mode_fixture(order="fifo")
        hit(cache, "a", 4 * PAGE)
        drain(cache, 1)
        self.assertEqual(cache.cache_controller.mem_pool_host.allocs, [4 * PAGE])
        self.assertEqual(launched(cache), ["a"])
        # Cache mode charges occupancy at enqueue, not at launch (unchanged).
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)

    def test_parked_hit_holds_no_host_memory_until_dispatch(self):
        cache = cache_mode_fixture()
        handle, op = hit(cache, "a", 4 * PAGE)
        drain(cache, 1)
        self.assertEqual(cache.read_scheduler.parked_handles(), [handle])
        self.assertEqual(cache.cache_controller.mem_pool_host.allocs, [])
        self.assertEqual(launched(cache), [])
        self.assertEqual(cache.storage_read_backlog_pages(), 0)
        # Parked: the timeout clock has not started.
        self.assertEqual(op.start_time, -1.0)

        cache._dispatch_storage_reads()
        self.assertEqual(cache.cache_controller.mem_pool_host.allocs, [4 * PAGE])
        self.assertEqual(launched(cache), ["a"])
        self.assertEqual(cache.read_scheduler.parked_count(), 0)
        # Occupancy and the in-flight count start at dispatch, sized to the
        # allocation, and the tracked key is trimmed to the same span so the
        # release paths subtract exactly what was charged.
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 4 * PAGE)
        self.assertEqual(len(cache.ongoing_prefetch[handle].prefetch_key), 4 * PAGE)
        self.assertEqual(cache.storage_read_backlog_pages(), 4)
        cache.revoke_pending_prefetch(handle)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)
        # The timeout policy's clock is reset to the dispatch, not the lookup.
        self.assertGreater(op.start_time, 0.0)

    def test_dispatch_is_by_deadline_within_the_window(self):
        """Three parked reads of two pages each, a window of four pages: the
        "now" read goes first, then the earliest deadline; the third waits for
        the in-flight pages to land."""
        cache = cache_mode_fixture(window_pages=4)
        late, _ = hit(cache, "late", 2 * PAGE)
        early, _ = hit(cache, "early", 2 * PAGE)
        now, _ = hit(cache, "now", 2 * PAGE)
        cache.deadlines.update({late: 300.0, early: 120.0})
        drain(cache, 3)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["now", "early"])
        self.assertEqual(cache.read_scheduler.parked_handles(), [late])
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), [])
        finish_read(cache, now)
        finish_read(cache, early)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["late"])

    def test_budget_counts_pinned_tokens_but_not_for_a_now_read(self):
        cache = cache_mode_fixture(budget_tokens=10 * PAGE, pin_fraction=0.5)
        cache._pin_prefetch(CacheRequestHandle("landed", 0), 7, 8 * PAGE)
        timed, _ = hit(cache, "timed", 4 * PAGE)
        cache.deadlines[timed] = 60.0
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), [])
        self.assertEqual(cache.read_scheduler.stats.budget_holds, 1)
        hit(cache, "urgent", 4 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["urgent"])
        cache._unpin_prefetch(CacheRequestHandle("landed", 0), reason="admit")
        finish_read(cache, CacheRequestHandle("urgent", 0))
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["timed"])

    def test_room_refused_is_reported_once_whole_for_a_revoke_and_by_the_cut(self):
        """The price of host memory follows these reports: a read trimmed to the
        pages that fit reports the part cut, one that finds none reports whole, each
        under its request and handed over once; nothing is kept until someone asks."""
        cache = cache_mode_fixture()
        pool = cache.cache_controller.mem_pool_host
        pool.available = 0
        hit(cache, "unasked", 6 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertIsNone(cache.room_refusals)
        self.assertEqual(cache.take_room_refusals(), [])
        pool.available = 2 * PAGE
        cut, _ = hit(cache, "cut", 6 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["cut"])
        self.assertEqual(cache.take_room_refusals(), [(cut, 4 * PAGE)])
        pool.available = 0
        revoked, _ = hit(cache, "revoked", 6 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), [])
        self.assertEqual(
            (cache.take_room_refusals(), cache.take_room_refusals()),
            ([(revoked, 6 * PAGE)], []),
        )

    def test_soft_pins_free_a_staged_read_so_the_asking_read_launches_whole(self):
        """Without them the read of a turn that has arrived is cut or revoked while a
        read staged for a turn not here keeps its pin. The staged read named by the
        prefetcher is released and eviction then finds its pages; the asking read is
        not reported as refused. Only the pages it is short of are asked for."""
        cache = cache_mode_fixture(pin_fraction=0.5)
        cache.room_refusals = []
        pool = cache.cache_controller.mem_pool_host
        pool.available = 2 * PAGE
        staged = CacheRequestHandle("staged", 0)
        cache._pin_prefetch(staged, 7, 6 * PAGE)
        cache.prefetch_loaded_tokens_by_reqid[staged] = 6 * PAGE
        asked = []

        def victims(handle, tokens):
            asked.append((handle.rid, tokens))
            return [staged]

        def evict_host(tokens):
            # Only an unpinned span can be evicted.
            if staged not in cache.pinned_prefetch:
                pool.available = 6 * PAGE

        cache.soft_pin_victims_fn = victims
        cache.evict_host = Mock(side_effect=evict_host)
        hit(cache, "arrived", 6 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(asked, [("arrived", 4 * PAGE)])
        self.assertEqual(cache.evict_host.call_args_list[-1].args, (4 * PAGE,))
        self.assertEqual(launched(cache), ["arrived"])
        self.assertEqual(pool.allocs, [6 * PAGE])
        self.assertEqual((cache.tree_core.locks[7], cache.pinned_prefetch_tokens), (0, 0))
        self.assertNotIn(staged, cache.prefetch_loaded_tokens_by_reqid)
        self.assertEqual(cache.take_room_refusals(), [])

    def test_revoked_or_aborted_parked_read_is_dropped_at_dispatch(self):
        """A parked handle that left ongoing_prefetch (abort, revoke) must not be
        launched: its bounce would be allocated for a request nobody waits on."""
        cache = cache_mode_fixture()
        gone, _ = hit(cache, "gone", 2 * PAGE)
        kept, _ = hit(cache, "kept", 2 * PAGE)
        drain(cache, 2)
        cache.revoke_pending_prefetch(gone)
        self.assertFalse(cache.read_scheduler.is_parked(gone))
        del cache.ongoing_prefetch[hit(cache, "raced", 2 * PAGE)[0]]
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["kept"])
        self.assertEqual(cache.read_scheduler.parked_count(), 0)
        self.assertEqual(cache.cache_controller.mem_pool_host.allocs, [2 * PAGE])

    def test_release_subtracts_exactly_the_dispatched_charge(self):
        """A short host pool trims the read to what fits; the charge, the tracked
        key and every release agree on that span, and revoking a parked read
        (never charged) leaves the dispatched reads' occupancy alone."""
        cache = cache_mode_fixture()
        cache.cache_controller.mem_pool_host.available = 2 * PAGE
        a, op = hit(cache, "a", 4 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["a"])
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 2 * PAGE)
        self.assertEqual(len(cache.ongoing_prefetch[a].prefetch_key), 2 * PAGE)
        self.assertEqual(len(op.hash_value), 2)
        b, _ = hit(cache, "b", 4 * PAGE)
        drain(cache, 1)
        cache.revoke_pending_prefetch(b)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 2 * PAGE)
        cache.revoke_pending_prefetch(a)
        self.assertEqual(cache.cache_controller.prefetch_tokens_occupied, 0)

    def test_timeout_policy_never_cuts_a_parked_read(self):
        """Nothing has been read while parked and the timeout clock runs from
        dispatch, so a parked op is cut only once it is in flight."""
        cache = cache_mode_fixture()
        cache.prefetch_stop_policy = "timeout"
        cache.pp_rank = 0
        cache._prefetch_timeout_check_linear_func = lambda op: True
        cache._all_reduce = lambda tensor, op: None
        handle, op = hit(cache, "a", 2 * PAGE)
        drain(cache, 1)
        self.assertFalse(cache._can_terminate_prefetch(op))
        cache._dispatch_storage_reads()
        self.assertTrue(cache._can_terminate_prefetch(op))


class TestReadOwnerHooks(CustomTestCase):
    """The cache's two hooks for the next-turn prefetcher's gate and hold."""

    def gated_cache(self, *, allow):
        cache = cache_mode_fixture()
        cache.enable_storage = True
        cache.tree_components = ()
        cache._build_sidecar_transfers = Mock(return_value=[])
        cache.tree_core.is_eagle = False
        cache.tree_core.prefetch_anchor_info = lambda node: (None, None)
        cache.cache_controller.get_prefetch_submission = lambda rid: None
        cache.cache_controller.submit_prefetch = Mock(
            return_value=PrefetchSubmission(operation=Mock(assume_stored=False))
        )
        cache.cache_controller.pp_prefetch_decisions = {}
        cache.asked = []

        def gate(handle, matched_tokens):
            cache.asked.append((handle.rid, matched_tokens))
            return allow

        cache.prefetch_gate_fn = gate
        return cache

    def prefetch(self, cache, *, tokens=8 * PAGE, matched=2 * PAGE):
        handle = CacheRequestHandle("a", 0)
        cache.prefetch_from_storage(
            handle,
            7,
            array("q", range(tokens)),
            matched_prefix_tokens=list(range(matched)),
        )
        return handle

    def test_a_declined_lookup_leaves_nothing_for_admission_to_wait_on(self):
        """Admission waits for a read it knows about; a declined one must leave no entry
        or the turn would never be admitted, and no anchor lock to leak."""
        cache = self.gated_cache(allow=False)
        handle = self.prefetch(cache)
        self.assertEqual(cache.asked, [("a", 2 * PAGE)])
        cache.cache_controller.submit_prefetch.assert_not_called()
        self.assertTrue(cache.check_prefetch_progress(handle))
        self.assertEqual(dict(cache.tree_core.locks), {})

    def test_joining_a_read_under_way_does_not_ask_the_gate(self):
        """The gate counts a turn's lookups: arrival joining the staged read must not
        spend the budget the turn needs if that read is later lost."""
        cache = self.gated_cache(allow=True)
        handle = self.prefetch(cache)
        self.assertTrue(cache.has_ongoing_prefetch(handle))
        self.prefetch(cache)
        self.assertEqual(len(cache.asked), 1)

    def test_a_call_that_would_not_look_up_does_not_ask_the_gate(self):
        """Too short or rate limited: no lookup would be issued, so none is counted."""
        cache = self.gated_cache(allow=True)
        self.prefetch(cache, tokens=1)
        cache.cache_controller.prefetch_rate_limited = lambda: True
        self.prefetch(cache)
        self.assertEqual(cache.asked, [])

    def test_the_gate_that_reads_the_prompt_gets_it_whole_from_the_match_on(self):
        """The prefetcher cuts a plan's key to what the prompt repeats of it. A retry that
        knows where the last hit ended asks storage for that span only; handed the same
        cut, the gate would take the prompt for one that leaves the key early. The gate
        bound for the prompt is asked in place of the two-argument one."""
        cache = self.gated_cache(allow=False)
        asked = []

        def gate(handle, matched_tokens, prompt_tail):
            asked.append((handle.rid, matched_tokens, list(prompt_tail)))
            return True

        cache.prefetch_gate_prompt_fn = gate
        cache.prefetch_from_storage(
            CacheRequestHandle("a", 0),
            7,
            array("q", range(2 * PAGE, 10 * PAGE)),
            matched_prefix_tokens=list(range(2 * PAGE)),
            storage_hit_end=6 * PAGE,
        )
        self.assertEqual(asked, [("a", 2 * PAGE, list(range(2 * PAGE, 10 * PAGE)))])
        self.assertEqual(cache.asked, [])
        (submitted,) = cache.cache_controller.submit_prefetch.call_args_list
        self.assertEqual(len(submitted.args[1]), 4 * PAGE)
        # Its "recompute" is a decline like any other: no read for admission to wait on.
        cache.prefetch_gate_prompt_fn = lambda handle, matched_tokens, prompt_tail: (
            False
        )
        declined = CacheRequestHandle("b", 0)
        cache.prefetch_from_storage(
            declined, 7, array("q", range(8 * PAGE)), matched_prefix_tokens=[]
        )
        self.assertFalse(cache.has_ongoing_prefetch(declined))
        self.assertEqual(cache._prefetch_outcome_stats["declined_gate"], 1)

    def test_every_answered_lookup_is_reported_once_with_what_it_asked_and_found(self):
        """The prefetcher gives a lookup back when it found nothing worth reading, and
        marks where storage ends. A lookup terminated before it ran also carries a count
        of zero, which is no answer; nor is one whose request is gone."""
        cache = cache_mode_fixture()
        cache.prefetch_threshold = 2 * PAGE
        reported = []
        cache.prefetch_lookup_fn = lambda handle, **answer: reported.append(
            (handle.rid, answer)
        )
        for rid, hit_tokens in (("empty", 0), ("short", PAGE), ("found", 4 * PAGE)):
            _, operation = hit(cache, rid, hit_tokens)
            operation.storage_start = 6 * PAGE
            operation.stats_requested_tokens = 8 * PAGE
        _, cut_short = hit(cache, "terminated", 0)
        cut_short.is_terminated = lambda: True
        del cache.ongoing_prefetch[hit(cache, "gone", 4 * PAGE)[0]]
        drain(cache, 5)
        self.assertEqual(
            reported,
            [
                (
                    rid,
                    dict(
                        storage_start=6 * PAGE,
                        requested_tokens=8 * PAGE,
                        hit_tokens=hit_tokens,
                    ),
                )
                for rid, hit_tokens in (
                    ("empty", 0),
                    ("short", PAGE),
                    ("found", 4 * PAGE),
                )
            ],
        )
        self.assertEqual(
            cache.read_scheduler.parked_handles(), [CacheRequestHandle("found", 0)]
        )

    def test_a_held_read_holds_no_memory_and_keeps_its_turn_waiting(self):
        """The turn must not be admitted past a read that is only delayed: it would
        recompute history the SSD was about to return."""
        cache = cache_mode_fixture()
        cache.cache_controller.pp_prefetch_decisions = {}
        cache.pp_rank = 0
        cache._all_reduce = lambda tensor, op: None
        cache.read_not_before_fn = lambda handle, read, ahead_pages: float("inf")
        handle, _ = hit(cache, "a", 4 * PAGE)
        drain(cache, 1)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), [])
        self.assertEqual(cache.cache_controller.mem_pool_host.allocs, [])
        self.assertFalse(cache.check_prefetch_progress(handle))
        cache.read_not_before_fn = lambda handle, read, ahead_pages: None
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["a"])


class TestReadLine(CustomTestCase):
    """What the next-turn prefetcher's read-time estimate reads from the cache: the line
    ahead of a read, and the read path's lags once it installs their measuring."""

    def test_the_line_is_the_launched_reads_and_the_parked_ones_ordered_no_later(self):
        """A read staged now is dispatched after the reads needed now and the timed ones
        due before it, and waits behind what the read thread already holds; the reads due
        later do not delay it."""
        cache = cache_mode_fixture(window_pages=4)
        urgent, _ = hit(cache, "urgent", 2 * PAGE)
        early, _ = hit(cache, "early", 3 * PAGE)
        late, _ = hit(cache, "late", 5 * PAGE)
        cache.deadlines.update({early: 120.0, late: 300.0})
        drain(cache, 3)
        self.assertEqual(
            [cache.storage_read_line_pages(due) for due in (None, 200.0, 400.0)],
            [2, 2 + 3, 2 + 3 + 5],
        )
        # The window lets the first two go: launched pages stay ahead of a read needed now.
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["urgent", "early"])
        self.assertEqual(
            [cache.storage_read_line_pages(due) for due in (None, 400.0)], [5, 5 + 5]
        )

    def test_only_a_parked_read_can_be_dropped_and_its_turn_is_then_free_to_admit(self):
        """The starved-GPU rule gives a read up so its turn recomputes the span. A parked
        read holds no host memory and nothing of it is in flight; one handed to the read
        thread must be left to land. After the drop admission finds nothing to wait on."""
        cache = cache_mode_fixture(window_pages=2)
        cache.cache_controller.pp_prefetch_decisions = {}
        dispatched, _ = hit(cache, "dispatched", 2 * PAGE)
        parked, operation = hit(cache, "parked", 4 * PAGE)
        operation.storage_start = 6 * PAGE
        cache.deadlines[parked] = 60.0
        drain(cache, 2)
        cache._dispatch_storage_reads()
        self.assertEqual(launched(cache), ["dispatched"])
        self.assertEqual(
            (cache.parked_read_span(dispatched), cache.parked_read_span(parked)),
            (None, (6 * PAGE, 4 * PAGE)),
        )
        self.assertEqual(cache.parked_read_pages_ahead(parked), 2)
        self.assertFalse(cache.drop_parked_read(dispatched))
        self.assertTrue(cache.has_ongoing_prefetch(dispatched))
        self.assertTrue(cache.drop_parked_read(parked))
        self.assertFalse(cache.drop_parked_read(parked))
        self.assertEqual(cache.read_scheduler.parked_count(), 0)
        self.assertTrue(cache.check_prefetch_progress(parked))
        self.assertEqual(cache.cache_controller.mem_pool_host.allocs, [2 * PAGE])

    def test_the_lags_are_taken_at_park_dispatch_and_landing_once_installed(self):
        """Lookup lag: submit to park. Fixed lag: dispatch to insertion, less the pages
        ahead at dispatch and its own at the read thread's rate, so a read behind another
        is not charged that read's time. A cache nobody asked measures nothing."""
        plain = cache_mode_fixture()
        _, untouched = hit(plain, "a", 4 * PAGE)
        drain(plain, 1)
        plain._dispatch_storage_reads()
        self.assertNotIn("dispatched_at", vars(untouched))

        cache = cache_mode_fixture(write_hold_tokens=10 * PAGE)
        cache.read_lags = lags = ReadPathLags()
        cache.cache_controller.storage_read_ms_per_page = 10.0
        cache.cache_controller.mem_pool_host.free = Mock()
        cache._check_hybrid_prefetch_result = Mock(return_value=True)
        cache._apply_cache_actions = Mock()
        cache.dec_host_lock_ref = Mock()
        cache.tree_core.commit_hicache_transfers = Mock()
        cache.tree_core.insert_host = lambda *args: SimpleNamespace(
            cache_actions=[],
            host_insert_dropped=False,
            prefix_len=0,
            inserted_host_node=5,
        )

        def read(rid, clock, *, landed_after):
            """One read of four pages: looked up 30 ms before it parks at 200 s."""
            _, operation = hit(cache, rid, 4 * PAGE)
            operation.start_time = 100.0
            operation.completed_tokens = 4 * PAGE
            operation.pool_storage_result = None
            clock.monotonic.return_value = 100.03
            clock.perf_counter.return_value = 200.0
            drain(cache, 1)
            cache._dispatch_storage_reads()
            clock.perf_counter.return_value = 200.0 + landed_after
            return operation

        with patch(f"{LOGGER}.time") as clock:
            for turn in range(LAG_MIN_SAMPLES - 1):
                cache._handle_prefetch_result(
                    read(f"r{turn}", clock, landed_after=0.05)
                )
            self.assertIsNone(lags.fixed.seconds)
            first = read("first", clock, landed_after=0.09)
            second = read("second", clock, landed_after=0.09)
            self.assertEqual((first.ahead_pages, second.ahead_pages), (0, 4))
            cache._handle_prefetch_result(second)
            # The write hold comes on and goes off: one hold of 4 s.
            for pending, now in ((10 * PAGE, 300.0), (0, 304.0)):
                cache.pending_backup_tokens = pending
                clock.perf_counter.return_value = now
                cache._dispatch_storage_reads()
        self.assertAlmostEqual(lags.lookup.seconds, 0.03)
        self.assertAlmostEqual(lags.fixed.seconds, 0.01)
        self.assertEqual(
            (lags.write_hold.samples, lags.write_hold.ewma_s, lags.write_hold_since),
            (1, 4.0, None),
        )


class TestPinnedStaging(CustomTestCase):
    def pin_events(self, cache, fn):
        with self.assertLogs(LOGGER, level="INFO") as logs:
            fn()
            logging.getLogger(LOGGER).info("HICACHE_EVT end rid=-")
        kinds = [line.split("HICACHE_EVT ")[1].split()[0] for line in logs.output]
        return kinds[:-1]

    def test_pin_is_one_host_lock_released_once(self):
        cache = cache_mode_fixture(pin_fraction=0.5)
        handle = CacheRequestHandle("a", 0)
        cache._pin_prefetch(handle, 3, 200)
        self.assertEqual((cache.tree_core.locks[3], cache.pinned_prefetch_tokens), (1, 200))
        cache._unpin_prefetch(handle, reason="admit")
        cache._unpin_prefetch(handle, reason="abort")
        self.assertEqual((cache.tree_core.locks[3], cache.pinned_prefetch_tokens), (0, 0))

    def test_repin_replaces_the_previous_pin_of_the_handle(self):
        cache = cache_mode_fixture(pin_fraction=0.5)
        handle = CacheRequestHandle("a", 0)
        cache._pin_prefetch(handle, 3, 200)
        cache._pin_prefetch(handle, 4, 100)
        self.assertEqual(dict(cache.tree_core.locks), {3: 0, 4: 1})
        self.assertEqual(cache.pinned_prefetch_tokens, 100)

    def test_span_over_the_budget_is_left_unpinned(self):
        cache = cache_mode_fixture(pin_fraction=0.5)
        cache._pin_prefetch(CacheRequestHandle("a", 0), 3, 300)
        kinds = self.pin_events(
            cache, lambda: cache._pin_prefetch(CacheRequestHandle("b", 0), 4, 300)
        )
        self.assertEqual(kinds, ["srs_pin_skipped"])
        self.assertEqual((cache.tree_core.locks[4], cache.pinned_prefetch_tokens), (0, 300))
        self.assertNotIn(CacheRequestHandle("b", 0), cache.pinned_prefetch)

    def test_pin_fraction_zero_takes_no_lock(self):
        """The default leaves the host tree exactly as before v2."""
        cache = cache_mode_fixture(pin_fraction=0.0)
        cache._pin_prefetch(CacheRequestHandle("a", 0), 3, 200)
        self.assertEqual((dict(cache.tree_core.locks), cache.pinned_prefetch), ({}, {}))

    def test_stale_pin_is_released_as_a_leak(self):
        cache = cache_mode_fixture(pin_fraction=0.5)
        handle = CacheRequestHandle("a", 0)
        cache._pin_prefetch(handle, 3, 200)
        pinned_at = cache.pinned_prefetch[handle].pinned_at
        cache._release_stale_pins(pinned_at + PREFETCH_PIN_MAX_AGE_S - 1.0)
        self.assertEqual(cache.tree_core.locks[3], 1)
        kinds = self.pin_events(
            cache,
            lambda: cache._release_stale_pins(pinned_at + PREFETCH_PIN_MAX_AGE_S + 1.0),
        )
        self.assertEqual(kinds, ["srs_unpin"])
        self.assertEqual((cache.tree_core.locks[3], cache.pinned_prefetch_tokens), (0, 0))


def backup_ready_cache(node_pages):
    """A cache whose tree backs up each node of `node_pages` (node id -> pages), queued
    by the controller's own write_storage."""
    cache = cache_mode_fixture()
    cache.enable_storage = True
    cache.hicache_storage_pass_prefix_keys = False
    cache.sidecar_pool_specs = []
    specs = {
        node_id: StorageBackupSpec(
            host_value=torch.arange(pages * PAGE),
            token_ids=array("q", range(pages * PAGE)),
            hash_value=[f"n{node_id}-{p}" for p in range(pages)],
            prefix_keys=None,
            comp_xfers={},
        )
        for node_id, pages in node_pages.items()
    }
    cache.tree_core.build_storage_backup_spec = lambda node_id, _: specs[node_id]
    cc = cache.cache_controller
    cc.backup_queue = Queue()
    cc.write_storage = MethodType(HybridCacheController.write_storage, cc)
    return cache


def finish_writes(cache, *written_tokens):
    """The write thread acks its oldest writes, each with the tokens it wrote."""
    cc = cache.cache_controller
    for tokens in written_tokens:
        operation = cc.backup_queue.get_nowait()
        operation.completed_tokens = tokens
        cc.ack_backup_queue.put(operation)


class _MarkingTreeCore(_LockedTreeCore):
    """Also records the stored marks the cache asks the tree for."""

    def __init__(self):
        super().__init__()
        self.queued = {}
        self.finished = []

    def mark_storage_write(self, node_id, *, op_id):
        self.queued[node_id] = op_id

    def finish_storage_write(self, node_id, *, op_id, stored_at, retire):
        self.finished.append((node_id, op_id, stored_at is not None))
        return []

    def unstored_ancestors(self, node_id):
        return []


def stored_tracking_cache(node_pages, **switches):
    """backup_ready_cache on one rank with SGLANG_HICACHE_STORAGE_TRACK_STORED resolved
    on, and the other Retain switches as given."""
    cache = backup_ready_cache(node_pages)
    build_spec = cache.tree_core.build_storage_backup_spec
    cache.tree_core = _MarkingTreeCore()
    cache.tree_core.build_storage_backup_spec = build_spec
    cache.enable_storage = True
    cache.retention = RetentionSwitches(track_stored=True, **switches)
    cache._storage_skip_nodes = set()
    cache._retire_at_ack = set()
    cache.tp_world_size = cache.pp_size = 1
    cache.cache_controller.backup_skip = False
    cache.cache_controller.storage_backend = Mock(spec=HiCacheStorage)
    return cache


def event_lines(logs, kind):
    """The HICACHE_EVT lines of one kind, without the logger prefix."""
    return [
        line.split("HICACHE_EVT ")[1]
        for line in logs.output
        if f"HICACHE_EVT {kind} " in line
    ]


class TestWriteBacklogHold(CustomTestCase):
    """Host tokens locked by SSD writes not yet acked, and the read hold they drive."""

    def test_pending_write_tokens_leave_with_each_acked_or_released_write(self):
        """The write hold reads this count, so a release path that forgot it would keep
        the hold on for good. An ack returns what its write locked, not what it wrote:
        a write cut short still unlocks all its pages."""
        cache = backup_ready_cache({1: 4, 2: 6, 3: 2})
        for node_id in (1, 2, 3):
            cache.write_backup_storage(node_id)
        self.assertEqual(cache.pending_backup_tokens, 12 * PAGE)
        self.assertEqual(dict(cache.tree_core.locks), {1: 1, 2: 1, 3: 1})
        finish_writes(cache, 2 * PAGE)
        drain(cache, 0, write_acks=1)
        self.assertEqual(cache.pending_backup_tokens, 8 * PAGE)
        self.assertEqual(dict(cache.tree_core.locks), {1: 0, 2: 1, 3: 1})
        # Detach and exit drop the writes a stopped write thread never acks.
        StorageAttachment(cache)._release_pending_storage_ops()
        self.assertEqual((cache.pending_backup_tokens, cache.ongoing_backup), (0, {}))
        self.assertEqual(dict(cache.tree_core.locks), {1: 0, 2: 0, 3: 0})

    def test_a_write_acked_short_unlocks_its_pages_and_leaves_them_unstored(self):
        """Any ack returns the lock and the pending tokens (a store that fails every
        write must not pin host memory); only one covering every page marks the node."""
        cache = stored_tracking_cache({1: 4, 2: 6})
        with self.assertLogs(LOGGER, level="INFO") as logs:
            for node_id in (1, 2):
                cache.write_backup_storage(node_id)
            op = dict(cache.tree_core.queued)
            finish_writes(cache, 4 * PAGE, 6 * PAGE - 1)
            drain(cache, 0, write_acks=2)
        self.assertEqual(
            cache.tree_core.finished, [(1, op[1], True), (2, op[2], False)]
        )
        self.assertEqual(
            (cache.pending_backup_tokens, dict(cache.tree_core.locks)),
            (0, {1: 0, 2: 0}),
        )
        done = event_lines(logs, "h2s_done")
        self.assertEqual([line.split()[-1] for line in done], ["stored=1", "stored=0"])
        self.assertEqual(len(event_lines(logs, "h2s_short")), 1)

    def test_with_the_switch_off_no_node_is_marked(self):
        """Off, the tree is asked for no mark (this tree core has none to give) and the
        ack line keeps exactly the fields earlier campaigns' scripts read."""
        cache = backup_ready_cache({1: 4})
        with self.assertLogs(LOGGER, level="INFO") as logs:
            cache.write_backup_storage(1)
            finish_writes(cache, 2 * PAGE)
            drain(cache, 0, write_acks=1)
        (done,) = event_lines(logs, "h2s_done")
        self.assertRegex(done, r"^h2s_done op=\d+ tokens=4 pending=0$")
        self.assertEqual(event_lines(logs, "h2s_short"), [])

    def test_a_write_short_on_any_rank_is_unstored_on_every_rank(self):
        """completed_tokens is a rank's own count: the verdict is the minimum over TP
        ranks. A rank that writes nothing votes yes; the detach path asks no other."""
        cache = stored_tracking_cache({1: 4, 2: 4})
        cache.tp_world_size = 2
        reduced_with = []

        def the_other_rank_cut_the_second_write(votes, op):
            reduced_with.append(op)
            votes.copy_(torch.minimum(votes, torch.tensor([1, 0], dtype=votes.dtype)))

        cache._all_reduce = the_other_rank_cut_the_second_write
        for node_id in (1, 2):
            cache.write_backup_storage(node_id)
        finish_writes(cache, 4 * PAGE, 4 * PAGE)
        drain(cache, 0, write_acks=2)
        self.assertEqual(
            [stored for *_, stored in cache.tree_core.finished], [True, False]
        )
        self.assertEqual(reduced_with, [torch.distributed.ReduceOp.MIN])

        cache._all_reduce = Mock()
        cache.cache_controller.backup_skip = True
        cache.write_backup_storage(1)
        finish_writes(cache, 0)
        drain(cache, 0, write_acks=1)
        self.assertTrue(cache.tree_core.finished[-1][2])
        cache._all_reduce.assert_called_once()

        cache.cache_controller.backup_skip = False
        cache.write_backup_storage(2)
        finish_writes(cache, 4 * PAGE)
        drain(cache, 0, write_acks=None)
        self.assertTrue(cache.tree_core.finished[-1][2])
        cache._all_reduce.assert_called_once()

    def test_a_split_of_a_skipped_node_skips_both_fragments(self):
        """A node split while its host copy is in flight gets a write per fragment: the
        skip or the retire decided for it follows both, or half an ended turn stays."""
        cache = stored_tracking_cache(
            {}, skip_ended_writes=True, retire_ended_pages=True
        )
        cache.ongoing_write_through = {
            ack_id: _OngoingWriteThrough(ack_id, None, [ack_id]) for ack_id in (7, 8, 9)
        }
        cache._storage_skip_nodes = {7}
        cache._retire_at_ack = {8}
        for node_id in (7, 8, 9):
            cache._replace_pending_write_through_node(
                node_id, node_id, [10 * node_id, node_id]
            )
        self.assertEqual(
            (cache._storage_skip_nodes, cache._retire_at_ack), ({7, 70}, {8, 80})
        )
        self.assertEqual(cache.ongoing_write_through[9].publish_node_ids, [90, 9])

    def test_timed_read_stays_parked_until_the_write_backlog_halves(self):
        """Fed the cache's pending write tokens, the hold parks a timed read from the
        threshold down to half while a "now" read launches, logging one line coming on
        and one going off."""
        hold = 10 * PAGE
        cache = cache_mode_fixture(write_hold_tokens=hold)
        timed, _ = hit(cache, "timed", 2 * PAGE)
        cache.deadlines[timed] = 60.0
        hit(cache, "urgent", 2 * PAGE)
        drain(cache, 2)
        cache.pending_backup_tokens = hold
        with self.assertLogs(LOGGER, level="INFO") as logs:
            cache._dispatch_storage_reads()
            self.assertEqual(launched(cache), ["urgent"])
            cache._dispatch_storage_reads()
            self.assertEqual(cache.read_scheduler.parked_handles(), [timed])
            cache.pending_backup_tokens = hold // 2
            cache._dispatch_storage_reads()
            self.assertEqual(launched(cache), ["timed"])
        states = [
            line.split("srs_write_hold state=")[1].split()[0]
            for line in logs.output
            if "srs_write_hold" in line
        ]
        self.assertEqual(states, ["on", "off"])


class TestRetentionSwitches(CustomTestCase):
    ALL_ON = RetentionSwitches(
        track_stored=True,
        skip_ended_writes=True,
        touch_interval_s=300.0,
        retire_ended_pages=True,
    )

    def switch_cache(self, *, prefetch="on") -> UnifiedRadixCache:
        """A real CPU cache meeting every condition, attached to a store that offers
        the touch interval and the retire."""
        from test_session_unified_radix_cache import make_params

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", hicache_storage_next_turn_prefetch=prefetch),
            role="scheduler",
        )
        cache = UnifiedRadixCache(make_params(enable_session=False, page_size=2))
        store = Mock(
            spec=HiCacheStorage, touch_interval_s=300.0, retire_ended_pages=True
        )
        cache.cache_controller = SimpleNamespace(storage_backend=store)
        return cache

    def pin_env_switches(self, *, track_stored: bool, skip_ended_writes: bool) -> None:
        # Closed as a cleanup: override() skips its restore when the test body raises.
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            envs.SGLANG_HICACHE_STORAGE_TRACK_STORED.override(track_stored)
        )
        stack.enter_context(
            envs.SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES.override(skip_ended_writes)
        )

    def test_a_switch_missing_what_it_needs_stays_off_and_says_so(self):
        """Each switch acts through something else: granted without it, a skipped write
        leaves a hole no child rewrites and a retire zeroes half a page's files."""
        wanted = self.ALL_ON
        met = dict(
            host_cache_mode=True,
            python_tree_core=True,
            single_pp_rank=True,
            next_turn_prefetch=True,
            full_kv_only=True,
        )
        self.assertEqual(resolve_retention_switches(wanted, **met), (wanted, []))
        marks_and_touch = RetentionSwitches(track_stored=True, touch_interval_s=300.0)
        no_retire = RetentionSwitches(
            track_stored=True, skip_ended_writes=True, touch_interval_s=300.0
        )
        cases = {
            "host_cache_mode": (RetentionSwitches(), 4),
            "python_tree_core": (RetentionSwitches(), 4),
            "single_pp_rank": (RetentionSwitches(), 4),
            "next_turn_prefetch": (marks_and_touch, 2),
            "full_kv_only": (no_retire, 1),
        }
        for unmet, (granted, refusals) in cases.items():
            with self.subTest(unmet=unmet):
                resolved, refused = resolve_retention_switches(
                    wanted, **{**met, unmet: False}
                )
                self.assertEqual((resolved, len(refused)), (granted, refusals))
        # A retired page read back from the SSD is refreshed, or it stays first to go.
        no_touch = RetentionSwitches(
            track_stored=True, skip_ended_writes=True, retire_ended_pages=True
        )
        resolved, refused = resolve_retention_switches(no_touch, **met)
        self.assertEqual(
            (resolved, len(refused)),
            (RetentionSwitches(track_stored=True, skip_ended_writes=True), 1),
        )
        # Without the marks nothing else may act; switches not asked for say nothing.
        no_marks = RetentionSwitches(skip_ended_writes=True, touch_interval_s=300.0)
        resolved, refused = resolve_retention_switches(no_marks, **met)
        self.assertEqual((resolved, len(refused)), (RetentionSwitches(), 2))
        self.assertEqual(
            resolve_retention_switches(
                RetentionSwitches(), **dict.fromkeys(met, False)
            ),
            (RetentionSwitches(), []),
        )

    def test_the_cache_reads_two_switches_from_the_envs_and_two_from_its_store(self):
        """The marks and the write skip are environment switches; the touch interval and
        the retire come from the attached store, and a test double claims neither."""
        cache = self.switch_cache()
        store = cache.cache_controller.storage_backend
        self.pin_env_switches(track_stored=False, skip_ended_writes=False)
        # Nothing asked for: every switch off, and no line in the server log.
        cache.cache_controller.storage_backend = None
        with self.assertNoLogs(LOGGER, level="INFO"):
            cache._init_storage_retention()
        self.assertEqual(cache.retention, RetentionSwitches())

        cache.cache_controller.storage_backend = store
        self.pin_env_switches(track_stored=True, skip_ended_writes=True)
        with self.assertLogs(LOGGER, level="INFO") as logs:
            cache._init_storage_retention()
        self.assertEqual(cache.retention, self.ALL_ON)
        self.assertIn(
            "HiCache retention: track_stored=1 skip_ended_writes=1 "
            "touch_interval_s=300 retire_ended_pages=1",
            logs.output[-1],
        )
        cache.cache_controller.storage_backend = Mock()
        cache._init_storage_retention()
        self.assertEqual(
            cache.retention,
            RetentionSwitches(track_stored=True, skip_ended_writes=True),
        )

    def test_the_cache_passes_the_resolver_what_the_server_really_has(self):
        """Every condition is read off the server: one passed as met when it is not
        grants the write skip without a prefetcher, or a retire that misses files."""
        self.pin_env_switches(track_stored=True, skip_ended_writes=True)
        marks_and_touch = RetentionSwitches(track_stored=True, touch_interval_s=300.0)
        unmet = {
            "the prefetcher": ("off", {}, marks_and_touch),
            "host memory as a cache": (
                "on",
                {"host_memory_mode": "buffer_only"},
                RetentionSwitches(),
            ),
            "the python tree core": (
                "on",
                {"tree_core": SimpleNamespace()},
                RetentionSwitches(),
            ),
            "PP = 1": ("on", {"pp_size": 2}, RetentionSwitches()),
        }
        for name, (prefetch, changed, granted) in unmet.items():
            with self.subTest(unmet=name):
                cache = self.switch_cache(prefetch=prefetch)
                vars(cache).update(changed)
                with self.assertLogs(LOGGER, level="WARNING") as logs:
                    cache._init_storage_retention()
                self.assertEqual(cache.retention, granted)
                self.assertTrue(
                    all("stays off: it needs" in line for line in logs.output)
                )

        # A draft model's sidecar pools are registered after the cache is built: each of
        # its pages then has a file the touch does not name.
        cache = self.switch_cache()
        cache._init_storage_retention()
        self.assertEqual(cache.retention, self.ALL_ON)
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            cache.register_sidecar_pool(SimpleNamespace(name="draft"))
        self.assertEqual(
            cache.retention,
            RetentionSwitches(
                track_stored=True, skip_ended_writes=True, touch_interval_s=300.0
            ),
        )
        self.assertIn(
            "SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES stays off", logs.output[0]
        )

    def test_a_runtime_attach_resolves_the_switches_for_the_store_attached(self):
        """A store attached at runtime states its own touch and retire support: switches
        kept from before would refresh files its cleaner does not order, or stay off."""
        cache = self.switch_cache()
        self.pin_env_switches(track_stored=True, skip_ended_writes=False)
        controller = SimpleNamespace(
            storage_backend=None, mem_pool_host=SimpleNamespace(entries=[])
        )

        def attach_storage_backend(**kwargs):
            controller.storage_backend = Mock(
                spec=HiCacheStorage, touch_interval_s=300.0, retire_ended_pages=False
            )

        controller.attach_storage_backend = attach_storage_backend
        cache.cache_controller = controller
        cache._init_storage_retention()
        self.assertEqual(cache.retention, RetentionSwitches(track_stored=True))
        attached, _ = StorageAttachment(cache).attach(storage_backend="nixl")
        self.assertTrue(attached)
        self.assertEqual(
            cache.retention,
            RetentionSwitches(track_stored=True, touch_interval_s=300.0),
        )


class TestAdmitSiteReleasesThePin(CustomTestCase):
    def test_cache_finished_req_releases_the_requests_pin(self):
        """The landed span's host lock must go when its request takes the span,
        or the pin lives on until the leak guard and keeps the limiter charged."""
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH
        from test_session_unified_radix_cache import make_params

        cache = UnifiedRadixCache(make_params(enable_session=False, page_size=2))
        cache.inc_host_lock_ref = Mock(
            return_value=SimpleNamespace(to_dec_params=lambda: "dec-params")
        )
        cache.dec_host_lock_ref = Mock()
        cache.prefetch_pin_budget_tokens = 10**6
        handle = CacheRequestHandle(rid="r", attempt_id=0)
        cache._pin_prefetch(handle, 42, 6)
        self.assertEqual(cache.pinned_prefetch_tokens, 6)

        prompt, output = array("q", range(1, 10)), array("q", [100, 101])
        n = len(prompt) + len(output)
        cache.req_to_token_pool.req_to_token[0, :n] = (
            cache.token_to_kv_pool_allocator.alloc(12)[:n]
        )
        req = SimpleNamespace(
            rid="r",
            kv_hints=None,
            finished=lambda: True,
            finished_reason=FINISH_LENGTH(length=8),
            time_stats=SimpleNamespace(wait_queue_entry_time=0.0, forward_entry_time=0.0),
            storage_hit_length=0,
            storage_prefetch_last_match_len=None,
            cache_request_handle=handle,
            session=None,
            origin_input_ids=prompt,
            output_ids=output,
            extra_key=None,
            cache_salt=None,
            priority=0,
            session_id=None,
            kv_rotation_base=None,
            last_node=None,
            swa_prefix_lock_released=False,
            kv=SimpleNamespace(
                req_pool_idx=0,
                cache_protected_len=0,
                swa_evicted_seqlen=0,
                swa_dead_lo=lambda page_size: 0,
            ),
        )
        cache.cache_finished_req(req, owned_kv_len=n)
        self.assertEqual((cache.pinned_prefetch, cache.pinned_prefetch_tokens), ({}, 0))
        cache.dec_host_lock_ref.assert_called_once_with(42, "dec-params")


class TestControllerReadRate(CustomTestCase):
    def controller(self):
        cc = HiCacheController.__new__(HiCacheController)
        cc.host_memory_mode = "cache"
        cc.prefetch_tokens_occupied = 0
        cc.prefetch_capacity_limit = 10
        cc.extra_occupancy_fn = None
        cc.storage_read_ms_per_page = None
        cc._storage_read_ms_ewma = None
        cc._storage_read_samples = 0
        return cc

    def test_rate_is_published_only_after_the_minimum_samples(self):
        cc = self.controller()
        for _ in range(STORAGE_READ_RATE_MIN_SAMPLES - 1):
            cc._observe_storage_read(pages=10, seconds=0.05)
        self.assertIsNone(cc.storage_read_ms_per_page)
        cc._observe_storage_read(pages=10, seconds=0.05)
        self.assertAlmostEqual(cc.storage_read_ms_per_page, 5.0)
        cc._observe_storage_read(pages=0, seconds=1.0)
        self.assertAlmostEqual(cc.storage_read_ms_per_page, 5.0)

    def test_limiter_counts_the_pinned_tokens(self):
        cc = self.controller()
        cc.prefetch_tokens_occupied = 4
        self.assertFalse(cc.prefetch_rate_limited())
        cc.extra_occupancy_fn = lambda: 6
        self.assertTrue(cc.prefetch_rate_limited())


class _RecordingStore:
    """The store as the write thread sees it: records each batch_set_v1 call as (keys,
    host indices, prefix keys) and fails the calls whose index is listed."""

    def __init__(self, *, fail_calls=(), on_call=None):
        self.calls = []
        self.sidecar_calls = []
        self.fail_calls = set(fail_calls)
        self.on_call = on_call

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        if self.on_call is not None:
            self.on_call()
        ok = len(self.calls) not in self.fail_calls
        self.calls.append((list(keys), host_indices.tolist(), extra_info.prefix_keys))
        return [ok] * len(keys)

    def batch_set_v2(self, transfers, extra_info=None):
        self.sidecar_calls.append([transfer.name for transfer in transfers])
        return {transfer.name: [True] * len(transfer.keys) for transfer in transfers}

    def take_stored_write_skips(self):
        return 0


def write_controller(*, coalesce_pages, store, backend="nixl"):
    """Only the state the storage write thread reads."""
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = PAGE
    cc.storage_backend_type = backend
    cc.storage_backend = store
    cc.backup_skip = False
    cc.write_coalesce_pages = coalesce_pages
    cc.page_set_func = cc._page_set_zero_copy
    cc.mem_pool_host = SimpleNamespace(layout_lease=nullcontext)
    cc.backup_queue = Queue()
    cc.ack_backup_queue = Queue()
    return cc


def queue_write(cc, name, pages, *, first_host_index=0, pool_transfers=None):
    tokens = pages * PAGE
    operation = StorageOperation(
        host_indices=torch.arange(first_host_index, first_host_index + tokens),
        token_ids=list(range(tokens)),
        hash_value=[f"{name}{p}" for p in range(pages)],
        prefix_keys=[f"{name}-prefix"],
        pool_transfers=pool_transfers,
    )
    cc.backup_queue.put(operation)
    return operation


def run_write_thread(cc):
    """The write thread's own loop, run here until its queue is empty."""
    cc.storage_stop_event = SimpleNamespace(is_set=cc.backup_queue.empty)
    cc.backup_thread_func()


def acked(cc):
    operations = []
    while not cc.ack_backup_queue.empty():
        operations.append(cc.ack_backup_queue.get())
    return operations


class TestStorageWriteCoalescing(CustomTestCase):
    """The write thread merges queued NIXL writes into one page stream."""

    def test_queued_writes_merge_into_one_stream_in_queue_order(self):
        """Each page's hash must stay on its own host page and each op is credited only
        its own pages, with acks in queue order; a group stops once it holds the cap."""
        store = _RecordingStore()
        cc = write_controller(coalesce_pages=6, store=store)
        ops = [
            queue_write(cc, "a", 2, first_host_index=0),
            queue_write(cc, "b", 3, first_host_index=100),
            queue_write(cc, "c", 1, first_host_index=200),
            queue_write(cc, "d", 1, first_host_index=300),
        ]
        run_write_thread(cc)
        a, b, c, d = (op.host_indices.tolist() for op in ops)
        self.assertEqual(
            [(keys, host_indices) for keys, host_indices, _ in store.calls],
            [(["a0", "a1", "b0", "b1", "b2", "c0"], a + b + c), (["d0"], d)],
        )
        done = acked(cc)
        self.assertEqual([op.id for op in done], [op.id for op in ops])
        self.assertEqual(
            [op.completed_tokens for op in done], [2 * PAGE, 3 * PAGE, PAGE, PAGE]
        )

    def test_a_failed_batch_costs_the_other_ops_nothing(self):
        """After a failed batch of the merged stream every op not yet written goes alone,
        as it would unmerged: ops wholly before the failure are done, the op it cuts and
        the later ops get their own writes, and acks keep queue order. Stopping the stream
        instead would drop up to write_coalesce_pages pages of other sessions' writes."""
        half = STORAGE_BATCH_SIZE // 2
        store = _RecordingStore(fail_calls={1})
        cc = write_controller(coalesce_pages=4 * STORAGE_BATCH_SIZE, store=store)
        # Merged batches: a and half of b; then the rest of b and half of c, which fails.
        ops = [
            queue_write(cc, "a", half),
            queue_write(cc, "b", STORAGE_BATCH_SIZE),
            queue_write(cc, "c", STORAGE_BATCH_SIZE),
        ]
        run_write_thread(cc)
        self.assertEqual(
            [keys[0] for keys, _, _ in store.calls], ["a0", f"b{half}", "b0", "c0"]
        )
        done = acked(cc)
        self.assertEqual([op.id for op in done], [op.id for op in ops])
        self.assertEqual(
            [op.completed_tokens for op in done],
            [half * PAGE, STORAGE_BATCH_SIZE * PAGE, STORAGE_BATCH_SIZE * PAGE],
        )

    def test_an_op_is_acked_once_its_own_pages_are_written(self):
        """A merged op's ack, which releases its host lock and its share of the pending
        writes, does not wait for the group's later batches: when the second batch goes
        out, the op that filled the first is already acked."""
        acks_at_call = []
        store = _RecordingStore()
        cc = write_controller(coalesce_pages=4 * STORAGE_BATCH_SIZE, store=store)
        store.on_call = lambda: acks_at_call.append(cc.ack_backup_queue.qsize())
        queue_write(cc, "a", STORAGE_BATCH_SIZE)
        queue_write(cc, "b", STORAGE_BATCH_SIZE)
        run_write_thread(cc)
        self.assertEqual(acks_at_call, [0, 1])
        self.assertEqual(len(acked(cc)), 2)

    def test_a_write_with_sidecar_pools_ends_the_group_and_goes_next_alone(self):
        """Merged writes skip batch_set_v2, so a write carrying sidecar pools must close
        the group and be written whole right after it: merged, its sidecars would be
        lost; passed over, acks would leave queue order."""
        store = _RecordingStore()
        cc = write_controller(coalesce_pages=64, store=store)
        state = PoolTransfer(
            name=PoolName.MAMBA, host_indices=torch.arange(1), keys=["s-state"]
        )
        ops = [
            queue_write(cc, "a", 2),
            queue_write(cc, "b", 1),
            queue_write(cc, "s", 1, pool_transfers=[state]),
            queue_write(cc, "c", 1),
        ]
        run_write_thread(cc)
        self.assertEqual(
            [keys for keys, _, _ in store.calls], [["a0", "a1", "b0"], ["s0"], ["c0"]]
        )
        self.assertEqual(store.sidecar_calls, [[PoolName.MAMBA]])
        self.assertEqual([op.id for op in acked(cc)], [op.id for op in ops])

    def test_writes_go_one_per_call_when_off_or_the_store_reads_prefix_keys(self):
        """The default 0 and any store but NIXL keep the old path: each op is written
        alone with its own prefix keys, which a store such as UMBP ("mori") reads."""
        for backend, coalesce_pages in (("nixl", 0), ("mori", 64)):
            with self.subTest(backend=backend, coalesce_pages=coalesce_pages):
                store = _RecordingStore()
                cc = write_controller(
                    coalesce_pages=coalesce_pages, store=store, backend=backend
                )
                queue_write(cc, "a", 2)
                queue_write(cc, "b", 3)
                run_write_thread(cc)
                self.assertEqual(
                    [(keys, prefix_keys) for keys, _, prefix_keys in store.calls],
                    [(["a0", "a1"], ["a-prefix"]), (["b0", "b1", "b2"], ["b-prefix"])],
                )


if __name__ == "__main__":
    unittest.main()
