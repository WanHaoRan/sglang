"""Cache-mode storage hits under the deadline read order: park without host
memory, dispatch by deadline into the read window, allocate at dispatch, pin the
landed span until admission (agent_cache/ORACLE_V2_PLAN.md 4.2-4.3). No GPU."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import logging
import unittest
from array import array
from collections import defaultdict
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.cache_controller import (
    STORAGE_READ_RATE_MIN_SAMPLES,
    HiCacheController,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.storage_read_scheduler import StorageReadScheduler
from sglang.srt.mem_cache.unified_radix_cache import (
    PREFETCH_PIN_MAX_AGE_S,
    UnifiedRadixCache,
    _OngoingPrefetch,
)
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
    *, order="deadline", window_pages=10**6, budget_tokens=10**9, pin_fraction=0.0
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
    cache.read_scheduler = (
        None
        if order == "fifo"
        else StorageReadScheduler(
            window_pages=window_pages, budget_tokens=budget_tokens, page_size=PAGE
        )
    )
    cache.pinned_prefetch = {}
    cache.pinned_prefetch_tokens = 0
    cache.prefetch_pin_budget_tokens = int(pin_fraction * HOST_POOL_TOKENS)
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


def drain(cache, n):
    cache._drain_storage_control_queues_impl(
        n_storage_hit=n,
        n_ack_prefetch=0,
        n_backup=0,
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


if __name__ == "__main__":
    unittest.main()
