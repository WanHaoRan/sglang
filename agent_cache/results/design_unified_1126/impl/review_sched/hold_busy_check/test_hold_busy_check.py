"""Review scratch (not CI): does the busy memory check see a prefill batch held for its load?

Drives the real Scheduler.get_next_batch_to_run, _hold_prefill_batch_for_load and
_take_held_prefill_batch on a stub scheduler, then the event-loop tail
(scheduler.py:2051-2098: running_batch = plan.running_batch; last_batch = batch;
self_check_during_busy) against a real PagedTokenToKVPoolAllocator (page 64, as the fork
serves), a real UnifiedRadixCache, a real SchedulerPoolStatsObserver and the real
SchedulerInvariantChecker. Only update_running_batch, the DP adapter and the ngram hook
are stubbed, each as the identity they are on one rank with no DP attention
(scheduler.py:4296-4378 returns its input batch on every path; dp_attn.py:572-592).
"""

import types
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import torch  # noqa: E402

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import NextBatchPlan, Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components import invariant_checker  # noqa: E402
from sglang.srt.managers.scheduler_components.invariant_checker import (  # noqa: E402
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (  # noqa: E402
    SchedulerPoolStatsObserver,
)
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)
from sglang.srt.mem_cache.allocator.paged import (  # noqa: E402
    PagedTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache.components import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

PAGE = 64
POOL_TOKENS = 4 * MIN_LOAD_TOKENS
D_PREFIX = 2 * PAGE  # decode request: cached, locked prefix
D_TAIL = 10  # decode request: own tokens past the prefix (one page)
P_PREFIX = MIN_LOAD_TOKENS  # held request: prefix loaded from host, locked in the tree
P_EXTEND = 150  # held request: extend tokens allocated at build time (3 pages)


class _Fixture:
    def __init__(self, max_steps: int):
        self.allocator = PagedTokenToKVPoolAllocator(
            size=POOL_TOKENS,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        self.req_pool = ReqToTokenPool(
            size=8, max_context_len=POOL_TOKENS, device="cpu", enable_memory_saver=False
        )
        with envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override("python"):
            self.cache = UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=self.req_pool,
                    token_to_kv_pool_allocator=self.allocator,
                    page_size=PAGE,
                    tree_components=(ComponentType.FULL,),
                )
            )
        # Decode request D: prefix in the tree and locked, one page of its own.
        d_req = self._req("d", rpi=0, key_base=0, prefix=D_PREFIX, own=D_TAIL)
        self.decode = ScheduleBatch(reqs=[d_req])
        self.decode.forward_mode = ForwardMode.DECODE
        # Prefill request P: 8,192 tokens loaded from host (in the tree, locked), and
        # its extend tokens allocated by prepare_for_extend at build time.
        p_req = self._req("p", rpi=1, key_base=10**6, prefix=P_PREFIX, own=P_EXTEND)
        p_req.host_loaded_length = P_PREFIX
        self.prefill = ScheduleBatch(reqs=[p_req])
        self.prefill.forward_mode = ForwardMode.EXTEND
        self.prefill.hicache_consumer_index = 0
        self.p_uncached = -(-(P_PREFIX + P_EXTEND) // PAGE) * PAGE - P_PREFIX

        self.load_done = False
        self.sched = self._scheduler(max_steps)
        observer = SchedulerPoolStatsObserver(
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            session_controller=None,
            hisparse_coordinator=None,
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hisparse=False,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.allocator.size,
            get_last_batch=lambda: self.sched.last_batch,
            get_running_batch=lambda: self.sched.running_batch,
        )
        # Wired exactly as Scheduler.init_invariant_checker (scheduler.py:2448-2465).
        self.checker = SchedulerInvariantChecker(
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            disaggregation_mode=DisaggregationMode.NULL,
            page_size=PAGE,
            full_tokens_per_layer=None,
            swa_tokens_per_layer=None,
            max_total_num_tokens=self.allocator.size,
            tree_cache=self.cache,
            token_to_kv_pool_allocator=self.allocator,
            req_to_token_pool=self.req_pool,
            pool_stats_observer=observer,
            get_last_batch=lambda: self.sched.last_batch,
            get_running_batch=lambda: self.sched.running_batch,
            get_chunked_req=lambda: self.sched.chunked_req,
            scheduler_stage_metrics=None,
        )

    def _req(self, rid, *, rpi, key_base, prefix, own):
        prefix_locs = self.allocator.alloc(prefix)
        key = RadixKey(array("q", range(key_base, key_base + prefix)))
        self.cache.insert(InsertParams(key=key, value=prefix_locs))
        matched = self.cache.match_prefix(MatchPrefixParams(key=key))
        assert len(matched.device_indices) == prefix
        self.cache.inc_lock_ref(matched.last_device_node)
        own_locs = self.allocator.alloc(-(-own // PAGE) * PAGE)
        locs = torch.cat((prefix_locs, own_locs))
        self.req_pool.req_to_token[rpi, : len(locs)] = locs.to(torch.int32)
        return SimpleNamespace(
            rid=rid,
            kv=ReqKvInfo(
                req_pool_idx=rpi,
                cache_protected_len=prefix,
                kv_committed_len=prefix + own,
                kv_allocated_len=prefix + own,
            ),
            beam_group=None,
            host_loaded_length=0,
        )

    def _scheduler(self, max_steps):
        fx = self
        s = SimpleNamespace(
            enable_fpm=False,
            dllm_config=None,
            chunked_req=None,
            enable_hisparse=False,
            is_mixed_chunk=False,
            require_mlp_sync=False,
            prefill_decode_interval=0,
            scheduler_stage_metrics=None,
            prefill_load_hold=PrefillLoadHold(max_steps=max_steps),
            running_batch=self.decode,
            last_batch=self.decode,
            built=0,
        )
        s.process_pending_chunked_abort = lambda: None
        s._process_hicache_events = lambda: None
        s._should_defer_prefill = lambda: False

        def get_new_batch_prefill(running_batch):
            s.built += 1
            batch = fx.prefill if s.built == 1 else None
            return NextBatchPlan(batch_to_run=batch, running_batch=running_batch)

        s.get_new_batch_prefill = get_new_batch_prefill
        s.update_running_batch = lambda batch: batch
        s.dp_attn_adapter = SimpleNamespace(
            maybe_prepare_mlp_sync_batch=lambda batch, need_sync=None: batch,
            maybe_convert_decode_to_extend=lambda batch: batch,
        )
        s.ngram_embedding_manager = SimpleNamespace(
            prepare_for_forward=lambda batch, chunked_req=None: batch
        )
        s.tree_cache = SimpleNamespace(
            is_load_back_event_done=lambda consumer_index: fx.load_done
        )
        for name in (
            "_hold_prefill_batch_for_load",
            "_take_held_prefill_batch",
            "_arm_prefill_decode_interval",
        ):
            setattr(s, name, types.MethodType(getattr(Scheduler, name), s))
        return s

    def one_pass(self):
        """get_next_batch_to_run, then the loop tail of event_loop_overlap/normal."""
        s = self.sched
        plan = Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        s.running_batch = plan.running_batch
        batch = plan.batch_to_run
        # (run_batch / process_batch_result do not move P's or D's pages here)
        s.last_batch = batch
        return batch

    def busy_check(self):
        self.checker.self_check_during_busy()

    def fixed_busy_check_leak(self):
        """The proposed fix: also count the held batch's requests as uncached."""
        ps = self.checker.pool_stats_observer.get_pool_stats()
        full_uncached, _ = self.checker._get_total_uncached_sizes()
        held = self.sched.prefill_load_hold.held
        if held is not None:
            for req in held.reqs:
                allocated = -(-req.kv.kv_allocated_len // PAGE) * PAGE
                full_uncached += allocated - req.kv.cache_protected_len
        return self.checker._check_full_pool(ps, uncached=full_uncached)


class TestHeldBatchUnderBusyCheck(CustomTestCase):
    def setUp(self):
        super().setUp()
        for p in (
            patch.object(
                invariant_checker,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
            envs.SGLANG_CHECK_KV_PAGE_INVARIANTS.override(False),
        ):
            p.__enter__()
            self.addCleanup(p.__exit__, None, None, None)

    def test_hold_off_the_pass_that_builds_the_prefill_passes(self):
        fx = _Fixture(max_steps=0)
        batch = fx.one_pass()
        self.assertIs(batch, fx.prefill)
        self.assertIs(fx.sched.running_batch, fx.decode)
        fx.busy_check()  # no AssertionError

    def test_hold_on_the_pass_that_holds_and_every_held_pass_assert_a_leak(self):
        fx = _Fixture(max_steps=16)
        batch = fx.one_pass()
        self.assertIs(batch, fx.decode)
        self.assertIs(fx.sched.prefill_load_hold.held, fx.prefill)
        self.assertIs(fx.sched.last_batch, fx.sched.running_batch)
        # Only D's own page is counted; P's three extend pages are not.
        self.assertEqual(fx.checker._get_total_uncached_sizes(), (PAGE, 0))
        with self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak Detected") as cm:
            fx.busy_check()
        print("\nheld pass 1:", cm.exception)
        leak, msg = fx.fixed_busy_check_leak()
        self.assertFalse(leak, msg)

        # Next pass, copy still in flight: the batch stays held and the check fails again.
        batch = fx.one_pass()
        self.assertIs(batch, fx.decode)
        self.assertIs(fx.sched.prefill_load_hold.held, fx.prefill)
        with self.assertRaisesRegex(AssertionError, "Full Pool Mem Leak Detected"):
            fx.busy_check()
        self.assertFalse(fx.fixed_busy_check_leak()[0])

        # Copy landed: the batch is released, becomes last_batch and the check passes.
        fx.load_done = True
        batch = fx.one_pass()
        self.assertIs(batch, fx.prefill)
        self.assertIsNone(fx.sched.prefill_load_hold.held)
        fx.busy_check()

    def test_shortfall_is_exactly_the_held_batchs_page_aligned_extend(self):
        fx = _Fixture(max_steps=16)
        fx.one_pass()
        ps = fx.checker.pool_stats_observer.get_pool_stats()
        uncached, _ = fx.checker._get_total_uncached_sizes()
        accounted = (
            ps.full_available_size
            + ps.full_evictable_size
            + fx.cache.protected_size()
            + uncached
        )
        self.assertEqual(fx.allocator.size - accounted, fx.p_uncached)
        print(
            f"\ntotal={fx.allocator.size} accounted={accounted} "
            f"shortfall={fx.allocator.size - accounted} held extend (page-aligned)={fx.p_uncached}"
        )


if __name__ == "__main__":
    unittest.main()
