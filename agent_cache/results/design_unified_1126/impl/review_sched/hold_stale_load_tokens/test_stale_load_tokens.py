"""Scratch check (review lens: hold): does a stale host_loaded_length hold a batch whose
own copy is below MIN_LOAD_TOKENS? Drives the working-tree PrefillAdder and
Scheduler._hold_prefill_batch_for_load; nothing under python/ or test/ is changed."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.schedule_policy import PrefillAdder  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)
from sglang.srt.mem_cache.base_prefix_cache import (  # noqa: E402
    CacheRequestHandle,
    DecLockRefResult,
    IncLockRefResult,
)
from sglang.srt.mem_cache.prefill_budget import PrefillBudget  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.srt.utils.common import Range  # noqa: E402

PAGE = 64
CHUNK = 8192  # --chunked-prefill-size of every campaign


class StaleLoadTokens(CustomTestCase):
    def setUp(self):
        print("sglang from", sglang.__file__)
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        tree = MagicMock()
        tree.full_evictable_size.return_value = 0
        tree.swa_evictable_size.return_value = 0
        tree.evictable_size.return_value = 0
        tree.disable = False
        tree.inc_lock_ref.return_value = IncLockRefResult()
        tree.dec_lock_ref.return_value = DecLockRefResult()
        tree.buffer_pipeline = None
        tree.is_tree_cache.return_value = False
        tree.supports_mamba.return_value = False
        self.copies = []  # tokens queued for H2D in the current walk (load_queue)

        def init_load_back(params):
            n = params.host_hit_length
            self.copies.append(n)
            return torch.arange(n, dtype=torch.int64), params.req.last_node

        tree.init_load_back.side_effect = init_load_back
        self.tree = tree
        alloc = MagicMock()
        alloc.full_available_size.return_value = 1_000_000
        alloc.swa_available_size.return_value = 1_000_000
        alloc.available_size.return_value = 1_000_000
        alloc.size_swa = 1_000_000
        alloc.swa_req_ring = False
        alloc.page_size = PAGE
        alloc.create_prefill_budget.side_effect = lambda tree_cache, **kw: (
            PrefillBudget(alloc, tree_cache, **kw)
        )
        self.alloc = alloc

    def _adder(self):
        running = MagicMock()
        running.reqs = []
        return PrefillAdder(
            page_size=PAGE,
            tree_cache=self.tree,
            token_to_kv_pool_allocator=self.alloc,
            running_batch=running,
            new_token_ratio=1.0,
            rem_input_tokens=16384,  # max_prefill_tokens of the campaigns
            rem_chunk_tokens=CHUNK,
            num_mixed_decode_tokens=0,
            priority_scheduling_preemption_threshold=0,
        )

    def _req(self, rid, *, host_hit, new_tokens):
        req = MagicMock(spec=Req)
        req.rid = rid
        req.cache_request_handle = CacheRequestHandle(rid, 0)
        req.priority = 0
        req.prefix_indices = torch.empty(0, dtype=torch.int64)
        req.full_untruncated_fill_ids = list(range(host_hit + new_tokens))
        req.output_ids = []
        req.sampling_params = SimpleNamespace(max_new_tokens=8, ignore_eos=False)
        req.time_stats = SimpleNamespace(wait_queue_entry_time=0)
        req.retracted_stain = False
        req.host_hit_length = host_hit
        req.swa_host_hit_length = 0
        req.mamba_host_hit_length = 0
        req.storage_hit_length = 0
        req.storage_hit_start = None
        req.host_hit_is_storage = False
        req.host_loaded_length = 0
        req.needs_host_load_back.return_value = host_hit > 0
        req.materialized_host_hit_len.side_effect = lambda: min(
            req.host_hit_length, req.host_loaded_length
        )
        req.fulfilled_storage_hit_len.return_value = 0
        req.finished.return_value = False
        req.last_node = MagicMock()
        req.best_match_node = req.last_node
        req.kv = SimpleNamespace(cache_protected_len=0)
        req.set_extend_range = MagicMock(
            side_effect=lambda s, e: setattr(req, "extend_range", Range(s, e))
        )
        return req

    def _hold(self, reqs):
        sched = Scheduler.__new__(Scheduler)
        sched.prefill_load_hold = PrefillLoadHold(max_steps=16)
        sched.chunked_req = None  # the last chunk finished the chunked request
        sched.is_mixed_chunk = False  # --enable-mixed-chunk is off in every campaign
        batch = SimpleNamespace(reqs=reqs, hicache_consumer_index=0)
        running = SimpleNamespace(is_empty=lambda: False, is_prefill_only=False)
        return sched, sched._hold_prefill_batch_for_load(batch, running), batch

    def test_last_chunk_next_to_a_small_load_is_held(self):
        # Pass 1: C loads 30,016 host tokens and is chunked (12,032 new > 8,192).
        c = self._req("C", host_hit=30016, new_tokens=12032)
        adder = self._adder()
        adder.add_one_req(c, has_chunked_req=False, truncation_align_size=None)
        self.assertIs(adder.new_chunked_req, c)
        self.assertEqual(c.host_loaded_length, 30016)
        # (the scheduler's chunked=True keeps this batch from being held)
        # The first chunk ran: its KV is now part of C's prefix.
        c.prefix_indices = torch.arange(30016 + CHUNK)

        # Pass 2: C's last chunk via add_chunked_req, then X with a 2,048-token load.
        self.copies.clear()  # start_loading merged and cleared the queue
        adder = self._adder()
        self.assertIsNone(adder.add_chunked_req(c))  # last chunk
        x = self._req("X", host_hit=2048, new_tokens=256)
        adder.add_one_req(x, has_chunked_req=False, truncation_align_size=None)
        self.assertEqual(adder.can_run_list, [c, x])
        self.assertIsNone(adder.new_chunked_req)
        print(
            "pass 2: copy queued =", sum(self.copies), "tokens; host_loaded_length:",
            {r.rid: r.host_loaded_length for r in adder.can_run_list},
        )
        self.assertLess(sum(self.copies), MIN_LOAD_TOKENS)
        self.assertEqual(c.host_loaded_length, 30016)  # stale, never reset

        sched, out, batch = self._hold(list(adder.can_run_list))
        print("held:", out is None)
        self.assertIsNone(out)
        self.assertIs(sched.prefill_load_hold.held, batch)

        # Control: the same 2,048-token load without C's continuation is not held.
        self.copies.clear()
        y = self._req("Y", host_hit=2048, new_tokens=256)
        adder = self._adder()
        adder.add_one_req(y, has_chunked_req=False, truncation_align_size=None)
        _, out, batch = self._hold(list(adder.can_run_list))
        self.assertIs(out, batch)

    def test_real_req_keeps_host_loaded_length_across_continuation_and_retract(self):
        sp = SamplingParams(max_new_tokens=8)
        sp.normalize(None)
        req = Req(
            rid="r",
            origin_input_text="",
            origin_input_ids=array("q", range(64)),
            sampling_params=sp,
            vocab_size=128,
        )
        req.host_loaded_length = 30016  # set by add_one_req at admission
        req.init_next_round_input()  # chunked continuation: no tree cache, no re-match
        self.assertEqual(req.host_loaded_length, 30016)
        req.reset_for_retract()
        self.assertEqual(req.host_loaded_length, 30016)
        print("real Req after continuation and retract:", req.host_loaded_length)


if __name__ == "__main__":
    unittest.main()
