"""Feasibility probe (not for the tree): the load hold's wiring in get_next_batch_to_run
on a bare Scheduler, built the way test_scheduler_chunked_req_gate.py builds one."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)


def _bare_scheduler(*, prefill_batch, load_done):
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s.disaggregation_mode = DisaggregationMode.NULL
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    s.enable_hierarchical_cache = True
    s.enable_hicache_storage = False
    s.enable_unified_cache_external_linker = False
    s.is_mixed_chunk = False
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.chunked_req = None
    s._pending_chunked_abort_req = None
    s.prefill_decode_interval = 0
    s._prefill_decode_interval_remaining = 0
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
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.dp_attn_adapter.maybe_convert_decode_to_extend = MagicMock(
        side_effect=lambda batch: batch
    )
    s.ngram_embedding_manager = MagicMock()
    s.ngram_embedding_manager.prepare_for_forward = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    s.tree_cache = MagicMock()
    s.tree_cache.is_load_back_event_done = MagicMock(side_effect=lambda i: load_done[0])
    # Per instance: never tune the class-level default.
    s.prefill_load_hold = PrefillLoadHold(max_steps=4)
    return s


def _loading_batch(tokens=MIN_LOAD_TOKENS, consumer_index=1):
    batch = MagicMock()
    batch.hicache_consumer_index = consumer_index
    batch.reqs = [SimpleNamespace(host_loaded_length=tokens)]
    batch.forward_mode.is_extend.return_value = True
    return batch


def _step(s):
    return Scheduler.get_next_batch_to_run(
        s, running_batch=s.running_batch, last_batch=None
    ).batch_to_run


class TestHoldWiring(CustomTestCase):
    def test_held_batch_decodes_beside_its_load_and_admits_nothing_meanwhile(self):
        batch, load_done = _loading_batch(), [False]
        s = _bare_scheduler(prefill_batch=batch, load_done=load_done)
        self.assertIs(_step(s), s.running_batch)  # held: the running batch decodes
        self.assertIs(_step(s), s.running_batch)  # still loading
        self.assertEqual(s.get_new_batch_prefill.call_count, 1)  # no admission
        load_done[0] = True
        self.assertIs(_step(s), batch)
        s.tree_cache.is_load_back_event_done.assert_called_with(1)
        self.assertIsNone(s.prefill_load_hold.held)

    def test_a_batch_that_starts_a_chunked_prefill_is_not_held(self):
        batch, load_done = _loading_batch(), [False]
        s = _bare_scheduler(prefill_batch=batch, load_done=load_done)
        plan = s.get_new_batch_prefill.return_value

        def admit_and_chunk(running_batch):
            s.chunked_req = MagicMock()  # adder.new_chunked_req became chunked_req
            return plan

        s.get_new_batch_prefill = MagicMock(side_effect=admit_and_chunk)
        self.assertIs(_step(s), batch)
        self.assertIsNone(s.prefill_load_hold.held)

    def test_a_mixed_batch_is_not_held_since_nothing_is_left_to_decode(self):
        # _get_new_batch_prefill_raw hands back an empty running batch once it mixes.
        batch = _loading_batch()
        s = _bare_scheduler(prefill_batch=batch, load_done=[False])
        s.is_mixed_chunk = True
        empty = MagicMock()
        empty.is_empty.return_value = True
        empty.is_prefill_only = False
        empty.reqs = []
        s.get_new_batch_prefill = MagicMock(
            return_value=NextBatchPlan(batch_to_run=batch, running_batch=empty)
        )
        self.assertIs(_step(s), batch)
        self.assertIsNone(s.prefill_load_hold.held)

    def test_mixed_chunk_on_never_holds_even_an_unmixed_batch(self):
        # Documents current behaviour: a batch left unmixed (logprob, input_embeds,
        # beam rows) beside a decoding batch is still never held under mixed chunk.
        batch = _loading_batch()
        s = _bare_scheduler(prefill_batch=batch, load_done=[False])
        s.is_mixed_chunk = True
        self.assertIs(_step(s), batch)

    def test_no_hold_without_a_host_load(self):
        for batch in (_loading_batch(consumer_index=-1), _loading_batch(tokens=1)):
            s = _bare_scheduler(prefill_batch=batch, load_done=[False])
            self.assertIs(_step(s), batch)


if __name__ == "__main__":
    unittest.main()
