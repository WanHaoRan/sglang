"""CPU-only check: does a decode step run during a hold carry the running batch's stale
hicache_consumer_index, and would an eager forward then wait on the held copy's events?

Drives the real Scheduler.get_next_batch_to_run / _hold_prefill_batch_for_load /
_take_held_prefill_batch / update_running_batch, the real PrefillLoadHold, the real
LayerDoneCounter, TpModelWorker.set_hicache_consumer and MHATokenToKVPool.get_key_buffer.
Only the batches, the CUDA events/streams and the attention call are faked.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.managers.cache_controller as cc  # noqa: E402
from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)
from sglang.srt.managers.tp_worker import TpModelWorker  # noqa: E402
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402

NUM_LAYERS = 4


class FakeEvent:
    def __init__(self):
        self.recorded_by = None  # which load last recorded this event

    def record(self, *a, **k):
        self.recorded_by = FakeDevice.current_load

    def query(self):
        return True


class FakeStream:
    def __init__(self):
        self.waits = []

    def wait_event(self, ev):
        self.waits.append((ev, ev.recorded_by))


class FakeDevice:
    current_load = None
    forward_stream = FakeStream()

    @staticmethod
    def Event(*a, **k):
        return FakeEvent()

    @staticmethod
    def current_stream():
        return FakeDevice.forward_stream


class FakeBatch:
    def __init__(self, name, mode, consumer, load_tokens=0, n=1):
        self.name = name
        self.forward_mode = mode
        self.hicache_consumer_index = consumer
        self.reqs = [SimpleNamespace(host_loaded_length=load_tokens, finished=lambda: False) for _ in range(n)]
        self.chunked_req = None
        self.is_prefill_only = False
        self.batch_is_full = False

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return len(self.reqs) == 0

    def filter_batch(self, chunked_req_to_exclude=None, keep_indices=None):
        pass

    def merge_batch(self, other):
        self.reqs = self.reqs + other.reqs

    def check_decode_mem(self):
        return True

    def prepare_for_decode(self):
        # The real prepare_for_decode does not touch hicache_consumer_index.
        self.forward_mode = ForwardMode.DECODE

    def __repr__(self):
        return f"<{self.name} idx={self.hicache_consumer_index}>"


def make_scheduler(prefill_queue, load_done):
    s = Scheduler.__new__(Scheduler)
    s.scheduler_stage_metrics = None
    s.process_pending_chunked_abort = Mock()
    s.enable_hierarchical_cache = True
    s.enable_unified_cache_external_linker = False
    s.enable_hicache_storage = False
    s.tree_cache = SimpleNamespace(
        check_hicache_events=lambda: None,
        is_load_back_event_done=lambda idx: load_done(idx),
    )
    s.enable_fpm = False
    s.dllm_config = None
    s.chunked_req = None
    s.is_mixed_chunk = False
    s.enable_hisparse = False
    s.require_mlp_sync = False
    s.prefill_decode_interval = 0
    s._prefill_decode_interval_remaining = 0
    s.prefill_load_hold = PrefillLoadHold(max_steps=8)
    s.new_token_ratio_tracker = Mock()
    s.forward_ct = 1
    s.dp_attn_adapter = Mock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch.side_effect = lambda ret, need_sync=True: ret
    s.dp_attn_adapter.maybe_convert_decode_to_extend.side_effect = lambda ret: ret
    s.ngram_embedding_manager = Mock()
    s.ngram_embedding_manager.prepare_for_forward.side_effect = lambda ret, chunked_req=None: ret

    def get_new_batch_prefill(running_batch):
        nb = prefill_queue.pop(0) if prefill_queue else None
        return NextBatchPlan(batch_to_run=nb, running_batch=running_batch)

    s.get_new_batch_prefill = get_new_batch_prefill
    return s


class TestStaleConsumer(unittest.TestCase):
    def test_decode_during_hold_waits_on_the_held_copy_in_eager_mode(self):
        with patch.object(cc, "device_module", FakeDevice):
            counter = cc.LayerDoneCounter(NUM_LAYERS)

            def submit_load(tag):
                # cache_controller.start_loading: next slot, then layer events recorded at submit.
                slot = counter.update_producer()
                FakeDevice.current_load = tag
                for layer in range(NUM_LAYERS):
                    counter.events[slot].complete(layer)
                FakeDevice.current_load = None
                return slot

            # Load A: the prefill that will refill an empty running batch.
            slot_a = submit_load("A")
            p_a = FakeBatch("P_A", ForwardMode.EXTEND, slot_a, load_tokens=100)
            # Loads B and C ran as prefills that merged into the running batch.
            slot_b = submit_load("B")
            slot_c = submit_load("C")

            prefills = []
            sched = make_scheduler(prefills, load_done=lambda idx: False)
            empty = FakeBatch("EMPTY", ForwardMode.DECODE, -1, n=0)

            # Step 1: P_A ran last; running batch empty -> running_batch = last_batch.
            plan = sched.get_next_batch_to_run(running_batch=empty, last_batch=p_a)
            running = plan.running_batch
            self.assertIs(running, p_a)
            self.assertIs(plan.batch_to_run, p_a)
            self.assertTrue(plan.batch_to_run.forward_mode.is_decode())
            # B and C merge in later (merge_batch leaves the index alone).
            p_b = FakeBatch("P_B", ForwardMode.EXTEND, slot_b, load_tokens=100)
            plan = sched.get_next_batch_to_run(running_batch=running, last_batch=p_b)
            running = plan.running_batch
            p_c = FakeBatch("P_C", ForwardMode.EXTEND, slot_c, load_tokens=100)
            plan = sched.get_next_batch_to_run(running_batch=running, last_batch=p_c)
            running = plan.running_batch
            self.assertIs(running, p_a)
            self.assertEqual(running.hicache_consumer_index, slot_a)

            # Load D (big) takes the next slot, which is slot_a again; it gets held.
            slot_d = submit_load("D")
            self.assertEqual(slot_d, slot_a)
            p_d = FakeBatch("P_D", ForwardMode.EXTEND, slot_d, load_tokens=MIN_LOAD_TOKENS)
            prefills.append(p_d)
            last = plan.batch_to_run
            plan = sched.get_next_batch_to_run(running_batch=running, last_batch=last)
            self.assertIs(sched.prefill_load_hold.held, p_d)
            decode = plan.batch_to_run
            self.assertIs(decode, p_a)
            self.assertTrue(decode.forward_mode.is_decode())
            print("decode batch run beside the held copy:", decode, "held:", p_d)

            # tp_worker.forward_batch_generation: set_hicache_consumer(batch.hicache_consumer_index)
            worker = TpModelWorker.__new__(TpModelWorker)
            worker.hicache_layer_transfer_counter = counter
            worker.set_hicache_consumer(decode.hicache_consumer_index)
            self.assertEqual(counter.consumer_index, slot_d)

            # Eager attention: FA3 forward_decode -> get_kv_buffer -> get_key_buffer per layer.
            pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
            pool.layer_transfer_counter = counter
            pool.start_layer = 0
            pool._get_key_buffer = lambda layer_id: None
            FakeDevice.forward_stream.waits.clear()
            for layer in range(NUM_LAYERS):
                pool.get_key_buffer(layer)
            waited_on = [tag for _, tag in FakeDevice.forward_stream.waits]
            print("eager decode layer waits recorded by load:", waited_on)
            self.assertEqual(waited_on, ["D"] * NUM_LAYERS)
            held_events = counter.events[slot_d].load_events
            self.assertEqual(
                [ev for ev, _ in FakeDevice.forward_stream.waits], held_events
            )

            # Next step: the held batch stays held (copy not landed) and decode runs again
            # with the same stale index.
            plan = sched.get_next_batch_to_run(running_batch=plan.running_batch, last_batch=decode)
            self.assertIs(plan.batch_to_run, p_a)
            self.assertEqual(plan.batch_to_run.hicache_consumer_index, slot_d)

            # Control: with the index reset to -1, the same eager layer reads wait on nothing.
            worker.set_hicache_consumer(-1)
            FakeDevice.forward_stream.waits.clear()
            for layer in range(NUM_LAYERS):
                pool.get_key_buffer(layer)
            self.assertEqual(FakeDevice.forward_stream.waits, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
