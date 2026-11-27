"""A refused ignore_eos candidate leaves its entry in PrefillAdder.req_states.

With the walk going on behind the refusal (pass-over), the next ignore_eos candidate is checked
against that entry too. Real PrefillAdder and PrefillBudget; the cache is a disabled (chunk)
cache, the only case add_one_req takes the ignore_eos path. CPU only.
"""

import sys
from types import SimpleNamespace

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.test.test_utils import maybe_stub_sgl_kernel  # noqa: E402

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Range  # noqa: E402
from sglang.srt.managers.schedule_policy import PrefillAdder  # noqa: E402
from sglang.srt.mem_cache.prefill_budget import PrefillBudget  # noqa: E402


class Alloc:
    page_size = 1

    def __init__(self, free):
        self.free = free

    def available_size(self):
        return self.free

    def create_prefill_budget(self, tree_cache, *, num_mixed_decode_tokens=0):
        return PrefillBudget(self, tree_cache, num_mixed_decode_tokens=num_mixed_decode_tokens)


class ChunkLikeCache:
    disable = True

    def supports_mamba(self):
        return False

    def evictable_size(self):
        return 0


class Req:
    def __init__(self, rid, input_len, max_new):
        self.rid = rid
        self.origin_input_ids = list(range(input_len))
        self.output_ids = []
        self.full_untruncated_fill_ids = self.origin_input_ids
        self.prefix_indices = []
        self.sampling_params = SimpleNamespace(ignore_eos=True, max_new_tokens=max_new)
        self.kv = SimpleNamespace(holds_mamba=False)
        self.retracted_stain = False

    def set_extend_range(self, start, end):
        self.extend_range = Range(start, end)


def adder():
    running = SimpleNamespace(reqs=[], batch_size=lambda: 0)
    return PrefillAdder(1, ChunkLikeCache(), Alloc(10000), running, 0.5, 16384, None)


big, small = Req("big", 4000, 12000), Req("small", 100, 100)
a = adder()
print("big   :", a.add_one_req(big, has_chunked_req=False, truncation_align_size=None))
print("req_states after the refusal:", a.req_states)
print("small, same pass (walk went on behind big):", a.add_one_req(small, has_chunked_req=False, truncation_align_size=None))
print("small, a pass without big's entry       :", adder().add_one_req(Req("small", 100, 100), has_chunked_req=False, truncation_align_size=None))
