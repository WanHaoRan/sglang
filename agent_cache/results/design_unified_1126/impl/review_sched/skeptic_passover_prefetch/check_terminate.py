"""Does the pass-over walk terminate a storage read that HEAD's walk leaves running?

Runs the real UnifiedRadixCache.check_prefetch_progress, _can_terminate_prefetch and
pop_prefetch_loaded_span on a bare cache object, inside the real admission walks of HEAD and
of the working tree, under each prefetch policy. _handle_prefetch_result is replaced by a
recorder that inserts what was read so far (its real body needs the radix tree).
CPU only; nothing in python/ is modified.
"""

import sys
import textwrap
import time
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.test.test_utils import maybe_stub_sgl_kernel  # noqa: E402

maybe_stub_sgl_kernel()

import torch  # noqa: E402

import sglang.srt.managers.scheduler as sched_mod  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.cache_controller import HiCacheController  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    AdmissionPassOver,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle  # noqa: E402
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (  # noqa: E402
    PrefetchOperation,
)
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import (  # noqa: E402
    UnifiedRadixCache,
    _OngoingPrefetch,
)

HERE = "/sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/review_sched/skeptic_passover_prefetch"
PAGE = 64

# ---- HEAD's walk, from `git show HEAD:python/sglang/srt/managers/scheduler.py` ----
lines = open(f"{HERE}/head_scheduler.py.txt").read().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith("    def _get_new_batch_prefill_raw("))
end = next(i for i, l in enumerate(lines[start + 1 :], start + 1) if l.startswith("    def "))
head_ns = dict(vars(sched_mod))
exec(compile(textwrap.dedent("".join(lines[start:end])), "head_walk", "exec"), head_ns)
WALKS = {"HEAD": head_ns["_get_new_batch_prefill_raw"],
         "tree": sched_mod.Scheduler._get_new_batch_prefill_raw}


class FakeAdder:
    """Token budget: a request fits when size < room; NO_TOKEN once room <= 0."""

    room = 0

    def __init__(self, *args, **kwargs):
        self.can_run_list, self.preempt_list = [], []
        self.new_chunked_req = self.rem_chunk_tokens = self.chunked_req_limit = None
        self.room = FakeAdder.room

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        if req.size >= self.room:
            return AddReqResult.NO_TOKEN
        self.room -= req.size
        self.can_run_list.append(req)
        return AddReqResult.NO_TOKEN if self.room <= 0 else AddReqResult.CONTINUE


class FakeReq:
    def __init__(self, rid, size):
        self.rid, self.size = rid, size
        self.beam_group = None
        self.kv = SimpleNamespace(holds_mamba=False, mamba_cow_src_index=None, mamba_needs_clear=False)
        self.cache_request_handle = CacheRequestHandle(rid=rid, attempt_id=0)
        self.time_stats = mock.MagicMock()
        self.storage_hit_length = 0

    def init_next_round_input(self, tree_cache):
        pass


def make_cache(policy, parked=frozenset()):
    """A bare UnifiedRadixCache with the fields check_prefetch_progress reads (one rank)."""
    c = UnifiedRadixCache.__new__(UnifiedRadixCache)
    c.cache_controller = SimpleNamespace(
        pp_prefetch_decisions={},
        terminate_prefetch=lambda op: HiCacheController.terminate_prefetch(None, op),
    )
    c.ongoing_prefetch = {}
    c.prefetch_stop_policy = policy
    c.read_scheduler = (SimpleNamespace(is_parked=lambda h: h.rid in parked) if parked else None)
    c.pp_rank, c.pp_size, c.pp_group = 0, 1, None
    c.tp_world_size, c.tp_group, c.attn_cp_group, c.attn_tp_group = 1, None, None, None
    # storage_attachment.apply_runtime_config with l3_extra.json's 1.0 and 0.25 per 1k tokens
    c.prefetch_timeout_base = 1.0
    c.prefetch_timeout_per_page = PAGE / 1024 * 0.25
    c.prefetch_loaded_tokens_by_reqid = {}
    c.prefetch_loaded_storage_start_by_reqid = {}
    c.storage_prefetch_retries = StoragePrefetchRetries()
    c.revoked = []
    c.revoke_pending_prefetch = lambda h: (c.revoked.append(h.rid), c.ongoing_prefetch.pop(h, None))
    c.inserted = {}

    def handle_result(op):  # stand-in for _handle_prefetch_result: insert what was read, drop the entry
        c.inserted[op.handle.rid] = (op.completed_tokens, len(op.hash_value) * PAGE)
        c.prefetch_loaded_tokens_by_reqid[op.handle] = op.completed_tokens
        del c.ongoing_prefetch[op.handle]

    c._handle_prefetch_result = handle_result
    return c


def add_read(cache, req, *, pages, done_pages, age_s, launched=True):
    op = PrefetchOperation(req.cache_request_handle, list(range(pages * PAGE)))
    op.hash_value = [f"{req.rid}{i}" for i in range(pages)]
    op.completed_tokens = done_pages * PAGE
    op.start_time = time.monotonic() - age_s
    op.host_indices = torch.arange(pages * PAGE) if launched else None
    cache.ongoing_prefetch[req.cache_request_handle] = _OngoingPrefetch(
        0, None, op.host_indices, op, None, {})
    return op


def make_sched(queue, cache, lookahead):
    s = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    s.grammar_manager = mock.MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.waiting_queue = list(queue)
    s.chunked_req = None
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda running_bs, beam_width=None, running_batch=None: 64
    s.policy = mock.MagicMock()
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 8192
    s.dynamic_chunk_sizer = None
    s.tp_worker = mock.MagicMock()
    s.tp_worker.model_runner.prefill_aware_swa = False
    s.page_size = PAGE
    s.tree_cache = mock.MagicMock()
    s.tree_cache.buffer_pipeline = None
    s.tree_cache.storage_prefetch_retries = None
    s.examined = []

    def check(handle):
        s.examined.append(handle.rid)
        return cache.check_prefetch_progress(handle)  # the real method

    s.tree_cache.check_prefetch_progress.side_effect = check
    s.tree_cache.pop_prefetch_loaded_span.side_effect = cache.pop_prefetch_loaded_span
    s.token_to_kv_pool_allocator = mock.MagicMock()
    s.new_token_ratio_tracker = mock.MagicMock()
    s.max_prefill_tokens = 16384
    s.is_mixed_chunk = False
    s.priority_scheduling_preemption_threshold = 0
    s.max_prefill_bs = 0
    s.max_running_requests = 64
    s.dllm_config = None
    s.enable_lora = False
    s.req_to_token_pool = SimpleNamespace()
    s.disaggregation_mode = DisaggregationMode.NULL
    s.enable_hicache_storage = True
    s._prefetch_after_device_hit_loss = lambda req: False
    s.truncation_align_size = None
    s.enable_hierarchical_cache = True
    s.enable_unified_cache_external_linker = False
    s.admission_pass_over = AdmissionPassOver(lookahead=lookahead, passes={})
    s.model_config = mock.MagicMock()
    s.enable_overlap = True
    s.spec_algorithm = mock.MagicMock()
    s.load_inquirer = mock.MagicMock()
    s.enable_priority_scheduling = False
    return s


class Running:
    def __init__(self, n):
        self.reqs = [object() for _ in range(n)]
        self.batch_is_full = False
        self.return_logprob = False

    def is_empty(self):
        return not self.reqs


def walk(name, policy, *, b_age, b_done, b_launched=True, parked=frozenset(), lookahead=3):
    queue = [FakeReq("big", 160), FakeReq("a", 20), FakeReq("b", 80), FakeReq("c", 20)]
    cache = make_cache(policy, parked)
    op = add_read(cache, queue[2], pages=64, done_pages=b_done, age_s=b_age, launched=b_launched)
    s = make_sched(queue, cache, lookahead if name == "tree" else 0)
    FakeAdder.room = 60
    fn = WALKS[name]
    patches = [
        mock.patch.dict(fn.__globals__, {
            "PrefillAdder": FakeAdder, "PrefillStats": mock.MagicMock(),
            "set_time_batch": lambda *a, **k: None,
            "get_schedule": lambda: SimpleNamespace(prefill_max_requests=None)}),
        mock.patch.object(sched_mod.ScheduleBatch, "init_new",
                          side_effect=lambda reqs, *a, **k: mock.MagicMock(reqs=reqs)),
    ]
    for p in patches:
        p.start()
    try:
        batch, _ = fn(s, prefill_delayer_single_pass=None, running_batch=Running(2))
    finally:
        for p in reversed(patches):
            p.stop()
    admitted = [] if batch is None else [r.rid for r in batch.reqs]
    still = "b" in [h.rid for h in cache.ongoing_prefetch]
    print(f"  {name:4s}: examined {s.examined}; admitted {admitted}; waiting "
          f"{[r.rid for r in s.waiting_queue]}; b read terminated={op.is_terminated()}, "
          f"still in flight={still}, b inserted (read, of)={cache.inserted.get('b')}, "
          f"b revoked={'b' in cache.revoked}, b miss retry armed={'b' in cache.storage_prefetch_retries._pending}")


timeout_s = 1.0 + 64 * PAGE / 1024 * 0.25
cases = [
    (f"timeout, b read 4096 tokens dispatched 10 s ago (timeout {timeout_s:.1f} s), 1024 read",
     "timeout", dict(b_age=10.0, b_done=16)),
    ("timeout, b read dispatched 0.5 s ago (inside its timeout), 256 read",
     "timeout", dict(b_age=0.5, b_done=4)),
    ("wait_complete, b read dispatched 10 s ago, 1024 read (the deployed arms)",
     "wait_complete", dict(b_age=10.0, b_done=16)),
    ("best_effort, b read just dispatched, nothing read yet",
     "best_effort", dict(b_age=0.01, b_done=0)),
    ("best_effort, b read parked by the read scheduler (no host pages yet)",
     "best_effort", dict(b_age=0.01, b_done=0, b_launched=False, parked=frozenset({"b"}))),
    ("timeout, b read parked by the read scheduler (no host pages yet)",
     "timeout", dict(b_age=10.0, b_done=0, b_launched=False, parked=frozenset({"b"}))),
]
for title, policy, kw in cases:
    print(title)
    for name in ("HEAD", "tree"):
        walk(name, policy, **kw)
