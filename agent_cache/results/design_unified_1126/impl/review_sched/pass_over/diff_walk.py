"""Differential check of Scheduler._get_new_batch_prefill_raw: HEAD vs the working tree.

Runs both versions of the admission walk (HEAD's extracted from `git show HEAD:...scheduler.py`,
saved next to this file) against a scripted PrefillAdder, on random queues, and compares what
each admits and the batch_is_full it leaves. CPU only; nothing in python/ is modified.
"""

import random
import sys
import textwrap
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.test.test_utils import maybe_stub_sgl_kernel  # noqa: E402

maybe_stub_sgl_kernel()

import sglang.srt.managers.scheduler as sched_mod  # noqa: E402
from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    AdmissionPassOver,
)

HERE = "/sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/review_sched/pass_over"

# ---- HEAD's walk, extracted and compiled against the same module globals ----
lines = open(f"{HERE}/head_scheduler.py.txt").read().splitlines(keepends=True)
start = next(i for i, l in enumerate(lines) if l.startswith("    def _get_new_batch_prefill_raw("))
end = next(i for i, l in enumerate(lines) if l.startswith("    def can_schedule_lora_req("))
head_src = textwrap.dedent("".join(lines[start:end]))
head_ns = dict(vars(sched_mod))
exec(compile(head_src, "head_get_new_batch_prefill_raw", "exec"), head_ns)
head_fn = head_ns["_get_new_batch_prefill_raw"]
new_fn = sched_mod.Scheduler._get_new_batch_prefill_raw


class FakeAdder:
    """Room is a token budget: a request fits when size < room (check_prefill's strict <);
    after a commit, NO_TOKEN once room <= 0 (budget_state)."""

    room = 0

    def __init__(self, *args, **kwargs):
        self.can_run_list = []
        self.preempt_list = []
        self.new_chunked_req = None
        self.rem_chunk_tokens = None
        self.chunked_req_limit = None
        self.room = FakeAdder.room

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        if req.kind == "other":
            return AddReqResult.OTHER
        if req.size >= self.room:
            return AddReqResult.NO_TOKEN
        self.room -= req.size
        self.can_run_list.append(req)
        return AddReqResult.NO_TOKEN if self.room <= 0 else AddReqResult.CONTINUE


class FakeReq:
    def __init__(self, rid, size, kind):
        self.rid, self.size, self.kind = rid, size, kind
        self.beam_group = None
        self.kv = SimpleNamespace(
            holds_mamba=False, mamba_cow_src_index=None, mamba_needs_clear=False
        )
        self.cache_request_handle = rid
        self.time_stats = mock.MagicMock()
        self.storage_hit_length = 0

    def init_next_round_input(self, tree_cache):
        pass

    def __repr__(self):
        return f"{self.rid}:{self.kind}{self.size}"


class FakeRunning:
    def __init__(self, n):
        self.reqs = [object() for _ in range(n)]
        self.batch_is_full = False
        self.return_logprob = False

    def is_empty(self):
        return not self.reqs


def make_sched(queue, *, hierarchical, max_alloc, lookahead):
    s = sched_mod.Scheduler.__new__(sched_mod.Scheduler)
    s.grammar_manager = mock.MagicMock()
    s.grammar_manager.has_waiting_grammars.return_value = False
    s.enable_priority_preemption = False
    s.is_hybrid_swa = False
    s.waiting_queue = list(queue)
    s.chunked_req = None
    s.min_free_slots_delayer = None
    s.get_num_allocatable_reqs = lambda running_bs, beam_width=None, running_batch=None: max_alloc
    s.policy = mock.MagicMock()
    s.processed_tokens_counter = 0
    s.chunked_prefill_size = 8192
    s.dynamic_chunk_sizer = None
    s.tp_worker = mock.MagicMock()
    s.tp_worker.model_runner.prefill_aware_swa = False
    s.page_size = 64
    s.tree_cache = mock.MagicMock()
    s.tree_cache.buffer_pipeline = None
    s.tree_cache.storage_prefetch_retries = None
    s.tree_cache.check_prefetch_progress.side_effect = lambda h: not h.startswith("skip")
    s.tree_cache.pop_prefetch_loaded_span.return_value = (0, None)
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
    s.enable_hierarchical_cache = hierarchical
    s.enable_unified_cache_external_linker = False
    s.admission_pass_over = AdmissionPassOver(lookahead=lookahead)
    s.model_config = mock.MagicMock()
    s.enable_overlap = True
    s.spec_algorithm = mock.MagicMock()
    s.load_inquirer = mock.MagicMock()
    s.enable_priority_scheduling = False
    return s


def run(fn, queue, *, room, running, hierarchical, max_alloc, lookahead, passes=None):
    FakeAdder.room = room
    s = make_sched(queue, hierarchical=hierarchical, max_alloc=max_alloc, lookahead=lookahead)
    if passes:
        s.admission_pass_over.passes.update(passes)
    rb = FakeRunning(running)
    patches = [
        mock.patch.dict(fn.__globals__, {"PrefillAdder": FakeAdder,
                                          "PrefillStats": mock.MagicMock(),
                                          "set_time_batch": lambda *a, **k: None,
                                          "get_schedule": lambda: SimpleNamespace(prefill_max_requests=None)}),
        mock.patch.object(sched_mod.ScheduleBatch, "init_new", side_effect=lambda reqs, *a, **k: mock.MagicMock(reqs=reqs)),
    ]
    for p in patches:
        p.start()
    try:
        batch, rb_out = fn(s, prefill_delayer_single_pass=None, running_batch=rb)
    finally:
        for p in reversed(patches):
            p.stop()
    admitted = [] if batch is None else [r.rid for r in batch.reqs]
    return admitted, rb_out.batch_is_full, s


def random_queue(rng, n):
    q = []
    for i in range(n):
        kind = rng.choices(["normal", "other", "skip"], weights=[8, 1, 1])[0]
        rid = f"skip{i}" if kind == "skip" else f"r{i}"
        q.append(FakeReq(rid, rng.choice([5, 10, 20, 40, 80, 160]), "normal" if kind == "skip" else kind))
    return q


rng = random.Random(0)
off_mismatch = 0
cases = 0
diverge = {}
for trial in range(4000):
    n = rng.randint(1, 10)
    queue = random_queue(rng, n)
    kw = dict(
        room=rng.choice([1, 30, 60, 100, 200, 400]),
        running=rng.choice([0, 0, 2]),
        hierarchical=rng.choice([True, False]),
        max_alloc=rng.choice([1, 2, 3, 64]),
    )
    head = run(head_fn, queue, lookahead=0, **kw)
    off = run(new_fn, queue, lookahead=0, **kw)
    cases += 1
    if head[:2] != off[:2] or [r.rid for r in head[2].waiting_queue] != [r.rid for r in off[2].waiting_queue]:
        off_mismatch += 1
        if off_mismatch <= 3:
            print("OFF MISMATCH", queue, kw, head[:2], off[:2])
    for lookahead in (1, 3):
        on = run(new_fn, queue, lookahead=lookahead, **kw)
        if on[1] != head[1]:
            key = (
                f"hier={kw['hierarchical']} running={kw['running'] > 0} "
                f"head_admitted={len(head[0]) > 0} on_admitted={len(on[0]) > 0}: "
                f"head_full={head[1]} on_full={on[1]}"
            )
            if key not in diverge:
                diverge[key] = (queue, kw, lookahead, head[0], on[0])
print(f"switch off: {cases} random walks, {off_mismatch} differ from HEAD (admitted, batch_is_full, queue)")
print("switch on, batch_is_full different from HEAD, by kind:")
for key, ex in sorted(diverge.items()):
    print("  ", key, "  e.g.", ex)

# Pass-over only appends behind the first refusal: HEAD's admitted list is a prefix of it.
rng = random.Random(1)
not_prefix = 0
more = 0
for trial in range(3000):
    queue = random_queue(rng, rng.randint(1, 10))
    kw = dict(room=rng.choice([30, 60, 100, 200]), running=rng.choice([0, 2]),
              hierarchical=True, max_alloc=rng.choice([1, 2, 3, 64]))
    head = run(head_fn, queue, lookahead=0, **kw)[0]
    on = run(new_fn, queue, lookahead=3, **kw)[0]
    if on[: len(head)] != head:
        not_prefix += 1
    more += len(on) > len(head)
    if len(on) > kw["max_alloc"]:
        print("over max_alloc", queue, kw, on)
print(f"switch on (lookahead 3): HEAD's admitted list not a prefix in {not_prefix} of 3000; more admitted in {more}")
