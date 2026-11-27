"""Which waiting requests the walk calls check_prefetch_progress on: HEAD vs pass-over (lookahead 3).

Under the timeout policy that call terminates a read past its timeout, under best_effort any read
in flight (unified_radix_cache.py check_prefetch_progress / _can_terminate_prefetch)."""
import sys
sys.path.insert(0, "/sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/review_sched/pass_over")
import diff_walk as d  # noqa: E402  (runs its own checks first)

queue = [d.FakeReq("big", 160, "normal"), d.FakeReq("a", 20, "normal"),
         d.FakeReq("b", 80, "normal"), d.FakeReq("c", 20, "normal")]
for name, fn, lookahead in (("HEAD", d.head_fn, 0), ("pass-over", d.new_fn, 3)):
    admitted, full, s = d.run(fn, queue, room=60, running=2, hierarchical=True, max_alloc=64, lookahead=lookahead)
    checked = [c.args[0] for c in s.tree_cache.check_prefetch_progress.call_args_list]
    print(f"{name:9s}: check_prefetch_progress on {checked}, admitted {admitted}, "
          f"left waiting {[r.rid for r in s.waiting_queue]}, batch_is_full={full}")
