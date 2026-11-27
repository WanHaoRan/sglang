"""Mutation check of the unit tests added with the FETCH and PRICE changes: apply one small
regression to the copy (impl_unified_A), run the test file that should catch it inside the
container (CPU, niced, GPU hidden), restore the file. One line per mutation: CAUGHT (a test
failed) or MISSED. The copy is restored after every run, also when a run raises; do not edit
the copy while this runs.

Run from the host:  nice -n 10 python3 -I mutation_check.py [M7 M12 ...] > mutation_check.out"""

import subprocess
import sys

COPY = "/home/wanhr/sglang/agent_cache/frozen/impl_unified_A"
IN_CONTAINER = "/sgl-workspace/sglang/agent_cache/frozen/impl_unified_A"
MEM = "python/sglang/srt/mem_cache/"

MUTATIONS = [
    # (name, file, old, new, test file)
    ("M1 c ignores ready turns left", MEM + "next_turn_meter.py",
     "ready_left=ready_waiting > 0", "ready_left=False", "test_next_turn_prefetch"),
    ("M2 waiting leaves out unplanned queued turns", MEM + "next_turn_prefetch.py",
     "            sessions_waiting=len(self._plans)\n            + sum(req.rid not in self._plans for req in waiting_queue),",
     "            sessions_waiting=len(self._plans),", "test_next_turn_prefetch"),
    ("M3 late reads by deadline alone", MEM + "next_turn_meter.py",
     "late += deadline is None or deadline - own_s <= now", "late += deadline is None or deadline <= now",
     "test_next_turn_prefetch"),
    ("M4 frontier takes the lowest rate", MEM + "next_turn_meter.py",
     "rate > self._evicted_rate", "rate < self._evicted_rate", "test_next_turn_prefetch"),
    ("M5 prefill fit fed by queued turns", MEM + "next_turn_meter.py",
     "or not 0.0 <= waited < at_once_s", "or False", "test_next_turn_prefetch"),
    ("M6 readers feed the admission error", MEM + "next_turn_meter.py",
     "        if not reader:\n            self.admission_error.observe(predicted_wait_s=predicted, wait_s=wait)",
     "        if True:\n            self.admission_error.observe(predicted_wait_s=predicted, wait_s=wait)",
     "test_next_turn_prefetch"),
    ("M7 staged lookup sets the shared length", MEM + "next_turn_prefetch.py",
     "if not self._free_empty_lookups or plan.staging_call:", "if not self._free_empty_lookups:",
     "test_next_turn_prefetch"),
    ("M8 empty answer keeps the refetch charged", MEM + "next_turn_prefetch.py",
     "            if plan.refetch_charged:\n                plan.refetches -= 1\n                refunded = True\n",
     "", "test_next_turn_prefetch"),
    ("M9 gate gets the prompt cut to the known hit", MEM + "unified_radix_cache.py",
     "self.prefetch_gate_prompt_fn(request, storage_start, prompt_tail)",
     "self.prefetch_gate_prompt_fn(request, storage_start, new_input_tokens)", "test_storage_read_dispatch"),
    ("M10 terminated lookups are reported", MEM + "unified_radix_cache.py",
     "if self.prefetch_lookup_fn is not None and not operation.is_terminated():",
     "if self.prefetch_lookup_fn is not None:", "test_storage_read_dispatch"),
    ("M11 lateness weight never applied", MEM + "next_turn_prefetch.py",
     "stall_weight = self._prices.c(now) if self._lateness_weight else 1.0", "stall_weight = 1.0",
     "test_next_turn_prefetch"),
    ("M12 staging priced by the frontier alone", MEM + "next_turn_prefetch.py",
     "        return max(price, self._meter.next_victim_price(self._load, now))", "        return price",
     "test_next_turn_prefetch"),
    ("M13 line pages not counted", MEM + "next_turn_prefetch.py",
     "        ahead_pages = cache.storage_read_line_pages(deadline)\n", "        ahead_pages = backlog_pages\n",
     "test_next_turn_prefetch"),
    ("M14 just-in-time counts the line at saturation", MEM + "next_turn_prefetch.py",
     "if self._read_line and self._prices.c(now) < READ_LINE_MIN_C:", "if False:", "test_next_turn_prefetch"),
    ("M15 hold keeps the seeded lags", MEM + "next_turn_prefetch.py",
     "load_s = self._line_load.seconds(own_pages=read.pages, backlog_pages=ahead_pages)",
     "load_s = self._load.seconds(own_pages=read.pages, backlog_pages=ahead_pages)", "test_next_turn_prefetch"),
    ("M16 fixed lag charged the pages ahead", MEM + "unified_radix_cache.py",
     "pages = operation.ahead_pages + completed_tokens // self.page_size",
     "pages = completed_tokens // self.page_size", "test_storage_read_dispatch"),
    ("M17 hold margin takes the error's sign", MEM + "next_turn_prices.py",
     "return fallback_s if error is None else -error", "return fallback_s if error is None else error",
     "test_next_turn_prefetch"),
    ("M18 recompute priced behind another turn", MEM + "next_turn_prefetch.py",
     "if others > 0 or board.ready_waiting > 0:", "if False:", "test_next_turn_prefetch"),
    ("M19 recompute priced on the seeded lags", MEM + "next_turn_prefetch.py",
     "            # On the load model's seeds a recompute would win at every size.\n            or not self._line_measured\n",
     "", "test_next_turn_prefetch"),
    ("M20 starved rule fires with a ready turn", MEM + "next_turn_prefetch.py",
     "if not waiting_queue or self._meter.ready_since:", "if not waiting_queue:", "test_next_turn_prefetch"),
    ("M21 starved rule counts reads parked later", MEM + "next_turn_prefetch.py",
     "backlog_pages=cache.parked_read_pages_ahead(handle),",
     "backlog_pages=cache.storage_read_line_pages(self.read_deadline(handle)),", "test_next_turn_prefetch"),
    ("M22 gate forgets a starved turn recomputes", MEM + "next_turn_prefetch.py",
     "        if plan.recompute:\n            decision = \"recompute\"\n        elif chain - matched_tokens < threshold:",
     "        if chain - matched_tokens < threshold:", "test_next_turn_prefetch"),
    ("M23 a dispatched read is dropped", MEM + "unified_radix_cache.py",
     "        if self.parked_read_span(request) is None:\n            return False\n        self.revoke_pending_prefetch(request)",
     "        self.revoke_pending_prefetch(request)", "test_storage_read_dispatch"),
    ("M24 mu at saturation is the ready queue", MEM + "next_turn_prices.py",
     "        return max(light, float(self.sessions_waiting))", "        return light", "test_next_turn_estimators"),
    ("M25 measured prices decide in observe mode", MEM + "next_turn_switches.py",
     "acting and envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES.get()",
     "envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES.get()", "test_next_turn_prefetch"),
    ("M26 quote ignores arrivals during the recompute", MEM + "next_turn_prices.py",
     "return self._mu_light(now) + self.arrival_rate * job_s / 2.0", "return self._mu_light(now)",
     "test_next_turn_estimators"),
    ("M27 observer told after the eviction", MEM + "unified_cache/components/full.py",
     "            if observer is not None:\n                observer(x, len(x.component_data[ct].host_value))\n            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)",
     "            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)\n            if observer is not None:\n                observer(x, 0)",
     "test_next_use_eviction"),
    ("M28 queued victim priced as unknown", MEM + "next_use_eviction.py",
     "            return None if stale else queued\n        if next_use is None:",
     "            return None\n        if next_use is None:", "test_next_turn_prefetch"),
    ("M29 pages due by ignores the deadline", MEM + "storage_read_scheduler.py",
     "if due is None or (deadline is not None and due <= deadline):", "if True:", "test_storage_read_scheduler"),
    ("M30 absent mark ignored by the queue check", MEM + "next_turn_prefetch.py",
     "        return chain if plan.absent_from is None else min(chain, plan.absent_from)",
     "        return chain", "test_next_turn_prefetch"),
    ("M31 quote prices the key's tail the prompt lacks", MEM + "next_turn_prefetch.py",
     "        missing = min(missing, len(new_input_tokens))\n", "", "test_next_turn_prefetch"),
    ("M32 read line counts later deadlines for a staged start", MEM + "storage_read_scheduler.py",
     "if due is None or (deadline is not None and due <= deadline):",
     "if due is None or deadline is not None:", "test_next_turn_prefetch"),
    ("M33 next victim named from the other end of the order", MEM + "unified_cache/components/full.py",
     "        _, victim = min((self.session_ref_eviction_strategy(n), n) for n in leaves)",
     "        _, victim = max((self.session_ref_eviction_strategy(n), n) for n in leaves)",
     "test_next_use_eviction"),
]


def run(test_file):
    command = [
        "docker", "exec",
        "-e", f"PYTHONPATH={IN_CONTAINER}/python",
        "-e", "CUDA_VISIBLE_DEVICES=9",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "sglang_hicache", "bash", "-c",
        f"cd {IN_CONTAINER} && nice -n 10 python3 -m pytest -q -p no:cacheprovider -x "
        f"test/registered/unit/mem_cache/{test_file}.py 2>&1 | tail -3",
    ]
    return subprocess.run(command, capture_output=True, text=True).stdout


def main(selected):
    for name, rel, old, new, test_file in MUTATIONS:
        if selected and not any(name.startswith(tag + " ") for tag in selected):
            continue
        path = f"{COPY}/{rel}"
        original = open(path).read()
        if original.count(old) != 1:
            print(f"SKIP   {name}: snippet found {original.count(old)} times")
            continue
        try:
            open(path, "w").write(original.replace(old, new))
            tail = run(test_file)
        finally:
            open(path, "w").write(original)
        failed = " failed" in tail or "error" in tail.lower()
        summary = tail.strip().splitlines()[-1] if tail.strip() else "no output"
        print(f"{'CAUGHT' if failed else 'MISSED'} {name}: {summary}")


if __name__ == "__main__":
    main(sys.argv[1:])
