"""Reviewer's throwaway check (style and tests lens): which one-line regressions do the unit
tests of the FETCH and PRICE change let through?

Each regression is applied in memory only: one function's source is re-compiled with a
snippet replaced and bound in place of the original for the run, then restored. No file of
the copy is edited. For every regression the five test modules of the change run in this
process; SURVIVED means no test failed.

Run (from the host, the copy's test folder holding a copy of this file):
  docker exec -e PYTHONPATH=<copy>/python -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 \
    sglang_hicache bash -c 'cd <copy> && nice -n 10 python3 -m pytest -q -s -p no:cacheprovider \
    test/registered/unit/mem_cache/zz_review_style_mutations.py 2>&1 | tail -80'
"""

import __future__

import contextlib
import inspect
import io
import logging
import sys
import unittest

import test_next_turn_estimators
import test_next_turn_prefetch
import test_next_use_eviction
import test_storage_read_dispatch
import test_storage_read_scheduler

from sglang.srt.mem_cache import next_turn_meter, next_turn_prefetch, next_turn_switches

TEST_MODULES = (
    test_next_turn_prefetch,
    test_next_turn_estimators,
    test_next_use_eviction,
    test_storage_read_dispatch,
    test_storage_read_scheduler,
)

P = next_turn_prefetch.NextTurnPrefetcher
M = next_turn_meter.PriceMeter
S = next_turn_switches.Switches

# (label, owner class, function name, snippet, replacement)
MUTATIONS = [
    (
        "CONTROL (the engineer's M12) staging priced by the frontier alone",
        P,
        "_staging_price",
        "return max(price, self._meter.next_victim_price(self._load, now))",
        "return price",
    ),
    (
        "G1 a staging ignores the frontier's price lam (victim price only)",
        P,
        "_staging_price",
        "return max(price, self._meter.next_victim_price(self._load, now))",
        "return self._meter.next_victim_price(self._load, now)",
    ),
    (
        "G2 the harm of a refused read keeps the constant per token (not the prefill fit)",
        P,
        "_follow_room",
        "self._meter.prefill_cost.seconds(\n"
        "                        new_tokens=lost_tokens, total_tokens=lost_tokens\n"
        "                    )",
        "(lost_tokens * RECOMPUTE_S_PER_TOKEN)",
    ),
    (
        "G5a a timed read's estimate never adds what is left of the write hold",
        P,
        "_line_seconds",
        "        if deadline is not None:\n"
        "            seconds += cache.read_lags.write_hold_left_s(now)\n",
        "",
    ),
    (
        "G5b a read needed now also waits for the write hold in the estimate",
        P,
        "_line_seconds",
        "        if deadline is not None:\n",
        "        if True:\n",
    ),
    (
        "G5c the starved-GPU rule never adds the write hold to a timed read's landing",
        P,
        "_relieve_starved_gpu",
        "            if self.read_deadline(handle) is not None:\n"
        "                land_s += cache.read_lags.write_hold_left_s(now)\n",
        "",
    ),
    (
        "G6 the queue check leaves the read line out (ntp_due backlog_pages, load_ms)",
        P,
        "_check_queued",
        "        load_s, backlog_pages = self._line_seconds(\n"
        "            plan.missing_tokens,\n"
        "            # As read_deadline() answers for this turn's own read.\n"
        "            deadline=None if position <= 1 else admission_at,\n"
        "            backlog_pages=backlog_pages,\n"
        "            now=now,\n"
        "        )\n",
        "        load_s = self._load_seconds(plan.missing_tokens, backlog_pages)\n",
    ),
    (
        "G7a a free room retry keeps an earlier lookup marked as charged",
        P,
        "prefetch_gate",
        "            self._take_room_retry(plan)\n            plan.lookup_charged = False\n",
        "            self._take_room_retry(plan)\n",
    ),
    (
        "G7b a room retry re-issued by the queue check is marked as a charged refetch",
        P,
        "on_refetch_result",
        "plan.refetch_charged = not decision.room_retry",
        "plan.refetch_charged = True",
    ),
    (
        "G9 a hit under the prefetch threshold counts as found (lookup stays charged)",
        P,
        "on_lookup_result",
        "found = hit_tokens >= self._tree_cache.prefetch_threshold",
        "found = hit_tokens > 0",
    ),
    (
        "G10a the prices line carries the hold counters as totals, not per-line deltas",
        M,
        "_read_path_fields",
        "**{name: count - before for name, count, before in zip(names, counts, logged)},",
        "**{name: count for name, count, before in zip(names, counts, logged)},",
    ),
    (
        "G10b the prices line drops the hold counters and the measured lags (A4)",
        M,
        "_price_fields",
        "            **self._read_path_fields(),\n",
        "",
    ),
    (
        "G12a the starved-GPU rule counts ready turns with the 0.5 s debounce (A3 says without)",
        P,
        "_relieve_starved_gpu",
        "if not waiting_queue or self._meter.ready_since:",
        "if not waiting_queue or self._prices.ready_waiting > 0:",
    ),
    (
        "G12b the queue check keeps re-issuing the prefetch of a turn that gave its read up",
        P,
        "_check_queued",
        "        if plan.recompute:\n            return False\n",
        "",
    ),
    (
        "G12c the starved-GPU rule acts on the seeded lags (before the lags were measured)",
        P,
        "_relieve_starved_gpu",
        "if now - self._starved_since < NTP_GPU_BOUND_S or not self._line_measured:",
        "if now - self._starved_since < NTP_GPU_BOUND_S:",
    ),
    (
        "G16 a landing adopts into a strategy that is not there (no return_chance guard)",
        P,
        "on_read_landed",
        "if plan is None or self._return_chance is None:",
        "if plan is None:",
    ),
    (
        "G17 landing times are kept for every request (table never emptied)",
        P,
        "on_read_landed",
        "if self._prices is not None and (plan is not None or handle.rid in self._queued):",
        "if self._prices is not None:",
    ),
    (
        "G18 queued turns with no plan get no predicted wait (their admissions feed nothing)",
        P,
        "tick",
        "if self._prices is not None and req.rid not in self._pred_wait:",
        "if False:",
    ),
    (
        "G19 a reader whose landing was not seen feeds the error of the GPU turn",
        M,
        "observe_admission",
        "        if not reader:\n",
        "        if not reader or landed_after is None:\n",
    ),
    (
        "G21 the next victim is priced once and never again",
        M,
        "next_victim_price",
        "if now - self._victim_priced_at >= VICTIM_PRICE_INTERVAL_S:",
        'if self._victim_priced_at == float("-inf"):',
    ),
    (
        "G22 measured prices accepted with --hicache-storage-next-turn-price-learning on",
        S,
        "_unmet",
        'if self.measured_prices and price_learning != "off":',
        'if self.measured_prices and price_learning == "harm":',
    ),
    (
        "G25 the board never gets the admission rate (mu_for loses the arrivals, A1)",
        P,
        "_follow_prices",
        "arrival_rate=0.0 if rate is None else rate.per_s,",
        "arrival_rate=0.0,",
    ),
    (
        "G28 a staged read's line counts only the reads needed now (expected_cost)",
        P,
        "_staged_start",
        "deadline=self.read_deadline(_handle(plan)) if self._read_line else None,",
        "deadline=None,",
    ),
    (
        "G39 an aborted request leaving the queue counts as an admission in c",
        P,
        "_on_departure",
        '                    "ntp_expire", req.rid, staged=int(plan.staged), reason="abort"\n'
        "                )\n"
        "            return 0\n",
        '                    "ntp_expire", req.rid, staged=int(plan.staged), reason="abort"\n'
        "                )\n"
        "            return 1\n",
    ),
    (
        "CONTROL2 the knob price stays the room price under measured prices",
        P,
        "_memory_price",
        "return self._prices.lam(now) if self._measured_prices else self._price.price",
        "return self._price.price",
    ),
    (
        "G41 the gate prices a recompute for the staged lookup's own call",
        P,
        "_recompute_quote",
        "            or plan.staging_call\n",
        "",
    ),
    (
        "G42 sessions waiting leaves the plans out (queued turns without a plan only)",
        P,
        "_follow_prices",
        "            sessions_waiting=len(self._plans)\n"
        "            + sum(req.rid not in self._plans for req in waiting_queue),",
        "            sessions_waiting=sum(req.rid not in self._plans for req in waiting_queue),",
    ),
]


def mutated(owner, name, old, new):
    """The function `name` of `owner` re-compiled with `old` replaced by `new`, in the
    globals of the module that defines it. Snippets carry the file's own indentation."""
    original = owner.__dict__[name]
    function = original.__func__ if isinstance(original, (staticmethod, classmethod)) else original
    source = inspect.getsource(function)
    if source.count(old) != 1:
        raise LookupError(f"snippet found {source.count(old)} times in {name}")
    source = source.replace(old, new)
    if source[0] in " \t":
        # A method's source is indented: give it a block to sit in.
        source = "if True:\n" + source
    code = compile(
        source,
        f"<mutated {name}>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    namespace = {}
    exec(code, sys.modules[owner.__module__].__dict__, namespace)
    made = namespace[name]
    if isinstance(original, staticmethod):
        made = staticmethod(made)
    elif isinstance(original, classmethod):
        made = classmethod(made)
    return original, made


def run_tests():
    """Ids of the tests (and subtests) that fail or error across the five modules."""
    failed = []
    out = io.StringIO()
    previous = logging.root.manager.disable
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        for module in TEST_MODULES:
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            result = unittest.TextTestRunner(stream=out, verbosity=0).run(suite)
            failed += [test.id() for test, _ in result.failures + result.errors]
    logging.disable(previous)
    return failed


def test_which_regressions_survive():
    clean = run_tests()
    lines = [f"clean run: {len(clean)} failing {clean[:3]}"]
    for label, owner, name, old, new in MUTATIONS:
        try:
            original, made = mutated(owner, name, old, new)
        except LookupError as error:
            lines.append(f"SKIP     {label}: {error}")
            continue
        setattr(owner, name, made)
        try:
            failed = run_tests()
        finally:
            setattr(owner, name, original)
        if failed:
            short = sorted({test_id.split(".", 1)[1].split(" ")[0] for test_id in failed})
            lines.append(f"KILLED   {label}: {len(failed)} failing, e.g. {short[0]}")
        else:
            lines.append(f"SURVIVED {label}")
    after = run_tests()
    lines.append(f"clean run after restoring: {len(after)} failing {after[:3]}")
    print("\n".join(["", "==== reviewer mutation check ====", *lines, "==== end ===="]))
    assert not clean and not after
