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
        "CONTROL (the engineer's M19) recompute priced on the seeded lags",
        P,
        "_recompute_quote",
        "            # On the load model's seeds a recompute would win at every size.\n"
        "            or not self._line_measured\n",
        "",
    ),
    (
        "B3a the gate's read time for a recompute quote leaves out the line ahead (A2)",
        P,
        "_recompute_quote",
        "read_s, _ = self._line_seconds(missing, deadline=None, backlog_pages=0, now=now)",
        "read_s = self._line_load.seconds(own_pages=math.ceil(missing / self._page_size), backlog_pages=0)",
    ),
    (
        "B3n the starved clock is not reset when a ready turn appears",
        P,
        "_relieve_starved_gpu",
        "        if not waiting_queue or self._meter.ready_since:\n"
        "            self._starved_since = None\n            return\n",
        "        if not waiting_queue or self._meter.ready_since:\n            return\n",
    ),
    (
        "B3s a starved turn whose read lands sooner lets the next turn be weighed (not one a pass)",
        P,
        "_relieve_starved_gpu",
        "                        parked=parked,\n                    )\n            return\n",
        "                        parked=parked,\n                    )\n                return\n",
    ),
    (
        "B3t the hold's landing quantile takes c for 1 (lateness at full weight)",
        P,
        "_hold_margin",
        "            lateness_weight=self._prices.c(now),\n",
        "            lateness_weight=1.0,\n",
    ),
    (
        "B3u horizon and just-in-time starts count the line while c is below 0.5 too",
        P,
        "_check_pending_deadline",
        "if self._read_line and self._prices.c(now) < READ_LINE_MIN_C:",
        "if False:",
    ),
    (
        "B3v the read line is counted for a staged start but the write hold is doubled",
        P,
        "_line_seconds",
        "            seconds += cache.read_lags.write_hold_left_s(now)\n",
        "            seconds += 2 * cache.read_lags.write_hold_left_s(now)\n",
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
