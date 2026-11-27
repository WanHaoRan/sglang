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

from sglang.srt.mem_cache import next_turn_prices, next_use_eviction

NU = next_use_eviction.NextUseStrategy
RC = next_use_eviction.ReturnChanceStrategy
FIT = next_turn_prices.PrefillCostModel
BOARD = next_turn_prices.PriceBoard

# (label, owner class, function name, snippet, replacement)
MUTATIONS = [
    (
        "CONTROL (the engineer's M28) queued victim priced as unknown",
        NU,
        "holding_rate",
        "            return None if stale else queued\n",
        "            return None\n",
    ),
    (
        "Q1 next_use: a registration queued past QUEUED_MAX_AGE_S still gets the queued rate",
        NU,
        "holding_rate",
        "            return None if stale else queued\n",
        "            return queued\n",
    ),
    (
        "Q2 return_chance: a registration queued past QUEUED_MAX_AGE_S still gets the queued rate",
        RC,
        "holding_rate",
        "            return None if stale else queued\n",
        "            return queued\n",
    ),
    (
        "Q3 the prefill fit never forgets (every sample weighs the same for ever)",
        FIT,
        "observe",
        "        fade = self.forget\n",
        "        fade = 1.0\n",
    ),
    (
        "Q13 the next victim's price has no ceiling",
        M,
        "next_victim_price",
        "else min(LAM_CEILING, rate * self.restore_s_per_token(load, now))",
        "else rate * self.restore_s_per_token(load, now)",
    ),
    (
        "Q16 mu at light load drops the mean of ready turns (always 1)",
        BOARD,
        "_mu_light",
        "        return 1.0 + self.ready.value(\n"
        "            now=now, prior=float(self.ready_waiting), prior_weight=1e-3\n"
        "        )\n",
        "        return 1.0\n",
    ),
    (
        "Q17 the meter's ready count has no 0.5 s debounce (c falls on every arrival)",
        M,
        "follow",
        "            now - since >= self._ready_after_s for since in self.ready_since.values()\n",
        "            True for since in self.ready_since.values()\n",
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
