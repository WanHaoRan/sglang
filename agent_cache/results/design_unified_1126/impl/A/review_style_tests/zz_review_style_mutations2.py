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
        "N2 a refunded lookup leaves the turn capped (refetch_capped not cleared)",
        P,
        "on_lookup_result",
        "                plan.free_lookups += 1\n                plan.refetch_capped = False\n",
        "                plan.free_lookups += 1\n",
    ),
    (
        "N4 the shared length is computed even when the match is past the key (IndexError)",
        P,
        "_history_tokens",
        "            and new_input_tokens is not None\n            and matched_tokens < len(key)\n",
        "            and new_input_tokens is not None\n",
    ),
    (
        "X3 the ntp_lookup line is not logged",
        P,
        "on_lookup_result",
        '        _log_event(\n            "ntp_lookup",\n            handle.rid,\n'
        "            start=storage_start,\n            asked=requested_tokens,\n"
        "            hit=hit_tokens,\n            refunded=int(refunded),\n"
        "            lookups=plan.lookups,\n        )\n",
        "        pass\n",
    ),
    (
        "G43 the prices line carries no mu (A1: log both mu= and waiting=)",
        M,
        "_price_fields",
        '            mu=f"{board.mu(now):.2f}",\n',
        "",
    ),
    (
        "G44 the prices line carries no kappa",
        M,
        "_price_fields",
        '            kappa=f"{board.kappa(now):.3f}",\n',
        "",
    ),
    (
        "G45 ntp_due logs the launched pages, not the line, for a just-in-time start",
        P,
        "_check_pending_deadline",
        "            load_s, backlog_pages = self._line_seconds(",
        "            load_s, _ = self._line_seconds(",
    ),
    (
        "G46 the staged lookup's gate call stays marked after the call (staging_call not reset)",
        P,
        "_stage",
        "            plan.staging_call = False\n",
        "",
    ),
    (
        "G47 the harm of a refused read is priced at the queue count, not mu",
        P,
        "_follow_room",
        "harm_s = self._prices.mu(now) * self._meter.prefill_cost.seconds(",
        "harm_s = max(1, queue_ahead) * self._meter.prefill_cost.seconds(",
    ),
    (
        "G48 a retracted request's negative wait feeds the admission error",
        M,
        "observe_admission",
        "if predicted is None or wait < 0.0:",
        "if predicted is None:",
    ),
    (
        "G49 an unpriced victim (no holding rate) resets the frontier's highest rate",
        M,
        "on_host_evicted",
        "if rate is not None and (self._evicted_rate is None or rate > self._evicted_rate):",
        "if self._evicted_rate is None or (rate is not None and rate > self._evicted_rate):",
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
