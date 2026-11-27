"""Review check (default-off lens), throwaway: run the copy's own prefetcher tests against a
mutant of next_turn_prefetch.py kept in the review folder (the copy is not edited). The
mutant named by REVIEW_MUTANT appends a field to an existing HICACHE_EVT line with every
switch off; a green run means no in-repo test guards that line's fields."""

import importlib.util
import os
import sys

MUTANT = os.environ["REVIEW_MUTANT"]
NAME = "sglang.srt.mem_cache.next_turn_prefetch"

import sglang.srt.mem_cache as package

spec = importlib.util.spec_from_file_location(NAME, MUTANT)
module = importlib.util.module_from_spec(spec)
sys.modules[NAME] = module
spec.loader.exec_module(module)
package.next_turn_prefetch = module

from test_next_turn_prefetch import *  # noqa: E402,F401,F403


def test_the_mutant_is_what_the_tests_ran_on():
    import test_next_turn_prefetch

    assert test_next_turn_prefetch.NextTurnPrefetcher.__module__ == NAME
    assert sys.modules[NAME].__file__ == MUTANT
