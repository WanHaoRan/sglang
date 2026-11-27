"""Mutation check for the cleaner re-check test, in this process only (never on disk):
argv[1] names the mutation; the keep_touched test then runs once."""

import inspect
import sys
import textwrap
import unittest

sys.path.insert(0, "test/registered/unit/mem_cache")

import test_hicache_nixl_cleaner as cleaner_tests  # noqa: E402

import sglang.srt.mem_cache.storage.nixl.nixl_cleaner as cleaner_module  # noqa: E402
from sglang.srt.mem_cache.storage.nixl.nixl_cleaner import HiCacheL3Cleaner  # noqa: E402

RECHECK = (
    "                if self.keep_touched:\n"
    "                    untouched = [g for g in batch if not _touched_since_scan(g)]\n"
    "                    kept_groups += len(batch) - len(untouched)\n"
    "                    batch = untouched\n"
    "                paths = list(self._iter_group_paths(batch))\n"
)
MUTATIONS = {
    # idx advanced by the groups left after the re-check, as before the change.
    "idx_after_filter": [
        ("                idx += len(batch)\n", ""),
        (
            "                deleted_groups += len(batch)\n",
            "                deleted_groups += len(batch)\n                idx += len(batch)\n",
        ),
    ],
    # Per-file re-check: the touched file is kept, the rest of its page deleted.
    "per_file": [
        (
            RECHECK,
            "                paths = [p for g in batch for p in sorted(g.paths)\n"
            "                         if not (self.keep_touched and os.path.exists(p)\n"
            "                                 and os.stat(p).st_mtime > g.mtime)]\n",
        )
    ],
}

name = sys.argv[1]
source = inspect.getsource(HiCacheL3Cleaner._tick)
for old, new in MUTATIONS[name]:
    assert source.count(old) == 1, (name, old)
    source = source.replace(old, new)
namespace = {}
exec(compile(textwrap.dedent(source), f"<mutant {name}>", "exec"),
     cleaner_module.__dict__, namespace)
HiCacheL3Cleaner._tick = namespace["_tick"]

suite = unittest.TestSuite(
    [
        cleaner_tests.TestHiCacheL3Cleaner(
            "test_a_group_touched_after_the_scan_is_kept_whole_only_under_keep_touched"
        )
    ]
)
result = unittest.TextTestRunner(verbosity=0).run(suite)
print(f"{name}: failures={len(result.failures)} errors={len(result.errors)}")
