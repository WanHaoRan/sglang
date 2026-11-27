"""Fixer's variant of the reviewer's probe f4 (rc_retire_heal.py): the resolver now grants the retire only with
a touch interval, and with one the page read back at mtime zero is refreshed at its next request match."""

from array import array
from types import SimpleNamespace

from test_next_use_eviction import (
    computed_node,
    read_back_node,
    recording_store,
    stored_marks_cache,
    stored_node,
)

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.storage_retention import (
    RetentionSwitches,
    resolve_retention_switches,
)
from sglang.test.test_utils import CustomTestCase


class FixF4(CustomTestCase):
    def test_retire_is_not_granted_without_the_refresh(self):
        wanted = RetentionSwitches(track_stored=True, retire_ended_pages=True)
        resolved, refused = resolve_retention_switches(
            wanted,
            host_cache_mode=True,
            python_tree_core=True,
            single_pp_rank=True,
            next_turn_prefetch=True,
            full_kv_only=True,
        )
        self.assertEqual(resolved, RetentionSwitches(track_stored=True))
        self.assertEqual(len(refused), 1)

    def test_with_the_refresh_a_page_read_back_is_touched_at_its_next_request(self):
        cache = stored_marks_cache(retire_ended_pages=True, touch_interval_s=300.0)
        store = recording_store(cache)
        first = stored_node(cache, [1, 2])
        back = read_back_node(cache, first, [3, 4])
        computed_node(cache, [1, 2, 3, 4, 5])
        request = SimpleNamespace(session=None)
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2, 3, 4, 5])), req=request)
        )
        refreshed = [
            key
            for call in store.touch_pages.call_args_list
            if call.kwargs["newest_ns"] is not None
            for key in call.args[0]
        ]
        for key in cache.tree_core.get_hash_values(back):
            self.assertIn(key, refreshed)
