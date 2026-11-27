"""Reviewer's throwaway checks (lens: repository rules and tests). Not part of the change.

Each "gap" case breaks one behaviour of the engine at run time (a patch on the class, the
copy's source is not edited), runs the engineer's own suites, and prints which of their
cases fail. No failing case = no test guards that behaviour.
Run with -s to see the report lines (prefix REVIEW).
"""

import importlib
import io
import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import Mock

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import unified_radix_cache as urc_module
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.storage_retention import RetentionSwitches, stored_votes
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs

CACHE_SUITES = (
    "test_next_use_eviction",
    "test_storage_read_dispatch",
    "test_next_turn_prefetch",
)
ALL_SUITES = CACHE_SUITES + ("test_hicache_nixl_cleaner", "test_hicache_nixl_storage")


def failing_cases(names=CACHE_SUITES):
    failed = []
    ran = 0
    for name in names:
        module = importlib.import_module(name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        ran += result.testsRun
        failed += sorted({str(test).split(" ")[0] for test, _ in result.failures + result.errors})
    return ran, failed


def report(label, names=CACHE_SUITES):
    ran, failed = failing_cases(names)
    print(f"REVIEW {label}: ran={ran} failing={failed}", flush=True)
    return failed


class ReviewGaps(unittest.TestCase):
    def test_00_baseline_is_green(self):
        self.assertEqual(report("baseline (no break)", ALL_SUITES), [])

    def test_01_prompt_insert_site_never_flags(self):
        original = UnifiedRadixCache._turn_ends_session

        def only_at_finish(self, req, *, reply_known):
            return reply_known and original(self, req, reply_known=reply_known)

        with mock.patch.object(UnifiedRadixCache, "_turn_ends_session", only_at_finish):
            report("R1: cache_unfinished_req never flags the insert")

    def test_02_prompt_insert_site_reads_the_reply(self):
        original = UnifiedRadixCache._turn_ends_session

        def always_reply_known(self, req, *, reply_known):
            return original(self, req, reply_known=True)

        with mock.patch.object(
            UnifiedRadixCache, "_turn_ends_session", always_reply_known
        ):
            report("R1: cache_unfinished_req passes reply_known=True")

    def test_03_pp_ranks_do_not_agree(self):
        def tp_only(self, operations, *, synced):
            votes = stored_votes(
                operations,
                page_size=self.page_size,
                writes_nothing=self.cache_controller.backup_skip,
            )
            if not (synced and votes and self.tp_world_size > 1):
                return votes
            agreed = torch.tensor(votes, dtype=torch.int)
            self._all_reduce(agreed, torch.distributed.ReduceOp.MIN)
            return [bool(vote) for vote in agreed.tolist()]

        with mock.patch.object(UnifiedRadixCache, "_agree_stored", tp_only):
            report("R3: no reduce when only PP > 1")

    def test_04_split_drops_the_refresh_stamp(self):
        original = UnifiedTreeCore._split_node

        def split(self, key, child, split_len):
            new_node, action = original(self, key, child, split_len)
            new_node.storage_touched_at = -math.inf
            return new_node, action

        with mock.patch.object(UnifiedTreeCore, "_split_node", split):
            report("R4: a split does not copy storage_touched_at")

    def test_05_retire_with_storage_detached(self):
        def retire(self, tail_node_id, *, keep_tokens, own_leaves):
            if not self.retention.retire_ended_pages:
                return
            retired = self.tree_core.take_retired_storage_pages(
                tail_node_id, keep_tokens=keep_tokens, own_leaves=own_leaves
            )
            pending = [
                node_id
                for node_id in retired.pending_node_ids
                if node_id not in self._storage_skip_nodes
            ]
            self._retire_at_ack.update(pending)
            if retired.page_keys:
                self.cache_controller.storage_backend.touch_pages(
                    retired.page_keys, newest_ns=None
                )
            urc_module.logger.info(
                "HICACHE_EVT l3_retire node=%d pages=%d at_ack=%d keep_tokens=%d",
                tail_node_id,
                len(retired.page_keys),
                sum(len(self.tree_core.get_hash_values(n)) for n in pending),
                keep_tokens,
            )

        with mock.patch.object(UnifiedRadixCache, "retire_storage_chain", retire):
            report("R4: retire does not check enable_storage")

    def test_06_match_hook_without_the_interval_guard(self):
        original = UnifiedRadixCache.match_prefix

        def match(self, params):
            result = original(self, params)
            if (
                self.retention.touch_interval_s <= 0.0
                and params.req is not None
                and self.enable_storage
            ):
                self._refresh_served_chain(result.last_host_node)
            return result

        with mock.patch.object(UnifiedRadixCache, "match_prefix", match):
            report("R4: the match hook runs with the touch interval at 0")

    def test_07_resolver_wiring_always_met(self):
        original = urc_module.resolve_retention_switches
        for forced in (
            "host_cache_mode",
            "python_tree_core",
            "next_turn_prefetch",
            "full_kv_only",
        ):

            def resolve(requested, _forced=forced, **needs):
                return original(requested, **{**needs, _forced: True})

            with mock.patch.object(urc_module, "resolve_retention_switches", resolve):
                report(f"switches: the cache always passes {forced}=True")

    def test_08_no_next_reason_misnamed(self):
        original = NextTurnPrefetcher._turn_outlook

        def outlook(self, req, *, reply_known, named_session):
            found = original(
                self, req, reply_known=reply_known, named_session=named_session
            )
            if found.end_why == "no_next":
                return msgspec.structs.replace(found, end_why="entry")
            return found

        with mock.patch.object(NextTurnPrefetcher, "_turn_outlook", outlook):
            report("X1: ntp_end why=no_next is logged as why=entry")

    def test_09_extents_never_bounded(self):
        with mock.patch(
            "sglang.srt.mem_cache.next_turn_prefetch.REGISTRY_MAX_ENTRIES", 10**12
        ):
            report("R4: _session_extents has no bound")


class ReviewBehaviour(unittest.TestCase):
    def test_override_is_not_restored_when_its_body_raises(self):
        name = "SGLANG_HICACHE_STORAGE_TRACK_STORED"
        with mock.patch.dict(os.environ):
            os.environ.pop(name, None)
            try:
                with envs.SGLANG_HICACHE_STORAGE_TRACK_STORED.override(True):
                    raise AssertionError("an assertion in the body")
            except AssertionError:
                pass
            print(f"REVIEW override after a raising body: {name}={os.environ.get(name)!r}")
            self.assertEqual(os.environ.get(name), "True")

    def test_a_sidecar_pool_registered_after_the_switches_were_resolved(self):
        from test_session_unified_radix_cache import make_params

        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", hicache_storage_next_turn_prefetch="on"),
            role="scheduler",
        )
        cache = UnifiedRadixCache(make_params(enable_session=False, page_size=2))
        store = Mock(spec=HiCacheStorage, touch_interval_s=0.0, retire_ended_pages=True)
        cache.cache_controller = SimpleNamespace(storage_backend=store)
        with mock.patch.dict(os.environ):
            os.environ["SGLANG_HICACHE_STORAGE_TRACK_STORED"] = "1"
            cache._init_storage_retention()
            at_init = cache.retention
            cache.register_sidecar_pool(SimpleNamespace(name="draft"))
            after_register = cache.retention
            cache._init_storage_retention()
            if_resolved_again = cache.retention
        print(
            "REVIEW sidecar after init: at_init.retire=%s after_register.retire=%s "
            "if_resolved_again.retire=%s"
            % (
                at_init.retire_ended_pages,
                after_register.retire_ended_pages,
                if_resolved_again.retire_ended_pages,
            )
        )
        self.assertTrue(after_register.retire_ended_pages)
        self.assertFalse(if_resolved_again.retire_ended_pages)


if __name__ == "__main__":
    unittest.main()
