"""Oracle v2 server flags: the defaults keep the FIFO read path, and a bad
combination fails at argument resolution (agent_cache/ORACLE_V2_PLAN.md 4.1)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import re
import unittest

import sglang.srt.server_args as server_args_module
from sglang.srt.arg_groups.hicache_hook import (
    handle_hicache,
    handle_hicache_ratio_default,
)
from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

THREE_TIER = dict(enable_hierarchical_cache=True, hicache_storage_backend="nixl")


def make_args(**overrides) -> ServerArgs:
    # A dummy model path takes the pipeline's early return, so the hooks under
    # test are run by hand after their one prerequisite.
    args = ServerArgs(model_path="dummy", **overrides)
    handle_hicache_ratio_default(args)
    return args


class TestOracleV2Flags(CustomTestCase):
    def test_defaults_keep_the_fifo_path_and_pass_validation(self):
        args = make_args(**THREE_TIER)
        self.assertEqual(
            (
                args.hicache_storage_read_order,
                args.hicache_storage_read_window_pages,
                args.hicache_storage_prefetch_pin_fraction,
                args.radix_eviction_policy,
            ),
            ("fifo", 512, 0.0, "lru"),
        )
        handle_hicache(args)
        handle_cache_compatibility(args)

    def test_deadline_order_prerequisites(self):
        """The parked reads land in the resident host tier on one rank."""
        cases = [
            (
                dict(enable_hierarchical_cache=True, hicache_storage_read_order="deadline"),
                "--hicache-storage-backend",
            ),
            (
                dict(
                    **THREE_TIER,
                    hicache_storage_read_order="deadline",
                    hicache_host_memory_mode="buffer_only",
                ),
                "--hicache-host-memory-mode",
            ),
            (dict(**THREE_TIER, hicache_storage_read_order="deadline", tp_size=2), "TP = PP = 1"),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    handle_hicache(make_args(**overrides))
        handle_hicache(
            make_args(
                **THREE_TIER,
                hicache_storage_read_order="deadline",
                hicache_storage_prefetch_pin_fraction=0.3,
            )
        )

    def test_pin_and_window_bounds(self):
        cases = [
            (
                dict(**THREE_TIER, hicache_storage_prefetch_pin_fraction=0.3),
                "needs --hicache-storage-read-order deadline",
            ),
            (
                dict(
                    **THREE_TIER,
                    hicache_storage_read_order="deadline",
                    hicache_storage_prefetch_pin_fraction=0.6,
                ),
                "[0, 0.5]",
            ),
            (
                dict(
                    **THREE_TIER,
                    hicache_storage_read_order="deadline",
                    hicache_storage_read_window_pages=0,
                ),
                "--hicache-storage-read-window-pages",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    handle_hicache(make_args(**overrides))

    def test_next_use_needs_the_prefetcher_and_a_non_negative_horizon(self):
        with self.assertRaisesRegex(ValueError, "next-turn-prefetch"):
            handle_cache_compatibility(
                make_args(**THREE_TIER, radix_eviction_policy="next_use")
            )
        for bad in ({"horizon_s": -5}, {"overdue_grace_s": "soon"}):
            with self.assertRaisesRegex(ValueError, next(iter(bad))):
                handle_cache_compatibility(
                    make_args(
                        **THREE_TIER,
                        radix_eviction_policy="next_use",
                        hicache_storage_next_turn_prefetch="on",
                        radix_eviction_policy_config=bad,
                    )
                )
        handle_cache_compatibility(
            make_args(
                **THREE_TIER,
                radix_eviction_policy="next_use",
                hicache_storage_next_turn_prefetch="on",
                radix_eviction_policy_config={"horizon_s": 300},
            )
        )

    def test_cli_spellings_reach_their_fields(self):
        parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        ns = parser.parse_args(
            [
                "--model-path", "dummy",
                "--hicache-storage-read-order", "deadline",
                "--hicache-storage-read-window-pages", "256",
                "--hicache-storage-prefetch-pin-fraction", "0.3",
                "--radix-eviction-policy", "next_use",
                "--radix-eviction-policy-config", '{"horizon_s": 300}',
            ]
        )
        self.assertEqual(
            (
                ns.hicache_storage_read_order,
                ns.hicache_storage_read_window_pages,
                ns.hicache_storage_prefetch_pin_fraction,
                ns.radix_eviction_policy,
                ns.radix_eviction_policy_config,
            ),
            ("deadline", 256, 0.3, "next_use", {"horizon_s": 300}),
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(
                ["--model-path", "dummy", "--hicache-storage-read-order", "random"]
            )


if __name__ == "__main__":
    unittest.main()
