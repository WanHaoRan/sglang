"""Dry run on CPU of the unified arm's command line through argument parsing and resolution.

    python3 dry_resolve.py <argv file, one token a line> [extra flags...]

Prints the resolved eviction policy, its config and the next-turn settings, or the error the boot would raise.
"""
import sys

from sglang.srt.server_args import prepare_server_args

argv = [t for t in open(sys.argv[1]).read().split("\n") if t][3:] + sys.argv[2:]
args = prepare_server_args(argv)
args.resolve_once()
for name in (
    "attention_backend",
    "radix_eviction_policy",
    "radix_eviction_policy_config",
    "hicache_storage_next_turn_prefetch",
    "hicache_storage_next_turn_read_timing",
    "hicache_storage_next_turn_price_learning",
    "hicache_storage_next_turn_soft_pins",
    "hicache_storage_next_turn_memory_price",
    "hicache_storage_next_turn_arrival_gate",
    "hicache_storage_next_turn_arrival_hold",
    "max_total_tokens",
    "host",
):
    print(f"{name} = {getattr(args, name)!r}")
