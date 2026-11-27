"""Retain switches of the three-tier cache: which radix nodes count as stored on the
SSD, which host-to-storage writes are skipped, and how the SSD cleaner's order is
steered through file modification times
(agent_cache/results/design_unified_1126/DESIGN.md, section 4).

Every switch is off by default. The cache resolves them at startup and at each runtime
attach of a storage backend; a switch that lacks what it needs stays off.
"""

from __future__ import annotations

from typing import Sequence

import msgspec


class RetentionSwitches(msgspec.Struct, frozen=True):
    """The Retain switches as one cache resolved them; the default is all off."""

    # SGLANG_HICACHE_STORAGE_TRACK_STORED.
    track_stored: bool = False
    # SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES.
    skip_ended_writes: bool = False
    # SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S or the store's touch_interval_s key; seconds.
    touch_interval_s: float = 0.0
    # SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES or the store's retire_ended_pages key.
    retire_ended_pages: bool = False

    @property
    def explicit_ends(self) -> bool:
        """Whether any switch acts on a session's end, which must then be explicit."""
        return self.skip_ended_writes or self.retire_ended_pages


def resolve_retention_switches(
    requested: RetentionSwitches,
    *,
    host_cache_mode: bool,
    python_tree_core: bool,
    next_turn_prefetch: bool,
    full_kv_only: bool,
) -> tuple[RetentionSwitches, list[str]]:
    """The requested switches that have what they need, and a line per refusal."""
    refused: list[str] = []

    def granted(name: str, wanted: bool, *needs: tuple[bool, str]) -> bool:
        unmet = [what for met, what in needs if not met]
        if wanted and unmet:
            refused.append(f"{name} stays off: it needs {', '.join(unmet)}")
        return wanted and not unmet

    track_stored = granted(
        "SGLANG_HICACHE_STORAGE_TRACK_STORED",
        requested.track_stored,
        (host_cache_mode, "--hicache-host-memory-mode cache"),
        (python_tree_core, "the python tree core"),
    )
    marks = (track_stored, "SGLANG_HICACHE_STORAGE_TRACK_STORED")
    # Its create() admits TP = PP = 1 only, which the two switches below rely on.
    prefetcher = (next_turn_prefetch, "--hicache-storage-next-turn-prefetch")
    skip_ended_writes = granted(
        "SGLANG_HICACHE_STORAGE_SKIP_ENDED_WRITES",
        requested.skip_ended_writes,
        marks,
        prefetcher,
    )
    touch = granted(
        "SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S",
        requested.touch_interval_s > 0.0,
        marks,
    )
    retire_ended_pages = granted(
        "SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES",
        requested.retire_ended_pages,
        marks,
        prefetcher,
        # A page leaves first only once every one of its files is at mtime zero, and
        # only the KV files are touched.
        (full_kv_only, "a FULL-only cache without sidecar pools"),
    )
    return (
        RetentionSwitches(
            track_stored=track_stored,
            skip_ended_writes=skip_ended_writes,
            touch_interval_s=requested.touch_interval_s if touch else 0.0,
            retire_ended_pages=retire_ended_pages,
        ),
        refused,
    )


def stored_votes(
    operations: Sequence, *, page_size: int, writes_nothing: bool
) -> list[bool]:
    """This rank's verdict per acked write: every page of the operation was written.
    A rank that writes nothing for a replicated pool has nothing to fail."""
    return [
        writes_nothing
        or operation.completed_tokens >= len(operation.hash_value) * page_size
        for operation in operations
    ]
