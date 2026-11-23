"""What holds host (L2) memory: the HICACHE_EVT host_census line (agent_cache/OPEN_ITEMS.md section 7).

A storage read refused host room means evict_host found nothing to free. The census walks
the tree once and splits the host pool into free pages, the release queue, read staging,
the device mirror (host copies of nodes still on the GPU; host eviction never takes them,
except duplicate copies under write_back) and host-only nodes. A host-only node is
evictable (now, or once its subtree is gone) or held. Eviction takes childless nodes only,
so a held node also keeps its host-only ancestors: each held token is attributed to the
reasons of every lock in its node's subtree, and fields h_<reason> are tokens that only
that reason holds (h_multi: several reasons).

Scheduler thread only: the tree and the holder maps are read without locks.
"""

from __future__ import annotations

import enum
import logging
import math
import time
from typing import Callable, Optional

import msgspec

from sglang.srt.environ import envs
from sglang.srt.mem_cache.unified_cache.component_type import BASE_COMPONENT_TYPE

logger = logging.getLogger(__name__)


class HoldReason(enum.IntFlag):
    WRITE = 1  # SSD write submitted, its ack not drained yet
    LOOKUP = 2  # anchor of a storage lookup, not parked or launched yet
    PARKED = 4  # anchor of a hit parked for dispatch
    READ = 8  # anchor of a launched read
    PIN = 16  # landed read pinned for an arrived turn
    STAGED = 32  # any read anchor or pin of a turn not yet arrived (an early read)
    LOAD_BACK = 64  # host anchor of a host->device load-back
    SEGMENT = 128  # a device (segment) lock on a host-only node
    OTHER = 256  # host_lock_ref beyond the holders the cache knows


# Census field per single reason, in log order.
REASON_FIELDS = tuple((f"h_{reason.name.lower()}", reason) for reason in HoldReason)
# The walk works on plain ints: IntFlag operators would dominate its cost.
_OTHER = int(HoldReason.OTHER)
_SEGMENT = int(HoldReason.SEGMENT)


class HostTreeTally(msgspec.Struct):
    n_nodes: int = 0
    n_host_nodes: int = 0
    mirror: int = 0
    mirror_locked: int = 0
    dev_unbacked: int = 0
    evict_leaf: int = 0
    evict_casc: int = 0
    held: dict[int, int] = {}
    h_multi: int = 0
    anc: int = 0
    mismatch: int = 0

    def host_only(self) -> int:
        return (
            self.evict_leaf + self.evict_casc + sum(self.held.values()) + self.h_multi
        )


def tally_host_tree(
    root, *, holder_bits: dict[int, int], holder_counts: dict[int, int]
) -> HostTreeTally:
    """Walk the tree below root once (iterative post-order). holder_bits maps a node id to
    the OR of its holders' reasons, holder_counts to how many host locks they account for."""
    tally = HostTreeTally()
    subtree_bits: dict[int, int] = {}
    stack = [(root, False)]
    while stack:
        node, children_done = stack.pop()
        if not children_done:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children.values())
            continue
        bits = 0
        for child in node.children.values():
            bits |= subtree_bits.pop(child.id, 0)
        if node is root:
            continue
        tally.n_nodes += 1
        subtree_bits[node.id] = _tally_node(
            node, tally, below=bits, holder_bits=holder_bits, holder_counts=holder_counts
        )
    return tally


def _tally_node(
    node, tally: HostTreeTally, *, below: int, holder_bits, holder_counts
) -> int:
    """Count one node; returns the reasons its subtree holds host memory for."""
    data = node.component_data[BASE_COMPONENT_TYPE]
    own = int(holder_bits.get(node.id, 0))
    expected = holder_counts.get(node.id, 0)
    if data.host_lock_ref != expected:
        tally.mismatch += 1
        if data.host_lock_ref > expected:
            own |= _OTHER
    if data.host_lock_ref == 0:
        # A holder without a lock holds nothing: eviction can take the node.
        own = 0
    if data.value is not None:
        # On the GPU: no host-only node sits above a GPU-resident one.
        if data.host_value is None:
            tally.dev_unbacked += len(data.value)
            return 0
        tally.n_host_nodes += 1
        tally.mirror += len(data.host_value)
        if data.lock_ref > 0:
            tally.mirror_locked += len(data.host_value)
        return 0
    if data.host_value is None:
        return below
    tally.n_host_nodes += 1
    tokens = len(data.host_value)
    if data.lock_ref > 0:
        own |= _SEGMENT
    held_by = own | below
    if held_by == 0:
        if node.children:
            tally.evict_casc += tokens
        else:
            tally.evict_leaf += tokens
    elif held_by & (held_by - 1) == 0:
        tally.held[held_by] = tally.held.get(held_by, 0) + tokens
    else:
        tally.h_multi += tokens
    if own == 0 and held_by != 0:
        tally.anc += tokens
    return held_by


def tally_fields(tally: HostTreeTally) -> dict[str, int]:
    fields = {
        "mirror": tally.mirror,
        "mirror_locked": tally.mirror_locked,
        "dev_unbacked": tally.dev_unbacked,
        "host_only": tally.host_only(),
        "evict_leaf": tally.evict_leaf,
        "evict_casc": tally.evict_casc,
    }
    for name, reason in REASON_FIELDS:
        fields[name] = tally.held.get(int(reason), 0)
    fields.update(
        h_multi=tally.h_multi,
        anc=tally.anc,
        n_nodes=tally.n_nodes,
        n_host_nodes=tally.n_host_nodes,
        mismatch=tally.mismatch,
    )
    return fields


class HostCensusLog:
    """host_census lines: one per interval from the cache tick, and at most one per
    interval when a read is refused room, each carrying the refusals counted since the
    previous line. A refusal line also restarts the tick's interval, so one state is not
    walked twice in a pass. A refusal inside insert (a device-to-host backup) only counts."""

    def __init__(self, *, interval_s: float, clock: Callable[[], float] = time.monotonic):
        self.interval_s = interval_s
        self._clock = clock
        self._next_tick = clock() + interval_s
        self._next_refusal = -math.inf
        self._counts = self._zero_counts()

    @classmethod
    def maybe_create(cls, *, enabled: bool) -> Optional[HostCensusLog]:
        interval_s = envs.SGLANG_LOG_HICACHE_HOST_CENSUS_INTERVAL.get()
        return cls(interval_s=interval_s) if enabled and interval_s > 0 else None

    @staticmethod
    def _zero_counts() -> dict[str, int]:
        return dict.fromkeys(
            (
                "refused",
                "refused_tok",
                "cut",
                "cut_tok",
                "backup_refused",
                "backup_refused_tok",
            ),
            0,
        )

    def tick(self, take: Callable[[], dict[str, int]]) -> None:
        now = self._clock()
        if now < self._next_tick:
            return
        self._next_tick = now + self.interval_s
        self._emit("tick", take())

    def on_read_refusal(
        self, take: Callable[[], dict[str, int]], *, tokens: int, cut: bool
    ) -> None:
        kind = "cut" if cut else "refused"
        self._counts[kind] += 1
        self._counts[f"{kind}_tok"] += tokens
        now = self._clock()
        if now < self._next_refusal:
            return
        self._next_refusal = now + self.interval_s
        self._next_tick = now + self.interval_s
        self._emit("refusal", take())

    def on_backup_refusal(self, tokens: int) -> None:
        self._counts["backup_refused"] += 1
        self._counts["backup_refused_tok"] += tokens

    def _emit(self, why: str, fields: dict[str, int]) -> None:
        fields.update(self._counts)
        self._counts = self._zero_counts()
        logger.info(
            "HICACHE_EVT host_census why=%s %s",
            why,
            " ".join(f"{name}={value}" for name, value in fields.items()),
        )
