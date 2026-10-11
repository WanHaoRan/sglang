"""Next-use eviction: order device and host eviction by the hinted next arrival
(agent_cache/ORACLE_V2_PLAN.md section 4.5).

The next-turn prefetcher registers each committed turn's tail node with the
predicted admission of the session's next turn, None when the session ended.
Eviction then takes, in order: ended sessions, sessions whose predicted turn
is overdue by more than the grace (oldest first: the turn is not coming),
sessions returning beyond the horizon (farthest first), unregistered nodes by
plain LRU, and last the sessions returning within the horizon (farthest
first). Priorities compare as tuples, smallest evicted first, like every other
EvictionStrategy.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.evict_policy import EvictionStrategy

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

TIER_ENDED = 0
TIER_OVERDUE = 1
TIER_FAR = 2
TIER_UNKNOWN = 3
TIER_SOON = 4
# Arbitrary; one ended entry per finished request under un-hinted traffic, so
# the bound, not the request count, sizes the registry (~100 B per entry).
REGISTRY_MAX_ENTRIES = 1 << 16


class NextUseRegistry:
    """Node id -> predicted next use (perf_counter seconds); None = ended.
    Past max_entries the oldest registration is dropped: node ids are
    monotonic, so the oldest entries are the likeliest to name evicted nodes."""

    def __init__(self, max_entries: int = REGISTRY_MAX_ENTRIES):
        self._next_use: OrderedDict[int, Optional[float]] = OrderedDict()
        self._known_sentinel = object()
        self.max_entries = max_entries

    def register(self, node_id: int, next_use: Optional[float]) -> None:
        self._next_use.pop(node_id, None)
        self._next_use[node_id] = next_use
        while len(self._next_use) > self.max_entries:
            self._next_use.popitem(last=False)

    def forget(self, node_id: int) -> None:
        self._next_use.pop(node_id, None)

    def lookup(self, node_id: int) -> tuple[bool, Optional[float]]:
        value = self._next_use.get(node_id, self._known_sentinel)
        if value is self._known_sentinel:
            return False, None
        return True, value

    def __len__(self) -> int:
        return len(self._next_use)


class NextUseStrategy(EvictionStrategy):
    def __init__(
        self,
        horizon_s: float = 300.0,
        overdue_grace_s: float = 60.0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if horizon_s < 0 or overdue_grace_s < 0:
            raise ValueError(
                f"horizon_s {horizon_s} and overdue_grace_s {overdue_grace_s} "
                "must be non-negative"
            )
        self.horizon_s = float(horizon_s)
        # 60 s matches the prefetcher's minimum plan expiry (MIN_EXPIRY_S).
        self.overdue_grace_s = float(overdue_grace_s)
        self._clock = clock
        self.registry = NextUseRegistry()

    def get_priority(self, node: TreeNode) -> tuple[int, float]:
        known, next_use = self.registry.lookup(node.id)
        if not known:
            return (TIER_UNKNOWN, node.last_access_time)
        if next_use is None:
            return (TIER_ENDED, node.last_access_time)
        delta = next_use - self._clock()
        if delta < -self.overdue_grace_s:
            return (TIER_OVERDUE, next_use)
        if delta > self.horizon_s:
            return (TIER_FAR, -next_use)
        return (TIER_SOON, -next_use)
