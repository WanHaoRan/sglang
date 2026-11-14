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

ReturnChanceStrategy orders the same registrations by a distribution instead of a
point (agent_cache/PAPER_PLAN.md, "Formulation", rule 3): a tail registered with a
ReturnForecast is scored by the chance its session returns within the horizon
given its idle time, so no forecast ends a session. A registration there covers
the session, not one node: the nodes under and above the registered tail that only
that session uses take its registration.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Optional, Union

import msgspec

from sglang.srt.mem_cache.evict_policy import EvictionStrategy
from sglang.srt.mem_cache.next_turn_policy import (
    MIN_GAP_SAMPLES,
    GapDistribution,
    sampled_chance,
)

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

RETURN_TIER_ENDED = 0
RETURN_TIER_UNKNOWN = 1
RETURN_TIER_SCORED = 2
RETURN_TIER_QUEUED = 3
# An exact hint's turn reaches the scheduler up to this long after its gap
# (ARRIVAL_OFFSET_RANGE_S in next_turn_estimators.py): due, not gone.
POINT_GRACE_S = 5.0
# A session registered as queued for longer than this ranks as un-hinted again: a turn
# aborted before it commits leaves the registration behind. Arbitrary; as the pin backstop.
QUEUED_MAX_AGE_S = 600.0


RETURN_SCORES = ("entry", "samples", "lru")


class ReturnForecast(msgspec.Struct, frozen=True):
    """A session's gap distribution and when the gap started (perf_counter seconds),
    with the gaps the engine has seen for the entry's key and for every key, ascending;
    both empty unless the strategy scores from samples."""

    dist: GapDistribution
    idle_from: float
    key_gaps: tuple[float, ...] = ()
    all_gaps: tuple[float, ...] = ()


Registration = Union[None, float, ReturnForecast]


class NextUseRegistry:
    """Node id -> predicted next use (perf_counter seconds), or the forecast it
    comes from; None = ended. Past max_entries the oldest registration is dropped:
    node ids are monotonic, so the oldest entries are the likeliest to name evicted
    nodes."""

    def __init__(self, max_entries: int = REGISTRY_MAX_ENTRIES):
        self._next_use: OrderedDict[int, Registration] = OrderedDict()
        # Previous tail of a session -> the tail registered in its place.
        self._moved: OrderedDict[int, int] = OrderedDict()
        self._known_sentinel = object()
        self.max_entries = max_entries

    def register(self, node_id: int, next_use: Registration) -> None:
        self._next_use.pop(node_id, None)
        self._next_use[node_id] = next_use
        while len(self._next_use) > self.max_entries:
            self._next_use.popitem(last=False)

    def forget(self, node_id: int) -> None:
        self._next_use.pop(node_id, None)

    def move(self, previous_id: int, node_id: int) -> None:
        """A session's tail is now node_id: forget the previous one, and remember
        where it went for a strategy that maps other nodes to a session's tail."""
        self._next_use.pop(previous_id, None)
        self._moved.pop(previous_id, None)
        self._moved[previous_id] = node_id
        while len(self._moved) > self.max_entries:
            self._moved.popitem(last=False)

    def current(self, node_id: int) -> int:
        """The tail a session's earlier tail id stands for now."""
        # A session moves once per turn, so a chain is as long as its turns since
        # the mapping was made; the bound only guards against a cycle.
        for _ in range(4096):
            moved = self._moved.get(node_id)
            if moved is None:
                break
            node_id = moved
        return node_id

    def lookup(self, node_id: int) -> tuple[bool, Registration]:
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


class ReturnChanceStrategy(EvictionStrategy):
    """Evicts first the session least likely back within horizon_s given its idle
    time. A session registered with a time has a turn that has arrived and goes last;
    a node of no registered session (un-hinted traffic, a shared prefix) goes by LRU
    before any scored session.

    score "entry" takes the chance from the hint entry's distribution; "samples" from
    the gaps the engine has seen for the entry's key, then for every key, and from the
    entry only until enough were seen; "lru" uses no forecast at all, only the end
    signal and the queue: ended sessions first, queued turns last, LRU between."""

    def __init__(
        self,
        horizon_s: float = 30.0,
        score: str = "entry",
        clock: Callable[[], float] = time.perf_counter,
    ):
        if horizon_s <= 0:
            raise ValueError(f"horizon_s {horizon_s} must be positive")
        if score not in RETURN_SCORES:
            raise ValueError(f"score {score!r} must be one of {RETURN_SCORES}")
        self.horizon_s = float(horizon_s)
        self.score = score
        self._clock = clock
        self.registry = NextUseRegistry()
        # Node id -> the registered tail whose session the node belongs to. Bounded
        # like the registry, oldest first.
        self._tail_of: OrderedDict[int, int] = OrderedDict()

    def _session_registration(self, node: TreeNode) -> tuple[bool, Registration]:
        """The registration of the node's session. The prefetcher registers one tail
        per session; a session's body once that tail is evicted, and a span read back
        under it, are other nodes. A node takes the registration of its only child
        (found when the child was ranked) or of the parent it is the only child of:
        with one child there is one session on that path."""
        registry = self.registry
        parent = node.parent
        only_child = parent is not None and len(parent.children) == 1
        if parent is not None and not only_child:
            # A branch point belongs to no single session, even if it once had one child.
            self._tail_of.pop(parent.id, None)
        tail_id = node.id
        known, forecast = registry.lookup(tail_id)
        if not known:
            tail_id = self._tail_of.get(node.id)
            if tail_id is None and only_child:
                tail_id = (
                    parent.id
                    if registry.lookup(parent.id)[0]
                    else self._tail_of.get(parent.id)
                )
            if tail_id is None:
                return False, None
            # The session has committed turns since: its tail moved on.
            tail_id = registry.current(tail_id)
            known, forecast = registry.lookup(tail_id)
            if not known:
                return False, None
            self.adopt(node.id, tail_id)
        if only_child and not registry.lookup(parent.id)[0]:
            self.adopt(parent.id, tail_id)
        return True, forecast

    def adopt(self, node_id: int, tail_id: int) -> None:
        """node_id belongs to the session whose registered tail is tail_id."""
        self._tail_of.pop(node_id, None)
        self._tail_of[node_id] = tail_id
        while len(self._tail_of) > self.registry.max_entries:
            self._tail_of.popitem(last=False)

    def get_priority(self, node: TreeNode) -> tuple[int, float, float]:
        known, forecast = self._session_registration(node)
        if not known:
            return (RETURN_TIER_UNKNOWN, node.last_access_time, 0.0)
        if forecast is None:
            return (RETURN_TIER_ENDED, node.last_access_time, 0.0)
        if not isinstance(forecast, ReturnForecast):
            if self._clock() - forecast > QUEUED_MAX_AGE_S:
                return (RETURN_TIER_UNKNOWN, node.last_access_time, 0.0)
            return (RETURN_TIER_QUEUED, node.last_access_time, 0.0)
        if self.score == "lru":
            return (RETURN_TIER_UNKNOWN, node.last_access_time, 0.0)
        idle = self._clock() - forecast.idle_from
        dist = forecast.dist
        if dist.is_point and idle >= dist.median_s:
            # Past an exact hint: due within the grace, and then not coming.
            if idle < dist.median_s + POINT_GRACE_S:
                return (RETURN_TIER_SCORED, 1.0, 0.0)
            return (RETURN_TIER_SCORED, 0.0, -math.inf)
        chance = self._sampled_chance(forecast, idle)
        if chance is None:
            chance = dist.chance_within(idle_s=idle, horizon_s=self.horizon_s)
        # Among equal chances the session expected back later goes first, so exact
        # hints give farthest-next-use order.
        return (RETURN_TIER_SCORED, chance, idle - dist.median_s)

    def _sampled_chance(self, forecast: ReturnForecast, idle: float) -> Optional[float]:
        """The chance from the key's own gaps, else from every key's; 0 when enough
        gaps were seen and almost none lasted this long; None with too few seen."""
        for gaps in (forecast.key_gaps, forecast.all_gaps):
            chance = sampled_chance(gaps=gaps, idle_s=idle, horizon_s=self.horizon_s)
            if chance is not None:
                return chance
        return 0.0 if len(forecast.all_gaps) >= MIN_GAP_SAMPLES else None
