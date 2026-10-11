"""Deadline-ordered dispatch of storage (L3) reads onto the single read thread
(agent_cache/ORACLE_V2_PLAN.md section 4.2).

A storage hit parks here holding no host memory until it is dispatched; the
cache then allocates the read's host pages and hands the operation to the I/O
thread. Dispatch order is by deadline: None means "needed now" (a request's own
lookup or re-query, a queued request at the head) and goes before any timed
deadline; a timed deadline is the predicted admission of a next-turn prefetch.
The I/O thread holds at most `window_pages` of dispatched work and dispatched
plus landed-and-pinned pages stay within `budget_tokens`. A "now" read ignores
both limits, as the FIFO path allocates it unconditionally and an admission may
be waiting on it.

Pure policy: no clock, no tree, no controller; every time is passed in.
"""

from __future__ import annotations

from typing import Any, Callable, Hashable, Optional

import msgspec


class ParkedRead(msgspec.Struct, frozen=True):
    handle: Hashable
    operation: Any
    pages: int
    parked_at: float


class DispatchDecision(msgspec.Struct, frozen=True):
    read: ParkedRead
    deadline: Optional[float]
    waited_s: float


class ReadSchedulerStats(msgspec.Struct):
    parked_total: int = 0
    dispatched_total: int = 0
    dispatched_now: int = 0
    window_holds: int = 0
    budget_holds: int = 0
    max_wait_s: float = 0.0


class StorageReadScheduler:
    """Parked storage hits and the rule that hands them to the I/O thread."""

    def __init__(self, *, window_pages: int, budget_tokens: int, page_size: int):
        if window_pages < 1 or page_size < 1 or budget_tokens < 0:
            raise ValueError(
                f"window_pages {window_pages}, page_size {page_size} and "
                f"budget_tokens {budget_tokens} must be positive"
            )
        self.window_pages = window_pages
        self.budget_tokens = budget_tokens
        self.page_size = page_size
        # Insertion order is the tiebreak among equal deadlines (park order).
        self._parked: dict[Hashable, ParkedRead] = {}
        self.stats = ReadSchedulerStats()

    def park(self, *, handle: Hashable, operation: Any, pages: int, now: float) -> None:
        if handle in self._parked:
            raise ValueError(f"read for {handle} is already parked")
        self._parked[handle] = ParkedRead(
            handle=handle, operation=operation, pages=pages, parked_at=now
        )
        self.stats.parked_total += 1

    def clear(self) -> None:
        self._parked.clear()

    def discard(self, handle: Hashable) -> bool:
        return self._parked.pop(handle, None) is not None

    def is_parked(self, handle: Hashable) -> bool:
        return handle in self._parked

    def parked_count(self) -> int:
        return len(self._parked)

    def parked_pages(self) -> int:
        return sum(read.pages for read in self._parked.values())

    def parked_handles(self) -> list[Hashable]:
        return list(self._parked)

    def dispatch(
        self,
        *,
        now: float,
        deadline_of: Callable[[Hashable], Optional[float]],
        in_flight_pages: int,
        held_tokens: int,
    ) -> list[DispatchDecision]:
        """Pop the reads to launch now, earliest deadline first.

        `in_flight_pages` is what the I/O thread already holds; `held_tokens`
        is the host memory charged to prefetching (the in-flight reads plus
        the pinned spans), so a read is charged once, here on top of it.
        """
        decisions: list[DispatchDecision] = []
        if not self._parked:
            return decisions
        deadlines = {handle: deadline_of(handle) for handle in self._parked}
        held = held_tokens
        while self._parked:
            handle = min(
                self._parked,
                key=lambda h: (deadlines[h] is not None, deadlines[h] or 0.0),
            )
            read = self._parked[handle]
            deadline = deadlines[handle]
            if deadline is not None:
                if in_flight_pages >= self.window_pages:
                    self.stats.window_holds += 1
                    break
                if held + read.pages * self.page_size > self.budget_tokens:
                    self.stats.budget_holds += 1
                    break
            else:
                self.stats.dispatched_now += 1
            del self._parked[handle]
            in_flight_pages += read.pages
            held += read.pages * self.page_size
            waited = max(0.0, now - read.parked_at)
            self.stats.dispatched_total += 1
            self.stats.max_wait_s = max(self.stats.max_wait_s, waited)
            decisions.append(
                DispatchDecision(read=read, deadline=deadline, waited_s=waited)
            )
        return decisions
