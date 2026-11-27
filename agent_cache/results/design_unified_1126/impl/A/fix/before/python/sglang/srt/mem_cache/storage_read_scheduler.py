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
be waiting on it. An optional hold passes a read over until its start time. An
optional write hold stops timed reads while pending SSD writes hold at least
`write_hold_tokens` host tokens, until they fall to half of it: a pending write
keeps its host pages locked, so the write thread must catch up before reads can
find room. "Now" reads are never write-held.

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
    hold_skips: int = 0
    write_holds: int = 0
    max_wait_s: float = 0.0


class StorageReadScheduler:
    """Parked storage hits and the rule that hands them to the I/O thread."""

    def __init__(
        self,
        *,
        window_pages: int,
        budget_tokens: int,
        page_size: int,
        write_hold_tokens: int = 0,
    ):
        if window_pages < 1 or page_size < 1 or budget_tokens < 0:
            raise ValueError(
                f"window_pages {window_pages}, page_size {page_size} and "
                f"budget_tokens {budget_tokens} must be positive"
            )
        if write_hold_tokens < 0:
            raise ValueError(f"write_hold_tokens {write_hold_tokens} is negative")
        self.window_pages = window_pages
        self.budget_tokens = budget_tokens
        self.page_size = page_size
        # 0 disables the write hold; the resume level at half is arbitrary.
        self.write_hold_tokens = write_hold_tokens
        self.write_resume_tokens = write_hold_tokens // 2
        self.write_held = False
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

    def parked_reads(self) -> list[ParkedRead]:
        return list(self._parked.values())

    def pages_due_by(
        self,
        deadline: Optional[float],
        *,
        deadline_of: Callable[[Hashable], Optional[float]],
    ) -> int:
        """Pages of the parked reads that dispatch orders no later than a read with this
        deadline (None = needed now): the "now" reads, and for a timed read the timed
        ones due no later."""
        pages = 0
        for handle, read in self._parked.items():
            due = deadline_of(handle)
            if due is None or (deadline is not None and due <= deadline):
                pages += read.pages
        return pages

    def pages_ahead_of(
        self, handle: Hashable, *, deadline_of: Callable[[Hashable], Optional[float]]
    ) -> int:
        """Pages of the parked reads that dispatch orders before this parked one: "now"
        reads first, then by deadline, park order among equals."""

        def order(deadline: Optional[float]) -> tuple[bool, float]:
            return (deadline is not None, deadline or 0.0)

        own = order(deadline_of(handle))
        pages = 0
        parked_earlier = True
        for other, read in self._parked.items():
            if other == handle:
                parked_earlier = False
                continue
            ahead = order(deadline_of(other))
            if ahead < own or (ahead == own and parked_earlier):
                pages += read.pages
        return pages

    def dispatch(
        self,
        *,
        now: float,
        deadline_of: Callable[[Hashable], Optional[float]],
        in_flight_pages: int,
        held_tokens: int,
        not_before_of: Optional[
            Callable[[Hashable, ParkedRead, int], Optional[float]]
        ] = None,
    ) -> list[DispatchDecision]:
        """Pop the reads to launch now, earliest deadline first.

        `in_flight_pages` is what the I/O thread already holds; `held_tokens`
        is the host memory charged to prefetching (the in-flight reads plus
        the pinned spans), so a read is charged once, here on top of it.
        `not_before_of(handle, read, ahead_pages)` gives a read's start time. A held
        read is passed over, unlike the window and budget holds that stop the loop,
        so it never blocks a due read, but its pages still count as ahead of the next.
        The write hold is whatever update_write_hold() last set.
        """
        decisions: list[DispatchDecision] = []
        if not self._parked:
            return decisions
        deadlines = {handle: deadline_of(handle) for handle in self._parked}
        order = sorted(
            self._parked,
            key=lambda h: (deadlines[h] is not None, deadlines[h] or 0.0),
        )
        held = held_tokens
        ahead_pages = in_flight_pages
        for handle in order:
            read = self._parked[handle]
            deadline = deadlines[handle]
            if not_before_of is not None:
                not_before = not_before_of(handle, read, ahead_pages)
                ahead_pages += read.pages
                if not_before is not None and now < not_before:
                    self.stats.hold_skips += 1
                    continue
            if deadline is not None:
                if self.write_held:
                    self.stats.write_holds += 1
                    break
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

    def update_write_hold(self, write_backlog_tokens: int) -> bool:
        """Apply the write hold's hysteresis to the host memory pending SSD writes hold;
        True if the hold flipped. The cache calls it every pass, before dispatch, so a
        crossing while nothing is parked still counts."""
        if self.write_hold_tokens <= 0:
            return False
        was_held = self.write_held
        if write_backlog_tokens >= self.write_hold_tokens:
            self.write_held = True
        elif write_backlog_tokens <= self.write_resume_tokens:
            self.write_held = False
        return self.write_held != was_held
