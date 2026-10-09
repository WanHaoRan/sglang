"""Estimators for the oracle next-turn prefetch (agent_cache/ORACLE_PLAN.md, section 2).

For a hinted next turn they predict when it reaches the scheduler, how long it then
waits in the queue, and how long an SSD read of its evicted pages takes; the read is
started at predicted admission minus load time. Every time is a time.perf_counter()
value passed in by the caller, so nothing here reads a clock.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import msgspec

# Weight of a new sample in both EWMAs: roughly an 8-sample memory, picked, not tuned.
EWMA_ALPHA = 1 / 8

# Offsets outside this range are client stalls or mispairings, not delivery delay
# (campaign 11 has 138 s and 162 s outliers); they are dropped, never clipped.
ARRIVAL_OFFSET_RANGE_S = (0.0, 5.0)

# Campaign 11 three_tier_wc: 33 of 117 one-minute windows held fewer than 5 admissions.
RATE_WINDOW_S = 60.0
RATE_FALLBACK_WINDOW_S = 300.0
MIN_ADMISSIONS = 5


class ArrivalOffset(msgspec.Struct):
    """EWMA of how late a hinted turn reaches the scheduler: its enqueue time minus
    the previous turn's commit minus the hinted gap (reply delivery, client
    bookkeeping, the HTTP request and tokenization)."""

    # Seed between the sparse (campaign 10, p50 0.088 s) and the dense
    # (campaign 11, p50 0.125 s) median offsets.
    seconds: float = 0.10
    samples: int = 0
    discarded: int = 0

    def observe(self, offset_s: float) -> bool:
        """Fold in one measured offset; False (and counted) when it is out of range."""
        low, high = ARRIVAL_OFFSET_RANGE_S
        if not low <= offset_s <= high:
            self.discarded += 1
            return False
        self.seconds += EWMA_ALPHA * (offset_s - self.seconds)
        self.samples += 1
        return True

    def predict_arrival(self, *, t_commit: float, gap_s: float) -> float:
        return t_commit + gap_s + self.seconds


class Rate(msgspec.Struct, frozen=True):
    """Admissions per second and the window they were counted over."""

    per_s: float
    window_s: float


class AdmissionRate(msgspec.Struct):
    """Recent admissions (the rate the queue drains at) and an EWMA of queue waits."""

    # Tick times of the admissions inside the fallback window, oldest first.
    stamps: deque[float] = msgspec.field(default_factory=deque)
    # None until the first sample. Requests credited with their own SSD read are left
    # out: their wait includes the read the oracle exists to hide.
    wait_ewma: Optional[float] = None

    def record_admission(self, *, now: float, wait_s: float, storage_hit: bool) -> None:
        """`now` is the detecting tick's clock and must not decrease between calls."""
        self.stamps.append(now)
        while self.stamps[0] <= now - RATE_FALLBACK_WINDOW_S:
            self.stamps.popleft()
        if storage_hit:
            return
        if self.wait_ewma is None:
            self.wait_ewma = wait_s
        else:
            self.wait_ewma += EWMA_ALPHA * (wait_s - self.wait_ewma)

    def rate(self, now: float) -> Optional[Rate]:
        """Over the last minute, or the last five when the minute holds fewer than
        MIN_ADMISSIONS; None when the five minutes do too."""
        recent = total = 0
        for stamp in reversed(self.stamps):
            if stamp <= now - RATE_FALLBACK_WINDOW_S:
                break
            total += 1
            if stamp > now - RATE_WINDOW_S:
                recent += 1
        if recent >= MIN_ADMISSIONS:
            return Rate(per_s=recent / RATE_WINDOW_S, window_s=RATE_WINDOW_S)
        if total >= MIN_ADMISSIONS:
            return Rate(
                per_s=total / RATE_FALLBACK_WINDOW_S, window_s=RATE_FALLBACK_WINDOW_S
            )
        return None


def residual_wait(*, position: int, rate: Optional[Rate]) -> Optional[float]:
    """Wait left for the request at `position` (0 = head) of the FCFS queue: the ones
    ahead drain at `rate`, plus half an inter-admission gap; None without a rate."""
    if rate is None:
        return None
    return (position + 0.5) / rate.per_s


def wait_at_arrival(
    *, queue_len: int, rate: Optional[Rate], wait_ewma: Optional[float]
) -> tuple[Optional[float], Optional[float]]:
    """(by queue length, by recent waits) for a turn arriving behind `queue_len`
    requests. An empty queue admits on the next pass, so both are 0."""
    if queue_len == 0:
        return 0.0, 0.0
    return residual_wait(position=queue_len, rate=rate), wait_ewma


class L3LoadModel(msgspec.Struct, frozen=True, kw_only=True):
    """Seconds from issuing an SSD read until its pages are inserted in host memory.

    Dense-load seeds per 64-token page of Qwen3-30B-A3B bf16 KV (6 MiB), from 1,199
    campaign-11 three_tier_wc reads; at sparse load they overestimate by ~0.8 s.
    """

    # OLS of s2h_io time against pages: ms = 43.5 + 2.477 x pages (~2.4 GB/s).
    read_fixed_s: float = 0.044
    read_s_per_page: float = 0.0025
    # Pages queued ahead on the one FIFO read thread. Campaign 7's backlog drained at
    # ~3.1 ms/page (reads overlapping SSD writes are slower); revisit after E1.
    backlog_s_per_page: float = 0.0025
    # Lookup answered -> I/O start with an idle read thread (sparse load ~0.03 s).
    alloc_lag_s: float = 0.5
    # I/O end -> pages inserted in the tree (sparse load ~0.02 s).
    drain_lag_s: float = 0.36

    def seconds(self, *, own_pages: int, backlog_pages: int) -> float:
        return (
            self.alloc_lag_s
            + self.read_fixed_s
            + self.read_s_per_page * own_pages
            + self.backlog_s_per_page * backlog_pages
            + self.drain_lag_s
        )


def trigger_time(*, admission_at: float, load_s: float, margin_s: float = 0.0) -> float:
    """Start the read here so its pages land as the request is admitted."""
    return admission_at - load_s - margin_s
