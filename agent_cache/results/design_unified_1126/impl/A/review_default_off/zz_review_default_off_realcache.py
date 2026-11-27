"""Review check (default-off lens), throwaway: with only the prices log on, the meter's reads
resolve against the real UnifiedRadixCache (attribute names, not the fakes'), fifo and
deadline read order, and a prices line is logged. Also a smoke of the new real-cache
accessors the switched code calls."""

import logging
from types import SimpleNamespace

import test_next_turn_prefetch as fixtures

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.storage_read_scheduler import StorageReadScheduler

LOGGER = "sglang.srt.mem_cache.next_turn_prefetch"


class Lines(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run(*, deadline_order):
    cache = fixtures.cpu_unified_cache()
    cache.cache_controller = SimpleNamespace(
        storage_read_ms_per_page=2.5, prefetch_rate_limited=lambda: False
    )
    if deadline_order:
        cache.read_scheduler = StorageReadScheduler(
            window_pages=512, budget_tokens=10**9, page_size=2
        )
    handler = Lines()
    logger = logging.getLogger(LOGGER)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(1.0):
            ntp = NextTurnPrefetcher(
                mode="on",
                tree_cache=cache,
                page_size=2,
                arrival_gate="on",
                arrival_hold="on" if deadline_order else "off",
            )
        assert ntp._meter is not None and cache.tree_core.host_eviction_observer is not None
        assert (cache.read_lags is not None) == deadline_order
        ntp.on_turn_committed(
            req=fixtures.make_req("s-t0", kv_hints=fixtures.hints("s-t1")),
            key=fixtures.key_of(8),
            rotation_tail_declined=False,
            now=0.0,
        )
        queue = [fixtures.make_req(f"q{i}", entered=0.0) for i in range(3)]
        if deadline_order:
            cache.read_scheduler.park(
                handle=CacheRequestHandle("q2", 0), operation=None, pages=4, now=0.0
            )
        for step in range(8):
            ntp.tick(waiting_queue=queue, now=0.5 * step)
        queue[0].time_stats.forward_entry_time = 4.2
        ntp.tick(waiting_queue=queue[1:], now=4.5)
    finally:
        logger.removeHandler(handler)
    prices = [line for line in handler.lines if line.startswith("HICACHE_EVT prices ")]
    assert len(prices) >= 4, handler.lines
    return cache, prices[-1]


def test_the_meter_reads_the_real_cache_under_fifo_order():
    _, line = run(deadline_order=False)
    print(line)
    assert " parked=0 " in line and " lookup_ms=na " in line


def test_the_meter_reads_the_real_cache_under_the_deadline_order():
    cache, line = run(deadline_order=True)
    print(line)
    assert " parked=1 " in line and " late=1 " in line
    # The accessors the switched code calls, on the real cache.
    handle = CacheRequestHandle("q2", 0)
    assert cache.storage_read_line_pages(None) == 4
    assert cache.next_host_victim() is None
    assert cache.parked_read_span(handle) is None  # parked in the scheduler only
    assert cache.drop_parked_read(handle) is False
