"""What the next-turn prefetcher measures ("Four actions, three prices",
agent_cache/results/design_unified_1126/): the waiting queue, host eviction, admissions and
prefills as the scheduler thread shows them, kept as the prices and estimators of
next_turn_prices.py, with the fields of the HICACHE_EVT prices line the prefetcher logs every
interval.

The meter only reads the cache and the eviction strategy, and decides nothing: the prefetcher
binds its hooks, feeds it every pass and reads its board. Every time is a
time.perf_counter() value passed in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_estimators import L3LoadModel, Rate, residual_wait
from sglang.srt.mem_cache.next_turn_policy import RECOMPUTE_S_PER_TOKEN
from sglang.srt.mem_cache.next_turn_prices import (
    LAM_CEILING,
    MIN_ERROR_SAMPLES,
    AdmissionErrorTracker,
    PrefillCostModel,
    PriceBoard,
    landing_quantile,
    quantile_of,
)
from sglang.srt.mem_cache.next_use_eviction import (
    HOLDING_MIN_HORIZON_S,
    NextUseStrategy,
    ReturnChanceStrategy,
    queued_holding_rate,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

# The next eviction victim is priced at most this often: one priority per evictable host
# leaf each time. Arbitrary; as the prefetcher's probe interval.
VICTIM_PRICE_INTERVAL_S = 0.5


def _ms(seconds: Optional[float]) -> str:
    return "na" if seconds is None else str(round(seconds * 1000))


def ready_turns(
    waiting_queue: list[Req],
    *,
    tree_cache: UnifiedRadixCache,
    seen: dict[str, float],
    now: float,
) -> dict[str, float]:
    """rid -> when each queued turn with no read of its own under way was first seen so,
    `seen` being the previous pass's answer."""
    return {
        req.rid: seen.get(req.rid, now)
        for req in waiting_queue
        if not tree_cache.has_ongoing_prefetch(req.cache_request_handle)
    }


class PriceMeter:
    """The price board and the estimators beside it, fed from one prefetcher's passes."""

    def __init__(
        self,
        *,
        tree_cache: UnifiedRadixCache,
        page_size: int,
        ready_after_s: float,
        hold_margin_s: float,
        log_interval_s: float,
        read_deadline: Callable[[CacheRequestHandle], Optional[float]],
    ):
        self.board = PriceBoard()
        self.admission_error = AdmissionErrorTracker()
        self.prefill_cost = PrefillCostModel(seed_a=RECOMPUTE_S_PER_TOKEN)
        # rid -> when the queued turn was first seen with no read of its own, on every pass.
        # The prefetcher's copy follows only passes that hold a plan, so the two stay apart.
        self.ready_since: dict[str, float] = {}
        self._tree_cache = tree_cache
        self._page_size = page_size
        # A queued turn with no read of its own for this long is waiting for the GPU.
        self._ready_after_s = ready_after_s
        # The fixed hold margin, logged until the admission error has enough samples.
        self._hold_margin_s = hold_margin_s
        self._log_interval_s = log_interval_s
        self._read_deadline = read_deadline
        self._next_log = float("-inf")
        # Host eviction since the last pass: the tokens given up, and the highest holding
        # rate (per second) among the victims that have one.
        self._evicted_tokens = 0
        self._evicted_rate: Optional[float] = None
        # Per second, what a queued turn's history is worth when host eviction takes it.
        self._queued_rate = 1.0 / HOLDING_MIN_HORIZON_S
        # The price of the host leaf eviction would take next, and when it was priced.
        self._victim_price = 0.0
        self._victim_priced_at = float("-inf")
        # The read scheduler's window, budget, hold and write-hold counts at the last line.
        self._hold_counts_logged = (0, 0, 0, 0)
        strategy = tree_cache.tree_core.eviction_strategy
        # None under an eviction policy that holds no forecast: room then has no price.
        self._holding_strategy = (
            strategy
            if isinstance(strategy, (NextUseStrategy, ReturnChanceStrategy))
            else None
        )

    def follow(
        self,
        waiting_queue: list[Req],
        *,
        admitted: int,
        plans: int,
        sessions_waiting: int,
        rate: Optional[Rate],
        load: L3LoadModel,
        now: float,
    ) -> Optional[dict[str, object]]:
        """One scheduler pass: `admitted` turns left the queue for the GPU at `rate` and
        waiting_queue is what stays. sessions_waiting are the sessions whose next turn is
        not running; `plans` is only logged. Folds in the host eviction since the last
        pass. Returns the fields of a prices line once every interval, else None."""
        board = self.board
        self.ready_since = ready_turns(
            waiting_queue, tree_cache=self._tree_cache, seen=self.ready_since, now=now
        )
        ready_waiting = sum(
            now - since >= self._ready_after_s for since in self.ready_since.values()
        )
        board.observe_admissions(count=admitted, ready_left=ready_waiting > 0, now=now)
        board.observe_queue(
            ready_waiting=ready_waiting,
            sessions_waiting=sessions_waiting,
            arrival_rate=0.0 if rate is None else rate.per_s,
            now=now,
        )
        if self._holding_strategy is not None:
            self._queued_rate = queued_holding_rate(
                self._holding_strategy.queued_order,
                # The last arrival waits for everyone ahead of it; None without a rate.
                queue_wait_s=(
                    residual_wait(position=len(waiting_queue) - 1, rate=rate)
                    if waiting_queue
                    else 0.0
                ),
            )
        if self._evicted_tokens > 0:
            holding_rate, self._evicted_rate = self._evicted_rate, None
            tokens, self._evicted_tokens = self._evicted_tokens, 0
            board.observe_eviction(
                marginal_index=(
                    None
                    if holding_rate is None
                    else holding_rate * self.restore_s_per_token(load=load, now=now)
                ),
                tokens=tokens,
                now=now,
            )
        if self._log_interval_s <= 0 or now < self._next_log:
            return None
        self._next_log = now + self._log_interval_s
        return self._price_fields(
            queued=len(waiting_queue), plans=plans, load=load, now=now
        )

    def restore_s_per_token(self, *, load: L3LoadModel, now: float) -> float:
        """Seconds of TTFT a read costs per token it restores: its turn's wait at c and the
        device's time at what the late reads behind it cost. Times a holding rate it prices
        host room, so it holds no price of room: with kappa's write term lam feeds on lam."""
        board = self.board
        cache = self._tree_cache
        late = 0
        if cache.read_scheduler is not None:
            for read in cache.read_scheduler.parked_reads():
                deadline = self._read_deadline(read.handle)
                own_s = load.seconds(own_pages=read.pages, backlog_pages=0)
                # Past its latest start: dispatched now it still lands after its turn is due.
                late += deadline is None or deadline - own_s <= now
        board.observe_device(
            late_reads=late, pending_write_tokens=cache.pending_backup_tokens
        )
        return (
            (board.c(now) + board.kappa_of_reads(now))
            * load.read_s_per_page
            / self._page_size
        )

    def on_host_evicted(self, node, tokens: int) -> None:
        """Host eviction is taking node with `tokens` host tokens. Called inside it, so it
        only records, and asks the strategy nothing that changes how it ranks: the next
        pass folds the victims into the price of host room."""
        self._evicted_tokens += tokens
        if self._holding_strategy is None:
            return
        rate = self._holding_strategy.holding_rate(node, queued=self._queued_rate)
        if rate is not None and (
            self._evicted_rate is None or rate > self._evicted_rate
        ):
            self._evicted_rate = rate

    def next_victim_price(self, *, load: L3LoadModel, now: float) -> float:
        """The index of the holder host eviction would take next, as a price of host room:
        what the frontier's price will answer only once that eviction is made."""
        if self._holding_strategy is None:
            return 0.0
        if now - self._victim_priced_at >= VICTIM_PRICE_INTERVAL_S:
            self._victim_priced_at = now
            victim = self._tree_cache.next_host_victim()
            rate = (
                None
                if victim is None
                # A read staged for a turn not here must not take a queued turn's history,
                # however long that turn still waits: the top rate.
                else self._holding_strategy.holding_rate(
                    victim, queued=1.0 / HOLDING_MIN_HORIZON_S
                )
            )
            self._victim_price = (
                0.0
                if rate is None
                else min(
                    LAM_CEILING, rate * self.restore_s_per_token(load=load, now=now)
                )
            )
        return self._victim_price

    def observe_prefill(self, req: Req, *, at_once_s: float) -> None:
        """Feed the prefill fit a finished turn that was admitted within at_once_s: one
        that queued shared its passes with others, which is not the GPU cost of its tokens."""
        stats = req.time_stats
        waited = stats.forward_entry_time - stats.wait_queue_entry_time
        if (
            stats.forward_entry_time <= 0
            or stats.prefill_finished_time <= stats.forward_entry_time
            # Negative for a retracted request, which re-entered the queue after its prefill.
            or not 0.0 <= waited < at_once_s
        ):
            return
        total = len(req.origin_input_ids)
        self.prefill_cost.observe(
            new_tokens=total - req.cached_tokens,
            total_tokens=total,
            seconds=stats.prefill_finished_time - stats.forward_entry_time,
        )

    def observe_admission(
        self,
        *,
        predicted: Optional[float],
        wait: float,
        reader: bool,
        landed_after: Optional[float],
    ) -> None:
        """Feed the admission error with a turn's predicted and measured queue wait;
        landed_after is from its arrival to the landing of its last storage read."""
        # A retracted request re-enters the queue after its first forward: nothing to compare.
        if predicted is None or wait < 0.0:
            return
        if reader:
            if landed_after is not None:
                self.admission_error.observe(
                    predicted_wait_s=predicted,
                    wait_s=wait,
                    landed_after_arrival_s=landed_after,
                )
        elif predicted > 0.0:
            # 0 is an empty queue at arrival, by rule and not by the admission rate: such
            # turns would fill the bin the hold reads for waits under 2 s with zeros.
            self.admission_error.observe(predicted_wait_s=predicted, wait_s=wait)

    def _price_fields(
        self, *, queued: int, plans: int, load: L3LoadModel, now: float
    ) -> dict[str, object]:
        # One HICACHE_EVT prices line. waiting plus the batch lines' running count is the
        # users in session. Values stay within [\w./-]+, as the scripts read every line.
        board = self.board
        srs = self._tree_cache.read_scheduler
        restore_s = self.restore_s_per_token(load=load, now=now)
        lam = board.lam(now)
        a, b = self.prefill_cost.coefficients()
        errors = self.admission_error.pooled()
        q10, q50, q90 = (
            quantile_of(errors, q=q) if errors else None for q in (0.1, 0.5, 0.9)
        )
        # What a held read of 20K tokens would get: the landing quantile and its margin.
        q20k = landing_quantile(price=lam, tokens=20_000, lateness_weight=board.c(now))
        margin = (
            -quantile_of(errors, q=q20k)
            if len(errors) >= MIN_ERROR_SAMPLES
            else self._hold_margin_s
        )
        return dict(
            c=f"{board.c(now):.3f}",
            mu=f"{board.mu(now):.2f}",
            waiting=board.sessions_waiting,
            ready=board.ready_waiting,
            queued=queued,
            plans=plans,
            kappa=f"{board.kappa(now):.3f}",
            late=board.late_reads,
            parked=0 if srs is None else srs.parked_count(),
            pend_w=board.pending_write_tokens,
            lam=f"{lam:.3g}",
            evict_tok=board.take_evicted_tokens(),
            k_us=f"{restore_s * 1e6:.2f}",
            rec_a_us=f"{a * 1e6:.2f}",
            rec_b_ns=f"{b * 1e9:.3f}",
            rec_n=self.prefill_cost.samples,
            read_ms_page=f"{load.read_s_per_page * 1000:.2f}",
            err_n=len(errors),
            err_q10_ms=_ms(q10),
            err_q50_ms=_ms(q50),
            err_q90_ms=_ms(q90),
            rd_lag_ms=_ms(self.admission_error.reader_lag()),
            q20k=f"{q20k:.2f}",
            margin_ms=_ms(margin),
            **self._read_path_fields(),
        )

    def _read_path_fields(self) -> dict[str, object]:
        # The read path's measured lags, and how often each limit (window, budget, arrival
        # hold, write hold) kept a read parked since the previous line; "na" under fifo.
        names = ("win_holds", "budget_holds", "held", "write_holds")
        cache = self._tree_cache
        srs = cache.read_scheduler
        if srs is None:
            return dict.fromkeys(("lookup_ms", "lag_ms", "whold_ms") + names, "na")
        lags = cache.read_lags
        stats = srs.stats
        counts = (
            stats.window_holds,
            stats.budget_holds,
            stats.hold_skips,
            stats.write_holds,
        )
        logged, self._hold_counts_logged = self._hold_counts_logged, counts
        return {
            "lookup_ms": _ms(lags.lookup.seconds),
            "lag_ms": _ms(lags.fixed.seconds),
            "whold_ms": _ms(lags.write_hold.seconds),
            **{
                name: count - before
                for name, count, before in zip(names, counts, logged)
            },
        }
