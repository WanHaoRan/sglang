"""Oracle next-turn prefetch from storage (agent_cache/ORACLE_PLAN.md).

A client hints each turn with its session's next request id and the gap before it
is sent (kv_hints action sglang.next_turn v1). When a hinted turn commits, a plan
keeps the committed key and the next turn's predicted arrival. Every scheduler
tick the prefetcher follows the waiting queue, whose arrivals and admissions feed
the estimators, and works out when an SSD read of the key's evicted pages must
start to land as the turn is admitted. Mode observe logs those decisions as
HICACHE_EVT ntp_* lines and changes nothing; mode on also acts on them. Case 1,
before the turn arrives, stages the read under the next request's own handle;
case 2, while it waits in the queue, has the Scheduler re-issue its prefetch.

Under --hicache-storage-read-order deadline (agent_cache/ORACLE_V2_PLAN.md 4.4)
the prefetcher stops timing reads: it issues a plan's lookup as soon as the span
is missing and the predicted admission is within NTP_LEAD_HORIZON_S, labels the
parked read with its predicted admission through read_deadline(), and the cache's
read scheduler decides the dispatch order. With --radix-eviction-policy next_use it
also registers each committed turn's tail node with the predicted next admission.

The decisions use this rank's wall clock, so create() refuses multi-rank setups.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
from typing import TYPE_CHECKING, Annotated, Optional

import msgspec
import msgspec.structs

from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.next_turn_estimators import (
    RATE_FALLBACK_WINDOW_S,
    AdmissionRate,
    ArrivalOffset,
    L3LoadModel,
    Rate,
    residual_wait,
    trigger_time,
    wait_at_arrival,
)
from sglang.srt.mem_cache.next_use_eviction import NextUseStrategy
from sglang.srt.mem_cache.radix_cache import RadixKey

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.kv_hints import KvHintsEnvelope
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

NEXT_TURN_ACTION_TYPE = "sglang.next_turn"
NEXT_TURN_ACTION_VERSION = "1"

# Ticks run once per scheduler pass (well under 1 ms when idle); a plan's
# residency is probed at most this often.
PROBE_INTERVAL_S = 0.5
# A pending plan expires this long after its predicted arrival, or one gap later
# if that is longer; real arrival offsets stay under 5 s.
MIN_EXPIRY_S = 60.0
# Plan decision 2 (open until E1): start the read exactly at admission - load.
TRIGGER_MARGIN_S = 0.0
# Reads per queued request: under heavy eviction a refetch for one request can
# evict the next one's pages, so re-evicted pages are fetched again only once.
MAX_REFETCHES = 2
# Deadline read order: a lookup is issued at most this long before the predicted
# admission, and at most this many parked reads may wait for dispatch.
# 120 s is about three drains of the prefetch cap (42 s at 302 pages/s, measured
# in campaign 15); 256 is arbitrary, it bounds the anchor locks parked lookups hold.
NTP_LEAD_HORIZON_S = 120.0
NTP_MAX_PARKED = 256


class NextTurnHint(msgspec.Struct, frozen=True):
    """Payload of the sglang.next_turn v1 action."""

    session: str
    next_rid: str
    next_gap_s: Annotated[float, msgspec.Meta(ge=0)]


def read_next_turn_hint(
    kv_hints: Optional[KvHintsEnvelope],
) -> Optional[NextTurnHint]:
    """The request's next-turn hint; other action types are not ours to read."""
    if kv_hints is None:
        return None
    for action in kv_hints.actions:
        if (
            action.action_type != NEXT_TURN_ACTION_TYPE
            or action.action_version != NEXT_TURN_ACTION_VERSION
        ):
            continue
        try:
            return msgspec.convert(action.payload, NextTurnHint)
        except msgspec.ValidationError as e:
            logger.warning("Ignoring a malformed %s hint: %s", NEXT_TURN_ACTION_TYPE, e)
            return None
    return None


class RefetchDecision(msgspec.Struct, frozen=True):
    """Why a queued request was handed back for a re-issued prefetch."""

    missing_tokens: int
    position: int
    wait_s: float
    load_s: float


class TurnPlan(msgspec.Struct):
    """A committed turn's key, waiting for its hinted next turn to be admitted."""

    next_rid: str
    session: str
    key: RadixKey
    t_commit: float
    gap_s: float
    predicted_arrival: float
    expires_at: float
    next_check: float
    # Queue entry time; None while the next turn has not reached the scheduler.
    arrival_t: Optional[float] = None
    pred_wait_ql: Optional[float] = None
    pred_wait_ewma: Optional[float] = None
    last_probe_t: float = -math.inf
    # At the last probe: tokens of the key in neither device nor host memory.
    missing_tokens: int = 0
    # 0 until the trigger time passes with enough pages to read; then the case.
    due_case: int = 0
    trigger_at: Optional[float] = None
    predicted_admission: Optional[float] = None
    # Case 1 issued a read under the next request's handle before it arrived.
    staged: bool = False
    # Case 2: re-issued prefetches that started a read, and the latest attempt.
    refetches: int = 0
    last_refetch_t: float = -math.inf
    refetch_capped: bool = False
    # Handed to the Scheduler this tick; reported back by on_refetch_result.
    pending_refetch: Optional[RefetchDecision] = None


def _handle(plan: TurnPlan) -> CacheRequestHandle:
    # Req's first attempt handle: the arriving request joins a staged read by it.
    return CacheRequestHandle(rid=plan.next_rid, attempt_id=0)


def _ms(seconds: Optional[float]) -> str:
    return "na" if seconds is None else str(round(seconds * 1000))


def _log_event(kind: str, rid: str, **fields) -> None:
    # Parsed by agent_cache/scripts: rid first, values match [\w./-]+.
    logger.info(
        "HICACHE_EVT %s rid=%s %s",
        kind,
        rid,
        " ".join(f"{name}={value}" for name, value in fields.items()),
    )


class NextTurnPrefetcher:
    """Plans an SSD read per hinted next turn; the Scheduler ticks it every pass."""

    def __init__(self, *, mode: str, tree_cache: UnifiedRadixCache, page_size: int):
        self._mode = mode
        self._tree_cache = tree_cache
        self._page_size = page_size
        self._offset = ArrivalOffset()
        self._admissions = AdmissionRate()
        self._load = L3LoadModel()
        self._plans: dict[str, TurnPlan] = {}
        # (next_check, seq, rid) of plans not yet arrived; stale entries are skipped.
        self._pending: list[tuple[float, int, str]] = []
        self._seq = itertools.count()
        # The previous tick's waiting queue, to tell admissions and aborts apart.
        self._queued: dict[str, Req] = {}
        # Admission tick by rid, oldest first: a retracted request re-enters the
        # queue and leaves it again, but counts once.
        self._admitted: dict[str, float] = {}
        # Deadline read order: the cache parks hits and asks read_deadline().
        self._deadline_mode = tree_cache.read_scheduler is not None
        self._last_now = -math.inf
        self._last_rate: Optional[Rate] = None
        self._positions: dict[str, int] = {}
        if self._deadline_mode:
            tree_cache.read_deadline_fn = self.read_deadline
        # Next-use eviction: the strategy's registry, when that policy is on.
        strategy = tree_cache.tree_core.eviction_strategy
        self._next_use = strategy.registry if isinstance(strategy, NextUseStrategy) else None
        self._session_tail: dict[str, int] = {}
        self._rid_session: dict[str, str] = {}

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        tree_cache,
        page_size: int,
        enable_hicache_storage: bool,
        tp_size: int,
        pp_size: int,
        enable_dp_attention: bool,
        disaggregation_mode: DisaggregationMode,
    ) -> NextTurnPrefetcher:
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
            UnifiedTreeCore,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        unmet = []
        if not enable_hicache_storage:
            unmet.append("a HiCache storage backend")
        if not isinstance(tree_cache, UnifiedRadixCache) or tree_cache.disable:
            unmet.append(f"UnifiedRadixCache (got {type(tree_cache).__name__})")
        else:
            if tree_cache.host_memory_mode != "cache":
                unmet.append("--hicache-host-memory-mode cache")
            # The residency probe is implemented by the Python tree core only.
            if not isinstance(tree_cache.tree_core, UnifiedTreeCore):
                unmet.append("the python tree core")
            elif tree_cache.tree_core.is_eagle:
                unmet.append("no EAGLE (bigram keys)")
        if tp_size != 1 or pp_size != 1 or enable_dp_attention:
            unmet.append("TP = PP = 1 without DP attention")
        if disaggregation_mode != DisaggregationMode.NULL:
            unmet.append("no PD disaggregation")
        if unmet:
            raise ValueError(
                f"--hicache-storage-next-turn-prefetch {mode} needs " + ", ".join(unmet)
            )
        logger.info("HiCache next-turn prefetch: mode=%s", mode)
        return cls(mode=mode, tree_cache=tree_cache, page_size=page_size)

    def on_turn_committed(
        self,
        *,
        req: Req,
        key: RadixKey,
        rotation_tail_declined: bool,
        now: float,
        tail_node_id: Optional[int] = None,
    ) -> None:
        """Called once a finished turn's KV is in the tree; keeps the key only."""
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        if rotation_tail_declined or not req.finished():
            return
        # The session recorded for this rid by the previous turn's hint, if any.
        session = self._rid_session.pop(req.rid, None)
        # A running request that is aborted also finishes through insertion.
        if len(key) == 0 or isinstance(req.finished_reason, FINISH_ABORT):
            self._end_session(rid=req.rid, session=session, now=now)
            return
        hint = read_next_turn_hint(req.kv_hints)
        if hint is None:
            # The session ended (or the client sends no hints): evict it first.
            self._register_next_use(
                rid=req.rid,
                session=session,
                tail_node_id=tail_node_id,
                next_use=None,
                now=now,
            )
            if session is not None:
                self._session_tail.pop(session, None)
            return
        predicted = self._offset.predict_arrival(t_commit=now, gap_s=hint.next_gap_s)
        wait, _ = self._wait_after_arrival(
            queue_len=len(self._queued), rate=self._last_rate
        )
        self._register_next_use(
            rid=req.rid,
            session=hint.session,
            tail_node_id=tail_node_id,
            next_use=predicted + (wait or 0.0),
            now=now,
        )
        self._rid_session[hint.next_rid] = hint.session
        plan = TurnPlan(
            next_rid=hint.next_rid,
            session=hint.session,
            key=key,
            t_commit=now,
            gap_s=hint.next_gap_s,
            predicted_arrival=predicted,
            expires_at=predicted + max(MIN_EXPIRY_S, hint.next_gap_s),
            next_check=now,
        )
        self._plans[plan.next_rid] = plan
        heapq.heappush(self._pending, (now, next(self._seq), plan.next_rid))
        _log_event(
            "ntp_plan",
            plan.next_rid,
            sess=plan.session,
            gap_ms=_ms(plan.gap_s),
            chain=len(key),
            pred_arr_ms=_ms(predicted - now),
        )

    def tick(self, *, waiting_queue: list[Req], now: float) -> list[Req]:
        """One scheduler pass: follow the queue and advance every plan that is due
        a check. Returns the queued requests to prefetch again (none in observe)."""
        queued = {req.rid: req for req in waiting_queue}
        for rid, req in self._queued.items():
            if rid not in queued:
                self._on_departure(req, now=now)
        self._queued = queued
        self._last_now = now
        self._positions = {req.rid: position for position, req in enumerate(waiting_queue)}
        if self._deadline_mode:
            self._refresh_load_model()
        if not self._plans:
            return []

        rate = self._admissions.rate(now)
        self._last_rate = rate
        backlog_pages = self._tree_cache.storage_read_backlog_pages()
        refetch = []
        for position, req in enumerate(waiting_queue):
            plan = self._plans.get(req.rid)
            if plan is None:
                continue
            if plan.arrival_t is None:
                self._on_arrival(
                    plan,
                    req=req,
                    position=position,
                    queue_len=len(waiting_queue),
                    rate=rate,
                    now=now,
                )
            if self._check_queued(
                plan,
                req=req,
                position=position,
                rate=rate,
                backlog_pages=backlog_pages,
                now=now,
            ):
                refetch.append(req)
        self._advance_pending(
            queue_len=len(waiting_queue),
            rate=rate,
            backlog_pages=backlog_pages,
            now=now,
        )
        return refetch

    def on_refetch_result(self, *, req: Req, issued: bool) -> None:
        """The Scheduler re-issued the prefetch tick() asked for; `issued` says
        whether a read started (a decline or a gate keeps the plan armed)."""
        plan = self._plans.get(req.rid)
        if plan is None or plan.pending_refetch is None:
            return
        decision, plan.pending_refetch = plan.pending_refetch, None
        if issued:
            plan.refetches += 1
        _log_event(
            "ntp_refetch",
            req.rid,
            issued=int(issued),
            shrink=decision.missing_tokens,
            pos=decision.position,
            resid_ms=_ms(decision.wait_s),
            load_ms=_ms(decision.load_s),
        )

    def _on_departure(self, req: Req, *, now: float) -> None:
        plan = self._plans.pop(req.rid, None)
        stats = req.time_stats
        if stats.forward_entry_time <= 0:
            if plan is not None:
                self._end_session(rid=req.rid, session=plan.session, now=now)
                _log_event(
                    "ntp_expire", req.rid, staged=int(plan.staged), reason="abort"
                )
            return
        if req.rid not in self._admitted:
            self._record_admission(req, now=now)
        if plan is None:
            return
        wait = stats.forward_entry_time - stats.wait_queue_entry_time
        due = plan.due_case > 0
        _log_event(
            "ntp_admit",
            req.rid,
            wait_ms=_ms(wait),
            pred_wait_ql_ms=_ms(plan.pred_wait_ql),
            pred_wait_ewma_ms=_ms(plan.pred_wait_ewma),
            adm_err_ms=_ms(
                stats.forward_entry_time - plan.predicted_admission if due else None
            ),
            trig_lead_ms=_ms(
                stats.forward_entry_time - plan.trigger_at if due else None
            ),
            case=plan.due_case,
        )

    def _record_admission(self, req: Req, *, now: float) -> None:
        self._admitted[req.rid] = now
        # Oldest first (insertion order); the entry just added is never pruned.
        while True:
            oldest = next(iter(self._admitted))
            if self._admitted[oldest] > now - RATE_FALLBACK_WINDOW_S:
                break
            del self._admitted[oldest]
        stats = req.time_stats
        self._admissions.record_admission(
            now=now,
            wait_s=stats.forward_entry_time - stats.wait_queue_entry_time,
            storage_hit=req.storage_hit_length > 0,
        )

    def _on_arrival(
        self,
        plan: TurnPlan,
        *,
        req: Req,
        position: int,
        queue_len: int,
        rate: Optional[Rate],
        now: float,
    ) -> None:
        arrival = req.time_stats.wait_queue_entry_time or now
        offset = arrival - plan.t_commit - plan.gap_s
        self._offset.observe(offset)
        plan.arrival_t = arrival
        self._register_next_use(
            rid=plan.next_rid,
            session=plan.session,
            tail_node_id=self._session_tail.get(plan.session),
            next_use=now,
            now=now,
        )
        plan.pred_wait_ql, plan.pred_wait_ewma = wait_at_arrival(
            queue_len=position, rate=rate, wait_ewma=self._admissions.wait_ewma
        )
        _log_event(
            "ntp_arrive",
            plan.next_rid,
            arr_err_ms=_ms(arrival - plan.predicted_arrival),
            delta_ms=_ms(offset),
            pos=position,
            qlen=queue_len,
        )

    def _check_queued(
        self,
        plan: TurnPlan,
        *,
        req: Req,
        position: int,
        rate: Optional[Rate],
        backlog_pages: int,
        now: float,
    ) -> bool:
        """Case 2: the turn waits in the queue and its pages must be back by
        admission. True asks the Scheduler to re-issue the request's prefetch."""
        wait = residual_wait(position=position, rate=rate)
        est = None if rate is None else f"res{rate.window_s:.0f}"
        if wait is None and position <= 1:
            # No rate to drain the queue with: the head is the only safe bet.
            wait, est = 0.0, "head"
        if wait is None:
            return False
        admission_at = now + wait
        baseline = max(len(plan.key), req.storage_prefetch_last_match_len or 0)
        # Not even a read of the whole key would need to start yet: skip the probe.
        if not self._deadline_mode and now < trigger_time(
            admission_at=admission_at,
            load_s=self._load_seconds(baseline, backlog_pages),
            margin_s=TRIGGER_MARGIN_S,
        ):
            return False
        if now - plan.last_probe_t >= PROBE_INTERVAL_S:
            plan.missing_tokens = baseline - self._resident_tokens(plan, now=now)
        load_s = self._load_seconds(plan.missing_tokens, backlog_pages)
        trigger_at = trigger_time(
            admission_at=admission_at, load_s=load_s, margin_s=TRIGGER_MARGIN_S
        )
        if plan.missing_tokens < self._tree_cache.prefetch_threshold or (
            not self._deadline_mode and now < trigger_at
        ):
            return False
        if plan.due_case == 0:
            self._mark_due(
                plan,
                case=2,
                backlog_pages=backlog_pages,
                load_s=load_s,
                wait_s=wait,
                admission_at=admission_at,
                trigger_at=trigger_at,
                est=est,
                now=now,
            )
        if self._mode != "on":
            return False
        return self._wants_refetch(
            plan,
            req=req,
            baseline=baseline,
            position=position,
            wait_s=wait,
            load_s=load_s,
            now=now,
        )

    def _wants_refetch(
        self,
        plan: TurnPlan,
        *,
        req: Req,
        baseline: int,
        position: int,
        wait_s: float,
        load_s: float,
        now: float,
    ) -> bool:
        cache = self._tree_cache
        # Any read for the request (its own, a re-query or a staged one) joins
        # admission's wait; a second one would fetch the same pages twice.
        if (
            cache.has_ongoing_prefetch(req.cache_request_handle)
            or now - plan.last_refetch_t < PROBE_INTERVAL_S
            or plan.refetch_capped
            or cache.cache_controller.prefetch_rate_limited()
        ):
            return False
        # The last probe may predate a read that has since landed.
        plan.missing_tokens = baseline - self._resident_tokens(plan, now=now)
        if plan.missing_tokens < cache.prefetch_threshold:
            return False
        if plan.refetches > 0:
            # The fetched pages were evicted again before admission.
            plan.refetch_capped = plan.refetches >= MAX_REFETCHES
            _log_event(
                "ntp_reevict",
                req.rid,
                evicted=plan.missing_tokens,
                refetches=plan.refetches,
                capped=int(plan.refetch_capped),
            )
            if plan.refetch_capped:
                return False
        plan.last_refetch_t = now
        plan.pending_refetch = RefetchDecision(
            missing_tokens=plan.missing_tokens,
            position=position,
            wait_s=wait_s,
            load_s=load_s,
        )
        return True

    def _advance_pending(
        self,
        *,
        queue_len: int,
        rate: Optional[Rate],
        backlog_pages: int,
        now: float,
    ) -> None:
        # Pop every due check first: one that asks for the next tick comes back
        # with next_check = now and must not run again in this one.
        due = []
        while self._pending and self._pending[0][0] <= now:
            due.append(heapq.heappop(self._pending))
        for check_at, _, rid in due:
            plan = self._plans.get(rid)
            if (
                plan is None
                or plan.arrival_t is not None
                or plan.next_check != check_at
            ):
                continue
            next_check = self._check_pending(
                plan,
                queue_len=queue_len,
                rate=rate,
                backlog_pages=backlog_pages,
                now=now,
            )
            if next_check is None:
                del self._plans[rid]
                continue
            plan.next_check = next_check
            heapq.heappush(self._pending, (next_check, next(self._seq), rid))

    def _check_pending(
        self,
        plan: TurnPlan,
        *,
        queue_len: int,
        rate: Optional[Rate],
        backlog_pages: int,
        now: float,
    ) -> Optional[float]:
        """Case 1: the turn has not arrived. Returns the next check time, or None
        once the plan is gone."""
        if plan.staged and not self._tree_cache.has_ongoing_prefetch(_handle(plan)):
            # An ended staged read may arm a paced retry for this rid, which is not
            # queued; pop_ready would then scan the queue for it every pass.
            self._tree_cache.storage_prefetch_retries.cancel(plan.next_rid)
        if now >= plan.expires_at:
            return self._expire(plan, now=now)
        recheck = min(now + PROBE_INTERVAL_S, plan.expires_at)
        wait, est = self._wait_after_arrival(queue_len=queue_len, rate=rate)
        if wait is None:
            return recheck
        admission_at = plan.predicted_arrival + wait
        if self._deadline_mode:
            return self._check_pending_deadline(
                plan,
                admission_at=admission_at,
                wait=wait,
                est=est,
                backlog_pages=backlog_pages,
                now=now,
                recheck=recheck,
            )
        window = trigger_time(
            admission_at=admission_at,
            load_s=self._load_seconds(len(plan.key), backlog_pages),
            margin_s=TRIGGER_MARGIN_S,
        )
        if now < window:
            return min(window, recheck)
        if now - plan.last_probe_t >= PROBE_INTERVAL_S:
            plan.missing_tokens = len(plan.key) - self._resident_tokens(plan, now=now)
        load_s = self._load_seconds(plan.missing_tokens, backlog_pages)
        trigger_at = trigger_time(
            admission_at=admission_at, load_s=load_s, margin_s=TRIGGER_MARGIN_S
        )
        enough = plan.missing_tokens >= self._tree_cache.prefetch_threshold
        if plan.due_case == 0:
            if not enough:
                return recheck
            if now < trigger_at:
                # Come back at the trigger itself, not up to a probe interval late.
                return min(trigger_at, recheck)
            self._mark_due(
                plan,
                case=1,
                backlog_pages=backlog_pages,
                load_s=load_s,
                wait_s=wait,
                admission_at=admission_at,
                trigger_at=trigger_at,
                est=est,
                now=now,
            )
        if self._mode != "on" or plan.staged or not enough:
            return recheck
        if self._tree_cache.cache_controller.prefetch_rate_limited():
            # Declining here would arm a paced retry for a rid that is not queued.
            return now
        self._stage(plan)
        return recheck

    def _check_pending_deadline(
        self,
        plan: TurnPlan,
        *,
        admission_at: float,
        wait: float,
        est: str,
        backlog_pages: int,
        now: float,
        recheck: float,
    ) -> float:
        """Case 1 under the deadline order: issue the lookup within the lead
        horizon; the cache's read scheduler orders the dispatch."""
        if admission_at - now > NTP_LEAD_HORIZON_S:
            return min(admission_at - NTP_LEAD_HORIZON_S, recheck)
        if now - plan.last_probe_t >= PROBE_INTERVAL_S:
            plan.missing_tokens = len(plan.key) - self._resident_tokens(plan, now=now)
        enough = plan.missing_tokens >= self._tree_cache.prefetch_threshold
        load_s = self._load_seconds(plan.missing_tokens, backlog_pages)
        if plan.due_case == 0 and enough:
            self._mark_due(
                plan,
                case=1,
                backlog_pages=backlog_pages,
                load_s=load_s,
                wait_s=wait,
                admission_at=admission_at,
                trigger_at=now,
                est=est,
                now=now,
            )
        if self._mode != "on" or plan.staged or not enough:
            return recheck
        cache = self._tree_cache
        if (
            cache.read_scheduler.parked_count() >= NTP_MAX_PARKED
            or cache.cache_controller.prefetch_rate_limited()
        ):
            # Declining in the cache would arm a paced retry for a rid that is
            # not queued; wait here instead.
            return min(now + PROBE_INTERVAL_S, recheck)
        self._stage(plan)
        return recheck

    def read_deadline(self, handle: CacheRequestHandle) -> Optional[float]:
        """Dispatch deadline of a parked read, for the cache; None = needed now."""
        plan = self._plans.get(handle.rid)
        if plan is None:
            return None
        now = self._last_now
        if plan.arrival_t is None:
            wait, _ = self._wait_after_arrival(
                queue_len=len(self._queued), rate=self._last_rate
            )
            deadline = plan.predicted_arrival + (wait or 0.0)
            return deadline if deadline > now else None
        position = self._positions.get(handle.rid)
        if position is None or position <= 1:
            return None
        wait = residual_wait(position=position, rate=self._last_rate)
        return None if wait is None else now + wait

    def _refresh_load_model(self) -> None:
        """Replace the load model's per-page seeds with the read thread's EWMA."""
        ms_per_page = self._tree_cache.cache_controller.storage_read_ms_per_page
        if ms_per_page is None:
            return
        s_per_page = ms_per_page / 1000.0
        if self._load.read_s_per_page == s_per_page:
            return
        self._load = msgspec.structs.replace(
            self._load, read_s_per_page=s_per_page, backlog_s_per_page=s_per_page
        )

    def _end_session(self, *, rid: str, session: Optional[str], now: float) -> None:
        """The session's next turn is not coming: its tail is evicted first."""
        if session is None:
            return
        self._register_next_use(
            rid=rid,
            session=session,
            tail_node_id=self._session_tail.get(session),
            next_use=None,
            now=now,
        )
        self._session_tail.pop(session, None)

    def _register_next_use(
        self,
        *,
        rid: str,
        session: Optional[str],
        tail_node_id: Optional[int],
        next_use: Optional[float],
        now: float,
    ) -> None:
        if self._next_use is None or tail_node_id is None:
            return
        if session is not None:
            previous = self._session_tail.get(session)
            if previous is not None and previous != tail_node_id:
                self._next_use.forget(previous)
            self._session_tail[session] = tail_node_id
        self._next_use.register(tail_node_id, next_use)
        _log_event(
            "nue_register",
            rid,
            node=tail_node_id,
            next_use_ms="ended" if next_use is None else _ms(next_use - now),
        )

    def _stage(self, plan: TurnPlan) -> None:
        """Case 1: read the evicted span under the next request's own handle; if the
        request arrives while the read runs, its own prefetch joins it."""
        cache = self._tree_cache
        handle = _handle(plan)
        plan.staged = True
        # Stamps the session's path as recently used: accepted at trigger time.
        match = cache.match_prefix(MatchPrefixParams(key=plan.key))
        anchor = match.last_host_node
        matched = len(match.device_indices) + match.host_hit_length
        tokens = plan.key.raw_token_ids()
        # Scheduler._prefetch_kvcache's gate: under write-through an unbacked anchor
        # means nothing below it is in storage, and a fetch there is dropped.
        gate = cache.is_backuped(anchor) or cache.is_root(anchor)
        if gate and not cache.has_ongoing_prefetch(handle):
            cache.prefetch_from_storage(
                handle,
                anchor,
                tokens[matched:],
                cache.get_last_hash_value(anchor),
                (
                    cache.get_prefix_hash_values(anchor)
                    if cache.hicache_storage_pass_prefix_keys
                    else None
                ),
                matched_prefix_tokens=tokens[:matched],
                extra_key=plan.key.extra_key,
                cache_salt=plan.key.cache_salt,
                storage_hit_end=None,
            )
        _log_event(
            "ntp_stage",
            plan.next_rid,
            tokens=len(tokens) - matched,
            matched=matched,
            gate=int(gate),
            inflight=int(cache.has_ongoing_prefetch(handle)),
        )

    def _expire(self, plan: TurnPlan, *, now: float) -> Optional[float]:
        """Drop a plan whose turn never arrived; None once it is gone."""
        if plan.staged:
            handle = _handle(plan)
            if self._tree_cache.has_ongoing_prefetch(handle):
                # Releasing a read mid-I/O frees its pages uninserted: let it land.
                return now
            # Pops the loaded-span records and any paced retry left for the rid.
            self._tree_cache.finish(handle, CacheRequestOutcome.ABORT)
        self._rid_session.pop(plan.next_rid, None)
        self._end_session(rid=plan.next_rid, session=plan.session, now=now)
        _log_event(
            "ntp_expire", plan.next_rid, staged=int(plan.staged), reason="timeout"
        )
        return None

    def _wait_after_arrival(
        self, *, queue_len: int, rate: Optional[Rate]
    ) -> tuple[Optional[float], str]:
        """Queue wait of a turn arriving now, and the estimator that gave it."""
        w_ql, w_ewma = wait_at_arrival(
            queue_len=queue_len, rate=rate, wait_ewma=self._admissions.wait_ewma
        )
        if queue_len == 0:
            return 0.0, "empty"
        if w_ql is not None:
            return w_ql, f"ql{rate.window_s:.0f}"
        if w_ewma is not None:
            return w_ewma, "ewma"
        return None, "none"

    def _resident_tokens(self, plan: TurnPlan, *, now: float) -> int:
        """Read-only: tokens of the key in device or host memory."""
        plan.last_probe_t = now
        _, resident = self._tree_cache.resident_prefix_len(plan.key)
        return resident

    def _load_seconds(self, tokens: int, backlog_pages: int) -> float:
        return self._load.seconds(
            own_pages=math.ceil(tokens / self._page_size), backlog_pages=backlog_pages
        )

    def _mark_due(
        self,
        plan: TurnPlan,
        *,
        case: int,
        backlog_pages: int,
        load_s: float,
        wait_s: float,
        admission_at: float,
        trigger_at: float,
        est: str,
        now: float,
    ) -> None:
        plan.due_case = case
        plan.trigger_at = trigger_at
        plan.predicted_admission = admission_at
        _log_event(
            "ntp_due",
            plan.next_rid,
            case=case,
            l3_pages=math.ceil(plan.missing_tokens / self._page_size),
            backlog_pages=backlog_pages,
            load_ms=_ms(load_s),
            wait_ms=_ms(wait_s),
            lead_ms=_ms(admission_at - now),
            est=est,
        )
