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
Read timing just_in_time delays that lookup until the load estimate says it must start.

Two more switches make the prefetcher the one owner of reads (agent_cache/DIST_HINT_DESIGN.md).
The arrival gate is consulted by every read trigger: a planned turn whose match covers its
key gets no SSD lookup, and a turn's reads are capped. The arrival hold keeps a parked read
of a queued turn until its queue position says the read must start, at the head at once.

A request may instead carry a table of gaps per reply key (action v2,
agent_cache/DIST_HINT_DESIGN.md). The reply is not known when the request is sent, so
the commit resolves it: the reply's key picks an entry, and that entry's p50 is the gap.

Read timing expected_cost plans on the entry's whole distribution instead of its p50
(agent_cache/PAPER_PLAN.md, "Formulation"; next_turn_policy.py). Host memory has a price:
a read starts where the stall it risks equals the rent it would pay pinned, or is left
to the turn's arrival; a landed read gives its pin back once its entry has passed;
with soft pins the read of an arrived turn takes room from staged reads; and the price
follows how much room reads were refused. No forecast ends such a plan.

The decisions use this rank's wall clock, so create() refuses multi-rank setups.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING, Annotated, Optional, TypeVar

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
from sglang.srt.mem_cache.next_turn_policy import (
    RECOMPUTE_S_PER_TOKEN,
    GapDistribution,
    GapSamples,
    PriceController,
    read_start,
    remaining_gaps,
)
from sglang.srt.mem_cache.next_use_eviction import (
    NextUseStrategy,
    Queued,
    Registration,
    ReturnChanceStrategy,
    ReturnForecast,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.reply_key import KEY_SEP, reply_key

if TYPE_CHECKING:
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.kv_hints import KvHintsEnvelope
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.storage_read_scheduler import ParkedRead
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

NEXT_TURN_ACTION_TYPE = "sglang.next_turn"
NEXT_TURN_ACTION_VERSION = "1"
NEXT_TURN_TABLE_VERSION = "2"
ANY_GAP_KEY = "*"

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
# Just-in-time reads start this much before the load estimate requires. Arbitrary, but
# well above the arrival error of exact hints (p90 0.2 s, campaign 19).
NTP_JIT_MARGIN_S = 1.0
# Held reads of queued turns start this much before the load estimate requires. Arbitrary;
# a late read costs only its own turn, which admission passes over until it lands.
NTP_HOLD_MARGIN_S = 2.0
# Expected-cost timing keeps a plan until its turn arrives or its session ends; past this
# age it is dropped as bookkeeping. Arbitrary (the replay client caps a gap at 1200 s).
NTP_PLAN_MAX_AGE_S = 1800.0
# A landed staged read keeps its pin this long past the end of its entry: a turn reaches
# the scheduler up to ARRIVAL_OFFSET_RANGE_S after its gap.
NTP_RELEASE_GRACE_S = 5.0
# A queued turn with no read of its own under way for this long is waiting for the GPU.
# Arbitrary; an idle GPU admits such a turn within a pass or two (tens of ms).
NTP_GPU_BOUND_S = 0.5
# Staged reads per plan under expected-cost timing: one that was released, bumped or
# refused room may be staged again. Each takes one of the turn's MAX_REFETCHES + 1
# lookups at the arrival gate, so one is left for the turn's own read at arrival.
NTP_MAX_STAGINGS = 2


class NextTurnHint(msgspec.Struct, frozen=True):
    """Payload of the sglang.next_turn v1 action."""

    session: str
    next_rid: str
    next_gap_s: Annotated[float, msgspec.Meta(ge=0)]
    # How much sooner than next_gap_s the turn may plausibly come.
    early_s: Annotated[float, msgspec.Meta(ge=0)] = 0.0


class GapEntry(msgspec.Struct, frozen=True):
    """One row of a v2 table; its n is not read."""

    key: str
    p10: Optional[Annotated[float, msgspec.Meta(ge=0)]] = None
    p50: Optional[Annotated[float, msgspec.Meta(ge=0)]] = None
    # Unconstrained: a table with an odd p90 was read before this field was, and still is.
    p90: Optional[float] = None
    ended: bool = False


class NextTurnTable(msgspec.Struct, frozen=True):
    """Payload of the sglang.next_turn v2 action."""

    session: str
    next_rid: str
    gaps: list[GapEntry]
    # A replay's generated reply is not the recorded one, so the replay names the
    # recorded reply's key; read only under reply_key_source "hint".
    replay_reply_key: Optional[str] = None


_Payload = TypeVar("_Payload", NextTurnHint, NextTurnTable)


def _read_action(
    kv_hints: Optional[KvHintsEnvelope], version: str, payload_type: type[_Payload]
) -> Optional[_Payload]:
    """The action of this version; other action types are not ours to read."""
    if kv_hints is None:
        return None
    for action in kv_hints.actions:
        if (
            action.action_type != NEXT_TURN_ACTION_TYPE
            or action.action_version != version
        ):
            continue
        try:
            return msgspec.convert(action.payload, payload_type)
        except msgspec.ValidationError as e:
            logger.warning("Ignoring a malformed %s hint: %s", NEXT_TURN_ACTION_TYPE, e)
            return None
    return None


def read_next_turn_hint(
    kv_hints: Optional[KvHintsEnvelope],
) -> Optional[NextTurnHint]:
    return _read_action(kv_hints, NEXT_TURN_ACTION_VERSION, NextTurnHint)


def read_next_turn_table(
    kv_hints: Optional[KvHintsEnvelope],
) -> Optional[NextTurnTable]:
    return _read_action(kv_hints, NEXT_TURN_TABLE_VERSION, NextTurnTable)


def pick_gap_entry(table: NextTurnTable, key: Optional[str]) -> Optional[GapEntry]:
    """The entry of the reply's key, else of its tool, else the default."""
    by_key = {entry.key: entry for entry in table.gaps}
    if key is not None:
        for candidate in (key, key.split(KEY_SEP, 1)[0]):
            if candidate in by_key:
                return by_key[candidate]
    return by_key.get(ANY_GAP_KEY)


def entry_distribution(entry: GapEntry) -> GapDistribution:
    """The entry's gap distribution; a point when it carries its median only."""
    if entry.p10 is None or entry.p90 is None:
        return GapDistribution.point(entry.p50)
    return GapDistribution.from_quantiles(p10=entry.p10, p50=entry.p50, p90=entry.p90)


class ResolvedHint(msgspec.Struct, frozen=True):
    """A finished turn's hint once its reply picked the entry: the point the horizon
    and just-in-time routines plan on, and the distribution it came from."""

    session: str
    next_rid: str
    next_gap_s: float
    # Negative when a landing quantile above the median times the read later.
    early_s: float
    dist: GapDistribution
    # The table entry the reply picked; None for an exact hint.
    entry_key: Optional[str] = None


def _event_value(key: Optional[str]) -> str:
    """A key as a HICACHE_EVT value; a key can come from model output."""
    if key is None:
        return "none"
    return re.sub(r"[^\w./-]", "_", key.replace(ANY_GAP_KEY, "any"))


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
    early_s: float = 0.0
    # Queue entry time; None while the next turn has not reached the scheduler.
    arrival_t: Optional[float] = None
    pred_wait_ql: Optional[float] = None
    pred_wait_ewma: Optional[float] = None
    last_probe_t: float = -math.inf
    # At the last probe: tokens of the key in neither device nor host memory.
    missing_tokens: int = 0
    # At the last probe: tokens of the key not in device memory. A pinned tail keeps
    # all of them in the host room, since host eviction takes childless nodes only.
    room_tokens: int = 0
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
    # Arrival gate: SSD lookups let through for this turn, by any trigger.
    lookups: int = 0
    # Handed to the Scheduler this tick; reported back by on_refetch_result.
    pending_refetch: Optional[RefetchDecision] = None
    dist: Optional[GapDistribution] = None
    entry_key: Optional[str] = None
    # Expected-cost timing: the distribution's equal-weight gaps, seconds after the
    # commit, ascending; reads staged so far; the last start decision logged.
    support: tuple[float, ...] = ()
    stagings: int = 0
    decision: str = ""


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

    def __init__(
        self,
        *,
        mode: str,
        tree_cache: UnifiedRadixCache,
        page_size: int,
        reply_key_source: str = "reply",
        read_timing: str = "horizon",
        arrival_gate: str = "off",
        arrival_hold: str = "off",
        landing_quantile: Optional[float] = None,
        memory_price: float = 5e-5,
        price_learning: str = "off",
        soft_pins: str = "off",
    ):
        self._mode = mode
        self._tree_cache = tree_cache
        self._page_size = page_size
        self._reply_key_source = reply_key_source
        self._just_in_time = read_timing == "just_in_time"
        # "arrival" keeps expected-cost timing's plans but never stages a read: the
        # plan then serves the arrival gate, the hold and the queued turn's protection.
        self._expected_cost = read_timing in ("expected_cost", "arrival")
        self._stages_reads = read_timing != "arrival"
        self._landing_quantile = landing_quantile
        self._price = PriceController(
            price=memory_price,
            rule={"off": "fixed", "on": "room", "harm": "harm"}[price_learning],
        )
        if self._expected_cost:
            tree_cache.room_refusals = []
        # Pins of turns not here yet at the last tick: what a refusal since then could
        # have used.
        self._staged_pin_tokens_seen = 0
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
        if mode == "on" and arrival_gate == "on":
            tree_cache.prefetch_gate_fn = self.prefetch_gate
        self._hold_arrived = (
            mode == "on" and arrival_hold == "on" and self._deadline_mode
        )
        if self._hold_arrived:
            tree_cache.read_not_before_fn = self.read_not_before
        if mode == "on" and soft_pins == "on" and self._deadline_mode:
            tree_cache.soft_pin_victims_fn = self.soft_pin_victims
        # rid -> when the queued turn was first seen with no read of its own under way.
        self._ready_since: dict[str, float] = {}
        # Requests with an ntp_hold event logged: one per request, not one per pass.
        self._hold_logged: set[str] = set()
        # Hint-driven eviction: the strategy's registry, when such a policy is on.
        strategy = tree_cache.tree_core.eviction_strategy
        self._next_use = (
            strategy.registry
            if isinstance(strategy, (NextUseStrategy, ReturnChanceStrategy))
            else None
        )
        self._scores_return_chance = isinstance(strategy, ReturnChanceStrategy)
        self._return_chance = strategy if self._scores_return_chance else None
        if self._scores_return_chance:
            tree_cache.prefetch_landed_fn = self.on_read_landed
        # Gaps seen per entry key, kept only when eviction scores from them.
        self._scores_samples = (
            isinstance(strategy, ReturnChanceStrategy) and strategy.score == "samples"
        )
        self._gap_samples = GapSamples()
        # The horizon and just-in-time routines under next-use eviction end a session
        # whose plan expires; a distribution keeps its tail instead.
        self._forecast_ends_session = not (
            self._expected_cost or self._scores_return_chance
        )
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
        reply_key_source: str = "reply",
        read_timing: str = "horizon",
        arrival_gate: str = "off",
        arrival_hold: str = "off",
        landing_quantile: Optional[float] = None,
        memory_price: float = 5e-5,
        price_learning: str = "off",
        soft_pins: str = "off",
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
        if read_timing in ("expected_cost", "arrival") or soft_pins == "on":
            logger.info(
                "HiCache next-turn pricing: read_timing=%s memory_price=%g "
                "price_learning=%s soft_pins=%s",
                read_timing,
                memory_price,
                price_learning,
                soft_pins,
            )
        return cls(
            mode=mode,
            tree_cache=tree_cache,
            page_size=page_size,
            reply_key_source=reply_key_source,
            read_timing=read_timing,
            arrival_gate=arrival_gate,
            arrival_hold=arrival_hold,
            landing_quantile=landing_quantile,
            memory_price=memory_price,
            price_learning=price_learning,
            soft_pins=soft_pins,
        )

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
        hint = self._resolve_hint(req)
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
            next_use=(
                self._return_forecast(hint, now=now)
                if self._scores_return_chance
                else predicted + (wait or 0.0)
            ),
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
            expires_at=(
                now + NTP_PLAN_MAX_AGE_S
                if self._expected_cost
                # A landing quantile above the median times the read past the p50.
                else predicted
                + max(0.0, -hint.early_s)
                + max(MIN_EXPIRY_S, hint.next_gap_s)
            ),
            next_check=now,
            early_s=hint.early_s,
            dist=hint.dist,
            entry_key=hint.entry_key,
            support=hint.dist.support() if self._expected_cost else (),
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

    def _resolve_hint(self, req: Req) -> Optional[ResolvedHint]:
        """The finished turn's hint. A v2 table wins: its v1 companion, if any, only
        serves engines that cannot read tables."""
        table = read_next_turn_table(req.kv_hints)
        if table is None:
            hint = read_next_turn_hint(req.kv_hints)
            if hint is None:
                return None
            return ResolvedHint(
                session=hint.session,
                next_rid=hint.next_rid,
                next_gap_s=hint.next_gap_s,
                early_s=hint.early_s,
                dist=GapDistribution.point(hint.next_gap_s),
            )
        if self._reply_key_source == "hint":
            key = table.replay_reply_key
        elif req.tokenizer is None:
            key = None
        else:
            key = reply_key(req.tokenizer.decode(req.output_ids))
        entry = pick_gap_entry(table, key)
        _log_event(
            "ntp_pick",
            table.next_rid,
            key=_event_value(key),
            entry=_event_value(None if entry is None else entry.key),
        )
        if entry is None or entry.ended or entry.p50 is None:
            return None
        dist = entry_distribution(entry)
        if self._landing_quantile is not None:
            early_s = entry.p50 - dist.quantile(self._landing_quantile)
        else:
            early_s = 0.0 if entry.p10 is None else max(0.0, entry.p50 - entry.p10)
        return ResolvedHint(
            session=table.session,
            next_rid=table.next_rid,
            next_gap_s=entry.p50,
            early_s=early_s,
            dist=dist,
            entry_key=entry.key,
        )

    def _return_forecast(self, hint: ResolvedHint, *, now: float) -> ReturnForecast:
        idle_from = now + self._offset.seconds
        if not self._scores_samples or hint.entry_key is None:
            return ReturnForecast(dist=hint.dist, idle_from=idle_from)
        return ReturnForecast(
            dist=hint.dist,
            idle_from=idle_from,
            key_gaps=self._gap_samples.ascending(hint.entry_key),
            all_gaps=self._gap_samples.ascending(ANY_GAP_KEY),
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
        # The hold times reads of turns with no plan, so it needs a fresh rate regardless.
        if not self._plans and not self._hold_arrived:
            return []

        rate = self._admissions.rate(now)
        self._last_rate = rate
        queue_ahead = self._queue_ahead(waiting_queue, now=now)
        if self._expected_cost:
            self._follow_room(queue_ahead=queue_ahead, now=now)
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
            queue_len=queue_ahead,
            rate=rate,
            backlog_pages=backlog_pages,
            now=now,
        )
        return refetch

    def _queue_ahead(self, waiting_queue: list[Req], *, now: float) -> int:
        """Queued turns that a turn arriving now would wait behind. Under expected-cost
        timing the queue counts only while the GPU is what it waits for: turns waiting
        on their own storage read are passed over by admission and delay no one, so
        without a turn that has nothing to read and is still queued, the wait is none."""
        if not self._expected_cost:
            return len(waiting_queue)
        cache = self._tree_cache
        ready_since = {}
        for req in waiting_queue:
            if not cache.has_ongoing_prefetch(req.cache_request_handle):
                ready_since[req.rid] = self._ready_since.get(req.rid, now)
        self._ready_since = ready_since
        gpu_bound = any(now - since >= NTP_GPU_BOUND_S for since in ready_since.values())
        return len(waiting_queue) if gpu_bound else 0

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
        self._hold_logged.discard(req.rid)
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
        if self._scores_samples and plan.entry_key is not None:
            # Net of the delivery offset, as the forecast's idle time is.
            gap_s = max(0.0, arrival - plan.t_commit - self._offset.seconds)
            self._gap_samples.add(key=plan.entry_key, gap_s=gap_s)
            if plan.entry_key != ANY_GAP_KEY:
                self._gap_samples.add(key=ANY_GAP_KEY, gap_s=gap_s)
        self._offset.observe(offset)
        plan.arrival_t = arrival
        self._register_next_use(
            rid=plan.next_rid,
            session=plan.session,
            tail_node_id=self._session_tail.get(plan.session),
            next_use=Queued(since=now),
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
        if self._expected_cost:
            return self._check_pending_expected_cost(
                plan,
                wait=wait,
                est=est,
                backlog_pages=backlog_pages,
                now=now,
                recheck=recheck,
            )
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
        horizon, or just in time; the cache's read scheduler orders the dispatch."""
        if admission_at - now > NTP_LEAD_HORIZON_S:
            return min(admission_at - NTP_LEAD_HORIZON_S, recheck)
        if now - plan.last_probe_t >= PROBE_INTERVAL_S:
            plan.missing_tokens = len(plan.key) - self._resident_tokens(plan, now=now)
        enough = plan.missing_tokens >= self._tree_cache.prefetch_threshold
        load_s = self._load_seconds(plan.missing_tokens, backlog_pages)
        if self._just_in_time:
            start_at = trigger_time(
                admission_at=admission_at,
                load_s=load_s,
                margin_s=plan.early_s + NTP_JIT_MARGIN_S,
            )
            if now < start_at:
                return min(start_at, recheck)
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

    def _check_pending_expected_cost(
        self,
        plan: TurnPlan,
        *,
        wait: float,
        est: str,
        backlog_pages: int,
        now: float,
        recheck: float,
    ) -> float:
        """Case 1 by expected cost: start the read where the stall it risks equals the
        rent it would pay pinned, or leave it to the turn's arrival; a landed read
        keeps its pin until its entry has passed."""
        if not self._stages_reads:
            return plan.expires_at
        cache = self._tree_cache
        handle = _handle(plan)
        idle_s = now - plan.t_commit - self._offset.seconds
        passed = self._mode == "on" and not remaining_gaps(
            support=plan.support, idle_s=idle_s - NTP_RELEASE_GRACE_S
        )
        if plan.staged and cache.has_ongoing_prefetch(handle):
            if passed and cache.read_scheduler.is_parked(handle):
                # Still waiting for the window or the budget: dispatched now it would
                # take room only to be released.
                self._release_staged(plan, tokens=0, now=now)
            return recheck
        remaining = remaining_gaps(support=plan.support, idle_s=idle_s)
        pinned_tokens = cache.pinned_prefetch_tokens_of(handle)
        if pinned_tokens > 0:
            # Pinned until admission, as the room model that set the price holds it;
            # only an entry that has passed gives the pin back.
            if passed:
                self._release_staged(plan, tokens=pinned_tokens, now=now)
            return recheck
        if plan.staged:
            # Over without a pin: refused room, a storage miss, or resident after all.
            # Left to the turn's arrival, which keeps its lookups for that.
            return recheck
        if now - plan.last_probe_t >= PROBE_INTERVAL_S:
            plan.last_probe_t = now
            in_device, resident = cache.resident_prefix_len(plan.key)
            plan.missing_tokens = len(plan.key) - resident
            plan.room_tokens = len(plan.key) - in_device
        if (
            plan.missing_tokens < cache.prefetch_threshold
            or plan.stagings >= NTP_MAX_STAGINGS
        ):
            return recheck
        read_s = self._read_seconds(plan.missing_tokens, backlog_pages)
        start_in = read_start(
            remaining=remaining,
            read_s=read_s,
            wait_s=wait,
            rent_per_s=self._price.price * plan.room_tokens,
        )
        self._log_start_decision(
            plan, start_in=start_in, read_s=read_s, wait_s=wait, now=now
        )
        if start_in is None:
            return recheck
        if start_in > 0.0:
            return min(now + start_in, recheck)
        if plan.due_case == 0:
            self._mark_due(
                plan,
                case=1,
                backlog_pages=backlog_pages,
                load_s=read_s - NTP_JIT_MARGIN_S,
                wait_s=wait,
                admission_at=now + read_s,
                trigger_at=now,
                est=est,
                now=now,
            )
        if self._mode != "on":
            return recheck
        if (
            cache.read_scheduler.parked_count() >= NTP_MAX_PARKED
            or cache.cache_controller.prefetch_rate_limited()
        ):
            # Declining in the cache would arm a paced retry for a rid that is
            # not queued; wait here instead.
            return min(now + PROBE_INTERVAL_S, recheck)
        plan.stagings += 1
        self._stage(plan)
        # No lookup went out (an unbacked anchor, less missing than the probe saw):
        # the attempt counts, but the plan is not left as staged with nothing staged.
        plan.staged = cache.has_ongoing_prefetch(handle)
        return recheck

    def on_read_landed(self, handle: CacheRequestHandle, node_id: int) -> None:
        """A storage read's span was inserted as node_id. Under return_chance eviction
        it belongs to the session of the plan it was read for; without this a span
        read back under a prompt that sessions share ranks as un-hinted."""
        plan = self._plans.get(handle.rid)
        if plan is None:
            return
        tail_id = self._session_tail.get(plan.session)
        if tail_id is not None:
            self._return_chance.adopt(node_id, tail_id)

    def _read_seconds(self, tokens: int, backlog_pages: int) -> float:
        """From issuing a staged read to its pages being usable, margin included."""
        return self._load_seconds(tokens, backlog_pages) + NTP_JIT_MARGIN_S

    def _log_start_decision(
        self,
        plan: TurnPlan,
        *,
        start_in: Optional[float],
        read_s: float,
        wait_s: float,
        now: float,
    ) -> None:
        """One ec_start event per change of decision, not one per probe."""
        if start_in is None:
            decision = "arrival"
        else:
            decision = "now" if start_in == 0.0 else "wait"
        if decision == plan.decision:
            return
        plan.decision = decision
        _log_event(
            "ec_start",
            plan.next_rid,
            decision=decision,
            tokens=plan.missing_tokens,
            start_ms=_ms(start_in),
            idle_ms=_ms(now - plan.t_commit),
            read_ms=_ms(read_s),
            wait_ms=_ms(wait_s),
            price=f"{self._price.price:.3g}",
        )

    def _release_staged(self, plan: TurnPlan, *, tokens: int, now: float) -> None:
        """Give a landed staged read's pin back; its pages stay cached until evicted."""
        self._tree_cache.finish(_handle(plan), CacheRequestOutcome.ABORT)
        plan.staged = False
        plan.decision = ""
        _log_event(
            "ec_release", plan.next_rid, tokens=tokens, idle_ms=_ms(now - plan.t_commit)
        )

    def soft_pin_victims(
        self, handle: CacheRequestHandle, tokens_needed: int
    ) -> list[CacheRequestHandle]:
        """Rule 2: the staged reads to give up so the read of a turn that has arrived
        finds room, least likely back within their own read time first. Empty unless
        the asking turn is queued."""
        if handle.rid not in self._positions:
            return []
        now = self._last_now
        staged = []
        for plan in self._plans.values():
            if plan.arrival_t is not None or plan.dist is None:
                continue
            # A turn that reached the scheduler since the last tick may already read
            # under this handle; bumping it would revoke that read with the pin.
            if self._tree_cache.has_ongoing_prefetch(_handle(plan)):
                continue
            tokens = self._tree_cache.pinned_prefetch_tokens_of(_handle(plan))
            if tokens <= 0:
                continue
            chance = plan.dist.chance_within(
                idle_s=now - plan.t_commit - self._offset.seconds,
                horizon_s=self._read_seconds(tokens, 0),
            )
            staged.append((chance, tokens, plan))
        staged.sort(key=lambda entry: entry[0])
        victims, freed = [], 0
        for chance, tokens, plan in staged:
            if freed >= tokens_needed:
                break
            freed += tokens
            plan.staged = False
            plan.decision = ""
            victims.append(_handle(plan))
            # Rule 4: a bumped read was room refused, and it stalls its own turn by
            # one read if the turn comes back before it is read again.
            self._price.refuse(tokens)
            self._price.charge(chance * self._read_seconds(tokens, 0))
            _log_event(
                "ec_bump",
                plan.next_rid,
                for_rid=handle.rid,
                tokens=tokens,
                chance=f"{chance:.3f}",
            )
        return victims

    def _follow_room(self, *, queue_ahead: int, now: float) -> None:
        """Rule 4: feed the price what the room refused, what it left idle, and what
        staged pins cost the turns that had arrived."""
        cache = self._tree_cache
        room_tokens, held_tokens = cache.host_room_tokens()
        staged_left = self._staged_pin_tokens_seen
        for handle, tokens in cache.take_room_refusals():
            self._price.refuse(tokens)
            if handle.rid in self._queued:
                # The part of the refused read that staged pins were holding is
                # recomputed, while its own turn and those waiting for the GPU wait.
                lost_tokens = min(tokens, staged_left)
                staged_left -= lost_tokens
                self._price.charge(
                    lost_tokens * RECOMPUTE_S_PER_TOKEN * max(1, queue_ahead)
                )
        staged_tokens = sum(
            cache.pinned_prefetch_tokens_of(_handle(plan))
            for plan in self._plans.values()
            if plan.arrival_t is None
        )
        self._staged_pin_tokens_seen = staged_tokens
        self._price.observe(
            now=now,
            idle_tokens=max(0, room_tokens - held_tokens),
            staged_tokens=staged_tokens,
        )
        update = self._price.step(room_tokens=cache.host_room_floor_tokens)
        if update is None:
            return
        logger.info(
            "HICACHE_EVT ec_price price=%.3g refused=%d idle=%d excess=%.3f room=%d "
            "held=%d staged=%d harm_ms=%d",
            update.price,
            update.refused_tokens,
            round(update.idle_tokens),
            update.excess,
            room_tokens,
            held_tokens,
            round(update.staged_tokens),
            round(update.harm_s * 1000),
        )

    def read_deadline(self, handle: CacheRequestHandle) -> Optional[float]:
        """Dispatch deadline of a parked read, for the cache; None = needed now."""
        plan = self._plans.get(handle.rid)
        # Under the hold a turn with no plan is timed from its queue position too.
        if plan is None and not self._hold_arrived:
            return None
        now = self._last_now
        if plan is not None and plan.arrival_t is None:
            wait, _ = self._wait_after_arrival(
                queue_len=len(self._queued), rate=self._last_rate
            )
            deadline = plan.predicted_arrival + (wait or 0.0)
            if self._expected_cost or plan.early_s < 0.0:
                # Started on purpose past the p50, so it stays a timed read: "now"
                # would lift the window and the budget for a turn not here yet.
                return max(deadline - min(0.0, plan.early_s), now + NTP_JIT_MARGIN_S)
            return deadline if deadline > now else None
        position = self._positions.get(handle.rid)
        if position is None or position <= 1:
            return None
        wait = residual_wait(position=position, rate=self._last_rate)
        return None if wait is None else now + wait

    def prefetch_gate(self, handle: CacheRequestHandle, matched_tokens: int) -> bool:
        """May this prefetch look up the SSD? Every read trigger ends here. A read
        already under way has been joined by now, so a new one means history is missing."""
        plan = self._plans.get(handle.rid)
        # A first turn, a client without hints, a plan gone: only a lookup can tell.
        if plan is None:
            return True
        chain = len(plan.key)
        if chain - matched_tokens < self._tree_cache.prefetch_threshold:
            # The history is resident and the rest is new. It is on the SSD only if
            # another session shares this prefix, which is not worth a lookup per turn.
            decision = "covered"
        elif plan.lookups > MAX_REFETCHES:
            # A re-read beats a recompute, but not without limit: reads that keep being
            # evicted before admission only evict others.
            plan.refetch_capped = True
            decision = "capped"
        else:
            decision = "read" if plan.lookups == 0 else "reread"
            plan.lookups += 1
        _log_event(
            "ntp_gate",
            handle.rid,
            decision=decision,
            matched=matched_tokens,
            chain=chain,
            lookups=plan.lookups,
        )
        return decision in ("read", "reread")

    def read_not_before(
        self, handle: CacheRequestHandle, read: ParkedRead, ahead_pages: int
    ) -> Optional[float]:
        """When a parked read of a queued turn may start; None = no hold. `ahead_pages`
        counts the pages the read thread serves first."""
        position = self._positions.get(handle.rid)
        if position is None:
            # Parked since the last tick, so the queue position is not known yet: wait
            # one pass. Still unknown after a tick means not queued, which includes a
            # staged read before its turn arrives: the prefetcher already timed that.
            return math.inf if read.parked_at > self._last_now else None
        if position <= 1:
            return None
        wait = residual_wait(position=position, rate=self._last_rate)
        if wait is None:
            return None
        load_s = self._load.seconds(own_pages=read.pages, backlog_pages=ahead_pages)
        not_before = trigger_time(
            admission_at=self._last_now + wait,
            load_s=load_s,
            margin_s=NTP_HOLD_MARGIN_S,
        )
        if not_before > self._last_now and handle.rid not in self._hold_logged:
            self._hold_logged.add(handle.rid)
            _log_event(
                "ntp_hold",
                handle.rid,
                pos=position,
                pages=read.pages,
                wait_ms=_ms(wait),
                load_ms=_ms(load_s),
                hold_ms=_ms(not_before - self._last_now),
            )
        return not_before

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
        next_use: Registration,
        now: float,
    ) -> None:
        if self._next_use is None or tail_node_id is None:
            return
        if session is not None:
            previous = self._session_tail.get(session)
            if previous is not None and previous != tail_node_id:
                self._next_use.move(previous, tail_node_id)
            self._session_tail[session] = tail_node_id
        self._next_use.register(tail_node_id, next_use)
        if next_use is None:
            due = "ended"
        elif isinstance(next_use, ReturnForecast):
            due = _ms(next_use.idle_from + next_use.dist.median_s - now)
        elif isinstance(next_use, Queued):
            due = _ms(next_use.since - now)
        else:
            due = _ms(next_use - now)
        _log_event("nue_register", rid, node=tail_node_id, next_use_ms=due)

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
        if self._forecast_ends_session:
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
