"""With every new switch off the prefetcher of impl_unified_A must behave as the baseline's,
and with only the prices log on it must still decide as the baseline does.

Replays seeded random scenarios against two prefetchers side by side, the baseline module
(baseline/next_turn_prefetch.py, a snapshot of the file before the change) and the copy's, each
on its own fake cache, and compares every HICACHE_EVT line they log, every answer they give
(tick, gate, deadline, hold, soft-pin victims) and what they asked of the cache.

CPU only. Run inside the container, GPU hidden, from this folder:
  COPY=/sgl-workspace/sglang/agent_cache/frozen/impl_unified_A
  docker exec -e PYTHONPATH=$COPY/python:$COPY/test/registered/unit/mem_cache \
      -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
      'cd /sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/A/checks &&
       nice -n 10 python3 -m pytest -q -p no:cacheprovider test_switches_off_equivalence.py'
"""

import collections
import importlib.util
import logging
import os
import random
import re
import sys
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import test_next_turn_prefetch as fixtures

from sglang.srt.environ import envs
from sglang.srt.mem_cache import next_turn_prefetch as changed
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_use_eviction import (
    NextUseStrategy,
    ReturnChanceStrategy,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_baseline():
    spec = importlib.util.spec_from_file_location(
        "ntp_baseline", os.path.join(HERE, "baseline", "next_turn_prefetch.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ntp_baseline"] = module
    spec.loader.exec_module(module)
    return module


baseline = load_baseline()
PAGE = fixtures.PAGE
NEW_SWITCHES = (
    "SGLANG_LOG_HICACHE_PRICES_INTERVAL",
    "SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES",
    "SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS",
    "SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT",
    "SGLANG_HICACHE_NEXT_TURN_READ_LINE",
    "SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE",
    "SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE",
    "SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED",
)
GAP_TABLES = (
    [{"key": "*", "p10": 45.0, "p50": 50.0, "p90": 56.0}],
    [{"key": "*", "p10": 0.85, "p50": 22.0, "p90": 58.0}],
    [{"key": "tool", "p10": 1.0, "p50": 4.0, "p90": 9.0}, {"key": "*", "p50": 12.0}],
    [{"key": "_end", "ended": True}, {"key": "*", "p10": 2.0, "p50": 6.0, "p90": 30.0}],
)


class Messages(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class Side:
    """One prefetcher on its own fake cache, with its own request objects."""

    def __init__(self, module, config, clock, *, prices_interval_s=0.0):
        self.module = module
        self.messages = Messages()
        self.logger = logging.getLogger(module.__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(self.messages)
        cache_class = {
            "fifo": fixtures.ActingCache,
            "deadline": fixtures.DeadlineCache,
            "priced": fixtures.PricedCache,
        }[config["cache"]]
        self.cache = cache_class()
        if config["cache"] != "fifo":
            strategy = (
                ReturnChanceStrategy(horizon_s=30.0, score=config["score"], clock=clock)
                if config["policy"] == "return_chance"
                else NextUseStrategy(horizon_s=300.0, clock=clock)
            )
            self.cache.tree_core = SimpleNamespace(eviction_strategy=strategy)
        else:
            # Its own, as above: the meter binds its eviction observer on it.
            self.cache.tree_core = SimpleNamespace(eviction_strategy=None)
        # Read by the meter only; the baseline never asks.
        self.cache.pending_backup_tokens = 0
        with envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.override(prices_interval_s):
            self.ntp = module.NextTurnPrefetcher(
                mode=config["mode"],
                tree_cache=self.cache,
                page_size=PAGE,
                reply_key_source="hint",
                read_timing=config["timing"],
                arrival_gate=config["gate"],
                arrival_hold=config["hold"],
                landing_quantile=config["landing_quantile"],
                memory_price=config["price"],
                price_learning=config["learning"],
                soft_pins=config["soft_pins"],
            )
        self.reqs = {}
        self.queue = []
        self.answers = []

    def close(self):
        self.logger.removeHandler(self.messages)

    def req(self, rid, **kwargs):
        if rid not in self.reqs:
            self.reqs[rid] = fixtures.make_req(rid, **kwargs)
            # Read by the prefill fit; no prefill is timed in these scenarios.
            self.reqs[rid].time_stats.prefill_finished_time = 0.0
        return self.reqs[rid]

    def apply(self, event, now):
        kind, args = event[0], event[1:]
        cache, ntp = self.cache, self.ntp
        handle = CacheRequestHandle(args[0], 0) if args and isinstance(args[0], str) else None
        if kind == "commit":
            rid, kv_hints, tokens, tail = args
            req = self.req(rid)
            req.kv_hints = kv_hints
            ntp.on_turn_committed(
                req=req,
                key=fixtures.key_of(tokens),
                rotation_tail_declined=False,
                now=now,
                tail_node_id=tail,
            )
        elif kind == "arrive":
            req = self.req(args[0], entered=now)
            req.time_stats.wait_queue_entry_time = now
            self.queue.append(req)
        elif kind == "admit":
            req = self.reqs[args[0]]
            req.time_stats.forward_entry_time = now
            req.storage_hit_length = args[1]
            self.queue = [queued for queued in self.queue if queued is not req]
        elif kind == "abort":
            self.queue = [queued for queued in self.queue if queued.rid != args[0]]
        elif kind == "tick":
            returned = ntp.tick(waiting_queue=list(self.queue), now=now)
            self.answers.append(("tick", [req.rid for req in returned]))
            for req in returned:
                gate = cache.prefetch_gate_fn
                issued = args[0] and (
                    gate is None or gate(req.cache_request_handle, cache.resident)
                )
                if issued:
                    cache.ongoing.add(req.cache_request_handle)
                ntp.on_refetch_result(req=req, issued=bool(issued))
        elif kind == "resident":
            cache.resident = cache.matched_tokens = args[0]
            cache.in_device = args[1]
        elif kind == "ongoing":
            (cache.ongoing.add if args[1] else cache.ongoing.discard)(handle)
        elif kind == "land" and isinstance(cache, fixtures.PricedCache):
            cache.land(handle, args[1])
        elif kind == "park" and cache.read_scheduler is not None:
            if not cache.read_scheduler.is_parked(handle):
                cache.read_scheduler.park(
                    handle=handle, operation=None, pages=args[1], now=now
                )
        elif kind == "dispatch" and cache.read_scheduler is not None:
            decisions = cache.read_scheduler.dispatch(
                now=now,
                deadline_of=ntp.read_deadline,
                in_flight_pages=args[0],
                held_tokens=0,
                not_before_of=cache.read_not_before_fn,
            )
            self.answers.append(
                ("dispatch", [(d.read.handle.rid, d.deadline) for d in decisions])
            )
        elif kind == "gate" and cache.prefetch_gate_fn is not None:
            self.answers.append(("gate", cache.prefetch_gate_fn(handle, args[1])))
        elif kind == "deadline" and cache.read_scheduler is not None:
            self.answers.append(("deadline", ntp.read_deadline(handle)))
        elif kind == "refuse":
            if isinstance(cache, fixtures.PricedCache):
                cache.refusals.append((handle, args[1]))
            if cache.prefetch_refused_fn is not None:
                cache.prefetch_refused_fn(handle, tokens=args[1], cut=args[2], now=now)
        elif kind == "victims" and isinstance(cache, fixtures.PricedCache):
            if cache.soft_pin_victims_fn is None:
                return
            victims = cache.soft_pin_victims_fn(handle, args[1])
            self.answers.append(("victims", [victim.rid for victim in victims]))
            for victim in victims:
                cache.finish(victim, "bumped")
        elif kind == "landed" and self.ntp._scores_return_chance:
            cache.prefetch_landed_fn(handle, args[1])
        elif kind == "limiter":
            cache.rate_limited = args[0]
        elif kind == "read_rate":
            cache.cache_controller.storage_read_ms_per_page = args[0]
        elif kind == "staged":
            self.answers.append(("staged", ntp.is_staged_handle(handle)))

    def state(self, lines_of=list):
        cache = self.cache
        prefetches = [
            (handle.rid, anchor, len(tokens), last_hash, sorted(kw))
            for handle, anchor, tokens, last_hash, _, kw in cache.prefetches
        ]
        finished = [(handle.rid, str(outcome)) for handle, outcome in cache.finished]
        return (
            lines_of(self.messages.lines),
            self.answers,
            prefetches,
            finished,
            cache.cancelled,
        )


def random_config(rng):
    cache = rng.choice(("fifo", "deadline", "priced", "priced"))
    timings = {
        "fifo": ("horizon",),
        "deadline": ("horizon", "just_in_time"),
        "priced": ("horizon", "just_in_time", "expected_cost", "arrival"),
    }[cache]
    return dict(
        cache=cache,
        mode=rng.choice(("on", "on", "observe")),
        timing=rng.choice(timings),
        gate=rng.choice(("on", "off")),
        hold=rng.choice(("on", "off")),
        soft_pins=rng.choice(("on", "off")),
        learning=rng.choice(("off", "on", "harm")),
        landing_quantile=rng.choice((None, None, 0.7)),
        price=rng.choice((5e-5, 5e-6)),
        policy=rng.choice(("next_use", "return_chance")),
        score=rng.choice(("entry", "samples", "lru")),
        room_retry=rng.choice((False, True)),
    )


def random_events(rng, count):
    """(dt, event) pairs. Six sessions cycle through commit (with a v1 hint, a v2 table or
    none), arrival of the hinted turn, queueing, and admission or abort, beside turns with
    no plan, while the cache's state moves under them."""
    sessions = [f"s{i}" for i in range(6)]
    turn = dict.fromkeys(sessions, 0)
    # commit: the session's current turn can commit; wait: its hinted turn has not arrived.
    phase = dict.fromkeys(sessions, "commit")
    queued = []
    node = iter(range(1, 10**6))
    events = []
    for _ in range(count):
        dt = rng.choice((0.0, 0.02, 0.1, 0.5, 0.5, 2.0, 7.0))
        roll = rng.random()
        committing = [s for s in sessions if phase[s] == "commit"]
        waiting = [s for s in sessions if phase[s] == "wait"]
        if roll < 0.14 and committing:
            session = rng.choice(committing)
            rid, next_rid = f"{session}-t{turn[session]}", f"{session}-t{turn[session] + 1}"
            turn[session] += 1
            pick = rng.random()
            if pick < 0.15:
                kv_hints = None
            elif pick < 0.5:
                payload = {
                    "session": session,
                    "next_rid": next_rid,
                    "next_gap_s": rng.choice((0.5, 5.0, 30.0, 200.0)),
                }
                kv_hints = fixtures.hints(next_rid, payload=payload)
            else:
                payload = {
                    "session": session,
                    "next_rid": next_rid,
                    "gaps": rng.choice(GAP_TABLES),
                    "replay_reply_key": rng.choice((None, "tool", "tool/x", "_end")),
                }
                kv_hints = fixtures.hints(next_rid, version="2", payload=payload)
            if kv_hints is not None:
                phase[session] = "wait"
            tokens = rng.choice((0, 8, 16, 40, 1000)) * PAGE
            tail = rng.choice((None, next(node)))
            events.append((dt, ("commit", rid, kv_hints, tokens, tail)))
        elif roll < 0.26 and waiting:
            session = rng.choice(waiting)
            phase[session] = "queue"
            queued.append(f"{session}-t{turn[session]}")
            events.append((dt, ("arrive", queued[-1])))
        elif roll < 0.30:
            queued.append(f"plain-{len(events)}")
            events.append((dt, ("arrive", queued[-1])))
        elif roll < 0.44 and queued:
            rid = queued.pop(0 if rng.random() < 0.8 else rng.randrange(len(queued)))
            session = rid.split("-")[0]
            admitted = rng.random() < 0.85
            if session in phase:
                phase[session] = "commit"
                # An aborted turn never commits: the session goes on with a new one.
                turn[session] += 0 if admitted else 1
            if admitted:
                events.append((dt, ("admit", rid, rng.choice((0, 0, 640)))))
            else:
                events.append((dt, ("abort", rid)))
        elif roll < 0.70:
            events.append((dt, ("tick", rng.random() < 0.8)))
        else:
            known = queued + [f"{s}-t{turn[s]}" for s in waiting]
            rid = rng.choice(known) if known else "nobody"
            resident = rng.choice((0, 4, 16, 1000)) * PAGE
            events.append(
                (
                    dt,
                    rng.choice(
                        (
                            ("resident", resident, rng.choice((None, 4 * PAGE))),
                            ("ongoing", rid, rng.random() < 0.5),
                            ("land", rid, rng.choice((640, 20_000))),
                            ("park", rid, rng.choice((1, 10, 400))),
                            ("dispatch", rng.choice((0, 300, 600))),
                            ("gate", rid, resident),
                            ("deadline", rid),
                            ("refuse", rid, rng.choice((640, 30_000)), rng.random() < 0.5),
                            ("victims", rid, rng.choice((640, 40_000))),
                            ("landed", rid, next(node)),
                            ("limiter", rng.random() < 0.2),
                            ("read_rate", rng.choice((None, 2.0, 5.0))),
                            ("staged", rid),
                        )
                    ),
                )
            )
    return events


def test_the_new_switches_are_off_by_default():
    for name in NEW_SWITCHES:
        assert not getattr(envs, name).get(), name


# Every kind of line the prefetcher logged before the change; each must be compared.
KINDS = {
    "ntp_plan", "ntp_pick", "ntp_arrive", "ntp_admit", "ntp_expire", "ntp_due", "ntp_stage",
    "ntp_refetch", "ntp_reevict", "ntp_gate", "ntp_hold", "ntp_room", "nue_register",
    "ec_start", "ec_release", "ec_bump", "ec_price",
}


def test_with_the_switches_off_every_line_and_answer_equals_the_baselines():
    rng = random.Random(20261126)
    kinds = collections.Counter()
    answers = collections.Counter()
    for scenario in range(240):
        config = random_config(rng)
        events = random_events(rng, 600)
        clock = SimpleNamespace(now=0.0)
        with envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(config["room_retry"]):
            sides = [
                Side(module, config, lambda: clock.now) for module in (baseline, changed)
            ]
        try:
            for step, (dt, event) in enumerate(events):
                clock.now += dt
                for side in sides:
                    side.apply(event, clock.now)
                before, after = (side.state() for side in sides)
                assert before == after, (scenario, config, step, event)
            for line in sides[0].messages.lines:
                kinds[line.split()[1]] += 1
            for answer in sides[0].answers:
                answers[answer[0]] += 1
        finally:
            for side in sides:
                side.close()
    print("lines compared by kind:", dict(sorted(kinds.items())))
    print("answers compared by kind:", dict(sorted(answers.items())))
    # The scenarios must reach every line the prefetcher wrote before the change.
    assert KINDS <= set(kinds), KINDS - set(kinds)
    assert sum(kinds.values()) > 40_000, sum(kinds.values())


# What the prices log adds to the lines: its own, and two fields at the end of ntp_admit.
ADMIT_ADDED = re.compile(r" reader=[01] land_ms=[\w-]+$")


def without_the_prices_log(lines):
    return [
        ADMIT_ADDED.sub("", line) if line.startswith("HICACHE_EVT ntp_admit ") else line
        for line in lines
        if not line.startswith("HICACHE_EVT prices ")
    ]


def test_with_only_the_prices_log_on_every_decision_equals_the_baselines():
    """SGLANG_LOG_HICACHE_PRICES_INTERVAL measures and logs: the meter follows every pass,
    host eviction and every landing, and no decision may read it. Its lines and the two
    fields ntp_admit gains aside, every line, answer and cache call must equal the
    baseline's."""
    rng = random.Random(20261127)
    prices_lines = admits = 0
    for scenario in range(240):
        config = random_config(rng)
        events = random_events(rng, 600)
        clock = SimpleNamespace(now=0.0)
        with envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(config["room_retry"]):
            before_side = Side(baseline, config, lambda: clock.now)
            after_side = Side(changed, config, lambda: clock.now, prices_interval_s=1.0)
        assert after_side.ntp._meter is not None
        try:
            for step, (dt, event) in enumerate(events):
                clock.now += dt
                before_side.apply(event, clock.now)
                after_side.apply(event, clock.now)
                before = before_side.state()
                after = after_side.state(without_the_prices_log)
                assert before == after, (scenario, config, step, event)
            lines = after_side.messages.lines
            prices_lines += sum(line.startswith("HICACHE_EVT prices ") for line in lines)
            admits += sum(bool(ADMIT_ADDED.search(line)) for line in lines)
        finally:
            before_side.close()
            after_side.close()
    print("prices lines logged:", prices_lines, "ntp_admit lines with the two fields:", admits)
    assert prices_lines > 5_000 and admits > 1_000, (prices_lines, admits)
