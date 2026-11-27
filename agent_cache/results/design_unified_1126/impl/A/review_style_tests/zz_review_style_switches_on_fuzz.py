"""Reviewer's throwaway check (style and tests lens): with the new switches ON in random
valid combinations, does any random sequence of scheduler events make the prefetcher raise?

The engineer's side-by-side replay covers the switches off and the prices log alone; every
other unit test builds one switch's own setup. Nothing runs the switches together. This
drives one prefetcher on the test file's MeasuredCache through seeded random scenarios
(events after checks/test_switches_off_equivalence.py, plus what the switched hooks take:
the prompt gate, lookup answers, host evictions, lag samples, write holds) and reports
every exception. An exception here would be a scheduler-thread crash on the server.

Nothing of the copy is edited. Run with the test recipe on a copy of this file placed in
the copy's test folder.
"""

import collections
import logging
import random
import traceback
from array import array
from types import SimpleNamespace

import test_next_turn_prefetch as T

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.next_turn_prices import PrefillCostModel
from sglang.srt.mem_cache.next_use_eviction import (
    NextUseStrategy,
    ReturnChanceStrategy,
)

PAGE = T.PAGE
BOOL_SWITCHES = (
    envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES,
    envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS,
    envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT,
    envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE,
    envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE,
    envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE,
    envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,
    envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES,
    envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY,
)
GAP_TABLES = (
    [{"key": "*", "p10": 45.0, "p50": 50.0, "p90": 56.0}],
    [{"key": "*", "p10": 0.85, "p50": 22.0, "p90": 58.0}],
    [{"key": "tool", "p10": 1.0, "p50": 4.0, "p90": 9.0}, {"key": "*", "p50": 12.0}],
    [{"key": "_end", "ended": True}, {"key": "*", "p10": 2.0, "p50": 6.0, "p90": 30.0}],
)


def tree_node(node_id):
    return SimpleNamespace(id=node_id, last_access_time=0.0, parent=None, children={})


class Run:
    """One prefetcher on a MeasuredCache, with its own request objects."""

    def __init__(self, config, rng, clock):
        self.rng = rng
        self.config = config
        self.cache = cache = T.MeasuredCache()
        strategy = (
            ReturnChanceStrategy(horizon_s=30.0, score=config["score"], clock=clock)
            if config["policy"] == "return_chance"
            else NextUseStrategy(horizon_s=300.0, clock=clock)
        )
        cache.tree_core = SimpleNamespace(
            eviction_strategy=strategy, host_eviction_observer=None
        )
        self.ntp = NextTurnPrefetcher(
            mode=config["mode"],
            tree_cache=cache,
            page_size=PAGE,
            reply_key_source="hint",
            read_timing=config["timing"],
            arrival_gate=config["gate"],
            arrival_hold=config["hold"],
            price_learning=config["learning"],
            soft_pins=config["soft_pins"],
        )
        self.reqs = {}
        self.queue = []
        self.nodes = []
        # In half of the runs the lags are measured from the start and prefill costs what
        # FA3 does, so that read-or-recompute and the starved-GPU rule have what they need.
        if rng.random() < 0.5 and self.ntp._meter is not None:
            self.ntp._meter.prefill_cost = PrefillCostModel(seed_a=12.8e-6, seed_b=0.71e-9)
            if "read_lags" in vars(cache):
                T.measure_lags(cache)

    def req(self, rid, **kwargs):
        if rid not in self.reqs:
            req = T.make_req(rid, **kwargs)
            req.time_stats.prefill_finished_time = 0.0
            req.cached_tokens = 0
            req.origin_input_ids = array("q")
            self.reqs[rid] = req
        return self.reqs[rid]

    def ask_gate(self, handle, matched):
        """The cache asking the gate, with the prompt where the prompt gate is bound."""
        cache, rng = self.cache, self.rng
        if cache.prefetch_gate_prompt_fn is not None:
            shared = rng.choice((0, 4, 16, 40, 1000)) * PAGE
            tail = array(
                "q", [*range(matched, max(matched, shared)), *range(9000, 9000 + rng.choice((0, 64, 400)))]
            )
            return cache.prefetch_gate_prompt_fn(handle, matched, tail)
        if cache.prefetch_gate_fn is not None:
            return cache.prefetch_gate_fn(handle, matched)
        return True

    def apply(self, event, now):
        kind, args = event[0], event[1:]
        cache, ntp, rng = self.cache, self.ntp, self.rng
        handle = CacheRequestHandle(args[0], 0) if args and isinstance(args[0], str) else None
        lags = vars(cache).get("read_lags")
        if kind == "commit":
            rid, kv_hints, tokens, tail, prefill = args
            req = self.req(rid)
            req.kv_hints = kv_hints
            if prefill is not None:
                waited, seconds, cached, total = prefill
                stats = req.time_stats
                stats.forward_entry_time = stats.wait_queue_entry_time + waited
                stats.prefill_finished_time = stats.forward_entry_time + seconds
                req.cached_tokens = cached
                req.origin_input_ids = array("q", range(total))
            if tail is not None:
                self.nodes.append(tail)
            ntp.on_turn_committed(
                req=req,
                key=T.key_of(tokens),
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
            for req in ntp.tick(waiting_queue=list(self.queue), now=now):
                issued = args[0] and self.ask_gate(req.cache_request_handle, cache.resident)
                if issued:
                    cache.ongoing.add(req.cache_request_handle)
                ntp.on_refetch_result(req=req, issued=bool(issued))
        elif kind == "resident":
            cache.resident = cache.matched_tokens = args[0]
            cache.in_device = args[1]
        elif kind == "ongoing":
            if args[1]:
                cache.ongoing.add(handle)
            elif not cache.read_scheduler.is_parked(handle):
                cache.ongoing.discard(handle)
        elif kind == "land":
            cache.read_scheduler.discard(handle)
            cache.land(handle, args[1])
            if cache.prefetch_landed_fn is not None:
                cache.prefetch_landed_fn(handle, args[2])
        elif kind == "park":
            if not cache.read_scheduler.is_parked(handle):
                cache.park(handle, start=args[2], tokens=args[1] * PAGE)
        elif kind == "dispatch":
            cache.read_scheduler.dispatch(
                now=now,
                deadline_of=ntp.read_deadline,
                in_flight_pages=args[0],
                held_tokens=0,
                not_before_of=cache.read_not_before_fn,
            )
        elif kind == "gate":
            self.ask_gate(handle, args[1])
        elif kind == "lookup":
            if cache.prefetch_lookup_fn is not None:
                cache.prefetch_lookup_fn(
                    handle, storage_start=args[1], requested_tokens=args[2], hit_tokens=args[3]
                )
        elif kind == "deadline":
            ntp.read_deadline(handle)
        elif kind == "refuse":
            cache.refusals.append((handle, args[1]))
            if cache.prefetch_refused_fn is not None:
                cache.prefetch_refused_fn(handle, tokens=args[1], cut=args[2], now=now)
        elif kind == "victims":
            if cache.soft_pin_victims_fn is not None:
                for victim in cache.soft_pin_victims_fn(handle, args[1]):
                    cache.finish(victim, "bumped")
        elif kind == "evict":
            observer = cache.tree_core.host_eviction_observer
            if observer is not None:
                node_id = rng.choice(self.nodes) if self.nodes and rng.random() < 0.7 else 10**7
                observer(tree_node(node_id), args[0])
        elif kind == "victim":
            cache.victim = (
                None if args[0] is None or not self.nodes else tree_node(rng.choice(self.nodes))
            )
        elif kind == "lags":
            if lags is not None:
                lags.lookup.observe(args[0])
                lags.fixed.observe(args[1])
        elif kind == "write_hold":
            if lags is not None:
                lags.note_write_hold(held=args[0], now=now)
        elif kind == "backlog":
            cache.backlog_pages = args[0]
            cache.pending_backup_tokens = args[1]
        elif kind == "limiter":
            cache.rate_limited = args[0]
        elif kind == "read_rate":
            cache.cache_controller.storage_read_ms_per_page = args[0]
        elif kind == "staged":
            ntp.is_staged_handle(handle)


def random_events(rng, count):
    sessions = [f"s{i}" for i in range(6)]
    turn = dict.fromkeys(sessions, 0)
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
                kv_hints = T.hints(next_rid, payload=payload)
            else:
                payload = {
                    "session": session,
                    "next_rid": next_rid,
                    "gaps": rng.choice(GAP_TABLES),
                    "replay_reply_key": rng.choice((None, "tool", "tool/x", "_end")),
                }
                kv_hints = T.hints(next_rid, version="2", payload=payload)
            if kv_hints is not None:
                phase[session] = "wait"
            tokens = rng.choice((0, 8, 16, 40, 1000)) * PAGE
            tail = rng.choice((None, next(node)))
            total = rng.choice((600, 9000, 40_000))
            prefill = rng.choice(
                (
                    None,
                    (
                        rng.choice((0.0, 0.01, 3.0)),
                        rng.choice((0.02, 0.3, 2.5)),
                        rng.choice((0, 500, total)),
                        total,
                    ),
                )
            )
            events.append((dt, ("commit", rid, kv_hints, tokens, tail, prefill)))
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
                turn[session] += 0 if admitted else 1
            if admitted:
                events.append((dt, ("admit", rid, rng.choice((0, 0, 640)))))
            else:
                events.append((dt, ("abort", rid)))
        elif roll < 0.66:
            events.append((dt, ("tick", rng.random() < 0.8)))
        else:
            known = queued + [f"{s}-t{turn[s]}" for s in waiting]
            rid = rng.choice(known) if known else "nobody"
            resident = rng.choice((0, 4, 16, 1000)) * PAGE
            start = rng.choice((0, 4, 16, 40)) * PAGE
            events.append(
                (
                    dt,
                    rng.choice(
                        (
                            ("resident", resident, rng.choice((None, 4 * PAGE))),
                            ("ongoing", rid, rng.random() < 0.5),
                            ("land", rid, rng.choice((640, 20_000)), next(node)),
                            ("park", rid, rng.choice((1, 10, 400)), start),
                            ("park", rid, rng.choice((1, 10, 400)), start),
                            ("park", rid, rng.choice((1, 10, 400)), start),
                            ("park", rid, rng.choice((1, 10, 400)), start),
                            ("dispatch", rng.choice((0, 300, 600))),
                            ("gate", rid, resident),
                            ("gate", rid, resident),
                            ("lookup", rid, start, 40 * PAGE, rng.choice((0, 0, 64, 320, 2560))),
                            ("lookup", rid, start, 40 * PAGE, rng.choice((0, 0, 64, 320, 2560))),
                            ("deadline", rid),
                            ("refuse", rid, rng.choice((640, 30_000)), rng.random() < 0.5),
                            ("victims", rid, rng.choice((640, 40_000))),
                            ("evict", rng.choice((64, 640, 20_000))),
                            ("evict", rng.choice((64, 640, 20_000))),
                            ("victim", rng.choice((None, 1))),
                            ("lags", rng.choice((0.005, 0.02, 0.4)), rng.choice((-0.03, 0.01, 0.2))),
                            ("lags", rng.choice((0.005, 0.02, 0.4)), rng.choice((-0.03, 0.01, 0.2))),
                            ("write_hold", rng.random() < 0.5),
                            ("backlog", rng.choice((0, 300, 3000)), rng.choice((0, 60_000))),
                            ("limiter", rng.random() < 0.2),
                            ("read_rate", rng.choice((None, 2.0, 5.0))),
                            ("staged", rid),
                        )
                    ),
                )
            )
    return events


def random_config(rng):
    switches = tuple(rng.random() < 0.75 for _ in BOOL_SWITCHES)
    return dict(
        mode=rng.choice(("on", "on", "on", "observe")),
        timing=rng.choice(("horizon", "just_in_time", "expected_cost", "arrival")),
        gate=rng.choice(("on", "on", "off")),
        hold=rng.choice(("on", "on", "off")),
        soft_pins=rng.choice(("on", "off")),
        policy=rng.choice(("next_use", "return_chance")),
        score=rng.choice(("entry", "samples", "lru", "queue")),
        interval=rng.choice((0.0, 1.0, 1.0)),
        switches=switches,
        # Measured prices refuse a learned price.
        learning="off" if switches[0] else rng.choice(("off", "harm")),
    )


ACTING = (
    "_measured_prices",
    "_free_empty_lookups",
    "_lateness_weight",
    "_read_line",
    "_hold_quantile",
    "_recompute_by_price",
    "_recompute_when_starved",
)


class Kinds(logging.Handler):
    """Counts the HICACHE_EVT lines logged, by kind (and by decision for the gate)."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.kinds = collections.Counter()

    def emit(self, record):
        words = record.getMessage().split()
        if len(words) > 1 and words[0] == "HICACHE_EVT":
            kind = words[1]
            if kind == "ntp_gate":
                kind += ":" + dict(w.split("=", 1) for w in words[2:])["decision"]
            if kind == "ntp_hold" and any(w.startswith("q=") for w in words):
                kind += ":quantile"
            if kind == "ec_start" and any(w.startswith("c=") for w in words):
                kind += ":weighed"
            self.kinds[kind] += 1


def test_no_event_sequence_raises_with_the_switches_on():
    counter = Kinds()
    logger = logging.getLogger(T.LOGGER)
    logger.setLevel(logging.INFO)
    logger.addHandler(counter)
    try:
        run_scenarios(counter)
    finally:
        logger.removeHandler(counter)


def run_scenarios(counter):
    rng = random.Random(20261126)
    refused = collections.Counter()
    built = collections.Counter()
    acting = collections.Counter()
    failures = []
    for scenario in range(600):
        clock = SimpleNamespace(now=0.0)
        config = random_config(rng)
        events = random_events(rng, 500)
        overrides = []
        try:
            wanted = [(envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, config["interval"])]
            wanted += list(zip(BOOL_SWITCHES, config["switches"]))
            for switch, value in wanted:
                manager = switch.override(value)
                manager.__enter__()
                overrides.append(manager)
            try:
                run = Run(config, rng, lambda: clock.now)
            except ValueError as error:
                refused[str(error).split(";")[0].split(" needs ")[0].split(" replaces ")[0]] += 1
                continue
            built[(config["mode"], config["timing"])] += 1
            # As on the server: the Scheduler ticks every pass from startup, so no hook is
            # ever called before the first tick (on_read_landed stamps the last tick's clock).
            run.ntp.tick(waiting_queue=[], now=0.0)
            for name in ACTING:
                acting[name] += bool(vars(run.ntp)[name])
            for step, (dt, event) in enumerate(events):
                clock.now += dt
                try:
                    run.apply(event, clock.now)
                except Exception:
                    failures.append((scenario, step, event[0], config, traceback.format_exc()))
                    break
        finally:
            for manager in reversed(overrides):
                manager.__exit__(None, None, None)
    print()
    print("==== reviewer switches-on fuzz ====")
    print("scenarios built by (mode, timing):", dict(sorted(built.items())))
    print("refused at construction:", dict(refused))
    print("scenarios in which each switch acted:", dict(acting))
    print("lines logged by kind:", dict(sorted(counter.kinds.items())))
    print("scenarios that raised:", len(failures))
    seen = set()
    for scenario, step, kind, config, text in failures:
        last = text.strip().splitlines()[-1]
        where = [line.strip() for line in text.splitlines() if line.strip().startswith("File")][-1]
        if (last, where) in seen:
            continue
        seen.add((last, where))
        print("---- scenario", scenario, "step", step, "event", kind)
        print("config:", {k: v for k, v in config.items() if k != "switches"})
        print(
            "switches on:",
            [s.name.replace("SGLANG_HICACHE_", "") for s, on in zip(BOOL_SWITCHES, config["switches"]) if on],
        )
        print(text)
    print("==== end ====")
    assert not failures
