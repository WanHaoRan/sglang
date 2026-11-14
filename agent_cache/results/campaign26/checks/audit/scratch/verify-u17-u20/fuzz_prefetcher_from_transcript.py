"""Random session timelines through the prefetcher (fake cache) under every new mode: any exception is a finding."""
import random, sys, collections
sys.path.insert(0, "/sgl-workspace/sglang/test/registered/unit/mem_cache")
import test_next_turn_prefetch as T
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.next_use_eviction import NextUseStrategy, ReturnChanceStrategy
from types import SimpleNamespace

ENTRIES = [
    {"key": "a", "p10": 45.0, "p50": 50.0, "p90": 56.0, "n": 40},
    {"key": "b", "p10": 0.85, "p50": 22.0, "p90": 58.0, "n": 500},
    {"key": "c", "p10": 6.9, "p50": 7.3, "p90": 13.0, "n": 90},
    {"key": "d", "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 2},
    {"key": "e", "p10": 3.0, "p50": 3.0, "p90": 3.0, "n": 2},
    {"key": "f", "p50": 12.0},
    {"key": "_end", "ended": True},
]

def run(seed, **kw):
    rng = random.Random(seed)
    cache = T.PricedCache()
    strategy = kw.pop("strategy")
    clock = [0.0]
    cache.tree_core = SimpleNamespace(eviction_strategy=strategy(clock) if strategy else None)
    ntp = NextTurnPrefetcher(mode="on", tree_cache=cache, page_size=T.PAGE, reply_key_source="hint", arrival_gate="on", arrival_hold="on", **kw)
    cache.prefetch_gate_fn = ntp.prefetch_gate
    sessions = {}   # sid -> dict(turn, state, t_next, req)
    queue = []
    now = 0.0
    stats = collections.Counter()
    for step in range(4000):
        now += rng.choice((0.01, 0.05, 0.3, 0.5, 1.0, 3.0))
        clock[0] = now
        # start or advance sessions
        if len(sessions) < 12 and rng.random() < 0.3:
            sid = f"s{len(sessions)}_{seed}"
            sessions[sid] = dict(turn=0, state="idle_new", t_next=now)
        for sid, s in list(sessions.items()):
            if s["state"] in ("idle", "idle_new") and now >= s["t_next"]:
                rid = f"{sid}-t{s['turn']}"
                req = T.make_req(rid, entered=now)
                s.update(state="queued", req=req)
                queue.append(req)
                # the request's own prefetch at arrival goes through the gate
                h = req.cache_request_handle
                if not cache.has_ongoing_prefetch(h) and ntp.prefetch_gate(h, rng.choice((0, 4 * T.PAGE, T.BIG_KEY_TOKENS))):
                    cache.ongoing.add(h); stats["arrival_reads"] += 1
            elif s["state"] == "queued" and rng.random() < 0.3 and queue and queue[0] is s["req"]:
                req = queue.pop(0)
                req.time_stats.forward_entry_time = now
                cache.ongoing.discard(req.cache_request_handle); cache.pinned.pop(req.cache_request_handle, None)
                s.update(state="running", t_done=now + rng.uniform(0.2, 4.0))
            elif s["state"] == "running" and now >= s["t_done"]:
                entry = rng.choice(ENTRIES)
                nxt = f"{sid}-t{s['turn'] + 1}"
                if rng.random() < 0.2:
                    kv = T.hints(nxt, rng.choice((0.0, 2.0, 30.0, 400.0)))           # exact v1
                else:
                    table = {"session": sid, "next_rid": nxt, "gaps": ENTRIES, "replay_reply_key": entry["key"]}
                    kv = T.hints("-", version="2", payload=table)
                done = T.make_req(f"{sid}-t{s['turn']}", kv_hints=kv)
                ntp.on_turn_committed(req=done, key=T.key_of(rng.choice((T.KEY_TOKENS, T.BIG_KEY_TOKENS))), rotation_tail_declined=False, now=now, tail_node_id=rng.randrange(10**6))
                s["turn"] += 1
                gap = rng.choice((0.0, 0.3, 5.0, 20.0, 60.0, 200.0, 2500.0, 1e9))
                s.update(state="idle", t_next=now + gap)
        cache.resident = rng.choice((0, 4 * T.PAGE, T.KEY_TOKENS, T.BIG_KEY_TOKENS))
        cache.in_device = rng.choice((None, 0, 4 * T.PAGE))
        cache.rate_limited = rng.random() < 0.05
        cache.backlog_pages = rng.choice((0, 0, 100, 4000))
        # staged reads land, get refused, or stay
        for h in list(cache.ongoing):
            r = rng.random()
            if r < 0.3:
                cache.land(h, rng.choice((300, 20_000, 60_000)))
            elif r < 0.4:
                cache.ongoing.discard(h)
        cache.refusals = rng.choice(([], [], [(CacheRequestHandle("nobody", 0), 30_000)], [(queue[0].cache_request_handle, 20_000)] if queue else []))
        cache.room = (rng.choice((0, 44_000, 90_000)), rng.choice((0, 20_000, 120_000)))
        refetch = ntp.tick(waiting_queue=list(queue), now=now)
        for req in refetch:
            ntp.on_refetch_result(req=req, issued=rng.random() < 0.7)
        for h in list(cache.pinned) + [CacheRequestHandle("nobody", 0)]:
            ntp.read_deadline(h)
        if queue and cache.soft_pin_victims_fn is not None and rng.random() < 0.2:
            for v in cache.soft_pin_victims_fn(queue[-1].cache_request_handle, rng.choice((1000, 50_000, 10**7))):
                cache.finish(v, None); stats["bumped"] += 1
        stats["staged"] = len(cache.prefetches)
        stats["released"] = len(cache.finished)
        # every registration must give a priority that sorts with the others
        strat = cache.tree_core.eviction_strategy
        if strat is not None and step % 50 == 0:
            ids = list(strat.registry._next_use)[-200:]
            sorted(strat.get_priority(SimpleNamespace(id=i, last_access_time=float(i % 7), parent=None, children={})) for i in ids)
            stats["priorities"] += len(ids)
    return stats

modes = [
    dict(read_timing="expected_cost", strategy=lambda c: NextUseStrategy(horizon_s=300.0, clock=lambda: c[0])),
    dict(read_timing="expected_cost", memory_price=0.0, strategy=lambda c: NextUseStrategy(horizon_s=300.0, clock=lambda: c[0])),
    dict(read_timing="expected_cost", memory_price=1e-3, price_learning="on", soft_pins="on", strategy=lambda c: ReturnChanceStrategy(horizon_s=30.0, clock=lambda: c[0])),
    dict(read_timing="just_in_time", strategy=lambda c: ReturnChanceStrategy(horizon_s=30.0, clock=lambda: c[0])),
    dict(read_timing="expected_cost", price_learning="harm", soft_pins="on", strategy=lambda c: ReturnChanceStrategy(horizon_s=30.0, score="samples", clock=lambda: c[0])),
    dict(read_timing="arrival", strategy=lambda c: ReturnChanceStrategy(score="lru", clock=lambda: c[0])),
    dict(read_timing="just_in_time", landing_quantile=0.9, strategy=lambda c: NextUseStrategy(horizon_s=300.0, clock=lambda: c[0])),
    dict(read_timing="just_in_time", strategy=lambda c: NextUseStrategy(horizon_s=300.0, clock=lambda: c[0])),
    dict(read_timing="horizon", strategy=None),
]
import logging
logging.disable(logging.CRITICAL)
for i, m in enumerate(modes):
    tot = collections.Counter()
    for seed in range(12):
        tot.update(run(seed * 7 + i, **dict(m)))
    label = {k: v for k, v in m.items() if k != "strategy"}
    print("ok", label, dict(tot))
