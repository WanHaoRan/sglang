"""Oracle next-turn prefetcher: plan registration, queue tracking and triggers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import logging
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from test_session_unified_radix_cache import make_params

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.mem_cache.next_turn_estimators import ArrivalOffset, L3LoadModel
from sglang.srt.mem_cache.next_turn_prefetch import NextTurnPrefetcher
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

LOGGER = "sglang.srt.mem_cache.next_turn_prefetch"
PAGE = 64
KEY_TOKENS = 16 * PAGE


def hints(
    next_rid, gap_s=5.0, action_type="sglang.next_turn", version="1", payload=None
):
    if payload is None:
        payload = {"session": "s", "next_rid": next_rid, "next_gap_s": gap_s}
    action = KvHintAction(
        action_id="next_turn",
        action_type=action_type,
        action_version=version,
        payload=payload,
    )
    return KvHintsEnvelope(protocol_version="1", message_id="m", actions=[action])


def make_req(
    rid, *, kv_hints=None, finished=True, reason=None, entered=0.0, last_match=None
):
    return SimpleNamespace(
        rid=rid,
        kv_hints=kv_hints,
        finished=lambda: finished,
        finished_reason=FINISH_LENGTH(length=8) if reason is None else reason,
        time_stats=SimpleNamespace(
            wait_queue_entry_time=entered, forward_entry_time=0.0
        ),
        storage_hit_length=0,
        storage_prefetch_last_match_len=last_match,
        cache_request_handle=CacheRequestHandle(rid=rid, attempt_id=0),
    )


def key_of(tokens):
    return RadixKey(array("q", range(tokens)))


class ReadOnlyCache:
    """The only calls an observe-mode prefetcher may make; any other is an error."""

    prefetch_threshold = 256

    def __init__(self):
        self.resident = 0
        self.backlog_pages = 0
        self.probes = 0

    def resident_prefix_len(self, key):
        self.probes += 1
        resident = min(self.resident, len(key))
        return resident, resident

    def storage_read_backlog_pages(self):
        return self.backlog_pages


ANCHOR = 7


class ActingCache(ReadOnlyCache):
    """Adds the state-changing calls of mode on, recorded instead of performed."""

    hicache_storage_pass_prefix_keys = False

    def __init__(self):
        super().__init__()
        self.rate_limited = False
        self.backed_up = True
        self.matched_tokens = 0
        self.ongoing = set()
        self.prefetches = []
        self.finished = []
        self.cancelled = []
        self.cache_controller = SimpleNamespace(
            prefetch_rate_limited=lambda: self.rate_limited
        )
        self.storage_prefetch_retries = SimpleNamespace(cancel=self.cancelled.append)

    def match_prefix(self, params):
        return SimpleNamespace(
            last_host_node=ANCHOR,
            device_indices=range(self.matched_tokens),
            host_hit_length=0,
        )

    def is_backuped(self, node):
        return self.backed_up

    def is_root(self, node):
        return False

    def get_last_hash_value(self, node):
        return "hash-of-anchor"

    def has_ongoing_prefetch(self, handle):
        return handle in self.ongoing

    def prefetch_from_storage(self, handle, anchor, tokens, last_hash, keys, **kw):
        kw["matched_prefix_tokens"] = list(kw["matched_prefix_tokens"])
        self.prefetches.append((handle, anchor, list(tokens), last_hash, keys, kw))
        self.ongoing.add(handle)

    def finish(self, handle, outcome):
        self.finished.append((handle, outcome))


class PrefetcherTestCase(CustomTestCase):
    mode = "observe"

    def setUp(self):
        self.cache = self.make_cache()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE
        )

    def make_cache(self):
        return ReadOnlyCache()

    def events(self, fn):
        """(kind, fields) of every HICACHE_EVT line logged while fn runs."""
        with self.assertLogs(LOGGER, level="INFO") as logs:
            fn()
            logging.getLogger(LOGGER).info("HICACHE_EVT end rid=-")
        found = []
        for line in logs.output:
            message = line.split(":", 2)[2]
            if message.startswith("HICACHE_EVT "):
                kind, *fields = message.split()[1:]
                found.append((kind, dict(field.split("=", 1) for field in fields)))
        return found[:-1]

    def commit(self, rid, next_rid, *, now, gap_s=5.0, tokens=KEY_TOKENS):
        self.ntp.on_turn_committed(
            req=make_req(rid, kv_hints=hints(next_rid, gap_s)),
            key=key_of(tokens),
            rotation_tail_declined=False,
            now=now,
        )

    def admit_through_queue(self, rids, *, now):
        """Pass requests through the queue: seen at `now`, admitted a tick later."""
        reqs = [make_req(rid, entered=now) for rid in rids]
        self.ntp.tick(waiting_queue=reqs, now=now)
        for req in reqs:
            req.time_stats.forward_entry_time = now + 0.01
        self.ntp.tick(waiting_queue=[], now=now + 0.01)


def cpu_unified_cache():
    return UnifiedRadixCache(make_params(enable_session=False, page_size=2))


class TestCreate(CustomTestCase):
    def test_refuses_setups_where_its_decisions_would_be_wrong(self):
        """Wall-clock triggers are rank-local and the probe exists only in the
        python tree core: every other setup is refused at startup."""
        ok = dict(
            mode="observe",
            tree_cache=cpu_unified_cache(),
            page_size=2,
            enable_hicache_storage=True,
            tp_size=1,
            pp_size=1,
            enable_dp_attention=False,
            disaggregation_mode=DisaggregationMode.NULL,
        )
        self.assertIsInstance(NextTurnPrefetcher.create(**ok), NextTurnPrefetcher)
        for change in (
            {"enable_hicache_storage": False},
            {"tp_size": 2},
            {"pp_size": 2},
            {"enable_dp_attention": True},
            {"disaggregation_mode": DisaggregationMode.PREFILL},
            {"tree_cache": SimpleNamespace(disable=False)},
        ):
            with self.subTest(change):
                with self.assertRaises(ValueError):
                    NextTurnPrefetcher.create(**{**ok, **change})


class TestCommitHook(CustomTestCase):
    def test_finished_turn_hands_its_page_aligned_committed_key_over(self):
        """The plan key is prompt + output as inserted, page-aligned: the prefix
        the next turn extends. A retraction (no insert) registers nothing."""
        cache = cpu_unified_cache()
        ntp = NextTurnPrefetcher(mode="observe", tree_cache=cache, page_size=2)
        cache.next_turn_prefetch = ntp
        prompt, output = array("q", range(1, 10)), array("q", [100, 101])
        n = len(prompt) + len(output)
        cache.req_to_token_pool.req_to_token[0, :n] = (
            cache.token_to_kv_pool_allocator.alloc(12)[:n]
        )
        req = make_req("s-t0", kv_hints=hints("s-t1"))
        req.__dict__.update(
            session=None,
            origin_input_ids=prompt,
            output_ids=output,
            extra_key=None,
            cache_salt=None,
            priority=0,
            session_id=None,
            kv_rotation_base=None,
            last_node=None,
            swa_prefix_lock_released=False,
            kv=SimpleNamespace(
                req_pool_idx=0,
                cache_protected_len=0,
                swa_evicted_seqlen=0,
                swa_dead_lo=lambda page_size: 0,
            ),
        )
        cache.cache_finished_req(req, is_insert=False, owned_kv_len=0)
        self.assertNotIn("s-t1", ntp._plans)

        cache.cache_finished_req(req, owned_kv_len=n)
        key = ntp._plans["s-t1"].key
        self.assertEqual(list(key.token_ids), list(prompt) + [100])
        self.assertEqual(cache.resident_prefix_len(key), (10, 10))


class TestPlanRegistration(PrefetcherTestCase):
    def test_only_finished_hinted_turns_register_a_plan(self):
        """The replay client stops turns with max_tokens, so FINISH_LENGTH must
        register; aborts, retractions, foreign or malformed hints, empty keys and
        declined rotation tails must not."""
        cases = {
            "length stop": (make_req("a", kv_hints=hints("a2")), KEY_TOKENS, False),
            "abort": (
                make_req("b", kv_hints=hints("b2"), reason=FINISH_ABORT()),
                KEY_TOKENS,
                False,
            ),
            "retracted": (
                make_req("c", kv_hints=hints("c2"), finished=False),
                KEY_TOKENS,
                False,
            ),
            "no hint": (make_req("d"), KEY_TOKENS, False),
            "other action": (
                make_req("e", kv_hints=hints("e2", action_type="vendor.x")),
                KEY_TOKENS,
                False,
            ),
            "other version": (
                make_req("e", kv_hints=hints("e3", version="2")),
                KEY_TOKENS,
                False,
            ),
            "negative gap": (
                make_req("f", kv_hints=hints("f2", gap_s=-1.0)),
                KEY_TOKENS,
                False,
            ),
            "empty key": (make_req("g", kv_hints=hints("g2")), 0, False),
            "declined tail": (make_req("h", kv_hints=hints("h2")), KEY_TOKENS, True),
        }
        for name, (req, tokens, declined) in cases.items():
            with self.subTest(name):
                found = self.events(
                    lambda: self.ntp.on_turn_committed(
                        req=req,
                        key=key_of(tokens),
                        rotation_tail_declined=declined,
                        now=0.0,
                    )
                )
                plans = [fields for kind, fields in found if kind == "ntp_plan"]
                if name == "length stop":
                    self.assertEqual(len(plans), 1)
                    # Keyed by the next turn, which is what arrives later.
                    self.assertEqual(plans[0]["rid"], "a2")
                    self.assertEqual(plans[0]["chain"], str(KEY_TOKENS))
                else:
                    self.assertEqual(plans, [])


class TestQueueTracking(PrefetcherTestCase):
    def setUp(self):
        super().setUp()
        self.cache.resident = KEY_TOKENS  # nothing evicted: no trigger noise

    def test_arrival_admission_and_abort_are_told_apart(self):
        self.commit("a1", "a2", now=0.0)
        self.commit("b1", "b2", now=0.0)
        a2, b2 = make_req("a2", entered=5.15), make_req("b2", entered=5.25)
        found = self.events(lambda: self.ntp.tick(waiting_queue=[a2, b2], now=5.3))
        arrivals = {f["rid"]: f for kind, f in found if kind == "ntp_arrive"}
        self.assertEqual(set(arrivals), {"a2", "b2"})
        # Raw offset = queue entry - commit - hinted gap.
        self.assertEqual(arrivals["a2"]["delta_ms"], "150")
        self.assertEqual((arrivals["b2"]["pos"], arrivals["b2"]["qlen"]), ("1", "2"))

        a2.time_stats.forward_entry_time = 5.9  # admitted; b2 aborted while queued
        found = self.events(lambda: self.ntp.tick(waiting_queue=[], now=6.0))
        self.assertEqual(
            [(kind, f["rid"]) for kind, f in found],
            [("ntp_admit", "a2"), ("ntp_expire", "b2")],
        )
        self.assertEqual(found[0][1]["wait_ms"], "750")
        self.assertEqual(found[1][1]["reason"], "abort")

    def test_retracted_request_is_admitted_once(self):
        """Retraction puts a request back in the queue with its first forward
        entry kept; leaving the queue again is not a second admission."""
        req = make_req("x", entered=1.0)
        self.ntp.tick(waiting_queue=[req], now=1.0)
        req.time_stats.forward_entry_time = 2.0
        self.ntp.tick(waiting_queue=[], now=2.0)
        req.time_stats.wait_queue_entry_time = 3.0
        self.ntp.tick(waiting_queue=[req], now=3.0)
        self.ntp.tick(waiting_queue=[], now=4.0)
        self.assertEqual(list(self.ntp._admissions.stamps), [2.0])

    def test_pending_plan_expires_and_a_late_turn_is_not_matched(self):
        self.commit("a1", "a2", now=0.0, gap_s=5.0)
        expires_at = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=5.0) + 60.0
        found = self.events(
            lambda: [
                self.ntp.tick(waiting_queue=[], now=t)
                for t in (10.0, expires_at - 0.1, expires_at + 0.1)
            ]
        )
        self.assertEqual(
            [(kind, f["rid"], f["reason"]) for kind, f in found],
            [("ntp_expire", "a2", "timeout")],
        )
        late = make_req("a2", entered=expires_at + 5)
        found = self.events(
            lambda: self.ntp.tick(waiting_queue=[late], now=expires_at + 5)
        )
        self.assertEqual(found, [])


class TestTriggers(PrefetcherTestCase):
    def due_events(self, now, queue=()):
        found = self.events(
            lambda: self.assertEqual(
                self.ntp.tick(waiting_queue=list(queue), now=now), []
            )
        )
        return [fields for kind, fields in found if kind == "ntp_due"]

    def test_case1_is_due_when_now_crosses_arrival_minus_load_of_evicted_pages(self):
        """Empty queue, so admission = predicted arrival P; the trigger is P minus
        the load time of the pages the probe finds evicted, never earlier."""
        arrival = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=5.0)
        for resident_pages in (0, 8):
            with self.subTest(resident_pages=resident_pages):
                self.setUp()
                self.cache.resident = resident_pages * PAGE
                missing_pages = KEY_TOKENS // PAGE - resident_pages
                trigger = arrival - L3LoadModel().seconds(
                    own_pages=missing_pages, backlog_pages=0
                )
                self.commit("a1", "a2", now=0.0)
                self.assertEqual(self.due_events(trigger - 0.01), [])
                (due,) = self.due_events(trigger + 0.01)
                self.assertEqual(
                    (due["rid"], due["case"], due["l3_pages"], due["est"]),
                    ("a2", "1", str(missing_pages), "empty"),
                )

    def test_a_plan_whose_turn_is_far_off_costs_no_probes(self):
        """The tick runs every scheduler pass: until a whole-key read would have
        to start, a pending plan is not probed at all."""
        self.commit("a1", "a2", now=0.0, gap_s=600.0)
        for now in range(1, 300):
            self.ntp.tick(waiting_queue=[], now=float(now))
        self.assertEqual(self.cache.probes, 0)

    def test_nothing_to_read_below_the_prefetch_threshold(self):
        self.cache.resident = KEY_TOKENS - 3 * PAGE  # 192 tokens missing < 256
        self.commit("a1", "a2", now=0.0)
        self.assertEqual(
            [due for t in (1.0, 4.5, 5.0, 6.0) for due in self.due_events(t)], []
        )

    def test_case2_without_a_rate_fires_at_the_queue_head_with_rate_limited_probes(
        self,
    ):
        """With no admission rate yet, a queued turn is due only near the head;
        however often the scheduler ticks, its residency is probed every 0.5 s."""
        self.cache.resident = KEY_TOKENS
        self.commit("a1", "a2", now=0.0)
        others = [make_req(f"o{i}", entered=5.0) for i in range(3)]
        a2 = make_req("a2", entered=5.1)
        self.assertEqual(self.due_events(5.1, queue=others + [a2]), [])

        self.cache.resident = 0
        probes = self.cache.probes
        (due,) = self.due_events(5.2, queue=[others[0], a2])
        self.assertEqual((due["rid"], due["case"], due["est"]), ("a2", "2", "head"))
        for i in range(40):
            self.due_events(5.21 + i * 0.01, queue=[others[0], a2])
        self.assertEqual(self.cache.probes - probes, 1)


class TestCase1Staging(PrefetcherTestCase):
    mode = "on"

    def make_cache(self):
        return ActingCache()

    def setUp(self):
        super().setUp()
        self.matched = self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.arrival = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=5.0)
        self.trigger = self.arrival - L3LoadModel().seconds(
            own_pages=(KEY_TOKENS - self.matched) // PAGE, backlog_pages=0
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.commit("s-t0", "s-t1", now=0.0)

    def tick(self, now, queue=()):
        return self.events(lambda: self.ntp.tick(waiting_queue=list(queue), now=now))

    def test_stages_once_at_the_trigger_under_the_next_turns_handle(self):
        """The read starts at the trigger with the arguments the request's own
        arrival prefetch would use, under its handle, so the request joins it."""
        self.tick(self.trigger - 0.01)
        self.assertEqual(self.cache.prefetches, [])
        found = self.tick(self.trigger + 0.01)
        tokens = list(range(KEY_TOKENS))
        self.assertEqual(
            self.cache.prefetches,
            [
                (
                    self.handle,
                    ANCHOR,
                    tokens[self.matched :],
                    "hash-of-anchor",
                    None,
                    {
                        "matched_prefix_tokens": tokens[: self.matched],
                        "extra_key": None,
                        "cache_salt": None,
                        "storage_hit_end": None,
                    },
                )
            ],
        )
        (stage,) = [f for kind, f in found if kind == "ntp_stage"]
        self.assertEqual(
            (stage["matched"], stage["inflight"]), (str(self.matched), "1")
        )
        self.tick(self.trigger + 1.0)
        self.assertEqual(len(self.cache.prefetches), 1)

    def test_rate_limit_defers_staging_to_a_later_tick(self):
        """A decline while rate-limited would arm a paced retry for a rid that is
        not queued; the prefetcher waits for the limiter instead."""
        self.cache.rate_limited = True
        self.tick(self.trigger + 0.01)
        self.assertEqual(self.cache.prefetches, [])
        self.cache.rate_limited = False
        self.tick(self.trigger + 0.02)
        self.assertEqual(len(self.cache.prefetches), 1)

    def test_unbacked_anchor_is_not_fetched(self):
        self.cache.backed_up = False
        found = self.tick(self.trigger + 0.01)
        self.assertEqual(self.cache.prefetches, [])
        (stage,) = [f for kind, f in found if kind == "ntp_stage"]
        self.assertEqual((stage["gate"], stage["inflight"]), ("0", "0"))

    def test_never_stages_once_the_turn_has_arrived(self):
        """After arrival the request owns its handle; case 1 must not touch it,
        even once a case-1 schedule (P + queue wait - load) would be due."""
        self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)
        early = make_req("s-t1", entered=self.arrival - 2.0)
        self.tick(self.arrival - 2.0, queue=[early])
        self.tick(self.arrival + 2.5, queue=[early])
        self.assertEqual(self.cache.prefetches, [])

    def test_ended_staged_read_cancels_the_retry_it_may_have_armed(self):
        self.tick(self.trigger + 0.01)
        self.cache.ongoing.clear()
        self.tick(self.trigger + 0.6)
        self.assertEqual(self.cache.cancelled, ["s-t1"])

    def test_expired_staged_plan_waits_for_its_read_before_releasing_it(self):
        """Releasing a read mid-I/O frees its pages uninserted; expiry waits for
        the read to land, then pops its records with finish(ABORT)."""
        self.tick(self.trigger + 0.01)
        expires_at = self.arrival + 60.0
        found = self.tick(expires_at + 0.1)
        self.assertEqual((self.cache.finished, found), ([], []))
        self.cache.ongoing.clear()
        found = self.tick(expires_at + 0.2)
        self.assertEqual(
            self.cache.finished, [(self.handle, CacheRequestOutcome.ABORT)]
        )
        self.assertEqual(
            [(kind, f["staged"]) for kind, f in found], [("ntp_expire", "1")]
        )


class TestCase2Refetch(PrefetcherTestCase):
    mode = "on"

    def make_cache(self):
        return ActingCache()

    def setUp(self):
        super().setUp()
        self.cache.resident = KEY_TOKENS
        self.commit("s-t0", "s-t1", now=0.0)
        self.req = make_req("s-t1", entered=5.1)

    def scheduler_pass(self, now, *, queue=None, issued=True):
        """tick(), then what Scheduler._on_next_turn_refetch does per returned req."""
        queue = [self.req] if queue is None else queue
        returned = []

        def run():
            for req in self.ntp.tick(waiting_queue=queue, now=now):
                returned.append(req.rid)
                if issued:
                    self.cache.ongoing.add(req.cache_request_handle)
                self.ntp.on_refetch_result(req=req, issued=issued)

        return returned, self.events(run)

    def test_evicted_queued_turn_is_refetched_and_not_again_while_it_reads(self):
        self.assertEqual(self.scheduler_pass(5.1)[0], [])
        self.cache.resident = 0  # evicted while queued
        returned, found = self.scheduler_pass(5.7)
        self.assertEqual(returned, ["s-t1"])
        (refetch,) = [f for kind, f in found if kind == "ntp_refetch"]
        self.assertEqual((refetch["issued"], refetch["shrink"]), ("1", str(KEY_TOKENS)))
        for now in (5.8, 6.5, 7.5):
            self.assertEqual(self.scheduler_pass(now)[0], [])

    def test_held_back_below_threshold_behind_a_running_read_or_the_limiter(self):
        self.scheduler_pass(5.1)
        self.cache.resident = KEY_TOKENS - 3 * PAGE
        self.assertEqual(self.scheduler_pass(5.7)[0], [])
        self.cache.resident = 0
        self.cache.ongoing.add(self.req.cache_request_handle)  # its own lookup
        self.assertEqual(self.scheduler_pass(6.3)[0], [])
        self.cache.ongoing.clear()
        self.cache.rate_limited = True
        self.assertEqual(self.scheduler_pass(6.9)[0], [])
        self.cache.rate_limited = False
        self.assertEqual(self.scheduler_pass(7.0)[0], ["s-t1"])

    def test_unissued_refetch_stays_armed_without_counting_as_a_read(self):
        self.scheduler_pass(5.1)
        self.cache.resident = 0
        returned, found = self.scheduler_pass(5.7, issued=False)
        self.assertEqual(returned, ["s-t1"])
        self.assertEqual(self.scheduler_pass(5.8, issued=False)[0], [])
        returned, found = self.scheduler_pass(6.3)
        self.assertEqual(returned, ["s-t1"])
        self.assertNotIn("ntp_reevict", [kind for kind, _ in found])

    def test_reevicted_pages_are_read_again_up_to_the_cap(self):
        """A fetched page is an evictable host leaf; evicted again before
        admission it is read again, but at most MAX_REFETCHES times."""
        self.scheduler_pass(5.1)
        self.cache.resident = 0
        self.assertEqual(self.scheduler_pass(5.7)[0], ["s-t1"])
        self.cache.ongoing.clear()
        self.cache.resident = KEY_TOKENS  # the read landed
        self.assertEqual(self.scheduler_pass(6.3)[0], [])
        self.cache.resident = 0  # and was evicted again
        returned, found = self.scheduler_pass(6.9)
        self.assertEqual(returned, ["s-t1"])
        self.assertEqual(
            [(f["refetches"], f["capped"]) for k, f in found if k == "ntp_reevict"],
            [("1", "0")],
        )
        self.cache.ongoing.clear()
        returned, found = self.scheduler_pass(7.5)
        self.assertEqual(returned, [])
        self.assertEqual(
            [(f["refetches"], f["capped"]) for k, f in found if k == "ntp_reevict"],
            [("2", "1")],
        )
        self.assertEqual(self.scheduler_pass(8.1), ([], []))

    def test_waits_until_the_remaining_queue_wait_fits_the_load_time(self):
        """Due when admission (now + residual wait at its position) minus the load
        time has passed: deep in the queue it waits, at the head it reads."""
        self.admit_through_queue([f"x{i}" for i in range(10)], now=1.0)
        self.cache.resident = 0
        self.cache.backlog_pages = 1000  # load ~3.4 s; residual 3 s at the head
        others = [make_req(f"o{i}", entered=5.0) for i in range(2)]
        self.assertEqual(self.scheduler_pass(5.1, queue=others + [self.req])[0], [])
        self.assertEqual(self.scheduler_pass(5.7)[0], ["s-t1"])

    def test_a_landed_read_is_not_taken_for_a_re_eviction(self):
        """A residency probe taken while a re-fetch is in flight goes stale when
        the read lands; acting on it re-fetched pages that were already back
        (logged as ntp_reevict) and spent the read cap on them."""
        self.scheduler_pass(5.1)
        self.cache.resident = 0
        self.assertEqual(self.scheduler_pass(5.7)[0], ["s-t1"])  # read in flight
        self.scheduler_pass(6.25)  # periodic probe: still missing, read running
        self.cache.ongoing.clear()
        self.cache.resident = KEY_TOKENS  # the read lands
        returned, found = self.scheduler_pass(6.5)
        self.assertEqual(returned, [])
        self.assertNotIn("ntp_reevict", [kind for kind, _ in found])

    def test_admitted_turn_drops_its_plan(self):
        """After admission a retracted re-entry is a different story: no refetch."""
        self.scheduler_pass(5.1)
        self.cache.resident = 0
        self.req.time_stats.forward_entry_time = 5.5
        self.scheduler_pass(5.6, queue=[])
        self.req.time_stats.wait_queue_entry_time = 6.0
        self.assertEqual(self.scheduler_pass(6.7)[0], [])


if __name__ == "__main__":
    unittest.main()
