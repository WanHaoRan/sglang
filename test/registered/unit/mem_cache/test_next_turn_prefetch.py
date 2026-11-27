"""Oracle next-turn prefetcher: plan registration, queue tracking and triggers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import contextlib
import logging
import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from test_session_unified_radix_cache import make_params

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
)
from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang.srt.mem_cache.next_turn_estimators import (
    LAG_MIN_SAMPLES,
    ArrivalOffset,
    L3LoadModel,
    ReadPathLags,
    residual_wait,
)
from sglang.srt.mem_cache.next_turn_policy import (
    GapDistribution,
    read_start,
    remaining_gaps,
)
from sglang.srt.mem_cache.next_turn_prefetch import (
    MAX_REFETCHES,
    MIN_EXPIRY_S,
    NTP_GPU_BOUND_S,
    NTP_HOLD_MARGIN_S,
    NTP_JIT_MARGIN_S,
    NTP_LEAD_HORIZON_S,
    NTP_MAX_FREE_LOOKUPS,
    NTP_MAX_PARKED,
    NTP_MAX_ROOM_RETRIES,
    NTP_MAX_STAGINGS,
    NTP_ROOM_RETRY_S,
    PROBE_INTERVAL_S,
    NextTurnPrefetcher,
)
from sglang.srt.mem_cache.next_turn_prices import LAM_CEILING, PrefillCostModel
from sglang.srt.mem_cache.next_use_eviction import (
    HOLDING_MIN_HORIZON_S,
    NextUseStrategy,
    Queued,
    ReturnChanceStrategy,
    ReturnForecast,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.reply_key import NO_CALL_KEY, reply_key
from sglang.srt.mem_cache.storage_read_scheduler import (
    ParkedRead,
    StorageReadScheduler,
)
from sglang.srt.mem_cache.storage_retention import RetentionSwitches
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
    rid,
    *,
    kv_hints=None,
    finished=True,
    reason=None,
    entered=0.0,
    last_match=None,
    tokenizer=None,
):
    return SimpleNamespace(
        rid=rid,
        kv_hints=kv_hints,
        tokenizer=tokenizer,
        output_ids=array("q"),
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
    # v1 wiring: no read scheduler (fifo order) and an eviction strategy without a next-use registry
    read_scheduler = None
    tree_core = SimpleNamespace(eviction_strategy=None)
    # The default: every trigger looks up, parked reads are not held, and refused reads
    # are reported to no one.
    prefetch_gate_fn = None
    read_not_before_fn = None
    prefetch_refused_fn = None

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
            prefetch_rate_limited=lambda: self.rate_limited,
            storage_read_ms_per_page=None,
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

    def test_the_factory_follows_the_end_switches_the_cache_resolved(self):
        """Tests hand the prefetcher its end switches as arguments; a server builds it
        through the factory, which must read them off the cache or both stay inert."""
        cache = cpu_unified_cache()
        create = lambda: NextTurnPrefetcher.create(
            mode="observe",
            tree_cache=cache,
            page_size=2,
            enable_hicache_storage=True,
            tp_size=1,
            pp_size=1,
            enable_dp_attention=False,
            disaggregation_mode=DisaggregationMode.NULL,
        )
        made = create()
        self.assertEqual((made._explicit_ends, made._retire_ended), (False, False))
        cache.retention = RetentionSwitches(track_stored=True, skip_ended_writes=True)
        made = create()
        self.assertEqual((made._explicit_ends, made._retire_ended), (True, False))
        cache.retention = RetentionSwitches(track_stored=True, retire_ended_pages=True)
        made = create()
        self.assertEqual((made._explicit_ends, made._retire_ended), (True, True))


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

    def test_an_explicit_end_skips_the_last_turns_write_and_retires_the_sessions_pages(
        self,
    ):
        """End to end on the real tree: the ended turn's inserts are flagged and the
        earlier turns' pages past the first prompt, dead replies too, are retired."""
        cache = cpu_unified_cache()
        cache.enable_storage = True
        store = mock.Mock(spec=HiCacheStorage)
        cache.cache_controller = SimpleNamespace(
            storage_backend=store, pp_prefetch_command_group=None
        )
        cache.retention = RetentionSwitches(
            track_stored=True,
            skip_ended_writes=True,
            touch_interval_s=300.0,
            retire_ended_pages=True,
        )
        ntp = NextTurnPrefetcher(
            mode="observe",
            tree_cache=cache,
            page_size=2,
            reply_key_source="hint",
            explicit_ends=True,
            retire_ended=True,
        )
        cache.next_turn_prefetch = ntp
        tree = cache.tree_core
        flagged = []
        insert = cache.insert

        def recording_insert(params):
            flagged.append(params.skip_storage_backup)
            return insert(params)

        cache.insert = recording_insert

        def turn(rid, next_rid, reply, prompt, output):
            """One finished request: returns the node its reply was committed as."""
            n = len(prompt) + len(output)
            cache.req_to_token_pool.req_to_token[0, :n] = (
                cache.token_to_kv_pool_allocator.alloc(n)
            )
            table = {
                "session": "s",
                "next_rid": next_rid,
                "gaps": [{"key": "tool", "p50": 5.0}, {"key": "_end", "ended": True}],
                "replay_reply_key": reply,
            }
            req = make_req(rid, kv_hints=hints(next_rid, version="2", payload=table))
            req.__dict__.update(
                session=None,
                origin_input_ids=array("q", prompt),
                output_ids=array("q", output),
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
            cache.cache_finished_req(req, owned_kv_len=n)
            return req.last_node

        def ack_writes(*node_ids):
            for node_id in node_ids:
                tree.mark_storage_write(node_id, op_id=node_id)
                tree.finish_storage_write(node_id, op_id=node_id, stored_at=0.0)

        first_prompt = list(range(1, 9))
        reply_0 = turn("s-t0", "s-t1", "tool", first_prompt, [100, 101])
        ack_writes(reply_0, tree.node_by_id(reply_0).parent.id)
        second_prompt = first_prompt + [9, 10, 11, 12]
        reply_1 = turn("s-t1", "s-t2", "tool", second_prompt, [102, 103])
        added_1 = tree.node_by_id(reply_1).parent.id
        ack_writes(reply_1, added_1)
        self.assertEqual(set(flagged), {False})
        store.touch_pages.assert_not_called()

        flagged.clear()
        turn("s-t2", "s-t3", "_end", second_prompt + [13, 14], [104, 105])
        self.assertEqual(set(flagged), {True})
        (retire,) = store.touch_pages.call_args_list
        self.assertEqual(retire.kwargs, {"newest_ns": None})
        own = (added_1, reply_1, reply_0)
        self.assertEqual(
            sorted(retire.args[0]),
            sorted(key for node_id in own for key in tree.get_hash_values(node_id)),
        )
        # The first prompt, which later sessions reuse, keeps its files.
        self.assertTrue(tree.node_by_id(added_1).parent.storage_stored)

    def test_the_prompt_insert_of_an_ending_turn_is_flagged_once_its_end_is_known(self):
        """The prompt is most of an ended turn and is inserted before the reply exists:
        flagged when the hint names the end at arrival, never from a reply to come."""
        cache = cpu_unified_cache()
        cache.enable_storage = True
        cache.retention = RetentionSwitches(track_stored=True, skip_ended_writes=True)
        flagged = []
        insert = cache.insert

        def recording_insert(params):
            flagged.append(params.skip_storage_backup)
            return insert(params)

        cache.insert = recording_insert

        def table(reply, gaps):
            return {
                "session": "s",
                "next_rid": "next",
                "gaps": gaps,
                "replay_reply_key": reply,
            }

        def prompt_flag(reply_key_source, payload, *, tokenizer=None):
            """The flag of one request's prompt insert, at the end of its prefill."""
            cache.next_turn_prefetch = NextTurnPrefetcher(
                mode="observe",
                tree_cache=cache,
                page_size=2,
                reply_key_source=reply_key_source,
                explicit_ends=True,
            )
            prompt = array("q", (100 * len(flagged) + i for i in range(1, 5)))
            n = len(prompt)
            cache.req_to_token_pool.req_to_token[0, :n] = (
                cache.token_to_kv_pool_allocator.alloc(n)
            )
            # Nothing matched at admission: the request holds its lock on the root.
            root = cache.root_node_handle()
            req = make_req(
                "a",
                kv_hints=hints("next", version="2", payload=payload),
                tokenizer=tokenizer,
            )
            req.__dict__.update(
                session=None,
                origin_input_ids=prompt,
                get_fill_ids=lambda: prompt,
                extra_key=None,
                cache_salt=None,
                priority=0,
                session_id=None,
                kv_rotation_base=None,
                last_node=root,
                lock_receipt=cache.inc_lock_ref(root).to_dec_params(),
                swa_prefix_lock_released=False,
                kv=SimpleNamespace(
                    req_pool_idx=0,
                    cache_protected_len=0,
                    swa_evicted_seqlen=0,
                    swa_dead_lo=lambda page_size: 0,
                ),
            )
            cache.cache_unfinished_req(req)
            return flagged[-1]

        gaps = [{"key": "tool", "p50": 5.0}, {"key": "_end", "ended": True}]
        self.assertTrue(prompt_flag("hint", table("_end", gaps)))
        self.assertFalse(prompt_flag("hint", table("tool", gaps)))
        # Read off the reply, the end is known only once the reply is: an empty reply
        # taken for one without a tool call would skip a live session's prompt.
        no_call = SimpleNamespace(decode=lambda ids: "Just an answer.")
        ends_without_a_call = table(None, [{"key": NO_CALL_KEY, "ended": True}])
        self.assertFalse(prompt_flag("reply", ends_without_a_call, tokenizer=no_call))
        # With the store detached no write comes to spend a skip: none is decided.
        cache.enable_storage = False
        self.assertFalse(prompt_flag("hint", table("_end", gaps)))


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

    def test_an_exact_hint_session_ends_at_the_turn_with_no_next(self):
        """An exact hint ends its session by naming no next turn. A request no hint
        named, a turn a table named and a hint that cannot be read state nothing."""
        ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            reply_key_source="hint",
            explicit_ends=True,
        )

        def commit(rid, envelope):
            ntp.on_turn_committed(
                req=make_req(rid, kv_hints=envelope),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=0.0,
            )

        commit("s-t0", hints("s-t1"))
        table = {
            "session": "u",
            "next_rid": "u-t1",
            "gaps": [{"key": "tool", "p50": 5.0}],
            "replay_reply_key": "tool",
        }
        commit("u-t0", hints("u-t1", version="2", payload=table))
        self.assertTrue(ntp.ends_session(make_req("s-t1"), reply_known=False))
        not_ends = {
            "never named": make_req("lone"),
            "named by a table": make_req("u-t1"),
            "names a next turn": make_req("s-t1", kv_hints=hints("s-t2")),
            "unreadable": make_req("s-t1", kv_hints=hints("s-t2", gap_s=-1.0)),
        }
        for name, req in not_ends.items():
            with self.subTest(name):
                self.assertFalse(ntp.ends_session(req, reply_known=True))
        found = self.events(lambda: commit("s-t1", None))
        self.assertEqual(
            [fields for kind, fields in found if kind == "ntp_end"],
            [{"rid": "s-t1", "sess": "s", "explicit": "1", "why": "no_next"}],
        )


class TestGapTable(PrefetcherTestCase):
    """Action v2: a table of gaps per reply key, resolved when the turn commits."""

    GAPS = [
        {"key": "execute_bash/pytest", "p10": 5.0, "p50": 80.0, "p90": 300.0, "n": 20},
        {"key": "execute_bash", "p10": 1.0, "p50": 7.0, "p90": 30.0, "n": 200},
        {"key": "*", "p10": 0.5, "p50": 6.0, "p90": 20.0, "n": 900},
        {"key": "_end", "ended": True},
    ]

    def setUp(self):
        super().setUp()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            reply_key_source="hint",
        )

    def table(self, reply_key_=None, gaps=None):
        return {
            "session": "s",
            "next_rid": "next",
            "gaps": self.GAPS if gaps is None else gaps,
            "replay_reply_key": reply_key_,
        }

    def plan_gap_ms(self, req, ntp=None):
        ntp = self.ntp if ntp is None else ntp
        found = self.events(
            lambda: ntp.on_turn_committed(
                req=req,
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=0.0,
            )
        )
        plans = [fields["gap_ms"] for kind, fields in found if kind == "ntp_plan"]
        return plans[0] if plans else None

    def test_the_reply_key_picks_its_entry_then_its_tool_then_the_default(self):
        """Mirrors the client's pick_entry: the v1 point it sends to older engines
        must equal what this engine plans. No usable entry means no plan."""
        cases = {
            "execute_bash/pytest": "80000",
            "execute_bash/ls": "7000",
            "execute_bash": "7000",
            "think": "6000",
            None: "6000",
            "_end": None,
        }
        for key, gap_ms in cases.items():
            with self.subTest(key):
                req = make_req(
                    "a",
                    kv_hints=hints("next", version="2", payload=self.table(key)),
                )
                self.assertEqual(self.plan_gap_ms(req), gap_ms)
        without_default = self.table("think", gaps=self.GAPS[:2])
        req = make_req(
            "a", kv_hints=hints("next", version="2", payload=without_default)
        )
        self.assertIsNone(self.plan_gap_ms(req))

    def test_a_table_wins_over_its_v1_companion_and_a_malformed_one_does_not(self):
        """The client sends both while older engines exist; a malformed table must
        not cost the turn its plan when a valid v1 action came with it."""

        def both(table_payload):
            table = hints("next", version="2", payload=table_payload).actions[0]
            point = hints("next", gap_s=2.0).actions[0]
            return KvHintsEnvelope(
                protocol_version="1", message_id="m", actions=[table, point]
            )

        req = make_req("a", kv_hints=both(self.table("execute_bash")))
        self.assertEqual(self.plan_gap_ms(req), "7000")
        req = make_req("a", kv_hints=both({"session": "s", "next_rid": "next"}))
        self.assertEqual(self.plan_gap_ms(req), "2000")

    def test_reply_mode_reads_the_generated_reply_not_the_replay_key(self):
        """Hint mode must not touch the reply, and reply mode must not trust the
        replay key: only one source may decide. No tokenizer means the default."""
        replying = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE
        )
        tokenizer = SimpleNamespace(
            decode=lambda ids: (
                "<function=execute_bash><parameter=command>pytest -x</parameter>"
                "</function>"
            )
        )
        envelope = hints("next", version="2", payload=self.table("execute_bash"))
        req = make_req("a", kv_hints=envelope, tokenizer=tokenizer)
        self.assertEqual(self.plan_gap_ms(req, ntp=replying), "80000")
        self.assertEqual(self.plan_gap_ms(req), "7000")
        self.assertEqual(
            self.plan_gap_ms(make_req("a", kv_hints=envelope), ntp=replying), "6000"
        )

    def test_only_an_explicit_end_ends_the_session(self):
        """Only an entry marked ended ends a session: one with no entry for its reply
        sends its next turn seconds later; a reply not generated yet picks no entry."""
        self.commit("a0", "a", now=0.0)

        def ends(ntp, payload, *, reply_known=True, tokenizer=None):
            envelope = hints("next", version="2", payload=payload)
            req = make_req("a", kv_hints=envelope, tokenizer=tokenizer)
            return ntp.ends_session(req, reply_known=reply_known)

        self.assertTrue(ends(self.ntp, self.table("_end")))
        not_ends = {
            "an entry with a gap": self.table("execute_bash"),
            "the default entry": self.table("think"),
            "no entry and no default": self.table("think", gaps=self.GAPS[:2]),
            "an entry without a gap": self.table("x", gaps=[{"key": "x"}]),
            "a table that cannot be read": {"session": "s", "next_rid": "next"},
        }
        for name, payload in not_ends.items():
            with self.subTest(name):
                self.assertFalse(ends(self.ntp, payload))

        replying = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE
        )
        no_call = SimpleNamespace(decode=lambda ids: "Just an answer.")
        over = self.table(gaps=[{"key": NO_CALL_KEY, "ended": True}])
        self.assertFalse(ends(replying, over, reply_known=False, tokenizer=no_call))
        self.assertTrue(ends(replying, over, reply_known=True, tokenizer=no_call))


class TestReplyKey(CustomTestCase):
    def test_same_rule_as_the_client_table(self):
        """The keys must equal agent_cache/scripts/gap_hints.py's, or the engine's lookups
        miss the client's entries and silently fall back to the tool's gap."""
        bash = "<function=execute_bash><parameter=command>%s</parameter></function>"
        cases = {
            bash % "pytest -x tests/": "execute_bash/pytest",
            bash % "cd /repo && FOO=1 BAR=2 git status": "execute_bash/git",
            bash % "/usr/bin/ls -la": "execute_bash/ls",
            bash % "python -m pytest x": "execute_bash/python-m-pytest",
            bash % "python3 -c 'print(1)'": "execute_bash/python-c",
            bash % "python run.py": "execute_bash/python",
            bash % "   ": "execute_bash",
            bash % "!!!": "execute_bash",
            "<function=think><parameter=thought>x</parameter></function>": "think",
            "<think><function=a></function></think>no call here": NO_CALL_KEY,
            "<think>hm</think><function=b></function>": "b",
            "Just an answer.": NO_CALL_KEY,
            '<tool_call>\n{"name": "bash", "arguments": {"command": "make -j8"}}'
            "\n</tool_call>": "bash/make",
            '<tool_call>{"name": "bash", "arguments": "{\\"cmd\\": \\"ls\\"}"}'
            "</tool_call>": "bash/ls",
            '<tool_call>{"name": "search"}</tool_call>': "search",
            "<tool_call>{not json}</tool_call>": NO_CALL_KEY,
            '<tool_call>{"name": 3}</tool_call>': NO_CALL_KEY,
        }
        for text, key in cases.items():
            with self.subTest(text):
                self.assertEqual(reply_key(text), key)


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


class DeadlineCache(ActingCache):
    """ActingCache under the deadline read order: a real read scheduler and a
    next-use registry on the eviction strategy (ORACLE_V2_PLAN.md 4.4, 4.5)."""

    def __init__(self):
        super().__init__()
        self.read_scheduler = StorageReadScheduler(
            window_pages=512, budget_tokens=10**9, page_size=PAGE
        )
        self.tree_core = SimpleNamespace(
            eviction_strategy=NextUseStrategy(horizon_s=300.0, clock=lambda: 0.0)
        )


class TestDeadlineMode(PrefetcherTestCase):
    mode = "on"

    def make_cache(self):
        return DeadlineCache()

    def setUp(self):
        super().setUp()
        self.matched = self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.arrival = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=5.0)
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)

    def tick(self, now, queue=()):
        return self.events(lambda: self.ntp.tick(waiting_queue=list(queue), now=now))

    def commit_tail(self, rid, next_rid, *, now, tail_node_id, gap_s=5.0):
        self.ntp.on_turn_committed(
            req=make_req(rid, kv_hints=None if next_rid is None else hints(next_rid, gap_s)),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=now,
            tail_node_id=tail_node_id,
        )

    def registry(self):
        return self.cache.tree_core.eviction_strategy.registry

    def test_stages_as_soon_as_the_plan_is_within_the_lead_horizon(self):
        """v1 waits for admission minus the load estimate; under the deadline
        order the lookup is issued at once and the dispatcher times the read."""
        self.commit("s-t0", "s-t1", now=0.0)
        found = self.tick(0.01)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])
        (due,) = [f for kind, f in found if kind == "ntp_due"]
        self.assertGreater(int(due["lead_ms"]), int(due["load_ms"]))
        self.assertEqual(self.ntp.read_deadline(self.handle), self.arrival)

    def test_plan_beyond_the_horizon_waits_for_it(self):
        gap = NTP_LEAD_HORIZON_S + 100.0
        self.commit("s-t0", "s-t1", now=0.0, gap_s=gap)
        arrival = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=gap)
        self.tick(arrival - NTP_LEAD_HORIZON_S - 1.0)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(arrival - NTP_LEAD_HORIZON_S + 0.01)
        self.assertEqual(len(self.cache.prefetches), 1)

    def test_parked_cap_defers_the_lookup(self):
        self.commit("s-t0", "s-t1", now=0.0)
        for i in range(NTP_MAX_PARKED):
            self.cache.read_scheduler.park(handle=i, operation=None, pages=1, now=0.0)
        self.tick(0.01)
        self.assertEqual(self.cache.prefetches, [])
        self.cache.read_scheduler.clear()
        # The plan is rechecked a probe interval later, not on the next tick.
        self.tick(0.3)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(0.6)
        self.assertEqual(len(self.cache.prefetches), 1)

    def test_just_in_time_read_waits_for_the_load_estimate(self):
        """The read must land by admission, so it starts one load estimate plus the
        margin before it, and a backlog ahead of it moves that earlier."""
        jit = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            read_timing="just_in_time",
        )
        self.ntp = jit
        gap = 50.0
        arrival = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=gap)
        missing_pages = (KEY_TOKENS - self.matched) // PAGE
        load = L3LoadModel().seconds(own_pages=missing_pages, backlog_pages=0)
        start_at = arrival - load - NTP_JIT_MARGIN_S
        self.commit("s-t0", "s-t1", now=0.0, gap_s=gap)
        self.tick(0.01)
        self.tick(start_at - 0.01)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(start_at + 0.01)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])

        self.ntp = jit = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            read_timing="just_in_time",
        )
        self.cache.prefetches.clear()
        self.cache.ongoing.clear()
        self.cache.backlog_pages = 8000
        backlog_s = L3LoadModel().backlog_s_per_page * 8000
        self.commit("s-t0", "s-t1", now=100.0, gap_s=gap)
        self.tick(100.0 + start_at - backlog_s - 0.5)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(100.0 + start_at - backlog_s + 0.5)
        self.assertEqual(len(self.cache.prefetches), 1)

    def test_just_in_time_starts_early_enough_for_the_tables_p10(self):
        """A table's gap is a distribution: the read must land by the early end of it,
        while eviction and the deadline keep using the p50."""
        self.ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            read_timing="just_in_time",
            reply_key_source="hint",
        )
        table = {
            "session": "s",
            "next_rid": "s-t1",
            "gaps": [{"key": "*", "p10": 20.0, "p50": 50.0, "p90": 90.0, "n": 9}],
        }
        self.ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1", version="2", payload=table)),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
        )
        arrival = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=50.0)
        missing_pages = (KEY_TOKENS - self.matched) // PAGE
        load = L3LoadModel().seconds(own_pages=missing_pages, backlog_pages=0)
        start_at = arrival - load - NTP_JIT_MARGIN_S - 30.0
        self.tick(start_at - 0.01)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(start_at + 0.01)
        self.assertEqual(len(self.cache.prefetches), 1)
        self.assertEqual(self.ntp.read_deadline(self.handle), arrival)

    def test_deadline_after_arrival_follows_the_queue_position(self):
        """Queued behind others with a measured rate: now plus the residual wait
        at its position; at the head: needed now."""
        self.commit("s-t0", "s-t1", now=0.0)
        self.tick(0.01)
        self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)
        req = make_req("s-t1", entered=self.arrival)
        others = [make_req(f"y{i}", entered=self.arrival) for i in range(5)]
        self.tick(self.arrival, queue=others + [req])
        rate = self.ntp._last_rate
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(
            self.ntp.read_deadline(self.handle),
            self.arrival + residual_wait(position=5, rate=rate),
        )
        self.tick(self.arrival + 0.1, queue=[others[0], req])
        self.assertIsNone(self.ntp.read_deadline(self.handle))  # position 1
        self.assertIsNone(self.ntp.read_deadline(CacheRequestHandle("unknown", 0)))

    def test_overdue_staged_plan_is_needed_now(self):
        """Past its predicted admission a staged read has no deadline left."""
        self.commit("s-t0", "s-t1", now=0.0)
        self.tick(0.01)
        self.assertEqual(self.ntp.read_deadline(self.handle), self.arrival)
        self.tick(self.arrival + 1.0)
        self.assertIsNone(self.ntp.read_deadline(self.handle))

    def test_online_read_rate_replaces_the_load_model_seeds(self):
        self.cache.cache_controller.storage_read_ms_per_page = 5.0
        self.ntp.tick(waiting_queue=[], now=0.0)
        self.assertEqual(
            (self.ntp._load.read_s_per_page, self.ntp._load.backlog_s_per_page),
            (0.005, 0.005),
        )
        self.cache.cache_controller.storage_read_ms_per_page = None
        self.ntp.tick(waiting_queue=[], now=0.1)
        self.assertEqual(self.ntp._load.read_s_per_page, 0.005)

    def test_commit_registers_the_tail_and_forgets_the_sessions_previous_one(self):
        self.commit_tail("s-t0", "s-t1", now=0.0, tail_node_id=7)
        self.assertEqual(self.registry().lookup(7), (True, self.arrival))
        self.commit_tail("s-t1", "s-t2", now=10.0, tail_node_id=9)
        self.assertEqual(self.registry().lookup(7), (False, None))
        self.assertTrue(self.registry().lookup(9)[0])
        # The session ends: its tail is registered as ended, evicted first.
        self.commit_tail("s-t2", None, now=20.0, tail_node_id=11)
        self.assertEqual(self.registry().lookup(11), (True, None))
        self.assertEqual(self.registry().lookup(9), (False, None))

    def test_expired_or_aborted_plan_ends_the_sessions_registration(self):
        """A tail left registered with a past next use would sit in the most
        protected tier for ever; the dead session must become 'ended'."""
        self.commit_tail("s-t0", "s-t1", now=0.0, tail_node_id=7)
        self.ntp.tick(waiting_queue=[], now=self.arrival + 61.0)
        self.assertEqual(self.registry().lookup(7), (True, None))
        self.assertNotIn("s", self.ntp._session_tail)
        # A queued turn aborted before admission ends it as well.
        self.commit_tail("u-t0", "u-t1", now=100.0, tail_node_id=9)
        req = make_req("u-t1", entered=100.0 + self.arrival)
        req.__dict__["kv_hints"] = hints("u-t2", payload={"session": "u", "next_rid": "u-t2", "next_gap_s": 5.0})
        self.ntp.tick(waiting_queue=[req], now=100.0 + self.arrival)
        self.assertEqual(
            self.registry().lookup(9), (True, Queued(since=100.0 + self.arrival))
        )
        self.ntp.tick(waiting_queue=[], now=101.0 + self.arrival)
        self.assertEqual(self.registry().lookup(9), (True, None))

    def test_a_turn_with_no_usable_entry_is_ranked_as_un_hinted_not_as_ended(self):
        """Once the cache acts on ends, only a hint that states the end registers one:
        no entry, no hint, an expired plan and an abort forget the tail and say why."""
        self.ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            reply_key_source="hint",
            explicit_ends=True,
        )

        def commit(rid, next_rid, reply, *, tail, now, session="s"):
            table = {
                "session": session,
                "next_rid": next_rid,
                "gaps": [{"key": "tool", "p50": 5.0}, {"key": "_end", "ended": True}],
                "replay_reply_key": reply,
            }
            req = make_req(rid, kv_hints=hints(next_rid, version="2", payload=table))
            found = self.events(
                lambda: self.ntp.on_turn_committed(
                    req=req,
                    key=key_of(KEY_TOKENS),
                    rotation_tail_declined=False,
                    now=now,
                    tail_node_id=tail,
                )
            )
            return [fields for kind, fields in found if kind == "ntp_end"]

        def ends_logged(fn):
            return [fields for kind, fields in self.events(fn) if kind == "ntp_end"]

        registered = lambda node_id: self.registry().lookup(node_id)
        self.assertEqual(commit("s-t0", "s-t1", "tool", tail=7, now=0.0), [])
        self.assertTrue(registered(7)[0])
        self.assertEqual(
            commit("s-t1", "s-t2", "other", tail=9, now=10.0),
            [{"rid": "s-t1", "sess": "s", "explicit": "0", "why": "no_entry"}],
        )
        self.assertEqual((registered(7), registered(9)), ((False, None), (False, None)))

        commit("s-t2", "s-t3", "tool", tail=11, now=20.0)
        self.assertEqual(
            commit("s-t3", "-", "_end", tail=13, now=30.0),
            [{"rid": "s-t3", "sess": "s", "explicit": "1", "why": "entry"}],
        )
        self.assertEqual(
            (registered(11), registered(13)), ((False, None), (True, None))
        )

        commit("u-t0", "u-t1", "tool", tail=15, now=100.0, session="u")
        expired = ends_logged(
            lambda: self.ntp.tick(waiting_queue=[], now=100.0 + self.arrival + 61.0)
        )
        self.assertIn(
            {"rid": "u-t1", "sess": "u", "explicit": "0", "why": "expired"}, expired
        )
        self.assertEqual(registered(15), (False, None))

        commit("v-t0", "v-t1", "tool", tail=17, now=300.0, session="v")
        queued = make_req("v-t1", entered=300.0 + self.arrival)
        self.ntp.tick(waiting_queue=[queued], now=300.0 + self.arrival)
        aborted = ends_logged(
            lambda: self.ntp.tick(waiting_queue=[], now=301.0 + self.arrival)
        )
        self.assertEqual([fields["why"] for fields in aborted], ["abort"])
        self.assertEqual(registered(17), (False, None))

        # A turn the table named that comes without its table states nothing.
        commit("w-t0", "w-t1", "tool", tail=19, now=400.0, session="w")
        unhinted = ends_logged(
            lambda: self.ntp.on_turn_committed(
                req=make_req("w-t1"),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=410.0,
                tail_node_id=21,
            )
        )
        self.assertEqual(
            unhinted, [{"rid": "w-t1", "sess": "w", "explicit": "0", "why": "no_hint"}]
        )
        self.assertEqual(
            (registered(19), registered(21)), ((False, None), (False, None))
        )

    def retiring_prefetcher(self):
        """A prefetcher with the retire on over a cache that records what it is asked to
        retire, and a commit of one table-hinted turn (session: the rid's stem)."""
        cache = DeadlineCache()
        cache.get_last_hash_value = lambda node_id: f"key-{node_id}"
        cache.retired = []
        cache.retire_storage_chain = lambda tail, *, keep_tokens, own_leaf_keys: (
            cache.retired.append((tail, keep_tokens, set(own_leaf_keys)))
        )
        ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=cache,
            page_size=PAGE,
            reply_key_source="hint",
            retire_ended=True,
        )

        def commit_req(req, *, tail, prompt_tokens, now):
            req.origin_input_ids = array("q", range(prompt_tokens))
            ntp.on_turn_committed(
                req=req,
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=now,
                tail_node_id=tail,
            )

        def commit(rid, next_rid, reply, *, tail, prompt_tokens, now=0.0):
            table = {
                "session": rid.split("-")[0],
                "next_rid": next_rid,
                "gaps": [{"key": "tool", "p50": 5.0}, {"key": "_end", "ended": True}],
                "replay_reply_key": reply,
            }
            req = make_req(rid, kv_hints=hints(next_rid, version="2", payload=table))
            commit_req(req, tail=tail, prompt_tokens=prompt_tokens, now=now)

        return cache, ntp, commit, commit_req

    def test_only_an_explicit_end_retires_and_only_past_the_sessions_first_prompt(self):
        """The cache is told where the session's first prompt ends, which later sessions
        reuse, and the last page key of each turn it committed, usable entry or not."""
        cache, ntp, commit, commit_req = self.retiring_prefetcher()
        commit("s-t0", "s-t1", "tool", tail=7, prompt_tokens=3 * PAGE + 10)
        commit("s-t1", "s-t2", "other", tail=9, prompt_tokens=5 * PAGE)
        commit("s-t2", "s-t3", "tool", tail=11, prompt_tokens=7 * PAGE)
        # A session whose plan expires is live: it keeps its pages.
        commit("u-t0", "u-t1", "tool", tail=31, prompt_tokens=2 * PAGE)
        ntp.tick(waiting_queue=[], now=self.arrival + 61.0)
        self.assertEqual(cache.retired, [])
        commit("s-t3", "-", "_end", tail=13, prompt_tokens=9 * PAGE, now=100.0)
        self.assertEqual(cache.retired, [(13, 3 * PAGE, {"key-7", "key-9", "key-11"})])
        # A session seen for one turn only has no page known to be its own.
        commit("one-t0", "-", "_end", tail=21, prompt_tokens=4 * PAGE, now=100.0)
        self.assertEqual(len(cache.retired), 1)
        # An exact hint states the end by naming no next turn.
        exact = {"session": "x", "next_rid": "x-t1", "next_gap_s": 5.0}
        named = make_req("x-t0", kv_hints=hints("x-t1", payload=exact))
        commit_req(named, tail=41, prompt_tokens=2 * PAGE, now=200.0)
        commit_req(make_req("x-t1"), tail=43, prompt_tokens=6 * PAGE, now=210.0)
        self.assertEqual(cache.retired[1:], [(43, 2 * PAGE, {"key-41"})])

    def test_what_is_kept_per_session_for_its_end_is_bounded_oldest_first(self):
        """A session that never states its end would keep its record for good: past the
        registry's bound the oldest record goes, and that session's end retires none."""
        cache, _, commit, _ = self.retiring_prefetcher()
        with mock.patch(
            "sglang.srt.mem_cache.next_turn_prefetch.REGISTRY_MAX_ENTRIES", 2
        ):
            for index, session in enumerate(("a", "b", "c")):
                commit(
                    f"{session}-t0",
                    f"{session}-t1",
                    "tool",
                    tail=50 + index,
                    prompt_tokens=PAGE,
                )
        commit("a-t1", "-", "_end", tail=60, prompt_tokens=2 * PAGE, now=10.0)
        self.assertEqual(cache.retired, [])
        commit("c-t1", "-", "_end", tail=61, prompt_tokens=2 * PAGE, now=10.0)
        self.assertEqual(cache.retired, [(61, PAGE, {"key-52"})])

    def test_turns_first_seen_in_one_tick_are_queued_at_their_own_arrivals(self):
        """queued_order farthest evicts the latest arrival's history first; turns that
        arrive during one long forward are first seen in the same tick and must keep
        their arrival order instead of tying on the tick's time."""
        self.commit_tail("s-t0", "s-t1", now=0.0, tail_node_id=7)
        self.ntp.on_turn_committed(
            req=make_req(
                "u-t0",
                kv_hints=hints(
                    "u-t1",
                    payload={"session": "u", "next_rid": "u-t1", "next_gap_s": 5.0},
                ),
            ),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
            tail_node_id=9,
        )
        self.tick(0.01)
        first = make_req("s-t1", entered=self.arrival + 1.0)
        second = make_req("u-t1", entered=self.arrival + 2.0)
        self.tick(self.arrival + 3.0, queue=[first, second])
        self.assertEqual(
            (self.registry().lookup(7), self.registry().lookup(9)),
            (
                (True, Queued(since=self.arrival + 1.0)),
                (True, Queued(since=self.arrival + 2.0)),
            ),
        )

    def test_arrival_registers_the_tail_as_queued_not_as_a_time(self):
        """Under next_use a plain time is a predicted next use, overdue once the turn
        has waited past the grace: the arrival needs its own marker."""
        self.commit_tail("s-t0", "s-t1", now=0.0, tail_node_id=7)
        self.tick(0.01)
        req = make_req("s-t1", entered=self.arrival + 1.0)
        self.tick(self.arrival + 1.0, queue=[req])
        self.assertEqual(
            self.registry().lookup(7), (True, Queued(since=self.arrival + 1.0))
        )


class PricedCache(DeadlineCache):
    """DeadlineCache plus the pin and room state that expected-cost timing reads."""

    soft_pin_victims_fn = None
    room_refusals = None
    host_room_floor_tokens = 44_000

    def __init__(self):
        super().__init__()
        self.pinned = {}
        self.room = (44_000, 0)
        self.refusals = []
        # Tokens of a key in device memory; None = all the resident ones.
        self.in_device = None

    def resident_prefix_len(self, key):
        self.probes += 1
        resident = min(self.resident, len(key))
        return (resident if self.in_device is None else self.in_device), resident

    def pinned_prefetch_tokens_of(self, handle):
        return self.pinned.get(handle, 0)

    def host_room_tokens(self):
        return self.room

    def take_room_refusals(self):
        refusals, self.refusals = self.refusals, []
        return refusals

    def land(self, handle, tokens):
        self.ongoing.discard(handle)
        self.pinned[handle] = tokens

    def finish(self, handle, outcome):
        super().finish(handle, outcome)
        self.pinned.pop(handle, None)


class MeasuredCache(PricedCache):
    """PricedCache plus what the measured prices read from the cache, and the hooks the
    switched changes bind. Like the real cache, it asks the gate before a lookup."""

    prefetch_landed_fn = None
    prefetch_gate_prompt_fn = None
    prefetch_lookup_fn = None
    pending_backup_tokens = 0

    def __init__(self):
        super().__init__()
        self.tree_core.host_eviction_observer = None
        # The host leaf eviction would take next.
        self.victim = None
        # handle -> (first token, tokens) of its storage hit, and the parked reads dropped.
        self.spans = {}
        self.dropped = []

    def next_host_victim(self):
        return self.victim

    def park(self, handle, *, start, tokens):
        """A storage hit of `tokens` from `start` on, parked for dispatch."""
        self.ongoing.add(handle)
        self.spans[handle] = (start, tokens)
        self.read_scheduler.park(
            handle=handle, operation=None, pages=tokens // PAGE, now=0.0
        )

    def parked_read_span(self, handle):
        return self.spans[handle] if self.read_scheduler.is_parked(handle) else None

    def parked_read_pages_ahead(self, handle):
        return self.backlog_pages + self.read_scheduler.pages_ahead_of(
            handle, deadline_of=self.read_deadline_fn
        )

    def drop_parked_read(self, handle):
        if self.parked_read_span(handle) is None:
            return False
        self.read_scheduler.discard(handle)
        self.ongoing.discard(handle)
        self.dropped.append(handle)
        return True

    def storage_read_line_pages(self, deadline):
        return self.backlog_pages + self.read_scheduler.pages_due_by(
            deadline, deadline_of=self.read_deadline_fn
        )

    def prefetch_from_storage(self, handle, anchor, tokens, last_hash, keys, **kw):
        matched = len(kw["matched_prefix_tokens"])
        if self.prefetch_gate_prompt_fn is not None:
            allowed = self.prefetch_gate_prompt_fn(handle, matched, tokens)
        else:
            allowed = self.prefetch_gate_fn is None or self.prefetch_gate_fn(
                handle, matched
            )
        if allowed:
            super().prefetch_from_storage(handle, anchor, tokens, last_hash, keys, **kw)


def hold_switches(test, *overrides):
    """Enter (switch, value) overrides. The returned stack's close() restores them; it
    also runs as a cleanup, since override() skips its restore when the body raises."""
    stack = contextlib.ExitStack()
    test.addCleanup(stack.close)
    for switch, value in overrides:
        stack.enter_context(switch.override(value))
    return stack


# Every env switch of the priced decisions but the prices log, and what two of them need.
DECISION_SWITCHES = (
    envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES,
    envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS,
    envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT,
    envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE,
    envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE,
    envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE,
    envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,
    envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES,
)
BIG_KEY_TOKENS = 1000 * PAGE
SHARP = {"key": "*", "p10": 45.0, "p50": 50.0, "p90": 56.0, "n": 40}
WIDE = {"key": "*", "p10": 0.85, "p50": 22.0, "p90": 58.0, "n": 500}


class TestExpectedCost(PrefetcherTestCase):
    """Read timing expected_cost: the start rule, the release of a landed read, soft
    pins and the price (agent_cache/PAPER_PLAN.md, "Formulation")."""

    mode = "on"

    def make_cache(self):
        return PricedCache()

    def setUp(self):
        super().setUp()
        self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.missing = BIG_KEY_TOKENS - 4 * PAGE
        self.read_s = (
            L3LoadModel().seconds(own_pages=self.missing // PAGE, backlog_pages=0)
            + NTP_JIT_MARGIN_S
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.ntp = self.make_ntp()

    def make_ntp(self, **kwargs):
        return NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            read_timing="expected_cost",
            reply_key_source="hint",
            **kwargs,
        )

    def commit_entry(self, entry, *, session="s", now=0.0, tail_node_id=None):
        table = {"session": session, "next_rid": f"{session}-t1", "gaps": [entry]}
        self.ntp.on_turn_committed(
            req=make_req(
                f"{session}-t0", kv_hints=hints("-", version="2", payload=table)
            ),
            key=key_of(BIG_KEY_TOKENS),
            rotation_tail_declined=False,
            now=now,
            tail_node_id=tail_node_id,
        )

    def tick(self, now, queue=()):
        return self.events(lambda: self.ntp.tick(waiting_queue=list(queue), now=now))

    def rule_start(self, entry, *, price):
        """Rule 1's start for the entry, seconds after a commit at 0."""
        dist = GapDistribution.from_quantiles(
            p10=entry["p10"], p50=entry["p50"], p90=entry["p90"]
        )
        return read_start(
            remaining=remaining_gaps(
                support=dist.support(), idle_s=-ArrivalOffset().seconds
            ),
            read_s=self.read_s,
            wait_s=0.0,
            rent_per_s=price * self.missing,
        )

    def test_a_read_starts_where_rule_1_puts_it_from_the_engines_own_estimates(self):
        """The read time is the load model's plus the margin, the idle time is net of
        the arrival offset and the rent is the price times the missing tokens; a
        lower price starts the same read earlier."""
        starts = {}
        for price in (5e-5, 5e-6):
            with self.subTest(price=price):
                self.cache.prefetches.clear()
                self.cache.ongoing.clear()
                self.ntp = self.make_ntp(memory_price=price)
                starts[price] = start = self.rule_start(SHARP, price=price)
                self.commit_entry(SHARP)
                self.tick(start - 0.3)
                self.assertEqual(self.cache.prefetches, [])
                found = self.tick(start + 0.01)
                self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])
                (due,) = [f for kind, f in found if kind == "ntp_due"]
                self.assertEqual(
                    int(due["load_ms"]), round((self.read_s - NTP_JIT_MARGIN_S) * 1000)
                )
        self.assertLess(starts[5e-6], starts[5e-5])

    def test_an_exact_hint_starts_at_the_just_in_time_moment(self):
        self.ntp = self.make_ntp()
        self.ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1", 50.0)),
            key=key_of(BIG_KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
        )
        start = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=50.0) - self.read_s
        self.tick(start - 0.3)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(start + 0.01)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])

    def test_a_wide_entry_is_left_to_the_turns_arrival(self):
        """A read staged for pip's entry would sit pinned through its long mode, so
        none is; the plan still matches the turn when it comes."""
        self.commit_entry(WIDE)
        for step in range(400):
            self.tick(0.5 * step)
        self.assertEqual(self.cache.prefetches, [])
        late = make_req("s-t1", entered=200.0)
        found = self.tick(200.5, queue=[late])
        self.assertIn("ntp_arrive", [kind for kind, _ in found])

    def test_arrival_timing_never_stages_and_still_follows_the_turn(self):
        """The rival that uses no gap forecast: even an exact hint stages nothing, and
        the plan still matches its turn for the gate, the hold and its protection."""
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, read_timing="arrival"
        )
        self.ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1", 20.0)),
            key=key_of(BIG_KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
        )
        for step in range(100):
            self.tick(0.5 * step)
        self.assertEqual((self.cache.prefetches, self.cache.probes), ([], 0))
        found = self.tick(60.5, queue=[make_req("s-t1", entered=60.0)])
        self.assertIn("ntp_arrive", [kind for kind, _ in found])

    def test_a_landed_read_gives_its_pin_back_once_its_entry_has_passed(self):
        """No forecast ends the plan: only the pin goes, and a turn that comes after
        all is still the plan's."""
        self.commit_entry(SHARP, tail_node_id=5)
        self.tick(self.rule_start(SHARP, price=5e-5) + 0.01)
        self.cache.land(self.handle, self.missing)
        self.tick(54.0)
        self.assertEqual(self.cache.finished, [])
        found = self.tick(90.0)
        self.assertEqual(self.cache.finished, [(self.handle, CacheRequestOutcome.ABORT)])
        (release,) = [f for kind, f in found if kind == "ec_release"]
        self.assertEqual(int(release["tokens"]), self.missing)
        found = self.tick(50.0 + MIN_EXPIRY_S + 60.0)
        self.assertEqual(
            [kind for kind, _ in found if kind in ("ntp_expire", "nue_register")], []
        )
        late = make_req("s-t1", entered=300.0)
        found = self.tick(300.5, queue=[late])
        self.assertIn("ntp_arrive", [kind for kind, _ in found])

    def test_a_read_refused_room_is_left_to_arrival_and_a_bumped_one_may_stage_again(self):
        """A refused staging tried again half a second later meets the same full room
        and spends the lookups the turn needs after it arrives. Only a read that was
        bumped stages again, and each staging takes one of the gate's lookups, so the
        cap leaves one for the turn's own read."""
        self.assertLessEqual(NTP_MAX_STAGINGS, MAX_REFETCHES)
        self.ntp = self.make_ntp(soft_pins="on")
        start = self.rule_start(SHARP, price=5e-5)
        self.commit_entry(SHARP)
        self.tick(start + 0.01)
        self.cache.ongoing.clear()
        for probe in range(1, 4):
            self.tick(start + 0.6 * probe)
        self.assertEqual(len(self.cache.prefetches), 1)

        self.ntp = self.make_ntp(soft_pins="on")
        self.cache.prefetches.clear()
        self.commit_entry(SHARP, now=1000.0)
        asking = make_req("c-t1", entered=1000.0)
        for round_ in range(1, 5):
            at = 1000.0 + start + 0.6 * round_
            self.tick(at)
            if self.handle in self.cache.ongoing:
                self.cache.land(self.handle, self.missing)
                # The asking turn is queued by the time its read needs the room.
                self.tick(at + 0.01, queue=[asking])
                for victim in self.ntp.soft_pin_victims(asking.cache_request_handle, 1):
                    self.cache.finish(victim, CacheRequestOutcome.ABORT)
        self.assertEqual(len(self.cache.prefetches), NTP_MAX_STAGINGS)

    def test_rent_is_charged_on_what_the_pin_keeps_in_the_host_room(self):
        """A pinned tail keeps the session's whole chain out of host eviction, so the
        rent is on every token not in device memory, not on the span being read. The
        same 4K-token read is staged when the rest of the chain is on the device and
        left to arrival when 60K tokens of it are host-only."""
        staged = {}
        for in_device in (4 * PAGE, None):
            self.cache.prefetches.clear()
            self.cache.ongoing.clear()
            self.cache.resident = BIG_KEY_TOKENS - 64 * PAGE
            self.cache.in_device = in_device
            self.ntp = self.make_ntp()
            self.commit_entry(SHARP)
            for step in range(140):
                self.tick(0.5 * step)
            staged[in_device] = len(self.cache.prefetches)
        self.assertEqual(staged, {4 * PAGE: 0, None: 1})

    def test_a_turn_waiting_on_its_own_read_does_not_lengthen_the_wait_estimate(self):
        """Admission passes over a queued turn whose read is still under way, so a
        turn arriving now does not wait behind it; counting it made rule 1 leave reads
        to arrival whenever a few such turns were queued."""
        start = self.rule_start(SHARP, price=5e-5)
        self.commit_entry(SHARP)
        blocked = [make_req(f"b{i}-t1", entered=0.0) for i in range(3)]
        for req in blocked:
            self.cache.ongoing.add(req.cache_request_handle)
        admitted = make_req("warm-t1", entered=0.0)
        self.tick(0.0, queue=[admitted])
        admitted.time_stats.forward_entry_time = 0.2
        for step in range(1, 8):
            more = make_req(f"warm{step}-t1", entered=0.2 * step)
            self.tick(0.2 * step, queue=[more] + blocked)
            more.time_stats.forward_entry_time = 0.2 * step + 0.1
        self.tick(start - 0.3, queue=blocked)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(start + 0.01, queue=blocked)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])

    def test_a_queue_of_turns_with_nothing_to_read_does_lengthen_it(self):
        """When turns that need no read stay queued the GPU is the bottleneck, and the
        whole queue is ahead of a turn arriving now: its wait then covers the read, so
        nothing is staged where it would be with an idle GPU."""
        start = self.rule_start(SHARP, price=5e-5)
        self.commit_entry(SHARP)
        admitted = make_req("warm-t1", entered=0.0)
        self.tick(0.0, queue=[admitted])
        admitted.time_stats.forward_entry_time = 0.2
        for step in range(1, 8):
            more = make_req(f"warm{step}-t1", entered=0.2 * step)
            self.tick(0.2 * step, queue=[more])
            more.time_stats.forward_entry_time = 0.2 * step + 0.1
        stuck = [make_req(f"q{i}-t1", entered=start - 5.0) for i in range(12)]
        self.tick(start - 5.0, queue=stuck)
        for step in range(40):
            self.tick(start - 4.5 + 0.25 * step, queue=stuck)
        self.assertEqual(self.cache.prefetches, [])

    def test_a_read_still_parked_when_its_entry_passes_is_given_up(self):
        """Dispatched after that, it would take room, land, and be released at the
        next probe: a read for nothing."""
        self.commit_entry(SHARP)
        self.tick(self.rule_start(SHARP, price=5e-5) + 0.01)
        self.cache.read_scheduler.park(
            handle=self.handle, operation=None, pages=1, now=60.0
        )
        self.tick(62.0)
        self.assertEqual(self.cache.finished, [])
        self.tick(90.0)
        self.assertEqual(self.cache.finished, [(self.handle, CacheRequestOutcome.ABORT)])

    def test_a_staging_that_issues_no_lookup_does_not_end_staging(self):
        """An anchor not yet backed up cannot be read under; the attempt counts
        against the cap, but the plan may stage once the anchor is."""
        start = self.rule_start(SHARP, price=5e-5)
        self.commit_entry(SHARP)
        self.cache.backed_up = False
        self.tick(start + 0.01)
        self.assertEqual(self.cache.prefetches, [])
        self.cache.backed_up = True
        for step in range(1, 8):
            self.tick(start + 0.3 * step)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])

    def test_an_overdue_staged_read_stays_a_timed_read(self):
        """Under the p50 routines an overdue plan's read is "needed now" and skips the
        window and the budget; a read started on purpose for a turn not here keeps
        both."""
        self.commit_entry(SHARP)
        self.tick(100.0)
        self.assertEqual(self.ntp.read_deadline(self.handle), 100.0 + NTP_JIT_MARGIN_S)

    def test_soft_pins_bump_the_staged_read_least_likely_back_for_a_queued_turn(self):
        self.ntp = self.make_ntp(memory_price=1e-6, soft_pins="on")
        self.assertEqual(self.cache.soft_pin_victims_fn, self.ntp.soft_pin_victims)
        self.commit_entry(SHARP, session="a")
        self.commit_entry({**SHARP, "p10": 40.0, "p50": 52.0, "p90": 70.0}, session="b")
        staged_a = CacheRequestHandle(rid="a-t1", attempt_id=0)
        staged_b = CacheRequestHandle(rid="b-t1", attempt_id=0)
        self.cache.pinned = {staged_a: 30_000, staged_b: 20_000}
        asking = make_req("c-t1", entered=49.0)
        self.assertEqual(
            self.ntp.soft_pin_victims(asking.cache_request_handle, 10_000), []
        )
        self.tick(49.0, queue=[asking])
        found = self.events(
            lambda: self.assertEqual(
                self.ntp.soft_pin_victims(asking.cache_request_handle, 10_000),
                [staged_b],
            )
        )
        self.assertEqual([f["for_rid"] for kind, f in found if kind == "ec_bump"], ["c-t1"])
        self.assertEqual(
            self.ntp.soft_pin_victims(asking.cache_request_handle, 40_000),
            [staged_b, staged_a],
        )

    def test_the_price_follows_refused_room_only_when_it_is_learned(self):
        for learning, moved in (("on", True), ("off", False)):
            with self.subTest(learning=learning):
                self.ntp = self.make_ntp(price_learning=learning)
                self.commit_entry(WIDE)
                self.cache.room = (44_000, 44_000)
                self.cache.refusals = [(CacheRequestHandle("x", 0), 30_000)]
                self.tick(0.0)
                found = self.tick(5.01)
                (price,) = [f for kind, f in found if kind == "ec_price"]
                self.assertEqual(int(price["refused"]), 30_000)
                self.assertEqual(float(price["price"]) > 5e-5, moved)

    def test_harm_is_charged_only_for_what_a_staged_pin_held_from_an_arrived_turn(self):
        """A read refused room costs its turn a recomputation only where a staged
        read's pin was holding that room, and each turn in the queue waits behind
        it; a refusal with nothing staged, or of a turn not here, charges nothing."""
        self.ntp = self.make_ntp(price_learning="harm")
        self.commit_entry(WIDE)
        waiting = [make_req("a-t1", entered=0.0), make_req("b-t1", entered=0.0)]
        asking = waiting[0].cache_request_handle

        def harm_ms(refusals, pinned):
            self.cache.pinned = pinned
            start = self.ntp._last_now if self.ntp._last_now > 0 else 0.0
            self.tick(start + 0.1, queue=waiting)
            self.cache.refusals = refusals
            found = self.tick(start + 5.2, queue=waiting)
            (price,) = [f for kind, f in found if kind == "ec_price"]
            return int(price["harm_ms"])

        staged = {self.handle: 20_000}
        self.assertEqual(harm_ms([(asking, 30_000)], {}), 0)
        self.assertEqual(harm_ms([(CacheRequestHandle("z-t9", 0), 30_000)], staged), 0)
        # 20,000 of the 30,000 refused tokens were under the staged pin, and the two
        # queued turns have been waiting for the GPU.
        self.assertEqual(harm_ms([(asking, 30_000)], staged), round(20_000 * 1.1e-4 * 2 * 1000))
        # Two refusals in one pass cannot be charged for more than the pins held.
        twice = [(asking, 15_000), (waiting[1].cache_request_handle, 15_000)]
        self.assertEqual(harm_ms(twice, staged), round(20_000 * 1.1e-4 * 2 * 1000))


class TestReturnChanceRegistration(PrefetcherTestCase):
    """Under return_chance eviction a commit registers the entry's distribution, and
    a plan that expires does not end its session."""

    mode = "on"

    def make_cache(self):
        cache = DeadlineCache()
        cache.tree_core = SimpleNamespace(
            eviction_strategy=ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 0.0)
        )
        return cache

    def test_the_distribution_outlives_the_plan_and_only_an_end_signal_ends_it(self):
        self.ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            read_timing="just_in_time",
            reply_key_source="hint",
        )
        registry = self.cache.tree_core.eviction_strategy.registry
        table = {"session": "s", "next_rid": "s-t1", "gaps": [SHARP, {"key": "_end", "ended": True}]}
        self.ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("-", version="2", payload=table)),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
            tail_node_id=5,
        )
        known, forecast = registry.lookup(5)
        self.assertIsInstance(forecast, ReturnForecast)
        self.assertAlmostEqual(forecast.dist.quantile(0.9), 56.0)
        # A span read back for the plan's turn joins the session at landing.
        strategy = self.cache.tree_core.eviction_strategy
        self.assertEqual(self.cache.prefetch_landed_fn, self.ntp.on_read_landed)
        self.ntp.on_read_landed(CacheRequestHandle(rid="s-t1", attempt_id=0), 77)
        span = SimpleNamespace(id=77, last_access_time=0.0, parent=None, children={})
        self.assertEqual(strategy.get_priority(span)[1:], strategy.get_priority(
            SimpleNamespace(id=5, last_access_time=0.0, parent=None, children={})
        )[1:])
        found = self.events(
            lambda: self.ntp.tick(waiting_queue=[], now=50.0 + MIN_EXPIRY_S + 60.0)
        )
        self.assertIn("ntp_expire", [kind for kind, _ in found])
        self.assertIs(registry.lookup(5)[1], forecast)
        table["replay_reply_key"] = "_end"
        self.ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("-", version="2", payload=table)),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=200.0,
            tail_node_id=5,
        )
        self.assertEqual(registry.lookup(5), (True, None))


    def test_samples_score_hands_eviction_the_gaps_seen_for_the_entrys_key(self):
        """The engine collects a key's gaps from its own arrivals, net of the delivery
        offset, and a later commit with that key carries them to the strategy."""
        self.cache.tree_core = SimpleNamespace(
            eviction_strategy=ReturnChanceStrategy(score="samples", clock=lambda: 0.0)
        )
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, reply_key_source="hint"
        )
        registry = self.cache.tree_core.eviction_strategy.registry
        table = {"session": "s", "next_rid": "s-t1", "gaps": [{**SHARP, "key": "tool/x"}],
                 "replay_reply_key": "tool/x"}

        def commit(now, next_rid):
            self.ntp.on_turn_committed(
                req=make_req("s-t0", kv_hints=hints("-", version="2", payload={**table, "next_rid": next_rid})),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=now,
                tail_node_id=5,
            )

        commit(0.0, "s-t1")
        self.assertEqual(registry.lookup(5)[1].key_gaps, ())
        offset = ArrivalOffset().seconds
        self.ntp.tick(waiting_queue=[make_req("s-t1", entered=40.0 + offset)], now=40.2)
        commit(50.0, "s-t2")
        forecast = registry.lookup(5)[1]
        self.assertEqual(len(forecast.key_gaps), 1)
        self.assertAlmostEqual(forecast.key_gaps[0], 40.0)
        self.assertEqual(forecast.all_gaps, forecast.key_gaps)


class TestOwnerSwitches(CustomTestCase):
    def test_gate_and_hold_bind_only_when_on_and_only_in_mode_on(self):
        """Off, the baseline and oracle arms keep the arrival fetch; observe must never
        change what the engine reads, whatever the switches say."""
        for mode, switch, bound in (
            ("on", "off", False),
            ("observe", "on", False),
            ("on", "on", True),
        ):
            with self.subTest(mode=mode, switch=switch):
                cache = DeadlineCache()
                NextTurnPrefetcher(
                    mode=mode,
                    tree_cache=cache,
                    page_size=PAGE,
                    arrival_gate=switch,
                    arrival_hold=switch,
                )
                self.assertEqual(cache.prefetch_gate_fn is not None, bound)
                self.assertEqual(cache.read_not_before_fn is not None, bound)
        # The hold times parked reads, which only the deadline order has.
        fifo = ActingCache()
        NextTurnPrefetcher(mode="on", tree_cache=fifo, page_size=PAGE, arrival_hold="on")
        self.assertIsNone(fifo.read_not_before_fn)

    def test_room_retry_binds_only_when_on_and_only_in_mode_on(self):
        """Off, the cache reports refused reads to no one and they spend the turn's reads as
        before; observe must never change what the engine reads."""
        for mode, switch, bound in (
            ("on", False, False),
            ("observe", True, False),
            ("on", True, True),
        ):
            with self.subTest(mode=mode, switch=switch):
                cache = DeadlineCache()
                with envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(switch):
                    NextTurnPrefetcher(mode=mode, tree_cache=cache, page_size=PAGE)
                self.assertEqual(cache.prefetch_refused_fn is not None, bound)

    def test_prices_are_measured_only_for_the_log_or_the_switch_and_decide_only_in_mode_on(
        self,
    ):
        """Off, host eviction reports to no one and the landing hook stays as the eviction
        policy left it. Observe may log the prices but takes no decision from them. A
        learned price would be logged and never used, so the switch refuses it."""
        for mode, interval, switch, measuring, deciding in (
            ("on", 0.0, False, False, False),
            ("on", 5.0, False, True, False),
            ("on", 0.0, True, True, True),
            ("observe", 0.0, True, False, False),
            ("observe", 5.0, True, True, False),
        ):
            stack = hold_switches(
                self,
                (envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, interval),
                (envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, switch),
            )
            with self.subTest(mode=mode, interval=interval, switch=switch):
                cache = MeasuredCache()
                ntp = NextTurnPrefetcher(mode=mode, tree_cache=cache, page_size=PAGE)
                self.assertEqual(
                    cache.tree_core.host_eviction_observer is not None, measuring
                )
                self.assertEqual(cache.prefetch_landed_fn is not None, measuring)
                self.assertEqual(ntp._measured_prices, deciding)
            stack.close()
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, True))
        for learning in ("on", "harm"):
            with self.subTest(learning=learning), self.assertRaises(ValueError):
                NextTurnPrefetcher(
                    mode="on",
                    tree_cache=MeasuredCache(),
                    page_size=PAGE,
                    read_timing="expected_cost",
                    price_learning=learning,
                )

    def test_in_mode_observe_only_the_prices_log_works(self):
        """Observe logs what the prefetcher would decide without the switches: none of
        them may weigh a logged decision there, bind a hook or ask for what it needs."""
        for interval in (0.0, 5.0):
            with self.subTest(interval=interval):
                stack = hold_switches(
                    self,
                    (envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, interval),
                    *((switch, True) for switch in DECISION_SWITCHES),
                )
                cache = MeasuredCache()
                ntp = NextTurnPrefetcher(
                    mode="observe",
                    tree_cache=cache,
                    page_size=PAGE,
                    read_timing="expected_cost",
                    arrival_gate="on",
                    arrival_hold="on",
                    price_learning="harm",
                )
                stack.close()
                self.assertEqual(ntp._meter is not None, interval > 0)
                self.assertFalse(
                    ntp._measured_prices
                    or ntp._free_empty_lookups
                    or ntp._lateness_weight
                    or ntp._read_line
                    or ntp._hold_quantile
                    or ntp._recompute_by_price
                    or ntp._recompute_when_starved
                )
                self.assertIsNone(cache.prefetch_gate_prompt_fn)
                self.assertIsNone(cache.prefetch_lookup_fn)

    def test_measured_prices_need_an_eviction_policy_that_prices_its_holders(self):
        """Under an eviction policy with no forecast the room's price is 0 for ever: the switch
        is refused where a staged or held read would then pay no rent, and only there."""
        hold_switches(
            self,
            (envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, True),
            (envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE, True),
        )

        def unpriced():
            cache = MeasuredCache()
            cache.tree_core.eviction_strategy = None
            return cache

        for setup in ({"read_timing": "expected_cost"}, {"arrival_hold": "on"}):
            with self.subTest(setup), self.assertRaises(ValueError):
                NextTurnPrefetcher(
                    mode="on", tree_cache=unpriced(), page_size=PAGE, **setup
                )
        ntp = NextTurnPrefetcher(mode="on", tree_cache=unpriced(), page_size=PAGE)
        self.assertTrue(ntp._measured_prices)


class TestArrivalGate(PrefetcherTestCase):
    mode = "on"

    def make_cache(self):
        return DeadlineCache()

    def setUp(self):
        super().setUp()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_gate="on"
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.threshold = self.cache.prefetch_threshold
        self.commit("s-t0", "s-t1", now=0.0)

    def test_no_lookup_once_the_match_covers_the_key_to_within_the_threshold(self):
        """What is left past the committed key is new; a missing span below the threshold
        is not worth a read, the same rule prefetch_from_storage applies to any lookup."""
        for matched, allowed in (
            (KEY_TOKENS + 500, False),
            (KEY_TOKENS, False),
            (KEY_TOKENS - self.threshold + 1, False),
            (KEY_TOKENS - self.threshold, True),
            (0, True),
        ):
            with self.subTest(matched=matched):
                self.assertEqual(self.ntp.prefetch_gate(self.handle, matched), allowed)

    def test_a_turn_with_no_plan_still_looks_up(self):
        """Only a lookup can find a first turn's shared prefix or a late turn's history.
        The match length says nothing without a plan's key to compare it with."""
        for rid in ("first-turn", "plan-expired"):
            self.assertTrue(
                self.ntp.prefetch_gate(CacheRequestHandle(rid, 0), KEY_TOKENS)
            )

    def test_a_turns_reads_are_capped_and_the_queued_refetch_stops_asking(self):
        """Covered calls spend nothing. Past the cap the turn recomputes, and the plan
        must stop asking, or it would be re-issued every probe interval until admission."""
        decisions = self.events(
            lambda: [
                self.ntp.prefetch_gate(self.handle, matched)
                for matched in (0, KEY_TOKENS, 0, 0, 0)
            ]
        )
        self.assertEqual(
            [f["decision"] for kind, f in decisions if kind == "ntp_gate"],
            ["read", "covered", "reread", "reread", "capped"],
        )
        self.assertEqual(MAX_REFETCHES, 2)
        req = make_req("s-t1", entered=5.0, last_match=0)
        self.assertEqual(self.ntp.tick(waiting_queue=[req], now=5.0), [])


class TestGateOnTheRealCache(CustomTestCase):
    def test_a_declined_lookup_survives_the_real_cache(self):
        """A declined lookup killed the scheduler at the first planned arrival: the real
        counters are a plain dict with fixed keys, which fixtures with a defaultdict hide."""
        cache = cpu_unified_cache()
        cache.enable_storage = True
        cache.prefetch_threshold = 2
        cache.cache_controller = SimpleNamespace(
            get_prefetch_submission=lambda rid: None,
            prefetch_rate_limited=lambda: False,
        )
        ntp = NextTurnPrefetcher(
            mode="on", tree_cache=cache, page_size=2, arrival_gate="on"
        )
        ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1")),
            key=key_of(8),
            rotation_tail_declined=False,
            now=0.0,
        )
        handle = CacheRequestHandle("s-t1", 0)
        cache.prefetch_from_storage(
            handle,
            cache.root_node_handle(),
            array("q", range(8, 16)),
            matched_prefix_tokens=list(range(8)),
        )
        self.assertEqual(cache._prefetch_outcome_stats["declined_gate"], 1)
        self.assertFalse(cache.has_ongoing_prefetch(handle))

    def test_the_real_cache_hands_the_prompt_to_the_gate_that_reads_it(self):
        """Of an 8-token key the prompt repeats 7, all in memory: judged by the key alone one
        token is missing, and the turn looks storage up for a reply no tier holds."""
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, True))
        cache = cpu_unified_cache()
        cache.enable_storage = True
        cache.prefetch_threshold = 1
        cache.cache_controller = SimpleNamespace(
            get_prefetch_submission=lambda rid: None,
            prefetch_rate_limited=lambda: False,
        )
        ntp = NextTurnPrefetcher(
            mode="on", tree_cache=cache, page_size=2, arrival_gate="on"
        )
        ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1")),
            key=key_of(8),
            rotation_tail_declined=False,
            now=0.0,
        )
        handle = CacheRequestHandle("s-t1", 0)
        cache.prefetch_from_storage(
            handle,
            cache.root_node_handle(),
            array("q", [107, 108, 109, 110]),
            matched_prefix_tokens=list(range(7)),
        )
        self.assertEqual(cache._prefetch_outcome_stats["declined_gate"], 1)
        self.assertEqual(ntp._plans["s-t1"].shared_tokens, 7)
        self.assertFalse(cache.has_ongoing_prefetch(handle))

    def test_the_prices_are_measured_on_the_real_cache_in_both_modes(self):
        """What the meter reads must exist on the real cache, not only on a fake: a prices line
        is logged, and with nothing parked or evictable the cache's new answers are empty."""
        hold_switches(self, (envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0))
        for mode in ("observe", "on"):
            with self.subTest(mode=mode):
                cache = cpu_unified_cache()
                ntp = NextTurnPrefetcher(mode=mode, tree_cache=cache, page_size=2)
                self.assertIsNotNone(cache.tree_core.host_eviction_observer)
                with self.assertLogs(LOGGER, level="INFO") as logs:
                    ntp.tick(waiting_queue=[make_req("u-t1", entered=0.0)], now=0.0)
                prices = [line for line in logs.output if "HICACHE_EVT prices " in line]
                self.assertEqual(len(prices), 1)
                nobody = CacheRequestHandle("nobody", 0)
                self.assertIsNone(cache.next_host_victim())
                self.assertEqual(cache.storage_read_line_pages(None), 0)
                self.assertIsNone(cache.parked_read_span(nobody))
                self.assertFalse(cache.drop_parked_read(nobody))


class TestArrivalHold(PrefetcherTestCase):
    mode = "on"

    def make_cache(self):
        return DeadlineCache()

    def setUp(self):
        super().setUp()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_hold="on"
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.parked = ParkedRead(handle=self.handle, operation=None, pages=10, parked_at=0.0)
        self.req = make_req("s-t1", entered=2.0)
        self.others = [make_req(f"y{i}", entered=2.0) for i in range(5)]
        # 30 admissions give the queue a measured drain rate.
        self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)

    def test_a_queued_read_waits_for_admission_minus_load_minus_margin(self):
        """Pages ahead move the start earlier. Near the head the read goes at once, even
        if the rate says otherwise: admission is already passing over this turn."""
        found = self.events(
            lambda: self.ntp.tick(waiting_queue=self.others + [self.req], now=2.0)
        )
        self.assertEqual([k for k, _ in found if k == "ntp_hold"], [])
        wait = residual_wait(position=5, rate=self.ntp._last_rate)
        load = L3LoadModel().seconds(own_pages=10, backlog_pages=40)
        self.assertAlmostEqual(
            self.ntp.read_not_before(self.handle, self.parked, 40),
            2.0 + wait - load - NTP_HOLD_MARGIN_S,
        )
        self.assertLess(
            self.ntp.read_not_before(self.handle, self.parked, 4000),
            self.ntp.read_not_before(self.handle, self.parked, 40),
        )
        self.ntp.tick(waiting_queue=[self.others[0], self.req], now=3.0)
        self.assertIsNone(self.ntp.read_not_before(self.handle, self.parked, 0))

    def test_a_read_is_logged_held_once_not_once_per_pass(self):
        """Dispatch asks for every parked read on every pass: 100 parked reads at 50
        passes a second would otherwise write 5,000 lines a second."""
        self.ntp.tick(waiting_queue=self.others + [self.req], now=2.0)
        found = self.events(
            lambda: [self.ntp.read_not_before(self.handle, self.parked, 0) for _ in range(3)]
        )
        self.assertEqual([k for k, _ in found], ["ntp_hold"])

    def test_without_a_measured_rate_a_queued_read_goes(self):
        """Nothing says how long the wait is, and a parked read blocks its turn."""
        ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_hold="on"
        )
        ntp.tick(waiting_queue=self.others + [self.req], now=2.0)
        self.assertIsNone(ntp.read_not_before(self.handle, self.parked, 0))

    def test_a_staged_read_goes_until_its_turn_arrives_then_follows_the_queue(self):
        """Before arrival the prefetcher already chose the time. A read still parked when
        the turn queues behind others has become an arrived read."""
        self.commit("s-t0", "s-t1", now=2.0)
        self.ntp.tick(waiting_queue=[], now=2.01)
        self.assertTrue(self.ntp._plans["s-t1"].staged)
        self.assertIsNone(self.ntp.read_not_before(self.handle, self.parked, 0))
        self.ntp.tick(waiting_queue=self.others + [self.req], now=7.0)
        self.assertGreater(self.ntp.read_not_before(self.handle, self.parked, 0), 7.0)

    def test_an_unknown_position_waits_one_pass_and_never_more(self):
        """The hit can be drained before the tick that first sees the queued turn: holding
        it one pass avoids reading it early. A rid that is still unknown is not queued,
        and holding it for ever would block admission of the turn it belongs to."""
        self.ntp.tick(waiting_queue=[], now=2.0)
        fresh = ParkedRead(handle=self.handle, operation=None, pages=10, parked_at=2.5)
        self.assertEqual(self.ntp.read_not_before(self.handle, fresh, 0), float("inf"))
        self.ntp.tick(waiting_queue=[], now=3.0)
        self.assertIsNone(self.ntp.read_not_before(self.handle, fresh, 0))

    def test_a_turn_with_no_plan_is_timed_from_its_queue_position(self):
        """First turns and expired plans get the same treatment, and the drain rate must
        stay fresh with no plan left to refresh it. With the hold off they stay "now"."""
        self.assertEqual(self.ntp._plans, {})
        self.ntp.tick(waiting_queue=self.others + [self.req], now=2.0)
        rate = self.ntp._last_rate
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(
            self.ntp.read_deadline(self.handle),
            2.0 + residual_wait(position=5, rate=rate),
        )
        self.assertGreater(self.ntp.read_not_before(self.handle, self.parked, 0), 2.0)
        unheld = NextTurnPrefetcher(mode=self.mode, tree_cache=self.cache, page_size=PAGE)
        unheld.tick(waiting_queue=self.others + [self.req], now=2.0)
        self.assertIsNone(unheld.read_deadline(self.handle))

    def test_a_queue_of_held_reads_drains_and_each_starts_near_its_admission(self):
        """The two properties the hold rests on, through the real dispatcher. No stall: a
        parked read blocks its turn, so every held read must eventually go. No early
        start: without the hold all twelve would start at once, 20 s before the last."""
        srs = self.cache.read_scheduler
        queue = [make_req(f"q{i}", entered=2.0) for i in range(12)]
        for req in queue:
            srs.park(handle=req.cache_request_handle, operation=None, pages=10, now=2.0)
        started, landed, admitted = {}, {}, {}
        now = 2.0
        while queue and now < 200.0:
            self.ntp.tick(waiting_queue=queue, now=now)
            for d in srs.dispatch(
                now=now,
                deadline_of=self.ntp.read_deadline,
                in_flight_pages=0,
                held_tokens=0,
                not_before_of=self.ntp.read_not_before,
            ):
                started[d.read.handle.rid] = now
                landed[d.read.handle.rid] = now + 1.0
            now += 0.5
            # One admission per pass at most, and only once its read has landed.
            ready = [r for r in queue if landed.get(r.rid, float("inf")) <= now]
            if ready:
                ready[0].time_stats.forward_entry_time = now
                admitted[ready[0].rid] = now
                queue = [r for r in queue if r is not ready[0]]
        self.assertEqual(queue, [])
        self.assertEqual(srs.parked_count(), 0)
        leads = [admitted[rid] - started[rid] for rid in admitted]
        self.assertLessEqual(max(leads), 6.0)
        self.assertGreater(started["q11"] - started["q0"], 3.0)


def gate_decisions(found):
    return [(f["decision"], f["lookups"]) for kind, f in found if kind == "ntp_gate"]


def reevictions(found):
    return [(f["refetches"], f["capped"]) for kind, f in found if kind == "ntp_reevict"]


class RoomRetryTestCase(PrefetcherTestCase):
    """A queued turn whose storage reads find no host room, with
    SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY on for the whole test. The turn waits behind
    queue_ahead others, with a fixed 30 s wait estimate away from the head."""

    mode = "on"
    arrival_gate = "off"
    queue_ahead = 2

    def make_cache(self):
        return DeadlineCache()

    def setUp(self):
        # Closed as a cleanup: override() skips its restore when the test body raises.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY.override(True))
        stack.enter_context(
            mock.patch(
                "sglang.srt.mem_cache.next_turn_prefetch.residual_wait",
                side_effect=lambda *, position, rate: None if position <= 1 else 30.0,
            )
        )
        self.cache = self.make_cache()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            arrival_gate=self.arrival_gate,
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.req = make_req("s-t1", entered=5.0)
        self.ahead = [make_req(f"q-t{i}", entered=4.0) for i in range(self.queue_ahead)]
        self.commit("s-t0", "s-t1", now=0.0)

    def lookup(self):
        """The turn's prefetch reaching the cache, where a bound gate decides if it reads."""
        gate = self.cache.prefetch_gate_fn
        if gate is None or gate(self.handle, self.cache.resident):
            self.cache.ongoing.add(self.handle)

    def scheduler_pass(self, now):
        """tick(), then what Scheduler._on_next_turn_refetch does per returned req."""
        returned = []

        def run():
            for req in self.ntp.tick(waiting_queue=self.ahead + [self.req], now=now):
                returned.append(req.rid)
                self.lookup()
                self.ntp.on_refetch_result(
                    req=req, issued=self.cache.has_ongoing_prefetch(self.handle)
                )

        return returned, self.events(run)

    def refuse(self, now):
        """_launch_storage_hit finding no host room: the read is reported, then revoked."""
        if self.cache.prefetch_refused_fn is not None:
            self.cache.prefetch_refused_fn(
                self.handle, tokens=KEY_TOKENS, cut=False, now=now
            )
        self.cache.ongoing.discard(self.handle)


class TestRoomRetry(RoomRetryTestCase):
    arrival_gate = "on"

    def test_a_read_refused_room_is_retried_later_at_no_cost_to_the_turn(self):
        """A read refused host room is retried NTP_ROOM_RETRY_S later and spends neither a
        lookup nor a refetch: refusals 25 ms apart must not use up MAX_REFETCHES and get
        the turn admitted without its history."""
        self.cache.resident = 0
        self.assertEqual(gate_decisions(self.events(self.lookup)), [("read", "1")])
        self.assertEqual(self.scheduler_pass(5.0)[0], [])  # its own read is under way
        now = 5.0
        for _ in range(2):  # the read, then its retry, refused room
            self.refuse(now)
            retry_at = now + NTP_ROOM_RETRY_S
            for early in (now + 0.025, retry_at - 0.125):
                self.assertEqual(self.scheduler_pass(early)[0], [])
            returned, found = self.scheduler_pass(retry_at)
            self.assertEqual(
                (returned, gate_decisions(found), reevictions(found)),
                (["s-t1"], [("retry", "1")], []),
            )
            now = retry_at
        self.assertEqual(self.ntp._plans["s-t1"].refetches, 0)

    def test_behind_turns_waiting_on_their_reads_a_refused_read_cannot_wait(self):
        """Admission passes over a turn whose own read is under way, so a refused turn
        behind only such turns is the next one admitted: like the head, its read is
        re-issued at once and charged, or admission takes it without its history."""
        for req in self.ahead:
            self.cache.ongoing.add(req.cache_request_handle)
        self.cache.resident = 0
        self.lookup()
        self.refuse(5.0)
        returned, found = self.scheduler_pass(5.0)
        self.assertEqual(
            (returned, gate_decisions(found)), (["s-t1"], [("reread", "2")])
        )

    def test_a_charged_retry_ends_the_free_one(self):
        """A charged lookup that retries a refused read (here at the head, after the
        turns ahead left) takes the refusal's free retry with it: if that read lands and
        is evicted again, the next re-read is charged, as for any landed read."""
        self.cache.resident = 0
        self.lookup()
        self.scheduler_pass(5.0)
        self.refuse(5.0)  # at position 2: the turn waits
        self.assertEqual(self.scheduler_pass(5.5)[0], [])
        self.ahead = []  # the two ahead were admitted
        returned, found = self.scheduler_pass(6.0)
        self.assertEqual(
            (returned, gate_decisions(found)), (["s-t1"], [("reread", "2")])
        )
        self.cache.ongoing.clear()
        self.cache.resident = KEY_TOKENS  # it landed
        self.cache.resident = 0  # and was evicted before a probe saw it
        returned, found = self.scheduler_pass(5.0 + NTP_ROOM_RETRY_S)
        self.assertEqual(
            (returned, gate_decisions(found), reevictions(found)),
            (["s-t1"], [("reread", "3")], [("1", "0")]),
        )

    def test_a_lookup_sooner_than_the_wait_is_charged(self):
        """A lookup earns the free retry only NTP_ROOM_RETRY_S after the refusal, whatever
        triggers it: one sooner (the arrival lookup, the admission re-query) spends a
        lookup as before, so the gate cannot be used to skip the wait."""
        self.cache.resident = 0
        self.lookup()
        self.scheduler_pass(5.0)
        self.refuse(5.0)
        self.assertEqual(gate_decisions(self.events(self.lookup)), [("reread", "2")])

    def test_past_its_budget_a_refused_read_is_charged_as_before(self):
        """The free retries stop at NTP_MAX_ROOM_RETRIES: a later refusal spends lookups and
        refetches again, at the probe interval, until the refetch is capped."""
        self.cache.resident = 0
        self.lookup()
        self.scheduler_pass(5.0)
        now = 5.0
        for _ in range(NTP_MAX_ROOM_RETRIES):
            self.refuse(now)
            now += NTP_ROOM_RETRY_S
            returned, found = self.scheduler_pass(now)
            self.assertEqual(
                (returned, gate_decisions(found)), (["s-t1"], [("retry", "1")])
            )
        charged = []
        for _ in range(3):
            self.refuse(now)
            now += PROBE_INTERVAL_S
            returned, found = self.scheduler_pass(now)
            charged.append((returned, gate_decisions(found), reevictions(found)))
        self.assertEqual(
            charged,
            [
                (["s-t1"], [("reread", "2")], []),
                (["s-t1"], [("reread", "3")], [("1", "0")]),
                ([], [], [("2", "1")]),
            ],
        )
        self.assertEqual(self.scheduler_pass(now + PROBE_INTERVAL_S), ([], []))

    def test_a_landed_read_evicted_again_is_charged_as_before(self):
        """Only a refusal is free: with the switch on, a read that landed and was evicted
        again still spends a lookup and a refetch, up to the cap."""
        self.cache.resident = KEY_TOKENS
        self.scheduler_pass(5.0)
        self.cache.resident = 0
        returned, found = self.scheduler_pass(5.6)
        self.assertEqual((returned, gate_decisions(found)), (["s-t1"], [("read", "1")]))
        self.cache.ongoing.clear()
        self.cache.resident = KEY_TOKENS  # the read landed
        self.assertEqual(self.scheduler_pass(6.2)[0], [])
        self.cache.resident = 0  # and was evicted again
        returned, found = self.scheduler_pass(6.8)
        self.assertEqual(
            (returned, gate_decisions(found), reevictions(found)),
            (["s-t1"], [("reread", "2")], [("1", "0")]),
        )
        self.cache.ongoing.clear()
        returned, found = self.scheduler_pass(7.4)
        self.assertEqual((returned, reevictions(found)), ([], [("2", "1")]))
        self.assertEqual(self.scheduler_pass(8.0), ([], []))


class TestRoomRetryWithoutGate(RoomRetryTestCase):
    def test_the_refetch_takes_the_retry_and_refusals_never_cap_the_turn(self):
        """Without the arrival gate (the oracle_v2 arm) the refetch takes the retry: more
        refusals than MAX_REFETCHES spend no refetch, and once a retry has landed, a new
        eviction is charged."""
        self.cache.resident = 0
        self.assertEqual(self.scheduler_pass(5.0)[0], ["s-t1"])  # the read at arrival
        now = 5.0
        for _ in range(MAX_REFETCHES + 1):
            self.refuse(now)
            now += NTP_ROOM_RETRY_S
            returned, found = self.scheduler_pass(now)
            self.assertEqual((returned, reevictions(found)), (["s-t1"], []))
            self.assertEqual(
                [f["room"] for kind, f in found if kind == "ntp_refetch"], ["1"]
            )
        self.cache.ongoing.clear()
        self.cache.resident = KEY_TOKENS  # the last retry landed
        self.assertEqual(self.scheduler_pass(now + PROBE_INTERVAL_S)[0], [])
        self.cache.resident = 0  # and was evicted again
        returned, found = self.scheduler_pass(now + 2 * PROBE_INTERVAL_S)
        self.assertEqual((returned, reevictions(found)), (["s-t1"], [("1", "0")]))

    def test_a_history_back_by_another_read_leaves_no_free_retry(self):
        """A refusal's free retry ends when the history comes back by another read (the
        turn's arrival read, say): a later re-eviction is charged, so the gate-less arm
        cannot re-read past MAX_REFETCHES."""
        self.cache.resident = 0
        self.assertEqual(self.scheduler_pass(5.0)[0], ["s-t1"])
        self.refuse(5.0)
        self.cache.ongoing.clear()
        self.cache.resident = KEY_TOKENS  # back by another read before the retry
        self.assertEqual(self.scheduler_pass(5.0 + PROBE_INTERVAL_S)[0], [])
        self.cache.resident = 0  # evicted again, past the retry's wait
        returned, found = self.scheduler_pass(5.0 + NTP_ROOM_RETRY_S + PROBE_INTERVAL_S)
        self.assertEqual((returned, reevictions(found)), (["s-t1"], [("1", "0")]))


class TestRoomRetryAtHead(RoomRetryTestCase):
    arrival_gate = "on"
    queue_ahead = 0

    def test_at_the_head_a_refused_read_is_reissued_at_once_and_charged(self):
        """A turn at the queue head cannot wait NTP_ROOM_RETRY_S, or admission takes it
        without its history before the retry goes out: its read is re-issued in the
        refusal's own pass and charged as with the switch off."""
        self.cache.resident = 0
        self.lookup()
        self.refuse(5.0)
        returned, found = self.scheduler_pass(5.0)
        self.assertEqual(
            (returned, gate_decisions(found)), (["s-t1"], [("reread", "2")])
        )


ADMIT_FIELDS = [
    "rid",
    "wait_ms",
    "pred_wait_ql_ms",
    "pred_wait_ewma_ms",
    "adm_err_ms",
    "trig_lead_ms",
    "case",
]
# As logged with every switch off: the scripts read these lines by position.
EC_START_FIELDS = [
    "rid",
    "decision",
    "tokens",
    "start_ms",
    "idle_ms",
    "read_ms",
    "wait_ms",
    "price",
]
HOLD_FIELDS = ["rid", "pos", "pages", "wait_ms", "load_ms", "hold_ms"]
PRICES_FIELDS = [
    "rid",
    "c",
    "mu",
    "waiting",
    "ready",
    "queued",
    "plans",
    "kappa",
    "late",
    "parked",
    "pend_w",
    "lam",
    "evict_tok",
    "k_us",
    "rec_a_us",
    "rec_b_ns",
    "rec_n",
    "read_ms_page",
    "err_n",
    "err_q10_ms",
    "err_q50_ms",
    "err_q90_ms",
    "rd_lag_ms",
    "q20k",
    "margin_ms",
    "lookup_ms",
    "lag_ms",
    "whold_ms",
    "win_holds",
    "budget_holds",
    "held",
    "write_holds",
]


class TestMeasuredPrices(PrefetcherTestCase):
    """The prices measured from the queues (next_turn_prices.py), with the prices line on."""

    mode = "on"

    def make_cache(self):
        return MeasuredCache()

    def setUp(self):
        hold_switches(self, (envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, 5.0))
        super().setUp()
        self.cache.resident = KEY_TOKENS  # nothing evicted: no staging noise
        self.board = self.ntp._prices

    def admit_one_per_pass(self, staying, *, passes, start):
        """`passes` ticks 0.25 s apart: on each a new turn arrives, to be admitted before
        the next, while `staying` remain queued. Returns the time of the last tick."""
        now = start
        for step in range(passes):
            turn = make_req(f"t{start}-{step}", entered=now)
            self.ntp.tick(waiting_queue=staying + [turn], now=now)
            turn.time_stats.forward_entry_time = now + 0.1
            now += 0.25
        self.ntp.tick(waiting_queue=staying, now=now)
        return now

    def test_c_falls_only_when_admissions_leave_a_turn_with_nothing_to_read_waiting(
        self,
    ):
        """Admission passes over a queued turn whose own read is under way, so that turn
        delays nobody. Counted as waiting for the GPU, a queue held up by the SSD would
        read as a saturated GPU and switch every early read off."""
        blocked = [make_req(f"b{i}", entered=0.0) for i in range(3)]
        for req in blocked:
            self.cache.ongoing.add(req.cache_request_handle)
        now = self.admit_one_per_pass(blocked, passes=12, start=0.0)
        self.assertEqual(self.board.c(now), 1.0)
        self.cache.ongoing.clear()  # their reads landed: now the GPU is what they wait for
        now = self.admit_one_per_pass(blocked, passes=60, start=now)
        self.assertLess(self.board.c(now), 0.3)

    def test_a_prices_line_per_interval_carries_what_rebuilds_the_users_in_session(
        self,
    ):
        """Users in session = waiting + the running count of the batch lines: every plan
        held (a session between turns or queued) plus the queued turns with no plan. The
        scripts read values as [\\w./-]+, which an exponent's plus sign would break."""
        self.commit("s-t0", "s-t1", now=0.0)
        self.commit("v-t0", "v-t1", now=0.0, gap_s=500.0)
        queue = [make_req("u-t1", entered=0.0), make_req("s-t1", entered=0.0)]
        found = self.events(
            lambda: [
                self.ntp.tick(waiting_queue=queue, now=now)
                for now in (0.0, 1.0, 4.9, 5.0, 5.1)
            ]
        )
        lines = [fields for kind, fields in found if kind == "prices"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(
            [lines[1][name] for name in ("rid", "plans", "queued", "waiting", "ready")],
            ["-", "2", "2", "3", "2"],
        )
        for name, value in lines[1].items():
            self.assertRegex(value, r"\A[\w./-]+\Z", name)

    def test_the_prices_line_names_every_price_and_counts_holds_since_the_line_before(
        self,
    ):
        """A field dropped is a column lost for a whole run. The hold counts are per line, not
        running totals; under the fifo read order they and the lags read na."""
        holds = ("win_holds", "budget_holds", "held", "write_holds")

        def line_at(ntp, now):
            found = self.events(lambda: ntp.tick(waiting_queue=[], now=now))
            (line,) = [fields for kind, fields in found if kind == "prices"]
            return line

        stats = self.cache.read_scheduler.stats
        stats.window_holds, stats.hold_skips = 3, 7
        first = line_at(self.ntp, 0.0)
        self.assertEqual(list(first), PRICES_FIELDS)
        self.assertEqual([first[name] for name in holds], ["3", "0", "7", "0"])
        stats.window_holds, stats.write_holds = 5, 4
        second = line_at(self.ntp, 5.0)
        self.assertEqual([second[name] for name in holds], ["2", "0", "0", "4"])

        fifo = self.make_cache()
        fifo.read_scheduler = None
        line = line_at(
            NextTurnPrefetcher(mode=self.mode, tree_cache=fifo, page_size=PAGE), 0.0
        )
        self.assertEqual(list(line), PRICES_FIELDS)
        self.assertEqual(
            [line[name] for name in ("parked", "late", "lookup_ms", *holds)],
            ["0", "0", "na", "na", "na", "na", "na"],
        )

    def test_ntp_admit_keeps_its_fields_and_gains_the_reader_flag_and_the_landing(self):
        """The scripts read ntp_admit by position: the two fields go at its end, and only
        while prices are measured. A reader's admission is timed from its last landing."""
        for measuring, added in ((True, ["reader", "land_ms"]), (False, [])):
            with self.subTest(measuring=measuring):
                stack = hold_switches(
                    self,
                    (
                        envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL,
                        5.0 if measuring else 0.0,
                    ),
                )
                ntp = NextTurnPrefetcher(
                    mode=self.mode, tree_cache=self.cache, page_size=PAGE
                )
                stack.close()
                ntp.on_turn_committed(
                    req=make_req("a1", kv_hints=hints("a2")),
                    key=key_of(KEY_TOKENS),
                    rotation_tail_declined=False,
                    now=0.0,
                )
                a2 = make_req("a2", entered=5.15)
                ntp.tick(waiting_queue=[a2], now=5.2)
                ntp.on_read_landed(a2.cache_request_handle, 7)
                a2.storage_hit_length = 640
                a2.time_stats.forward_entry_time = 5.9
                found = self.events(lambda: ntp.tick(waiting_queue=[], now=6.0))
                (admit,) = [fields for kind, fields in found if kind == "ntp_admit"]
                self.assertEqual(list(admit), ADMIT_FIELDS + added)
                if measuring:
                    self.assertEqual((admit["reader"], admit["land_ms"]), ("1", "700"))

    def test_the_admission_error_is_fed_by_turns_that_read_nothing_with_readers_apart(
        self,
    ):
        """The error bins hold measured minus predicted wait where the queue's rate predicted
        it: not a reader's wait, a retracted request's, or a turn that found the queue empty."""
        tracker = self.ntp._meter.admission_error
        for rid in ("a", "b", "c"):
            self.commit(f"{rid}1", f"{rid}2", now=0.0)
        self.commit("z1", "z2", now=0.0, gap_s=500.0)  # keeps the queue followed
        # 30 admissions: the queue drains one turn every 2 s.
        self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)
        self.assertEqual([len(errors) for errors in tracker.errors], [0] * 5)

        def through_queue(*rids, entered, admitted, reader=None):
            reqs = [make_req(rid, entered=entered) for rid in rids]
            self.ntp.tick(waiting_queue=reqs, now=entered)
            for req in reqs:
                req.storage_hit_length = 640 if req.rid == reader else 0
                req.time_stats.forward_entry_time = admitted
            return reqs

        # u found the queue empty; a2 was predicted 3 s behind it and v, with no plan, 5 s.
        _, a2, _ = through_queue("u", "a2", "v", entered=5.0, admitted=5.3)
        self.ntp.tick(waiting_queue=[], now=5.4)
        self.assertEqual([len(errors) for errors in tracker.errors], [0, 1, 1, 0, 0])
        self.assertAlmostEqual(tracker.errors[1][0], 0.3 - 3.0)

        b2 = make_req("b2", entered=6.0)
        self.ntp.tick(waiting_queue=[b2], now=6.0)
        self.ntp.tick(waiting_queue=[b2], now=7.0)
        self.ntp.on_read_landed(b2.cache_request_handle, 9)
        b2.storage_hit_length = 640
        b2.time_stats.forward_entry_time = 7.4
        self.ntp.tick(waiting_queue=[], now=7.5)
        self.assertAlmostEqual(tracker.reader_lags[0], 0.4)
        # A reader whose landing was not seen has no lag, and is no non-reader either.
        through_queue("u2", "c2", entered=8.0, admitted=8.3, reader="c2")
        self.ntp.tick(waiting_queue=[], now=8.4)
        # Retracted: a2 is queued again after its first forward, with no plan now.
        a2.time_stats.wait_queue_entry_time = 9.0
        self.ntp.tick(waiting_queue=[make_req("u3", entered=9.0), a2], now=9.0)
        self.ntp.tick(waiting_queue=[], now=9.5)
        self.assertEqual([len(errors) for errors in tracker.errors], [0, 1, 1, 0, 0])
        self.assertEqual(len(tracker.reader_lags), 1)

    def test_an_aborted_turn_is_not_an_admission(self):
        """c is the share of admissions that left no ready turn waiting: an abort that left
        the queue beside a ready turn would otherwise read as the GPU passing that turn over."""
        ready, gone = make_req("ready", entered=0.0), make_req("gone", entered=0.0)
        self.ntp.tick(waiting_queue=[ready, gone], now=0.0)
        self.ntp.tick(waiting_queue=[ready, gone], now=1.0)
        self.ntp.tick(waiting_queue=[ready], now=1.1)
        self.assertEqual(self.board.c(1.1), 1.0)

    def test_a_landing_is_stamped_for_a_planned_or_queued_turn_under_any_eviction_policy(
        self,
    ):
        """Under next_use there is no return_chance strategy to adopt a landed span into, and a
        landing for a request never seen queued would stay in the table for ever."""
        self.assertIsInstance(self.cache.tree_core.eviction_strategy, NextUseStrategy)
        self.ntp.on_turn_committed(
            req=make_req("s-t0", kv_hints=hints("s-t1")),
            key=key_of(KEY_TOKENS),
            rotation_tail_declined=False,
            now=0.0,
            tail_node_id=5,
        )
        self.ntp.tick(waiting_queue=[make_req("queued", entered=1.0)], now=1.0)
        for rid in ("s-t1", "queued", "unknown"):
            self.cache.prefetch_landed_fn(CacheRequestHandle(rid, 0), 9)
        self.assertEqual(self.ntp._landed_at, {"s-t1": 1.0, "queued": 1.0})

    def test_host_eviction_prices_the_room_by_the_victim_worth_most(self):
        """lam is the highest holding rate among a pass's victims times what a read costs per
        token restored; an un-hinted prefix is evicted tokens with no price of its own."""
        registry = self.cache.tree_core.eviction_strategy.registry
        registry.register(1, 100.0)
        registry.register(2, 10.0)
        self.ntp.tick(waiting_queue=[], now=0.0)
        self.assertEqual(self.board.lam(0.0), 0.0)
        for node_id in (1, 2, 4):
            self.cache.tree_core.host_eviction_observer(
                SimpleNamespace(id=node_id), 640
            )
        self.ntp.tick(waiting_queue=[], now=0.1)
        read_s_per_token = L3LoadModel().read_s_per_page / PAGE
        self.assertAlmostEqual(self.board.lam(0.1), (1.0 / 10.0) * read_s_per_token)
        self.assertEqual(self.board.evicted_tokens, 3 * 640)

    def test_a_queued_turns_history_is_priced_by_the_wait_it_has_left(self):
        """Unpriced, room reads as free while eviction takes queued turns' history. "farthest"
        takes the last arrival (9 s of queue here); "lru" the turn nearest admission."""
        read_s_per_token = L3LoadModel().read_s_per_page / PAGE
        queue = [make_req(f"q{i}", entered=2.0) for i in range(5)]
        for order, rate in (
            ("farthest", 1.0 / 9.0),
            ("lru", 1.0 / HOLDING_MIN_HORIZON_S),
        ):
            with self.subTest(order=order):
                cache = self.make_cache()
                cache.tree_core.eviction_strategy = NextUseStrategy(
                    horizon_s=300.0, clock=lambda: 2.0, queued_order=order
                )
                cache.tree_core.eviction_strategy.registry.register(
                    7, Queued(since=2.0)
                )
                self.ntp = NextTurnPrefetcher(
                    mode=self.mode, tree_cache=cache, page_size=PAGE
                )
                # 30 admissions: the queue drains one turn every 2 s.
                self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)
                self.ntp.tick(waiting_queue=queue, now=2.0)
                cache.tree_core.host_eviction_observer(SimpleNamespace(id=7), 640)
                self.ntp.tick(waiting_queue=queue, now=2.1)
                self.assertAlmostEqual(
                    self.ntp._prices.lam(2.1), rate * read_s_per_token
                )

    def test_the_room_price_does_not_feed_on_the_writes_it_prices(self):
        """With kappa's pending-write term in the restore cost behind lam, 50K pending tokens
        multiply each sample by 1.95 and lam climbs to its ceiling whatever room is worth."""
        registry = self.cache.tree_core.eviction_strategy.registry
        self.cache.pending_backup_tokens = 50_000
        for second in range(1, 61):
            # The strategy's clock stands at 0: due back in a second, a rate of 1 a second.
            registry.register(second, 1.0)
            self.cache.tree_core.host_eviction_observer(SimpleNamespace(id=second), 640)
            self.ntp.tick(waiting_queue=[], now=float(second))
        self.assertAlmostEqual(
            self.board.lam(60.0), L3LoadModel().read_s_per_page / PAGE
        )

    def test_kappa_counts_the_parked_reads_past_their_latest_start(self):
        """A parked read is late once even dispatched now it lands after its turn is due,
        and a read needed now always is. Counted by deadline alone, a read due in 30 s
        that takes 40 s would read as on time."""
        srs = self.cache.read_scheduler
        self.commit("s-t0", "s-t1", now=0.0, gap_s=50.0)
        due = ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=50.0)
        own_s = L3LoadModel().seconds(own_pages=10, backlog_pages=0)
        srs.park(
            handle=CacheRequestHandle("s-t1", 0), operation=None, pages=10, now=0.0
        )
        srs.park(
            handle=CacheRequestHandle("unplanned", 0), operation=None, pages=1, now=0.0
        )

        def late_at(now):
            found = self.events(lambda: self.ntp.tick(waiting_queue=[], now=now))
            (line,) = [fields for kind, fields in found if kind == "prices"]
            return (line["late"], line["parked"])

        self.assertEqual(late_at(due - own_s - 5.0), ("1", "2"))
        self.assertEqual(late_at(due - own_s + 0.1), ("2", "2"))

    def test_the_prefill_fit_is_fed_only_by_turns_admitted_at_once(self):
        """A turn that queued shares its prefill passes with the turns admitted beside it,
        and a retracted one re-entered the queue after its prefill: neither time is the
        GPU cost of the turn's own tokens (fed with all turns the fit ran 23 % high)."""

        def finish(rid, *, entered, forward, done):
            req = make_req(rid, kv_hints=hints(f"{rid}-next"), entered=entered)
            req.time_stats.forward_entry_time = forward
            req.time_stats.prefill_finished_time = done
            req.cached_tokens = 1000
            req.origin_input_ids = array("q", range(9000))
            self.ntp.on_turn_committed(
                req=req, key=key_of(KEY_TOKENS), rotation_tail_declined=False, now=10.0
            )

        cost = self.ntp._meter.prefill_cost
        finish("queued", entered=1.0, forward=3.0, done=3.3)
        finish("retracted", entered=5.0, forward=1.01, done=1.31)
        self.assertEqual(cost.samples, 0)
        for turn in range(cost.min_samples):
            finish(f"at-once-{turn}", entered=1.0, forward=1.01, done=1.31)
        self.assertEqual(cost.samples, cost.min_samples)
        # 8,000 new tokens on 1,000 cached ones took 0.3 s each time.
        self.assertAlmostEqual(cost.seconds(new_tokens=8000, total_tokens=9000), 0.3)


class TestFreeEmptyLookups(PrefetcherTestCase):
    """SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS: a lookup that finds nothing in storage
    is not one of the turn's reads (agent_cache/OPEN_ITEMS.md section 1)."""

    mode = "on"
    # The turn's prompt repeats the key this far, the previous prompt; the rest of the key
    # is the reply as generated, which the client sends back in its own rendering.
    SHARED = KEY_TOKENS - 256

    def make_cache(self):
        return MeasuredCache()

    def setUp(self):
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS, True))
        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = KEY_TOKENS
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_gate="on"
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.req = make_req("s-t1", entered=5.1)
        self.commit("s-t0", "s-t1", now=0.0)
        self.plan = self.ntp._plans["s-t1"]

    def gate(self, matched):
        """The turn's prefetch reaching the gate with `matched` tokens of its prompt in
        memory; returns the events logged."""
        tail = array("q", [*range(matched, self.SHARED), *range(9000, 9400)])
        return self.events(
            lambda: self.cache.prefetch_gate_prompt_fn(self.handle, matched, tail)
        )

    def answer(self, *, start, hit):
        """The cache draining the answer of the lookup under way."""
        self.cache.ongoing.discard(self.handle)
        self.cache.prefetch_lookup_fn(
            self.handle,
            storage_start=start,
            requested_tokens=KEY_TOKENS,
            hit_tokens=hit,
        )

    def test_a_reply_the_next_prompt_does_not_repeat_is_not_missing_history(self):
        """The client sends the reply back in its own rendering, so the prompt leaves the
        key before its end. Counted as missing, that tail cost each such turn a lookup for
        text no tier holds (92 % of campaign 34's empty arrival lookups) and one of its
        three reads; history that did leave memory is still read."""
        found = self.gate(self.SHARED)
        self.assertEqual(
            [(f["decision"], f["chain"]) for kind, f in found if kind == "ntp_gate"],
            [("covered", str(self.SHARED))],
        )
        self.assertEqual(gate_decisions(self.gate(0)), [("read", "1")])

    def test_an_empty_lookup_is_given_back_and_not_asked_again_until_the_match_shrinks(
        self,
    ):
        """A lookup stops at the first page missing, so nothing is stored past its hit:
        asking again from the same match can only find nothing again. Once the match
        falls below that point, the span up to it is worth a lookup."""
        self.assertEqual(gate_decisions(self.gate(512)), [("read", "1")])
        self.answer(start=512, hit=0)
        self.assertEqual((self.plan.lookups, self.plan.free_lookups), (0, 1))
        self.assertEqual(gate_decisions(self.gate(512)), [("absent", "0")])
        found = self.gate(128)
        self.assertEqual(
            [(f["decision"], f["chain"]) for kind, f in found if kind == "ntp_gate"],
            [("read", "512")],
        )
        # This one finds pages: the lookup stays charged, and storage ends where its hit does.
        self.answer(start=128, hit=320)
        self.assertEqual((self.plan.lookups, self.plan.absent_from), (1, 448))

    def test_a_turn_whose_lookups_found_nothing_still_reads_its_history_once_it_leaves_memory(
        self,
    ):
        """Three lookups for a reply's tail that no tier held used up a turn's reads; when
        its history left memory two minutes later the gate answered capped and the whole
        prompt was computed again (OPEN_ITEMS.md section 1). Here the gate is not told the
        prompt, so only the lookup's answer protects the turn."""
        history = KEY_TOKENS - 256
        self.assertTrue(self.cache.prefetch_gate_fn(self.handle, history))
        self.answer(start=history, hit=0)
        found = self.events(
            lambda: [
                self.cache.prefetch_gate_fn(self.handle, history) for _ in range(4)
            ]
        )
        self.assertEqual(gate_decisions(found), [("absent", "0")] * 4)
        found = self.events(
            lambda: self.assertTrue(self.cache.prefetch_gate_fn(self.handle, 0))
        )
        self.assertEqual(gate_decisions(found), [("read", "1")])

    def test_lookups_are_given_back_only_so_many_times(self):
        """A turn whose match keeps shrinking onto spans the SSD cleaner emptied would look
        up without end; past the allowance an empty lookup is charged like a read."""
        self.commit("b-t0", "b-t1", now=0.0, tokens=20 * 256)
        handle = CacheRequestHandle(rid="b-t1", attempt_id=0)
        decisions = []
        for step in range(NTP_MAX_FREE_LOOKUPS + 4):
            matched = (19 - step) * 256
            found = self.events(lambda: self.cache.prefetch_gate_fn(handle, matched))
            decisions += gate_decisions(found)
            self.cache.prefetch_lookup_fn(
                handle, storage_start=matched, requested_tokens=256, hit_tokens=0
            )
        self.assertEqual(
            decisions,
            [("read", "1")] * (NTP_MAX_FREE_LOOKUPS + 1)
            + [("reread", "2"), ("reread", "3"), ("capped", "3")],
        )
        self.assertEqual(self.ntp._plans["b-t1"].free_lookups, NTP_MAX_FREE_LOOKUPS)

    def test_the_staged_lookup_does_not_set_what_the_prompt_repeats(self):
        """The prefetcher's own lookup asks for the key's tail before the turn exists; taken
        for the prompt, it would say the turn repeats the whole key and the reply's tail
        would be missing history again when the turn arrives."""
        self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        found = self.events(lambda: self.ntp.tick(waiting_queue=[], now=0.01))
        self.assertEqual(gate_decisions(found), [("read", "1")])
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])
        self.assertIsNone(self.plan.shared_tokens)
        self.answer(start=4 * PAGE, hit=KEY_TOKENS - 4 * PAGE)
        self.assertEqual(gate_decisions(self.gate(self.SHARED)), [("covered", "1")])

    def test_an_empty_re_read_of_a_queued_turn_is_not_a_re_eviction(self):
        """The queue check counts every re-issued prefetch as a read; one whose lookup found
        nothing read nothing, and counting it capped the turn after two (ntp_reevict
        capped=1) with its history still on the SSD."""

        def scheduler_pass(now, *, matched):
            returned = []

            def run():
                for req in self.ntp.tick(waiting_queue=[self.req], now=now):
                    returned.append(req.rid)
                    if self.cache.prefetch_gate_fn(self.handle, matched):
                        self.cache.ongoing.add(self.handle)
                    self.ntp.on_refetch_result(
                        req=req, issued=self.cache.has_ongoing_prefetch(self.handle)
                    )

            return returned, self.events(run)

        self.assertEqual(scheduler_pass(5.1, matched=KEY_TOKENS)[0], [])
        self.cache.resident = 0  # evicted while queued
        self.assertEqual(scheduler_pass(5.7, matched=0)[0], ["s-t1"])
        self.assertEqual((self.plan.lookups, self.plan.refetches), (1, 1))
        self.answer(start=0, hit=0)
        self.assertEqual((self.plan.lookups, self.plan.refetches), (0, 0))
        for now in (6.3, 6.9, 7.5):
            returned, found = scheduler_pass(now, matched=0)
            self.assertEqual((returned, reevictions(found)), ([], []))

    def test_a_match_past_the_key_is_covered_without_comparing_the_prompt(self):
        """Another session can hold more of the prompt than the plan's key is long. No
        key is left to compare then, and comparing raised in the arrival path."""
        found = self.gate(KEY_TOKENS + 500)
        self.assertEqual(gate_decisions(found), [("covered", "0")])
        self.assertIsNone(self.plan.shared_tokens)

    def test_a_hit_too_short_to_read_is_given_back_and_logged(self):
        """The cache revokes a hit under the read threshold, so the turn read nothing:
        kept as charged, such answers use the turn's reads up as empty ones did."""
        self.assertEqual(gate_decisions(self.gate(512)), [("read", "1")])
        ((kind, lookup),) = self.events(lambda: self.answer(start=512, hit=128))
        self.assertEqual(kind, "ntp_lookup")
        self.assertEqual(
            list(lookup.items()),
            [
                ("rid", "s-t1"),
                ("start", "512"),
                ("asked", str(KEY_TOKENS)),
                ("hit", "128"),
                ("refunded", "1"),
                ("lookups", "0"),
            ],
        )
        self.assertEqual((self.plan.absent_from, self.plan.free_lookups), (512, 1))

    def test_a_match_that_grew_past_the_mark_opens_what_lies_beyond_it(self):
        """A lookup from token 0 that finds nothing says the first page left storage, not the
        pages after it: once another session computed that head, the rest is asked for again."""
        self.cache.resident = 0
        self.assertEqual(gate_decisions(self.gate(0)), [("read", "1")])
        self.answer(start=0, hit=0)
        self.assertEqual(self.ntp.tick(waiting_queue=[self.req], now=5.2), [])
        self.cache.resident = 256
        self.assertEqual(self.ntp.tick(waiting_queue=[self.req], now=5.8), [self.req])
        self.assertEqual(gate_decisions(self.gate(256)), [("read", "1")])

        # The gate alone, for a turn the queue check has not probed since.
        self.commit("b-t0", "b-t1", now=0.0)
        handle = CacheRequestHandle(rid="b-t1", attempt_id=0)
        tail = array("q", range(self.SHARED))
        self.cache.prefetch_gate_prompt_fn(handle, 0, tail)
        self.cache.prefetch_lookup_fn(
            handle, storage_start=0, requested_tokens=KEY_TOKENS, hit_tokens=0
        )
        found = self.events(
            lambda: self.cache.prefetch_gate_prompt_fn(handle, 256, tail[256:])
        )
        self.assertEqual(gate_decisions(found), [("read", "1")])

    def test_a_free_room_retry_that_finds_nothing_gives_back_no_read(self):
        """The queue check re-issues a refused read without counting it: marked as counted, its
        empty answer would take a re-read off the turn's count that was never added."""
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY, True))
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_gate="on"
        )
        self.commit("s-t0", "s-t1", now=0.0)
        plan = self.ntp._plans["s-t1"]
        self.cache.resident = 0
        self.assertEqual(gate_decisions(self.gate(0)), [("read", "1")])
        self.answer(start=0, hit=KEY_TOKENS)
        self.ntp.on_read_refused(self.handle, tokens=KEY_TOKENS, cut=False, now=5.2)
        (again,) = self.ntp.tick(waiting_queue=[self.req], now=5.2 + NTP_ROOM_RETRY_S)
        self.assertEqual(gate_decisions(self.gate(0)), [("retry", "1")])
        self.cache.ongoing.add(self.handle)
        self.ntp.on_refetch_result(req=again, issued=True)
        self.answer(start=0, hit=0)
        self.assertEqual((plan.lookups, plan.refetches, plan.free_lookups), (1, 0, 0))

    def test_the_switch_binds_only_with_the_gate_in_mode_on(self):
        """Without the gate nothing counts lookups, and observe must never change what
        the engine reads: the cache then keeps asking the two-argument gate, or none."""
        for mode, gate, bound in (
            ("on", "on", True),
            ("on", "off", False),
            ("observe", "on", False),
        ):
            with self.subTest(mode=mode, gate=gate):
                cache = MeasuredCache()
                NextTurnPrefetcher(
                    mode=mode, tree_cache=cache, page_size=PAGE, arrival_gate=gate
                )
                self.assertEqual(cache.prefetch_gate_prompt_fn is not None, bound)
                self.assertEqual(cache.prefetch_lookup_fn is not None, bound)


class MeasuredTimingTestCase(PrefetcherTestCase):
    """Read timing expected_cost on a MeasuredCache with `switches` held for the test: one
    plan of BIG_KEY_TOKENS for s-t1, all but four pages of it evicted."""

    mode = "on"
    switches = ()

    def make_cache(self):
        return MeasuredCache()

    def setUp(self):
        hold_switches(self, *self.switches)
        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.missing = BIG_KEY_TOKENS - 4 * PAGE
        self.read_s = (
            L3LoadModel().seconds(own_pages=self.missing // PAGE, backlog_pages=0)
            + NTP_JIT_MARGIN_S
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.ntp = self.make_ntp()

    def make_ntp(self, **kwargs):
        kwargs.setdefault("read_timing", "expected_cost")
        return NextTurnPrefetcher(
            mode=self.mode,
            tree_cache=self.cache,
            page_size=PAGE,
            reply_key_source="hint",
            **kwargs,
        )

    def commit_entry(self, entry, *, session="s", now=0.0):
        table = {"session": session, "next_rid": f"{session}-t1", "gaps": [entry]}
        self.ntp.on_turn_committed(
            req=make_req(
                f"{session}-t0", kv_hints=hints("-", version="2", payload=table)
            ),
            key=key_of(BIG_KEY_TOKENS),
            rotation_tail_declined=False,
            now=now,
        )

    def tick(self, now, queue=()):
        return self.events(lambda: self.ntp.tick(waiting_queue=list(queue), now=now))

    def sharp_start(self, *, price=5e-5):
        """Rule 1's start for SHARP with an idle GPU, seconds after a commit at 0."""
        dist = GapDistribution.from_quantiles(
            p10=SHARP["p10"], p50=SHARP["p50"], p90=SHARP["p90"]
        )
        return read_start(
            remaining=remaining_gaps(
                support=dist.support(), idle_s=-ArrivalOffset().seconds
            ),
            read_s=self.read_s,
            wait_s=0.0,
            rent_per_s=price * self.missing,
        )

    def busy_spell(self, queued, *, until=20.0):
        """Passes 0.25 s apart up to `until`, each admitting one new turn while `queued`
        stay in the queue; then `queued` are admitted too and the queue is empty. Returns
        the events logged."""
        found = []
        now = 0.0
        while now < until:
            turn = make_req(f"busy-{now}", entered=now)
            found += self.tick(now, queue=queued + [turn])
            turn.time_stats.forward_entry_time = now + 0.1
            now += 0.25
        for req in queued:
            req.time_stats.forward_entry_time = now
        self.cache.ongoing.clear()
        return found + self.tick(now)


class TestLatenessWeight(MeasuredTimingTestCase):
    """SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT: a second a turn waits on its read costs c,
    the share of admissions that left no ready turn waiting."""

    switches = ((envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT, True),)

    def test_after_a_saturated_minute_a_read_is_not_started_where_an_idle_gpu_starts_it(
        self,
    ):
        """Once the queue has drained no wait hides the read, so with the stall at full
        weight the read starts as for an idle GPU; but ready turns were just being passed
        to the GPU in this turn's place, and a late read still delays nobody. ec_start
        carries the weight used, at its end."""
        start = self.sharp_start()
        self.commit_entry(SHARP)
        found = self.busy_spell([make_req(f"q{i}-t1", entered=0.0) for i in range(3)])
        now = 20.25
        while now < start + 3.75:
            found += self.tick(now)
            now += 0.25
        self.assertEqual(self.cache.prefetches, [])
        self.assertLess(self.ntp._prices.c(start), 0.2)
        weights = [fields for kind, fields in found if kind == "ec_start"]
        self.assertEqual(list(weights[-1])[-1], "c")
        self.assertLess(float(weights[-1]["c"]), 0.2)

    def test_a_minute_of_turns_waiting_on_their_own_reads_leaves_the_start_where_it_was(
        self,
    ):
        """The same admissions with the queued turns held up by the SSD, not the GPU: no
        ready turn was passed over, c stays 1, and the read starts where rule 1 puts it."""
        start = self.sharp_start()
        self.commit_entry(SHARP)
        blocked = [make_req(f"b{i}-t1", entered=0.0) for i in range(3)]
        for req in blocked:
            self.cache.ongoing.add(req.cache_request_handle)
        self.busy_spell(blocked)
        self.assertEqual(self.ntp._prices.c(start), 1.0)
        self.tick(start - 0.3)
        self.assertEqual(self.cache.prefetches, [])
        self.tick(start + 0.01)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])

    def test_the_weight_acts_only_on_reads_staged_by_expected_cost(self):
        """Just-in-time and arrival timing weigh no stall against rent; with nothing to
        act on the switch is off, and measures nothing."""
        for timing, weighed in (
            ("expected_cost", True),
            ("arrival", False),
            ("just_in_time", False),
        ):
            with self.subTest(timing=timing):
                ntp = self.make_ntp(read_timing=timing)
                self.assertEqual(ntp._lateness_weight, weighed)
                self.assertEqual(ntp._prices is not None, weighed)


class TestMeasuredRoomPrice(MeasuredTimingTestCase):
    """SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES under read timing expected_cost: the rent
    of a staged read and the harm of a refused one."""

    switches = ((envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, True),)

    def ec_start(self, now):
        found = self.tick(now)
        (start,) = [fields for kind, fields in found if kind == "ec_start"]
        return (start["decision"], start["price"])

    def test_a_staging_pays_for_the_holder_it_would_evict_next(self):
        """The frontier's price answers evictions already made and is zero until one
        happens, while a staged pin lasts until its turn arrives: priced by the frontier
        alone, every plan would stage at once into a full room. A queued turn's history
        is the dearest victim: it is needed within its wait."""
        registry = self.cache.tree_core.eviction_strategy.registry
        registry.register(1, 10.0)
        registry.register(2, Queued(since=0.0))
        per_token = L3LoadModel().read_s_per_page / PAGE
        for victim, decision, price in (
            (None, "now", 0.0),
            (SimpleNamespace(id=1), "wait", per_token / 10.0),
            (SimpleNamespace(id=2), "wait", per_token / 0.5),
            (SimpleNamespace(id=3), "now", 0.0),
        ):
            with self.subTest(victim=victim):
                self.ntp = self.make_ntp()
                self.cache.prefetches.clear()
                self.cache.ongoing.clear()
                self.cache.victim = victim
                self.commit_entry(SHARP)
                self.assertEqual(self.ec_start(0.01), (decision, f"{price:.3g}"))
                self.assertEqual(len(self.cache.prefetches), int(decision == "now"))

    def test_a_staging_pays_the_frontier_price_when_no_victim_is_dearer(self):
        """With the frontier dropped from a staging's price, a room that just evicted a holder
        due back in 10 s reads as free and the read starts at once."""
        self.cache.tree_core.eviction_strategy.registry.register(1, 10.0)
        self.cache.tree_core.host_eviction_observer(SimpleNamespace(id=1), 640)
        self.commit_entry(SHARP)
        per_token = L3LoadModel().read_s_per_page / PAGE
        self.assertEqual(self.ec_start(0.01), ("wait", f"{per_token / 10.0:.3g}"))

    def test_under_return_chance_a_staging_pays_for_the_session_of_the_next_victim(
        self,
    ):
        """The arms that stage by expected cost evict by return chance: priced only under
        next_use, their stagings would pay for no victim at all."""
        strategy = ReturnChanceStrategy(horizon_s=30.0, clock=lambda: 0.0)
        strategy.registry.register(
            1, ReturnForecast(dist=GapDistribution.point(10.0), idle_from=0.0)
        )
        self.cache.tree_core.eviction_strategy = strategy
        self.cache.victim = SimpleNamespace(id=1, parent=None, children={})
        self.ntp = self.make_ntp()
        self.commit_entry(SHARP)
        per_token = L3LoadModel().read_s_per_page / PAGE
        self.assertEqual(self.ec_start(0.01), ("wait", f"{per_token / 10.0:.3g}"))

    def test_the_next_victim_is_priced_again_only_every_half_second_and_within_the_ceiling(
        self,
    ):
        """Pricing costs one priority per evictable host leaf, so not every pass; and a victim
        due at once behind 13 late reads is worth 1.09e-3, above the ceiling."""
        registry = self.cache.tree_core.eviction_strategy.registry
        for node_id, next_use in ((1, 10.0), (2, 100.0), (3, 0.0)):
            registry.register(node_id, next_use)
        meter, load = self.ntp._meter, L3LoadModel()
        per_token = load.read_s_per_page / PAGE
        self.cache.victim = SimpleNamespace(id=1)
        self.assertAlmostEqual(
            meter.next_victim_price(load=load, now=0.0), per_token / 10.0
        )
        self.cache.victim = SimpleNamespace(id=2)
        self.assertAlmostEqual(
            meter.next_victim_price(load=load, now=0.4), per_token / 10.0
        )
        self.assertAlmostEqual(
            meter.next_victim_price(load=load, now=0.6), per_token / 100.0
        )
        for late in range(13):
            self.cache.read_scheduler.park(
                handle=CacheRequestHandle(f"late{late}", 0),
                operation=None,
                pages=1,
                now=0.0,
            )
        self.cache.victim = SimpleNamespace(id=3)
        self.assertEqual(meter.next_victim_price(load=load, now=1.2), LAM_CEILING)

    def test_the_harm_of_a_refused_read_is_its_recompute_at_the_fitted_cost_times_mu(
        self,
    ):
        """What a staged pin costs an arrived turn refused room: the prefill of the tokens
        the pin held, at the price of GPU time. The constant it replaces is Triton's at
        one length, times a queue count that is 0 or the whole queue."""
        # FA3's cost (campaigns 33 and 34): 20K tokens from scratch take 0.54 s.
        self.ntp._meter.prefill_cost = PrefillCostModel(seed_a=12.8e-6, seed_b=0.71e-9)
        self.commit_entry(WIDE)
        waiting = [make_req("a-t1", entered=0.0), make_req("b-t1", entered=0.0)]
        self.cache.pinned = {self.handle: 20_000}
        for step in range(10):
            self.tick(0.5 * step, queue=waiting)
        self.cache.refusals = [(waiting[0].cache_request_handle, 30_000)]
        found = self.tick(5.2, queue=waiting)
        (price,) = [fields for kind, fields in found if kind == "ec_price"]
        mu = self.ntp._prices.mu(5.2)
        self.assertGreater(mu, 2.5)
        prefill_s = 12.8e-6 * 20_000 + 0.71e-9 * 20_000**2
        self.assertEqual(int(price["harm_ms"]), round(mu * prefill_s * 1000))
        with self.assertRaises(ValueError):
            self.make_ntp(price_learning="harm")


class TestPricesLogOnly(MeasuredTimingTestCase):
    """SGLANG_LOG_HICACHE_PRICES_INTERVAL alone, under expected-cost timing with the
    arrival hold: one script run without it and with it."""

    def script(self, prices_interval_s):
        """A saturated spell (12 ready turns queued for 19 s while 190 others are admitted), a
        held read, a read staged once the queue drained, and a turn refused room under its
        pin. Returns the lines logged and the answers given."""
        stack = hold_switches(
            self, (envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL, prices_interval_s)
        )
        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.ntp = self.make_ntp(arrival_hold="on")
        stack.close()
        self.commit_entry(SHARP)
        stuck = [make_req(f"q{i}", entered=0.0) for i in range(12)]
        found, answers = [], []
        for step in range(190):
            turn = make_req(f"busy-{step}", entered=0.1 * step)
            found += self.tick(0.1 * step, queue=stuck + [turn])
            turn.time_stats.forward_entry_time = 0.1 * step + 0.05
        found += self.tick(19.0, queue=stuck)
        last = stuck[-1].cache_request_handle
        held = ParkedRead(handle=last, operation=None, pages=10, parked_at=0.0)
        found += self.events(
            lambda: answers.append(self.ntp.read_not_before(last, held, 0))
        )
        for req in stuck:
            req.time_stats.forward_entry_time = 19.05
        now = 19.25
        while now < self.sharp_start() + 1.0:
            found += self.tick(now)
            answers.append((now, [h for h, *_ in self.cache.prefetches]))
            now += 0.25
        self.cache.land(self.handle, self.missing)
        found += self.tick(now)
        refused = make_req("r-t1", entered=now)
        self.cache.refusals = [(refused.cache_request_handle, 30_000)]
        arrived = make_req("s-t1", entered=now)
        for step in range(1, 25):
            found += self.tick(now + 0.25 * step, queue=[refused, arrived])
        arrived.time_stats.forward_entry_time = now + 6.2
        found += self.tick(now + 6.25, queue=[refused])
        return found, answers

    def test_the_prices_log_alone_changes_no_decision_and_no_other_line(self):
        """With only the log on c, mu and the admission error exist to be read: no start, hold
        or harm charge may read them, and every line but ntp_admit keeps its fields."""
        off, off_answers = self.script(0.0)
        on, on_answers = self.script(5.0)
        self.assertEqual(on_answers, off_answers)

        def decisions(found):
            lines = []
            for kind, fields in found:
                if kind == "ntp_admit":
                    fields = {name: fields[name] for name in ADMIT_FIELDS}
                if kind != "prices":
                    lines.append((kind, fields))
            return lines

        self.assertEqual(decisions(on), decisions(off))
        by_kind = {}
        for kind, fields in off:
            by_kind.setdefault(kind, []).append(fields)
        for kind, names in (("ec_start", EC_START_FIELDS), ("ntp_hold", HOLD_FIELDS)):
            self.assertEqual(
                {tuple(fields) for fields in by_kind[kind]}, {tuple(names)}
            )
        # The script reached what it compares: the hold held, the read was staged where
        # an idle GPU stages it, the refusal was charged and the turn admitted.
        self.assertGreater(off_answers[0], 19.0)
        self.assertEqual(off_answers[-1][1], [self.handle])
        self.assertIn("now", [fields["decision"] for fields in by_kind["ec_start"]])
        self.assertGreater(max(int(f["harm_ms"]) for f in by_kind["ec_price"]), 0)
        self.assertEqual(len(by_kind["ntp_admit"]), 1)
        self.assertIn("prices", [kind for kind, _ in on])


def measure_lags(cache, *, lookup_s=0.02, fixed_s=0.01):
    """Enough park and landing samples for the read path's lags to be published."""
    for _ in range(LAG_MIN_SAMPLES):
        cache.read_lags.lookup.observe(lookup_s)
        cache.read_lags.fixed.observe(fixed_s)


def hold_writes(cache, *, since, usual_s=4.0):
    """Enough write holds of usual_s for their length to be published, and one on since
    `since`: timed reads are write-held."""
    lags = cache.read_lags
    for hold in range(LAG_MIN_SAMPLES):
        lags.note_write_hold(held=True, now=since - 100.0 * (hold + 1))
        lags.note_write_hold(held=False, now=since - 100.0 * (hold + 1) + usual_s)
    lags.note_write_hold(held=True, now=since)


class TestReadLine(PrefetcherTestCase):
    """SGLANG_HICACHE_NEXT_TURN_READ_LINE under just-in-time timing and the arrival hold."""

    mode = "on"

    def make_cache(self):
        return MeasuredCache()

    def setUp(self):
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE, True))
        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        self.ntp = self.make_ntp()
        # s-t1 is due at 50.1 s; one plan's read is due before it and one after.
        self.commit_gaps(e=30.0, l=80.0, s=50.0)
        srs = self.cache.read_scheduler
        srs.park(
            handle=CacheRequestHandle("e-t1", 0), operation=None, pages=400, now=0.0
        )
        srs.park(
            handle=CacheRequestHandle("l-t1", 0), operation=None, pages=4000, now=0.0
        )
        own_pages = (KEY_TOKENS - 4 * PAGE) // PAGE
        # The start without the switch: admission minus the load of the read alone, minus
        # the margin.
        self.start_alone = (
            ArrivalOffset().predict_arrival(t_commit=0.0, gap_s=50.0)
            - L3LoadModel().seconds(own_pages=own_pages, backlog_pages=0)
            - NTP_JIT_MARGIN_S
        )

    def make_ntp(self, **kwargs):
        kwargs.setdefault("read_timing", "just_in_time")
        return NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, **kwargs
        )

    def commit_gaps(self, **gaps):
        """A plan of KEY_TOKENS per session, its next turn <session>-t1 due after the gap."""
        for session, gap_s in gaps.items():
            self.ntp.on_turn_committed(
                req=make_req(
                    f"{session}-t0",
                    kv_hints=hints(
                        "-",
                        payload={
                            "session": session,
                            "next_rid": f"{session}-t1",
                            "next_gap_s": gap_s,
                        },
                    ),
                ),
                key=key_of(KEY_TOKENS),
                rotation_tail_declined=False,
                now=0.0,
            )

    def staged(self):
        return self.handle in [handle for handle, *_ in self.cache.prefetches]

    def test_a_pending_read_counts_the_line_and_a_timed_one_the_write_hold_too(self):
        """ntp_due says what was counted: a timed read waits for the parked reads due no later
        (400 pages) and for the write hold (2.8 s left); a read needed now for neither."""
        self.ntp = self.make_ntp(read_timing="horizon")
        self.commit_gaps(e=30.0, l=80.0, s=50.0, o=0.0)
        hold_writes(self.cache, since=-1.0)
        found = self.events(lambda: self.ntp.tick(waiting_queue=[], now=0.2))
        due = {
            fields["rid"]: (fields["backlog_pages"], fields["load_ms"])
            for kind, fields in found
            if kind == "ntp_due"
        }
        seeds_s = L3LoadModel().seconds(own_pages=12, backlog_pages=0)
        self.assertEqual(
            (due["s-t1"], due["o-t1"]),
            (
                ("400", str(round((seeds_s + 400 * 0.0025 + 2.8) * 1000))),
                ("0", str(round(seeds_s * 1000))),
            ),
        )

    def test_the_queue_check_counts_the_line_once_a_turn_and_not_the_turns_own_read(
        self,
    ):
        """Only logged under the deadline order, the estimate must not walk the parked reads per
        queued turn per pass; a turn's own parked read stands in that line, not behind it."""
        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = 4 * PAGE
        self.ntp = self.make_ntp()
        self.commit_gaps(s=5.0)
        srs = self.cache.read_scheduler
        srs.park(
            handle=CacheRequestHandle("urgent", 0), operation=None, pages=200, now=0.0
        )
        srs.park(handle=self.handle, operation=None, pages=12, now=0.0)
        self.cache.ongoing.add(self.handle)
        asked = []
        line_pages = self.cache.storage_read_line_pages
        pages_ahead = self.cache.parked_read_pages_ahead
        self.cache.storage_read_line_pages = lambda due: (
            asked.append(due) or line_pages(due)
        )
        self.cache.parked_read_pages_ahead = lambda own: (
            asked.append(own) or pages_ahead(own)
        )
        req = make_req("s-t1", entered=1.0)
        found = []
        for step in range(10):
            found += self.events(
                lambda: self.ntp.tick(waiting_queue=[req], now=1.0 + 0.1 * step)
            )
        (due,) = [fields for kind, fields in found if kind == "ntp_due"]
        load_s = L3LoadModel().seconds(own_pages=12, backlog_pages=200)
        self.assertEqual(
            (due["case"], due["backlog_pages"], due["load_ms"]),
            ("2", "200", str(round(load_s * 1000))),
        )
        self.assertEqual(asked, [self.handle])

    def test_a_parked_read_due_earlier_moves_a_start_earlier_and_one_due_later_does_not(
        self,
    ):
        """Dispatch serves reads by deadline, so a read staged now waits for the parked
        ones due before it. Counting launched reads only, a planned 8 s read took 30 to
        52 s (campaign 34); counting the later ones too would start it 10 s early here."""
        line_s = 400 * L3LoadModel().backlog_s_per_page
        self.ntp.tick(waiting_queue=[], now=self.start_alone - line_s - 0.2)
        self.assertFalse(self.staged())
        self.ntp.tick(waiting_queue=[], now=self.start_alone - line_s + 0.2)
        self.assertTrue(self.staged())

    def test_once_ready_turns_mostly_wait_just_in_time_leaves_the_line_out(self):
        """This timing weighs no cost against an early start: with the GPU saturated a
        late read delays nobody, and the line (5 s of it here) would only pin more history
        early. The start is then the one without the switch, on the seeded lags."""
        self.cache.read_scheduler.park(
            handle=CacheRequestHandle("urgent", 0), operation=None, pages=2000, now=0.0
        )
        measure_lags(self.cache)
        self.ntp._prices.observe_admissions(count=100, ready_left=True, now=40.0)
        self.ntp.tick(waiting_queue=[], now=self.start_alone - 0.2)
        self.assertFalse(self.staged())
        self.ntp.tick(waiting_queue=[], now=self.start_alone + 0.2)
        self.assertTrue(self.staged())

    def test_measured_lags_replace_the_seeds_only_where_the_line_is_counted_in_pages(
        self,
    ):
        """The seeds stand for a busy read thread: 0.9 s with one read ahead folded in.
        Kept beside a line counted in pages they count that read twice; a negative
        residual is no lag; and before both lags were measured the seeds must stay."""
        self.cache.read_lags.lookup.observe(0.02)
        self.ntp.tick(waiting_queue=[], now=0.0)
        self.assertIs(self.ntp._line_load, self.ntp._load)
        measure_lags(self.cache, lookup_s=0.02, fixed_s=-0.03)
        self.ntp.tick(waiting_queue=[], now=0.1)
        line = self.ntp._line_load
        self.assertEqual(
            (line.alloc_lag_s, line.drain_lag_s, line.read_fixed_s), (0.02, 0.0, 0.0)
        )
        self.assertEqual(self.ntp._load, L3LoadModel())

    def test_a_held_read_starts_by_its_pages_and_the_measured_lags(self):
        """The hold's page count already holds every read ahead, so on the seeds a held
        read started 0.9 s early; with the switch off the hold keeps the seeds."""
        queue = [make_req(f"y{i}", entered=2.0) for i in range(5)]
        queue.append(make_req("s-t1", entered=2.0))
        parked = ParkedRead(handle=self.handle, operation=None, pages=10, parked_at=0.0)
        for switch, lag_s in ((True, 0.02 + 0.01), (False, 0.5 + 0.044 + 0.36)):
            with self.subTest(switch=switch):
                stack = hold_switches(
                    self, (envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE, switch)
                )
                self.ntp = self.make_ntp(arrival_hold="on")
                stack.close()
                measure_lags(self.cache)
                # 30 admissions give the queue a measured drain rate.
                self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)
                self.ntp.tick(waiting_queue=queue, now=2.0)
                wait = residual_wait(position=5, rate=self.ntp._last_rate)
                load_s = lag_s + 50 * L3LoadModel().read_s_per_page
                self.assertAlmostEqual(
                    self.ntp.read_not_before(self.handle, parked, 40),
                    2.0 + wait - load_s - NTP_HOLD_MARGIN_S,
                )

    def test_the_switch_needs_parked_reads_and_under_expected_cost_the_lateness_weight(
        self,
    ):
        """Without the deadline order nothing parks, so the switch is off. With the stall
        at full weight the line turned 349 of 1,525 late-run decisions of campaign 34
        into reads started at once: under expected-cost staging it is refused alone."""
        fifo = NextTurnPrefetcher(mode="on", tree_cache=ActingCache(), page_size=PAGE)
        self.assertFalse(fifo._read_line)
        self.assertTrue(self.make_ntp(read_timing="arrival")._read_line)
        with self.assertRaises(ValueError):
            self.make_ntp(read_timing="expected_cost")
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT, True))
        self.assertTrue(self.make_ntp(read_timing="expected_cost")._read_line)


class TestReadLineByExpectedCost(MeasuredTimingTestCase):
    """The read line under read timing expected_cost, which needs the lateness weight."""

    switches = (
        (envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT, True),
        (envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE, True),
    )
    # A staged read due 20 s before the plan's, parked: 5 s of line.
    LINE_PAGES = 2000

    def setUp(self):
        super().setUp()
        # Session e is due at 30.1 s; all of its short key is in memory, so the ticks
        # stage nothing for it.
        self.ntp.on_turn_committed(
            req=make_req("e-t0", kv_hints=hints("e-t1", 30.0)),
            key=key_of(4 * PAGE),
            rotation_tail_declined=False,
            now=0.0,
        )
        self.cache.read_scheduler.park(
            handle=CacheRequestHandle("e-t1", 0),
            operation=None,
            pages=self.LINE_PAGES,
            now=0.0,
        )
        self.line_s = self.LINE_PAGES * L3LoadModel().backlog_s_per_page
        self.start = self.sharp_start()

    def test_with_an_idle_gpu_a_line_ahead_starts_a_staged_read_earlier(self):
        """Without the line the staged read lands a line later than planned. A longer read
        also moves the start rule 1 picks, so the start is earlier by the line and more;
        ec_start says how many pages were counted, at its end."""
        self.commit_entry(SHARP)
        found = self.tick(self.start - self.line_s - 4.0)
        self.assertEqual(self.cache.prefetches, [])
        now = self.start - self.line_s - 4.0
        while now < self.start - self.line_s:
            now += 0.25
            found += self.tick(now)
        self.assertEqual([h for h, *_ in self.cache.prefetches], [self.handle])
        starts = [fields for kind, fields in found if kind == "ec_start"]
        self.assertEqual(list(starts[-1])[-2:], ["c", "line_pages"])
        self.assertEqual(starts[-1]["line_pages"], str(self.LINE_PAGES))

    def test_after_a_saturated_spell_the_line_starts_nothing_early(self):
        """The line makes the read longer than any queue wait, which at full weight reads
        as a stall worth pinning for; a late read delays nobody while ready turns wait."""
        self.commit_entry(SHARP)
        self.busy_spell([make_req(f"q{i}-t1", entered=0.0) for i in range(3)])
        now = 20.25
        while now < self.start - 2.5:
            self.tick(now)
            now += 0.25
        self.assertLess(self.ntp._prices.c(now), 0.2)
        self.assertEqual(self.cache.prefetches, [])


class TestHoldQuantile(PrefetcherTestCase):
    """SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE: a held read lands at a quantile of its
    turn's admission time, by the measured error of the admission estimate."""

    mode = "on"

    def make_cache(self):
        return MeasuredCache()

    def setUp(self):
        hold_switches(
            self,
            (envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, True),
            (envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE, True),
        )
        self.cache = self.make_cache()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_hold="on"
        )
        self.handle = CacheRequestHandle(rid="s-t1", attempt_id=0)
        # 1,000 pages: at a room price of 1e-6 the read's rent is 0.064 a second.
        self.parked = ParkedRead(
            handle=self.handle, operation=None, pages=1000, parked_at=0.0
        )
        queue = [make_req(f"y{i}", entered=2.0) for i in range(5)]
        queue.append(make_req("s-t1", entered=2.0))
        # 30 admissions in the minute: at position 5 the wait predicted is 11 s.
        self.admit_through_queue([f"x{i}" for i in range(30)], now=1.0)
        self.ntp.tick(waiting_queue=queue, now=2.0)
        # Admissions against a wait predicted in that bin: 5 s early to 4.9 s late.
        for k in range(100):
            self.ntp._meter.admission_error.observe(
                predicted_wait_s=11.0, wait_s=11.0 + (k - 50) / 10
            )
        wait = residual_wait(position=5, rate=self.ntp._last_rate)
        load = L3LoadModel().seconds(own_pages=1000, backlog_pages=0)
        self.fixed_start = 2.0 + wait - load - NTP_HOLD_MARGIN_S

    def not_before(self):
        return self.ntp.read_not_before(self.handle, self.parked, 0)

    def test_free_room_lands_a_held_read_early_and_a_saturated_gpu_at_the_median(self):
        """q = rent / (c + rent). With room free the read lands as early as the measured
        error asks, its 5 % quantile: 4.5 s before the estimate here, where the fixed
        margin gave 2 s. With the GPU saturated a late read delays nobody, so it lands
        later, but never past the median of the admission time."""
        self.assertAlmostEqual(self.not_before(), self.fixed_start - 2.5)
        board = self.ntp._prices
        board.observe_eviction(marginal_index=1e-6, tokens=64, now=2.0)
        self.assertAlmostEqual(self.not_before(), self.fixed_start - 2.4)
        board.observe_admissions(count=10_000, ready_left=True, now=2.0)
        self.assertAlmostEqual(self.not_before(), self.fixed_start + 2.0)

    def test_ntp_hold_gains_the_margin_and_the_quantile_at_its_end_once(self):
        """Dispatch asks every pass; the scripts read the line by position."""
        found = self.events(lambda: [self.not_before() for _ in range(3)])
        (hold,) = [fields for kind, fields in found if kind == "ntp_hold"]
        self.assertEqual(
            list(hold),
            ["rid", "pos", "pages", "wait_ms", "load_ms", "hold_ms", "margin_ms", "q"],
        )
        self.assertEqual((hold["margin_ms"], hold["q"]), ("4500", "0.05"))

    def test_held_reads_still_all_go_when_the_margin_starts_them_past_the_estimate(
        self,
    ):
        """Admissions that ran 3 s late at every predicted wait and a saturated GPU give a
        margin of minus 3 s. A parked read blocks its turn, so each must still start: at
        the head it goes at once, whatever the margin."""
        tracker = self.ntp._meter.admission_error
        for predicted in (1.0, 3.0, 7.0, 15.0, 30.0):
            for _ in range(40):
                tracker.observe(predicted_wait_s=predicted, wait_s=predicted + 3.0)
        board = self.ntp._prices
        board.observe_eviction(marginal_index=1e-6, tokens=64, now=2.0)
        board.observe_admissions(count=10_000, ready_left=True, now=2.0)
        srs = self.cache.read_scheduler
        queue = [make_req(f"q{i}", entered=2.0) for i in range(12)]
        for req in queue:
            srs.park(
                handle=req.cache_request_handle, operation=None, pages=1000, now=2.0
            )
        started, landed = {}, {}
        now = 2.0
        while queue and now < 200.0:
            self.ntp.tick(waiting_queue=queue, now=now)
            for decision in srs.dispatch(
                now=now,
                deadline_of=self.ntp.read_deadline,
                in_flight_pages=0,
                held_tokens=0,
                not_before_of=self.ntp.read_not_before,
            ):
                started[decision.read.handle.rid] = now
                landed[decision.read.handle.rid] = now + 1.0
            now += 0.5
            # One admission per pass at most, and only once its read has landed.
            ready = [req for req in queue if landed.get(req.rid, float("inf")) <= now]
            if ready:
                ready[0].time_stats.forward_entry_time = now
                queue = [req for req in queue if req is not ready[0]]
        self.assertEqual((queue, srs.parked_count()), ([], 0))
        self.assertGreater(started["q11"] - started["q0"], 3.0)

    def test_the_switch_needs_the_hold_and_measured_prices(self):
        """Without the hold nothing is held, so the switch is off. At the knob's 5e-5 a
        20K-token read would land at the median of its admission, late half the time:
        on that price the switch is refused."""
        unheld = NextTurnPrefetcher(
            mode="on", tree_cache=MeasuredCache(), page_size=PAGE
        )
        self.assertFalse(unheld._hold_quantile)
        hold_switches(self, (envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES, False))
        with self.assertRaises(ValueError):
            NextTurnPrefetcher(
                mode="on", tree_cache=MeasuredCache(), page_size=PAGE, arrival_hold="on"
            )


# What both recompute switches need beside the arrival gate.
RECOMPUTE_NEEDS = (
    envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES,
    envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE,
    envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES,
)
# 126 pages: read back in 30 ms of lags + 315 ms; on FA3 recomputed from scratch in 149 ms.
SHORT_HISTORY = 126 * PAGE


class RecomputeTestCase(PrefetcherTestCase):
    """The arrival gate on a MeasuredCache with `switch` and what it needs held: FA3's
    prefill cost (campaigns 33 and 34), the read path's lags measured at 20 + 10 ms, and
    nothing that the ticks would stage."""

    mode = "on"
    # The switch under test, in a tuple: an EnvField cannot be a class attribute.
    switched = ()

    def make_cache(self):
        return MeasuredCache()

    def setUp(self):
        (self.switch,) = self.switched
        hold_switches(
            self, *((need, True) for need in RECOMPUTE_NEEDS), (self.switch, True)
        )
        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = 10**9
        self.ntp = self.make_ntp()

    def make_ntp(self):
        ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_gate="on"
        )
        ntp._meter.prefill_cost = PrefillCostModel(seed_a=12.8e-6, seed_b=0.71e-9)
        measure_lags(self.cache)
        ntp.tick(waiting_queue=[], now=0.0)
        return ntp

    def plan_turn(self, session, tokens=SHORT_HISTORY):
        """A plan for <session>-t1 whose key is `tokens` long; returns the turn's handle."""
        payload = {"session": session, "next_rid": f"{session}-t1", "next_gap_s": 5.0}
        self.ntp.on_turn_committed(
            req=make_req(f"{session}-t0", kv_hints=hints("-", payload=payload)),
            key=key_of(tokens),
            rotation_tail_declined=False,
            now=0.0,
        )
        return CacheRequestHandle(f"{session}-t1", 0)


class TestRecomputeByPrice(RecomputeTestCase):
    """SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE: read-or-recompute at the arrival gate."""

    switched = (envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE,)

    def test_both_recompute_switches_need_the_gate_and_refuse_what_they_cannot_price_on(
        self,
    ):
        """Without the gate nothing can decline a read, so the switches are off. On the
        load model's seeds a recompute beats a read at every size, on the constant prefill
        cost at none, and without the skip a recomputed history is written to the SSD
        again: with any of the three missing, each switch is refused."""
        for switch in (
            envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE,
            envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,
        ):
            held = hold_switches(self, (self.switch, False), (switch, True))
            ungated = NextTurnPrefetcher(
                mode="on", tree_cache=MeasuredCache(), page_size=PAGE
            )
            self.assertFalse(
                ungated._recompute_by_price or ungated._recompute_when_starved
            )
            for need in RECOMPUTE_NEEDS:
                with self.subTest(switch=switch.name, without=need.name):
                    stack = hold_switches(self, (need, False))
                    with self.assertRaises(ValueError):
                        NextTurnPrefetcher(
                            mode="on",
                            tree_cache=MeasuredCache(),
                            page_size=PAGE,
                            arrival_gate="on",
                        )
                    stack.close()
            held.close()

    def gate(self, handle, *, history=SHORT_HISTORY):
        """The turn's own prefetch at the gate with nothing of its prompt in memory: the
        prompt repeats `history` tokens of the key and adds 512. Returns the ntp_gate line."""
        tail = array("q", [*range(history), *range(900_000, 900_512)])
        found = self.events(lambda: self.cache.prefetch_gate_prompt_fn(handle, 0, tail))
        (line,) = [fields for kind, fields in found if kind == "ntp_gate"]
        return line

    def test_alone_at_an_idle_gpu_a_short_history_is_recomputed_and_a_long_one_read(
        self,
    ):
        """FA3 prefill grows with the square of the length while a read grows with the
        pages: 8K tokens are computed in 149 ms against 345 ms to read, 60K in 3.3 s
        against 2.4 s. The priced answer spends no lookup and says what it weighed."""
        short = self.gate(self.plan_turn("s"))
        self.assertEqual((short["decision"], short["lookups"]), ("recompute", "0"))
        self.assertEqual(list(short)[-4:], ["rec_ms", "read_ms", "mu", "c"])
        self.assertEqual(
            (short["rec_ms"], short["read_ms"], short["mu"], short["c"]),
            ("149", "345", "1.00", "1.00"),
        )
        long = self.gate(self.plan_turn("l", 938 * PAGE), history=938 * PAGE)
        self.assertEqual((long["decision"], long["lookups"]), ("read", "1"))
        self.assertGreater(int(long["rec_ms"]), int(long["read_ms"]))

    def test_a_reply_the_prompt_does_not_repeat_is_not_priced_as_history_to_restore(
        self,
    ):
        """Without free empty lookups the plan's key still ends with the reply as generated.
        Neither a read nor a recompute restores text the prompt does not have: priced on
        the key's 126 pages instead of the prompt's 102, both sides come out too long."""
        handle = self.plan_turn("s")
        tail = array("q", [*range(100 * PAGE), *range(900_000, 900_128)])
        found = self.events(lambda: self.cache.prefetch_gate_prompt_fn(handle, 0, tail))
        (line,) = [fields for kind, fields in found if kind == "ntp_gate"]
        self.assertEqual(
            (line["decision"], line["rec_ms"], line["read_ms"]),
            ("recompute", "114", "285"),
        )

    def test_behind_another_turn_or_once_it_has_waited_the_turn_reads(self):
        """Behind a queue the read is hidden by the wait and GPU time is dear. A recompute
        answer is not final: the turn is asked again while it queues, and once it has
        itself waited for the GPU it reads, as it would without the switch."""
        handle = self.plan_turn("s")
        req = make_req("s-t1", entered=1.0)
        self.ntp.tick(waiting_queue=[req], now=1.0)
        self.assertEqual(self.gate(handle)["decision"], "recompute")
        self.ntp.tick(waiting_queue=[req], now=1.6)
        waited = self.gate(handle)
        self.assertEqual((waited["decision"], waited["lookups"]), ("read", "1"))
        self.assertNotIn("rec_ms", waited)

        other = self.plan_turn("o")
        self.ntp.tick(waiting_queue=[make_req("busy", entered=2.0)], now=2.0)
        self.assertEqual(self.gate(other)["decision"], "read")

    def test_the_turns_that_arrive_during_a_recompute_are_priced_into_it(self):
        """At 2.7 admissions a second a 0.7 s recompute delays the turn or two that arrive while
        it runs: 24K tokens that an idle engine recomputes are read."""
        handle = self.plan_turn("s", 375 * PAGE)
        self.assertEqual(self.gate(handle, history=375 * PAGE)["decision"], "recompute")
        self.admit_through_queue([f"x{i}" for i in range(160)], now=1.0)
        busy = self.gate(handle, history=375 * PAGE)
        self.assertEqual(busy["decision"], "read")
        self.assertGreater(float(busy["mu"]), 1.9)

    def test_the_read_is_priced_behind_the_line_ahead_and_not_behind_the_write_hold(
        self,
    ):
        """A read needed at arrival waits for the reads needed now parked before it, never for
        the write hold: 60K tokens read at an idle thread are recomputed behind 5 s of reads."""
        hold_writes(self.cache, since=0.0)
        self.assertEqual(self.gate(self.plan_turn("s"))["read_ms"], "345")
        long_history = 938 * PAGE
        idle = self.gate(self.plan_turn("l", long_history), history=long_history)
        self.assertEqual(idle["decision"], "read")
        self.cache.read_scheduler.park(
            handle=CacheRequestHandle("urgent", 0), operation=None, pages=2000, now=0.0
        )
        behind = self.gate(self.plan_turn("m", long_history), history=long_history)
        self.assertEqual(
            (behind["decision"], behind["read_ms"]),
            ("recompute", str(round((0.03 + (938 + 2000) * 0.0025) * 1000))),
        )

    def test_a_falling_c_unmeasured_lags_and_the_staged_lookup_are_not_recomputed(self):
        """Below c = 0.8 the rule was wrong on a quarter to half of its recomputes: a queue
        formed behind them. On the load model's seeds a recompute would win at every size.
        And the staged lookup comes before the turn exists: priced, a short history would
        never be staged."""
        handle = self.plan_turn("s")
        self.ntp._prices.observe_admissions(count=10, ready_left=True, now=0.0)
        falling = self.gate(handle)
        self.assertEqual((falling["decision"], falling["c"]), ("read", "0.09"))

        self.cache = self.make_cache()
        self.ntp = NextTurnPrefetcher(
            mode=self.mode, tree_cache=self.cache, page_size=PAGE, arrival_gate="on"
        )
        self.ntp._meter.prefill_cost = PrefillCostModel(seed_a=12.8e-6, seed_b=0.71e-9)
        unmeasured = self.gate(self.plan_turn("u"))
        self.assertEqual(unmeasured["decision"], "read")
        self.assertNotIn("rec_ms", unmeasured)

        self.cache = self.make_cache()
        self.cache.resident = self.cache.matched_tokens = 0
        self.ntp = self.make_ntp()
        self.plan_turn("p")
        found = self.events(lambda: self.ntp.tick(waiting_queue=[], now=0.01))
        self.assertEqual(gate_decisions(found), [("read", "1")])


class TestRecomputeWhenStarved(RecomputeTestCase):
    """SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED: every queued turn waits on a read
    of its own while the GPU has nothing to admit. Queue: a, whose read went to the read
    thread; b, whose read is parked; c, with no plan, whose read parked before b's; 5 s
    of launched reads ahead of them."""

    switched = (envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED,)

    def setUp(self):
        super().setUp()
        self.handles = {
            name: CacheRequestHandle(f"{name}-t1", 0) for name in ("a", "b", "c", "d")
        }
        for name in ("a", "b", "d"):
            self.plan_turn(name)
        self.cache.ongoing.add(self.handles["a"])
        self.cache.spans[self.handles["a"]] = (0, SHORT_HISTORY)
        for name in ("c", "b"):
            self.cache.park(self.handles[name], start=0, tokens=SHORT_HISTORY)
        self.cache.backlog_pages = 2000
        self.queue = [make_req(f"{name}-t1", entered=1.0) for name in ("a", "b", "c")]

    def passes(self, *times, ntp=None):
        ntp = self.ntp if ntp is None else ntp
        found = []
        for now in times:
            found += self.events(lambda: ntp.tick(waiting_queue=self.queue, now=now))
        return [fields for kind, fields in found if kind == "ntp_starve"]

    def test_the_first_planned_turn_with_a_parked_read_recomputes_and_reads_no_more(
        self,
    ):
        """Not before the queue has had no ready turn for NTP_GPU_BOUND_S: a turn between
        its lookup and its read looks the same for a pass. A read on the read thread is
        left to land, and a turn with no plan could not be kept from looking up again.
        The gate then declines the turn's lookups without charging them."""
        self.assertEqual(self.passes(1.0, 1.4), [])
        (starve,) = self.passes(1.6)
        self.assertEqual(self.cache.dropped, [self.handles["b"]])
        # 2,000 launched pages and c's 126 before b's 126, at 2.5 ms, plus 30 ms of lags.
        self.assertEqual(
            starve,
            {
                "rid": "b-t1",
                "land_ms": "5660",
                "rec_ms": "149",
                "tokens": str(SHORT_HISTORY),
                "parked": "2",
            },
        )
        found = self.events(
            lambda: [
                self.cache.prefetch_gate_fn(self.handles["b"], 0) for _ in range(2)
            ]
        )
        self.assertEqual(gate_decisions(found), [("recompute", "0")] * 2)
        # b now has nothing to read: a turn is ready, so nothing more is dropped.
        self.assertEqual(self.passes(1.7, 2.3, 2.9), [])
        self.assertEqual(self.cache.dropped, [self.handles["b"]])

    def test_one_turn_with_nothing_to_read_keeps_every_read(self):
        """While a ready turn waits the GPU is not starved, however many others wait on
        the SSD: recomputing then takes GPU time that turn is queued for."""
        self.queue.append(make_req("ready-t1", entered=1.0))
        self.assertEqual(self.passes(1.0, 1.6, 2.2, 2.8), [])
        self.assertEqual(self.cache.dropped, [])

    def test_only_the_first_such_turn_is_weighed_and_a_read_that_lands_sooner_is_kept(
        self,
    ):
        """b's span sits 40K tokens deep, where FA3 takes 607 ms for it, and with nothing
        ahead its read lands in 345 ms: kept. d's read parked after b's and does not delay
        it; d would recompute in 149 ms, but one turn a pass is weighed, the one nearest
        admission."""
        self.cache.backlog_pages = 0
        self.cache.read_scheduler.discard(self.handles["c"])
        self.cache.spans[self.handles["b"]] = (40_000, SHORT_HISTORY)
        self.cache.park(self.handles["d"], start=0, tokens=SHORT_HISTORY)
        self.queue = [make_req(f"{name}-t1", entered=1.0) for name in ("a", "b", "d")]
        self.assertEqual(self.passes(1.0, 1.6, 2.2), [])
        self.assertEqual(self.cache.dropped, [])

    def test_a_ready_turn_stops_the_rule_at_once_and_its_wait_starts_again(self):
        """A turn with nothing to read is work for the GPU from the pass it is seen; once it is
        admitted the queue must again go half a second with no ready turn."""
        self.assertEqual(self.passes(1.0, 1.4), [])
        ready = make_req("ready-t1", entered=1.5)
        self.queue.append(ready)
        self.assertEqual(self.passes(1.5, 1.6, 1.7), [])
        ready.time_stats.forward_entry_time = 1.75
        self.queue.remove(ready)
        self.assertEqual(self.passes(1.8, 2.2), [])
        (starve,) = self.passes(2.4)
        self.assertEqual(starve["rid"], "b-t1")

    def test_unmeasured_lags_keep_every_read(self):
        """On the load model's seeds a parked read looks 0.9 s slower than it is, and
        nearly any recompute beats it."""
        self.cache.read_lags = ReadPathLags()
        self.assertEqual(self.passes(1.0, 1.6, 2.2), [])
        self.assertEqual(self.cache.dropped, [])

    def test_a_timed_read_is_also_weighed_behind_the_write_hold(self):
        """b's span costs 607 ms to compute against 345 ms to read. Write-held with 3.3 s left,
        a read due at b's admission lands later; a read needed now is never held."""
        self.admit_through_queue([f"x{i}" for i in range(30)], now=0.5)
        self.cache.backlog_pages = 0
        self.cache.read_scheduler.discard(self.handles["c"])
        self.cache.spans[self.handles["b"]] = (40_000, SHORT_HISTORY)
        self.cache.ongoing.add(self.handles["d"])
        hold_writes(self.cache, since=1.0)
        self.queue = [make_req(f"{name}-t1", entered=1.0) for name in ("a", "b")]
        self.assertEqual(self.passes(1.0, 1.6), [])
        self.queue.insert(1, make_req("d-t1", entered=1.0))
        (starve,) = self.passes(1.7)
        self.assertEqual(
            (starve["rid"], starve["land_ms"], starve["rec_ms"]),
            ("b-t1", "3645", "607"),
        )

    def test_a_turn_that_gave_its_read_up_is_not_handed_back_for_another(self):
        """Its history is missing by choice: the queue check would re-issue its prefetch
        every probe interval, for the gate to decline each time."""
        self.cache.resident = self.cache.matched_tokens = 0
        self.passes(1.0, 1.4, 1.6)
        self.assertEqual(self.cache.dropped, [self.handles["b"]])
        for now in (2.2, 2.8):
            self.assertEqual(self.ntp.tick(waiting_queue=self.queue, now=now), [])

    def test_after_a_drop_the_queue_must_starve_for_another_half_second(self):
        """Admission takes the dropped turn within its pass, so the next pass sees the same
        starved queue: without a new wait a read was given up on every pass."""
        self.cache.park(self.handles["d"], start=0, tokens=SHORT_HISTORY)
        self.queue.append(make_req("d-t1", entered=1.0))
        dropped_at = {}
        for step in range(70):
            now = 1.0 + 0.025 * step
            for starve in self.passes(now):
                dropped_at[starve["rid"]] = now
                (turn,) = [req for req in self.queue if req.rid == starve["rid"]]
                turn.time_stats.forward_entry_time = now + 0.001
                self.queue.remove(turn)
        self.assertEqual(list(dropped_at), ["b-t1", "d-t1"])
        self.assertGreaterEqual(
            dropped_at["d-t1"] - dropped_at["b-t1"], NTP_GPU_BOUND_S - 1e-9
        )

    def test_a_turn_is_weighed_by_the_deadline_it_has_once_it_is_queued(self):
        """At the head x's read is needed now. Weighed in the pass x arrives by its staged
        deadline, the 345 ms read looked 2.6 s away and was given up for a 607 ms recompute."""
        srs = self.cache.read_scheduler
        srs.clear()
        self.cache.backlog_pages = 0
        staged = [self.plan_turn(f"p{i}") for i in range(3)]
        x = self.plan_turn("x")
        for handle in staged:
            self.cache.park(handle, start=0, tokens=300 * PAGE)
        self.cache.park(x, start=40_000, tokens=SHORT_HISTORY)
        self.queue = [make_req("a-t1", entered=1.0)]
        self.assertEqual(self.passes(1.0, 1.4), [])
        self.queue.append(make_req("x-t1", entered=1.6))
        self.assertEqual(self.passes(1.6, 1.7, 2.3), [])
        self.assertTrue(srs.is_parked(x))

    def test_with_the_switch_off_a_starved_queue_keeps_its_reads(self):
        stack = hold_switches(self, (self.switch, False))
        ntp = self.make_ntp()
        stack.close()
        for name in ("a", "b"):
            ntp.on_turn_committed(
                req=make_req(
                    f"{name}-t0",
                    kv_hints=hints(
                        "-",
                        payload={
                            "session": name,
                            "next_rid": f"{name}-t1",
                            "next_gap_s": 5.0,
                        },
                    ),
                ),
                key=key_of(SHORT_HISTORY),
                rotation_tail_declined=False,
                now=0.0,
            )
        self.assertEqual(self.passes(1.0, 1.6, 2.2, ntp=ntp), [])
        self.assertEqual(self.cache.dropped, [])
        self.assertTrue(self.cache.prefetch_gate_fn(self.handles["b"], 0))


if __name__ == "__main__":
    unittest.main()
