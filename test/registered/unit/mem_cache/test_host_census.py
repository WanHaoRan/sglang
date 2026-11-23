"""HiCache host census (agent_cache/OPEN_ITEMS.md section 7): how the tree walk splits host
memory into the GPU mirror and evictable or held host-only tokens, and how often the
HICACHE_EVT host_census line takes the walk. No GPU."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import logging
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.host_census import (
    HoldReason,
    HostCensusLog,
    tally_fields,
    tally_host_tree,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
    UnifiedTreeCore,
    UnifiedTreeNode,
)
from sglang.test.test_utils import CustomTestCase

LOGGER = "sglang.srt.mem_cache.host_census"
FULL = ComponentType.FULL
# The refusal counters every host_census line carries.
COUNTERS = (
    "refused",
    "refused_tok",
    "cut",
    "cut_tok",
    "backup_refused",
    "backup_refused_tok",
)


def new_root():
    return UnifiedTreeNode((FULL,))


def attach(parent, tokens, *, gpu=False, host=True, host_lock_ref=0, lock_ref=0):
    """A child of parent holding `tokens` FULL tokens on the GPU, the host, or both."""
    node = UnifiedTreeNode((FULL,))
    data = node.component_data[FULL]
    data.value = torch.zeros(tokens, dtype=torch.int64) if gpu else None
    data.host_value = torch.zeros(tokens, dtype=torch.int64) if host else None
    data.host_lock_ref = host_lock_ref
    data.lock_ref = lock_ref
    node.parent = parent
    parent.children[node.id] = node
    return node


def descendants(root):
    stack, found = list(root.children.values()), []
    while stack:
        node = stack.pop()
        found.append(node)
        stack.extend(node.children.values())
    return found


def host_tokens(node):
    host_value = node.component_data[FULL].host_value
    return 0 if host_value is None else len(host_value)


def is_host_only(node):
    return node.component_data[FULL].value is None and host_tokens(node) > 0


def census(root, holders=()):
    """The census fields of the tree under root; holders are (node, reason) pairs, each
    accounting for one host lock, as UnifiedRadixCache._census_holders lists them."""
    bits, counts = {}, {}
    for node, reason in holders:
        bits[node.id] = bits.get(node.id, 0) | reason
        counts[node.id] = counts.get(node.id, 0) + 1
    return tally_fields(tally_host_tree(root, holder_bits=bits, holder_counts=counts))


def mixed_tree():
    """One host-only node of each kind below a GPU node with a host copy. The sizes are
    distinct powers of two, so a sum of sizes names the nodes it counts."""
    root = new_root()
    mirror = attach(root, 1, gpu=True)
    branch = attach(mirror, 2)
    write_leaf = attach(branch, 4, host_lock_ref=1)
    pin_leaf = attach(branch, 8, host_lock_ref=1)
    inner = attach(mirror, 16)
    attach(inner, 32)
    attach(mirror, 64, lock_ref=1)  # a live segment's device lock
    attach(mirror, 128, host_lock_ref=1)  # a host lock no holder accounts for
    attach(mirror, 256, gpu=True, host=False)
    attach(root, 512)
    return root, [(write_leaf, HoldReason.WRITE), (pin_leaf, HoldReason.PIN)]


class TestHostTreeTally(CustomTestCase):
    def test_a_locked_host_only_leaf_keeps_its_host_only_parent(self):
        """Host eviction takes childless unlocked nodes only, so a leaf waiting for its SSD
        write also holds its host-only parent: both count for the write, the parent as an
        ancestor. Released, the leaf is evictable at once and the parent after it."""
        root = new_root()
        mirror = attach(root, 64, gpu=True)
        parent = attach(mirror, 128)
        leaf = attach(parent, 256, host_lock_ref=1)
        names = ("mirror", "h_write", "anc", "evict_leaf", "evict_casc")
        held = census(root, holders=[(leaf, HoldReason.WRITE)])
        self.assertEqual([held[name] for name in names], [64, 128 + 256, 128, 0, 0])
        leaf.component_data[FULL].host_lock_ref = 0
        released = census(root)
        self.assertEqual([released[name] for name in names], [64, 0, 0, 256, 128])

    def test_every_host_token_lands_in_exactly_one_field(self):
        """The line's resid is the pool less free, release queue, staging, mirror and
        host_only, so a host token counted twice or missed would show there. A branch held
        below by two reasons counts once, in h_multi."""
        root, holders = mixed_tree()
        fields = census(root, holders)
        self.assertEqual(
            fields["mirror"] + fields["host_only"],
            sum(host_tokens(node) for node in descendants(root)),
        )
        held = sum(value for name, value in fields.items() if name.startswith("h_"))
        self.assertEqual(
            fields["evict_leaf"] + fields["evict_casc"] + held, fields["host_only"]
        )
        branch = (fields["h_multi"], fields["h_write"], fields["h_pin"])
        self.assertEqual(branch, (2, 4, 8))

    def test_a_lock_and_its_holders_that_disagree_are_counted(self):
        """A host lock no holder accounts for still keeps its node, as OTHER; a holder whose
        node has no host lock is flagged as well but holds nothing, since host eviction can
        take that node. Each is a mismatch, which says the holders the cache tracks do not
        explain the locks."""
        root = new_root()
        attach(root, 64, host_lock_ref=1)
        unlocked = attach(root, 128)
        fields = census(root, holders=[(unlocked, HoldReason.READ)])
        self.assertEqual(
            (fields["h_other"], fields["h_read"], fields["evict_leaf"], fields["mismatch"]),
            (64, 0, 128, 2),
        )

    def test_evict_leaf_is_what_host_eviction_can_take_now(self):
        """evict_leaf must count exactly the nodes UnifiedTreeCore._is_host_leaf lets host
        eviction take, before and after every lock is released."""
        root, holders = mixed_tree()
        core = SimpleNamespace(root_node=root)
        for released in (False, True):
            if released:
                for node in descendants(root):
                    data = node.component_data[FULL]
                    data.host_lock_ref = data.lock_ref = 0
                holders = []
            with self.subTest(released=released):
                takeable = sum(
                    host_tokens(node)
                    for node in descendants(root)
                    if is_host_only(node) and UnifiedTreeCore._is_host_leaf(core, node)
                )
                self.assertEqual(census(root, holders)["evict_leaf"], takeable)


class TestHostCensusLog(CustomTestCase):
    def setUp(self):
        self.now = 0.0
        self.log = HostCensusLog(interval_s=10.0, clock=lambda: self.now)
        self.takes = 0

    def take(self):
        self.takes += 1
        return {"free": 7}

    def lines(self, fn):
        """The host_census lines logged while fn runs, as field dicts."""
        with self.assertLogs(LOGGER, level="INFO") as logs:
            fn()
            logging.getLogger(LOGGER).info("end")
        found = []
        for line in logs.output[:-1]:
            words = line.split(":", 2)[2].split()
            self.assertEqual(words[:2], ["HICACHE_EVT", "host_census"])
            fields = dict(word.split("=", 1) for word in words[2:])
            found.append({k: v if k == "why" else int(v) for k, v in fields.items()})
        return found

    def test_a_tick_line_comes_once_per_interval(self):
        """The tick runs every scheduler pass and the walk visits every node, so only a pass
        an interval after the last line walks the tree."""
        for now, lines in ((9.9, 0), (10.0, 1), (19.9, 0), (20.0, 1)):
            self.now = now
            with self.subTest(now=now):
                found = self.lines(lambda: self.log.tick(self.take))
                self.assertEqual([line["why"] for line in found], ["tick"] * lines)
        self.assertEqual(self.takes, 2)

    def test_refusals_in_one_interval_walk_once_and_reach_the_next_line(self):
        """A refusal storm refuses reads every pass: the first refusal's line walks the
        tree, the rest only count, and the next line carries their counts, once. A D2H
        backup refusal runs inside insert, where the tree must not be walked. A refusal
        line restarts the tick's interval, so one state is not walked twice in a pass."""
        self.now = 1.0
        first = self.lines(
            lambda: self.log.on_read_refusal(self.take, tokens=640, cut=False)
        )
        self.now = 2.0

        def quiet():
            self.log.on_read_refusal(self.take, tokens=128, cut=True)
            self.log.on_read_refusal(self.take, tokens=64, cut=False)
            self.log.on_backup_refusal(32)

        self.assertEqual(self.lines(quiet), [])
        self.assertEqual(self.takes, 1)
        self.now = 10.0
        self.assertEqual(self.lines(lambda: self.log.tick(self.take)), [])
        self.now = 11.0
        carried = self.lines(lambda: self.log.tick(self.take))
        self.now = 21.0
        reset = self.lines(lambda: self.log.tick(self.take))
        counters = [
            (line["why"], line["free"], [line[name] for name in COUNTERS])
            for line in first + carried + reset
        ]
        self.assertEqual(
            counters,
            [
                ("refusal", 7, [1, 640, 0, 0, 0, 0]),
                ("tick", 7, [1, 64, 1, 128, 1, 32]),
                ("tick", 7, [0, 0, 0, 0, 0, 0]),
            ],
        )

    def test_the_census_is_off_unless_an_interval_is_set(self):
        """Interval 0, the default, keeps the walk out of the scheduler, and so does a cache
        the census does not apply to."""
        for interval, enabled, interval_s in (
            (0.0, True, None),
            (30.0, False, None),
            (30.0, True, 30.0),
        ):
            with self.subTest(interval=interval, enabled=enabled):
                with envs.SGLANG_LOG_HICACHE_HOST_CENSUS_INTERVAL.override(interval):
                    log = HostCensusLog.maybe_create(enabled=enabled)
                self.assertEqual(None if log is None else log.interval_s, interval_s)


if __name__ == "__main__":
    unittest.main()
