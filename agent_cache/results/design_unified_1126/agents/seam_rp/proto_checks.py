"""CPU checks of the premises the Retain / Preempt plans rest on, against the engine's own pure-Python modules
as committed at HEAD, loaded by path (nothing in the repository is edited, no torch, no GPU).

Usage: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B proto_checks.py
  A  R4 on the unchanged cleaner (nixl_cleaner.py): files at mtime zero go first, only if every file of the page
     is zeroed; a refresh with the head newest loses its tail first; a file touched after the cleaner's scan is
     still deleted (the race); utime on a deleted file raises and creates nothing.
  B  P1 on the engine's eviction strategies (next_use_eviction.py): a class bound as a tuple stops a min-heap
     walk before the queued tier; a prototype Queued with an admission time orders by slack and shields the head.
  C  P2: the reserve rule as a pure function, on campaign 34's ranges and on a 70 GB host.
  D  R2: a priority queue of (rank, op id) keeps FIFO among equal ranks and takes a comparable stop sentinel.
  E  R3: the ack walk that marks a node and the split fragments above it.
"""
import enum
import heapq
import importlib.util
import os
import queue
import re
import shutil
import sys
import time
import types

REPO = "/home/wanhr/sglang"
HERE = os.path.dirname(os.path.abspath(__file__))
# The engine files as committed at HEAD, read with `git show` into a scratch folder here: the working tree
# may differ (next_use_eviction.py and next_turn_policy.py were edited by someone else at 18:23 UTC on
# 2026-11-26, while these checks were being written).
SRT = os.path.join(HERE, "head_src")
FILES = (
    "mem_cache/evict_policy.py",
    "mem_cache/next_turn_policy.py",
    "mem_cache/next_use_eviction.py",
    "mem_cache/hicache_storage.py",
    "mem_cache/storage/nixl/nixl_routing.py",
    "mem_cache/storage/nixl/nixl_cleaner.py",
)


def fetch_head_sources():
    import subprocess

    shutil.rmtree(SRT, ignore_errors=True)
    for rel in FILES:
        dst = os.path.join(SRT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        blob = subprocess.run(
            ["git", "-C", REPO, "show", f"HEAD:python/sglang/srt/{rel}"], check=True, capture_output=True
        ).stdout
        with open(dst, "wb") as f:
            f.write(blob)
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], check=True, capture_output=True)
    return head.stdout.decode().strip()


def _package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def load(name, rel):
    """Load <SRT>/<rel> as module `name`, with bare parent packages."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        _package(".".join(parts[:i]))
    spec = importlib.util.spec_from_file_location(name, os.path.join(SRT, rel))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stub_hicache_storage():
    """hicache_storage imports torch; the cleaner needs only PoolName's values, read from its source."""
    text = open(os.path.join(SRT, "mem_cache/hicache_storage.py")).read()
    body = text.split("class PoolName(str, Enum):", 1)[1].split("\nclass ", 1)[0]
    members = dict(re.findall(r'^\s{4}([A-Z0-9_]+) = "([^"]+)"', body, flags=re.M))
    pool_name = enum.Enum("PoolName", members, type=str)
    module = types.ModuleType("sglang.srt.mem_cache.hicache_storage")
    module.PoolName = pool_name
    _package("sglang")
    _package("sglang.srt")
    _package("sglang.srt.mem_cache")
    sys.modules[module.__name__] = module
    return len(members)


def check(label, ok):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    return ok


# ----- A: R4 on the unchanged cleaner -----

def check_cleaner():
    n = stub_hicache_storage()
    routing = load("sglang.srt.mem_cache.storage.nixl.nixl_routing", "mem_cache/storage/nixl/nixl_routing.py")
    cleaner_mod = load("sglang.srt.mem_cache.storage.nixl.nixl_cleaner", "mem_cache/storage/nixl/nixl_cleaner.py")
    print(f"A. R4 against nixl_cleaner.py at HEAD ({n} pool names stubbed)")
    root = os.path.join(HERE, "proto_tmp")
    ok = True

    def fresh():
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        return [root]

    def path_of(key):
        _, bucket = routing.route_key(key, 1)
        return os.path.join(root, bucket, key)

    def write(page, mtime, suffixes=("_k", "_v")):
        paths = []
        for suffix in suffixes:
            p = path_of(f"{page}_model_0_1{suffix}")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(b"x" * 16)
            os.utime(p, (mtime, mtime))
            paths.append(p)
        return paths

    def one_group_pass(after_scan=None):
        cleaner = cleaner_mod.HiCacheL3Cleaner(fresh_dirs, tp_rank=0, high_watermark=80.0, low_watermark=70.0,
                                               recheck_groups=1, unlink_workers=1)
        calls = {}

        def usage(p):
            calls[p] = calls.get(p, 0) + 1
            return 90.0 if calls[p] == 1 else 60.0

        cleaner._disk_usage_pct = usage
        if after_scan is not None:
            scan = cleaner._scan_base_dir

            def scan_then(base_dir, groups):
                scan(base_dir, groups)
                after_scan()

            cleaner._scan_base_dir = scan_then
        return cleaner._tick()

    # A1: a retired page (every file at mtime zero) goes before an older untouched page
    fresh_dirs = fresh()
    old = write("page-old", 100.0)
    ended = write("page-ended", 200.0)
    for p in ended:
        os.utime(p, ns=(0, 0))
    one_group_pass()
    ok &= check("a page whose files are all at mtime zero is deleted before an older untouched page",
                not any(map(os.path.exists, ended)) and all(map(os.path.exists, old)))
    # A2: zeroing one of the two files is not enough: the group's mtime is the max over its files
    fresh_dirs = fresh()
    old = write("page-old", 100.0)
    half = write("page-half", 200.0)
    os.utime(half[0], ns=(0, 0))
    one_group_pass()
    ok &= check("with only the K file zeroed the page keeps its newest mtime and the older page goes instead",
                all(map(os.path.exists, half)) and not any(map(os.path.exists, old)))
    # A3: refresh with the head newest: the tail goes first, an untouched older page before both
    fresh_dirs = fresh()
    chain = [write(f"page-c{i}", 100.0 + i) for i in range(3)]     # written head first: the head is the oldest
    newest = time.time_ns()
    for i, files in enumerate(chain):                              # head gets the newest stamp
        for p in files:
            os.utime(p, ns=(newest - i * 1_000_000,) * 2)
    one_group_pass()
    ok &= check("after a refresh that stamps the head newest, the cleaner takes the chain's tail and keeps its head",
                all(map(os.path.exists, chain[0])) and all(map(os.path.exists, chain[1]))
                and not any(map(os.path.exists, chain[2])))
    fresh_dirs = fresh()
    chain = [write(f"page-c{i}", 100.0 + i) for i in range(3)]
    one_group_pass()
    ok &= check("without the refresh the same pass takes the head (today's order)",
                not any(map(os.path.exists, chain[0])) and all(map(os.path.exists, chain[2])))
    # A4: the race: a touch that lands after the scan does not save the file
    fresh_dirs = fresh()
    victim = write("page-victim", 100.0)
    other = write("page-other", 200.0)

    def touch_victim():
        for p in victim:
            os.utime(p)

    one_group_pass(after_scan=touch_victim)
    ok &= check("a page touched after the cleaner's scan is still deleted from the stale order (race B3)",
                not any(map(os.path.exists, victim)) and all(map(os.path.exists, other)))
    # A5: utime on a deleted file
    try:
        os.utime(victim[0], ns=(0, 0))
        raised = False
    except FileNotFoundError:
        raised = True
    ok &= check("os.utime on a deleted file raises FileNotFoundError and creates nothing",
                raised and not os.path.exists(victim[0]))
    shutil.rmtree(root, ignore_errors=True)
    return ok


# ----- B: P1 on the engine's strategies -----

def check_bounds():
    load("sglang.srt.mem_cache.evict_policy", "mem_cache/evict_policy.py")
    load("sglang.srt.mem_cache.next_turn_policy", "mem_cache/next_turn_policy.py")
    nue = load("sglang.srt.mem_cache.next_use_eviction", "mem_cache/next_use_eviction.py")
    print("B. P1 against next_use_eviction.py at HEAD")
    ok = True
    now = 1000.0
    node = lambda i, t=0.0: types.SimpleNamespace(id=i, last_access_time=t, parent=None, children={})
    for name, strategy, queued_tier in (
        ("next_use", nue.NextUseStrategy(clock=lambda: now), nue.TIER_QUEUED),
        ("return_chance", nue.ReturnChanceStrategy(clock=lambda: now), nue.RETURN_TIER_QUEUED),
    ):
        reg = strategy.registry
        reg.register(1, None)
        reg.register(2, nue.Queued(since=now))
        if name == "next_use":
            reg.register(3, now + 10.0)
            reg.register(4, now + 1000.0)
        else:
            dist = sys.modules["sglang.srt.mem_cache.next_turn_policy"].GapDistribution.from_quantiles(
                p10=1.0, p50=20.0, p90=60.0)
            reg.register(3, nue.ReturnForecast(dist=dist, idle_from=now - 5.0))
            reg.register(4, nue.ReturnForecast(dist=dist, idle_from=now - 900.0))
        nodes = [node(i, float(i)) for i in (1, 2, 3, 4, 5)]
        bound = (queued_tier,)
        heap = [(strategy.get_priority(n), n.id) for n in nodes]
        heapq.heapify(heap)
        taken = []
        while heap:
            priority, node_id = heapq.heappop(heap)
            if priority >= bound:
                break
            taken.append(node_id)
        ok &= check(f"{name}: a walk bounded at ({queued_tier},) takes every idle class and stops at the queued turn "
                    f"(took {taken})", sorted(taken) == [1, 3, 4, 5] and 2 not in taken)

    # prototype: a queued registration that carries its admission time and re-read time
    class QueuedSlack(nue.Queued, frozen=True):
        admit_at: float = 0.0
        reread_s: float = 0.0

    tier_head = nue.TIER_QUEUED + 1
    min_slack = 2.0

    def queued_priority(reg_value, last_access):
        slack = reg_value.admit_at - now - reg_value.reread_s
        if slack < min_slack:
            return (tier_head, last_access)
        return (nue.TIER_QUEUED, -slack)

    far = QueuedSlack(since=now, admit_at=now + 30.0, reread_s=3.0)       # slack 27
    mid = QueuedSlack(since=now, admit_at=now + 12.0, reread_s=3.0)       # slack 9
    head = QueuedSlack(since=now, admit_at=now + 0.3, reread_s=3.0)       # slack < 0
    tight = QueuedSlack(since=now, admit_at=now + 4.0, reread_s=3.0)      # slack 1 < 2
    order = sorted([("head", queued_priority(head, 1.0)), ("mid", queued_priority(mid, 2.0)),
                    ("far", queued_priority(far, 3.0)), ("tight", queued_priority(tight, 4.0))], key=lambda x: x[1])
    ok &= check(f"queued turns yield largest slack first, and a turn whose re-read does not fit is in the head class "
                f"({[x[0] for x in order]})", [x[0] for x in order] == ["far", "mid", "head", "tight"])
    ok &= check("a walk bounded before the head class may take slack-rich queued turns only",
                all((p >= (tier_head,)) == (name in ("head", "tight")) for name, p in order))
    ok &= check("isinstance(QueuedSlack(...), Queued) holds, so today's tier code still sees a queued turn",
                isinstance(far, nue.Queued))
    return ok


# ----- C: P2 as a pure rule -----

def reserve_hold(*, pending_tokens, reserve_tokens, alloc_tokens_per_s, write_tokens_per_s):
    """Hold timed reads while the host room that can be handed out without losing an only copy is less than
    what backups and landings will ask for until the pending writes have drained."""
    if pending_tokens <= 0 or not write_tokens_per_s:
        return False
    drain_s = pending_tokens / write_tokens_per_s
    return reserve_tokens < alloc_tokens_per_s * drain_s


def check_reserve_rule():
    print("C. P2 reserve rule")
    ok = True
    # campaign 34 late phase: reserve 0.6-0.9 M (free + evictable leaf + cascade), pending <= 87 K,
    # writes 4-7 K tokens/s while reads take the disk, allocation about 18 K tokens/s
    ok &= check("campaign 34's worst pending (87 K) against its usual reserve (0.6 M): no hold",
                not reserve_hold(pending_tokens=87_000, reserve_tokens=600_000, alloc_tokens_per_s=18_000,
                                 write_tokens_per_s=4_000))
    ok &= check("the same pending against the leaf-only count (40 K): hold, as the fixed 50 K rule would",
                reserve_hold(pending_tokens=87_000, reserve_tokens=40_000, alloc_tokens_per_s=18_000,
                             write_tokens_per_s=4_000))
    # campaign 32's loop: pending 0.4-0.6 M
    ok &= check("campaign 32's backlog (400 K pending) against 0.6 M of reserve: hold",
                reserve_hold(pending_tokens=400_000, reserve_tokens=600_000, alloc_tokens_per_s=18_000,
                             write_tokens_per_s=7_000))
    # a 70 GB host: 44 K of room beyond the mirror; the fixed rule's 50 K exceeds the whole room
    ok &= check("70 GB host (44 K of room): 15 K pending already holds, where the fixed rule waits for 50 K",
                reserve_hold(pending_tokens=15_000, reserve_tokens=44_000, alloc_tokens_per_s=18_000,
                             write_tokens_per_s=5_000))
    ok &= check("nothing pending, or no write rate measured yet: no hold",
                not reserve_hold(pending_tokens=0, reserve_tokens=0, alloc_tokens_per_s=18_000, write_tokens_per_s=5_000)
                and not reserve_hold(pending_tokens=50_000, reserve_tokens=0, alloc_tokens_per_s=18_000,
                                     write_tokens_per_s=None))
    return ok


# ----- D: R2 priority queue -----

def check_write_queue():
    print("D. R2 write queue order")
    ok = True
    q = queue.PriorityQueue()
    stop = ((-1,), -1, None)                  # sorts before every rank, comparable with them
    ops = [((1, (3, 5.0)), 10, "mirror-a"), ((0, (3, 7.0)), 11, "hostonly-b"), ((1, (3, 5.0)), 12, "mirror-c"),
           ((0, (1, 2.0)), 13, "hostonly-overdue-d"), ((1, (0, 1.0)), 14, "mirror-ended-e")]
    for item in ops:
        q.put(item)
    out = [q.get()[2] for _ in ops]
    ok &= check(f"host-only pages first, then by the eviction priority, FIFO among equals ({out})",
                out == ["hostonly-overdue-d", "hostonly-b", "mirror-ended-e", "mirror-a", "mirror-c"])
    q.put(ops[0])
    q.put(stop)
    ok &= check("the stop sentinel compares with queued ops and comes out first", q.get()[2] is None)
    try:
        q2 = queue.PriorityQueue()
        q2.put(ops[0])
        q2.put(None)
        q2.get()
        plain_none_ok = True
    except TypeError:
        plain_none_ok = False
    ok &= check("a bare None, as _stop_storage_threads puts today, is not orderable in a priority queue",
                not plain_none_ok)
    return ok


# ----- E: R3 ack walk -----

class _Node:
    def __init__(self, node_id, parent=None):
        self.id, self.parent = node_id, parent
        self.storage_write_op = None
        self.storage_stored = False


def finish_storage_write(node, root, op_id, stored):
    marked = []
    while node is not None and node is not root and node.storage_write_op == op_id:
        node.storage_write_op = None
        node.storage_stored = stored
        marked.append(node.id)
        node = node.parent
    return marked


def check_ack_walk():
    print("E. R3 ack walk over split fragments")
    ok = True
    root = _Node(0)
    parent = _Node(1, root)
    parent.storage_write_op = 6               # its own write, still queued
    child = _Node(3, parent)
    child.storage_write_op = 7
    # the node splits twice while op 7 is queued: fragments 4 and 5 sit between parent and child
    frag_a = _Node(4, parent)
    frag_a.storage_write_op = 7
    frag_b = _Node(5, frag_a)
    frag_b.storage_write_op = 7
    child.parent = frag_b
    marked = finish_storage_write(child, root, 7, stored=True)
    ok &= check(f"a full ack marks the node and both fragments stored and leaves the parent's own write pending "
                f"(marked {marked})", marked == [3, 5, 4] and child.storage_stored and frag_a.storage_stored
                and parent.storage_write_op == 6 and not parent.storage_stored)
    short = _Node(8, root)
    short.storage_write_op = 9
    finish_storage_write(short, root, 9, stored=False)
    ok &= check("a short ack clears the pending mark and leaves the pages unstored",
                short.storage_write_op is None and not short.storage_stored)
    return ok


if __name__ == "__main__":
    print(f"engine sources: git show HEAD ({fetch_head_sources()})")
    results = [check_cleaner(), check_bounds(), check_reserve_rule(), check_write_queue(), check_ack_walk()]
    shutil.rmtree(SRT, ignore_errors=True)
    print("ALL OK" if all(results) else "SOME CHECKS FAILED")
