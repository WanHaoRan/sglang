"""Quick probe (read-only, verifier U15): the eviction replay with ONE pooled entry for every key.
Run from agent_cache/:  nice -n 10 python3 -I -B <this file>
It executes frozen/ec_policy_20261111/evict_replay/evict_replay.py's own definitions (text up to its last dispatch,
nothing of it is changed or written) and adds rows the replay does not have: chance of return within 30 s scored from
the pooled entry alone, per source and over all sources, with the end signalled and never signalled. The deck's limit
slide says this order is LRU. Not a verified analysis."""
import collections, os, sys

PATH = "frozen/ec_policy_20261111/evict_replay/evict_replay.py"
source = open(PATH).read()
ns = {"__name__": "evict_replay_probe"}
sys.argv = sys.argv[:1]
exec(compile(source[:source.index('\nif len(sys.argv) > 1 and sys.argv[1] == "noend":')], PATH, "exec"), ns)
load, simulate, share, Dist, chance_own, lru, lru_end, exact, return_chance, RUNS, LRU_TARGET = (
    ns[k] for k in ("load", "simulate", "share", "Dist", "chance_own", "lru", "lru_end", "exact", "return_chance", "RUNS", "LRU_TARGET"))

turns = load(RUNS["jit_p50"])
by = {(r["conv"], r["turn"]): r for r in turns}
gaps = collections.defaultdict(list)
for r in turns:
    nxt = by.get((r["conv"], r["turn"] + 1))
    if nxt:
        g = nxt["t_send"] - r["t_send"] - r["latency"]
        gaps[r["source"]].append(g)
        gaps["*"].append(g)
pooled = {}
for src, xs in gaps.items():
    xs.sort()
    q = lambda p: max(xs[min(len(xs) - 1, int(p * len(xs)))], 0.01)
    pooled[src] = Dist(q(0.1), q(0.5), q(0.9))
    print(f"pooled {src}: n {len(xs)} p10 {q(.1):.2f} p50 {q(.5):.2f} p90 {q(.9):.2f}")
lo, hi = 200_000, 3_000_000
for _ in range(18):
    mid = (lo + hi) // 2
    if share(simulate(turns, mid, lru)) > LRU_TARGET:
        lo = mid
    else:
        hi = mid
cap = hi


def pooled_only(h, *, per_source, ended_first):
    """Every session is scored from the pooled entry, whatever its key; an un-hinted reply too."""
    def fn(s, now):
        if ended_first and s.ended:
            return (0, s.last_used, 0.0)
        d = pooled[s.key[0]] if per_source else pooled["*"]
        a = now - s.t_end - 0.1
        return (2, chance_own(d, None, a, h), a - d.m)
    return fn


rows = [("LRU", lru), ("LRU, ended sessions first", lru_end),
        ("return chance H=30, own entries (the replay's row)", return_chance(30, chance_own)),
        ("return chance H=30, pooled entry per source, ends ignored", pooled_only(30, per_source=True, ended_first=False)),
        ("return chance H=30, one pooled entry, ends ignored", pooled_only(30, per_source=False, ended_first=False)),
        ("return chance H=30, pooled entry per source, ended first", pooled_only(30, per_source=True, ended_first=True)),
        ("return chance H=30, one pooled entry, ended first", pooled_only(30, per_source=False, ended_first=True)),
        ("exact next use (Belady)", exact)]
print(f"pool {cap} tokens (LRU at {100 * LRU_TARGET:.1f} %): share of returning turns that read from SSD")
for label, prio in rows:
    o = simulate(turns, cap, prio)
    print(f"  {label:62s} {100 * share(o):5.1f} %  ({o['read_turns']} of {o['returning']})")
