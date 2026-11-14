"""Quick probe (read-only, verifier U15): the deck's limit "one pooled distribution -> the hint-free arm".
Runs the ENGINE's own policy code (python/sglang/srt/mem_cache/next_turn_policy.py, read as text and executed with a
stand-in for msgspec.Struct, since the host has the standard library only). Nothing is written.

Part A, rule 1: with the pooled entry sent for every key, does read_start ever start a read before arrival at 5e-5?
  The engine re-solves every 0.5 s on the remaining distribution; the first idle time at which it answers "now" is
  printed, under the deck's read time r = bL/beta and under the engine's (load model + 1 s margin, no backlog).
Part B, rule 3: is chance_within(idle, 30 s) falling with idle time for the pooled entry (then the order is LRU)?
Not a verified analysis."""
import math, sys, types

SRC = "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py"


class _Struct:
    """Just enough of msgspec.Struct for the policy module: fields from annotations, defaults from class attributes."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        names = [n for klass in reversed(type(self).__mro__) for n in vars(klass).get("__annotations__", {})]
        for name, value in zip(names, args):
            object.__setattr__(self, name, value)
        for name in names[len(args):]:
            if name in kwargs:
                value = kwargs[name]
            else:
                value = getattr(type(self), name)
                if isinstance(value, (dict, list)):
                    value = type(value)(value)
            object.__setattr__(self, name, value)
        post = getattr(self, "__post_init__", None)
        if post:
            post()


stub = types.ModuleType("msgspec")
stub.Struct = _Struct
sys.modules["msgspec"] = stub
ns = {"__name__": "next_turn_policy_probe"}
exec(compile(open(SRC).read(), SRC, "exec"), ns)
GapDistribution, remaining_gaps, read_start = ns["GapDistribution"], ns["remaining_gaps"], ns["read_start"]

PRICE = 5e-5
BYTES_PER_TOKEN, BETA = 98304, 2.11e9
PAGE = 64
# L3LoadModel seeds (next_turn_estimators.py) plus NTP_JIT_MARGIN_S (next_turn_prefetch.py), no backlog.
engine_read_s = lambda tokens: 0.5 + 0.044 + 0.0025 * math.ceil(tokens / PAGE) + 0.36 + 1.0
deck_read_s = lambda tokens: tokens * BYTES_PER_TOKEN / BETA

# Pooled gaps: evict_replay.out (campaign 23 jit_p50) per source, and ec_r13's own client log over every source.
POOLED = {
    "swebench (evict_replay.out)": (0.58, 7.05, 12.75),
    "all sources, ec_r13 run": (0.58, 7.05, 17.50),
    "gaia (evict_replay.out)": (3.00, 22.38, 95.27),
    "wildclaw (evict_replay.out)": (0.31, 5.51, 100.31),
}


def first_now(dist, *, read_s, rent):
    """Idle time at which the engine's re-solve first says 'start now', or None (left to arrival)."""
    support = dist.support()
    idle = 0.0
    while idle < support[-1]:
        remaining = remaining_gaps(support=support, idle_s=idle)
        if not remaining:
            return None
        start = read_start(remaining=remaining, read_s=read_s, wait_s=0.0, rent_per_s=rent)
        if start == 0.0:
            return idle
        idle += 0.5
    return None


print("Part A. Rule 1 on one pooled entry, price 5e-5, no queue wait; rent on the read's own tokens")
print("  entry (p10 p50 p90)                    tokens | deck r, q -> first 'now' at idle | engine r -> first 'now' at idle, and the share of gaps already over by then")
for name, (p10, p50, p90) in POOLED.items():
    dist = GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
    for tokens in (8000, 16000, 32000, 64000):
        rent = PRICE * tokens
        q = rent / (1 + rent)
        cells = []
        for read_s in (deck_read_s(tokens), engine_read_s(tokens)):
            at = first_now(dist, read_s=read_s, rent=rent)
            cells.append(f"r {read_s:4.2f} s -> " + ("arrival" if at is None else f"idle {at:5.1f} s (gaps over by then {100 * (1 - dist.survival(at)):4.1f} %)"))
        print(f"  {name:38s} {tokens:6d} | q {q:4.2f} {cells[0]:46s} | {cells[1]}")

print("\nPart B. Rule 3 on one pooled entry: chance of return within 30 s by idle time (LRU order needs it to fall with idle time)")
grid = (0, 1, 2, 4, 6, 8, 10, 13, 20, 30, 60, 120, 300, 600, 1200)
print("  idle s:                                " + " ".join(f"{a:6d}" for a in grid))
for name, (p10, p50, p90) in POOLED.items():
    dist = GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
    vals = [dist.chance_within(idle_s=float(a), horizon_s=30.0) for a in grid]
    fine = [dist.chance_within(idle_s=a / 10, horizon_s=30.0) for a in range(0, 12001)]
    peak = max(range(len(fine)), key=fine.__getitem__) / 10
    rising = sum(1 for x, y in zip(fine, fine[1:]) if y > x + 1e-15) / 10
    # Idle time past the peak at which the chance is back down to its value at idle 0: sessions younger than that
    # rank above a session that has just finished, i.e. not in LRU order among themselves.
    back = next((a / 10 for a in range(int(peak * 10), 12001) if fine[a] <= fine[0]), None)
    print(f"  {name:38s} " + " ".join(f"{v:6.4f}" for v in vals) + f" | peak at idle {peak:.1f} s, rising over {rising:.1f} s of idle time, back to its idle-0 value at {back} s")
