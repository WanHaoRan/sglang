"""Expected cost of read_start's candidate starts (the engine's own cost formula, re-evaluated
here per start) for the FreeInference tool and user gap entries; shows where the global
minimum sits against starting now and against reading at arrival."""
import sys, types, bisect, itertools
shim = types.ModuleType("msgspec")
class Struct:
    def __init_subclass__(cls, **kw): pass
    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)
shim.Struct = Struct; sys.modules["msgspec"] = shim
ns = {"__name__": "ntp_policy"}
exec(compile(open(sys.argv[1]).read(), sys.argv[1], "exec"), ns)
GD, remaining_gaps, read_start = ns["GapDistribution"], ns["remaining_gaps"], ns["read_start"]

def cost(rem, start, read_s, wait_s, rent):
    # The same piecewise cost as next_turn_policy.read_start (lines 185-198).
    stall_at_arrival = max(0.0, read_s - wait_s)
    prefix = list(itertools.accumulate(rem, initial=0.0)); total = prefix[-1]; n = len(rem)
    lead = read_s - wait_s
    arrived = bisect.bisect_right(rem, start); late = bisect.bisect_right(rem, start + lead)
    stall = arrived * stall_at_arrival + ((late - arrived) * (start + lead) - (prefix[late] - prefix[arrived]))
    pinned = (total - prefix[late]) - (n - late) * (start + lead)
    return (stall + rent * pinned) / n, stall / n, pinned / n

for name, d in (("tool", GD.from_quantiles(p10=0.414, p50=1.692, p90=16.795)),
                ("user", GD.from_quantiles(p10=3.084, p50=96.228, p90=1040.492))):
    rem = remaining_gaps(support=d.support(), idle_s=0.0)
    print(name, "support top 4", [round(x, 1) for x in rem[-4:]], "sigma_low/high", round(d.sigma_low, 2), round(d.sigma_high, 2))
    for tokens, read_s, wait_s in ((5000, 4.8, 3.7), (30000, 4.8, 3.7), (30000, 4.8, 0.0)):
        rent = 5e-5 * tokens
        best = read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent)
        c0 = cost(rem, 0.0, read_s, wait_s, rent)
        cb = cost(rem, best, read_s, wait_s, rent) if best is not None else None
        lead = read_s - wait_s
        f0 = bisect.bisect_left(rem, lead) / len(rem)
        print(f"  L={tokens} r={read_s} W={wait_s} q={rent/(1+rent):.2f} F(x|0)={f0:.2f} | arrival cost={lead:.3f} "
              f"| start now: cost={c0[0]:.3f} (stall {c0[1]:.3f}, pinned s {c0[2]:.1f}) "
              f"| read_start={None if best is None else round(best,1)} cost={None if cb is None else round(cb[0],3)}")
