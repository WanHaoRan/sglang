"""Check, on the engine's own next_turn_policy.py, how read_start relates to
(a) the paper's newsvendor threshold F(x | idle + s) >= q = rent/(1 + rent) and
(b) a quantile rule "land at the q-quantile of the remaining gap".
Loads the policy module by path with a tiny msgspec stand-in; changes nothing."""
import math, sys, types, bisect

shim = types.ModuleType("msgspec")
class Struct:
    def __init_subclass__(cls, **kw):
        pass
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
shim.Struct = Struct
sys.modules["msgspec"] = shim
src = open(sys.argv[1]).read()
ns = {"__name__": "ntp_policy"}
exec(compile(src, sys.argv[1], "exec"), ns)
GD, remaining_gaps, read_start = ns["GapDistribution"], ns["remaining_gaps"], ns["read_start"]

def newsvendor_start(rem, lead, rent):
    """Earliest support-grid start s >= 0 where the share of remaining points in
    [s, s + lead) among those >= s reaches q (the paper's rule at time now + s)."""
    q = rent / (1.0 + rent)
    cands = sorted({0.0, *(max(0.0, g - lead) for g in rem)})
    for s in cands:
        alive = [g for g in rem if g >= s]
        if not alive:
            return None
        f = sum(1 for g in alive if g < s + lead) / len(alive)
        if f >= q:
            return s
    return None

# FreeInference trace waits (data/converted/freeinference_16w.programs.jsonl.gz.stats.json)
dists = {
    "tool p10/50/90 0.41/1.69/16.8": GD.from_quantiles(p10=0.414, p50=1.692, p90=16.795),
    "user p10/50/90 3.1/96/1040": GD.from_quantiles(p10=3.084, p50=96.228, p90=1040.492),
    "point 30 s (exact)": GD.point(30.0),
}
price = 5e-5
print("price", price)
for name, d in dists.items():
    sup = d.support()
    for tokens in (5000, 30000):
        rent = price * tokens
        for read_s, wait_s in ((4.8, 0.0), (4.8, 3.7), (4.8, 7.1)):
            lead = read_s - wait_s
            for idle in (0.0, 10.0, 120.0):
                rem = remaining_gaps(support=sup, idle_s=idle)
                s_engine = read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent)
                s_nv = newsvendor_start(rem, lead, rent) if lead > 0 and rem else None
                print(f"{name:32s} L={tokens:6d} q={rent/(1+rent):.2f} r={read_s} W={wait_s} idle={idle:6.1f} "
                      f"points_left={len(rem):2d} read_start={s_engine if s_engine is None else round(s_engine,2)} "
                      f"newsvendor={s_nv if s_nv is None else round(s_nv,2)}")
