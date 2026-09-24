import dup, json, csv, bisect, statistics as st
from dup import *
def dists(arm):
    rd, cd = RUNS[arm]
    turns = []
    for line in open(R + rd + "/" + cd + "/client.jsonl"):
        o = json.loads(line)
        if o.get("kind") == "turn": turns.append(o)
    tier = {}
    for r in csv.DictReader(open(R + f"analysis_gapscale/turn_components_x70 (campaign 7)_{arm}.csv")):
        tier[(int(r["conv"]), int(r["turn"]))] = r["tier"]
    fin = sorted(((o["t_send"] + o["latency"], o["conv"], o["prompt_tokens"] + o["completion_tokens"]) for o in turns))
    fts = [x[0] for x in fin]
    res = []
    for o in sorted(turns, key=lambda o: o["t_send"]):
        if o["turn"] == 0: continue
        j = bisect.bisect_right(fts, o["t_send"])
        la, ctx = {}, {}
        for tf, c, sz in fin[:j]:
            la[c] = tf; ctx[c] = sz - SHARED
        if o["conv"] not in la: continue
        me = la[o["conv"]]
        d = SHARED + sum(ctx[c] for c in la if la[c] >= me)
        res.append((d, tier.get((o["conv"], o["turn"]), "?"), o["t_send"]))
    return res
for arm in ("three_tier_to",):
    res = dists(arm)
    miss = sum(1 for d, tg, t in res if tg in ("storage", "cold"))
    ds = sorted((d for d, _, _ in res), reverse=True)
    ceff = ds[miss - 1]
    print(arm, "observed storage/cold", miss, "of", len(res), f"C_eff (capacity giving same miss count) = {ceff:,}  = {ceff/L2:.2f} L2")
    for extra in (0, 489604, 651464):
        C = ceff + extra
        m = sum(1 for d in ds if d > C)
        print(f"  C_eff+{extra:,} = {C:,}: predicted misses {m} ({m/len(res):.1%})")
    # rank agreement: among turns with d > ceff, share observed miss
    hi = [tg for d, tg, t in res if d > ceff]
    print("  of turns with d > C_eff, observed storage/cold:", sum(1 for x in hi if x in ('storage','cold')), "/", len(hi))
    # distance distribution of observed misses vs hits
    mm = [d for d, tg, t in res if tg in ("storage","cold")]; hh = [d for d, tg, t in res if tg in ("host","device")]
    print(f"  dist p50 observed-miss {pct(mm,.5):,.0f}  observed-hit {pct(hh,.5):,.0f}; p10 miss {pct(mm,.1):,.0f}")
