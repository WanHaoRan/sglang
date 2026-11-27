"""ev_fetch 'F3 and F2 together': what the engine's expected-cost start would say for the stalled readers of
campaign 34 ec_full if it were given the true exposed read time (x = the reader's measured stall) and today's
stall weight of 1.  read_start and the two-piece lognormal come from the engine's next_turn_policy.py, loaded by
path (pure Python).  The entry is the hint the client sent with the previous request (p10/p50/p90)."""
import sys, os, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("ntp", "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py")
ntp = importlib.util.module_from_spec(spec); sys.modules["ntp"] = ntp; spec.loader.exec_module(ntp)
import vlib, runtab
from vlib import mean
for arm in ("ec_full",):
    tab = runtab.build(34, arm)
    reqs, reads, stall = tab["reqs"], tab["reads"], tab["stall"]
    hint_for = {}
    for rid, q in reqs.items():
        h = q.get("hint")
        if h and h.get("next_rid"):
            hint_for[h["next_rid"]] = h
    for lo, hi in [(70, 110), (70, 90), (90, 110), (30, 70)]:
        rd = [(rid, q) for rid, q in reqs.items() if q["k"] >= 1 and lo * 60 <= q["t_send"] < hi * 60 and rid in stall and stall[rid]["stall"] > 0.05]
        ret = sum(1 for q in reqs.values() if q["k"] >= 1 and lo * 60 <= q["t_send"] < hi * 60)
        print(f"c34 {arm} minutes {lo}-{hi}: stalled readers {len(rd)} of {ret} returning requests; with a table entry sent: {sum(1 for rid, q in rd if rid in hint_for and hint_for[rid].get('p50') is not None)}")
        for lam in (2e-5, 5e-5, 1e-4):
            now = 0; credit = 0.0; pin = 0.0; n = 0
            for rid, q in rd:
                h = hint_for.get(rid)
                if not h or h.get("p50") is None: continue
                n += 1
                x = stall[rid]["stall"]
                tokens = sum(reads[i]["pages"] * 64 for i in q["reads"] if not reads[i].get("dead"))
                dist = ntp.GapDistribution.from_quantiles(p10=h["p10"], p50=h["p50"], p90=h["p90"])
                sup = dist.support()
                gap = q["wait_s"] or 0.0
                idle = 0.0; staged_at = None
                while idle < gap:
                    rem = ntp.remaining_gaps(support=sup, idle_s=idle)
                    s = ntp.read_start(remaining=rem, read_s=x, wait_s=0.0, rent_per_s=lam * tokens)
                    if s is not None and s <= 0.25:
                        staged_at = idle; break
                    if s is None and not rem: break
                    idle += 0.5
                if staged_at is not None:
                    now += 1
                    lead = gap - staged_at            # the read started this long before the arrival
                    credit += min(lead, x)            # stall the rule's own accounting removes
                    pin += max(lead - x, 0.0) * tokens
            print(f"   price {lam:g}: the rule stages a read before arrival for {now} of {n} ({100*now/max(n,1):.0f} %; {100*now/len(rd):.0f} % of all stalled readers); "
                  f"its own accounting: stall removed {credit/ret:.2f} s per returning request, rent {lam*pin/ret:.2f} s")
