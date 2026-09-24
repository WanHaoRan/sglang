#!/usr/bin/env python3
"""(3) loss vs server queue wait (c9) and (4) host-eviction volume/rate vs queue wait."""
import csv, os, pickle, bisect, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = 7808
HOST = 1627648; DEV = 668160
D = pickle.load(open(os.path.join(HERE, "t2.pkl"), "rb"))
rows = list(csv.DictReader(open(os.path.join(HERE, "t2_turns.csv"))))
num = lambda x: None if x in ("", None) else float(x)
for r in rows:
    for k in list(r):
        if k not in ("scale", "arm", "qhow"): r[k] = num(r[k])

def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else float("nan")

for scale in ("x1", "x10"):
    for arm in ("three_tier_to", "three_tier_wc"):
        run = D[(scale, arm)]; t0 = run["t0"]
        eh = run["glob"]["evict_host"]; et = [x[0] for x in eh]; cum = [0]
        for x in eh: cum.append(cum[-1] + x[1])
        ed = run["glob"]["evict_device"]; edt = [x[0] for x in ed]; cumd = [0]
        for x in ed: cumd.append(cumd[-1] + x[1])
        def hev(a, b): return cum[bisect.bisect_right(et, b)] - cum[bisect.bisect_right(et, a)]
        def dev_ev(a, b): return cumd[bisect.bisect_right(edt, b)] - cumd[bisect.bisect_right(edt, a)]
        R = [r for r in rows if r["scale"] == scale and r["arm"] == arm]
        print(f"\n===== {scale} {arm}")
        first_ev = eh[0][0] - t0 if eh else None
        tend = max(x[0] for x in run["glob"]["batch"]) - t0
        print(f"  first evict_host at t={first_ev:.0f}s; run end t={tend:.0f}s; total host evicted={cum[-1]:,} tokens; device evicted={cumd[-1]:,}")
        # per-minute host eviction rate
        rates = []
        for m in range(0, int(tend // 60) + 1):
            a = t0 + 60 * m; b = a + 60
            rates.append(hev(a, b) / 60.0)
        steady = [x for x in rates[10:-10]] if len(rates) > 25 else rates
        print(f"  host-evict rate tok/s by minute: first 12 = {[int(x) for x in rates[:12]]}")
        print(f"  steady (min 10..end-10): mean={st.mean(steady):,.0f} p50={pct(steady,50):,.0f} p10={pct(steady,10):,.0f} p90={pct(steady,90):,.0f}")
        # measured windows: need entry & adm
        W = [r for r in R if r["entry"] and r["adm"] and r["m_arr"] is not None]
        if scale == "x1":
            W = [r for r in W if r["qhow"] == "time_stats"]
        for r in W:
            r["H"] = hev(r["entry"], r["adm"]); r["Dv"] = dev_ev(r["entry"], r["adm"])
            r["lossfrac"] = max(0.0, r["m_arr"] - r["C"]) / max(1.0, r["m_arr"] - SHARED)
            r["lost"] = (r["m_arr"] - r["C"]) > 64 and r["m_arr"] >= r["P"] - 1 - 1024
            r["eligible"] = r["m_arr"] >= r["P"] - 1 - 1024 and r["m_arr"] > SHARED + 256
        E = [r for r in W if r["eligible"]]
        # server queue distribution
        qv = [r["adm"] - r["entry"] for r in E]
        print(f"  eligible turns (context in L1+L2 at arrival, private>256): {len(E)}; server queue (entry->adm) p10/p50/p90 = {pct(qv,10):.1f}/{pct(qv,50):.1f}/{pct(qv,90):.1f}s  ({'ReqTimeStats' if scale=='x1' else 'x10: adm from load_back_init or t_send+estimated queue; entry=prefetch_start'})")
        bins = [0, 1, 5, 10, 20, 30, 45, 60, 75, 90, 105, 120, 150, 1e9]
        print("  queue bin (s)   n    lost%   mean_lossfrac  hostEvicted_in_wait p50 (tok)  devEvicted p50")
        for a, b in zip(bins, bins[1:]):
            v = [r for r in E if a <= r["adm"] - r["entry"] < b]
            if not v: continue
            print(f"  [{a:>4},{b if b<1e9 else 'inf':>4})  {len(v):5d}  {sum(r['lost'] for r in v)/len(v)*100:6.1f}  {st.mean(r['lossfrac'] for r in v):8.3f}   {pct([r['H'] for r in v],50):>14,.0f}   {pct([r['Dv'] for r in v],50):>12,.0f}")
        hb = [0, 100e3, 250e3, 500e3, 750e3, 900e3, 1.0e6, 1.1e6, 1.25e6, 1.5e6, 2e6, 1e12]
        print("  host tokens evicted during wait   n   lost%")
        for a, b in zip(hb, hb[1:]):
            v = [r for r in E if a <= r["H"] < b]
            if not v: continue
            print(f"  [{a/1e3:>6.0f}K,{(b/1e3 if b<1e12 else float('inf')):>6.0f}K)  {len(v):5d}  {sum(r['lost'] for r in v)/len(v)*100:6.1f}")
        # time course: lost% by 10-min window of arrival
        print("  arrival window  n  lost%  mean server queue")
        for m in range(0, int(tend // 600) + 1):
            v = [r for r in E if 600 * m <= r["entry"] - t0 < 600 * (m + 1)]
            if not v: continue
            print(f"   [{m*10:3d},{m*10+10:3d}) min  {len(v):4d}  {sum(r['lost'] for r in v)/len(v)*100:5.1f}  {st.mean(r['adm']-r['entry'] for r in v):6.1f}s")
        # H/queue ratio as effective turnover estimate among lost turns
        lost = [r for r in E if r["lost"]]; kept = [r for r in E if not r["lost"]]
        print(f"  lost turns: n={len(lost)} host-evicted-in-wait p10/p50/p90 = {pct([r['H'] for r in lost],10):,.0f}/{pct([r['H'] for r in lost],50):,.0f}/{pct([r['H'] for r in lost],90):,.0f}")
        print(f"  kept turns: n={len(kept)} host-evicted-in-wait p10/p50/p90 = {pct([r['H'] for r in kept],10):,.0f}/{pct([r['H'] for r in kept],50):,.0f}/{pct([r['H'] for r in kept],90):,.0f}")
