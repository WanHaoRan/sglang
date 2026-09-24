#!/usr/bin/env python3
import csv, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = 7808
rows = list(csv.DictReader(open(os.path.join(HERE, "t2_turns.csv"))))
def f(x):
    return None if x in ("", None) else float(x)
for r in rows:
    for k in ("P", "C", "dev", "host", "sto", "T", "m_arr", "q_req", "q_hit", "io_pages", "pf_loaded", "lb_host_hit", "rts_input", "rts_cached", "queue", "t_send", "ttft", "entry", "adm", "delta"):
        r[k] = f(r[k])
def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p / 100 * len(v)))] if v else None
def cls(r):
    P, C, m = r["P"], r["C"], r["m_arr"]
    if m is None: return "no_arrival_evt"
    arrive_ok = m >= P - 1 - 1024        # everything but <=1K-token new tail matched in L1+L2 at arrival
    lost = m - C                           # tokens matched at arrival but not cached at admission
    if arrive_ok and lost <= 64: return "kept"
    if arrive_ok and C <= SHARED + 64: return "L12_arrival_full_loss"
    if arrive_ok and lost > 64: return "L12_arrival_partial_loss"
    # not (mostly) in L1+L2 at arrival
    if C >= P - 1 - 1024: return "missing_at_arrival_recovered"
    return "missing_at_arrival_recompute"

for scale in ("x10", "x1"):
    for arm in ("three_tier_to", "three_tier_wc"):
        R = [r for r in rows if r["scale"] == scale and r["arm"] == arm]
        n = len(R)
        print(f"\n===== {scale} {arm}: returning turns n={n}")
        # sanity: P vs rts input
        bad = [r for r in R if r["rts_input"] is not None and r["rts_input"] != r["P"]]
        print(f"  P!=rts_input: {len(bad)}; rts_cached!=C: {sum(1 for r in R if r['rts_cached'] is not None and r['rts_cached']!=r['C'])}; n_ps>1: {sum(1 for r in R if r['n_ps'] not in ('1',1,'1.0'))}")
        fracA = [r["m_arr"] / r["P"] for r in R if r["m_arr"] is not None]
        fracC = [r["C"] / r["P"] for r in R]
        print("  matched_at_arrival/P  p10/p25/p50/p75/p90:", [round(pct(fracA, p), 3) for p in (10, 25, 50, 75, 90)], " mean", round(st.mean(fracA), 3))
        print("  cached_at_admission/P p10/p25/p50/p75/p90:", [round(pct(fracC, p), 3) for p in (10, 25, 50, 75, 90)], " mean", round(st.mean(fracC), 3))
        sumP = sum(r["P"] for r in R); sumM = sum(r["m_arr"] for r in R if r["m_arr"] is not None); sumC = sum(r["C"] for r in R)
        print(f"  token-weighted: matched_at_arrival {sumM/sumP:.3f} of prompt tokens, cached_at_admission {sumC/sumP:.3f}")
        cnt = {}
        for r in R: cnt.setdefault(cls(r), []).append(r)
        for k in ("kept", "L12_arrival_partial_loss", "L12_arrival_full_loss", "missing_at_arrival_recovered", "missing_at_arrival_recompute", "no_arrival_evt"):
            v = cnt.get(k, [])
            lost = sum(r["m_arr"] - r["C"] for r in v if r["m_arr"] is not None)
            print(f"  {k:30s} n={len(v):5d} ({len(v)/n*100:5.1f}%) lost_tokens_sum={lost:,.0f}")
        rec = [r for r in R if r["C"] < 0.5 * r["P"]]
        rec_L12 = [r for r in rec if r["m_arr"] is not None and r["m_arr"] >= r["P"] - 1 - 1024]
        rec_L12b = [r for r in rec if r["m_arr"] is not None and r["m_arr"] >= 0.9 * r["P"]]
        print(f"  recomputed (C<0.5P): n={len(rec)} ({len(rec)/n*100:.1f}%); of these matched>=P-1025 at arrival: {len(rec_L12)} ({len(rec_L12)/max(1,len(rec))*100:.1f}%); matched>=0.9P: {len(rec_L12b)} ({len(rec_L12b)/max(1,len(rec))*100:.1f}%)")
        onlyshared = [r for r in rec if r["C"] <= SHARED + 64]
        print(f"  recomputed with C<=shared prompt(7808+64): {len(onlyshared)}")
        # tokens: total prefill due to loss
        tot_unc = sum(r["P"] - r["C"] for r in R); tot_loss = sum(max(0, r["m_arr"] - r["C"]) for r in R if r["m_arr"] is not None)
        tot_newtail = sum(r["P"] - r["m_arr"] for r in R if r["m_arr"] is not None)
        print(f"  uncached tokens prefilled total={tot_unc:,.0f}; of which lost-after-arrival={tot_loss:,.0f} ({tot_loss/tot_unc*100:.1f}%); not-matched-at-arrival (new tail + missing)={tot_newtail:,.0f}")
        # L3 activity
        qd = [r for r in R if r["q_req"] is not None]
        qhit = [r for r in qd if r["q_hit"] and r["q_hit"] > 0]
        io = [r for r in R if r["io_pages"] and r["io_pages"] > 0]
        print(f"  s2h_query issued: {len(qd)} (T>=256 page-aligned); storage_hit>0: {len(qhit)}; s2h_io (real L3 read): {len(io)}; pf_loaded>0: {sum(1 for r in R if r['pf_loaded'] and r['pf_loaded']>0)}")
        Tv = [r["T"] for r in R if r["T"] is not None]
        print(f"  arrival unmatched tail T: p50={pct(Tv,50):.0f} p90={pct(Tv,90):.0f}; T<256: {sum(1 for t in Tv if t<256)/len(Tv)*100:.1f}% ; T<1024: {sum(1 for t in Tv if t<1024)/len(Tv)*100:.1f}%")
        for r in io:
            lost = r["m_arr"] + r["pf_loaded"] - r["C"]
            print(f"    L3 read c{int(r['conv'])}-t{int(r['turn'])}: P={r['P']:.0f} m_arr={r['m_arr']:.0f} T={r['T']:.0f} hit={r['q_hit']:.0f} pages={r['io_pages']:.0f} loaded={r['pf_loaded']:.0f} C={r['C']:.0f} (dev={r['dev']:.0f} host={r['host']:.0f} sto={r['sto']:.0f}) lost_after_prefetch={lost:.0f} queue={r['queue']} ({r['qhow']}) ttft={r['ttft']:.1f}")
        # storage_hit>0 but no io
        for r in qhit:
            if not (r["io_pages"] and r["io_pages"] > 0):
                print(f"    hit-without-io c{int(r['conv'])}-t{int(r['turn'])}: hit={r['q_hit']:.0f} T={r['T']:.0f} C={r['C']:.0f}")
