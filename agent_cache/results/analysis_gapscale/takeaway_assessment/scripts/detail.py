"""Details: arrival==prefetch timing, anchor-lock hold time, eviction volume
during the queue wait, class-(b) timing, x70 storage-hit critical path,
h2s backlog, device token usage."""
import bisect, re, sys, os, glob, json
sys.path.insert(0, os.path.dirname(__file__))
from arrival_loss import load, pct, TS, ts
import statistics as st

H2S_SUB = re.compile(TS + r"HICACHE_EVT h2s_submit op=(\d+) node=\d+ tokens=(\d+)")
H2S_DONE = re.compile(TS + r"HICACHE_EVT h2s_done op=(\d+) tokens=(\d+)")
PB = re.compile(TS + r"Prefill batch.*token usage: ([0-9.]+), #running-req: (\d+), #queue-req: (\d+)")


def extra(run):
    sub, done, pb = {}, {}, []
    with open(os.path.join(run, "server.log"), errors="replace") as f:
        for line in f:
            if "h2s_submit" in line:
                m = H2S_SUB.match(line)
                if m: sub[int(m[2])] = (ts(m[1]), int(m[3]))
            elif "h2s_done" in line:
                m = H2S_DONE.match(line)
                if m: done[int(m[2])] = ts(m[1])
            elif "Prefill batch" in line:
                m = PB.match(line)
                if m: pb.append((ts(m[1]), float(m[2]), int(m[3]), int(m[4])))
    return sub, done, pb


def main():
    for scale, arm in [("x70", "three_tier_to"), ("x70", "three_tier_wc"), ("x10", "three_tier_to"), ("x1", "three_tier_to"), ("x1", "three_tier_wc")]:
        run, ev, evict, turns, t0 = load(scale, arm)
        print(f"\n=== {scale} {arm}")
        et = [t for t, _, _ in evict]
        cum = [0]
        for _, tok, _ in evict:
            cum.append(cum[-1] + tok)

        def evicted_between(a, b):
            i, j = bisect.bisect_left(et, a), bisect.bisect_right(et, b)
            return cum[j] - cum[i]
        # arrival timing: prefetch_start vs scheduler wait-queue entry (ReqTimeStats entry_time)
        d = [abs(e["pf_t"] - e["rts_entry"]) for r, e in ev.items() if r.startswith("cmp_") and "pf_t" in e and "rts_entry" in e]
        if d:
            print(f"  |prefetch_start - wait-queue entry_time|: n={len(d)} p50 {pct(d,50)*1000:.1f} ms p99 {pct(d,99)*1000:.1f} ms max {max(d)*1000:.0f} ms (log ts ms resolution)")
        # anchor lock hold for L3-miss queries: prefetch_start -> s2h_query
        h = [e["q_t"] - e["pf_t"] for r, e in ev.items() if r.startswith("cmp_") and "q_t" in e and e.get("q_hit", 0) < 256]
        print(f"  prefetch_start -> s2h_query (L3 miss; anchor lock released at next drain): n={len(h)} p50 {pct(h,50)*1000:.0f} ms p99 {pct(h,99)*1000:.0f} ms")
        rows = []
        for r in turns:
            if r["turn"] < 1:
                continue
            rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
            e = ev.get(rid, {})
            if "pf_tok" not in e:
                continue
            A = r["prompt_tokens"] - 1 - e["pf_tok"]
            C = r.get("cached_tokens") or 0
            loaded = e.get("loaded", 0)
            adm = e["rts_entry"] + e["rts_q"] if "rts_q" in e else (e.get("lb_t"))
            rows.append(dict(e=e, A=A, C=C, loaded=loaded, lost=max(0, A + loaded - C), adm=adm, ttft=r["ttft"], r=r))
        m = [x for x in rows if x["adm"] is not None]
        if m and scale == "x1":
            lost = [evicted_between(x["e"]["pf_t"], x["adm"]) for x in m if x["lost"] >= 1024]
            kept = [evicted_between(x["e"]["pf_t"], x["adm"]) for x in m if x["lost"] < 1024]
            print(f"  host tokens evicted between arrival and admission: lost turns n={len(lost)} p10 {pct(lost,10):.0f} p50 {pct(lost,50):.0f};  kept turns n={len(kept)} p50 {pct(kept,50):.0f} p90 {pct(kept,90):.0f} max {max(kept):.0f}")
            for lo, hi in [(0, 1e5), (1e5, 2e5), (2e5, 3e5), (3e5, 5e5), (5e5, 1e6), (1e6, 1e12)]:
                g = [x for x in m if lo <= evicted_between(x["e"]["pf_t"], x["adm"]) < hi]
                if g:
                    print(f"    evicted-during-wait [{lo:.0e},{hi:.0e}): lost {sum(x['lost']>=1024 for x in g)}/{len(g)}")
        # class (b): L3-loaded but not fully used
        b = [x for x in rows if x["loaded"] > 0 and x["lost"] >= 1024]
        if b:
            gaps = [(x["adm"] - x["e"]["ld_t"]) for x in b if x["adm"] is not None and "ld_t" in x["e"]]
            print(f"  class (b) n={len(b)}: loaded p50 {pct([x['loaded'] for x in b],50):.0f}, lost p50 {pct([x['lost'] for x in b],50):.0f}, lost>=loaded (whole prefetched span gone): {sum(x['lost']>=x['loaded']-64 for x in b)}; insert->admission gap measured n={len(gaps)} p50 {pct(gaps,50):.1f}s")
        ok = [x for x in rows if x["loaded"] > 0 and x["lost"] < 1024]
        if ok:
            l3 = [x["e"]["ld_t"] - x["e"]["pf_t"] for x in ok if "ld_t" in x["e"]]
            adm = [x["adm"] - x["e"]["ld_t"] for x in ok if x["adm"] is not None and "ld_t" in x["e"]]
            print(f"  L3-loaded & used n={len(ok)}: arrival->L3 insert p50 {pct(l3,50):.2f}s mean {st.mean(l3):.2f}s; L3 insert->admission p50 {pct(adm,50):.2f}s (n={len(adm)}); TTFT p50 {pct([x['ttft'] for x in ok],50):.2f}s")
            oth = [x["ttft"] for x in rows if x["loaded"] == 0 and x["lost"] < 1024]
            print(f"  returning turns without L3 load: TTFT p50 {pct(oth,50):.2f}s mean {st.mean(oth):.2f}s")
        sub, done, pb = extra(run)
        # h2s backlog (tokens host-locked by pending L3 writes)
        evs = sorted([(t, tok) for op, (t, tok) in sub.items()] + [(done[op], -sub[op][1]) for op in sub if op in done])
        cur = mx = 0
        samples = []
        for t, dlt in evs:
            cur += dlt; mx = max(mx, cur); samples.append(cur)
        tot = sum(tok for _, tok in sub.values())
        dur = [done[op] - sub[op][0] for op in sub if op in done]
        print(f"  h2s: submitted {len(sub)} ops {tot} tok; pending-L3-write backlog max {mx} tok, p50 {pct(samples,50):.0f}; submit->done p50 {pct(dur,50):.2f}s p99 {pct(dur,99):.2f}s")
        if pb:
            sat = [x for x in pb if x[3] >= 10]
            if sat:
                print(f"  device token usage while queue>=10: n={len(sat)} p50 {pct([x[1] for x in sat],50):.2f} (x 668,160 = {pct([x[1] for x in sat],50)*668160:.0f} tok); queue-req p50 {pct([x[3] for x in sat],50):.0f}")
            # saturated eviction rate
            if sat and evict:
                a, bnd = sat[0][0], sat[-1][0]
                vol = sum(tok for t, tok, _ in evict if a <= t <= bnd)
                print(f"  evict_host rate over queue>=10 window ({bnd-a:.0f}s): {vol/(bnd-a):.0f} tok/s")


if __name__ == "__main__":
    main()
