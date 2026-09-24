"""Per returning turn: arrival-time L1+L2 match (from prefetch_start tokens) vs
admission-time cached tokens (client cached_tokens), joined with L3 query/load.
Read-only over results dirs."""
import glob, json, os, re, sys, datetime as dt
from collections import Counter, defaultdict
import statistics as st

R = "/home/wanhr/sglang/agent_cache/results"
RUNS = {
    "x70": {"hbm_host": "20260922_023714_hbm_host_NAT160", "three_tier_to": "20260922_051155_three_tier_to_NAT160",
            "three_tier_wc": "20260922_064515_three_tier_wc_NAT160", "hbm_lru": "20260922_081425_hbm_lru_NAT160"},
    "x10": {"hbm_host": "20260922_165655_hbm_host_NAT160", "three_tier_to": "20260922_192912_three_tier_to_NAT160",
            "three_tier_wc": "20260922_220250_three_tier_wc_NAT160", "hbm_lru": "20260923_003655_hbm_lru_NAT160"},
    "x1": {"hbm_host": "20260923_031116_hbm_host_NAT160", "three_tier_to": "20260923_053724_three_tier_to_NAT160",
           "three_tier_wc": "20260923_080528_three_tier_wc_NAT160", "hbm_lru": "20260923_103328_hbm_lru_NAT160"},
}
TS = r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] "
P_START = re.compile(TS + r"HICACHE_EVT prefetch_start rid=(\S+) tokens=(\d+)")
S2H_Q = re.compile(TS + r"HICACHE_EVT s2h_query rid=(\S+) tokens_req=(\d+) storage_hit=(\d+)")
PF_OK = re.compile(TS + r"HiCache prefetch (\w+) req=(\S+) completed=(\d+) matched=(\d+) loaded=(\d+)")
LBI = re.compile(TS + r"HICACHE_EVT load_back_init rid=(\S+) tokens=(\d+) host_hit=(\d+)")
RTS = re.compile(TS + r"ReqTimeStats\(rid=([^,]+), input_len=(\d+), cached_input_len=(\d+).*queue_duration=([0-9.]+)(ms|s), forward_duration=([0-9.]+)(ms|s), entry_time=([0-9.]+)")
EVH = re.compile(TS + r"HICACHE_EVT evict_host tokens=(\d+) requested=(\d+)")


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()


def load(scale, arm):
    run = os.path.join(R, RUNS[scale][arm])
    ev = defaultdict(dict)
    evict = []
    with open(os.path.join(run, "server.log"), errors="replace") as f:
        for line in f:
            if "prefetch_start" in line:
                m = P_START.match(line)
                if m:
                    ev[m[2]]["pf_t"] = ts(m[1]); ev[m[2]]["pf_tok"] = int(m[3])
            elif "s2h_query" in line:
                m = S2H_Q.match(line)
                if m:
                    ev[m[2]]["q_t"] = ts(m[1]); ev[m[2]]["q_hit"] = int(m[4]); ev[m[2]]["q_req"] = int(m[3])
            elif "HiCache prefetch" in line:
                m = PF_OK.match(line)
                if m:
                    ev[m[3]]["ld_t"] = ts(m[1]); ev[m[3]]["loaded"] = int(m[6]); ev[m[3]]["pf_kind"] = m[2]
            elif "load_back_init" in line:
                m = LBI.match(line)
                if m:
                    ev[m[2]]["lb_t"] = ts(m[1]); ev[m[2]]["lb_tok"] = int(m[3])
            elif "ReqTimeStats" in line:
                m = RTS.match(line)
                if m:
                    q = float(m[5]) / (1000 if m[6] == "ms" else 1)
                    ev[m[2]].update(rts_in=int(m[3]), rts_cached=int(m[4]), rts_q=q, rts_entry=float(m[9]))
            elif "evict_host" in line:
                m = EVH.match(line)
                if m:
                    evict.append((ts(m[1]), int(m[2]), int(m[3])))
    cl = glob.glob(os.path.join(run, "client_*", "client.jsonl"))[0]
    turns = []
    t0 = None
    with open(cl) as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "run":
                t0 = r["t_start"]
            elif r.get("kind") == "turn":
                turns.append(r)
    return run, ev, evict, turns, t0


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo = int(k); hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    scales = sys.argv[1].split(",") if len(sys.argv) > 1 else ["x70", "x10", "x1"]
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["three_tier_to", "three_tier_wc"]
    for scale in scales:
        for arm in arms:
            run, ev, evict, turns, t0 = load(scale, arm)
            rows = []
            for r in turns:
                if r["turn"] < 1 or r.get("ttft") is None:
                    continue
                rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
                e = ev.get(rid, {})
                if "pf_tok" not in e:
                    continue
                prompt = r["prompt_tokens"]
                A = prompt - 1 - e["pf_tok"]  # arrival-time L1+L2 matched length
                C = r.get("cached_tokens") or 0
                cd = r.get("cached_details") or {}
                loaded = e.get("loaded", 0)
                qhit = e.get("q_hit")
                q = e.get("rts_q")
                if q is None and "lb_t" in e:
                    q = e["lb_t"] - e["pf_t"]
                rows.append(dict(rid=rid, prompt=prompt, A=A, C=C, tail=e["pf_tok"], loaded=loaded, qhit=qhit,
                                 dev=cd.get("device", 0), host=cd.get("host", 0), sto=cd.get("storage", 0), q=q,
                                 t_send=r["t_send"], lost=max(0, A + loaded - C)))
            n = len(rows)
            lost = [x for x in rows if x["lost"] >= 1024]
            print(f"\n=== {scale} {arm}  ({os.path.basename(run)})  returning turns with prefetch_start: {n}")
            if not n:
                continue
            print(f"  arrival match A: p50 {pct([x['A'] for x in rows],50):.0f}  mean {st.mean(x['A'] for x in rows):.0f};  "
                  f"admitted cached C: p50 {pct([x['C'] for x in rows],50):.0f} mean {st.mean(x['C'] for x in rows):.0f};  prompt mean {st.mean(x['prompt'] for x in rows):.0f}")
            print(f"  arrival unmatched tail: <256: {sum(x['tail']<256 for x in rows)}  256..1023: {sum(256<=x['tail']<1024 for x in rows)}  >=1024: {sum(x['tail']>=1024 for x in rows)}")
            qh = [x for x in rows if x["qhit"] is not None]
            print(f"  s2h_query issued: {len(qh)}; storage_hit>=256: {sum(x['qhit']>=256 for x in qh)}; loaded>0: {sum(x['loaded']>0 for x in rows)} (sum loaded {sum(x['loaded'] for x in rows)})")
            print(f"  turns losing >=1024 tokens between arrival and admission: {len(lost)} ({100*len(lost)/n:.1f}%), lost tokens total {sum(x['lost'] for x in lost)} = {100*sum(x['lost'] for x in lost)/max(1,sum(x['A']+x['loaded'] for x in rows)):.1f}% of arrival-time reusable tokens")
            a = [x for x in lost if x["loaded"] == 0]
            b = [x for x in lost if x["loaded"] > 0]
            print(f"    (a) no L3 load for this turn (arrival L1/L2 match lost): {len(a)}  lost tokens {sum(x['lost'] for x in a)}")
            print(f"        of which arrival tail <256 (prefetch declined, anchor never locked): {sum(x['tail']<256 for x in a)}; tail>=256 & L3 query hit<256 (anchor lock dropped at query drain): {sum(x['tail']>=256 and (x['qhit'] or 0)<256 for x in a)}")
            print(f"    (b) L3 load>0 and loaded span not fully used at admission: {len(b)}  lost tokens {sum(x['lost'] for x in b)}; of these lost > loaded (also arrival match lost): {sum(x['lost']>x['loaded'] for x in b)}")
            fulllost = [x for x in lost if x["C"] <= 7808 + 64]
            print(f"    lost-turns admitted with only <= shared prompt (C<=7872): {len(fulllost)};  C==0: {sum(x['C']==0 for x in lost)}")
            if lost:
                print(f"    lost-turn A p50 {pct([x['A'] for x in lost],50):.0f}, C p50 {pct([x['C'] for x in lost],50):.0f}, tail p50 {pct([x['tail'] for x in lost],50):.0f}")
            qk = [x["q"] for x in rows if x["q"] is not None and x["lost"] < 1024]
            ql = [x["q"] for x in lost if x["q"] is not None]
            print(f"  queue wait (s) kept: n={len(qk)} p50 {pct(qk,50):.1f} p90 {pct(qk,90):.1f};  lost: n={len(ql)} p50 {pct(ql,50):.1f} p10 {pct(ql,10):.1f} min {min(ql) if ql else float('nan'):.1f}")
            # loss vs queue-time buckets
            bk = [(0, 1), (1, 10), (10, 30), (30, 60), (60, 120), (120, 1e9)]
            s = []
            for lo, hi in bk:
                g = [x for x in rows if x["q"] is not None and lo <= x["q"] < hi]
                if g:
                    s.append(f"[{lo},{hi if hi<1e9 else 'inf'}): {sum(x['lost']>=1024 for x in g)}/{len(g)}")
            print("  lost/total by queue wait:", "  ".join(s))
            # host eviction rate
            if evict:
                tot = sum(t for _, t, _ in evict)
                span = evict[-1][0] - evict[0][0]
                short = sum(1 for _, t, rq in evict if t < rq)
                print(f"  evict_host: events {len(evict)}, tokens {tot}, span {span:.0f}s, mean rate {tot/max(1,span):.0f} tok/s -> 1,627,648-token pool turned over in {1627648/max(1e-9,tot/max(1,span)):.0f}s; events with tokens<requested: {short}")


if __name__ == "__main__":
    main()
