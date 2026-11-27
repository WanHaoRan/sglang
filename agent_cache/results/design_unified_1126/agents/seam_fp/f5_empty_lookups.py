"""F5 check (CPU, read-only): lookups that find nothing, the three-lookup cap, and what they cost.

Per run: s2h_query lines by result (0, below the 256-token threshold, a hit), gate decisions (ntp_gate), requests
that reached decision=capped, and from the client records the history each request computed again
(ideal_cached - cached_tokens, counted when >= 256 tokens). 'Refused by the cap' = the gate answered capped before
admission and the request was admitted with >= 256 tokens of history missing.
Also: what an empty lookup at arrival costs a request that needed no read: its queue wait against that of
requests the gate called covered (no lookup), for requests that arrived at an empty queue.
Usage: python3 f5_empty_lookups.py <label> [minute_hi]
"""
import collections, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]
hi = float(sys.argv[2]) if len(sys.argv) > 2 else 110.0
r = lib.Run(label)
TH = 256

nq = collections.Counter(); tok = collections.Counter()
for e in r.kind["s2h_query"]:
    if r.minute(e["t"]) >= hi:
        continue
    c = "empty" if e["storage_hit"] == 0 else ("below" if e["storage_hit"] < TH else "hit")
    nq[c] += 1; tok[c] += e["tokens_req"]
total = sum(nq.values())
print(f"== {label} (to minute {hi:.0f}): {total} SSD lookups: found nothing {nq['empty']} ({100*nq['empty']/max(1,total):.1f} %), "
      f"under {TH} tokens {nq['below']}, a hit {nq['hit']}; tokens asked by empty lookups {tok['empty']/1e6:.2f} M")
gd = collections.Counter(e["decision"] for e in r.kind["ntp_gate"] if r.minute(e["t"]) < hi)
print(f"   gate decisions: {dict(gd)}")

rows = []
for rid, rec in r.requests():
    rts = rec["rts"]
    if r.minute(rts["entry"]) >= hi:
        continue
    adm = rts["entry"] + rts["queue_s"]
    cl = r.client.get(rid)
    qs = [e for e in rec.get("s2h_query", []) if e["t"] <= adm + 0.05]
    gates = [e for e in rec.get("ntp_gate", []) if e["t"] <= adm + 0.05]
    capped = any(e["decision"] == "capped" for e in gates)
    lookups = max([e["lookups"] for e in gates], default=0)
    empty = sum(1 for e in qs if e["storage_hit"] < TH)
    hits = sum(1 for e in qs if e["storage_hit"] >= TH)
    ideal = cl.get("ideal_cached") if cl else None
    cached = cl.get("cached_tokens") if cl else None
    again = (ideal - cached) if ideal is not None and cached is not None else None
    arr = rec.get("ntp_arrive")
    rows.append(dict(rid=rid, capped=capped, lookups=lookups, empty=empty, hits=hits, again=again, wait=rts["queue_s"],
                     planned=bool(arr), qlen=(arr[0]["qlen"] - 1) if arr else None, pos=arr[0]["pos"] if arr else None,
                     reader=bool(rec.get("s2h_io")), gates=[e["decision"] for e in gates], nq=len(qs),
                     first_q=(qs[0]["t"] - rts["entry"]) if qs else None, m=r.minute(rts["entry"]),
                     n=rts["input_len"], cached_len=rts["cached_input_len"]))
n = len(rows)
planned = [x for x in rows if x["planned"]]
print(f"   requests {n}, planned (returning, hinted) {len(planned)}")
with_empty = [x for x in rows if x["empty"] > 0]
all3 = [x for x in planned if x["lookups"] >= 3]
all3_empty = [x for x in all3 if x["hits"] == 0]
print(f"   requests with at least one empty lookup: {len(with_empty)} ({100*len(with_empty)/max(1,n):.1f} %); "
      f"planned requests that used all 3 lookups: {len(all3)}, of them every lookup empty: {len(all3_empty)}")
capped = [x for x in rows if x["capped"]]
have = [x for x in rows if x["again"] is not None]
again_all = sum(max(0, x["again"]) for x in have if x["again"] >= TH)
refused = [x for x in capped if x["again"] is not None and x["again"] >= TH]
again_refused = sum(x["again"] for x in refused)
print(f"   gate answered capped before admission: {len(capped)} requests; admitted with >= {TH} tokens of history missing: {len(refused)} "
      f"({again_refused/1e6:.3f} M tokens of the {again_all/1e6:.3f} M computed again by all requests = {100*again_refused/max(1,again_all):.1f} %)")
if refused:
    print(f"     of these {len(refused)}: empty lookups median {lib.q([x['empty'] for x in refused],.5)}, lookups with a hit median {lib.q([x['hits'] for x in refused],.5)}, "
          f"minute median {lib.q([x['m'] for x in refused],.5):.0f} [p10 {lib.q([x['m'] for x in refused],.1):.0f}, p90 {lib.q([x['m'] for x in refused],.9):.0f}], history lost median {lib.q([x['again'] for x in refused],.5):.0f}")
    only_empty = [x for x in refused if x["hits"] == 0]
    print(f"     refused with no lookup that hit (all of the allowance went on empty lookups): {len(only_empty)} ({sum(x['again'] for x in only_empty)/1e6:.3f} M tokens)")
# by window
for lo_m, hi_m in ((0, 30), (30, 60), (60, 90), (90, 110)):
    w = [x for x in have if lo_m <= x["m"] < hi_m]
    a = sum(x["again"] for x in w if x["again"] >= TH)
    rf = [x for x in w if x["capped"] and x["again"] >= TH]
    print(f"     minutes {lo_m:>3}-{hi_m:<3}: computed again {a/1e6:.3f} M tokens ({len([x for x in w if x['again']>=TH])} requests), by cap-refused {sum(x['again'] for x in rf)/1e6:.3f} M ({len(rf)} requests)")
# cost of the empty lookup at arrival for requests that needed no read, at an empty queue
cov = [x["wait"] for x in planned if x["qlen"] == 0 and x["gates"][:1] == ["covered"] and not x["reader"] and x["nq"] == 0]
emp = [x["wait"] for x in planned if x["qlen"] == 0 and x["gates"][:1] == ["read"] and not x["reader"] and x["empty"] >= 1 and x["hits"] == 0]
print(f"   at an empty queue, planned non-readers: gate covered (no lookup) n {len(cov)} queue wait median {lib.q(cov,.5)*1e3:.1f} ms [p90 {lib.q(cov,.9)*1e3:.1f}]; "
      f"one or more empty lookups n {len(emp)} median {lib.q(emp,.5)*1e3:.1f} ms [p90 {lib.q(emp,.9)*1e3:.1f}]")
allw_emp = [x for x in planned if x["gates"][:1] == ["read"] and not x["reader"] and x["empty"] >= 1 and x["hits"] == 0]
print(f"   planned non-readers whose arrival lookup found nothing: {len(allw_emp)} ({100*len(allw_emp)/max(1,len(planned)):.1f} % of planned)")
first = [x for x in rows if not x["planned"]]
fe = [x for x in first if x["empty"] >= 1 and x["hits"] == 0]
print(f"   unplanned requests (first turns, no plan): {len(first)}; with only empty lookups {len(fe)}; with a hit {len([x for x in first if x['hits']>0])}")
