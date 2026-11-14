"""Read-only quick look: for every 'device prefix shrank before admission' line of a run, how long the turn had been queued,
the GPU pool use and running requests at the last prefill batch before it, and whether the gate had capped its lookups."""
import datetime as dt, re, sys, statistics as st, collections
RES = "/home/wanhr/sglang/agent_cache/results"
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
RID = r"(?:rid|req)=\S+?-(c\d+-t\d+)"
ARR = re.compile(r"HICACHE_EVT ntp_arrive " + RID)
REG = re.compile(r"HICACHE_EVT nue_register " + RID + r" node=(\d+) next_use_ms=(\S+)")
SHR = re.compile(r"device prefix shrank before admission " + RID + r" lookup_match=(\d+) current_match=(\d+)")
GATE = re.compile(r"HICACHE_EVT ntp_gate " + RID + r" decision=(\w+)")
PRE = re.compile(r"Prefill batch.*token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)")
DEC = re.compile(r"Decode batch.*token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)")
EVD = re.compile(r"HICACHE_EVT evict_device tokens=(\d+) requested=(\d+)")
for label, run in (a.split("=") for a in sys.argv[1:]):
    arrive, reg, last_gate = {}, {}, {}
    pool = None
    rows = []
    evd_hist = collections.deque(maxlen=400)
    for line in open(f"{RES}/{run}/server.log", errors="replace"):
        m = TS.match(line)
        if not m:
            continue
        t = None
        if (p := PRE.search(line)) or (p := DEC.search(line)):
            pool = (float(p[1]), int(p[2]), int(p[3]))
            continue
        if "HICACHE_EVT evict_device" in line and (e := EVD.search(line)):
            evd_hist.append((stamp(m[1]), int(e[1]), int(e[2])))
            continue
        if "ntp_arrive" in line and (a := ARR.search(line)):
            arrive[a[1]] = stamp(m[1]); continue
        if "nue_register" in line and (r := REG.search(line)):
            reg[r[1]] = (stamp(m[1]), r[3]); continue
        if "ntp_gate" in line and (g := GATE.search(line)):
            last_gate[g[1]] = g[2]; continue
        if "shrank before admission" in line and (s := SHR.search(line)):
            t = stamp(m[1])
            k = s[1]
            lost = int(s[2]) - int(s[3])
            ev = [(x, tok, req) for x, tok, req in evd_hist if x >= arrive.get(k, t) - 0.001]
            rows.append(dict(k=k, t=t, age=t - arrive.get(k, float("nan")), lost=lost, pool=pool, gate_before=last_gate.get(k), reg=reg.get(k, (None, None))[1],
                             evd_tok=sum(x[1] for x in ev), evd_req=sum(x[2] for x in ev), evd_n=len(ev)))
    turns = collections.defaultdict(list)
    for r in rows:
        turns[r["k"]].append(r)
    ages = [r["age"] for r in rows if r["age"] == r["age"]]
    print(f"\n{label} ({run[:15]}): {len(rows)} shrank lines on {len(turns)} turns; tokens lost from the GPU prefix {sum(r['lost'] for r in rows):,}")
    if not rows:
        continue
    print(f"  queue age at the line, s: min {min(ages):.1f} median {st.median(ages):.1f} max {max(ages):.1f}  (the queued registration lapses at 600 s)")
    use = [r["pool"][0] for r in rows if r["pool"]]
    run_n = [r["pool"][1] for r in rows if r["pool"]]
    print(f"  GPU pool use at the last batch line before it: min {min(use):.2f} median {st.median(use):.2f} max {max(use):.2f}; running requests min {min(run_n)} median {st.median(run_n)} max {max(run_n)}")
    print(f"  gate decision last seen for the turn before the line: {dict(collections.Counter(r['gate_before'] for r in rows))}")
    print(f"  registration at arrival (next_use_ms): {dict(collections.Counter(r['reg'] for r in rows))}")
    big = sorted(turns.items(), key=lambda kv: -sum(r['lost'] for r in kv[1]))[:6]
    for k, v in big:
        print(f"    {k}: {len(v)} lines, lost {sum(r['lost'] for r in v):6d} tokens, queued {v[0]['age']:.1f}-{v[-1]['age']:.1f} s, pool {v[0]['pool'][0]:.2f}-{v[-1]['pool'][0]:.2f}, running {v[0]['pool'][1]}-{v[-1]['pool'][1]}, "
              f"device evictions since arrival up to last line: {v[-1]['evd_n']} calls, {v[-1]['evd_tok']:,} tokens (asked {v[-1]['evd_req']:,})")
