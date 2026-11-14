"""Quick probe (read-only): rule 1's read-time estimate at a 'now' decision against the measured time from staging to
the read's pages being pinned in host memory (first srs_pin after ntp_stage for the same rid). Not a verified analysis."""
import datetime as dt, os, re, sys, statistics as st

RES = "/home/wanhr/sglang/agent_cache/results"
LINE = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_start|ntp_stage|srs_pin|s2h_io) rid=(\S+) (.*)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
for run in sys.argv[1:]:
    est, staged_at, rows, tok = {}, {}, [], {}
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT" not in line:
                continue
            m = LINE.match(line)
            if not m:
                continue
            t, kind, rid, rest = stamp(m[1]), m[2], m[3], m[4]
            if kind == "ec_start" and "decision=now" in rest:
                kv = dict(p.split("=", 1) for p in rest.split() if "=" in p)
                est[rid] = float(kv["read_ms"]) / 1000
                tok[rid] = int(kv["tokens"])
            elif kind == "ntp_stage":
                staged_at[rid] = t
            elif kind == "srs_pin" and rid in staged_at and rid in est:
                rows.append((est[rid], t - staged_at.pop(rid), tok[rid]))
    e = [r[0] for r in rows]; a = [r[1] for r in rows]
    print(f"{run}: staged reads with a landing {len(rows)}")
    print(f"  estimate used by rule 1, s: p10 {pct(e,.1):.2f} p50 {pct(e,.5):.2f} p90 {pct(e,.9):.2f}")
    print(f"  measured stage -> pinned, s: p10 {pct(a,.1):.2f} p50 {pct(a,.5):.2f} p90 {pct(a,.9):.2f}")
    d = [x - y for x, y, _ in rows]
    print(f"  estimate minus measured, s: p10 {pct(d,.1):.2f} p50 {pct(d,.5):.2f} p90 {pct(d,.9):.2f}; estimate above measured in {sum(1 for x in d if x > 0)} of {len(d)}")
    for lo, hi in ((0, 4000), (4000, 16000), (16000, 32000), (32000, 10**9)):
        sub = [r for r in rows if lo <= r[2] < hi]
        if sub:
            print(f"    tokens {lo:6d}-{hi if hi < 10**9 else 'up':>6}: n {len(sub):4d} | estimate p50 {pct([r[0] for r in sub], .5):.2f} | measured p50 {pct([r[1] for r in sub], .5):.2f} | deck r = bL/beta at the median size {st.median(r[2] for r in sub) * 98304 / 2.11e9:.2f}")
