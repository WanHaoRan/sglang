"""Is an s2h_io line's time the end of the read and ms its duration? From the raw server.log:
 (a) implied start (line time - ms) against the launched srs_dispatch of the same request id just before it;
 (b) delay from the s2h_io line to the 'HiCache prefetch success' line of the same request id, and whether it tracks ms;
 (c) delay from the s2h_io line to admission (entry_time + queue_duration) for turns whose read is their last wait.
  nice -n 19 python3 -I -B r2_io_semantics.py"""
import calendar, collections, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_cache_names import RUNS, RES, CURRENT

TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] (.*)$")
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
for a in ("nofc", "nofc_noend", "ec_r13", "jit_exact", "jit_p50_ctl", "hintfree_ctl", "baseline"):
    assert RUNS[a] != CURRENT
    disp, gap, succ_delay, succ_minus_ms, pin_delay = {}, [], [], [], []
    pending = {}
    entry = {}
    last_io = {}
    neg = 0
    with open(os.path.join(RES, RUNS[a], "server.log"), errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            body = m[8]
            if not (body.startswith("HICACHE_EVT s") or body.startswith("HiCache prefetch success") or body.startswith("ReqTimeStats")):
                continue
            t = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000
            if body.startswith("HICACHE_EVT srs_dispatch"):
                rid = re.search(r"rid=(\S+)", body)[1]
                if "launched=1" in body:
                    disp[rid] = t
            elif body.startswith("HICACHE_EVT s2h_io"):
                rid = re.search(r"rid=(\S+)", body)[1]
                ms = int(re.search(r" ms=(\d+)", body)[1]) / 1000
                if rid in disp:
                    g = (t - ms) - disp.pop(rid)
                    gap.append(g)
                    neg += g < -0.003
                pending[rid] = (t, ms)
                last_io[rid] = t
            elif body.startswith("HICACHE_EVT srs_pin"):
                rid = re.search(r"rid=(\S+)", body)[1]
                if rid in pending:
                    pin_delay.append(t - pending[rid][0])
            elif body.startswith("HiCache prefetch success"):
                rid = re.search(r"req=(\S+)", body)[1]
                if rid in pending:
                    t_io, ms = pending.pop(rid)
                    succ_delay.append(t - t_io)
                    succ_minus_ms.append(t - t_io - ms)
            elif body.startswith("ReqTimeStats"):
                r = re.match(r"ReqTimeStats\(rid=([^,]+), .*queue_duration=([\d.]+)ms.*entry_time=([\d.]+)", body)
                if r:
                    entry[r[1]] = float(r[3]) + float(r[2]) / 1000
    adm = sorted(entry[r] - t for r, t in last_io.items() if r in entry)
    print(f"== {a}: reads with a launched dispatch before them {len(gap)}")
    if gap:
        print(f"   (a) implied start minus dispatch time: min {min(gap) * 1000:.0f} ms, p1 {pct(gap, .01) * 1000:.0f}, median {st.median(gap) * 1000:.0f}, p90 {pct(gap, .9) * 1000:.0f}, max {max(gap) * 1000:.0f}; negative by over 3 ms: {neg}")
    print(f"   (b) s2h_io line to 'prefetch success': n {len(succ_delay)}, median {st.median(succ_delay) * 1000:.0f} ms, p90 {pct(succ_delay, .9) * 1000:.0f}, p99 {pct(succ_delay, .99) * 1000:.0f}, max {max(succ_delay) * 1000:.0f}; "
          f"within 3 ms of the read's own ms: {sum(abs(x) < 0.003 for x in succ_minus_ms)}; over 100 ms: {sum(x > 0.1 for x in succ_delay)}")
    print(f"   (c) last s2h_io line of a request to its admission: n {len(adm)}, negative {sum(x < 0 for x in adm)}, min {adm[0] * 1000:.0f} ms, p10 {pct(adm, .1) * 1000:.0f}, median {st.median(adm) * 1000:.0f}, share under 100 ms {100 * sum(0 <= x < 0.1 for x in adm) / len(adm):.0f} %, under 20 ms {100 * sum(0 <= x < 0.02 for x in adm) / len(adm):.0f} %")
