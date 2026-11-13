"""How long a launched read waits for the read thread: from the raw server.log, per s2h_io line, the time from the
launched srs_dispatch of the same request id to the read's start (line time - ms), beside the read's own duration.
Whole run and inside minutes 50-70 and 84-104 of the client's clock (the two load episodes on the leading arms' clocks).
  nice -n 19 python3 -I -B r2_read_wait.py"""
import calendar, json, glob, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_cache_names import RUNS, RES, CURRENT

TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] HICACHE_EVT (srs_dispatch|s2h_io) rid=(\S+) (.*)$")
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))]
print("arm: reads | wait for the thread after launch: mean s, median, p90, share over 0.1 s, over 1 s | own duration mean s | summed wait / summed duration | "
      "episodes (minutes 50-70, 84-104): reads, mean wait, mean duration, share over 0.1 s")
for a in ("jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "jit_p50_ctl", "hintfree_ctl"):
    assert RUNS[a] != CURRENT
    cpath = sorted(glob.glob(os.path.join(RES, RUNS[a], "client_*", "client.jsonl")))[-1]
    with open(cpath) as f:
        t0 = next(json.loads(l)["t_start"] for l in f if '"kind": "run"' in l)
    disp, rows = {}, []
    with open(os.path.join(RES, RUNS[a], "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT s" not in line:
                continue
            m = TS.match(line)
            if not m:
                continue
            t = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000
            if m[8] == "srs_dispatch":
                if "launched=1" in m[10]:
                    disp[m[9]] = t
            else:
                ms = int(re.search(r"ms=(\d+)", m[10])[1]) / 1000
                if m[9] in disp:
                    rows.append((t - t0, max(0.0, t - ms - disp.pop(m[9])), ms))
    w = [r[1] for r in rows]
    d = [r[2] for r in rows]
    ep = [r for r in rows if 50 * 60 <= r[0] < 70 * 60 or 84 * 60 <= r[0] < 104 * 60]
    print(f"  {a:13s} {len(rows):5d} | {st.mean(w):.3f} {st.median(w):.3f} {pct(w, .9):.2f} {100 * sum(x > 0.1 for x in w) / len(w):4.1f} % {100 * sum(x > 1 for x in w) / len(w):4.1f} % | {st.mean(d):.3f} | {sum(w) / sum(d):.2f} | "
          f"{len(ep):5d} {st.mean(r[1] for r in ep):.3f} {st.mean(r[2] for r in ep):.3f} {100 * sum(r[1] > 0.1 for r in ep) / len(ep):4.1f} %")
