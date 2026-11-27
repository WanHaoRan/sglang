"""SSD (vdc) by arm from iostat -dxt logs: throughput and latency, to compare the volume before and after the host moved.
usage: ssd_unit.py <campaign folder> [...]   (arm windows from launcher.log LAUNCH/DONE lines)"""
import sys, re, datetime as dt, statistics as st
def q(v, p): return sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
for c in sys.argv[1:]:
    win = {}
    for l in open(f"{c}/launcher.log"):
        m = re.match(r"C\d+ (\S+) (LAUNCH|DONE) (\S+)Z", l)
        if m:
            win.setdefault(m[1], [None, None])[0 if m[2] == "LAUNCH" else 1] = dt.datetime.fromisoformat(m[3])
    samples, ts, hdr = [], None, None
    for l in open(f"{c}/iostat_vdc.log"):
        if re.match(r"\d\d/\d\d/\d{4} ", l):
            ts = dt.datetime.strptime(l.strip(), "%m/%d/%Y %I:%M:%S %p")
        elif l.startswith("Device"):
            hdr = l.split()
        elif l.startswith("vdc") and hdr and ts:
            samples.append((ts, dict(zip(hdr[1:], map(float, l.split()[1:])))))
    for arm, (a, b) in win.items():
        if not (a and b):
            continue
        s = [x for t, x in samples if a + dt.timedelta(minutes=5) <= t <= b]
        rd = [x["rkB/s"] / 1024 for x in s]; wr = [x["wkB/s"] / 1024 for x in s]
        busy = [x for x in s if x["%util"] >= 95]
        tot_busy = [(x["rkB/s"] + x["wkB/s"]) / 1024 for x in busy]
        ra = [x["r_await"] for x in s if x["rkB/s"] / 1024 >= 200]; wa = [x["w_await"] for x in s if x["wkB/s"] / 1024 >= 200]
        print(f"{c[-10:]} {arm:10s} n={len(s):4d} read MB/s p50 {q(rd,.5):6.0f} p90 {q(rd,.9):6.0f} max {max(rd):6.0f} | write p50 {q(wr,.5):6.0f} p90 {q(wr,.9):6.0f} max {max(wr):6.0f} "
              f"| at >=95% util: {len(busy):4d} samples, r+w MB/s p50 {q(tot_busy,.5):6.0f} | r_await ms (read>=200MB/s) p50 {q(ra,.5):6.1f} | w_await p50 {q(wa,.5):6.1f}")
