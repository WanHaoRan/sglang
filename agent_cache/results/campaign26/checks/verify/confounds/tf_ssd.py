"""Test (f5): the SSD across the day, from results/campaign26/iostat_vdc.log (10 s samples of /dev/vdc, 2026-11-11).
Per arm of that day, over its first 60 minutes and its whole run: mean %util, read MB/s, write MB/s, read wait per request
(r_await, weighted by r/s), write wait, queue size; and the 10 s samples with the device over 90 % busy.
  nice -n 19 python3 -I -B tf_ssd.py"""
import calendar, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STAMP = re.compile(r"^(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d) (AM|PM)")
rows = []
t = None
with open(os.path.join(C.RES, "campaign26", "iostat_vdc.log"), errors="replace") as f:
    for line in f:
        m = STAMP.match(line)
        if m:
            h = int(m[4]) % 12 + (12 if m[7] == "PM" else 0)
            t = calendar.timegm((int(m[3]), int(m[1]), int(m[2]), h, int(m[5]), int(m[6])))
            if t > calendar.timegm((2026, 11, 11, 16, 40, 0)):
                break
        elif line.startswith("vdc") and t is not None:
            p = line.split()
            # r/s rkB/s rrqm/s %rrqm r_await rareq-sz w/s wkB/s wrqm/s %wrqm w_await wareq-sz ... aqu-sz %util
            rows.append((t, float(p[1]), float(p[2]), float(p[5]), float(p[6]), float(p[7]), float(p[8]), float(p[11]), float(p[12]), float(p[-2]), float(p[-1])))
print(f"{len(rows)} samples from {rows[0][0]} to {rows[-1][0]} (epoch s); first sample is the since-boot average and is skipped")
rows = rows[1:]
day = ["jit_p50_ctl", "hintfree_ctl", "ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_p50_d_cut"]
print("arm            window     samples | %util mean, p90 | read MB/s | write MB/s | r_await ms (weighted) | read request KB | w_await ms (weighted) | queue size | samples over 90 % busy")
for a in day:
    d = C.load(a)
    t0 = d["t0"]
    end = max(x["t_send"] for x in d["turns"])
    for label, lo, hi in (("0-60 min", 0, 3600), ("whole run", 0, end), ("0-74.7 min", 0, 4482)):
        x = [r for r in rows if lo <= r[0] - t0 < hi]
        if not x:
            continue
        rs = sum(r[1] for r in x)
        ws = sum(r[5] for r in x)
        print(f"  {a:13s} {label:10s} {len(x):5d} | {C.mean(r[10] for r in x):5.1f} {C.pct([r[10] for r in x], .9):5.1f} | {C.mean(r[2] for r in x) / 1024:7.1f} | {C.mean(r[6] for r in x) / 1024:7.1f} | "
              f"{sum(r[3] * r[1] for r in x) / rs if rs else 0:6.2f} | {sum(r[4] * r[1] for r in x) / rs if rs else 0:7.1f} | {sum(r[7] * r[5] for r in x) / ws if ws else 0:6.2f} | {C.mean(r[9] for r in x):5.2f} | {sum(1 for r in x if r[10] > 90)}")
print("\nbetween runs (server restarts, model load): gaps are not measured; activity outside any client window on that day, 02:49-16:32Z:")
spans = [(C.load(a)["t0"], C.load(a)["t0"] + max(x["t_send"] for x in C.load(a)["turns"]) + 120) for a in day]
x = [r for r in rows if not any(lo - 240 <= r[0] < hi for lo, hi in spans)]
print(f"  {len(x)} samples outside the runs: %util mean {C.mean(r[10] for r in x):.1f}, read {C.mean(r[2] for r in x) / 1024:.1f} MB/s, write {C.mean(r[6] for r in x) / 1024:.1f} MB/s")
