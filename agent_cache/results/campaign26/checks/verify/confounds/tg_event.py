"""Test (f15): was the stretch that carries the repeat's deficit (order-run positions 6250-6499) disturbed from outside?
In each run's own wall-clock window for that bin: the SSD as iostat saw it (results/campaign26/iostat_vdc.log; runs of
2026-11-11 only), the SSD as the server saw it (ms per page of its reads), the decode step, the prefill rate, and the
client's own timing. The neighbouring ten minutes before the window are shown beside it.
  nice -n 19 python3 -I -B tg_event.py"""
import calendar, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

rank = C.order_rank()
SIX = C.LEAD5 + ["ec_r13_rep"]
D = {a: C.load(a) for a in SIX}
B = {a: C.by_key(C.kept(D[a])) for a in SIX}
keys = C.shared_keys(B, SIX, rank)
bin_keys = [k for k in keys if 6250 <= rank.get(k, -1) < 6500]

STAMP = re.compile(r"^(\d\d)/(\d\d)/(\d{4}) (\d\d):(\d\d):(\d\d) (AM|PM)")
rows = []
t = None
with open(os.path.join(C.RES, "campaign26", "iostat_vdc.log"), errors="replace") as f:
    for line in f:
        m = STAMP.match(line)
        if m:
            h = int(m[4]) % 12 + (12 if m[7] == "PM" else 0)
            t = calendar.timegm((int(m[3]), int(m[1]), int(m[2]), h, int(m[5]), int(m[6])))
            if t > calendar.timegm((2026, 11, 12, 0, 49, 0)):
                break
        elif line.startswith("vdc") and t is not None:
            p = line.split()
            try:
                rows.append((t, float(p[1]), float(p[2]), float(p[5]), float(p[7]), float(p[8]), float(p[11]), float(p[-2]), float(p[-1])))  # t r/s rkB/s r_await w/s wkB/s w_await aqu %util
            except (ValueError, IndexError):
                pass
rows = rows[1:]
print(f"iostat samples used: {len(rows)}, from {rows[0][0]} to {rows[-1][0]} (epoch s)")


def io(a, lo, hi):
    t0 = D[a]["t0"]
    x = [r for r in rows if lo <= r[0] - t0 < hi]
    if not x:
        return "iostat: no samples (run of another day)"
    rs = sum(r[1] for r in x)
    ws = sum(r[4] for r in x)
    return (f"iostat {len(x):3d} samples: busy {C.mean(r[8] for r in x):5.1f} % (max {max(r[8] for r in x):5.1f}), read {C.mean(r[2] for r in x) / 1024:6.1f} MB/s at {sum(r[3] * r[1] for r in x) / rs if rs else 0:5.1f} ms per request, "
            f"write {C.mean(r[5] for r in x) / 1024:6.1f} MB/s at {sum(r[6] * r[4] for r in x) / ws if ws else 0:5.1f} ms, queue {C.mean(r[7] for r in x):5.1f}")


def server(a, lo, hi):
    d = D[a]
    t0 = d["t0"]
    w = lambda t: lo <= t - t0 < hi
    rd = [x for x in d["s2h"] if w(x[0]) and x[3] >= 100]
    dec = [(1000.0 * r / thr, r, q) for t, r, tok, u, thr, q in d["dec"] if w(t) and thr > 0]
    cells = []
    for blo, bhi in L.BUCKETS:
        v = [x[0] for x in dec if blo <= x[1] <= bhi and x[2] == 0]
        cells.append(f"{blo}-{bhi} {L.med(v):6.2f} ({len(v):3d})")
    pre = []
    for tn in d["turns"]:
        if lo <= tn["t_send"] < hi and tn["ttft"] is not None:
            r = d["req"].get(f"{tn['conv']}-{tn['turn']}")
            unc = (tn["pt"] or 0) - (tn["cached"] or 0)
            if r and r[0] < 50 and unc > 2000 and not any(1 for x in ()):
                pre.append((tn["ttft"] - r[0] / 1000) / unc * 1000)
    ent = [1000 * (d["req"][f"{tn['conv']}-{tn['turn']}"][2] - (t0 + tn["t_send"])) for tn in d["turns"] if lo <= tn["t_send"] < hi and f"{tn['conv']}-{tn['turn']}" in d["req"]]
    return (f"server reads of 100 pages or more: {len(rd):3d}, median {L.med([x[5] / x[3] for x in rd]):5.2f} ms per page, total {sum(x[5] for x in rd) / 1000:6.1f} s | decode ms per step, no queue: " + " ".join(cells)
            + f" | prefill ms per uncached token on turns with no queue wait and over 2K uncached: {L.med(pre):.4f} ({len(pre)}) | client send to server entry, median {L.med(ent):5.1f} ms")


for a in SIX:
    ts = [B[a][k]["t_send"] for k in bin_keys]
    lo, hi = min(ts), max(ts)
    print(f"\n{a}: positions 6250-6499 were sent in minutes {lo / 60:.1f}-{hi / 60:.1f}; mean TTFT there {C.mean(B[a][k]['ttft'] for k in bin_keys):.3f} s")
    print(f"  in the window:      {io(a, lo, hi)}")
    print(f"                      {server(a, lo, hi)}")
    print(f"  the 10 min before:  {io(a, lo - 600, lo)}")
    print(f"                      {server(a, lo - 600, lo)}")
