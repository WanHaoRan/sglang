"""Predicted against actual queue wait by position at arrival, for turns that read from SSD and turns that did not."""
import collections, re, statistics as st, sys
ARR = re.compile(r"HICACHE_EVT ntp_arrive rid=\S+?-c(\d+)-t(\d+) .*pos=(\d+) qlen=(\d+)")
ADM = re.compile(r"HICACHE_EVT ntp_admit rid=\S+?-c(\d+)-t(\d+) wait_ms=(\d+) pred_wait_ql_ms=(\S+) ")
IO = re.compile(r"HICACHE_EVT s2h_io rid=\S+?-c(\d+)-t(\d+) ")
DISP = re.compile(r"HICACHE_EVT srs_dispatch rid=\S+?-c(\d+)-t(\d+) pages=\d+ waited_ms=(\d+) .*launched=(\d)")
for name, run in (a.split("=") for a in sys.argv[1:]):
    pos, adm, read, held = {}, {}, set(), collections.Counter()
    for line in open(f"results/{run}/server.log", errors="replace"):
        if "HICACHE_EVT" not in line:
            continue
        if m := ARR.search(line):
            pos[(int(m[1]), int(m[2]))] = int(m[3])
        elif m := ADM.search(line):
            adm[(int(m[1]), int(m[2]))] = (int(m[3]) / 1000, None if m[4] in ("na", "None") else float(m[4]) / 1000)
        elif m := IO.search(line):
            read.add((int(m[1]), int(m[2])))
        elif m := DISP.search(line):
            held[(int(m[1]), int(m[2]))] += int(m[3])
    print(f"== {name}: {len(adm)} admitted turns, {len(read)} that read")
    print("   position at arrival | turns that did not read: n, predicted wait, actual wait mean, median | turns that read: n, predicted, actual mean, median, held before dispatch mean")
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 7), (8, 99)):
        cells = []
        for is_read in (False, True):
            ks = [k for k in adm if k in pos and lo <= pos[k] <= hi and (k in read) == is_read and k[1] > 0]
            pred = [adm[k][1] for k in ks if adm[k][1] is not None]
            act = [adm[k][0] for k in ks]
            cell = f"{len(ks):5d} {st.mean(pred) if pred else float('nan'):6.2f} {st.mean(act) if act else float('nan'):6.2f} {st.median(act) if act else float('nan'):6.2f}"
            if is_read:
                cell += f" {st.mean([held[k] / 1000 for k in ks]) if ks else float('nan'):6.2f}"
            cells.append(cell)
        print(f"   {lo:2d}-{hi:2d} | " + " | ".join(cells))
