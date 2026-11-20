"""Campaign 28 or 29: how busy the storage read thread is. On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/read_thread.py results/<campaign> > results/<campaign>/checks/read_thread.out
Per arm of the manifest and ten minutes of the server's clock: reads done ("s2h_io" lines, one per read, with its
pages and its I/O time), tokens read, the share of the wall time the thread spent in I/O, the read speed, and the
median time a read stayed parked before its dispatch ("srs_dispatch waited_ms", deadline-order arms). Stdlib only."""
import collections
import datetime
import re
import sys

cdir = sys.argv[1]
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
IO = re.compile(r"s2h_io rid=\S+ pages=(\d+) tokens=\d+ ms=(\d+)")
PAGE = 64   # tokens a page (hicache page size of these runs)


def stamp(line):
    return datetime.datetime.strptime(TS.match(line)[1], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def med(values):
    v = sorted(values)
    return v[len(v) // 2] if v else float("nan")


for row in open(f"{cdir}/manifest.txt"):
    label, run, _ = row.split()
    bins = collections.defaultdict(lambda: [0, 0, 0.0, []])
    t0 = None
    with open(f"results/{run}/server.log", errors="replace") as f:
        for line in f:
            if t0 is None and "Prefill batch" in line and TS.match(line):
                t0 = stamp(line)
            if t0 is None:
                continue
            if (m := IO.search(line)):
                b = bins[int((stamp(line) - t0) // 600)]
                b[0] += 1
                b[1] += int(m[1]) * PAGE
                b[2] += int(m[2]) / 1e3
            elif " srs_dispatch " in line:
                bins[int((stamp(line) - t0) // 600)][3].append(int(re.search(r"waited_ms=(\d+)", line)[1]) / 1e3)
    print(f"{label}: minutes since the first prefill | reads | tokens read M | read thread busy | K tokens a second of I/O | parked before dispatch, median s")
    for k in sorted(bins):
        n, tokens, busy, parked = bins[k]
        print(f"  {10 * k:4d}-{10 * k + 10:<4d} | {n:5d} | {tokens / 1e6:6.2f} | {100 * busy / 600:5.1f}% | {tokens / 1e3 / busy if busy else float('nan'):6.1f} | "
              f"{med(parked):6.1f}")
    n = sum(b[0] for b in bins.values())
    tokens = sum(b[1] for b in bins.values())
    busy = sum(b[2] for b in bins.values())
    late = [bins[k] for k in bins if 3 <= k < 11]
    print(f"  whole run: {n} reads, {tokens / 1e6:.1f} M tokens, {busy:.0f} s of I/O; minutes 30 to 110: busy {100 * sum(b[2] for b in late) / max(1, 600 * len(late)):.1f} %, "
          f"{sum(b[1] for b in late) / 1e3 / max(1e-9, sum(b[2] for b in late)):.1f} K tokens a second of I/O")
