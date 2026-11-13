"""Reads parked after their turn arrived: how long, and how much of that with the read thread idle."""
import bisect, collections, datetime as dt, re, statistics as st, sys
TS = re.compile(r"^\[(.{23})\] ")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
ENTRY = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), .*queue_duration=([\d.]+)ms.*entry_time=([\d.]+)")
EV = re.compile(r"HICACHE_EVT (srs_park|srs_dispatch|s2h_io) rid=\S+?-c(\d+)-t(\d+) (.*)")
for name, run in (a.split("=") for a in sys.argv[1:]):
    entry, parks, ios, windows, last_io = {}, {}, [], [], {}
    for line in open(f"results/{run}/server.log", errors="replace"):
        m = TS.match(line)
        if not m:
            continue
        body = line[m.end():]
        if body.startswith("ReqTimeStats"):
            if e := ENTRY.match(body):
                entry[(int(e[1]), int(e[2]))] = (float(e[4]), float(e[3]) / 1000)
        elif body.startswith("HICACHE_EVT s"):
            if e := EV.match(body):
                k, t = (int(e[2]), int(e[3])), stamp(m[1])
                if e[1] == "srs_park":
                    parks[k] = t
                elif e[1] == "srs_dispatch":
                    if k in parks:
                        windows.append((k, parks.pop(k), t, "launched=1" in e[4]))
                else:
                    ms = int(re.search(r"ms=(\d+)", e[4])[1]) / 1000
                    ios.append((t - ms, t))
                    last_io[k] = t
    ios.sort()
    # merged busy intervals of the read thread
    busy = []
    for a, b in ios:
        if busy and a <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], b)
        else:
            busy.append([a, b])
    starts = [a for a, _ in busy]
    def busy_in(lo, hi):
        if hi <= lo:
            return 0.0
        i = max(0, bisect.bisect_right(starts, lo) - 1)
        tot = 0.0
        while i < len(busy) and busy[i][0] < hi:
            tot += max(0.0, min(hi, busy[i][1]) - max(lo, busy[i][0]))
            i += 1
        return tot
    t0, t1 = min(a for a, _ in ios), max(b for _, b in ios)
    n = len([k for k in entry if k[1] > 0])
    held = idle = pre = 0.0
    turns = collections.Counter()
    idle_by = collections.Counter()
    for k, park, disp, launched in windows:
        if k not in entry:
            continue
        arrive, wait = entry[k]
        lo, hi = max(park, arrive), min(disp, arrive + wait)
        pre += max(0.0, min(disp, arrive) - park)
        if hi > lo:
            b = busy_in(lo, hi)
            held += hi - lo
            idle += hi - lo - b
            turns[k] += hi - lo
            idle_by[k] += hi - lo - b
    over = [k for k in turns if turns[k] > 0.2]
    print(f"== {name}: {n} returning turns with stats, read thread busy {sum(b - a for a, b in busy):.0f} s of {t1 - t0:.0f} s ({100 * sum(b - a for a, b in busy) / (t1 - t0):.0f} %)")
    print(f"   parked after arrival: {held:.0f} s in all = {held / n:.3f} s per returning turn; of it the read thread idle {idle:.0f} s = {idle / n:.3f} s per returning turn"
          f" | turns parked over 0.2 s: {len(over)}, their mean parked {st.mean([turns[k] for k in over]):.2f} s, idle part {st.mean([idle_by[k] for k in over]):.2f} s"
          f" | parked before arrival (staged) {pre:.0f} s")
    # busy fraction per 5 minutes
    print("   read thread busy % per 5 min: " + " ".join(f"{100 * busy_in(t0 + 300 * i, t0 + 300 * (i + 1)) / 300:.0f}" for i in range(int((t1 - t0) // 300))))
    lost = [(k, idle_by[k], entry[k][0] + entry[k][1] - last_io[k]) for k in idle_by if idle_by[k] > 0.2 and k in last_io]
    quick = [x for x in lost if x[2] < 0.1]
    print(f"   turns parked over 0.2 s with the read thread idle: {len(lost)}, idle-parked {sum(x[1] for x in lost):.0f} s; of them admitted within 0.1 s of their read landing: {len(quick)} turns, "
          f"idle-parked {sum(x[1] for x in quick):.0f} s = {sum(x[1] for x in quick) / n:.3f} s per returning turn; the others waited {st.mean([x[2] for x in lost if x[2] >= 0.1]) if len(lost) > len(quick) else 0:.2f} s more on average")
