"""Campaign 34: do requests that read from the SSD wait longer than those that do not?

Streams one server.log. Per request (conv c, turn t): entry_time and queue_duration
(ReqTimeStats), and its s2h_io landings with tokens > 0. Per 10-minute window of entry
time (from the first request of the run), mean queue wait of requests with an SSD read
against those without, and the share of the wait after the request's last landing.
"""
import collections
import datetime as dt
import re
import statistics as st
import sys

ADMIT = re.compile(r"rid=\S+?-c(\d+)-t(\d+)[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
READ = re.compile(r"^\[([\d\- :.]+)\] HICACHE_EVT s2h_io rid=\S+-c(\d+)-t(\d+) pages=(\d+) tokens=(\d+) ms=(\d+)")


def stamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()


def main(path):
    entry, queue, lands, tokens = {}, {}, collections.defaultdict(list), collections.Counter()
    with open(path, errors="replace") as f:
        for line in f:
            if "s2h_io" in line:
                m = READ.match(line)
                if m and int(m[5]) > 0:
                    k = (int(m[2]), int(m[3]))
                    lands[k].append(stamp(m[1]))
                    tokens[k] += int(m[5])
            elif "queue_duration=" in line:
                m = ADMIT.search(line)
                if m:
                    k = (int(m[1]), int(m[2]))
                    entry[k] = float(m[5])
                    queue[k] = float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
    t0 = min(entry.values())
    rows = collections.defaultdict(lambda: {"read": [], "none": [], "after": [], "tok": []})
    for k, e in entry.items():
        if k[1] == 0:
            continue  # first requests
        w = int((e - t0) // 600) * 10
        q = queue[k]
        ends = [x for x in lands.get(k, ()) if x <= e + q + 1.0]
        if ends:
            rows[w]["read"].append(q)
            rows[w]["after"].append(max(0.0, e + q - max(ends)))
            rows[w]["tok"].append(tokens[k])
        else:
            rows[w]["none"].append(q)
    print(f"{path.split('/')[-2]}")
    print("minutes | returning with an SSD read: n, mean wait s, mean wait after last landing s, mean tokens read K"
          " | without: n, mean wait s")
    for w in sorted(rows):
        r = rows[w]
        if not r["none"]:
            continue
        rd = (f"{len(r['read']):5d} {st.mean(r['read']):7.2f} {st.mean(r['after']):7.2f} {st.mean(r['tok'])/1000:6.1f}"
              if r["read"] else f"{0:5d}       -       -      -")
        print(f"{w:3d}-{w+10:<3d} | {rd} | {len(r['none']):5d} {st.mean(r['none']):7.2f}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
