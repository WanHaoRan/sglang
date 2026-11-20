"""Campaign 28 or 29, the four-rule arm: what the rules did, by ten minutes of the client clock.
On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/rules_activity.py results/<server run> <programs dir> > results/campaign28/checks/rules_activity_<label>.out
Reads four kinds of server lines: ec_price (every 5 s: the memory price, tokens refused room, idle room, tokens under
early-read pins, harm charged), ec_start (a change of the read-timing decision for a hinted request: now, wait, or
leave it to the arrival), ec_release (an early read's pin given back, its request did not come in time) and ec_bump
(an early read given up for the read of a queued request). Stdlib only."""
import collections
import datetime as dt
import json
import re
import statistics as st
import sys

run, programs = sys.argv[1], sys.argv[2]
with open(f"{run}/{programs}/client.jsonl") as f:
    t0 = json.loads(f.readline())["t_start"]
STAMP = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_\w+) (.*)")
PRICE = re.compile(r"price=(\S+) refused=(\d+) idle=(\d+) excess=(\S+) room=(\d+) held=(\d+) staged=(\d+) harm_ms=(\d+)")
price, starts, releases, bumps = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
with open(f"{run}/server.log", errors="replace") as f:
    for line in f:
        if "HICACHE_EVT ec_" not in line or not (m := STAMP.match(line)):
            continue
        t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() - t0
        if t < 0:
            continue
        w, kv = int(t // 600), dict(x.split("=", 1) for x in m[3].split() if "=" in x)
        if m[2] == "ec_price" and (p := PRICE.search(m[3])):
            price[w].append([float(x) for x in p.groups()])
        elif m[2] == "ec_start":
            starts[w].append((kv["decision"], int(kv["tokens"]), int(kv["read_ms"]), int(kv["wait_ms"])))
        elif m[2] == "ec_release":
            releases[w].append(int(kv["tokens"]))
        elif m[2] == "ec_bump":
            bumps[w].append(int(kv["tokens"]))

every = [x for v in price.values() for x in v]
print(f"{run.split('/')[-1]}: {len(every)} price updates; price first {every[0][0]:.2e}, median {st.median(x[0] for x in every):.2e}, "
      f"last {every[-1][0]:.2e}, lowest {min(x[0] for x in every):.2e}, highest {max(x[0] for x in every):.2e}")
print("minutes | price median | refused room K | idle room K | held K | early-read pins K | harm s | "
      "decisions now, wait, arrival | releases (K) | bumps (K)")
for w in sorted(price):
    v, s = price[w], collections.Counter(d for d, *_ in starts[w])
    print(f"{10 * w:3d}-{10 * w + 10:3d} | {st.median(x[0] for x in v):12.2e} | {sum(x[1] for x in v) / 1e3:14.0f} | {st.mean(x[2] for x in v) / 1e3:11.0f} | "
          f"{st.mean(x[5] for x in v) / 1e3:6.0f} | {st.mean(x[6] for x in v) / 1e3:17.1f} | {sum(x[7] for x in v) / 1e3:6.0f} | "
          f"{s['now']:6d} {s['wait']:6d} {s['arrival']:6d} | {len(releases[w]):5d} ({sum(releases[w]) / 1e3:5.0f}) | {len(bumps[w]):4d} ({sum(bumps[w]) / 1e3:5.0f})")
allstarts = [x for v in starts.values() for x in v]
for kind in ("now", "wait", "arrival"):
    rows = [x for x in allstarts if x[0] == kind]
    if rows:
        print(f"decision {kind}: {len(rows)}; tokens to read median {st.median(x[1] for x in rows):.0f}, "
              f"read estimate median {st.median(x[2] for x in rows) / 1e3:.2f} s, queue wait estimate median {st.median(x[3] for x in rows) / 1e3:.2f} s")
print(f"releases {sum(len(v) for v in releases.values())} ({sum(sum(v) for v in releases.values()) / 1e3:.0f} K tokens); "
      f"bumps {sum(len(v) for v in bumps.values())} ({sum(sum(v) for v in bumps.values()) / 1e3:.0f} K tokens)")
