"""Campaigns 28 and 29: when a queued request of the just-in-time engine uses up its read allowance on lookups
that find nothing, and whether the arrival gate then refuses it a read. On the host, from agent_cache/:
  python3 -I -B results/campaign29/checks/cap_path.py results/campaign29 jit_exact jit_p50 ec_full
A plan allows MAX_REFETCHES = 2 re-reads ("ntp_reevict ... capped=1" marks the allowance used up) and the gate
three lookups. A request is at risk when that mark comes before any of its lookups found pages ("s2h_query ...
storage_hit" over 0) and 30 s or more before its admission. Prints the at-risk requests by the time since arrival
of the mark and how many the gate later refused ("ntp_gate decision=capped").
Under --radix-eviction-policy next_use a turn queued for over 60 s ranks as overdue (next_use_eviction.py,
overdue_grace_s) and its session's tail is evicted: the 55-70 s row. Reads the server log only; stdlib only."""
import re
import sys
from datetime import datetime

cdir, labels = sys.argv[1], sys.argv[2:]
manifest = dict((line.split()[0], line.split()[1]) for line in open(f"{cdir}/manifest.txt"))
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
REEVICT = re.compile(r"ntp_reevict rid=(\S+) evicted=(\d+) refetches=(\d+) capped=(\d)")
GATE = re.compile(r"ntp_gate rid=(\S+) decision=(\w+) ")
QUERY = re.compile(r"s2h_query rid=(\S+) tokens_req=(\d+) storage_hit=(\d+)")
ARRIVE = re.compile(r"ntp_arrive rid=(\S+) ")
ADMIT = re.compile(r"ntp_admit rid=(\S+) wait_ms=(\d+)")
BINS = ((0, 5), (5, 55), (55, 70), (70, 1e9))


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


print(f"{cdir.split('/')[-1]}: requests that used up the read allowance on lookups that found nothing, 30 s or more before admission")
print(f"{'arm':10s} | {'queued':>6s} {'wait p50 s':>10s} | {'at risk':>7s} {'share':>6s} {'refused':>7s} | "
      + " | ".join(f"marked {lo:.0f}-{hi if hi < 1e8 else 'up'} s: n, refused" for lo, hi in BINS))
for label in labels:
    arrive, admit, mark, refused, hit = {}, {}, {}, set(), {}
    with open(f"results/{manifest[label]}/server.log", errors="replace") as f:
        for line in f:
            if "ntp_" not in line and "s2h_query" not in line:
                continue
            m = STAMP.match(line)
            if not m:
                continue
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
            if (g := REEVICT.search(line)):
                if g.group(4) == "1":
                    mark.setdefault(g.group(1), t)
            elif (g := GATE.search(line)):
                if g.group(2) == "capped":
                    refused.add(g.group(1))
            elif (g := QUERY.search(line)):
                if int(g.group(3)) > 0:
                    hit.setdefault(g.group(1), t)
            elif (g := ARRIVE.search(line)):
                arrive[g.group(1)] = t
            elif (g := ADMIT.search(line)):
                admit[g.group(1)] = (t, int(g.group(2)) / 1e3)
    rids = [r for r in arrive if r in admit]
    risk = [r for r in rids if r in mark and (r not in hit or hit[r] > mark[r]) and admit[r][0] - mark[r] >= 30]
    cells = []
    for lo, hi in BINS:
        s = [r for r in risk if lo <= mark[r] - arrive[r] < hi]
        cells.append(f"{len(s):22d}, {sum(r in refused for r in s):4d}")
    print(f"{label:10s} | {len(rids):6d} {q([admit[r][1] for r in rids], .5):10.0f} | {len(risk):7d} {100 * len(risk) / max(1, len(rids)):5.1f}% "
          f"{sum(r in refused for r in risk):7d} | " + " | ".join(cells))
