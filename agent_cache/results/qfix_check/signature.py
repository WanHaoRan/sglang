"""Defect 2's signature in a server log (OPEN_ITEMS.md §2): requests that used up their read allowance ("ntp_reevict
... capped=1") 55 to 70 s after arriving, and how many the arrival gate then refused a read ("ntp_gate
decision=capped"). The counts of results/campaign29/checks/cap_path.py, restricted to requests that reach the server
within WINDOW seconds of its first arrival, under several readings of that window. On the host, from agent_cache/:
  python3 -I -B results/qfix_check/signature.py results/<run>/server.log [window s, default 1800]
The fix check reads the first line, "arrived before 1800 s": campaign 29's jit_exact gives 88 marks at 55 to 70 s and
57 of them refused. Stdlib only.
"""
import re
import sys
from datetime import datetime

path = sys.argv[1]
window = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
REEVICT = re.compile(r"ntp_reevict rid=(\S+) evicted=(\d+) refetches=(\d+) capped=(\d)")
GATE = re.compile(r"ntp_gate rid=(\S+) decision=(\w+) ")
QUERY = re.compile(r"s2h_query rid=(\S+) tokens_req=(\d+) storage_hit=(\d+)")
ARRIVE = re.compile(r"ntp_arrive rid=(\S+) ")
ADMIT = re.compile(r"ntp_admit rid=(\S+) wait_ms=(\d+)")
arrive, admit, mark, refused, hit, first = {}, {}, {}, {}, {}, None
with open(path, errors="replace") as f:
    for line in f:
        m = STAMP.match(line)
        if not m:
            continue
        if first is None:
            first = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        if "ntp_" not in line and "s2h_query" not in line:
            continue
        t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
        if (g := REEVICT.search(line)):
            if g.group(4) == "1":
                mark.setdefault(g.group(1), t)
        elif (g := GATE.search(line)):
            if g.group(2) == "capped":
                refused.setdefault(g.group(1), t)
        elif (g := QUERY.search(line)):
            if int(g.group(3)) > 0:
                hit.setdefault(g.group(1), t)
        elif (g := ARRIVE.search(line)):
            arrive.setdefault(g.group(1), t)
        elif (g := ADMIT.search(line)):
            admit[g.group(1)] = t
t0 = min(arrive.values())
print(f"first log line to first arrival: {t0 - first:.0f} s; arrivals {len(arrive)}, marks {len(mark)}, refusals {len(refused)}")


def report(name, rids):
    m5570 = [r for r in rids if r in mark and r in arrive and 55 <= mark[r] - arrive[r] < 70]
    # At risk as in cap_path.py: marked before any lookup found pages and 30 s or more before admission.
    risk = [r for r in m5570 if (r not in hit or hit[r] > mark[r]) and r in admit and admit[r] - mark[r] >= 30]
    print(f"{name:52s} | requests {len(rids):5d} | marks {sum(r in mark for r in rids):4d}, at 55-70 s {len(m5570):4d} "
          f"(at risk {len(risk):4d}) | refused {sum(r in refused for r in rids):4d}, of the 55-70 s marks "
          f"{sum(r in refused for r in m5570):4d}")


report(f"arrived before {window:.0f} s", [r for r in arrive if arrive[r] - t0 < window])
report(f"arrived and admitted before {window:.0f} s",
       [r for r in arrive if arrive[r] - t0 < window and r in admit and admit[r] - t0 < window])
report(f"arrived before {window:.0f} s, marked before it",
       [r for r in arrive if arrive[r] - t0 < window and r in mark and mark[r] - t0 < window])
report(f"arrived before {window:.0f} s from the first log line", [r for r in arrive if arrive[r] - first < window])
