"""Campaign 28 or 29, the arms on the just-in-time engine: when their SSD reads start. On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/read_starts.py results/campaign28 > results/campaign28/checks/read_starts.out
Counts the server's "ntp_due" lines by case (1: the read starts before its request arrives, on the hint; 2: it starts
while the request waits in the queue, timed from its queue position), the reads staged before arrival ("ntp_stage"),
the plans that expired before their request came ("ntp_expire") and the gate's decisions ("ntp_gate"). Stdlib only."""
import collections
import re
import sys

cdir = sys.argv[1]
with open(f"{cdir}/manifest.txt") as f:
    arms = [line.split()[:3] for line in f if line.strip()]
EVENT = re.compile(r"HICACHE_EVT (ntp_due|ntp_stage|ntp_expire|ntp_gate|ntp_plan) (.*)")
print(f"{cdir.split('/')[-1]}: reads by when they start, whole run")
print(f"{'arm':10s} | {'plans':>6s} | {'before arrival':>14s} {'in the queue':>12s} {'share before':>12s} | {'staged':>6s} {'expired':>7s} | gate decisions")
for label, run, _ in arms:
    n, gate = collections.Counter(), collections.Counter()
    with open(f"results/{run}/server.log", errors="replace") as f:
        for line in f:
            if "HICACHE_EVT ntp_" not in line or not (m := EVENT.search(line)):
                continue
            if m[1] == "ntp_due":
                n["case" + re.search(r"case=(\d+)", m[2])[1]] += 1
            elif m[1] == "ntp_gate":
                gate[re.search(r"decision=(\w+)", m[2])[1]] += 1
            else:
                n[m[1]] += 1
    if not n["ntp_plan"]:
        print(f"{label:10s} | no next-turn plans (not the just-in-time engine's events)")
        continue
    due = n["case1"] + n["case2"]
    print(f"{label:10s} | {n['ntp_plan']:6d} | {n['case1']:14d} {n['case2']:12d} {100 * n['case1'] / max(1, due):11.1f}% | "
          f"{n['ntp_stage']:6d} {n['ntp_expire']:7d} | {dict(sorted(gate.items()))}")
