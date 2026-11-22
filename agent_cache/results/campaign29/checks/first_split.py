"""A pair of arms, B - A on the requests both sent before the cut, split into users' first requests (stream 0, index 0)
and returning requests, with compare_arms.py's pairing and interval over users. On the host, from agent_cache/:
  python3 -I -B results/campaign29/checks/first_split.py results/campaign29 ec_full ec_exact [cut s, default 6600]
"""
import os
import random
import sys

AC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(AC, "frozen", "oracle_scripts_staging", "c28_users"))
import compare_arms as ca  # noqa: E402

cdir, la, lb = sys.argv[1:4]
cut = float(sys.argv[4]) if len(sys.argv) > 4 else 6600.0
runs = {}
with open(os.path.join(cdir, "manifest.txt")) as f:
    for line in f:
        if line.strip():
            label, run, programs = line.split()[:3]
            runs[label] = os.path.join(AC, "results", run, programs, "client.jsonl")
ttft = {}
for label in (la, lb):
    arm = ca.load_arm(runs[label])
    ttft[label] = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in arm["reqs"] if r["t_send"] < cut}
print(f"{os.path.basename(cdir.rstrip('/'))}: A = {la}, B = {lb}, requests both sent before {cut:,.0f} s")
for name, keep in (("all", lambda k: True), ("first", lambda k: k[1] == 0 and k[2] == 0),
                   ("returning", lambda k: not (k[1] == 0 and k[2] == 0))):
    a = {k: v for k, v in ttft[la].items() if keep(k)}
    b = {k: v for k, v in ttft[lb].items() if keep(k)}
    p = ca.paired(a, b, random.Random(0))
    print(f"{name:9s} {p['n']:5d} requests {p['users']:4d} users | A {p['a']:7.2f} B {p['b']:7.2f} | B - A {p['diff']:+7.2f} "
          f"[{p['lo']:+7.2f}, {p['hi']:+7.2f}] | B faster {100 * p['b_faster']:.1f}%")
