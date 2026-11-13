"""Test (f16): which cache mechanisms left events in each run's server log (counts of HICACHE_EVT lines by type), to
check the arms' labels against what the engine did: "no hints" (nofc_noend), "reads only at arrival" (nofc, nofc_noend).
One example line is printed for every event type whose name starts with ntp_ (next-turn prefetch).
  nice -n 19 python3 -I -B tg_evtypes.py"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import lib2 as L

RUNS = ["hintfree_ctl", "jit_p50_ctl", "jit_exact", "ec_r13", "ec_r13_rep", "ec_r1", "nofc", "nofc_noend"]
EVT = re.compile(r"HICACHE_EVT (\w+)")
counts, example = {}, {}
for a in RUNS:
    assert os.path.basename(L.run_dir(a)) != C.current_run()
    c = {}
    with open(os.path.join(L.run_dir(a), "server.log"), errors="replace") as f:
        for line in f:
            i = line.find("HICACHE_EVT ")
            if i < 0:
                continue
            m = EVT.match(line, i)
            if m:
                c[m[1]] = c.get(m[1], 0) + 1
                if m[1].startswith("ntp_") and (a, m[1]) not in example:
                    example[(a, m[1])] = line[i:i + 260].rstrip()
    counts[a] = c
types = sorted(set().union(*(set(c) for c in counts.values())))
print("event type".ljust(26) + " ".join(f"{a:>13s}" for a in RUNS))
for t in types:
    print(t.ljust(26) + " ".join(f"{counts[a].get(t, 0):13d}" for a in RUNS))
print("\nexamples of ntp_ events in the two rivals and in rules 1 and 3:")
for a in ("nofc_noend", "nofc", "ec_r13"):
    for t in types:
        if (a, t) in example:
            print(f"  {a:11s} {example[(a, t)]}")

print("\nwhat the two rivals took from the client's per-turn table (the runs' client args: hints=" + ", ".join(f"{a} {C.load(a)['args'].get('hints')}" for a in ("nofc", "nofc_noend", "ec_r13", "hintfree_ctl")) + ")")
GATE = re.compile(r"HICACHE_EVT ntp_gate \S+ decision=(\w+)")
PICK = re.compile(r"HICACHE_EVT ntp_pick \S+ key=(\S+) entry=(\S+)")
NUE = re.compile(r"HICACHE_EVT nue_register \S+ node=\S+ next_use_ms=(\S+)")
for a in ("nofc", "nofc_noend", "ec_r13"):
    gate, pick, nue = {}, {"entry": 0, "none": 0}, {"ended": 0, "time": 0}
    with open(os.path.join(L.run_dir(a), "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT n" not in line:
                continue
            m = GATE.search(line)
            if m:
                gate[m[1]] = gate.get(m[1], 0) + 1
                continue
            m = PICK.search(line)
            if m:
                pick["none" if m[2] == "none" else "entry"] += 1
                continue
            m = NUE.search(line)
            if m:
                nue["ended" if m[1] == "ended" else "time"] += 1
    turns = len(C.load(a)["turns"])
    print(f"  {a:11s} turns {turns} | ntp_pick: table read on {pick['entry'] + pick['none']} turns, an entry found on {pick['entry']} | plans made (ntp_plan) {counts[a].get('ntp_plan', 0)} = {100 * counts[a].get('ntp_plan', 0) / turns:.1f} % of turns | "
          f"arrivals matched to a plan (ntp_arrive) {counts[a].get('ntp_arrive', 0)} | gate decisions {dict(sorted(gate.items()))} | next-use registrations: {nue['time']} with a time or forecast, {nue['ended']} as ended")
