"""Test (f1): is the workload the same turn by turn, and is anything outside the cache policy different between runs?
1. Token counts, gaps and output lengths of the same (conv, turn) across arms. 2. Cached tokens on first turns (a cache
that survived from an earlier run would show here). 3. Client-side overhead: time from the client's send to the server's
entry, and client latency minus the server's queue + forward time. 4. Arm settings: run.txt and server_args.txt lines
that differ between runs.
  nice -n 19 python3 -I -B tf_workload.py"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
ARMS = [a for a in C.RUNS if a != "order"]
D = {a: C.load(a) for a in ARMS}
B = {a: C.by_key(C.kept(D[a])) for a in ARMS}

print("1. the same turn across arms")
for title, names in (("LEAD5", C.LEAD5), ("ALL11", C.ALL11), ("CUT7", C.CUT7)):
    keys = C.shared_keys(B, names, rank)
    for f in ("pt", "ct", "olen", "rpt", "gap", "true_gap", "delta"):
        diff = [k for k in keys if len(set(B[a][k][f] for a in names)) > 1]
        mag = [max(B[a][k][f] or 0 for a in names) - min(B[a][k][f] or 0 for a in names) for k in diff]
        print(f"  {title} {len(keys)} shared returning turns, field {f:8s}: differs on {len(diff):5d} turns ({100 * len(diff) / len(keys):5.1f} %)"
              + (f"; spread mean {C.mean(mag):.2f} max {max(mag):.2f}" if diff else ""))
    print(f"  {title} sums over the shared turns per arm: prompt tokens (M), completion tokens (K), uncached prompt tokens (M), gap slept (h)")
    for a in names:
        print(f"    {a:14s} {sum(B[a][k]['pt'] for k in keys) / 1e6:8.3f} {sum(B[a][k]['ct'] for k in keys) / 1e3:8.1f} "
              f"{sum(B[a][k]['pt'] - (B[a][k]['cached'] or 0) for k in keys) / 1e6:8.3f} {sum(B[a][k]['gap'] or 0 for k in keys) / 3600:7.2f}")
    # completion tokens against the requested output length
    for a in names[:1] + names[-1:]:
        short = sum(1 for k in keys if B[a][k]["ct"] != B[a][k]["olen"])
        print(f"    {a}: completion_tokens != output_length on {short} turns; prompt_tokens != replay_prompt_tokens on {sum(1 for k in keys if B[a][k]['pt'] != B[a][k]['rpt'])}")

print("\n2. first turns (turn 0) of the sessions every LEAD5+ctl arm started: cached tokens (mean, max, turns with any), storage tokens > 0, mean ttft, mean prompt tokens")
names = C.SEVEN
first = sorted(set.intersection(*(set(k for k in B[a] if k[1] == 0) for a in names)))
for a in names:
    c = [B[a][k]["cached"] or 0 for k in first]
    print(f"  {a:14s} {len(first)} first turns: cached mean {C.mean(c):7.1f} max {max(c):6d} with any {sum(1 for x in c if x > 0):4d} | storage>0 {sum(1 for k in first if (B[a][k]['sto'] or 0) > 0)} | "
          f"ttft {C.mean(B[a][k]['ttft'] for k in first):6.3f} median {C.pct([B[a][k]['ttft'] for k in first], .5):6.3f} | prompt {C.mean(B[a][k]['pt'] for k in first):8.1f}")
print("  first turns in the first 9 minutes (before the first SSD read of any arm): mean ttft per arm on the same turns")
early = [k for k in first if all(B[a][k]["t_send"] < 540 for a in names)]
print("    " + " ".join(f"{a} {C.mean(B[a][k]['ttft'] for k in early):.3f}" for a in names) + f"  ({len(early)} turns)")

print("\n3. client-side overhead on each arm's kept turns: server entry_time - (t_start + t_send), ms: mean, median, p99, max | client latency - (queue + forward), ms: mean, median, p99 | turns without a ReqTimeStats line")
for a in ARMS:
    off, lat, miss = [], [], 0
    for k, t in B[a].items():
        r = D[a]["req"].get(f"{k[0]}-{k[1]}")
        if not r:
            miss += 1
            continue
        off.append(1000 * (r[2] - (D[a]["t0"] + t["t_send"])))
        lat.append(1000 * t["latency"] - (r[0] + r[1]))
    print(f"  {a:14s} {C.mean(off):7.1f} {C.pct(off, .5):7.1f} {C.pct(off, .99):7.1f} {max(off):8.1f} | {C.mean(lat):7.1f} {C.pct(lat, .5):7.1f} {C.pct(lat, .99):7.1f} | {miss}")

print("\n4. run.txt per run, and server_args.txt fields that differ between runs")
txt = {}
for a in ARMS:
    p = os.path.join(C.RES, C.RUNS[a], "run.txt")
    txt[a] = dict(line.strip().split("=", 1) for line in open(p) if "=" in line)
keys = sorted(set().union(*(set(v) for v in txt.values())))
for k in keys:
    vals = {a: txt[a].get(k, "<absent>") for a in ARMS}
    if len(set(vals.values())) > 1 and k != "started_utc":
        print(f"  run.txt {k}:")
        for a in ARMS:
            print(f"      {a:14s} {vals[a]}")
args = {}
for a in ARMS:
    s = open(os.path.join(C.RES, C.RUNS[a], "server_args.txt")).read()
    # server_args.txt is a printed ServerArgs(...) or key=value dump; pull key=value pairs out of it
    args[a] = dict(re.findall(r"\b([a-z_][a-z0-9_]*)=((?:'[^']*'|\"[^\"]*\"|\[[^\]]*\]|\{[^}]*\}|[^,\s()]+))", s))
keys = sorted(set().union(*(set(v) for v in args.values())))
print(f"  server_args.txt: {len(keys)} fields parsed")
for k in keys:
    vals = {a: args[a].get(k, "<absent>") for a in ARMS}
    if len(set(vals.values())) > 1:
        groups = {}
        for a in ARMS:
            groups.setdefault(vals[a], []).append(a)
        if k in ("port", "api_key"):
            continue
        print(f"  {k}: " + " | ".join(f"{v[:90]} <- {', '.join(g)}" for v, g in groups.items()))
