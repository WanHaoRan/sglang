"""Quick probe (read-only): per reply key, how often rule 1 staged a read early and how long the read then waited for
its turn (true gap minus idle time at the 'now' decision minus the read time). Not a verified analysis."""
import glob, json, os, re, sys, collections, statistics as st

RES = "/home/wanhr/sglang/agent_cache/results"
EV = re.compile(r"HICACHE_EVT ec_start rid=(\S+) decision=(\w+) tokens=(\d+) start_ms=(\S+) idle_ms=(\S+) read_ms=(\S+) wait_ms=(\S+)")
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else float("nan")
for run in sys.argv[1:]:
    cl = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    hint = {}
    for line in open(cl):
        if '"dist_hint"' not in line:
            continue
        h = (json.loads(line).get("dist_hint") or {})
        if h.get("next_rid"):
            hint[h["next_rid"]] = h
    plans = collections.defaultdict(list)
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            if "ec_start" in line and (m := EV.search(line)):
                plans[m[1]].append((m[2], int(m[3]), float(m[5]) / 1000, float(m[6]) / 1000))
    rows = collections.defaultdict(lambda: dict(n=0, now=0, lead=[], tok=[]))
    for rid, decs in plans.items():
        h = hint.get(rid)
        if not h:
            continue
        key = h.get("reply_key") or "?"
        for name in (key, "ALL"):
            row = rows[name]
            row["n"] += 1
            first_now = next((d for d in decs if d[0] == "now"), None)
            if first_now:
                row["now"] += 1
                if h.get("true_gap_s") is not None:
                    row["lead"].append(h["true_gap_s"] - first_now[2] - first_now[3])
                    row["tok"].append(first_now[1])
    print(f"\n{run}")
    print("  key                               plans staged  %   | read landed before the turn came back, s: p10 p50 p90 | over 30 s | came back before it landed | median tokens")
    top = sorted(rows.items(), key=lambda kv: -kv[1]["n"])[:12]
    for key, row in top:
        lead = row["lead"]
        if lead:
            cell = f"{pct(lead,.1):7.1f} {pct(lead,.5):6.1f} {pct(lead,.9):7.1f} | {sum(1 for x in lead if x > 30):4d} | {sum(1 for x in lead if x < 0):4d} | {int(st.median(row['tok'])):6d}"
        else:
            cell = "-"
        print(f"  {key:33s} {row['n']:5d} {row['now']:5d} {100*row['now']/row['n']:5.1f} | {cell}")
