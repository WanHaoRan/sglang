"""Quick probe (read-only): rule 1's start decisions by the shape of the entry the turn was sent with.
Joins ec_start events (server.log) to the client's dist_hint record by next_rid. Not a verified analysis."""
import glob, json, os, re, sys, collections, statistics as st

RES = "/home/wanhr/sglang/agent_cache/results"
EV = re.compile(r"HICACHE_EVT ec_start rid=(\S+) decision=(\w+) tokens=(\d+) start_ms=(\S+) idle_ms=(\S+) read_ms=(\S+) wait_ms=(\S+)")
for run in sys.argv[1:]:
    cl = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    hint = {}
    for line in open(cl):
        if '"dist_hint"' not in line:
            continue
        r = json.loads(line)
        h = r.get("dist_hint") or {}
        if h.get("next_rid"):
            hint[h["next_rid"]] = h
    last = {}
    order = collections.defaultdict(list)
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            if "ec_start" not in line:
                continue
            m = EV.search(line)
            if m:
                order[m[1]].append(m[2])
    # a plan's "outcome": staged if it ever decided "now"
    cls = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    gaps = collections.defaultdict(list)
    keyc = collections.defaultdict(collections.Counter)
    for rid, decs in order.items():
        h = hint.get(rid)
        outcome = "now" if "now" in decs else ("wait_only" if "wait" in decs else "arrival")
        if not h or h.get("p50") is None:
            shape = "no entry"
        else:
            p10, p50, p90 = h["p10"], h["p50"], h["p90"]
            lo = p50 / max(p10, 0.01); hi = p90 / max(p50, 0.01)
            shape = "sharp (p90/p10<2)" if p90 / max(p10, 0.01) < 2 else ("low side wide (p50/p10>=5)" if lo >= 5 else "other")
            if outcome == "now" and h.get("true_gap_s") is not None:
                gaps[shape].append((h["true_gap_s"], p10, p50, p90))
            keyc[shape][h.get("reply_key")] += 1
        by[shape][outcome] += 1
        cls[outcome] += 1
    print(f"\n{run}: plans with a start decision {len(order)}; {dict(cls)}; hints with next_rid {len(hint)}")
    for shape in sorted(by):
        c = by[shape]
        tot = sum(c.values())
        extra = ""
        if gaps.get(shape):
            g = gaps[shape]
            below = sum(1 for t, a, b, d in g if t < a); above = sum(1 for t, a, b, d in g if t > d)
            extra = f" | of the staged: true gap below p10 {below}, above p90 {above}, of {len(g)}"
        top = ", ".join(f"{k}:{v}" for k, v in keyc[shape].most_common(4))
        print(f"  {shape:28s} n={tot:5d} staged(now) {c['now']:4d} ({100*c['now']/tot:4.1f} %) wait only {c['wait_only']:4d} arrival {c['arrival']:4d}{extra} | top keys {top}")
