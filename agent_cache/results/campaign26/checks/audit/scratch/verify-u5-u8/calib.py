"""Audit scratch (read-only): the reviewer's calibration split, re-derived with my own parser. For each returning turn kept
(sent at least 300 s before the run's last send): the hint the previous turn of the session carried (dist_hint p10/p50/p90,
true_gap_s) against the true gap, by whether the turn read from SSD (an s2h_io line with its request id).
  nice -n 10 python3 -I -B calib.py label=run ..."""
import collections, json, os, re, sys
RES = "/home/wanhr/sglang/agent_cache/results"
IO = re.compile(r"HICACHE_EVT s2h_io rid=\S+?-c(\d+)-t(\d+) pages=(\d+)")
for arg in sys.argv[1:]:
    name, run = arg.split("=")
    client = sorted(d for d in os.listdir(f"{RES}/{run}") if d.startswith("client_"))[-1]
    every = {}
    for line in open(f"{RES}/{run}/{client}/client.jsonl"):
        if line.strip():
            r = json.loads(line)
            if r.get("kind") == "turn" and r.get("ttft") is not None:
                every[(r["conv"], r["turn"])] = r
    end = max(r["t_send"] for r in every.values()) - 300
    pages = collections.Counter()
    for line in open(f"{RES}/{run}/server.log", errors="replace"):
        if "s2h_io" in line:
            m = IO.search(line)
            if m:
                pages[(int(m.group(1)), int(m.group(2)))] += int(m.group(3))
    rows = {"read": [], "no read": []}
    for k, r in every.items():
        if k[1] == 0 or r["t_send"] >= end:
            continue
        p = every.get((k[0], k[1] - 1))
        if p is None:
            continue
        rows["read" if pages[k] > 0 else "no read"].append((p.get("dist_hint") or {}))
    print(f"{name}: kept returning turns with a previous turn {sum(len(v) for v in rows.values())}")
    for cls, v in rows.items():
        have = [h for h in v if h.get("p50") is not None and h.get("true_gap_s") is not None]
        three = [h for h in have if h.get("p10") is not None and h.get("p90") is not None]
        w2 = sum(abs(h["true_gap_s"] - h["p50"]) <= 2 for h in have)
        w5 = sum(abs(h["true_gap_s"] - h["p50"]) <= 5 for h in have)
        above = sum(h["true_gap_s"] > h["p90"] for h in three)
        below = sum(h["true_gap_s"] < h["p10"] for h in three)
        print(f"  {cls:8s} n {len(v):5d}, with p50 and true gap {len(have):5d} | |true - p50| <= 2 s {100 * w2 / max(1, len(have)):5.1f} %, <= 5 s {100 * w5 / max(1, len(have)):5.1f} % | "
              f"three quantiles {len(three):5d}: above p90 {100 * above / max(1, len(three)):5.1f} %, below p10 {100 * below / max(1, len(three)):5.1f} %")
