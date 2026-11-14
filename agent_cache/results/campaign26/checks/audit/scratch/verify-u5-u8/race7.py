"""Audit scratch (read-only): the bumps of ec_full stamped at or after the bumped turn's scheduler entry: what the turn got."""
import datetime as dt, json, os, re, sys, collections
RES = "/home/wanhr/sglang/agent_cache/results"
run = "20261112_214834_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"
EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_bump|ntp_arrive|s2h_io) rid=\S+?-c(\d+)-t(\d+)(.*)")
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+),.*queue_duration=([\d.]+)ms.*entry_time=([\d.]+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
bumps, arrive, entry, queue, reads = [], {}, {}, {}, collections.defaultdict(list)
for line in open(f"{RES}/{run}/server.log", errors="replace"):
    m = EVT.match(line)
    if m:
        t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
        if kind == "ec_bump":
            bumps.append((t, key, m.group(5).strip()))
        elif kind == "ntp_arrive":
            arrive.setdefault(key, t)
        else:
            reads[key].append((t, int(re.search(r"pages=(\d+)", m.group(5)).group(1))))
        continue
    r = RTS.search(line)
    if r:
        k = (int(r.group(1)), int(r.group(2)))
        entry.setdefault(k, float(r.group(4))); queue.setdefault(k, float(r.group(3)) / 1000)
client = sorted(d for d in os.listdir(f"{RES}/{run}") if d.startswith("client_"))[-1]
turns = {}
for line in open(f"{RES}/{run}/{client}/client.jsonl"):
    if line.strip():
        r = json.loads(line)
        if r.get("kind") == "turn" and r.get("ttft") is not None:
            turns[(r["conv"], r["turn"])] = r
print("bumps", len(bumps), "distinct bumped turns", len({k for _, k, _ in bumps}))
n = 0
for t, k, rest in bumps:
    if k in entry and entry[k] <= t + 0.0005:   # log stamp is truncated to the ms
        n += 1
        r = turns.get(k, {})
        cd = r.get("cached_details") or {}
        before = [(round(at - t, 3), p) for at, p in reads[k] if at <= t]
        after = [(round(at - t, 3), p) for at, p in reads[k] if at > t]
        print(f"  c{k[0]}-t{k[1]}: bump - entry {t - entry[k]:+.4f} s, ntp_arrive - bump {arrive.get(k, float('nan')) - t:+.4f} s | {rest[:70]} | reads before {before[-2:]} after {after[:3]} | "
              f"ttft {r.get('ttft', float('nan')):.2f} s queue {queue.get(k, float('nan')):.2f} s storage {cd.get('storage')} host {cd.get('host')} device {cd.get('device')} recomputed {r.get('reply_reprefilled')}")
print("bumps at or after the bumped turn's scheduler entry (ms resolution):", n)
close = sum(1 for t, k, _ in bumps if k in entry and 0.0005 < entry[k] - t <= 0.1)
print("bumps up to 0.1 s before the entry:", close)
