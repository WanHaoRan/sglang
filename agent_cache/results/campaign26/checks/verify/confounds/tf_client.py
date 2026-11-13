"""Test (f8): the client as a measuring instrument. Slack between when a turn should be sent (previous send + previous
latency + the gap slept) and when it was sent, per run; a loaded client would show up here and in every TTFT it stamps.
Also the client-stamped latency against the server's own queue + forward time for the same request.
  nice -n 19 python3 -I -B tf_client.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ARMS = [a for a in C.RUNS if a != "order"]
print("arm            returning turns | send slack ms: mean, median, p99, max | share over 50 ms | client latency - server (queue + forward) ms: median, p99 | client ttft - server queue wait, on turns with no SSD read and under 64 recomputed tokens: median ms")
for a in ARMS:
    d = C.load(a)
    conv = {}
    for t in sorted(d["turns"], key=lambda t: t["t_send"]):
        conv.setdefault(t["conv"], []).append(t)
    slack = []
    for v in conv.values():
        for p, t in zip(v, v[1:]):
            if p["latency"] is not None and t["turn"] == p["turn"] + 1:
                slack.append(1000 * (t["t_send"] - (p["t_send"] + p["latency"] + (t["gap"] or 0))))
    lat, rest = [], []
    read = set((x[1], x[2]) for x in d["s2h"] if x[3] > 0)
    for t in d["turns"]:
        r = d["req"].get(f"{t['conv']}-{t['turn']}")
        if r and t["ttft"] is not None:
            lat.append(1000 * t["latency"] - (r[0] + r[1]))
            if t["turn"] > 0 and (t["conv"], t["turn"]) not in read and C.rec(t) <= 64:
                rest.append(1000 * t["ttft"] - r[0])
    print(f"  {a:14s} {len(slack):6d} | {C.mean(slack):7.2f} {C.pct(slack, .5):7.2f} {C.pct(slack, .99):7.2f} {max(slack):8.1f} | {100 * sum(1 for x in slack if x > 50) / len(slack):5.2f} % | "
          f"{C.pct(lat, .5):7.1f} {C.pct(lat, .99):7.1f} | {C.pct(rest, .5):7.1f}")
