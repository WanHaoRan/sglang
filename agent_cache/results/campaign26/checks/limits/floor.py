"""Where a returning turn's TTFT goes once the cache policy has done its part: before the scheduler sees the request, in its queue, after admission."""
import glob, json, os, re, statistics as st, sys

RES = "results"
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")
pct = lambda v, q: sorted(v)[min(len(v) - 1, int(q * len(v)))] if v else float("nan")
for run in sys.argv[1:]:
    path = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    t0 = next(r["t_start"] for r in rows if r["kind"] == "run")
    turns = {(r["conv"], r["turn"]): r for r in rows if r["kind"] == "turn" and r.get("ttft") is not None}
    srv = {}
    for line in open(os.path.join(RES, run, "server.log"), errors="replace"):
        if "ReqTimeStats(" in line and (m := RTS.search(line)):
            srv[(int(m[1]), int(m[2]))] = dict(inp=int(m[3]), cached=int(m[4]), out=int(m[5]), queue=float(m[6]) / 1000, fwd=float(m[7]) / 1000, entry=float(m[8]))
    rec = []
    for k, r in turns.items():
        if k[1] == 0 or r.get("mid_start") or k not in srv:
            continue
        s = srv[k]
        cd = r.get("cached_details") or {}
        pre = s["entry"] - (t0 + r["t_send"])
        tier = "storage" if (cd.get("storage") or 0) > 0 else "host" if (cd.get("host") or 0) > 0 else "device"
        rec.append(dict(pre=pre, queue=s["queue"], after=r["ttft"] - pre - s["queue"], ttft=r["ttft"], prompt=s["inp"], new=s["inp"] - s["cached"], tier=tier,
                        host=int(cd.get("host") or 0), storage=int(cd.get("storage") or 0), t=r["t_send"]))
    n = len(rec)
    mean = lambda key, rs=rec: st.mean(x[key] for x in rs)
    print(f"\n{run}: {n} returning turns with server stats")
    print(f"  mean TTFT {mean('ttft'):.3f} = before entry {mean('pre'):.3f} + scheduler queue {mean('queue'):.3f} + after admission {mean('after'):.3f}")
    print(f"  median TTFT {st.median(x['ttft'] for x in rec):.3f}; before entry median {st.median(x['pre'] for x in rec):.3f} p90 {pct([x['pre'] for x in rec], .9):.3f}; "
          f"after admission median {st.median(x['after'] for x in rec):.3f} p90 {pct([x['after'] for x in rec], .9):.3f}")
    print("  before entry by prompt length (tokens): n, median s, mean s")
    for lo, hi in ((0, 8000), (8000, 16000), (16000, 32000), (32000, 64000), (64000, 200000)):
        v = [x["pre"] for x in rec if lo <= x["prompt"] < hi]
        if v:
            print(f"    {lo:6d}-{hi:6d}: {len(v):5d} {st.median(v):.3f} {st.mean(v):.3f}")
    print("  after admission by where the prefix was, turns with no queue wait over 50 ms: n, median s, mean s, mean new tokens, mean tokens loaded from host or storage")
    calm = [x for x in rec if x["queue"] < 0.05]
    for tier in ("device", "host", "storage"):
        v = [x for x in calm if x["tier"] == tier]
        if v:
            print(f"    {tier:8s} {len(v):5d} {st.median(x['after'] for x in v):.3f} {mean('after', v):.3f} {mean('new', v):7.0f} {st.mean(x['host'] + x['storage'] for x in v):8.0f}")
    print("  device-prefix turns with no queue wait, after admission by new tokens: n, median s")
    dev = [x for x in calm if x["tier"] == "device"]
    for lo, hi in ((0, 128), (128, 512), (512, 2048), (2048, 8192), (8192, 10**6)):
        v = [x["after"] for x in dev if lo <= x["new"] < hi]
        if v:
            print(f"    {lo:5d}-{hi:7d}: {len(v):5d} {st.median(v):.3f}")
    print(f"  share of all TTFT seconds: before entry {sum(x['pre'] for x in rec) / sum(x['ttft'] for x in rec):.0%}, queue {sum(x['queue'] for x in rec) / sum(x['ttft'] for x in rec):.0%}, "
          f"after admission {sum(x['after'] for x in rec) / sum(x['ttft'] for x in rec):.0%}")
