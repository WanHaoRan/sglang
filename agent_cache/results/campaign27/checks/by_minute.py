"""A replay_programs.py run minute by minute: lanes in flight, requests sent, TTFT of returning requests, storage hits,
and the server's running and queued requests. From agent_cache/:
  python3 -I -B results/campaign27/checks/by_minute.py results/<server run> [programs folder]"""
import collections, glob, json, os, re, statistics as st, sys

run = sys.argv[1]
path = os.path.join(run, sys.argv[2], "client.jsonl") if len(sys.argv) > 2 else sorted(glob.glob(os.path.join(run, "programs_*", "client.jsonl")))[-1]
rows = [json.loads(line) for line in open(path) if line.strip()]
t_start = next(r["t_start"] for r in rows if r["kind"] == "run")
lanes = {int(r["t"] // 60) - 1: (r["live"], r["in_flight"]) for r in rows if r["kind"] == "lanes"}
by = collections.defaultdict(list)
for r in rows:
    if r["kind"] == "req" and r.get("ttft") is not None:
        by[int(r["t_send"] // 60)].append(r)
DEC = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.\d+\] Decode batch, #running-req: (\d+), #token: \d+, token usage: ([\d.]+), .*#queue-req: (\d+)")
import datetime as dt
srv = collections.defaultdict(list)
for line in open(os.path.join(run, "server.log"), errors="replace"):
    m = DEC.match(line)
    if m:
        t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() - t_start
        if t >= 0:
            srv[int(t // 60)].append((int(m[2]), int(m[4]), float(m[3])))
print("minute | in session, in flight | sent, first requests | returning: mean TTFT s, p50, from SSD | server: running, queued, GPU pool used (medians)")
for m in sorted(by):
    v = by[m]
    ret = [r for r in v if r["k"] > 0 and r["phase"] == "profile"]
    ssd = sum(1 for r in ret if int((r.get("cached_details") or {}).get("storage") or 0) > 0)
    s = srv.get(m) or [(0, 0, 0.0)]
    live = lanes.get(m, ("", ""))
    print(f"  {m:4d} | {str(live[0]):>4s} {str(live[1]):>4s} | {len(v):4d} {len(v) - len(ret):3d} | "
          f"{st.mean(r['ttft'] for r in ret) if ret else float('nan'):7.2f} {st.median(r['ttft'] for r in ret) if ret else float('nan'):6.2f} {100 * ssd / max(1, len(ret)):4.0f}% | "
          f"{st.median(x[0] for x in s):4.0f} {st.median(x[1] for x in s):4.0f} {st.median(x[2] for x in s):5.2f}")
