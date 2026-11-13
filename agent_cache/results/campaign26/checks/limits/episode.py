"""What the scheduler queue waits for in a run: per minute, queue seconds accrued, GPU pool use, prefill load, first turns."""
import collections, datetime as dt, glob, json, os, re, statistics as st, sys

RES = "results"
TS = re.compile(r"^\[(.{23})\] ")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), .*#queue-req: (\d+)")
PRE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)")
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
run = sys.argv[1]
path = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
rows = [json.loads(l) for l in open(path) if l.strip()]
t0 = next(r["t_start"] for r in rows if r["kind"] == "run")
first = {(r["conv"], r["turn"]) for r in rows if r["kind"] == "turn" and (r["turn"] == 0 or r.get("mid_start"))}
B = 60
use, queue_len, pre_tok, pre_n, qsec, big, qsec_first, newtok_first, n_first, running = (collections.defaultdict(list), collections.defaultdict(list), collections.Counter(), collections.Counter(),
    collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.defaultdict(list))
sample = None
for line in open(os.path.join(RES, run, "server.log"), errors="replace"):
    m = TS.match(line)
    if not m:
        continue
    body = line[m.end():]
    if body.startswith("Decode batch"):
        if d := DEC.match(body):
            b = int((stamp(m[1]) - t0) // B)
            use[b].append(float(d[3])); queue_len[b].append(int(d[4])); running[b].append(int(d[1]))
    elif body.startswith("Prefill batch"):
        if sample is None:
            sample = body.strip()[:200]
        if p := PRE.match(body):
            b = int((stamp(m[1]) - t0) // B)
            pre_tok[b] += int(p[2]); pre_n[b] += 1
    elif body.startswith("ReqTimeStats"):
        if r := RTS.match(body):
            k = (int(r[1]), int(r[2]))
            b = int((float(r[8]) - t0) // B)
            q, new = float(r[6]) / 1000, int(r[3]) - int(r[4])
            qsec[b] += q
            if k in first:
                qsec_first[b] += q; newtok_first[b] += new; n_first[b] += 1
            if new >= 8192:
                big[b] += 1
print("prefill line:", sample)
tot = sum(qsec.values())
full = sum(v for b, v in qsec.items() if use[b] and st.mean(use[b]) > 0.9)
print(f"{run}: scheduler queue seconds in all {tot:.0f}; in minutes with mean GPU pool use over 0.9: {full / tot:.0%}; of first turns {sum(qsec_first.values()) / tot:.0%}")
top = sorted(qsec, key=qsec.get, reverse=True)[:14]
print(f"the {len(top)} minutes with the most queue seconds hold {sum(qsec[b] for b in top) / tot:.0%}")
print("minute: queue s accrued | GPU pool use mean, max | running mean | queued mean | prefill tokens K, batches | turns with 8K+ new tokens | first turns, their new tokens K")
for b in sorted(top):
    print(f"  {b:4d}: {qsec[b]:7.0f} | {st.mean(use[b]):.2f} {max(use[b]):.2f} | {st.mean(running[b]):5.1f} | {st.mean(queue_len[b]):5.1f} | {pre_tok[b] / 1000:6.0f} {pre_n[b]:4d} | {big[b]:3d} | {n_first[b]:3d} {newtok_first[b] / 1000:6.0f}")
calm = [b for b in qsec if b not in top and 20 <= b < 118]
print(f"the other minutes from 20 on, means: queue s {st.mean(qsec[b] for b in calm):.1f} | pool use {st.mean(st.mean(use[b]) for b in calm if use[b]):.2f} | running {st.mean(st.mean(running[b]) for b in calm if running[b]):.1f} | "
      f"prefill K {st.mean(pre_tok[b] for b in calm) / 1000:.0f} | 8K+ turns {st.mean(big[b] for b in calm):.1f} | first turns {st.mean(n_first[b] for b in calm):.1f}")
