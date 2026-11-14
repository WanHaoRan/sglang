"""Read-only: requests whose prefetch_start (arrival at the server) has no client record, for a cut run.
usage: python3 -I -B censor_check2.py <run folder> [<run folder> ...]"""
import calendar, glob, json, os, re, sys
RES = "/home/wanhr/sglang/agent_cache/results"
cur = open(os.path.join(RES, ".current_run")).read().strip()
TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] ")
PS = re.compile(r"HICACHE_EVT prefetch_start rid=\S*?-c(\d+)-t(\d+)\b")
for run in sys.argv[1:]:
    assert run != cur, "refusing to read the running arm"
    cdir = sorted(glob.glob(os.path.join(RES, run, "client_*")))[-1]
    t0 = None; turns = []
    for line in open(os.path.join(cdir, "client.jsonl")):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        if r.get("kind") == "run": t0 = r["t_start"]
        elif r.get("kind") == "turn": turns.append(r)
    by = {(r["conv"], r["turn"]): r for r in turns}
    ok = sorted((r for r in turns if r.get("ttft") is not None), key=lambda r: r["t_send"])
    last = ok[-1]["t_send"]; cut = last - 300
    arr = {}; n = 0; sample = None
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            if "prefetch_start" not in line: continue
            m = TS.match(line); p = PS.search(line)
            if not m: continue
            n += 1
            if not p:
                if sample is None: sample = line[:200]
                continue
            t = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6]))) + int(m[7]) / 1000.0
            arr.setdefault((int(p[1]), int(p[2])), t - t0)
    missing = sorted((t, k) for k, t in arr.items() if k not in by)
    print(f"{run}\n  client turn records {len(turns)}; last send {last:.1f} s; kept cutoff {cut:.1f} s (minute {cut/60:.2f})")
    print(f"  prefetch_start lines {n}, keyed {len(arr)}" + (f"; unkeyed sample: {sample}" if sample else ""))
    print(f"  arrived at the server with no client record: {len(missing)}; of them before the kept cutoff: {sum(t < cut for t, _ in missing)}")
    for t, k in missing:
        if t < cut:
            print(f"     c{k[0]}-t{k[1]} arrived {t:.1f} s ({(cut - t):.0f} s before the cutoff, {(last - t):.0f} s before the last recorded send)")
    print("  all missing arrivals (s): " + ", ".join(f"{t:.0f}:c{k[0]}-t{k[1]}" for t, k in missing))
    dt = [arr[k] - by[k]["t_send"] for k in arr if k in by]
    dt.sort()
    if dt: print(f"  arrival minus client t_send on recorded turns: median {dt[len(dt)//2]:.3f} s, max {dt[-1]:.3f} s")
