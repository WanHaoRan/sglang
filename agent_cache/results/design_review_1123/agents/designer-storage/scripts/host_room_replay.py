"""Open-loop replay of one campaign 34 run's timeline (arrival, admission, finish of every request held as
recorded) through a session-level host tier, to compare host accounting and eviction orders by the SSD
reads they need. A session's prefix is one unit (its latest prompt; outputs are never re-sent in this
replay). Running units sit on the GPU; under write-through they also hold host memory (the mirror), under
exclusive tiering they do not. A unit missing from host at its request's admission needs an SSD read of
that request's history (ideal_cached). Shared prefixes, partial eviction, pins and read timing are ignored.
Usage: python host_room_replay.py server.log client.jsonl"""
import json, re, sys, collections
ADMIT = re.compile(r"rid=(\S+?-c(\d+)-t(\d+))[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
qw = {}
for line in open(sys.argv[1], errors="replace"):
    if "queue_duration=" in line:
        m = ADMIT.search(line)
        if m: qw[m[1]] = (float(m[6]), float(m[4]) / (1000.0 if m[5] == "ms" else 1.0))
t_start = None; reqs = []
for line in open(sys.argv[2]):
    if '"kind":"run"' in line and t_start is None: t_start = json.loads(line)["t_start"]; continue
    if '"kind":"req"' not in line: continue
    r = json.loads(line)
    if r.get("ttft") is None or r["rid"] not in qw: continue
    m = re.match(r".*-c(\d+)-t(\d+)$", r["rid"])
    arr, q = qw[r["rid"]]
    reqs.append(dict(c=int(m[1]), k=int(m[2]), arr=arr, adm=arr + q, fin=t_start + r["t_send"] + r["latency"],
                     hist=r.get("ideal_cached") or 0, prompt=r.get("prompt_tokens") or 0))
t0 = min(r["arr"] for r in reqs)
nxt = {}   # (c, k) -> next request's admission time (oracle next use)
byc = collections.defaultdict(list)
for r in reqs: byc[r["c"]].append(r)
for c, lst in byc.items():
    lst.sort(key=lambda r: r["k"])
    for a, b in zip(lst, lst[1:]): nxt[(c, a["k"])] = b["adm"]
CAP = 1627648 - 17000   # host pool minus the census' median free

def run(exclusive, order, cap=CAP):
    ev = []
    for r in reqs:
        ev.append((r["arr"], 1, r)); ev.append((r["adm"], 2, r)); ev.append((r["fin"], 0, r))
    ev.sort(key=lambda e: (e[0], e[1]))
    host = {}            # c -> size, units in host (idle or queued)
    last = {}            # c -> last use time
    queued = {}          # c -> admission time of its waiting request
    running = {}         # c -> size on GPU
    nextuse = {}         # c -> oracle next admission
    reads = collections.Counter(); nread = collections.Counter(); hist_tot = collections.Counter()
    def used(): return sum(host.values()) + (0 if exclusive else sum(running.values()))
    def evict(now):
        while used() > cap:
            idle = [c for c in host if c not in queued]
            if idle:
                if order == "lru": v = min(idle, key=lambda c: last[c])
                else: v = max(idle, key=lambda c: nextuse.get(c, float("inf")))
            else:
                qs = list(host)
                if not qs: return
                v = max(qs, key=lambda c: queued[c]) if order != "lru" else min(qs, key=lambda c: last[c])
            del host[v]
    for t, kind, r in ev:
        c = r["c"]; w = int((t - t0) // 600) * 10
        if kind == 1:      # arrival: joins the queue
            queued[c] = r["adm"]; last[c] = t
        elif kind == 2:    # admission: the history must be in host (or on the GPU) now
            queued.pop(c, None)
            if r["k"] >= 1 and r["hist"] >= 1024:
                hist_tot[w] += r["hist"]
                if c not in host and c not in running:
                    reads[w] += r["hist"]; nread[w] += 1
            host.pop(c, None); running[c] = r["prompt"]; last[c] = t
            evict(t)
        else:              # finish: KV leaves the GPU into host (or, write-through, the mirror becomes host-only)
            running.pop(c, None); host[c] = r["prompt"]; last[c] = t
            nextuse[c] = nxt.get((c, r["k"]), float("inf"))
            if nextuse[c] == float("inf") and order == "oracle_end": host.pop(c, None)
            evict(t)
    return reads, nread, hist_tot

arms = [("write-through, LRU (queued last)", False, "lru"), ("exclusive, LRU (queued last)", True, "lru"),
        ("write-through, next-use oracle + ends", False, "oracle_end"), ("exclusive, next-use oracle + ends", True, "oracle_end")]
res = {name: run(ex, od) for name, ex, od in arms}
print("SSD read tokens needed, M (requests with a read) by 10-minute window of admission")
print("win     " + " | ".join(f"{n[:34]:>34}" for n, _, _ in arms))
for w in range(0, 130, 10):
    print(f"{w:3d}-{w+10:<3d} " + " | ".join(f"{res[n][0][w]/1e6:24.2f} ({res[n][1][w]:4d})" for n, _, _ in arms))
for n, _, _ in arms:
    rd, nr, ht = res[n]
    print(f"{n}: total {sum(rd.values())/1e6:.1f} M in {sum(nr.values())} requests; minutes 30-110 {sum(rd[w] for w in range(30,110,10))/1e6:.1f} M of {sum(ht[w] for w in range(30,110,10))/1e6:.1f} M history")
