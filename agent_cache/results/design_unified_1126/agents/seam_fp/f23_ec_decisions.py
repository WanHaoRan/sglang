"""F2 / F3 check (CPU, read-only): the engine's own read_start() on a recorded run's expected-cost decisions, with
(F3) a read time that counts the line in front of the read thread and (F2) the lateness weight c.

Each ec_start line gives a pending plan's missing tokens, idle time, read time r (margin included), queue wait W
and price at one change of decision. The plan's gap entry is rebuilt from the client records: the entry key the
engine picked (ntp_pick) and the empirical p10/p50/p90 of the gaps the run realised for that key (the table's final
state; the online table at commit time is not recorded, so this is an approximation, checked by how often the
recomputed decision equals the logged one). Rent = price x missing tokens (room_tokens is not logged; a lower bound).
Variants:
  base    read_start as the engine ran it (r, W logged).
  F3      r + line(t), line(t) = median park -> dispatch wait of timed reads dispatched within +-60 s of the decision.
  F2      stall weighted by c(t): the per-minute c rebuilt by c_series.py.
  F2+F3   both.
Usage: <venv python> f23_ec_decisions.py <label>
"""
import bisect, collections, importlib.util, json, os, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

spec = importlib.util.spec_from_file_location("ntp_policy", "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py")
pol = importlib.util.module_from_spec(spec); sys.modules["ntp_policy"] = pol; spec.loader.exec_module(pol)

def read_start_c(*, remaining, read_s, wait_s, rent_per_s, c):
    """next_turn_policy.read_start with each second of stall weighted c (c = 1 is the engine's rule)."""
    stall_at_arrival = max(0.0, read_s - wait_s)
    count = len(remaining)
    if count == 0 or stall_at_arrival == 0.0:
        return None
    prefix = [0.0]
    for g in remaining:
        prefix.append(prefix[-1] + g)
    total = prefix[-1]; lead = read_s - wait_s
    best_cost, best_start = c * stall_at_arrival, None
    for start in sorted({0.0, *(max(0.0, g - lead) for g in remaining)}):
        arrived = bisect.bisect_right(remaining, start)
        late = bisect.bisect_right(remaining, start + lead)
        stall = arrived * stall_at_arrival + ((late - arrived) * (start + lead) - (prefix[late] - prefix[arrived]))
        pinned = (total - prefix[late]) - (count - late) * (start + lead)
        cost = (c * stall + rent_per_s * pinned) / count
        if cost < best_cost:
            best_cost, best_start = cost, start
    return best_start

label = sys.argv[1]
r = lib.Run(label)
cser = json.load(open(os.path.join(lib.HERE, f"c_series_{label}.json")))
def c_at(t):
    m = int(r.minute(t))
    for d in (0, -1, 1, -2, 2):
        v = cser.get(str(m + d))
        if v and v[0] is not None:
            return v[0]
    return 1.0

# gaps realised per key, from the client records
by_next = {}
for rid, cl in r.client.items():
    h = cl.get("hint") or cl.get("dist_hint")
    if h and h.get("next_rid"):
        by_next[h["next_rid"]] = h.get("reply_key")
gaps_key = collections.defaultdict(list); gaps_tool = collections.defaultdict(list); gaps_any = []
for rid, cl in r.client.items():
    key = by_next.get(rid)
    g = cl.get("wait_s")
    if key is None or g is None:
        continue
    gaps_key[key].append(g); gaps_tool[key.split("/", 1)[0]].append(g); gaps_any.append(g)
def quant(xs):
    xs = sorted(xs)
    if len(xs) < 2:
        return None
    return tuple(xs[min(len(xs) - 1, int(p * len(xs)))] for p in (0.1, 0.5, 0.9))
entry_of = {}
for e in r.kind["ntp_pick"]:
    entry_of[e["rid"]] = (e.get("key"), e.get("entry"))
def dist_of(rid):
    key, entry = entry_of.get(rid, (None, None))
    if entry in (None, "none", "_end"):
        return None
    if entry == "any":
        qs = quant(gaps_any)
    elif entry in gaps_key and entry == key:
        qs = quant(gaps_key[entry])
    elif entry in gaps_tool:
        qs = quant(gaps_tool[entry])
    else:
        qs = quant(gaps_key.get(entry, []))
    if qs is None:
        return None
    return pol.GapDistribution.from_quantiles(p10=qs[0], p50=qs[1], p90=qs[2])

# line in front of the read thread: park -> dispatch of timed reads
disp = sorted((e["t"], e["waited_ms"] / 1e3) for e in r.kind["srs_dispatch"])
park_dl = {}
for e in r.kind["srs_park"]:
    park_dl.setdefault(e["rid"], []).append(e)
timed = []
for e in r.kind["srs_dispatch"]:
    ps = [p for p in park_dl.get(e["rid"], []) if p["t"] <= e["t"]]
    if ps and ps[-1]["deadline_ms"] != "now":
        timed.append((e["t"], e["waited_ms"] / 1e3))
timed.sort(); tt = [x[0] for x in timed]
def line_at(t):
    lo_i = bisect.bisect_left(tt, t - 60); hi_i = bisect.bisect_right(tt, t + 60)
    w = [x[1] for x in timed[lo_i:hi_i]]
    return lib.q(w, .5) if w else 0.0

def cls(s):
    if s is None:
        return "arrival"
    return "now" if s == 0.0 else ("soon" if s <= 5.0 else "wait")
rows = []
skipped = 0
for e in r.kind["ec_start"]:
    d = dist_of(e["rid"])
    if d is None:
        skipped += 1
        continue
    idle = e["idle_ms"] / 1e3 - 0.10
    rem = pol.remaining_gaps(support=d.support(), idle_s=idle)
    read_s = e["read_ms"] / 1e3; wait_s = e["wait_ms"] / 1e3
    rent = float(e["price"]) * e["tokens"]
    c = c_at(e["t"]); line = line_at(e["t"])
    base = pol.read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent)
    f3 = pol.read_start(remaining=rem, read_s=read_s + line, wait_s=wait_s, rent_per_s=rent)
    f2 = read_start_c(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent, c=c)
    f23 = read_start_c(remaining=rem, read_s=read_s + line, wait_s=wait_s, rent_per_s=rent, c=c)
    rows.append(dict(m=r.minute(e["t"]), logged=e["decision"], base=cls(base), f3=cls(f3), f2=cls(f2), f23=cls(f23),
                     c=c, line=line, read_s=read_s, wait_s=wait_s, tokens=e["tokens"], idle=idle))
print(f"== {label}: {len(rows)} ec_start decisions with a rebuilt entry ({skipped} skipped: no entry)")
agree = sum((x["logged"] == x["base"]) or (x["logged"] == "wait" and x["base"] == "soon") for x in rows)
print(f"   recomputed base equals the logged decision for {agree}/{len(rows)} ({100*agree/max(1,len(rows)):.1f} %)")
for lo_m, hi_m in ((0, 30), (30, 70), (70, 110), (0, 121)):
    w = [x for x in rows if lo_m <= x["m"] < hi_m]
    if not w:
        continue
    print(f"   minutes {lo_m}-{hi_m}: n {len(w)}; c median {lib.q([x['c'] for x in w],.5):.2f}; line median {lib.q([x['line'] for x in w],.5):.1f} s [p90 {lib.q([x['line'] for x in w],.9):.1f}]; "
          f"r median {lib.q([x['read_s'] for x in w],.5):.1f} s; W median {lib.q([x['wait_s'] for x in w],.5):.1f} s")
    for name in ("logged", "base", "f3", "f2", "f23"):
        cnt = collections.Counter(x[name] for x in w)
        print(f"       {name:6s}: " + ", ".join(f"{k} {cnt.get(k,0)}" for k in ("arrival", "wait", "soon", "now")))
