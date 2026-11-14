"""ec_r134: the learned price by minute beside what the room did, against the fixed-price run's log in the same minutes."""
import collections, datetime as dt, glob, json, os, re, statistics as st
P = re.compile(r"^\[(.{23})\] HICACHE_EVT ec_price price=(\S+) refused=(\d+) idle=(\d+) excess=(\S+) room=(\d+) held=(\d+) staged=(\d+) harm_ms=(\d+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
def load(run):
    path = sorted(glob.glob(f"results/{run}/client_*/client.jsonl"))[-1]
    t0 = json.loads(open(path).readline())["t_start"]
    by = collections.defaultdict(list)
    for line in open(f"results/{run}/server.log", errors="replace"):
        if "ec_price" in line and (m := P.match(line)):
            by[int((stamp(m[1]) - t0) // 60)].append([float(x) for x in m.groups()[1:]])
    return by
learned, fixed = load("20261112_153600_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"), load("20261111_070012_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70")
row = lambda v: (st.mean(x[0] for x in v), sum(x[1] for x in v) / 1000, st.mean(x[2] for x in v) / 1000, st.mean(x[6] for x in v) / 1000, sum(x[7] for x in v) / 1000)
print("minute | learned price: price, refused K in the minute, idle room K, staged pins K, harm s | fixed price run: refused K, idle room K, staged pins K, harm s")
for m in list(range(44, 68, 2)) + list(range(78, 96, 2)):
    a = [x for k in (m, m + 1) for x in learned.get(k, [])]
    b = [x for k in (m, m + 1) for x in fixed.get(k, [])]
    if a and b:
        pa, pb = row(a), row(b)
        print(f"  {m:3d}-{m + 1:3d} | {pa[0]:.1e} {pa[1]:6.0f} {pa[2]:6.0f} {pa[3]:5.1f} {pa[4]:6.0f} | {pb[1]:6.0f} {pb[2]:6.0f} {pb[3]:5.1f} {pb[4]:6.0f}")
calm = [x for k, v in learned.items() if not (52 <= k <= 64 or 85 <= k <= 92) for x in v]
busy = [x for k, v in learned.items() if 52 <= k <= 64 or 85 <= k <= 92 for x in v]
for name, v in (("minutes 52-64 and 85-92", busy), ("the other minutes", calm)):
    print(f"learned-price run, {name}: price median {st.median(x[0] for x in v):.1e}, idle room K mean {st.mean(x[2] for x in v) / 1000:.0f}, refused K per minute {sum(x[1] for x in v) / 1000 / (len(v) / 12):.0f}, "
          f"staged pins K {st.mean(x[6] for x in v) / 1000:.1f}, harm s per minute {sum(x[7] for x in v) / 1000 / (len(v) / 12):.1f}")
