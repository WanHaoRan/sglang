"""SSD write thread service rate per 10 minutes from h2s_io / h2s_group lines, and read thread rate from s2h_io."""
import sys, re, json, datetime as dt, collections
run, client = sys.argv[1], sys.argv[2]
t0 = json.loads(open(client).readline())['t_start']
TS = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\] HICACHE_EVT (h2s_io|h2s_group|s2h_io|h2s_submit) (.*)')
w = collections.defaultdict(lambda: [0, 0.0, 0, 0])   # pages, ms, ops, skipped
r = collections.defaultdict(lambda: [0, 0.0, 0])
pend_max = collections.defaultdict(int)
for line in open(run, errors='replace'):
    if 'HICACHE_EVT' not in line:
        continue
    m = TS.match(line)
    if not m:
        continue
    t = dt.datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp() - t0
    b = int(t // 600)
    kv = dict(x.split('=', 1) for x in m[3].split() if '=' in x)
    if m[2] == "h2s_io":
        w[b][0] += int(kv["pages"]); w[b][1] += float(kv["ms"]); w[b][2] += 1; w[b][3] += int(kv.get("skipped", 0) or 0)
    elif False:
        # in campaign 34 a group line follows its member ios; count io lines only when no group lines exist
        pass
    elif m[2] == 's2h_io':
        r[b][0] += int(kv['pages']); r[b][1] += float(kv['ms']); r[b][2] += 1
    elif m[2] == 'h2s_submit':
        pend_max[b] = max(pend_max[b], int(kv.get('pending', 0)))
print('10-min bin | write groups pages ms/page busy% Ktok/s-capacity | reads pages ms/page busy% | pending writes max K')
for b in sorted(set(w) | set(r)):
    pw, mw, nw, sk = w[b]; pr, mr, nr = r[b]
    cw = f"{nw:5d} {pw:7d} {mw/pw if pw else 0:6.2f} {mw/6000:6.1f}% {64*pw/mw if mw else 0:7.1f}" 
    cr = f"{pr:7d} {mr/pr if pr else 0:6.2f} {mr/6000:6.1f}%"
    print(f"{b*10:4d}-{b*10+10:<4d} | {cw} | {cr} | {pend_max[b]/1000:7.1f}")
