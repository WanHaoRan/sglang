"""Census by 10-minute window: medians and p90 of what holds host memory (c34 arms).
Usage: python census_windows.py census_<run>.txt"""
import re, sys, statistics as st
from datetime import datetime
F = ["free","mirror","mirror_locked","dev_unbacked","host_only","evict_leaf","evict_casc","h_write","h_lookup","h_parked","h_read","h_pin","h_staged","h_load_back","h_other","h_multi","anc","pending_write","wq","pinned","refused_tok","backup_refused_tok"]
rows = []
for line in open(sys.argv[1]):
    m = re.match(r"\[(\S+ \S+)\]", line)
    t = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").timestamp()
    d = dict(re.findall(r"(\w+)=(-?\d+)", line))
    rows.append((t, {k: int(d.get(k, 0)) for k in F}))
t0 = None
for t, d in rows:
    if d["mirror"] + d["host_only"] > 0:
        t0 = t; break
def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs)-1, int(p*len(xs)))]
print("minutes from first nonzero census | median K tokens (p90)")
show = ["free","mirror","host_only","evict_leaf","evict_casc","h_write","h_pin","h_staged","h_parked","h_lookup","h_read","h_load_back","anc","pending_write"]
print("win   " + " ".join(f"{k:>14}" for k in show))
for w0 in range(0, 130, 10):
    sel = [d for t, d in rows if w0*60 <= t - t0 < (w0+10)*60]
    if not sel: continue
    print(f"{w0:3d}-{w0+10:<3d}" + " ".join(f"{st.median([d[k] for d in sel])/1e3:7.0f}({q([d[k] for d in sel],0.9)/1e3:5.0f})" for k in show))
