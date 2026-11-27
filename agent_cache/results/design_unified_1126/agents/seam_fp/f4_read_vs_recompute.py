"""F4 check (CPU, read-only): read-or-recompute at light load, from run records.

Prefill model, fitted per run on its own cold prefills at an empty queue (cached_input_len == 0, no SSD read,
queue_duration < 20 ms, >= 1K prompt tokens):   ttft(client) = k0 + a n + b n^2   (n = prompt tokens).
A reader is a request with its own SSD read (s2h_io before admission). For readers that arrived alone (ntp_arrive
pos 0, qlen 1; read issued at arrival, not staged) the script compares
  measured TTFT (client)                                   -- the read path as it ran, and
  T_rec = k0 + a (n - M) + b (n^2 - M^2)                   -- the same request prefilling the L read tokens itself,
with n = prompt tokens, L = tokens the read delivered, M = cached_input_len - L (history that was in memory).
It also reports the measured read-path pieces: lookup -> park, park -> dispatch, I/O, landing -> admission, and the
queue wait as a line in L.
Usage: python3 f4_read_vs_recompute.py <label> [minute_lo minute_hi]
"""
import bisect, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]
lo = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
hi = float(sys.argv[3]) if len(sys.argv) > 3 else 110.0
r = lib.Run(label)

def solve3(A, y):
    # normal equations for 3 unknowns, Gaussian elimination
    M = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(3)] + [sum(A[k][i] * y[k] for k in range(len(A)))] for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda k: abs(M[k][i])); M[i], M[p] = M[p], M[i]
        for k in range(i + 1, 3):
            f = M[k][i] / M[i][i]
            M[k] = [M[k][j] - f * M[i][j] for j in range(4)]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (M[i][3] - sum(M[i][j] * x[j] for j in range(i + 1, 3))) / M[i][i]
    return x

def fit_line(pts):
    n = len(pts)
    if n < 3:
        return float("nan"), float("nan")
    sx = sum(x for x, _ in pts); sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts); sxy = sum(x * y for x, y in pts)
    d = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / d
    return (sy - b * sx) / n, b

cold = []
for rid, rec in r.requests():
    rts = rec["rts"]; cl = r.client.get(rid)
    if cl is None or rec.get("s2h_io") or rts["queue_s"] > 0.02 or rts["cached_input_len"] != 0:
        continue
    if rts["input_len"] < 1000 or cl.get("ttft") is None:
        continue
    cold.append((rts["input_len"], cl["ttft"]))
if len(cold) >= 10:
    k0, a, b = solve3([[1.0, n, n * n] for n, _ in cold], [t for _, t in cold])
    res = [t - (k0 + a * n + b * n * n) for n, t in cold]
else:
    k0 = a = b = float("nan"); res = []
print(f"== {label}: cold prefills at an empty queue: n {len(cold)}; ttft = {k0*1e3:.0f} ms + {a*1e6:.1f} us/token x n + {b*1e9:.3f} ns/token^2 x n^2 "
      f"(residual p10 {lib.q(res,.1)*1e3:.0f} ms, p90 {lib.q(res,.9)*1e3:.0f} ms)")
for lo_t, hi_t in ((1000, 8000), (8000, 16000), (16000, 32000), (32000, 70000)):
    xs = [t / n for n, t in cold if lo_t <= n < hi_t]
    print(f"   cold {lo_t:>5}-{hi_t:<5} tokens: n {len(xs):4d}  ttft/token median {lib.q(xs,.5)*1e6:6.1f} us  [p10 {lib.q(xs,.1)*1e6:6.1f}, p90 {lib.q(xs,.9)*1e6:6.1f}]")
def t_cold(n):
    return k0 + a * n + b * n * n
for n in (8000, 16000, 24000, 32000, 48000, 64000):
    print(f"   model: cold prefill of {n:>5} tokens = {t_cold(n):.2f} s ({t_cold(n)/n*1e6:.1f} us/token; marginal {((a+2*b*n))*1e6:.1f} us/token)")

bt = [x[0] for x in r.batches]
def queue_at(t):
    i = bisect.bisect_right(bt, t) - 1
    return r.batches[i][4] if i >= 0 else 0
def running_at(t):
    i = bisect.bisect_right(bt, t) - 1
    return r.batches[i][3] if i >= 0 else 0

rows = []
n_all = 0
for rid, rec in r.requests():
    rts = rec["rts"]; adm = rts["entry"] + rts["queue_s"]
    m = r.minute(rts["entry"])
    if not lo <= m < hi:
        continue
    n_all += 1
    ios = [e for e in rec.get("s2h_io", []) if e["t"] <= adm + 0.05]
    if not ios:
        continue
    L = sum(e["loaded"] for e in rec.get("landed", []) if e["t"] <= adm + 0.05)
    arr = rec.get("ntp_arrive")
    qlen = arr[0]["qlen"] - 1 if arr else queue_at(rts["entry"])
    pos = arr[0]["pos"] if arr else qlen
    parks = [e for e in rec.get("srs_park", []) if e["t"] <= adm + 0.05]
    disp = [e for e in rec.get("srs_dispatch", []) if e["t"] <= adm + 0.05]
    staged = any(e["t"] < rts["entry"] for e in parks)
    cl = r.client.get(rid, {})
    rows.append(dict(rid=rid, m=m, L=L, wait=rts["queue_s"], pos=pos, qlen=qlen, io_s=sum(e["ms"] for e in ios) / 1e3,
                     pages=sum(e["pages"] for e in ios), waited=sum(e["waited_ms"] for e in disp) / 1e3,
                     lookup=(parks[0]["t"] - rts["entry"]) if parks and not staged else float("nan"),
                     land_to_adm=adm - max(e["t"] for e in ios), staged=staged, ttft=cl.get("ttft"),
                     n=rts["input_len"], cached=rts["cached_input_len"], running=running_at(rts["entry"])))
print(f"   window {lo:.0f}-{hi:.0f} min: {n_all} requests, {len(rows)} with their own SSD read before admission "
      f"({100*len(rows)/max(1,n_all):.1f} %), {sum(x['staged'] for x in rows)} of them staged before arrival")
alone = [x for x in rows if x["pos"] == 0 and x["qlen"] <= 0 and not x["staged"] and x["L"] > 0 and x["ttft"] is not None]
print(f"   'alone' readers (arrived at an empty queue, read issued at arrival): {len(alone)}; running requests at arrival median {lib.q([x['running'] for x in alone],.5)}")
if alone:
    a_r, b_r = fit_line([(x["L"], x["wait"]) for x in alone])
    a_i, b_i = fit_line([(x["pages"], x["io_s"]) for x in alone])
    print(f"     queue wait = {a_r*1e3:.0f} ms + {b_r*1e6:.1f} us/token x L (OLS);   I/O = {a_i*1e3:.0f} ms + {b_i*1e3:.2f} ms/page ({b_i/64*1e6:.1f} us/token)")
    print(f"     lookup->park median {lib.q([x['lookup'] for x in alone if x['lookup']==x['lookup']],.5)*1e3:.0f} ms; park->dispatch median {lib.q([x['waited'] for x in alone],.5)*1e3:.0f} ms; "
          f"landing->admission median {lib.q([x['land_to_adm'] for x in alone],.5)*1e3:.0f} ms; ttft - queue wait median {lib.q([x['ttft']-x['wait'] for x in alone],.5)*1e3:.0f} ms")
    def t_rec(x):
        M = max(0, x["cached"] - x["L"])
        return k0 + a * (x["n"] - M) + b * (x["n"] ** 2 - M ** 2)
    if k0 == k0:
        gains = [x["ttft"] - t_rec(x) for x in alone]
        pos_g = [g for g in gains if g > 0]
        print(f"     measured TTFT median {lib.q([x['ttft'] for x in alone],.5):.2f} s vs recompute model median {lib.q([t_rec(x) for x in alone],.5):.2f} s")
        print(f"     recompute faster for {len(pos_g)}/{len(alone)}; gain = TTFT - T_rec: median {lib.q(gains,.5):+.2f} s [p10 {lib.q(gains,.1):+.2f}, p90 {lib.q(gains,.9):+.2f}]; "
              f"sum of positive gains {sum(pos_g):.1f} s = {sum(pos_g)/max(1,n_all)*1e3:.2f} ms per request of the window; sum of all {sum(gains):+.1f} s")
        for lo_t, hi_t in ((256, 4000), (4000, 12000), (12000, 24000), (24000, 40000), (40000, 70000)):
            sel = [x for x in alone if lo_t <= x["L"] < hi_t]
            if sel:
                g = [x["ttft"] - t_rec(x) for x in sel]
                print(f"       L {lo_t:>5}-{hi_t:<5}: n {len(sel):4d}  TTFT median {lib.q([x['ttft'] for x in sel],.5):.2f} s, T_rec median {lib.q([t_rec(x) for x in sel],.5):.2f} s, gain median {lib.q(g,.5):+.2f} s, recompute faster {sum(v>0 for v in g)}/{len(sel)}")
busy = [x for x in rows if not (x["pos"] == 0 and x["qlen"] <= 0) and not x["staged"]]
if busy:
    print(f"   readers that arrived behind others: {len(busy)}; wait median {lib.q([x['wait'] for x in busy],.5):.1f} s; park->dispatch median {lib.q([x['waited'] for x in busy],.5):.1f} s; L median {lib.q([x['L'] for x in busy],.5):.0f}")
