#!/usr/bin/env python3
"""Data sanity pass over the cached runs (reads cache/*.json only).

Run as:  nice -n 19 python3 -I -B explore.py
"""
import json
import os
from collections import Counter, defaultdict

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
CACHE = OUT + "/cache"
LABELS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl",
          "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "order_run"]

lines = []


def P(s=""):
    lines.append(s)
    print(s)


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


for lab in LABELS:
    with open(os.path.join(CACHE, lab + ".json")) as f:
        d = json.load(f)
    T = d["turns"]
    P("== %s  (%s)" % (lab, d["folder"]))
    n = len(T)
    null_ttft = [t for t in T if t["ttft"] is None]
    err = [t for t in T if t["error"]]
    no_tsend = [t for t in T if t["t_send"] is None]
    keys = Counter((t["conv"], t["turn"]) for t in T)
    dupk = {k: c for k, c in keys.items() if c > 1}
    ok = [t for t in T if t["ttft"] is not None]
    max_ts_ok = max(t["t_send"] for t in ok)
    max_ts_all = max(t["t_send"] for t in T if t["t_send"] is not None)
    kept = [t for t in sorted(ok, key=lambda t: t["t_send"]) if t["t_send"] < max_ts_ok - 300]
    kept_all = [t for t in sorted(ok, key=lambda t: t["t_send"]) if t["t_send"] < max_ts_all - 300]
    ts_ties = sum(1 for c in Counter(t["t_send"] for t in ok).values() if c > 1)
    neg = [t for t in T if t["ttft"] is not None and t["ttft"] < 0]
    lat_null = [t for t in ok if t["latency"] is None]
    P("  turn records=%d  ttft null=%d  error set=%d  t_send null=%d  dup (conv,turn)=%d  ttft<0=%d  latency null among ok=%d"
      % (n, len(null_ttft), len(err), len(no_tsend), len(dupk), len(neg), len(lat_null)))
    P("  max t_send: with ttft=%.3f  all=%.3f  kept=%d (returning %d, first %d)  kept if max over all records=%d  t_send tie groups=%d"
      % (max_ts_ok, max_ts_all, len(kept), sum(1 for t in kept if t["turn"] > 0),
         sum(1 for t in kept if t["turn"] == 0), len(kept_all), ts_ties))
    if null_ttft:
        P("  sample failed turn: %s" % json.dumps(null_ttft[0])[:400])
    if err:
        P("  error values: %s" % Counter(str(t["error"])[:80] for t in err).most_common(5))
    rr = Counter()
    for t in T:
        v = t["reply_reprefilled"]
        rr["null" if v is None else ("neg" if v < 0 else ("zero" if v == 0 else "pos"))] += 1
    P("  reply_reprefilled: %s" % dict(rr))
    # file order vs t_send order
    inv = sum(1 for a, b in zip(T, T[1:]) if a["t_send"] is not None and b["t_send"] is not None and b["t_send"] < a["t_send"])
    P("  file-order inversions in t_send=%d" % inv)

    # server side
    rts = d["rts"]
    rk = Counter((r[1], r[2]) for r in rts)
    rdup = {k: c for k, c in rk.items() if c > 1}
    ckeys = set(keys)
    okkeys = set((t["conv"], t["turn"]) for t in ok)
    P("  ReqTimeStats keyed lines=%d  distinct keys=%d  keys with >1 line=%d (max %d)  client keys without RTS=%d (of which with ttft %d)  RTS keys not in client=%d"
      % (len(rts), len(rk), len(rdup), max(rk.values()) if rk else 0,
         len(ckeys - set(rk)), len(okkeys - set(rk)), len(set(rk) - ckeys)))
    if rdup:
        for k in list(rdup)[:5]:
            P("    dup sample %s: %s" % (k, [r for r in rts if (r[1], r[2]) == k]))
    att = Counter(r[9] for r in rts)
    P("  attempts values: %s" % dict(att))
    q = [r[3] for r in rts]
    P("  queue_duration ms: min=%.3f p50=%.3f p99=%.1f max=%.1f  negatives=%d"
      % (min(q), pct(q, 0.5), pct(q, 0.99), max(q), sum(1 for x in q if x < 0)))
    s2h = d["s2h"]
    sk = Counter((r[1], r[2]) for r in s2h)
    P("  s2h_io lines=%d distinct keys=%d  keys not in client=%d  keys w/o RTS=%d  turn0 lines=%d  pages sum=%d  tokens==0 lines=%d  ms min/p50/max=%s"
      % (len(s2h), len(sk), len(set(sk) - ckeys), len(set(sk) - set(rk)),
         sum(1 for r in s2h if r[2] == 0), sum(r[3] for r in s2h),
         sum(1 for r in s2h if r[4] == 0),
         ("%.0f/%.0f/%.0f" % (min(r[5] for r in s2h), pct([r[5] for r in s2h], 0.5), max(r[5] for r in s2h))) if s2h else "-"))
    P("  s2h lines per key histogram: %s" % dict(sorted(Counter(sk.values()).items())))
    ntp = d["ntp"]
    nk = Counter((r[1], r[2]) for r in ntp)
    P("  ntp_stage lines=%d distinct keys=%d keys not in client=%d  per-key hist=%s"
      % (len(ntp), len(nk), len(set(nk) - ckeys), dict(sorted(Counter(nk.values()).items()))))
    srs = d["srs"]
    P("  srs_dispatch lines=%d launched hist=%s  keys=%d  refused keys=%d"
      % (len(srs), dict(Counter(r[7] for r in srs)), len(set((r[1], r[2]) for r in srs)),
         len(set((r[1], r[2]) for r in srs if r[7] == 0))))
    P("  evict_device lines=%d tokens=%d   evict_host lines=%d tokens=%d"
      % (len(d["evd"]), sum(r[1] for r in d["evd"]), len(d["evh"]), sum(r[1] for r in d["evh"])))
    P("  t_start=%.3f  first log=%.3f (%.1f s before t_start)  last log=%.3f (%.1f s after t_start)  last ok t_send=%.1f"
      % (d["t_start"], d["first_t"], d["t_start"] - d["first_t"], d["last_t"],
         d["last_t"] - d["t_start"], max_ts_ok))
    P()

with open(os.path.join(OUT, "explore.out"), "w") as f:
    f.write("\n".join(lines) + "\n")
