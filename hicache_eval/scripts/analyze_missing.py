"""Summarise the three previously-missing sub-experiments."""
import json
import os
import re

import pandas as pd

R = os.environ["RESULTS"]
out = {}

d = pd.read_csv(os.path.join(R, "exp1", "ttft_by_tier.csv"))

# --- A: timeout vs wait_complete, L3 -------------------------------------
t = d[(d.prefetch_policy == "timeout") & (d.tier == "L3")]
w = d[(d.prefetch_policy == "wait_complete") & (d.tier == "L3")
      & (d.background_load == "none")]
rows = []
for L in sorted(set(t.L) | set(w.L)):
    a, b = t[t.L == L], w[w.L == L]
    rows.append({
        "L": int(L),
        "timeout_median_s": round(a.ttft_s.median(), 3) if len(a) else None,
        "wait_complete_median_s": round(b.ttft_s.median(), 3) if len(b) else None,
        "timeout_discards": int(a.discards.sum()) if len(a) else 0,
        "n_timeout": len(a), "n_wait": len(b)})
out["A_timeout_vs_wait_complete_L3"] = rows

log = os.path.join(R, "exp1_timeout", "run.log")
if os.path.exists(log):
    txt = open(log).read()
    ev = [(int(m.group(1)), int(m.group(2)))
          for m in re.finditer(r"discard \(bad attribution\) L=(\d+) rep=\d+: "
                               r"dev=\d+ host=\d+ stor=(\d+)", txt)]
    out["A_deadline_misses"] = {
        "partial_load_events": len(ev),
        "by_length": {str(L): sorted(s for l, s in ev if l == L)
                      for L in sorted({l for l, _ in ev})},
        "_meaning": ("Under --hicache-storage-prefetch-policy timeout the prefetch "
                     "returned FEWER storage tokens than the prefix and the rest was "
                     "recomputed. Each event is one deadline-truncated read: the read "
                     "was paid for and only partly used.")}

# --- B: background-load variant ------------------------------------------
bg = d[d.background_load != "none"]
if len(bg):
    rows = []
    for (tier, L), g in bg.groupby(["tier", "L"]):
        base = d[(d.tier == tier) & (d.L == L) & (d.background_load == "none")]
        rows.append({"tier": tier, "L": int(L), "n": len(g),
                     "bg_median_s": round(g.ttft_s.median(), 4),
                     "single_median_s": round(base.ttft_s.median(), 4) if len(base) else None,
                     "ratio_bg_over_single": (round(g.ttft_s.median() / base.ttft_s.median(), 2)
                                              if len(base) and base.ttft_s.median() else None)})
    out["B_background_load"] = sorted(rows, key=lambda r: (r["tier"], r["L"]))
    rec = bg[bg.tier == "recompute"]
    if len(rec):
        big = rec[rec.L == rec.L.max()]
        if len(big):
            out["B_prefill_throughput_under_load"] = {
                "L": int(big.L.iloc[0]),
                "median_ttft_s": round(big.ttft_s.median(), 3),
                "P_tok_per_s": round(big.L.iloc[0] / big.ttft_s.median(), 0),
                "_vs_single_request_P": 15745}

# --- C: exp2 timeout sweep ------------------------------------------------
p = os.path.join(R, "exp2", "interference.csv")
if os.path.exists(p):
    e = pd.read_csv(p)
    c = e[e.control == "timeout_policy"]
    if len(c):
        rows = []
        for r, g in c.groupby("requested_rps"):
            rows.append({"requested_rps": float(r), "n": len(g),
                         "ttft_median_s": round(g.ttft_s.median(), 3),
                         "l3_hits": int(g.cached_storage.notna().sum()),
                         "median_cached_storage": (float(g.cached_storage.median())
                                                   if g.cached_storage.notna().any() else None),
                         "unfulfilled_tokens": float(g.unfulfilled_tokens.max()),
                         "write_gbps_actual": round(float(g.write_gbps_actual.max()), 4)})
        out["C_exp2_timeout_sweep"] = rows

json.dump(out, open(os.path.join(R, "missing_runs_summary.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
