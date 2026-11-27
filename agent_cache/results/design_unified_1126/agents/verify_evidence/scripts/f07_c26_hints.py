"""ev_fetch 1c: were campaign 26's hints sharp?  Entries the client sent (dist_hint p10/p50/p90) and the true gaps,
arm nofc_noend (LMCache agentic trace, gaps x10)."""
import sys, os, json, math, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import pct
run, prog = vlib.manifest(26)["nofc_noend"]
p = f"{vlib.RES}/{run}/{prog}/client.jsonl"
ratios = []; lv = collections.Counter(); gaps = collections.defaultdict(list); inside = 0; n = 0
for line in open(p):
    d = json.loads(line)
    if d.get("kind") != "turn": continue
    h = d.get("dist_hint") or {}
    if h.get("true_gap_s") is not None and h.get("reply_key"):
        gaps[h["reply_key"]].append(h["true_gap_s"])
    if h.get("p50") is None: continue
    n += 1; lv[h.get("matched")] += 1
    ratios.append(h["p90"] / max(h["p10"], 1e-9))
    if h.get("true_gap_s") is not None and h["p10"] <= h["true_gap_s"] <= h["p90"]: inside += 1
print(f"c26 nofc_noend: entries sent {n}; matched level {dict(lv)}; p90/p10 median {pct(ratios,.5):.2f} p10 {pct(ratios,.1):.2f} p90 {pct(ratios,.9):.2f}; true gap inside p10-p90: {100*inside/n:.1f} %")
allg = [g for v in gaps.values() for g in v]
print(f"true gaps (x10): n {len(allg)} p10/p50/p90 {pct(allg,.1):.2f}/{pct(allg,.5):.2f}/{pct(allg,.9):.2f} s")
BINS = [0, 1, 2, 4, 6, 10, 20, 60, 300, 1200]
def hz(g):
    out = []
    for lo, hi in zip(BINS, BINS[1:]):
        risk = sum(1 for x in g if x >= lo); evn = sum(1 for x in g if lo <= x < hi)
        out.append(-math.log(1 - evn / risk) / (hi - lo) if 0 < risk and evn < risk else float("nan"))
    return out
print("pooled hazard per s: " + " ".join(f"{lo}-{hi}:{h:.3f}" for (lo, hi), h in zip(zip(BINS, BINS[1:]), hz(allg))))
big = sorted(gaps.items(), key=lambda x: -len(x[1]))[:8]
for k, v in big:
    print(f"   {k:28s} n {len(v):5d} p10/p50/p90 {pct(v,.1):6.2f}/{pct(v,.5):6.2f}/{pct(v,.9):6.2f} ratio {pct(v,.9)/max(pct(v,.1),1e-9):5.1f}  hazard " + " ".join(f"{h:.2f}" for h in hz(v)[:7]))
