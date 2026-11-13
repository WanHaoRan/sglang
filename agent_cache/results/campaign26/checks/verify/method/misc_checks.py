"""Smaller figures of the Results section that no .out file holds any more, recomputed from the cache.
  nice -n 19 python3 -I -B misc_checks.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

rk = rank()
print("[1] jit_p50_ctl entry: 'on the 9,660 returning turns that it, the hint-free arm and JIT exact all share: 1.46 against 1.60 s (hint-free 1.75, JIT exact 0.72)' and 'positions 5,500-7,000: 3.92 against 5.24 s'")
for names in (["jit_p50_ctl", "jit_p50", "hintfree", "jit_exact"], ["jit_p50_ctl", "jit_p50", "hintfree", "jit_exact", "baseline", "base_rep"]):
    keys = shared(names)
    ks = [k for k in keys if 5500 <= rk.get(k, -1) < 7000]
    print(f"  {len(keys)} shared by {names}: " + "  ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in keys):.2f}" for a in names)
          + f" | positions 5500-7000 ({len(ks)} turns): " + "  ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in ks):.2f}" for a in names))

print("\n[2] evictions inside each arm's kept window (M tokens, GPU and host), turns that shrank while queued (own returning kept turns), from the cache")
import re
for a in ("nofc_noend", "hintfree_ctl", "nofc", "ec_r13", "jit_p50_ctl"):
    d = arm(a)
    ev = d["server"]["evts"]
    dev = sum(e["tokens"] for e in ev["evict_device"] if 0 <= e["t"] - d["t0"] < d["end"]) / 1e6
    host = sum(e["tokens"] for e in ev["evict_host"] if 0 <= e["t"] - d["t0"] < d["end"]) / 1e6
    shr = collections.Counter()
    for t, body in d["server"]["misc"].get("shrank", []):
        m = re.search(r"req=\S+?-c(\d+)-t(\d+) ", body)
        if m:
            shr[(int(m[1]), int(m[2]))] += 1
    own = [k for k in d["by"] if k[1] > 0]
    print(f"  {a:13s} evicted GPU {dev:5.1f} M host {host:5.1f} M | shrank-while-queued turns {sum(1 for k in own if shr[k]):4d} (log lines {len(d['server']['misc'].get('shrank', []))})")

print("\n[3] 'recomputation is level (145 against 131 tokens per turn on the 6,715, 203 against 185 on the 9,702)'")
for names, label in ((ALL11, "6,715 set"), (SEVEN, "9,702 set"), (["nofc_noend", "nofc", "ec_r13", "hintfree_ctl"], "9,703 set")):
    keys = shared(names)
    print(f"  {label} ({len(keys)}): nofc_noend {mean(rec(arm('nofc_noend')['by'][k]) for k in keys):.1f}  nofc {mean(rec(arm('nofc')['by'][k]) for k in keys):.1f}")

print("\n[4] the cut run: what share of the first load episode its kept turns hold")
full = shared(["jit_p50", "jit_p50_ctl", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"])
cut = set(shared(CUT7))
ep_full = [k for k in full if 5500 <= rk.get(k, -1) < 7000]
ep_cut = [k for k in ep_full if k in cut]
print(f"  returning turns at order positions 5500-6999 shared by the six full runs: {len(ep_full)}; of them in the cut run's kept set: {len(ep_cut)} ({100 * len(ep_cut) / len(ep_full):.1f} %)")
for a in ("jit_p50", "jit_p50_ctl", "hintfree_ctl", "nofc_noend", "ec_r13", "jit_exact"):
    d = arm(a)
    miss = [k for k in ep_full if k not in cut]
    print(f"    {a:13s} mean TTFT on the {len(ep_full)}: {mean(d['by'][k]['ttft'] for k in ep_full):.2f}; on the {len(ep_cut)} the cut run kept: {mean(d['by'][k]['ttft'] for k in ep_cut):.2f}; on the {len(miss)} it lacks: {mean(d['by'][k]['ttft'] for k in miss):.2f}; "
          f"summed TTFT of the episode held by the kept part {100 * sum(d['by'][k]['ttft'] for k in ep_cut) / sum(d['by'][k]['ttft'] for k in ep_full):.0f} %")
print("  own-clock minute at which each JIT p50 run sent its last turn at order positions 5500-6999, and its mean TTFT by 2.5 min of its own clock from minute 50:")
for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut"):
    d = arm(a)
    ts = sorted(r["t_send"] / 60 for k, r in d["by"].items() if 5500 <= rk.get(k, -1) < 7000)
    cells = []
    for m in range(50, 90, 5):
        v = [r["ttft"] for r in d["turns"] if m * 60 <= r["t_send"] < (m + 5) * 60 and r["turn"] > 0]
        cells.append(f"{m}: {mean(v):.2f} ({len(v)})" if v else f"{m}: -")
    print(f"    {a:13s} first {ts[0]:.1f}, 95 % by {pct(ts, .95):.1f}, last {ts[-1]:.1f} ({len(ts)} turns) | " + "  ".join(cells))
d = arm("jit_p50_d_cut")
own_pos = len(d["turns"])
print(f"  the cut run's kept turns end at own position {own_pos}; the two full JIT p50 runs reach own position {own_pos} at minute "
      + ", ".join(f"{arm(a)['turns'][own_pos]['t_send'] / 60:.1f} ({a})" for a in ("jit_p50", "jit_p50_ctl")) + " and position 7000 at "
      + ", ".join(f"{arm(a)['turns'][7000]['t_send'] / 60:.1f}" for a in ("jit_p50", "jit_p50_ctl")))

print("\n[5] two old-build JIT p50 runs against each other on sets that end at different points of the episode (later minus earlier, s per turn)")
for names, label in ((CUT7, "the cut set, 6,579"), (CTL9, "the 6,715"), (["jit_p50", "jit_p50_ctl"], "their own shared set"), (["jit_p50", "jit_p50_ctl", "jit_exact", "ec_r13", "nofc_noend"], "with the leading arms")):
    keys = shared(names)
    ep1 = [k for k in keys if 5500 <= rk.get(k, -1) < 7000]
    print(f"  {label} ({len(keys)} turns, {len(ep1)} at positions 5500-6999): {mean(arm('jit_p50_ctl')['by'][k]['ttft'] - arm('jit_p50')['by'][k]['ttft'] for k in keys):+.3f}; "
          f"at positions 5500-6999 {mean(arm('jit_p50_ctl')['by'][k]['ttft'] - arm('jit_p50')['by'][k]['ttft'] for k in ep1):+.3f}")

print("\n[6] hintfree (campaign 21, cut at 117 min): requests that arrived with no client record")
d = arm("hintfree")
arrived = {}
for e in d["server"]["evts"].get("prefetch_start", []):
    if e["conv"] is not None:
        arrived.setdefault((e["conv"], e["turn"]), e["t"] - d["t0"])
missing = sorted((t, k) for k, t in arrived.items() if k not in d["by_all"])
print(f"  {len(missing)} such requests; arrived before the kept cutoff ({d['end']:.0f} s): {sum(t < d['end'] for t, _ in missing)}; earliest at {missing[0][0]:.0f} s")
