"""Independent recomputation of checks/holds.out, checks/hw_window.out, and the write-up's figures that cite no .out file:
the idle room of the price log (median, in the two load episodes, refused per interval), the first staged read of the
JIT p50 runs, and park/dispatch pairing.
  nice -n 19 python3 -I -B repro_holds.py"""
import bisect, collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

rk = rank()
print("[1] holds.out: arm | returning turns with stats | read thread busy s of span (%) | parked after arrival s/turn, of it thread idle s/turn | turns parked >0.2 s, mean parked, idle part | staged park s | "
      "parked>0.2 s idle with a read: turns, idle-parked s; admitted within 0.1 s of landing: turns, s/turn; others waited more s | busy % per 5 min max | park/dispatch pairing")
for a in ("nofc", "ec_r13", "ec_r1", "jit_exact", "jit_p50_ctl"):
    d = arm(a)
    ev = d["server"]["evts"]
    seq = sorted([(e["t"], 0, "park", (e["conv"], e["turn"]), e) for e in ev["srs_park"]] + [(e["t"], 1, "disp", (e["conv"], e["turn"]), e) for e in ev["srs_dispatch"]], key=lambda x: (x[0], x[1]))
    # the log's own line order matters when times tie; rebuild it from the cache's event order per type is not possible, so
    # pair in time order with park before dispatch on ties (as the lines appear in the log samples)
    parks, windows, orphans, overwritten = {}, [], 0, 0
    for t, _, kind, k, e in seq:
        if kind == "park":
            if k in parks:
                overwritten += 1
            parks[k] = t
        elif k in parks:
            windows.append((k, parks.pop(k), t, e["launched"] == 1))
        else:
            orphans += 1
    ios = sorted((t - ms, t) for v in d["ios"].values() for t, ms, _ in v)
    busy = []
    for lo, hi in ios:
        if busy and lo <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], hi)
        else:
            busy.append([lo, hi])
    starts = [b[0] for b in busy]

    def busy_in(lo, hi):
        if hi <= lo:
            return 0.0
        i = max(0, bisect.bisect_right(starts, lo) - 1)
        tot = 0.0
        while i < len(busy) and busy[i][0] < hi:
            tot += max(0.0, min(hi, busy[i][1]) - max(lo, busy[i][0]))
            i += 1
        return tot

    t0, t1 = min(x[0] for x in ios), max(x[1] for x in ios)
    n = sum(1 for k in d["entry"] if k[1] > 0)
    held = idle = pre = 0.0
    turns, idle_by = collections.Counter(), collections.Counter()
    for k, park, disp, launched in windows:
        if k not in d["entry"]:
            continue
        arrive, wait = d["entry"][k], d["queue"][k]
        lo, hi = max(park, arrive), min(disp, arrive + wait)
        pre += max(0.0, min(disp, arrive) - park)
        if hi > lo:
            b = busy_in(lo, hi)
            held += hi - lo
            idle += hi - lo - b
            turns[k] += hi - lo
            idle_by[k] += hi - lo - b
    over = [k for k in turns if turns[k] > 0.2]
    last_io = {k: max(t for t, _, _ in v) for k, v in d["ios"].items()}
    lost = [(k, idle_by[k], d["entry"][k] + d["queue"][k] - last_io[k]) for k in idle_by if idle_by[k] > 0.2 and k in last_io]
    quick = [x for x in lost if x[2] < 0.1]
    rest = [x[2] for x in lost if x[2] >= 0.1]
    tot_busy = sum(b[1] - b[0] for b in busy)
    per5 = [100 * busy_in(t0 + 300 * i, t0 + 300 * (i + 1)) / 300 for i in range(int((t1 - t0) // 300))]
    run_span = d["every"][-1]["t_send"]
    print(f"  {a:12s} {n:6d} | {tot_busy:5.0f} of {t1 - t0:5.0f} ({100 * tot_busy / (t1 - t0):3.0f} %; of the {run_span:.0f} s run {100 * tot_busy / run_span:3.0f} %) | {held / n:.3f} {idle / n:.3f} | {len(over):4d} {mean(turns[k] for k in over):.2f} {mean(idle_by[k] for k in over):.2f} | {pre:5.0f} | "
          f"{len(lost):4d} {sum(x[1] for x in lost):5.0f}; {len(quick):3d} {sum(x[1] for x in quick) / n:.3f}; {mean(rest):.2f} | {max(per5):3.0f} | windows {len(windows)}, dispatch without a park {orphans}, park overwritten {overwritten}")

print("\n[2] idle room from the price log (ec_price, one line per 5 s): arm | lines | idle K median, mean | in the two load episodes (by the arm's own clock span of its turns at order positions 5500-6999 and 8750-10999): "
      "idle K median, share of intervals with idle < 10K, refused K per interval mean | outside: refused K per interval")
for a in ("ec_r13", "nofc", "ec_r1", "nofc_noend"):
    d = arm(a)
    pr = d["server"]["evts"].get("ec_price", [])
    spans = []
    for lo, hi in EPISODES:
        ts = [r["t_send"] for k, r in d["by"].items() if lo <= rk.get(k, -1) < hi]
        spans.append((d["t0"] + pct(ts, .01), d["t0"] + pct(ts, .99)))
    inside = [e for e in pr if any(lo <= e["t"] < hi for lo, hi in spans)]
    outside = [e for e in pr if not any(lo <= e["t"] < hi for lo, hi in spans)]
    cells = []
    for i, (lo, hi) in enumerate(spans):
        x = [e for e in pr if lo <= e["t"] < hi]
        cells.append(f"episode {i + 1} (min {(lo - d['t0']) / 60:.0f}-{(hi - d['t0']) / 60:.0f}): idle median {st.median(e['idle'] for e in x) / 1000:.1f}K, under 10K {100 * sum(e['idle'] < 10000 for e in x) / len(x):.0f} %, refused {mean(e['refused'] for e in x) / 1000:.1f}K")
    print(f"  {a:11s} {len(pr):5d} | idle median {st.median(e['idle'] for e in pr) / 1000:.1f}K mean {mean(e['idle'] for e in pr) / 1000:.1f}K | both episodes: idle median {st.median(e['idle'] for e in inside) / 1000:.1f}K, "
          f"under 10K {100 * sum(e['idle'] < 10000 for e in inside) / len(inside):.0f} %, refused {mean(e['refused'] for e in inside) / 1000:.1f}K | " + " | ".join(cells)
          + f" | outside refused {mean(e['refused'] for e in outside) / 1000:.1f}K | whole run refused {mean(e['refused'] for e in pr) / 1000:.1f}K")

print("\n[3] first staged read (ntp_stage) of each JIT p50 run: key, seconds after the first arrival (first ReqTimeStats entry_time of a c-t request), seconds after the client's t_start")
for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact"):
    d = arm(a)
    stg = sorted(d["server"]["evts"]["ntp_stage"], key=lambda e: e["t"])
    first_arr = min(d["entry"].values())
    first_ps = min(e["t"] for e in d["server"]["evts"]["prefetch_start"] if e["conv"] is not None)
    print(f"  {a:14s} first: c{stg[0]['conv']}-t{stg[0]['turn']} at {stg[0]['t'] - first_arr:7.2f} s after the first arrival = {int((stg[0]['t'] - first_arr) // 60)} min {(stg[0]['t'] - first_arr) % 60:4.1f} s "
          f"(after first prefetch_start {stg[0]['t'] - first_ps:7.2f} s; after t_start {stg[0]['t'] - d['t0']:7.2f} s); second: c{stg[1]['conv']}-t{stg[1]['turn']} at {stg[1]['t'] - first_arr:7.2f} s")

print("\n[4] hw_window: decode ms per step with no queue by running requests, first-turn prefill ms/token, SSD ms/page for reads of 500+ pages, first 74.7 min of each run's clock")
END = 74.7 * 60
for a in ("baseline", "base_rep", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "ec_r13", "nofc_noend", "hintfree_ctl"):
    d = arm(a)
    dec = [(n, 1000 * n / thr) for t, n, _, _, thr, q in d["server"]["dec"] if 0 <= t - d["t0"] < min(END, d["end"]) and q == 0 and thr > 0]
    cells = []
    for lo, hi in ((8, 15), (16, 23), (24, 31)):
        v = [ms for n, ms in dec if lo <= n <= hi]
        cells.append(f"{lo}-{hi} {st.median(v):5.2f} ({len(v):4d})")
    first = [1000 * (r["ttft"] - d["queue"].get((r["conv"], r["turn"]), 0.0)) / (r["prompt_tokens"] - (r["cached_tokens"] or 0)) for r in d["turns"]
             if r["turn"] == 0 and r["t_send"] < END and d["queue"].get((r["conv"], r["turn"]), 0.0) < 0.05 and (r["prompt_tokens"] or 0) - (r["cached_tokens"] or 0) > 2000]
    ssd = [1000 * ms / p for v in d["ios"].values() for t, ms, p in v if t - d["t0"] < END and p >= 500]
    ssd_all = [1000 * ms / p for v in d["ios"].values() for t, ms, p in v if t - d["t0"] < END and p >= 100]
    print(f"  {a:14s} decode " + " ".join(cells) + f" | prefill {st.median(first):.3f} ({len(first)}) | SSD {st.median(ssd):.2f} ({len(ssd)}) | SSD reads of 100+ pages {st.median(ssd_all):.2f} ({len(ssd_all)})")
