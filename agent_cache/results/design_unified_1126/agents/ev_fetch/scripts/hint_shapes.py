"""Part 1: shapes of the gap hints, per key (tool plus first word; _no_call) and per class.

Rebuilds the gap samples as the replay client's table does (replay_programs.py hints_for, --hint-key command
--hint-no-call, gap x1 capped at 1,200 s): the sample of key(reply k-1) is the wait before request k of the same stream.
Population A = trees the campaigns could draw (--max-context 65536), B = the whole trace.

For each key with enough samples and for the pooled classes:
  - the table entry (p10/p50/p90 of ln gap) and the engine's two-piece lognormal through it (next_turn_policy.py);
  - the empirical hazard per idle-time bin (returning gaps only; and with ends counted as never returning);
  - the fitted hazard per bin and the age at which it peaks;
  - F(x|s) = P(G <= s + x | G > s) for x in 1, 2, 4, 6 s: its value at s = 0, its maximum over s and where;
  - how often the stream ends after a reply of this key.

  python hint_shapes.py A|B
"""
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402

BINS = [(0, 1), (1, 2), (2, 4), (4, 6), (6, 10), (10, 20), (20, 60), (60, 300), (300, 1200)]
RANGES = {"0-10": [0, 1, 2, 3, 4], "10-60": [5, 6], "60+": [7, 8]}
XS = (1.0, 2.0, 4.0, 6.0)
S_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 120.0, 300.0)
MIN_AT_RISK = 30
MIN_EVENTS = 5
MIN_KEY_N = 100


def emp_bin_hazard(g, n_inf=0):
    """Per bin: (hazard per s, events, at risk). g ascending numpy array of gaps (capped at 1,200 s: a capped gap
    survives the last bin); n_inf extra sessions that never return (survive every bin)."""
    out = []
    n = len(g)
    for lo, hi in BINS:
        at_risk = n - np.searchsorted(g, lo, side="left") + n_inf
        # events: lo <= gap < hi; a gap equal to the cap is censored, not an event
        ev = np.searchsorted(g, hi, side="left") - np.searchsorted(g, lo, side="left")
        if at_risk < MIN_AT_RISK or ev < MIN_EVENTS or ev >= at_risk:
            out.append((float("nan"), int(ev), int(at_risk)))
            continue
        out.append((-math.log(1.0 - ev / at_risk) / (hi - lo), int(ev), int(at_risk)))
    return out


def fit_bin_hazard(dist):
    out = []
    for lo, hi in BINS:
        s_lo, s_hi = dist.survival(lo), dist.survival(hi)
        out.append(-math.log(max(s_hi, 1e-300) / s_lo) / (hi - lo) if s_lo > 0 else float("nan"))
    return out


def fit_hazard_peak(dist):
    """Age (s) at which the fitted hazard f/S peaks, on a log grid 0.01..1200 s (numerical derivative of -ln S)."""
    ts = np.exp(np.linspace(math.log(0.01), math.log(1200.0), 2400))
    lnS = np.array([math.log(max(dist.survival(float(t)), 1e-300)) for t in ts])
    h = -(lnS[1:] - lnS[:-1]) / (ts[1:] - ts[:-1])
    mid = np.sqrt(ts[1:] * ts[:-1])
    # beyond the fitted p99 the hazard of a near-point entry is numerical noise: look below it only
    h = np.where(lnS[1:] >= math.log(0.01), h, -1.0)
    i = int(np.argmax(h))
    return float(mid[i]), float(h[i])


def trend(hz, idx):
    """Direction of the hazard across the bins idx: slope of ln h on ln(mid age), weighted by events; with z."""
    pts = [(math.log(math.sqrt(max(BINS[i][0], 0.3) * BINS[i][1])), math.log(hz[i][0]), hz[i][1])
           for i in idx if not math.isnan(hz[i][0]) and hz[i][0] > 0]
    if len(pts) < 2:
        return "na", float("nan"), float("nan")
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts]); w = np.array([p[2] for p in pts], float)
    xm = (w * x).sum() / w.sum()
    sxx = (w * (x - xm) ** 2).sum()
    slope = (w * (x - xm) * y).sum() / sxx
    # var(ln h) ~ 1/events per bin (Poisson); slope variance = 1 / sum(w (x - xm)^2) with weights w = events
    z = slope * math.sqrt(sxx)
    lab = "rising" if z > 2 else ("falling" if z < -2 else "flat")
    return lab, float(slope), float(z)


def emp_F(g, x, s):
    """Empirical P(G <= s + x | G > s); (value, at risk)."""
    n = len(g)
    first = np.searchsorted(g, s, side="right")
    at_risk = n - first
    if at_risk < MIN_AT_RISK:
        return float("nan"), int(at_risk)
    return float((np.searchsorted(g, s + x, side="right") - first) / at_risk), int(at_risk)


def describe(name, gaps, n_end, hist, P):
    g = np.sort(np.asarray(gaps, float))
    n = len(g)
    p10, p50, p90 = C.entry_quantiles(g)
    dist = P.GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
    hz = emp_bin_hazard(g)
    hz_end = emp_bin_hazard(g, n_inf=n_end)
    fz = fit_bin_hazard(dist)
    peak_t, peak_h = fit_hazard_peak(dist)
    valid = [(hz[i][0], i) for i in range(len(BINS)) if not math.isnan(hz[i][0])]
    emp_peak_bin = max(valid)[1] if valid else None
    # significant rise between consecutive bins up to 6-10 s: ln ratio z > 2
    rises = []
    for i in range(0, 4):
        a, b = hz[i], hz[i + 1]
        if math.isnan(a[0]) or math.isnan(b[0]):
            continue
        z = (math.log(b[0]) - math.log(a[0])) / math.sqrt(1.0 / a[1] + 1.0 / b[1])
        if z > 2:
            rises.append((f"{BINS[i][0]}-{BINS[i][1]}>{BINS[i + 1][0]}-{BINS[i + 1][1]}", round(b[0] / a[0], 2), round(z, 1)))
    rec = {"key": name, "n": n, "n_end": n_end, "p_end": n_end / (n + n_end) if n + n_end else float("nan"),
           "p10": p10, "p50": p50, "p90": p90, "sharp": p90 / p10, "sigma_low": dist.sigma_low,
           "sigma_high": dist.sigma_high, "mean": float(g.mean()), "share_capped": float((g >= C.GAP_CAP_S).mean()),
           "hist_p50": float(np.median(hist)) if len(hist) else float("nan"),
           "hist_mean": float(np.mean(hist)) if len(hist) else float("nan"),
           "emp_hz": [h[0] for h in hz], "emp_ev": [h[1] for h in hz], "emp_risk": [h[2] for h in hz],
           "emp_hz_end": [h[0] for h in hz_end], "fit_hz": fz, "fit_peak_s": peak_t, "fit_peak_h": peak_h,
           "emp_peak_bin": emp_peak_bin, "emp_rises": rises}
    for rname, idx in RANGES.items():
        lab, slope, z = trend(hz, idx)
        rec[f"emp_trend_{rname}"] = lab; rec[f"emp_slope_{rname}"] = slope; rec[f"emp_z_{rname}"] = z
        fpts = [(math.log(math.sqrt(max(BINS[i][0], 0.3) * BINS[i][1])), math.log(fz[i])) for i in idx if fz[i] > 0]
        rec[f"fit_slope_{rname}"] = float(np.polyfit([p[0] for p in fpts], [p[1] for p in fpts], 1)[0]) if len(fpts) > 1 else float("nan")
    for x in XS:
        f0, _ = emp_F(g, x, 0.0)
        best, best_s = f0, 0.0
        curve = []
        for s in S_GRID:
            v, risk = emp_F(g, x, s)
            curve.append(v)
            if not math.isnan(v) and v > best:
                best, best_s = v, s
        fit0 = dist.chance_within(idle_s=0.0, horizon_s=x)
        fbest, fbest_s = fit0, 0.0
        for s in np.exp(np.linspace(math.log(0.05), math.log(600.0), 400)):
            v = dist.chance_within(idle_s=float(s), horizon_s=x)
            if v > fbest:
                fbest, fbest_s = v, float(s)
        rec[f"F{x:g}_emp0"] = f0; rec[f"F{x:g}_emp_max"] = best; rec[f"F{x:g}_emp_smax"] = best_s
        rec[f"F{x:g}_fit0"] = fit0; rec[f"F{x:g}_fit_max"] = fbest; rec[f"F{x:g}_fit_smax"] = fbest_s
        rec[f"F{x:g}_emp_curve"] = curve
    return rec


def main():
    pop = sys.argv[1] if len(sys.argv) > 1 else "A"
    # "A_sys" / "B_sys": one entry per (agent system, key), as the replay's online table keeps them
    by_system = pop.endswith("_sys")
    P = C.policy()
    trees = C.load_trace(C.MAX_CONTEXT if pop.startswith("A") else None)
    gaps = collections.defaultdict(list); hist = collections.defaultdict(list)
    cls_gaps = collections.defaultdict(list); cls_hist = collections.defaultdict(list)
    tool_gaps = collections.defaultdict(list)
    for (t, s, k, key, gap, kind, keep, prompt, system) in C.iter_gaps(trees):
        base_key = key
        if by_system:
            key = "|".join(system) + "|" + key
        gaps[key].append(gap); hist[key].append(keep)
        key = base_key
        cls = "human(_no_call)" if key == C.NO_CALL else "tool(all tool keys)"
        cls_gaps[cls].append(gap); cls_hist[cls].append(keep)
        cls_gaps["all(*)"].append(gap); cls_hist["all(*)"].append(keep)
        cls_gaps[f"gapkind={kind}"].append(gap); cls_hist[f"gapkind={kind}"].append(keep)
        tool_gaps[C.tool_of(key)].append(gap)
    ends = collections.Counter(); cls_ends = collections.Counter()
    for key, system in C.iter_ends(trees):
        ends[("|".join(system) + "|" + key) if by_system else key] += 1
        cls_ends["human(_no_call)" if key == C.NO_CALL else "tool(all tool keys)"] += 1
        cls_ends["all(*)"] += 1
    n_all = len(cls_gaps["all(*)"])
    recs = []
    for name in sorted(cls_gaps):
        recs.append(dict(describe(name, cls_gaps[name], cls_ends.get(name, 0), cls_hist[name], P), kind="class"))
    for key, g in sorted(gaps.items(), key=lambda kv: -len(kv[1])):
        if len(g) >= MIN_KEY_N:
            recs.append(dict(describe(key, g, ends.get(key, 0), hist[key], P), kind="key"))
    for r in recs:
        r["share"] = r["n"] / n_all
    path = os.path.join(C.OUT, f"hint_shapes_{pop}.json")
    with open(path, "w") as f:
        json.dump({"pop": pop, "trees": len(trees), "gaps": n_all, "bins": BINS, "s_grid": S_GRID, "records": recs,
                   "small_keys": {"n_keys_total": len(gaps),
                                  "n_keys_lt_min": sum(1 for g in gaps.values() if len(g) < MIN_KEY_N),
                                  "gaps_in_small": sum(len(g) for g in gaps.values() if len(g) < MIN_KEY_N)},
                   "ends_total": sum(ends.values())}, f)
    # small keys' sharpness needs their own table: entries exist from 5 samples (gap_hints.MIN_POOLED)
    small = []
    for key, g in gaps.items():
        if C.MIN_POOLED <= len(g) < MIN_KEY_N:
            p10, p50, p90 = C.entry_quantiles(g)
            small.append({"key": key, "n": len(g), "p10": p10, "p50": p50, "p90": p90, "sharp": p90 / p10,
                          "n_end": ends.get(key, 0)})
    with open(os.path.join(C.OUT, f"hint_shapes_{pop}_small.json"), "w") as f:
        json.dump(small, f)
    print("wrote", path, "records", len(recs), "small keys", len(small))


if __name__ == "__main__":
    main()
