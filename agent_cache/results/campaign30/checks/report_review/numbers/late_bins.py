"""Late-run checks for the doc's chart 2: (1) per 5-minute bin, recorded first requests vs returning ones for the two
Continuum arms, and users admitted (client 'lanes' plays) without a recorded first request; (2) ours (ec_full) per bin:
recomputed share of prompt tokens, history delivered, share of returning requests reading the SSD. Read-only."""
import collections
import json
import os

RESULTS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
CUT = 6600.0
END = 7230.0
RUNS = {
    "ec_full": "20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/programs_093918_ec_full",
    "continuum": "20261124_203658_continuum_continuum/programs_203842_continuum",
    "continuum_lmcache": "20261124_224027_continuum_continuum_lmcache/programs_224335_continuum_lmcache",
}


def load(label):
    reqs, lanes = [], []
    with open(os.path.join(RESULTS, RUNS[label], "client.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "req" and r["phase"] == "profile":
                reqs.append(r)
            elif r["kind"] == "lanes":
                lanes.append(r)
    return reqs, lanes


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


for label in ("continuum", "continuum_lmcache"):
    reqs, lanes = load(label)
    ok = [r for r in reqs if r.get("ttft") is not None and not r.get("error")]
    # a user's first request: stream 0, k 0; original send time for a retried one = t_send - 3600
    first_any = {}
    for r in reqs:
        if r["stream"] == 0 and r["k"] == 0:
            t0 = r["t_send"] - 3600 if r.get("retried") else r["t_send"]
            first_any[r["play"]] = (t0, r)
    plays_at = sorted((l["t"], l["plays"]) for l in lanes)
    print(f"== {label}: per 5-minute bin by send time (before {CUT:.0f} s)")
    print("  bin | recorded: first n mean, returning n mean | users admitted in bin (lanes) | first requests without a"
          " recorded answer before the cut (lower bound 7,230 s - bin end)")
    missing_lb = []
    for b in range(22):
        lo, hi = b * 300, (b + 1) * 300
        rs = [r for r in ok if lo <= r["t_send"] < hi and r["t_send"] < CUT]
        fr = [r["ttft"] for r in rs if r["stream"] == 0 and r["k"] == 0]
        rt = [r["ttft"] for r in rs if not (r["stream"] == 0 and r["k"] == 0)]
        p_lo = max([p for t, p in plays_at if t <= lo + 0.5] or [0])
        p_hi = max([p for t, p in plays_at if t <= hi + 0.5] or [0])
        admitted = p_hi - p_lo
        answered_first = sum(1 for t0, r in first_any.values() if lo <= t0 < hi and r.get("ttft") is not None
                             and not r.get("retried") and r["t_send"] < CUT)
        unanswered = admitted - answered_first
        missing_lb += [END - hi] * max(0, unanswered)
        print(f"  {b * 5:3d} | first {len(fr):3d} {mean(fr):8.1f} | returning {len(rt):4d} {mean(rt):6.2f} | admitted {admitted:3d}"
              f" | without a recorded answer {unanswered:3d}")
    rec_first = [r["ttft"] for r in ok if r["stream"] == 0 and r["k"] == 0 and r["t_send"] < CUT]
    print(f"  recorded first requests {len(rec_first)} mean {mean(rec_first):.1f} s; unanswered by bin count {len(missing_lb)}"
          f" (lower bounds mean {mean(missing_lb):.1f} s); all first requests with lower bounds:"
          f" {mean(rec_first + missing_lb):.1f} s ({mean(rec_first + missing_lb) / 60:.1f} min)")

reqs, _ = load("ec_full")
ok = [r for r in reqs if r.get("ttft") is not None and r["t_send"] < CUT]
print("\n== ec_full per 5-minute bin: mean TTFT, recomputed share of prompt tokens, history delivered of ideal,"
      " returning requests reading the SSD")
for b in range(22):
    rs = [r for r in ok if b * 300 <= r["t_send"] < (b + 1) * 300]
    prompt = sum(r["prompt_tokens"] for r in rs)
    cached = sum(sum(int((r.get("cached_details") or {}).get(k) or 0) for k in ("device", "host", "storage"))
                 if r.get("cached_details") else int(r.get("cached_tokens") or 0) for r in rs)
    ideal = sum(r.get("ideal_cached") or 0 for r in rs)
    later = [r for r in rs if not (r["stream"] == 0 and r["k"] == 0)]
    ssd = sum(1 for r in later if int((r.get("cached_details") or {}).get("storage") or 0) > 0)
    print(f"  {b * 5:3d}: n {len(rs):4d} mean {mean([r['ttft'] for r in rs]):6.2f} | recomputed {100 * (1 - cached / prompt):5.1f} %"
          f" | delivered {cached / ideal:6.3f} | SSD reads {100 * ssd / max(1, len(later)):5.1f} %")
