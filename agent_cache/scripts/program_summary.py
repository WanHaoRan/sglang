"""Metrics of replay_programs.py runs in the units the literature reports. Stdlib only.

  python3 program_summary.py client.jsonl [more.jsonl ...] [--skip-first S] [--json out.json]

Per run, over the profile phase (warmup primers excluded; --skip-first drops requests sent in the phase's first S s):
  latency     TTFT mean, p50, p90, p99 (every paper); TPOT; TTFT by the kind of wait before the request (tool, user,
              spawn, join) and by its length (returning turns, our campaigns' view)
  throughput  output and input tokens per second and requests per minute over the window from the first send to the
              last completion (MORI's output throughput, ThunderAgent's steps per minute)
  cache       served share of prompt tokens by tier (device, host, storage) and recomputed share (MORI's served and
              recomputed fractions), next to the share an infinite cache would hold in the same run, and their ratio
              (FlatCake's delivered hit rate)
  sessions    plays drained, requests skipped at the end, errors
"""
from __future__ import annotations

import argparse
import json
import sys

GAP_BINS = ((0, 1), (1, 10), (10, 60), (60, 300), (300, 1800), (1800, float("inf")))


def _q(xs: list, p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return ys[min(len(ys) - 1, int(p * (len(ys) - 1) + 0.5))]


def _lat(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(sum(xs) / len(xs), 4), "p50": round(_q(xs, 0.5), 4),
            "p90": round(_q(xs, 0.9), 4), "p99": round(_q(xs, 0.99), 4)}


def load(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows: list, *, skip_first: float = 0.0) -> dict:
    run = next((r for r in rows if r["kind"] == "run"), {})
    phase = next((r["t"] for r in rows if r["kind"] == "phase" and r["phase"] == "profile"), 0.0)
    reqs = [r for r in rows if r["kind"] == "req" and r["phase"] == "profile" and r.get("t_send") is not None
            and r["t_send"] >= phase + skip_first]
    ok = [r for r in reqs if not r.get("error") and r.get("ttft") is not None]
    out = {"recipe": (run.get("args") or {}).get("recipe"), "hints": (run.get("args") or {}).get("hints"),
           "trace": (run.get("trace_header") or {}).get("source"), "requests": len(reqs),
           "errors": sum(1 for r in reqs if r.get("error")),
           "skipped_at_end": next((r.get("skipped") for r in rows if r["kind"] == "summary"), None),
           "plays_drained": sum(1 for r in rows if r["kind"] == "play")}
    out["ttft_s"] = _lat([r["ttft"] for r in ok])
    tpot = [(r["latency"] - r["ttft"]) / (r["completion_tokens"] - 1)
            for r in ok if (r.get("completion_tokens") or 0) > 1]
    out["tpot_s"] = _lat(tpot)
    out["ttft_by_gap_kind"] = {k: _lat([r["ttft"] for r in ok if (r.get("gap") or "none") == k])
                               for k in sorted({r.get("gap") or "none" for r in ok})}
    out["ttft_by_wait"] = {_bin_name(lo, hi): _lat([r["ttft"] for r in ok if lo <= r["wait_s"] < hi])
                           for lo, hi in GAP_BINS}
    out.update(_throughput(ok))
    out["cache"] = _cache(ok)
    return out


def _bin_name(lo: float, hi: float) -> str:
    return f"{lo:g}s+" if hi == float("inf") else f"{lo:g}-{hi:g}s"


def _throughput(ok: list) -> dict:
    if not ok:
        return {}
    t0 = min(r["t_send"] for r in ok)
    t1 = max(r["t_send"] + r["latency"] for r in ok)
    span = max(1e-9, t1 - t0)
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    return {"window_s": round(span, 1), "output_tok_per_s": round(out_tok / span, 1),
            "input_tok_per_s": round(sum(r["prompt_tokens"] or 0 for r in ok) / span, 1),
            "requests_per_min": round(60 * len(ok) / span, 2)}


def _cache(ok: list) -> dict:
    prompt = sum(r["prompt_tokens"] or 0 for r in ok)
    if not prompt:
        return {}
    tiers = {"device": 0, "host": 0, "storage": 0}
    for r in ok:
        d = r.get("cached_details") or {}
        if d:
            for k in tiers:
                tiers[k] += int(d.get(k) or 0)
        else:
            tiers["device"] += int(r.get("cached_tokens") or 0)
    cached = sum(tiers.values())
    ideal = sum(r.get("ideal_cached") or 0 for r in ok)
    return {"prompt_tokens": prompt, **{f"served_{k}": round(v / prompt, 4) for k, v in tiers.items()},
            "recomputed": round(1 - cached / prompt, 4), "ideal_hit": round(ideal / prompt, 4),
            "delivered_over_ideal": round(cached / ideal, 4) if ideal else None}


def table(summaries: list, names: list) -> str:
    cols = ("requests", "errors", "TTFT mean", "p50", "p90", "p99", "out tok/s", "req/min", "device", "host", "storage",
            "recomp", "ideal", "deliv/ideal")
    lines = ["run | " + " | ".join(cols)]
    for name, s in zip(names, summaries):
        t, c = s.get("ttft_s") or {}, s.get("cache") or {}
        vals = (s["requests"], s["errors"], t.get("mean"), t.get("p50"), t.get("p90"), t.get("p99"),
                s.get("output_tok_per_s"), s.get("requests_per_min"), c.get("served_device"), c.get("served_host"),
                c.get("served_storage"), c.get("recomputed"), c.get("ideal_hit"), c.get("delivered_over_ideal"))
        lines.append(name + " | " + " | ".join("-" if v is None else str(v) for v in vals))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="client JSONL files of replay_programs.py")
    ap.add_argument("--skip-first", type=float, default=0.0,
                    help="ignore requests sent in the profile phase's first S seconds")
    ap.add_argument("--json", default=None, help="write the summaries here")
    args = ap.parse_args()
    sums = [summarize(load(p), skip_first=args.skip_first) for p in args.runs]
    print(table(sums, args.runs))
    for p, s in zip(args.runs, sums):
        print(f"\n{p}: TTFT by wait " + json.dumps({k: v.get("mean") for k, v in s["ttft_by_wait"].items()}))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(zip(args.runs, sums)), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
