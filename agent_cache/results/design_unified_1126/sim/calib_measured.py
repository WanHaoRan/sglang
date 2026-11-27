"""Measured anchors from the client records of campaigns 34 and 32 (read-only, streaming).

Prints, per arm, the statistics kvsim.summary() prints for a simulated run, computed the same way from
<run>/<programs>/client.jsonl: requests sent before the cut, TTFT mean / p50 / p90 / p99, by window of send time,
requests per minute, the shares of prompt tokens served from GPU / host / SSD, and the history computed again
(1 - cached / ideal_cached). Writes measured_anchors.json next to this file.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

RESULTS = "/home/wanhr/sglang/agent_cache/results"
HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = ((0, 1800), (1800, 3600), (3600, 5400), (5400, 6600))
CUT = 6600.0


def manifest(campaign: str) -> dict:
    out = {}
    with open(os.path.join(RESULTS, campaign, "manifest.txt")) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and not line.startswith("#"):
                out[parts[0]] = os.path.join(RESULTS, parts[1], parts[2], "client.jsonl")
    return out


def summarize(rows: np.ndarray, plays: list, *, cut: float = CUT) -> dict:
    """rows: columns t_send, ttft, prompt, cached, device, host, storage, ideal, first (0/1), out, latency."""
    r = rows[rows[:, 0] < cut]
    t, ttft = r[:, 0], r[:, 1]
    prompt = r[:, 2].sum()
    res = {
        "requests": int(len(r)),
        "req_per_min": len(r) / (cut / 60.0),
        "ttft_mean": float(ttft.mean()), "ttft_p50": float(np.percentile(ttft, 50)),
        "ttft_p90": float(np.percentile(ttft, 90)), "ttft_p99": float(np.percentile(ttft, 99)),
        "share_gpu": float(r[:, 4].sum() / prompt), "share_host": float(r[:, 5].sum() / prompt),
        "share_ssd": float(r[:, 6].sum() / prompt), "share_uncached": float(1.0 - r[:, 3].sum() / prompt),
        "history_recomputed": float(1.0 - r[:, 3].sum() / max(1.0, r[:, 7].sum())),
        "prompt_mean": float(r[:, 2].mean()), "out_mean": float(r[:, 9].mean()),
        "out_tok_per_s": float(r[:, 9].sum() / cut),
        "windows": [],
    }
    for lo, hi in WINDOWS:
        w = r[(t >= lo) & (t < hi)]
        ret = w[w[:, 8] == 0]
        res["windows"].append({
            "window": f"{lo // 60}-{hi // 60} min", "requests": int(len(w)),
            "req_per_min": len(w) / ((hi - lo) / 60.0),
            "ttft_mean": float(w[:, 1].mean()) if len(w) else None,
            "ttft_p50": float(np.percentile(w[:, 1], 50)) if len(w) else None,
            "ttft_p90": float(np.percentile(w[:, 1], 90)) if len(w) else None,
            "share_ssd": float(w[:, 6].sum() / max(1.0, w[:, 2].sum())),
            "share_gpu": float(w[:, 4].sum() / max(1.0, w[:, 2].sum())),
            "share_host": float(w[:, 5].sum() / max(1.0, w[:, 2].sum())),
            "history_recomputed": float(1.0 - w[:, 3].sum() / max(1.0, w[:, 7].sum())),
            "recomputed_tokens_M": float((w[:, 7] - w[:, 3]).clip(min=0).sum() / 1e6),
            "returning_reading": float((ret[:, 6] > 0).mean()) if len(ret) else None,
        })
    sat = r[(t >= 5400) & (t < 6540)]
    res["req_per_min_90_109"] = len(sat) / 19.0
    res["ttft_mean_90_109"] = float(sat[:, 1].mean()) if len(sat) else None
    if plays:
        waits = np.array([p["start"] - p["arrival_s"] for p in plays if p.get("start") is not None])
        res["users_started"] = int(len(waits))
        res["slot_wait_mean"] = float(waits.mean()) if len(waits) else None
    return res


def read_client(path: str):
    rows, plays = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('{"kind":"req"'):
                d = json.loads(line)
                if d.get("ttft") is None or d.get("error"):
                    continue
                det = d.get("cached_details") or {}
                rows.append((d["t_send"], d["ttft"], d.get("prompt_tokens") or d["prompt_sent"],
                             d.get("cached_tokens") or 0, det.get("device", 0), det.get("host", 0),
                             det.get("storage", 0), d.get("ideal_cached") or 0, 1 if d["k"] == 0 and d["stream"] == 0 else 0,
                             d.get("completion_tokens") or 0, d.get("latency") or 0.0))
            elif line.startswith('{"kind":"play"'):
                plays.append(json.loads(line))
    return np.array(rows, float), plays


def main() -> int:
    out = {}
    for campaign in ("campaign34", "campaign32"):
        for arm, path in manifest(campaign).items():
            if not os.path.exists(path):
                print(f"{campaign} {arm}: {path} missing", file=sys.stderr)
                continue
            rows, plays = read_client(path)
            s = summarize(rows, plays)
            out[f"{campaign}/{arm}"] = s
            print(f"{campaign} {arm:10s} n {s['requests']:6d} ({s['req_per_min']:.1f}/min; min 90-109 "
                  f"{s['req_per_min_90_109']:.1f}/min, TTFT {s['ttft_mean_90_109']:.2f}) TTFT mean {s['ttft_mean']:.2f} "
                  f"p50 {s['ttft_p50']:.2f} p90 {s['ttft_p90']:.2f} p99 {s['ttft_p99']:.2f} | gpu/host/ssd "
                  f"{s['share_gpu']:.3f}/{s['share_host']:.3f}/{s['share_ssd']:.3f} uncached {s['share_uncached']:.3f} "
                  f"hist recomputed {s['history_recomputed']:.4f} | users {s.get('users_started')} slot wait "
                  f"{s.get('slot_wait_mean') or 0:.1f}")
            for w in s["windows"]:
                print(f"      {w['window']:>12s}: n {w['requests']:5d} ({w['req_per_min']:.1f}/min) TTFT "
                      f"{w['ttft_mean']:.2f} p50 {w['ttft_p50']:.2f} p90 {w['ttft_p90']:.2f} | gpu/host/ssd "
                      f"{w['share_gpu']:.3f}/{w['share_host']:.3f}/{w['share_ssd']:.3f} hist recomputed "
                      f"{w['history_recomputed']:.4f} ({w['recomputed_tokens_M']:.2f} M) returning reading "
                      f"{w['returning_reading']:.3f}")
    with open(os.path.join(HERE, "measured_anchors.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
