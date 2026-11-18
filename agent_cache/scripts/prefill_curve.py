"""Prefill time against context length on the H200, and the CPU replay of room_model.py re-priced with it.

  python3 prefill_curve.py fit      # results/probe_moe sweep: full-prefill time by length, per-token cost by position
  python3 prefill_curve.py replay   # room_model.py's read-start rules, recomputation priced by position, not 9K tokens/s

Attention makes a prefill's cost grow with the square of the context, so a token's cost depends on its position p.
With T(L) = a + bL + cL^2 for a full prefill of L tokens, recomputing a history's span [s, e) costs T(e) - T(s)
(the fixed part a is paid by every turn anyway). The replay's recomputed tokens are taken as the tail of each turn's
history, [e - U, e), where e = prompt - new tokens. Writes results/cpu_calib_c44/prefill_curve.csv.
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np

import room_model as rm

SWEEP = os.path.join(rm.RES, "probe_moe", "probe.log")   # Qwen3-30B-A3B-2507, bf16 KV, triton, 8K chunks, 2026-09-22
FIT_FROM = 4096    # tokens: below this the fixed per-request time dominates, and recomputed spans start past 7.8K
FLAT_RHO = rm.RHO  # the flat rate the replay used, tokens/s
SSD_S_PER_TOKEN = 98304 / 2.11e9   # bf16 KV bytes per token over the SSD's shared read rate
ROOMS = (90000, 70000, 110000)
WEIGHTS = (1, 3, 7)   # campaign 23: with the queue it causes, a recomputed token costs 6-8 times its own prefill


def sweep() -> tuple:
    rows = [json.loads(line) for line in open(SWEEP) if line.startswith('{"tier": "recompute"')]
    return np.array([r["L"] for r in rows], float), np.array([r["ttft_s"] for r in rows])


def fit_curve() -> tuple:
    """(a, b, c) of T(L) = a + bL + cL^2, least squares on relative error over lengths >= FIT_FROM."""
    L, T = sweep()
    sel = L >= FIT_FROM
    A = np.vstack([np.ones(sel.sum()), L[sel], L[sel] ** 2]).T
    w = 1.0 / T[sel]
    return tuple(np.linalg.lstsq(A * w[:, None], T[sel] * w, rcond=None)[0])


def fit() -> None:
    a, b, c = fit_curve()
    L, T = sweep()
    print(f"T(L) = {a:.4f} + {b:.3e} L + {c:.4e} L^2 s, fitted on L >= {FIT_FROM}")
    print(f"{'L':>7s} {'measured':>9s} {'fit':>7s} {'avg tok/s':>10s}")
    for x in np.unique(L):
        m = float(np.median(T[L == x]))
        print(f"{x:7.0f} {m:8.3f}s {a + b * x + c * x * x:6.3f}s {x / m:10.0f}")
    print(f"\nper-token cost at position p, b + 2cp, against {SSD_S_PER_TOKEN * 1e6:.1f} us to read it from SSD:")
    for p in (8000, 14000, 20000, 27000, 42000, 60000, 80000):
        s = b + 2 * c * p
        print(f"  p {p:6d}: {s * 1e6:6.1f} us = {1 / s:6.0f} tokens/s, {s / SSD_S_PER_TOKEN:4.1f} x a read")
    for k in (1, 2):
        print(f"recomputing costs {k} x a read at p = {(k * SSD_S_PER_TOKEN - b) / (2 * c):.0f}")


# ===== The replay, re-priced =====

_DATA, _END, _CURVE = None, None, None


def history_ends(data: dict) -> np.ndarray:
    turns = {(t["conv"], t["turn"]): t for t in rm.load_run(data["name"])["turns"]}
    return np.array([turns[(j["conv"], j["turn"])]["prompt"] - (turns[(j["conv"], j["turn"])]["delta"] or 0)
                     for j in data["jobs"]], float)


def curve_seconds(recomputed: list) -> float:
    """GPU seconds to recompute each turn's tokens as the tail of its history; tokens the replay counts past the
    history's length (a re-read lost again) are charged the history's mean cost per token."""
    _, b, c = _CURVE
    U = np.asarray(recomputed, float)
    inside = np.minimum(U, _END)
    start = _END - inside
    return float(np.sum(b * inside + c * (_END ** 2 - start ** 2) + (U - inside) * (b + c * _END)))


def _run(args) -> dict:
    room, p = args
    sim = rm.simulate(_DATA, dict(p), room=room, per_job=True)
    return dict(room=room, policy=rm.label(p), rule=p["rule"], soft=bool(p.get("soft")), q=p.get("q"),
                lam=p.get("lam"), form=p.get("form"), stall_s=sim["stall_s"], recompute_tokens=sim["recompute_tokens"],
                flat_recompute_s=round(sim["recompute_tokens"] / FLAT_RHO, 1),
                curve_recompute_s=round(curve_seconds(sim["recompute_by_job"]), 1),
                mean_pinned=sim["mean_pinned"], revoked=sim["staged_revoked"] + sim["arrival_revoked"])


FAMILIES = {
    "JIT p50, as run": lambda r: r["rule"] == "asrun" and not r["soft"],
    "Read at arrival": lambda r: r["rule"] == "arrival",
    "Best fixed quantile": lambda r: r["rule"] == "quantile" and not r["soft"],
    "Rule 1, p10/p50/p90": lambda r: r["rule"] == "cost" and r["form"] == "split" and not r["soft"],
    "Rule 1, any form": lambda r: r["rule"] == "cost" and not r["soft"],
    "Fixed quantile + soft pins": lambda r: r["rule"] == "quantile" and r["soft"],
    "Rule 1 + soft pins": lambda r: r["rule"] == "cost" and r["soft"],
    "Exact arrival times": lambda r: r["rule"] == "oracle",
}


def report(rows: list, room: int, price: str, w: int) -> None:
    """The cheapest policy of each family at this room and weight on recomputation."""
    print(f"\n== room {room // 1000}K, recomputation at {price} x{w}")
    here = [r for r in rows if r["room"] == room]
    cost = lambda r: r["stall_s"] + w * r[price]
    arrival = min(cost(r) for r in here if FAMILIES["Read at arrival"](r))
    for name, sel in FAMILIES.items():
        best = min((r for r in here if sel(r)), key=cost)
        at = f"lam {best['lam']:g} {best['form']}" if best["lam"] else (f"q {best['q']:g}" if best["q"] else "")
        print(f"  {name:28s} {cost(best):7.0f} s  {at:20s} {100 * (cost(best) / arrival - 1):+6.1f} % vs arrival")


def replay() -> None:
    global _DATA, _END, _CURVE
    _CURVE = fit_curve()
    _DATA = rm.build_jobs("p50")
    _END = history_ends(_DATA)
    print(f"{len(_DATA['jobs'])} reading turns; history ends p10/p50/p90 {np.percentile(_END, [10, 50, 90]).round()}")
    base = [dict(rule="asrun"), dict(rule="arrival"), dict(rule="oracle")]
    quant = [dict(rule="quantile", q=q, form="split") for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
    cost = [dict(rule="cost", lam=lam, form=f) for f in rm.FORMS for lam in rm.LAMS]
    plist = base + quant + cost + [dict(p, soft=True) for p in [dict(rule="asrun")] + quant + cost]
    with Pool(min(os.cpu_count() or 1, 24)) as pool:
        rows = pool.map(_run, [(room, p) for room in ROOMS for p in plist], chunksize=1)
    for room in ROOMS:
        report(rows, room, "flat_recompute_s", 1)
        for w in WEIGHTS:
            report(rows, room, "curve_recompute_s", w)
    rm.write_csv("prefill_curve.csv", rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("fit", "replay"))
    {"fit": fit, "replay": replay}[ap.parse_args().step]()


if __name__ == "__main__":
    main()
