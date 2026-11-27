"""Per-run FA3 prefill cost, two ways.

(1) idle: requests admitted at once (queue wait < 50 ms) with no host or SSD part in their cached tokens and at
    least 1,000 new tokens; prefill time = client TTFT - server queue wait. OLS of that time on
    [1, new tokens, new x (prefix + new/2)] with at most R_MAX requests running at admission.
    Reported: per-token cost at 5K/10K/20K/30K new tokens on an empty prefix, and the cost of 20K tokens.
(2) loaded: intervals between consecutive Decode-batch log lines (40 decode steps) with >= 8 running at both
    ends; OLS of the interval on [1, running, K tokens in use, K prefill tokens, prefill batches,
    sum new x cached / 1e6] (the review's v_cost.py form). Reported: per-token cost at prefix 0 / 16K / 25K,
    ms per batch, and the cost of a 20K history computed again in 8,192-token chunks.

usage: prefill_fit.py camp:arm [...]; writes ../out/prefill_fit.json (merged)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import lib  # noqa: E402
import qstate  # noqa: E402

R_MAX = 12


def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    r2 = 1 - (res @ res) / (((y - y.mean()) ** 2).sum())
    return beta, r2, res


def idle_fit(st, t_hi=1e9):
    rows = []
    for r in st["rows"]:
        if r["arr"] is None or r["ttft"] is None or r["queue"] is None or r["arr"] > t_hi:
            continue
        cd = r["cd"] or {}
        if r["queue"] > 0.05 or (cd.get("host") or 0) > 0 or (cd.get("storage") or 0) > 0:
            continue
        new = r["in_len"] - r["cached_srv"]
        if new < 1000:
            continue
        run = st["running"].at(r["adm"])
        if run > R_MAX:
            continue
        rows.append((new, r["cached_srv"], r["ttft"] - r["queue"], run))
    if len(rows) < 20:
        return None
    X = [[1.0, n, n * (p + n / 2) / 1e6] for n, p, _, _ in rows]
    y = [t for _, _, t, _ in rows]
    beta, r2, res = ols(X, y)
    # robust pass: drop residuals beyond 3 MAD (requests that shared the GPU with another prefill)
    mad = np.median(np.abs(res - np.median(res))) or 1e-9
    keep = np.abs(res - np.median(res)) < 5 * 1.4826 * mad
    beta, r2, res = ols(np.asarray(X)[keep], np.asarray(y)[keep])
    a, b, c = beta
    out = {"n": int(keep.sum()), "n_all": len(rows), "a_s": float(a), "b_s_per_tok": float(b),
           "c_s_per_tok_Mtok": float(c), "r2": float(r2)}
    for L in (5000, 10000, 20000, 30000):
        out[f"ms_per_tok_at_{L // 1000}K"] = float((a + b * L + c * L * L / 2e6) / L * 1e3)
    out["t20K_s"] = float(a + b * 20000 + c * 20000 * 20000 / 2e6)
    # direct medians, no model: cold prompts (cached < 512)
    for lo, hi in ((4000, 12000), (12000, 20000), (20000, 30000), (30000, 70000)):
        xs = [t / n * 1e3 for n, p, t, _ in rows if p < 512 and lo <= n < hi]
        out[f"cold_ms_per_tok_{lo // 1000}-{hi // 1000}K"] = (float(np.median(xs)) if xs else None, len(xs))
    return out


def loaded_fit(st, lo=60.0, hi=6600.0):
    batch = st["srv"]["batch"]
    X, y = [], []
    prev = None
    new = nb = 0
    cx = 0.0
    for b in batch:
        if b[1] == "P":
            new += b[6]
            nb += 1
            cx += b[6] * b[7]
            continue
        if prev is not None and lo <= prev[0] and b[0] < hi:
            dt = b[0] - prev[0]
            if 0 < dt <= 30 and prev[2] >= 8 and b[2] >= 8:
                run = (prev[2] + b[2]) / 2
                tokk = (prev[6] + b[6]) / 2 / 1e3
                X.append([1.0, run, tokk, new / 1e3, nb, cx / 1e6])
                y.append(dt)
        prev, new, nb, cx = b, 0, 0, 0.0
    if len(y) < 200:
        return None
    beta, r2, _ = ols(X, y)
    c0, c_run, c_tok, c_pre, c_b, c_x = beta
    out = {"n": len(y), "r2": float(r2), "ms_per_tok_prefix0": float(c_pre), "ms_per_batch": float(c_b * 1e3),
           "ms_per_tok_per_Kprefix": float(c_x), "ms_per_tok_prefix16K": float(c_pre + c_x * 16),
           "ms_per_tok_prefix25K": float(c_pre + c_x * 25),
           "decode_step_ms_22run_635K": float((c0 + c_run * 22 + c_tok * 635) / 40 * 1e3)}
    for n in (20000, 30000):
        ms = 0.0
        for s0 in range(0, n, 8192):
            a = min(8192, n - s0)
            ms += c_pre * a + c_b * 1e3 + c_x * a * s0 / 1e3
        out[f"t{n // 1000}K_s"] = float(ms / 1e3)
        out[f"ms_per_tok_{n // 1000}K"] = float(ms / n)
    tot_new = sum(r[3] for r in X) * 1e3
    tot_nb = sum(r[4] for r in X)
    out["allin_ms_per_tok"] = float((c_pre * tot_new / 1e3 + c_b * tot_nb) / tot_new * 1e3) if tot_new else None
    return out


def main():
    path = f"{lib.OUT}/prefill_fit.json"
    os.makedirs(lib.OUT, exist_ok=True)
    res = json.load(open(path)) if os.path.exists(path) else {}
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        st = qstate.build(camp, arm)
        res[spec] = {"idle": idle_fit(st), "idle_first30min": idle_fit(st, 1800.0), "loaded": loaded_fit(st)}
        i, i30, l = res[spec]["idle"], res[spec]["idle_first30min"], res[spec]["loaded"]
        print(f"== {spec}")
        for name, f in (("idle (whole run)", i), ("idle (first 30 min)", i30)):
            if f:
                print(f"  {name}: n {f['n']}/{f['n_all']} R2 {f['r2']:.3f}; T = {f['a_s']*1e3:.1f} ms + {f['b_s_per_tok']*1e3:.5f} ms/tok x new "
                      f"+ {f['c_s_per_tok_Mtok']*1e3:.5f} ms/tok per K of (prefix + new/2); per token at 5/10/20/30K cold: "
                      f"{f['ms_per_tok_at_5K']:.4f} / {f['ms_per_tok_at_10K']:.4f} / {f['ms_per_tok_at_20K']:.4f} / "
                      f"{f['ms_per_tok_at_30K']:.4f} ms; 20K = {f['t20K_s']:.3f} s")
                print("     cold medians ms/token (n): " + "; ".join(
                    f"{k[16:]}: {v[0]:.4f} ({v[1]})" for k, v in f.items() if k.startswith("cold_ms") and v[0] is not None))
        if l:
            print(f"  loaded: n {l['n']} R2 {l['r2']:.3f}; per token {l['ms_per_tok_prefix0']:.5f} ms at prefix 0, "
                  f"{l['ms_per_tok_prefix16K']:.5f} at 16K, {l['ms_per_tok_prefix25K']:.5f} at 25K; {l['ms_per_batch']:.1f} ms a batch; "
                  f"20K again {l['t20K_s']:.3f} s ({l['ms_per_tok_20K']:.4f} ms/tok), 30K {l['t30K_s']:.3f} s "
                  f"({l['ms_per_tok_30K']:.4f}); all-in {l['allin_ms_per_tok']:.4f} ms/tok; decode step {l['decode_step_ms_22run_635K']:.1f} ms")
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
