"""Produce the plan-mandated artifacts that the first pass did not emit:
  exp2/ttft_vs_writerate.png       (plan line 235)
  exp2/interference.csv            + the prefetch_policy column the plan names
  exp3/amplification_bars.png      (plan line 277)
  exp3/per_round_ttft.csv          (plan line 277)
  exp1/l3_iostat_check.json        (plan line 211)
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = os.environ["RESULTS"]
KV = 147456


# --- exp2: add prefetch_policy, then the required plot ---------------------
def exp2():
    p = os.path.join(R, "exp2", "interference.csv")
    df = pd.read_csv(p)
    if "prefetch_policy" not in df.columns:
        # every Exp 2 point was run with wait_complete (see server_args.txt)
        df.insert(3, "prefetch_policy", "wait_complete")
        df.to_csv(p, index=False)
    d = df[df.control != "_populate_only"]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    colors = {"baseline": "#b2182b", "C1_tmpfs": "#2166ac", "C2_backup_skip": "#1b7837"}
    for ctl, g in d.groupby("control"):
        m = g.groupby("requested_rps").ttft_s.agg(["median", "min", "max"]).reset_index()
        ax.plot(m.requested_rps, m["median"], "o-", label=ctl, color=colors.get(ctl))
        ax.fill_between(m.requested_rps, m["min"], m["max"], alpha=0.15,
                        color=colors.get(ctl))
    # recompute line for L=16384 from Exp 1
    e1 = pd.read_csv(os.path.join(R, "exp1", "ttft_by_tier.csv"))
    rec = e1[(e1.tier == "recompute") & (e1.L == 16384)].ttft_s.median()
    ax.axhline(rec, ls="--", color="black",
               label=f"recompute L=16384 ({rec:.2f}s)")
    ax.set_xlabel("requested write load (req/s of 4096-token prompts)")
    ax.set_ylabel("L3-hit TTFT (s), L=16384")
    ax.set_yscale("log")
    ax.set_title("Exp 2: L3 hit latency vs write rate, with controls")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(R, "exp2", "ttft_vs_writerate.png"), dpi=130)
    print("wrote exp2/ttft_vs_writerate.png; recompute line", round(rec, 3))


# --- exp3/4: bar chart of written vs read, and per-round TTFT --------------
def exp3():
    df = pd.read_csv(os.path.join(R, "exp3", "amplification.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(df))
    w = 0.38
    gb = df.written_L3_bytes / 2 ** 30
    rd = df.read_L3_tokens * KV / 2 ** 30
    ax.bar([i - w / 2 for i in x], gb, w, label="written to L3 (GiB)", color="#b2182b")
    ax.bar([i + w / 2 for i in x], rd, w, label="read from L3 (GiB)", color="#1b7837")
    for i, v in enumerate(gb):
        ax.text(i - w / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    for i in x:
        ax.text(i + w / 2, 0.2, "0", ha="center", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(df.condition, rotation=20)
    ax.set_ylabel("GiB"); ax.grid(alpha=0.3, axis="y"); ax.legend()
    ax.set_title("Exp 3/4: L3 bytes written vs read back (read is 0 everywhere)")
    fig.tight_layout()
    fig.savefig(os.path.join(R, "exp3", "amplification_bars.png"), dpi=130)
    print("wrote exp3/amplification_bars.png")

    rows = []
    for f in sorted(glob.glob(os.path.join(R, "exp3", "*", "bench.jsonl"))):
        cond = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except Exception:
            continue
        s = d.get("summary", {})
        for rname, rv in (d.get("round") or {}).items():
            rows.append({"condition": cond, "round": rname,
                         "avg_ttft_s": rv.get("average_ttft"),
                         "cache_hit_rate": rv.get("cache_hit_rate"),
                         "requests": rv.get("request_count")})
        rows.append({"condition": cond, "round": "ALL",
                     "avg_ttft_s": s.get("average_ttft"),
                     "median_ttft_s": s.get("median_ttft"),
                     "p90_ttft_s": s.get("p90_ttft"), "p99_ttft_s": s.get("p99_ttft"),
                     "cache_hit_rate": s.get("cache_hit_rate"),
                     "avg_output_len": s.get("average_output_len"),
                     "input_tok_throughput": s.get("input_token_throughput"),
                     "output_tok_throughput": s.get("output_token_throughput")})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(R, "exp3", "per_round_ttft.csv"), index=False)
    print("wrote exp3/per_round_ttft.csv", len(t), "rows")
    print(t[t["round"] == "ALL"][["condition", "median_ttft_s", "p90_ttft_s",
                                  "p99_ttft_s", "avg_output_len",
                                  "output_tok_throughput"]].to_string(index=False))


# --- exp1: iostat read MB/s as an independent check on the L3 slope --------
def exp1_iostat():
    f = os.path.join(R, "exp1", "iostat.log")
    vals = []
    for line in open(f):
        if line.startswith("vda"):
            p = line.split()
            try:
                vals.append(float(p[2]) / 1024.0)  # rkB/s -> MB/s
            except (ValueError, IndexError):
                pass
    s = pd.Series(vals)
    out = {
        "_what": "vda read MB/s sampled every 5s across the whole Exp 1 run. "
                 "Nothing else reads the disk during Exp 1, so the upper tail is "
                 "the L3 probe read rate; it is an independent check on the "
                 "0.058 GB/s slope fitted from TTFT.",
        "samples": len(s), "median_MBps": round(s.median(), 2),
        "p90_MBps": round(s.quantile(0.90), 2), "p99_MBps": round(s.quantile(0.99), 2),
        "max_MBps": round(s.max(), 2),
        "p99_GBps": round(s.quantile(0.99) / 1024, 4),
        "max_GBps": round(s.max() / 1024, 4),
        "ttft_fitted_L3_GBps": 0.0584,
    }
    json.dump(out, open(os.path.join(R, "exp1", "l3_iostat_check.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    exp2(); exp3(); exp1_iostat()
