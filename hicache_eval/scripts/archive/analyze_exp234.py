"""Consolidate the nixl Exp 2/3/4 sweep across all three models.

Reads every exp2_<key>/interference.csv and exp3_<key>/amplification.csv under
the run directory and emits merged tables, the headline crossover numbers, and
the two plots the plan asks for.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RES = os.environ["RESULTS"]
MODELS = ["qwen8b", "qwen32b", "llama70b"]
LABEL = {"qwen8b": "Qwen3-8B bf16KV", "qwen32b": "Qwen3-32B-FP8 fp8KV",
         "llama70b": "Llama-3.3-70B-AWQ fp8KV"}
# Recompute TTFT at L=16384 from the Exp 1 runs; the Exp 2 headline is the write
# rate at which an L3 hit crosses this line.
RECOMPUTE_16384 = {"qwen8b": 0.9199, "qwen32b": 4.2098, "llama70b": 14.0575}
if os.environ.get("EXP1_RECOMPUTE_QWEN8B"):      # measured on the box under test (Exp 1 median at L=16384)
    RECOMPUTE_16384["qwen8b"] = float(os.environ["EXP1_RECOMPUTE_QWEN8B"])


def load(pattern, tag):
    frames = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"  skip {p}: {e}")
            continue
        if d.empty:
            continue
        d["_src"] = os.path.relpath(p, RES)
        frames.append(d)
    if not frames:
        print(f"no {tag} data found")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def exp2_report(d):
    if d.empty:
        return
    d = d[d.ttft_s.notna()].copy()
    print("\n" + "=" * 78)
    print("EXP 2 - write-through interference on L3 hit latency (nixl)")
    print("=" * 78)
    for m in MODELS:
        s = d[d.model == m]
        if s.empty:
            continue
        print(f"\n--- {LABEL[m]}   (recompute @16384 = {RECOMPUTE_16384[m]:.3f} s) ---")
        for arm in ["baseline", "timeout_policy", "C1_tmpfs", "C2_backup_skip",
                    "C2_populate"]:
            hit = s[s.control == arm]
            rec = s[s.control == arm + "_recompute"]
            if hit.empty:
                continue
            g = hit.groupby("requested_rps").agg(
                achieved=("achieved_rps", "median"), L3=("ttft_s", "median"),
                read_mb=("probe_read_mb", "median"),
                bg_read=("iostat_r_mbps", "median"),
                unful=("unfulfilled_tokens", "median"), util=("util_pct", "median"))
            if not rec.empty:
                g["recompute"] = rec.groupby("requested_rps").ttft_s.median()
                g["speedup"] = (g.recompute / g.L3).round(2)
                # A recompute control that reports a storage hit is not a
                # control; flag it rather than quietly averaging it in.
                bad = rec.cached_storage.notna().sum()
                if bad:
                    g.attrs["warn"] = f"{bad}/{len(rec)} recompute rows were L3 hits"
            print(f"\n  [{arm}]")
            print("  " + g.round(3).to_string().replace("\n", "\n  "))
            if g.attrs.get("warn"):
                print(f"  !! INVALID CONTROL: {g.attrs['warn']}")
            if "speedup" in g and (g.speedup < 1).any():
                lost = g.index[g.speedup < 1].min()
                print(f"  L3 falls behind its paired recompute at R = {lost}")
            elif "speedup" in g:
                print(f"  L3 never falls behind its paired recompute "
                      f"(min speed-up {g.speedup.min():.2f}x)")
    d.to_csv(os.path.join(RES, "exp2_all_models.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=False)
    for ax, m in zip(axes, MODELS):
        s = d[(d.model == m)]
        if s.empty:
            ax.set_title(f"{LABEL[m]} (no data)")
            continue
        for ctl, mark in [("baseline", "o-"), ("C1_tmpfs", "s--"),
                          ("C2_backup_skip", "^--"), ("timeout_policy", "d:")]:
            t = s[s.control == ctl]
            if t.empty:
                continue
            g = t.groupby("requested_rps").ttft_s.median()
            ax.plot(g.index, g.values, mark, label=ctl, markersize=6)
        ax.axhline(RECOMPUTE_16384[m], color="crimson", ls="-.", lw=1.4,
                   label=f"recompute {RECOMPUTE_16384[m]:.2f}s")
        ax.set_xlabel("requested write rate (req/s)")
        ax.set_ylabel("L3-hit TTFT (s), L=16384")
        ax.set_title(LABEL[m])
        ax.set_yscale("log")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7)
    fig.suptitle("Exp 2 - L3 hit latency vs concurrent write rate (nixl backend)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "exp2_ttft_vs_writerate.png"), dpi=130)
    print(f"\nwrote exp2_ttft_vs_writerate.png")


def exp34_report(d):
    if d.empty:
        return
    print("\n" + "=" * 78)
    print("EXP 3/4 - write amplification on a multi-turn workload (nixl)")
    print("=" * 78)
    cols = ["condition", "hicache_size", "write_policy", "duration_s",
            "written_L3_tokens", "written_L3_bytes", "l3_files", "read_L3_tokens",
            "generation_tokens", "prompt_tokens", "cache_hit_rate",
            "dead_fraction_L3", "drain_s"]
    for m in MODELS:
        s = d[d.model == m]
        if s.empty:
            continue
        print(f"\n--- {LABEL[m]} ---")
        print(s[[c for c in cols if c in s.columns]].to_string(index=False))
        wt = s[s.condition == "wt100"]
        wts = s[s.condition == "wts100"]
        orc = s[s.condition == "oracle100"]
        if not wt.empty and not wts.empty:
            a, b = wt.written_L3_bytes.iloc[0], wts.written_L3_bytes.iloc[0]
            print(f"  write_through_selective vs write_through: "
                  f"{100*(b-a)/a:+.1f}% L3 bytes")
        if not wt.empty and not orc.empty:
            a, b = wt.written_L3_bytes.iloc[0], orc.written_L3_bytes.iloc[0]
            print(f"  strip-thinking-cache oracle vs write_through: "
                  f"{100*(b-a)/a:+.1f}% L3 bytes")
    d.to_csv(os.path.join(RES, "exp34_all_models.csv"), index=False)

    present = [m for m in MODELS if not d[d.model == m].empty]
    if present:
        fig, axes = plt.subplots(1, len(present), figsize=(5.4 * len(present), 4.4))
        if len(present) == 1:
            axes = [axes]
        for ax, m in zip(axes, present):
            s = d[d.model == m]
            x = range(len(s))
            ax.bar([i - .2 for i in x], s.written_L3_bytes / 2**30, .4, label="written")
            ax.bar([i + .2 for i in x],
                   s.read_L3_tokens * 147456 / 2**30, .4, label="read back")
            ax.set_xticks(list(x))
            ax.set_xticklabels(s.condition, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("L3 GiB")
            ax.set_title(LABEL[m])
            ax.grid(alpha=.3, axis="y")
            ax.legend(fontsize=8)
        fig.suptitle("Exp 3/4 - L3 bytes written vs ever read back (nixl)")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "exp34_written_vs_read.png"), dpi=130)
        print("\nwrote exp34_written_vs_read.png")


if __name__ == "__main__":
    exp2_report(load("exp2_*/interference.csv", "exp2"))
    exp34_report(load("exp3_*/amplification.csv", "exp3/4"))
