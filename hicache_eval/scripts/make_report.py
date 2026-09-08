"""Build the consolidated Exp 2/3/4 report for the nixl sweep.

Applies the caveats established during the run: Exp 2 rows are split into the
trustworthy subset (idle, plus steady-state reps) and the queue-ramp rows, and
any recompute control that reports a storage hit is flagged rather than averaged.
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
LABEL = {"qwen8b": "Qwen3-8B (bf16 KV)",
         "qwen32b": "Qwen3-32B-FP8 (fp8 KV)",
         "llama70b": "Llama-3.3-70B-AWQ (fp8 KV)"}
KV = {"qwen8b": 147456, "qwen32b": 131072, "llama70b": 163840}
# Idle recompute at L=16384 from the Exp 1 fits, for cross-validation only.
EXP1_RECOMPUTE = {"qwen8b": 0.9199, "qwen32b": 4.2098, "llama70b": 14.0575}


def read(pattern):
    out = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if not d.empty:
            out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def exp2_tables(d):
    """Per model/arm: idle row, steady-state reps, and the ramp rows, kept apart."""
    rows = []
    for m in MODELS:
        s = d[d.model == m]
        for arm in ["baseline", "timeout_policy", "C1_tmpfs", "C2_backup_skip",
                    "C2_populate"]:
            hit = s[s.control == arm]
            rec = s[s.control == arm + "_recompute"]
            if hit.empty:
                continue
            bad_ctrl = int(rec.cached_storage.notna().sum()) if not rec.empty else 0
            for rate in sorted(hit.requested_rps.unique()):
                a = hit[hit.requested_rps == rate]
                b = rec[rec.requested_rps == rate]
                # rep 2 sits in the queue ramp / drain transient (F16, F18)
                steady = a[a.rep < 2] if rate > 0 else a
                bs = b[b.rep < 2] if rate > 0 else b
                rows.append({
                    "model": m, "arm": arm, "R": rate,
                    "achieved": a.achieved_rps.median(),
                    "L3_steady": steady.ttft_s.median() if not steady.empty else None,
                    "rec_steady": bs.ttft_s.median() if not bs.empty else None,
                    "L3_all": a.ttft_s.median(),
                    "L3_ramp": a[a.rep == 2].ttft_s.median() if (a.rep == 2).any() else None,
                    "storage_hits": f"{int(a.cached_storage.notna().sum())}/{len(a)}",
                    "read_mb": a.probe_read_mb.median(),
                    "bad_control_rows": bad_ctrl,
                })
    t = pd.DataFrame(rows)
    if not t.empty:
        t["speedup_steady"] = (t.rec_steady / t.L3_steady).round(2)
    return t


def main():
    e2 = read("exp2_*/interference.csv")
    e3 = read("exp3_*/amplification.csv")
    t2 = exp2_tables(e2) if not e2.empty else pd.DataFrame()
    if not t2.empty:
        t2.to_csv(os.path.join(RES, "exp2_summary.csv"), index=False)
    if not e3.empty:
        e3.to_csv(os.path.join(RES, "exp34_all_models.csv"), index=False)

    print("=" * 78)
    print("EXP 2 - idle L3 advantage (the trustworthy subset)")
    print("=" * 78)
    idle = t2[(t2.R == 0) & t2.arm.isin(["baseline", "timeout_policy"])] if not t2.empty else pd.DataFrame()
    if not idle.empty:
        p = idle.pivot_table(index="model", columns="arm",
                             values=["L3_steady", "rec_steady", "speedup_steady"])
        print(p.round(3).to_string())
        print("\ncross-check vs the independent Exp 1 fit at L=16384:")
        for m in MODELS:
            r = idle[(idle.model == m) & (idle.arm == "baseline")]
            if r.empty:
                continue
            got, want = r.rec_steady.iloc[0], EXP1_RECOMPUTE[m]
            print(f"  {LABEL[m]:34s} measured {got:7.3f} s  vs Exp1 {want:7.3f} s"
                  f"  ({100*abs(got-want)/want:.1f}% apart)")

    print("\n" + "=" * 78)
    print("EXP 3/4 - L3 bytes written vs ever read back")
    print("=" * 78)
    if not e3.empty:
        for m in MODELS:
            s = e3[e3.model == m]
            if s.empty:
                continue
            print(f"\n--- {LABEL[m]} ---")
            for _, r in s.iterrows():
                w = r.written_L3_bytes / 2 ** 30
                rd = r.read_L3_tokens * KV[m] / 2 ** 30
                print(f"  {r.condition:10s} written {w:7.2f} GiB   read back {rd:5.2f} GiB"
                      f"   hit {r.cache_hit_rate:.4f}")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        for ax, m in zip(axes, MODELS):
            s = e3[e3.model == m]
            if s.empty:
                ax.set_title(f"{LABEL[m]}\n(no data)")
                ax.axis("off")
                continue
            x = range(len(s))
            ax.bar([i - .2 for i in x], s.written_L3_bytes / 2**30, .4, label="written")
            ax.bar([i + .2 for i in x], s.read_L3_tokens * KV[m] / 2**30, .4,
                   label="read back")
            ax.set_xticks(list(x))
            ax.set_xticklabels(s.condition, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("L3 GiB")
            ax.set_title(LABEL[m], fontsize=10)
            ax.grid(alpha=.3, axis="y")
            ax.legend(fontsize=8)
        fig.suptitle("Exp 3/4 - every byte written to L3 is dead (nixl backend)")
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "exp34_written_vs_read.png"), dpi=130)
        print("\nwrote exp34_written_vs_read.png")

    if not idle.empty:
        fig, ax = plt.subplots(figsize=(7, 4.4))
        w = 0.35
        xs = range(len(MODELS))
        for i, arm in enumerate(["baseline", "timeout_policy"]):
            v = [idle[(idle.model == m) & (idle.arm == arm)].speedup_steady.max()
                 for m in MODELS]
            ax.bar([x + (i - .5) * w for x in xs], v, w,
                   label="wait_complete" if arm == "baseline" else "timeout")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([LABEL[m].split(" (")[0] for m in MODELS], fontsize=9)
        ax.set_ylabel("idle L3 speed-up vs recompute (x)")
        ax.axhline(1, color="crimson", ls="--", lw=1)
        ax.set_title("Idle L3 advantage at L=16384, by model and prefetch policy")
        ax.grid(alpha=.3, axis="y")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "exp2_idle_speedup.png"), dpi=130)
        print("wrote exp2_idle_speedup.png")


if __name__ == "__main__":
    main()
