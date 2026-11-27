"""One figure across the four settings: mean TTFT of each arm that ran, with the paired difference to the current
design written on each bar (the campaign figure style: one legend above, a footnote with the numbers).

In the container, from agent_cache/scripts/ (no GPU is used):
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=. python3 -B ../results/compare_unified/summary.py
Reads results/compare_unified/<setting>/compare_arms.out (update.sh writes them) and writes summary.png/.pdf/.csv
into results/compare_unified/.
"""
import csv
import os
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import jit_vs_hicache as jv  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = (
    ("A_128u_160g", "A: 128 users,\n160 GB, 2 h"),
    ("B_cleaner50", "B: A, cleaner\nfrom minute 50"),
    ("C_112g", "C: 128 users,\n112 GB, 2 h"),
    ("D_64u_88g", "D: 64 users,\n88 GB, 1 h"),
)
ARMS = (
    ("ec_full", "Current design: the four rules", "#eb6834"),
    ("ec_full_load", "Current design + load hold", "#2a78d6"),
    ("unified", "Unified cache rules, no hold", "#1baf7a"),
    ("unified_load", "Unified design: cache rules + load hold", "#4a3aa7"),
    ("unified_load_po", "Unified design + admission pass-over", "#a0307a"),
    ("unified_load_fixedprice", "Unified design, fixed price of host room", "#8a8a80"),
)
PAIR = re.compile(r"^(\S+)\s+(\S+)\s+\|\s+(\d+)\s+\d+\s+\|\s+([\d.]+)\s+([\d.]+)\s+\|\s+([+-][\d.]+)\s+\[\s*([+-][\d.]+),\s*([+-][\d.]+)\]")


def pairs(setting):
    out = {}
    path = os.path.join(HERE, setting, "compare_arms.out")
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = PAIR.match(line)
        if m and m.group(1) == "ec_full":
            out[m.group(2)] = dict(
                n=int(m.group(3)), base=float(m.group(4)), mean=float(m.group(5)),
                diff=float(m.group(6)), lo=float(m.group(7)), hi=float(m.group(8)),
            )
    return out


def main():
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": jv.AXIS, "xtick.color": jv.INK2, "ytick.color": jv.INK2,
                         "axes.labelcolor": jv.INK2})
    data = {s: pairs(s) for s, _ in SETTINGS}
    present = [a for a in ARMS if a[0] == "ec_full" or any(a[0] in data[s] for s, _ in SETTINGS)]
    fig, ax = plt.subplots(1, 1, figsize=(10.4, 5.6), facecolor=jv.SURFACE)
    jv.style(ax, "Mean time to first token, same users and requests per setting", "seconds")
    width = 0.8 / len(present)
    rows = []
    for k, (key, label, color) in enumerate(present):
        xs, ys = [], []
        for i, (setting, _) in enumerate(SETTINGS):
            d = data[setting]
            if not d:
                continue
            if key == "ec_full":
                y = next(iter(d.values()))["base"]
                text = f"{y:.2f}"
            elif key in d:
                y = d[key]["mean"]
                text = f"{d[key]['diff']:+.2f}"
                rows.append((setting, key, d[key]["base"], y, d[key]["diff"], d[key]["lo"], d[key]["hi"], d[key]["n"]))
            else:
                continue
            x = i - 0.4 + width * (k + 0.5)
            xs.append(x)
            ys.append(y)
            ax.text(x, y + 0.15, text, ha="center", va="bottom", fontsize=7.5, color=jv.INK2, rotation=90)
        ax.bar(xs, ys, width=width * 0.92, color=color, label=label)
    ax.set_xticks(range(len(SETTINGS)))
    ax.set_xticklabels([t for _, t in SETTINGS])
    ax.set_ylim(0, max(r[2] for r in rows) * 1.25 if rows else 1)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.065, 0.93), ncol=3, frameon=False,
               fontsize=9, handlelength=1.6, columnspacing=1.4, labelcolor=jv.INK)
    fig.text(0.07, 0.975, "The unified design against the current one, four settings", fontsize=12, color=jv.INK,
             va="top", weight="semibold")
    jv.footnote(fig, 0.07, textwrap.fill("Numbers on the bars: the current design's mean, and each other arm's paired "
                "difference with it (same user, stream and request; requests both sent before the last ten minutes). "
                "FreeInference sessions replayed as users at 0.1 users/s with recorded waits, Qwen3-30B-A3B on one H200, "
                "FA3 attention.", 150))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, f"summary.{ext}"), dpi=180, facecolor=jv.SURFACE)
    plt.close(fig)
    with open(os.path.join(HERE, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["setting", "arm", "current_mean_s", "arm_mean_s", "paired_diff_s", "ci_lo", "ci_hi", "pairs"])
        w.writerows(rows)
    for r in rows:
        print(" ".join(str(v) for v in r))


if __name__ == "__main__":
    main()
