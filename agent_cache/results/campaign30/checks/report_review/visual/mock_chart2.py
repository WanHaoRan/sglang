"""Mock-up of chart 2 as small multiples at the doc's 760-unit width: one panel per baseline with ours repeated,
2 rows (CPU memory too / GPU memory only) x 4 columns, the 8th cell a key. Reads bins.json (bins.py).
Writes chart2_small_multiples.png here."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, INK, INK2, MUTED, GRID = "#2A78D6", "#1F1E1C", "#52514E", "#C3C2B7", "#E6E6E6"
PT = 0.72
bins = json.load(open(os.path.join(HERE, "bins.json")))
NAMES = {"vllm_lmcache": "vLLM + LMCache", "hicache": "SGLang + HiCache", "continuum_lmcache": "Continuum + LMCache",
         "tokencake": "TokenCake", "continuum": "Continuum", "sglang_vanilla": "SGLang (GPU only)",
         "vllm_vanilla": "vLLM (GPU only)"}
GRIDS = [["vllm_lmcache", "hicache", "continuum_lmcache", "tokencake"],
         ["continuum", "sglang_vanilla", "vllm_vanilla", None]]
# first bin from which only returning users are (mostly) counted: no or few first requests answered after it
CENSORED = {"continuum": 70, "continuum_lmcache": 95}
W, H = 760, 500
CAP = 150
left0, right0, colgap = 52, 744, 18
pw = (right0 - left0 - 3 * colgap) / 4
ph = 104
rows_y = [118, 292]  # plot tops

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
base = fig.add_axes([0, 0, 1, 1]); base.set_xlim(0, W); base.set_ylim(H, 0); base.axis("off")


def txt(x, y, s, size=12, color=INK, **kw):
    base.text(x, y, s, fontsize=size * PT, color=color, va="baseline", **kw)


txt(24, 30, "Each baseline falls behind ours between minute 10 and minute 40", 16, INK,
    fontweight="bold")
txt(24, 52, "Mean wait for the first token by when the request was sent, 5-minute bins; ours (blue) repeated in every panel",
    13, INK2)
txt(24, 80, "Keep history in CPU memory too", 11, INK2, fontweight="bold")
txt(24, 254, "Keep history in GPU memory only", 11, INK2, fontweight="bold")

ours = bins["ec_full"]
mid = lambda b: b["minute"] + 2.5
for r, row in enumerate(GRIDS):
    for c, arm in enumerate(row):
        x0, y0 = left0 + c * (pw + colgap), rows_y[r]
        if arm is None:
            # key cell
            ky = y0 + 4
            def key_line(y, color, lw, ls, label):
                base.plot([x0, x0 + 22], [y, y], color=color, lw=lw, ls=ls, solid_capstyle="butt")
                txt(x0 + 28, y + 3.5, label, 10, INK2)
            key_line(ky, BLUE, 2, "-", "Ours")
            key_line(ky + 16, INK2, 1.5, "-", "Baseline in the title")
            base.fill_between([x0, x0 + 22], [ky + 28] * 2, [ky + 38] * 2, color=MUTED, alpha=0.45, lw=0)
            txt(x0 + 28, ky + 37, "Baseline slower than ours", 10, INK2)
            base.scatter([x0 + 11], [ky + 50], s=22, facecolors="white", edgecolors=INK, linewidths=1.2, zorder=3)
            txt(x0 + 28, ky + 53.5, "First 5-min bin > 2 s above ours", 10, INK2)
            key_line(ky + 66, INK2, 1.5, (0, (1.5, 1.5)), "Returning users only †")
            continue
        ax = fig.add_axes([x0 / W, 1 - (y0 + ph) / H, pw / W, ph / H])
        ax.set_xlim(0, 110); ax.set_ylim(0, CAP)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#9A998F")
        ax.set_yticks([0, 50, 100, 150]); ax.set_xticks([0, 30, 60, 90])
        ax.tick_params(length=0, labelsize=10 * PT, colors=INK2, pad=2)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        if c == 0:
            ax.set_yticklabels(["0", "50", "100", "150 s"])
        else:
            ax.set_yticklabels([])
        b = bins[arm]
        xs = [mid(p) for p in b]
        yb = [p["mean"] for p in b]
        yo = [p["mean"] for p in ours]
        ax.fill_between(xs, yo, yb, where=[q > o for q, o in zip(yb, yo)], interpolate=True, color=MUTED,
                        alpha=0.45, lw=0)
        cut = CENSORED.get(arm)
        if cut is None:
            ax.plot(xs, yb, color=INK2, lw=1.5, solid_joinstyle="round")
        else:
            k = next(i for i, p in enumerate(b) if p["minute"] >= cut)
            ax.plot(xs[:k + 1], yb[:k + 1], color=INK2, lw=1.5)
            ax.plot(xs[k:], yb[k:], color=INK2, lw=1.5, ls=(0, (1.5, 1.5)))
            ax.text(109, 112 if arm == "continuum_lmcache" else 50, "returning\nusers only †", fontsize=9.5 * PT,
                    color=INK2, ha="right", va="bottom", linespacing=1.0)
        ax.plot(xs, yo, color=BLUE, lw=2)
        # off-scale peak
        top = max(b, key=lambda p: p["mean"])
        if top["mean"] > CAP:
            ax.text(mid(top) + 9, CAP - 4, f"peak {top['mean']:.0f} s", fontsize=9.5 * PT, color=INK2, va="top")
            ax.plot([mid(top)], [CAP + 4], marker="^", color=INK2, ms=4, clip_on=False)
        # onset marker
        on = next(i for i, (p, o) in enumerate(zip(b, ours)) if p["mean"] - o["mean"] > 2)
        ax.scatter([xs[on]], [yb[on]], s=22, facecolors="white", edgecolors=INK, linewidths=1.2, zorder=4, clip_on=False)
        lx = xs[on] + 4
        ly = max(yb[on] + 14, 18)
        if arm == "continuum":
            lx, ly = xs[on] + 22, 70
        ax.text(lx, ly, f"from min {b[on]['minute']}", fontsize=10 * PT, color=INK, ha="left", va="bottom",
                fontweight="bold")
        txt(x0, y0 - 6, NAMES[arm], 12, INK)

txt((left0 + right0) / 2, rows_y[1] + ph + 30, "Minutes into the run (when the request was sent)", 11, INK2, ha="center")
txt(24, rows_y[1] + ph + 52, "† Means count answered requests only. No new user's first request sent after minute 70 was "
    "answered by Continuum", 10, INK2)
txt(24, rows_y[1] + ph + 66, "before the run ended, and few after minute 95 by Continuum + LMCache; from there these "
    "lines cover returning users only.", 10, INK2)
out = os.path.join(HERE, "chart2_small_multiples.png")
fig.savefig(out, dpi=1344 / 7.6)
print(out)
