"""Mock-ups of chart 1 at the doc's 760-unit width (1 unit = 1 px at dpi 100; saved at 1344 px like the screenshot):
A = the current gap plot fixed (axis, legible marks, caps, lower-bound arrows, tier groups, shared-set note);
B = the dumbbell alternative (ours' mean on each shared set -> the baseline's mean on it).
Numbers: campaign30/checks/vs_c34/compare_arms.out, the ec_full pairs. Writes chart1_A.png and chart1_B.png here."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, INK, INK2, MUTED, GRID = "#2A78D6", "#1F1E1C", "#52514E", "#C3C2B7", "#E6E6E6"
PT = 0.72  # font units (px at dpi 100) -> points

# system, tier, shared requests, ours mean A, baseline mean B, gap, lo, hi, lower bound
ROWS = [("vLLM + LMCache", "cpu", 11103, 6.13, 31.37, 25.24, 22.55, 27.99, False),
        ("SGLang + HiCache", "cpu", 9967, 5.17, 31.88, 26.71, 23.52, 30.28, False),
        ("Continuum + LMCache", "cpu", 10081, 4.96, 37.90, 32.95, 28.53, 37.98, True),
        ("TokenCake", "cpu", 7501, 3.14, 57.94, 54.80, 49.18, 60.40, False),
        ("Continuum", "gpu", 5278, 1.63, 65.84, 64.21, 53.00, 78.12, True),
        ("SGLang (GPU only)", "gpu", 6762, 3.04, 68.61, 65.56, 60.30, 70.87, False),
        ("vLLM (GPU only)", "gpu", 6775, 3.05, 68.73, 65.68, 60.49, 70.63, False)]


def canvas(h):
    fig = plt.figure(figsize=(7.6, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 760)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, size=13, color=INK, **kw):
    ax.text(x, y, s, fontsize=size * PT, color=color, va="baseline", **kw)


def draw(variant):
    H = 540
    fig, ax = canvas(H)
    left, right, xmax = 210, 600, 80
    X = lambda v: left + v / xmax * (right - left)
    title = "On the same requests, every baseline waits 25 to 66 s longer than ours"
    txt(ax, 24, 30, title, 16, INK, fontweight="bold")
    if variant == "A":
        sub = "Mean extra wait for the first token, baseline minus ours, on the requests both answered"
    else:
        sub = "Mean wait for the first token on the requests both answered: ours (blue) and the baseline (grey)"
    txt(ax, 24, 52, sub, 13, INK2)
    y0, rowH, gapH = 100, 40, 26
    ys, y = [], y0
    for i, r in enumerate(ROWS):
        if i == 0 or r[1] != ROWS[i - 1][1]:
            label = "Keep history in CPU memory too" if r[1] == "cpu" else "Keep history in GPU memory only"
            if i:
                y += 8
            txt(ax, 24, y + 4, label, 11, INK2, fontweight="bold")
            ax.plot([24, right + 140], [y + 10, y + 10], color=GRID, lw=1)
            y += gapH
        ys.append(y + rowH / 2)
        y += rowH
    yb = y + 6
    # axis and gridlines
    ticks = [0, 20, 40, 60, 80]
    for t in ticks:
        zero = BLUE if variant == "A" else "#9A998F"
        ax.plot([X(t), X(t)], [y0 + 14, yb], color=GRID if t else zero, lw=1 if t else 2, zorder=0)
        txt(ax, X(t), yb + 16, ("0 = ours" if (t == 0 and variant == "A") else f"{t}"), 11,
            BLUE if (t == 0 and variant == "A") else INK2, ha="center", fontweight="bold" if (t == 0 and variant == "A") else "normal")
    axis_title = "Extra wait vs ours (s)" if variant == "A" else "Mean wait for the first token (s)"
    txt(ax, (left + right) / 2, yb + 34, axis_title, 11, INK2, ha="center")
    for r, yc in zip(ROWS, ys):
        name, tier, n, a, b, gap, lo, hi, lb = r
        txt(ax, left - 14, yc - 2, name, 13, INK, ha="right")
        sub2 = f"{n:,} shared requests" + (f" · ours {a:.1f} s" if variant == "A" else "")
        txt(ax, left - 14, yc + 13, sub2, 10.5, INK2, ha="right")
        if variant == "A":
            c, L, Hh = gap, lo, hi
        else:
            c, L, Hh = b, a + lo, a + hi
            ax.plot([X(a), X(b)], [yc, yc], color=MUTED, lw=2.5, zorder=1, solid_capstyle="butt")
            ax.scatter([X(a)], [yc], s=36, color=BLUE, zorder=4, linewidths=0)
        ax.plot([X(L), X(Hh)], [yc, yc], color=INK2, lw=1.6, zorder=2)
        ax.plot([X(L), X(L)], [yc - 4, yc + 4], color=INK2, lw=1.6, zorder=2)
        if lb:
            ax.add_patch(FancyArrowPatch((X(Hh) - 1, yc), (X(Hh) + 9, yc), arrowstyle="-|>", mutation_scale=9,
                                         color=INK2, lw=1.6, zorder=2))
        else:
            ax.plot([X(Hh), X(Hh)], [yc - 4, yc + 4], color=INK2, lw=1.6, zorder=2)
        ax.scatter([X(c)], [yc], s=36, color=INK2, zorder=3, linewidths=0)
        lab = ("≥ " if lb else "") + f"+{gap:.1f} s"
        txt(ax, X(Hh) + (16 if lb else 8), yc + 4, lab, 12.5, INK)
    foot = ("Each row compares ours with one baseline on the requests both answered, so each row has its own set. "
            "Bars: 95 % interval,\nresampling users. ≥: lower bound, requests still waiting when the run ended "
            "are not counted.")
    txt(ax, 24, yb + 58, foot, 10.5, INK2)
    out = os.path.join(HERE, f"chart1_{variant}.png")
    fig.savefig(out, dpi=1344 / 7.6)
    print(out)


for v in ("A", "B"):
    draw(v)
