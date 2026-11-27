"""Python port of chart1_widget.jsx and chart2_widget.jsx geometry, drawn with matplotlib in viewBox units (y down),
to preview the drafts without the doc. DejaVu Sans is wider than the doc's font, so text that fits here fits there.
Writes preview_chart1.png and preview_chart2.png (1344 px wide, like the doc screenshots)."""
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
C = {"text-primary": "#1F1E1C", "text-secondary": "#52514E", "chart-muted": "#C3C2B7", "chart-grid": "#E6E6E6",
     "chart-axis": "#9A998F", "cat1": "#2A78D6"}
PT = 0.72


class Canvas:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0); self.ax.axis("off")

    def text(self, x, y, s, size=12, color="text-primary", anchor="start", weight="normal"):
        ha = {"start": "left", "end": "right", "middle": "center"}[anchor]
        self.ax.text(x, y, s, fontsize=size * PT, color=C[color], ha=ha, va="baseline", fontweight=weight)

    def line(self, x1, x2, y1, y2, color, lw=1, dash=None):
        self.ax.plot([x1, x2], [y1, y2], color=C[color], lw=lw * PT * 1.39, ls=(0, dash) if dash else "-",
                     solid_capstyle="butt")

    def path(self, d, color, lw=1, fill=None, alpha=1, dash=None):
        segs, cur = [], []
        for cmd, xs, ys in re.findall(r"([MLZ])\s*(?:(-?[\d.]+),(-?[\d.]+))?", d):
            if cmd == "M":
                if cur:
                    segs.append(cur)
                cur = [(float(xs), float(ys))]
            elif cmd == "L":
                cur.append((float(xs), float(ys)))
            elif cmd == "Z":
                cur.append(cur[0])
        if cur:
            segs.append(cur)
        if fill:
            for s in segs:
                self.ax.add_patch(Polygon(s, closed=True, facecolor=C[fill], alpha=alpha, lw=0))
        else:
            for s in segs:
                self.ax.plot([p[0] for p in s], [p[1] for p in s], color=C[color], lw=lw * PT * 1.39,
                             ls=(0, dash) if dash else "-", solid_joinstyle="round")

    def circle(self, x, y, r, color):
        self.ax.add_patch(Circle((x, y), r, facecolor=C[color], lw=0, zorder=5))

    def rect(self, x, y, w, h, color, alpha):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=C[color], alpha=alpha, lw=0))

    def save(self, name):
        out = os.path.join(HERE, name)
        self.fig.savefig(out, dpi=1344 / 7.6)
        print(out)


def chart1():
    rows = [("vLLM + LMCache", "cpu", 25.24, 22.55, 27.99, 11103, 6.13, False),
            ("SGLang + HiCache", "cpu", 26.71, 23.52, 30.28, 9967, 5.17, False),
            ("Continuum + LMCache", "cpu", 32.95, 28.53, 37.98, 10081, 4.96, True),
            ("TokenCake", "cpu", 54.8, 49.18, 60.4, 7501, 3.14, False),
            ("Continuum", "gpu", 64.21, 53, 78.12, 5278, 1.63, True),
            ("SGLang (GPU only)", "gpu", 65.56, 60.3, 70.87, 6762, 3.04, False),
            ("vLLM (GPU only)", "gpu", 65.68, 60.49, 70.63, 6775, 3.05, False)]
    left, right, xmax, rowH, cpuHead = 250, 600, 80, 40, 86
    x = lambda v: left + v / xmax * (right - left)
    cpu = sorted([r for r in rows if r[1] == "cpu"], key=lambda r: r[2])
    gpu = sorted([r for r in rows if r[1] == "gpu"], key=lambda r: r[2])
    cpu0 = cpuHead + 12; cpu1 = cpu0 + len(cpu) * rowH; gpuHead = cpu1 + 16; gpu0 = gpuHead + 12
    yb = gpu0 + len(gpu) * rowH
    placed = [(r, cpu0 + i * rowH + rowH / 2) for i, r in enumerate(cpu)] + [(r, gpu0 + i * rowH + rowH / 2) for i, r in enumerate(gpu)]
    cv = Canvas(760, 500)
    cv.text(24, 30, "On the requests both answered, every baseline waits 25 to 66 s longer than ours", 16, weight="bold")
    cv.text(24, 52, "Mean extra wait for the first token, baseline minus ours; whiskers: 95 % interval from resampling users", 13, "text-secondary")
    cv.text(left - 14, cpuHead, "Keep history in CPU memory too", 11, "text-secondary", "end", "bold")
    cv.text(left - 14, gpuHead, "Keep history in GPU memory only", 11, "text-secondary", "end", "bold")
    for t in (0, 20, 40, 60, 80):
        if t:
            cv.line(x(t), x(t), cpu0, yb, "chart-grid")
        cv.text(x(t), yb + 18, f"{t}", 11, "text-secondary", "middle")
    cv.text((left + right) / 2, yb + 38, "Extra wait vs ours (s)", 11, "text-secondary", "middle")
    cv.line(x(0), x(0), cpu0 - 6, cpu1, "cat1", 2); cv.line(x(0), x(0), gpu0, yb, "cat1", 2)
    cv.text(x(0) + 6, cpu0 - 2, "Ours (0 s)", 12, "cat1", weight="bold")
    for (name, tier, gap, lo, hi, n, ours, lower), yc in placed:
        cv.text(left - 14, yc - 2, name, 13, "text-primary", "end")
        cv.text(left - 14, yc + 14, f"{n:,} shared requests · ours {ours:.1f} s", 11, "text-secondary", "end")
        cv.line(x(lo), x(hi), yc, yc, "text-secondary", 1.6)
        cv.line(x(lo), x(lo), yc - 4, yc + 4, "text-secondary", 1.6)
        if lower:
            cv.path(f"M{x(hi):.1f},{yc - 4.5}L{x(hi) + 7:.1f},{yc}L{x(hi):.1f},{yc + 4.5}Z", None, fill="text-secondary")
        else:
            cv.line(x(hi), x(hi), yc - 4, yc + 4, "text-secondary", 1.6)
        cv.circle(x(gap), yc, 5, "text-secondary")
        cv.text(x(hi) + (13 if lower else 9), yc + 4, ("≥ " if lower else "") + f"+{gap:.1f} s", 13)
    cv.text(24, yb + 62, "Each row pairs ours with one baseline on the requests both answered, so each row has its own set; ours = our mean on it.", 11, "text-secondary")
    cv.text(24, yb + 77, "≥: lower bound; requests still waiting when the run ended are not counted.", 11, "text-secondary")
    cv.save("preview_chart1.png")


def chart2():
    raw = open(os.path.join(HERE, "rows_chart2.txt")).read()
    rows = [dict(slug=s, minute=int(m), ttft=float(t), n=int(n)) for s, m, t, n in
            re.findall(r"slug:'([^']+)',minute:(\d+),ttft:([\d.]+),n:(\d+)", raw)]
    systems = [("ours", "Ours", "ours", None), ("vllm-lmcache", "vLLM + LMCache", "offload", None),
               ("hicache", "SGLang + HiCache", "offload", None), ("continuum-lmcache", "Continuum + LMCache", "offload", 95),
               ("tokencake", "TokenCake", "offload", None), ("continuum", "Continuum", "gpu", 70),
               ("sglang", "SGLang (GPU only)", "gpu", None), ("vllm", "vLLM (GPU only)", "gpu", None)]
    cap, behind, pw, ph, gx, x00, rowY = 150, 2, 154, 100, 20, 60, [128, 318]
    ours = [r for r in rows if r["slug"] == "ours"]
    oursAt = lambda m: next(o for o in ours if o["minute"] == m)
    base = [s for s in systems if s[2] != "ours"]
    panels = []
    for s in base:
        col = [t for t in base if t[2] == s[2]].index(s)
        panels.append(dict(s=s, col=col, x0=x00 + col * (pw + gx), y0=rowY[1 if s[2] == "gpu" else 0],
                           pts=[r for r in rows if r["slug"] == s[0]]))
    X = lambda p, m: p["x0"] + (m + 2.5) / 110 * pw
    Y = lambda p, v: p["y0"] + ph - min(v, cap) / cap * ph
    f = lambda v: f"{v:.1f}"

    def line(p, pts):
        acc = ""
        for i, q in enumerate(pts):
            a = pts[i - 1] if i else None
            qx, qy = X(p, q["minute"]), Y(p, q["ttft"])
            if a is None:
                acc = f"M{f(qx)},{f(qy)}" if q["ttft"] <= cap else ""
                continue
            ax_ = X(p, a["minute"])
            if a["ttft"] <= cap and q["ttft"] <= cap:
                acc += f"{'L' if acc else 'M'}{f(qx)},{f(qy)}"
            elif a["ttft"] <= cap:
                acc += f"L{f(ax_ + (cap - a['ttft']) / (q['ttft'] - a['ttft']) * (qx - ax_))},{f(p['y0'])}"
            elif q["ttft"] <= cap:
                acc += f"M{f(ax_ + (a['ttft'] - cap) / (a['ttft'] - q['ttft']) * (qx - ax_))},{f(p['y0'])}L{f(qx)},{f(qy)}"
        return acc

    def area(p):
        up = [f"{f(X(p, q['minute']))},{f(Y(p, max(q['ttft'], oursAt(q['minute'])['ttft'])))}" for q in p["pts"]]
        down = [f"{f(X(p, q['minute']))},{f(Y(p, oursAt(q['minute'])['ttft']))}" for q in p["pts"]][::-1]
        return f"M{'L'.join(up)}L{'L'.join(down)}Z"

    cv = Canvas(760, 500)
    cv.text(24, 30, "Each baseline falls behind ours between minute 10 and minute 40", 16, weight="bold")
    cv.text(24, 52, "Mean wait for the first token of each system's own requests by when they were sent, 5-minute bins", 13, "text-secondary")
    cv.text(24, rowY[0] - 44, "Keep history in CPU memory too", 11, "text-secondary", weight="bold")
    cv.text(24, rowY[1] - 44, "Keep history in GPU memory only", 11, "text-secondary", weight="bold")
    for p in panels:
        on = next((q for q in p["pts"] if q["ttft"] - oursAt(q["minute"])["ttft"] > behind), None)
        pk = max(p["pts"], key=lambda q: q["ttft"])
        cut = p["s"][3]
        early = p["pts"] if cut is None else [q for q in p["pts"] if q["minute"] <= cut]
        late = [] if cut is None else [q for q in p["pts"] if q["minute"] >= cut]
        last = p["pts"][-1]
        cv.text(p["x0"], p["y0"] - 24, p["s"][1], 12)
        if on:
            cv.text(p["x0"], p["y0"] - 10, f"behind from minute {on['minute']}", 10.5, "text-secondary")
        for t in (0, 50, 100, 150):
            cv.line(p["x0"], p["x0"] + pw, Y(p, t), Y(p, t), "chart-grid" if t else "chart-axis")
            if p["col"] == 0:
                cv.text(p["x0"] - 6, Y(p, t) + 4, f"{t} s" if t == cap else f"{t}", 10, "text-secondary", "end")
        for t in (0, 30, 60, 90):
            cv.text(p["x0"] + t / 110 * pw, p["y0"] + ph + 14, f"{t}", 10, "text-secondary", "middle")
        cv.path(area(p), None, fill="chart-muted", alpha=0.45)
        cv.path(line(p, early), "text-secondary", 1.6)
        if late:
            cv.path(line(p, late), "text-secondary", 1.6, dash=(1.5, 2.5))
            cv.text(X(p, last["minute"]) + 4, Y(p, last["ttft"]) + 4, "†", 11, "text-secondary")
        cv.path(line(p, ours), "cat1", 2)
        if pk["ttft"] > cap:
            cv.path(f"M{f(X(p, pk['minute']) - 4)},{f(p['y0'] + 7)}L{f(X(p, pk['minute']))},{f(p['y0'])}L{f(X(p, pk['minute']) + 4)},{f(p['y0'] + 7)}Z",
                    None, fill="text-secondary")
            cv.text(X(p, pk["minute"]) + 24, p["y0"] + 12, f"peak {round(pk['ttft'])} s", 10, "text-secondary")
        if on:
            cv.circle(X(p, on["minute"]), Y(p, on["ttft"]), 3.5, "text-primary")
    kx, ky = x00 + 3 * (pw + gx), rowY[1] + 6
    cv.line(kx, kx + 22, ky, ky, "cat1", 2); cv.text(kx + 28, ky + 4, "Ours", 10.5, "text-secondary")
    cv.line(kx, kx + 22, ky + 17, ky + 17, "text-secondary", 1.6); cv.text(kx + 28, ky + 21, "Baseline in the panel title", 10.5, "text-secondary")
    cv.rect(kx, ky + 29, 22, 10, "chart-muted", 0.45); cv.text(kx + 28, ky + 38, "Extra wait vs ours", 10.5, "text-secondary")
    cv.circle(kx + 11, ky + 51, 3.5, "text-primary"); cv.text(kx + 28, ky + 55, "Falls behind: first 5-min bin", 10.5, "text-secondary")
    cv.text(kx + 28, ky + 68, "more than 2 s above ours", 10.5, "text-secondary")
    cv.line(kx, kx + 22, ky + 81, ky + 81, "text-secondary", 1.6, dash=(1.5, 2.5)); cv.text(kx + 28, ky + 85, "Mostly returning users †", 10.5, "text-secondary")
    cv.text((x00 + kx + pw) / 2, rowY[1] + ph + 34, "Minutes into the run (when the request was sent)", 12, "text-secondary", "middle")
    cv.text(24, rowY[1] + ph + 58, "† Means count answered requests only. Continuum answered no new user's first request sent after minute 70", 11, "text-secondary")
    cv.text(24, rowY[1] + ph + 73, "before the run ended, Continuum + LMCache few after minute 95; from there these lines show mostly returning users.", 11, "text-secondary")
    cv.save("preview_chart2.png")


chart1()
chart2()
