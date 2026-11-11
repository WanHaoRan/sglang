"""Original SGLang HiCache against the just-in-time routine (exact and p50 hints), in two settings (gaps x10, 2 h each):

  python3 jit_vs_hicache.py [--setting c44 c128]

  c44   44 users, 70 GB of host memory (campaign 23): host memory is the scarce resource
  c128  128 users, 160 GB of host memory (HiCache from campaign 19, the just-in-time arms from campaign 23): the GPU is

Figures, beside each setting's overview figure in results/compare_c23_<setting>/:
  hicache_vs_jit.png / .pdf              panels over 5-minute bins by send time, chosen per setting
  hicache_vs_jit_ttft_kv.png / .pdf      the TTFT CDF of returning turns, and where their history's KV came from
  hicache_vs_jit_host_memory.png / .pdf  c44 only: how long SSD-loaded KV waits in host memory for its turn's admission,
                                         the host memory holding it would take over time, and the TTFT CDF
  hicache_vs_jit_gpu.png / .pdf          c128 only: the TTFT CDF, SSD bandwidth over time, and the share of the GPU's
                                         prefill spent recomputing history over time

Mean TTFT, turns per second, recomputed history per turn and the SSD read rate over time come from the setting's
arms_overview.csv. Per-turn numbers come from each arm's client.jsonl, as timeline.py reads it. SSD reads are the
server's s2h_io lines (by rid; logged when the data has landed in host memory; 64 tokens a page), SSD writes its h2s_io
lines, and admission is ReqTimeStats' entry_time + queue_duration. A turn's history is its prompt minus the new tokens it adds (delta_tokens);
recomputed history is the uncached part of the prompt minus those new tokens, and a turn's prefill is its uncached part.
CDFs, KV sources and footnotes are over the returning turns all three arms sent in their first 2 h.
"""
import argparse
import collections
import csv
import datetime as dt
import json
import os
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
BIN_S, END_S, PAGE = 300, 7200, 64
TREND_S, STEADY_S = 120, 2400   # the GPU figure's bins, and the minute-40 start of its steady-state means
BYTES_PER_TOKEN = 98304   # bf16 KV of Qwen3-30B-A3B, as the runs' h2d_done lines give it: 6 MiB a page
SSD_LIMIT_GB_S = 2.11   # measured on this volume (campaigns 14-17): reads and writes share it
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
READ = re.compile(r"^\[([\d\- :.]+)\] HICACHE_EVT s2h_io rid=\S+-c(\d+)-t(\d+) pages=(\d+)")
WRITE = re.compile(r"^\[([\d\- :.]+)\] HICACHE_EVT h2s_io .*?pages=(\d+)")
ADMIT = re.compile(r"rid=\S+-c(\d+)-t(\d+)[,)].*?queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")
FOOT_HINTS = ("Exact hints are the true next-request times; p50 hints are the median gap for the tool the reply calls, "
              "from a table the agent keeps online.")

Arm = collections.namedtuple("Arm", "label short color overview run client")
NAMES = (("Original SGLang HiCache", "HiCache", "#2a78d6"), ("Just-in-time, exact hints", "JIT, exact", "#1baf7a"),
         ("Just-in-time, p50 hints", "JIT, p50", "#a0307a"))   # colors as in the overview figures


def arms(*runs):
    return [Arm(*name, *run) for name, run in zip(NAMES, runs)]


PANELS = {   # key: (title, unit)
    "ttft": ("Mean time to first token (log scale)", "seconds"),
    "tps": ("Turns completed per second", "turns/s"),
    "recomputed": ("History recomputed per returning turn", "tokens"),
    "unused": ("History read from SSD but not used, per returning turn", "tokens"),
    "prefill_share": ("Share of the GPU's prefill spent recomputing history", "share of prefill tokens"),
    "ssd": ("SSD read rate", "GB/s"),
}
SETTINGS = {
    "c44": dict(
        name="44 users, 70 GB of host memory", out="compare_c23_c44", source="Campaign 23",
        arms=arms(("three_tier_wc", "20261101_082229_three_tier_wc_custom_L1nat_H70", "client_082359_cmp_three_tier_wc"),
                  ("jit_exact_c44", "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
                   "client_202308_cmp_three_tier_wc_oracle_v2_nu_owned"),
                  ("jit_p50_c44", "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
                   "client_222615_cmp_three_tier_wc_oracle_v2_nu_owned")),
        panels=("ttft", "tps", "recomputed", "unused"),
        timeline_title="44 users, 70 GB of host memory: original HiCache against just-in-time fetching",
        ttft_table=(0.47, 0.36), scarce="host", host_room=44_000),   # tokens free beyond the GPU's write-through copy
    "c128": dict(
        name="128 users, 160 GB of host memory", out="compare_c23_c128",
        source="HiCache from campaign 19, the JIT arms from campaign 23",
        arms=arms(("baseline", "20261018_103544_three_tier_wc_NAT160", "client_103852_cmp_three_tier_wc"),
                  ("jit_exact_c128", "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
                   "client_002856_cmp_three_tier_wc_oracle_v2_nu_owned"),
                  ("jit_p50_c128", "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160",
                   "client_023345_cmp_three_tier_wc_oracle_v2_nu_owned")),
        panels=("ttft", "tps", "prefill_share", "ssd"),
        timeline_title="128 users, 160 GB: the GPU is full in every arm; HiCache spends it recomputing",
        ttft_table=(0.03, 0.88), scarce="gpu"),
}


# ----------------------------------------------------------------------------- data
def stamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()   # UTC logs


def client_turns(arm):
    """Turns sent in the first 2 h, keyed (conv, turn), as timeline.parse_client reads them; and the client's start."""
    rows = [json.loads(l) for l in open(os.path.join(RES, arm.run, arm.client, "client.jsonl")) if l.strip()]
    t_start = next(r["t_start"] for r in rows if r.get("kind") == "run")
    turns = {}
    for r in rows:
        if r.get("kind") != "turn" or "t_send" not in r or r["t_send"] >= END_S:
            continue
        cd = r.get("cached_details") or {}
        prompt, cached = int(r.get("prompt_tokens") or 0), int(r.get("cached_tokens") or 0)
        turns[(r["conv"], r["turn"])] = dict(
            t0=r["t_send"], ttft=r.get("ttft") or 0.0, prompt=prompt, cached=cached, uncached=prompt - cached,
            host=int(cd.get("host") or 0), storage=int(cd.get("storage") or 0), device0=cd.get("device"),
            new=r.get("delta_tokens"), first=r["turn"] == 0 or bool(r.get("mid_start")))
    return t_start, turns


def server_events(arm):
    """SSD reads as (landing time, tokens) and the admission time, per (conv, turn), and SSD writes as (time, tokens),
    in epoch seconds."""
    reads, admitted, writes = collections.defaultdict(list), {}, []
    for line in open(os.path.join(RES, arm.run, "server.log"), errors="replace"):
        if "s2h_io" in line and (m := READ.match(line)):
            reads[(int(m[2]), int(m[3]))].append((stamp(m[1]), PAGE * int(m[4])))
        elif "h2s_io" in line and (m := WRITE.match(line)):
            writes.append((stamp(m[1]), PAGE * int(m[2])))
        elif "queue_duration=" in line and (m := ADMIT.search(line)):
            admitted[(int(m[1]), int(m[2]))] = float(m[5]) + float(m[3]) / (1000.0 if m[4] == "ms" else 1.0)
    return reads, admitted, writes


def load(setting):
    data = {}
    for arm in setting["arms"]:
        t_start, turns = client_turns(arm)
        reads, admitted, writes = server_events(arm)
        data[arm.overview] = dict(turns=turns, returning={k: t for k, t in turns.items() if not t["first"]},
                                  reads=reads, admitted=admitted, writes=writes, t_start=t_start,
                                  read_tok={k: sum(t for _, t in v) for k, v in reads.items()})
    shared = sorted(set.intersection(*(set(d["returning"]) for d in data.values())))
    new = {}   # the workload is the same in every arm, so a turn's new tokens can come from any arm that logged them
    for d in data.values():
        for k, t in d["returning"].items():
            if t["new"] is not None:
                new.setdefault(k, t["new"])
    overview = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(RES, setting["out"], "arms_overview.csv"))):
        overview[r["arm"]][int(r["minute"])] = r
    return data, shared, new, overview


def recomputed(t, new):
    return max(0, t["uncached"] - new)


def q(v, p):
    return sorted(v)[int(p * (len(v) - 1))]


# ----------------------------------------------------------------------------- styling
def style(ax, title, unit):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=6)
    ax.set_ylabel(unit)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(which="both", length=0)


def header(fig, ax, title, x):
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(x, 0.955), ncol=3, frameon=False,
               fontsize=9.5, handlelength=1.8, columnspacing=1.6, labelcolor=INK)
    fig.text(x, 0.985, title, fontsize=12, color=INK, va="top", weight="semibold")


def footnote(fig, x, text):
    fig.text(x, 0.012, text, fontsize=8, color=INK2, va="bottom", linespacing=1.45)


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(RES, out, f"{name}.{ext}"), dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", os.path.join(RES, out, f"{name}.png"), "and .pdf")


def kilo(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))


def percent(axis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f} %"))


# ----------------------------------------------------------------------------- figure 1: panels over time
def per_bin(d, new, minutes):
    """Per 5-minute bin by send time: SSD-read history not used per returning turn, and the recomputed share of prefill."""
    read, used, n, rec, prefill = (collections.Counter() for _ in range(5))
    for k, t in d["turns"].items():
        b = int(t["t0"] // BIN_S) * BIN_S // 60
        prefill[b] += t["uncached"]
        if not t["first"]:
            read[b] += d["read_tok"].get(k, 0); used[b] += t["storage"]; n[b] += 1
            rec[b] += recomputed(t, new[k]) if k in new else 0
    nan = float("nan")
    return dict(unused=[max(0, read[m] - used[m]) / n[m] if n[m] else nan for m in minutes],
                prefill_share=[rec[m] / prefill[m] if prefill[m] else nan for m in minutes])


def end_labels(ax, x, ends):
    """Direct labels at the line ends, in ink beside a dot of the series color; ends that nearly meet share one label."""
    groups = []
    for arm, y in sorted(ends, key=lambda e: e[1]):
        if groups and y / groups[-1][1] < 1.2:
            groups[-1][0].append(arm)
        else:
            groups.append(([arm], y))
    for members, y in groups:
        for arm in members:
            ax.plot([x], [dict(ends)[arm]], "o", ms=5, color=arm.color, mec=SURFACE, mew=1.2, clip_on=False, zorder=3)
        label = members[0].short if len(members) == 1 else "JIT, exact and p50"
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2,
                    annotation_clip=False)


def plot_timeline(setting, data, shared, new, overview):
    arms_ = setting["arms"]
    minutes = sorted(overview[arms_[0].overview])
    x = [m + BIN_S / 120 for m in minutes]   # bin centres in minutes
    fig, axes = plt.subplots(len(setting["panels"]), 1, figsize=(8.6, 9.6), sharex=True, facecolor=SURFACE)
    series = {}
    for a in arms_:
        ov = overview[a.overview]
        g = lambda col: [float(ov[m][col]) for m in minutes]
        series[a.overview] = dict(ttft=g("ttft_mean_s"), tps=g("turns_per_s"), recomputed=g("recomputed_per_turn"),
                                  ssd=g("ssd_read_gb_s"), **per_bin(data[a.overview], new, minutes))
    for ax, key in zip(axes, setting["panels"]):
        style(ax, *PANELS[key])
        for a in arms_:
            ax.plot(x, series[a.overview][key], color=a.color, lw=1.8, label=a.label, solid_capstyle="round")
        if key == "ttft":
            ax.set_yscale("log")
            ticks = [t for t in (0.3, 1, 3, 10, 30, 100) if t <= max(max(s["ttft"]) for s in series.values())]
            ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(bottom=0)
        if key in ("recomputed", "unused"):
            kilo(ax)
        if key == "prefill_share":
            ax.set_ylim(0, 1); percent(ax.yaxis)
    end_labels(axes[0], x[-1], [(a, series[a.overview]["ttft"][-1]) for a in arms_])
    axes[-1].set_xlabel("minutes since the client started (send time, 5-minute bins)")
    axes[-1].set_xlim(0, 120)
    header(fig, axes[0], setting["timeline_title"], x=0.085)
    lines = timeline_footnote(setting, data, shared, new, overview)
    footnote(fig, 0.085, "\n".join(lines))
    fig.subplots_adjust(top=0.895, bottom=0.075 + 0.015 * len(lines), left=0.085, right=0.86, hspace=0.42)
    save(fig, setting["out"], "hicache_vs_jit")


def timeline_footnote(setting, data, shared, new, overview):
    arms_ = setting["arms"]
    three = lambda v, f: f"{f(v[0])}, {f(v[1])} and {f(v[2])}"
    ttft = [st.mean(data[a.overview]["returning"][k]["ttft"] for k in shared) for a in arms_]
    served = three([len(data[a.overview]["turns"]) for a in arms_], lambda v: f"{v:,}")
    lines = [f"Over the {len(shared):,} returning turns all three arms sent (HiCache, JIT exact, JIT p50): mean TTFT "
             f"{three(ttft, lambda v: f'{v:.2f}')} s;"]
    if "unused" in setting["panels"]:
        used = [sum(data[a.overview]["returning"][k]["storage"] for k in shared)
                / sum(data[a.overview]["read_tok"].get(k, 0) for k in shared) for a in arms_]
        lines.append(f"share of SSD reads used {three(used, lambda v: f'{v * 100:.0f}')} %. Turns served in 2 h: "
                     f"{served}. {setting['source']}, x10 gaps, one run per arm.")
    else:
        rec = [st.mean(recomputed(data[a.overview]["returning"][k], new[k]) for k in shared) for a in arms_]
        pool = [st.mean(float(overview[a.overview][m]["gpu_pool_use"]) for m in range(30, 120, 5)) for a in arms_]
        lines += [f"recomputed history {three(rec, lambda v: f'{v:,.0f}')} tokens per turn. The GPU KV pool is "
                  f"{three(pool, lambda v: f'{v * 100:.0f}')} % full from minute 30.",
                  f"Turns served in 2 h: {served}. {setting['source']}; x10 gaps, one run per arm."]
    return lines + [FOOT_HINTS]


# ----------------------------------------------------------------------------- figure 2: TTFT CDF and KV sources
def cdf_panel(ax, arms_, values, title, xlabel, xlim, xticks, table_at, table_title, yticks=(0, 0.25, 0.5, 0.75, 0.9, 1)):
    """CDF per arm on a log time axis from yticks[0] up, with a p50/p90/p99 table at table_at (axes fraction: left,
    header row)."""
    style(ax, title, "share of turns")
    pct = {}
    for a in arms_:
        v = sorted(values[a.overview])
        ax.plot(v, [(i + 1) / len(v) for i in range(len(v))], color=a.color, lw=1.8, label=a.label)
        pct[a.short] = [q(v, p) for p in (0.5, 0.9, 0.99)]
    ax.set_xscale("log"); ax.set_xlim(*xlim); ax.set_ylim(yticks[0], 1.0)
    ax.set_xticks(xticks); ax.set_xticklabels([f"{t:g}" for t in xticks]); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(yticks); percent(ax.yaxis)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel(xlabel)
    (x0, top), dy = table_at, 0.075
    cols = (x0 + 0.28, x0 + 0.38, x0 + 0.48)
    ax.text(x0, top + dy, table_title, transform=ax.transAxes, fontsize=9, color=INK2)
    for x, h in zip(cols, ("p50", "p90", "p99")):
        ax.text(x, top, h, transform=ax.transAxes, ha="right", fontsize=9, color=INK2)
    for i, a in enumerate(arms_):   # a line key beside each row, text in ink
        y = top - (i + 1) * dy
        ax.plot([x0, x0 + 0.03], [y + 0.012] * 2, transform=ax.transAxes, color=a.color, lw=1.8)
        ax.text(x0 + 0.045, y, a.short, transform=ax.transAxes, fontsize=9, color=INK)
        for x, v in zip(cols, pct[a.short]):
            ax.text(x, y, f"{v:.2f}" if v < 10 else f"{v:.1f}" if v < 100 else f"{v:.0f}", transform=ax.transAxes,
                    ha="right", fontsize=9, color=INK)
    return pct


def ttft_cdf(ax, arms_, data, shared, table_at):
    values = {a.overview: [data[a.overview]["returning"][k]["ttft"] for k in shared] for a in arms_}
    return cdf_panel(ax, arms_, values, "Time to first token of returning turns (CDF)", "seconds (log scale)",
                     (0.08, 300), [0.1, 0.3, 1, 3, 10, 30, 100], table_at, "TTFT percentiles, seconds")


def kv_sources(d, shared, new):
    """Mean tokens per returning turn: history from GPU, host and SSD, recomputed history, and new tokens."""
    rows = [(d["returning"][k], new[k]) for k in shared]
    mean = lambda f: sum(f(t, n) for t, n in rows) / len(rows)
    return dict(gpu=mean(lambda t, n: t["cached"] - t["host"] - t["storage"]), host=mean(lambda t, n: t["host"]),
                ssd=mean(lambda t, n: t["storage"]), recomputed=mean(lambda t, n: recomputed(t, n)),
                new=mean(lambda t, n: n), prompt=mean(lambda t, n: t["prompt"]))


def source_bars(ax, arms_, src):
    cats = [("gpu", "GPU memory"), ("host", "Host memory"), ("ssd", "SSD"), ("recomputed", "Recomputed")]
    style(ax, "Where each returning turn's history KV came from", "")
    h = 0.24
    for j, a in enumerate(arms_):
        s = src[a.overview]; hist = s["prompt"] - s["new"]
        for i, (key, _) in enumerate(cats):
            v = 100 * s[key] / hist
            ax.barh(i + (j - 1) * (h + 0.04), v, height=h, color=a.color)
            ax.text(v + 1.2, i + (j - 1) * (h + 0.04), f"{v:.1f} %", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(cats))); ax.set_yticklabels([c[1] for c in cats], fontsize=9.5, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_xlabel("share of the turn's history (tokens)")


def plot_ttft_and_sources(setting, data, shared, new):
    arms_ = setting["arms"]
    fig, (cdf, bars) = plt.subplots(1, 2, figsize=(12.4, 5.0), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.1, 1]})
    pct = ttft_cdf(cdf, arms_, data, shared, setting["ttft_table"])
    src = {a.overview: kv_sources(data[a.overview], shared, new) for a in arms_}
    source_bars(bars, arms_, src)
    header(fig, cdf, f"{setting['name']}: TTFT and where each returning turn's KV came from", x=0.06)
    s0 = src[arms_[0].overview]
    shared_prompt = st.median(t["device0"] for t in data[arms_[0].overview]["turns"].values()
                              if t["first"] and t["device0"] is not None)
    footnote(fig, 0.06,
             f"The {len(shared):,} returning turns all three arms sent in their first 2 h. A returning turn's prompt "
             f"averages {s0['prompt'] / 1000:.1f}K tokens: {(s0['prompt'] - s0['new']) / 1000:.1f}K of history "
             f"(right), including a {shared_prompt / 1000:.1f}K-token system prompt every session shares,\nand "
             f"{s0['new']:.0f} new tokens that every arm computes. {setting['source']}, x10 gaps, one run per arm.\n"
             f"{FOOT_HINTS}")
    fig.subplots_adjust(top=0.84, bottom=0.23, left=0.06, right=0.98, wspace=0.28)
    save(fig, setting["out"], "hicache_vs_jit_ttft_kv")
    return pct, src


# ----------------------------------------------------------------------------- figure 3 (c44): host memory
def landing_waits(d, shared):
    """Per returning turn that read from SSD: seconds from its first read landing in host memory to its admission."""
    return {k: d["admitted"][k] - min(t for t, _ in d["reads"][k])
            for k in shared if k in d["reads"] and k in d["admitted"]}


def held_tokens(d):
    """Tokens of SSD-read data in host memory waiting for their turn's admission, per second since the client started."""
    diff = [0] * (END_S + 2)
    for k, reads in d["reads"].items():
        if k not in d["admitted"]:
            continue
        adm = d["admitted"][k] - d["t_start"]
        for land, tok in reads:
            lo, hi = int(max(0.0, land - d["t_start"])), int(min(END_S, adm))
            if lo < hi:
                diff[lo] += tok; diff[hi] -= tok
    held, run = [], 0
    for s in range(END_S):
        run += diff[s]; held.append(run)
    return held


def unused_share(d, waits, lo, hi):
    keys = [k for k, w in waits.items() if lo <= w < hi]
    return 1 - sum(d["returning"][k]["storage"] for k in keys) / sum(d["read_tok"][k] for k in keys), len(keys)


def plot_host_memory(setting, data, shared):
    arms_, room = setting["arms"], setting["host_room"]
    fig, (wait_ax, mem_ax, ttft_ax) = plt.subplots(1, 3, figsize=(17.2, 5.2), facecolor=SURFACE,
                                                   gridspec_kw={"width_ratios": [1, 1.05, 1]})
    waits = {a.overview: landing_waits(data[a.overview], shared) for a in arms_}
    # upper half only: below the median the arms' order says nothing about host memory
    wpct = cdf_panel(wait_ax, arms_, {k: list(v.values()) for k, v in waits.items()},
                     "SSD-loaded KV: landing in host memory to admission (CDF, 50-100 %)", "seconds (log scale)",
                     (0.5, 1000), [1, 3, 10, 30, 100, 300, 1000], (0.50, 0.45), "Wait percentiles, seconds",
                     yticks=(0.5, 0.6, 0.7, 0.8, 0.9, 1))
    wait_ax.set_ylabel("share of returning turns that read from SSD")
    style(mem_ax, "Host memory needed to hold SSD-loaded KV until admission", "tokens")
    held = {}
    for a in arms_:
        h = held[a.short] = held_tokens(data[a.overview])
        mem_ax.plot([m + 0.5 for m in range(END_S // 60)], [sum(h[m * 60:(m + 1) * 60]) / 60 for m in range(END_S // 60)],
                    color=a.color, lw=1.5, label=a.label)
    line = mem_ax.axhline(room, color=INK, lw=0.9)
    mem_ax.legend([line], [f"host memory free beyond the GPU's copy: {room / 1000:.0f}K tokens"], loc="upper left",
                  frameon=False, fontsize=8.5, handlelength=1.8, labelcolor=INK)
    mem_ax.set_xlim(0, 120); mem_ax.set_ylim(bottom=0); kilo(mem_ax)
    mem_ax.set_xlabel("minutes since the client started (1-minute means)")
    ttft_cdf(ttft_ax, arms_, data, shared, setting["ttft_table"])
    header(fig, wait_ax, f"{setting['name']}: just-in-time reads land when they are needed, so keeping them is cheap",
           x=0.05)
    base = data[arms_[0].overview]
    short_used, n_short = unused_share(base, waits[arms_[0].overview], 0, 0.5)
    long_used, n_long = unused_share(base, waits[arms_[0].overview], 5, 1e9)
    heavy = {s: sum(v[2700:]) / len(v[2700:]) for s, v in held.items()}
    over = {s: sum(x > room for x in v[2700:]) / len(v[2700:]) for s, v in held.items()}
    footnote(fig, 0.05,
             f"Left and right: the {len(shared):,} returning turns all three arms sent in their first 2 h (left: those that "
             f"read from SSD; a read lands when the I/O thread logs it). Middle: every SSD read in the first 2 h, counted "
             f"from landing to its turn's admission\nwhether or not the engine kept it. Minutes 45-120: mean "
             f"{heavy['HiCache'] / 1000:.0f}K, {heavy['JIT, exact'] / 1000:.0f}K and {heavy['JIT, p50'] / 1000:.0f}K "
             f"tokens (HiCache, JIT exact, JIT p50), above {room / 1000:.0f}K for {over['HiCache']:.0%}, "
             f"{over['JIT, exact']:.0%} and {over['JIT, p50']:.0%} of the seconds. HiCache does not keep landed reads: "
             f"{short_used:.0%} of the data of its reads admitted within 0.5 s\nwent unused ({n_short:,} turns), against "
             f"{long_used:.0%} of those that waited 5 s or more ({n_long:,} turns). {setting['source']}, x10 gaps, one "
             f"run per arm. {FOOT_HINTS}")
    fig.subplots_adjust(top=0.83, bottom=0.2, left=0.05, right=0.985, wspace=0.27)
    save(fig, setting["out"], "hicache_vs_jit_host_memory")
    return wpct, heavy, over


# ----------------------------------------------------------------------------- figure 4 (c128): GPU time
def binned(events):
    """Sums of (seconds since the client started, value) pairs per TREND_S bin over the first 2 h."""
    sums = collections.Counter()
    for t, v in events:
        if 0 <= t < END_S:
            sums[int(t // TREND_S)] += v
    return [sums[i] for i in range(END_S // TREND_S)]


def gpu_trends(d, new):
    """Per TREND_S bin: SSD reads (at landing) and writes in GB/s, and prefill and recomputed-history tokens by
    first-token time, when the turn's prefill finished."""
    gb_s = lambda tokens: [v * BYTES_PER_TOKEN / TREND_S / 1e9 for v in tokens]
    return dict(read=gb_s(binned((t - d["t_start"], tok) for v in d["reads"].values() for t, tok in v)),
                write=gb_s(binned((t - d["t_start"], tok) for t, tok in d["writes"])),
                prefill=binned((t["t0"] + t["ttft"], t["uncached"]) for t in d["turns"].values()),
                recomputed=binned((t["t0"] + t["ttft"], recomputed(t, new[k]))
                                  for k, t in d["returning"].items() if k in new))


def plot_gpu_time(setting, data, shared, new):
    arms_ = setting["arms"]
    fig, (ttft_ax, ssd_ax, rec_ax) = plt.subplots(1, 3, figsize=(17.2, 5.2), facecolor=SURFACE)
    ttft_cdf(ttft_ax, arms_, data, shared, setting["ttft_table"])
    x = [(i + 0.5) * TREND_S / 60 for i in range(END_S // TREND_S)]
    style(ssd_ax, "SSD bandwidth: reads, and reads plus writes", "GB/s")
    style(rec_ax, "Share of the GPU's prefill spent recomputing history", "share of prefill tokens")
    trends = {}
    for a in arms_:
        tr = trends[a.short] = gpu_trends(data[a.overview], new)
        ssd_ax.plot(x, tr["read"], color=a.color, lw=1.8)
        ssd_ax.plot(x, [r + w for r, w in zip(tr["read"], tr["write"])], color=a.color, lw=1.1, ls=(0, (3, 2)))
        rec_ax.plot(x, [r / p if p else float("nan") for r, p in zip(tr["recomputed"], tr["prefill"])],
                    color=a.color, lw=1.8)
    limit = ssd_ax.axhline(SSD_LIMIT_GB_S, color=INK, lw=0.9)
    ssd_ax.legend([Line2D([], [], color=INK2, lw=1.8), Line2D([], [], color=INK2, lw=1.1, ls=(0, (3, 2))), limit],
                  ["reads", "reads plus writes", f"the device's limit, reads and writes together: {SSD_LIMIT_GB_S} GB/s"],
                  loc="upper left", frameon=False, fontsize=8.5, handlelength=2.2, labelcolor=INK)
    ssd_ax.set_ylim(0, 3.0)
    rec_ax.set_ylim(0, 1); percent(rec_ax.yaxis)
    rec_ax.text(36, 0.95, "HiCache", ha="right", va="center", fontsize=9, color=INK2)
    rec_ax.text(80, 0.12, "JIT, exact and p50", ha="center", fontsize=9, color=INK2)
    for ax, how in ((ssd_ax, "reads by landing time"), (rec_ax, "by first-token time")):
        ax.set_xlim(0, 120)
        ax.set_xlabel(f"minutes since the client started (2-minute bins, {how})")
    header(fig, ttft_ax, f"{setting['name']}: just-in-time fetching spends SSD bandwidth to save the full GPU's time",
           x=0.05)
    steady, secs = slice(STEADY_S // TREND_S, None), END_S - STEADY_S
    means = {s: dict(read=st.mean(tr["read"][steady]), write=st.mean(tr["write"][steady]),
                     prefill=sum(tr["prefill"][steady]) / secs,
                     share=sum(tr["recomputed"][steady]) / sum(tr["prefill"][steady])) for s, tr in trends.items()}
    three = lambda key, f: f"{f(means['HiCache'][key])}, {f(means['JIT, exact'][key])} and {f(means['JIT, p50'][key])}"
    footnote(fig, 0.05,
             f"Left: the {len(shared):,} returning turns all three arms sent in their first 2 h. Middle: every SSD read "
             f"(s2h_io, logged at landing) and write (h2s_io) in the first 2 h, at 6 MiB a page; the limit was measured on "
             f"this volume. Right: recomputed history over all prefill\ntokens; the few percent every arm recomputes before "
             f"minute 38 is each turn's partial last page (under 64 tokens), which the cache does not hold. Minutes 40-120 "
             f"(HiCache, JIT exact, JIT p50): the GPU prefills {three('prefill', lambda v: f'{v / 1000:.1f}K')} tokens/s,\n"
             f"{three('share', lambda v: f'{v * 100:.0f}')} % of them recomputed history; the SSD reads "
             f"{three('read', lambda v: f'{v:.2f}')} GB/s and writes {three('write', lambda v: f'{v:.2f}')} GB/s. The "
             f"writes match the KV each arm computes, so HiCache writes the history it recomputes to SSD again.\n"
             f"{setting['source']}, x10 gaps, one run per arm. {FOOT_HINTS}")
    fig.subplots_adjust(top=0.83, bottom=0.2, left=0.05, right=0.985, wspace=0.27)
    save(fig, setting["out"], "hicache_vs_jit_gpu")
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": AXIS, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK2})
    for name in args.setting:
        setting = SETTINGS[name]
        data, shared, new, overview = load(setting)
        missing = sum(1 for k in shared if k not in new)
        print(f"{name}: {len(shared):,} shared returning turns, {missing} without new-token counts")
        plot_timeline(setting, data, shared, new, overview)
        pct, src = plot_ttft_and_sources(setting, data, shared, new)
        print(" TTFT p50/p90/p99", {k: [round(v, 2) for v in p] for k, p in pct.items()})
        print(" sources", {k: {c: round(v) for c, v in s.items()} for k, s in src.items()})
        if setting["scarce"] == "host":
            print(" host memory", plot_host_memory(setting, data, shared))
        else:
            print(" GPU time", plot_gpu_time(setting, data, shared, new))


if __name__ == "__main__":
    main()
