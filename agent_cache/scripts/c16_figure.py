"""One figure for ORACLE_FINDINGS.md: campaign 14's three_tier_wc, campaign 15's three_tier_wc_oracle and the three
campaign 16 arms over client time, in 5-min bins of the turn's send time: mean TTFT, turns completed per second, history
tokens recomputed per returning turn, and users waiting for admission (the client's cap records).

  python3 scripts/c16_figure.py [--results agent_cache/results] --out results/campaign16/all/c16_overview.png
  (matplotlib: run it in the container)
"""
import argparse, collections, json, os, statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from c16_summary import ARMS, runs  # noqa: E402

COLORS = {"c14 wc": "#1f77b4", "c15 wc_oracle": "#e377c2", "c16 _dl": "#ff7f0e", "c16 _oracle_v2": "#2ca02c",
          "c16 _oracle_v2_nu": "#9467bd"}
BIN = 300


def series(run, client):
    ttft, done, recomp, waiting = (collections.defaultdict(list) for _ in range(4))
    with open(os.path.join(run, client, "client.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            k = d.get("kind")
            if k == "turn" and d.get("ttft") is not None:
                ttft[int(d["t_send"] // BIN)].append(d["ttft"])
                done[int((d["t_send"] + d["latency"]) // BIN)].append(1)
                if "reply_reprefilled" in d:
                    recomp[int(d["t_send"] // BIN)].append(max(d["reply_reprefilled"], 0))
            elif k in ("cap", "users"):   # the backlog cap's updates, or the per-minute population record
                waiting[int(d["t"] // BIN)].append(d["waiting"])
    xs = range(18000 // BIN)
    mid = [(i + 0.5) * BIN / 60 for i in xs]
    return (mid, [st.mean(ttft[i]) if ttft[i] else None for i in xs], [len(done[i]) / BIN for i in xs],
            [st.mean(recomp[i]) if recomp[i] else None for i in xs], [st.mean(waiting[i]) if waiting[i] else None for i in xs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", nargs=3, action="append", metavar=("LABEL", "CAMPAIGN", "ARM"),
                    help="a run to draw (repeatable; default: campaigns 14-16)")
    ap.add_argument("--title", default="Campaign 14/15 baselines vs campaign 16 arms")
    ap.add_argument("--window", nargs=2, type=float, default=(2700, 16800), metavar=("START", "END"),
                    help="client seconds to shade (0 0: none)")
    a = ap.parse_args()
    fig, ax = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    arms = [tuple(x) for x in a.arm] if a.arm else ARMS
    spare = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#e377c2", "#8c564b", "#17becf"]
    for i, (name, run, client) in enumerate(runs(a.results, arms)):
        mid, ttft, tps, rec, wait = series(run, client)
        kw = dict(color=COLORS.get(name, spare[i % len(spare)]), label=name, lw=1.6)
        ax[0].plot(mid, ttft, **kw)
        ax[1].plot(mid, tps, **kw)
        ax[2].plot(mid, [max(v, 1) if v is not None else None for v in rec], **kw)
        ax[3].plot(mid, wait, **kw)
    ax[0].set_yscale("log"); ax[0].set_ylabel("mean TTFT (s)")
    ax[1].set_ylabel("turns completed / s")
    ax[2].set_yscale("log"); ax[2].set_ylabel("history tokens recomputed\nper returning turn")
    ax[3].set_ylabel("users waiting\nfor admission")
    ax[3].set_xlabel("minutes since the client started (5-min bins)")
    for x in ax:
        x.grid(alpha=0.3)
        if a.window[1] > a.window[0]:
            x.axvspan(a.window[0] / 60, a.window[1] / 60, color="0.92", zorder=0)
    shade = f" (shaded: the {a.window[0]:,.0f}-{a.window[1]:,.0f} s window)" if a.window[1] > a.window[0] else ""
    fig.suptitle(a.title + shade)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
