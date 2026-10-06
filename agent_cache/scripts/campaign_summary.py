#!/usr/bin/env python3
"""campaign_summary.py --manifest M [--iostat LOG] [--out DIR]

Per-arm result table of one campaign directory, in the format of the campaign READMEs (compare_20260922_023714 onward):
whole-run statistics of the returning turns plus the tier traffic that timeline.py writes to compare.csv. Run
timeline.py on the same manifest first: this script reads compare.csv from --out (default: next to the manifest).
Writes summary.csv (one row per arm) and paired.csv (mean TTFT difference over the same (conversation, turn) with a 95 %
bootstrap CI: three_tier_to - hbm_host, three_tier_wc - three_tier_to, hbm_lru - hbm_host) and prints both as Markdown.

Definitions (shared with vs_baseline.py and live_timeline.py where they overlap):
- returning turn: turn > 0 with a first token; TTFT as measured by the client.
- tier: recompute when more than half of the private prompt (prompt - 7,808 shared tokens) was uncached, otherwise
  storage / host / device by where the cached tokens came from. compare.csv's hits_* columns use timeline.py's older rule
  (cold when less than half of the whole prompt was cached), so they differ.
- worst 10-min window: the highest median TTFT over fixed 10-min bins of send time that hold >= 20 returning turns.
- SSD restore: a returning turn with storage-hit tokens that is not a recompute; TTFT p50 / p90, median storage tokens.
- server queue wait: ReqTimeStats queue_duration from server.log.
- SSD (iostat 10-s samples inside the arm's client window): bytes read and written, %util mean and p90; "while active" =
  samples reading more than 1 MiB/s: their mean and peak read rate and the share of them at >= 95 % util.
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vs_baseline import load_arm, load_iostat, paired, q, tier_of  # noqa: E402

WINDOW_S, MIN_TURNS = 600, 20      # worst-window bin width (s) and minimum returning turns per bin
ACTIVE_KBS = 1024.0                # iostat rkB/s above which the SSD counts as reading
COMPARE_COLS = ("t95_done_s", "ret_uncached_mean", "d2h_tokens", "h2s_tokens", "h2d_tokens", "s2h_tokens",
                "evict_device_tokens", "evict_host_tokens", "peak_queue")
PAIRS = (("three_tier_to", "hbm_host"), ("three_tier_wc", "three_tier_to"), ("hbm_lru", "hbm_host"))


def extra_stats(results, run, client, iostat):
    """Worst 10-min window, SSD restores and active-read iostat statistics for one arm."""
    rows = [json.loads(l) for l in open(os.path.join(results, run, client, "client.jsonl")) if l.strip()]
    run_rec = next(r for r in rows if r.get("kind") == "run")
    summ = next((r for r in rows if r.get("kind") == "summary"), {})
    ret = [r for r in rows if r.get("kind") == "turn" and "t_send" in r and r["turn"] > 0 and r.get("ttft")]
    bins = {}
    for r in ret:
        bins.setdefault(int(r["t_send"] // WINDOW_S), []).append(r["ttft"])
    worst, worst_bin = max(((st.median(v), k) for k, v in bins.items() if len(v) >= MIN_TURNS),
                           default=(float("nan"), None))
    rest = [r for r in ret if tier_of(r) == "storage"]
    stok = [(r.get("cached_details") or {}).get("storage") or 0 for r in rest]
    t0 = run_rec["t_start"]
    wall = summ.get("elapsed_s") or max(r["t_send"] + (r.get("latency") or 0) for r in ret)
    active = [x for x in iostat if t0 < x[0] <= t0 + wall + 10 and x[1] > ACTIVE_KBS] if iostat else []
    return dict(
        worst10_p50=round(worst, 2), worst10_start_min=worst_bin * WINDOW_S // 60 if worst_bin is not None else "",
        restores=len(rest),
        restore_ttft_p50=round(q(xs=[r["ttft"] for r in rest], f=.5), 2) if rest else "",
        restore_ttft_p90=round(q(xs=[r["ttft"] for r in rest], f=.9), 2) if rest else "",
        restore_tokens_p50=q(xs=stok, f=.5) if rest else "",
        ssd_active_samples=len(active),
        ssd_active_read_mibs=round(st.mean(x[1] for x in active) / 1024) if active else "",
        ssd_peak_read_mibs=round(max(x[1] for x in active) / 1024) if active else "",
        ssd_active_util95=round(sum(x[3] >= 95 for x in active) / len(active), 3) if active else "")


# ----------------------------------------------------------------------------- Markdown
def n(v):
    return f"{int(round(float(v))):,}"


def s(v):
    """Seconds: two decimals below 10 s, one above."""
    v = float(v)
    return f"{v:.2f}" if abs(v) < 10 else f"{v:,.1f}"


def mtok(v):
    v = float(v)
    return "–" if v == 0 else f"{v / 1e6:.2f}M" if v < 1e7 else f"{v / 1e6:.1f}M"


def markdown(rows):
    na = "–"
    def ssd(r, fmt):
        return fmt(r) if r.get("ssd_active_samples") else na
    lines = [
        ("turns / errors / conversations cut by the wall cap", lambda r: f"{n(r['turns'])} / {r['errors']} / {r['capped_convs']}"),
        ("client wall (s)", lambda r: n(r["wall_s"])),
        ("90 % of conversations ended (min)", lambda r: f"{float(r['conv90_min']):.1f}"),
        ("returning-turn TTFT mean / p50 / p90 / p99 (s)",
         lambda r: " / ".join(s(r[k]) for k in ("ttft_mean", "ttft_p50", "ttft_p90", "ttft_p99"))),
        ("worst 10-min window, median TTFT (s)", lambda r: f"{s(r['worst10_p50'])} (from min {r['worst10_start_min']})"),
        ("turns done at 95 % (s)", lambda r: n(r["t95_done_s"])),
        ("server queue wait, mean (s)", lambda r: s(r["queue_mean"]) if r["queue_mean"] != "" else na),
        ("returning turns: device / host / SSD / recompute",
         lambda r: " / ".join(n(r[f"n_{k}"]) for k in ("device", "host", "storage", "recompute"))),
        ("uncached tokens per returning turn (mean)", lambda r: n(r["ret_uncached_mean"])),
        ("SSD restore TTFT p50 / p90 (s), tokens per restore (median)",
         lambda r: f"{s(r['restore_ttft_p50'])} / {s(r['restore_ttft_p90'])}, {float(r['restore_tokens_p50']) / 1e3:.1f}K"
         if r["restores"] else na),
        ("tokens: d2h / h2s / h2d / s2h", lambda r: " / ".join(mtok(r[k]) for k in ("d2h_tokens", "h2s_tokens", "h2d_tokens", "s2h_tokens"))),
        ("tokens evicted from device / host", lambda r: f"{mtok(r['evict_device_tokens'])} / {mtok(r['evict_host_tokens'])}"),
        ("peak scheduler queue", lambda r: n(r["peak_queue"])),
        ("SSD read / written (TB); %util mean / p90",
         lambda r: ssd(r, lambda r: f"{float(r['ssd_read_tb']):.2f} / {float(r['ssd_written_tb']):.2f}; "
                                    f"{float(r['util_mean']):.0f} % / {float(r['util_p90']):.0f} %")),
        ("SSD reads while active, mean / peak (MiB/s); samples at ≥ 95 % util",
         lambda r: ssd(r, lambda r: f"{n(r['ssd_active_read_mibs'])} / {n(r['ssd_peak_read_mibs'])}; "
                                    f"{100 * float(r['ssd_active_util95']):.0f} %")),
    ]
    out = ["| | " + " | ".join(r["arm"] for r in rows) + " |", "|---" * (len(rows) + 1) + "|"]
    for label, f in lines:
        out.append(f"| {label} | " + " | ".join(f(r) for r in rows) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="lines '<arm> <run dir name> <client dir name>'")
    ap.add_argument("--iostat", default="", help="iostat -x log of the SSD covering the arms' client windows")
    ap.add_argument("--out", default="", help="directory with timeline.py's compare.csv (default: next to the manifest)")
    a = ap.parse_args()
    results = os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
    out = a.out or os.path.dirname(os.path.abspath(a.manifest))
    compare = {r["arm"]: r for r in csv.DictReader(open(os.path.join(out, "compare.csv")))}
    iostat = load_iostat(a.iostat) if a.iostat else []
    rows, ttft = [], {}
    for arm, run, client in (l.split()[:3] for l in open(a.manifest) if l.strip()):
        row, ttft[arm] = load_arm(results=results, arm=arm, run=run, client=client, iostat=iostat)
        row.update(extra_stats(results=results, run=run, client=client, iostat=iostat))
        row.update({k: compare[arm][k] for k in COMPARE_COLS})
        rows.append(row)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    prs = [p for p in (paired(label=f"{x} - {y}", a=x, b=y, ta=ttft[x], tb=ttft[y])
                       for x, y in PAIRS if x in ttft and y in ttft) if p]
    if prs:
        with open(os.path.join(out, "paired.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prs[0].keys()))
            w.writeheader()
            w.writerows(prs)
    print(markdown(rows))
    print()
    for p in prs:
        print(f"- `{p['a']}` − `{p['b']}`: {p['diff_mean']:+.2f} s [{p['ci_lo']:+.2f}, {p['ci_hi']:+.2f}], "
              f"p50 {p['diff_p50']:+.2f} s ({p['pairs']:,} pairs)")
    print(f"\nwrote {os.path.join(out, 'summary.csv')}" + (f" and {os.path.join(out, 'paired.csv')}" if prs else ""))


if __name__ == "__main__":
    main()
