"""Campaign 25's figures in timeline.py's standard set, beside each part's overview figure in results/compare_c25_<group>/
(results/campaign25/DECISIONS.md).

  python3 c25_figures.py [--group c128 a05 h]

  c128  the hint-free arm at 128 users with the arms of campaigns 19, 20 and 23, over the first 2 h (the hint-free arm's)
  a05   the four arms at ARRIVAL=0.5 users/s, 2 h each
  h     option H, read pins against paced retries at 44 users, over the first 105 min (the two new arms')

Each campaign-25 arm gets timeline_<arm>.png, events_<arm>.csv and turns_<arm>.csv over its whole run; the other arms'
are in their own compare dirs. The comparisons keep in every arm the turns sent, and the server events stamped, before
the group's cut, as arms_overview.png does: compare.png / .csv, compare_steady_1800s.png / .csv, ttft_over_time_60s.png
/ .csv (timeline.py --tier-rule private --steady-state 1800 --ttft-over-time 60) and paired_cdf_<arm>_vs_<ref>.png.
A cut figure says so in a line above it. Colors follow the overview figures.
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt

import timeline as tl

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DPI = 130   # timeline.py's savefig dpi
BASE, HINTFREE, ORACLE, P50, EXACT, JIT_P50 = "#2a78d6", "#796006", "#4a3aa7", "#eb6834", "#1baf7a", "#a0307a"
SKY = "#56b4e9"   # a paired figure's dots when the arm has its reference's color (the baseline's repeat)
# group: (cut in seconds since the client started or None, the line above a cut figure,
#         arms: (label, run dir, client dir, color, run in campaign 25), paired figures: (arm, reference))
GROUPS = {
    "c128": (7200.0, "First 2 h of every arm: the turns sent before 7,200 s and the server events stamped before it "
                     "(the hint-free arm ran 2 h, campaigns 19 and 20 ran 4.7-5.1 h)", [
        ("baseline", "20261018_103544_three_tier_wc_NAT160", "client_103852_cmp_three_tier_wc", BASE, False),
        ("hintfree_c128", "20261104_213242_three_tier_wc_dl_poll0_NAT160", "client_213731_cmp_three_tier_wc_dl_poll0",
         HINTFREE, True),
        ("oracle", "20261018_154254_three_tier_wc_oracle_v2_nu_NAT160", "client_154608_cmp_three_tier_wc_oracle_v2_nu",
         ORACLE, False),
        ("p50", "20261031_204526_three_tier_wc_oracle_v2_nu_NAT160", "client_204850_cmp_three_tier_wc_oracle_v2_nu", P50, False),
        ("jit_exact_c128", "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
         "client_002856_cmp_three_tier_wc_oracle_v2_nu_owned", EXACT, False),
        ("jit_p50_c128", "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160",
         "client_023345_cmp_three_tier_wc_oracle_v2_nu_owned", JIT_P50, False)],
        [("hintfree_c128", "oracle"), ("hintfree_c128", "baseline")]),
    "a05": (None, "", [
        ("base_a05", "20261104_234044_three_tier_wc_custom_L1nat_H70", "client_234447_cmp_three_tier_wc", BASE, True),
        ("hintfree_a05", "20261105_014608_three_tier_wc_dl_poll0_custom_L1nat_H70",
         "client_014738_cmp_three_tier_wc_dl_poll0", HINTFREE, True),
        ("jit_exact_a05", "20261105_034841_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
         "client_035011_cmp_three_tier_wc_oracle_v2_nu_owned", EXACT, True),
        ("jit_p50_a05", "20261105_055051_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
         "client_055219_cmp_three_tier_wc_oracle_v2_nu_owned", JIT_P50, True)],
        [("hintfree_a05", "base_a05"), ("jit_p50_a05", "base_a05"), ("jit_exact_a05", "base_a05"),
         ("hintfree_a05", "jit_p50_a05")]),
    "h": (6300.0, "First 105 min of every arm: the turns sent before 6,300 s and the server events stamped before it "
                  "(poll0_a01 and dl_a01 ran 105 min, the others 117-120 min)", [
        ("base_a01", "20261101_082229_three_tier_wc_custom_L1nat_H70", "client_082359_cmp_three_tier_wc", BASE, False),
        ("base_a01_rep", "20261105_075438_three_tier_wc_custom_L1nat_H70", "client_075607_cmp_three_tier_wc", BASE, True),
        ("poll0_a01", "20261105_095747_three_tier_wc_poll0_custom_L1nat_H70", "client_095915_cmp_three_tier_wc_poll0",
         P50, True),
        ("dl_a01", "20261105_114546_three_tier_wc_dl_custom_L1nat_H70", "client_114716_cmp_three_tier_wc_dl", ORACLE, True),
        ("hintfree_a01", "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70",
         "client_122833_cmp_three_tier_wc_dl_poll0", HINTFREE, False),
        ("jit_exact_a01", "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
         "client_202308_cmp_three_tier_wc_oracle_v2_nu_owned", EXACT, False)],
        [("base_a01_rep", "base_a01"), ("poll0_a01", "base_a01"), ("dl_a01", "base_a01"), ("dl_a01", "hintfree_a01")]),
}
DASHED = {"base_a01_rep": (0, (4, 3))}   # the box's overview figures draw the repeat dashed in the baseline's color


def cut_view(cl: dict, sv: dict, t_cut: float) -> tuple:
    """timeline.steady_view's other end: the client and server records cut to the turns sent, and the events stamped,
    before t_cut. The client sends nothing after its --timeout, so a cut at the shortest arm's timeout is a span every
    arm ran."""
    turns = [t for t in cl["turns"] if t["t0"] < t_cut]
    rids = {t["rid"] for t in turns}
    before = lambda xs, key: [x for x in xs if x[key] < t_cut]
    return dict(cl, turns=turns, caps=before(cl["caps"], "t")), dict(
        sv, batches=before(sv["batches"], "t"), d2h=before(sv["d2h"], "t0"), h2d=before(sv["h2d"], "t0"),
        h2s={op_id: op for op_id, op in sv["h2s"].items() if op.get("t_submit", t_cut) < t_cut},
        s2h={rid: op for rid, op in sv["s2h"].items() if rid in rids},
        evict_device=before(sv["evict_device"], "t"), evict_host=before(sv["evict_host"], "t"),
        load_back_init=before(sv["load_back_init"], "t"))


def add_note(png: str, note: str) -> None:
    """A line above a figure timeline.py already saved and closed."""
    img = plt.imread(png)
    h, w = img.shape[:2]
    band = 34   # pixels
    fig = plt.figure(figsize=(w / DPI, (h + band) / DPI), dpi=DPI)
    fig.patch.set_facecolor("#fcfcfb")
    fig.figimage(img, xo=0, yo=0, origin="upper")
    fig.text(14 / w, 1 - band / 2 / (h + band), note, ha="left", va="center", fontsize=10, color=tl.INK)
    fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def load_arms(name: str, arms: list, cut: float, out_dir: str) -> list:
    """timeline.py's result tuples (arm, client, server, metrics), cut; each campaign-25 arm's timeline drawn whole."""
    results = []
    for arm, run, client, _, new in arms:
        cl = tl.parse_client(os.path.join(RESULTS, run, client, "client.jsonl"), "private")
        sv = tl.parse_server(os.path.join(RESULTS, run, "server.log"), cl["t_start"], {t["rid"]: t["t0"] for t in cl["turns"]})
        if new:
            tl.draw_timeline(arm, cl, sv, os.path.join(out_dir, f"timeline_{arm}.png"))
            tl.write_csvs(arm, cl, sv, out_dir)
        if cut is not None:
            cl, sv = cut_view(cl, sv, cut)
        m = tl.arm_metrics(arm, cl, sv)
        results.append((arm, cl, sv, m))
        print(f"{name} {arm}: {m['turns']} turns / {m['conversations']} users; returning TTFT mean {m['ret_ttft_mean']} "
              f"p50 {m['ret_ttft_p50']} p90 {m['ret_ttft_p90']} p99 {m['ret_ttft_p99']} s; device/host/storage/recompute "
              f"{m['hits_device']}/{m['hits_host']}/{m['hits_storage']}/{m['cold_returns']}; uncached/turn {m['ret_uncached_mean']}",
              flush=True)
    return results


def draw_pairs(results: list, pairs: list, out_dir: str) -> list:
    by_arm = {r[0]: r for r in results}
    pngs = []
    for arm, ref in pairs:
        png = os.path.join(out_dir, f"paired_cdf_{arm}_vs_{ref}.png")
        own = tl.ARM_COLOR[arm]
        if own == tl.ARM_COLOR[ref]:
            tl.ARM_COLOR[arm] = SKY
        tl.draw_paired_cdf(by_arm[ref], by_arm[arm], png)
        tl.ARM_COLOR[arm] = own
        pngs.append(png)
    return pngs


def plot_group(name: str) -> None:
    cut, note, arms, pairs = GROUPS[name]
    out_dir = os.path.join(RESULTS, f"compare_c25_{name}")
    os.makedirs(out_dir, exist_ok=True)
    tl.ARM_COLOR.update({arm: color for arm, _, _, color, _ in arms})
    tl.ARM_DASH.update(DASHED)
    results = load_arms(name, arms, cut, out_dir)
    pngs = [os.path.join(out_dir, "compare.png")]
    tl.draw_compare(results, pngs[0])
    with open(os.path.join(out_dir, "compare.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0][3].keys()))
        w.writeheader(); w.writerows([r[3] for r in results])
    pngs += draw_pairs(results, pairs, out_dir)
    pngs.append(os.path.join(out_dir, "ttft_over_time_60s.png"))
    rows = tl.draw_ttft_over_time(results, pngs[-1], 60.0)
    with open(os.path.join(out_dir, "ttft_over_time_60s.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "t_from_s", "turns", "ttft_mean_s"])
        w.writeheader(); w.writerows(rows)
    tl.write_steady(results, out_dir, start=1800.0)
    pngs.append(os.path.join(out_dir, "compare_steady_1800s.png"))
    for png in pngs if cut is not None else ():
        add_note(png, note)
    print(f"wrote {out_dir}: " + ", ".join(sorted(x for x in os.listdir(out_dir) if x.endswith((".png", ".csv")))), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    for name in ap.parse_args().group:
        plot_group(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
