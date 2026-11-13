"""More figures recomputed from the cache: the idle room under several definitions of 'in the two load episodes',
whether the arrival-only arms ever start a read before the turn arrives, checks/predwait.out, mean pinned tokens and
the expected-cost decisions, and the reference point of '9 min 54 s after the first arrival'.
  nice -n 19 python3 -I -B misc_checks2.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

rk = rank()
print("[1] idle room (ec_price) inside the two load episodes, three ways to place the episodes on an arm's clock:")
print("    A = from the first to the last send of the arm's kept turns at order positions 5500-6999 / 8750-10318; B = own-order positions 5500-6999 / 8750-10999 (the 'minute each arm reached' rows); "
      "C = minutes with a mean TTFT over 1 s among returning turns")
for a in ("ec_r13", "nofc", "ec_r1", "nofc_noend"):
    d = arm(a)
    pr = d["server"]["evts"]["ec_price"]
    t = d["turns"]
    wins = {}
    wins["A"] = [(min(r["t_send"] for k, r in d["by"].items() if lo <= rk.get(k, -1) < hi), max(r["t_send"] for k, r in d["by"].items() if lo <= rk.get(k, -1) < hi)) for lo, hi in EPISODES]
    wins["B"] = [(t[5500]["t_send"], t[6999]["t_send"]), (t[8750]["t_send"], t[min(10999, len(t) - 1)]["t_send"])]
    hot = []
    for m in range(int(d["end"] // 60)):
        v = [r["ttft"] for r in t if m * 60 <= r["t_send"] < (m + 1) * 60 and r["turn"] > 0]
        if v and mean(v) > 1.0:
            hot.append((m * 60, (m + 1) * 60))
    wins["C"] = hot
    cells = []
    for name, w in wins.items():
        x = [e for e in pr if any(lo <= e["t"] - d["t0"] < hi for lo, hi in w)]
        cells.append(f"{name}: {len(x)} intervals, idle median {st.median(e['idle'] for e in x) / 1000:.1f}K mean {mean(e['idle'] for e in x) / 1000:.1f}K, under 10K {100 * sum(e['idle'] < 10000 for e in x) / len(x):.0f} %, "
                     f"refused mean {mean(e['refused'] for e in x) / 1000:.1f}K")
    print(f"  {a:11s} whole run idle median {st.median(e['idle'] for e in pr) / 1000:.1f}K | " + " | ".join(cells))
    per = []
    for i, (lo, hi) in enumerate(wins["B"]):
        x = [e for e in pr if lo <= e["t"] - d["t0"] < hi]
        per.append(f"B episode {i + 1} (min {lo / 60:.1f}-{hi / 60:.1f}): idle median {st.median(e['idle'] for e in x) / 1000:.1f}K, under 10K {100 * sum(e['idle'] < 10000 for e in x) / len(x):.0f} %, refused {mean(e['refused'] for e in x) / 1000:.1f}K")
    print("       " + " | ".join(per))

print("\n[2] do the arrival-only arms start a read before its turn arrives? per arm: s2h_io reads | whose start (end - ms) is over 10 ms before the turn's entry_time | parks over 10 ms before entry | reads for keys with no ReqTimeStats")
for a in ("nofc", "nofc_noend", "hintfree_ctl", "ec_r13", "jit_exact"):
    d = arm(a)
    early = sum(1 for k, v in d["ios"].items() if k in d["entry"] for t, ms, _ in v if t - ms < d["entry"][k] - 0.01)
    nostat = sum(len(v) for k, v in d["ios"].items() if k not in d["entry"])
    parks = sum(1 for e in d["server"]["evts"]["srs_park"] if (e["conv"], e["turn"]) in d["entry"] and e["t"] < d["entry"][(e["conv"], e["turn"])] - 0.01)
    print(f"  {a:12s} {sum(len(v) for v in d['ios'].values()):5d} | {early:5d} | {parks:5d} | {nostat:3d}")

print("\n[3] checks/predwait.out, recomputed: position at arrival | no read: n, predicted, actual mean, median | read: n, predicted, actual mean, median, held before dispatch mean")
for a in ("nofc", "ec_r13", "jit_p50_ctl"):
    d = arm(a)
    ev = d["server"]["evts"]
    pos = {(e["conv"], e["turn"]): e["pos"] for e in ev["ntp_arrive"]}
    adm = {(e["conv"], e["turn"]): (e["wait_ms"] / 1000, None if e["pred_wait_ql_ms"] in ("na", "None") else float(e["pred_wait_ql_ms"]) / 1000) for e in ev["ntp_admit"]}
    read = set(d["ios"])
    held = collections.Counter()
    for e in ev["srs_dispatch"]:
        held[(e["conv"], e["turn"])] += e["waited_ms"]
    print(f"  == {a}: {len(adm)} admitted turns, {len(read)} that read")
    for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 7), (8, 99)):
        cells = []
        for is_read in (False, True):
            ks = [k for k in adm if k in pos and lo <= pos[k] <= hi and (k in read) == is_read and k[1] > 0]
            pred = [adm[k][1] for k in ks if adm[k][1] is not None]
            act = [adm[k][0] for k in ks]
            cell = f"{len(ks):5d} {mean(pred):6.2f} {mean(act):6.2f} {st.median(act) if act else float('nan'):6.2f}"
            if is_read:
                cell += f" {mean(held[k] / 1000 for k in ks):6.2f}"
            cells.append(cell)
        print(f"     {lo:2d}-{hi:2d} | " + " | ".join(cells))

print("\n[4] mean pinned tokens over the kept window, unpins by reason, expected-cost decisions, releases")
for a in ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "hintfree_ctl"):
    d = arm(a)
    ev = d["server"]["evts"]
    series = sorted([(e["t"] - d["t0"], e["pinned"]) for name in ("srs_pin", "srs_unpin") for e in ev.get(name, []) if 0 <= e["t"] - d["t0"] < d["end"]])
    # the two event types interleave in the log; equal timestamps keep pin before unpin only by luck, so integrate both orders
    total, t0, v0 = 0.0, 0.0, 0
    for t, v in series:
        total += v0 * (t - t0)
        t0, v0 = t, v
    level = (total + v0 * (d["end"] - t0)) / d["end"]
    print(f"  {a:13s} mean pinned {level:7.0f} | unpins {dict(collections.Counter(e['reason'] for e in ev.get('srs_unpin', [])))} | decisions {dict(collections.Counter(e['decision'] for e in ev.get('ec_start', [])))} | "
          f"releases {len(ev.get('ec_release', []))} bumps {len(ev.get('ec_bump', []))}")

print("\n[5] 'first staged read ... 9 min 54 s after the first arrival': the first ntp_stage against several first events")
for a in ("jit_p50_ctl", "jit_p50_d_cut", "jit_p50"):
    d = arm(a)
    ev = d["server"]["evts"]
    stg = min(ev["ntp_stage"], key=lambda e: e["t"])
    refs = {"client t_start": d["t0"], "first c-t ReqTimeStats entry_time": min(d["entry"].values()), "first ntp_arrive line": min(e["t"] for e in ev["ntp_arrive"]),
            "first ntp_plan line": min(e["t"] for e in ev["ntp_plan"]), "first ntp_pick line": min(e["t"] for e in ev.get("ntp_pick", [{"t": float("nan")}]))}
    print(f"  {a:13s} c{stg['conv']}-t{stg['turn']}: " + "; ".join(f"{k} {int((stg['t'] - v) // 60)} min {(stg['t'] - v) % 60:.1f} s" for k, v in refs.items()))

print("\n[6] own-set against shared-set means: does the intersection move an arm's mean (returning turns)")
for names, label in ((LEAD5, "LEAD5"), (SEVEN, "SEVEN"), (ALL11, "ALL11")):
    keys = shared(names)
    print(f"  {label} ({len(keys)}): " + "  ".join(f"{a} own {mean(r['ttft'] for k, r in arm(a)['by'].items() if k[1] > 0):.2f} ({sum(1 for k in arm(a)['by'] if k[1] > 0)}) shared {mean(arm(a)['by'][k]['ttft'] for k in keys):.2f}" for a in names))
