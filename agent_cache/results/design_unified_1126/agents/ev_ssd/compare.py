"""Steps 2 and 3: cleaning orders on the same timeline.
Usage: python3 -I -B compare.py <arm> <volume scale> <high mark> <low mark> [policy names...]
Writes out/compare_<arm>_s<scale>_<hi>_<lo>.json and prints a table."""
import sys, json, collections, time
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *

POLICIES = collections.OrderedDict([
    ("a_oldest_written", {}),
    ("b_end_chain", dict(end_zero="chain")),
    ("b_end_own", dict(end_zero="own")),
    ("b_end_index", dict(end_zero="index")),
    ("b_end_tail", dict(end_zero="tail")),
    ("b_end_private", dict(end_zero="priv")),
    ("touch_on_read", dict(touch_on_hit=True)),
    ("c_refresh", dict(refresh="end")),
    ("c_refresh_60s", dict(refresh="end", refresh_limit=60.0)),
    ("c_refresh_300s", dict(refresh="end", refresh_limit=300.0)),
    ("c_refresh_headfirst", dict(refresh="end", head_last=False)),
    ("c_refresh+hit", dict(refresh="end", touch_on_hit=True)),
    ("d_chain+refresh", dict(end_zero="chain", refresh="end")),
    ("d_own+refresh", dict(end_zero="own", refresh="end")),
    ("d_index+refresh", dict(end_zero="index", refresh="end")),
    ("d_tail+refresh", dict(end_zero="tail", refresh="end")),
    ("d_tail+refresh_60s", dict(end_zero="tail", refresh="end", refresh_limit=60.0)),
    ("d_tail+refresh_300s", dict(end_zero="tail", refresh="end", refresh_limit=300.0)),
    ("d_tail+refresh+hit", dict(end_zero="tail", refresh="end", touch_on_hit=True)),
    ("d_tail+touch_on_read", dict(end_zero="tail", touch_on_hit=True)),
    ("d_tier+refresh", dict(end_zero="tier", refresh="end")),
    ("d_tier+refresh_300s", dict(end_zero="tier", refresh="end", refresh_limit=300.0)),
    ("d_tier+touch_on_read", dict(end_zero="tier", touch_on_hit=True)),
    ("d_tier_index+refresh", dict(end_zero="tier_index", refresh="end")),
    ("d_tier_index+refresh_300s", dict(end_zero="tier_index", refresh="end", refresh_limit=300.0)),
    ("d_tier_index+refresh+hit", dict(end_zero="tier_index", refresh="end", touch_on_hit=True)),
    ("d_tier_index+refresh+hit+recheck", dict(end_zero="tier_index", refresh="end", touch_on_hit=True, recheck=True)),
    ("d_tier_shared+refresh+hit+recheck", dict(end_zero="tier_shared", refresh="end", touch_on_hit=True, recheck=True)),
    ("a+hit+recheck", dict(touch_on_hit=True, recheck=True)),
    ("d_private+refresh", dict(end_zero="priv", refresh="end")),
    ("d_private+refresh+hit", dict(end_zero="priv", refresh="end", touch_on_hit=True)),
    ("e_oracle_next_use", dict(order="oracle_use")),
    ("e_oracle_ssd_need", dict(order="oracle_need")),
])

def main():
    arm, scale, hi, lo = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    names = sys.argv[5:] or list(POLICIES)
    d = load_arm(arm)
    d["sharing"] = page_sharing(d)
    d["shared_since"] = page_shared_since(d)
    d["first_req"] = first_requests(d)
    cap = int(390100 * scale)
    ref = Sim(d, {}, passes="none").run()
    res = {}
    base = Sim(d, {}, passes="volume", cap=cap, hi=hi, lo=lo).run()
    first = base.pass_log[0]["t_snap"] if base.pass_log else 7300.0
    for name in names:
        t0 = time.time()
        sim = Sim(d, POLICIES[name], passes="volume", cap=cap, hi=hi, lo=lo).run()
        w_all = summarize(sim, ref.out, first, 6600.0)
        w_late = summarize(sim, ref.out, 5400.0, 6600.0)
        w_run = summarize(sim, ref.out, 0.0, 6600.0)
        deleted = sum(p["deleted"] for p in sim.pass_log)
        live = sum(p["live_use"] for p in sim.pass_log)
        live_sh = sum(p["live_shared"] for p in sim.pass_log)
        need = sum(p["live_need"] for p in sim.pass_log)
        pl = [p for p in sim.pass_log if p["t"] < 6600.0]
        tch_late = sum(n for t, n in sim.touches if 5400.0 <= t < 6600.0) / 1200.0
        tch_all = sum(n for t, n in sim.touches if 1800.0 <= t < 6600.0) / 4800.0
        # peak touches per second over 10 s bins
        bins = collections.Counter()
        for t, n in sim.touches:
            bins[int(t // 10)] += n
        peak = max(bins.values()) / 10.0 if bins else 0.0
        def lk(lo, hi):
            z = sum(1 for t, k in sim.lookup_losses if lo <= t < hi and k == "zero")
            pt = sum(1 for t, k in sim.lookup_losses if lo <= t < hi and k == "part")
            rf = sum(1 for x in sim.readfail_pred if lo <= x[5] < hi)
            return dict(zero=z, part=pt, reads_failed=rf)
        res[name] = dict(policy=POLICIES[name], passes=len(sim.pass_log), passes_before_6600=len(pl),
                         lookups_after_first_pass=lk(first, 6600.0), lookups_minutes_90_110=lk(5400.0, 6600.0),
                         first_pass=sim.pass_log[0]["t_snap"] if sim.pass_log else None, deleted=deleted,
                         deleted_before_6600=sum(p["deleted"] for p in pl),
                         live_deleted_before_6600=sum(p["live_use"] for p in pl),
                         live_shared_before_6600=sum(p["live_shared"] for p in pl),
                         need_deleted_before_6600=sum(p["live_need"] for p in pl),
                         live_deleted=live, live_shared=live_sh, need_deleted=need, zeroed=sim.zeroed,
                         after_first_pass=w_all, minutes_90_110=w_late, to_6600=w_run,
                         touch_pages_per_s_late=tch_late, touch_pages_per_s_30_110=tch_all, touch_pages_per_s_peak10s=peak,
                         pass_log=sim.pass_log, secs=round(time.time() - t0, 1))
    out = f"out/compare_{arm}_s{scale:g}_{int(hi*100)}_{int(lo*100)}.json"
    json.dump(dict(arm=arm, cap_pages=cap, hi=hi, lo=lo, first_pass_a=first, results=res), open(out, "w"))
    print(f"{arm}: volume {cap} pages ({cap*64/1e6:.1f} M tokens), marks {hi:.0%}/{lo:.0%}; first pass of (a) at {first:.0f} s "
          f"({first/60:.1f} min); reference = nothing ever deleted")
    hdr = (f"{'policy':34s} {'passes':>6s} {'deleted':>8s} {'live del':>8s} {'(shared)':>8s} | after the first pass to 6,600 s: "
           f"{'lost':>4s} {'zero_h':>6s} {'zero_m':>6s} {'part':>5s} {'in_fl':>5s} {'extra K':>8s} {'TTFT s':>7s} | min 90-110: "
           f"{'lost':>4s} {'extra K':>8s} {'TTFT s':>7s} | {'touch/s':>8s}")
    print(hdr)
    for name, r in res.items():
        a, l = r["after_first_pass"], r["minutes_90_110"]
        k = a["by_kind"]
        print(f"{name:34s} {r['passes_before_6600']:6d} {r['deleted_before_6600']:8d} {r['live_deleted_before_6600']:8d} "
              f"{r['live_shared_before_6600']:8d} | {'':32s}{a['lost_requests']:4d} {k.get('zero_head',0):6d} {k.get('zero_mid',0):6d} "
              f"{k.get('partial',0):5d} {k.get('read_in_flight',0):5d} {a['extra_tokens']/1e3:8.1f} {a['ttft_s']:7.2f} | {'':11s}"
              f"{l['lost_requests']:4d} {l['extra_tokens']/1e3:8.1f} {l['ttft_s']:7.2f} | {r['touch_pages_per_s_late']:8.1f}")

main()
