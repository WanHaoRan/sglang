"""Collect out/compare_*.json into one table: per arm and volume setting, minutes 90-110 (the 128-user cap binds).
Usage: python3 -I -B summary.py"""
import json, glob, os
SEL = ["a_oldest_written", "b_end_chain", "b_end_own", "b_end_index", "b_end_tail", "touch_on_read", "c_refresh", "c_refresh_60s",
       "c_refresh_300s", "c_refresh_headfirst", "d_chain+refresh", "d_index+refresh", "d_tail+refresh", "d_tier+refresh",
       "d_tier+refresh_300s", "d_tier+touch_on_read", "d_tier_index+refresh", "d_tier_index+refresh+hit+recheck",
       "d_tier_shared+refresh+hit+recheck", "e_oracle_next_use"]
SC = [("s1_90_80", "25.0 M, 90/80 (as run)"), ("s1_80_70", "25.0 M, 80/70"), ("s0.5_90_80", "12.5 M, 90/80"), ("s0.25_90_80", "6.2 M, 90/80")]
ARMS = ["ec_full", "jit_exact", "jit_p50"]
data = {}
for a in ARMS:
    for key, _ in SC:
        p = f"out/compare_{a}_{key}.json"
        if os.path.exists(p):
            data[(a, key)] = json.load(open(p))
for key, label in SC:
    print(f"\n=== volume {label}: minutes 90-110 (requests sent 5,400-6,600 s): requests losing / K tokens recomputed beyond a store that deletes nothing / "
          f"TTFT at stake by N g (N = 128, 0.043 ms a token), s; and live pages deleted before 6,600 s")
    print(f"{'order':34s} " + " | ".join(f"{a:^34s}" for a in ARMS))
    for name in SEL:
        cells = []
        for a in ARMS:
            d = data.get((a, key))
            if d is None or name not in d["results"]:
                cells.append(f"{'-':^34s}"); continue
            r = d["results"][name]; w = r["minutes_90_110"]
            cells.append(f"{w['lost_requests']:4d} /{w['extra_tokens']/1e3:8.1f} /{w['ttft_s']:6.2f} s; live {r['live_deleted_before_6600']:6d}")
        print(f"{name:34s} " + " | ".join(cells))
    for a in ARMS:
        d = data.get((a, key))
        if d:
            r = d["results"]["a_oldest_written"]
            print(f"   {a}: order a: first pass at {d['first_pass_a']:.0f} s, {r['passes_before_6600']} passes before 6,600 s, "
                  f"file times deleted up to {[round(p['newest_deleted']) for p in r['pass_log']][:4]}... s at passes {[round(p['t']) for p in r['pass_log']][:4]}...; "
                  f"requests in the window {r['minutes_90_110']['requests']}")
print("\n=== lookups a deleted page cut short (found nothing / found part) and reads that failed on a deleted file, first pass of order a to 6,600 s")
for key, label in SC:
    print(f"  volume {label}")
    for name in SEL:
        cells = []
        for a in ARMS:
            d = data.get((a, key))
            if d is None or name not in d["results"]:
                cells.append("-"); continue
            x = d["results"][name]["lookups_after_first_pass"]; w = d["results"][name]["after_first_pass"]
            cells.append(f"{x['zero']:4d} /{x['part']:4d} /{x['reads_failed']:3d}; requests {w['lost_requests']:4d}, {w['extra_tokens']/1e3:8.1f} K")
        print(f"    {name:34s} " + " | ".join(cells))
print("\n=== file touches the refresh needs (pages a second; two files a page), minutes 90-110 mean and the peak 10 s")
for a in ARMS:
    d = data.get((a, "s1_90_80"))
    if not d: continue
    for name in ("c_refresh", "c_refresh_60s", "c_refresh_300s", "touch_on_read", "c_refresh+hit"):
        r = d["results"][name]
        print(f"   {a:10s} {name:18s}: {r['touch_pages_per_s_late']:7.1f} pages/s = {2*r['touch_pages_per_s_late']:7.1f} files/s late; minutes 30-110 "
              f"{r['touch_pages_per_s_30_110']:7.1f} pages/s; peak 10 s {r['touch_pages_per_s_peak10s']:7.1f} pages/s")
