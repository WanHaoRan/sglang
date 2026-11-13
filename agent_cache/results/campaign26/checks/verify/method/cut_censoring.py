"""The cut run (jit_p50_d_cut) against the full JIT p50 runs: does it matter that only the cut run is truncated at
minute 74.7 of its own clock while the other runs keep 115 minutes?
1. Turns the cut run's server saw arrive but its client never recorded (in flight at the outage).
2. The published comparison (shared set = cut run's kept turns) against a symmetric one (every JIT p50 run truncated at the
   same minute of its own clock), and the placebo: the same two treatments applied to the two full JIT p50 runs, cutting
   one of them at a time.
  nice -n 19 python3 -I -B cut_censoring.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

rk = rank()
d = arm("jit_p50_d_cut")
T = d["end"]
print(f"cut run: last recorded send {d['every'][-1]['t_send']:.1f} s, kept sends before {T:.1f} s = minute {T / 60:.2f}; kept turns {len(d['turns'])}, returning {sum(1 for k in d['by'] if k[1] > 0)}")
s = d["server"]
last_line = max(max(e["t"] for v in s["evts"].values() for e in v), max(r["t"] for r in s["rts"]))
print(f"cut run: last server log event at {last_line - d['t0']:.1f} s after the client's start (minute {(last_line - d['t0']) / 60:.2f})")
arrived = {}
for e in s["evts"].get("prefetch_start", []):
    if e["conv"] is not None:
        arrived.setdefault((e["conv"], e["turn"]), e["t"] - d["t0"])
missing = sorted((t, k) for k, t in arrived.items() if k not in d["by_all"])
print(f"cut run: {len(missing)} request ids arrived at the server with no client record; arrival second and key:")
print("   " + ", ".join(f"{t:.0f}s c{k[0]}-t{k[1]}" for t, k in missing))
print(f"   of them arrived before the kept cutoff ({T:.0f} s): {sum(t < T for t, _ in missing)}; these are missing from the 'kept' turns although sent before the cutoff")
lat = sorted((r["latency"] for r in d["every"]), reverse=True)
print(f"   cut run latencies over 300 s among recorded turns: {sum(x > 300 for x in lat)}; in jit_p50_ctl: {sum(r['latency'] > 300 for r in arm('jit_p50_ctl')['every'])}, of them sent before {T:.0f} s: "
      f"{sum(r['latency'] > 300 and r['t_send'] < T for r in arm('jit_p50_ctl')['every'])}; with t_send + latency > {T + 300:.0f} s and t_send < {T:.0f} s: "
      f"{sum(r['t_send'] + r['latency'] > T + 300 and r['t_send'] < T for r in arm('jit_p50_ctl')['every'])}")
for a in ("jit_p50_ctl", "jit_p50"):
    x = [r for r in arm(a)["every"] if r["t_send"] < T and r["t_send"] + r["latency"] > T + 300]
    print(f"   {a}: turns sent before {T:.0f} s and still running {300} s after it: {len(x)}; their ttft: {[round(r['ttft'], 1) for r in x]}; turn numbers {[r['turn'] for r in x]}")


def cmp(names, cuts, label, show=True):
    """names: arms; cuts: {arm: own-clock cutoff in s or None for the arm's usual kept turns}."""
    arms = {a: arm(a, cut_at=cuts.get(a)) for a in names}
    keys = shared(names, arms)
    ep = [k for k in keys if in_episode(k)]
    calm = [k for k in keys if not in_episode(k)]
    out = {}
    for a in names:
        out[a] = (mean(arms[a]["by"][k]["ttft"] for k in keys), mean(arms[a]["by"][k]["ttft"] for k in ep), mean(arms[a]["by"][k]["ttft"] for k in calm))
    if show:
        print(f"  {label}: {len(keys)} shared returning turns ({len(ep)} in the episode positions, {len(calm)} elsewhere)")
        print("     " + "   ".join(f"{a} {out[a][0]:.3f} (ep {out[a][1]:.3f}, else {out[a][2]:.3f})" for a in names))
    return keys, out, arms


JIT = ["jit_p50", "jit_p50_ctl", "jit_p50_d_cut"]
print("\n[1] the three JIT p50 runs alone")
k_pub, o_pub, _ = cmp(JIT, {}, "published treatment: only the cut run is truncated (others keep 115 min)")
k_sym, o_sym, _ = cmp(JIT, {a: T for a in JIT}, f"symmetric: every run truncated at {T:.0f} s of its own clock")
for x, y in (("jit_p50_d_cut", "jit_p50_ctl"), ("jit_p50_d_cut", "jit_p50"), ("jit_p50_ctl", "jit_p50")):
    print(f"     {x} - {y}: published {o_pub[x][0] - o_pub[y][0]:+.3f} (episode {o_pub[x][1] - o_pub[y][1]:+.3f}, elsewhere {o_pub[x][2] - o_pub[y][2]:+.3f}) | "
          f"symmetric {o_sym[x][0] - o_sym[y][0]:+.3f} (episode {o_sym[x][1] - o_sym[y][1]:+.3f}, elsewhere {o_sym[x][2] - o_sym[y][2]:+.3f})")
only_pub = [k for k in k_pub if k not in set(k_sym)]
print(f"     turns in the published set and not in the symmetric one: {len(only_pub)}; their mean TTFT: " + "  ".join(f"{a} {mean(arm(a)['by'][k]['ttft'] for k in only_pub):.2f}" for a in JIT))
print("     when each run sent those turns (own-clock minute, min / median / max): " + "  ".join(
    f"{a} {min(arm(a)['by'][k]['t_send'] for k in only_pub) / 60:.1f}/{st.median(arm(a)['by'][k]['t_send'] for k in only_pub) / 60:.1f}/{max(arm(a)['by'][k]['t_send'] for k in only_pub) / 60:.1f}" for a in JIT))

print("\n[2] CUT7 (the write-up's set), published and symmetric (the three JIT p50 runs truncated alike; the other four arms as usual)")
k_pub7, o_pub7, _ = cmp(CUT7, {}, "published")
k_sym7, o_sym7, _ = cmp(CUT7, {a: T for a in JIT}, "symmetric")
for x, y in (("jit_p50_d_cut", "jit_p50_ctl"), ("jit_p50_d_cut", "jit_p50"), ("jit_p50_ctl", "jit_p50"), ("nofc_noend", "jit_p50_d_cut"), ("ec_r13", "jit_p50_d_cut"), ("jit_exact", "jit_p50_d_cut"),
             ("nofc_noend", "jit_p50_ctl"), ("jit_exact", "jit_p50")):
    print(f"     {x} - {y}: published {o_pub7[x][0] - o_pub7[y][0]:+.3f} (episode {o_pub7[x][1] - o_pub7[y][1]:+.3f}) | symmetric {o_sym7[x][0] - o_sym7[y][0]:+.3f} (episode {o_sym7[x][1] - o_sym7[y][1]:+.3f})")

print("\n[3] placebo on the two full JIT p50 runs: cut one of them at minute M of its own clock, keep the other whole (the published treatment), against cutting both")
print("     M | A cut, B whole: n, B - A | B cut, A whole: n, B - A | both cut: n, B - A      (A = jit_p50, B = jit_p50_ctl)")
for M in (55, 60, 65, 70, 72, 74.7, 77, 80, 85, 90, 95, 100, 105, 110):
    c = M * 60
    ka, oa, _ = cmp(["jit_p50", "jit_p50_ctl"], {"jit_p50": c}, "", show=False)
    kb, ob, _ = cmp(["jit_p50", "jit_p50_ctl"], {"jit_p50_ctl": c}, "", show=False)
    ks, os_, _ = cmp(["jit_p50", "jit_p50_ctl"], {"jit_p50": c, "jit_p50_ctl": c}, "", show=False)
    f = lambda o: o["jit_p50_ctl"][0] - o["jit_p50"][0]
    print(f"     {M:5.1f} | {len(ka):5d} {f(oa):+.3f} | {len(kb):5d} {f(ob):+.3f} | {len(ks):5d} {f(os_):+.3f}    spread between the two one-sided cuts {f(oa) - f(ob):+.3f}")

print("\n[4] the same placebo with the cut run as one side: jit_p50_d_cut against each full run cut at minute M (both sides cut alike) or kept whole")
for other in ("jit_p50_ctl", "jit_p50"):
    for M in (55, 60, 65, 70, 72, 74.7):
        c = M * 60
        ka, oa, _ = cmp(["jit_p50_d_cut", other], {"jit_p50_d_cut": c}, "", show=False)
        ks, os_, _ = cmp(["jit_p50_d_cut", other], {"jit_p50_d_cut": c, other: c}, "", show=False)
        f = lambda o: o["jit_p50_d_cut"][0] - o[other][0]
        print(f"     d_cut - {other:12s} M {M:5.1f} | d_cut cut only: n {len(ka):5d} {f(oa):+.3f} | both cut: n {len(ks):5d} {f(os_):+.3f}")
