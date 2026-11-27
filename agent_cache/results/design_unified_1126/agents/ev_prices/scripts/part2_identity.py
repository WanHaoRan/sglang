"""Part 2: the first-order identity of DESIGN.md section 1.5 on the data.

Per window of minutes and run:
  N      users in session (time average)            X   requests finished per second       g = 1 / X
  T      mean TTFT of the requests sent in the window
  D      mean decode time (latency - TTFT) of the requests finished in the window
  LT, LD time-average number of requests waiting for their first token / decoding (client clocks)
  Z      pause per request implied by the state counts: (N - LT - LD) / X
  id     N / X - D - Z  (Little's law prediction of T; equals LT / X by construction, shown against T)
  U      history computed again per request finished, tokens (ideal_cached - cached_tokens, floor 0),
         and its GPU seconds at the run's loaded cost (prefill_fit.json: per token at prefix 16K + per batch share)
  pf     all prefill tokens per request (input - cached, server)
  Q, Qr  queue and ready queue
Pairs (A, B): dT = T_A - T_B against N x dg (dg = g_A - g_B), against the recompute GPU seconds per request, and
the implied mu = dT / d(recompute GPU s per request); 1 + Qr, 1 + Q and N beside it.
usage: part2_identity.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

FIT = json.load(open(f"{lib.OUT}/prefill_fit.json"))


def gamma(spec):
    f = FIT.get(spec, {}).get("loaded")
    return (f["ms_per_tok_20K"] / 1e3) if f else 0.043e-3


def state(camp, arm):
    st = qstate.build(camp, arm)
    rows = readers.annotate(st)
    d_t, d_d = [], []
    for r in rows:
        if r["ttft"] is None or r["latency"] is None:
            continue
        s, ft, e = r["t_send"], r["t_send"] + r["ttft"], r["t_send"] + r["latency"]
        d_t += [(s, 1), (ft, -1)]
        d_d += [(ft, 1), (e, -1)]
    st["LT"] = qstate.Step.from_deltas(d_t)
    st["LD"] = qstate.Step.from_deltas(d_d)
    st["arows"] = rows
    return st


def window(st, spec, lo, hi):
    rows = st["arows"]
    w = hi - lo
    sent = [r for r in rows if lo <= r["t_send"] < hi and r["ttft"] is not None]
    fin = [r for r in rows if r["latency"] is not None and lo <= r["t_send"] + r["latency"] < hi]
    adm = [r for r in rows if lo <= r["adm"] < hi]
    o = {}
    o["N"] = st["n_live"].mean(lo, hi)
    o["n_sent"], o["n_fin"] = len(sent), len(fin)
    o["X"] = len(fin) / w
    o["g"] = w / len(fin) if fin else float("nan")
    o["T"] = lib.mean(r["ttft"] for r in sent)
    o["T_fin"] = lib.mean(r["ttft"] for r in fin)
    o["D"] = lib.mean(r["latency"] - r["ttft"] for r in fin)
    o["LT"], o["LD"] = st["LT"].mean(lo, hi), st["LD"].mean(lo, hi)
    o["Z"] = (o["N"] - o["LT"] - o["LD"]) / o["X"] if o["X"] else float("nan")
    o["id"] = o["N"] / o["X"] - o["D"] - o["Z"] if o["X"] else float("nan")
    o["Q"], o["Qr"] = st["tot"].mean(lo, hi), st["rdy"].mean(lo, hi)
    u = sum(max(0, r["ideal"] - r["cached"]) for r in adm)
    o["U"] = u / len(adm) if adm else float("nan")
    o["Ugpu"] = o["U"] * gamma(spec)
    o["pf"] = sum((r["in_len"] or 0) - (r["cached_srv"] or 0) for r in adm) / len(adm) if adm else float("nan")
    rd = [r for r in adm if r["k"] > 0 and r["n_reads"] > 0]
    o["stall"] = sum(r["stall"] for r in adm) / len(adm) if adm else float("nan")
    o["rd_share"] = len(rd) / max(1, sum(1 for r in adm if r["k"] > 0))
    o["out"] = lib.mean(r["out"] or 0 for r in fin)
    return o


def main():
    specs = ["c32:ec_full", "c34:ec_full", "c32:jit_exact", "c34:jit_exact", "c32:jit_p50", "c34:jit_p50",
             "c32:ec_exact", "c34:ec_exact", "c34:hicache", "c32:hicache"]
    sts = {s: state(*s.split(":")) for s in specs}
    wins = [(m * 60.0, (m + 5) * 60.0) for m in range(30, 110, 5)]
    tab = {s: [window(sts[s], s, lo, hi) for lo, hi in wins] for s in specs}
    json.dump({s: tab[s] for s in specs}, open(f"{lib.OUT}/part2_windows.json", "w"))
    print("A. state counts per 5-minute window (Little's law: LT / X against the mean TTFT of the requests sent)")
    for s in ("c34:ec_full", "c32:ec_full", "c34:hicache"):
        print(f"== {s} (loaded recompute cost {gamma(s)*1e3:.4f} ms/token)")
        print("  minutes |    N  req/min   g s |  T sent  T fin   LT/X |    D s    Z s | N*g = T+D+Z? |  LT   LD   Q   Qr | U tok/req  GPU s/req | pf tok/req")
        for (lo, hi), o in zip(wins, tab[s]):
            print(f"  {lo/60:3.0f}-{hi/60:3.0f} | {o['N']:5.1f} {o['X']*60:7.1f} {o['g']:5.3f} | {o['T']:6.2f} {o['T_fin']:6.2f} {o['LT']/o['X']:6.2f} | "
                  f"{o['D']:6.2f} {o['Z']:6.2f} | {o['N']*o['g']:6.1f} {o['T_fin']+o['D']+o['Z']:6.1f} | {o['LT']:4.1f} {o['LD']:4.1f} {o['Q']:4.1f} {o['Qr']:4.1f} | "
                  f"{o['U']:8.0f} {o['Ugpu']:9.4f} | {o['pf']:7.0f}")
    print()
    print("B. pairs: A minus B per 5-minute window")
    pairs = [("c32:ec_full", "c34:ec_full"), ("c32:jit_exact", "c34:jit_exact"), ("c32:jit_p50", "c34:jit_p50"),
             ("c32:ec_exact", "c34:ec_exact"), ("c34:hicache", "c34:ec_full"), ("c34:jit_p50", "c34:ec_full"),
             ("c34:jit_exact", "c34:ec_full")]
    for a, b in pairs:
        print(f"== {a} - {b}")
        print("  minutes | N_A   N_B | dT s | dg s  N_B*dg | dD s  dZ s | dU GPU s/req | mu = dT/dUgpu | dT/dg | 1+Qr_A 1+Q_A | d stall s/req")
        for (lo, hi), x, y in zip(wins, tab[a], tab[b]):
            dT, dg, dD, dZ = x["T"] - y["T"], x["g"] - y["g"], x["D"] - y["D"], x["Z"] - y["Z"]
            dU = x["Ugpu"] - y["Ugpu"]
            mu = dT / dU if abs(dU) > 0.02 else float("nan")
            sl = dT / dg if abs(dg) > 0.02 else float("nan")
            print(f"  {lo/60:3.0f}-{hi/60:3.0f} | {x['N']:5.1f} {y['N']:5.1f} | {dT:+7.2f} | {dg:+6.3f} {y['N']*dg:+7.1f} | {dD:+6.2f} {dZ:+6.2f} | "
                  f"{dU:+8.4f} | {mu:8.1f} | {sl:7.1f} | {1+x['Qr']:5.1f} {1+x['Q']:5.1f} | {x['stall']-y['stall']:+7.2f}")
    # the saturated window as one piece, minutes 90-110 (and 85-110)
    print()
    print("C. whole windows")
    for lo, hi in ((90 * 60.0, 110 * 60.0), (85 * 60.0, 110 * 60.0), (30 * 60.0, 60 * 60.0), (60 * 60.0, 85 * 60.0)):
        print(f"  minutes {lo/60:.0f}-{hi/60:.0f}")
        ws = {s: window(sts[s], s, lo, hi) for s in specs}
        for s in specs:
            o = ws[s]
            print(f"    {s:14s} N {o['N']:5.1f} req/min {o['X']*60:6.1f} g {o['g']:.3f} | T {o['T']:6.2f} (fin {o['T_fin']:6.2f}) D {o['D']:6.2f} Z {o['Z']:6.2f} "
                  f"sum {o['T_fin']+o['D']+o['Z']:6.1f} vs N*g {o['N']*o['g']:6.1f} | U {o['U']:7.0f} tok = {o['Ugpu']:.4f} GPU s | "
                  f"pf {o['pf']:6.0f} | Q {o['Q']:5.1f} Qr {o['Qr']:5.1f} | stall/req {o['stall']:5.2f} | out {o['out']:.0f}")
        for a, b in pairs:
            x, y = ws[a], ws[b]
            dT, dg, dD, dZ, dU = x["T"] - y["T"], x["g"] - y["g"], x["D"] - y["D"], x["Z"] - y["Z"], x["Ugpu"] - y["Ugpu"]
            print(f"    {a} - {b}: dT {dT:+.2f} s; dg {dg:+.4f} s -> N*dg {y['N']*dg:+.1f} s ({dT/(y['N']*dg) if abs(dg)>1e-3 else float('nan'):.2f} of it in TTFT), "
                  f"dD {dD:+.2f}, dZ {dZ:+.2f}; d recompute GPU {dU:+.4f} s/req ({dU/dg if abs(dg)>1e-3 else float('nan'):.2f} of dg); "
                  f"mu = dT / dUgpu = {dT/dU if abs(dU)>1e-3 else float('nan'):.1f}; d stall/req {x['stall']-y['stall']:+.2f} s")


if __name__ == "__main__":
    main()
