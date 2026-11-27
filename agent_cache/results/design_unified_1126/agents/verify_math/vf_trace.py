"""verify_math, part 2: math_fetch R7 (hazards of the two entries), R8 (J_F on the entries),
R9 (the start rules on the hints the replay sent), R13 (the engine's 64-point read_start).

Reads, read-only: the campaign 34 ec_full client log and python/sglang/srt/mem_cache/next_turn_policy.py.
Run: CUDA_VISIBLE_DEVICES= nice -n 10 <venv python> -I -B vf_trace.py > vf_trace.out
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys

import numpy as np
from scipy import optimize, stats

sys.path.insert(0, ".")
from vlib import Phibar, TwoPiece, Z90, phi  # noqa: E402

CLIENT = (
    "/home/wanhr/sglang/agent_cache/results/20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/"
    "programs_093918_ec_full/client.jsonl"
)
ENGINE = "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py"
CAP = 1200.0
L = 25_000
rng = np.random.default_rng(991177)


# ---------------------------------------------------------------- R7
def r7():
    print("== R7: hazard of the two entries in the engine's two-piece lognormal ==")
    for name, (p10, p50, p90) in (("tool", (0.41, 1.69, 16.8)), ("human", (3.1, 96.0, 1040.0))):
        law = TwoPiece(p10, p50, p90)
        t = np.geomspace(1e-4, 5e3, 2_000_001)
        h = law.hazard_raw(t)
        below = t < law.med
        ipk = int(np.argmax(np.where(below, h, -1)))
        ipk_all = int(np.argmax(h))
        print(f"   {name}: sigma_low {law.sl:.3f}, sigma_high {law.sh:.3f}")
        print(f"      hazard peak below the median: {h[ipk]:.4f} /s at {t[ipk]:.3f} s (percentile {100 * (1 - law.sf_raw(t[ipk])):.1f}); overall peak at {t[ipk_all]:.3f} s")
        hm_lo = float(law.hazard_raw(law.med * (1 - 1e-9)))
        hm_hi = float(law.hazard_raw(law.med * (1 + 1e-9)))
        print(f"      at the median: {hm_lo:.4f} -> {hm_hi:.4f} /s, factor {hm_hi / hm_lo:.3f} (sigma_low / sigma_high = {law.sl / law.sh:.3f})")
        # monotone pieces
        d = np.diff(h)
        rising = t[1:][(d > 0)]
        if len(rising):
            segs = []
            start = rising[0]
            prev = rising[0]
            for v in rising[1:]:
                if v > prev * 1.00002:
                    segs.append((start, prev))
                    start = v
                prev = v
            segs.append((start, prev))
            print(f"      hazard rises on: {[(round(a, 4), round(b, 4)) for a, b in segs[:4]]} (plus the jump at the median if upward)")
        vals = {5.0: None, 16.8: None, 60.0: None, 96.0: None}
        print("      hazard at 5 / 16.8 / 60 s: " + " / ".join(f"{float(law.hazard_raw(v)):.4f}" for v in (5.0, 16.8, 60.0)))
        print(f"      model p99 {float(law.quantile(0.99)):.0f} s; P(G > 1,200 s) = {100 * float(law.sf_raw(CAP)):.2f} %")
        # where F(x|s) rises in s: sign of h(s + x) - h(s)
        for x in (0.5, 1.0, 2.0, 6.0):
            s = np.geomspace(1e-4, 2e3, 400_001)
            S, Sx = law.sf_raw(s), law.sf_raw(s + x)
            Fx = 1 - Sx / S
            dF = np.diff(Fx)
            up = s[1:][dF > 1e-15]
            if len(up):
                segs = []
                a0 = up[0]
                pv = up[0]
                for v in up[1:]:
                    if v > pv * 1.0002:
                        segs.append((a0, pv))
                        a0 = v
                    pv = v
                segs.append((a0, pv))
                print(f"      x = {x}: F(x|s) rises on {[(round(a, 3), round(b, 3)) for a, b in segs[:4]]}")
    # general fact
    g = lambda z: phi(z) / Phibar(z) - z
    s0 = float(g(0.0))
    print(f"   lognormal: hazard peak at the median iff sigma = phi(0)/0.5 = {s0:.4f}, i.e. p90/p50 = {math.exp(s0 * Z90):.3f}")
    for sig in (0.3, 0.5, 1.0, 1.79):
        zs = optimize.brentq(lambda z: float(g(z)) - sig, -12.0, 12.0)
        print(f"      sigma {sig}: peak at median x {math.exp(sig * zs):.4f}, quantile {100 * float(1 - Phibar(zs)):.1f} %")
    # check numerically that a lognormal hazard has one peak at that place
    for sig in (0.4, 1.2):
        t = np.geomspace(1e-5, 1e5, 2_000_001)
        z = np.log(t) / sig
        h = phi(z) / (sig * t * Phibar(z))
        zs = optimize.brentq(lambda z: float(g(z)) - sig, -12.0, 12.0)
        i = int(np.argmax(h))
        nchg = int(np.sum(np.diff(np.sign(np.diff(h[h > 1e-200]))) != 0))
        print(f"      sigma {sig}: grid peak at {t[i]:.5f}, formula {math.exp(sig * zs):.5f}, sign changes of the slope: {nchg}")


def load_rows():
    rows = []
    gaps = {"tool": [], "user": []}
    with open(CLIENT) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "req":
                continue
            if d.get("gap") in gaps and d.get("wait_s") is not None:
                gaps[d["gap"]].append(float(d["wait_s"]))
            h = d.get("hint")
            if not h or "p10" not in h or h.get("true_gap_s") is None:
                continue
            kind = "no_call" if h.get("reply_key") == "_no_call" else "tool"
            rows.append((float(h["p10"]), float(h["p50"]), float(h["p90"]), float(h["true_gap_s"]), kind))
    return rows, gaps


def r7_empirical(gaps):
    print("\n== R7 (empirical): life-table hazard of the gaps the replay sent ==")
    for name in ("tool", "user"):
        g = np.sort(np.array(gaps[name]))
        n = len(g)
        edges = 0.01 * 2.0 ** (np.arange(0, 36) / 2.0)
        edges = edges[edges < 1300]
        print(f"   {name}: n = {n}, at the cap (>= 1,199.9 s): {int(np.sum(g >= 1199.9))}; p10/p50/p90 = {np.quantile(g, 0.1):.2f}/{np.quantile(g, 0.5):.2f}/{np.quantile(g, 0.9):.1f}")
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            at_risk = np.sum(g > lo)
            ev = np.sum((g > lo) & (g <= hi))
            if at_risk < 30:
                continue
            exposure = np.sum(np.minimum(g[g > lo], hi) - lo)
            out.append((lo, hi, ev / exposure, np.mean(g <= hi)))
        ipk = int(np.argmax([o[2] for o in out]))
        print("      bin (s]: hazard /s  [cdf at the bin's end]")
        for (lo, hi, hz, cdf) in out:
            if hi < 0.15:
                continue
            print(f"      ({lo:7.2f}, {hi:7.2f}]: {hz:8.4f}  [{100 * cdf:5.1f} %]" + ("  <- peak" if (lo, hi) == out[ipk][:2] else ""))


# ---------------------------------------------------------------- R8
def gains_table():
    print("\n== R8: J_F on the two entries (engine law capped at 1,200 s, L = 25,000, copy held to admission) ==")
    tool = TwoPiece(0.41, 1.69, 16.8, cap=CAP)
    human = TwoPiece(3.1, 96.0, 1040.0, cap=CAP)
    tool_raw = TwoPiece(0.41, 1.69, 16.8)
    sgrid = np.concatenate([[0.0], np.geomspace(1e-3, CAP, 20_000)])
    for name, law in (("tool", tool), ("human", human)):
        print(f"   {name} entry:")
        for lam in (0.0, 1e-6, 2e-5, 1e-4):
            lamL = lam * L
            for c in (1.0, 0.3):
                row = []
                for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                    grid = np.unique(np.concatenate([sgrid, [CAP - x]]))
                    J = c * x - c * law.M(grid) + (c + lamL) * law.M(grid + x)
                    i = int(J.argmin())
                    Jarr = c * x
                    gain_best = Jarr - J[i]
                    gain_now = Jarr - J[0]
                    row.append((x, grid[i] if J[i] < Jarr - 1e-12 else None, gain_best, gain_now))
                txt = "; ".join(f"x={x:.0f}: best {'arrival' if s is None else f'{s:.2f}'} gain {gb:.4f} (now: {gn:+.3f})" for (x, s, gb, gn) in row)
                print(f"      lambda = {lam:g}, c = {c}: {txt}")
    print("   exact boundary on the uncapped tool entry: read now (held to admission) pays iff lambda L / c < E[min(G,x)] / E[(G-x)^+]")
    mean = float(tool_raw.M(0.0))
    for x in (1.0, 2.0, 3.0, 4.0, 6.0):
        Mx = float(tool_raw.M(x))
        ratio = (mean - Mx) / Mx
        print(f"      x = {x:.0f}: E[min] = {mean - Mx:.4f}, E[(G-x)^+] = {Mx:.4f}, ratio {ratio:.4f} -> lambda < {ratio / L:.2e} at c = 1")
    # minimum at an end? (x > 0, c > 0), and the threshold rule on the same cells
    for name, law in (("tool", tool), ("human", human)):
        cells = 0
        interior = 0
        thr_early = 0
        thr_worse = 0
        worst = (0.0, 0.0)
        for r in (2.0, 3.0, 4.0, 5.0, 6.0):
            for W in (0.0, 2.0, 5.0):
                x = r - W
                if x <= 0:
                    continue
                for c in (1.0, 0.3):
                    for lam in (0.0, 1e-6, 2e-5, 1e-4):
                        lamL = lam * L
                        q = lamL / (c + lamL)
                        grid = np.unique(np.concatenate([sgrid, [CAP - x]]))
                        J = c * x - c * law.M(grid) + (c + lamL) * law.M(grid + x)
                        Jarr = c * x
                        i = int(J.argmin())
                        cells += 1
                        at_cap_side = grid[i] >= CAP - x - 1e-6
                        if 0 < i and not at_cap_side and J[i] < min(J[0], Jarr) - 1e-9:
                            interior += 1
                        S, Sx = law.sf(grid), law.sf(grid + x)
                        Fx = np.where(S > 0, 1 - Sx / np.maximum(S, 1e-300), 1.0)
                        ab = (Fx >= q) & (S > 0)
                        if ab.any():
                            j = int(np.argmax(ab))
                            if grid[j] <= 1.0:
                                thr_early += 1
                                if J[j] > Jarr + 1e-9:
                                    thr_worse += 1
                                    if J[j] - Jarr > worst[0] - worst[1]:
                                        worst = (J[j], Jarr)
        print(f"   {name}: {cells} cells with x > 0 and c > 0; minimum strictly inside (not at the turn's end, not at the cap, not the arrival): {interior}; threshold rule starts within 1 s of the turn's end in {thr_early}, and costs more than arrival in {thr_worse} (worst {worst[0]:.1f} against {worst[1]:.1f} s)")
    # with a drop time, in-flight rent charged: G(0, theta), W = 0, K_io = lamL * min(r, 0.9 + L / beta)
    beta = 21_400.0
    print("   with a drop time (W = 0, K_io = lamL x min(r, 0.9 + L/beta)): best gain of reading at the turn's end")
    for name, law, lam, xs in (("tool", tool, 2e-5, (2, 3, 4, 5, 6)), ("human", human, 1e-6, (3, 4, 5, 6)), ("tool", tool, 1e-4, (2, 4, 6))):
        lamL = lam * L
        c = 1.0
        out = []
        for x in xs:
            r = float(x)
            Kio = lamL * min(r, 0.9 + L / beta)
            th = np.geomspace(r, CAP * 0.999, 6000)
            A = float(law.M(0.0)) - float(law.M(r))  # integral_0^x S
            Sth = law.sf(th)
            B = float(law.M(r)) - law.M(th)  # integral_x^theta S
            G = c * (A - r * Sth) - lamL * B - Kio * Sth
            k = int(G.argmax())
            out.append(f"x={x}: drop at {th[k]:.1f} s gain {G[k]:+.2f}")
        print(f"      {name}, lambda = {lam:g}: " + "; ".join(out))


# ---------------------------------------------------------------- R9
def r9(rows):
    print(f"\n== R9: the {len(rows):,} hints the replay sent ==")
    p10 = np.array([r[0] for r in rows])
    p50 = np.array([r[1] for r in rows])
    p90 = np.array([r[2] for r in rows])
    g = np.array([r[3] for r in rows])
    kind = np.array([r[4] for r in rows])
    med = np.maximum(p50, 0.01)
    lo = np.maximum(np.minimum(p10, med), 0.01)
    hi = np.maximum(p90, med)
    sl = np.clip(np.log(med / lo) / Z90, 0.05, 6.0)
    sh = np.clip(np.log(hi / med) / Z90, 0.05, 6.0)
    print(f"   sigma_high < 0.798 (hazard still rising at the median): {100 * np.mean(sh < 0.798):.1f} % of entries ({100 * np.mean(sh[kind == 'tool'] < 0.798):.1f} % of tool-call entries, {100 * np.mean(sh[kind == 'no_call'] < 0.798):.1f} % of no-call entries); < 0.5: {100 * np.mean(sh < 0.5):.1f} %; < 0.2: {100 * np.mean(sh < 0.2):.1f} %")
    # calibration: where does the true gap fall in its own entry's law
    z = np.log(np.maximum(g, 1e-9) / med) / np.where(g < med, sl, sh)
    u = stats.norm.cdf(z)
    print(f"   true gap below the entry's p10: {100 * np.mean(u < 0.1):.1f} %; inside [p10, p90]: {100 * np.mean((u >= 0.1) & (u <= 0.9)):.1f} %; above the p90: {100 * np.mean(u > 0.9):.1f} %; above the model's p99: {100 * np.mean(u > 0.99):.1f} %")
    dec = np.histogram(u, bins=np.linspace(0, 1, 11))[0] / len(u)
    print("   share of true gaps per decile of the entry's law: " + " ".join(f"{100 * v:.1f}" for v in dec))
    means = np.array([float(TwoPiece(a, b, c_, cap=CAP).M(0.0)) for a, b, c_ in zip(p10, p50, p90)])
    over2 = np.array([float(TwoPiece(a, b, c_, cap=CAP).M(2.0)) for a, b, c_ in zip(p10, p50, p90)])
    print(f"   mean true gap {g.mean():.2f} s against {means.mean():.2f} s for the entries' capped model means; E[(gap - 2 s)^+]: realised {np.maximum(g - 2, 0).mean():.2f} s, model {over2.mean():.2f} s")

    # realised priced cost of start rules; W = 0, c = 1, K_io = lamL * r, prefix on the SSD at the turn's end
    n = len(rows)
    idx = np.arange(n) if "all" in sys.argv else rng.choice(n, size=6000, replace=False)
    sgrid = np.concatenate([[0.0], np.geomspace(1e-3, CAP, 1500)])
    thgrid_base = np.geomspace(1e-2, CAP * 0.999, 1500)

    def realised_fetch(gap, s, x, c, lamL, Kio):
        if gap <= s:
            return c * x + Kio
        return c * max(x - (gap - s), 0.0) + lamL * max(gap - s - x, 0.0) + Kio

    for (x, lam) in ((2.0, 1e-6), (2.0, 2e-5), (4.0, 2e-5), (4.0, 1e-4)):
        lamL = lam * L
        c = 1.0
        r = x
        Kio = lamL * r
        q = lamL / (c + lamL)
        tot = dict(arrival=0.0, exact=0.0, argmin=0.0, threshold=0.0, p50=0.0, p10=0.0, now_drop=0.0, keep=0.0)
        exp_gain_argmin = 0.0
        early = dict(argmin=0, threshold=0, now_drop=0)
        for i in idx:
            law = TwoPiece(p10[i], p50[i], p90[i], cap=CAP)
            gap = g[i]
            grid = np.unique(np.concatenate([sgrid, [max(CAP - x, 0.0)]]))
            Ms = law.M(grid)
            Msx = law.M(grid + x)
            J = c * x - c * Ms + (c + lamL) * Msx
            k = int(J.argmin())
            tot["arrival"] += c * x + Kio
            tot["exact"] += (Kio if gap >= x else c * (x - gap) + Kio)
            if J[k] < c * x - 1e-12:
                s_arg = grid[k]
                early["argmin"] += 1
                exp_gain_argmin += c * x - J[k]
            else:
                s_arg = float("inf")
            tot["argmin"] += realised_fetch(gap, s_arg, x, c, lamL, Kio)
            S, Sx = law.sf(grid), law.sf(grid + x)
            Fx = np.where(S > 0, 1 - Sx / np.maximum(S, 1e-300), 1.0)
            ab = (Fx >= q) & (S > 0)
            s_thr = grid[int(np.argmax(ab))] if ab.any() else float("inf")
            early["threshold"] += int(np.isfinite(s_thr))
            tot["threshold"] += realised_fetch(gap, s_thr, x, c, lamL, Kio)
            tot["p50"] += realised_fetch(gap, max(p50[i] - x, 0.0), x, c, lamL, Kio)
            tot["p10"] += realised_fetch(gap, max(p10[i] - x - 1.0, 0.0), x, c, lamL, Kio)
            # read now with the entry's best drop time (maximise G(0, theta) of R4c), else read at arrival
            th = thgrid_base[thgrid_base >= r]
            A = float(law.M(0.0)) - float(law.M(x))
            Sth = law.sf(th)
            B = float(law.M(x)) - law.M(th)
            G = c * (A - x * Sth) - lamL * B - Kio * Sth
            kk = int(G.argmax())
            if G[kk] > 0:
                early["now_drop"] += 1
                theta = th[kk]
                if gap <= theta:
                    tot["now_drop"] += c * max(x - gap, 0.0) + lamL * max(gap - x, 0.0) + Kio
                else:
                    tot["now_drop"] += lamL * (theta - r) + Kio + c * x + Kio
            else:
                tot["now_drop"] += c * x + Kio
            # a copy that never left memory, kept until the entry's best drop time
            Knet = c * x + Kio
            thk = np.concatenate([[0.0], thgrid_base])
            Gk = Knet * (1.0 - law.sf(thk)) - lamL * (float(law.M(0.0)) - law.M(thk))
            kq = int(Gk.argmax())
            thq = thk[kq]
            tot["keep"] += (lamL * gap if gap <= thq else lamL * thq + c * x + Kio)
        m = len(idx)
        print(f"   x = {x:.0f} s, lambda = {lam:g} (lamL = {lamL:g}): " + ", ".join(f"{k_} {v / m:.2f}" for k_, v in tot.items()))
        print(f"      reads started before arrival of {m}: argmin {early['argmin']}, threshold {early['threshold']}, now-with-drop {early['now_drop']}; argmin's expected gain on its own model {exp_gain_argmin / m:+.2f} s, realised {(tot['arrival'] - tot['argmin']) / m:+.2f} s")


# ---------------------------------------------------------------- R13
def r13():
    print("\n== R13: the engine's 64-point read_start ==")
    spec = importlib.util.spec_from_file_location("ntp_readonly", ENGINE)
    ntp = importlib.util.module_from_spec(spec)
    sys.modules["ntp_readonly"] = ntp
    spec.loader.exec_module(ntp)
    none_count = 0
    last_minus = 0
    diff_sw = 0
    tested = 0
    tested_sw = 0
    for trial in range(20_000):
        p50 = float(np.exp(rng.uniform(math.log(0.05), math.log(400))))
        p10 = p50 * float(np.exp(-rng.uniform(0.0, 3.0)))
        p90 = p50 * float(np.exp(rng.uniform(0.0, 3.5)))
        dist = ntp.GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
        sup = dist.support()
        idle = float(rng.uniform(0.0, 1.0) * sup[-1] * rng.choice([0.0, 0.3, 0.9]))
        rem = ntp.remaining_gaps(support=sup, idle_s=idle)
        if not rem:
            continue
        read_s = float(rng.uniform(0.2, 12.0))
        wait_s = float(rng.uniform(0.0, read_s * 0.99))
        rent = float(np.exp(rng.uniform(math.log(1e-3), math.log(20.0))))
        s0 = ntp.read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent)
        tested += 1
        if s0 is None:
            none_count += 1
        elif abs(s0 - max(0.0, rem[-1] - (read_s - wait_s))) < 1e-9:
            last_minus += 1
        sw = float(rng.choice([0.05, 0.3, 0.7, 2.0]))
        s1 = ntp.read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent, stall_weight=sw)
        s2 = ntp.read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent / sw)
        tested_sw += 1
        if (s1 is None) != (s2 is None) or (s1 is not None and abs(s1 - s2) > 1e-9):
            diff_sw += 1
    print(f"   {tested} random entries with x > 0 and a non-empty remaining support: read_start returned None {none_count} times; returned 'last support point minus x' in {100 * last_minus / tested:.1f} %")
    print(f"   stall_weight = c against rent / c: {diff_sw} differences in {tested_sw} cases")
    tool = ntp.GapDistribution.from_quantiles(p10=0.41, p50=1.69, p90=16.8)
    human = ntp.GapDistribution.from_quantiles(p10=3.1, p50=96.0, p90=1040.0)
    st, sh_ = tool.support(), human.support()
    print(f"   tool entry: last four support points {[round(v, 1) for v in st[-4:]]}; remaining_gaps is empty from idle {st[-4]:.1f} s on: {ntp.remaining_gaps(support=st, idle_s=st[-4]) == ()}")
    print(f"   human entry: support points at the 1,200 s cap: {sum(1 for v in sh_ if v >= 1200.0)} of {len(sh_)}")
    for nm, d_, sup in (("tool", tool, st), ("human", human, sh_)):
        for idle in (0.0, 5.0, 60.0):
            rem = ntp.remaining_gaps(support=sup, idle_s=idle)
            s0 = ntp.read_start(remaining=rem, read_s=3.0, wait_s=0.0, rent_per_s=2e-5 * L) if rem else "no plan"
            print(f"      {nm}, idle {idle:.0f} s, r = 3 s, W = 0, lambda = 2e-5: read_start -> {s0 if isinstance(s0, str) or s0 is None else round(s0 + idle, 1)} (idle age of the planned start)")


if __name__ == "__main__":
    which = sys.argv[1:] or ["r7", "emp", "r8", "r9", "r13"]
    rows = gaps = None
    if "emp" in which or "r9" in which:
        rows, gaps = load_rows()
    for w in which:
        if w == "all":
            continue
        if w == "r7":
            r7()
        elif w == "emp":
            r7_empirical(gaps)
        elif w == "r8":
            gains_table()
        elif w == "r9":
            r9(rows)
        elif w == "r13":
            r13()
