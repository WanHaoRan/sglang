#!/usr/bin/env python3
"""Numeric checks of the retention index of DESIGN.md section 4.1 against the Gittins-index literature.

CPU only.  Run:
  CUDA_VISIBLE_DEVICES= nice -n 10 /home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B index_checks.py > index_checks.out

Symbols.  G = gap (seconds) with survival S(t) = P(G > t); a = idle age; R = G - a given G > a (remaining gap);
F(h|a) = P(R <= h); tau(h|a) = E[min(R, h)]; hazard h(a) = f(a)/S(a); mrl(a) = E[R] (mean remaining gap).
g(a) = sup_{h>0} F(h|a)/tau(h|a)  is the draft's index without the factor K/L  (unit: 1/second).
Literature form (Gittins 1989; Aalto, Ayesta, Righter 2009; Scully & Harchol-Balter 2021, Example V.2 with holding
cost 1): index(a) = sup_{y>a} P(S <= y | S > a) / E[min(S, y) - a | S > a]  -- the same expression with S = G.
"""
import warnings
warnings.simplefilter('ignore')
import numpy as np
from scipy import stats, optimize

np.set_printoptions(linewidth=160, precision=4, suppress=True)

# ----------------------------------------------------------------------------- grid machinery
class Gap:
    """A gap distribution on a fine grid, with an optional mass p_inf at infinity (session ended)."""
    def __init__(self, name, dist, tmax, n=200001, p_inf=0.0):
        self.name, self.p_inf = name, p_inf
        self.t = np.linspace(0.0, tmax, n)
        self.dt = self.t[1] - self.t[0]
        self.S = p_inf + (1 - p_inf) * dist.sf(self.t)          # survival incl. the mass at infinity
        self.f = (1 - p_inf) * dist.pdf(self.t)
        self.haz = self.f / np.maximum(self.S, 1e-300)
        # cumulative integral of S (trapezoid), IS[i] = int_0^{t_i} S
        self.IS = np.concatenate([[0.0], np.cumsum(0.5 * (self.S[1:] + self.S[:-1]) * self.dt)])

    def index(self, i, return_h=False):
        """g(a_i) = sup over grid horizons of F(h|a)/tau(h|a); the h -> 0 limit is the hazard."""
        S, IS = self.S, self.IS
        num = (S[i] - S[i + 1:])                 # P(a < G <= a+h)
        den = (IS[i + 1:] - IS[i])               # int_a^{a+h} S
        ratio = num / np.maximum(den, 1e-300)
        j = int(np.argmax(ratio))
        best = max(ratio[j], self.haz[i])
        if return_h:
            hstar = 0.0 if self.haz[i] >= ratio[j] else self.t[i + 1 + j] - self.t[i]
            return best, hstar
        return best

    def mrl(self, i):
        """E[R | not ended]/..., the mean remaining gap ignoring the tail beyond the grid; inf if p_inf > 0."""
        if self.p_inf > 0:
            return np.inf
        return (self.IS[-1] - self.IS[i]) / self.S[i]

def trunc_normal(mu, sd):
    return stats.truncnorm((0 - mu) / sd, np.inf, loc=mu, scale=sd)

class Mixture:
    def __init__(self, comps, weights):
        self.c, self.w = comps, weights
    def sf(self, t): return sum(w * c.sf(t) for c, w in zip(self.c, self.w))
    def pdf(self, t): return sum(w * c.pdf(t) for c, w in zip(self.c, self.w))

def table_special_cases():
    print("== 1. g(a) = sup_h F(h|a)/tau(h|a) against the hazard h(a) and 1/mrl(a)")
    print("   claim (draft 4.1, and Aalto-Ayesta-Righter 2009 for the same index in the M/G/1 queue):")
    print("   falling hazard -> g = h(a) (sup at h -> 0);  rising hazard -> g = 1/mrl(a) (sup at h -> infinity)")
    gaps = [
        Gap("Weibull k=0.5, scale 3 s (falling hazard)", stats.weibull_min(0.5, scale=3.0), 4000.0, 400001),
        Gap("Lomax shape 1.5, scale 2 s (falling hazard, heavy tail)", stats.lomax(1.5, scale=2.0), 20000.0, 400001),
        Gap("Weibull k=3, scale 20 s (rising hazard)", stats.weibull_min(3.0, scale=20.0), 80.0),
        Gap("Normal(20 s, 2 s) cut at 0 (rising hazard: a sharp hint)", trunc_normal(20.0, 2.0), 40.0),
        Gap("lognormal median 1.69 s, sigma 1.79 (hazard rises then falls)", stats.lognorm(1.79, scale=1.69), 20000.0, 400001),
        Gap("two modes: 0.3 Exp(mean 1 s) + 0.7 Normal(30 s, 3 s)", Mixture([stats.expon(scale=1.0), trunc_normal(30.0, 3.0)], [0.3, 0.7]), 60.0),
    ]
    for G in gaps:
        print(f"   -- {G.name}")
        print("        age a      g(a)     hazard    1/mrl     best horizon h*   g/hazard   g*mrl")
        tmax = G.t[-1]
        ages = [0.0, 0.5, 2.0, 5.0, 15.0, 25.0] if tmax <= 100 else [0.05, 0.5, 2.0, 10.0, 60.0, 300.0]
        for a in ages:
            i = int(round(a / G.dt))
            if G.S[i] < 1e-9:
                continue
            g, hs = G.index(i, return_h=True)
            m = G.mrl(i)
            # mrl on a truncated grid underestimates heavy tails; report it only when the grid tail is negligible
            tail_ok = G.S[-1] * tmax < 1e-3 * (G.IS[-1] - G.IS[i])
            inv = 1.0 / m if tail_ok else float('nan')
            ratio_h = f"{g / G.haz[i]:8.3f}" if G.haz[i] > 1e-9 else "     inf"
            print(f"      {G.t[i]:7.2f}  {g:9.5f} {G.haz[i]:9.5f} {inv:9.5f}   {hs:12.2f}     {ratio_h}  {g * m if tail_ok else float('nan'):7.3f}")
    return gaps

# ----------------------------------------------------------------------------- 2. optimal stopping check
def stopping_check(gaps):
    print("== 2. 'keep while nu(a) >= lambda' against a direct solution of the priced single-session stopping problem")
    print("   problem (K = 1): keeping costs rho per second (rho = lambda L / K) until the session returns; dropping at")
    print("   age a costs P(return | G > a) (pay K when it returns); once dropped the prefix stays out until the turn.")
    print("   dynamic program on the age grid, compared with the index rule keep iff g(a) >= rho, at every grid age.")
    for G in gaps:
        n = len(G.t)
        # coarsen: use every k-th point so the DP is cheap but still fine
        k = max(1, n // 20000)
        idx = np.arange(0, n, k)
        S = G.S[idx]; IS = G.IS[idx]; t = G.t[idx]
        m = len(idx)
        drop = (S - G.p_inf) / S                           # P(return | G > a)
        g = np.array([G.index(i) for i in idx[:-1]] + [np.nan])
        for rho_label, rho in (("low", None), ("mid", None), ("high", None)):
            # choose rho from the index range so the rule is not trivial
            finite = g[:-1][S[:-1] > 1e-6]
            qs = {"low": 0.2, "mid": 0.5, "high": 0.8}[rho_label]
            rho = float(np.quantile(finite, qs))
            V = np.empty(m)
            keep = np.zeros(m, dtype=bool)
            V[-1] = drop[-1] if G.p_inf > 0 or True else 0.0
            # beyond the grid: for a light tail keeping costs about rho * (remaining mean) ~ 0; use min(drop, that)
            tail_keep = rho * 0.0
            V[-1] = min(drop[-1], tail_keep) if S[-1] < 1e-9 else drop[-1]
            for i in range(m - 2, -1, -1):
                if S[i] < 1e-300:
                    V[i] = 0.0
                    continue
                hold = rho * (IS[i + 1] - IS[i]) / S[i]
                cont = hold + (S[i + 1] / S[i]) * V[i + 1]
                if cont < drop[i]:
                    V[i], keep[i] = cont, True
                else:
                    V[i] = drop[i]
            live = S[:-1] > 1e-6
            rule = g[:-1] >= rho
            dis = np.sum(keep[:-1][live] != rule[live])
            # disagreements within a relative band of the threshold are discretisation
            band = np.abs(g[:-1] / rho - 1) < 0.02
            dis_out = np.sum((keep[:-1] != rule) & live & ~band)
            print(f"   {G.name[:52]:52s} rho={rho:9.5f} ({rho_label}): ages with DP != index rule: {dis:5d} of {int(live.sum())}; "
                  f"outside a 2 % band around g = rho: {dis_out}")

# ----------------------------------------------------------------------------- 3. sharp hint: keep / drop / drop and re-read
def three_plans():
    print("== 3. A sharp (rising-hazard) key: keep, drop and read at arrival, or drop and read back in time?")
    print("   gap G ~ Normal(20 s, 2 s); prefix L = 25,500 tokens; read time r = 0.9 + L/21,400 = 2.09 s; no queue (W = 0,")
    print("   x = r); c = 1 (idle GPU); device price kappa as listed; holding the whole prefix costs lamL = lambda*L per second.")
    mu, sd, L, beta = 20.0, 2.0, 25500.0, 21400.0
    r = 0.9 + L / beta
    D = trunc_normal(mu, sd)
    t = np.linspace(0, 60, 600001)
    f = D.pdf(t); dt = t[1] - t[0]
    def JF(s, c, lamL):
        # read started at s (or at arrival if the turn comes first), lands at l = min(s, A) + r, turn due at A (W = 0)
        late = np.where(t <= s, r, np.maximum(0.0, r - (t - s)))
        early = np.where(t <= s, 0.0, np.maximum(0.0, t - s - r))
        return float(np.sum((c * late + lamL * early) * f) * dt)
    print("     lamL   kappa |  A keep      B drop, read   C drop, timed read (start s*, stall, early hold)   | best | index rule nu(0)>=lambda says")
    for lamL in (0.01, 0.05, 0.2, 1.0):
        for kappa in (0.0, 0.5):
            c = 1.0
            dev = kappa * L / beta
            A = lamL * float(np.sum(t * f) * dt)
            K_read = c * r + dev + lamL * r
            B = K_read
            res = optimize.minimize_scalar(lambda s: JF(s, c, lamL), bounds=(0.0, 40.0), method="bounded", options={"xatol": 1e-4})
            s_star = res.x
            C = res.fun + dev + lamL * r
            late = float(np.sum(np.where(t <= s_star, r, np.maximum(0.0, r - (t - s_star))) * f) * dt)
            early = float(np.sum(np.where(t <= s_star, 0.0, np.maximum(0.0, t - s_star - r)) * f) * dt)
            best = min((A, "A"), (B, "B"), (C, "C"))[1]
            # the draft's index at age 0 for a rising hazard: nu = K_read / (L E[G]); keep iff nu >= lambda  <=>  K_read >= lamL E[G]
            keep = K_read >= lamL * mu
            print(f"   {lamL:6.2f} {kappa:6.2f} | {A:7.3f}    {B:7.3f}        {C:7.3f}  (s*={s_star:5.2f} s, stall {late:5.3f} s, early {early:5.2f} s)"
                  f"          |  {best}   | {'keep (A)' if keep else 'drop (B)'}")
    print("   Reading: with a sharp key and a cheap device, plan C costs less than both plans the index compares, so the")
    print("   index rule is the optimum only of the problem in which a dropped prefix is not read back before its turn.")

# ----------------------------------------------------------------------------- 4. TTL: full-TTL charge vs charge until return
def ttl_compare():
    print("== 4. Time-to-live for a falling-hazard key: charging the memory for the whole TTL (Continuum's Eq. 1) against")
    print("   charging it until the session returns (the draft's tau(h|a)).  gap: lognormal, median 1.69 s, p90 16.8 s.")
    D = stats.lognorm(1.79, scale=1.69)
    t = np.linspace(0, 3000, 3000001)
    F = D.cdf(t); Sv = 1 - F
    dt = t[1] - t[0]
    IS = np.concatenate([[0.0], np.cumsum(0.5 * (Sv[1:] + Sv[:-1]) * dt)])
    print("     K (s)   lamL |  TTL full-charge   TTL until-return |  priced cost at each TTL (K P(miss) + lamL E[min(G,TTL)])   | loss of the full-charge TTL")
    for K in (2.2, 10.0):
        for lamL in (0.02, 0.1, 0.5):
            obj_full = K * F - lamL * t
            obj_true = K * F - lamL * IS
            i1, i2 = int(np.argmax(obj_full)), int(np.argmax(obj_true))
            cost = lambda i: K * (1 - F[i]) + lamL * IS[i]
            print(f"   {K:6.1f} {lamL:6.2f} |   {t[i1]:9.2f} s        {t[i2]:9.2f} s    |      {cost(i1):7.4f} s           {cost(i2):7.4f} s"
                  f"                    | {cost(i1) - cost(i2):7.4f} s ({100 * (cost(i1) / cost(i2) - 1):5.1f} %)")
    print("   first-order conditions: full charge  K f(h) = lamL  (density);  until return  K f(h)/S(h) = lamL  (hazard).")

if __name__ == "__main__":
    gaps = table_special_cases()
    stopping_check(gaps)
    three_plans()
    ttl_compare()
