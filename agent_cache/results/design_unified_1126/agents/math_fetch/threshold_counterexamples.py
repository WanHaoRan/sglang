"""Check 1a: when is the threshold rule "start when F(x|s) >= q" optimal?

Exact criterion (proved in the notes): with tau = first s with F(x|s) >= q, the rule is optimal iff
    I(s) = integral_tau^s S(u) [F(x|u) - q] du >= 0   for every s > tau
(J_F(s) - J_F(tau) = (c + lamL) I(s)). A single up-crossing is sufficient, not necessary.

(a) two sharp modes: F(x|s) crosses q three times, and the first crossing is still the optimum.
(b) the same law at a higher q: three crossings, and the first crossing is NOT the optimum.
(c) a tool that fails fast or runs long: the threshold fires at the turn's end and pins through the long mode.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # python -I drops the script's folder

import numpy as np

from fetchlib import J_F, F_cond, Phi, Phibar, phi


class NormalMix:
    """Mixture of normals far from zero (truncation at zero is negligible)."""

    def __init__(self, w, mu, sd):
        self.w, self.mu, self.sd, self.cap = np.asarray(w, float), np.asarray(mu, float), np.asarray(sd, float), None

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        z = (t[..., None] - self.mu) / self.sd
        return (self.w * Phibar(z)).sum(-1)

    def M(self, t):
        t = np.asarray(t, dtype=float)
        z = (t[..., None] - self.mu) / self.sd
        return (self.w * self.sd * (phi(z) - z * Phibar(z))).sum(-1)


def report(name, law, x, c, lamL, hi):
    q = lamL / (c + lamL)
    g = np.linspace(0.0, hi, 200001)
    fx = F_cond(law, x, g)
    j = J_F(law, g, x, c, lamL)
    above = fx >= q
    cross = np.flatnonzero(np.diff(above.astype(int)) != 0)
    ups = [g[k + 1] for k in cross if not above[k]]
    downs = [g[k + 1] for k in cross if above[k]]
    tau = g[np.flatnonzero(above)[0]] if above.any() else None
    k = int(np.argmin(j))
    print(f"   {name}: x = {x}, c = {c}, lamL = {lamL}, q = {q:.3f}")
    print(f"      F(x|s) crosses q upward at s = {[round(float(u), 2) for u in ups]}{' (and is above q at s = 0)' if above[0] else ''}, downward at s = {[round(float(u), 2) for u in downs]}")
    if tau is not None:
        S = law.sf(g)
        integ = np.concatenate([[0.0], np.cumsum(0.5 * (S[1:] * (fx[1:] - q) + S[:-1] * (fx[:-1] - q)) * np.diff(g))])
        i0 = int(np.flatnonzero(above)[0])
        I = integ - integ[i0]
        print(f"      threshold start tau = {tau:.2f} s: J = {float(j[i0]):.4f};  global minimum at s = {g[k]:.2f} s: J = {float(j[k]):.4f};  arrival: {c * x:.4f}")
        print(f"      min over s > tau of I(s) = {float(I[i0:].min()):+.5f}  ->  the threshold rule is {'optimal' if I[i0:].min() >= -1e-9 else 'NOT optimal'}"
              f" (J(tau) - min J = {float(j[i0] - j[k]):.4f})")


if __name__ == "__main__":
    print("== two sharp modes, 10 +- 0.5 s and 14 +- 0.5 s, equal weights ==")
    two = NormalMix([0.5, 0.5], [10.0, 14.0], [0.5, 0.5])
    report("(a)", two, 3.0, 1.0, 0.4286, 30.0)       # q = 0.30
    report("(b)", two, 3.0, 1.0, 0.8182, 30.0)       # q = 0.45
    print("== a tool that fails in about 1 s (30 %) or runs about 40 s (70 %) ==")
    ff = NormalMix([0.3, 0.7], [1.0, 40.0], [0.2, 3.0])
    report("(c)", ff, 3.0, 1.0, 0.25, 80.0)          # q = 0.20
    report("(c')", ff, 3.0, 1.0, 0.025, 80.0)        # q = 0.024: cheap memory
