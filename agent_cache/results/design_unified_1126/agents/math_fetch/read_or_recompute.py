"""Check 3: read or recompute (DESIGN.md section 2), and the hybrid restore.

Priced cost of restoring a prefix of L tokens for a turn that has arrived (time 0 = arrival):
  read:       K_read = c (r - W)^+ + kappa L / beta + lam L r_m,   r = V + lags + L / beta
  recompute:  K_rec  = mu * T_rec(L),   T_rec = gamma L  (the turn's own extra prefill is the "1" in mu = 1 + turns delayed)
Read iff K_read < K_rec.

Closed form (linear recompute, lam = 0), with D0 = V + lags - W:
  D0 >= 0:  read iff L > L_x = c D0 / (mu gamma - (c + kappa) / beta)     if mu gamma > (c + kappa) / beta, else never
  D0 <  0:  read for every L if mu gamma >= (c + kappa) / beta [and mu gamma beta > kappa];
            else read iff L < c (-D0) / ((c + kappa) / beta - mu gamma)
Per token the read is cheaper once mu > mu_x = (c + kappa) / (gamma beta).

Constants (campaign 34, FA3, Qwen3-30B-A3B bf16): beta = 21.4 K tokens/s, lags 0.9 s,
gamma 0.037-0.044 ms/token, 27-35 ms per 8,192-token prefill chunk.
"""

import math

import numpy as np
from scipy import optimize

BETA = 21_400.0
LAGS = 0.9
CHUNK = 8192


def k_read(L, c, W, kappa=0.0, lam=0.0, V=0.0, lags=LAGS, beta=BETA):
    r = V + lags + L / beta
    r_m = lags + L / beta            # pages are taken at dispatch, not while parked in the device's line
    return c * max(r - W, 0.0) + kappa * L / beta + lam * L * r_m


def t_rec(L, gamma, batch=0.0, rho=0.0):
    return gamma * L + batch * math.ceil(L / CHUNK) + rho * L * L


def crossover_linear(mu, gamma, c, W, kappa=0.0, V=0.0, lags=LAGS, beta=BETA):
    """Returns a text verdict and the crossover length (None when there is none)."""
    D0 = V + lags - W
    slope = mu * gamma - (c + kappa) / beta
    if D0 >= 0:
        if slope <= 0:
            return "recompute at every length", None
        Lx = c * D0 / slope
        return ("read at every length" if Lx <= 0 else f"read above {Lx:,.0f} tokens"), Lx
    if slope >= 0:
        return ("read at every length" if mu * gamma * beta > kappa else "recompute at every length"), None
    Lx = c * (-D0) / (-slope)
    return f"read below {Lx:,.0f} tokens", Lx


def crossover_numeric(mu, gamma, c, W, kappa, lam, batch, rho, V=0.0, lags=LAGS, Lmax=65_536):
    """Sign changes of K_rec - K_read on [64, Lmax] for the full model (batch overhead, prefix term, rent)."""
    Ls = np.arange(64, Lmax + 1, 64)
    d = np.array([mu * t_rec(L, gamma, batch, rho) - k_read(L, c, W, kappa, lam, V, lags) for L in Ls])
    sg = np.sign(d)
    ch = np.flatnonzero(np.diff(sg) != 0)
    first = "read" if d[0] > 0 else "recompute"
    if len(ch) == 0:
        return f"{first} at every length up to {Lmax:,}"
    pts = ", ".join(f"{Ls[k + 1]:,}" for k in ch)
    return f"{first} below, switching at {pts} tokens"


def main():
    print("== per-token threshold mu_x = (c + kappa) / (gamma beta): below it recompute is cheaper per token ==")
    for gamma in [0.037e-3, 0.040e-3, 0.044e-3]:
        print(f"   gamma = {gamma * 1e3:.3f} ms/token: gamma*beta = {gamma * BETA:.3f}; mu_x = {1 / (gamma * BETA):.3f} (c = 1, kappa = 0),"
              f" {0.3 / (gamma * BETA):.3f} (c = 0.3), 0 (c = 0)")
    print()
    print("== K_read and K_rec for L = 25,000 tokens (seconds of summed TTFT); kappa = 0, lambda = 0, no device line ==")
    L = 25_000
    print("   gamma(ms)  mu    K_rec   | c=1: K_read at W=0, 2, 5        | c=0.3: W=0, 2, 5          | c=0")
    for gamma in [0.037e-3, 0.044e-3]:
        for mu in [1, 2, 5, 128]:
            kr = mu * t_rec(L, gamma)
            cells = []
            for c in [1.0, 0.3, 0.0]:
                cells.append("  ".join(f"{k_read(L, c, W):6.3f}" for W in [0.0, 2.0, 5.0]))
            print(f"   {gamma * 1e3:5.3f}     {mu:4d}  {kr:7.2f}  | {cells[0]}        | {cells[1]}     | {cells[2].split()[0]}")
    print()
    print("== crossover length, linear model (kappa = 0, lambda = 0, V = 0, lags 0.9 s) ==")
    print("   gamma(ms)  mu    c     W=0                               W=2                               W=5")
    for gamma in [0.037e-3, 0.044e-3]:
        for mu in [1, 2, 5, 128]:
            for c in [1.0, 0.3, 0.0]:
                cells = [crossover_linear(mu, gamma, c, W)[0] for W in [0.0, 2.0, 5.0]]
                print(f"   {gamma * 1e3:5.3f}     {mu:4d}  {c:3.1f}   {cells[0]:32s}  {cells[1]:32s}  {cells[2]:32s}")
    print()
    print("== the same with the measured extras: 31 ms per 8,192-token chunk, prefix term 1.25e-10 s/token^2 (loaded fit), up to 65,536 tokens ==")
    print("   gamma(ms)  mu    c     W=0")
    for gamma in [0.0374e-3]:
        for mu in [1, 2, 5, 128]:
            for c in [1.0, 0.3]:
                print(f"   {gamma * 1e3:6.4f}    {mu:4d}  {c:3.1f}   {crossover_numeric(mu, gamma, c, 0.0, 0.0, 0.0, 0.031, 1.25e-10)}")
    print("   with the rent of the read's pages in flight, lambda = 2e-5 per token-second (kappa = 0):")
    for mu in [1, 2, 5, 128]:
        for c in [1.0, 0.3, 0.0]:
            print(f"   {0.0374:6.4f}    {mu:4d}  {c:3.1f}   {crossover_numeric(mu, 0.0374e-3, c, 0.0, 0.0, 2e-5, 0.031, 1.25e-10)}")
    print()
    print("== mu = 1, c = 1: the two restore times (s), loaded constants ==")
    print("   L        read = 0.9 + L/beta   recompute (0.0374 ms + chunk + prefix term)   recompute at 0.037 / 0.044 ms flat")
    for L in [5_000, 10_000, 25_000, 46_500, 65_536]:
        print(f"   {L:6,d}   {LAGS + L / BETA:6.3f}                {t_rec(L, 0.0374e-3, 0.031, 1.25e-10):6.3f}"
              f"                                        {t_rec(L, 0.037e-3):6.3f} / {t_rec(L, 0.044e-3):6.3f}")
    # where would the loaded fit cross if contexts were longer
    f = lambda L: t_rec(L, 0.0374e-3, 0.0, 1.25e-10) + 0.031 * L / CHUNK - (LAGS + L / BETA)
    print(f"   the loaded fit crosses at {optimize.brentq(f, 1e3, 1e7):,.0f} tokens (beyond the 65,536 limit)")
    # idle-box constants (facts_calib: inferred; one request above 30 K)
    fi = lambda L: (-0.012 + 2.31e-5 * L + 3.58e-10 * L * L) - (0.044 + 0.05 + 2.477e-3 * L / 64)
    print(f"   idle-box fit (cold first requests; read 43.5 ms + 2.477 ms/page + 0.05 s lags): crosses at {optimize.brentq(fi, 1e3, 1e7):,.0f} tokens"
          " -> on an idle box the read wins above about 50 K (inferred from one request above 30 K)")
    print()
    print("== device price at which recompute would win at saturation (c = 0, W hides the read): kappa > mu gamma beta - lambda beta r_m ==")
    for mu in [94, 128]:
        for gamma in [0.037e-3, 0.044e-3]:
            for lam in [0.0, 2e-5]:
                r_m = LAGS + 25_000 / BETA
                print(f"   mu = {mu}, gamma = {gamma * 1e3:.3f} ms, lambda = {lam:g}: kappa > {mu * gamma * BETA - lam * BETA * r_m:6.1f} TTFT-seconds per device-second")
    print()
    hybrid()


def hybrid():
    """Recompute the head [0, a) on the GPU while the SSD reads the tail [a, L).
    K_hyb(a) = mu gamma a + c (r_tail(a) - W - gamma a)^+ + kappa (L - a) / beta,  r_tail(a) = V + lags + (L - a) / beta, a < L;
    K_hyb(L) = mu gamma L (no read at all)."""
    print("== hybrid restore: recompute the head while the SSD reads the tail ==")

    def K(a, L, mu, c, gamma, W=0.0, kappa=0.0, lags=LAGS):
        if a >= L:
            return mu * gamma * L
        r_tail = lags + (L - a) / BETA
        return mu * gamma * a + c * max(r_tail - W - gamma * a, 0.0) + kappa * (L - a) / BETA

    print("   condition for the equal-finish split: kappa/beta <= mu gamma < c (gamma + 1/beta) + kappa/beta,")
    for gamma in [0.037e-3, 0.044e-3]:
        print(f"   i.e. mu < c (1 + 1/(gamma beta)) = {1 + 1 / (gamma * BETA):.2f} c at gamma = {gamma * 1e3:.3f} ms, kappa = 0")
    worst = 0.0
    rng = np.random.default_rng(3)
    for _ in range(3000):
        L = int(rng.integers(2_000, 65_000))
        mu = float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 5.0]))
        c = float(rng.choice([1.0, 0.6, 0.3]))
        gamma = float(rng.uniform(0.037e-3, 0.044e-3))
        W = float(rng.choice([0.0, 0.5, 2.0]))
        kappa = float(rng.choice([0.0, 0.5, 2.0]))
        lags = float(rng.choice([0.1, 0.9]))
        grid = np.concatenate([np.linspace(0, L, 6001), [L]])
        vals = np.array([K(a, L, mu, c, gamma, W, kappa, lags) for a in grid])
        k = int(np.argmin(vals))
        # the rule
        a_eq = (lags - W + L / BETA) / (gamma + 1 / BETA)
        if mu * gamma < kappa / BETA:
            a_rule = L
        elif mu * gamma < c * (gamma + 1 / BETA) + kappa / BETA:
            a_rule = min(max(a_eq, 0.0), L)
        else:
            a_rule = 0.0
        best_rule = min(K(a_rule, L, mu, c, gamma, W, kappa, lags), K(L, L, mu, c, gamma, W, kappa, lags))
        worst = max(worst, best_rule - vals[k])
    print(f"   3,000 random cases: the rule (corner or equal finish, or no read at all) is within {worst:.2e} s of a 6,001-point search")
    print("   mu = 1, c = 1, W = 0, kappa = 0: restore time (s)")
    print("   lags  gamma(ms)  L        read    recompute  hybrid   head recomputed   gain over the better of the two")
    for lags in [0.9, 0.1]:
        for gamma in [0.037e-3, 0.044e-3]:
            for L in [10_000, 25_000, 46_500, 65_536]:
                a_eq = min((lags + L / BETA) / (gamma + 1 / BETA), L)
                t_h = gamma * a_eq
                t_r, t_c = lags + L / BETA, gamma * L
                print(f"   {lags:3.1f}   {gamma * 1e3:5.3f}     {L:6,d}   {t_r:6.3f}  {t_c:6.3f}     {t_h:6.3f}   {a_eq:8,.0f} ({100 * a_eq / L:3.0f} %)   {min(t_r, t_c) - t_h:6.3f}")
    print(f"   with no lags the hybrid restores at 1/gamma + beta = {1 / 0.040e-3 + BETA:,.0f} tokens/s"
          f" against {1 / 0.040e-3:,.0f} (recompute) and {BETA:,.0f} (read)")


if __name__ == "__main__":
    main()
