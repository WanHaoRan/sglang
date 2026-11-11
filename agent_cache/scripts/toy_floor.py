"""Toy check of the "Pricing the Room" rules against the weak-duality floor. Made-up keys, not trace data.

One session, one idle gap, a price lam on host room (seconds of stall per token-second):
  - the history is in the room when the turn ends and can be dropped at any time (it is on SSD);
  - a read of r = b L / beta seconds brings it back, and holds its tokens in the room from launch;
  - a turn that arrives with the history present stalls 0; mid-read, the rest of the read; absent, r.
Cls.solve is the exact dynamic program over idle time for that priced problem.

Many sessions share one hard room. Each cycles: turn (tau seconds, off the room) -> idle gap -> arrival.
No queue, no SSD contention, open-loop arrivals. bound() is the floor: per-session optima minus the room's rent.

Usage:
  python3 toy_floor.py single
  python3 toy_floor.py sweep --per 9000 --lams 2e-6,2.5e-6,3e-6,3.5e-6,4e-6,5e-6,7e-6,1e-5 --mix 2.5e-6,3e-6,0.3,0.45,0.55 --seeds 3
  python3 toy_floor.py sweep --per 6000 --lams 2e-6,5e-6,1e-5,2e-5,5e-5,1e-4,2e-4 --seeds 2 --variants
A sweep takes about 5 minutes on 22 cores and writes ../results/toy_floor/sweep_<per>.json.
"""
import argparse
import json
import pathlib
import time
from multiprocessing import Pool

import numpy as np
from scipy.special import ndtr

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "toy_floor"

# ===== One session at a given price =====

B_TOK, BETA = 98304, 2.11e9
C_READ = B_TOK / BETA  # seconds of SSD time per token, about 47 us


def LN(med, s):
    return lambda x: ndtr((np.log(np.maximum(x, 1e-12)) - np.log(med)) / s)


KEYS = {
    "sharp": LN(20.0, 0.10),  # a test run, about 20 s +- 2 s
    "pip": lambda x: 0.2 * LN(0.85, 0.5)(x) + 0.8 * LN(26.6, 0.6)(x),  # pip-like, two modes
    "editor": LN(2.0, 0.5),  # a quick file edit
    "human": LN(300.0, 1.0),  # waits on a person
}
MEAN_GAP = {
    "sharp": 20.0 * np.exp(0.1**2 / 2),
    "pip": 0.2 * 0.85 * np.exp(0.5**2 / 2) + 0.8 * 26.6 * np.exp(0.6**2 / 2),
    "editor": 2.0 * np.exp(0.5**2 / 2),
    "human": 300.0 * np.exp(1.0**2 / 2),
}


class Cls:
    """One (key, L) class on a uniform idle-time grid."""

    def __init__(self, key, L, dt=0.05, tmax=1200.0):
        self.key, self.L, self.dt = key, int(L), dt
        self.M = int(round(tmax / dt))
        t = np.arange(self.M + 1) * dt
        cdf = KEYS[key](t)
        cdf[0] = 0.0
        self.t = t
        self.S = 1.0 - cdf  # survival at the start of step j
        self.f = np.diff(cdf)  # mass arriving in step j
        self.nr = max(1, int(round(C_READ * self.L / dt)))
        self.r = self.nr * dt  # read time on the grid
        self.mean_gap = MEAN_GAP[key]
        # prefix sums for the stall of a read launched at step j
        self.Fc = np.concatenate([[0.0], np.cumsum(self.f)])
        self.M1 = np.concatenate([[0.0], np.cumsum(self.f * np.arange(self.M))])
        Sbar = 0.5 * (self.S[1:] + self.S[:-1])
        self.IS = np.concatenate([[0.0], np.cumsum(Sbar) * dt])  # integral of survival up to t_j

    # chance of return within x seconds given idle t (scalar helpers)
    def F(self, x, t):
        Phi = KEYS[self.key]
        return float((Phi(t + x) - Phi(t)) / (1 - Phi(t)))

    def hazard(self):
        return self.f / (np.maximum(self.S[:-1], 1e-300) * self.dt)

    def rate_index(self, n_h=80):
        """max over horizons H of  c F(H|t) / E[min(G - t, H) | G > t]  at every grid time.

        Seconds of stall saved per token-second of room by holding the history from t."""
        M = self.M
        ks = np.unique(np.round(np.logspace(0, np.log10(M), n_h)).astype(int))
        j = np.arange(M)
        best = np.zeros(M)
        arg = np.zeros(M)
        for k in ks:
            e = np.minimum(j + k, M)
            num = self.S[j] - self.S[e]
            den = self.IS[e] - self.IS[j]
            v = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
            upd = v > best
            best[upd] = v[upd]
            arg[upd] = (e - j)[upd] * self.dt
        return C_READ * best, arg

    def solve(self, lam):
        """Exact optimum of the priced one-session problem. Returns a dict of tables."""
        L, dt, M, nr, r = self.L, self.dt, self.M, self.nr, self.r
        S, f = self.S, self.f
        rent_read = lam * L * r
        absent_arrival = r + rent_read  # read launched at arrival: stall r, room for r
        p = f / np.maximum(S[:-1], 1e-300)  # chance of arrival in step j given alive
        # stall part of a read launched at step j, given alive at j
        j = np.arange(M + 1)
        e = np.minimum(j + nr, M)
        stall_sum = dt * ((j + nr - 0.5) * (self.Fc[e] - self.Fc[j]) - (self.M1[e] - self.M1[j]))
        Va = np.empty(M + 1)
        Vp = np.empty(M + 1)
        Sa = np.empty(M + 1)  # expected stall under the optimal policy, absent / present
        Sp = np.empty(M + 1)
        start = np.zeros(M + 1, bool)
        drop = np.zeros(M + 1, bool)
        Va[M] = Vp[M] = absent_arrival
        Sa[M] = Sp[M] = r
        drop[M] = True
        hold_rent = lam * L * dt
        for jj in range(M - 1, -1, -1):
            pj = p[jj]
            wait = pj * absent_arrival + (1 - pj) * Va[jj + 1]
            wait_s = pj * r + (1 - pj) * Sa[jj + 1]
            if jj + nr <= M and S[jj] > 0:
                surv = S[jj + nr] / S[jj]
                rd = rent_read + stall_sum[jj] / S[jj] + surv * Vp[jj + nr]
                rd_s = stall_sum[jj] / S[jj] + surv * Sp[jj + nr]
            else:
                rd, rd_s = np.inf, np.inf
            if rd < wait - 1e-12:
                Va[jj], Sa[jj], start[jj] = rd, rd_s, True
            else:
                Va[jj], Sa[jj] = wait, wait_s
            hold = hold_rent * (1 - pj / 2) + (1 - pj) * Vp[jj + 1]
            hold_s = (1 - pj) * Sp[jj + 1]
            if Va[jj] <= hold + 1e-12:
                Vp[jj], Sp[jj], drop[jj] = Va[jj], Sa[jj], True
            else:
                Vp[jj], Sp[jj] = hold, hold_s
        return dict(V=Vp[0], stall=Sp[0], occ=(Vp[0] - Sp[0]) / lam if lam > 0 else np.nan,
                    Va=Va, Vp=Vp, start=start, drop=drop)

    def single_read(self, lam, sigma):
        """Cost of: drop at once, start one read at idle time sigma (None = at arrival), hold until arrival."""
        L, dt, M, nr, r = self.L, self.dt, self.M, self.nr, self.r
        if sigma is None:
            return r, 0.0
        js = int(round(sigma / dt))
        m = np.arange(M)
        tm = (m + 0.5) * dt
        stall = np.where(m < js, r, np.maximum((js + nr) * dt - tm, 0.0))
        held = np.where(m >= js + nr, tm - (js + nr) * dt, 0.0)
        st = float(stall @ self.f) + self.S[M] * 0.0
        pn = float(held @ self.f) + self.S[M] * max(M * dt - (js + nr) * dt, 0.0)
        return st, pn


# ===== Many sessions, one hard room =====

ABSENT, PRESENT, READING = 0, 1, 2
IDLE, WAIT, TURN = 0, 1, 2

PATTERN_KEYS = ["sharp"] * 5 + ["pip"] * 5 + ["editor"] * 4 + ["human"] * 2
PATTERN_L = [16384, 32768, 65536]


def population(N):
    """The same mix at every N (N a multiple of 48 repeats it exactly)."""
    keys = [PATTERN_KEYS[i % 16] for i in range(N)]
    Ls = [PATTERN_L[(i // 16 + i) % 3] for i in range(N)]
    return keys, Ls


def sample_gap(key, rng, n):
    if key == "sharp":
        return 20.0 * np.exp(0.10 * rng.standard_normal(n))
    if key == "editor":
        return 2.0 * np.exp(0.5 * rng.standard_normal(n))
    if key == "human":
        return 300.0 * np.exp(1.0 * rng.standard_normal(n))
    short = rng.random(n) < 0.2
    z = rng.standard_normal(n)
    return np.where(short, 0.85 * np.exp(0.5 * z), 26.6 * np.exp(0.6 * z))


class Tables:
    """Plan and score tables for every (key, L) class at one price."""

    def __init__(self, dt=0.05, tmax=2400.0):
        self.dt, self.tmax = dt, tmax
        self.cls = {}
        for k in ("sharp", "pip", "editor", "human"):
            for L in PATTERN_L:
                self.cls[(k, L)] = Cls(k, L, dt=dt, tmax=tmax)
        self.names = list(self.cls)
        self.M = next(iter(self.cls.values())).M
        self._rate = {n: c.rate_index()[0] for n, c in self.cls.items()}
        t = np.arange(self.M) * dt
        self._f30 = {}
        for n, c in self.cls.items():
            e = np.minimum(np.arange(self.M) + int(round(30.0 / dt)), self.M)
            self._f30[n] = C_READ * (c.S[:-1] - c.S[e]) / np.maximum(c.S[:-1], 1e-300)
        self._sol = {}

    def solve(self, lam):
        if lam not in self._sol:
            self._sol[lam] = {n: c.solve(lam) for n, c in self.cls.items()}
        return self._sol[lam]

    def V(self, lam):
        return {n: s["V"] for n, s in self.solve(lam).items()}

    def build(self, policy, lam):
        """Arrays [class, idle step]: start, drop, score."""
        C, M = len(self.names), self.M
        start = np.zeros((C, M + 1), bool)
        drop = np.zeros((C, M + 1), bool)
        score = np.zeros((C, M + 1))
        t = np.arange(M + 1) * self.dt
        for ci, n in enumerate(self.names):
            c = self.cls[n]
            if policy["plan"] == "priced":
                s = self.solve(lam)[n]
                start[ci], drop[ci] = s["start"], s["drop"]
            elif policy["plan"] == "none":  # keep everything, read at arrival
                pass
            elif policy["plan"] == "dropall":  # keep nothing, read at arrival
                drop[ci, :] = True
            kind = policy["score"]
            if kind == "lru":
                score[ci] = -t
            elif kind == "rate":
                score[ci, :M] = self._rate[n]
            elif kind == "psi":  # rate, counting the room a re-read would take
                score[ci, :M] = self._rate[n] * (1 + lam * c.L)
            elif kind == "f30":
                score[ci, :M] = self._f30[n]
            elif kind == "adv":
                s = self.solve(lam)[n]
                score[ci] = (s["Va"] - s["Vp"]) / c.L
            else:
                raise ValueError(kind)
            drop[ci, M] = True
        return start, drop, score


def bound(tables, keys, Ls, R, tau, lams):
    """Weak-duality floor on stall per second, maximised over the prices given. Returns (per turn, lam*)."""
    rate = 0.0
    cyc = {}
    for k, L in zip(keys, Ls):
        cyc[(k, L)] = cyc.get((k, L), 0) + 1
    best = (-np.inf, None)
    turn_rate = sum(n / (tau + tables.cls[c].mean_gap) for c, n in cyc.items())
    out = []
    for lam in lams:
        V = tables.V(lam)
        b = sum(n * V[c] / (tau + tables.cls[c].mean_gap) for c, n in cyc.items()) - lam * R
        out.append((lam, b / turn_rate))
        if b > best[0]:
            best = (b, lam)
    return best[0] / turn_rate, best[1], out


def run(N, R, policy, lam, tables, seed=0, T=3000.0, warm=300.0, tau=4.0):
    dt = tables.dt
    rng = np.random.default_rng(seed)
    keys, Ls = population(N)
    cls_id = np.array([tables.names.index((k, L)) for k, L in zip(keys, Ls)])
    L = np.array(Ls, float)
    rd = np.array([tables.cls[(k, l)].r for k, l in zip(keys, Ls)])
    mix = policy.get("mix")  # (lam_lo, lam_hi, share of gaps planned at lam_lo): the floor's optimum mixes two plans
    if mix:
        s0, d0, score_tab = tables.build(policy, mix[0])
        s1, d1, _ = tables.build(policy, mix[1])
        start_tab, drop_tab = np.stack([s0, s1]), np.stack([d0, d1])
    else:
        s0, d0, score_tab = tables.build(policy, lam)
        start_tab, drop_tab = np.stack([s0, s0]), np.stack([d0, d0])
    which = np.zeros(N, int)
    M = tables.M
    soft = policy.get("pins", "soft") == "soft"
    compete = policy.get("arrivals") == "compete"

    phase = np.full(N, TURN)
    busy = rng.random(N) * tau  # staggered first turn ends
    e = np.zeros(N)
    A = np.full(N, np.inf)
    loc = np.full(N, ABSENT)
    land = np.full(N, np.inf)
    arrived = np.zeros(N, bool)
    staged = np.zeros(N, bool)  # in the room through an early read
    used = 0.0
    pending = []  # arrived turns waiting for room, FIFO

    n_turn = 0
    stall_sum = 0.0
    room_ts = 0.0
    stats = dict(blocked=0, evict=0, early=0, early_denied=0, keep_denied=0, overrun=0, bumped=0)
    per_key = {k: [0, 0.0] for k in KEYS}

    def idle_step(i, t):
        return min(int((t - e[i]) / dt + 1e-9), M)

    def score_of(i, t):
        return score_tab[cls_id[i], idle_step(i, t)]

    def release(i):
        nonlocal used
        used -= L[i]
        loc[i] = ABSENT
        land[i] = np.inf
        staged[i] = False

    def make_room(need, req_score, t, allow_staged):
        """Evict lowest-score occupants scoring below req_score until `need` fits. All or nothing."""
        nonlocal used
        if used + need <= R + 1e-6:
            return True
        cand = np.flatnonzero((loc != ABSENT) & ~arrived & (phase == IDLE))
        if not allow_staged:
            cand = cand[~staged[cand]]
        if cand.size == 0:
            return False
        js = np.minimum(((t - e[cand]) / dt + 1e-9).astype(int), M)
        sc = score_tab[cls_id[cand], js]
        ok = sc < req_score
        cand, sc = cand[ok], sc[ok]
        order = np.argsort(sc, kind="stable")
        freed, take = 0.0, []
        for o in order:
            take.append(cand[o])
            freed += L[cand[o]]
            if used - freed + need <= R + 1e-6:
                break
        else:
            return False
        for i in take:
            if staged[i]:
                stats["bumped"] += 1
            stats["evict"] += 1
            release(i)
        return True

    def admit(i, t_adm, t):
        nonlocal n_turn, stall_sum
        s = max(0.0, t_adm - A[i])
        if A[i] >= warm:
            n_turn += 1
            stall_sum += s
            pk = per_key[keys[i]]
            pk[0] += 1
            pk[1] += s
        if loc[i] != ABSENT:
            release(i)
        arrived[i] = False
        phase[i] = TURN
        busy[i] = A[i] + tau
        if t_adm > busy[i]:
            busy[i] = t_adm
            stats["overrun"] += 1

    nsteps = int(T / dt)
    for n in range(nsteps):
        t = n * dt
        # (a) landings
        idx = np.flatnonzero((loc == READING) & (land <= t + 1e-9))
        for i in idx:
            if arrived[i]:
                admit(i, land[i], t)
            else:
                loc[i] = PRESENT
        # (b) turns end: the history asks to stay
        keep_req = []
        idx = np.flatnonzero((phase == TURN) & (busy <= t + 1e-9))
        for i in idx:
            phase[i] = IDLE
            e[i] = busy[i]
            A[i] = e[i] + sample_gap(keys[i], rng, 1)[0]
            if mix:
                which[i] = 0 if rng.random() < mix[2] else 1
            if not drop_tab[which[i], cls_id[i], 0]:
                keep_req.append(i)
        # (c) arrivals
        idx = np.flatnonzero((phase == IDLE) & (A <= t + 1e-9))
        for i in idx:
            if i in keep_req:
                keep_req.remove(i)
            if loc[i] == PRESENT:
                admit(i, A[i], t)
            else:
                arrived[i] = True
                phase[i] = WAIT
                if loc[i] == ABSENT:
                    pending.append(i)
        # (d) plans of idle sessions
        idle = np.flatnonzero((phase == IDLE) & ~arrived)
        early_req = []
        if idle.size:
            js = np.minimum(((t - e[idle]) / dt + 1e-9).astype(int), M)
            c = cls_id[idle]
            w = which[idle]
            dmask = (loc[idle] == PRESENT) & drop_tab[w, c, js]
            for i in idle[dmask]:
                release(i)
            smask = (loc[idle] == ABSENT) & start_tab[w, c, js]
            early_req = [i for i in idle[smask] if i not in keep_req]
        # (e) grant: arrived turns first, then keeps and early reads by score
        still = []
        for i in pending:
            if make_room(L[i], (1.0 / L[i]) if compete else np.inf, t, soft):
                loc[i] = READING
                land[i] = (A[i] if t - A[i] <= dt + 1e-9 else t) + rd[i]
                used += L[i]
            else:
                still.append(i)
                stats["blocked"] += 1
        pending = still
        reqs = [(score_tab[cls_id[i], 0], i, "keep") for i in keep_req]
        for i in early_req:
            jl = min(idle_step(i, t) + int(round(rd[i] / dt)), M)
            reqs.append((score_tab[cls_id[i], jl], i, "early"))
        reqs.sort(key=lambda x: -x[0])
        for sc, i, kind in reqs:
            if make_room(L[i], sc, t, False):
                used += L[i]
                if kind == "keep":
                    loc[i] = PRESENT
                else:
                    loc[i] = READING
                    land[i] = e[i] + idle_step(i, t) * dt + rd[i]
                    staged[i] = True
                    stats["early"] += 1
            else:
                stats["early_denied" if kind == "early" else "keep_denied"] += 1
        assert used <= R + 1e-3, (used, R)
        if t >= warm:
            room_ts += used * dt
    span = nsteps * dt - warm
    return dict(stall=stall_sum / max(n_turn, 1), turns=n_turn, room=room_ts / span,
                per_key={k: (v[0], v[1] / max(v[0], 1)) for k, v in per_key.items()}, **stats)


# ===== Checks and sweeps =====

TAU = 4.0  # seconds a turn keeps its history off the room
SIZES = (48, 96, 192, 384, 768)
POLICIES = {
    "lru": dict(plan="none", score="lru"),
    "rules": dict(plan="priced", score="psi"),
    "rules_f30": dict(plan="priced", score="f30"),
    "rules_compete": dict(plan="priced", score="psi", arrivals="compete"),
}
_tables = None
_mix = None


def single():
    """Rule 1 against the exact per-session optimum, and the rule 3 example by return rate."""
    for lam in (5e-5, 5e-6):
        print(f"lambda = {lam:g}")
        for key in ("sharp", "pip", "editor", "human"):
            for kilo in (8, 32, 64, 128):
                c = Cls(key, kilo * 1024, dt=0.05, tmax=1200.0)
                sol = c.solve(lam)
                best = (c.r + lam * c.L * c.r, None)
                for s in np.arange(0, 200.01, 0.05):
                    st, pn = c.single_read(lam, s)
                    cost = st + lam * c.L * (pn + c.r)
                    if cost < best[0] - 1e-9:
                        best = (cost, round(float(s), 2))
                print(f"  {key:6s} {kilo:>4}K r={c.r:5.2f} | exact {sol['V']:7.3f} | best single read {best[0]:7.3f} at {best[1]}")
    print("rule 3 example: return rate per second, max over horizons")
    for sid, key, idle, kilo in (("A", "editor", 1.0, 10), ("B", "sharp", 12.0, 20), ("C", "pip", 3.0, 24),
                                 ("D", "pip", 45.0, 30), ("E", "human", 20.0, 12)):
        c = Cls(key, kilo * 1024, dt=0.05, tmax=3000.0)
        idx, _ = c.rate_index()
        print(f"  {sid} {key:6s} idle {idle:5.1f} s  F(30 s|t) {c.F(30, idle):.3f}  rate {idx[int(round(idle / c.dt))] / C_READ:.4f}")


def _job(a):
    global _tables
    if _tables is None:
        _tables = Tables(dt=0.05, tmax=2400.0)
    per, n, name, lam, seed = a
    pol = POLICIES[name] if name in POLICIES else dict(plan="priced", score="psi", mix=(_mix[0], _mix[1], lam))
    r = run(n, per * n, pol, lam, _tables, seed=seed, T=3300.0, warm=300.0, tau=TAU)
    return (n, name, lam, r["stall"])


def sweep(per, lams, mix, n_seeds, variants):
    """Stall per turn by number of sessions: LRU, the rules at their best price or mix, and the floor."""
    global _mix
    _mix = mix
    t0 = time.time()
    tb = Tables(dt=0.05, tmax=2400.0)
    keys, sizes = population(48)
    fine = list(np.concatenate([[1e-7, 3e-7], np.logspace(-6, -3, 121)]))
    floor, lam_star, _ = bound(tb, keys, sizes, per * 48, TAU, fine)
    seeds = list(range(1, n_seeds + 1))
    names = ["rules"] + (["rules_f30", "rules_compete"] if variants else [])
    jobs = [(per, n, "lru", 0.0, s) for n in SIZES for s in seeds]
    jobs += [(per, n, name, lam, s) for n in SIZES for name in names for lam in lams for s in seeds]
    jobs += [(per, n, "mix", share, s) for n in SIZES for share in (mix[2:] if mix else []) for s in seeds]
    with Pool(22) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    agg = {}
    for n, name, lam, stall in res:
        agg.setdefault((n, name, lam), []).append(stall)
    mean = {k: float(np.mean(v)) for k, v in agg.items()}
    rows = []
    print(f"room per session {per:.0f} tokens: floor {floor:.4f} s per turn at lambda* = {lam_star:.2e}")
    for n in SIZES:
        row = dict(N=n, lru=mean[(n, "lru", 0.0)])
        cands = [(mean[(n, "rules", lam)], f"lam {lam:g}") for lam in lams]
        cands += [(mean[(n, "mix", share)], f"mix {share:g}") for share in (mix[2:] if mix else [])]
        row["rules"], row["rules_at"] = min(cands)
        for name in names[1:]:
            row[name] = min(mean[(n, name, lam)] for lam in lams)
        rows.append(row)
        extra = "".join(f" | {name} {row[name]:.3f}" for name in names[1:])
        print(f"N={n:4d} | LRU {row['lru']:.3f} | rules {row['rules']:.3f} ({row['rules_at']}){extra} | x floor {row['rules'] / floor:.2f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"sweep_{int(per)}.json").write_text(json.dumps(dict(per=per, floor=floor, lam_star=lam_star, rows=rows), indent=1))
    print(f"done in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("single", "sweep"))
    ap.add_argument("--per", type=float, default=9000.0, help="tokens of room per session")
    ap.add_argument("--lams", default="2e-6,2.5e-6,3e-6,3.5e-6,4e-6,5e-6,7e-6,1e-5")
    ap.add_argument("--mix", default="", help="lam_lo,lam_hi,share,share,...: gaps planned at lam_lo with that share")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--variants", action="store_true", help="also the fixed-horizon order and competing arrivals")
    args = ap.parse_args()
    if args.step == "single":
        single()
        return
    mix = [float(x) for x in args.mix.split(",")] if args.mix else None
    sweep(args.per, [float(x) for x in args.lams.split(",")], mix, args.seeds, args.variants)


if __name__ == "__main__":
    main()
