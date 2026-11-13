"""Test (a): unequal load at the same turn.
For each arm, at every turn's send time: sessions in progress (first send <= t < last send + its latency), the summed
prompt+completion tokens of the latest sent turn of each session in progress (live context), requests in flight
(t_send <= t < t_send + latency) and their summed prompt tokens. Arms are then compared turn by turn on LEAD5's shared
returning turns, and the paired TTFT differences are redone on load-matched turns and standardized to one load mix.
  nice -n 19 python3 -I -B ta_load.py"""
import heapq, itertools, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ARMS = [a for a in C.RUNS if a != "order"]
rank = C.order_rank()
D = {a: C.load(a) for a in ARMS}
B = {a: C.by_key(C.kept(D[a])) for a in ARMS}


def load_state(d, censor_fix=False):
    """(conv, turn) -> (sessions in progress, live context, requests in flight, in-flight prompt tokens) at the turn's send.
    Task definition: a session is in progress from its first send to its last send + that turn's latency. With censor_fix a
    session the client cut at the run's end (conv_end.capped, or no conv_end record in a run cut by an outage) stays in
    progress to the end: under the task definition a session parked in a long gap when the run stops looks ended early."""
    turns = sorted(d["turns"], key=lambda t: t["t_send"])
    last = {}
    for t in turns:
        last[t["conv"]] = t
    s_end = {c: t["t_send"] + (t["latency"] or 0.0) for c, t in last.items()}
    if censor_fix:
        for c in s_end:
            ce = d["conv_end"].get(str(c))
            if ce is None or ce.get("capped"):
                s_end[c] = float("inf")
    live, live_sum = {}, 0
    sess_heap, req_heap = [], []
    n_flight = flight_sum = 0
    out = {}
    for t in turns:
        now = t["t_send"]
        while sess_heap and sess_heap[0][0] <= now:
            _, c = heapq.heappop(sess_heap)
            live_sum -= live.pop(c)
        while req_heap and req_heap[0][0] <= now:
            _, p = heapq.heappop(req_heap)
            n_flight -= 1
            flight_sum -= p
        c = t["conv"]
        size = (t["pt"] or 0) + (t["ct"] or 0)
        if c not in live:
            if s_end[c] > now:
                heapq.heappush(sess_heap, (s_end[c], c))
                live[c] = 0
        if c in live:
            live_sum += size - live[c]
            live[c] = size
        n_flight += 1
        flight_sum += t["pt"] or 0
        heapq.heappush(req_heap, (now + (t["latency"] or 0.0), t["pt"] or 0))
        out[(c, t["turn"])] = (len(live), live_sum, n_flight, flight_sum)
    return out


L = {a: load_state(D[a]) for a in ARMS}
LF = {a: load_state(D[a], censor_fix=True) for a in ARMS}
with open(os.path.join(C.CACHE, "load_state.json"), "w") as f:
    json.dump({a: {f"{k[0]}-{k[1]}": v for k, v in L[a].items()} for a in ARMS}, f)
with open(os.path.join(C.CACHE, "load_state_fix.json"), "w") as f:
    json.dump({a: {f"{k[0]}-{k[1]}": v for k, v in LF[a].items()} for a in ARMS}, f)

keys = C.shared_keys(B, C.LEAD5, rank)
ep1 = [k for k in keys if 5500 <= rank.get(k, -1) < 7000]
ep2 = [k for k in keys if 8750 <= rank.get(k, -1) < 11000]
ep = ep1 + ep2
calm = [k for k in keys if not C.in_episode(rank, k)]
absent = [k for k in keys if k not in rank]
SETS = (("all", keys), ("episodes", ep), ("ep1 5500-6999", ep1), ("ep2 8750-10999", ep2), ("elsewhere", calm), ("absent from order", absent))
print(f"LEAD5 shared returning turns: {len(keys)}; episodes {len(ep)}; elsewhere {len(calm)}; absent from the order run {len(absent)}")

print("\n1. Load at the send time of the same turns, per arm: mean sessions in progress | live context K tokens (mean, p90) | requests in flight | in-flight prompt K | mean TTFT s | minute of send (mean)")
for name, ks in SETS:
    print(f"  {name} ({len(ks)} turns)")
    for a in C.LEAD5:
        print(f"    {a:11s} {C.mean(L[a][k][0] for k in ks):6.2f} | {C.mean(L[a][k][1] for k in ks) / 1000:7.1f} {C.pct([L[a][k][1] for k in ks], .9) / 1000:7.1f} | "
              f"{C.mean(L[a][k][2] for k in ks):6.2f} | {C.mean(L[a][k][3] for k in ks) / 1000:7.1f} | {C.mean(B[a][k]['ttft'] for k in ks):6.3f} | {C.mean(B[a][k]['t_send'] for k in ks) / 60:6.2f}")

print("\n2. Paired by turn, A minus B: TTFT s | sessions | live context K (and % of B's mean) | in flight | minutes of send (mean, p10, p50, p90) ; all shared turns, then the episodes")
for a, b in itertools.combinations(C.LEAD5, 2):
    for name, ks in (("all", keys), ("episodes", ep), ("ep1", ep1), ("ep2", ep2)):
        dt = [(B[b][k]["t_send"] - B[a][k]["t_send"]) / 60 for k in ks]
        cb = C.mean(L[a][k][1] for k in ks)
        print(f"  {b:10s} - {a:10s} {name:8s} ttft {C.mean(B[b][k]['ttft'] - B[a][k]['ttft'] for k in ks):+6.3f} | sess {C.mean(L[b][k][0] - L[a][k][0] for k in ks):+5.2f} | "
              f"ctx {C.mean(L[b][k][1] - L[a][k][1] for k in ks) / 1000:+7.1f}K ({100 * C.mean(L[b][k][1] - L[a][k][1] for k in ks) / cb:+5.2f} %) | "
              f"flight {C.mean(L[b][k][2] - L[a][k][2] for k in ks):+5.2f} | minute {C.mean(dt):+5.2f} [{C.pct(dt, .1):+5.2f} {C.pct(dt, .5):+5.2f} {C.pct(dt, .9):+5.2f}]")

print("\n3. Wall-clock minute at which each arm sends the same turn: median t_send (min) of the shared turns within +-100 positions of each order-run position; and at shared-turn deciles of the mean send order")
for p in (1000, 3000, 5500, 6250, 7000, 8750, 9500, 10300):
    ks = [k for k in keys if abs(rank.get(k, -10 ** 9) - p) <= 100]
    print(f"  position {p:5d} ({len(ks):3d} turns): " + "  ".join(f"{a} {C.pct([B[a][k]['t_send'] for k in ks], .5) / 60:6.2f}" for a in C.LEAD5))
cons = sorted(keys, key=lambda k: C.mean(B[a][k]["t_send"] for a in C.LEAD5))
for i in range(1, 11):
    ks = cons[max(0, int(len(cons) * i / 10) - 100):int(len(cons) * i / 10)]
    print(f"  decile {i:2d} of the shared turns by mean send time: " + "  ".join(f"{a} {C.mean(B[a][k]['t_send'] for k in ks) / 60:6.2f}" for a in C.LEAD5))
print("  last kept send (min) and kept turns: " + "  ".join(f"{a} {max(t['t_send'] for t in B[a].values()) / 60:6.2f}/{len(B[a])}" for a in C.LEAD5))

M = LF if "fix" in sys.argv[1:] else L
print("\nsections 4-6 use " + ("the censoring-fixed load state (sessions cut by the run's end stay in progress)" if M is LF else "the task's definition of sessions in progress"))
print("\n4. Is the lead explained by load? Paired TTFT difference (B - A) on load-matched turns: |live context difference| <= 2 % and |in-flight difference| <= 1")
for a, b in itertools.combinations(C.LEAD5, 2):
    cells = []
    for name, ks in (("all", keys), ("episodes", ep)):
        m = [k for k in ks if abs(M[b][k][1] - M[a][k][1]) <= 0.02 * 0.5 * (M[b][k][1] + M[a][k][1]) and abs(M[b][k][2] - M[a][k][2]) <= 1]
        raw = C.mean(B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks)
        cells.append(f"{name}: raw {raw:+6.3f} on {len(ks)}, matched {C.mean(B[b][k]['ttft'] - B[a][k]['ttft'] for k in m):+6.3f} on {len(m)} "
                     f"(ctx diff there {C.mean(M[b][k][1] - M[a][k][1] for k in m) / 1000:+5.1f}K)")
    print(f"  {b:10s} - {a:10s} " + " | ".join(cells))

print("\n5. TTFT against the arm's own live context at the send (bins of 50K tokens): turns and mean TTFT per arm")
edges = [0, 700, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300, 10 ** 6]
print("  bin K        " + "  ".join(f"{a:>17s}" for a in C.LEAD5))
cell = {}
for lo, hi in zip(edges, edges[1:]):
    row = []
    for a in C.LEAD5:
        ks = [k for k in keys if lo * 1000 <= M[a][k][1] < hi * 1000]
        cell[(a, lo)] = (len(ks), C.mean(B[a][k]["ttft"] for k in ks))
        row.append(f"{len(ks):5d} {C.mean(B[a][k]['ttft'] for k in ks):7.3f}    ")
    print(f"  {lo:5d}-{hi if hi < 10 ** 6 else 'up':>5} " + "  ".join(row))
print("  load-standardized mean TTFT (each arm's bin means weighted by the pooled five-arm share of turns per bin), against the raw mean:")
tot = {lo: sum(cell[(a, lo)][0] for a in C.LEAD5) for lo in edges[:-1]}
for a in C.LEAD5:
    num = den = 0.0
    for lo in edges[:-1]:
        n, m = cell[(a, lo)]
        if n:
            num += tot[lo] * m
            den += tot[lo]
    print(f"    {a:11s} raw {C.mean(B[a][k]['ttft'] for k in keys):6.3f}  standardized {num / den:6.3f}  share of own turns at 1050K or more {100 * sum(cell[(a, lo)][0] for lo in edges[:-1] if lo >= 1050) / len(keys):5.1f} %")

print("\n6. Regression across shared turns of the paired TTFT difference on the paired live-context difference (per 10K tokens) and in-flight difference: intercept = difference at equal load")
for a, b in itertools.combinations(C.LEAD5, 2):
    for name, ks in (("all", keys), ("episodes", ep)):
        y = [B[b][k]["ttft"] - B[a][k]["ttft"] for k in ks]
        x1 = [(M[b][k][1] - M[a][k][1]) / 10000 for k in ks]
        x2 = [M[b][k][2] - M[a][k][2] for k in ks]
        n = len(ks)
        my, m1, m2 = C.mean(y), C.mean(x1), C.mean(x2)
        s11 = sum((u - m1) ** 2 for u in x1)
        s22 = sum((u - m2) ** 2 for u in x2)
        s12 = sum((u - m1) * (v - m2) for u, v in zip(x1, x2))
        s1y = sum((u - m1) * (v - my) for u, v in zip(x1, y))
        s2y = sum((u - m2) * (v - my) for u, v in zip(x2, y))
        det = s11 * s22 - s12 * s12
        b1 = (s1y * s22 - s2y * s12) / det
        b2 = (s2y * s11 - s1y * s12) / det
        b0 = my - b1 * m1 - b2 * m2
        syy = sum((v - my) ** 2 for v in y)
        r2 = (b1 * s1y + b2 * s2y) / syy if syy else 0
        print(f"  {b:10s} - {a:10s} {name:8s} raw {my:+6.3f} = intercept {b0:+6.3f} + ctx {b1 * m1:+6.3f} ({b1:+.3f} s per 10K x {m1:+5.2f}) + flight {b2 * m2:+6.3f} ({b2:+.3f} s per request x {m2:+5.2f}); R2 {r2:.3f}")

print("\n7. The same with sessions cut by the run's end kept in progress to the end (censoring fix): mean sessions | live context K | paired live-context difference against jit_exact K (%)")
for name, ks in SETS:
    print(f"  {name} ({len(ks)} turns)")
    for a in C.LEAD5:
        ref = C.mean(LF["jit_exact"][k][1] for k in ks)
        print(f"    {a:11s} {C.mean(LF[a][k][0] for k in ks):6.2f} | {C.mean(LF[a][k][1] for k in ks) / 1000:7.1f} | {C.mean(LF[a][k][1] - LF['jit_exact'][k][1] for k in ks) / 1000:+6.1f}K ({100 * C.mean(LF[a][k][1] - LF['jit_exact'][k][1] for k in ks) / ref:+5.2f} %)")
print("  largest pairwise gap in mean live context between two LEAD5 arms, task definition and with the fix:")
for name, ks in SETS:
    v = [C.mean(L[a][k][1] for k in ks) for a in C.LEAD5]
    w = [C.mean(LF[a][k][1] for k in ks) for a in C.LEAD5]
    print(f"    {name:18s} {100 * (max(v) - min(v)) / min(v):5.2f} %   fix {100 * (max(w) - min(w)) / min(w):5.2f} %")

print("\n8. The other sets: load at the same turns per arm (censoring fix): turns | sessions | live context K | in flight | minute of send | mean TTFT")
for title, names in (("SEVEN", C.SEVEN), ("ALL11", C.ALL11), ("CUT7", C.CUT7)):
    ks = C.shared_keys(B, names, rank)
    for sub, kk in (("all", ks), ("ep1 5500-6999", [k for k in ks if 5500 <= rank.get(k, -1) < 7000]), ("elsewhere", [k for k in ks if not C.in_episode(rank, k)])):
        print(f"  {title} {sub} ({len(kk)} turns)")
        for a in names:
            print(f"    {a:14s} {C.mean(LF[a][k][0] for k in kk):6.2f} | {C.mean(LF[a][k][1] for k in kk) / 1000:7.1f} | {C.mean(LF[a][k][2] for k in kk):6.2f} | {C.mean(B[a][k]['t_send'] for k in kk) / 60:6.2f} | {C.mean(B[a][k]['ttft'] for k in kk):6.3f}")

print("\n9. Turn by turn (censoring fix): distribution of the paired live-context difference B - A in % of A's value: p5, p50, p95; share of turns over 5 % apart; the same inside the episodes; paired sessions difference p5..p95")
for a, b in itertools.combinations(C.LEAD5, 2):
    cells = []
    for name, ks in (("all", keys), ("episodes", ep)):
        r = [100.0 * (LF[b][k][1] - LF[a][k][1]) / LF[a][k][1] for k in ks]
        cells.append(f"{name}: {C.pct(r, .05):+5.2f} {C.pct(r, .5):+5.2f} {C.pct(r, .95):+5.2f} %, over 5 % apart {100 * sum(1 for x in r if abs(x) > 5) / len(r):4.1f} %")
    s = [LF[b][k][0] - LF[a][k][0] for k in keys]
    print(f"  {b:10s} - {a:10s} " + " | ".join(cells) + f" | sessions {C.pct(s, .05):+d}..{C.pct(s, .95):+d}")
