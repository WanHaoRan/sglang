"""Offline check of queue-wait (W) predictors on a server log (read-only, CPU only).

Per request (ReqTimeStats of the arm's own rids): arrival = entry_time, wait = queue_duration,
admission = arrival + wait, input_len, cached_input_len.
At each arrival, and at fixed horizons h before each admission, predict the wait left with
  P1 (engine): (requests queued ahead + 0.5) / admissions per second over the last 60 s
  P2 (tokens): (input tokens queued ahead + own/2) / admitted input tokens per second (60 s)
  P3 (recent): mean wait of the last 20 admissions (a proxy of the wait EWMA)
  P5 (300 s):  as P1 with a 300 s rate window
and compare with the measured wait. Also: how much W changes over a gap G (|W(t+G) - W(t)|).
Usage: python3 w_estimators.py <server.log> <rid prefix> [t_from_min t_to_min]
"""
import bisect, re, statistics as st, sys

RX = re.compile(r"ReqTimeStats\(rid=(\S+?), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
                r"queue_duration=([0-9.]+)(ms|s), .*?entry_time=([0-9.]+)")

def load(path, prefix):
    reqs = []
    with open(path, errors="replace") as f:
        for line in f:
            if "ReqTimeStats" not in line:
                continue
            m = RX.search(line)
            if not m or not m[1].startswith(prefix + "-c"):
                continue
            w = float(m[5]) / (1000.0 if m[6] == "ms" else 1.0)
            a = float(m[7])
            reqs.append((a, w, a + w, int(m[2]), int(m[3])))
    reqs.sort()
    return reqs

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))] if xs else float("nan")

def summ(name, errs):
    if not errs:
        return f"{name}: none"
    ab = [abs(e) for e in errs]
    return (f"{name}: n {len(errs)} | err (actual-pred) p10 {q(errs,.1):+.1f} p50 {q(errs,.5):+.1f} "
            f"p90 {q(errs,.9):+.1f} | |err| p50 {q(ab,.5):.1f} p90 {q(ab,.9):.1f} mean {st.mean(ab):.1f}")

def main():
    path, prefix = sys.argv[1], sys.argv[2]
    reqs = load(path, prefix)
    t0 = reqs[0][0]
    lo = float(sys.argv[3]) * 60 if len(sys.argv) > 3 else 0.0
    hi = float(sys.argv[4]) * 60 if len(sys.argv) > 4 else 1e9
    arr = [r[0] for r in reqs]
    adm = sorted((r[2], r[3]) for r in reqs)            # admissions by time, with tokens
    adm_t = [x[0] for x in adm]
    adm_tok_cum = [0]
    for _, tok in adm:
        adm_tok_cum.append(adm_tok_cum[-1] + tok)
    adm_wait = sorted((r[2], r[1]) for r in reqs)
    adm_wait_t = [x[0] for x in adm_wait]

    def rates(t, win):
        i0, i1 = bisect.bisect_right(adm_t, t - win), bisect.bisect_right(adm_t, t)
        return (i1 - i0) / win, (adm_tok_cum[i1] - adm_tok_cum[i0]) / win, i1 - i0

    def queued_ahead(i, t):
        """Requests that arrived before i and are not admitted by t (FCFS ahead), and their tokens."""
        n = tok = 0
        for j in range(i - 1, -1, -1):
            if reqs[j][2] > t:
                n += 1
                tok += reqs[j][3]
            if t - reqs[j][0] > 2000:      # nobody waits that long here
                break
        return n, tok

    bins = [(0, 2), (2, 10), (10, 30), (30, 60), (60, 1e9)]
    out = {k: {b: [] for b in bins} for k in ("P1", "P2", "P3", "P5")}
    allerr = {k: [] for k in out}
    for i, (a, w, d, tok, cached) in enumerate(reqs):
        if not lo <= a - t0 < hi:
            continue
        r60, tr60, n60 = rates(a, 60.0)
        r300, _, n300 = rates(a, 300.0)
        n, tq = queued_ahead(i, a)
        if n == 0 or n60 < 5:
            continue       # the engine predicts 0 on an empty queue; leave those out
        k = bisect.bisect_right(adm_wait_t, a)
        recent = [x[1] for x in adm_wait[max(0, k - 20):k]]
        preds = {"P1": (n + 0.5) / r60, "P2": (tq + tok / 2) / tr60 if tr60 > 0 else None,
                 "P3": st.mean(recent) if recent else None, "P5": (n + 0.5) / r300 if n300 >= 5 else None}
        b = next(bb for bb in bins if bb[0] <= w < bb[1])
        for name, p in preds.items():
            if p is None:
                continue
            out[name][b].append(w - p)
            allerr[name].append(w - p)
    print(f"{path.split('/')[-2]} prefix {prefix}: {len(reqs)} requests; window {lo/60:.0f}-{min(hi, 1e7)/60:.0f} min; at arrival, queue non-empty")
    for name in ("P1", "P2", "P3", "P5"):
        print("  " + summ(name + " all", allerr[name]))
        for b in bins:
            print("    " + summ(f"{name} actual {b[0]}-{b[1] if b[1] < 1e9 else 'inf'} s", out[name][b]))

    # Horizons: the wait left h seconds before admission, predicted from the state at that time.
    print("  wait left h s before admission (requests queued at that time):")
    for h in (2.0, 5.0, 10.0, 20.0):
        e1, e2 = [], []
        for i, (a, w, d, tok, cached) in enumerate(reqs):
            t = d - h
            if t < a or not lo <= a - t0 < hi:
                continue
            r60, tr60, n60 = rates(t, 60.0)
            if n60 < 5:
                continue
            n, tq = queued_ahead(i, t)
            e1.append(h - (n + 0.5) / r60)
            if tr60 > 0:
                e2.append(h - (tq + tok / 2) / tr60)
        print("    " + summ(f"h {h:.0f} s P1", e1))
        print("    " + summ(f"h {h:.0f} s P2", e2))

    # How much the wait moves over a gap G: requests arriving G s apart.
    print("  |W(t+G) - W(t)| over arrivals (all requests in window):")
    for G in (1, 2, 5, 17, 64, 300):
        diffs = []
        for i, (a, w, d, tok, cached) in enumerate(reqs):
            if not lo <= a - t0 < hi:
                continue
            j = bisect.bisect_left(arr, a + G)
            if j < len(reqs):
                diffs.append(abs(reqs[j][1] - w))
        print(f"    G {G:>4} s: n {len(diffs)} p50 {q(diffs,.5):.1f} p90 {q(diffs,.9):.1f} mean {st.mean(diffs):.1f}")

if __name__ == "__main__":
    main()
