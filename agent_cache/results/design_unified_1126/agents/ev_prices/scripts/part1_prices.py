"""Part 1: the prices over time, per minute of a run (DESIGN.md sections 1.5 and 6).

Per minute (by server time since the client's start):
  N        users in session (client play records), time average
  Q, Qr    queue length and READY queue length (no lookup/read of its own under way), time averages
  mu       min(1 + Qr, N)                                  (the draft's section 6 definition)
  mu_bp    1 + the admissions that follow an admission before the ready queue next empties, capped at
           N - B (users in session minus running requests), averaged over the minute's admissions: the
           turns one more GPU-second at that moment delays (first order, open loop)
  c_adm    share of admissions after which no ready turn was left waiting          (task definition)
  c_room   ... and the pool had room before it (last batch line before the admission: use < 0.90)
  c_arr    share of arriving turns with no SSD read that were admitted within 0.25 s
  c_time   share of time with an empty ready queue
  use, run median GPU pool use and #running-req on the batch lines
  parked   reads waiting for dispatch (hicache: for the read thread), time average
  late     of them past their latest start (due - own read time), time average: the draft's kappa count
  rd_busy  read thread busy share (sum of s2h_io ms)
  readers  returning requests admitted in the minute with a landed SSD read; their mean stall
  host     free / host-only tokens (census, campaign 34 only), host eviction tokens/s, turnover
           tau = host_only / eviction rate
usage: part1_prices.py camp:arm [...]  -> ../out/prices_<camp>_<arm>.csv and a printed summary."""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

MIN = 120


def overlap(a, b, lo, hi):
    return max(0.0, min(b, hi) - max(a, lo))


def per_minute(camp, arm):
    st = qstate.build(camp, arm)
    rows = readers.annotate(st)
    srv = st["srv"]
    out = [dict(minute=m) for m in range(MIN)]
    # queue state
    for m in range(MIN):
        lo, hi = m * 60.0, m * 60.0 + 60.0
        o = out[m]
        o["N"] = st["n_live"].mean(lo, hi)
        o["Q"] = st["tot"].mean(lo, hi)
        o["Qr"] = st["rdy"].mean(lo, hi)
        o["mu"] = min(1 + o["Qr"], max(1.0, o["N"]))
        o["c_time"] = st["rdy"].time_at(lo, hi, lambda v: v == 0) / 60.0
    by_m = [[] for _ in range(MIN)]
    for b in srv["batch"]:
        m = int(b[0] // 60)
        if 0 <= m < MIN:
            by_m[m].append(b)
    for m in range(MIN):
        bs = by_m[m]
        out[m]["use"] = lib.quantile([b[4] for b in bs], .5) if bs else float("nan")
        out[m]["run"] = lib.quantile([b[2] for b in bs], .5) if bs else float("nan")
    # busy-period count per admission
    import bisect
    rts_, rvs_ = st["rdy"].ts, st["rdy"].vs
    zero_idx = [i for i, v in enumerate(rvs_) if v == 0]
    adm_times = sorted(r["adm"] for r in rows)
    bp = [[] for _ in range(MIN)]
    for t in adm_times:
        m = int(t // 60)
        if not 0 <= m < MIN:
            continue
        i = bisect.bisect_right(rts_, t + 0.05) - 1
        if i < 0 or rvs_[i] == 0:
            bp[m].append(1)
            continue
        j = bisect.bisect_right(zero_idx, i)
        t_end = rts_[zero_idx[j]] if j < len(zero_idx) else float("inf")
        bp[m].append(1 + bisect.bisect_right(adm_times, t_end) - bisect.bisect_right(adm_times, t + 0.05))
    for m in range(MIN):
        o = out[m]
        cap = max(1.0, o["N"] - (o["run"] if o["run"] == o["run"] else 0))
        o["mu_bp"] = lib.mean(min(x, cap) for x in bp[m]) if bp[m] else float("nan")
        o["NB"] = cap
    # admissions and arrivals
    adm = [[] for _ in range(MIN)]
    arrn = [[] for _ in range(MIN)]
    rdr = [[] for _ in range(MIN)]
    sent = [[] for _ in range(MIN)]
    usage = st["usage"]
    for r in rows:
        m = int(r["adm"] // 60)
        if 0 <= m < MIN:
            room = usage.before(r["adm"] - 0.01) < 0.90
            adm[m].append((r["left_rdy"] == 0, r["left_rdy"] == 0 and room))
            if r["k"] > 0 and r["n_reads"] > 0:
                rdr[m].append(r)
        m = int(r["arr"] // 60)
        if 0 <= m < MIN and not r["reads"]:
            arrn[m].append(r["queue"])
        m = int(r["t_send"] // 60)
        if 0 <= m < MIN and r["ttft"] is not None:
            sent[m].append(r)
    for m in range(MIN):
        o = out[m]
        a = adm[m]
        o["n_adm"] = len(a)
        o["c_adm"] = sum(x[0] for x in a) / len(a) if a else float("nan")
        o["c_room"] = sum(x[1] for x in a) / len(a) if a else float("nan")
        q = arrn[m]
        o["c_arr"] = sum(1 for x in q if x < 0.25) / len(q) if q else float("nan")
        o["c_arr1"] = sum(1 for x in q if x < 1.0) / len(q) if q else float("nan")
        w = rdr[m]
        o["readers"] = len(w)
        o["stall"] = lib.mean(r["stall"] for r in w) if w else float("nan")
        o["read_K"] = lib.mean(r["read_tok"] for r in w) / 1e3 if w else float("nan")
        s = sent[m]
        o["sent"] = len(s)
        ret = [r for r in s if r["k"] > 0]
        o["ttft_ret"] = lib.mean(r["ttft"] for r in ret) if ret else float("nan")
        o["ttft_all"] = lib.mean(r["ttft"] for r in s) if s else float("nan")
        o["rd_share"] = sum(1 for r in ret if r["n_reads"] > 0) / len(ret) if ret else float("nan")
    # read line: parked, late, busy
    parked = [0.0] * MIN
    late = [0.0] * MIN
    busy = [0.0] * MIN
    for r in st["rows"]:
        if not r["reads"]:
            continue
        due = r.get("due")
        for rd in r["reads"]:
            if rd["land"] is None:
                continue
            io_s = (rd["ms"] or 0) / 1000.0
            if rd["park"] is not None and rd["disp"] is not None:
                p0, p1 = rd["park"], rd["disp"]
            else:
                p0, p1 = rd["t0"], rd["land"] - io_s
            done = rd["done"] if rd["done"] is not None else rd["land"]
            own = max(0.0, done - p1)
            ls = (due - own) if due is not None else None
            for m in range(max(0, int(p0 // 60)), min(MIN - 1, int(p1 // 60)) + 1):
                lo, hi = m * 60.0, m * 60.0 + 60.0
                parked[m] += overlap(p0, p1, lo, hi)
                if ls is not None and p1 > max(p0, ls):
                    late[m] += overlap(max(p0, ls), p1, lo, hi)
            b0, b1 = rd["land"] - io_s, rd["land"]
            for m in range(max(0, int(b0 // 60)), min(MIN - 1, int(b1 // 60)) + 1):
                busy[m] += overlap(b0, b1, m * 60.0, m * 60.0 + 60.0)
    for m in range(MIN):
        out[m]["parked"] = parked[m] / 60.0
        out[m]["late"] = late[m] / 60.0
        out[m]["rd_busy"] = busy[m] / 60.0
    # host memory
    ev = [0] * MIN
    for t, f in srv["glob"]["evict_host"]:
        m = int(t // 60)
        if 0 <= m < MIN:
            ev[m] += f.get("tokens", 0)
    cen = [[] for _ in range(MIN)]
    for t, f in srv["glob"]["host_census"]:
        m = int(t // 60)
        if 0 <= m < MIN:
            cen[m].append(f)
    price = [[] for _ in range(MIN)]
    for t, f in srv["glob"]["ec_price"]:
        m = int(t // 60)
        if 0 <= m < MIN:
            price[m].append(f.get("price"))
    for m in range(MIN):
        o = out[m]
        o["evict_tok_s"] = ev[m] / 60.0
        c = cen[m]
        o["host_free"] = lib.mean(x.get("free", 0) for x in c) if c else float("nan")
        o["host_only"] = lib.mean(x.get("host_only", 0) for x in c) if c else float("nan")
        o["mirror"] = lib.mean(x.get("mirror", 0) for x in c) if c else float("nan")
        o["pend_write"] = lib.mean(x.get("pending_write", 0) for x in c) if c else float("nan")
        o["h_pin"] = lib.mean(x.get("h_pin", 0) for x in c) if c else float("nan")
        ho = o["host_only"] if c else lib.POOL_HOST - lib.POOL_GPU * (o["use"] if o["use"] == o["use"] else 0)
        o["tau_host"] = ho / o["evict_tok_s"] if o["evict_tok_s"] > 0 else float("inf")
        o["ec_price"] = price[m][-1] if price[m] else float("nan")
    return out, st, rows


COLS = ["minute", "N", "sent", "ttft_ret", "Q", "Qr", "mu", "mu_bp", "NB", "use", "run", "n_adm", "c_adm", "c_room", "c_arr", "c_arr1",
        "c_time", "rd_share", "readers", "stall", "read_K", "parked", "late", "rd_busy", "evict_tok_s", "host_free",
        "host_only", "mirror", "pend_write", "h_pin", "tau_host", "ec_price"]


def fmt(v):
    if isinstance(v, float):
        if v != v:
            return ""
        if v == float("inf"):
            return "inf"
        return f"{v:.4g}"
    return str(v)


def window(out, lo, hi, key, how="mean"):
    xs = [o[key] for o in out[lo:hi] if o[key] == o[key] and o[key] != float("inf")]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def main():
    os.makedirs(lib.OUT, exist_ok=True)
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        out, st, rows = per_minute(camp, arm)
        with open(f"{lib.OUT}/prices_{camp}_{arm}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for o in out:
                w.writerow([fmt(o[c]) for c in COLS])
        print(f"== {spec}: per-minute table -> out/prices_{camp}_{arm}.csv")
        print("  minutes |   N   sent/min TTFT_ret |    Q    Qr    mu mu_bp   N-B | use  run | c_adm c_room c_arr c_time | rd%  stall s | parked late rd_busy |"
              " evict tok/s host_free host_only tau_host s")
        wins = [(0, 8), (8, 18), (18, 28), (28, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110), (110, 120)]
        for lo, hi in wins:
            g = lambda k: window(out, lo, hi, k)  # noqa: E731
            n_adm = sum(o["n_adm"] for o in out[lo:hi])
            wa = lambda k: (sum(o[k] * o["n_adm"] for o in out[lo:hi] if o[k] == o[k]) / n_adm) if n_adm else float("nan")  # noqa: E731
            rd = sum(o["readers"] for o in out[lo:hi])
            stall = (sum(o["stall"] * o["readers"] for o in out[lo:hi] if o["stall"] == o["stall"]) / rd) if rd else float("nan")
            ev = g("evict_tok_s")
            ho = g("host_only")
            tau = ho / ev if ev and ev > 0 and ho == ho else float("nan")
            print(f"  {lo:3d}-{hi:3d} | {g('N'):5.1f} {g('sent'):7.1f} {g('ttft_ret'):8.2f} | {g('Q'):5.1f} {g('Qr'):5.1f} {g('mu'):5.1f} {g('mu_bp'):5.1f} {g('NB'):5.1f} | "
                  f"{g('use'):.2f} {g('run'):4.1f} | {wa('c_adm'):5.2f} {wa('c_room'):5.2f} {g('c_arr'):5.2f} {g('c_time'):6.2f} | "
                  f"{100*g('rd_share'):4.1f} {stall:7.2f} | {g('parked'):6.2f} {g('late'):5.2f} {g('rd_busy'):6.2f} | "
                  f"{ev:9.0f} {g('host_free'):9.0f} {ho:9.0f} {tau:8.1f}")


if __name__ == "__main__":
    main()
