"""Part 3 (F4): read or recompute, request by request.

For every returning request whose history came (in part) from an SSD read that landed before its admission:
  state at arrival   other ready turns waiting (q_rdy), GPU pool use, #running, trailing-60 s c and arrival rate
  stall              admission - due (readers.annotate): the wait its own read added beyond the GPU's turn
  c_i                share of [due, admission] in which no other ready turn waited
  X                  tokens its reads put in host memory; P = cached tokens at admission (the read is its tail)
  g_idle             seconds an idle FA3 GPU would take to compute those X tokens again:
                       b X + c X (P - X/2)   (per-run idle fit of prefill_fit.py)
  g_load             the same at the run's loaded cost: X (c_pre + c_x (P - X/2)) + extra batches x c_b
  ext                first-order delay of the others if g seconds of GPU work were added at `due`:
                       sum over later admissions j of (g - idle time between due and adm_j)+, at most N - B terms
                       ("idle" = no ready turn waiting);  ext_chunk caps each term at 1.5 chunks (0.375 s) when
                       c_i >= 0.5 (arrivals join the next prefill batch between 8,192-token chunks)
  B_i                gain in summed TTFT from recomputing instead of reading = c_i x stall - g - ext
Rules (decided with what is known when the lookup is answered) are scored against sign(B_i).
usage: part3_f4.py camp:arm[:t_hi_minutes] [...]"""
import bisect
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import qstate  # noqa: E402
import readers  # noqa: E402

FIT = json.load(open(f"{lib.OUT}/prefill_fit.json"))
CHUNK_CAP = 0.375


def g_idle(fit, X, P):
    return fit["b_s_per_tok"] * X + fit["c_s_per_tok_Mtok"] * X * max(0.0, P - X / 2) / 1e6


def g_load(fit, X, P, delta):
    batches = math.ceil((X + delta) / 8192) - math.ceil(max(delta, 1) / 8192)
    return (X * (fit["ms_per_tok_prefix0"] + fit["ms_per_tok_per_Kprefix"] * max(0.0, P - X / 2) / 1e3)
            + batches * fit["ms_per_batch"]) / 1e3


class Idle:
    """Cumulative time with an empty ready queue."""

    def __init__(self, step):
        self.ts = step.ts
        self.cum = [0.0] * len(step.ts)
        acc = 0.0
        for i in range(1, len(step.ts)):
            if step.vs[i - 1] == 0:
                acc += step.ts[i] - step.ts[i - 1]
            self.cum[i] = acc
        self.vs = step.vs

    def at(self, t):
        i = bisect.bisect_right(self.ts, t) - 1
        if i < 0:
            return 0.0
        return self.cum[i] + ((t - self.ts[i]) if self.vs[i] == 0 else 0.0)


def analyse(camp, arm):
    spec = f"{camp}:{arm}"
    st = qstate.build(camp, arm)
    rows = readers.annotate(st)
    fit_i, fit_l = FIT[spec]["idle"], FIT[spec]["loaded"]
    idle = Idle(st["rdy"])
    adm_sorted = sorted((r["adm"], r["rid"]) for r in rows)
    adm_t = [a for a, _ in adm_sorted]
    arr_t = sorted(r["arr"] for r in rows)
    adm_left0 = sorted((r["adm"], 1 if r["left_rdy"] == 0 else 0) for r in rows)
    al_t = [a for a, _ in adm_left0]
    al_c = [0]
    for _, v in adm_left0:
        al_c.append(al_c[-1] + v)
    out = []
    for r in rows:
        if r["k"] == 0 or r["n_reads"] == 0:
            continue
        X = r["read_tok"]
        P = max(r["cached_srv"] or 0, X)
        delta = max(0, (r["in_len"] or 0) - (r["cached_srv"] or 0))
        gi, gl = g_idle(fit_i, X, P), g_load(fit_l, X, P, delta)
        a, d, due = r["arr"], r["adm"], r["due"]
        stall = r["stall"]
        ci = (idle.at(d) - idle.at(due)) / stall if stall > 1e-6 else 1.0
        ci = max(0.0, min(1.0, ci))
        nb = max(1, int(st["n_live"].at(due) - st["running"].at(due)))
        idle_state = r["q_rdy_arr"] == 0 and (1 - r["use_arr"]) * lib.POOL_GPU >= (r["in_len"] or 0) + (r["out_len"] or 0) + readers.ROOM_MARGIN
        g = gi if idle_state else gl
        base = idle.at(due)
        j = bisect.bisect_right(adm_t, due)
        ext = ext_c = 0.0
        n_del = 0
        while j < len(adm_t) and n_del < nb:
            if adm_sorted[j][1] != r["rid"]:
                e = g - (idle.at(adm_t[j]) - base)
                if e <= 0:
                    break
                ext += e
                ext_c += min(e, CHUNK_CAP) if ci >= 0.5 else e
                n_del += 1
            j += 1
        # what a rule may know at arrival
        i0, i1 = bisect.bisect_left(al_t, a - 60.0), bisect.bisect_left(al_t, a)
        c60 = (al_c[i1] - al_c[i0]) / (i1 - i0) if i1 > i0 else 1.0
        lam60 = (bisect.bisect_left(arr_t, a) - bisect.bisect_left(arr_t, a - 60.0)) / 60.0
        pred_w = (r["ntp_admit"][2] or 0) / 1000.0 if r["ntp_admit"] and isinstance(r["ntp_admit"][2], (int, float)) else None
        pages = sum(rd["pages"] or 0 for rd in r["reads"] if rd.get("done") is not None and rd["done"] <= d + 0.05)
        before = r["first_t0"] is not None and r["first_t0"] < a - 0.05
        out.append(dict(rid=r["rid"], arr=a, due=due, adm=d, stall=stall, ci=ci, X=X, P=P, delta=delta, gi=gi, gl=gl, g=g,
                        ext=ext, ext_c=ext_c, n_del=n_del, q_rdy=r["q_rdy_arr"], q_tot=r["q_tot_arr"], use=r["use_arr"],
                        run=r["run_arr"], idle_state=idle_state, c60=c60, lam60=lam60, pred_w=pred_w, pages=pages,
                        before=before, ttft=r["ttft"], queue=r["queue"], n_reads=r["n_reads"], gap=r["gap"],
                        wait_s=r["wait_s"], nb=nb, po=r["po"] is not None))
    return out, rows, st


def fit_read(rs):
    """stall = a0 + a1 x pages on readers that arrived idle with the read issued at arrival."""
    pts = [(x["pages"], x["stall"]) for x in rs if x["idle_state"] and not x["before"] and x["n_reads"] == 1 and x["ci"] > 0.9]
    if len(pts) < 8:
        return None
    n = len(pts)
    mx, my = sum(p for p, _ in pts) / n, sum(s for _, s in pts) / n
    sxx = sum((p - mx) ** 2 for p, _ in pts)
    a1 = sum((p - mx) * (s - my) for p, s in pts) / sxx if sxx else 0.0
    return my - a1 * mx, a1, n


def score(name, rs, choose):
    ch = [x for x in rs if choose(x)]
    gain = sum(x["B"] for x in ch)
    wrong = [x for x in ch if x["B"] < 0]
    best = sum(max(0.0, x["B"]) for x in rs)
    missed = sum(1 for x in rs if x["B"] > 0 and not choose(x))
    return (f"    {name:58s} recomputes {len(ch):5d} of {len(rs):5d}; wrong {len(wrong):4d} ({100*len(wrong)/max(1,len(ch)):4.1f} %) costing "
            f"{-sum(x['B'] for x in wrong):8.1f} s; net {gain:+9.1f} TTFT-s ({100*gain/best if best > 0 else float('nan'):5.1f} % of the best "
            f"{best:7.1f}); misses {missed}")


def main():
    os.makedirs(lib.OUT, exist_ok=True)
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        camp, arm = parts[0], parts[1]
        t_hi = float(parts[2]) * 60 if len(parts) > 2 else 6600.0
        rs_all, rows, st = analyse(camp, arm)
        rs = [x for x in rs_all if x["arr"] < t_hi]
        for x in rs:
            x["B"] = x["ci"] * x["stall"] - x["g"] - x["ext"]
            x["Bc"] = x["ci"] * x["stall"] - x["g"] - x["ext_c"]
        with open(f"{lib.OUT}/f4_readers_{camp}_{arm}.csv", "w", newline="") as f:
            cols = ["rid", "arr", "due", "adm", "stall", "ci", "X", "P", "delta", "gi", "gl", "ext", "ext_c", "n_del", "q_rdy",
                    "q_tot", "use", "run", "idle_state", "c60", "lam60", "pred_w", "pages", "before", "ttft", "queue", "n_reads",
                    "gap", "wait_s", "B", "Bc"]
            w = csv.writer(f)
            w.writerow(cols)
            for x in rs_all:
                x.setdefault("B", x["ci"] * x["stall"] - x["g"] - x["ext"])
                x.setdefault("Bc", x["ci"] * x["stall"] - x["g"] - x["ext_c"])
                w.writerow([f"{x[c]:.4f}" if isinstance(x[c], float) else x[c] for c in cols])
        ret = [r for r in rows if r["k"] > 0 and r["arr"] < t_hi]
        n_ret = len(ret)
        sum_ttft = sum(r["ttft"] for r in ret if r["ttft"] is not None)
        print(f"== {spec} to minute {t_hi/60:.0f}: returning requests {n_ret} (mean TTFT {sum_ttft/n_ret:.3f} s); with an SSD read landed before "
              f"admission {len(rs)} ({100*len(rs)/n_ret:.2f} %); tokens read {sum(x['X'] for x in rs)/1e6:.2f} M")
        pre = [x for x in rs if x["before"]]
        post = [x for x in rs if not x["before"]]
        print(f"   read issued before the turn arrived: {len(pre)} (stall mean {lib.mean(x['stall'] for x in pre) if pre else 0:.2f} s); "
              f"issued at or after arrival: {len(post)}")
        idle = [x for x in post if x["idle_state"]]
        busy = [x for x in post if not x["idle_state"]]

        def line(tag, xs):
            if not xs:
                print(f"   {tag}: 0")
                return
            print(f"   {tag}: {len(xs)} ({100*len(xs)/n_ret:.2f} % of returning); tokens read mean {lib.mean(x['X'] for x in xs)/1e3:.1f} K "
                  f"(p10 {lib.quantile([x['X'] for x in xs], .1)/1e3:.1f}, p90 {lib.quantile([x['X'] for x in xs], .9)/1e3:.1f}); "
                  f"stall mean {lib.mean(x['stall'] for x in xs):.2f} s p50 {lib.quantile([x['stall'] for x in xs], .5):.2f}; c_i mean {lib.mean(x['ci'] for x in xs):.2f}; "
                  f"recompute idle {lib.mean(x['gi'] for x in xs):.2f} s ({1e3*sum(x['gi'] for x in xs)/sum(x['X'] for x in xs):.4f} ms/tok), "
                  f"loaded {lib.mean(x['gl'] for x in xs):.2f} s ({1e3*sum(x['gl'] for x in xs)/sum(x['X'] for x in xs):.4f} ms/tok); "
                  f"#running {lib.mean(x['run'] for x in xs):.1f}, pool use {lib.mean(x['use'] for x in xs):.2f}")
        line("arrived with an idle GPU (no ready turn waiting, pool room)", idle)
        line("arrived with ready turns waiting or no pool room", busy)
        if idle:
            own = [x["stall"] - x["gi"] for x in idle]
            net = [x["B"] for x in idle]
            netc = [x["Bc"] for x in idle]
            print(f"   idle arrivals, recompute instead of read: own TTFT {lib.mean(own):+.3f} s per request (saved if +), "
                  f"{sum(1 for v in own if v > 0)} of {len(own)} faster; others delayed {lib.mean(x['ext'] for x in idle):.3f} s per recompute "
                  f"(chunk-capped {lib.mean(x['ext_c'] for x in idle):.3f}; {lib.mean(x['n_del'] for x in idle):.2f} turns each); "
                  f"stall hidden by other work (1 - c_i) {lib.mean((1-x['ci'])*x['stall'] for x in idle):.3f} s")
            print(f"      net summed TTFT if ALL idle arrivals recomputed: {sum(net):+.1f} s (chunk-capped {sum(netc):+.1f} s) = "
                  f"{sum(net)/n_ret:+.4f} s per returning request ({sum(netc)/n_ret:+.4f}); own part alone {sum(own):+.1f} s = {sum(own)/n_ret:+.4f} s; "
                  f"only those with B > 0: {sum(v for v in net if v > 0):+.1f} s on {sum(1 for v in net if v > 0)} requests")
            for lo, hi in ((0, 8000), (8000, 16000), (16000, 24000), (24000, 32000), (32000, 48000), (48000, 70000)):
                b = [x for x in idle if lo <= x["X"] < hi]
                if b:
                    print(f"      read {lo//1000:2d}-{hi//1000:2d} K: n {len(b):4d}; stall {lib.mean(x['stall'] for x in b):.2f} s; recompute {lib.mean(x['gi'] for x in b):.2f} s; "
                          f"own gain {lib.mean(x['stall']-x['gi'] for x in b):+.2f} s; others {lib.mean(x['ext'] for x in b):.2f} s; net {lib.mean(x['B'] for x in b):+.2f} s; "
                          f"B > 0 in {sum(1 for x in b if x['B'] > 0)}")
        fr = fit_read(post)
        if fr:
            print(f"   read time on idle arrivals (single read, c_i > 0.9): stall = {fr[0]:.3f} s + {fr[1]*1e3:.3f} ms x pages (n {fr[2]}); "
                  f"per token {fr[1]/64*1e3:.4f} ms")
        a0, a1 = (fr[0], fr[1]) if fr else (0.1, 0.003)
        fi = FIT[f"{camp}:{arm}"]["idle"]
        # rules
        for x in post:
            x["r_meas"] = a0 + a1 * x["pages"]
            x["r_eng"] = 0.9 + 0.003 * x["pages"]
            x["g_hat"] = g_idle(fi, x["X"], x["P"])
            x["w_hat"] = x["pred_w"] if x["pred_w"] is not None else (0.0 if x["q_tot"] == 0 else 1e9)
        print(f"   rules on the {len(post)} reads issued at or after arrival (B from the serial model; in brackets what a rule knows):")
        print(score("always read (engine today)", post, lambda x: False))
        print(score("oracle: recompute iff B > 0", post, lambda x: x["B"] > 0))
        print(score("recompute iff idle at arrival", post, lambda x: x["idle_state"]))
        print(score("idle and g_hat < measured read time", post, lambda x: x["idle_state"] and x["g_hat"] < x["r_meas"]))
        print(score("draft: (1+q_rdy) g_hat < c60 (r_meas - W)+", post,
                    lambda x: (1 + x["q_rdy"]) * x["g_hat"] < x["c60"] * max(0.0, x["r_meas"] - x["w_hat"])))
        print(score("draft with the engine's read estimate (0.9 s + 3 ms/page)", post,
                    lambda x: (1 + x["q_rdy"]) * x["g_hat"] < x["c60"] * max(0.0, x["r_eng"] - x["w_hat"])))
        print(score("mu = 1 + q_rdy + lam60 g/2: mu g_hat < c60 (r_meas - W)+", post,
                    lambda x: (1 + x["q_rdy"] + x["lam60"] * x["g_hat"] / 2) * x["g_hat"] < x["c60"] * max(0.0, x["r_meas"] - x["w_hat"])))
        print(score("same, and q_rdy = 0 at arrival", post,
                    lambda x: x["q_rdy"] == 0 and (1 + x["lam60"] * x["g_hat"] / 2) * x["g_hat"] < x["c60"] * max(0.0, x["r_meas"] - x["w_hat"])))
        print(score("same with c60 >= 0.8 required", post,
                    lambda x: x["q_rdy"] == 0 and x["c60"] >= 0.8 and (1 + x["lam60"] * x["g_hat"] / 2) * x["g_hat"] < x["c60"] * x["r_meas"]))
        wrong = [x for x in post if x["idle_state"] and x["B"] < 0]
        if wrong:
            q_formed = sum(1 for x in wrong if x["ci"] < 0.5 or x["n_del"] >= 3)
            longr = sum(1 for x in wrong if x["g"] >= x["stall"])
            print(f"   idle arrivals where recomputing loses (B < 0): {len(wrong)}; a queue formed behind it (c_i < 0.5 or 3+ turns delayed) in {q_formed}; "
                  f"recompute slower than the read itself in {longr}; mean loss {-lib.mean(x['B'] for x in wrong):.2f} s")


if __name__ == "__main__":
    main()
