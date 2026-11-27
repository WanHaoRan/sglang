"""ev_fetch 3 and 4 (F1, F3): held reads, the pool at each hold, lateness and pin of held reads, the error of
the engine's predicted wait at the first hold, the read line by regime, the planner's read time against the
actual one.  Reads of queued turns only (parked at or after the turn's entry)."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, runtab
from vlib import kv, pct, mean

WINS = [(0, 30), (30, 70), (70, 90), (90, 110)]
def win(t):
    for lo, hi in WINS:
        if lo * 60 <= t < hi * 60: return (lo, hi)
    return None

for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "ec_exact"), (34, "jit_p50")]:
    tab = runtab.build(camp, arm); ev = vlib.load(camp, arm)
    reqs, reads, T, U, NR, stall = tab["reqs"], tab["reads"], tab["T"], tab["U"], tab["NR"], tab["stall"]
    print(f"===== c{camp} {arm}: reads parked {len(reads)}; staged before arrival {sum(1 for r in reads if r['staged'])}; ntp_stage lines {len(ev.get('ntp_stage', []))}; ntp_hold lines {len(ev.get('ntp_hold', []))}")
    Q = [r for r in reads if not r["staged"] and r["rid"] in reqs and r["pin"] is not None and not r.get("dead")]
    held = [r for r in Q if r["hold"] is not None]
    # pool usage at each ntp_hold line
    us = []
    for t, r in ev.get("ntp_hold", []):
        j = bisect.bisect_right(T, t) - 1
        us.append(U[j] if j >= 0 else 0.0)
    print(f"  holds: {len(us)} lines; pool usage at the hold: min {min(us):.2f} p1 {pct(us,.01):.2f} p10 {pct(us,.1):.2f}; under 0.90: {sum(1 for u in us if u < 0.90)}")
    print(f"  reads of queued turns that landed: {len(Q)}; with a hold line: {len(held)} (hold_ms > 100: {sum(1 for r in held if r['hold']['hold'] > 0.1)}), mean tokens {mean(r['pages']*64 for r in held)/1e3:.1f} K; before 6,600 s: {sum(1 for r in held if r['park'] < 6600)}")
    # lateness and pin of held reads (actual)
    def lp(rs):
        late = st = pin = pts = 0.0; n = 0
        for r in rs:
            s = stall.get(r["rid"])
            if s is None: continue
            q = reqs[r["rid"]]
            D = s["D"]
            n += 1
            if r["pin"] > D + 1e-6: late += 1
            st += max(q["adm"] - D, 0.0)
            pin += max(D - r["pin"], 0.0)
            pts += max(D - r["pin"], 0.0) * r["pages"] * 64
        return n, late / max(n, 1), st / max(n, 1), pin / max(n, 1), pts / max(n, 1)
    n, late, st, pin, pts = lp(held)
    print(f"  held reads, actual: n {n}  landed after the GPU turn {100*late:.1f} %  stall {st:.2f} s  pinned before the turn {pin:.2f} s = {pts/1e3:.1f} K token-s per read")
    for w in WINS:
        sub = [r for r in held if win(r["park"]) == w]
        allr = [r for r in Q if win(r["park"]) == w]
        n, late, st, pin, pts = lp(sub)
        n2, late2, st2, pin2, pts2 = lp(allr)
        print(f"    minutes {w[0]}-{w[1]}: held {n} of {n2} reads; held: late {100*late:.0f} % stall {st:.2f} s pin {pin:.2f} s ({pts/1e3:.0f} K token-s); all reads: late {100*late2:.0f} % stall {st2:.2f} s pin {pin2:.2f} s")
    # error of the predicted wait at the first hold: e = GPU turn - (hold time + predicted wait)
    for w in [(30, 70), (70, 90), (90, 110), None]:
        e = []
        for r in held:
            if w is not None and win(r["park"]) != w: continue
            s = stall.get(r["rid"])
            if s is None: continue
            e.append(s["D"] - (r["hold"]["t"] + r["hold"]["wait"]))
        if e:
            print(f"    error of the predicted GPU turn at the first hold, {'all' if w is None else 'minutes %d-%d' % w} (n {len(e)}): p5 {pct(e,.05):+.2f} p20 {pct(e,.2):+.2f} p50 {pct(e,.5):+.2f} p80 {pct(e,.8):+.2f} p95 {pct(e,.95):+.2f} s")
    # the line by regime
    print("  the line (medians [p90]), by park time: n | hold (first hold_ms; 0 if none) | park->dispatch minus hold | dispatch->io start | own io | io end->pin | (park+hold)->pin")
    for w in WINS:
        sub = [r for r in Q if win(r["park"]) == w]
        if not sub: continue
        hold = [r["hold"]["hold"] if r["hold"] else 0.0 for r in sub]
        line = [max(r["disp"] - r["park"] - (r["hold"]["hold"] if r["hold"] else 0.0), 0.0) for r in sub]
        fifo = [max(r["io_end"] - r["io_ms"] - r["disp"], 0.0) for r in sub]
        io = [r["io_ms"] for r in sub]
        tail = [r["pin"] - r["io_end"] for r in sub]
        tot = [r["pin"] - r["park"] - (r["hold"]["hold"] if r["hold"] else 0.0) for r in sub]
        f = lambda x: f"{pct(x,.5):6.2f} [{pct(x,.9):6.2f}]"
        print(f"    minutes {w[0]:3d}-{w[1]:<3d}: {len(sub):4d} | {f(hold)} | {f(line)} | {f(fifo)} | {f(io)} | {f(tail)} | {f(tot)}   held {sum(1 for r in sub if r['hold'])}  nowhold median park->dispatch {pct([r['disp']-r['park'] for r in sub if not r['hold']] or [0],.5):.2f}")
    # planner's read time (last ec_start before the turn's entry), without its 1 s margin, against the actual
    if "ec_start" in ev:
        last = {}
        ecs = [(t, kv(r)) for t, r in sorted(ev["ec_start"], key=lambda x: x[0])]
        by = {}
        for t, d in ecs:
            by.setdefault(d["rid"], []).append((t, d))
        print("  planner (last ec_start before the reading turn's entry): minutes | n | planned read (read_ms - 1 s) median | actual (park+hold)->pin median | ratio | planned W (wait_ms) median | actual entry->GPU turn median | decision counts")
        for w in WINS:
            pl = []; ac = []; pw = []; aw = []; decs = {}
            for r in Q:
                if win(r["park"]) != w: continue
                q = reqs[r["rid"]]
                cand = [x for x in by.get(r["rid"], []) if x[0] <= q["entry"]]
                if not cand: continue
                t, d = cand[-1]
                pl.append(float(d["read_ms"]) / 1e3 - 1.0)
                ac.append(r["pin"] - r["park"] - (r["hold"]["hold"] if r["hold"] else 0.0))
                pw.append(float(d["wait_ms"]) / 1e3)
                s = stall.get(r["rid"])
                if s: aw.append(s["D"] - q["entry"])
                decs[d["decision"]] = decs.get(d["decision"], 0) + 1
            if pl:
                print(f"    minutes {w[0]:3d}-{w[1]:<3d}: {len(pl):4d} | {pct(pl,.5):5.2f} | {pct(ac,.5):6.2f} | {pct(ac,.5)/pct(pl,.5):5.2f} | {pct(pw,.5):5.2f} | {pct(aw,.5):6.2f} | {decs}")
