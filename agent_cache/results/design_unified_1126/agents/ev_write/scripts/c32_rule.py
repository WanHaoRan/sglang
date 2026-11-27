"""Campaign 32 has no census. Estimate, second by second, the draft's rule 'hold timed reads when free + evictable
clean room < the allocation expected while the pending writes drain' and fix 3's rule (pending write tokens >= 50 K),
and see which fire before each burst of refusals.

Room estimate (inferred, constants from campaign 34's census): room = 900 K (campaign 34's median free + evictable
room from minute 30) - 0.77 x the chains under pending writes of finished requests - the resident prefixes of parked
reads (their anchors are locked) - reads in flight - pinned tokens. Allocation expected = (new KV + read launches
of the last 30 s) / 30 x drain time, drain time = the age of the oldest pending write (a FIFO queue serves the
newest submit about that long after its submit).

  python3 -I -B c32_rule.py <label> [<label> ...]
"""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import PAGE, load_trees
from model import load
from write_held import Pending

ROOM0, K = 900_000, 0.77
trees = load_trees()
for label in sys.argv[1:]:
    m = load("campaign32", label, trees)
    pd = Pending(m)
    refused = sorted(d["t"] for d in m.dispatches if not d["launched"])
    bursts = []
    for t in refused:
        if not bursts or t - bursts[-1][-1] > 120.0: bursts.append([t])
        else: bursts[-1].append(t)
    disp = sorted(m.dispatches, key=lambda d: d["t"]); dt = [d["t"] for d in disp]
    parks = sorted(((d["park"]["t"], d["t"], d["rid"]) for d in m.dispatches if d["park"]), key=lambda x: x[0])
    d2t = [x[0] for x in m.d2h]
    rd = sorted((r["t0"], r["t1"], r["pages"]) for r in m.reads)
    def state(t):
        s = pd.summary(t)
        i = bisect.bisect_right(dt, t) - 1
        pinned = disp[i]["pinned_tokens"] if i >= 0 else 0
        inflight = sum(pg for a, e, pg in rd[max(0, bisect.bisect_left(rd, (t - 30,))):] if a <= t < e) * PAGE
        anchors = 0
        lo = bisect.bisect_left(parks, (t - 200,))
        for pt, dtm, rid in parks[lo:]:
            if pt > t: break
            if dtm > t:
                got = m.arrival.get(rid)
                if got: anchors += got[1]
        j0, j1 = bisect.bisect_left(d2t, t - 30), bisect.bisect_right(d2t, t)
        newkv = sum(x[2] for x in m.d2h[j0:j1])
        launches = sum(d["pages"] * PAGE for d in disp[bisect.bisect_left(dt, t - 30):bisect.bisect_right(dt, t)] if d["launched"])
        room = ROOM0 - K * s["bound"] - anchors - inflight - pinned
        alloc = (newkv + launches) / 30.0 * s["oldest_age"]
        return s, room, alloc
    end = bursts[-1][0] if len(bursts[-1]) > 500 else 7200.0     # the sustained loop starts with the last long burst
    quiet_draft = quiet_fix3 = n_quiet = 0
    starts = [b[0] for b in bursts]
    for t in range(1800, int(end)):
        near = any(-60 <= t - s0 <= 60 for s0 in starts)
        s, room, alloc = state(float(t))
        if not near:
            n_quiet += 1
            quiet_draft += room < alloc
            quiet_fix3 += s["pending"] >= 50000
    print(f"campaign32 {label}: {len(bursts)} bursts of refusals; seconds from minute 30 to the sustained loop ({end:.0f} s), away from "
          f"bursts by over 60 s: {n_quiet}; the draft's rule would hold in {quiet_draft} of them ({quiet_draft / max(n_quiet, 1):.1%}), "
          f"fix 3's 50 K rule in {quiet_fix3} ({quiet_fix3 / max(n_quiet, 1):.1%})")
    for b in bursts[:6]:
        t0 = b[0]
        first_draft = first_fix3 = None
        line = []
        for off in range(-60, 1):
            s, room, alloc = state(t0 + off)
            if room < alloc and first_draft is None: first_draft = off
            if s["pending"] >= 50000 and first_fix3 is None: first_fix3 = off
            if off in (-30, -10, -5, 0):
                line.append(f"{off:+d} s: room {room / 1e3:5.0f}K alloc {alloc / 1e3:4.0f}K pending {s['pending'] / 1e3:3.0f}K")
        print(f"   burst at {t0:.0f} s ({len(b)} refusals): draft's rule first true {first_draft} s, 50 K rule first true {first_fix3} s (within the 60 s before) | " + "; ".join(line))
