"""Store composition at the first pass, cross-session sharing, and R1's write volume (my page model)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay
arm = sys.argv[1] if len(sys.argv) > 1 else "ec_full"
D = ssd_replay.Data(34, arm); ssd_replay.first_sets(D); ssd_replay.uses_index(D)
reqs = D.reqs
p1 = D.passes[0][0]
# replay writes up to the first pass (no deletion before it)
order = sorted(reqs.items(), key=lambda x: x[1]["t_send"])
store = {}          # page id -> writer session
holders = {}        # page id -> set of sessions whose prompt held it (requests sent before p1)
for rid, q in order:
    if q["ttft"] is None: continue
    s = D.sess[rid]
    if q["t_send"] + q["ttft"] < p1:
        ks = D.keys[rid]
        for j in range(q["cached"] // 64, len(ks)):
            store.setdefault(ks[j], s)
        for i in ks:
            holders.setdefault(i, set()).add(s)
    if q["t_send"] + q["latency"] < p1:
        for i in D.out_pages[rid]:
            store.setdefault(i, s)
ended = {s for s, t in D.ended_at.items() if t < p1}
# which prompt pages of open sessions are in the session's latest prompt (still needed) at p1
latest = {}
for rid, q in order:
    if q["t_send"] < p1:
        latest[D.sess[rid]] = rid
live_latest = set()
for s, rid in latest.items():
    if s not in ended:
        live_latest.update(D.keys[rid])
cls = {"ended prompt": 0, "ended output": 0, "ended, used by a later session": 0, "open prompt, in the latest prompt": 0,
       "open prompt, dropped": 0, "open output": 0}
for i, w in store.items():
    if D.is_out[i]:
        cls["ended output" if w in ended else "open output"] += 1
        continue
    hs = holders.get(i, {w})
    if all(h in ended for h in hs):
        if D.next_use(i, p1) < float("inf"):
            cls["ended, used by a later session"] += 1
        else:
            cls["ended prompt"] += 1
    else:
        if i in live_latest: cls["open prompt, in the latest prompt"] += 1
        else: cls["open prompt, dropped"] += 1
tot = len(store)
print(f"c34 {arm}: store at the first pass ({p1:.0f} s): {tot} pages = {tot*64/1e6:.2f} M tokens; sessions ended by then {len(ended)}")
for k, v in cls.items():
    print(f"   {k:36s} {v*64/1e6:6.2f} M  {100*v/tot:5.1f} %")
e = cls["ended prompt"] + cls["ended output"]
print(f"   ended sessions' share (prompt + outputs, not used later): {100*e/tot:.1f} %;  a pass deletes {D.passes[0][2]} pages = {D.passes[0][2]*64/1e6:.2f} M; the ended pool is {e/D.passes[0][2]:.1f} passes")
# ---- sharing over the whole run
hold_all = {}; in_first = set()
for rid, q in reqs.items():
    s = D.sess[rid]
    for i in D.keys[rid]:
        hold_all.setdefault(i, set()).add(s)
    if q["k"] == 0:
        in_first.update(D.keys[rid])
shared = [i for i, hs in hold_all.items() if len(hs) >= 2]
print(f"pages held by two or more sessions: {len(shared)}; in some session's first prompt: {100*sum(1 for i in shared if i in in_first)/len(shared):.1f} %")
# first requests: pages shared with earlier sessions (an unbounded cache would hold them: ideal_cached)
firsts = [(rid, q) for rid, q in reqs.items() if q["k"] == 0]
n_sh = sum(1 for rid, q in firsts if q["ideal"] > 0); tok_sh = sum(q["ideal"] for rid, q in firsts)
# were all earlier holders over when this session's first request was sent?
seen = {}     # page id -> list of (session)
over_n = 0; over_tok = 0
for rid, q in sorted(firsts, key=lambda x: x[1]["t_send"]):
    pass
first_send = {D.sess[rid]: q["t_send"] for rid, q in firsts}
# earlier holders of a page: sessions with a request holding the page that completed before this send (as ideal_cached counts)
done_hold = {}
evs = []
for rid, q in reqs.items():
    evs.append((q["t_send"] + q["latency"], 1, rid)); 
    if q["k"] == 0: evs.append((q["t_send"], 0, rid))
evs.sort()
for t, kind, rid in evs:
    s = D.sess[rid]
    if kind == 1:
        for i in D.keys[rid]:
            done_hold.setdefault(i, set()).add(s)
    else:
        q = reqs[rid]
        npg = q["ideal"] // 64
        if npg == 0: continue
        ks = D.keys[rid][:npg]
        hs = set()
        for i in ks: hs |= done_hold.get(i, set())
        hs.discard(s)
        if hs and all(D.ended_at.get(h, 1e18) < t for h in hs):
            over_n += 1; over_tok += q["ideal"]
print(f"first requests {len(firsts)}: {n_sh} share pages with earlier sessions ({tok_sh/1e6:.2f} M tokens); for {over_n} of them ({over_tok/1e6:.2f} M, {100*over_tok/max(tok_sh,1):.0f} %) every earlier holder was already over")
# ---- R1: writes of the last request of sessions that ended
pp = oo = 0; n_end = 0; later = 0
for rid, q in reqs.items():
    s = D.sess[rid]
    if s in D.ended_at and abs(q["t_send"] + q["latency"] - D.ended_at[s]) < 1e-6:
        n_end += 1
        ks = D.keys[rid]
        w = ks[q["cached"] // 64:]
        pp += len(w); oo += len(D.out_pages[rid])
        later += sum(1 for i in w if D.next_use(i, q["t_send"]) < float("inf"))
allw = sum(len(D.keys[r]) - q["cached"] // 64 + len(D.out_pages[r]) for r, q in reqs.items())
print(f"R1: sessions that ended {n_end}; their last request wrote {pp*64/1e6:.2f} M of prompt pages + {oo*64/1e6:.2f} M of outputs = {(pp+oo)*64/1e6:.2f} M ({100*(pp+oo)/allw:.1f} % of {allw*64/1e6:.2f} M written); a later request's prompt holds {later*64/1e3:.0f} K of the prompt pages")
