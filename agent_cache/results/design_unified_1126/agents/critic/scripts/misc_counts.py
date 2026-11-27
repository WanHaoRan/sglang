"""Counts of small mechanisms per run: pin ends, cuts and refusals, partial reads, prefix-shrank events,
dispatch window state, budget headroom, parked anchors, eviction overshoot, re-evictions."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

PF = re.compile(r"HiCache prefetch (success|dropped) req=(\S+) completed=(\d+) matched=(\d+) loaded=(\d+) occupied=(\d+)")
SH = re.compile(r"device prefix shrank before admission req=(\S+) lookup_match=(\d+) current_match=(\d+)")

for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    d = xlib.load(camp, arm)
    ev = d["ev"]
    print("==", spec)
    reasons = {}
    for t, f in ev.get("srs_unpin", []):
        reasons.setdefault(f.get("reason"), [0, 0])
        reasons[f.get("reason")][0] += 1; reasons[f.get("reason")][1] += f.get("tokens", 0)
    print("srs_unpin by reason (count, tokens):", {str(k): tuple(v) for k, v in reasons.items()}, "| srs_pin_skipped:", len(ev.get("srs_pin_skipped", [])))
    # pin durations
    pins = {}
    durs = []
    for t, f in ev.get("srs_pin", []):
        pins[f.get("rid")] = (t, f.get("tokens", 0))
    for t, f in ev.get("srs_unpin", []):
        p = pins.pop(f.get("rid"), None)
        if p:
            durs.append((t - p[0], p[1], f.get("reason")))
    if durs:
        ds = [x[0] for x in durs]
        print(f"pin duration s: p50 {xlib.pct(ds,.5):.2f} p90 {xlib.pct(ds,.9):.2f} p99 {xlib.pct(ds,.99):.1f} max {max(ds):.0f}; over 60 s: {sum(x>60 for x in ds)}, over 300 s: {sum(x>300 for x in ds)}, at 600 s or more: {sum(x>=599 for x in ds)}; token-seconds total {sum(x[0]*x[1] for x in durs)/1e6:.1f} M")
    cs = ev.get("host_census", [])
    if cs:
        last = cs[-1][1]
        keys = ["refused", "refused_tok", "cut", "cut_tok", "backup_refused", "backup_refused_tok"]
        tot = {k: sum(f.get(k, 0) for t, f in cs) for k in keys}
        print("census counters summed over lines (each line counts since the previous one):", tot)
        late = [f for t, f in cs if t >= 1680]
        for k in ("h_parked", "h_lookup", "h_read", "h_pin", "h_staged", "h_load_back", "h_write", "h_other", "h_multi", "anc", "relq", "staging", "occ", "n_parked", "free", "evict_leaf", "evict_casc", "mirror", "mirror_locked", "dev_unbacked", "host_only", "walk_ms", "n_leaves", "n_nodes"):
            v = [f.get(k, 0) for f in late]
            print(f"   census {k:14s} minute 28+: mean {xlib.mean(v):10.1f} p50 {xlib.pct(v,.5):10.1f} p90 {xlib.pct(v,.9):10.1f} max {max(v):10.1f}")
    disp = ev.get("srs_dispatch", [])
    if disp:
        before = [f.get("inflight_pages", 0) - f.get("pages", 0) for t, f in disp]
        after = [f.get("inflight_pages", 0) for t, f in disp]
        late = [(t, f) for t, f in disp if t >= 4200]
        print(f"srs_dispatch: {len(disp)}; launched=0: {sum(1 for t,f in disp if f.get('launched',1)==0)}; pages in flight before the dispatch p50 {xlib.pct(before,.5):.0f} p90 {xlib.pct(before,.9):.0f} max {max(before)}; after p50 {xlib.pct(after,.5):.0f} p90 {xlib.pct(after,.9):.0f} max {max(after)}")
        if late:
            b2 = [f.get("inflight_pages", 0) - f.get("pages", 0) for t, f in late]
            w2 = [f.get("waited_ms", 0) / 1e3 for t, f in late]
            print(f"   minute 70+: {len(late)} dispatches; in flight before: p10 {xlib.pct(b2,.1):.0f} p50 {xlib.pct(b2,.5):.0f} p90 {xlib.pct(b2,.9):.0f}; share dispatched with >= 512 already in flight {sum(x>=512 for x in b2)/len(b2):.2f} (these are 'now' reads), with 0 in flight {sum(x==0 for x in b2)/len(b2):.2f}; waited s p50 {xlib.pct(w2,.5):.1f} p90 {xlib.pct(w2,.9):.1f}")
        pinned = [f.get("pinned_tokens", 0) + f.get("inflight_pages", 0) * 64 for t, f in disp]
        print(f"   in flight + pinned tokens at dispatch: p50 {xlib.pct(pinned,.5):.0f} p99 {xlib.pct(pinned,.99):.0f} max {max(pinned)} (budget 813,824)")
    pf = [PF.search(x[1]) for x in d["other"].get("pf", [])]
    pf = [m for m in pf if m]
    if pf:
        kinds = {}
        for m in pf:
            kinds[m[1]] = kinds.get(m[1], 0) + 1
        print("prefetch result lines:", kinds)
    sh = [SH.search(x[1]) for x in d["other"].get("shrank", [])]
    sh = [m for m in sh if m]
    if sh:
        lost = [int(m[2]) - int(m[3]) for m in sh]
        print(f"device prefix shrank before admission: {len(sh)} lines, {len(set(m[1] for m in sh))} requests; tokens shrunk p50 {xlib.pct(lost,.5):.0f} sum {sum(lost)/1e3:.0f} K")
    print("read failures:", len(d["other"].get("readfail", [])), "| cleanup passes:", [(round(t), g, s) for t, g, f_, gib, s in d["cleanup"]])
    eh = ev.get("evict_host", [])
    if eh:
        req = sum(f.get("requested", 0) for t, f in eh); got = sum(f.get("tokens", 0) for t, f in eh)
        short = sum(1 for t, f in eh if f.get("tokens", 0) < f.get("requested", 0))
        print(f"evict_host: {len(eh)} calls, requested {req/1e6:.1f} M, freed {got/1e6:.1f} M ({got/max(req,1):.2f}x), short {short}")
    ed = ev.get("evict_device", [])
    if ed:
        req = sum(f.get("requested", 0) for t, f in ed); got = sum(f.get("tokens", 0) for t, f in ed)
        print(f"evict_device: {len(ed)} calls, requested {req/1e6:.1f} M, freed {got/1e6:.1f} M ({got/max(req,1):.2f}x), ms total {sum(f.get('ms',0) for t,f in ed)/1e3:.1f} s")
    re_ = ev.get("ntp_reevict", [])
    if re_:
        print(f"ntp_reevict: {len(re_)} events on {len(set(f.get('rid') for t,f in re_))} requests; capped=1: {sum(1 for t,f in re_ if f.get('capped')==1)}; tokens evicted sum {sum(f.get('evicted',0) for t,f in re_)/1e6:.2f} M")
    gate = ev.get("ntp_gate", [])
    if gate:
        dec = {}
        for t, f in gate:
            dec[f.get("decision")] = dec.get(f.get("decision"), 0) + 1
        print("ntp_gate decisions:", dec)
        cov = [f.get("chain", 0) - f.get("matched", 0) for t, f in gate if f.get("decision") == "covered"]
        print(f"   covered: missing tokens (chain - matched) p50 {xlib.pct(cov,.5):.0f} p90 {xlib.pct(cov,.9):.0f}; in [64,256): {sum(64<=x<256 for x in cov)}")
    wh = ev.get("srs_write_hold", [])
    print("srs_write_hold lines:", len(wh), "| srs_rate:", [(round(t/60), f.get("ms_per_page")) for t, f in ev.get("srs_rate", [])][:0])
