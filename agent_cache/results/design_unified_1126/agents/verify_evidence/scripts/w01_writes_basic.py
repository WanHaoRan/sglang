"""ev_write checks that need no page model: submitted tokens, submit->ack lag, fix-3 holds, short evictions,
the census's dirty set against room."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv, pct, mean

def wq(vals, wts, q):
    z = sorted(zip(vals, wts)); tot = sum(wts); acc = 0.0
    for v, w in z:
        acc += w
        if acc >= q * tot:
            return v
    return z[-1][0]

for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "jit_p50"), (34, "ec_exact"), (34, "hicache"), (32, "ec_full"), (32, "jit_exact")]:
    ev = vlib.load(camp, arm)
    print(f"== c{camp} {arm}")
    sub = {}
    for t, r in ev["h2s_submit"]:
        d = kv(r); sub[d["op"]] = (t, int(d["tokens"]))
    tot = sum(v[1] for v in sub.values())
    done = {}
    for t, r in ev["h2s_done"]:
        d = kv(r); done[d["op"]] = t
    lags = [(done[o] - sub[o][0], sub[o][1]) for o in sub if o in done]
    lv = [x[0] for x in lags]; lw = [x[1] for x in lags]
    print(f"  h2s_submit ops {len(sub)}  tokens {tot/1e6:.2f} M   acked ops {len(lags)}  never acked {len(sub)-len(lags)}")
    print(f"  submit->ack per token: p50 {wq(lv,lw,.5):.2f}  p90 {wq(lv,lw,.9):.2f}  p99 {wq(lv,lw,.99):.2f}  max {max(lv):.2f} s ;"
          f" per op: p50 {pct(lv,.5):.2f} p90 {pct(lv,.9):.2f}")
    # skipped pages (fix 1)
    sk = pg = 0
    for t, r in ev["h2s_io"]:
        d = kv(r); pg += int(d["pages"]); sk += int(d.get("skipped", 0))
    print(f"  h2s_io pages {pg}  skipped as stored {sk} ({100*sk/max(pg,1):.1f} %) = {sk*64/1e6:.2f} M tokens")
    # write holds
    hs = ev.get("srs_write_hold", [])
    on = None; durs = []
    for t, r in hs:
        d = kv(r)
        if d["state"] == "on":
            on = t
        elif on is not None:
            durs.append((on, t - on)); on = None
    b = [x for x in durs if x[0] < 7200]
    if durs:
        dd = [x[1] for x in b]
        print(f"  write holds: {len(durs)} in the log, {len(b)} before 7,200 s; total {sum(dd):.1f} s  median {pct(dd,.5):.2f}  max {max(dd):.2f} s")
    # short host evictions
    n = short = 0
    late = []
    for t, r in ev["evict_host"]:
        d = kv(r); n += 1
        if int(d["tokens"]) < int(d["requested"]):
            short += 1; late.append(round(t))
    print(f"  evict_host {n}  came back short {short}  at {late[:6]}{'...' if len(late)>6 else ''}")
    # census
    cs = [(t, kv(r)) for t, r in ev.get("host_census", [])]
    cs = [(t, d) for t, d in cs if 1800 <= t < 7200 and d.get("why") == "tick"]
    if cs:
        pw = [int(d["pending_write"]) for t, d in cs]; hw = [int(d["h_write"]) for t, d in cs]
        room = [int(d["free"]) + int(d["evict_leaf"]) + int(d["evict_casc"]) for t, d in cs]
        ho = [int(d["host_only"]) for t, d in cs]
        print(f"  census lines minute 30+: {len(cs)}; pending_write med {pct(pw,.5)/1e3:.1f} K p90 {pct(pw,.9)/1e3:.1f} K max {max(pw)/1e3:.1f} K;"
              f" h_write med {pct(hw,.5)/1e3:.1f} K p90 {pct(hw,.9)/1e3:.1f} K max {max(hw)/1e3:.1f} K; sum h_write / sum pending = {sum(hw)/max(sum(pw),1):.2f}")
        print(f"  room (free+evict_leaf+evict_casc) p10 {pct(room,.1)/1e3:.0f} K med {pct(room,.5)/1e3:.0f} K; pending/room p90 {pct([a/b for a,b in zip(pw,room)],.9):.3f} max {max(a/b for a,b in zip(pw,room)):.2f};"
              f" h_write/room p90 {pct([a/b for a,b in zip(hw,room)],.9):.3f} max {max(a/b for a,b in zip(hw,room)):.2f};"
              f" h_write/host_only p90 {100*pct([a/max(b,1) for a,b in zip(hw,ho)],.9):.1f} % max {100*max(a/max(b,1) for a,b in zip(hw,ho)):.1f} %")
        mir = [int(d["mirror"]) for t, d in cs]
        print(f"  mirror med {pct(mir,.5)/1e3:.0f} K  host_only med {pct(ho,.5)/1e3:.0f} K  free med {pct([int(d['free']) for t,d in cs],.5)/1e3:.1f} K")
