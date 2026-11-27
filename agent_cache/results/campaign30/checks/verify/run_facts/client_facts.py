"""Per arm, from the raw client.jsonl: t_start (UTC), req/error counts, summary, lanes at the cut, missing at the cut,
first requests among the missing, means before the cut and the +N*630 s adjustment. Read-only.
usage: client_facts.py <manifest> [cut]   (run from agent_cache/results)"""
import json, sys, datetime, collections

man, cut = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 6600.0
for line in open(man):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    fn = f"{run}/{prog}/client.jsonl"
    kinds = collections.Counter()
    reqs, lanes, run_rec, summ, canc, plays, nerr = [], [], None, None, [], [], []
    for l in open(fn):
        r = json.loads(l)
        k = r["kind"]
        kinds[k] += 1
        if k == "run":
            run_rec = r
        elif k == "req":
            reqs.append(r)
        elif k == "lanes":
            lanes.append(r)
        elif k == "summary":
            summ = r
        elif k == "cancelled":
            canc.append(r)
        elif k == "play":
            plays.append(r)
        elif k == "notify_error":
            nerr.append(r)
    ts = run_rec["t_start"]
    tsu = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    phases = collections.Counter(r.get("phase") for r in reqs)
    errs = [r for r in reqs if r.get("error")]
    ok = [r for r in reqs if r.get("ttft") is not None]
    print(f"== {label}: {fn}")
    print(f"  t_start {ts} = {tsu}; grace {run_rec['args'].get('grace')} duration {run_rec['args'].get('duration')} url {run_rec['args'].get('url')}")
    print(f"  kinds {dict(kinds)}; req phases {dict(phases)}; errors {len(errs)}; with ttft {len(ok)}; neither {len(reqs)-len(errs)-len(ok)}")
    for e in errs[:5]:
        print(f"  error: rid {e.get('rid')} play {e.get('play')} stream {e.get('stream')} k {e.get('k')} t_send {e.get('t_send')} err {e.get('error')}")
    print(f"  summary {summ}")
    print(f"  cancelled {canc}; notify_error lines {len(nerr)}")
    # lanes at the cut
    lc = [x for x in lanes if abs(x["t"] - cut) < 1.0]
    lanes_t = sorted(round(x['t'], 3) for x in lanes)
    print(f"  lanes count {len(lanes)}; first t {lanes_t[0] if lanes_t else None}, last t {lanes_t[-1] if lanes_t else None}; at cut {lc}")
    prof = [r for r in reqs if r.get("phase") == "profile"]
    before = [r for r in prof if r["t_send"] < cut]
    before_ok = [r for r in before if r.get("ttft") is not None]
    before_err = [r for r in before if r.get("error")]
    # in flight at the cut among recorded: sent before the cut, ended after the cut
    def end(r):
        if r.get("latency") is not None:
            return r["t_send"] + r["latency"]
        return None
    ended_after = [r for r in before if (end(r) is not None and end(r) > cut)]
    ended_after_err = [r for r in before if r.get("error")]
    fl = lc[0]["in_flight"] if lc else None
    miss = None if fl is None else fl - len(ended_after)
    print(f"  before cut: recorded {len(before)} (ttft {len(before_ok)}, errors {len(before_err)}); recorded ending after the cut {len(ended_after)}; in flight per lanes {fl}; missing {miss}")
    # errors with no latency: when did they end? (error requests carry no latency)
    for e in before_err:
        print(f"  error before cut has latency {e.get('latency')} (not counted in 'ending after')")
    if before_ok:
        m = sum(r["ttft"] for r in before_ok) / len(before_ok)
        print(f"  mean ttft before cut {m:.4f} on {len(before_ok)}")
        if miss:
            for extra in (630.0,):
                m2 = (sum(r["ttft"] for r in before_ok) + miss * extra) / (len(before_ok) + miss)
                print(f"  + {miss} at {extra:.0f} s: mean {m2:.4f}, raise {m2 - m:.4f}")
    # first requests: users let in by the cut with no recorded first request
    started_by_cut = [p for p in plays if p.get("start") is not None and p["start"] < cut]
    lanes_plays = lc[0]["plays"] if lc else None
    firsts = {r["play"] for r in prof if r["stream"] == 0 and r["k"] == 0 and r["t_send"] < cut}
    print(f"  plays started by cut per lanes {lanes_plays}; play records started < cut {len(started_by_cut)} (play lines {len(plays)}); plays with a recorded first request before cut {len(firsts)}; without: {None if lanes_plays is None else lanes_plays - len(firsts)}")
