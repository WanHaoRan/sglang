"""Shape checks on the cached parse: duplicates, missing keys, odd lines, field ranges.
  nice -n 19 python3 -I -B census_report.py"""
import collections, json, os, pickle, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LABELS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "ORDER"]
for label in LABELS:
    d = pickle.load(open(os.path.join(CACHE, label + ".pkl"), "rb"))
    c = d["client"]
    turns = c["turns"]
    keys = collections.Counter((r["conv"], r["turn"]) for r in turns)
    print(f"\n== {label}: client kinds {c['kinds']}, client dirs {c['n_client_dirs']}, bad json lines {c['bad_lines']}, file ends with newline {c['ends_with_newline']}")
    print(f"   t_start {c['run']['t_start']:.3f}; args: users {c['run']['args'].get('num_users')} conc {c['run']['args'].get('concurrency')} arrival {c['run']['args'].get('arrival_rate')} "
          f"gap_scale {c['run']['args'].get('gap_scale')} seed {c['run']['args'].get('seed')} timeout {c['run']['args'].get('timeout')} hints {c['run']['args'].get('hints')} "
          f"oracle_hints {c['run']['args'].get('oracle_hints')} trace {os.path.basename(str(c['run']['args'].get('trace')))} turns/user {c['run']['args'].get('turns')}")
    print(f"   turn records {len(turns)}, ttft null {sum(r['ttft'] is None for r in turns)}, with error {sum(bool(r['error']) for r in turns)}, duplicate keys {sum(v > 1 for v in keys.values())}, "
          f"t_send null {sum(r['t_send'] is None for r in turns)}, max t_send {max(r['t_send'] for r in turns):.1f} s, max ttft {max(r['ttft'] for r in turns if r['ttft'] is not None):.1f} s, "
          f"max latency {max(r['latency'] for r in turns if r['latency'] is not None):.1f} s")
    rr = [r["reply_reprefilled"] for r in turns]
    print(f"   reply_reprefilled: null {sum(v is None for v in rr)} (of them turn 0: {sum(1 for r in turns if r['reply_reprefilled'] is None and r['turn'] == 0)}), negative {sum(1 for v in rr if v is not None and v < 0)}, "
          f"cached_details null {sum(not r['has_cd'] for r in turns)} (turn 0: {sum(1 for r in turns if not r['has_cd'] and r['turn'] == 0)})")
    # is the order of sends consistent inside a session
    by = collections.defaultdict(list)
    for r in turns:
        by[r["conv"]].append(r)
    gaps = sum(1 for v in by.values() if sorted(x["turn"] for x in v) != list(range(len(v))))
    unsorted = sum(1 for v in by.values() if [x["turn"] for x in sorted(v, key=lambda x: x["t_send"])] != sorted(x["turn"] for x in v))
    print(f"   sessions {len(by)}, with a hole in their turn numbers {gaps}, with sends out of turn order {unsorted}")
    if "server" not in d:
        continue
    s = d["server"]
    print(f"   server census {s['census']}")
    for k, v in s["odd"].items():
        if k in ("errors",):
            print(f"   odd[{k}] first: {v[:3]}")
        elif k not in ("no_ts_ReqTimeStats",):
            print(f"   odd[{k}] {len(v)} samples; first: {v[0][:200]!r}")
    rts = s["rts"]
    rk = collections.Counter((r["conv"], r["turn"]) for r in rts if r["conv"] is not None)
    print(f"   ReqTimeStats {len(rts)}, with c-t rid {sum(rk.values())}, keys with several {sum(v > 1 for v in rk.values())}, attempts>0 {sum(1 for r in rts if r['attempts'])}, "
          f"queue null {sum(r['queue_ms'] is None for r in rts)}, entry null {sum(r['entry_time'] is None for r in rts)}, "
          f"client keys without stats {len(set(keys) - set(rk))}, stats keys not in client {len(set(rk) - set(keys))}")
    # entry + queue + forward against the line's own time
    off = [r["t"] - (r["entry_time"] + r["queue_ms"] / 1000 + r["forward_ms"] / 1000) for r in rts if r["conv"] is not None]
    print(f"   line time - (entry + queue + forward): median {st.median(off) * 1000:.1f} ms, min {min(off) * 1000:.1f}, max {max(off) * 1000:.1f}")
    t0 = c["run"]["t_start"]
    byk = {(r["conv"], r["turn"]): r for r in turns}
    lag = [r["entry_time"] - (t0 + byk[(r["conv"], r["turn"])]["t_send"]) for r in rts if r["conv"] is not None and (r["conv"], r["turn"]) in byk]
    print(f"   entry_time - (t_start + t_send): median {st.median(lag) * 1000:.1f} ms, p99 {sorted(lag)[int(.99 * len(lag))] * 1000:.1f}, min {min(lag) * 1000:.1f}, max {max(lag) * 1000:.1f}")
    dq = [byk[(r["conv"], r["turn"])]["ttft"] - r["queue_ms"] / 1000 for r in rts if r["conv"] is not None and (r["conv"], r["turn"]) in byk and byk[(r["conv"], r["turn"])]["ttft"] is not None]
    print(f"   ttft - queue_duration: min {min(dq):.3f} s, negative in {sum(v < 0 for v in dq)} turns, median {st.median(dq):.3f}")
    ev = s["evts"]
    print("   events: " + ", ".join(f"{k} {len(v)}" for k, v in sorted(ev.items())))
    for name in ("s2h_io", "ntp_stage", "srs_dispatch", "srs_park", "srs_pin", "srs_unpin", "prefetch_start"):
        v = ev.get(name, [])
        if v:
            print(f"   {name}: {len(v)}, rid without c-t {sum(e['conv'] is None for e in v)}, keys {len({(e['conv'], e['turn']) for e in v})}, "
                  f"keys not in client {len({(e['conv'], e['turn']) for e in v} - set(keys))}, fields {sorted(k for k in v[0] if k not in ('t', 'rid', 'conv', 'turn', 'raw'))}")
    io = ev.get("s2h_io", [])
    if io:
        per = collections.Counter((e["conv"], e["turn"]) for e in io)
        print(f"   s2h_io: pages=0 {sum(e['pages'] == 0 for e in io)}, tokens=0 {sum(e['tokens'] == 0 for e in io)}, ms=0 {sum(e['ms'] == 0 for e in io)}, non-integer ms {sum(not isinstance(e['ms'], int) for e in io)}, "
              f"keys with several {sum(v > 1 for v in per.values())}, max per key {max(per.values())}, pages total {sum(e['pages'] for e in io)}")
    dp = ev.get("srs_dispatch", [])
    if dp:
        print(f"   srs_dispatch: launched values {dict(collections.Counter(e.get('launched') for e in dp))}")
    for name in ("evict_device", "evict_host"):
        v = ev.get(name, [])
        print(f"   {name}: {len(v)} lines, tokens {sum(e['tokens'] for e in v) / 1e6:.1f} M, non-integer tokens {sum(not isinstance(e['tokens'], int) for e in v)}")
    print(f"   Decode lines {len(s['dec'])}, Prefill lines {len(s['pre'])}, prefetch success {len(s['misc'].get('succ', []))}, shrank {len(s['misc'].get('shrank', []))}, cap {len(s['misc'].get('cap', []))}")
