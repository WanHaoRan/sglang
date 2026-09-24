#!/usr/bin/env python3
"""Quantify L1/L2 duplication under write_through (campaign 7, GAP x70).

Read-only: parses server.log + client.jsonl + turn_components CSV.
"""
import json, re, sys, csv, statistics as st
from datetime import datetime, timezone

L1 = 668160
L2 = 1627648
SHARED = 7808
R = "/home/wanhr/sglang/agent_cache/results/"
RUNS = {
    "three_tier_to": ("20260922_051155_three_tier_to_NAT160", "client_051609_cmp_three_tier_to"),
    "hbm_host": ("20260922_023714_hbm_host_NAT160", "client_023901_cmp_hbm_host"),
}
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")
KV = re.compile(r"(\w+)=([0-9.]+)")


def ts(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main(arm):
    rd, cd = RUNS[arm]
    batches = []  # (t, kind, num_used, running, queue)
    ev = []  # (t, name, dict)
    with open(R + rd + "/server.log", errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            if "Decode batch" in line:
                t = ts(m[1])
                nu = int(re.search(r"#token: (\d+)", line)[1])
                run = int(re.search(r"#running-req: (\d+)", line)[1])
                q = int(re.search(r"#queue-req: (\d+)", line)[1])
                batches.append((t, "D", nu, run, q))
            elif "Prefill batch" in line:
                t = ts(m[1])
                u = float(re.search(r"token usage: ([0-9.]+)", line)[1])
                run = int(re.search(r"#running-req: (\d+)", line)[1])
                q = int(re.search(r"#queue-req: (\d+)", line)[1])
                batches.append((t, "P", u * L1, run, q))
            elif "HICACHE_EVT" in line:
                t = ts(m[1])
                name = line.split("HICACHE_EVT ")[1].split()[0]
                d = {k: float(v) for k, v in KV.findall(line)}
                ev.append((t, name, d))
    first_eh = next(t for t, n, d in ev if n == "evict_host")
    first_ed = next(t for t, n, d in ev if n == "evict_device")
    t_end = batches[-1][0]

    # host occupancy reconstruction (upper bound: allocs - evictions; ignores other frees)
    cum_d2h = sum(d["tokens"] for t, n, d in ev if n == "d2h_submit" and t < first_eh)
    first_eh_ev = next((d for t, n, d in ev if n == "evict_host"))

    # client
    turns = []
    with open(R + rd + "/" + cd + "/client.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if o.get("kind") == "run":
                t0 = o["t_start"]
            elif o.get("kind") == "turn":
                turns.append(o)
    # queue from turn_components CSV (returning turns); turn 0 -> queue = ttft (conservative: admits at first token)
    qmap = {}
    with open(R + f"analysis_gapscale/turn_components_x70 (campaign 7)_{arm}.csv") as f:
        for r in csv.DictReader(f):
            qmap[(int(r["conv"]), int(r["turn"]))] = float(r["queue"])
    reqs = []
    for o in turns:
        s = t0 + o["t_send"]
        q = qmap.get((o["conv"], o["turn"]), o["ttft"] * 0.9)
        a = s + q
        ft = s + o["ttft"]
        fin = s + o["latency"]
        reqs.append(dict(conv=o["conv"], turn=o["turn"], send=s, adm=a, ft=ft, fin=fin,
                         prompt=o["prompt_tokens"], cached=o["cached_tokens"], out=o["completion_tokens"]))

    # live set at first evict_host: per conv latest finished context (prompt + output)
    def live_set(t):
        ctx = {}
        for r in reqs:
            if r["fin"] <= t:
                c = r["prompt"] + r["out"]
                if c > ctx.get(r["conv"], 0):
                    ctx[r["conv"]] = c
        if not ctx:
            return 0, 0
        return SHARED + sum(c - SHARED for c in ctx.values()), len(ctx)

    ls_eh, nconv_eh = live_set(first_eh)
    ls_ed, _ = live_set(first_ed)

    # model running set at time t
    def model(t):
        nontree = 0.0
        locked_priv = 0.0
        nrun = 0
        for r in reqs:
            if r["adm"] <= t < r["fin"]:
                nrun += 1
                unc = r["prompt"] - r["cached"]
                if t < r["ft"]:
                    gen = 0
                else:
                    span = max(1e-6, r["fin"] - r["ft"])
                    gen = r["out"] * min(1.0, (t - r["ft"]) / span)
                nontree += unc + gen
                locked_priv += max(0, r["cached"] - SHARED)
        locked = (SHARED + locked_priv) if nrun else 0
        return nontree, locked, nrun

    # 1-second grid over the post-evict_host phase
    post = [b for b in batches if b[0] >= first_eh]
    grid = []
    bi = 0
    t = first_eh
    idx = 0
    bt = [b[0] for b in batches]
    import bisect
    while t <= t_end:
        j = bisect.bisect_right(bt, t) - 1
        if j >= 0 and t - bt[j] <= 5.0:
            nu = batches[j][2]
        else:
            nu = 0.0  # idle: no batch logged in 5 s
        nontree, locked, nrun = model(t)
        grid.append((t, nu, nontree, locked, nrun))
        t += 1.0

    nu_s = [g[1] for g in grid]
    nt_s = [g[2] for g in grid]
    lk_s = [g[3] for g in grid]
    # log-sample stats (decode lines exact #token)
    dec_post = [b[2] for b in post if b[1] == "D"]
    pre_post = [b[2] for b in post if b[1] == "P"]

    # device eviction overshoot (bounds free slack 'available')
    over = [d["tokens"] - d["requested"] for tt, n, d in ev if n == "evict_device" and tt >= first_eh]
    ed_post = [tt for tt, n, d in ev if n == "evict_device" and tt >= first_eh]
    gaps = [b - a for a, b in zip(ed_post, ed_post[1:])]

    print(f"=== {arm}")
    print(f"first evict_device {datetime.fromtimestamp(first_ed, timezone.utc)}  (+{first_ed - t0:.0f}s)")
    print(f"first evict_host   {datetime.fromtimestamp(first_eh, timezone.utc)}  (+{first_eh - t0:.0f}s)  {first_eh_ev}")
    print(f"end of batches     +{t_end - t0:.0f}s ; post-evict_host phase {t_end - first_eh:.0f}s")
    print(f"cum d2h_submit tokens before first evict_host: {cum_d2h:,.0f}  (L2={L2:,}, ratio {cum_d2h / L2:.3f})")
    print(f"live set (7808 + sum private ctx of finished turns) at first evict_device: {ls_ed:,}; at first evict_host: {ls_eh:,} over {nconv_eh} convs")
    print(f"  live/L2 {ls_eh / L2:.3f}  live/(L1+L2) {ls_eh / (L1 + L2):.3f}")
    print(f"num_used (#token=non-evictable device) decode-line samples post: n={len(dec_post)} mean {st.mean(dec_post):,.0f} p50 {pct(dec_post, .5):,.0f} p90 {pct(dec_post, .9):,.0f} max {max(dec_post):,.0f}")
    print(f"num_used prefill-line samples post: n={len(pre_post)} mean {st.mean(pre_post):,.0f} p90 {pct(pre_post, .9):,.0f}")
    print(f"num_used 1s-grid (0 when idle>5s): mean {st.mean(nu_s):,.0f} p50 {pct(nu_s, .5):,.0f} p90 {pct(nu_s, .9):,.0f}; idle share {sum(1 for x in nu_s if x == 0) / len(nu_s):.3f}")
    print(f"model running non-tree (uncached prompt + generated): mean {st.mean(nt_s):,.0f} p50 {pct(nt_s, .5):,.0f} p90 {pct(nt_s, .9):,.0f} max {max(nt_s):,.0f}")
    print(f"model locked tree (shared once + running cached private): mean {st.mean(lk_s):,.0f} p90 {pct(lk_s, .9):,.0f}")
    both = [(g[1], g[2] + g[3]) for g in grid if g[1] > 0]
    print(f"validation: grid points with batch: mean logged num_used {st.mean(b[0] for b in both):,.0f} vs model locked+nontree {st.mean(b[1] for b in both):,.0f}")
    print(f"evict_device post: n={len(over)} overshoot(tokens-requested) mean {st.mean(over):,.0f} p50 {pct(over, .5):,.0f} p90 {pct(over, .9):,.0f} max {max(over):,.0f}; inter-eviction gap p50 {pct(gaps, .5):.2f}s p90 {pct(gaps, .9):.2f}s")
    # evictable bounds
    ev_lb = [L1 - x for x in nu_s]  # upper bound on evictable+available
    # duplicates estimate: device tree tokens = L1 - avail - nontree ; avail in [0, overshoot p90]
    avail_hi = pct(over, .5)
    dup_hi = [L1 - x for x in nt_s]
    dup_lo_evictable = [L1 - avail_hi - x for x in nu_s]  # evictable only (drop locked tree), with slack
    dup_mid = [L1 - avail_hi - x for x in nt_s]
    for name, arr in (("dup upper (avail=0, all device tree backed)", dup_hi),
                      ("dup mid (avail=overshoot p50)", dup_mid),
                      ("dup lower (evictable only, avail=overshoot p50)", dup_lo_evictable)):
        m_, p10, p50 = st.mean(arr), pct(arr, .1), pct(arr, .5)
        print(f"{name}: mean {m_:,.0f} ({m_ / L2:.1%} of L2) p10 {p10:,.0f} ({p10 / L2:.1%}) p50 {p50:,.0f}")
    # write-through backlog: d2h pending tokens (submit - done) sampled at submits
    ub2 = [L1 - avail_hi - g[1] + min(g[3], g[1]) for g in grid]
    print(f"dup alt-upper (evictable + modeled locked tree, capped at num_used): mean {st.mean(ub2):,.0f} ({st.mean(ub2) / L2:.1%}) p10 {pct(ub2, .1):,.0f} p50 {pct(ub2, .5):,.0f} p90 {pct(ub2, .9):,.0f}")
    for name, arr in (("dup_mid", dup_mid), ("dup_lo", dup_lo_evictable)):
        print(f"  {name} p90 {pct(arr, .9):,.0f} ({pct(arr, .9) / L2:.1%}) p25 {pct(arr, .25):,.0f}")
    busy = [g for g in grid if g[1] > 0]
    print(f"busy-only (batch logged): n={len(busy)} dup_lo mean {st.mean(L1 - avail_hi - g[1] for g in busy):,.0f}; idle points dup = L1-avail = {L1 - avail_hi:,}")
    return dict(arm=arm, dup_mid=st.mean(dup_mid), dup_lo=st.mean(dup_lo_evictable), dup_hi=st.mean(dup_hi))


if __name__ == "__main__":
    for arm in sys.argv[1:] or RUNS:
        main(arm)


def extra(arm):
    """Host occupancy over time, private ctx at last turn, LRU stack-distance check."""
    rd, cd = RUNS[arm]
    evs = []
    with open(R + rd + "/server.log", errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            if "d2h_submit" in line:
                evs.append((ts(m[1]), +int(re.search(r"tokens=(\d+)", line)[1])))
            elif "HICACHE_EVT evict_host" in line:
                evs.append((ts(m[1]), -int(re.search(r"tokens=(\d+)", line)[1]), ))
            elif "HiCache prefetch success" in line:
                evs.append((ts(m[1]), +int(re.search(r"loaded=(\d+)", line)[1])))
    evs.sort()
    first_eh = None
    used = 0
    post = []
    with open(R + rd + "/server.log", errors="replace") as f:
        pass
    # find first evict_host
    for t, d in evs:
        used += d
        if d < 0 and first_eh is None:
            first_eh = t
        if first_eh is not None:
            post.append(used)
    print(f"--- {arm}: reconstructed host used after first evict_host: min {min(post):,} mean {st.mean(post):,.0f} max {max(post):,} (L2 {L2:,})")
    turns = []
    with open(R + rd + "/" + cd + "/client.jsonl") as f:
        for line in f:
            o = json.loads(line)
            if o.get("kind") == "run":
                t0 = o["t_start"]
            elif o.get("kind") == "turn":
                turns.append(o)
    last = {}
    for o in turns:
        if o["turn"] >= last.get(o["conv"], {"turn": -1})["turn"]:
            last[o["conv"]] = o
    priv = [o["prompt_tokens"] + o["completion_tokens"] - SHARED for o in last.values()]
    nt = [o["turn"] for o in last.values()]
    print(f"private ctx at each conv's last turn: n={len(priv)} mean {st.mean(priv):,.0f} p50 {pct(priv, .5):,.0f} p10 {pct(priv, .1):,.0f} p90 {pct(priv, .9):,.0f}; last-turn index mean {st.mean(nt):.1f}; convs with 40 turns: {sum(1 for x in nt if x == 39)}")
    live_final = SHARED + sum(priv)
    print(f"final live set {live_final:,} = {live_final / L2:.2f} x L2, {live_final / (L1 + L2):.2f} x (L1+L2)")
    # session-granularity LRU stack distance at each returning turn's send time
    by = sorted(turns, key=lambda o: o["t_send"])
    fin = sorted(((o["t_send"] + o["latency"], o["conv"], o["prompt_tokens"] + o["completion_tokens"]) for o in turns))
    tier = {}
    with open(R + f"analysis_gapscale/turn_components_x70 (campaign 7)_{arm}.csv") as f:
        for r in csv.DictReader(f):
            tier[(int(r["conv"]), int(r["turn"]))] = r["tier"]
    import bisect
    fts = [x[0] for x in fin]
    res = []
    for o in by:
        if o["turn"] == 0:
            continue
        t = o["t_send"]
        j = bisect.bisect_right(fts, t)
        lastacc, ctx = {}, {}
        for tf, c, sz in fin[:j]:
            lastacc[c] = tf
            ctx[c] = sz - SHARED
        if o["conv"] not in lastacc:
            continue
        me = lastacc[o["conv"]]
        dist = SHARED + sum(ctx[c] for c in lastacc if lastacc[c] >= me)
        res.append((dist, tier.get((o["conv"], o["turn"]), "?")))
    for C, name in ((L1, "L1"), (L2, "L2 (inclusive)"), (L2 + 650000, "L2+650K (exclusive est.)"), (L1 + L2, "L1+L2")):
        within = sum(1 for d, _ in res if d <= C)
        print(f"  capacity {name:26s} {C:>9,}: returning turns with session stack distance <= C: {within}/{len(res)} = {within / len(res):.1%}")
    # agreement of the L2 predictor with observed tier
    obs_miss = {"storage", "cold"}
    pm = [(d > L2, tg in obs_miss) for d, tg in res if tg != "?"]
    tp = sum(1 for a, b in pm if a and b); fp = sum(1 for a, b in pm if a and not b)
    fn = sum(1 for a, b in pm if not a and b); tn = sum(1 for a, b in pm if not a and not b)
    print(f"  predictor 'dist > L2' vs observed storage/cold tier: TP {tp} FP {fp} FN {fn} TN {tn}")
    pm2 = [(d > L2 + 650000, tg in obs_miss) for d, tg in res if tg != "?"]
    print(f"  turns observed storage/cold: {sum(1 for a, b in pm if b)}; of these with dist <= L2+650K (would stay in DRAM under exclusive): {sum(1 for a, b in pm2 if b and not a)}")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "extra":
    pass
