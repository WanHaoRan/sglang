"""Comparisons for a finite workload (ORACLE_FINDINGS.md): by default campaign 14's three_tier_wc, campaign 15's
three_tier_wc_oracle and the campaign 16 arms; --arm and --pair pick other runs (campaign 17: the x10 arms). The workload
is 768 users (~20,232 turns at x70) that the better arms finish before the 5 h stop, so a fixed time window holds
different parts of it per arm; these tables compare the same turns.

  python3 scripts/c16_summary.py [--results agent_cache/results] [--boot 2000] [--seed 0] [--no-server]
      [--arm LABEL CAMPAIGN ARM ...] [--pair LABEL_A LABEL_B ...]

  paired   TTFT over the (conv, turn) pairs every run completed: mean / p50 / p90 / p95 / p99 per run, and paired mean
           differences with a 95 % bootstrap interval that resamples users (turns of one user are correlated)
  makespan time at which the N-th user finished
  history  history tokens per returning turn served by device / host / SSD and recomputed (the client's
           cached_details and reply_reprefilled = uncached tokens beyond the turn's new messages)
  fairness per-user mean TTFT across users; TTFT by the gap slept before the turn
  phases   turns sent and mean TTFT per 30 min of client time
  lost     turns whose SSD read completed ("HiCache prefetch success ... loaded=N" in server.log) but whose admission
           credited less than half of it as storage and recomputed more than a page: the span was lost between landing
           and admission (reads the server logs; --no-server skips it)
"""
import argparse, collections, json, os, random, re, statistics as st

ARMS = [("c14 wc", "campaign14", "three_tier_wc"), ("c15 wc_oracle", "campaign15", "three_tier_wc_oracle"),
        ("c16 _dl", "campaign16", "three_tier_wc_dl"), ("c16 _oracle_v2", "campaign16", "three_tier_wc_oracle_v2"),
        ("c16 _oracle_v2_nu", "campaign16", "three_tier_wc_oracle_v2_nu")]
PAIRS = [("c16 _dl", "c14 wc"), ("c16 _oracle_v2", "c14 wc"), ("c16 _oracle_v2_nu", "c14 wc"),
         ("c16 _dl", "c15 wc_oracle"), ("c16 _oracle_v2", "c16 _dl"), ("c16 _oracle_v2_nu", "c16 _oracle_v2"),
         ("c15 wc_oracle", "c14 wc")]
SUCCESS = re.compile(r"HiCache prefetch success req=(\S+) completed=\d+ matched=\d+ loaded=(\d+)")


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def runs(results, arms=ARMS):
    """(label, run dir, client dir) per arm, from the campaign manifests (`<arm> <run dir> <client dir>`)."""
    out = []
    for label, campaign, arm in arms:
        with open(os.path.join(results, campaign, "manifest.txt")) as f:
            run, client = next(l.split()[1:3] for l in f if l.split() and l.split()[0] == arm)
        out.append((label, os.path.join(results, run), client))
    return out


def load(run, client):
    turns, ends, tag = {}, [], None
    with open(os.path.join(run, client, "client.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            k = d.get("kind")
            if k == "run":
                tag = d["tag"]
            elif k == "turn" and d.get("ttft") is not None:
                turns[(d["conv"], d["turn"])] = d
            elif k == "conv_end" and not d.get("aborted") and not d.get("capped"):   # capped: cut by the timeout
                ends.append(d["end_s"])
    return tag, turns, sorted(ends)


def boot_ci(by_user, n_boot, rng):
    sums = {u: (sum(v), len(v)) for u, v in by_user.items()}
    users, means = list(sums), []
    for _ in range(n_boot):
        s = n = 0
        for u in rng.choices(users, k=len(users)):
            s += sums[u][0]; n += sums[u][1]
        means.append(s / n)
    return pct(means, 0.025), pct(means, 0.975)


def paired(data, common, n_boot, rng, pairs):
    print(f"paired: {len(common)} (conv, turn) pairs completed in all {len(data)} runs "
          f"(per run: {', '.join(str(len(t)) for _, t, _ in data.values())})")
    print(f"  {'run':18s} {'mean':>6s} {'p50':>6s} {'p90':>6s} {'p95':>6s} {'p99':>6s}  TTFT (s) over the common turns")
    for name, (_, t, _) in data.items():
        v = [t[k]["ttft"] for k in common]
        print(f"  {name:18s} {st.mean(v):6.2f} {pct(v, .5):6.2f} {pct(v, .9):6.2f} {pct(v, .95):6.2f} {pct(v, .99):6.2f}")
    print("  paired mean difference (s), 95 % interval resampling users:")
    for x, y in pairs:
        by = collections.defaultdict(list)
        for k in common:
            by[k[0]].append(data[x][1][k]["ttft"] - data[y][1][k]["ttft"])
        lo, hi = boot_ci(by, n_boot, rng)
        print(f"    {x:18s} - {y:18s} {st.mean(v for vs in by.values() for v in vs):+7.2f}  [{lo:+.2f}, {hi:+.2f}]")


def makespan(data):
    ns = (384, 600, 700, 750, 760, 765, 766, 768)
    print("\nmakespan: client seconds at which the N-th user finished")
    print(f"  {'run':18s} " + " ".join(f"{n:>7d}" for n in ns))
    for name, (_, _, ends) in data.items():
        print(f"  {name:18s} " + " ".join(f"{ends[n - 1]:7.0f}" if len(ends) >= n else f"{'-':>7s}" for n in ns))


def history(data):
    print("\nhistory tokens per returning turn (all turns with a previous turn)")
    print(f"  {'run':18s} {'device':>7s} {'host':>7s} {'SSD':>7s} {'recomp':>7s}   shares %         turns recomputing > 1 page")
    for name, (_, t, _) in data.items():
        tot = {"device": 0, "host": 0, "storage": 0}
        rep = n = big = 0
        for d in t.values():
            if "reply_reprefilled" not in d:
                continue
            n += 1; rep += d["reply_reprefilled"]; big += d["reply_reprefilled"] > 64
            for k in tot:
                tot[k] += int((d.get("cached_details") or {}).get(k) or 0)
        allt = sum(tot.values()) + rep
        sh = " / ".join(f"{100 * v / allt:.0f}" for v in (*tot.values(), rep))
        print(f"  {name:18s} " + " ".join(f"{v / n:7.0f}" for v in (*tot.values(), rep)) + f"   {sh:16s} {100 * big / n:.1f} %")


def fairness(data, common):
    print("\nfairness: per-user mean TTFT over the common turns (s), across users")
    print(f"  {'run':18s} {'p50':>6s} {'p90':>6s} {'p99':>6s} {'max':>6s}   TTFT by gap slept before the turn: "
          "<60 s / 60-300 s / >=300 s (mean, p90)")
    for name, (_, t, _) in data.items():
        per, g = collections.defaultdict(list), {0: [], 1: [], 2: []}
        for k in common:
            per[k[0]].append(t[k]["ttft"])
            gs = t[k].get("gap_slept") or 0
            g[0 if gs < 60 else 1 if gs < 300 else 2].append(t[k]["ttft"])
        um = [st.mean(v) for v in per.values()]
        gtxt = " / ".join(f"{st.mean(v):.1f}, {pct(v, .9):.1f} (n={len(v)})" for v in g.values())
        print(f"  {name:18s} {pct(um, .5):6.1f} {pct(um, .9):6.1f} {pct(um, .99):6.1f} {max(um):6.1f}   {gtxt}")


def phases(data):
    edges = list(range(0, 18001, 1800))
    print("\nphases: turns sent / mean TTFT per 30 min of client time")
    print(f"  {'run':18s} " + " ".join(f"{lo // 60:>4d}m" + " " * 6 for lo in edges[:-1]))
    for name, (_, t, _) in data.items():
        cells = []
        for lo, hi in zip(edges, edges[1:]):
            v = [d["ttft"] for d in t.values() if lo <= d["t_send"] < hi]
            cells.append(f"{len(v):5d}/{st.mean(v):4.0f}s" if v else f"{'-':>11s}")
        print(f"  {name:18s} " + " ".join(cells))


def lost(data, paths):
    print("\nlost after landing: a completed SSD read whose span was gone by admission")
    for name, (tag, t, _) in data.items():
        loaded = collections.defaultdict(int)
        with open(os.path.join(paths[name], "server.log"), errors="replace") as f:
            for line in f:
                m = SUCCESS.search(line)
                if m:
                    loaded[m.group(1)] += int(m.group(2))
        rec = sum(max(d.get("reply_reprefilled", 0), 0) for d in t.values())
        hit = [d for (c, i), d in t.items() if d.get("reply_reprefilled", 0) > 64
               and loaded.get(f"{tag}-c{c}-t{i}", 0) > 2 * int((d.get("cached_details") or {}).get("storage") or 0)]
        lt = sum(d["reply_reprefilled"] for d in hit)
        print(f"  {name:18s} {len(hit):5d} turns, {lt / 1e6:5.1f}M of {rec / 1e6:5.1f}M recomputed history tokens "
              f"({100 * lt / max(rec, 1):.0f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-server", action="store_true", help="skip the server-log pass (lost after landing)")
    ap.add_argument("--arm", nargs=3, action="append", metavar=("LABEL", "CAMPAIGN", "ARM"),
                    help="a run to compare, by campaign directory and arm name (repeatable; default: campaigns 14-16)")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("LABEL_A", "LABEL_B"),
                    help="report A - B on the shared turns (repeatable; default: campaign 16's pairs)")
    a = ap.parse_args()
    arms = [tuple(x) for x in a.arm] if a.arm else ARMS
    rs = runs(a.results, arms)
    data = {name: load(run, client) for name, run, client in rs}
    common = set.intersection(*(set(t) for _, t, _ in data.values()))
    paired(data, common, a.boot, random.Random(a.seed), [tuple(x) for x in a.pair] if a.pair else PAIRS)
    makespan(data)
    history(data)
    fairness(data, common)
    phases(data)
    if not a.no_server:
        lost(data, {name: run for name, run, _ in rs})


if __name__ == "__main__":
    main()
