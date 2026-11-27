"""Four smaller checks behind the Retain / Preempt notes; each prints one block.

  python3 -I -B more_stats.py false_ends   <client.jsonl> <ext/arm.evt.gz>
      sessions the engine registered as ended (nue_register ... next_use_ms=ended) against the trace's explicit
      end: how many sent their next turn anyway, why, when, and how soon it came (bug X1).
  python3 -I -B more_stats.py queued_loss  <ext/arm.evt.gz>
      per planned turn, the arrival gate's first lookup (matched m1 of chain c) against each later lookup (m2);
      m2 < m1 means history resident at the first lookup was evicted while the turn waited (what P1 protects).
  python3 -I -B more_stats.py rereads      <ext/arm.evt.gz>
      arrival gate decisions and the tokens missing at them, per 30 minutes, with the SSD tokens read.
  python3 -I -B more_stats.py p2_whatif    <census extract> [...]
      on the review's census lines (design_review_1123/agents/designer-storage/census_*.txt), how often the
      reserve rule would hold timed reads, with leaf + cascade room and with leaf-only room.
"""
import collections
import gzip
import json
import re
import sys
from datetime import datetime, timezone

KV = re.compile(r"(\w+)=([^\s]+)")
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def open_text(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, errors="replace")


def ts(line):
    return datetime.strptime(TS.match(line).group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def events(path, wanted):
    for line in open_text(path):
        if "HICACHE_EVT " not in line:
            continue
        ev, _, rest = line.split("HICACHE_EVT ", 1)[1].partition(" ")
        if ev in wanted:
            yield ts(line), ev, dict(KV.findall(rest))


def false_ends(client_path, evt_path):
    recs, t_start = {}, None
    for line in open(client_path):
        if '"kind":"run"' in line:
            t_start = json.loads(line)["t_start"]
        elif '"kind":"req"' in line:
            r = json.loads(line)
            recs[r["rid"]] = r
    ended, expired = {}, set()
    for t, ev, f in events(evt_path, ("nue_register", "ntp_expire")):
        if ev == "ntp_expire":
            expired.add(f["rid"])
        elif f.get("next_use_ms") == "ended":
            ended.setdefault(f["rid"], t - t_start)
    kinds = collections.Counter()
    per10 = collections.Counter()
    gaps = []
    for rid, t in ended.items():
        m = re.search(r"-t(\d+)$", rid)
        nxt = re.sub(r"-t\d+$", f"-t{int(m.group(1)) + 1}", rid)
        hint = (recs.get(rid) or {}).get("hint") or {}
        if hint.get("reply_key") == "_end":
            kinds["explicit end"] += 1
        elif nxt in recs:
            why = "plan expired" if rid in expired else ("no entry for the reply key" if hint.get("matched") is None else "other")
            kinds[f"next turn came: {why}"] += 1
            per10[(why, int(t // 600) * 10)] += 1
            if hint.get("true_gap_s") is not None:
                gaps.append(hint["true_gap_s"])
        else:
            kinds["no next turn in the record (run end, abort, exact hint's last turn)"] += 1
    print(f"{evt_path}: sessions registered as ended {len(ended)}: {dict(kinds)}")
    print("  next turn came, per 10 minutes: " + ", ".join(f"{why}@{m}: {n}" for (why, m), n in sorted(per10.items())))
    if gaps:
        print(f"  the gap that followed: p10 {q(gaps, .1):.1f} s, p50 {q(gaps, .5):.1f}, p90 {q(gaps, .9):.1f} (n={len(gaps)})")


def queued_loss(evt_path):
    t0, first = None, {}
    lost_read = collections.defaultdict(list)
    lost_covered = collections.defaultdict(list)
    same = collections.defaultdict(list)
    arrived = collections.Counter()
    for t, ev, f in events(evt_path, ("ntp_gate", "ntp_arrive", "h2s_submit")):
        if t0 is None:
            t0 = t
        w = int((t - t0) // 1800)
        if ev == "ntp_arrive":
            arrived[w] += 1
        elif ev == "ntp_gate":
            rid, m, c = f["rid"], int(f["matched"]), int(f["chain"])
            if rid not in first:
                first[rid] = (m, f["decision"], t)
                continue
            m1, d1, t1 = first[rid]
            if m < m1:
                (lost_covered if d1 == "covered" else lost_read)[w].append((m1 - m, rid, t - t1))
            else:
                same[w].append(c - m)
    print(evt_path)
    for w in sorted(arrived):
        a, b, s = lost_read[w], lost_covered[w], same[w]
        both = a + b
        print(f"  min {w * 30:3d}-{w * 30 + 30:3d}: planned turns arrived {arrived[w]:5d}; turns whose resident history shrank before a "
              f"later lookup: {len({x[1] for x in both}):4d} ({len({x[1] for x in both}) / max(arrived[w], 1):.1%}), "
              f"{sum(x[0] for x in both) / 1e6:5.2f} M tokens (after a first read {len({x[1] for x in a})}, after 'covered' "
              f"{len({x[1] for x in b})}); lost per turn p50 {q([x[0] for x in both], .5)}; found p50 {q([x[2] for x in both], .5):.1f} s "
              f"after the first lookup; later lookups with no shrink {len(s)} (missing then p50 {q(s, .5)}, sum {sum(s) / 1e6:.2f} M)")


def rereads(evt_path):
    t0 = None
    gate = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    read = collections.Counter()
    for t, ev, f in events(evt_path, ("ntp_gate", "s2h_io", "h2s_submit")):
        if t0 is None:
            t0 = t
        w = int((t - t0) // 1800)
        if ev == "ntp_gate":
            g = gate[w][f["decision"]]
            g[0] += 1
            g[1] += int(f["chain"]) - int(f["matched"])
        elif ev == "s2h_io":
            read[w] += int(f["tokens"])
    print(evt_path)
    for w in sorted(read):
        g = gate[w]
        print(f"  min {w * 30:3d}-{w * 30 + 30:3d}: SSD tokens read {read[w] / 1e6:6.2f} M; gate read {g['read'][0]:5d} "
              f"({g['read'][1] / 1e6:6.2f} M missing), reread {g['reread'][0]:4d} ({g['reread'][1] / 1e6:5.2f} M), "
              f"capped {g['capped'][0]:3d} ({g['capped'][1] / 1e6:.2f} M), covered {g['covered'][0]:5d}")


def p2_whatif(paths):
    for path in paths:
        rows = []
        for line in open_text(path):
            d = dict(kv.split("=", 1) for kv in line.split("host_census ", 1)[1].split())
            if int(d["host_only"]) > 0:
                rows.append({k: int(v) for k, v in d.items() if k != "why"})
        n = len(rows)
        fixed = sum(1 for r in rows if r["pending_write"] >= 50000)
        print(f"{path}: {n} samples with host-only pages; pending_write >= 50,000 in {fixed} ({fixed / n:.1%}), max "
              f"{max(r['pending_write'] for r in rows)}")
        for alloc, write in ((10000, 4000), (20000, 4000), (20000, 7000), (20000, 21000)):
            full = sum(1 for r in rows if r["pending_write"] > 0
                       and r["free"] + r["evict_leaf"] + r["evict_casc"] < alloc * r["pending_write"] / write)
            leaf = sum(1 for r in rows if r["pending_write"] > 0
                       and r["free"] + r["evict_leaf"] < alloc * r["pending_write"] / write)
            print(f"  alloc {alloc / 1e3:.0f} K tokens/s, write {write / 1e3:.0f} K tokens/s: the rule holds in {full} samples "
                  f"({full / n:.1%}) with leaf + cascade room, {leaf} ({leaf / n:.1%}) with leaf-only room")
        res = [r["free"] + r["evict_leaf"] + r["evict_casc"] for r in rows]
        leaf_only = [r["free"] + r["evict_leaf"] for r in rows]
        print(f"  reserve, leaf + cascade: p10 {q(res, .1)}, p50 {q(res, .5)}, min {min(res)}; leaf only: p10 {q(leaf_only, .1)}, "
              f"p50 {q(leaf_only, .5)}, min {min(leaf_only)}; walk_ms p50 {q([r['walk_ms'] for r in rows], .5)}, max "
              f"{max(r['walk_ms'] for r in rows)}; n_nodes max {max(r['n_nodes'] for r in rows)}; n_leaves p50 "
              f"{q([r['n_leaves'] for r in rows], .5)}, max {max(r['n_leaves'] for r in rows)}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "false_ends":
        false_ends(sys.argv[2], sys.argv[3])
    elif what == "queued_loss":
        queued_loss(sys.argv[2])
    elif what == "rereads":
        rereads(sys.argv[2])
    elif what == "p2_whatif":
        p2_whatif(sys.argv[2:])
    else:
        sys.exit(__doc__)
