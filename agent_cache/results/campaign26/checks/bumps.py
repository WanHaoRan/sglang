"""Rule 2 (soft pins): what the bumped staged reads cost, and what the turns they made room for got.
From agent_cache/:  nice -n 10 python3 -I -B results/campaign26/checks/bumps.py label=run [reference=run]
An ec_bump line is a staged read given up so that the read of a turn that has arrived finds room. For the first run:
the bumps and the SSD pages their reads had loaded, when the bumped turns arrived and whether they read again, and the
SSD pages by what became of the read. With a reference run (the same arm without soft pins): TTFT of the bumped
turns, of the turns the room went to and of the other returning turns, beside the same turns there.
Times are seconds; a turn is a (user, turn) pair, and only turns both runs recorded are set side by side."""
import collections, datetime as dt, json, os, re, statistics as st, sys

EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_bump|s2h_io|ntp_arrive|ntp_stage) rid=\S+?-c(\d+)-t(\d+)(.*)")
FOR = re.compile(r"for_rid=\S+?-c(\d+)-t(\d+) tokens=(\d+) chance=([\d.]+)")
PAGES = re.compile(r"pages=(\d+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
three = lambda v: f"{q(v, .1):.1f} {q(v, .5):.1f} {q(v, .9):.1f}"


def turns(run):
    """(user, turn) -> the client's record, for the turns that got a first token."""
    client = sorted(d for d in os.listdir(f"results/{run}") if d.startswith("client_"))[-1]
    rows = (json.loads(line) for line in open(f"results/{run}/{client}/client.jsonl") if line.strip())
    return {(r["conv"], r["turn"]): r for r in rows if r.get("kind") == "turn" and r.get("ttft") is not None}


def beside(name, keys, own, other):
    """Mean TTFT of the keys in the run, and against the reference on the keys both have."""
    have = [k for k in keys if k in own]
    line = f"  {name}: {len(have)} turns, mean TTFT {st.mean(own[k]['ttft'] for k in have):.2f} s, median {st.median(own[k]['ttft'] for k in have):.2f}"
    both = [k for k in have if k in other]
    if both:
        d = [own[k]["ttft"] - other[k]["ttft"] for k in both]
        line += (f" | on the {len(both)} also in the reference: {st.mean(own[k]['ttft'] for k in both):.2f} s against {st.mean(other[k]['ttft'] for k in both):.2f} s, "
                 f"paired {st.mean(d):+.3f} s (faster on {sum(x < -0.05 for x in d)}, slower on {sum(x > 0.05 for x in d)})")
    print(line)


(name, run), ref = sys.argv[1].split("="), (sys.argv[2].split("=") if len(sys.argv) > 2 else None)
bumps, arrive = [], {}
reads, stages = collections.defaultdict(list), collections.Counter()
for line in open(f"results/{run}/server.log", errors="replace"):
    m = EVT.match(line)
    if not m:
        continue
    t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
    if kind == "ec_bump":
        f = FOR.search(m.group(5))
        bumps.append((t, key, (int(f.group(1)), int(f.group(2))), int(f.group(3)), float(f.group(4))))
    elif kind == "s2h_io":
        reads[key].append((t, int(PAGES.search(m.group(5)).group(1))))
    elif kind == "ntp_arrive":
        arrive[key] = t
    else:
        stages[key] += 1
own = turns(run)
total = sum(p for v in reads.values() for _, p in v)
per_turn = collections.Counter(key for _, key, *_ in bumps)
print(f"{name} {run}")
print(f"  bumps {len(bumps)} of {sum(stages.values())} staged reads, on {len(per_turn)} turns ({sum(n > 1 for n in per_turn.values())} bumped more than once), "
      f"for {len({b[2] for b in bumps})} arrived turns; tokens given up {sum(b[3] for b in bumps):,} (median {st.median(b[3] for b in bumps):.0f} per bump); "
      f"chance of return within the read time p10 p50 p90 {q([b[4] for b in bumps], .1):.2f} {q([b[4] for b in bumps], .5):.2f} {q([b[4] for b in bumps], .9):.2f}")

# A read belongs to the bump that follows it: the pages it loaded were given up.
lost, sat, gap, late = 0, [], [], collections.Counter()
for t, key, _, _, _ in bumps:
    before = [(at, p) for at, p in reads[key] if at <= t]
    if before:
        lost += before[-1][1]
        sat.append(t - before[-1][0])
    after = [at for at, _ in reads[key] if at > t]
    if key not in arrive:
        late["never arrived"] += 1
        continue
    gap.append(arrive[key] - t)
    late["read again before arrival" if any(at <= arrive[key] for at in after) else "read at arrival" if after else "no further read"] += 1
print(f"  SSD pages their reads had loaded: {lost:,} ({100 * lost / total:.1f} % of the run's {total:,}); the data had sat {three(sat)} s (p10 p50 p90) when bumped")
print(f"  bump to the turn's arrival {three(gap)} s; within 1 s {sum(g <= 1 for g in gap)}, within 5 s {sum(g <= 5 for g in gap)} of {len(gap)}; afterwards: "
      + ", ".join(f"{k} {n}" for k, n in late.most_common()))

# Every read of the run by what became of it.
fate = collections.Counter()
for key, v in reads.items():
    cut = sorted(t for t, k, *_ in bumps if k == key)
    for at, p in v:
        staged = key not in arrive or at <= arrive[key]
        fate["at or after arrival" if not staged else "staged, then bumped" if any(t >= at for t in cut) else "staged and kept"] += p
print("  SSD pages by read: " + ", ".join(f"{k} {n:,} ({100 * n / total:.0f} %)" for k, n in fate.most_common()))

other = turns(ref[1]) if ref else {}
if ref:
    print(f"  reference {ref[0]} {ref[1]}")
bumped, helped = set(per_turn), {b[2] for b in bumps}
beside("bumped turns", bumped, own, other)
beside("turns the room went to", helped - bumped, own, other)
beside("other returning turns", {k for k in own if k[1] > 0} - bumped - helped, own, other)
