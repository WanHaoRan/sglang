"""Campaign 26: every pair of runs against each other on the turns they all share, and what a repeat of an arm moves.
From agent_cache/, on the host:
  nice -n 10 python3 -I -B frozen/oracle_scripts_staging/c26_ec/pairs_ec.py [arms...] | tee results/campaign26/pairs_ec.out
Default arms: the leading arms and their repeats that have landed. Uses compare_ec.py's loader for the turns kept.
The interval beside a difference is a block bootstrap over BLOCK consecutive positions of the oracle's send order: it says
whether the difference is larger than the turns' own scatter inside these two runs. It leaves out run-to-run spread, which
only a repeat of the same arm shows (the last two tables). Turns past the order run's last position count as "the other
positions"; they are the leading arms' last 15 minutes or so."""
import itertools, os, random, statistics as st, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LEAD = ["jit_exact", "jit_exact_d", "ec_r13", "ec_r13_rep", "ec_r134", "ec_full", "ec_r1", "nofc", "nofc_rep", "nofc_noend", "nofc_noend_rep"]
SAME = (("original HiCache", ("baseline", "base_rep")), ("hint-free", ("hintfree", "hintfree_ctl")), ("JIT p50", ("jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_p50_d")),
        ("JIT exact", ("jit_exact", "jit_exact_d")), ("rules 1 and 3", ("ec_r13", "ec_r13_rep")), ("no gap forecast", ("nofc", "nofc_rep")),
        ("no hints", ("nofc_noend", "nofc_noend_rep")))
BLOCK = 500
DRAWS = 2000
EPISODES = ((5500, 7000), (8750, 11000))
source = open(os.path.join(HERE, "compare_ec.py")).read()
ns = {"__name__": "compare_ec_loader"}
asked = sys.argv[1:]
sys.argv = sys.argv[:1]
exec(compile(source[:source.index("\nD = {a: load(")], "compare_ec.py", "exec"), ns)
RUNS, load, pct, client, rec, ORDER_RUN = (ns[k] for k in ("RUNS", "load", "pct", "client", "rec", "ORDER_RUN"))
ARMS = [a for a in (asked or LEAD) if a in RUNS]
W = max(len(a) for a in ARMS)
D = {a: load(RUNS[a]) for a in ARMS}
rank = {(r["conv"], r["turn"]): i for i, r in enumerate(client(ORDER_RUN)[1])}
# The order run stops near position 10,300: later turns take the first arm's own send order for their bootstrap block.
late = {k: i for i, k in enumerate(sorted((k for k in D[ARMS[0]]["by"] if k not in rank), key=lambda k: D[ARMS[0]]["by"][k]["t_send"]))}
keys = sorted(set.intersection(*(set(k for k in D[a]["by"] if k[1] > 0) for a in ARMS)), key=lambda k: rank.get(k, 1 << 30))
mean = lambda v: st.mean(v) if v else float("nan")
ttft = lambda a, k: D[a]["by"][k]["ttft"]
in_episode = lambda k: any(lo <= rank.get(k, -1) < hi for lo, hi in EPISODES)


def interval(a, b, ks):
    """Paired mean TTFT difference a - b on ks, with the 5th and 95th percentile of the block bootstrap."""
    blocks = {}
    for k in ks:
        s = blocks.setdefault(rank[k] // BLOCK if k in rank else ("late", late.get(k, 0) // BLOCK), [0.0, 0])
        s[0] += ttft(a, k) - ttft(b, k)
        s[1] += 1
    cells = list(blocks.values())
    rng = random.Random(0)
    draws = []
    for _ in range(DRAWS):
        pick = rng.choices(cells, k=len(cells))
        draws.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    return sum(c[0] for c in cells) / sum(c[1] for c in cells), pct(draws, .05), pct(draws, .95)


def matrix(title, ks):
    print(f"\n{title}: {len(ks)} turns; row minus column, seconds per turn [5 %, 95 % of the block bootstrap]")
    print("  " + " " * W + "".join(f" {b:>22s}" for b in ARMS[:-1]))
    for a in ARMS[1:]:
        cells = []
        for b in ARMS[:-1]:
            if ARMS.index(b) < ARMS.index(a):
                d, lo, hi = interval(a, b, ks)
                cells.append(f" {d:+6.3f} [{lo:+5.2f},{hi:+5.2f}]".rjust(23))
            else:
                cells.append(" " * 23)
        print(f"  {a:{W}s}" + "".join(cells))


print(f"{len(keys)} returning turns shared by {', '.join(ARMS)}")
print("\nper arm on them: mean TTFT, median, p90, p99, turns over 10 s | in the two load episodes, elsewhere | turns reading from SSD, SSD pages, recomputed tokens per turn")
episode = [k for k in keys if in_episode(k)]
calm = [k for k in keys if not in_episode(k)]
for a in ARMS:
    v = [ttft(a, k) for k in keys]
    print(f"  {a:{W}s} {mean(v):6.2f} {st.median(v):6.2f} {pct(v, .9):6.2f} {pct(v, .99):6.1f} {sum(x > 10 for x in v):5d} | "
          f"{mean([ttft(a, k) for k in episode]):6.2f} {mean([ttft(a, k) for k in calm]):6.2f} | "
          f"{sum(D[a]['pages'][k] > 0 for k in keys):5d} {sum(D[a]['pages'][k] for k in keys):8d} {mean([rec(D[a]['by'][k]) for k in keys]):6.0f}")

matrix("all shared returning turns", keys)
matrix(f"the two load episodes (positions {', '.join(f'{lo}-{hi}' for lo, hi in EPISODES)})", episode)
matrix("the other positions", calm)

print("\nthe same arm run again, on the turns shared here: paired mean difference later minus earlier run, [5 %, 95 %], in the episodes, elsewhere")
spread = {}
for name, runs in SAME:
    for a, b in itertools.combinations([r for r in runs if r in ARMS], 2):
        d, lo, hi = interval(b, a, keys)
        spread[(a, b)] = d
        print(f"  {name:16s} {b:{W}s} - {a:{W}s} {d:+6.3f} [{lo:+5.2f},{hi:+5.2f}] {interval(b, a, episode)[0]:+6.3f} {interval(b, a, calm)[0]:+6.3f}")
if not spread:
    print("  none of the arms named here has a repeat among them")

group = {r: name for name, runs in SAME for r in runs}
label = lambda a: group.get(a, a)
print("\nbetween arms, over every pair of their runs: paired mean difference, smallest and largest (pairs); the order holds across runs when both have one sign")
for x, y in itertools.combinations(dict.fromkeys(label(a) for a in ARMS), 2):
    d = [interval(b, a, keys)[0] for a in ARMS for b in ARMS if label(a) == x and label(b) == y]
    print(f"  {y:16s} - {x:16s} {min(d):+6.3f} {max(d):+6.3f} ({len(d)})")
