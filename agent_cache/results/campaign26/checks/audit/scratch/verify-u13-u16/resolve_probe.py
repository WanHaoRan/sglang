"""Quick probe (read-only, verifier U13): rule 1's logged start decisions re-solved with another read time.
Run from agent_cache/:  nice -n 10 python3 -I -B <this file> label=run [label=run ...]
For every ec_start line (one per change of decision) the ENGINE's own read_start (next_turn_policy.py, executed with a
stand-in for msgspec.Struct) is run again on the entry the client sent for that turn (client.jsonl dist_hint, joined on
the request id), the logged idle time, queue wait and price.
Known gaps of this probe: the rent uses the logged missing tokens (the engine charges every token of the session not in
GPU memory, which is not logged), the arrival offset is its 0.10 s seed, and each line is re-solved at its own instant
only (a decision that changes would also change the later ones). The first column says how often the logged decision is
reproduced with the logged read time; read the other columns only as far as that rate allows. Not a verified analysis."""
import collections, glob, json, math, os, re, sys, types


class _Struct:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

    def __init__(self, *args, **kwargs):
        names = [n for klass in reversed(type(self).__mro__) for n in vars(klass).get("__annotations__", {})]
        for name, value in zip(names, args):
            object.__setattr__(self, name, value)
        for name in names[len(args):]:
            value = kwargs[name] if name in kwargs else getattr(type(self), name)
            object.__setattr__(self, name, value)


stub = types.ModuleType("msgspec")
stub.Struct = _Struct
sys.modules["msgspec"] = stub
SRC = "/home/wanhr/sglang/python/sglang/srt/mem_cache/next_turn_policy.py"
ns = {"__name__": "next_turn_policy_probe"}
exec(compile(open(SRC).read(), SRC, "exec"), ns)
GapDistribution, remaining_gaps, read_start = ns["GapDistribution"], ns["remaining_gaps"], ns["read_start"]

EV = re.compile(r"HICACHE_EVT ec_start rid=(\S+) decision=(\w+) tokens=(\d+) start_ms=(\S+) idle_ms=(\S+) read_ms=(\S+) wait_ms=(\S+) price=(\S+)")
OFFSET_S = 0.10
MARGIN_S = 1.0
BYTES_PER_TOKEN, BETA = 98304, 2.11e9
kind = lambda start: "arrival" if start is None else ("now" if start == 0.0 else "wait")

for name, run in (a.split("=") for a in sys.argv[1:]):
    hint = {}
    for line in open(sorted(glob.glob(os.path.join("results", run, "client_*", "client.jsonl")))[-1]):
        if '"dist_hint"' in line:
            h = json.loads(line).get("dist_hint") or {}
            if h.get("next_rid"):
                hint[h["next_rid"]] = h
    rows, no_entry = [], 0
    with open(os.path.join("results", run, "server.log"), errors="replace") as f:
        for line in f:
            if "ec_start" not in line:
                continue
            m = EV.search(line)
            if not m:
                continue
            h = hint.get(m[1])
            if not h or h.get("p50") is None:
                no_entry += 1
                continue
            rows.append((m[2], int(m[3]), float(m[5]) / 1000, float(m[6]) / 1000, float(m[7]) / 1000, float(m[8]), h))
    variants = {"logged read time": lambda t, r: r, "minus the 1 s margin": lambda t, r: max(0.05, r - MARGIN_S),
                "minus 1.9 s (measured)": lambda t, r: max(0.05, r - 1.9),
                "deck r = bL/beta": lambda t, r: t * BYTES_PER_TOKEN / BETA}
    table = {v: collections.Counter() for v in variants}
    for decision, tokens, idle, read_s, wait_s, price, h in rows:
        dist = GapDistribution.from_quantiles(p10=h["p10"], p50=h["p50"], p90=h["p90"])
        remaining = remaining_gaps(support=dist.support(), idle_s=idle - OFFSET_S)
        for v, fn in variants.items():
            start = read_start(remaining=remaining, read_s=fn(tokens, read_s), wait_s=wait_s, rent_per_s=price * tokens)
            table[v][(decision, kind(start))] += 1
    n = len(rows)
    same = sum(c for (a, b), c in table["logged read time"].items() if a == b)
    print(f"{name} {run}: ec_start lines with an entry {n} (without one {no_entry}); logged decision reproduced with the logged read time in {same} ({100 * same / max(1, n):.1f} %)")
    for v in variants:
        t = table[v]
        cells = " | ".join(f"logged {a:7s} -> " + ", ".join(f"{b} {t[(a, b)]}" for b in ("now", "wait", "arrival")) for a in ("now", "wait", "arrival"))
        print(f"  {v:22s} {cells}")
    # The logged "now" lines again: with the other read time, would the read still start before its turn arrives?
    for v, fn in variants.items():
        still, late, never, delay, lead_s = 0, 0, 0, [], []
        for decision, tokens, idle, read_s, wait_s, price, h in rows:
            if decision != "now" or h.get("true_gap_s") is None:
                continue
            dist = GapDistribution.from_quantiles(p10=h["p10"], p50=h["p50"], p90=h["p90"])
            remaining = remaining_gaps(support=dist.support(), idle_s=idle - OFFSET_S)
            r = fn(tokens, read_s)
            start = read_start(remaining=remaining, read_s=r, wait_s=wait_s, rent_per_s=price * tokens)
            left = h["true_gap_s"] + OFFSET_S - idle   # seconds until the turn in fact arrives
            if start is None:
                never += 1
            elif start < left:
                still += 1
                delay.append(start)
                lead_s.append(left - start)
            else:
                late += 1
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
        print(f"  logged 'now' lines under {v:22s}: still started before the arrival {still}, the arrival comes first {late}, left to arrival {never}"
              f" | start delayed by p50 {q(delay, .5):.2f} p90 {q(delay, .9):.2f} s | started this long before the arrival p10 {q(lead_s, .1):.1f} p50 {q(lead_s, .5):.1f} p90 {q(lead_s, .9):.1f} s")
    reads = sorted(r[3] for r in rows if r[0] == "now")
    deck = sorted(r[1] * BYTES_PER_TOKEN / BETA for r in rows if r[0] == "now")
    if reads:
        print(f"  'now' lines: read time used p50 {reads[len(reads) // 2]:.2f} s; deck r for the same tokens p50 {deck[len(deck) // 2]:.2f} s")
