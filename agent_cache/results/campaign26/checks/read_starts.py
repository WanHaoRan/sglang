"""When a pricing arm starts its SSD reads: rule 1's start decisions, the price log and the hold on arrived turns' reads.
From agent_cache/:  nice -n 10 python3 -I -B results/campaign26/checks/read_starts.py label=run [label=run ...]
Per run, from server.log's ec_start, ec_price and ntp_hold lines: the decisions by kind (now: read started before its
turn arrived; wait: timed for later; arrival: left to the turn's arrival) with the estimates they were taken on; the
price's range with the most tokens read pins held, staged and refused and the harm charged; the reads held after
arrival. Times are seconds since the client started. A run without pricing prints its holds only."""
import datetime as dt, json, os, re, sys

TS = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_start|ec_price|ntp_hold) (.*)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
fields = lambda text: dict(part.split("=", 1) for part in text.split() if "=" in part)

for name, run in (a.split("=") for a in sys.argv[1:]):
    client = sorted(d for d in os.listdir(f"results/{run}") if d.startswith("client_"))[-1]
    rows = (json.loads(line) for line in open(f"results/{run}/{client}/client.jsonl") if line.strip())
    t_start = next(r["t_start"] for r in rows if r.get("kind") == "run")
    starts, prices, holds = [], [], []
    for line in open(f"results/{run}/server.log", errors="replace"):
        m = TS.match(line)
        if m:
            {"ec_start": starts, "ec_price": prices, "ntp_hold": holds}[m.group(2)].append((stamp(m.group(1)) - t_start, fields(m.group(3))))
    print(f"{name} {run}")
    for kind in sorted({kv["decision"] for _, kv in starts}):
        sub = [(t, kv) for t, kv in starts if kv["decision"] == kind]
        read, wait = ([float(kv[k]) / 1000 for _, kv in sub] for k in ("read_ms", "wait_ms"))
        print(f"  start {kind:8s} {len(sub):5d} at {min(t for t, _ in sub):5.0f}-{max(t for t, _ in sub):5.0f} s | read s p10 p50 p90 {q(read, .1):5.2f} "
              f"{q(read, .5):5.2f} {q(read, .9):5.2f} | queue wait after arrival s {q(wait, .1):5.2f} {q(wait, .5):5.2f} {q(wait, .9):5.2f} | "
              f"read shorter than the wait in {sum(r < w for r, w in zip(read, wait))}")
    if prices:
        col = lambda k: [float(kv[k]) for _, kv in prices]
        print(f"  price lines {len(prices)}: price {min(col('price')):.3g} to {max(col('price')):.3g} | most tokens held by read pins {max(col('held')):.0f}, "
              f"staged {max(col('staged')):.0f}, refused {max(col('refused')):.0f} | harm charged {sum(col('harm_ms')) / 1000:.1f} s | "
              f"room at the end {prices[-1][1]['room']}")
    if holds:
        held, wait, load = ([float(kv[k]) / 1000 for _, kv in holds] for k in ("hold_ms", "wait_ms", "load_ms"))
        print(f"  reads held after arrival {len(holds)}: held s p10 p50 p90 {q(held, .1):5.1f} {q(held, .5):5.1f} {q(held, .9):5.1f} | expected wait s p50 "
              f"{q(wait, .5):5.1f} | load estimate s p50 p90 {q(load, .5):5.1f} {q(load, .9):5.1f}")
