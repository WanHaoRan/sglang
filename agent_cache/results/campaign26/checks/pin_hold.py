"""Host memory held by read pins, split by whether the read landed before its turn arrived.
From agent_cache/:  nice -n 10 python3 -I -B results/campaign26/checks/pin_hold.py label=run [label=run ...]
Per run, from server.log's srs_pin, srs_unpin and ntp_arrive lines: every pin is paired with the next unpin of the same
turn and counted as tokens x seconds held. An early pin is one held over EARLY s before its turn arrived (to its unpin
when the turn never came); its hold is split at the arrival into the wait for the turn and the wait in the queue. The
mean level is all token-seconds over the span from the first to the last pin event, and each part's share of it is
printed in tokens. Times are seconds; quantiles are p10 p50 p90."""
import collections, datetime as dt, re, sys

EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (srs_pin|srs_unpin|ntp_arrive) rid=\S+?-c(\d+)-t(\d+) (.*)")
TOKENS = re.compile(r"tokens=(\d+)")
REASON = re.compile(r"reason=(\w+)")
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
three = lambda v: f"{q(v, .1):.1f} {q(v, .5):.1f} {q(v, .9):.1f}"
EARLY = 0.05  # s; a read at arrival logs its pin within a few ms of the arrival line, on either side

for name, run in (a.split("=") for a in sys.argv[1:]):
    arrive, open_pins, holds, first, last = {}, {}, [], None, None
    for line in open(f"results/{run}/server.log", errors="replace"):
        m = EVT.match(line)
        if not m:
            continue
        t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
        if kind == "ntp_arrive":
            arrive[key] = t
            continue
        first, last = first or t, t
        if kind == "srs_pin":
            open_pins[key] = (t, int(TOKENS.search(m.group(5)).group(1)))
        elif key in open_pins:
            at, tokens = open_pins.pop(key)
            holds.append((key, at, t, tokens, REASON.search(m.group(5)).group(1)))
    span = last - first
    # Seconds a pin was held before its turn arrived; negative when it was taken after the arrival.
    wait = lambda h: min(h[2], arrive.get(h[0], h[2])) - h[1]
    early = [h for h in holds if wait(h) > EARLY]
    late = [h for h in holds if wait(h) <= EARLY]
    area = lambda hs: sum(tokens * (end - at) for _, at, end, tokens, _ in hs)
    before = sum(h[3] * wait(h) for h in early)
    waits = [wait(h) for h in early]
    reasons = collections.Counter(h[4] for h in early)
    print(f"{name} {run}")
    print(f"  pins {len(holds)} paired ({len(open_pins)} still held at the end), mean level {area(holds) / span:,.0f} tokens over {span / 60:.0f} min")
    print(f"  early pins {len(early)} on {len({h[0] for h in early})} turns: {three([h[3] for h in early])} tokens; held before the arrival {three(waits)} s, "
          f"over 30 s {sum(w > 30 for w in waits)}, over 120 s {sum(w > 120 for w in waits)}; ended by " + ", ".join(f"{k} {n}" for k, n in reasons.most_common()))
    print(f"    token-seconds: before the arrival {before / 1e6:,.1f} M ({before / span:,.0f} tokens of the mean level), after it {(area(early) - before) / 1e6:,.1f} M ({(area(early) - before) / span:,.0f})")
    print(f"  the other pins {len(late)}: held {three([h[2] - h[1] for h in late])} s; token-seconds {area(late) / 1e6:,.1f} M ({area(late) / span:,.0f} tokens of the mean level)")
