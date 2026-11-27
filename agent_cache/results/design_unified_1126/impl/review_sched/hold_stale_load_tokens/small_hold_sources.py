"""For each hold whose own copy is under MIN_LOAD_TOKENS, look for the stale source.

With no retractions, a stale host_loaded_length can only come from a chunked request
whose first chunk loaded in an earlier, unheld batch (chunked=True) and whose last chunk
rides in the held batch. Reports, per small-copy hold, the most recent earlier loading
request with tokens >= MIN_LOAD_TOKENS - copy whose batch was not held, and how many
prefill batches ran in between ("Prefill batch" lines, logged after each forward).

usage: python3 small_hold_sources.py <server.log>
"""

import re
import sys

MIN_LOAD_TOKENS = 8192
TS = r"\[(?P<ts>[0-9-]+ [0-9:.]+)\]"
SUBMIT = re.compile(TS + r" HICACHE_EVT h2d_submit nodes=\d+ tokens=(?P<tok>\d+)")
INIT = re.compile(TS + r" HICACHE_EVT load_back_init rid=(?P<rid>\S+) tokens=(?P<tok>\d+)")
HOLD = re.compile(TS + r" HICACHE_EVT load_hold steps=(?P<steps>\d+)")
PREFILL = re.compile(TS + r" Prefill batch.*#new-seq: (?P<seq>\d+), #new-token: (?P<new>\d+), #cached-token: (?P<cached>\d+)")
STATS = re.compile(r"ReqTimeStats\(rid=(?P<rid>\S+), input_len=(?P<inp>\d+), cached_input_len=(?P<cached>\d+)")


def main(path):
    events = []  # (kind, ts, data)
    pending = []
    stats = {}
    with open(path, errors="replace", newline="\n") as f:
        for line in f:
            m = STATS.search(line)
            if m:
                stats[m["rid"]] = (int(m["inp"]), int(m["cached"]))
                continue
            m = INIT.search(line)
            if m:
                pending.append((m["rid"], int(m["tok"])))
                continue
            m = SUBMIT.search(line)
            if m:
                events.append(["submit", m["ts"], int(m["tok"]), pending, False])
                pending = []
                continue
            m = HOLD.search(line)
            if m:
                # the last submit is the held batch's copy
                for ev in reversed(events):
                    if ev[0] == "submit":
                        ev[4] = True
                        break
                events.append(["hold", m["ts"], int(m["steps"])])
                continue
            m = PREFILL.search(line)
            if m:
                events.append(["prefill", m["ts"], int(m["seq"]), int(m["new"]), int(m["cached"])])
    submits = [i for i, ev in enumerate(events) if ev[0] == "submit"]
    explained = 0
    small = 0
    for k, i in enumerate(submits):
        ev = events[i]
        if not ev[4] or ev[2] >= MIN_LOAD_TOKENS:
            continue
        small += 1
        need = MIN_LOAD_TOKENS - ev[2]
        source = None
        for j in reversed(submits[:k]):
            prev = events[j]
            cands = [(rid, tok) for rid, tok in prev[3] if tok >= need]
            if cands and not prev[4]:
                between = sum(1 for e in events[j:i] if e[0] == "prefill")
                source = (prev[1], cands, between)
                break
            if prev[4]:
                break  # an earlier held batch ran in between; stop looking further back
        if source is not None:
            explained += 1
            ts, cands, between = source
            extra = [(rid, tok, stats.get(rid)) for rid, tok in cands]
            print(f"hold copy={ev[2]:5d} @{ev[1]}  <- unheld load {extra} @{ts}, prefill results between={between}")
        else:
            print(f"hold copy={ev[2]:5d} @{ev[1]}  <- no unheld earlier load >= {need}")
    print(f"small-copy holds={small} with an unheld earlier load big enough={explained}")


if __name__ == "__main__":
    main(sys.argv[1])
