"""Pair each load-hold release with the H2D copy its held batch started.

Read-only over a server.log. While a batch is held, get_new_batch_prefill does not run,
so no start_loading happens between the hold and its release: the last h2d_submit before
a `load_hold` line is the held batch's copy. A hold whose copy is below MIN_LOAD_TOKENS
(8192) was triggered by tokens the copy does not contain (stale host_loaded_length).

usage: python3 holds_vs_copy.py <server.log> [<server.log> ...]
"""

import re
import sys

MIN_LOAD_TOKENS = 8192
TS = r"\[(?P<ts>[0-9-]+ [0-9:.]+)\]"
SUBMIT = re.compile(TS + r" HICACHE_EVT h2d_submit nodes=(?P<nodes>\d+) tokens=(?P<tok>\d+)")
INIT = re.compile(TS + r" HICACHE_EVT load_back_init rid=(?P<rid>\S+) tokens=(?P<tok>\d+)")
HOLD = re.compile(TS + r" HICACHE_EVT load_hold steps=(?P<steps>\d+) landed=(?P<landed>\d)")


def scan(path):
    holds = []
    pending_inits = []  # load_back_init lines since the last h2d_submit
    last_submit = None  # (ts, tokens, nodes, inits, line_no)
    with open(path, errors="replace") as f:
        for no, line in enumerate(f, 1):
            if "HICACHE_EVT" not in line:
                continue
            m = INIT.search(line)
            if m:
                pending_inits.append((m["rid"], int(m["tok"])))
                continue
            m = SUBMIT.search(line)
            if m:
                last_submit = (m["ts"], int(m["tok"]), int(m["nodes"]), pending_inits, no)
                pending_inits = []
                continue
            m = HOLD.search(line)
            if m:
                holds.append((no, m["ts"], int(m["steps"]), int(m["landed"]), last_submit))
                last_submit = None  # each copy belongs to at most one held batch
    return holds


def main():
    for path in sys.argv[1:]:
        holds = scan(path)
        small = [h for h in holds if h[4] is not None and h[4][1] < MIN_LOAD_TOKENS]
        orphan = [h for h in holds if h[4] is None]
        print(f"== {path}")
        print(f"holds={len(holds)} copy<{MIN_LOAD_TOKENS}={len(small)} no_copy_seen={len(orphan)}")
        if holds:
            steps_all = sum(h[2] for h in holds) / len(holds)
            print(f"mean steps held (all)={steps_all:.2f}")
        if small:
            steps_small = sum(h[2] for h in small) / len(small)
            print(f"mean steps held (copy<{MIN_LOAD_TOKENS})={steps_small:.2f}")
        for no, ts, steps, landed, sub in small[:40]:
            s_ts, tok, nodes, inits, s_no = sub
            print(
                f"  hold@L{no} {ts} steps={steps} landed={landed} | submit@L{s_no} "
                f"tokens={tok} nodes={nodes} inits={inits}"
            )


if __name__ == "__main__":
    main()
