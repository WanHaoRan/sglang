"""Read-only: the shrank lines that came with the GPU pool under 0.90 at the last batch line, with the log around them."""
import re, sys, collections
RES = "/home/wanhr/sglang/agent_cache/results"
PRE = re.compile(r"(?:Prefill|Decode) batch.*token usage: ([\d.]+)")
SHR = re.compile(r"device prefix shrank before admission (?:rid|req)=\S+?-(c\d+-t\d+) lookup_match=(\d+) current_match=(\d+)")
for label, run in (a.split("=") for a in sys.argv[1:]):
    pool, n, low = None, 0, []
    hist = collections.deque(maxlen=2500)
    for i, line in enumerate(open(f"{RES}/{run}/server.log", errors="replace"), 1):
        hist.append((i, line))
        if p := PRE.search(line):
            pool = float(p[1]); continue
        if s := SHR.search(line):
            n += 1
            if pool is not None and pool < 0.90:
                k = s[1]
                mine = [(j, l.rstrip()[:230]) for j, l in hist if k + " " in l or k + "\n" in l or k + "," in l]
                ev = [(j, l.rstrip()[:120]) for j, l in hist if "evict_device" in l and mine and j >= mine[0][0]]
                low.append((i, k, pool, int(s[2]), int(s[3]), mine[-8:], ev[-6:], len(ev)))
    print(f"\n{label}: {n} shrank lines, {len(low)} with the pool under 0.90")
    for i, k, pool, a, b, mine, ev, nev in low:
        print(f"  line {i} {k} pool {pool:.2f} match {a} -> {b}; device evictions since the turn's first line in the window: {nev}")
        for j, l in mine:
            print(f"      {j}: {l}")
        for j, l in ev[-3:]:
            print(f"      evict {j}: {l}")
