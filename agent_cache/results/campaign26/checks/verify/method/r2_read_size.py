"""'A read loads 17-19K tokens' (DECISIONS.md, Where the wait is): tokens loaded per 'HiCache prefetch success' line
(one per read) against the per-turn sum compare_ec.py prints, own returning kept turns, from the cache.
  nice -n 19 python3 -I -B r2_read_size.py"""
import collections, os, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arm, mean

print("arm: success lines of own returning kept turns | tokens per read: mean, median | turns with a landed read | per-turn sum: mean (compare_ec.py's column), median | per-turn largest single read: mean | pages per s2h_io line x 64: mean")
for a in ("ec_r13", "nofc", "ec_r1", "nofc_noend", "jit_exact"):
    d = arm(a)
    per = collections.defaultdict(list)
    for t, body in d["server"]["misc"].get("succ", []):
        m = re.search(r"req=\S+?-c(\d+)-t(\d+) completed=\d+ matched=\d+ loaded=(\d+)", body)
        per[(int(m[1]), int(m[2]))].append(int(m[3]))
    own = [k for k in d["by"] if k[1] > 0 and sum(per.get(k, [])) > 0]
    reads = [x for k in own for x in per[k]]
    sums = [sum(per[k]) for k in own]
    io = [p * 64 for k in own for _, _, p in d["ios"].get(k, [])]
    print(f"  {a:11s} {len(reads):5d} | {mean(reads):7.0f} {st.median(reads):7.0f} | {len(own):5d} | {mean(sums):7.0f} {st.median(sums):7.0f} | {mean(max(per[k]) for k in own):7.0f} | {mean(io):7.0f}")
