import sys, runpy
# reuse cleaner_hits2 definitions without printing: exec the module body up to the helpers
src = open(sys.argv[1]).read().split('CL = (')[0]
sys.argv = [sys.argv[1]] + sys.argv[2:5]
g = {}
exec(compile(src, 'cleaner_hits2', 'exec'), g)
want = sys.argv[-1] if False else None
reqs, classify, first = g['reqs'], g['classify'], g['first']
import collections
out = collections.defaultdict(list)
for rid, r in reqs.items():
    if first <= r['t'] < 6600 and r['ideal'] - r['cached'] >= 1024:
        c, best = classify(rid, r)
        out[c].append((round(r['t']), rid, r['ideal'], r['cached'], r['storage'], best and (round(best[0]), best[1], best[2])))
for c in ('full', 'part', 'nolook'):
    for e in sorted(out[c])[:6]:
        print(c, e)
