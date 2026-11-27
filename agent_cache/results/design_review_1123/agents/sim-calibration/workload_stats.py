"""Workload as replayed: per-request fields from a replay_programs.py client.jsonl (trace fields, same in every arm)."""
import json, sys, statistics as st, collections

def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else None

path = sys.argv[1]
reqs, plays = [], []
for line in open(path):
    if '"kind":"req"' in line:
        reqs.append(json.loads(line))
    elif '"kind":"play"' in line:
        plays.append(json.loads(line))
print(path.split('/')[-2], 'requests', len(reqs), 'plays finished', len(plays))
by_kind = collections.defaultdict(list)
for r in reqs:
    by_kind[r.get('gap') or 'first'].append(r['wait_s'])
for k, v in sorted(by_kind.items()):
    capped = sum(1 for x in v if x >= 1199.999)
    print(f"  gap {k:6s} n {len(v):6d} share {len(v)/len(reqs):.3f}  wait p10 {q(v,.1):8.2f} p50 {q(v,.5):8.2f} p90 {q(v,.9):8.2f} p99 {q(v,.99):8.2f} mean {st.mean(v):8.2f}  at cap {capped}")
ret = [r for r in reqs if r['k'] > 0]
first = [r for r in reqs if r['k'] == 0]
P = [r['prompt_sent'] for r in reqs]
newt = [r['prompt_sent'] - r['ideal_cached'] for r in ret]
outs = [r['out_target'] for r in reqs]
print(f"  prompt tokens: mean {st.mean(P):.0f} p10 {q(P,.1)} p50 {q(P,.5)} p90 {q(P,.9)} p99 {q(P,.99)} max {max(P)}")
print(f"  first requests: n {len(first)} prompt mean {st.mean(r['prompt_sent'] for r in first):.0f} p50 {q([r['prompt_sent'] for r in first],.5)}")
print(f"  returning: n {len(ret)}; new tokens (prompt - ideal cached): mean {st.mean(newt):.0f} p10 {q(newt,.1)} p50 {q(newt,.5)} p90 {q(newt,.9)} p99 {q(newt,.99)}")
big = [x for x in newt if x > 8192]
print(f"  returning with > 8192 new tokens (context rewrites): {len(big)} ({len(big)/len(ret):.3f}), mean {st.mean(big) if big else 0:.0f}")
ideal = [r['ideal_cached'] for r in ret]
print(f"  returning ideal cached (history): mean {st.mean(ideal):.0f} p50 {q(ideal,.5)} p90 {q(ideal,.9)}")
print(f"  output tokens: mean {st.mean(outs):.0f} p10 {q(outs,.1)} p50 {q(outs,.5)} p90 {q(outs,.9)} p99 {q(outs,.99)}")
roles = collections.Counter(r.get('role') for r in reqs)
print('  roles', dict(roles))
per_inst = collections.Counter((r['play'], r['stream']) for r in reqs)
print(f"  streams {len(per_inst)}, requests per stream mean {st.mean(per_inst.values()):.1f} p50 {q(list(per_inst.values()),.5)} p90 {q(list(per_inst.values()),.9)}")
per_play = collections.Counter(r['play'] for r in reqs)
print(f"  plays with a request {len(per_play)}, requests per play mean {st.mean(per_play.values()):.1f}")
slot = [p['start'] - p['arrival_s'] for p in plays if 'arrival_s' in p]
if slot:
    print(f"  finished plays: slot wait mean {st.mean(slot):.1f} max {max(slot):.1f}; session span mean {st.mean(p['end']-p['start'] for p in plays):.0f} s")
# hints
h = [r['hint'] for r in reqs if r.get('hint')]
if h:
    has_q = [x for x in h if x.get('p50') is not None]
    matched = collections.Counter(x.get('matched') for x in h)
    print(f"  hints {len(h)}; with p10/p50/p90 {len(has_q)}; matched top {matched.most_common(4)}; end-of-session (reply_key _end) {sum(1 for x in h if x.get('reply_key')=='_end')}")
    if has_q:
        rel = [x['true_gap_s'] / x['p50'] for x in has_q if x.get('true_gap_s') is not None and x['p50'] > 0]
        inside = sum(1 for x in has_q if x.get('true_gap_s') is not None and x['p10'] <= x['true_gap_s'] <= x['p90'])
        print(f"  true gap / hinted p50: p10 {q(rel,.1):.2f} p50 {q(rel,.5):.2f} p90 {q(rel,.9):.2f}; true gap inside [p10, p90]: {inside/len(has_q):.3f}")
o = [r.get('oracle_gap_s') for r in reqs if r.get('oracle_gap_s') is not None]
if o:
    print(f"  oracle hints {len(o)}: exact gap p50 {q(o,.5):.2f} p90 {q(o,.9):.2f}")
