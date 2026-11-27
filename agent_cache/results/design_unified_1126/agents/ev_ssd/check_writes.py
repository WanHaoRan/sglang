"""Compare the client-derived write model with the logged h2s_submit stream.
Model: at prefill end a request writes its prompt pages [cached/64, prompt//64); at its end, the output pages
[prompt//64, (prompt+completion-1)//64).  Usage: python3 -I -B check_writes.py <arm>"""
import sys, collections, bisect
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from c34lib import *
arm = sys.argv[1]
t_start, reqs, plays, args = load_client(arm)
ev = load_events(arm)
print(arm, {k: len(v) for k, v in ev.items()})
sub = ev["h2s_submit"]
tot_sub = sum(d["tokens"] for d in sub)
# model
mp = mo = 0
per_min_model = collections.Counter(); per_min_log = collections.Counter()
not_aligned = 0
for r in reqs:
    if r.get("ttft") is None:
        continue
    P, C, c = r["prompt_tokens"], r["completion_tokens"], r["cached_tokens"] or 0
    if c % 64:
        not_aligned += 1
    wp = (P // 64) * 64 - c
    wo = ((P + C - 1) // 64) * 64 - (P // 64) * 64
    mp += max(0, wp); mo += max(0, wo)
    per_min_model[int((r["t_send"] + r["ttft"]) // 60)] += max(0, wp)
    per_min_model[int((r["t_send"] + r["latency"]) // 60)] += max(0, wo)
for d in sub:
    per_min_log[int((d["t"] - t_start) // 60)] += d["tokens"]
print(f"h2s_submit: {len(sub)} ops, {tot_sub/1e6:.3f} M tokens; model: prompt {mp/1e6:.3f} M + output {mo/1e6:.3f} M = {(mp+mo)/1e6:.3f} M;"
      f" cached not page aligned: {not_aligned}")
print("minute: logged K tokens | model K tokens  (every 10th minute and the last ones)")
for m in sorted(set(per_min_log) | set(per_min_model)):
    if m % 10 == 0 or m >= 118:
        print(f"  {m:4d}: {per_min_log[m]/1e3:8.1f} | {per_min_model[m]/1e3:8.1f}")
d10 = [abs(sum(per_min_log[m + i] for i in range(10)) - sum(per_min_model[m + i] for i in range(10))) /
       max(1, sum(per_min_log[m + i] for i in range(10))) for m in range(0, 120, 10)]
print("relative gap per 10-minute window:", " ".join(f"{x:.3%}" for x in d10))
io = ev["h2s_io"]
skipped = sum(d.get("skipped", 0) for d in io) + sum(d["skipped"] for d in ev.get("h2s_group", []))
pages = sum(d["pages"] for d in io)
print(f"h2s_io: {len(io)} ops, {pages} pages, skipped as stored (fix 1): {skipped} pages")
cl = ev.get("cleanup", [])
for c in cl:
    t0 = c["t"] - c["secs"]
    wr = sum(d["pages"] for d in io if d["t"] <= t0)
    sk = sum(d.get("skipped", 0) for d in io if d["t"] <= t0) + sum(d["skipped"] for d in ev.get("h2s_group", []) if d["t"] <= t0)
    print(f"pass ending {c['t']-t_start:7.1f} s (took {c['secs']} s): {c['groups']} groups, {c['gib']} GiB, {c['pct0']}% -> {c['pct1']}%;"
          f" pages written before its start {wr} (skipped {sk}) -> new pages {wr - sk}")
