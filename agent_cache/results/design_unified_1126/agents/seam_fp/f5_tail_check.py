"""F5 check (CPU, read-only): is the text the empty arrival lookups ask for the previous reply's tail?

For planned turns whose first gate decision was 'read' and whose arrival lookup found nothing, compare
  chain - matched (ntp_gate: the plan key's tokens beyond the match in memory)
with the previous turn's output (client completion_tokens of the same user and stream, k - 1; the key is
page-aligned, so the two agree to within a 64-token page when the missing part is exactly the reply the replay does
not send back). Way B of OPEN_ITEMS section 1 (cut the key to what the prompt shares with it) removes exactly those.
Usage: python3 f5_tail_check.py <label>"""
import collections, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib

label = sys.argv[1]
r = lib.Run(label)
prev = {}
for rid, cl in r.client.items():
    key = (cl.get("play"), cl.get("stream"), cl.get("k"))
    prev[key] = cl
rows = []
for rid, rec in r.requests():
    gates = rec.get("ntp_gate", [])
    qs = rec.get("s2h_query", [])
    cl = r.client.get(rid)
    if not gates or not qs or cl is None or gates[0]["decision"] != "read":
        continue
    first_q = qs[0]
    if first_q["storage_hit"] >= 256:
        continue
    p = prev.get((cl.get("play"), cl.get("stream"), (cl.get("k") or 0) - 1))
    if p is None:
        continue
    gap_tokens = gates[0]["chain"] - gates[0]["matched"]
    rows.append((gap_tokens, p.get("completion_tokens") or 0, first_q["tokens_req"], cl.get("ideal_cached", 0) - gates[0]["matched"]))
print(f"== {label}: {len(rows)} planned turns whose arrival lookup (gate 'read') found nothing")
d = [g - c for g, c, _, _ in rows]
within = sum(abs(x) <= 64 for x in d)
print(f"   (chain - matched) - previous output tokens: median {lib.q(d,.5):+.0f}, p10 {lib.q(d,.1):+.0f}, p90 {lib.q(d,.9):+.0f}; within one page (64) for {within}/{len(rows)} ({100*within/max(1,len(rows)):.1f} %)")
print(f"   chain - matched: median {lib.q([g for g,_,_,_ in rows],.5)}, p90 {lib.q([g for g,_,_,_ in rows],.9)}; tokens the lookup asked for: median {lib.q([t for _,_,t,_ in rows],.5)}")
hist_missing = [h for _, _, _, h in rows]
real = sum(h >= 256 for h in hist_missing)
print(f"   of them, history really missing from memory at the gate (client ideal_cached - matched >= 256): {real} ({100*real/max(1,len(rows)):.1f} %)")
