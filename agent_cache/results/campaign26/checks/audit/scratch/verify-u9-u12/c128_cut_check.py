"""Read-only: shared returning-turn sets of the five 128-user design runs under plot_pricing_vs_jit.py's rule and compare_c128.py's rule."""
import json, os, statistics as st
RES = "/home/wanhr/sglang/agent_cache/results"
MAN = [l.split() for l in open(os.path.join(RES, "compare_c26_designs_c128", "manifest.txt")) if l.strip()]
T = {}
for label, run, client in MAN:
    rows = [json.loads(l) for l in open(os.path.join(RES, run, client, "client.jsonl")) if l.strip()]
    T[label] = [r for r in rows if r.get("kind") == "turn" and r.get("ttft") is not None]
    last = max(r["t_send"] for r in T[label])
    print(f"{label:16s} turns {len(T[label]):6d} last send {last:9.1f} s; sent <6900: {sum(r['t_send'] < 6900 for r in T[label])}; sent <7200: {sum(r['t_send'] < 7200 for r in T[label])}")
def shared(rule):
    K = {}
    for a, t in T.items():
        last = max(r["t_send"] for r in t)
        K[a] = {(r["conv"], r["turn"]): r for r in t if rule(r, last)}
    s = set.intersection(*(set(v) for v in K.values()))
    return len(s), [round(st.mean(K[a][k]["ttft"] for k in s), 2) for a in T]
# plot_pricing_vs_jit.py client_turns(): t_send < END_S (7200), returning, not mid_start, t_send < last - 300
print("deck figure rule  :", shared(lambda r, last: r["t_send"] < 7200 and r["turn"] > 0 and not r.get("mid_start") and r["t_send"] < last - 300))
# compare_c128.py: t_send < 6900, returning
print("compare_c128 rule :", shared(lambda r, last: r["t_send"] < 6900 and r["turn"] > 0))
print("same + no mid_start:", shared(lambda r, last: r["t_send"] < 6900 and r["turn"] > 0 and not r.get("mid_start")))
print("record's wording  :", shared(lambda r, last: r["t_send"] < min(6900, last - 300) and r["turn"] > 0))
