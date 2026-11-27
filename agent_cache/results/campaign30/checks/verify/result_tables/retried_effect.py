"""Continuum's numbers with the four retried first requests (plays 216-219) placed at their original send (the session
start; play 218 has no play record, so retry time - 3600 s) with TTFT = retry time - original send + retry TTFT.
nice -n 10 python3 -I -B retried_effect.py > retried_effect.out"""
import json, os
RES = "/home/wanhr/sglang/agent_cache/results"
def load(rel):
    return [json.loads(x) for x in open(os.path.join(RES, rel, "client.jsonl")) if x.strip()]
c = load("20261124_203658_continuum_continuum/programs_203842_continuum")
e = load("20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/programs_093918_ec_full")
plays = {r["play"]: r for r in c if r["kind"] == "play"}
ok = [r for r in c if r["kind"] == "req" and r.get("phase") == "profile" and not r.get("error") and r.get("ttft") is not None]
cut = [dict(r) for r in ok if r["t_send"] < 6600]
fixed = []
for r in ok:
    if r.get("retried") and r["stream"] == 0 and r["k"] == 0 and r["t_send"] >= 6600:
        orig = plays[r["play"]]["start"] if r["play"] in plays else r["t_send"] - 3600
        fixed.append(dict(r, t_send=orig, ttft=r["t_send"] - orig + r["ttft"]))
for f in fixed:
    print(f"play {f['play']}: original send {f['t_send']:.3f} s, true TTFT {f['ttft']:.2f} s")
m = lambda v: sum(v) / len(v)
both = cut + fixed
print(f"all before the cut: {len(cut)} mean {m([r['ttft'] for r in cut]):.3f} -> {len(both)} mean {m([r['ttft'] for r in both]):.3f}")
fr = lambda rs: [r["ttft"] for r in rs if r["stream"] == 0 and r["k"] == 0]
print(f"first requests: {len(fr(cut))} mean {m(fr(cut)):.2f} -> {len(fr(both))} mean {m(fr(both)):.2f}")
w = lambda rs: [r["ttft"] for r in rs if 1800 <= r["t_send"] < 3600]
print(f"minutes 30-60: {len(w(cut))} mean {m(w(cut)):.2f} -> {len(w(both))} mean {m(w(both)):.2f}")
ef = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in e if r["kind"] == "req" and r.get("phase") == "profile"
      and not r.get("error") and r.get("ttft") is not None and r["t_send"] < 6600}
for label, rs in (("as recorded", cut), ("with the four", both)):
    cc = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in rs}
    keys = ef.keys() & cc.keys()
    print(f"continuum - ec_full {label}: n {len(keys)} diff {m([cc[k] - ef[k] for k in keys]):+.3f}")
