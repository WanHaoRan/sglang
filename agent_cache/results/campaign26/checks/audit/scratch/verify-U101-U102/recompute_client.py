"""Verifier's own client-side recomputation of headline TTFT numbers for campaign 26 (read-only).
Own parser; shares with the lead's scripts only the stated definitions: a run keeps the turns sent at least 300 s before
its last send; returning turn = turn index > 0; shared set = keys (conv, turn) present in every named run."""
import glob, json, os, re, statistics as st, sys

RES = "/home/wanhr/sglang/agent_cache/results"
C26 = os.path.join(RES, "campaign26")

def read_run(run, client=None):
    if client:
        path = os.path.join(RES, run, client, "client.jsonl")
    else:
        cands = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))
        assert cands, run
        path = cands[-1]
    turns, n_err, n_null = [], 0, 0
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("kind") != "turn":
                continue
            if r.get("error"):
                n_err += 1
            if r.get("ttft") is None:
                n_null += 1
                continue
            turns.append(r)
    last = max(r["t_send"] for r in turns)
    kept = {}
    dup = 0
    for r in turns:
        if r["t_send"] < last - 300:
            k = (r["conv"], r["turn"])
            if k in kept:
                dup += 1
            kept[k] = r
    return dict(kept=kept, n_all=len(turns), last=last, err=n_err, null=n_null, dup=dup)

# run map: five earlier runs as printed in the head of compare_ec.out, campaign 26 arms from its manifest
runs = {"baseline": ("20261101_082229_three_tier_wc_custom_L1nat_H70", None),
        "base_rep": ("20261105_075438_three_tier_wc_custom_L1nat_H70", None),
        "hintfree": ("20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70", None),
        "jit_p50": ("20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70", None),
        "jit_exact": ("20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70", None)}
c128 = {"baseline": ("20261018_103544_three_tier_wc_NAT160", None),
        "oracle": ("20261018_154254_three_tier_wc_oracle_v2_nu_NAT160", None),
        "p50": ("20261031_204526_three_tier_wc_oracle_v2_nu_NAT160", None),
        "jit_exact_c128": ("20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160", None),
        "jit_p50_c128": ("20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160", None),
        "hintfree_c128": ("20261104_213242_three_tier_wc_dl_poll0_NAT160", None)}
for line in open(os.path.join(C26, "manifest.txt")):
    f = line.split()
    if not f:
        continue
    if f[0].endswith("_c128"):
        c128[f[0]] = (f[1], f[2])
    elif re.search(r"_h\d+$", f[0]):
        continue
    else:
        runs[f[0]] = (f[1], f[2])

mean = lambda v: st.mean(v) if v else float("nan")
out = {}

print("== 44 users, 70 GB:", len(runs), "runs")
D = {a: read_run(*runs[a]) for a in runs}
for a, d in D.items():
    v = [r["ttft"] for r in d["kept"].values()]
    out[("c44", "per_arm", a)] = (len(v), round(mean(v), 2), round(st.median(v), 2))
    print(f"  {a:15s} kept {len(v):6d} mean {mean(v):6.2f} median {st.median(v):6.2f} | all turns {d['n_all']:6d} errors {d['err']} null-ttft {d['null']} dup {d['dup']}")
shared = set.intersection(*(set(k for k in d["kept"] if k[1] > 0) for d in D.values()))
print(f"  shared returning turns over all {len(D)} runs: {len(shared)}")
out[("c44", "shared_n", "all")] = len(shared)
for a, d in D.items():
    m = mean([d["kept"][k]["ttft"] for k in shared])
    out[("c44", "shared_mean", a)] = round(m, 2)
    print(f"     {a:15s} {m:6.3f}")
LEAD = ["jit_exact", "jit_exact_d", "ec_r13", "ec_r13_rep", "ec_r134", "ec_full", "ec_r1", "nofc", "nofc_rep", "nofc_noend", "nofc_noend_rep"]
lead = set.intersection(*(set(k for k in D[a]["kept"] if k[1] > 0) for a in LEAD))
print(f"  shared returning turns over the 11 leading runs: {len(lead)}")
out[("c44", "lead_n", "all")] = len(lead)
for a in LEAD:
    v = [D[a]["kept"][k]["ttft"] for k in lead]
    lat = [D[a]["kept"][k]["latency"] for k in lead]
    out[("c44", "lead_mean", a)] = (round(mean(v), 2), round(st.median(v), 2))
    print(f"     {a:15s} mean {mean(v):6.3f} median {st.median(v):6.3f} latency {mean(lat):6.2f} over10s {sum(x > 10 for x in v)}")

print("\n== 128 users, 160 GB:", len(c128), "runs, first 6900 s")
E = {a: read_run(*c128[a]) for a in c128}
T = {a: {k: r for k, r in E[a]["kept"].items() if r["t_send"] < 6900} for a in E}
for a in E:
    v = [r["ttft"] for r in T[a].values()]
    lat = [r["latency"] for r in T[a].values()]
    out[("c128", "per_arm", a)] = (len(v), round(mean(v), 2), round(st.median(v), 2), round(mean(lat), 2))
    print(f"  {a:16s} turns {len(v):6d} mean {mean(v):7.2f} median {st.median(v):6.2f} latency {mean(lat):7.2f} | errors {E[a]['err']} null {E[a]['null']} dup {E[a]['dup']}")
sh = set.intersection(*(set(k for k in T[a] if k[1] > 0) for a in T))
print(f"  shared returning turns over all {len(T)} runs: {len(sh)}")
out[("c128", "shared_n", "all")] = len(sh)
for a in T:
    m = mean([T[a][k]["ttft"] for k in sh]); l = mean([T[a][k]["latency"] for k in sh])
    out[("c128", "shared_mean", a)] = (round(m, 2), round(l, 2))
    print(f"     {a:16s} ttft {m:7.3f} latency {l:7.3f}")
for new in ("nofc_noend_c128", "ec_r134_c128", "nofc_c128"):
    for rival in ("oracle", "jit_exact_c128", "nofc_noend_c128", "ec_r134_c128"):
        if rival == new:
            continue
        ks = [k for k in T[new] if k in T[rival] and k[1] > 0]
        dif = [T[new][k]["ttft"] - T[rival][k]["ttft"] for k in ks]
        print(f"  {new} - {rival}: n {len(ks)} {mean([T[new][k]['ttft'] for k in ks]):.2f} vs {mean([T[rival][k]['ttft'] for k in ks]):.2f} paired {mean(dif):+.3f}")
json.dump({"|".join(k): v for k, v in out.items()}, open("recompute_client.json", "w"), indent=1)
