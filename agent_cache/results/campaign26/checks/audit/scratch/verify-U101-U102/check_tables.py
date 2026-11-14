"""Set the verifier's recomputed numbers against the printed tables. Read-only."""
import json, re, os
C26 = "/home/wanhr/sglang/agent_cache/results/campaign26"
mine = {tuple(k.split("|")): v for k, v in json.load(open("recompute_client.json")).items()}
ok = bad = 0
later = {"nofc_rep", "jit_exact_d", "ec_r134", "nofc_noend_rep", "ec_full", "fq90", "ec_r3", "nofc_noend_c128", "ec_r134_c128", "nofc_c128"}
ok_later = 0
def cmp(what, a, b, arm=None):
    global ok, bad, ok_later
    if a == b:
        ok += 1
        if arm in later: ok_later += 1
    else:
        bad += 1
        print("  MISMATCH", what, "mine", a, "printed", b)

# compare_ec.out: per-arm table and the all-arms set
txt = open(os.path.join(C26, "compare_ec.out")).read().split("\n")
i = next(n for n, l in enumerate(txt) if l.startswith("per arm, turns sent at least"))
for l in txt[i + 1:]:
    f = l.split()
    if not f: break
    arm = f[0]
    n, m, med = mine[("c44", "per_arm", arm)]
    cmp(f"compare_ec per-arm turns {arm}", n, int(f[1]), arm); cmp(f"compare_ec per-arm mean {arm}", f"{m:.2f}", f[5], arm); cmp(f"compare_ec per-arm median {arm}", f"{med:.2f}", f[6], arm)
i = next(n for n, l in enumerate(txt) if l.startswith("all arms:"))
cmp("compare_ec shared n", mine[("c44", "shared_n", "all")], int(txt[i].split()[2]))
for l in txt[i + 1:]:
    f = l.split()
    if not f or f[0] == "by": break
    cmp(f"compare_ec shared mean {f[0]}", f"{mine[('c44', 'shared_mean', f[0])]:.2f}", f[1], f[0])

# pairs_ec.out: the leading set
txt = open(os.path.join(C26, "pairs_ec.out")).read().split("\n")
cmp("pairs_ec lead n", mine[("c44", "lead_n", "all")], int(txt[0].split()[0]))
i = next(n for n, l in enumerate(txt) if l.startswith("per arm on them"))
for l in txt[i + 1:]:
    f = l.split()
    if not f: break
    m, med = mine[("c44", "lead_mean", f[0])]
    cmp(f"pairs_ec lead mean {f[0]}", f"{m:.2f}", f[1], f[0]); cmp(f"pairs_ec lead median {f[0]}", f"{med:.2f}", f[2], f[0])

# compare_c128.out
txt = open(os.path.join(C26, "compare_c128.out")).read().split("\n")
i = next(n for n, l in enumerate(txt) if l.startswith("per arm, turns sent in its first"))
for l in txt[i + 1:]:
    f = l.split()
    if not f: break
    n, m, med, lat = mine[("c128", "per_arm", f[0])]
    cmp(f"c128 per-arm turns {f[0]}", n, int(f[1]), f[0]); cmp(f"c128 mean {f[0]}", f"{m:.2f}", f[2], f[0]); cmp(f"c128 median {f[0]}", f"{med:.2f}", f[3], f[0]); cmp(f"c128 latency {f[0]}", f"{lat:.2f}", f[6], f[0])
row = next(l for l in txt if l.strip().startswith("all          :"))
cmp("c128 shared n", mine[("c128", "shared_n", "all")], int(row.split("|")[0].split(":")[1]))
names = list(k[2] for k in mine if k[0] == "c128" and k[1] == "shared_mean")
for cell in row.split("|")[1:]:
    f = cell.split()
    arm = next(a for a in names if a[:14] == f[0])
    cmp(f"c128 shared mean {arm}", f"{mine[('c128', 'shared_mean', arm)][0]:.2f}", f[1], arm)
row = next(l for l in txt if l.strip().startswith("latency      :"))
for cell in row.split("|")[1:]:
    f = cell.split()
    arm = next(a for a in names if a[:14] == f[0])
    cmp(f"c128 shared latency {arm}", f"{mine[('c128', 'shared_mean', arm)][1]:.2f}", f[1], arm)
print(f"matched {ok} of {ok + bad} printed numbers; of them {ok_later} belong to the ten arms landed after the 12 Nov recomputation")
