"""Per 5-minute bin (by send time) for ours (ec_full) and the seven baselines: requests answered, mean TTFT, and the
split into first requests (stream 0, k 0) and returning ones. Checks chart 2's embedded numbers and gives the
onset minute (first bin more than 2 s above ours). Read-only on the run folders; writes bins.json next to itself.
Usage, from agent_cache/results:
  /home/wanhr/venv312/bin/python -I -B campaign30/checks/report_review/visual/bins.py"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = "campaign30/checks/vs_c34/manifest.txt"
ARMS = ["ec_full", "vllm_lmcache", "hicache", "continuum_lmcache", "tokencake", "continuum", "sglang_vanilla",
        "vllm_vanilla"]
CUT, BIN = 6600.0, 300.0

runs = {}
for line in open(MAN):
    if line.strip():
        label, run, prog = line.split()[:3]
        runs[label] = os.path.join(run, prog, "client.jsonl")

out = {}
for arm in ARMS:
    reqs = []
    for l in open(runs[arm]):
        r = json.loads(l)
        if r.get("kind") == "req" and r.get("phase") == "profile" and r.get("ttft") is not None and r["t_send"] < CUT:
            reqs.append(r)
    bins = []
    for b in range(int(CUT // BIN)):
        a, z = b * BIN, (b + 1) * BIN
        inb = [r for r in reqs if a <= r["t_send"] < z]
        f = [r["ttft"] for r in inb if r["stream"] == 0 and r["k"] == 0]
        g = [r["ttft"] for r in inb if not (r["stream"] == 0 and r["k"] == 0)]
        m = lambda v: sum(v) / len(v) if v else None
        bins.append({"minute": int(a // 60), "n": len(inb), "mean": m([r["ttft"] for r in inb]),
                     "first_n": len(f), "first_mean": m(f), "ret_n": len(g), "ret_mean": m(g)})
    out[arm] = bins
    print(arm, "requests", len(reqs))

json.dump(out, open(os.path.join(HERE, "bins.json"), "w"), indent=1)

ours = out["ec_full"]
print("\nminute | " + " | ".join(f"{a[:12]:>12s}" for a in ARMS))
for i in range(len(ours)):
    print(f"{ours[i]['minute']:6d} | " + " | ".join(
        f"{out[a][i]['mean']:6.1f} {out[a][i]['n']:4d}" if out[a][i]['mean'] is not None else f"{'-':>11s}" for a in ARMS))

print("\nfirst requests per bin (n, mean s) | returning mean s")
for a in ["ec_full", "continuum", "continuum_lmcache"]:
    print(a)
    for x in out[a]:
        fm = f"{x['first_mean']:8.1f}" if x["first_mean"] is not None else f"{'-':>8s}"
        rm = f"{x['ret_mean']:6.2f}" if x["ret_mean"] is not None else "-"
        print(f"  {x['minute']:3d}: first {x['first_n']:3d} {fm} | returning {x['ret_n']:4d} {rm}")

print("\nonset: first bin more than T s above ours")
for T in (1.0, 2.0, 3.0):
    row = []
    for a in ARMS[1:]:
        on = next((x["minute"] for x, o in zip(out[a], ours) if x["mean"] - o["mean"] > T), None)
        row.append(f"{a}={on}")
    print(f"  T={T}: " + ", ".join(row))
print("\nbins where a baseline is below ours")
for a in ARMS[1:]:
    below = [(x["minute"], round(x["mean"] - o["mean"], 1)) for x, o in zip(out[a], ours) if x["mean"] < o["mean"] and x["minute"] >= 15]
    print(f"  {a}: {below}")
