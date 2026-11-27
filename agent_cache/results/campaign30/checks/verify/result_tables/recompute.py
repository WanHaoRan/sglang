"""Independent recomputation of campaign 30's Results numbers from the raw client records (stdlib only).
Run from anywhere: nice -n 10 python3 -I -B recompute.py > recompute.out"""
import collections
import json
import os
import random

RES = "/home/wanhr/sglang/agent_cache/results"
MAN = os.path.join(RES, "campaign30/checks/vs_c34/manifest.txt")
CUT = 6600.0
SEED, RATE = "0", 0.1
WINDOWS = ((0, 10), (10, 30), (30, 60), (60, 110))


def load(fname):
    arm = {"reqs": [], "errs": [], "plays": {}, "lanes": {}, "run": None, "cancelled": [], "summary": None}
    with open(fname) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            k = r["kind"]
            if k == "run":
                arm["run"] = r
            elif k == "req" and r.get("phase") == "profile":
                (arm["errs"] if (r.get("error") or r.get("ttft") is None) else arm["reqs"]).append(r)
            elif k == "play":
                arm["plays"][r["play"]] = r
            elif k == "lanes":
                arm["lanes"][round(r["t"])] = r
            elif k == "cancelled":
                arm["cancelled"].append(r)
            elif k == "summary":
                arm["summary"] = r
    return arm


def avg(v):
    return sum(v) / len(v) if v else float("nan")


def q_near(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))] if s else float("nan")


def q_lin(v, p):
    s = sorted(v)
    if not s:
        return float("nan")
    x = p * (len(s) - 1)
    i = int(x)
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (x - i)


def first_req(r):
    return r["stream"] == 0 and r["k"] == 0


def shares(rs):
    prompt = sum(r.get("prompt_tokens") or 0 for r in rs)
    cached = 0
    for r in rs:
        d = r.get("cached_details") or {}
        cached += sum(int(d.get(t) or 0) for t in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)
    ideal = sum(r.get("ideal_cached") or 0 for r in rs)
    return 1 - cached / prompt, cached / ideal


def arrivals(n):
    rng, t, out = random.Random(f"{SEED}:arrivals"), 0.0, []
    for _ in range(n):
        out.append(t)
        t += rng.expovariate(RATE)
    return out


arms = collections.OrderedDict()
for line in open(MAN):
    if line.strip():
        label, run, prog = line.split()[:3]
        arms[label] = load(os.path.join(RES, run, prog, "client.jsonl"))

for a in arms.values():
    a["cut"] = [r for r in a["reqs"] if r["t_send"] < CUT]
    a["tt"] = {(r["play"], r["stream"], r["k"]): r["ttft"] for r in a["cut"]}
    assert len(a["tt"]) == len(a["cut"]), "duplicate keys"

print("== per arm, own requests sent before 6600 s")
print("arm | users | reqs | per min | mean p50 p90 p99 (nearest rank) | p50 p90 p99 (linear) | first returning (n first) | "
      "recomputed delivered | errors(all, before cut) | t_start epoch")
for name, a in arms.items():
    rs = a["cut"]
    t = [r["ttft"] for r in rs]
    f = [r["ttft"] for r in rs if first_req(r)]
    g = [r["ttft"] for r in rs if not first_req(r)]
    rec, dl = shares(rs)
    eb = sum(1 for r in a["errs"] if r["t_send"] < CUT)
    print(f"{name:18s} | {len({r['play'] for r in rs}):4d} | {len(rs):6d} | {60 * len(rs) / CUT:6.2f} | "
          f"{avg(t):7.3f} {q_near(t, .5):7.3f} {q_near(t, .9):7.3f} {q_near(t, .99):8.3f} | "
          f"{q_lin(t, .5):7.3f} {q_lin(t, .9):7.3f} {q_lin(t, .99):8.3f} | {avg(f):8.3f} {avg(g):7.3f} ({len(f)}) | "
          f"{rec:.4f} {dl:.4f} | {len(a['errs'])}, {eb} | {a['run']['t_start']:.1f}")

print("\n== by send time: n, mean TTFT")
for lo, hi in WINDOWS:
    cells = []
    for name, a in arms.items():
        rs = [r["ttft"] for r in a["cut"] if lo * 60 <= r["t_send"] < hi * 60]
        cells.append(f"{name}={len(rs)},{avg(rs):.3f}")
    print(f"{lo}-{hi}: " + "  ".join(cells))

print("\n== common set")
common = set.intersection(*(set(a["tt"]) for a in arms.values()))
users = {k[0] for k in common}
print(f"requests {len(common)} users {len(users)}")
for name, a in arms.items():
    t = [a["tt"][k] for k in common]
    print(f"{name:18s} mean {avg(t):7.3f} p50 {q_near(t, .5):7.3f} p90 {q_near(t, .9):7.3f}  (linear p50 {q_lin(t, .5):.3f} p90 {q_lin(t, .9):.3f})")
# when the common requests were sent, by ec_exact's send times
ts = [r["t_send"] for r in arms["ec_exact"]["cut"] if (r["play"], r["stream"], r["k"]) in common]
for lo, hi in WINDOWS:
    n = sum(1 for x in ts if lo * 60 <= x < hi * 60)
    print(f"  common requests sent (ec_exact clock) in minutes {lo}-{hi}: {n} ({100 * n / len(ts):.1f} %)")
nf = sum(1 for k in common if k[1] == 0 and k[2] == 0)
print(f"  of them first requests: {nf}")

print("\n== paired B - A, own bootstrap over users (2000 draws, seed 12345)")
rng = random.Random(12345)


def pair(A, B):
    a, b = arms[A]["tt"], arms[B]["tt"]
    keys = a.keys() & b.keys()
    per = collections.defaultdict(lambda: [0.0, 0])
    for k in keys:
        per[k[0]][0] += b[k] - a[k]
        per[k[0]][1] += 1
    cells = [tuple(per[p]) for p in sorted(per)]
    draws = sorted(sum(c[0] for c in pick) / sum(c[1] for c in pick)
                   for pick in (rng.choices(cells, k=len(cells)) for _ in range(2000)))
    d = sum(c[0] for c in cells) / len(keys)
    return len(keys), len(cells), avg([a[k] for k in keys]), avg([b[k] for k in keys]), d, draws[50], draws[1949]


for A in ("ec_full", "ec_exact"):
    for B in ("hicache", "vllm_lmcache", "continuum_lmcache", "tokencake", "continuum", "sglang_vanilla", "vllm_vanilla"):
        n, u, ma, mb, d, lo, hi = pair(A, B)
        print(f"{B:18s} - {A:9s}: n {n:6d} users {u:4d} A {ma:6.2f} B {mb:6.2f} B-A {d:+7.3f} [{lo:+7.2f}, {hi:+7.2f}]")
for A, B in (("hicache", "vllm_lmcache"), ("sglang_vanilla", "vllm_vanilla")):
    n, u, ma, mb, d, lo, hi = pair(A, B)
    print(f"{B:18s} - {A:9s}: n {n:6d} users {u:4d} A {ma:6.2f} B {mb:6.2f} B-A {d:+7.3f} [{lo:+7.2f}, {hi:+7.2f}]")

print("\n== users: slot wait (first send - arrival) and session length")
maxplay = max(max([k[0] for k in a["tt"]] + list(a["plays"])) for a in arms.values()) + 1
arr = arrivals(maxplay + 50)
# cross-check the drawn arrivals against the play records' own arrival_s
worst = 0.0
for a in arms.values():
    for p, rec in a["plays"].items():
        if rec.get("arrival_s") is not None:
            worst = max(worst, abs(rec["arrival_s"] - arr[p]))
print(f"max |play record arrival_s - drawn arrival| over all play records: {worst:.4f} s")
starts = {n: {r["play"]: r["t_send"] for r in a["cut"] if first_req(r)} for n, a in arms.items()}
in_all = set.intersection(*(set(s) for s in starts.values()))
fin_all = set.intersection(*({p for p, rec in a["plays"].items() if rec["end"] < CUT} for a in arms.values()))
print(f"users started (recorded first request) before the cut in every arm: {len(in_all)}; sessions ended before it in every arm: {len(fin_all)}")
for name, a in arms.items():
    w = [t - arr[p] for p, t in starts[name].items()]
    ln = [a["plays"][p]["end"] - a["plays"][p]["start"] for p in fin_all]
    print(f"{name:18s} slot wait mean {avg(w):7.1f} max {max(w):7.1f} (n {len(w)}) | session mean on {len(fin_all)} shared ended {avg(ln):7.1f}")

print("\n== users let in by 6600 s (client lanes 'plays') and in flight")
for name, a in arms.items():
    ln = a["lanes"].get(6600)
    print(f"{name:18s} lanes plays {ln['plays'] if ln else None} in_flight {ln['in_flight'] if ln else None} | recorded users {len({r['play'] for r in a['cut']})}")
