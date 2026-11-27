"""Continuum: every user let in by 6,600 s and what the records hold for its first request; max TTFT per arm.
nice -n 10 python3 -I -B continuum_users.py > continuum_users.out"""
import collections, json, os, random
RES = "/home/wanhr/sglang/agent_cache/results"
MAN = os.path.join(RES, "campaign30/checks/vs_c34/manifest.txt")
for line in open(MAN):
    if not line.strip():
        continue
    label, run, prog = line.split()[:3]
    rows = [json.loads(x) for x in open(os.path.join(RES, run, prog, "client.jsonl")) if x.strip()]
    ok = [r for r in rows if r["kind"] == "req" and r.get("phase") == "profile" and r.get("ttft") is not None]
    t = sorted(r["ttft"] for r in ok)
    print(f"{label:18s} max TTFT {t[-1]:8.1f} s; TTFT >= 3000 s: {sum(1 for x in t if x >= 3000)}; >= 3500 s: {sum(1 for x in t if x >= 3500)}")
for label, rel in (("continuum", "20261124_203658_continuum_continuum/programs_203842_continuum"),
                   ("continuum_lmcache", "20261124_224027_continuum_continuum_lmcache/programs_224335_continuum_lmcache")):
    rows = [json.loads(x) for x in open(os.path.join(RES, rel, "client.jsonl")) if x.strip()]
    lanes = sorted((r["t"], r["plays"]) for r in rows if r["kind"] == "lanes")
    n_in = [n for tt, n in lanes if round(tt) == 6600][0]
    firsts = {r["play"]: r for r in rows if r["kind"] == "req" and r.get("phase") == "profile" and r["stream"] == 0 and r["k"] == 0}
    plays = {r["play"]: r for r in rows if r["kind"] == "play"}
    cls = collections.Counter()
    detail = collections.defaultdict(list)
    for p in range(n_in):
        f = firsts.get(p)
        if f is None:
            c = "no record"
        elif f.get("error"):
            c = "error"
        elif f.get("retried") and f["t_send"] >= 6600:
            c = "retried, recorded after the cut"
        elif f["t_send"] < 6600:
            c = "recorded before the cut"
        else:
            c = "other"
        cls[c] += 1
        # window of the session start (play record start, else lanes bound)
        start = plays[p]["start"] if p in plays else (f["t_send"] - (3600 if (f and f.get("retried") and f["t_send"] >= 6600) else 0) if f else
                                                      min(tt for tt, n in lanes if n > p))
        detail[c].append((p, round(start, 1)))
    print(f"== {label}: let in by 6600 s {n_in}: {dict(cls)}")
    for c, v in detail.items():
        if c != "recorded before the cut":
            w = collections.Counter("30-60" if s < 3600 else ("60-110" if s < 6600 else ">=110") if s >= 1800 else "<30" for _, s in v)
            print(f"   {c}: {len(v)} users, session start windows {dict(w)}; first few {v[:6]}")
