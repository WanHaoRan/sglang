import pickle, numpy as np, bisect
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb"))
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; ev = A["ev"]
    shr = {f"cmp_{arm}-c{x['conv']}-t{x['turn']}" for x in R[arm] if x["tier"] == "storage"}
    io_rids = [k["rid"] for _, k in A["g"]["s2h_io"]]
    extra = [r for r in io_rids if r not in shr]
    print(arm, "s2h_io rids not storage-hit rows:", len(extra), extra[:8])
    turnsd = {(r["conv"], r["turn"]): r for r in A["turns"]}
    kinds = {}
    for r in extra:
        c, t = r.split("-c")[1].split("-t"); rr = turnsd.get((int(c), int(t)))
        if rr is None: kinds.setdefault("noturn", []).append(r); continue
        cd = rr.get("cached_details") or {}
        key = "t0" if int(t) == 0 else ("sto>0" if cd.get("storage", 0) > 0 else "sto=0")
        kinds.setdefault(key, []).append((r, rr.get("prompt_tokens"), rr.get("cached_tokens"), cd, [k for k in ev.get(r, {})]))
    for k, v in kinds.items(): print("  ", k, len(v), v[:2])
