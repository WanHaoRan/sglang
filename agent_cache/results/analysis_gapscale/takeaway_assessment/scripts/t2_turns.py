#!/usr/bin/env python3
"""Build per-returning-turn table from t2.pkl; write t2_turns.csv."""
import pickle, os, csv, bisect
HERE = os.path.dirname(os.path.abspath(__file__))
A = "/home/wanhr/sglang/agent_cache/results/analysis_gapscale"
SHARED = 7808
CSVMAP = {"x10": "x10 (campaign 8)", "x1": "x1 (campaign 9)"}

def load_csv(scale, arm):
    p = os.path.join(A, f"turn_components_{CSVMAP[scale]}_{arm}.csv")
    return {(int(r["conv"]), int(r["turn"])): r for r in csv.DictReader(open(p))}

def build():
    D = pickle.load(open(os.path.join(HERE, "t2.pkl"), "rb"))
    rows = []
    for (scale, arm), run in D.items():
        comp = load_csv(scale, arm)
        t0 = run["t0"]; ev = run["ev"]
        bt = [b[0] for b in run["glob"]["batch"]]
        for r in run["turns"]:
            if r["turn"] == 0 or "t_send" not in r or not r.get("ttft"):
                continue
            rid = f"cmp_{arm}-c{r['conv']}-t{r['turn']}"
            e = ev.get(rid, {})
            P = int(r["prompt_tokens"]); C = int(r.get("cached_tokens") or 0)
            cd = r.get("cached_details") or {}
            ps = e.get("prefetch_start", [])
            T = int(ps[0][1]["tokens"]) if ps else None
            m_arr = (P - 1 - T) if T is not None else None
            q = e.get("s2h_query", [])
            io = e.get("s2h_io", [])
            pfr = e.get("pf_result", [])
            lb = e.get("load_back_init", [])
            rts = e.get("rts", [])
            send = t0 + r["t_send"]
            arr_t = ps[0][0] if ps else None
            if rts:
                entry = rts[-1][1]["entry"]; queue = rts[-1][1]["queue"]; adm = entry + queue; qhow = "time_stats"
            else:
                c = comp.get((r["conv"], r["turn"]))
                queue = float(c["queue"]) if c else None; qhow = c["how"] if c else None
                adm = (lb[-1][0] if lb else (send + queue if queue is not None else None))
                if lb:
                    queue = lb[-1][0] - (arr_t if arr_t else send); qhow = "load_back_init"
                entry = arr_t
            rows.append(dict(
                scale=scale, arm=arm, conv=r["conv"], turn=r["turn"], t_send=r["t_send"], P=P, C=C,
                dev=int(cd.get("device") or 0), host=int(cd.get("host") or 0), sto=int(cd.get("storage") or 0),
                T=T, n_ps=len(ps), m_arr=m_arr, delta=r.get("delta_tokens"),
                q_req=int(q[0][1]["tokens_req"]) if q else None, q_hit=int(q[0][1]["storage_hit"]) if q else None,
                io_pages=sum(int(x[1]["pages"]) for x in io), io_ms=sum(float(x[1]["ms"]) for x in io),
                pf_loaded=sum(x[1]["loaded"] for x in pfr), pf_completed=sum(x[1]["completed"] for x in pfr),
                lb_host_hit=int(lb[-1][1]["host_hit"]) if lb else 0,
                rts_input=rts[-1][1]["input_len"] if rts else None, rts_cached=rts[-1][1]["cached"] if rts else None,
                queue=queue, qhow=qhow, entry=entry, adm=adm, ttft=r["ttft"],
            ))
    return rows

if __name__ == "__main__":
    rows = build()
    with open(os.path.join(HERE, "t2_turns.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(len(rows))
