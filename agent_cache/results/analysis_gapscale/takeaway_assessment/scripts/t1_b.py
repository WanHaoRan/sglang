import pickle, bisect, numpy as np, csv
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb"))
BPT = 98304; PAGE = 64
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
def summ(a): a = np.asarray(a, float); return "n %d mean %.3f p10 %.3f p50 %.3f p90 %.3f p99 %.3f max %.3f" % (len(a), a.mean(), q(a,10), q(a,50), q(a,90), q(a,99), a.max()) if len(a) else "n 0"
res = {}
for arm in ("three_tier_to", "three_tier_wc", "hbm_host"):
    A = D[arm]; t0 = A["t0"]; ev = A["ev"]
    bt = [t for t, _ in A["g"]["batch"]]; bq = [k["q"] for _, k in A["g"]["batch"]]; brun = [k["run"] for _, k in A["g"]["batch"]]
    turns = {(r["conv"], r["turn"]): r for r in A["turns"] if "t_send" in r and r.get("ttft")}
    rows = []
    for (c, t), r in turns.items():
        if t == 0: continue
        rid = f"cmp_{arm}-c{c}-t{t}"; e = ev.get(rid, {})
        cd = r.get("cached_details") or {}; prompt = r["prompt_tokens"]; cached = r["cached_tokens"]
        tier = "storage" if cd.get("storage", 0) > 0 else ("cold" if cached < 0.5 * prompt else "host" if cd.get("host", 0) > 0 else "device")
        send = t0 + r["t_send"]; prev = turns.get((c, t - 1))
        end_prev = t0 + prev["t_send"] + prev["latency"] if prev else None
        i = bisect.bisect_right(bt, send) - 1
        x = dict(arm=arm, conv=c, turn=t, tier=tier, prompt=prompt, cached=cached, dev=cd.get("device", 0), host=cd.get("host", 0), sto=cd.get("storage", 0),
                 ttft=r["ttft"], send=send, trel=r["t_send"], gap=r["gap_slept"], end_prev=end_prev, qdepth=bq[i] if i >= 0 else 0, running=brun[i] if i >= 0 else 0)
        if "prefetch_start" in e: x["ps"] = e["prefetch_start"][0][0]
        if "s2h_query" in e: x["sq"] = e["s2h_query"][0][0]; x["sq_hit"] = int(e["s2h_query"][0][1]["storage_hit"]); x["sq_req"] = int(e["s2h_query"][0][1]["tokens_req"])
        if "s2h_io" in e:
            tt, kv = e["s2h_io"][0]; x["io_end"] = tt; x["io_ms"] = float(kv["ms"]); x["pages"] = int(kv["pages"]); x["io_start"] = tt - float(kv["ms"]) / 1000
        if "pf_success" in e: x["pf"] = e["pf_success"][0][0]; x["pf_loaded"] = int(e["pf_success"][0][1]["loaded"]); x["pf_completed"] = int(e["pf_success"][0][1]["completed"])
        if "load_back_init" in e: x["lb"] = e["load_back_init"][0][0]; x["lb_tokens"] = int(e["load_back_init"][0][1]["tokens"])
        if "lb" in x: x["queue"] = x["lb"] - send; x["adm2ft"] = r["ttft"] - x["queue"]
        rows.append(x)
    res[arm] = rows
pickle.dump(res, open(S + "t1_rows.pkl", "wb"))
for arm in ("three_tier_to", "three_tier_wc"):
    rows = res[arm]; sh = [x for x in rows if x["tier"] == "storage"]
    print("=====", arm, "storage-hit turns", len(sh))
    print(" TTFT            ", summ([x["ttft"] for x in sh]))
    print(" storage tokens  ", summ([x["sto"] for x in sh]))
    print(" prompt tokens   ", summ([x["prompt"] for x in sh]))
    print(" pages*64 (io)   ", summ([x["pages"] * 64 for x in sh]))
    print(" pf loaded       ", summ([x["pf_loaded"] for x in sh]), " sto==pf_loaded:", sum(1 for x in sh if x["sto"] == x["pf_loaded"]))
    print(" gap_slept       ", summ([x["gap"] for x in sh]))
    print(" send->ps        ", summ([x["ps"] - x["send"] for x in sh]))
    print(" ps->sq (query)  ", summ([x["sq"] - x["ps"] for x in sh]))
    print(" sq->io_start    ", summ([x["io_start"] - x["sq"] for x in sh]))
    print(" io ms           ", summ([x["io_ms"] / 1000 for x in sh]))
    print(" fetch ps->io_end", summ([x["io_end"] - x["ps"] for x in sh]))
    print(" send->io_end    ", summ([x["io_end"] - x["send"] for x in sh]))
    print(" io_end->pf      ", summ([x["pf"] - x["io_end"] for x in sh]))
    print(" io_end->lb      ", summ([x["lb"] - x["io_end"] for x in sh]))
    print(" queue (send->lb)", summ([x["queue"] for x in sh]))
    print(" adm->first tok  ", summ([x["adm2ft"] for x in sh]))
    print(" qdepth@arrival  ", summ([x["qdepth"] for x in sh]))
    print(" fetch share of queue (sum io_end-send / sum queue): %.3f" % (sum(min(x["queue"], x["io_end"] - x["send"]) for x in sh) / sum(x["queue"] for x in sh)))
    print(" fetch share of TTFT  (sum min(queue, send->io_end) / sum ttft): %.3f" % (sum(min(x["queue"], x["io_end"] - x["send"]) for x in sh) / sum(x["ttft"] for x in sh)))
    bw = [x["pages"] * 64 * 98304 / (x["io_ms"] / 1000) / 2**30 for x in sh if x["io_ms"] > 0]
    print(" per-read throughput GiB/s", summ(bw))
    for tier in ("device", "host", "cold"):
        tt = [x for x in rows if x["tier"] == tier]
        print(" tier", tier, "TTFT", summ([x["ttft"] for x in tt]))
