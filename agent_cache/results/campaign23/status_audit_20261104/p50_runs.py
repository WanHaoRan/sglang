"""Per p50 run: hint settings, no-entry turns (cold start), pick level, list lengths, and the envelope bytes reconstructed
by re-running the client's own SessionHinter/OnlineGapTable in t_send order (gap_hints is imported, not copied)."""
import sys, json, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
from gap_hints import OnlineGapTable, SessionHinter, next_turn_envelope, scaled_gap
R = "/home/wanhr/sglang/agent_cache/results/"
RUNS = {
 "c20_p50_x10_c128": "20261031_204526_three_tier_wc_oracle_v2_nu_NAT160/client_204850_cmp_three_tier_wc_oracle_v2_nu",
 "c21_p50_c44": "20261101_062014_three_tier_wc_oracle_v2_nu_custom_L1nat_H70/client_062142_cmp_three_tier_wc_oracle_v2_nu",
 "c23_table_smoke": "20261101_194908_three_tier_wc_oracle_v2_nu_table_custom_L1nat_H70/client_195035_cmp_three_tier_wc_oracle_v2_nu_table",
 "c23_jit_p50_c44": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70/client_222615_cmp_three_tier_wc_oracle_v2_nu_owned",
 "c23_jit_p50_c128": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160/client_023345_cmp_three_tier_wc_oracle_v2_nu_owned",
}
trace = "/home/wanhr/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json"
data = json.load(open(trace))
keys = json.load(open(trace + ".reply_keys.json"))["conversations"]
out = {}
for name, d in RUNS.items():
    recs = [json.loads(l) for l in open(R + d + "/client.jsonl")]
    run = [r for r in recs if r.get("kind") == "run"][0]
    a = run["args"]
    turns = [r for r in recs if r.get("kind") == "turn" and "dist_hint" in r and r.get("t_send") is not None]
    c = collections.Counter(); lev = collections.Counter(); ne_t = []
    for r in turns:
        dh = r["dist_hint"]
        if dh.get("ended"): c["ended"] += 1
        elif dh["matched"] is None:
            c["no entry"] += 1; ne_t.append(r["t_send"])
        else:
            c["picked"] += 1
            lev["key" if dh["matched"] == dh["reply_key"] else ("*" if dh["matched"] == "*" else "tool")] += 1
    ll = sorted(r["dist_hint"]["list_len"] for r in turns)
    # reconstruct envelopes in t_send order with the client's own classes
    tables = collections.defaultdict(OnlineGapTable); hinters = {}; sizes = []; mism = 0
    for r in sorted(turns, key=lambda r: (r["t_send"], r["conv"], r["turn"])):
        i, conv, tn = r["task_conv"], r["conv"], r["turn"]
        m = data["sessions"][i]
        if conv not in hinters:
            hinters[conv] = SessionHinter(data["conversations"][i], keys=keys[i], gap_of=lambda t, s=a["gap_scale"]: scaled_gap(t, scale=s),
                                          history=tables[(m["source"], m["orig_model"])], end=a["hint_end"], calls=m["n_turns_total"])
        h = hinters[conv].step(tn)
        env = next_turn_envelope(message_id=f"{a['tag']}-c{conv}-t{tn}", session=f"{a['tag']}-c{conv}",
                                 next_rid=f"{a['tag']}-c{conv}-t{tn+1}", hint=h, v1_point=a["hint_point"])
        sizes.append(len(json.dumps(env)))
        dh = r["dist_hint"]
        mism += (len(h.entries) != dh["list_len"]) or ((h.picked or {}).get("p50") != dh.get("p50"))
    sizes.sort()
    out[name] = {"hints": a["hints"], "source": a["hint_source"], "key": a["hint_key"], "point": a["hint_point"], "end": a["hint_end"],
                 "users_cap": a["concurrency"], "turns": len(turns), "counts": dict(c), "pick_level": dict(lev),
                 "no_entry_t_send_max_s": round(max(ne_t), 1) if ne_t else None,
                 "list_len_median": ll[len(ll)//2], "list_len_max": ll[-1],
                 "envelope_bytes_median": sizes[len(sizes)//2], "envelope_bytes_max": sizes[-1],
                 "reconstruction_mismatches": mism,
                 "final_table_entries": {f"{k[0]}/{k[1]}": len(v.entries()) for k, v in sorted(tables.items())}}
    print(name, json.dumps(out[name]))
json.dump(out, open(sys.argv[1], "w"), indent=1)
