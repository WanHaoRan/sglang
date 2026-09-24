import pickle, numpy as np
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb")); SIM = pickle.load(open(S + "t1_sim.pkl", "rb"))
BPT = 98304; GiB = 2**30
for arm in ("three_tier_to", "three_tier_wc"):
    A = D[arm]; t0 = A["t0"]; jobs, res = SIM[arm]
    rows = R[arm]; sh = [x for x in rows if x["tier"] == "storage"]
    print("=====", arm, " per 5-min: storage hits | read-thread busy % | read GiB/s | mean storage TTFT | mean fetch window | host-evict tok/s | returns | cold")
    for lo in range(0, 5400, 300):
        a, b = t0 + lo, t0 + lo + 300
        busy = sum(max(0, min(b, j["io_end"]) - max(a, j["io_end"] - j["dur"])) for j in jobs)
        byt = sum(j["pages"] * 64 * BPT * max(0, min(b, j["io_end"]) - max(a, j["io_end"] - j["dur"])) / max(j["dur"], 1e-6) for j in jobs)
        s = [x for x in sh if a <= x["send"] < b]
        ev = sum(int(k["tokens"]) for t, k in A["g"]["evict_host"] if a <= t < b)
        rt = [x for x in rows if a <= x["send"] < b]
        cold = sum(1 for x in rt if x["tier"] == "cold")
        if rt:
            print("  +%4d s: %3d | %5.1f%% | %.2f | %6.2f | %6.2f | %6.0f | %3d | %d" % (lo, len(s), 100 * busy / 300, byt / 300 / GiB, np.mean([x["ttft"] for x in s]) if s else float('nan'), np.mean([x["io_end"] - x["ps"] for x in s]) if s else float('nan'), ev / 300, len(rt), cold))
