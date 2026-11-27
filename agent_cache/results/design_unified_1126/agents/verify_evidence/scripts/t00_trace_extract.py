"""One pass over the FreeInference trace.
 (1) extracts/trace_gaps.jsonl.gz: per tree: id, system, scope, harness, peak tokens (prompt+out), and per stream
     the list of [wait, gap kind, reply key, out, prompt tokens] per request (small).
 (2) extracts/trees_played.jsonl.gz: the full tree record of every tree a campaign 34 / 32 client played.
"""
import sys, os, gzip, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib

want = set()
for camp, arms in [(34, ["ec_full", "jit_exact", "jit_p50", "ec_exact", "hicache"]), (32, ["ec_full", "jit_exact", "jit_p50"])]:
    for arm in arms:
        _, cp = vlib.paths(camp, arm)
        for line in open(cp):
            if '"kind":"req"' in line:
                d = json.loads(line)
                want.add(d["tree"])
print("trees played in the campaign 34/32 arms:", len(want), flush=True)

t0 = time.time()
n = 0
B = 16
with gzip.open(vlib.trace_path(), "rt", encoding="utf-8") as f, \
        gzip.open(vlib.EXT + "/trace_gaps.jsonl.gz", "wt", compresslevel=5) as g, \
        gzip.open(vlib.EXT + "/trees_played.jsonl.gz", "wt", compresslevel=5) as h:
    header = json.loads(f.readline())
    assert header["block_tokens"] == B
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        n += 1
        streams = []
        peak = 0
        for st in d["streams"]:
            rs = []
            for r in st["requests"]:
                ptok = (r["keep"] + len(r["new"])) * B + int(r.get("tail") or 0)
                peak = max(peak, ptok + r["out"])
                rs.append([r["wait"], r.get("gap"), r.get("key"), r["out"], ptok])
            streams.append({"role": st.get("role"), "reqs": rs})
        g.write(json.dumps({"tree": d["tree"], "system": d.get("system"), "scope": d.get("scope"),
                            "harness": d.get("harness"), "peak": peak, "streams": streams}, separators=(",", ":")) + "\n")
        if d["tree"] in want:
            h.write(line if line.endswith("\n") else line + "\n")
        if n % 1000 == 0:
            print(n, "trees", "%.0fs" % (time.time() - t0), flush=True)
print("done", n, "trees", "%.0fs" % (time.time() - t0))
