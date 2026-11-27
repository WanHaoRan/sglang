"""Compact extract of the FreeInference program trace for the gap/hint analysis (CPU, stdlib).

Reads the trace once (gz, 662 MB) and writes one JSON line per tree with what the hint table and the
length statistics need: per request t, dur, wait, gap kind, key, out, prompt tokens, whether the only
dependency is the previous request of the same stream. Block ids are dropped.

  python3 -I -B extract_trace.py <trace.programs.jsonl.gz> <out.jsonl.gz>
"""
import gzip
import json
import sys


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    n_tree = n_req = 0
    with gzip.open(src, "rt", encoding="utf-8") as f, gzip.open(dst, "wt", encoding="utf-8", compresslevel=4) as g:
        header = json.loads(f.readline())
        B = header["block_tokens"]
        g.write(json.dumps({"kind": "header", "block_tokens": B, "source": src}) + "\n")
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            peak = 0
            streams = []
            for s, st in enumerate(t["streams"]):
                reqs = []
                for k, r in enumerate(st["requests"]):
                    n = (r["keep"] + len(r["new"])) * B + int(r.get("tail") or 0)
                    peak = max(peak, n + r["out"])
                    deps = r["deps"]
                    self_dep = 1 if (k > 0 and deps == [[s, k - 1]]) else 0
                    reqs.append([r["t"], r["dur"], r["wait"], r.get("gap"), r.get("key"), r["out"], n,
                                 r["keep"] * B, self_dep, len(deps), r.get("mutation")])
                streams.append({"role": st.get("role"), "name": st.get("name"), "model": st.get("model"), "reqs": reqs})
                n_req += len(reqs)
            g.write(json.dumps({"i": n_tree, "tree": t["tree"], "system": t.get("system"), "harness": t.get("harness"),
                                "user": t.get("user"), "week": t.get("week"), "scope": t.get("scope"),
                                "peak": peak, "streams": streams}, separators=(",", ":")) + "\n")
            n_tree += 1
            if n_tree % 1000 == 0:
                print(n_tree, n_req, flush=True)
    print("done", n_tree, n_req, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
