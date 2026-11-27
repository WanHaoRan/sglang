"""Stream the FreeInference program trace once and keep the trees the campaign 34 clients played.
Usage: python3 -I -B extract_trees.py <out.jsonl.gz> <client.jsonl> [<client.jsonl> ...]"""
import gzip, json, sys, re
out_path, clients = sys.argv[1], sys.argv[2:]
want = set()
pat = re.compile(r'"tree":"([^"]+)"')
for c in clients:
    with open(c) as f:
        for line in f:
            if '"kind":"req"' in line or '"kind":"play"' in line:
                m = pat.search(line)
                if m:
                    want.add(m.group(1))
print("trees wanted:", len(want), flush=True)
TRACE = "/home/wanhr/sglang/agent_cache/data/converted/freeinference_16w.programs.jsonl.gz"
tpat = re.compile(r'"tree":"([^"]+)"')
n = kept = 0
with gzip.open(TRACE, "rt", encoding="utf-8") as f, gzip.open(out_path, "wt", encoding="utf-8", compresslevel=4) as g:
    g.write(f.readline())  # header
    for line in f:
        n += 1
        m = tpat.search(line[:400])
        if m and m.group(1) in want:
            g.write(line)
            kept += 1
            want.discard(m.group(1))
        if n % 2000 == 0:
            print(n, kept, flush=True)
print("trees read", n, "kept", kept, "missing", len(want))
