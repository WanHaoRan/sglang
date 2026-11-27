"""Copy from the trace the trees that the given client.jsonl files played (one pass over the 660 MB gz).

  python3 -I -B pick_trees.py <trace.jsonl.gz> <out.jsonl.gz> <client.jsonl> [<client.jsonl> ...]
"""
import gzip
import json
import re
import sys

trace, out_path, clients = sys.argv[1], sys.argv[2], sys.argv[3:]
want = set()
TREE = re.compile(r'"tree":"([^"]+)"')
for path in clients:
    with open(path) as f:
        for line in f:
            if '"kind":"req"' in line or '"kind":"play"' in line:
                m = TREE.search(line)
                if m:
                    want.add(m[1])
print(f"{len(want)} trees wanted", file=sys.stderr)
n = kept = 0
with gzip.open(trace, "rt", encoding="utf-8") as f, gzip.open(out_path, "wt", encoding="utf-8", compresslevel=3) as out:
    out.write(f.readline())   # header
    for line in f:
        n += 1
        m = TREE.search(line[:200])
        if m and m[1] in want:
            out.write(line)
            kept += 1
print(f"{n} trees read, {kept} kept -> {out_path}", file=sys.stderr)
