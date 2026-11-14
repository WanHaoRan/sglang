import re, sys
pat = re.compile(sys.argv[2], re.I)
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 160
for ln, line in enumerate(open(sys.argv[1], encoding='utf-8', errors='replace'), 1):
    for m in pat.finditer(line):
        s = max(0, m.start() - ctx); e = min(len(line), m.end() + ctx)
        print(f"{ln}: ...{line[s:e].strip()}...")
