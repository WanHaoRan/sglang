import sys, re
# usage: ctx.py FILE WIDTH MAXHITS pattern...
path, width, maxhits = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pats = sys.argv[4:]
lines = open(path, encoding='utf-8', errors='replace').read().split('\n')
for pat in pats:
    print('=== %s' % pat)
    n = 0
    rx = re.compile(pat, re.I)
    for i, line in enumerate(lines, 1):
        for m in rx.finditer(line):
            a = max(0, m.start() - width); b = min(len(line), m.end() + width)
            print('  L%d c%d: ...%s...' % (i, m.start(), line[a:b]))
            n += 1
            if n >= maxhits: break
        if n >= maxhits: break
    if n == 0: print('  (no match)')
