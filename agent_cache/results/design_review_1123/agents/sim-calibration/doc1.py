import sys, ast, os
for p in open(sys.argv[1]).read().split():
    try:
        src = open(p, errors='replace').read()
        d = ast.get_docstring(ast.parse(src)) or ''
        first = d.strip().splitlines()[0][:120] if d.strip() else '(no docstring)'
    except Exception:
        first = '(parse error)'
    print(f"{p:60s} {src.count(chr(10)):5d}  {first}")
