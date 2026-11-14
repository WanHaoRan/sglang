import re, sys, html
for p in sys.argv[1:]:
    t = open(p, encoding="utf-8", errors="replace").read()
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"(?s)<[^>]+>", "\n", t)
    t = html.unescape(t)
    lines = [re.sub(r"\s+", " ", x).strip() for x in t.split("\n")]
    lines = [x for x in lines if x]
    print("=====", p.rsplit("/", 1)[-1])
    print("\n".join(lines))
