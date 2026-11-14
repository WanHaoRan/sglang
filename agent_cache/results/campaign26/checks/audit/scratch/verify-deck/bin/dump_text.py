import sys, re, html, json, os
src = sys.argv[1]
out = sys.argv[2]
order = json.load(open(os.path.join(src, "project", "deck.json")))["order"]
lines = []
for i, sid in enumerate(order, 1):
    p = os.path.join(src, "project", "slides", sid + ".html")
    if not os.path.exists(p):
        lines.append("=== %d %s (MISSING FILE)" % (i, sid))
        continue
    t = open(p, encoding="utf-8").read()
    # split notes
    m = re.search(r"<aside>(.*?)</aside>", t, re.S)
    notes = m.group(1) if m else ""
    body = re.sub(r"<aside>.*?</aside>", "", t, flags=re.S)
    def clean(s):
        s = re.sub(r"<img[^>]*alt=\"([^\"]*)\"[^>]*>", r" [img:\1] ", s)
        s = re.sub(r"<svg[^>]*aria-label=\"([^\"]*)\"[^>]*>.*?</svg>", r" [svg:\1] ", s, flags=re.S)
        s = re.sub(r"</(p|h1|h2|h3|li|div|tr|td|th)>", "\n", s)
        s = re.sub(r"<[^>]+>", "", s)
        s = html.unescape(s)
        s = re.sub(r"[ \t]+", " ", s)
        s = "\n".join(x.strip() for x in s.split("\n") if x.strip())
        return s
    lines.append("=== %d %s" % (i, sid))
    lines.append(clean(body))
    lines.append("--- notes %d %s" % (i, sid))
    lines.append(clean(notes))
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(len(order), "slides ->", out)
