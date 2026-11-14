import sys, os, re
from html.parser import HTMLParser

ORDER = "cover setting motivation roadmap case-late case-early case-lost case-nextuse case-gpu b-model turncost problem b-policy price onegap conditional tradeoff rule1 rule1keys rule2 rule3 rule4 b-proof proof floor replay methods limits b-better settings better open next resources session refs".split()

BLOCK = {"p","div","li","h1","h2","h3","h4","h5","tr","section","aside","br","ul","ol","table","figcaption","dt","dd","text","tspan","g","svg","header","footer","blockquote","pre"}

class P(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0
    def handle_starttag(self, tag, attrs):
        if tag in ("style","script"):
            self.skip += 1
        if tag == "aside":
            self.out.append("\n[ASIDE / speaker notes]\n")
        if tag in BLOCK:
            self.out.append("\n")
        if tag in ("td","th"):
            self.out.append(" | ")
    def handle_endtag(self, tag):
        if tag in ("style","script"):
            self.skip = max(0, self.skip-1)
        if tag in BLOCK:
            self.out.append("\n")
    def handle_data(self, data):
        if self.skip: return
        self.out.append(data)

src = sys.argv[1]
dst = sys.argv[2]
with open(dst, "w") as f:
    for i, name in enumerate(ORDER, 1):
        p = os.path.join(src, name + ".html")
        raw = open(p, encoding="utf-8").read()
        h = P()
        h.feed(raw)
        txt = "".join(h.out)
        txt = re.sub(r"[ \t]+", " ", txt)
        txt = re.sub(r"\n\s*\n+", "\n", txt)
        f.write("\n===== SLIDE %d: %s =====\n" % (i, name))
        f.write(txt.strip() + "\n")
