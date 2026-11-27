"""Height estimate of each slide from the deck fonts' real advance widths (no browser on this host).

Usage: python3 estimate_heights.py <deck root> [slide id ...]
Needs font_metrics.json beside this file (ttf_widths.py writes it). Words wrap greedily at spaces; kerning and
`text-wrap: pretty` are ignored, so a ratio above 0.97 or a line within 2 % of its box is flagged.
"""
import json, math, os, re, sys
from html.parser import HTMLParser

M = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "font_metrics.json")))
FONT = {  # class: (size, line height, family, weight)
    "eyebrow": (24, 1.2, "plex", 600), "title": (60, 1.1, "serif", 600), "h3": (36, 1.2, "serif", 600),
    "body": (30, 1.4, "plex", 400), "strong": (30, 1.3, "plex", 600), "small": (24, 1.4, "plex", 400),
    "ink": (24, 1.4, "plex", 400), "ink-b": (24, 1.4, "plex", 600), "note": (24, 1.4, "plex", 400),
    "kick": (24, 1.4, "plex", 600), "tag": (24, 1.3, "plex", 600), "tag-blue": (24, 1.3, "plex", 600),
    "big": (60, 1.1, "plex", 600), "mid": (44, 1.1, "plex", 600), "math": (44, 1.3, "serif", 400),
    "math-s": (36, 1.3, "serif", 400), "title-xl": (120, 1.05, "serif", 600), "lead-dk": (36, 1.35, "plex", 400),
    "body-dk": (30, 1.4, "plex", 400), "small-dk": (24, 1.4, "plex", 400), "eyebrow-dk": (24, 1.2, "plex", 600),
    "title-dk": (84, 1.1, "serif", 600), "bx-t": (24, 1.3, "plex", 400), "bx-tb": (24, 1.3, "plex", 600),
    "bx-tw": (24, 1.3, "plex", 400), "bx-tp": (24, 1.3, "plex", 400), "bx-tm": (24, 1.3, "plex", 400),
    "step-name": (30, 1.3, "plex", 600), "step-text": (30, 1.4, "plex", 400), "step-text-s": (24, 1.4, "plex", 400),
    "dot-t": (30, 1.2, "plex", 600), "footer": (24, 1.4, "plex", 400),
}
PAD = {"card": (64, 64), "card-lt": (64, 64), "card-hot": (64, 64), "card-s": (40, 56), "card-s-lt": (40, 56),
       "formula": (56, 78), "step": (32, 64), "step-lt": (32, 64), "bx": (24, 36)}
GAP = {"head": 12, "col": 12, "col-s": 16, "grow": 16, "card": 8, "card-lt": 8, "card-hot": 8, "card-s": 6,
       "card-s-lt": 6, "formula": 12, "row": 24, "row-wide": 64, "chips": 12, "step": 32, "step-lt": 32}
ROWS = {"row", "row-wide", "chips", "step", "step-lt"}
INLINE = {"i", "b", "span", "em", "strong", "sup", "br", "a", "code"}
SKIP = {"aside", "noscript"}
NOTES = []


class Node:
    def __init__(self, tag, attrs):
        self.tag, self.attrs, self.kids, self.runs = tag, dict(attrs), [], []
        self.cls = (self.attrs.get("class") or "").split()
        self.style = (self.attrs.get("style") or "").replace(" ", "")

    def px(self, name):
        m = re.search(r"(?:^|;)" + name + r":(\d+(?:\.\d+)?)px", self.style)
        return float(m.group(1)) if m else None

    def pct(self, name):
        m = re.search(r"(?:^|;)" + name + r":(\d+(?:\.\d+)?)%", self.style)
        return float(m.group(1)) if m else None

    @property
    def text(self):
        return "".join(c for c, _, _ in self.runs)


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", [])
        self.stack = [self.root]
        self.it = self.bd = 0

    def handle_starttag(self, tag, attrs):
        if tag in INLINE:
            if tag == "br":
                self.stack[-1].runs.append(("\n", 0, 0))
            self.it += tag in ("i", "em")
            self.bd += tag in ("b", "strong")
            return
        n = Node(tag, attrs)
        self.stack[-1].kids.append(n)
        self.stack.append(n)

    def handle_endtag(self, tag):
        if tag in INLINE:
            self.it -= tag in ("i", "em")
            self.bd -= tag in ("b", "strong")
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        self.stack[-1].runs.extend((c, self.it > 0, self.bd > 0) for c in data)


def font_of(n, inherited):
    for c in n.cls:
        if c in FONT:
            return FONT[c]
    fs = n.px("font-size")
    if fs:
        return (fs,) + inherited[1:]
    return inherited


def char_w(ch, it, bd, font, upper=False):
    size, _, family, weight = font
    if upper:
        ch = ch.upper()
    heavy = bd or weight >= 600
    face = ("serif600" if heavy else "serif400") if family == "serif" else ("plex600" if heavy else ("plex400i" if it else "plex400"))
    return M[face].get(ch, 0.55) * size


def wrap_lines(runs, width, font, upper=False, spacing=0.0, where=""):
    """Greedy wrap at spaces; returns the number of lines."""
    lines = 0
    paragraphs, cur = [], []
    for r in runs:
        if r[0] == "\n":
            paragraphs.append(cur); cur = []
        else:
            cur.append(r)
    paragraphs.append(cur)
    for para in paragraphs:
        text = re.sub(r"\s+", " ", "".join(c if not c.isspace() else " " for c, _, _ in para)).strip()
        if not text:
            continue
        # rebuild with style flags by walking the runs in step with the collapsed text
        flags, prev_space = [], True
        for c, it, bd in para:
            sp = c.isspace()
            if sp and prev_space:
                continue
            flags.append((" " if sp else c, it, bd)); prev_space = sp
        while flags and flags[-1][0] == " ":
            flags.pop()
        words, w = [], []
        for f in flags:
            if f[0] == " ":
                words.append(w); w = []
            else:
                w.append(f)
        words.append(w)
        space = char_w(" ", False, False, font) + spacing
        line_w, n = 0.0, 1
        for wd in words:
            ww = sum(char_w(c, it, bd, font, upper) + spacing for c, it, bd in wd)
            if line_w == 0:
                line_w = ww
            elif line_w + space + ww <= width:
                line_w += space + ww
            else:
                n += 1; line_w = ww
        if n == 1 and line_w > 0.97 * width:
            NOTES.append(f"{where}: a single line fills {line_w / width:.0%} of its box and may wrap ({text[:40]}...)")
        lines += n
    return max(lines, 1)


def all_runs(n):
    out = list(n.runs)
    for c in n.kids:
        out += [(" ", 0, 0)] + all_runs(c)
    return out


def table_height(n, width, sid):
    rows = [r for r in walk(n) if r.tag == "tr"]
    size = 24
    first = [c for c in rows[0].kids if c.tag in ("th", "td")]
    pcts = [c.pct("width") for c in first]
    known = sum(p for p in pcts if p)
    unknown = [p for p in pcts if not p]
    rest = (100 - known) / len(unknown) if unknown else 0
    widths = [(p or rest) * width / 100 for p in pcts]
    total = 2
    for ri, r in enumerate(rows):
        cells = [c for c in r.kids if c.tag in ("th", "td")]
        h = 0
        for i, c in enumerate(cells):
            font = (size, 1.2, "plex", 600 if c.tag == "th" else 400)
            inner = widths[min(i, len(widths) - 1)] - 2 * 0.6 * size - 2
            lines = wrap_lines(all_runs(c), inner, font, where=f"{sid} table row {ri + 1} col {i + 1}")
            h = max(h, lines * size * 1.2)
        total += h + 2 * 0.35 * size + 2
    return total


def walk(n):
    yield n
    for c in n.kids:
        yield from walk(c)


def height(n, width, font, sid):
    if n.tag in SKIP or "footer" in n.cls or "position:absolute" in n.style:
        return 0
    font = font_of(n, font)
    wpx = n.px("width")
    if wpx:
        width = wpx
    pad_v = pad_h = 0
    for c in n.cls:
        if c in PAD:
            pad_v, pad_h = PAD[c]
    hpx = n.px("height")
    if hpx is not None:
        return hpx
    if n.tag == "table":
        return table_height(n, width, sid)
    inner = width - pad_h
    kids = [c for c in n.kids if c.tag not in SKIP and "footer" not in c.cls and "position:absolute" not in c.style]
    if not kids:
        if not n.text.strip():
            return pad_v
        upper = "eyebrow" in n.cls or "eyebrow-dk" in n.cls
        lines = wrap_lines(n.runs, inner, font, upper=upper, spacing=2.0 if upper else 0.0, where=f"{sid} {'.'.join(n.cls) or n.tag}")
        return lines * font[0] * font[1] + pad_v
    gap = n.px("gap")
    if gap is None:
        gap = next((GAP[c] for c in n.cls if c in GAP), 0)
    if any(c in ROWS for c in n.cls) or "flex-direction:row" in n.style:
        fixed = [c.px("width") for c in kids]
        flex = [i for i, c in enumerate(kids) if fixed[i] is None]
        remaining = inner - sum(w for w in fixed if w) - gap * (len(kids) - 1)
        share = remaining / len(flex) if flex else 0
        hs = [height(c, fixed[i] if fixed[i] else share, font, sid) for i, c in enumerate(kids)]
        if os.environ.get("VERBOSE") and n.tag == "div" and ("row" in n.cls or "row-wide" in n.cls):
            print(f"           {sid} {'.'.join(n.cls)} columns: {[round(h) for h in hs]}")
        return max(hs) + pad_v
    return sum(height(c, inner, font, sid) for c in kids) + gap * (len(kids) - 1) + pad_v


def main():
    root = sys.argv[1]
    order = json.load(open(root + "/project/deck.json"))["order"]
    ids = sys.argv[2:] or order
    bad = 0
    for sid in ids:
        NOTES.clear()
        t = Tree()
        t.feed(open(f"{root}/project/slides/{sid}.html", encoding="utf-8").read())
        sec = t.root.kids[0]
        foot_cls = "foot" in sec.cls
        usable = 1080 - 128 - (160 if foot_cls else 128)
        gap = sec.px("gap")
        gap = 40 if gap is None else gap
        kids = [c for c in sec.kids if c.tag not in SKIP and "footer" not in c.cls and "position:absolute" not in c.style]
        base = (32, 1.2, "plex", 400)
        hs = [height(c, 1664, base, sid) for c in kids]
        total = sum(hs) + gap * (len(kids) - 1)
        foot = next((c for c in sec.kids if "footer" in c.cls), None)
        fl, clash = 0, ""
        if foot is not None:
            fl = wrap_lines(foot.runs, 1664, FONT["footer"], where=f"{sid} footer")
            top = 1080 - 64 - fl * 24 * 1.4
            if 128 + total > top - 8:
                clash = f" FOOTER CLASH (content ends {128 + total:.0f}, footer starts {top:.0f})"
        flag = "OVER" if total > usable else ("tight" if total > 0.97 * usable else "")
        bad += bool(flag == "OVER" or clash)
        print(f"{sid:10s} {total:6.0f} / {usable}  {total / usable:4.2f} {flag:5s} footer lines {fl}{clash}  parts {[round(h) for h in hs]}")
        for nline in NOTES:
            print("           note:", nline)
    print("slides over or clashing:", bad)


main()
