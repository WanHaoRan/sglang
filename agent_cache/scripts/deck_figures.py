"""Redraw the charts and diagrams of the "Pricing the Room" deck as standalone SVGs.

On a slide a figure is a position:relative <div> of fixed size whose children are pinned to it: painted boxes,
labels, and for a chart one inline <svg> with the curves. Each becomes one SVG file: a box is a <rect>, a label a
<text> in the deck's face (IBM Plex Sans, by name: not embedded), and the inline <svg> is nested unchanged. Blocks
that follow one another in the same column, as a case slide's three timelines do, are stacked into one figure with the
column's gap, and a line of text right above a block comes along as its title.

Every label is drawn as one line, which holds for the deck as published on 2026-11-10 (measured with the face); a
label that wraps on its slide would need its lines split here.

The viewBox is in slide pixels and the size in points, half as many, as deck_formulas.py --svg sizes the formulas.
Files are named <slide number>_<slide>_<what>.

Usage: python3 deck_figures.py <deck project dir> [out_dir]   (default out_dir: ../results/pricing_the_room)
  <deck project dir> is the deck's project/ folder as read from its artifact: deck.json and slides/<id>.html.
"""
import html, json, pathlib, re, sys
from html.parser import HTMLParser

HERE = pathlib.Path(__file__).resolve().parent
SLIDE_PPI = 144  # 1920 px across PowerPoint's 13.33-in 16:9 slide
# hhea ascent and descent of IBMPlexSans-Regular.ttf, in em; a browser centers their sum in the line box.
ASCENT, DESCENT = 1.025, 0.275
# The deck format's <p>: 32 px, weight 400, line-height 1.4 (artifact-type/reference/format.md).
P_SIZE, P_WEIGHT, P_LINE_HEIGHT = 32.0, 400, 1.4
VOID = {"img", "br", "hr"}
SVG_RE = re.compile(r"<svg\b.*?</svg>", re.S)


class Node:
    def __init__(self, tag, attrs=(), parent=None, text=""):
        self.tag, self.attrs, self.parent, self.kids, self.text = tag, dict(attrs), parent, [], text
        self.style = {k.strip(): v.strip() for k, v in
                      (d.split(":", 1) for d in self.attrs.get("style", "").split(";") if ":" in d)}

    def elements(self):
        return [kid for kid in self.kids if kid.tag != "#text"]

    def walk(self):
        yield self
        for kid in self.kids:
            yield from kid.walk()

    def inherited(self, prop, default):
        node = self
        while node is not None:
            if prop in node.style:
                return node.style[prop]
            node = node.parent
        return default


class Tree(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.root = self.cur = Node("root")
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.cur)
        self.cur.kids.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_endtag(self, tag):
        node = self.cur
        while node.parent is not None and node.tag != tag:
            node = node.parent
        if node.parent is not None:
            self.cur = node.parent

    def handle_data(self, data):
        self.cur.kids.append(Node("#text", parent=self.cur, text=data))


def px(value, default=None):
    return default if value is None else float(value.replace("px", ""))


def num(value):
    return f"{value:.2f}".rstrip("0").rstrip(".")


def is_block(node):
    return node.tag == "div" and node.style.get("position") == "relative"


def is_pinned(node):
    return node.style.get("position") == "absolute"


def is_column(node):
    if node.tag == "div" and "display" not in node.style:
        return True
    return node.style.get("display") == "flex" and node.style.get("flex-direction") == "column"


# ===== Text =====

def font(node):
    size = px(node.inherited("font-size", None), P_SIZE)
    line = node.style.get("line-height")
    if line is None:
        line_px = P_LINE_HEIGHT * size
    elif line.endswith("px"):
        line_px = px(line)
    else:
        line_px = (float(line[:-1]) / 100 if line.endswith("%") else float(line)) * size
    return size, line_px


def baseline_in_line(size, line_px):
    """From the top of a line box to the baseline."""
    return (line_px - (ASCENT + DESCENT) * size) / 2 + ASCENT * size


def runs(node, bold=False, italic=False, color=None):
    for kid in node.kids:
        if kid.tag == "#text":
            yield [re.sub(r"\s+", " ", kid.text), bold, italic, color]
        else:
            yield from runs(kid, bold or kid.tag == "b", italic or kid.tag == "i", kid.style.get("color", color))


def text(node, x, baseline, anchor, family):
    size, _ = font(node)
    parts = [run for run in runs(node) if run[0]]
    for before, run in zip(parts, parts[1:]):
        if before[0].endswith(" "):
            run[0] = run[0].lstrip(" ")
    parts[0][0] = parts[0][0].lstrip(" ")
    parts[-1][0] = parts[-1][0].rstrip(" ")
    body = ""
    for words, bold, italic, color in parts:
        if not words:
            continue
        span = ((' font-weight="600"' if bold else "") + (' font-style="italic"' if italic else "")
                + (f' fill="{color}"' if color else ""))
        body += f"<tspan{span}>{html.escape(words, quote=False)}</tspan>" if span else html.escape(words, quote=False)
    weight = int(node.inherited("font-weight", P_WEIGHT))
    own_family = node.inherited("font-family", family)
    attrs = (f'x="{num(x)}" y="{num(baseline)}" font-size="{num(size)}"'
             + (f' font-weight="{weight}"' if weight != 400 else "")
             + (f' font-family="{own_family}"' if own_family != family else "")
             + f' fill="{node.inherited("color", "#000000")}"'
             + (f' text-anchor="{anchor}"' if anchor != "start" else ""))
    # preserve: the spaces between runs are exact here, and readers differ on trimming them.
    return f'<text {attrs} xml:space="preserve">{body}</text>'


def pinned_text(node, ox, oy, family):
    size, line_px = font(node)
    left, width = px(node.style.get("left"), 0.0), px(node.style.get("width"))
    align = node.style.get("text-align", "left")
    if width is None or align == "left":
        x, anchor = left, "start"
    elif align == "center":
        x, anchor = left + width / 2, "middle"
    else:
        x, anchor = left + width, "end"
    top = px(node.style.get("top"), 0.0)
    return text(node, ox + x, oy + top + baseline_in_line(size, line_px), anchor, family)


# ===== Boxes =====

def rect(x, y, w, h, style):
    """A painted <div>: its background and one stroke, drawn inside its border box as CSS does."""
    border = style.get("border", "").split()
    stroke_w = px(border[0]) if border else 0.0
    radius = px(style.get("border-radius"), 0.0)
    attrs = (f'x="{num(x + stroke_w / 2)}" y="{num(y + stroke_w / 2)}" width="{num(w - stroke_w)}"'
             f' height="{num(h - stroke_w)}"')
    if radius:
        attrs += f' rx="{num(max(0.0, radius - stroke_w / 2))}"'
    attrs += f' fill="{style.get("background", "none")}"'
    if border:
        attrs += f' stroke="{border[2]}" stroke-width="{num(stroke_w)}"'
        if border[1] == "dashed":
            # Chromium's dashes for a stroke under 3 px: three widths on, two off.
            attrs += f' stroke-dasharray="{num(3 * stroke_w)} {num(2 * stroke_w)}"'
    return f"<rect {attrs}/>"


def padding(style):
    sides = [px(side) for side in style.get("padding", "0").split()]
    top, right, bottom, left = (sides * 4)[:4] if len(sides) != 2 else sides * 2
    return (top, right, bottom, left) if len(sides) != 3 else (sides[0], sides[1], sides[2], sides[1])


def box(node, ox, oy, family):
    """A pinned <div> with a size: a rectangle, with at most one label inside."""
    style = node.style
    x, y = ox + px(style.get("left"), 0.0), oy + px(style.get("top"), 0.0)
    w, h = px(style["width"]), px(style["height"])
    out = [rect(x, y, w, h, style)] if "background" in style or "border" in style else []
    labels = node.elements()
    if not labels:
        return out
    if len(labels) > 1 or labels[0].tag != "p":
        sys.exit(f"box at left {style.get('left')}, top {style.get('top')}: more than one label is not drawn")
    size, line_px = font(labels[0])
    edge = px(style["border"].split()[0]) if "border" in style else 0.0
    top, right, bottom, left = padding(style)
    row = style.get("display") == "flex" and style.get("flex-direction", "row") == "row"
    across = style.get("justify-content" if row else "align-items", "start")
    down = style.get("align-items" if row else "justify-content", "start")
    inner_w, inner_h = w - 2 * edge - left - right, h - 2 * edge - top - bottom
    if across == "center":
        tx, anchor = x + edge + left + inner_w / 2, "middle"
    elif across in ("end", "flex-end"):
        tx, anchor = x + edge + left + inner_w, "end"
    else:
        tx, anchor = x + edge + left, "start"
    line_top = y + edge + top + ((inner_h - line_px) / 2 if down == "center" else 0.0)
    return out + [text(labels[0], tx, line_top + baseline_in_line(size, line_px), anchor, family)]


def legend_row(node, ox, oy, family):
    """A pinned <div> with no size: a row of swatches and one closing label, centered on the row's height."""
    style = node.style
    x, y, gap = ox + px(style.get("left"), 0.0), oy + px(style.get("top"), 0.0), px(style.get("gap"), 0.0)
    kids = node.elements()
    heights = [font(kid)[1] if kid.tag == "p" else px(kid.style["height"]) for kid in kids]
    out = []
    for kid, height in zip(kids, heights):
        top = y + ((max(heights) - height) / 2 if style.get("align-items") == "center" else 0.0)
        if kid.tag == "p":
            if kid is not kids[-1]:
                sys.exit("legend row: a label before the last item needs its width, which is not measured here")
            out.append(text(kid, x, top + baseline_in_line(*font(kid)), "start", family))
        else:
            out.append(rect(x, top, px(kid.style["width"]), height, kid.style))
            x += px(kid.style["width"]) + gap
    return out


def block(node, ox, oy, family, raw_svg):
    """One position:relative block at (ox, oy): its flow <svg> first, then its pinned children in order."""
    out = []
    kids = node.elements()
    for kid in kids:
        if kid.tag == "svg":
            out.append(raw_svg[id(kid)].replace("<svg", f'<svg x="{num(ox)}" y="{num(oy)}"', 1))
        elif not is_pinned(kid):
            sys.exit(f"<{kid.tag}> in a figure block is neither pinned nor an <svg>: not drawn")
    for kid in kids:
        if kid.tag == "p":
            out.append(pinned_text(kid, ox, oy, family))
        elif kid.tag == "div" and "width" in kid.style and "height" in kid.style:
            out += box(kid, ox, oy, family)
        elif kid.tag == "div":
            out += legend_row(kid, ox, oy, family)
        elif kid.tag != "svg":
            sys.exit(f"<{kid.tag}> pinned in a figure block: not drawn")
    return out


# ===== Figures =====

def figures(section):
    """The slide's figures, each a list of nodes to stack: blocks, and the flow <p> right above one as its title."""
    found = []
    for parent in section.walk():
        kids = parent.elements()
        column = parent.tag == "section" or is_column(parent)
        run = []
        for kid, after in zip(kids, kids[1:] + [None]):
            title = (column and kid.tag == "p" and not is_pinned(kid) and after is not None and is_block(after))
            if is_block(kid) or title:
                run.append(kid)
                if column:
                    continue
            if run:
                found.append(run)
                run = []
        if run:
            found.append(run)
    return found


def draw(parts, gap, family, raw_svg, title):
    width = max(px(part.style["width"]) for part in parts if is_block(part))
    body, y = [], 0.0
    for part in parts:
        if is_block(part):
            body += block(part, 0.0, y, family, raw_svg)
            y += px(part.style["height"]) + gap
        else:
            size, line_px = font(part)
            body.append(text(part, 0.0, y + baseline_in_line(size, line_px), "start", family))
            y += line_px + gap
    height = y - gap
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{num(width * 72 / SLIDE_PPI)}pt"'
            f' height="{num(height * 72 / SLIDE_PPI)}pt" viewBox="0 0 {num(width)} {num(height)}" font-family="{family}">')
    return "\n".join([head, f"<title>{html.escape(title, quote=False)}</title>"] + body + ["</svg>"]) + "\n", width, height


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    deck = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "results" / "pricing_the_room"
    out.mkdir(parents=True, exist_ok=True)
    order = json.loads((deck / "deck.json").read_text())["order"]
    written = 0
    for number, slide in enumerate(order, 1):
        source = (deck / "slides" / f"{slide}.html").read_text(encoding="utf-8")
        section = Tree(source).root.elements()[0]
        # html.parser lowercases names (viewBox, clipPath), so an inline <svg> is copied from the source text.
        svg_nodes = [node for node in section.walk() if node.tag == "svg"]
        raw_svg = {id(node): raw for node, raw in zip(svg_nodes, SVG_RE.findall(source))}
        heading = " ".join("".join(run[0] for run in runs(node)).strip()
                           for node in section.walk() if node.tag == "h2")
        found = figures(section)
        for index, parts in enumerate(found, 1):
            blocks = [part for part in parts if is_block(part)]
            charts = [node for part in blocks for node in part.elements() if node.tag == "svg"]
            what = "chart" if charts else "timelines" if len(blocks) > 1 else "diagram"
            name = f"{number:02d}_{slide}_{what}" + (f"_{index}" if len(found) > 1 else "")
            family = blocks[0].inherited("font-family", None) or next(
                node.style["font-family"] for node in blocks[0].walk() if "font-family" in node.style)
            title = charts[0].attrs["aria-label"] if charts and charts[0].attrs.get("aria-label") else heading
            gap = px(parts[0].parent.style.get("gap", "0").split()[0])
            svg, width, height = draw(parts, gap, family, raw_svg, title)
            (out / f"{name}.svg").write_text(svg, encoding="utf-8")
            written += 1
            print(f"{name:28s} {width:5.0f} x {height:4.0f} px on the slide, {len(blocks)} block(s)")
    print(written, "SVGs in", out)


if __name__ == "__main__":
    main()
