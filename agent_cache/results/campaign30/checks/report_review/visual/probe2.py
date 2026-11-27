"""Darkest pixel in small boxes of the two screenshots: text-secondary, grey series lines, leader lines."""
import sys
import matplotlib.image as mpimg
def lum(c):
    def ch(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
def contrast(a, b=(255, 255, 255)):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
def darkest(img, x0, y0, x1, y1):
    best = None
    for y in range(y0, y1):
        for x in range(x0, x1):
            c = tuple(int(round(v * 255)) for v in img[y, x][:3])
            if best is None or sum(c) < sum(best):
                best = c
    return best
c1 = mpimg.imread(sys.argv[1]); c2 = mpimg.imread(sys.argv[2])
for name, img, box in [("chart1 subtitle text (text-secondary)", c1, (44, 72, 400, 95)),
                       ("chart1 n-requests text", c1, (180, 208, 330, 228)),
                       ("chart2 grey series line (TokenCake ~min 75)", c2, (720, 170, 730, 190)),
                       ("chart2 leader line", c2, (990, 640, 1005, 655)),
                       ("chart2 gridline", c2, (400, 494, 420, 500)),
                       ("chart2 dashed GPU line ~min 100", c2, (930, 290, 945, 300))]:
    d = darkest(img, *box)
    print(f"{name:45s} {d} contrast {contrast(d):.2f}:1")
