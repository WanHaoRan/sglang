"""Sample pixel colours in the chart-1 screenshot: whisker, dot, background; WCAG contrast against the background."""
import sys
import matplotlib.image as mpimg
img = mpimg.imread(sys.argv[1])
def rgb(x, y):
    p = img[y, x]
    return tuple(int(round(c * 255)) for c in p[:3])
def lum(c):
    def ch(v):
        v = v / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(v) for v in c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
bg = rgb(700, 160)
print("image", img.shape, "background", bg)
for name, x, y in [("vllm-lmcache whisker", 552, 199), ("vllm-lmcache dot", 570, 199), ("continuum whisker", 830, 496),
                   ("continuum whisker r", 1000, 496), ("hicache whisker", 560, 273), ("ours line", 353, 400)]:
    c = rgb(x, y)
    print(f"{name:22s} {c}  contrast {contrast(c, bg):.2f}:1")
