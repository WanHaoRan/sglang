"""Crop a rectangle out of an 8-bit RGB/RGBA non-interlaced PNG and write it scaled up by an integer factor. Stdlib only.
usage: crop.py src.png dst.png x0 y0 x1 y1 [scale]"""
import struct, sys, zlib

src, dst, x0, y0, x1, y1 = sys.argv[1], sys.argv[2], *map(int, sys.argv[3:7])
scale = int(sys.argv[7]) if len(sys.argv) > 7 else 1
data = open(src, "rb").read()
assert data[:8] == b"\x89PNG\r\n\x1a\n"
pos, idat, ihdr = 8, [], None
while pos < len(data):
    n, kind = struct.unpack(">I4s", data[pos:pos + 8])
    body = data[pos + 8:pos + 8 + n]
    if kind == b"IHDR":
        ihdr = struct.unpack(">IIBBBBB", body)
    elif kind == b"IDAT":
        idat.append(body)
    pos += 12 + n
w, h, depth, ctype, _, _, interlace = ihdr
assert depth == 8 and ctype in (2, 6) and interlace == 0, ihdr
bpp = 3 if ctype == 2 else 4
raw = zlib.decompress(b"".join(idat))
stride = w * bpp
x1, y1 = min(x1, w), min(y1, h)
prev = bytearray(stride)
rows = []
for y in range(y1):
    f = raw[y * (stride + 1)]
    line = bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
    if f == 1:
        for i in range(bpp, stride):
            line[i] = (line[i] + line[i - bpp]) & 255
    elif f == 2:
        for i in range(stride):
            line[i] = (line[i] + prev[i]) & 255
    elif f == 3:
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            line[i] = (line[i] + ((a + prev[i]) >> 1)) & 255
    elif f == 4:
        for i in range(stride):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
    prev = line
    if y >= y0:
        rows.append(bytes(line[x0 * bpp:x1 * bpp]))
out = bytearray()
for r in rows:
    if scale > 1:
        r = b"".join(r[i:i + bpp] * scale for i in range(0, len(r), bpp))
    for _ in range(scale):
        out += b"\x00" + r
def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", (x1 - x0) * scale, len(rows) * scale, 8, ctype, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(bytes(out), 6)) + chunk(b"IEND", b"")
open(dst, "wb").write(png)
print("wrote", dst, (x1 - x0) * scale, "x", len(rows) * scale, "from", w, "x", h, "ctype", ctype)
