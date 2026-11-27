"""Advance widths (in em) of the characters the slides use, read from the deck's font files (cmap + hmtx).

Usage: python3 -I ttf_widths.py <folder with the .ttf files> <deck root> > font_metrics.json
"""
import glob, json, struct, sys


def tables(data):
    n = struct.unpack(">H", data[4:6])[0]
    out = {}
    for i in range(n):
        tag, _, off, length = struct.unpack(">4sIII", data[12 + 16 * i: 28 + 16 * i])
        out[tag.decode()] = (off, length)
    return out


def cmap(data, off):
    n = struct.unpack(">H", data[off + 2: off + 4])[0]
    best = None
    for i in range(n):
        pid, eid, sub = struct.unpack(">HHI", data[off + 4 + 8 * i: off + 12 + 8 * i])
        fmt = struct.unpack(">H", data[off + sub: off + sub + 2])[0]
        if fmt == 12 and pid in (0, 3):
            best = (12, off + sub)
            break
        if fmt == 4 and pid in (0, 3) and best is None:
            best = (4, off + sub)
    fmt, o = best
    m = {}
    if fmt == 12:
        groups = struct.unpack(">I", data[o + 12: o + 16])[0]
        for g in range(groups):
            s, e, gid = struct.unpack(">III", data[o + 16 + 12 * g: o + 28 + 12 * g])
            for c in range(s, e + 1):
                m[c] = gid + c - s
        return m
    segx2 = struct.unpack(">H", data[o + 6: o + 8])[0]
    seg = segx2 // 2
    ends = struct.unpack(f">{seg}H", data[o + 14: o + 14 + segx2])
    p = o + 16 + segx2
    starts = struct.unpack(f">{seg}H", data[p: p + segx2]); p += segx2
    deltas = struct.unpack(f">{seg}h", data[p: p + segx2]); p += segx2
    ro_at = p
    ros = struct.unpack(f">{seg}H", data[p: p + segx2])
    for i in range(seg):
        for c in range(starts[i], ends[i] + 1):
            if c == 0xFFFF:
                continue
            if ros[i] == 0:
                gid = (c + deltas[i]) & 0xFFFF
            else:
                q = ro_at + 2 * i + ros[i] + 2 * (c - starts[i])
                gid = struct.unpack(">H", data[q: q + 2])[0]
                if gid:
                    gid = (gid + deltas[i]) & 0xFFFF
            if gid:
                m[c] = gid
    return m


def widths(path, chars):
    data = open(path, "rb").read()
    t = tables(data)
    upm = struct.unpack(">H", data[t["head"][0] + 18: t["head"][0] + 20])[0]
    nh = struct.unpack(">H", data[t["hhea"][0] + 34: t["hhea"][0] + 36])[0]
    hm = t["hmtx"][0]
    cm = cmap(data, t["cmap"][0])
    out, missing = {}, []
    for ch in chars:
        gid = cm.get(ord(ch))
        if gid is None:
            missing.append(ch)
            continue
        i = min(gid, nh - 1)
        out[ch] = struct.unpack(">H", data[hm + 4 * i: hm + 4 * i + 2])[0] / upm
    return out, missing


def main():
    folder, root = sys.argv[1], sys.argv[2]
    chars = set(" ")
    for f in glob.glob(root + "/project/slides/*.html"):
        chars |= set(open(f, encoding="utf-8").read())
    chars |= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?-–—()[]{}%+=<>/*'\"’‘“”·×→←≈≤≥−λμκνγβτρσΔ∞²³⁻⁷⁸⁵⁶⁴¹⁰₀₁₂ …")
    chars = sorted(c for c in chars if c not in "\n\r\t")
    out = {}
    for name in ("plex400", "plex600", "plex400i", "serif400", "serif600"):
        w, missing = widths(f"{folder}/{name}.ttf", chars)
        out[name] = w
        print(name, "missing:", "".join(missing), file=sys.stderr)
    json.dump(out, sys.stdout, ensure_ascii=False, sort_keys=True)


main()
