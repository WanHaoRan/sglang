#!/usr/bin/env python3
"""Decode text that pypdf returned as '/65 /83/116...' glyph-code names (old dvips Type 3 fonts, OT1 encoding).
usage: python3 -I -B decode_slashcodes.py in.txt out.txt"""
import re, sys
LIG = {11: "ff", 12: "fi", 13: "fl", 14: "ffi", 15: "ffl", 16: "i", 17: "j", 18: "`", 19: "'", 34: '"', 92: '"', 123: "-", 124: "--", 3: "*", 0: "-", 1: ".", 2: "x", 15: "ffl"}
def dec(m):
    n = int(m.group(1))
    if n in LIG:
        return LIG[n]
    if 32 <= n < 127:
        return chr(n)
    return f"<{n}>"
src = open(sys.argv[1], errors="replace").read()
out = []
for line in src.splitlines():
    if line.startswith("/") or re.search(r"/\d+/\d+", line):
        words = line.split(" ")
        out.append(" ".join(re.sub(r"/(\d+)", dec, w) for w in words))
    else:
        out.append(line)
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("wrote", sys.argv[2])
