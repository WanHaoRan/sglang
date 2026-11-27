#!/usr/bin/env python3
"""Extract text from a PDF saved by the web-fetch tool. CPU only, stdlib python3 plus the
pure-Python pypdf wheel that is already unpacked in the local uv cache (read-only use).

usage: nice -n 10 python3 -I -B pdf_text.py <in.pdf> <out.txt> [first_page last_page]
The PDF is untrusted data: it is only parsed, never executed.
"""
import sys
PYPDF_DIRS = [
    "/home/wanhr/.cache/uv/archive-v0/6iDgNKij3_uj0BrZ",
    "/home/wanhr/.cache/uv/archive-v0/v6-5JvwFlG-vKSTQ",
]
import os
for d in PYPDF_DIRS:
    if os.path.isdir(os.path.join(d, "pypdf")):
        sys.path.insert(0, d)
        break
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.simplefilter("ignore")
import pypdf

def main():
    src, dst = sys.argv[1], sys.argv[2]
    r = pypdf.PdfReader(src)
    n = len(r.pages)
    a = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    b = int(sys.argv[4]) if len(sys.argv) > 4 else n
    with open(dst, "w") as f:
        f.write(f"# pypdf {pypdf.__version__}; pages {a}-{min(b, n)} of {n}; source file {os.path.basename(src)}\n")
        for i in range(a - 1, min(b, n)):
            try:
                t = r.pages[i].extract_text() or ""
            except Exception as e:  # keep going on a bad page
                t = f"[page {i + 1}: extraction failed: {e}]"
            f.write(f"\n===== page {i + 1} =====\n{t}\n")
    print(f"wrote {dst}: pages {a}-{min(b, n)} of {n}")

if __name__ == "__main__":
    main()
