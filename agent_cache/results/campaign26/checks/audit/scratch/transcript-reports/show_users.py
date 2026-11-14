#!/usr/bin/env python3
"""Print real user prompts from the extracted file from a date on; long ones truncated."""
import os
import re
import sys

OUT = os.path.dirname(os.path.abspath(__file__))
since = sys.argv[1] if len(sys.argv) > 1 else "2026-11-11"
maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 2500

txt = open(os.path.join(OUT, "user_prompts.txt")).read()
parts = re.split(r"(?m)^=== L(\d+) (\S+) side=(\S+) len=(\d+)\n", txt)
# parts: [pre, idx, ts, side, len, body, idx, ts, ...]
i = 1
while i < len(parts):
    idx, ts, side, ln, body = parts[i], parts[i + 1], parts[i + 2], parts[i + 3], parts[i + 4]
    i += 5
    if ts < since:
        continue
    body = body.rstrip()
    if len(body) > maxlen:
        body = body[:maxlen] + "\n...[TRUNCATED, total %s chars]" % ln
    print("=== L%s %s side=%s len=%s\n%s\n" % (idx, ts, side, ln, body))
