#!/usr/bin/env python3
"""Look at non user/assistant line types from 2026-11-11 on: queue-operation, attachment kinds, system."""
import json
import os
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
OUT = os.path.dirname(os.path.abspath(__file__))
CUT = "2026-11-11T00:00"


def short(o, n=600):
    s = json.dumps(o, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + "...[%d]" % len(s)


def main():
    att_kinds = {}
    fq = open(os.path.join(OUT, "queue_ops.txt"), "w")
    fa = open(os.path.join(OUT, "attachments.txt"), "w")
    fsys = open(os.path.join(OUT, "system.txt"), "w")
    last_ts = ""
    with open(SRC, "r", errors="replace") as f:
        for idx, line in enumerate(f):
            # cheap prefilter
            if '"type":"queue-operation"' not in line and '"type":"attachment"' not in line and '"type":"system"' not in line and '"timestamp"' not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = obj.get("type")
            ts = obj.get("timestamp") or ""
            if ts:
                last_ts = ts
            eff = ts or last_ts
            if eff < CUT:
                continue
            if t == "queue-operation":
                fq.write("=== L%d %s\n%s\n\n" % (idx, eff, short(obj, 3000)))
            elif t == "attachment":
                a = obj.get("attachment") or {}
                k = a.get("type")
                att_kinds[k] = att_kinds.get(k, 0) + 1
                if k in ("queued_command", "queued-command", "user_message", "prompt"):
                    fa.write("=== L%d %s kind=%s\n%s\n\n" % (idx, eff, k, short(a, 4000)))
            elif t == "system":
                fsys.write("=== L%d %s\n%s\n\n" % (idx, eff, short(obj, 500)))
    fq.close(); fa.close(); fsys.close()
    print("attachment kinds since cut:", att_kinds)


if __name__ == "__main__":
    main()
