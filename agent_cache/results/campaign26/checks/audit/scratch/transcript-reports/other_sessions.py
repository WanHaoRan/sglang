#!/usr/bin/env python3
"""List real user prompts (first 300 chars) of the other session files of this project, read-only."""
import json, os, sys, glob
D = "/home/wanhr/.claude/projects/-home-wanhr-sglang"
SKIP = ("<system-reminder>", "<task-notification>", "<local-command", "<command-")
for p in sorted(glob.glob(D + "/*.jsonl")):
    if "7c93b1d5" in p:
        continue
    n = 0; first = last = None; prompts = []; texts = 0
    for idx, line in enumerate(open(p, errors="replace")):
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp")
        if ts:
            first = first or ts; last = ts
        if o.get("type") == "user" and not o.get("isMeta"):
            c = (o.get("message") or {}).get("content")
            items = [c] if isinstance(c, str) else [b.get("text") or "" for b in (c or []) if isinstance(b, dict) and b.get("type") == "text"]
            for t in items:
                t = t.strip()
                if t and not t.startswith(SKIP):
                    prompts.append((idx, ts, t[:300].replace("\n", " ")))
        elif o.get("type") == "assistant":
            for b in (o.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "text" and len(b.get("text") or "") > 300:
                    texts += 1
    print("=== %s size=%d first=%s last=%s long_assistant_texts=%d user_prompts=%d" % (os.path.basename(p), os.path.getsize(p), first, last, texts, len(prompts)))
    for pr in prompts[-14:]:
        print("   L%d %s %s" % pr)
