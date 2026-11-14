import json, sys, glob, os
base = "/home/wanhr/.claude/projects/-home-wanhr-sglang/"
for name in ["454133c8-c6f1-4d1f-8d97-1e8cda557a0d.jsonl", "45edf7fb-7509-4ab1-95e0-795434f50dc8.jsonl", "566bc6cc-fbd6-4d85-9296-9718834f104f.jsonl", "ef917f01-705a-4292-8b9e-78b27c7008d3.jsonl", "e6eba98b-134c-4efd-88ea-79b3bfc41250.jsonl"]:
    print("=====", name)
    with open(base + name, "r", errors="replace") as f:
        for i, line in enumerate(f):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "user":
                continue
            if d.get("isMeta") or d.get("isCompactSummary"):
                continue
            ts = d.get("timestamp", "")
            m = d.get("message", {})
            c = m.get("content")
            texts = []
            if isinstance(c, str):
                texts.append(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        texts.append(b.get("text", ""))
            if not texts:
                continue
            t = "\n".join(texts)
            if t.startswith("<task-notification>") or t.startswith("[Request interrupted") or t.startswith("<system-reminder>") or t.startswith("<local-command") or t.startswith("<command-name>"):
                continue
            print(f"[{i}] {ts} len={len(t)} cwd={d.get('cwd','')[-40:]} :: {t[:260]!r}")
