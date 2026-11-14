# Reverse-apply the transcript's Edit calls (in memory) to the current plot scripts and compare md5 with scripts_before.md5.
import json, hashlib, sys, collections
T = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
S = "/tmp/claude-1001/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d/scratchpad/scripts_before.md5"
ROOT = "/home/wanhr/sglang/agent_cache/"
edits = collections.defaultdict(list)
with open(T, errors="replace") as f:
    for i, line in enumerate(f):
        if i < 30600: continue
        if i > 30700: break
        try: d = json.loads(line)
        except Exception: continue
        for c in ((d.get("message") or {}).get("content") or []):
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Edit":
                inp = c["input"]
                edits[inp["file_path"]].append((i, inp["old_string"], inp["new_string"], inp.get("replace_all")))
before = {}
for line in open(S):
    h, p = line.split()
    before[ROOT + p] = h
for path, es in edits.items():
    text = open(path, encoding="utf-8").read()
    cur = hashlib.md5(text.encode()).hexdigest()
    ok = True
    for i, old, new, ra in reversed(es):
        n = text.count(new)
        if n != 1:
            print("  edit at line", i, "new_string occurs", n, "times"); ok = False
        text = text.replace(new, old, 1)
    rev = hashlib.md5(text.encode()).hexdigest()
    print(path.replace(ROOT, ""), "edits:", [e[0] for e in es], "current", cur, "reverted", rev, "before", before.get(path), "MATCH" if rev == before.get(path) else "DIFFERENT")
    # is every edit purely additive (old_string is a prefix/suffix-preserved part of new_string)?
    for i, old, new, ra in es:
        additive = (old in new) or all(l in new for l in old.splitlines())
        added = [l for l in new.splitlines() if l not in old.splitlines()]
        removed = [l for l in old.splitlines() if l not in new.splitlines()]
        print("   edit", i, "lines added", len(added), "lines removed/reworded", len(removed))
        for l in removed: print("      - " + l[:200])
