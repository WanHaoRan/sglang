import sys, re
txt = open("usermsgs_main.txt", errors="replace").read()
blocks = re.split(r'(?m)^(?==== idx0=)', txt)
for b in blocks:
    if not b.strip():
        continue
    head, _, body = b.partition("\n")
    m = re.search(r'idx0=(\d+) ts=(\S+) meta=(\S+) len=(\d+)', head)
    if not m:
        continue
    idx, ts, meta, ln = m.groups()
    body = body.strip()
    # skip task notifications and system reminders
    if body.startswith("<task-notification") or body.startswith("<system-reminder") or body.startswith("<local-command") or body.startswith("<command-name") or "task-notification" in body[:200]:
        print("--- %s %s [skipped: %s]" % (idx, ts, body[:90].replace("\n", " ")))
        continue
    print("=== %s %s meta=%s len=%s" % (idx, ts, meta, ln))
    print(body[:900])
    print()
