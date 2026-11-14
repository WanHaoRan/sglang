import sys, re
src = sys.argv[1]
txt = open(src, encoding="utf-8").read()
blocks = re.split(r"(?m)^##### ", txt)
for b in blocks:
    if not b.strip():
        continue
    head, _, body = b.partition("\n")
    if "meta=True" in head:
        continue
    body = body.strip()
    if body.startswith("<task-notification>") or body.startswith("<artifact-content-authored-by-others/>") or body.startswith("This session is being continued") or body.startswith("[Request interrupted"):
        continue
    if body.startswith("<local-command") or body.startswith("<command-name>") or body.startswith("Caveat:"):
        print("#####", head, "\n   [command]", body[:300].replace("\n", " "), "\n")
        continue
    print("#####", head)
    print(body[:2500])
    print()
