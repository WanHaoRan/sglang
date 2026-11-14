"""Read-only: compare a frozen engine tree's python/ with a commit's python/ by git blob hash.
python3 -I -B treecmp.py <frozen_dir> <commit> [<commit> ...]   (run with cwd anywhere; repo fixed below)"""
import hashlib, os, subprocess, sys
REPO = "/home/wanhr/sglang"
frozen = sys.argv[1]


def blob(path):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def tree(commit):
    out = subprocess.run(["git", "--no-optional-locks", "-C", REPO, "ls-tree", "-r", commit, "--", "python/"],
                         capture_output=True, text=True, check=True).stdout
    d = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        d[path] = meta.split()[2]
    return d


mine = {}
root = os.path.join(frozen, "python")
for dp, dn, fn in os.walk(root):
    dn[:] = [x for x in dn if x != "__pycache__"]
    for f in fn:
        if f.endswith(".pyc"):
            continue
        p = os.path.join(dp, f)
        if os.path.islink(p):
            continue
        mine["python/" + os.path.relpath(p, root)] = blob(p)
print(f"{frozen}: {len(mine)} files (no pyc)")
for commit in sys.argv[2:]:
    t = tree(commit)
    changed = sorted(p for p in mine if p in t and t[p] != mine[p])
    new = sorted(p for p in mine if p not in t)
    gone = sorted(p for p in t if p not in mine)
    print(f"  vs {commit}: tree has {len(t)}; changed {len(changed)}, only in frozen {len(new)}, only in commit {len(gone)}")
    for p in changed:
        print("     M", p)
    for p in new[:40]:
        print("     A", p)
    if len(new) > 40:
        print("     ... and", len(new) - 40, "more only in frozen")
    for p in gone[:15]:
        print("     D", p)
    if len(gone) > 15:
        print("     ... and", len(gone) - 15, "more only in commit")
