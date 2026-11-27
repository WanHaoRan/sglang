"""Three-way merge of the two implementation copies (A: Fetch and Price; B: Retain and the store) and the working
tree (the load hold) over their common base, into a staging folder.

Usage: python3 merge.py <staging folder>
Reads agent_cache/frozen/impl_unified_{base,A,B} and the working tree; writes <staging>/python and
<staging>/test/registered/unit (the working tree's with the merged files), and prints one line per merged file.
Where two sides add lines at the same place both are kept (git merge-file --union) and the file is listed
(UNION) for review.
"""
import filecmp, os, shutil, subprocess, sys

ROOT = "/home/wanhr/sglang"
F = ROOT + "/agent_cache/frozen"
TREES = (("python/sglang", "python/sglang"), ("test/registered/unit/mem_cache", "test/registered/unit/mem_cache"))


def files(top):
    out = set()
    for d, _, names in os.walk(top):
        if "__pycache__" in d:
            continue
        for n in names:
            if n.endswith(".pyc"):
                continue
            out.add(os.path.relpath(os.path.join(d, n), top))
    return out


def same(a, b):
    return os.path.exists(a) and os.path.exists(b) and filecmp.cmp(a, b, shallow=False)


def main():
    stage = sys.argv[1]
    conflicts = 0
    for sub, _ in TREES:
        base, a, b, work = (f"{F}/impl_unified_base/{sub}", f"{F}/impl_unified_A/{sub}", f"{F}/impl_unified_B/{sub}",
                            f"{ROOT}/{sub}")
        dst = f"{stage}/{sub}"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(work, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for rel in sorted(files(a) | files(b) | files(base)):
            pb, pa, pB, pw, pd = (f"{base}/{rel}", f"{a}/{rel}", f"{b}/{rel}", f"{work}/{rel}", f"{dst}/{rel}")
            ca = not same(pa, pb) and os.path.exists(pa)
            cb = not same(pB, pb) and os.path.exists(pB)
            cw = not same(pw, pb) and os.path.exists(pw)
            if not ca and not cb:
                continue
            os.makedirs(os.path.dirname(pd), exist_ok=True)
            if not os.path.exists(pb):
                # A new file: one copy may add it.
                src = [p for p, c in ((pa, ca), (pB, cb)) if c]
                if len(src) == 2 and not same(pa, pB):
                    print(f"CONFLICT {sub}/{rel}: added by both copies with different content")
                    conflicts += 1
                shutil.copyfile(src[0], pd)
                print(f"new      {sub}/{rel} from {'A' if src[0] == pa else 'B'}")
                continue
            cur = pd if cw else None
            if cur is None:
                shutil.copyfile(pb, pd)
            who = []
            for name, other, changed in (("A", pa, ca), ("B", pB, cb)):
                if not changed:
                    continue
                # Probe for conflicts, then merge keeping both sides of each (both copies add at the same
                # places); every such file is listed for review.
                probe = subprocess.run(["git", "merge-file", "-p", pd, pb, other], capture_output=True)
                subprocess.run(["git", "merge-file", "--union", pd, pb, other], check=True)
                who.append(name)
                if probe.returncode != 0:
                    print(f"UNION    {sub}/{rel}: {probe.returncode} place(s) where {name} and the others add; both kept")
                    conflicts += 1
            print(f"merged   {sub}/{rel}: {'+'.join(who)}{' + working tree' if cw else ''}")
    print(f"files with places merged by union: {conflicts}")
    return 0


sys.exit(main())
