"""Read-only: sizes of what commit a5fe6121c8 added/changed, by folder, from the ls-tree dumps and the name-status list."""
import collections, os, sys
head, base = {}, {}
for fn, d in (("tree_head.txt", head), ("tree_base.txt", base)):
    for line in open(fn, errors="replace"):
        meta, path = line.rstrip("\n").split("\t", 1)
        mode, typ, sha, size = meta.split()
        d[path] = (sha, int(size) if size != "-" else 0)
changed = [l.rstrip("\n").split("\t", 1) for l in open("commit_names.txt", errors="replace")]
tot = sum(head[p][1] for s, p in changed)
print(f"commit touches {len(changed)} paths; blobs at the commit sum to {tot/1e6:.1f} MB ({tot/2**20:.1f} MiB)")
def group(p):
    for pre in ("agent_cache/results/_trial_c128/", "agent_cache/results/campaign26/checks/verify/", "agent_cache/results/campaign26/checks/",
                "agent_cache/results/campaign26/", "agent_cache/results/compare_c26_designs_c128/", "agent_cache/results/compare_c26_designs/",
                "agent_cache/results/compare_c26_ec/", "agent_cache/results/", "agent_cache/scripts/", "agent_cache/", "python/", "test/"):
        if p.startswith(pre):
            return pre
    return p.split("/")[0] + "/"
g = collections.defaultdict(lambda: [0, 0])
for s, p in changed:
    k = group(p)
    g[k][0] += 1
    g[k][1] += head[p][1]
for k, (n, b) in sorted(g.items(), key=lambda kv: -kv[1][1]):
    print(f"  {k:55s} {n:5d} files {b/1e6:9.1f} MB")
for pre in ("agent_cache/results/_trial_c128/", "agent_cache/results/campaign26/checks/verify/"):
    files = [(p, sz) for p, (sha, sz) in head.items() if p.startswith(pre)]
    b = sum(sz for _, sz in files)
    in_base = sum(1 for p, _ in files if p in base)
    print(f"{pre}: {len(files)} tracked at HEAD, {b/1e6:.1f} MB ({b/2**20:.1f} MiB); {in_base} of them already in b684fd6464")
    pk = [(p, sz) for p, sz in files if p.endswith(".pkl")]
    print(f"   .pkl: {len(pk)} files, {sum(sz for _, sz in pk)/1e6:.1f} MB ({sum(sz for _, sz in pk)/2**20:.1f} MiB)")
    big = sorted(files, key=lambda x: -x[1])[:8]
    for p, sz in big:
        print(f"   {sz/1e6:8.1f} MB ({sz/2**20:6.1f} MiB)  {p}")
    ext = collections.Counter(os.path.splitext(p)[1] for p, _ in files)
    print("   by extension:", dict(ext.most_common(12)))
over50 = sorted(((p, sz) for s, p in changed for sz in [head[p][1]] if sz > 50 * 2**20), key=lambda x: -x[1])
print("blobs over 50 MiB in the commit:", len(over50))
for p, sz in over50:
    print(f"   {sz/2**20:7.1f} MiB  {p}")
over100 = [x for x in over50 if x[1] > 100 * 2**20]
print("blobs over 100 MiB (GitHub's hard limit):", len(over100))
# what kinds of files did the commit add outside those two folders
print("tracked .out at HEAD:", sum(1 for p in head if p.endswith(".out")), "| tracked under agent_cache/frozen:", sum(1 for p in head if p.startswith("agent_cache/frozen/")),
      "| tracked .pdf under agent_cache:", sum(1 for p in head if p.startswith("agent_cache/") and p.endswith(".pdf")))
