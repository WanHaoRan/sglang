#!/usr/bin/env python3
"""r2_04_parse_extra.py -- parse the two arms that finished after the task text was written.

They are not in the task's list of runs.  They are read only because the regenerated
compare_ec.out / pairs_ec.out (2026-11-12 01:33Z) print numbers on arm sets that hold them,
and because ec_r13_rep is the one repeat of a leading arm.  Labels and folders are taken from
results/campaign26/manifest.txt.  Refuses the run named in results/.current_run.

Run as:  nice -n 19 python3 -I -B r2_04_parse_extra.py
"""
import glob
import hashlib
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"
EXTRA = [
    ("jit_p50_d", "20261111_204059_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ec_r13_rep", "20261111_224624_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
]


def main():
    with open(os.path.join(RESULTS, ".current_run")) as f:
        current = f.read().strip()
    print("current run (never read):", repr(current))
    manifest = {}
    with open(os.path.join(RESULTS, "campaign26", "manifest.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                manifest[p[0]] = (p[1], p[2])
    out = {}
    for label, folder in EXTRA:
        if current and (folder == current or current in folder or folder in current):
            print("REFUSING to read the running directory", folder)
            sys.exit(2)
        if manifest.get(label, (None,))[0] != folder:
            print("manifest does not name", folder, "as", label, ":", manifest.get(label))
            sys.exit(3)
        cdirs = sorted(c for c in glob.glob(os.path.join(RESULTS, folder, "client_*"))
                       if os.path.isdir(c))
        cdir = cdirs[-1]
        if os.path.basename(cdir) != manifest[label][1]:
            print("client folder differs from the manifest:", cdir, manifest[label][1])
            sys.exit(3)
        path = os.path.join(cdir, "client.jsonl")
        st = os.stat(path)
        h = hashlib.sha256()
        recs, bad, nlines = [], [], 0
        with open(path, "rb") as f:
            for i, raw in enumerate(f):
                h.update(raw)
                nlines += 1
                s = raw.strip()
                if not s:
                    continue
                try:
                    recs.append(json.loads(s))
                except Exception as e:
                    bad.append((i, repr(e), raw[:200]))
        kinds = {}
        for r in recs:
            kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
        out[label] = {"label": label, "folder": folder, "n_client_dirs": len(cdirs),
                      "client_dir": os.path.basename(cdir), "path": path, "size": st.st_size,
                      "mtime": st.st_mtime, "sha256": h.hexdigest(), "n_lines": nlines,
                      "bad_lines": bad, "kinds": kinds, "records": recs}
        print("%-11s %s client_dirs=%d lines=%d bad=%d size=%d sha=%s kinds=%s"
              % (label, folder, len(cdirs), nlines, len(bad), st.st_size, h.hexdigest()[:16],
                 json.dumps(kinds, sort_keys=True)))
    dst = os.path.join(HERE, "r2_parsed_extra.pkl")
    with open(dst + ".tmp", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(dst + ".tmp", dst)
    print("wrote", dst, os.path.getsize(dst), "bytes")


if __name__ == "__main__":
    main()
