#!/usr/bin/env python3
"""r2_01_parse.py -- fresh, independent parser of the client logs (resumed attempt).

Reads client_*/client.jsonl (last client_* folder by name) of the 13 runs named in the
task and caches every record in r2_parsed.pkl in this folder.  Read-only on the runs.
Refuses to touch the run directory named in results/.current_run.

Run as:  nice -n 19 python3 -I -B r2_01_parse.py
"""
import glob
import hashlib
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"

RUNS = [
    ("baseline", "20261101_082229_three_tier_wc_custom_L1nat_H70"),
    ("base_rep", "20261105_075438_three_tier_wc_custom_L1nat_H70"),
    ("hintfree", "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70"),
    ("hintfree_ctl", "20261111_045808_three_tier_wc_dl_poll0_custom_L1nat_H70"),
    ("jit_p50", "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("jit_p50_ctl", "20261111_024927_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("jit_p50_d_cut", "20261111_151101_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("jit_exact", "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ec_r13", "20261111_070012_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ec_r1", "20261111_110558_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("nofc", "20261111_090311_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("nofc_noend", "20261111_130801_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ORDER", "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70"),
]


def main():
    t0 = time.time()
    with open(os.path.join(RESULTS, ".current_run")) as f:
        current = f.read().strip()
    print("current run (never read):", repr(current))
    out = {}
    for label, folder in RUNS:
        if current and (folder == current or current in folder or folder in current):
            print("REFUSING to read the running directory", folder)
            sys.exit(2)
        cdirs = sorted(glob.glob(os.path.join(RESULTS, folder, "client_*")))
        cdirs = [c for c in cdirs if os.path.isdir(c)]
        if not cdirs:
            print("NO client folder for", label, folder)
            sys.exit(3)
        cdir = cdirs[-1]
        path = os.path.join(cdir, "client.jsonl")
        st = os.stat(path)
        h = hashlib.sha256()
        recs = []
        bad = []
        nlines = 0
        blank = 0
        with open(path, "rb") as f:
            for i, raw in enumerate(f):
                h.update(raw)
                nlines += 1
                s = raw.strip()
                if not s:
                    blank += 1
                    continue
                try:
                    recs.append(json.loads(s))
                except Exception as e:  # truncated / malformed line
                    bad.append((i, repr(e), raw[:200]))
        kinds = {}
        for r in recs:
            k = r.get("kind") if isinstance(r, dict) else "<non-dict>"
            kinds[k] = kinds.get(k, 0) + 1
        out[label] = {
            "label": label,
            "folder": folder,
            "n_client_dirs": len(cdirs),
            "client_dirs": [os.path.basename(c) for c in cdirs],
            "client_dir": os.path.basename(cdir),
            "path": path,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "sha256": h.hexdigest(),
            "n_lines": nlines,
            "n_blank": blank,
            "bad_lines": bad,
            "kinds": kinds,
            "records": recs,
        }
        print(
            "%-14s %s  client_dirs=%d  lines=%d bad=%d blank=%d size=%d sha=%s kinds=%s"
            % (label, folder, len(cdirs), nlines, len(bad), blank, st.st_size,
               h.hexdigest()[:16], json.dumps(kinds, sort_keys=True))
        )
    dst = os.path.join(HERE, "r2_parsed.pkl")
    with open(dst + ".tmp", "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(dst + ".tmp", dst)
    print("wrote", dst, os.path.getsize(dst), "bytes in %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
