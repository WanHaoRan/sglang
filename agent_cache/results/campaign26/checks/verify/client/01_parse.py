#!/usr/bin/env python3
"""Independent parser for campaign 26 client-side results.

Reads client_*/client.jsonl (last client_* folder by name) of each run and
caches the records in this folder (parsed.pkl).  Read-only on the run folders.
Run as:  nice -n 19 python3 -I -B 01_parse.py
"""
import glob
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
    with open(os.path.join(RESULTS, ".current_run")) as f:
        current = f.read().strip()
    out = {}
    for label, folder in RUNS:
        if current and (folder == current or current in folder or folder in current):
            print("REFUSING to read the run being written:", folder)
            sys.exit(2)
        cdirs = sorted(glob.glob(os.path.join(RESULTS, folder, "client_*")))
        cdirs = [c for c in cdirs if os.path.isdir(c)]
        if not cdirs:
            print("no client folder for", label, folder)
            sys.exit(2)
        path = os.path.join(cdirs[-1], "client.jsonl")
        st = os.stat(path)
        kinds = {}
        bad = []
        run_recs = []
        turns = []
        conv_start = []
        other = {}
        nlines = 0
        with open(path, "rb") as f:
            for lineno, raw in enumerate(f, 1):
                nlines += 1
                s = raw.decode("utf-8", errors="replace").strip()
                if not s:
                    kinds["<blank>"] = kinds.get("<blank>", 0) + 1
                    continue
                try:
                    rec = json.loads(s)
                except Exception as e:  # truncated or broken line
                    bad.append((lineno, len(s), s[:120], repr(e)[:120]))
                    continue
                if not isinstance(rec, dict):
                    bad.append((lineno, len(s), s[:120], "not a dict"))
                    continue
                k = rec.get("kind", "<nokind>")
                kinds[k] = kinds.get(k, 0) + 1
                rec["_line"] = lineno
                if k == "run":
                    run_recs.append(rec)
                elif k == "turn":
                    turns.append(rec)
                elif k == "conv_start":
                    conv_start.append(rec)
                else:
                    other.setdefault(k, []).append(rec)
        out[label] = {
            "label": label,
            "folder": folder,
            "client_dirs": [os.path.basename(c) for c in cdirs],
            "path": path,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "nlines": nlines,
            "kinds": kinds,
            "bad": bad,
            "run": run_recs,
            "turns": turns,
            "conv_start": conv_start,
            "other": other,
        }
        print(
            "%-14s lines=%6d kinds=%s bad=%d client_dirs=%d size=%d mtime=%s"
            % (
                label,
                nlines,
                json.dumps(kinds, sort_keys=True),
                len(bad),
                len(cdirs),
                st.st_size,
                time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(st.st_mtime)),
            )
        )
        for b in bad[:5]:
            print("    bad line:", b)
    with open(os.path.join(HERE, "parsed.pkl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("wrote", os.path.join(HERE, "parsed.pkl"))


if __name__ == "__main__":
    main()
