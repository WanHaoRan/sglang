#!/usr/bin/env python3
"""Parse the two arms that landed after the task was specified (jit_p50_d, ec_r13_rep)
into cache/, with the unchanged functions of parse_runs.py.

They are not in the task's run list; they are needed only because stall_ec.out and
compare_ec.out were regenerated with them at 01:33-01:35Z on 2026-11-12, after items
1-5 were frozen.  Everything computed from these two caches is "after opening".

Run as:  nice -n 19 python3 -I -B parse_new.py
"""
import json
import os
import sys
import time

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
sys.path.insert(0, OUT)
import parse_runs as PR  # noqa: E402

NEW = [
    ("jit_p50_d", "20261111_204059_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
    ("ec_r13_rep", "20261111_224624_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"),
]

with open(PR.RES + "/.current_run") as f:
    current = f.read().strip()
diag = ["parse_new.py  generated %s UTC" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "current run (never read): %r" % current]
for label, folder in NEW:
    if not current or folder == current or current in folder or folder in current:
        raise SystemExit("refusing to read the live run folder: " + folder)
    run_dir = os.path.join(PR.RES, folder)
    ds, cd = PR.last_client_dir(run_dir)
    diag.append("")
    diag.append("== %s  %s" % (label, folder))
    diag.append("  client dirs: %s  -> using %s" % (ds, cd))
    cpath = os.path.join(run_dir, cd, "client.jsonl")
    spath = os.path.join(run_dir, "server.log")
    t_start, run_rec, turns = PR.parse_client(cpath, diag)
    srv = PR.parse_server(spath, diag)
    diag.append("  client t_start=%r  turns=%d  ttft null=%d  errors=%d  dup keys=%d" % (
        t_start, len(turns), sum(1 for t in turns if t["ttft"] is None),
        sum(1 for t in turns if t["error"]),
        len(turns) - len(set((t["conv"], t["turn"]) for t in turns))))
    out = {
        "label": label, "folder": folder, "client_dir": cd,
        "client_size": os.path.getsize(cpath), "server_size": os.path.getsize(spath),
        "client_mtime": os.path.getmtime(cpath), "server_mtime": os.path.getmtime(spath),
        "t_start": t_start, "run_args": (run_rec or {}).get("args"),
        "run_turns_field": (run_rec or {}).get("turns"),
        "run_users_field": (run_rec or {}).get("users"),
        "turns": turns,
    }
    out.update(srv)
    with open(os.path.join(PR.CACHE, label + ".json"), "w") as f:
        json.dump(out, f)
    diag.append("  cached -> %s" % os.path.join(PR.CACHE, label + ".json"))
with open(os.path.join(OUT, "parse_diag_new.txt"), "w") as f:
    f.write("\n".join(diag) + "\n")
print("\n".join(diag))
