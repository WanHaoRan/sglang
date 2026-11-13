#!/usr/bin/env python3
"""Tally compare2.json (current files) and compare.json (versions of 21:07Z 2026-11-11).

Run as:  nice -n 19 python3 -I -B tally2.py
"""
import json
import os
from collections import Counter

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
lines = []


def P(s=""):
    lines.append(s)
    print(s)


for name in ("compare2.json", "compare.json"):
    with open(os.path.join(OUT, name)) as f:
        rows = json.load(f)
    num = [r for r in rows if r["mine_raw"] is not None]
    P("%s: rows %d, numeric %d (equal %d, different %d), non-numeric %d (equal %d)" % (
        name, len(rows), len(num), sum(r["match"] for r in num), sum(not r["match"] for r in num),
        len(rows) - len(num), sum(r["match"] for r in rows if r["mine_raw"] is None)))
    c = Counter((r["file"], r["prov"]) for r in num)
    ok = Counter((r["file"], r["prov"]) for r in num if r["match"])
    for f in ("stall_ec.out", "stall_ec_ctl.out", "split_ec_lead.out", "compare_ec.out"):
        P("   %-18s numeric: " % f + "  ".join("%s %d/%d" % (p, ok[(f, p)], c[(f, p)]) for p in ("frozen", "text", "after"))
          + "   total %d/%d" % (sum(ok[(f, p)] for p in ("frozen", "text", "after")), sum(c[(f, p)] for p in ("frozen", "text", "after"))))
    P("   all files numeric by provenance: " + "  ".join("%s %d/%d" % (
        p, sum(v for (f, q), v in ok.items() if q == p), sum(v for (f, q), v in c.items() if q == p)) for p in ("frozen", "text", "after")))
    if name == "compare2.json":
        # frozen-provenance rows by the task item they rest on (from the 'what' text)
        byitem = Counter()
        for r in num:
            if r["prov"] != "frozen":
                continue
            w = r["what"]
            tag = "other frozen figure"
            for key, t in (("(item 1)", "item 1"), ("(item 2)", "item 2"), ("(items 2,3)", "items 2+3"), ("(item 3)", "item 3"),
                           ("item 2 definition", "item 2")):
                if key in w:
                    tag = t
                    break
            byitem[(r["file"], tag)] += 1
        for k, v in sorted(byitem.items()):
            P("   frozen rows  %-18s %-22s %d" % (k[0], k[1], v))
        new = [r for r in num if r["arm"] in ("jit_p50_d", "ec_r13_rep")]
        P("   numeric rows of the two new arms: %d (equal %d); per-15-min rows: %d" % (
            len(new), sum(r["match"] for r in new), sum(1 for r in num if r["section"].startswith("per 15 min"))))
    P()
with open(os.path.join(OUT, "tally2.out"), "w") as f:
    f.write("\n".join(lines) + "\n")
