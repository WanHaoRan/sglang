"""Compare chart 2's inline rows (chart2_rows.txt, copied from the doc widget) with the recomputed bins (bins.json)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "chart2_rows.txt")) as f:
    text = f.read()
rows = re.findall(r"\{slug:'([^']+)',minute:(\d+),ttft:([0-9.]+),n:(\d+)\}", text)
with open(os.path.join(HERE, "bins.json")) as f:
    bins = json.load(f)
bad = 0
for slug, minute, ttft, n in rows:
    want_n, want_t = bins[slug][minute]
    if int(n) != want_n or abs(float(ttft) - want_t) > 0.005:
        bad += 1
        print(f"MISMATCH {slug} {minute}: doc {ttft} s n {n}; recomputed {want_t} s n {want_n}")
per = {}
for slug, minute, _, _ in rows:
    per.setdefault(slug, set()).add(int(minute))
print(f"rows {len(rows)}; series {sorted(per)}; bins per series {sorted({len(v) for v in per.values()})}; mismatches {bad}")
for slug in per:
    missing = set(int(m) for m in bins[slug]) - per[slug]
    if missing:
        print(f"{slug}: recomputed bins not in the doc: {sorted(missing)}")
