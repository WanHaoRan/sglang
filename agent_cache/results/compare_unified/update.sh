#!/usr/bin/env bash
# Paired comparison of the unified design's arms with the current design, per setting, across the sibling campaign
# folders (the launcher takes switches per campaign, so the arms of one setting live in several folders).
# For each setting it writes results/compare_unified/<setting>/manifest.txt from the arms that are done and runs
# compare_arms.py there (compare_arms.out). CPU only. Host side:
#   bash /home/wanhr/sglang/agent_cache/results/compare_unified/update.sh
set -uo pipefail
A=/home/wanhr/sglang/agent_cache
R=$A/results
O=$R/compare_unified
U=$A/frozen/oracle_scripts_staging/c28_users
# setting folder | settings.env taken from | "label campaign arm" ...
SETS=(
  "A_128u_160g|campaign35|ec_full_c34 campaign34 ec_full|ec_full campaign35 ec_full|ec_full_load campaign39 ec_full_load|unified campaign35 unified|unified_load campaign40 unified_load|unified_load_po campaign44 unified_load_po|unified_load_fixedprice campaign45 unified_load_fixedprice"
  "B_cleaner50|campaign38|ec_full campaign38 ec_full|unified campaign38 unified|unified_load campaign41 unified_load"
  "C_112g|campaign37|ec_full campaign37 ec_full|unified_load campaign42 unified_load"
  "D_64u_88g|campaign36|ec_full campaign36 ec_full|unified_load campaign43 unified_load"
)
for spec in "${SETS[@]}"; do
  IFS='|' read -r name from rest <<< "$spec"
  d=$O/$name; mkdir -p "$d/checks"
  cp "$R/$from/settings.env" "$d/settings.env"
  # The comparison reads the trace and the replay settings only; the switches differ by arm.
  sed -i '/^SGLANG_/d' "$d/settings.env"
  cp "$R/$from/checks/tree_systems.json" "$d/checks/" 2>/dev/null
  : > "$d/manifest.txt"
  IFS='|' read -ra arms <<< "$rest"
  for a in "${arms[@]}"; do
    read -r label camp arm <<< "$a"
    line=$(grep "^$arm " "$R/$camp/manifest.txt" 2>/dev/null | tail -1)
    [ -n "$line" ] || continue
    echo "$label ${line#* }" >> "$d/manifest.txt"
  done
  n=$(wc -l < "$d/manifest.txt")
  if [ "$n" -ge 2 ]; then
    ( cd "$A" && CUDA_VISIBLE_DEVICES= nice -n 10 python3 -I -B "$U/compare_arms.py" "results/compare_unified/$name" ) > "$d/compare_arms.out" 2> "$d/compare_arms.err"
    echo "== $name: $n arms"; grep -A12 "^pairs" "$d/compare_arms.out" | grep -v "^$" | cut -c1-140
  else
    echo "== $name: $n arm(s) done, nothing to pair yet"
  fi
done
# Figures in the campaign figure style for every setting with two arms or more (timeline.png/.pdf/.csv, ttft_cdf.png/.pdf).
docker exec -e CUDA_VISIBLE_DEVICES=7 -e PYTHONPATH=. -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
  'cd /sgl-workspace/sglang/agent_cache/scripts && nice -n 15 python3 -B ../results/compare_unified/figs.py 2>&1 | grep -v findfont'
# What each rule did per arm and window (compare_unified/<setting>/mechanisms.out).
for d in "$O"/*/; do
  [ "$(wc -l < "$d/manifest.txt" 2>/dev/null || echo 0)" -ge 2 ] || continue
  ( cd "$O" && nice -n 15 python3 -B mechanisms.py "$(basename "$d")" ) > "$d/mechanisms.out" 2>&1
done
# The cross-setting summary (compare_unified/summary.png/.pdf/.csv).
docker exec -e CUDA_VISIBLE_DEVICES=7 -e PYTHONPATH=. -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
  'cd /sgl-workspace/sglang/agent_cache/scripts && nice -n 15 python3 -B ../results/compare_unified/summary.py > /dev/null 2>&1'
