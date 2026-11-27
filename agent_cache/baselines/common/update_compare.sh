#!/usr/bin/env bash
# Add the finished arms of replay folders to a comparison folder's manifest and rerun compare_arms.py over it.
#   host: bash update_compare.sh <comparison folder> <cut s> <replay folder> [<replay folder> ...]
# A label already in the comparison's manifest is left as it is. Output: <comparison folder>/compare_arms.out.
set -uo pipefail
AC=/home/wanhr/sglang/agent_cache
CMP=$AC/results/$1 CUT=$2; shift 2
for f in "$@"; do
  while read -r label run prog; do
    [ -n "${label:-}" ] || continue
    grep -q "^$label " "$CMP/manifest.txt" || echo "$label $run $prog" >> "$CMP/manifest.txt"
  done < "$AC/results/$f/manifest.txt"
done
cd "$AC" && python3 -I -B frozen/oracle_scripts_staging/c28_users/compare_arms.py "$CMP" --cut "$CUT" > "$CMP/compare_arms.out" 2>&1
echo "arms: $(cut -d' ' -f1 "$CMP/manifest.txt" | tr '\n' ' ')"
sed -n '4,20p' "$CMP/compare_arms.out" | grep -E "^[a-z_]+ +\|" | cut -c1-180
