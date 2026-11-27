#!/usr/bin/env bash
# Format with ruff 0.15.1 (the repo's ruff-format) only the line ranges of a file that differ from git HEAD, bottom
# up, so lines HEAD already has keep HEAD's formatting. Host side: ruff_changed_ranges.sh <file> ...
set -uo pipefail
R=agent_cache/results/design_unified_1126/impl/ruff_pkg/bin/ruff
for f in "$@"; do
  # New-file line ranges of every hunk (start,count), last first.
  ranges=$(git diff -U0 HEAD -- "$f" | grep -o '^@@ -[0-9,]* +[0-9,]* @@' | sed -E 's/^@@ -[0-9,]* \+([0-9]+)(,([0-9]+))? @@/\1 \3/' \
    | awk '{c = ($2 == "" ? 1 : $2); if (c > 0) print $1":"$1 + c - 1}' | sort -t: -k1,1nr)
  for r in $ranges; do
    docker exec sglang_hicache bash -c "cd /sgl-workspace/sglang && $R format --quiet --range=$r $f" || echo "ruff failed on $f $r"
  done
  echo "$f: $(echo $ranges | wc -w) ranges formatted"
done
