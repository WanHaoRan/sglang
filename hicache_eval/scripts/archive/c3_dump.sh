#!/usr/bin/env bash
# C3 - GIL attribution by repeated py-spy dump on the scheduler under write load.
# `py-spy record -s` hangs indefinitely here: -s walks the launcher's children and
# the scheduler has ~96 GB RSS. `dump` is O(stack) and returns in <1 s, and it
# already annotates threads as "active+gil" / "active" / "idle", which is exactly
# what C3 asks for.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts
KEY="${1:?usage: c3_dump.sh <model key>}"; RATE="${2:-8}"
source "$S/models.sh" "$KEY" >/dev/null; source "$S/env.sh"
OUT="$RESULTS/exp2_${KEY}/c3"; mkdir -p "$OUT"
SCHED=$(pgrep -f "sglang::scheduler" | head -1)
[ -z "$SCHED" ] && { echo "no scheduler process"; exit 1; }
echo "scheduler=$SCHED rate=$RATE" | tee "$OUT/c3_meta.txt"

python3 "$S/writeload.py" --len 4096 --out 1 --rate "$RATE" --duration 100 \
  > "$OUT/writeload.json" 2>/dev/null &
W=$!
sleep 25
: > "$OUT/dumps_under_load.txt"
for i in $(seq 1 12); do
  echo "===== sample $i =====" >> "$OUT/dumps_under_load.txt"
  timeout 30 py-spy dump --pid "$SCHED" --nonblocking >> "$OUT/dumps_under_load.txt" 2>&1
  sleep 2
done
kill -TERM $W 2>/dev/null; wait $W 2>/dev/null

echo "--- GIL holder census across 12 samples ---" | tee "$OUT/c3_summary.txt"
grep -E "^Thread .*\(active\+gil\)" "$OUT/dumps_under_load.txt" \
  | sed -E 's/.*: "(.*)"/\1/' | sort | uniq -c | sort -rn | tee -a "$OUT/c3_summary.txt"
echo "--- top frame directly under each active+gil thread ---" | tee -a "$OUT/c3_summary.txt"
grep -A1 -E "^Thread .*\(active\+gil\)" "$OUT/dumps_under_load.txt" \
  | grep -vE "^Thread|^--" | sed 's/^ *//' | sort | uniq -c | sort -rn | head -12 \
  | tee -a "$OUT/c3_summary.txt"
echo "--- all thread states seen ---" | tee -a "$OUT/c3_summary.txt"
grep -oE "\((active\+gil|active|idle)\)" "$OUT/dumps_under_load.txt" | sort | uniq -c | tee -a "$OUT/c3_summary.txt"
