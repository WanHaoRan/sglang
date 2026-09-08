#!/usr/bin/env bash
# Exp 2 control C3 - GIL attribution under write load.
#   bash c3_pyspy.sh <model key>   (a server for that model must be running)
#
# Replaces the first attempt, which failed twice: `py-spy top --duration` is not
# a flag in py-spy 0.4.2, and the pid came from `pgrep -f sglang.launch_server |
# head -1`, which is the launcher -- its threads are all idle. The scheduler is
# a subprocess, so -s/--subprocesses is what reaches it.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts
KEY="${1:?usage: c3_pyspy.sh <model key>}"
source "$S/models.sh" "$KEY" >/dev/null
source "$S/env.sh"
OUT="$RESULTS/exp2_${KEY}/c3"; mkdir -p "$OUT"
case "$KEY" in qwen8b) R=8 ;; qwen32b) R=2 ;; llama70b) R=1 ;; esac

LPID=$(pgrep -f "sglang.launch_server" | head -1)
[ -z "$LPID" ] && { echo "C3: no server running"; exit 1; }
echo "launcher pid $LPID; descendants:" | tee "$OUT/pids.txt"
ps --ppid "$LPID" -o pid,rss,args --no-headers 2>/dev/null | cut -c1-120 | tee -a "$OUT/pids.txt"

python3 "$S/writeload.py" --len 4096 --out 1 --rate "$R" --duration 90 \
  > "$OUT/writeload.json" 2>/dev/null &
WPID=$!
sleep 20   # let the load reach steady state before sampling

# All sampled traces, then only those holding the GIL. The difference between
# the two is the C3 answer: if the scheduler's time sits in the GIL-holding set
# under backup/prefetch frames, the contention is the GIL rather than the SSD.
py-spy record -p "$LPID" -s -d 25 -r 100 -t --nonblocking \
  -f raw -o "$OUT/all_traces.txt"  > "$OUT/record_all.log" 2>&1
py-spy record -p "$LPID" -s -d 25 -r 100 -t --nonblocking -g \
  -f raw -o "$OUT/gil_traces.txt" > "$OUT/record_gil.log" 2>&1
for i in 1 2 3; do
  py-spy dump -p "$LPID" -s -j --nonblocking >> "$OUT/dumps.json" 2>&1
done
kill -TERM $WPID 2>/dev/null; wait $WPID 2>/dev/null

echo "--- top 15 GIL-holding stacks ---" | tee "$OUT/summary.txt"
sort -t' ' -k2 -rn -o /dev/null /dev/null 2>/dev/null
awk '{n=$NF; $NF=""; print n"\t"$0}' "$OUT/gil_traces.txt" 2>/dev/null \
  | sort -rn | head -15 | tee -a "$OUT/summary.txt"
echo "C3 done: $OUT"
