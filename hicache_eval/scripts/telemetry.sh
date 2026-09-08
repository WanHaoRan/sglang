#!/usr/bin/env bash
# telemetry.sh start|stop <dir>   — per-block iostat + PCIe capture.
# The Exp 1 collector was started with & and never stopped, which contaminated
# its whole log (D16.1). This version records PIDs and stops with the block.
source /sgl-workspace/sglang/hicache_eval/scripts/env.sh
case "$1" in
  start)
    D="$2"; mkdir -p "$D"
    iostat -x -d -t 5 "$NVME_DEV" > "$D/iostat.log" 2>/dev/null &
    echo $! > "$D/.iostat.pid"
    nvidia-smi dmon -s t -d 5 > "$D/pcie.log" 2>/dev/null &
    echo $! > "$D/.dmon.pid"
    echo "telemetry started -> $D"
    ;;
  stop)
    D="$2"
    for f in "$D/.iostat.pid" "$D/.dmon.pid"; do
      [ -f "$f" ] && { kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; }
    done
    echo "telemetry stopped for $D"
    ;;
esac
