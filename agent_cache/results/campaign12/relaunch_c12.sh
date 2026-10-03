#!/usr/bin/env bash
# One-off, 2026-10-01 ~21:40Z: the user relaxed "all 128 conversations overlap" to "high concurrency", so x10 and x1 move
# to the same 0.2/s as x70 (was 0.75/s and 1.0/s). The first driver was stopped while its x70 run_compare.sh kept running;
# this waits for that run_compare.sh to end, then relaunches the driver with 0.2/s everywhere. The relaunched driver merges
# x70's finished arms from driver_x70_*.log and skips them. Runs inside sglang_hicache (docker exec -d).
CD=/sgl-workspace/sglang/agent_cache/results/campaign12
cd /sgl-workspace/sglang/agent_cache/scripts || exit 1
nohup iostat -dxt vdc 10 >> "$CD/iostat_vdc_x70.log" 2>&1 < /dev/null & IOS=$!
while pgrep -f '^bash run_compare.sh' >/dev/null; do sleep 30; done
kill "$IOS" 2>/dev/null
echo "RELAUNCH $(date -u +%FT%TZ): x70 run_compare.sh ended; x10 and x1 now at 0.2/s" >> "$CD/driver.log"
ARRIVAL_X70=0.2 ARRIVAL_X10=0.2 ARRIVAL_X1=0.2 exec bash run_campaign12.sh "$CD" >> "$CD/driver.log" 2>&1
