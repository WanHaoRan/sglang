#!/usr/bin/env bash
# Stop any running launch_server and wait for the GPU and port to clear.
set -uo pipefail
pkill -f "sglang.launch_server" 2>/dev/null || true
for i in $(seq 1 90); do
  if ! pgrep -f "sglang.launch_server" >/dev/null 2>&1; then break; fi
  sleep 1
done
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 2
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$used" -eq 0 ] && break
  sleep 1
done
echo "stopped (gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l))"
