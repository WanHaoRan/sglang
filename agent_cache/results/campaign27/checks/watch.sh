#!/usr/bin/env bash
# Campaign 27 watcher. From agent_cache/, on the host:  bash results/campaign27/checks/watch.sh
# Exits on the next arm's end in the launcher log or on a problem: a traceback in the running server's log, client
# errors, a client that stops writing for 300 s (it writes a status record every minute), the launcher or container gone.
L=results/campaign27/launcher.log
P='C27 .*(DONE|STOP|IDLE)'
n0=$(grep -cE "$P" "$L")
while :; do
  n=$(grep -cE "$P" "$L")
  if [ "$n" -gt "$n0" ]; then echo "launcher: $(grep -E "$P" "$L" | tail -1 | cut -c1-240)"; exit 0; fi
  R=$(cat results/.current_run 2>/dev/null)
  if [ -f "results/$R/server.log" ]; then
    tb=$(grep -ac 'Traceback' "results/$R/server.log")
    [ "$tb" -gt 0 ] && { echo "ALERT: $tb tracebacks in results/$R/server.log"; exit 0; }
  fi
  for c in results/$R/programs_*/client.jsonl; do
    [ -f "$c" ] || continue
    grep -q '"kind":"summary"' "$c" && continue
    e=$(grep -ac '"error":"' "$c")
    [ "$e" -gt 0 ] && { echo "ALERT: $e client errors in $c"; exit 0; }
    age=$(( $(date +%s) - $(stat -c %Y "$c") ))
    [ "$age" -gt 300 ] && { echo "ALERT: $c has not grown for $age s"; exit 0; }
  done
  docker ps --format '{{.Names}}' | grep -q '^sglang_hicache$' || { echo "ALERT: container down"; exit 0; }
  docker exec sglang_hicache pgrep -f '[c]27.sh' > /dev/null || { echo "ALERT: launcher not running"; exit 0; }
  sleep 60
done
