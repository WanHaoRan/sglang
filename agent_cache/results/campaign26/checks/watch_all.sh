#!/usr/bin/env bash
# c26_ec/watch.sh with its problem checks widened to every host pool size (that one skips run folders ending in
# _H100, so the three 100 GB arms ran unchecked). From agent_cache/, on the host:  bash results/campaign26/checks/watch_all.sh
# Exits on the next arm's end in campaign 26's launcher.log or on a problem: a traceback in the running server's log,
# client errors, a client that stops writing for 300 s, or the container down.
L=results/campaign26/launcher.log
P='C26 .*(DONE|STOP|IDLE)'
n0=$(grep -cE "$P" "$L")
while :; do
  n=$(grep -cE "$P" "$L")
  if [ "$n" -gt "$n0" ]; then echo "launcher: $(grep -E "$P" "$L" | tail -1 | cut -c1-240)"; exit 0; fi
  R=$(cat results/.current_run 2>/dev/null)
  case "$R" in *_NAT160|*custom_L1nat_H*)
    if [ -f "results/$R/server.log" ]; then
      tb=$(grep -ac 'Traceback' "results/$R/server.log")
      [ "$tb" -gt 0 ] && { echo "ALERT: $tb tracebacks in results/$R/server.log"; exit 0; }
    fi
    for c in results/$R/client_*/client.jsonl; do
      [ -f "$c" ] || continue
      e=$(grep -ac '"error": "' "$c")
      [ "$e" -gt 0 ] && { echo "ALERT: $e client errors in $c"; exit 0; }
      age=$(( $(date +%s) - $(stat -c %Y "$c") ))
      [ "$age" -gt 300 ] && { echo "ALERT: $c has not grown for $age s"; exit 0; }
    done ;;
  esac
  docker ps --format '{{.Names}}' | grep -q '^sglang_hicache$' || { echo "ALERT: container down"; exit 0; }
  sleep 60
done
