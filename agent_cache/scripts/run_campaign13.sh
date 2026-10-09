#!/usr/bin/env bash
# Campaign 13: x70 steady state. Campaign 12's x70 workload (same trace window, lengths, Poisson arrivals at 0.2/s, live cap
# 128) with arrivals that never stop: above capacity the live population holds at C, so every arm runs saturated at 128
# users. Each arm stops when STOP_AFTER_USERS users have finished (3 population turnovers: 1 warm-up + 2 measured), with
# GUARD seconds as the wall cap (results/campaign13/DECISIONS.md).
#   host: docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign13 && ENGINE_PYTHONPATH=/sgl-workspace/sglang/agent_cache/frozen/engine_2abe005bb2/python nohup bash run_campaign13.sh ../results/campaign13 >> ../results/campaign13/driver.log 2>&1'
# Resumable: it runs only the arms with no line in <campaign dir>/manifest.txt. Finished arms are merged from the compare
# dirs that this campaign's driver logs name ("compare dir: ..."), each compare dir once (ledger .merged), so after a host
# reboot (sudo docker start sglang_hicache, then the same command) finished arms are kept and the interrupted arm restarts
# from scratch. To redo an arm, delete its line from manifest.txt and relaunch.
# Stop: pkill -f '^bash run_campaign13.sh' first, then pkill -f '^bash run_compare.sh'; pkill -f replay_agentic.py;
# then bash stop_server.sh. Do not edit agent_cache/scripts/ while it runs: every arm re-reads them; python/ is safe when
# ENGINE_PYTHONPATH points at a frozen copy.
# Knobs (env): ARMS="hbm_host three_tier_to three_tier_wc hbm_lru"  ARRIVAL=0.2  C=128  STOP_AFTER_USERS=384  GUARD=43200
#              TURNS_RANGE=30:50  TURNS_SEED=120  ENGINE_PYTHONPATH=<frozen python/ dir> (optional, passed to start_server.sh)
set -uo pipefail
CDIR=${1:?usage: run_campaign13.sh <campaign dir>}
mkdir -p "$CDIR"; CDIR=$(cd "$CDIR" && pwd)
AC=/sgl-workspace/sglang/agent_cache
cd "$AC/scripts" || { echo "STOP: $AC/scripts missing (run inside the container)"; exit 1; }
ALL_ARMS=${ARMS:-"hbm_host three_tier_to three_tier_wc hbm_lru"}
RATE=${ARRIVAL:-0.2}; CAP=${C:-128}; STOP_N=${STOP_AFTER_USERS:-384}; GUARD=${GUARD:-43200}
TR=${TURNS_RANGE:-30:50}; TS=${TURNS_SEED:-120}
RX='(^|/)bash ([^ ]*/)?(run_compare|start_client|start_server)\.sh( |$)|replay_agentic\.py'
pgrep -f "$RX" >/dev/null && { echo "STOP: a driver, client or server boot is already running: $(pgrep -af "$RX" | head -3)"; exit 1; }
MAN="$CDIR/manifest.txt"; touch "$MAN"

sync_manifest() {   # merge finished arms from this campaign's compare dirs into manifest.txt, each compare dir once
  local cmp line arm
  touch "$CDIR/.merged"
  for cmp in $(grep -ah '^compare dir: ' "$CDIR"/driver_*.log 2>/dev/null | cut -d' ' -f3 | sort -u); do
    grep -qxF "$cmp" "$CDIR/.merged" && continue
    [ -f "$cmp/manifest.txt" ] || continue
    while read -r line; do
      arm=${line%% *}
      [ -n "$arm" ] && ! grep -q "^$arm " "$MAN" && { echo "$line" >> "$MAN"; echo "  merged $arm from $(basename "$cmp")"; }
    done < "$cmp/manifest.txt"
    echo "$cmp" >> "$CDIR/.merged"
  done
  { for A in $ALL_ARMS; do grep "^$A " "$MAN"; done; } > "$MAN.tmp"; mv "$MAN.tmp" "$MAN"   # keep ALL_ARMS order
}

IOS=""
trap '[ -n "$IOS" ] && kill "$IOS" 2>/dev/null' EXIT
echo "LAUNCH $(date -u +%FT%TZ) commit $(git -c safe.directory="*" -C "$AC" rev-parse --short HEAD) arms: $ALL_ARMS arrival $RATE/s cap $CAP stop after $STOP_N users guard ${GUARD}s engine ${ENGINE_PYTHONPATH:-working tree}"
git -c safe.directory="*" -C "$AC" status --short -- scripts > "$CDIR/scripts_at_launch_$(date -u +%m%d_%H%M%S).txt"
sync_manifest
TODO=""
for A in $ALL_ARMS; do grep -q "^$A " "$MAN" || TODO="$TODO $A"; done
TODO=${TODO# }
if [ -z "$TODO" ]; then echo "CAMPAIGN DONE (nothing left) $(date -u +%FT%TZ)"; cat "$MAN"; exit 0; fi
echo "START arms: $TODO $(date -u +%FT%TZ)"
LOG="$CDIR/driver_$(date -u +%m%d_%H%M%S).log"
nohup iostat -dxt vdc 10 >> "$CDIR/iostat_vdc.log" 2>&1 < /dev/null & IOS=$!
# Campaign 12's x70 settings (results/campaign12/DECISIONS.md) plus: arrivals until the guard, the stop rule, the same guard.
ARMS="$TODO" LEVEL=NAT160 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= \
  TRACE=$AC/traces/lmcache_agentic_trace_262k.json L3_CLEANER_PCT=70,60 NCONV=128 C="$CAP" TURNS=50 GAP=70 \
  GAP_CAP=1200 OFFSET=32 SEED=0 K=0 CHECK_IDS=0 CLIENT_TIMEOUT="$GUARD" SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
  ARRIVAL="$RATE" ARRIVAL_UNTIL="$GUARD" STOP_AFTER_USERS="$STOP_N" TURNS_RANGE="$TR" TURNS_SEED="$TS" \
  ENGINE_PYTHONPATH="${ENGINE_PYTHONPATH:-}" bash run_compare.sh > "$LOG" 2>&1
rc=$?
kill "$IOS" 2>/dev/null; IOS=""
if [ "$rc" -ge 128 ]; then
  echo "STOP: run_compare.sh was killed (rc=$rc); stop the client (pkill -f replay_agentic.py) and the server (bash stop_server.sh), then relaunch to resume"
  exit 1
fi
sync_manifest
echo "END $(date -u +%FT%TZ): $(grep -ac 'STAGE DONE' "$LOG") done, $(grep -ac 'STAGE FAILED' "$LOG") failed ($LOG)"
M=""; for A in $ALL_ARMS; do grep -q "^$A " "$MAN" || M="$M $A"; done
cat "$MAN"
if [ -n "$M" ]; then echo "CAMPAIGN INCOMPLETE $(date -u +%FT%TZ): missing$M; relaunch to retry"; exit 1; fi
echo "CAMPAIGN DONE $(date -u +%FT%TZ)"
