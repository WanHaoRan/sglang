#!/usr/bin/env bash
# Campaign 14: campaign 13's x70 steady state with the client changes against its light phases
# (results/campaign14/DECISIONS.md): the live-user cap follows the server backlog (CAP_MODE=backlog, target CAP_TARGET
# requests awaiting their first token), each user replays a different one of the trace's 768 conversations in a seeded
# order (NCONV=768 ORDER=fresh: no conversation twice, no user after the 768th) and the first C users start mid-session
# (START_PHASE=stationary). Arrivals at 0.1/s until the 768th user; each arm runs RUN_SECONDS of
# wall time (the client's wall cap), so the measured window has a fixed length.
#   host: docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign14 && ENGINE_PYTHONPATH=/sgl-workspace/sglang/agent_cache/frozen/engine_2abe005bb2/python nohup bash run_campaign14.sh ../results/campaign14 >> ../results/campaign14/driver.log 2>&1'
# Resumable: it runs only the arms with no line in <campaign dir>/manifest.txt. Finished arms are merged from the compare
# dirs that this campaign's driver logs name ("compare dir: ..."), each compare dir once (ledger .merged), so after a host
# reboot (sudo docker start sglang_hicache, then the same command) finished arms are kept and the interrupted arm restarts
# from scratch. To redo an arm, delete its line from manifest.txt and relaunch.
# Stop: pkill -f '^bash run_campaign14.sh' first, then pkill -f '^bash run_compare.sh'; pkill -f replay_agentic.py;
# then bash stop_server.sh. Do not edit agent_cache/scripts/ while it runs: every arm re-reads them; python/ is safe when
# ENGINE_PYTHONPATH points at a frozen copy.
# Knobs (env): ARMS="three_tier_wc three_tier_to"  ARRIVAL=0.1  C=128 (the starting cap)  RUN_SECONDS=18000  TURNS_RANGE=30:50
#              TURNS_SEED=120  CAP_TARGET=32  CAP_INTERVAL=120  CAP_GAIN=0.5  CAP_STEP=8  CAP_MIN=16  CAP_MAX=512
#              ADMIT_SPACING=5  ORDER_SEED=0  START_SEED=0  START_SERVICE=20
#              ENGINE_PYTHONPATH=<frozen python/ dir> (optional, passed to start_server.sh)
set -uo pipefail
CDIR=${1:?usage: run_campaign14.sh <campaign dir>}
mkdir -p "$CDIR"; CDIR=$(cd "$CDIR" && pwd)
AC=/sgl-workspace/sglang/agent_cache
cd "$AC/scripts" || { echo "STOP: $AC/scripts missing (run inside the container)"; exit 1; }
ALL_ARMS=${ARMS:-"three_tier_wc three_tier_to"}
RATE=${ARRIVAL:-0.1}; CAP=${C:-128}; RUN_S=${RUN_SECONDS:-18000}
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
echo "LAUNCH $(date -u +%FT%TZ) commit $(git -c safe.directory="*" -C "$AC" rev-parse --short HEAD) arms: $ALL_ARMS arrival $RATE/s cap from $CAP by backlog (target ${CAP_TARGET:-32}) run ${RUN_S}s engine ${ENGINE_PYTHONPATH:-working tree}"
git -c safe.directory="*" -C "$AC" status --short -- scripts > "$CDIR/scripts_at_launch_$(date -u +%m%d_%H%M%S).txt"
sync_manifest
TODO=""
for A in $ALL_ARMS; do grep -q "^$A " "$MAN" || TODO="$TODO $A"; done
TODO=${TODO# }
if [ -z "$TODO" ]; then echo "CAMPAIGN DONE (nothing left) $(date -u +%FT%TZ)"; cat "$MAN"; exit 0; fi
echo "START arms: $TODO $(date -u +%FT%TZ)"
LOG="$CDIR/driver_$(date -u +%m%d_%H%M%S).log"
nohup iostat -dxt vdc 10 >> "$CDIR/iostat_vdc.log" 2>&1 < /dev/null & IOS=$!
# Campaign 13's settings (results/campaign13/DECISIONS.md) on all 768 conversations, with a time stop and the client changes.
ARMS="$TODO" LEVEL=NAT160 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= \
  TRACE=$AC/traces/lmcache_agentic_trace_262k.json L3_CLEANER_PCT=70,60 NCONV=768 C="$CAP" TURNS=50 GAP=70 \
  GAP_CAP=1200 OFFSET=0 SEED=0 K=0 CHECK_IDS=0 CLIENT_TIMEOUT="$RUN_S" SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
  ARRIVAL="$RATE" ARRIVAL_UNTIL="$RUN_S" STOP_AFTER_USERS=0 TURNS_RANGE="$TR" TURNS_SEED="$TS" \
  CAP_MODE=backlog CAP_TARGET="${CAP_TARGET:-32}" CAP_INTERVAL="${CAP_INTERVAL:-120}" CAP_GAIN="${CAP_GAIN:-0.5}" \
  CAP_STEP="${CAP_STEP:-8}" CAP_MIN="${CAP_MIN:-16}" CAP_MAX="${CAP_MAX:-512}" ADMIT_SPACING="${ADMIT_SPACING:-5}" \
  ORDER=fresh ORDER_SEED="${ORDER_SEED:-0}" START_PHASE=stationary START_SEED="${START_SEED:-0}" \
  START_SERVICE="${START_SERVICE:-20}" ENGINE_PYTHONPATH="${ENGINE_PYTHONPATH:-}" bash run_compare.sh > "$LOG" 2>&1
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
