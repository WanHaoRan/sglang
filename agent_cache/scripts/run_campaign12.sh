#!/usr/bin/env bash
# Campaign 12: campaigns 7/8/9 (x70 / x10 / x1) rerun with Poisson arrivals and varied conversation lengths.
# Same trace, conversations, pools and arms as campaigns 7-9; only the arrival and ending patterns differ
# (results/campaign12/DECISIONS.md).
#   host: docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign12 && ARRIVAL_X70=0.2 ARRIVAL_X10=0.2 ARRIVAL_X1=0.2 nohup bash run_campaign12.sh ../results/campaign12 >> ../results/campaign12/driver.log 2>&1'
# Resumable: per scale it runs only the arms with no line in <campaign dir>/manifest_x<scale>.txt. Finished arms are merged
# from the compare dirs that this campaign's driver logs name ("compare dir: ..."), each compare dir once (ledger
# .merged_x<scale>), so after a host reboot (sudo docker start sglang_hicache, then the same command) finished arms are
# kept and the interrupted arm restarts from scratch. To redo an arm, delete its line from manifest_x<scale>.txt and relaunch.
# Stop: pkill -f '^bash run_campaign12.sh' first, then pkill -f '^bash run_compare.sh'; pkill -f replay_agentic.py;
# then bash stop_server.sh. Do not edit agent_cache/scripts/ or python/ while it runs: every arm re-reads them.
# Outputs: per-scale manifests are the inputs for timeline.py / gapclass.py; give each scale its own --out
# (e.g. --manifest ../results/campaign12/manifest_x70.txt --out ../results/campaign12/x70), else the outputs overwrite.
# Knobs (env): SCALES="70 10 1"  ARMS="hbm_host three_tier_to three_tier_wc hbm_lru"
#              ARRIVAL_X70 / ARRIVAL_X10 / ARRIVAL_X1 (conversations/s, required)  TURNS_RANGE=30:50  TURNS_SEED=120
#              ENGINE_PYTHONPATH=<frozen python/ dir> (optional, passed to start_server.sh)
set -uo pipefail
CDIR=${1:?usage: run_campaign12.sh <campaign dir>}
mkdir -p "$CDIR"; CDIR=$(cd "$CDIR" && pwd)
AC=/sgl-workspace/sglang/agent_cache
cd "$AC/scripts" || { echo "STOP: $AC/scripts missing (run inside the container)"; exit 1; }
SCALES=${SCALES:-"70 10 1"}
ALL_ARMS=${ARMS:-"hbm_host three_tier_to three_tier_wc hbm_lru"}
TR=${TURNS_RANGE:-30:50}; TS=${TURNS_SEED:-120}
RX='(^|/)bash ([^ ]*/)?(run_compare|start_client|start_server)\.sh( |$)|replay_agentic\.py'
pgrep -f "$RX" >/dev/null && { echo "STOP: a driver, client or server boot is already running: $(pgrep -af "$RX" | head -3)"; exit 1; }

rate_of() {   # canonical rate string for scale $1 ('1.0', '1' -> 1; '.5' -> 0.5)
  local r
  case "$1" in 70) r=${ARRIVAL_X70:-} ;; 10) r=${ARRIVAL_X10:-} ;; 1) r=${ARRIVAL_X1:-} ;; *) r= ;; esac
  awk -v r="$r" 'BEGIN { if (r + 0 > 0) printf "%g", r }'
}
cap_of() {    # per-arm wall cap: campaign 7's x70 arms were in effect uncapped (longest 11,429 s); x10/x1 keep
  # campaigns 8/9's 9,000 s per conversation, plus the seed-0 arrival window (147.11 / rate s) for the last arrival
  case "$1" in 70) echo 12600 ;; *) awk -v r="$2" 'BEGIN { printf "%d", 9000 + 147.11 / r + 1 }' ;; esac
}
for S in $SCALES; do [ -n "$(rate_of "$S")" ] || { echo "STOP: ARRIVAL_X$S must be a rate > 0"; exit 1; }; done

sync_manifest() {   # merge finished arms from this campaign's compare dirs for scale $1 into $2, each compare dir once
  local cmp line arm
  touch "$CDIR/.merged_x$1"
  for cmp in $(grep -ah '^compare dir: ' "$CDIR"/driver_x"$1"_*.log 2>/dev/null | cut -d' ' -f3 | sort -u); do
    grep -qxF "$cmp" "$CDIR/.merged_x$1" && continue
    [ -f "$cmp/manifest.txt" ] || continue
    while read -r line; do
      arm=${line%% *}
      [ -n "$arm" ] && ! grep -q "^$arm " "$2" && { echo "$line" >> "$2"; echo "  merged $arm from $(basename "$cmp")"; }
    done < "$cmp/manifest.txt"
    echo "$cmp" >> "$CDIR/.merged_x$1"
  done
  { for A in $ALL_ARMS; do grep "^$A " "$2"; done; } > "$2.tmp"; mv "$2.tmp" "$2"   # keep ALL_ARMS order (gapclass pairs vs line 1)
}

IOS=""
trap '[ -n "$IOS" ] && kill "$IOS" 2>/dev/null' EXIT
echo "LAUNCH $(date -u +%FT%TZ) commit $(git -c safe.directory="*" -C "$AC" rev-parse --short HEAD) scales: $SCALES arms: $ALL_ARMS turns $TR seed $TS"
git -c safe.directory="*" -C "$AC" status --short -- scripts > "$CDIR/scripts_at_launch_$(date -u +%m%d_%H%M%S).txt"
for S in $SCALES; do
  RATE=$(rate_of "$S"); CT=$(cap_of "$S" "$RATE")
  MAN="$CDIR/manifest_x$S.txt"; touch "$MAN"
  sync_manifest "$S" "$MAN"
  TODO=""
  for A in $ALL_ARMS; do grep -q "^$A " "$MAN" || TODO="$TODO $A"; done
  TODO=${TODO# }
  if [ -z "$TODO" ]; then echo "SCALE x$S complete: $(wc -l < "$MAN") arms"; continue; fi
  echo "SCALE START x$S arms: $TODO  arrival $RATE/s  cap ${CT}s  $(date -u +%FT%TZ)"
  LOG="$CDIR/driver_x${S}_$(date -u +%m%d_%H%M%S).log"
  nohup iostat -dxt vdc 10 >> "$CDIR/iostat_vdc_x$S.log" 2>&1 < /dev/null & IOS=$!
  # Campaigns 7-9 settings (config.txt of compare_20260922_023714 / _165655 / compare_20260923_031116) plus campaign 11's
  # keep-alive; new: ARRIVAL, TURNS_RANGE/TURNS_SEED (TURNS=50 is the trace maximum, so only the range caps lengths) and
  # the per-scale wall cap above.
  ARMS="$TODO" LEVEL=NAT160 MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= \
    TRACE=$AC/traces/lmcache_agentic_trace_262k.json L3_CLEANER_PCT=70,60 NCONV=128 C=128 TURNS=50 GAP="$S" \
    GAP_CAP=1200 OFFSET=32 SEED=0 K=0 CHECK_IDS=0 CLIENT_TIMEOUT="$CT" SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
    ARRIVAL="$RATE" TURNS_RANGE="$TR" TURNS_SEED="$TS" ENGINE_PYTHONPATH="${ENGINE_PYTHONPATH:-}" \
    bash run_compare.sh > "$LOG" 2>&1
  rc=$?
  kill "$IOS" 2>/dev/null; IOS=""
  if [ "$rc" -ge 128 ]; then
    echo "STOP: run_compare.sh was killed (rc=$rc); stop the client (pkill -f replay_agentic.py) and the server (bash stop_server.sh), then relaunch to resume"
    exit 1
  fi
  sync_manifest "$S" "$MAN"
  echo "SCALE END x$S $(date -u +%FT%TZ): $(grep -ac 'STAGE DONE' "$LOG") done, $(grep -ac 'STAGE FAILED' "$LOG") failed ($LOG)"
done
MISSING=""
for S in $SCALES; do
  M=""; for A in $ALL_ARMS; do grep -q "^$A " "$CDIR/manifest_x$S.txt" || M="$M $A"; done
  [ -n "$M" ] && MISSING="$MISSING x$S:$M;"
  echo "x$S:"; cat "$CDIR/manifest_x$S.txt"
done
if [ -n "$MISSING" ]; then echo "CAMPAIGN INCOMPLETE $(date -u +%FT%TZ):$MISSING relaunch to retry"; exit 1; fi
echo "CAMPAIGN DONE $(date -u +%FT%TZ)"
