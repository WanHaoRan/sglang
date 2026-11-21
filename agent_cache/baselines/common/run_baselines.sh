#!/usr/bin/env bash
# One launcher for a comparison across systems: our SGLang arms and the baselines, on the same trace and schedule.
# Host side. For each arm it boots the server (SGLang in the sglang_hicache container, a baseline on the host), runs
# the trace replay from the container, stops the server and records the arm as run_campaigns.sh of campaigns 28 and 29
# does, so compare_arms.py and the checks read the folder unchanged.
#
#   host: setsid nohup bash /home/wanhr/sglang/agent_cache/baselines/common/run_baselines.sh <folder> [<folder> ...] \
#           >> /home/wanhr/sglang/agent_cache/results/<folder>/runner.log 2>&1 < /dev/null &
#
# A campaign folder under agent_cache/results/ holds
#   settings.env   TRACE RATE CAP HOST_GB DURATION MAX_CONTEXT GAP GAP_CAP SEED, the same for every arm (TRACE is the
#                  path the container sees, /sgl-workspace/sglang/agent_cache/...)
#   queue.txt      one arm per line: <label> <system> <hints> <client flags joined by commas, or -> [arguments ...]
#                    system "sglang"  one of our arms; arguments: <server arm> <engine folder in frozen/> [server flags]
#                    any other system a folder of agent_cache/baselines/ with a start_server.sh; arguments: KEY=VALUE
#                                     knobs of that script; <hints> is "-" (a baseline takes its signals from its
#                                     backend, baselines/<system>/client.env)
#   manifest.txt   written here, one finished arm per line: <label> <server run dir> <programs dir>
#   launcher.log   written here: <label> LAUNCH | DONE | STOP lines
#   inputs.sha256  written at the first launch: the trace and the scripts every arm must run on
# Unfinished queue lines run top to bottom; the queue is re-read after every arm. An arm in the manifest is skipped, so
# after a host outage the same command resumes with the interrupted arm from its start. With nothing left it exits.
# An arm is DONE when its client wrote a summary with errors on at most 1 % of its requests; otherwise the launcher
# stops there. Every arm boots on wiped SSD stores (/mnt/ssd/hicache_l3 and /mnt/ssd/baselines).
# Stop: pkill -f "[r]un_baselines.sh"; docker exec sglang_hicache pkill -f "[r]eplay_"; then stop the server
#       (bash baselines/common/stop_server.sh; docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh)
set -uo pipefail
AC=/home/wanhr/sglang/agent_cache
CAC=/sgl-workspace/sglang/agent_cache
BL=$AC/baselines
BOX=sglang_hicache
NEEDS="TRACE RATE CAP HOST_GB DURATION MAX_CONTEXT GAP GAP_CAP SEED"
PINNED="scripts/start_server.sh scripts/stop_server.sh scripts/start_programs.sh scripts/replay_programs.py scripts/program_trace.py scripts/gap_hints.py scripts/program_summary.py baselines/common/replay_baseline.py baselines/common/backends.py baselines/common/start_client.sh baselines/common/lib.sh"
SNAP_GLOB=/home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots
[ $# -ge 1 ] || { echo "STOP: name the campaign folders under $AC/results"; exit 1; }

PIDF=$AC/results/.run_baselines.pid; old=$(cat "$PIDF" 2>/dev/null)
if [ -n "$old" ] && [ "$old" != "$$" ] && grep -qa run_baselines.sh "/proc/$old/cmdline" 2>/dev/null; then
  echo "STOP: run_baselines.sh is already running as pid $old"; exit 1
fi
echo $$ > "$PIDF"
docker ps --format '{{.Names}}' | grep -qx "$BOX" || docker start "$BOX" > /dev/null \
  || { echo "STOP: container $BOX is not running and did not start"; exit 1; }
if docker exec "$BOX" pgrep -f 'python[0-9.]* [r]eplay_(programs|agentic|baseline)\.py' > /dev/null \
   || docker exec "$BOX" pgrep -f '[r]un_campaigns.sh' > /dev/null; then
  echo "STOP: a replay client or run_campaigns.sh is still running in $BOX and would share port 30000"; exit 1
fi

host_path() { echo "${1/#\/sgl-workspace\/sglang/\/home\/wanhr\/sglang}"; }
say() { echo "$2" | tee -a "$AC/results/$1/launcher.log"; }   # campaign folder, line

# Checks before the first boot: a bad setting, trace or queue line would otherwise show hours in.
for C in "$@"; do
  CD=$AC/results/$C
  [ -s "$CD/settings.env" ] && [ -f "$CD/queue.txt" ] || { echo "STOP: $CD needs settings.env and queue.txt"; exit 1; }
  (
    set -a; . "$CD/settings.env"; set +a
    for v in $NEEDS; do [ -n "${!v:-}" ] || { echo "STOP: $CD/settings.env sets no $v"; exit 1; }; done
    T=$(host_path "$TRACE")
    [ -s "$T" ] && gzip -t "$T" 2>/dev/null || { echo "STOP: $C: trace $T is missing or not a complete gzip"; exit 1; }
    [ -f "$CD/inputs.sha256" ] || ( cd "$AC" && sha256sum "$T" $PINNED > "$CD/inputs.sha256" )
  ) || exit 1
  while read -r L S H CF rest || [ -n "${L:-}" ]; do
    case "${L:-}" in ''|'#'*) continue ;; esac
    [ -n "${CF:-}" ] || { echo "STOP: $CD/queue.txt, arm $L: needs <label> <system> <hints> <client flags or ->"; exit 1; }
    if [ "$S" = sglang ]; then
      read -r A E _ <<< "$rest"
      [ -n "${E:-}" ] && [ -d "$AC/frozen/$E/python" ] \
        || { echo "STOP: $CD/queue.txt, arm $L: sglang needs <server arm> <engine folder in frozen/> (got '${E:-}')"; exit 1; }
    else
      [ -x "$BL/$S/start_server.sh" ] || { echo "STOP: $CD/queue.txt, arm $L: no $BL/$S/start_server.sh"; exit 1; }
      case "$S" in sglang_*) ;; *) [ -s "$BL/$S/client.env" ] || { echo "STOP: arm $L: no $BL/$S/client.env"; exit 1; } ;; esac
    fi
  done < "$CD/queue.txt"
  touch "$CD/manifest.txt"
done

stop_servers() {
  bash "$BL/common/stop_server.sh"
  docker exec "$BOX" bash "$CAC/scripts/stop_server.sh"
}

run_arm() {  # campaign folder, label, system, hints, client flags, then the system's arguments
  local C=$1 L=$2 S=$3 H=$4 CF=$5; shift 5
  local CD=$AC/results/$C LOG RUN OUT SUM ERR REQ
  [ "$CF" = - ] && CF=
  ( cd "$AC" && sha256sum --quiet -c "$CD/inputs.sha256" ) > /dev/null 2>&1 \
    || { say "$C" "$L STOP $(date -u +%FT%TZ): the trace or a script changed since this folder's first launch (cd $AC && sha256sum -c $CD/inputs.sha256)"; return 1; }
  LOG="$CD/driver_${L}_$(date -u +%m%d_%H%M%S).log"
  say "$C" "$L LAUNCH $(date -u +%FT%TZ) system $S hints $H client [${CF//,/ }] args [$*] settings [$(tr '\n' ' ' < "$CD/settings.env")] -> $LOG"
  (
    # A subshell, so one arm's settings never reach the next.
    set -a; . "$CD/settings.env"; set +a
    stop_servers
    : > "$AC/results/.current_run"
    docker exec "$BOX" bash -c 'mkdir -p /mnt/ssd/baselines && find /mnt/ssd/hicache_l3 /mnt/ssd/baselines -mindepth 1 -delete; chown 1001:1001 /mnt/ssd/baselines; echo "SSD stores wiped: $(find /mnt/ssd/hicache_l3 /mnt/ssd/baselines -type f | wc -l) files left"'
    # the page cache loses the weights during a run with a large host tier; a boot would then crawl
    cat "$SNAP_GLOB"/*/model-*.safetensors > /dev/null
    if [ "$S" = sglang ]; then
      read -r A E X <<< "$*"
      docker exec -e HOST_GB="$HOST_GB" -e MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 -e CTX=262144 -e KV_DTYPE=auto \
        -e TEMPLATE= -e L3_CLEANER_PCT=90,80 -e SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
        -e ENGINE_PYTHONPATH="$CAC/frozen/$E/python" -e SERVER_EXTRA="$X" \
        "$BOX" bash "$CAC/scripts/start_server.sh" "$A" NAT160 || { echo "STAGE FAILED $L (boot)"; stop_servers; exit 3; }
    else
      env HOST_GB="$HOST_GB" "$@" bash "$BL/$S/start_server.sh" || { echo "STAGE FAILED $L (boot)"; stop_servers; exit 3; }
    fi
    case "$S" in
      sglang|sglang_*)
        # shellcheck disable=SC2086
        docker exec -e TRACE="$TRACE" -e RECIPE=none -e RATE="$RATE" -e MAX_ACTIVE="$CAP" -e DURATION="$DURATION" \
          -e HINTS="${H/#-/none}" -e MAX_CONTEXT="$MAX_CONTEXT" -e SEED="$SEED" -e TAG="$L" \
          "$BOX" bash "$CAC/scripts/start_programs.sh" --gap-scale "$GAP" --gap-cap "$GAP_CAP" ${CF//,/ } ;;
      *)
        set -a; . "$BL/$S/client.env"; set +a
        # shellcheck disable=SC2086
        docker exec -e BACKEND="$BACKEND" -e BACKEND_OPTS="${BACKEND_OPTS:-}" -e TRACE="$TRACE" -e RECIPE=none \
          -e RATE="$RATE" -e MAX_ACTIVE="$CAP" -e DURATION="$DURATION" -e MAX_CONTEXT="$MAX_CONTEXT" -e SEED="$SEED" \
          -e TAG="$L" "$BOX" bash "$CAC/baselines/common/start_client.sh" --gap-scale "$GAP" --gap-cap "$GAP_CAP" ${CF//,/ } ;;
    esac
    echo "client rc=$?"
    echo "ssd used: $(df --output=pcent /mnt/ssd | tail -1 | tr -d ' '); baseline store: $(du -sh /mnt/ssd/baselines 2>/dev/null | cut -f1)"
    [ -x "$BL/$S/after_run.sh" ] && bash "$BL/$S/after_run.sh" "$AC/results/$(cat "$AC/results/.current_run")"
    stop_servers
  ) > "$LOG" 2>&1
  RUN=$(cat "$AC/results/.current_run" 2>/dev/null); OUT=
  [ -n "$RUN" ] && OUT=$(ls -d "$AC/results/$RUN"/programs_*_"$L" 2>/dev/null | tail -1)
  if [ -n "$OUT" ] && grep -q '"kind":"summary"' "$OUT/client.jsonl" 2>/dev/null; then
    SUM=$(grep -a '"kind":"summary"' "$OUT/client.jsonl" | tail -1)
    ERR=$(grep -o '"errors":[0-9]*' <<< "$SUM" | cut -d: -f2); REQ=$(grep -o '"profile":[0-9]*' <<< "$SUM" | cut -d: -f2)
    if [ -z "$ERR" ] || [ -z "$REQ" ] || [ $((ERR * 100)) -gt "$REQ" ]; then
      say "$C" "$L STOP $(date -u +%FT%TZ), ${ERR:-?} client errors in ${REQ:-?} requests: read $LOG"
      return 1
    fi
    echo "$L $RUN $(basename "$OUT")" >> "$CD/manifest.txt"
    say "$C" "$L DONE $(date -u +%FT%TZ) requests $REQ errors $ERR, $(grep -a '^ssd used' "$LOG" | tail -1), run $RUN/$(basename "$OUT")"
  else
    say "$C" "$L STOP $(date -u +%FT%TZ), no summary line: read $LOG"
    return 1
  fi
}

while :; do
  next=""; where=""
  for C in "$@"; do
    CD=$AC/results/$C
    while read -r L rest || [ -n "${L:-}" ]; do
      case "${L:-}" in ''|'#'*) continue ;; esac
      grep -q "^$L " "$CD/manifest.txt" || { next="$L $rest"; where=$C; break; }
    done < "$CD/queue.txt"
    [ -n "$next" ] && break
  done
  [ -n "$next" ] || { for C in "$@"; do say "$C" "ALL DONE $(date -u +%FT%TZ)"; done; exit 0; }
  # shellcheck disable=SC2086
  run_arm "$where" $next || exit 1
done
