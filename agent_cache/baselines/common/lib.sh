# Shared by every baseline's start_server.sh and by stop_server.sh; source it, do not run it. Host side.
# The baseline servers run on the host from their own uv environments (each upstream's install path); our SGLang
# servers keep running in the sglang_hicache container. Both write under agent_cache/results/, one directory per boot,
# and results/.current_run names the latest boot whichever side made it.

BL_AC=/home/wanhr/sglang/agent_cache
BL_RESULTS=$BL_AC/results
BL_PORT=${PORT:-30000}
BL_MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
# Qwen3-30B-A3B in bf16: 48 layers x 2 x 4 KV heads x 128 x 2 bytes.
BL_KV_BYTES_PER_TOKEN=98304
# The GPU KV pool our SGLang arms profile on this H200 (campaigns 28 and 29); every system gets the same bytes.
BL_GPU_KV_TOKENS=${GPU_KV_TOKENS:-668096}

bl_port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$BL_PORT") 2>/dev/null; }

# bl_new_run NAME: make results/<UTC>_NAME, point .current_run at it, print its path. Refuses a busy port.
bl_new_run() {
  if bl_port_busy; then
    echo "STOP: something is already listening on port $BL_PORT; stop it first (bash $BL_AC/baselines/common/stop_server.sh)" >&2
    return 1
  fi
  local run="$BL_RESULTS/$(date -u +%Y%m%d_%H%M%S)_$1"
  mkdir -p "$run" && basename "$run" > "$BL_RESULTS/.current_run" && echo "$run"
}

# bl_launch RUN [VAR=VALUE ...] -- COMMAND...: start the server detached, its output in RUN/server.log, pid in
# RUN/server.pid. With MEM_CAP_GB set it runs in a systemd user scope capped at that much RAM, page cache included,
# so a disk tier cannot be served from memory the system was not given.
bl_launch() {
  local run=$1; shift
  local envv=()
  while [ "$1" != "--" ]; do envv+=("$1"); shift; done
  shift
  local wrap=()
  if [ -n "${MEM_CAP_GB:-}" ]; then
    wrap=(systemd-run --user --scope -q -p "MemoryMax=${MEM_CAP_GB}G" -p MemorySwapMax=0 --)
  fi
  printf '%q ' env "${envv[@]}" "$@" > "$run/command.txt"; echo >> "$run/command.txt"
  env "${envv[@]}" setsid nohup "${wrap[@]}" "$@" > "$run/server.log" 2>&1 < /dev/null &
  echo $! > "$run/server.pid"
}

bl_alive() { kill -0 "$1" 2>/dev/null; }

# bl_wait_ready RUN SECONDS: until /health answers and /v1/models lists a model; fails when the server dies first.
bl_wait_ready() {
  local run=$1 limit=$2 pid i
  pid=$(cat "$run/server.pid")
  for i in $(seq 1 "$limit"); do
    if curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:$BL_PORT/health" \
       && curl -sf --max-time 3 "http://127.0.0.1:$BL_PORT/v1/models" | grep -q '"id"'; then
      echo "READY after ${i}s"; return 0
    fi
    if ! bl_alive "$pid"; then
      echo "STOP: the server died during startup; last lines of $run/server.log:"; tail -25 "$run/server.log"; return 1
    fi
    sleep 1
  done
  echo "STOP: not ready after ${limit}s; last lines of $run/server.log:"; tail -25 "$run/server.log"; return 1
}

# bl_stop: stop the server of the current run when it is one of ours (a host pid), wait for the port and the GPU.
bl_stop() {
  [ -s "$BL_RESULTS/.current_run" ] || { echo "nothing to stop: no .current_run"; return 0; }
  local run="$BL_RESULTS/$(cat "$BL_RESULTS/.current_run")" pid i
  [ -f "$run/server.pid" ] && [ -f "$run/command.txt" ] || { echo "nothing to stop here: $(basename "$run") is not a host-side baseline boot"; return 0; }
  pid=$(cat "$run/server.pid")
  if bl_alive "$pid"; then
    # setsid made the server a session leader: signal its whole group, engine core and workers included
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for i in $(seq 1 90); do bl_alive "$pid" || break; sleep 1; done
    if bl_alive "$pid"; then echo "pid $pid still alive after 90 s, SIGKILL"; kill -KILL -- "-$pid" 2>/dev/null; sleep 3; fi
  fi
  for i in $(seq 1 30); do bl_port_busy || break; sleep 1; done
  for i in $(seq 1 60); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 2000 ] && break; sleep 1
  done
  echo "stopped: $(basename "$run") (pid $pid); port $BL_PORT $(bl_port_busy && echo STILL BUSY || echo free); GPU used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
}
