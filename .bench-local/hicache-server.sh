#!/usr/bin/env bash
# =============================================================================
#  hicache-server.sh — start/stop the SGLang server for the HiCache multi-tier
#  evaluation, with the tier ladder pinned and reported at startup.
#
#  Run this INSIDE the sglang dev container. Pair it with hicache-phase.sh,
#  which drives the client from a second shell.
#
#  The tier ladder it pins (Qwen3-8B, 147,456 B/token):
#     L1 device   --max-total-tokens 131072   131K tokens
#     L2 host     --hicache-size 40           271K tokens
#     L3 file     200Gi cap                  ~1.46M tokens
#  A working set must exceed L1+L2 (~402K tokens) before anything is read
#  back from L3. `start` prints the real numbers so a mis-sized run is
#  obvious before you spend an hour on it.
#
#  Full protocol:
#     ./hicache-server.sh start --wipe-l3      # phase 0: cold everything
#     ./hicache-phase.sh -l cold
#     ./hicache-phase.sh -l populate
#     ./hicache-phase.sh -l l3read --flush --drop-caches
#     ./hicache-phase.sh -l warm
#
#  Usage:
#     ./hicache-server.sh <start|stop|restart|status> [options]
#
#  Commands:
#     start            Launch the server (refuses if one is already up)
#     stop             Stop it and wait for the port to clear
#     restart          stop, then start
#     status           Health, resolved tier sizes, L3 on-disk size
#
#  Options:
#     --wipe-l3            Delete the L3 files before starting. Only safe
#                          while stopped -- the evictor holds an in-memory
#                          LRU index built at startup.
#     --model <name>       Model (default Qwen/Qwen3-8B)
#     --port N             Server port (default 30000)
#     --l1 N               --max-total-tokens, the device pool (default 131072)
#     --l2 N               --hicache-size in GB, the host pool (default 40)
#     --l3-max-size <sz>   L3 cap, SI/IEC suffixes (default 200Gi)
#     --l3-min-free <sz>   Free-space watermark on the fs (default 100Gi)
#     --storage-dir <d>    L3 directory (default /var/hicache)
#     --storage-backend <b>  file | nixl (default file). nixl uses O_DIRECT
#                          and so bypasses the page cache; file does not.
#     --context-length N   default 32768
#     --page-size N        default 64
#     --mem-fraction F     --mem-fraction-static (default 0.85)
#     --cpus <list>        taskset CPU list for the server (default 0-19)
#     --wait-sec N         Startup health timeout (default 900)
#     --foreground         Run in the foreground instead of backgrounding
#     --dry-run            Print what would run; run nothing
#     -h, --help           Show this help
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
LOG_DIR="$SCRIPT_DIR/logs"
PID_FILE="$SCRIPT_DIR/logs/server.pid"

CMD=""
WIPE_L3=0
MODEL="Qwen/Qwen3-8B"
PORT=30000
L1_TOKENS=131072
L2_GB=40
L3_MAX_SIZE="200Gi"
L3_MIN_FREE="100Gi"
STORAGE_DIR=/var/hicache
STORAGE_BACKEND="file"
CONTEXT_LEN=32768
PAGE_SIZE=64
MEM_FRACTION=0.85
CPUS="0-19"
WAIT_SEC=900
FOREGROUND=0
DRY_RUN=0

if [ -t 1 ]; then
  C_B=$'\033[1m'; C_G=$'\033[32m'; C_Y=$'\033[33m'; C_R=$'\033[31m'; C_0=$'\033[0m'
else
  C_B=''; C_G=''; C_Y=''; C_R=''; C_0=''
fi
step() { printf '\n%s==> %s%s\n' "$C_B" "$*" "$C_0"; }
info() { printf '    %s\n' "$*"; }
ok()   { printf '    %s+%s %s\n' "$C_G" "$C_0" "$*"; }
warn() { printf '    %s!%s %s\n' "$C_Y" "$C_0" "$*" >&2; }
die()  { printf '\n%sError:%s %s\n' "$C_R" "$C_0" "$*" >&2; exit 1; }

usage() { sed -n '2,/^# ===\+$/p' "$0" | sed 's/^# \{0,1\}//' | sed '$d'; }

# ------------------------------------------------------------------- args ---
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
  CMD="$1"; shift
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --wipe-l3)          WIPE_L3=1; shift ;;
    --model)            MODEL="$2"; shift 2 ;;
    --port)             PORT="$2"; shift 2 ;;
    --l1)               L1_TOKENS="$2"; shift 2 ;;
    --l2)               L2_GB="$2"; shift 2 ;;
    --l3-max-size)      L3_MAX_SIZE="$2"; shift 2 ;;
    --l3-min-free)      L3_MIN_FREE="$2"; shift 2 ;;
    --storage-dir)      STORAGE_DIR="$2"; shift 2 ;;
    --storage-backend)  STORAGE_BACKEND="$2"; shift 2 ;;
    --context-length)   CONTEXT_LEN="$2"; shift 2 ;;
    --page-size)        PAGE_SIZE="$2"; shift 2 ;;
    --mem-fraction)     MEM_FRACTION="$2"; shift 2 ;;
    --cpus)             CPUS="$2"; shift 2 ;;
    --wait-sec)         WAIT_SEC="$2"; shift 2 ;;
    --foreground)       FOREGROUND=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *)                  die "Unknown option: $1  (try --help)" ;;
  esac
done

[ -n "$CMD" ] || { usage; exit 1; }
case "$CMD" in start|stop|restart|status) ;; *) die "Unknown command: $CMD" ;; esac
case "$STORAGE_BACKEND" in file|nixl) ;; *) die "--storage-backend must be file or nixl" ;; esac

BASE="http://127.0.0.1:$PORT"

# ---------------------------------------------------------------- helpers ---
server_pid() {
  [ -f "$PID_FILE" ] || return 1
  local pid; pid=$(cat "$PID_FILE" 2>/dev/null) || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  printf '%s\n' "$pid"
}

healthy() { curl -sf -o /dev/null --max-time 5 "$BASE/health" 2>/dev/null; }

l3_bytes() { du -sb "$STORAGE_DIR" 2>/dev/null | cut -f1 || echo 0; }

# L1 and L2 come straight from the server; L3's token capacity is derived from
# the bytes-per-token the host pool actually resolved, so it stays right if the
# model changes.
report_tiers() {
  python3 - "$BASE" "$L2_GB" "$L3_MAX_SIZE" <<'PY'
import json, sys, urllib.request

base, l2_gb, l3_cap_raw = sys.argv[1], float(sys.argv[2]), sys.argv[3]


def fetch(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.read().decode()


def parse_size(text):
    si = {"k": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}
    iec = {"Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}
    for suf, mult in sorted(iec.items()) + sorted(si.items()):
        if text.endswith(suf):
            return int(float(text[: -len(suf)]) * mult)
    return int(text)


l1 = json.loads(fetch("/get_server_info")).get("max_total_num_tokens")

l2 = 0.0
for line in fetch("/metrics").splitlines():
    if line.startswith("sglang:hicache_host_total_tokens"):
        l2 += float(line.rsplit(" ", 1)[1])
l2 = int(l2)

bytes_per_token = (l2_gb * 1e9 / l2) if l2 else 0
l3_bytes = parse_size(l3_cap_raw)
l3 = int(l3_bytes / bytes_per_token) if bytes_per_token else 0

rows = [
    ("L1 device", f"{l1:,} tokens", "--max-total-tokens"),
    ("L2 host", f"{l2:,} tokens", f"--hicache-size {l2_gb:g}"),
    ("L3 storage", f"~{l3:,} tokens", f"{l3_cap_raw} cap"),
]
for name, tokens, note in rows:
    print(f"    {name:<12} {tokens:>18}   {note}")
if l1 and l2:
    total = f"{l1 + l2:,} tokens"
    print(f"\n    {'L1+L2':<12} {total:>18}   <- a working set must exceed "
          "this for any L3 read")
    print(f"    {'':<12} {'':>18}   i.e. groups x prefix-len >= "
          f"{2 * (l1 + l2):,} for a 2x margin")
if bytes_per_token:
    # Derived from the resolved host pool, so page alignment rounds the token
    # count up and leaves this a few bytes low against the true size.
    print(f"\n    KV per token: ~{bytes_per_token:,.0f} B (derived)")
PY
}

# ------------------------------------------------------------------ start ---
do_stop() {
  step "Stopping"
  local pid
  if pid=$(server_pid); then
    info "Sending SIGTERM to $pid"
    kill "$pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 60 ]; do
      sleep 1; waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
      warn "Still alive after 60s; sending SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
      sleep 2
    fi
    rm -f "$PID_FILE"
    ok "Stopped"
  else
    # A server started outside this script still has to go, or the port is busy.
    if pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null; then
      info "Killed a launch_server not started by this script"
      sleep 3
      ok "Stopped"
    else
      info "Nothing running"
    fi
    rm -f "$PID_FILE"
  fi
}

do_start() {
  # --dry-run changes nothing, so a live server must not block it.
  if [ "$DRY_RUN" -eq 0 ] && healthy; then
    die "A server is already answering on $BASE. Use 'restart', or 'stop' first."
  fi

  if [ "$WIPE_L3" -eq 1 ]; then
    step "Wiping L3"
    # Safe only while stopped: the evictor's LRU index is built at startup and
    # would otherwise keep counting bytes for files that no longer exist.
    if [ "$DRY_RUN" -eq 1 ]; then
      info "[dry-run] rm -rf $STORAGE_DIR/*"
    else
      local before; before=$(l3_bytes)
      rm -rf "${STORAGE_DIR:?}"/* 2>/dev/null || true
      mkdir -p "$STORAGE_DIR"
      ok "Removed $before bytes from $STORAGE_DIR"
    fi
  fi

  export SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE="$L3_MIN_FREE"
  if [ "$STORAGE_BACKEND" = "file" ]; then
    export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$STORAGE_DIR"
    export SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE="$L3_MAX_SIZE"
  else
    export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="$STORAGE_DIR"
  fi

  local cmd=(python3 -m sglang.launch_server
    --model-path "$MODEL"
    --port "$PORT"
    --page-size "$PAGE_SIZE"
    --context-length "$CONTEXT_LEN"
    --chunked-prefill-size 8192
    --mem-fraction-static "$MEM_FRACTION"
    --max-total-tokens "$L1_TOKENS"
    --enable-hierarchical-cache
    --hicache-size "$L2_GB"
    --hicache-write-policy write_through
    --hicache-io-backend kernel
    --hicache-mem-layout page_first
    --hicache-storage-backend "$STORAGE_BACKEND"
    --hicache-storage-prefetch-policy timeout
    --radix-eviction-policy lru
    --enable-metrics)
  if command -v taskset >/dev/null 2>&1 && [ -n "$CPUS" ]; then
    cmd=(taskset -c "$CPUS" "${cmd[@]}")
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    step "Dry run"
    info "SGLANG_HICACHE_*_STORAGE_DIR=$STORAGE_DIR"
    [ "$STORAGE_BACKEND" = "file" ] && info "SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=$L3_MAX_SIZE"
    info "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE=$L3_MIN_FREE"
    printf '    %s\n' "${cmd[*]}"
    return 0
  fi

  mkdir -p "$LOG_DIR" "$STORAGE_DIR"

  if [ "$FOREGROUND" -eq 1 ]; then
    step "Starting in the foreground"
    exec "${cmd[@]}"
  fi

  local stamp log
  stamp=$(date +%Y%m%d-%H%M%S)
  log="$LOG_DIR/server-$stamp.log"

  step "Starting"
  info "backend: $STORAGE_BACKEND   L3: $STORAGE_DIR (cap $L3_MAX_SIZE)"
  info "cpus: $CPUS   log: $log"
  nohup "${cmd[@]}" > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"
  ln -sfn "$log" "$LOG_DIR/server-latest.log"
  info "pid $pid; waiting up to ${WAIT_SEC}s for /health"

  local waited=0
  while [ "$waited" -lt "$WAIT_SEC" ]; do
    if healthy; then
      ok "Healthy after ${waited}s"
      step "Tier ladder"
      report_tiers
      step "Ready"
      info "logs:   tail -f $LOG_DIR/server-latest.log"
      info "client: ./hicache-phase.sh -l <label>"
      return 0
    fi
    # Fail fast on a crash instead of burning the whole timeout.
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      printf '\n'; tail -30 "$log" >&2
      die "Server exited during startup. Full log: $log"
    fi
    sleep 3
    waited=$((waited + 3))
  done

  printf '\n'; tail -30 "$log" >&2
  die "Server did not become healthy within ${WAIT_SEC}s. Log: $log"
}

do_status() {
  step "Status"
  local pid
  if pid=$(server_pid); then
    ok "Running as pid $pid"
  elif healthy; then
    warn "Healthy, but not started by this script (no live pid file)"
  else
    info "Not running"
  fi

  if healthy; then
    step "Tier ladder"
    report_tiers
  fi

  step "L3 on disk"
  info "$STORAGE_DIR: $(l3_bytes) bytes ($(du -sh "$STORAGE_DIR" 2>/dev/null | cut -f1 || echo 0))"
}

case "$CMD" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
esac
