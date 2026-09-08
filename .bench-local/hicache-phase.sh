#!/usr/bin/env bash
# =============================================================================
#  hicache-phase.sh — run one phase of the HiCache multi-tier evaluation.
#
#  Run this INSIDE the sglang dev container, in a second shell, while the
#  server is up. It snapshots /metrics around the run and prints the tier
#  counter deltas, so each phase is self-documenting.
#
#  The four-phase protocol (see ../README.md in MLSys-Learn):
#     0 cold      stop server, rm -rf /var/hicache/*, restart, then:
#                    ./hicache-phase.sh -l cold
#     1 populate  ./hicache-phase.sh -l populate
#     2 L3 read   ./hicache-phase.sh -l l3read --flush --drop-caches
#     3 all warm  ./hicache-phase.sh -l warm
#
#  --flush clears L1+L2 but NOT the on-disk L3 (UnifiedRadixCache.reset only
#  clears the host pool and the radix tree), which is exactly the state that
#  isolates the disk read path.
#
#  Usage:
#     ./hicache-phase.sh -l <label> [options]
#
#  Options:
#     -l, --label <name>   Phase label, used for output filenames (required)
#     --flush              POST /flush_cache first (drops L1+L2, keeps L3)
#     --drop-caches        Drop the OS page cache first (needs privileged)
#     --groups N           Distinct shared prefixes (default 256)
#     --per-group N        Requests per prefix (default 8)
#     --prefix-len N       Shared prefix tokens (default 4096)
#     --question-len N     Unique suffix tokens (default 128)
#     --output-len N       Decode tokens (default 128)
#     --concurrency N      Max in-flight requests (default 32)
#     --dist <d>           uniform | zipf (default zipf)
#     --zipf-alpha F       Zipf skew (default 1.0)
#     --cpus <list>        taskset CPU list for the client (default 20-25)
#     --model <name>       Model (default Qwen/Qwen3-8B)
#     --port N             Server port (default 30000)
#     --storage-dir <d>    L3 directory to measure (default /var/hicache)
#     --out-dir <d>        Results directory (default ./results next to this)
#     --dry-run            Print the command; run nothing
#     -h, --help           Show this help
# =============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

LABEL=""
DO_FLUSH=0
DO_DROP=0
NUM_GROUPS=256
PER_GROUP=8
PREFIX_LEN=4096
QUESTION_LEN=128
OUTPUT_LEN=128
CONCURRENCY=32
DIST=zipf
ZIPF_ALPHA=1.0
CPUS="20-25"
MODEL="Qwen/Qwen3-8B"
PORT=30000
STORAGE_DIR=/var/hicache
OUT_DIR="$SCRIPT_DIR/results"
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

while [ $# -gt 0 ]; do
  case "$1" in
    -l|--label)      [ $# -ge 2 ] || die "--label needs a value"; LABEL="$2"; shift 2 ;;
    --label=*)       LABEL="${1#*=}"; shift ;;
    --flush)         DO_FLUSH=1; shift ;;
    --drop-caches)   DO_DROP=1; shift ;;
    --groups)        NUM_GROUPS="$2"; shift 2 ;;
    --per-group)     PER_GROUP="$2"; shift 2 ;;
    --prefix-len)    PREFIX_LEN="$2"; shift 2 ;;
    --question-len)  QUESTION_LEN="$2"; shift 2 ;;
    --output-len)    OUTPUT_LEN="$2"; shift 2 ;;
    --concurrency)   CONCURRENCY="$2"; shift 2 ;;
    --dist)          DIST="$2"; shift 2 ;;
    --zipf-alpha)    ZIPF_ALPHA="$2"; shift 2 ;;
    --cpus)          CPUS="$2"; shift 2 ;;
    --model)         MODEL="$2"; shift 2 ;;
    --port)          PORT="$2"; shift 2 ;;
    --storage-dir)   STORAGE_DIR="$2"; shift 2 ;;
    --out-dir)       OUT_DIR="$2"; shift 2 ;;
    --dry-run)       DRY_RUN=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *)               die "Unknown option: $1  (try --help)" ;;
  esac
done

[ -n "$LABEL" ] || die "--label is required (e.g. -l populate)"
case "$DIST" in uniform|zipf) ;; *) die "--dist must be uniform or zipf" ;; esac

BASE="http://127.0.0.1:$PORT"
STAMP=$(date +%Y%m%d-%H%M%S)
PREFIX="$OUT_DIR/${STAMP}_${LABEL}"

# The client is a single asyncio loop; taskset keeps it off the server's cores
# so its own scheduling latency does not land in the TTFT numbers.
CLIENT=(python3 -m sglang.benchmark.serving
  --backend sglang
  --host 127.0.0.1 --port "$PORT"
  --model "$MODEL"
  --dataset-name generated-shared-prefix
  --gsp-num-groups "$NUM_GROUPS"
  --gsp-prompts-per-group "$PER_GROUP"
  --gsp-system-prompt-len "$PREFIX_LEN"
  --gsp-question-len "$QUESTION_LEN"
  --gsp-output-len "$OUTPUT_LEN"
  --gsp-group-distribution "$DIST"
  --max-concurrency "$CONCURRENCY"
  --warmup-requests 0
  --seed 42
  --output-details
  --output-file "${PREFIX}.jsonl")
if [ "$DIST" = "zipf" ]; then
  CLIENT+=(--gsp-zipf-alpha "$ZIPF_ALPHA")
fi
if command -v taskset >/dev/null 2>&1 && [ -n "$CPUS" ]; then
  CLIENT=(taskset -c "$CPUS" "${CLIENT[@]}")
fi

if [ "$DRY_RUN" -eq 1 ]; then
  step "Dry run"
  printf '    %s\n' "${CLIENT[*]}"
  exit 0
fi

mkdir -p "$OUT_DIR"

step "Phase '$LABEL'"
curl -sf -o /dev/null "$BASE/health" || die "Server not healthy at $BASE"
REQUESTS=$((NUM_GROUPS * PER_GROUP))
PREFIX_TOKENS=$((NUM_GROUPS * PREFIX_LEN))
info "workload: $REQUESTS requests, $NUM_GROUPS x ${PREFIX_LEN}-token prefixes = $PREFIX_TOKENS unique prefix tokens"
info "dist: $DIST$([ "$DIST" = zipf ] && echo " (alpha=$ZIPF_ALPHA)")   concurrency: $CONCURRENCY"

if [ "$DO_FLUSH" -eq 1 ]; then
  # Clears L1 (radix tree) and L2 (host pool). Leaves the L3 files on disk.
  curl -sf -X POST "$BASE/flush_cache" >/dev/null || warn "flush_cache failed"
  ok "flushed L1+L2 (L3 files kept)"
fi

if [ "$DO_DROP" -eq 1 ]; then
  # The file backend opens with buffering=0 but no O_DIRECT, so without this
  # the page cache serves most of L3 and the numbers are not disk numbers.
  if sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
    ok "dropped the OS page cache"
  else
    warn "could not drop the page cache (needs a privileged container)"
  fi
fi

l3_bytes() { du -sb "$STORAGE_DIR" 2>/dev/null | cut -f1 || echo 0; }

L3_BEFORE=$(l3_bytes)
curl -s "$BASE/metrics" > "${PREFIX}_metrics_before.txt"

step "Running"
"${CLIENT[@]}"

curl -s "$BASE/metrics" > "${PREFIX}_metrics_after.txt"
L3_AFTER=$(l3_bytes)

step "Tier counters for '$LABEL'"
python3 - "${PREFIX}_metrics_before.txt" "${PREFIX}_metrics_after.txt" <<'PY'
import sys

COUNTERS = [
    "sglang:prefetched_tokens_total",
    "sglang:storage_prefetch_hit_tokens_total",
    "sglang:storage_prefetch_unfulfilled_tokens_total",
    "sglang:hicache_backup_tokens_total",
    "sglang:hicache_backup_bytes_total",
    "sglang:hicache_dropped_tokens_total",
    "sglang:hicache_backup_dropped_tokens_total",
]
GAUGES = [
    "sglang:hicache_host_used_tokens",
    "sglang:hicache_host_total_tokens",
]


def load(path):
    """Sum each metric family across its label sets."""
    totals = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            head, _, value = line.rpartition(" ")
            name = head.split("{", 1)[0].strip()
            try:
                totals[name] = totals.get(name, 0.0) + float(value)
            except ValueError:
                continue
    return totals


before, after = load(sys.argv[1]), load(sys.argv[2])
width = max(len(n) for n in COUNTERS + GAUGES)

for name in COUNTERS:
    delta = after.get(name, 0.0) - before.get(name, 0.0)
    print(f"    {name:<{width}}  +{delta:,.0f}")
print()
for name in GAUGES:
    print(f"    {name:<{width}}   {after.get(name, 0.0):,.0f}")

hit = after.get("sglang:storage_prefetch_hit_tokens_total", 0.0) - before.get(
    "sglang:storage_prefetch_hit_tokens_total", 0.0
)
pref = after.get("sglang:prefetched_tokens_total", 0.0) - before.get(
    "sglang:prefetched_tokens_total", 0.0
)
if pref > 0:
    print(f"\n    L3 prefetch hit rate: {100.0 * hit / pref:.1f}% of prefetched tokens")
elif hit == 0:
    print("\n    No L3 traffic this phase — the working set never overflowed L1+L2.")
PY

printf '    %-46s  %s -> %s bytes\n' "L3 on disk ($STORAGE_DIR)" "$L3_BEFORE" "$L3_AFTER"

step "Done"
info "results:  ${PREFIX}.jsonl"
info "metrics:  ${PREFIX}_metrics_{before,after}.txt"
