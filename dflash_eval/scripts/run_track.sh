#!/usr/bin/env bash
# Generic sequential runner for one track. Use the wrappers:
#   scripts/run_track_a.sh   (TRACK=A: Qwen3-8B, arms A0 A1 A2 A3, presets greedy|think)
#   scripts/run_track_b.sh   (TRACK=B: Qwen3.8-27B-FP8, arms B0 B1-4 B1-8 B2 B3, presets greedy|xhigh)
#
# For each arm: launch its server once (scripts/launch.sh), run every cell against it
# (scripts/bench.sh), stop it, move to the next arm. Cells already in summary.csv are
# skipped, so `RUN=<dir>` reruns resume. Failures go to failures.log and the run continues.
#
# Env knobs (all optional):
#   ARMS="B0 B2 B3"                 subset of arms
#   CELLS="gsm8k 1|humaneval 8"     '|'-separated "<dataset> <concurrency>" pairs
#   SAMPLING=greedy|think|xhigh     bench.sh preset (think = Track A thinking, xhigh = Track B reasoning)
#   ORDER=arm|cell                  per-arm blocks (default) or interleaved per cell (relaunch per cell)
#   NUM_PROMPTS=128 MAX_NEW=2048    forwarded to bench.sh
#   RUN=20260909_1855               reuse a results dir (resume)
#   DRY_RUN=1                       print the plan only
#
# Run from the host with ~/dflash-venv active, ideally detached:
#   nohup scripts/run_track_b.sh > /tmp/track_b.out 2>&1 &
#   tail -f results/$(cat .current_results)/track_b.log
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
EVAL_DIR=$(cd "$HERE/.." && pwd -P)

TRACK=${TRACK:?set TRACK=A or TRACK=B (or use run_track_a.sh / run_track_b.sh)}
case "$TRACK" in
  A) DEF_ARMS="A0 A1 A2 A3";        DEF_SAMPLING=greedy; PRESETS="greedy think"
     MODELS="models--Qwen--Qwen3-8B models--Tengyunw--qwen3_8b_eagle3 models--z-lab--Qwen3-8B-DFlash-b16 models--deepseek-ai--dspark_qwen3_8b_block7" ;;
  B) DEF_ARMS="B0 B1-4 B1-8 B2 B3"; DEF_SAMPLING=greedy; PRESETS="greedy xhigh"
     MODELS="models--Qwen--Qwen3.8-27B-FP8 models--RadixArk--Qwen3.8-27B-DSpark models--incoai--Qwen3.8-27B-DFlash2" ;;
  *) echo "TRACK must be A or B" >&2; exit 1 ;;
esac
ARMS=${ARMS:-$DEF_ARMS}
CELLS=${CELLS:-"gsm8k 1|gsm8k 8|humaneval 1|humaneval 8"}
SAMPLING=${SAMPLING:-$DEF_SAMPLING}
ORDER=${ORDER:-arm}
NUM_PROMPTS=${NUM_PROMPTS:-128}
MAX_NEW=${MAX_NEW:-2048}
DRY_RUN=${DRY_RUN:-0}
LOWER=$(echo "$TRACK" | tr 'A-Z' 'a-z')

# ---- results dir: reuse $RUN if given, else a fresh timestamped one ------------
RUN=${RUN:-$(date +%Y%m%d_%H%M)}
REAL_RESULTS="$EVAL_DIR/results/$RUN"
if [[ $DRY_RUN == 1 ]]; then
  export RESULTS=$(mktemp -d "${TMPDIR:-/tmp}/track_${LOWER}_dryrun.XXXXXX")   # env.sh skips its path guard when DRY_RUN=1
else
  export RESULTS="$REAL_RESULTS"
  mkdir -p "$RESULTS"
  echo "$RUN" > "$EVAL_DIR/.current_results"
fi
# shellcheck source=env.sh
source "$HERE/env.sh"
MASTER="$RESULTS/track_${LOWER}.log"
FAILURES="$RESULTS/failures.log"
SUMMARY="$REAL_RESULTS/summary.csv"      # real one even in a dry run, so the skip list is honest

log() { echo "$(date -Is) $*" | tee -a "$MASTER"; }

# ---- pre-flight --------------------------------------------------------------
preflight() {
  local ok=1
  for a in $ARMS; do
    [[ "$a" == "$TRACK"* ]] || { echo "arm '$a' is not a Track $TRACK arm" >&2; ok=0; }
    arm_flags "$a" >/dev/null || ok=0
  done
  [[ " $PRESETS " == *" $SAMPLING "* ]] || { echo "SAMPLING for Track $TRACK must be one of: $PRESETS (got '$SAMPLING')" >&2; ok=0; }
  if [[ $DRY_RUN == 0 ]]; then
    command -v dflash >/dev/null || { echo "dflash not on PATH: source ~/dflash-venv/bin/activate" >&2; ok=0; }
    if command -v dflash >/dev/null; then
      python3 -c 'import dflash.benchmark as b,sys; src=open(b.__file__).read(); sys.exit(0 if "choices" in src else 1)' \
        || { echo "dflash client is unpatched; run scripts/patch_dflash_client.sh" >&2; ok=0; }
    fi
    $DOCKER ps --format '{{.Names}}' | grep -qx "$CONTAINER" || { echo "container $CONTAINER is not running" >&2; ok=0; }
    if curl -sf "$BASE_URL/health" >/dev/null 2>&1; then
      echo "a server already answers on $BASE_URL; stop it first (scripts/launch.sh stop)" >&2; ok=0
    fi
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [[ "${used:-0}" -lt 2000 ]] || { echo "GPU already has ${used} MiB in use; another server is running" >&2; ok=0; }
    for m in $MODELS; do
      [[ -d "$HOME/.cache/huggingface/hub/$m" ]] || echo "note: $m is not in the HF cache yet; the first launch will download it (pre-download with 'hf download' inside the container to keep launches fast)" >&2
    done
  fi
  [[ $ok == 1 ]]
}

# already have a summary row for this cell? (columns: ts,arm,window,dataset,conc,sampling,...)
cell_done() {
  local arm=$1 ds=$2 conc=$3
  [[ -f "$SUMMARY" ]] && awk -F, -v a="$arm" -v d="$ds" -v c="$conc" -v s="$SAMPLING" \
    'NR>1 && $2==a && $4==d && $5==c && $6==s && $9!="" {found=1} END {exit found?0:1}' "$SUMMARY"
}

# ---- one arm block: launch, cells, stop ----------------------------------------
run_block() {            # run_block <arm> "<cell>|<cell>..."
  local arm=$1 cells=$2
  local pending=()
  IFS='|' read -r -a cell_list <<< "$cells"
  for cell in "${cell_list[@]}"; do
    read -r ds conc <<< "$cell"
    if cell_done "$arm" "$ds" "$conc"; then log "skip $arm $ds c$conc $SAMPLING (already in summary.csv)"; continue; fi
    pending+=("$ds $conc")
  done
  [[ ${#pending[@]} -eq 0 ]] && { log "arm $arm: nothing to do"; return 0; }

  log "=== arm $arm: launching (${#pending[@]} cells pending) ==="
  if [[ $DRY_RUN == 1 ]]; then
    "$HERE/launch.sh" print "$arm" | grep -E "^# (cd|boot)" | tee -a "$MASTER"
    for cell in "${pending[@]}"; do read -r ds conc <<< "$cell"; log "  would run: bench.sh $arm $ds $conc $SAMPLING $NUM_PROMPTS $MAX_NEW"; done
    return 0
  fi
  if ! "$HERE/launch.sh" "$arm" 2>&1 | tee -a "$MASTER"; then
    log "!!! launch failed for $arm; see $RESULTS/server_${arm}.log"
    echo "$(date -Is) launch $arm" >> "$FAILURES"
    "$HERE/launch.sh" stop >/dev/null 2>&1 || true
    return 1
  fi
  for cell in "${pending[@]}"; do
    read -r ds conc <<< "$cell"
    log "--- $arm $ds c$conc $SAMPLING ---"
    if ! "$HERE/bench.sh" "$arm" "$ds" "$conc" "$SAMPLING" "$NUM_PROMPTS" "$MAX_NEW" 2>&1 | tee -a "$MASTER"; then
      log "!!! bench failed: $arm $ds c$conc"
      echo "$(date -Is) bench $arm $ds c$conc $SAMPLING" >> "$FAILURES"
      curl -sf "$BASE_URL/health" >/dev/null 2>&1 || { log "server is down; abandoning arm $arm"; break; }
    fi
  done
  log "=== arm $arm: stopping ==="
  "$HERE/launch.sh" stop 2>&1 | tee -a "$MASTER"
  sleep 10
}

cleanup() { [[ $DRY_RUN == 0 ]] && "$HERE/launch.sh" stop >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

# ---- main -----------------------------------------------------------------------
preflight || exit 1
log "Track $TRACK run=$RUN arms=[$ARMS] cells=[$CELLS] sampling=$SAMPLING order=$ORDER num_prompts=$NUM_PROMPTS max_new=$MAX_NEW dry_run=$DRY_RUN"
log "results: $RESULTS"
START=$(date +%s)

if [[ $ORDER == arm ]]; then
  for arm in $ARMS; do run_block "$arm" "$CELLS"; done
elif [[ $ORDER == cell ]]; then
  IFS='|' read -r -a cell_list <<< "$CELLS"
  for cell in "${cell_list[@]}"; do
    for arm in $ARMS; do run_block "$arm" "$cell"; done
  done
else
  echo "ORDER must be arm or cell" >&2; exit 1
fi

log "done in $(( ($(date +%s) - START) / 60 )) min"
if [[ -f "$FAILURES" ]]; then log "FAILURES:"; cat "$FAILURES" | tee -a "$MASTER"; fi
if [[ -f "$SUMMARY" ]]; then
  log "summary ($SUMMARY):"
  column -s, -t < "$SUMMARY" 2>/dev/null | tee -a "$MASTER" || cat "$SUMMARY" | tee -a "$MASTER"
fi
