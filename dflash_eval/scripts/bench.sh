#!/usr/bin/env bash
# Run one measurement cell against the server that launch.sh started and append a
# row to $RESULTS/summary.csv. Runs on the HOST with ~/dflash-venv active.
#
#   scripts/bench.sh <ARM> <DATASET> <CONCURRENCY> <SAMPLING> [NUM_PROMPTS] [MAX_NEW_TOKENS]
#
#   DATASET   gsm8k | math500 | humaneval | mbpp | mt-bench   (mt-bench has 80 prompts)
#   SAMPLING  greedy  -> --temperature 0 --reasoning off
#             think   -> --temperature 1 --top-p 0.95 --top-k 20 --reasoning on     (Track A)
#             xhigh   -> --temperature 1 --top-p 0.95 --top-k 20 --reasoning xhigh  (Track B)
#
# Sequence (RUNBOOK.md §5.1): POST /flush_cache (zeroes the lifetime accept-length
# accumulator), dflash benchmark openai, GET /server_info, append summary row.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$HERE/env.sh"

ARM=${1:?}; DATASET=${2:?}; CONC=${3:?}; SAMPLING=${4:?}
NUM_PROMPTS=${5:-128}; MAX_NEW=${6:-2048}
[[ "$DATASET" == "mt-bench" && "$NUM_PROMPTS" -gt 80 ]] && NUM_PROMPTS=80

case "$SAMPLING" in
  greedy) SAMP="--temperature 0 --reasoning off" ;;
  think)  SAMP="--temperature 1 --top-p 0.95 --top-k 20 --reasoning on" ;;
  xhigh)  SAMP="--temperature 1 --top-p 0.95 --top-k 20 --reasoning xhigh" ;;
  *) echo "unknown SAMPLING: $SAMPLING" >&2; exit 1 ;;
esac
MODEL=$(arm_model "$ARM"); WINDOW=$(arm_window "$ARM")
command -v dflash >/dev/null || { echo "dflash not on PATH; source ~/dflash-venv/bin/activate" >&2; exit 1; }
curl -sf "$BASE_URL/health" >/dev/null || { echo "no server on $BASE_URL" >&2; exit 1; }

mkdir -p "$RESULTS"
TAG="${ARM}_${DATASET}_c${CONC}_${SAMPLING}"
OUT="$RESULTS/${TAG}.log"
SUMMARY="$RESULTS/summary.csv"
[[ -f "$SUMMARY" ]] || echo "timestamp,arm,window,dataset,concurrency,sampling,num_prompts,max_new_tokens,client_throughput_tok_s,client_accept_len,server_avg_spec_accept_length,log" > "$SUMMARY"

server_al() {
  python3 -c "import json,urllib.request;s=json.load(urllib.request.urlopen('$BASE_URL/server_info'))['internal_states'][0];print(s.get('avg_spec_accept_length',''))"
}

echo "# $(date -Is) $TAG" | tee "$OUT"
curl -s -X POST "$BASE_URL/flush_cache" | tee -a "$OUT"; echo | tee -a "$OUT"

# GPU clock/power trace for the duration of the cell (throttling check)
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used --format=csv -l 5 > "$RESULTS/${TAG}.gpu.csv" &
SMI_PID=$!
trap 'kill $SMI_PID 2>/dev/null || true' EXIT

CMD="dflash benchmark openai --base-url $BASE_URL --model $MODEL --dataset $DATASET --num-prompts $NUM_PROMPTS --concurrency $CONC --max-new-tokens $MAX_NEW $SAMP"
echo "# $CMD" | tee -a "$OUT"
$CMD 2>&1 | tee -a "$OUT"

kill $SMI_PID 2>/dev/null || true
AL_SERVER=$(server_al)
echo "server avg_spec_accept_length (since flush): $AL_SERVER" | tee -a "$OUT"

# The client prints "Throughput:       1,234.56 tok/s" (thousands comma) and
# "Accept length:    4.567" (only when the patched client found meta_info); parse loosely.
THR=$(grep -oiE "throughput:[^0-9]*[0-9.,]+" "$OUT" | tail -1 | grep -oE "[0-9.,]+$" | tr -d , || true)
AL_CLIENT=$(grep -oiE "accept length:[^0-9]*[0-9.]+" "$OUT" | tail -1 | grep -oE "[0-9.]+$" || true)
echo "$(date -Is),$ARM,$WINDOW,$DATASET,$CONC,$SAMPLING,$NUM_PROMPTS,$MAX_NEW,${THR:-},${AL_CLIENT:-},${AL_SERVER:-},${TAG}.log" >> "$SUMMARY"
echo "row appended to $SUMMARY"
