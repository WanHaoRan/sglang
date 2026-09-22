#!/usr/bin/env bash
# Rerun of campaign 2 (20260908_llama70b_awq_fp8kv) on the GCP A100 box: Exp 0 + Exp 1 for
# Llama-3.3-70B-AWQ and Qwen3-32B-FP8, fp8_e5m2 KV, nixl L3.
#
#   bash run_a100_rerun_c2.sh <stage> [<stage> ...]
#   stages, in campaign-2 order:
#     preflight_32b  p_measure_70b  exp0_70b exp1_70b exp1_70b_l2  exp0_32b exp1_32b exp1_32b_l2
#   ("all" = everything after the preflight)
#
# Flags, client parameters, tags and order are campaign 2's (run_70b_exp01.sh, run_32b_exp01.sh,
# run_p_measure.sh). What differs because of THIS box is set here:
#   * L3 on the local NVMe, device nvme0n1; HICACHE_FLUSH_TIMEOUT for the slower disk.
#   * ATTENTION_BACKEND=triton. Campaign 2 resolved to triton on every boot (fa3 rejects fp8_e5m2
#     KV and sglang rewrote it); SM80 would default to flashinfer and never hit that rewrite.
#   * START_TIMEOUT_S: the first boot of each model JIT-compiles its Marlin kernels on 12 vCPUs.
#   * preflight_32b: unmeasured boot that settles whether the FP8 checkpoint runs on SM80 at all
#     (weight-only FP8 Marlin here, DeepGEMM W8A8 on the H100) and absorbs that JIT cost;
#     p_measure_70b does the same for the 70B, exactly as it did in campaign 2.
#   * boot failures stop the stage; the old drivers ran the client against a dead server.
# REASONING_PARSER is left at its default (qwen3) for both models, as campaign 2 launched them.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts

export RESULTS=${RESULTS:-/sgl-workspace/sglang/hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv}
export L3_DIR=/mnt/nvme/hicache_l3
export NVME_DEV=nvme0n1
export HICACHE_FLUSH_TIMEOUT=${HICACHE_FLUSH_TIMEOUT:-1800}
export ATTENTION_BACKEND=${ATTENTION_BACKEND-triton}
export START_TIMEOUT_S=${START_TIMEOUT_S:-2400}
export HICACHE_BACKEND=nixl
case "$(basename "$RESULTS")" in 20260917_a100_gcp_qwen8b) echo "refusing to write into the 8B campaign dir" >&2; exit 1 ;; esac
source "$S/env.sh"
LOG="$RESULTS/run_c2.log"
mkdir -p "$RESULTS"
cd "$S"

HI="--kv-cache-dtype fp8_e5m2 --enable-hierarchical-cache --hicache-size 100 \
--hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first \
--hicache-storage-backend nixl --hicache-storage-prefetch-policy wait_complete --radix-eviction-policy lru"

model_of() { case "$1" in
  70b) echo casperhansen/llama-3.3-70b-instruct-awq ;; 32b) echo Qwen/Qwen3-32B-FP8 ;;
  *) echo "unknown model key $1" >&2; return 1 ;; esac; }
kv_of() { case "$1" in 70b) echo 163840 ;; 32b) echo 131072 ;; esac; }
use_model() { export MODEL; MODEL=$(model_of "$1") || exit 2; export KV_BYTES_PER_TOKEN; KV_BYTES_PER_TOKEN=$(kv_of "$1"); }

say()  { echo -e "\n========== [$(date -u +%FT%TZ)] $* ==========" | tee -a "$LOG"; }
gate() {
  [ "$(findmnt -n -o SOURCE -T "$L3_DIR")" = "/dev/$NVME_DEV" ] \
    || { echo "GATE FAILED: $L3_DIR is not on /dev/$NVME_DEV" | tee -a "$LOG"; exit 1; }
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "$avail" -ge 110 ] || { echo "GATE FAILED: only ${avail} GiB RAM available" | tee -a "$LOG"; exit 1; }
}
wipe() { find "$L3_DIR" -mindepth 1 -delete 2>/dev/null
         local n; n=$(find "$L3_DIR" -type f | wc -l); echo "L3 wiped: $n files remain" >>"$LOG"
         [ "$n" -eq 0 ] || { echo "WIPE FAILED: $n files remain in $L3_DIR" | tee -a "$LOG"; exit 1; }; }
boot() { # boot <tag> <args...>; a failed boot ends the stage instead of measuring a dead server
  local tag="$1"; shift
  if ! bash "$S/start_server.sh" "$tag" "$@" >>"$LOG" 2>&1; then
    echo "BOOT FAILED: $tag (see $RESULTS/$tag/server.log)" | tee -a "$LOG"; bash "$S/stop_server.sh" >>"$LOG" 2>&1; return 1
  fi
  local sl="$RESULTS/$tag/server.log"
  grep -aoE "existing=[0-9]+ B \([0-9]+ entries\)|KV Cache is allocated[^,]*|max_total_num_tokens=[0-9]+" "$sl" | head -3 > "$RESULTS/$tag/cold_check.txt"
  { echo "boot facts for $tag:"
    grep -ao -m1 "'attention_backend': '[a-z0-9_]*'" "$sl"
    grep -a -m1 -iE "marlin|awq_marlin|DeepGEMM" "$sl" | cut -c1-200
    grep -a -m1 "KV Cache is allocated" "$sl" | cut -c1-160
    grep -a -m1 "host pool" "$sl" | cut -c1-160
    free -g | sed -n 2p; } >>"$LOG"
}
postboot_hicache() {
  local f="$RESULTS/$1/startup_facts.txt"
  grep -q "HiCacheL3Cleaner started: dirs=\['$L3_DIR'\]" "$f" \
    || { echo "POSTBOOT FAILED: cleaner dir is not $L3_DIR in $f" | tee -a "$LOG"; return 1; }
  grep -q "O_DIRECT is active" "$f" || echo "POSTBOOT WARNING: no O_DIRECT line in $f" | tee -a "$LOG"
}
health() { df -h "$L3_DIR" | tail -1 | tee -a "$LOG"
  local t c; for t in "$@"; do c=$(grep -ac "NIXL L3 cleanup" "$RESULTS/$t/server.log" 2>/dev/null); echo "cleaner lines in $t/server.log: ${c:-0}" | tee -a "$LOG"; done
  free -g | sed -n 2p | tee -a "$LOG"; }
measured() { # measured <tag> <client command...>: telemetry + metrics around one client run
  local tag="$1"; shift
  bash "$S/telemetry.sh" start "$RESULTS/$tag" >>"$LOG" 2>&1
  python3 "$S/cachectl.py" scrape "$RESULTS/$tag/metrics_before.txt" >>"$LOG" 2>&1
  "$@"
  python3 "$S/cachectl.py" scrape "$RESULTS/$tag/metrics_after.txt" >>"$LOG" 2>&1
  bash "$S/telemetry.sh" stop "$RESULTS/$tag" >>"$LOG" 2>&1
}
sanity_generation() { # coherent text is the only check that catches wrong FP8 scales
  curl -s "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":48,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:300])"
}

stage_preflight_32b() {
  use_model 32b; say "PREFLIGHT 32B (unmeasured): does $MODEL run on SM80 with triton + fp8_e5m2 KV + HiCache?"
  gate; wipe; boot preflight/boot_32b $HI || return 1; postboot_hicache preflight/boot_32b
  local out="$RESULTS/preflight/boot_32b/checks.txt"
  { echo "# cold probe, then L1 re-probe (expect device 4032)"
    python3 "$S/probe.py" --len 4096 --seed 424242; python3 "$S/probe.py" --len 4096 --seed 424242
    echo "# drain + flush, L3 stats (expect 128 files, $((4096*131072)) bytes), then L3 probe (expect storage 4032)"
    python3 -c "import sys; sys.path.insert(0,'$S'); import hcommon; print('drain_s', hcommon.wait_until_flushable(1800)); print(hcommon.l3_stats()); print(hcommon.drop_page_cache())"
    python3 "$S/probe.py" --len 4096 --seed 424242
    echo "# sanity generations"
    sanity_generation "What is the capital of France? Answer in one short sentence."
    sanity_generation "Compute 17 * 23 and reply with just the number."
  } > "$out" 2>&1
  cat "$out" | cut -c1-260 | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; health preflight/boot_32b
}

stage_p_measure_70b() {
  use_model 70b; say "p_measure 70B: pure recompute, no HiCache (also absorbs the 70B's JIT-cold boot)"
  gate; wipe; boot p_measure --kv-cache-dtype fp8_e5m2 --radix-eviction-policy lru || return 1
  { echo "# sanity generations"; sanity_generation "What is the capital of France? Answer in one short sentence."
    sanity_generation "Compute 17 * 23 and reply with just the number."; } > "$RESULTS/p_measure/sanity.txt" 2>&1
  cat "$RESULTS/p_measure/sanity.txt" | tee -a "$LOG"
  python3 "$S/cachectl.py" scrape "$RESULTS/p_measure/metrics_before.txt" >>"$LOG" 2>&1
  EXP1_OUT=p_measure python3 "$S/exp1.py" --reps 3 --tiers recompute \
    --lengths 512,1024,2048,4096,8192,16384,32512 > "$RESULTS/p_measure/run.log" 2>&1 || echo "p_measure FAILED" | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; health p_measure
}

exp0() { local k="$1"; use_model "$k"; say "Exp 0 $k ($MODEL)"
  gate; wipe; boot "exp0_$k" $HI || return 1; postboot_hicache "exp0_$k"
  measured "exp0_$k" bash -c "EXP0_OUT=exp0_$k python3 '$S/exp0.py' > '$RESULTS/exp0_$k/run.log' 2>&1" || echo "exp0 $k FAILED" | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; health "exp0_$k"; }
exp1() { local k="$1"; use_model "$k"; say "Exp 1 main pass $k (recompute, L1, L3)"
  gate; wipe; boot "exp1_$k" $HI || return 1; postboot_hicache "exp1_$k"
  measured "exp1_$k" bash -c "EXP1_OUT=exp1_$k python3 '$S/exp1.py' --reps 3 --tiers recompute,L1,L3 > '$RESULTS/exp1_$k/run.log' 2>&1" || echo "exp1 $k FAILED" | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; health "exp1_$k"; }
exp1_l2() { local k="$1"; use_model "$k"; say "Exp 1 dedicated L2 pass $k (fresh server)"
  gate; wipe; boot "exp1_${k}_l2" $HI || return 1; postboot_hicache "exp1_${k}_l2"
  measured "exp1_${k}_l2" bash -c "EXP1_OUT=exp1_$k python3 '$S/exp1.py' --reps 3 --tiers L2 > '$RESULTS/exp1_$k/run_l2.log' 2>&1" || echo "exp1 L2 $k FAILED" | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; health "exp1_${k}_l2"; }
stage_exp0_70b() { exp0 70b; };    stage_exp1_70b() { exp1 70b; };    stage_exp1_70b_l2() { exp1_l2 70b; }
stage_exp0_32b() { exp0 32b; };    stage_exp1_32b() { exp1 32b; };    stage_exp1_32b_l2() { exp1_l2 32b; }

[ $# -ge 1 ] || { sed -n 2,8p "$0"; exit 2; }
stages=("$@"); [ "$1" = all ] && stages=(p_measure_70b exp0_70b exp1_70b exp1_70b_l2 exp0_32b exp1_32b exp1_32b_l2)
for st in "${stages[@]}"; do
  declare -F "stage_$st" >/dev/null || { echo "unknown stage: $st" >&2; exit 2; }
  if "stage_$st"; then say "STAGE DONE: $st"; else say "STAGE FAILED: $st"; fi
done
