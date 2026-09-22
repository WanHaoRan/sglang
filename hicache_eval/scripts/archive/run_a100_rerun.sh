#!/usr/bin/env bash
# Rerun of the Qwen3-8B / nixl HiCache campaign (backend A/B + Exp 0-4) on the GCP A100 box.
#
#   bash run_a100_rerun.sh <stage> [<stage> ...]
#   stages, in order: ab exp01 exp1_l2 exp2_base exp2_timeout exp3 exp4        ("all" = every stage)
#
# Server flags and client parameters are those of campaign 1 (20260907_203035/nixl, Exp 0/1 + A/B)
# and campaign 3 (20260908_nixl_exp234, Exp 2/3/4). Everything that differs because of THIS box is
# set here, not in the shared scripts:
#   * L3 on the local NVMe (/mnt/nvme/hicache_l3, device nvme0n1) instead of the old virtio disk.
#   * HICACHE_FLUSH_TIMEOUT: the fixed 120 s flush wait was sized for a ~1.7 GB/s disk.
#   * Attention backend: the H100 defaulted to fa3; SM80 defaults to flashinfer and fa3 cannot be pinned
#     here (decode CUDA-graph capture dies with "scheduler_metadata must have shape (metadata_size)",
#     see preflight/boot_fa3/server.log), so this campaign runs the default. ATTENTION_BACKEND=<x> overrides.
#   * Exp 1 main pass runs recompute,L1,L3 only; L2 comes from the dedicated pass (as it did before:
#     the combined run's L2 phase produced 0 rows and 63 discards on the old box).
#   * Exp 2 runs as wiped chunks of the same rate ladder. The nixl cleaner deletes oldest files first
#     at 80 % of the FILESYSTEM; on a 369 GB disk one un-chunked sweep would write past that and
#     delete the pre-populated probe prefixes mid-sweep. It also passes a longer writeload duration
#     (blocks are slower here) and the non-replaying writeload (HANDOFF §3, PENDING item 7).
#   * C1 / C2 are not run (HANDOFF §5: invalid; C2 also needs a patch under python/).
# Results go to $RESULTS (hicache_eval/.current_results); env.sh refuses frozen campaigns.
set -uo pipefail
S=/sgl-workspace/sglang/hicache_eval/scripts

export L3_DIR=/mnt/nvme/hicache_l3
export NVME_DEV=nvme0n1
export HICACHE_FLUSH_TIMEOUT=${HICACHE_FLUSH_TIMEOUT:-1800}
export ATTENTION_BACKEND=${ATTENTION_BACKEND-}
export HICACHE_BACKEND=nixl
export AB_STORE_DIR=$L3_DIR
source "$S/models.sh" qwen8b
source "$S/env.sh"
export EXP2_SUBDIR=${EXP2_SUBDIR:-exp2_qwen8b}
export EXP3_SUBDIR=${EXP3_SUBDIR:-exp3_qwen8b}
EXP2_DURATION=${EXP2_DURATION:-900}
EXP2_CHUNKS=(${EXP2_CHUNKS:-0,0.5 1 2 4 8 16})     # same ladder as campaign 3: 0,0.5,1,2,4,8,16
EXP2_TCHUNKS=(${EXP2_TCHUNKS:-0,2 8})              # same timeout ladder: 0,2,8
CLIENT_ARGS="--request-length 1024 --output-length 1024 --num-clients 16 --num-rounds 4 --max-parallel 8 --request-rate 2"
HI_COMMON="--enable-hierarchical-cache --hicache-size 100 --hicache-io-backend kernel \
--hicache-mem-layout page_first --hicache-storage-backend nixl --radix-eviction-policy lru \
--hicache-write-policy write_through"
LOG="$RESULTS/run_qwen8b.log"
mkdir -p "$RESULTS" "$RESULTS/$EXP2_SUBDIR" "$RESULTS/$EXP3_SUBDIR"
cd "$S"

say()  { echo -e "\n========== [$(date -u +%FT%TZ)] $* ==========" | tee -a "$LOG"; }
gate() { # the store must be the NVMe; an unmounted /mnt/nvme would silently land on the root disk
  [ "$(findmnt -n -o SOURCE -T "$L3_DIR")" = "/dev/$NVME_DEV" ] \
    || { echo "GATE FAILED: $L3_DIR is not on /dev/$NVME_DEV" | tee -a "$LOG"; exit 1; }
  bash "$S/stop_server.sh" >>"$LOG" 2>&1      # a stage always starts from no server: wipes must not race a draining one
  local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
  [ "$avail" -ge 110 ] || { echo "GATE FAILED: only ${avail} GiB RAM available (need 110 for the pinned pool)" | tee -a "$LOG"; exit 1; }
}
wipe() { find "$L3_DIR" -mindepth 1 -delete 2>/dev/null
         local n; n=$(find "$L3_DIR" -type f | wc -l); echo "L3 wiped: $n files remain" >>"$LOG"
         [ "$n" -eq 0 ] || { echo "WIPE FAILED: $n files remain in $L3_DIR" | tee -a "$LOG"; exit 1; }; }
postboot() { # the store the server opened must be the one the harness wipes and measures
  local f="$RESULTS/$1/startup_facts.txt"
  grep -q "HiCacheL3Cleaner started: dirs=\['$L3_DIR'\]" "$f" \
    || { echo "POSTBOOT FAILED: cleaner dir is not $L3_DIR in $f" | tee -a "$LOG"; return 1; }
  grep -q "O_DIRECT is active" "$f" || echo "POSTBOOT WARNING: no O_DIRECT line in $f" | tee -a "$LOG"
  echo "attention backend for $1: $(grep -ao -m1 "'attention_backend': '[a-z0-9_]*'" "$RESULTS/$1/server.log")" >>"$LOG"
}
boot() { local tag="$1"; shift
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  bash "$S/start_server.sh" "$tag" "$@" >>"$LOG" 2>&1 || return 1
  postboot "$tag"
}
health() { # after a stage: disk usage, cleaner activity, host memory
  df -h "$L3_DIR" | tail -1 | tee -a "$LOG"
  local t c
  for t in "$@"; do
    c=$(grep -ac "NIXL L3 cleanup" "$RESULTS/$t/server.log" 2>/dev/null); echo "cleaner lines in $t/server.log: ${c:-0}" | tee -a "$LOG"
  done
  free -g | sed -n 2p | tee -a "$LOG"
}

stage_ab() {
  say "Backend A/B, 4096-token probe: nixl"
  gate; wipe
  if boot ab_nixl $HI_COMMON --hicache-storage-prefetch-policy wait_complete; then
    AB_TAG=nixl python3 "$S/backend_ab.py" > "$RESULTS/ab_nixl/run.log" 2>&1 || echo "ab nixl FAILED" | tee -a "$LOG"
  else echo "ab nixl server FAILED" | tee -a "$LOG"; fi
  say "Backend A/B, 4096-token probe: file (reference backend)"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; wipe
  if bash "$S/start_server.sh" ab_file ${HI_COMMON/--hicache-storage-backend nixl/--hicache-storage-backend file} \
       --hicache-storage-prefetch-policy wait_complete >>"$LOG" 2>&1; then
    AB_TAG=file python3 "$S/backend_ab.py" > "$RESULTS/ab_file/run.log" 2>&1 || echo "ab file FAILED" | tee -a "$LOG"
  else echo "ab file server FAILED" | tee -a "$LOG"; fi
  bash "$S/stop_server.sh" >>"$LOG" 2>&1; health ab_nixl
}

stage_exp01() {
  say "Exp 0 + Exp 1 main pass (tiers recompute,L1,L3)"
  gate
  EXP1_TIERS=recompute,L1,L3 bash "$S/run_nixl_exp01.sh" >>"$LOG" 2>&1
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  postboot exp0_nixl; postboot exp1_nixl; health exp0_nixl exp1_nixl
}

stage_exp1_l2() {
  say "Exp 1 dedicated L2 pass"
  gate
  bash "$S/telemetry.sh" start "$RESULTS/exp1_nixl_l2" >>"$LOG" 2>&1
  bash "$S/run_nixl_l2.sh" >>"$LOG" 2>&1
  python3 "$S/cachectl.py" scrape "$RESULTS/exp1_nixl_l2/metrics_after.txt" >>"$LOG" 2>&1
  bash "$S/telemetry.sh" stop "$RESULTS/exp1_nixl_l2" >>"$LOG" 2>&1
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  postboot exp1_nixl_l2; health exp1_nixl_l2
}

exp2_arm() { # exp2_arm <control label> <prefetch policy> <log name> <chunk>...
  local control="$1" policy="$2" runlog="$3"; shift 3
  local i=0 chunk tag
  for chunk in "$@"; do
    i=$((i+1)); tag="${EXP2_SUBDIR}_${control}_c$i"
    say "Exp2 $control ($policy) chunk $i rates=$chunk writeload-duration=$EXP2_DURATION"
    gate; wipe
    if boot "$tag" $HI_COMMON --hicache-storage-prefetch-policy "$policy"; then
      python3 "$S/cachectl.py" scrape "$RESULTS/$tag/metrics_before.txt" >>"$LOG" 2>&1
      python3 "$S/exp2.py" --rates "$chunk" --reps 3 --control "$control" --duration "$EXP2_DURATION" \
        >>"$RESULTS/$EXP2_SUBDIR/$runlog" 2>&1 || echo "exp2 $control chunk $i FAILED" | tee -a "$LOG"
      python3 "$S/cachectl.py" scrape "$RESULTS/$tag/metrics_after.txt" >>"$LOG" 2>&1
    else echo "exp2 $control chunk $i server FAILED" | tee -a "$LOG"; fi
    health "$tag"
  done
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
}
stage_exp2_base()    { exp2_arm baseline       wait_complete run_baseline.log "${EXP2_CHUNKS[@]}"; }
stage_exp2_timeout() { exp2_arm timeout_policy timeout       run_timeout.log  "${EXP2_TCHUNKS[@]}"; }

stage_exp3() {
  say "Exp3 five conditions"
  gate; bash "$S/stop_server.sh" >>"$LOG" 2>&1
  python3 "$S/exp3.py" --conditions wt100,wt30,wts100,wts30,nohicache --client-args "$CLIENT_ARGS" \
    >>"$RESULTS/$EXP3_SUBDIR/run_exp3.log" 2>&1 || echo "exp3 FAILED" | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  for t in exp3_wt100 exp3_wt30 exp3_wts100 exp3_wts30; do postboot "$t"; done
  health exp3_wt100 exp3_wt30 exp3_wts100 exp3_wts30
}

stage_exp4() {
  say "Exp4 strip-thinking-cache oracle"
  gate; bash "$S/stop_server.sh" >>"$LOG" 2>&1
  python3 "$S/exp3.py" --conditions oracle100,oracle30 --client-args "$CLIENT_ARGS" \
    >>"$RESULTS/$EXP3_SUBDIR/run_exp4.log" 2>&1 || echo "exp4 FAILED" | tee -a "$LOG"
  bash "$S/stop_server.sh" >>"$LOG" 2>&1
  for t in exp3_oracle100 exp3_oracle30; do postboot "$t"; done
  health exp3_oracle100 exp3_oracle30
}

[ $# -ge 1 ] || { sed -n 2,6p "$0"; exit 2; }
stages=("$@"); [ "$1" = all ] && stages=(ab exp01 exp1_l2 exp2_base exp2_timeout exp3 exp4)
for st in "${stages[@]}"; do
  declare -F "stage_$st" >/dev/null || { echo "unknown stage: $st" >&2; exit 2; }
  "stage_$st"
  say "STAGE DONE: $st"
done
