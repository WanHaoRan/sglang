# Shared settings for dflash_eval scripts. Source this; do not execute it.
#
#   source /lambda/nfs/MLSys-Learn/sglang/dflash_eval/scripts/env.sh
#
# DOCKER: prefix used for every docker call. Shells that predate `usermod -aG docker`
# (including Claude Code / VS Code sessions) must keep using sudo (RUNBOOK.md §2.1).
export DOCKER=${DOCKER:-"sudo docker"}
export CONTAINER=${CONTAINER:-sglang_dev}
export PORT=${PORT:-30000}
export BASE_URL=${BASE_URL:-"http://127.0.0.1:${PORT}"}

# Repo path as seen from the host and from inside the container (same inode).
export DFLASH_EVAL_HOST=/lambda/nfs/MLSys-Learn/sglang/dflash_eval
export DFLASH_EVAL_CTR=/sgl-workspace/sglang/dflash_eval

# Active results directory (host path). Create one per campaign:
#   export RESULTS=$DFLASH_EVAL_HOST/results/$(date +%Y%m%d_%H%M)
export RESULTS=${RESULTS:-"$DFLASH_EVAL_HOST/results/$(cat "$DFLASH_EVAL_HOST/.current_results" 2>/dev/null || echo scratch)"}
# Canonicalize: ~/MLSys-Learn is a symlink to /lambda/nfs/MLSys-Learn, and the container
# only knows the /lambda/nfs path (as /sgl-workspace/sglang). A symlink-form RESULTS
# would otherwise map to a container path that does not exist and the server would
# silently never start.
RESULTS=$(readlink -f "$RESULTS")
export RESULTS
export RESULTS_CTR=${RESULTS_CTR:-"${RESULTS/$DFLASH_EVAL_HOST/$DFLASH_EVAL_CTR}"}
if [[ "${DRY_RUN:-0}" != 1 && "$RESULTS_CTR" == "$RESULTS" ]]; then
  echo "env.sh: RESULTS=$RESULTS is not under $DFLASH_EVAL_HOST; the container cannot see it" >&2
  return 1 2>/dev/null || exit 1
fi

# ---- Track A: Qwen3-8B --------------------------------------------------------
export COMMON_A="--model-path Qwen/Qwen3-8B --trust-remote-code --dtype bfloat16 \
--attention-backend fa3 --speculative-draft-attention-backend fa3 \
--disable-radix-cache --mem-fraction-static 0.80 \
--max-running-requests 32 --cuda-graph-max-bs-decode 32 \
--enable-metrics --decode-log-interval 10 --host 0.0.0.0 --port ${PORT}"

# ---- Track B: Qwen3.8-27B-FP8 -----------------------------------------------
# --max-running-requests <= 16 is mandatory for B2/B3 on 80 GB (RUNBOOK.md §6.1).
export COMMON_B="--model-path Qwen/Qwen3.8-27B-FP8 --trust-remote-code \
--kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 \
--attention-backend fa3 --speculative-draft-attention-backend fa3 \
--disable-radix-cache --mem-fraction-static 0.80 \
--max-running-requests 16 --cuda-graph-max-bs-decode 16 \
--chunked-prefill-size 8192 --max-prefill-tokens 8192 \
--reasoning-parser qwen3 --enable-metrics --decode-log-interval 10 --host 0.0.0.0 --port ${PORT}"
export ENV_B="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# arm name -> extra flags. Keep in sync with RUNBOOK.md §5 and §6.
arm_flags() {
  case "$1" in
    A0)      echo "" ;;
    A1)      echo "--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16" ;;
    A1-auto) echo "--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3" ;;
    A1-w8)   echo "--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8" ;;
    A1-card) echo "--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32" ;;
    A2)      echo "--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --speculative-num-draft-tokens 16" ;;
    A2-w8)   echo "--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --speculative-num-draft-tokens 8" ;;
    A3)      echo "--speculative-algorithm DSPARK --speculative-draft-model-path deepseek-ai/dspark_qwen3_8b_block7 --speculative-dspark-block-size 7" ;;
    B0)      echo "" ;;
    B1-4)    echo "--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4" ;;
    B1-8)    echo "--speculative-algorithm EAGLE --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8" ;;
    B2)      echo "--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark --speculative-dspark-block-size 7 --speculative-num-steps 1" ;;
    B3)      echo "--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8" ;;
    *) echo "unknown arm: $1" >&2; return 1 ;;
  esac
}

# arm name -> served model name the client should pass as --model.
arm_model() {
  case "$1" in
    A*) echo "Qwen/Qwen3-8B" ;;
    B*) echo "Qwen/Qwen3.8-27B-FP8" ;;
    *) return 1 ;;
  esac
}

# arm name -> verify window (for the results table).
arm_window() {
  case "$1" in
    A0|B0) echo 1 ;;
    A1|A2) echo 16 ;;
    A1-auto|B1-4) echo 4 ;;
    A1-w8|A2-w8|A3|B1-8|B2|B3) echo 8 ;;
    A1-card) echo 32 ;;
    *) return 1 ;;
  esac
}
