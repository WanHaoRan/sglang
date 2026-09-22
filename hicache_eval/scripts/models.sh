# Per-model constants. Source with a model key:
#   source models.sh qwen8b | qwen32b | llama70b
# Everything downstream reads MODEL / KV_BYTES_PER_TOKEN / MODEL_EXTRA_ARGS.
#
# KV_BYTES_PER_TOKEN is an architecture+dtype property and is the same on every box.
# L1_TOKENS is NOT: it is the profiled device pool at --mem-fraction-static 0.85 and it scales with HBM.
# The values below are the H100-80GB ones (campaigns 1-3). On the A100-80GB they were close but not equal
# (the 32B profiled 281,216, not 284,224). On THIS box (H200, 143,771 MiB) they are roughly 2.5x too small.
# Nothing in the Exp 0 / Exp 1 path reads L1_TOKENS -- it is documentation, and it is wrong here until a
# boot on this box fills it in. Read the real value off `max_total_num_tokens=` in any server.log.
case "${1:?usage: source models.sh <qwen8b|qwen32b|llama70b>}" in
  qwen8b)
    export MODEL=Qwen/Qwen3-8B
    export MODEL_KEY=qwen8b
    export KV_BYTES_PER_TOKEN=147456
    export MODEL_EXTRA_ARGS=""
    export REASONING_PARSER=qwen3
    export SUPPORTS_THINKING=1     # gates --default-chat-template-kwargs and Exp 4
    export L1_TOKENS_H100=374784   # H100 80GB, measured campaigns 1-3; NOT this box (see header)
    export L1_TOKENS=${L1_TOKENS:-}   # fill from max_total_num_tokens= on this box's first boot
    ;;
  qwen32b)
    export MODEL=Qwen/Qwen3-32B-FP8
    export MODEL_KEY=qwen32b
    export KV_BYTES_PER_TOKEN=131072
    export MODEL_EXTRA_ARGS="--kv-cache-dtype fp8_e5m2"
    export REASONING_PARSER=qwen3
    export SUPPORTS_THINKING=1
    export L1_TOKENS_H100=284224   # H100 80GB, measured campaigns 1-3; NOT this box (see header)
    export L1_TOKENS=${L1_TOKENS:-}   # fill from max_total_num_tokens= on this box's first boot
    ;;
  llama70b)
    export MODEL=casperhansen/llama-3.3-70b-instruct-awq
    export MODEL_KEY=llama70b
    export KV_BYTES_PER_TOKEN=163840
    export MODEL_EXTRA_ARGS="--kv-cache-dtype fp8_e5m2"
    # Llama 3.3 emits no reasoning channel. strip_thinking_cache is gated on
    # reasoning_tokens > 0 (schedule_batch.py:1381), so Exp 4 is a no-op here.
    export REASONING_PARSER=""
    export SUPPORTS_THINKING=0
    export L1_TOKENS_H100=193728   # H100 80GB, measured campaigns 1-3; NOT this box (see header)
    export L1_TOKENS=${L1_TOKENS:-}   # fill from max_total_num_tokens= on this box's first boot
    ;;
  *) echo "unknown model key: $1" >&2; return 1 ;;
esac
echo "model=$MODEL  b=$KV_BYTES_PER_TOKEN B/tok  thinking=$SUPPORTS_THINKING  L1_TOKENS=${L1_TOKENS:-<unmeasured on this box>}"
