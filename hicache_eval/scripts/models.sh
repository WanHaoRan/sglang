# Per-model constants for the nixl Exp 2/3/4 sweep. Source with a model key:
#   source models.sh qwen8b | qwen32b | llama70b
# Everything downstream reads MODEL / KV_BYTES_PER_TOKEN / MODEL_EXTRA_ARGS.
case "${1:?usage: source models.sh <qwen8b|qwen32b|llama70b>}" in
  qwen8b)
    export MODEL=Qwen/Qwen3-8B
    export MODEL_KEY=qwen8b
    export KV_BYTES_PER_TOKEN=147456
    export MODEL_EXTRA_ARGS=""
    export REASONING_PARSER=qwen3
    export SUPPORTS_THINKING=1     # gates --default-chat-template-kwargs and Exp 4
    export L1_TOKENS=374784
    ;;
  qwen32b)
    export MODEL=Qwen/Qwen3-32B-FP8
    export MODEL_KEY=qwen32b
    export KV_BYTES_PER_TOKEN=131072
    export MODEL_EXTRA_ARGS="--kv-cache-dtype fp8_e5m2"
    export REASONING_PARSER=qwen3
    export SUPPORTS_THINKING=1
    export L1_TOKENS=284224
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
    export L1_TOKENS=193728
    ;;
  *) echo "unknown model key: $1" >&2; return 1 ;;
esac
echo "model=$MODEL  b=$KV_BYTES_PER_TOKEN B/tok  thinking=$SUPPORTS_THINKING"
