# Common environment for the HiCache evaluation. Source this, do not run it.
export WORK=/sgl-workspace/sglang/hicache_eval
export SGLANG_REPO=/sgl-workspace/sglang
export L3_DIR=${L3_DIR:-/var/hicache_l3}
export RESULTS=$(cat $WORK/.current_results)
export PORT=30000
export BASE=http://127.0.0.1:$PORT
export MODEL=${MODEL:-Qwen/Qwen3-8B}          # override: MODEL=... before sourcing
export NVME_DEV=vda
export KV_BYTES_PER_TOKEN=147456
export PAGE_SIZE=64
mkdir -p "$RESULTS" "$L3_DIR"

# ARG_MAX-safe L3 wipe: rm -rf $L3_DIR/* silently fails past ~10k files.
l3_wipe() { find "$L3_DIR" -mindepth 1 -delete 2>/dev/null; echo "L3 wiped: $(find "$L3_DIR" -name "*.bin" | wc -l) files remain"; }
