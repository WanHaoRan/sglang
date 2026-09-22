# Common environment for the HiCache evaluation. Source this, do not run it.
export WORK=/sgl-workspace/sglang/hicache_eval
export SGLANG_REPO=/sgl-workspace/sglang
# L3 store. Default is this box's attached SSD (/dev/vdc, 2.27 TiB, ext4 at /mnt/ssd); the H100
# campaigns used /var/hicache_l3 on the root disk and the A100 box /mnt/nvme/hicache_l3.
export L3_DIR=${L3_DIR:-/mnt/ssd/hicache_l3}
export RESULTS=${RESULTS:-$(cat $WORK/.current_results)}
# Finished campaigns are frozen (one dir name per line in .frozen_results): refuse to
# run a driver against one, so a rerun can never append to or overwrite old evidence.
if [ -f "$WORK/.frozen_results" ] && [ -z "${HICACHE_ALLOW_FROZEN:-}" ] \
   && grep -qxF "$(basename "$RESULTS")" "$WORK/.frozen_results"; then
  echo "env.sh: RESULTS=$RESULTS is a frozen campaign; point .current_results (or RESULTS) at a new dir" >&2
  case $- in *i*) return 1 ;; *) exit 1 ;; esac
fi
export PORT=30000
export BASE=http://127.0.0.1:$PORT
export MODEL=${MODEL:-Qwen/Qwen3-8B}          # override: MODEL=... before sourcing
export NVME_DEV=${NVME_DEV:-vdc}       # block device behind L3_DIR (iostat + /proc/diskstats); vda on the H100 box, nvme0n1 on the A100
export KV_BYTES_PER_TOKEN=${KV_BYTES_PER_TOKEN:-147456}   # Qwen3-8B bf16 KV; 131072 / 163840 for the fp8-KV 32B / 70B
export PAGE_SIZE=64
mkdir -p "$RESULTS" "$L3_DIR"

# ARG_MAX-safe L3 wipe: rm -rf $L3_DIR/* silently fails past ~10k files.
l3_wipe() { find "$L3_DIR" -mindepth 1 -delete 2>/dev/null; echo "L3 wiped: $(find "$L3_DIR" -name "*.bin" | wc -l) files remain"; }
