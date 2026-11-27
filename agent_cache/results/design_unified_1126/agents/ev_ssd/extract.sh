#!/bin/zsh
# Usage: extract.sh <label> <run folder>
# One pass over a server log: keep the lines the SSD replay needs (original format).
label=$1; run=$2
out=/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd/extracts
nice -n 10 grep -E "HICACHE_EVT (h2s_[a-z]+|d2h_submit|s2h_[a-z]+|srs_park|srs_dispatch|srs_pin|srs_unpin|prefetch_start|ntp_gate|ntp_reevict|ntp_refetch|ntp_stage|load_back_init|nue_register|evict_host|evict_device) |NIXL L3 cleanup|HiCacheL3Cleaner|open failed|failed to retrieve|registration failed|Failed to register|HiCache prefetch|prefix shrank|ReqTimeStats|storage-hit accounting" "${run}/server.log" > "${out}/${label}.evt"
wc -l "${out}/${label}.evt"
