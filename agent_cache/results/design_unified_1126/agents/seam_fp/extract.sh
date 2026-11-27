#!/bin/zsh
# Stream each run's server.log once and keep the lines the seam_fp checks read (read-only on the run folders).
# Usage: nice -n 10 zsh extract.sh   (writes evt/<label>.evt.gz; skips labels already extracted)
set -u
here=${0:A:h}
res=/home/wanhr/sglang/agent_cache/results
pat='HICACHE_EVT (ntp_|ec_|s2h_|srs_|host_census|prefetch_start|load_back_init|evict_host)|ReqTimeStats\(|HiCache prefetch |device prefix shrank|NIXL L3 cleanup|Prefill batch|Decode batch|storage reads: order|next-turn pricing|max_total_num_tokens='
while IFS=$'\t' read -r label run prog; do
  out=${here}/evt/${label}.evt.gz
  [[ -s ${out} ]] && continue
  log=${res}/${run}/server.log
  [[ -r ${log} ]] || { echo "missing ${log}"; continue; }
  LC_ALL=C grep -a -E "${pat}" "${log}" | gzip -c > "${out}.tmp" && mv "${out}.tmp" "${out}"
  echo "${label} $(zcat ${out} | wc -l) lines"
done < ${here}/runs.tsv
