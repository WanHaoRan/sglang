#!/bin/sh
# Rebuild the event extracts the scripts here read: one gzipped file per arm, from the arm's server.log.
# Usage: sh extract.sh            (from this folder; CPU only, about a minute)
# Reads agent_cache/results/campaign34/manifest.txt and campaign32/manifest.txt; writes ext/<campaign>_<arm>.evt.gz.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
results=$(cd "${here}/../../.." && pwd)
mkdir -p "${here}/ext"
events='HICACHE_EVT (h2s_submit|h2s_done|h2s_io|h2s_group|srs_write_hold|evict_host|ntp_reevict|ntp_gate|ntp_room|nue_register|srs_dispatch|srs_park|srs_pin|srs_unpin|srs_pin_skipped|ec_bump|ec_release|ntp_stage|ntp_expire|ntp_refetch|ntp_arrive|ntp_admit|ntp_plan|s2h_io|s2h_query|d2h_done|ntp_hold)|NIXL L3 cleanup|Write page to storage|failed to retrieve page|HiCache prefetch (success|dropped)|HiCacheL3Cleaner started|host pool:|skip_stored_writes|HiCache storage reads:|HiCache next-turn'
for campaign in campaign34 campaign32; do
    tag=$(echo "${campaign}" | sed 's/campaign/c/')
    while read -r arm run prog; do
        [ -n "${arm}" ] || continue
        dst="${here}/ext/${tag}_${arm}.evt.gz"
        [ -s "${dst}" ] && continue
        nice -n 10 grep -E "${events}" "${results}/${run}/server.log" | gzip -6 > "${dst}"
        echo "${dst}"
    done < "${results}/${campaign}/manifest.txt"
done
