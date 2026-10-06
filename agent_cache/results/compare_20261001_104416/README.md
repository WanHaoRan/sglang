# Campaign 11, fifth arm `three_tier_wc_norq` — separate driver run (2026-10-01)

This directory is the driver run that added `three_tier_wc_norq` to campaign 11 after its four main arms. The arm is
`three_tier_wc` with `--hicache-storage-prefetch-retry-max-attempts 0`: no admission-time SSD re-query (#39283) and no
paced retries, which is campaign 8's behaviour on the merged engine. `../chain_c11_norq.sh` launched it when campaign 11
finished and appended its manifest line to `../compare_20261001_013443/manifest.txt`.

**All outputs of this arm are in `../compare_20261001_013443/`**: `timeline_three_tier_wc_norq.png`,
`turns_three_tier_wc_norq.csv`, `events_three_tier_wc_norq.csv`, its row in `compare.csv`, the five-arm `compare.png`,
and the README comparing it with the other arms. That directory's `iostat_vdc.log` covers this arm's window
(10:46-13:16Z). `config.txt` and `manifest.txt` here are this driver's own records.
