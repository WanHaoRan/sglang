# Oracle smoke C4 (2026-10-09)

C4: run C's oracle arm on the pre-fix code (frozen copy `frozen/engine_294c82b7fc_stale_probe`). Same knobs and client
as `../compare_20261009_082335/` (driver `frozen/oracle_scripts_staging/oracle_smoke_c4.sh`); one `three_tier_wc_oracle`
arm, paired with that directory's observe arm. Results and discussion: `../compare_20261009_082335/README.md`, section
"Reruns". Two-arm manifest for `analyze_c*.py`: `frozen/oracle_scripts_staging/c_pairs/c4/manifest.txt`.
