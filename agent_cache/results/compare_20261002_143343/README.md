# Campaign 12, x10 — relaunch after the outage (2026-10-02)

This directory is the second driver run of campaign 12's x10 scale. The first run, `../compare_20261002_060200/`
(launched 06:02Z), finished `hbm_host` and was cut off about 40 minutes into `three_tier_to` when the host went down
(last log line 08:33Z, back ~14:29Z). This run, launched at 14:33Z on the frozen engine `2abe005bb2`, ran the remaining
three arms: `three_tier_to`, `three_tier_wc` and `hbm_lru`.

**All x10 results are in `../compare_20261002_060200/`**: figures, tables, the SSD samples and the README with the
result table. Its `manifest.txt` lists all four arms, because this directory's three lines were appended there (as
campaign 11 did for its fifth arm). `config.txt` and `manifest.txt` here are this driver's own records.

The cut-off run, `../20261002_075232_three_tier_to_NAT160` (3,560 of 4,671 turns), is not used.
