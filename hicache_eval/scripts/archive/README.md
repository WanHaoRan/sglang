# Archived harness scripts

Moved here on 2026-09-21 when `hicache_eval/` was ported to the H200 box. **Nothing was deleted** — the frozen
campaign reports under `results/` cite several of these by path, and `git log --follow` still tracks each file.
A report that says `scripts/exp3.py` means `scripts/archive/exp3.py` for anything written before 2026-09-21.

The active set left in `scripts/` is the 13 files the Exp 0 / Exp 1 campaign actually runs
(`env.sh`, `models.sh`, `hcommon.py`, `probe.py`, `cachectl.py`, `start_server.sh`, `stop_server.sh`,
`telemetry.sh`, `exp0.py`, `exp1.py`, `writeload.py`, `compare_campaigns.py`, `plot_tier_ttft.py`) plus the
box driver `run_h200.sh`.

Nothing in the active set imports anything here. `compare_campaigns.py` and `plot_tier_ttft.py` name
`analyze_nixl.py` and `exp3.py` in comments, as the source of a formula; those are citations, not imports.

## Why each group is archived

### Exp 2 / 3 / 4 machinery — the experiments themselves are on hold
`exp2.py`, `exp3.py`, `exp1_bg_l3.py`, `analyze_exp234.py`, `analyze_missing.py`, `make_report.py`,
`close_gaps.py`, `peek.py`, and the drivers `run_exp2.sh`, `run_exp2b.sh`, `run_exp3.sh`, `run_exp234.sh`,
`run_all_exp234.sh`, `run_c1.sh`, `run_c2.sh`, `run_finish.sh`, `run_missing.sh`, `rerun_exp2_baseline.sh`,
`finish_all.sh`.

HANDOFF §5 is the reason: **Exp 3/4 cannot show a write-policy benefit at the plan's working-set size on any
model, because L3 is never read** — `nohicache` scores within 1-3 points of every HiCache condition. And Exp 2's
write load replays itself (HANDOFF §3, `PENDING.md` item 7), which invalidates the high-rate end of its rate
axis. Campaign 5 dropped Exp 2-4 for these reasons and campaign 6 (this box) does the same. Before any of this
is un-archived, do `PENDING.md` items 1 and 2: a per-invocation seed and index offset for the write load, and a
working set past L1+L2.

`writeload.py` stayed in the active set: `exp1.py --bg-rate` spawns it by absolute path.

### Superseded drivers — one driver per box, and this box has a new one
`run_32b_exp01.sh`, `run_70b_exp01.sh`, `run_p_measure.sh` (campaign 2, H100), `run_nixl_exp01.sh`,
`run_nixl_l2.sh`, `run_l2.sh` (campaign 1), `run_a100_rerun.sh` (campaign 4, the 8B A100 rerun),
`run_a100_rerun_c2.sh` (campaign 5, the 32B+70B A100 rerun).

`run_a100_rerun_c2.sh` is the direct ancestor of `scripts/run_h200.sh`: same stages, same flags, same order.
Read it when you want to know what campaign 5 did; run `run_h200.sh` when you want to measure this box.

### One-off analysis and plots — superseded by `compare_campaigns.py` and `plot_tier_ttft.py`
`analyze.py` (campaign 1 Exp 1 table), `analyze_nixl.py` (campaign 1 file-vs-nixl), `analyze_models.py`
(cross-model fits; its `polyfit` of per-length medians is the definition `compare_campaigns.py` reuses),
`plot_tier_gain.py` (per-tier gain curves, superseded by the TTFT figure), `backend_ab.py` (campaign 4's
file-vs-nixl A/B against the Exp 0 L3 probe).

`backend_ab.py` is the one worth reviving deliberately: the file-vs-nixl gap (34x on the H100 box, HANDOFF §2)
has never been measured on an SSD this fast. It needs no changes beyond `L3_DIR`.

### C3 GIL census
`c3_dump.sh`, `c3_pyspy.sh`. Thin (12 samples, 8B only) and directional; HANDOFF §6 item 3 wants it re-run on
all three models. Un-archive when that is the task — and note `py-spy record -s` hangs on the scheduler, only
`py-spy dump --nonblocking` works.
