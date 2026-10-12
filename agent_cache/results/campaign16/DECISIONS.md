# Campaign 16 — the Oracle v2 levers on campaign 14's workload (2026-10-11)

`ORACLE_V2_PLAN.md` §5, run by the procedure of `ORACLE_V2_HANDOFF.md` §H5. Driver: `../../scripts/run_campaign14.sh`
unchanged (resumable), with `ARMS`, `ORACLE_HINTS=1` and `ENGINE_PYTHONPATH` set at launch. Manifest `manifest.txt`,
driver logs `driver.log` and `driver_*.log`, SSD samples `iostat_vdc.log`, scripts as launched in `scripts_at_launch.tgz`.

## Arms (run in this order)

| arm | server, on top of `three_tier_wc` |
|---|---|
| `three_tier_wc_dl` | `--hicache-storage-read-order deadline --hicache-storage-read-window-pages 512 --hicache-storage-prefetch-pin-fraction 0.3`, oracle off (every read "now"); retry flags 8 / 8 |
| `three_tier_wc_oracle_v2` | the above + `--hicache-storage-next-turn-prefetch on`; paced retries off (poll interval 0, attempts 8) |
| `three_tier_wc_oracle_v2_nu` | the above + `--radix-eviction-policy next_use` (`horizon_s` 300) |

Baselines, not rerun (as in the handoff): campaign 14's `three_tier_wc` and campaign 15's `three_tier_wc_oracle`. Same
workload as campaigns 14 and 15 (768 fresh users, Poisson 0.1/s, backlog cap target 32, stationary start, x70 gaps capped
at 1,200 s, 5 h per arm), NAT160 pools, Qwen3-30B-A3B bf16 KV, TP 1, context 131,072, `ORACLE_HINTS=1`.

Engine: the frozen copy `agent_cache/frozen/engine_5df71c4bf0_v2/python` (HEAD `5df71c4bf0`, clean tree, 0 diff lines).

## `three_tier_wc_wb` is held

The plan's fourth arm (`--hicache-write-policy write_back`, oracle off) was smoked but not launched. In its smoke the L3
was written (1,106 host-to-storage writes, 33 GB) but never read: 0 of 1,281 lookups hit, and 133 of 352 returning turns
recomputed (`_dl`: 80, with 99 restored from the SSD). `Scheduler._prefetch_kvcache` issues an L3 lookup only when the
match's last host node is host-backed or is the root; its comment states the write-through assumption ("an unbacked
anchor means a guaranteed storage miss"). Under write-back a device-resident prefix has no host copy, so the anchor is
usually a device-only node and the lookup is skipped, and the #39283 re-query at admission meets the same gate. Of the
133 recomputed turns, 70 issued no lookup and 63 looked up only their tail (largest lookup per turn p50 576 tokens). The
arm would measure L1 + L2 write-back with L3 writes and no L3 reads, not the mirror ablation of plan §1.4. Add it
later (relaunch with it in `ARMS`) once write-back lookups are anchored on the device match, as buffer mode already
does, or if a two-tier write-back number is wanted.

## Pre-flight (2026-10-11)

- CPU gate in the container: the eight v2 test files pass with the laptop's counts (6, 6, 6, 15, 32, 4, 8, 5), and so
  do the unchanged suites the plan lists (`test_unified_radix_cache_unittest.py` 1,282 passed with the GPU visible).
- Smokes (`../oracle_v2_smokes_1011_1831/`, `../oracle_v2_smokes_1011_1848/`; driver
  `frozen/oracle_scripts_staging/v2_smokes.sh`): every arm on smoke C's small pools (L1 65,536 tokens, host 12 GB),
  32 sessions x 12 turns at x1 gaps, cold L3. At NAT160 a 32-session client never fills the host pool in a smoke's 8
  minutes (campaign 15's 128-user pre-flight needed ~17 min), so no read would have parked. All boot checks matched;
  384 turns and 0 errors per arm; 0 tracebacks.

| arm | parked (deadline now) | dispatched; launched=0 | pins / unpins (reasons) | other |
|---|---|---|---|---|
| `_dl` | 176 (176) | 176; 33 | 97 / 97 (admit) | read rate ~3 ms/page |
| `_oracle_v2` | 504 (27) | 504; 136 | 133 / 133 (116 admit, 17 replaced) | 99 % of triggers fired ahead of their read time; timed deadlines p50 23.5 s ahead; window held (in-flight p99 477 pages) |
| `_oracle_v2_nu` | 509 (34) | 509; 113 | 136 / 136 (108 admit, 28 replaced) | 736 `nue_register` |
| `_wb` | 0 | 0 | 0 | 0 L3 hits (above) |

`launched=0` is a dispatch whose host allocation failed (the read is revoked); every dispatch minus those equals the
`s2h_io` count. On the 12 GB pool it is expected: the host copy of L1 (65,536 tokens) plus the prefetch cap (61,056)
exceeds the pool (122,112). At NAT160 the same sum is 1.48M of 1.63M tokens, so the campaign should show few. Watch it
with `scripts/srs_summary.py`; under the oracle each one also costs a refetch and logs a spurious `ntp_reevict` (plan
§3.1).

## Fixes before launch

- **Zombie pids.** Since the host reboot of 2026-10-11 18:00Z the container's PID 1 is a bare `/bin/zsh` that does not
  reap orphans, so every exited server stays a zombie, `kill -0` reports it alive, `stop_server.sh` waits 120 s and
  `start_server.sh` refuses the next boot. Every arm after the first would have failed. `stop_server.sh`,
  `start_server.sh` and `start_client.sh` now treat a zombie as stopped (`alive()` checks the process state).
- **Launch command.** The handoff's §H5 line computes `ENGINE_PYTHONPATH` with `$(docker exec ...)` inside the
  container's `bash -c`, where there is no `docker`; the path would have been empty and every arm would have imported
  the working tree, not the freeze. The command below lists the freeze inside the container.

## Running it

```
docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign16 && ARMS="three_tier_wc_dl three_tier_wc_oracle_v2 three_tier_wc_oracle_v2_nu" ORACLE_HINTS=1 ENGINE_PYTHONPATH=$(ls -d /sgl-workspace/sglang/agent_cache/frozen/engine_5df71c4bf0_v2/python) nohup bash run_campaign14.sh ../results/campaign16 >> ../results/campaign16/driver.log 2>&1'
```

- Stop: `pkill -f '^bash run_campaign14.sh'`, then `pkill -f '^bash run_compare.sh'`, `pkill -f replay_agentic.py`,
  `bash stop_server.sh`, all in the container. Relaunching the same command resumes with the arms not yet in
  `manifest.txt`. Do not edit `agent_cache/scripts/` while it runs; `python/` is safe (frozen engine).
- About 5.1-5.5 h per arm (a NAT160 boot after another NAT160 arm reloads the weights from disk); ~16 h for the three.
- Analysis: `ORACLE_V2_HANDOFF.md` §H6 (`steady_state.py`, `oracle_eval.py`, `srs_summary.py`, `oracle_timeline.py`
  per arm against the two baselines; `frozen/oracle_scripts_staging/c15_diagnosis.py` takes `--baseline` and `--oracle`
  triples, not a manifest).

## Divergence from the cookbook

As campaigns 13-15: `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` recommends one GPU (TP 1 on H200), the default attention
backend and about 128K context; the study keeps its pinned server (`--attention-backend triton`, `--context-length
131072`, bf16 KV, HiCache pools as the variable) and adds the v2 flags.
