# Oracle v2 handoff: the GPU half (smokes and campaign 16)

Everything in `ORACLE_V2_PLAN.md` that a CPU can check is done (its §9). What is left needs the H200 box: boot each
new arm once, confirm the new events appear and nothing crashes, then run the four-arm campaign 16 and fill the go/no-go
table of the plan's §6. This document is the procedure, written for the box as campaign 15 left it (`results/campaign15/
DECISIONS.md`): the `sglang_hicache` container, `/sgl-workspace/sglang` = `/home/wanhr/sglang`, the SSD on `/dev/vdc`
at `/mnt/ssd/hicache_l3`, Qwen3-30B-A3B-Instruct-2507 cached, the 262K agentic trace under `agent_cache/traces/`.

Paths: host `/home/wanhr/sglang/agent_cache` is container `/sgl-workspace/sglang/agent_cache`; every command below says
which side it runs on. The engine freeze is the campaign-15 convention: a copy of `python/` under
`agent_cache/frozen/engine_<commit>_<tag>/python` with the uncommitted diff beside it, so a campaign keeps running the
code it started with while the working tree moves on.

## 0. The short version

1. §H1 box and checkout state (host, 5 min, GPU idle).
2. §H2 freeze the engine (container, 2 min, GPU idle).
3. §H3 CPU gate inside the container (container, 3 min, GPU idle).
4. §H4 smoke each of the four arms: boot, 32-session client, event summary, stop (container, ~20 min per arm, GPU
   busy; 80 min in all).
5. §H5 campaign 16: four arms of 5 h on campaign 14's workload (container, ~22 h, GPU busy; resumable).
6. §H6 evaluation against campaign 14's `three_tier_wc` and campaign 15's `three_tier_wc_oracle` (host, 30 min, GPU idle).
7. §H7 fill `ORACLE_V2_PLAN.md` §6 and decide what to keep.

## H1 Box and checkout state

**Status:** done 2026-10-11 18:24Z. The host had rebooted at 18:00Z: the container was `Exited (255)` (restarted with
`docker start sglang_hicache`), and `/mnt/ssd/hicache_l3` still held campaign 15's 1.6 TB at the cleaner's 70 % mark
(`start_server.sh` does not wipe it; the smoke driver of §H4 wipes it before every boot, as `run_compare.sh` does).
**Goal:** make sure the box runs the v2 code and the campaign-14 environment, so a smoke failure is a code failure and
not a stale checkout or an unmounted SSD.
**Runs on:** host, as `wanhr`
**Touches:** read-only (a `git pull` updates `/home/wanhr/sglang` to the pushed commit)
**Takes:** 5 min, GPU idle

```bash
# host: the v2 code must be in the box's checkout (pull the pushed commit first if the files are missing)
cd /home/wanhr/sglang && git log --oneline -1 && for f in python/sglang/srt/mem_cache/storage_read_scheduler.py \
  python/sglang/srt/mem_cache/next_use_eviction.py agent_cache/scripts/srs_summary.py; do echo "$f: $([ -f $f ] && echo present || echo MISSING)"; done
```
-> prints the commit and three `present` lines; a `MISSING` line means the pull did not happen.

```bash
# host: container up, SSD mounted, GPU idle, model cached
echo "container: $(docker ps --format '{{.Names}} {{.Status}}' | grep sglang_hicache)"
echo "ssd: $(findmnt -n -o SOURCE,FSTYPE -T /mnt/ssd/hicache_l3)"
echo "gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "model: $(ls -d /home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507 2>/dev/null || echo MISSING)"
echo "server running: $(docker exec sglang_hicache bash -c 'cat /sgl-workspace/sglang/agent_cache/results/.current_run 2>/dev/null; pgrep -af sglang.launch_server | head -1')"
```
-> prints `sglang_hicache Up ...`, `/dev/vdc ext4`, a used-memory figure under 1,000 MiB, the model directory and an
empty `server running:` line. A running server must be stopped (`bash stop_server.sh` in the container) before §H4.

**Expected result:** every line passes; no server is up.
**If it differs:** `ssd:` empty: the SSD is not mounted (campaign RUNBOOK §2.2); every L3 write would land on the root
disk and `start_server.sh` refuses to boot. Container `Exited`: `sudo docker start sglang_hicache`.
**Expected lessons:** none beyond the trap: a boot against the wrong checkout silently runs the v1 engine, because every
v2 flag defaults to the old behaviour.

## H2 Freeze the engine

**Status:** done 2026-10-11: `agent_cache/frozen/engine_5df71c4bf0_v2` (0 diff lines), run after §H3 so a failing test
would not have needed a second freeze.
**Goal:** a copy of `python/` that campaign 16 imports, so edits during the 22 h do not reach later arms.
**Runs on:** container `sglang_hicache` (`docker exec -it sglang_hicache bash`), in `/sgl-workspace/sglang`
**Touches:** creates `agent_cache/frozen/engine_<commit>_v2/`
**Takes:** 2 min, GPU idle

```bash
# container: freeze the working tree's engine next to the earlier freezes (campaign 15 used engine_294c82b7fc_ntpfix)
cd /sgl-workspace/sglang && C=$(git -c safe.directory='*' rev-parse --short HEAD) && F=agent_cache/frozen/engine_${C}_v2 \
  && [ ! -d "$F" ] && mkdir -p "$F" && cp -r python "$F/python" && git -c safe.directory='*' diff > "$F/worktree.diff" \
  && echo "frozen: $F ($(wc -l < $F/worktree.diff) diff lines)" || echo "STOP: $F exists; pick another tag"
```
-> prints `frozen: agent_cache/frozen/engine_<commit>_v2 (N diff lines)`; N is 0 when everything is committed.

```bash
# container: the freeze must carry the read scheduler, or start_server.sh refuses the v2 arms
ls agent_cache/frozen/engine_*_v2/python/sglang/srt/mem_cache/storage_read_scheduler.py
```
-> prints the path.

**Expected result:** the directory exists and the file is listed.
**Expected lessons:** none; this is the campaign-15 convention repeated.

## H3 CPU gate inside the container

**Status:** done 2026-10-11: 6, 6, 6, 15, 32, 4, 8, 5 passed (`test_oracle_v2_flags.py` has 6 since the §3.1 retry rule,
not the 5 below). The plan's unchanged suites pass as well; `test_unified_radix_cache_unittest.py` 1,282 passed with the
GPU visible.
**Goal:** the unit tests that pass on the laptop also pass on the box's Python (same sglang, the real `sgl_kernel`), so
a smoke failure is not an import-level mismatch.
**Runs on:** container, `/sgl-workspace/sglang`
**Touches:** read-only
**Takes:** 3 min, GPU idle

```bash
# container: every v2 test file in its own process, as CI does
cd /sgl-workspace/sglang && for t in test/registered/unit/mem_cache/test_storage_read_scheduler.py \
  test/registered/unit/mem_cache/test_next_use_eviction.py test/registered/unit/mem_cache/test_oracle_v2_flags.py \
  test/registered/unit/mem_cache/test_storage_read_dispatch.py test/registered/unit/mem_cache/test_next_turn_prefetch.py \
  test/registered/unit/mem_cache/test_next_turn_estimators.py test/registered/unit/mem_cache/test_storage_prefetch_lifecycle.py \
  test/registered/unit/managers/test_scheduler_hicache_events.py; do echo "== $t"; python3 -m pytest -q -p no:cacheprovider "$t" 2>&1 | tail -1; done
```
-> prints one `N passed` line per file with no `failed` or `error`; the laptop counts (2026-10-11) are 6, 6, 5, 15,
32, 4, 8 and 5 in the order listed.

**Expected result:** eight `passed` lines with those counts.
**If it differs:** a collection error names a missing module: the container's sglang install is older than the checkout
(`pip install -e python` in the container, as the RUNBOOK's setup did). A failing assertion is a real regression:
stop and fix before booting anything.
**Expected lessons:** none.

## H4 Smoke each arm

**Status:** done 2026-10-11, with three deviations (`results/campaign16/DECISIONS.md` has the table):
1. Small pools: at NAT160 a 32-session client never fills the host pool in 8 min, so no read would park. The smokes ran
   on smoke C's pools (`L1_TOKENS=65536 HOST_GB=12`), 32 sessions x 12 turns at x1 gaps, from
   `frozen/oracle_scripts_staging/v2_smokes.sh` (`results/oracle_v2_smokes_1011_1831/` and `_1848/`). All boot checks
   matched; 0 errors, 0 tracebacks. `srs_dispatch launched=0` was 33 / 136 / 113 on `_dl` / `_v2` / `_v2_nu`: the 12 GB
   pool is oversubscribed by construction (host copy of L1 65,536 + prefetch cap 61,056 > 122,112 tokens).
2. Zombie servers: since the reboot the container's PID 1 (`/bin/zsh`) does not reap, an exited server stays a zombie,
   `kill -0` calls it alive and the next boot is refused. `stop_server.sh`, `start_server.sh` and `start_client.sh` now
   treat a zombie as stopped.
3. `_wb` is held: its L3 was written (33 GB) but 0 of 1,281 lookups hit and 133 of 352 returning turns recomputed.
   `Scheduler._prefetch_kvcache` looks up L3 only when the match's last host node is host-backed or the root (a
   write-through assumption); under write-back that anchor is usually a device-only node, so the lookup and the #39283
   re-query are skipped (70 of the 133 issued none, the other 63 only their tail).
**Goal:** each new arm boots, the v2 events appear in the proportions the design implies, and nothing crashes in a
short run; this is the gate before 22 h of GPU time.
**Runs on:** container, `/sgl-workspace/sglang/agent_cache/scripts`
**Touches:** creates one boot directory per arm under `agent_cache/results/<UTC>_<arm>_NAT160/` with its client run;
writes to `/mnt/ssd/hicache_l3` (the L3 store; the cleaner keeps it under 70 % of the disk)
**Takes:** ~20 min per arm (boot 5-10 min, client ~8 min, drain 1 min), GPU busy

Paste this variables block first; it is campaign 14's server environment (`run_campaign14.sh` line 60):

```bash
# container: the campaign-14 server environment, used by every boot below
export MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= L3_CLEANER_PCT=70,60
export ENGINE_PYTHONPATH=$(ls -d /sgl-workspace/sglang/agent_cache/frozen/engine_*_v2/python | tail -1)
export TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json
cd /sgl-workspace/sglang/agent_cache/scripts && echo "engine: $ENGINE_PYTHONPATH"
```
-> prints the frozen engine path from §H2.

Then, for each ARM in `three_tier_wc_dl`, `three_tier_wc_oracle_v2`, `three_tier_wc_oracle_v2_nu`, `three_tier_wc_wb`:

```bash
# container: boot the arm (JIT-cold boot up to ~11 min, warm ~5 min; watch: tail -f ../results/$(cat ../results/.current_run)/server.log)
ARM=three_tier_wc_dl && bash start_server.sh $ARM NAT160
```
-> ends with `READY: <arm> after Ns` and the startup-facts block; for `_dl` and the two `_oracle_v2` arms the
`read order:` line must read `deadline, pin 0.3, banner 1`, for `_wb` `fifo, pin 0.0, banner 0`; `write policy:`
`write_back` only on `_wb`; `eviction policy:` `next_use` only on `_nu`; `next-turn mode:` `on` with `banner 1` on the
two `_oracle_v2` arms. The retries: the banner's `retry_poll_interval=` reads 0 on the two `_oracle_v2` arms (paced
miss retries off, plan §3.1) and 8 on `_dl`; `re-query cap:` reads 8 on every v2 arm, because the admission re-query
keeps its budget. A `retry_poll_interval=8` on an `_oracle_v2` arm means the engine predates the retry rule.

```bash
# container: 32 sessions x 12 turns with the next-turn hints, x10 gaps capped at 600 s (~8 min; stop early: pkill -f replay_agentic.py)
NCONV=32 C=32 TURNS=12 GAP=10 GAP_CAP=600 CHECK_IDS=0 ORACLE_HINTS=1 TAG=smoke bash start_client.sh
```
-> ends with the client's summary line; `errors` must be 0.

```bash
# container: the v2 event summary of the boot that just ran
R=../results/$(cat ../results/.current_run) && python3 srs_summary.py --run $R && echo "tracebacks: $(grep -ac Traceback $R/server.log)"
```
-> prints the counts table; the pass conditions per arm are:

| arm | `srs_park` | `srs_park` with deadline=now | `srs_dispatch` = `s2h_io` | `srs_pin` | `nue_register` | `unpin reasons` |
|---|---|---|---|---|---|---|
| `three_tier_wc_dl` | > 0 | all of them (the oracle is off) | yes | > 0 | 0 | `admit` only (an `abort` or two is fine) |
| `three_tier_wc_oracle_v2` | > 0 | fewer than all (timed deadlines exist) | yes | > 0 | 0 | as above |
| `three_tier_wc_oracle_v2_nu` | > 0 | fewer than all | yes | > 0 | > 0 | as above |
| `three_tier_wc_wb` | 0 | 0 of 0 | 0 = 0 | 0 | 0 | none |

and on every arm `tracebacks: 0`, `srs_dispatch launched=0: 0`, no `leak` among the unpin reasons, and on the
`deadline` arms `ms_per_page (srs_rate)` prints numbers once more than 64 reads ran (a short smoke may print `n/a`).

```bash
# container: stop the server before the next arm
bash stop_server.sh
```
-> prints that the pid is gone and port 30000 is free.

**Expected result:** four boot directories, each with a `client_*_smoke/client.jsonl` and a passing summary.
**If it differs:** `srs_dispatch` below `s2h_io` on a deadline arm: a read reached the I/O thread without passing the
scheduler; grep `server.log` for `HICACHE_EVT srs_park` around the first `s2h_io` whose rid has no `srs_dispatch` and
compare with `unified_radix_cache.py` `_drain_and_alloc_storage_hit`. A `leak` unpin: a request admitted without
`_unpin_prefetch` on its path (the three admit sites are `cache_finished_req`, `cache_unfinished_req`,
`init_load_back`); find its rid's lifecycle in the log. A traceback in `_dispatch_storage_reads` or `_launch_storage_hit`:
stop, fix on CPU with `test_storage_read_dispatch.py` extended by the failing shape, re-freeze (§H2), re-smoke.
**Expected lessons:** the smoke shows whether the parked-without-memory path works end to end on a real controller and
whether the pin releases balance; it says nothing about the gains, which need §H5's load.

## H5 Campaign 16

**Status:** running since 2026-10-11 19:17:35Z with three arms (`_dl`, `_oracle_v2`, `_oracle_v2_nu`; `_wb` held, §H4),
expected done ~11:30Z on 2026-10-12. The command below had a bug: `$(docker exec ...)` runs inside the container's
`bash -c`, where there is no `docker`, so `ENGINE_PYTHONPATH` would have been empty and the arms would have imported the
working tree. The launch used `ENGINE_PYTHONPATH=$(ls -d /sgl-workspace/sglang/agent_cache/frozen/engine_5df71c4bf0_v2/python)`
(`results/campaign16/DECISIONS.md`).
**Goal:** the four arms on campaign 14's workload, 5 h each, against the campaign 14 and 15 baselines.
**Runs on:** container, detached (`docker exec -d`), as in campaign 15
**Touches:** `agent_cache/results/campaign16/` (driver logs, manifest, iostat) and one boot directory per arm under
`agent_cache/results/`; `/mnt/ssd/hicache_l3`
**Takes:** ~22 h, GPU busy (4 x (boot + 5 h + drain)); resumable after a reboot

```bash
# host: launch the four arms (campaign 15's command with the v2 arms and the v2 freeze; driver log: results/campaign16/driver.log)
docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign16 && ARMS="three_tier_wc_dl three_tier_wc_oracle_v2 three_tier_wc_oracle_v2_nu three_tier_wc_wb" ORACLE_HINTS=1 ENGINE_PYTHONPATH=$(docker exec sglang_hicache bash -c "ls -d /sgl-workspace/sglang/agent_cache/frozen/engine_*_v2/python | tail -1") nohup bash run_campaign14.sh ../results/campaign16 >> ../results/campaign16/driver.log 2>&1'
```
-> the driver log starts with `LAUNCH ... arms: three_tier_wc_dl ...` and `START arms: ...`; one `STAGE DONE <arm>`
line per finished arm; `CAMPAIGN DONE` at the end.

```bash
# host: progress (one line per arm when done) and the live server's backlog
docker exec sglang_hicache bash -c 'grep -aE "STAGE (START|DONE|FAILED)|CAMPAIGN" /sgl-workspace/sglang/agent_cache/results/campaign16/driver.log; tail -2 /sgl-workspace/sglang/agent_cache/results/campaign16/driver_*.log | tail -2'
```
-> prints the stage lines so far.

Stop cleanly (campaign 15's sequence): `pkill -f '^bash run_campaign14.sh'`, then `pkill -f '^bash run_compare.sh'`,
`pkill -f replay_agentic.py`, then `bash stop_server.sh`, all inside the container. Relaunching the same command
resumes with the arms that have no line in `results/campaign16/manifest.txt`. Do not edit `agent_cache/scripts/` while
it runs; `python/` is safe because the arms import the freeze.

**Expected result:** `results/campaign16/manifest.txt` with four lines `<arm> <boot dir> <client dir>`; each client's
summary line has `errors` 0 and about 765-830 users finished (campaign 14: 765-766).
**If it differs:** `STAGE FAILED <arm>` with the server dead during startup: read the last 20 lines of that boot's
`server.log`; the v2 flags are validated at boot, so a validation message names the flag. A client `errors` count above
0: the server returned 5xx; the boot's `server.log` has the traceback.
**Expected lessons:** this is the measurement that decides §6 of the plan; the smoke cannot stand in for it because the
levers only act under a full read queue.

## H6 Evaluation

**Status:** not started
**Goal:** the numbers the go/no-go table of `ORACLE_V2_PLAN.md` §6 asks for, produced by the campaign-15 scripts plus
the v2 event summary, for every arm and both baselines.
**Runs on:** host, `/home/wanhr/sglang/agent_cache` (CPU; the scripts read the boot directories under `results/`)
**Touches:** writes `results/campaign16/<arm>/` (steady csvs and figures, oracle_eval csvs, timelines, event summaries)
**Takes:** 30 min, GPU idle

```bash
# host: one manifest per arm against the two baselines, as campaign 15 did (manifest_vs_c14.txt); baseline boot dirs from the campaign 14/15 manifests
cd /home/wanhr/sglang/agent_cache && B14=$(grep '^three_tier_wc ' results/campaign14/manifest.txt) && B15=$(grep '^three_tier_wc_oracle ' results/campaign15/manifest.txt) \
  && for A in three_tier_wc_dl three_tier_wc_oracle_v2 three_tier_wc_oracle_v2_nu three_tier_wc_wb; do mkdir -p results/campaign16/$A \
  && { echo "$B14"; echo "$B15"; grep "^$A " results/campaign16/manifest.txt; } > results/campaign16/$A/manifest_vs_baselines.txt; done && cat results/campaign16/three_tier_wc_dl/manifest_vs_baselines.txt
```
-> prints three manifest lines (campaign 14 wc, campaign 15 oracle, the arm).

```bash
# host: steady-state window 2,700-16,800 s per arm (throughput, TTFT, tier shares, Little's law), iostat of all three campaigns concatenated
cd /home/wanhr/sglang/agent_cache && cat results/campaign14/iostat_vdc.log results/campaign15/iostat_vdc.log results/campaign16/iostat_vdc.log > results/campaign16/iostat_all.log \
  && for A in three_tier_wc_dl three_tier_wc_oracle_v2 three_tier_wc_oracle_v2_nu three_tier_wc_wb; do python3 scripts/steady_state.py \
  --manifest results/campaign16/$A/manifest_vs_baselines.txt --results results --window-start 2700 --window-end 16800 \
  --iostat results/campaign16/iostat_all.log --out results/campaign16/$A; done && head -3 results/campaign16/three_tier_wc_oracle_v2_nu/steady.csv | cut -c1-200
```
-> prints `steady.csv` rows for the three arms of that manifest (columns `turns_per_s`, `ttft_p50`, `share_device`,
`share_host`, `share_storage`, ...).

```bash
# host: the oracle's own bookkeeping (plans, staged, late, re-evictions, load error) and the v2 events per arm
cd /home/wanhr/sglang/agent_cache && for A in three_tier_wc_dl three_tier_wc_oracle_v2 three_tier_wc_oracle_v2_nu three_tier_wc_wb; do \
  R=$(grep "^$A " results/campaign16/manifest.txt | cut -d' ' -f2); python3 scripts/oracle_eval.py --run $R --out results/campaign16/$A; \
  python3 scripts/srs_summary.py --run $R > results/campaign16/$A/srs_summary.txt; python3 scripts/srs_summary.py --run $R --window 7200 7320 >> results/campaign16/$A/srs_summary.txt; \
  python3 scripts/oracle_timeline.py --run $R --start 7200 --seconds 120 --out results/campaign16/$A/oracle_timeline_7200_120.png; done; cat results/campaign16/three_tier_wc_oracle_v2/srs_summary.txt
```
-> prints the event summary of the `_oracle_v2` arm (whole run, then the congested two minutes); `oracle_eval.csv`
has the `late`, `staged`, `reevicts`, `load_err_p50_ms` and `slack_p50_ms` columns the table below reads.

```bash
# host: the campaign-15 diagnosis (SSD utilisation, host pool held by reads, reads by case) on each new arm
cd /home/wanhr/sglang/agent_cache && ls frozen/oracle_scripts_staging/c15_diagnosis.py && for A in three_tier_wc_dl three_tier_wc_oracle_v2 three_tier_wc_oracle_v2_nu; do \
  python3 frozen/oracle_scripts_staging/c15_diagnosis.py --manifest results/campaign16/$A/manifest_vs_baselines.txt > results/campaign16/$A/diagnosis.txt 2>&1; head -8 results/campaign16/$A/diagnosis.txt; done
```
-> prints the SSD and host-pool blocks per arm; `c15_diagnosis.py` lives only on the box (it was never committed). It
takes `--baseline <run> <client> <iostat>` and `--oracle <run> <client> <iostat>`, not `--manifest`.

**Expected result:** per arm, `steady.csv`, `steady_paired.csv`, `steady_bins.csv`, `steady.png`, `oracle_eval.csv`,
`srs_summary.txt`, `oracle_timeline_7200_120.png`, `diagnosis.txt`.
**If it differs:** `steady_state.py` complains about the window: the arm stopped before 16,800 s (a crash or a stop);
use `--window-end` at the arm's stop and say so in the README.
**Expected lessons:** none; the lessons are in §H7.

## H7 Decide

**Status:** not started
**Goal:** fill `ORACLE_V2_PLAN.md` §6 with measured values and record what to keep.
**Runs on:** host, editing `ORACLE_V2_PLAN.md` and a new `results/campaign16/README.md` (campaign 15's README is the
template)
**Touches:** the two documents
**Takes:** 1 h, GPU idle

Where each §6 row comes from:

| §6 row | file and field |
|---|---|
| queued pages at the cap vs parked | `srs_summary.txt` window block: `inflight_pages at dispatch` (queued at the thread) against `pinned_tokens` / 64 (parked, landed) |
| re-evictions of landed spans | `oracle_eval.csv` `reevicts` |
| host pool held by queued reads | `diagnosis.txt` "host pool held by in-flight SSD reads" |
| late reads | `oracle_eval.csv` `late` / `actioned`; SSD busy share from `diagnosis.txt` |
| trigger bursts | `oracle_timeline_7200_120.png` (lookups per second) and `srs_summary.txt` `waited_ms` |
| load estimate error | `oracle_eval.csv` `load_err_p50_ms` |
| host-hit share of returns, SSD read volume | `steady.csv` `share_host`, `share_storage`; `diagnosis.txt` read GB/s x window |
| turns/s while congested, users finished | `steady.csv` `turns_per_s`, `finished_at_stop`; `steady_bins.csv` for the congested bins |
| `_wb` admission stalls | `queue_duration=` lines of the `_wb` boot's `server.log` (p99 of the per-request forward wait) |

Then apply the no-go rules written under the table in the plan's §6, in order: `_dl` first (does the cap hold work done
ahead), then `_v2` against `_dl` (do deadlines carry information), then `_nu` against `_v2` (does eviction order pay),
then `_wb` on its own.

**Expected result:** the plan's §6 has a measured column; `results/campaign16/README.md` says which arm is the new
policy of record and why, in the campaign-15 format.
**Expected lessons:** the three answers the design was built to get: whether the read queue can be made a scheduling
decision without memory attached, whether the oracle's deadlines are worth more than FIFO at this load, and whether
next-use eviction moves the host-hit share by what the CPU replay (plan §4.7) promised.

## Traps

- A boot without `ENGINE_PYTHONPATH` runs the working tree; a boot with a freeze that predates v2 is refused by
  `start_server.sh` (it checks for `storage_read_scheduler.py`), so an old freeze cannot silently run v1 under a v2 arm
  name (§H2).
- `ORACLE_HINTS=1` must be on for every arm, baselines included: without hints `next_use` degrades to "ended first, then
  LRU" and the oracle has no plans; campaign 15 already paid for the baselines with hints off, which is why its
  `three_tier_wc` is campaign 14's run (§H5).
- The pin budget is 30 % of the host pool (488,294 tokens at 160 GB; the flag refuses more than 50 %, the prefetch
  cap); a smoke with more `srs_pin_skipped` than `srs_pin` means the budget is too small for the window, not a bug (§H4).
- A `leak` unpin (600 s) in a smoke is a bug; in a campaign a handful can be client disconnects (§H4).
- Do not run two servers: `start_server.sh` refuses when `.current_run` has a live pid (§H4, §H5).

## Layout and hygiene

- Boot directories stay under `agent_cache/results/<UTC>_<arm>_NAT160/` (server.log, server_args.txt, run.txt, the
  client directory); campaign 16's own files under `agent_cache/results/campaign16/`.
- Commit the campaign README, manifests, csvs, txt summaries and pngs; never the server logs or client.jsonl files
  (hundreds of MB), as the earlier campaigns did.
- The frozen engine is not committed (`agent_cache/frozen/` is local to the box).
