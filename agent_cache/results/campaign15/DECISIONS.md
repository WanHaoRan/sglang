# Campaign 15 — the oracle next-turn prefetch on campaign 14's workload, `wait_complete` and `timeout`

Requested 2026-10-09: "compare the oracle with the original three_tier using the campaign 14 workload. With both wc and
to policy." The oracle is `agent_cache/ORACLE_PLAN.md` (steps 1-9), switched on with our flag
`--hicache-storage-next-turn-prefetch on` (added in `2abe005bb2`).

Driver: `../../scripts/run_campaign14.sh` unchanged (resumable), with `ARMS`, `ORACLE_HINTS=1` and `ENGINE_PYTHONPATH`
set at launch. Manifest `manifest.txt` here, driver logs `driver.log` and `driver_*.log`, SSD samples `iostat_vdc.log`.

## Arms

| arm | server | run in |
|---|---|---|
| `three_tier_wc_oracle` | `three_tier_wc` + `--hicache-storage-next-turn-prefetch on` | this campaign |
| `three_tier_to_oracle` | `three_tier_to` + `--hicache-storage-next-turn-prefetch on` (new arm, added for this campaign) | this campaign |
| `three_tier_wc` | L1 natural + L2 160 GB write-through + nixl L3 on `/mnt/ssd` (wiped before the arm), `wait_complete` | campaign 14 (`../campaign14/manifest.txt`) |
| `three_tier_to` | as `three_tier_wc` with the `timeout` prefetch policy | campaign 14 |

The oracle arms also pin `--hicache-storage-prefetch-retry-poll-interval 8 --hicache-storage-prefetch-retry-max-attempts
8`, which are the engine defaults the baselines ran with (campaign 14's `three_tier_wc` logged 8 and 8), so each oracle
arm differs from its baseline in the next-turn prefetch, plus the engine copy and the client hints below.

## Same as campaign 14

Every client knob of `run_campaign14.sh` (see `../campaign14/DECISIONS.md`): 768 fresh conversations (`ORDER=fresh`),
Poisson arrivals at 0.1/s, the user cap following the server backlog (target 32, every 120 s, gain 0.5, step 8, 16-512,
starting at 128), stationary start, turns min(recorded, U[30,50]), gaps x70 capped at 1,200 s, 5 h of wall time per arm
(`RUN_SECONDS=18000`). Server: `NAT160` pools, Qwen3-30B-A3B-Instruct-2507 bf16 KV, context 131,072, triton attention,
L3 cleaner 70/60. Campaign 14 concluded that target 32 is too deep and the controller too fast; the knobs are kept so
that this campaign's baselines can be checked against campaign 14's runs.

## What differs from campaign 14

- **Engine:** the frozen copy `agent_cache/frozen/engine_294c82b7fc_ntpfix/python`: `294c82b7fc` (oracle steps 1-7) plus
  the fresh re-eviction probe in `next_turn_prefetch.py`, uncommitted at the freeze (`worktree.diff` in the copy).
  Campaign 14 ran `engine_2abe005bb2`. The added code (`git diff --stat 2abe005bb2 294c82b7fc -- python/`: the
  prefetcher, its estimators, the read-only residency probe, and Scheduler and `cache_finished_req` hooks) is inactive
  with the flag off.
- **Client:** `ORACLE_HINTS=1`: each turn carries a `sglang.next_turn` kv_hint with the session's next rid and its
  recorded gap. Campaign 14's client sent none; otherwise the requests are the same. With the prefetcher off the server
  ignores the hints (smoke A, `../compare_20261009_070112/`: hints with the prefetcher off or observing leave serving
  unchanged).

## Baselines: campaign 14's runs, not rerun

The launch at 16:41Z also had the two baselines, to run them on the same engine and client. At 16:47Z the user dropped
them to save about 10 h: the oracle arms are compared with campaign 14's `three_tier_wc` and `three_tier_to`. The
engine and client differences above are inactive for a baseline, but each side is a single run: in smoke C two runs of
identical code differed by 0.98 s mean TTFT on the same turns at 32 sessions (`../compare_20261009_082335/README.md`,
"Reruns"), and campaign 14's arms moved through congestion and light phases. The run-to-run spread at this scale is not
measured, so differences of that order are not evidence.

## The `timeout` arms: a known gap

Plan decision 3 kept v1 to `wait_complete`. Under `timeout` a prefetch is cut at admission once `1.0 + 0.015625 x pages`
seconds have passed since the read started (`prefetch_timeout_base 1.0`, `prefetch_timeout_per_ki_token 0.25`). For an
oracle read that clock starts at staging (case 1) or at the re-fetch (case 2), before the turn's own arrival or
admission, so a read still running when its turn is first considered for admission can be cut early and load only part
of its pages. Smoke C's reads landed a median 1.0 s before admission, so the cut should be rare, but under campaign 14's
SSD-bound congestion reads queue behind each other. Report the truncated oracle reads separately.

## What to look at

- At the operating point the controller settles on: returning-turn TTFT, queue wait, storage-hit and recompute shares,
  live users, users per hour and turns per second. The oracle may let the controller hold more users at the same
  backlog, so throughput and the cap are results, not fixed inputs.
- The oracle's own record (`oracle_eval.py`): arrival error, wait estimators, late reads, staged reads never used,
  re-fetched tokens against tokens credited from the SSD, and re-evictions (the cascade risk: a re-fetched context
  evicted by another turn's read just before admission, `../compare_20261009_082335/README.md`).
- SSD reads (`iostat_vdc.log`): campaign 14's `three_tier_wc` was SSD-bound, so extra reads compete for the same ~2
  GB/s.
- Each oracle arm against its campaign-14 baseline: `manifest.txt` here plus `../campaign14/manifest.txt`.

## Divergence from the cookbook

As campaigns 13 and 14: `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` recommends one GPU (TP 1 on H200), the default
attention backend and about 128K context; the study keeps its pinned server (`--attention-backend triton`,
`--context-length 131072`, bf16 KV, HiCache pools as the variable) and adds the oracle flag.

## Running it

First launch (16:40:55Z), with the baselines:

```
docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign15 && ARMS="three_tier_wc_oracle three_tier_wc three_tier_to_oracle three_tier_to" ORACLE_HINTS=1 ENGINE_PYTHONPATH=/sgl-workspace/sglang/agent_cache/frozen/engine_294c82b7fc_ntpfix/python nohup bash run_campaign14.sh ../results/campaign15 >> ../results/campaign15/driver.log 2>&1'
```

Cut (16:47Z): `frozen/oracle_scripts_staging/c15_cut_baselines.sh`, running in the container (log
`cut_baselines.log`), waits for `three_tier_wc_oracle` to end, stops the driver chain before `three_tier_wc` boots and
relaunches it with the oracle arms only. `three_tier_wc_oracle` is merged from its compare dir and only
`three_tier_to_oracle` runs. A `three_tier_wc` run dir from the stopped boot may exist; it is in no manifest. To resume
after a reboot, relaunch with the oracle arms:

```
docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && ARMS="three_tier_wc_oracle three_tier_to_oracle" ORACLE_HINTS=1 ENGINE_PYTHONPATH=/sgl-workspace/sglang/agent_cache/frozen/engine_294c82b7fc_ntpfix/python nohup bash run_campaign14.sh ../results/campaign15 >> ../results/campaign15/driver.log 2>&1'
```

- Stop: `pkill -f '^bash run_campaign14.sh'`, then `pkill -f '^bash run_compare.sh'`, `pkill -f replay_agentic.py` and
  `bash stop_server.sh`. Do not edit `agent_cache/scripts/` while it runs; `python/` is safe (frozen engine).
- About 5.1 h per arm: `three_tier_wc_oracle` ends about 21:45Z, `three_tier_to_oracle` about 02:50Z on 2026-10-10.
- `scripts_at_launch.tgz`: `agent_cache/scripts/` as launched (uncommitted: the new arm, the plot colors, the step-8/9
  analysis and client changes).
- Analysis: `steady_state.py --manifest manifest.txt --iostat iostat_vdc.log`, `steady_figures.py`,
  `timeline.py --tier-rule private`, `oracle_eval.py --manifest manifest.txt --out .`,
  `frozen/oracle_scripts_staging/truncated_reads.py <run dir>`.

## Pre-flight

Both on the frozen engine above, logs in `../oracle_smokes_1009_0701/`.

- **S15** (`../compare_20261009_155943/`, `frozen/oracle_scripts_staging/oracle_smoke_c14.sh`): campaign 14's knobs for
  1,500 s on `three_tier_to_oracle`. All boot checks pass (`timeout`, oracle on, banner 1, retry cap 8, L1 668,160 and
  host 1,627,648 tokens as campaign 14). 968 turns, no client errors or retries, no tracebacks. 851 plans, 844 arrived
  and admitted (the rest still pending at the cap), none expired: the hints match under the stationary start, the fresh
  order and the backlog cap. Arrival error p50 61 ms, p90 259 ms. The host pool first filled after about 17 min
  (campaign 14: the same), so the oracle acted only in the last minutes: 5 staged reads, all used, none cut by the
  timeout.
- **C5** (`../compare_20261009_162731/`, `frozen/oracle_scripts_staging/oracle_smoke_c5.sh`): smoke C's dense small-pool
  config (32 x 12, L1 65,536 tokens, host 12 GB) on `three_tier_to_oracle`, to exercise re-fetches under `timeout`. 163
  re-fetches and 2 staged reads; 11 of 165 late (the `wait_complete` runs of this config: 1-3). The timeout cut 19 of
  211 oracle reads (82K tokens short) and 15 of 119 of the requests' own reads
  (`frozen/oracle_scripts_staging/truncated_reads.py`). TTFT mean 13.42 s (the four `wait_complete` oracle runs of this
  config: 13.1-14.1 s; no `timeout` baseline was run). 77 re-evictions, none of the stale kind; no client errors, no
  tracebacks.
