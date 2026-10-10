# Campaign 13 — x70 steady state: campaign 12's arrivals, but users keep coming

Requested 2026-10-07: run the four arms at gap x70 with the current arrival pattern, now that the client can keep users
arriving for the whole run. Question asked with it: stop on a maximum timeout or on a fixed number of users?

Driver: `../../scripts/run_campaign13.sh` (resumable). Manifest `manifest.txt` here, driver logs `driver.log` and
`driver_*.log`, SSD samples `iostat_vdc.log`.

## Same as campaign 12 x70

Trace `lmcache_agentic_trace_262k.json`, the same 128 conversations (`OFFSET=32 NCONV=128`, all django swebench), lengths
min(recorded, U[30,50]) (`TURNS_RANGE=30:50 TURNS_SEED=120`), Poisson arrivals at 0.2/s from seed 0, live cap `C=128`,
gaps x70 capped at 1,200 s, `NAT160` pools (L1 natural, L2 160 GB write-through, L3 nixl on `/mnt/ssd`, wiped before each
three-tier arm, cleaner 70/60), Qwen3-30B-A3B-Instruct-2507 bf16 KV, `CTX=131072`, stock template, arms `hbm_host
three_tier_to three_tier_wc hbm_lru` in that order, `K=0 CHECK_IDS=0`. Engine: the frozen copy of `2abe005bb2`
(`ENGINE_PYTHONPATH`), the engine of campaign 12's later arms; HEAD differs only by three uncalled helpers in mem_cache.

## What changes

- **Arrivals never stop** (`ARRIVAL_UNTIL` = the guard). 0.2/s is far above what any arm sustains at x70 (research,
  2026-10-07: about 0.04-0.06 users/s for the host and SSD arms, 0.016 for hbm_lru, when the live set is in the
  recompute or SSD regime), so after the first 128 users (12.3 min, as in campaign 12) the live population holds at
  `C=128` and a new user starts each time one leaves; later arrivals wait in the client. Every arm therefore runs
  saturated with the same 128 live users: a closed population, the stationary version of campaigns 7 and 12. At x70
  128 users hold about 2.3M private tokens, 1.4x L2, so L3 is exercised.
- **Users are reused with salts.** About 400-500 users start per arm, so each of the 128 recorded conversations is
  replayed 3-4 times; a reopened conversation gets a salt in its first user message where it stops sharing a prefix
  with other tasks (verified on the real server: a salted copy shares exactly the 7,808 tokens two different tasks share).
- **Stop rule: `STOP_AFTER_USERS=384` (3 x C), guard `GUARD=43200` s (12 h).** The run stops like the wall cap once 384
  users have finished: running sessions stop at their next turn boundary (in-flight requests complete) and waiting
  users are skipped.

## Why a number of finished users, not a timeout or a number of arrivals

- A fixed number of *arrivals* (campaigns 7-12) ends every arm with a drain: no new users, the population falls from 128
  to 0, and the arm spends its last lifetime in exactly the non-stationary transient a steady-state run is meant to
  avoid (campaign 12's departure phases). Slow arms also run unbounded (hbm_lru: 3.2 h for 128 users in campaign 7).
- A fixed *timeout* keeps the population at 128 to the end, but at x70 the arms' session lifetimes at 128 users differ
  about threefold (estimates: about 1.2 h for the SSD arms, 2.5 h for hbm_host, 3.5 h for hbm_lru), so one wall time gives
  the fast arms many turnovers and the slow arms barely one, or wastes many hours on the fast arms.
- A fixed number of *finished* users with arrivals continuing is a fixed number of population turnovers (C finished
  users = one turnover in a closed population), so every arm gets the same statistical depth: the first turnover is the
  warm-up (the live population moves from fresh users to a stationary mix of early and late turns), the next two are
  measured. The guard only matters if an arm is slower than estimated; it is reported (`stopped_by` in the summary).

Expected duration: about 3.5 h per SSD arm, 7-8 h for hbm_host, 10-12 h for hbm_lru, plus 5 min per boot: about 25 h.

## Analysis plan

- Steady-state window per arm: from the time the 128th user finished (end of the first turnover) to the stop. Report
  TTFT (mean, p50, p90, p99) of returning turns in the window, throughput (turns/s, finished users/s), the live and
  waiting counts, where returning turns found their prefix, recompute share, SSD traffic.
- Same users across arms: the user sequence is deterministic, so arms can be paired on (conv, turn) for turns both
  completed in their windows.
- The tools do not know the window yet: `live_timeline.py` phases assume one arrival burst and the whole-run statistics
  include the warm-up. They will need a window option before this campaign is analysed.

## Divergence from the cookbook

`docs/cookbook/autoregressive/Qwen/Qwen3.mdx` (H200, Qwen3-30B-A3B-Instruct-2507): one GPU, default attention backend,
default context 262K with the tip to use about 128K. This campaign keeps the study's pinned server: `--context-length
131072` (the cookbook's tip), `--attention-backend triton` (every campaign since 7; changing it would break the
comparison with campaigns 7-12), bf16 KV, and the HiCache pools that are the experiment's variable.

## Running it

- Launch (host): `sudo docker start sglang_hicache`, then the `docker exec -d` line in `run_campaign13.sh`.
- `run_compare.sh` first runs the client's `--validate` pre-flight, so a bad knob stops the campaign before any boot.
- Do not edit `agent_cache/scripts/` while it runs. `python/` edits are safe (frozen engine).
- After a host reboot: `sudo docker start sglang_hicache`, then the same launch line; finished arms are kept.
