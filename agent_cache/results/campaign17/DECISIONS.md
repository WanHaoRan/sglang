# Campaign 17 — the original HiCache baseline and oracle v2 + next-use eviction at x10 gaps (2026-10-12)

Requested 2026-10-12, after campaign 16 (`../campaign16/`, `../../ORACLE_FINDINGS.md`): "let's add the baseline without
our design, but the original hicache + retries ... we only do x10, so two runs with x10 gap, baseline + oracle (and nu)".
Campaign 16 ran at x70; this campaign asks whether the design still pays when sessions return 7x sooner.

Driver: `../../scripts/run_campaign14.sh` (resumable), launched with `ARMS`, `GAP=10`, `ORACLE_HINTS=1` and
`ENGINE_PYTHONPATH`. Manifest `manifest.txt`, driver logs `driver.log` and `driver_*.log`, SSD samples `iostat_vdc.log`,
scripts as launched in `scripts_at_launch.tgz`.

## Arms (run in this order)

| arm | server |
|---|---|
| `three_tier_wc` | the original HiCache, exactly campaign 14's baseline: L1 natural (668,160 tokens) + L2 160 GB write-through + nixl L3 on `/mnt/ssd`, `wait_complete`, paced storage retries at their defaults (max attempts 8, poll interval 8) and the upstream #39283 re-query at admission, LRU eviction, no oracle |
| `three_tier_wc_oracle_v2_nu` | campaign 16's best arm: the above + `--hicache-storage-read-order deadline` (window 512 pages) + `--hicache-storage-prefetch-pin-fraction 0.3` + `--hicache-storage-next-turn-prefetch on` (paced retries off: poll interval 0) + `--radix-eviction-policy next_use` (horizon 300 s) |

Both arms on the frozen engine `agent_cache/frozen/engine_5df71c4bf0_v2/python` and both with the client's next-turn
hints, so the client side is identical and the server configuration is the only difference.

## Workload: campaign 14's, with x10 gaps

768 users, each replaying one session of the 262K agentic trace once, fresh order (seed 0), Poisson arrivals at 0.1/s,
stationary start, user cap following the server backlog (target 32, from 128), sessions min(recorded, U[30,50]) turns
(seed 120), gaps = recorded gap **x 10** capped at 1,200 s (campaigns 14-16: x 70), 5 h wall cap per arm, NAT160 pools
(L3 wiped at each arm start, cleaner 70/60), Qwen3-30B-A3B-Instruct-2507 bf16 KV, TP 1, context 131,072.

The stationary start places the first 128 users at time-uniform points of their sessions, and the time a session takes
depends on its gaps. So at x10 a slightly different set of first requests is sent: 122 users start mid-session and
20,293 turns are planned, against 125 and 20,232 at x70. Both arms of this campaign send the same turns; their turns do
not pair one-to-one with campaigns 14-16.

## Changes before launch

- `run_campaign14.sh` passed `GAP=70` as a literal on the `run_compare.sh` call, so an exported `GAP` was ignored. It now
  takes `GAP` from the environment (default 70, so campaigns 14-16 relaunch unchanged) and prints it in the LAUNCH line.
  Checked with the client's own pre-flight (`VALIDATE_ONLY=1 bash start_client.sh` with the driver's settings, no
  server): `GAP=10` prints "valid: 768 users (0 salted), 20293 turns planned" and "122 of the first 128 users start
  mid-session"; `GAP=70` prints 20,232 and 125, as campaigns 14-16.
- Nothing else. The host rebooted at 18:40Z; the container was restarted with `docker start`.

## Pre-flight checks (2026-10-12, read-only)

- **Baseline parity.** `three_tier_wc` on the v2 engine resolves to campaign 14's server: the same launch line, and of
  the 517 recorded server arguments only five differ (the per-boot random seed, the path of `l3_extra.json` whose contents
  are identical, and three new keys at their defaults: read order `fifo`, window 512, pin fraction 0.0). Every v2 code
  path is behind a flag, an empty pin budget or an object that is never created with the oracle off; the only addition
  that runs is one `srs_rate` log line per 64 SSD reads. Hints are decoded onto the request and never read. Check after
  the boot: `server_args.txt` differs from `../20261008_204128_three_tier_wc_NAT160/server_args.txt` in exactly these keys.
- **Short gaps in the v2 engine.** No new failure path: the commit hook runs before the reply's last chunk is streamed,
  so a plan always exists before its next request arrives; pins, plans and registries are released on every exit path
  and are bounded. One dormant behaviour: next-use eviction ranks a queued request's context "overdue" after 60 s in the
  queue and evicts it early. It never fired in campaign 16 (longest TTFT 38 s). Check the longest `ntp_admit wait_ms`
  after the run.

## What to expect

| | x70 (campaigns 14-16) | x10 (this campaign) |
|---|---|---|
| gap before a returning turn, p50 / mean | 49 / 74 s | 7.1 / 14.1 s |
| gaps over 300 s | 4.3 % | 0.6 % |
| a session's total idle time, mean | 1,893 s | 364 s |
| live users (0.1/s x session length) | ~200-300, held by the cap | ~50 (light load, 3-5 s per turn) to ~140 (campaign 14's congested service times) |

- **The GPU, not the SSD, is likely the limit.** Users 128-767 offer 18,362 turns and 3.71M output tokens between
  1,473 s and 7,634 s: 2.98 turns/s or 602 decode tokens/s, the same at every gap scale. With its KV pool full the H200
  decoded a median 505-525 tokens/s in campaign 16 (`Decode batch` lines at 28-43 running requests), and no 600-s bin of
  campaign 16 exceeded 2.58 turns/s. So expect congestion from about 30 min until arrivals end at ~2.1 h. The pre-flight
  estimates disagreed (from ~50 live users and an idle cap at light-load service times to ~140 live users and a binding
  cap at congested ones); the demand-versus-ceiling numbers favour congestion. The cap only moves once live plus waiting
  users reach 128 (at x70 it rose from 128 to 251). Live KV (1.1-3.2M tokens) exceeds L1 and comes close to the host
  pool, so the SSD is still used, but far less than at x70 (an offline replay sends 1-10 % of returns to it, against
  62-70 % at x70). A higher TTFT than campaign 16's x70 runs is the load, not a regression.
- **The baseline's risk is campaign 11's failure, not campaign 14's.** At x10 a queued request's context ages under LRU
  like an idle one and can be evicted while it waits; its paced polls of the never-stored new tail use up the 8 lookup
  attempts within seconds, after which the #39283 re-query cannot run and the history is recomputed. Campaigns 11 and 12
  (x10, a different 128-session workload) logged 2,162 and 1,541 "reissue cap reached" and collapsed; lost-after-landing,
  campaign 14's failure, was only 2-4 % of their recompute. Expect the baseline to recompute once queue waits pass about
  10 s.
- **The oracle will be mostly idle.** Staging needs at least 256 tokens missing from both GPU and host memory, which at
  x10 is rare; next-use eviction mostly takes ended sessions first, as LRU does. So the oracle arm's difference at x10
  may come mostly from its pins and from paced retries being off (see the confound below).
- **Run length.** One user (u722: arrives at 7,245 s, five 1,200-s gaps) cannot finish before ~14,180 s, so each arm
  runs ~14,300-15,800 s (oracle arm) or up to 18,000 s (baseline, if it congests). About 8.5-10 h for both, with boots.

## Confound to report with the results

The oracle arm also turns off the paced storage retries (poll interval 0, forced by the engine with the deadline order
and the oracle on), and at x10 those retries are the documented cause of the baseline's recompute (campaign 11: 93-94 %
of its remaining recomputes had used up their budget on them). A gain at x10 therefore mixes that hint-free change with
the hint-driven levers. A third arm would separate them: `three_tier_wc` with `--hicache-storage-prefetch-retry-poll-interval 0`
(needs a new case in `start_server.sh`; it can be appended to this campaign later, since the driver runs only the arms
missing from `manifest.txt`). Not run unless requested.

## Analysis plan

- Pair the two arms on shared (user, turn) pairs: mean / p90 / p99 TTFT with intervals that resample users.
- Time until 600 / 700 / 750 users finished (not all 768: u722 sets the end), turns/s while congested, users waiting.
- History served by GPU / host / SSD / recomputed, recomputed tokens per returning turn, turns that lost a read after
  landing, "reissue cap reached" and "reissuing storage lookup" counts per arm: these name the mechanism.
- GPU state per 10-min bin from the server's "Decode batch" lines: KV usage >= 0.97 with >= 450 decode tokens/s is the
  healthy GPU limit; usage >= 0.97 with <= 250 tokens/s is a recompute storm.
- v2 events (`scripts/srs_summary.py`), oracle bookkeeping (`scripts/oracle_eval.py`), cleaner timing. The 2,700-16,800 s
  window of campaigns 14-16 does not carry over (the runs end earlier); `scripts/c16_summary.py` and `c16_figure.py` are
  wired to campaign 16's runs and need generalising before use here.

## Running it

```
docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && mkdir -p ../results/campaign17 && ARMS="three_tier_wc three_tier_wc_oracle_v2_nu" GAP=10 ORACLE_HINTS=1 ENGINE_PYTHONPATH=$(ls -d /sgl-workspace/sglang/agent_cache/frozen/engine_5df71c4bf0_v2/python) nohup bash run_campaign14.sh ../results/campaign17 >> ../results/campaign17/driver.log 2>&1'
```

- Stop: all four steps are needed (bash defers a SIGTERM until its running child exits), and the bracketed patterns keep
  `pkill -f` from killing the `bash -c` that runs it:
  `docker exec sglang_hicache bash -c 'pkill -f "^bash [r]un_campaign14.sh"; pkill -f "^bash [r]un_compare.sh"; pkill -f "[r]eplay_agentic.py"; cd /sgl-workspace/sglang/agent_cache/scripts && bash stop_server.sh; pgrep -af "[i]ostat -dxt" || echo no iostat left'`.
- Resume (after a stop or a host reboot: `sudo docker start sglang_hicache`, then check that
  `docker exec sglang_hicache findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3` prints `/dev/vdc`): rerun the command above
  unchanged. It must carry `GAP=10`: the driver resumes by arm name, so a relaunch without it would run the remaining arm
  at x70 into this campaign. The new LAUNCH line must say "gaps x10". The interrupted arm restarts from scratch; finished
  arms are kept.
- Do not edit `agent_cache/scripts/` while it runs: bash reads `run_campaign14.sh` as it executes, and every arm
  re-reads the others. `python/` is safe (frozen engine).

## Accepting an arm

`run_compare.sh` records an arm as done whenever the client wrote its summary, which it also does after a server crash
(every session then aborts). An arm counts only with: summary `errors` 0, no `conv_end` with `aborted: true`, no
`Traceback` in `server.log`, and the boot pools `max_total_num_tokens=668160` and `host pool: 1627648 tokens, 160.00 GB`.
For `three_tier_wc_oracle_v2_nu` also: `ntp_plan` = `ntp_arrive` = `ntp_admit` (19,525 if every user finishes) with
`ntp_expire` 0, no `leak` among the unpin reasons (`scripts/srs_summary.py`), and the largest `ntp_admit wait_ms` below
60,000 (above it, next-use eviction ranked the longest-waiting queued contexts "overdue" and evicted them early).

## Divergence from the cookbook

As campaigns 13-16: `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` recommends one GPU (TP 1 on H200), the default attention
backend and about 128K context; the study keeps its pinned server (`--attention-backend triton`, `--context-length
131072`, bf16 KV, HiCache pools as the variable).
