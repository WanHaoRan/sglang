# Campaign 18 — the original HiCache baseline and oracle v2 + next-use at x70 gaps, on the redesigned client (2026-10-18)

Requested 2026-10-18, after the replay client redesign (`../../REPLAY_REDESIGN.md`, commits `e32458cc9f` and
`593da72038`): "the real evaluation should only take into account of the QPS, gap and concurrency ... Let's rerun the
experiments: 1) gap 70, QPS 0.1, original wc (like campaign 14) + oracle v2 + nu (like campaign 16), 2) gap 10, QPS 0.1,
original wc + oracle v2 (like campaign 17)." This campaign is experiment 1; campaign 19 (`../campaign19/`) is
experiment 2 and starts when this one ends.

Launcher: `frozen/oracle_scripts_staging/c18_c19.sh` (copied here as `launcher.sh`), which runs
`../../scripts/run_compare.sh` once per campaign with the knobs below, logs the SSD with `iostat` (`iostat_vdc.log`),
and keeps this directory's `manifest.txt` (arm, run directory, client directory). Its own log is `launcher.log`; the
compare driver's log is `driver_*.log`. Scripts as launched: `scripts_at_launch.tgz`.

## Arms (run in this order)

| arm | server |
|---|---|
| `three_tier_wc` | the original HiCache, campaign 14's server: L1 natural (668,160 tokens) + L2 160 GB write-through + nixl L3 on `/mnt/ssd`, `wait_complete`, paced storage retries at their defaults (8 attempts, poll interval 8) with the upstream #39283 re-query at admission, LRU eviction, no oracle |
| `three_tier_wc_oracle_v2_nu` | campaign 16's best arm: the above + deadline read order (512-page window), pins (0.3 of the host pool), next-turn prefetch on (paced retries off, poll interval 0), next-use eviction (horizon 300 s) |

Both arms on the frozen engine `agent_cache/frozen/engine_5df71c4bf0_v2/python` (the working tree's `python/` is
identical to it: `git diff 5df71c4bf0 HEAD -- python/` is empty on 2026-10-18), and both with `ORACLE_HINTS=1`, so the
request bodies are identical and only the server configuration differs. The L3 store is wiped before each arm.

## Workload: the redesigned client

`USERS=768 TURNS=50 ARRIVAL=0.1 GAP=70 C=128 SEED=0 CLIENT_TIMEOUT=18000` on the 262K trace
(`traces/lmcache_agentic_trace_262k.json`):

- 768 users, each replaying a different recorded conversation from turn 0, all of its recorded turns (at most 50:
  24,879 turns planned), as a closed loop: send a turn, stream the reply, append it verbatim, sleep the recorded gap x 70
  (at most 1,200 s), send the next turn. Thinking off, `temperature 0`, `max_tokens` = the recorded reply length.
- Users arrive as a Poisson process at 0.1/s (seed 0), the last at ~7,633 s; the same conversations and arrival times as
  campaigns 14-17 (REPLAY_REDESIGN section 7.1).
- At most 128 users in session (`C=128`). A user holds its slot through every sleep until it ends, so the cap counts
  idle users; arriving users wait for a slot in arrival order, and that wait is outside every TTFT.
- 5 h per arm: after 18,000 s no user starts and no request is sent (`STAGE CAPPED` follows).

Not comparable to campaigns 14-17 (no mid-session start, no 30-50 turn draw, no backlog-driven cap): compare the two
arms of this campaign with each other (REPLAY_REDESIGN section 7.2).

## Pre-flight (2026-10-18)

- The host rebooted 2026-10-17 ~23:05Z; the container was restarted with `docker start`. The box's checkout was
  fast-forwarded to `593da72038` (it lacked the redesign document; the redesign itself, `e32458cc9f`, was already
  there). The container's Python (3.12.3) hands a freed semaphore slot to the first waiter (REPLAY_REDESIGN 8.5).
- Smoke (`results/20261018_001849_three_tier_wc_oracle_v2_nu_NAT160/`): the oracle arm booted on the frozen engine with
  every boot check matching; `TRACE=<262K> USERS=4 TURNS=10 C=4 GAP=1 ORACLE_HINTS=1 TAG=smoke bash start_client.sh` gave
  4 users, 40 turns, 0 errors, not capped, mean reply re-prefill 36 tokens (all returning turns under one 64-token
  page); the server logged 36 `ntp_plan` = 36 `ntp_arrive` = 36 `ntp_admit`, 0 `ntp_expire`, 0 tracebacks. The new
  client's hints reach the v2 server.

## What to expect (REPLAY_REDESIGN section 9.2)

`live` reaches 128 at ~1,500-1,700 s and stays there while `waiting` climbs to ~300-420 by ~7,600 s. Expect
`STAGE CAPPED` for both arms: with zero service time the queue alone leaves 3 users cut at 18,000 s, and at campaign
14's per-turn latency ~60-150 users are cut or never start. With 128 users in session the server carries less load than
campaigns 14-16 (whose backlog cap averaged 171-231 users), so whether either arm storms is open.

## Accepting an arm

An arm counts only with: summary `errors` 0, no `conv_end` with `aborted: true`, no `Traceback` in `server.log`, and the
boot pools `max_total_num_tokens=668160` and `host pool: 1627648 tokens, 160.00 GB`. For `three_tier_wc_oracle_v2_nu`
also: `ntp_plan` = `ntp_arrive` = `ntp_admit` with `ntp_expire` 0 (users cut by the timeout leave plans that never
arrive; count those separately), no `leak` among the unpin reasons (`scripts/srs_summary.py`), and the largest
`ntp_admit wait_ms` below 60,000 (above it, next-use eviction ranked the longest-waiting queued contexts overdue).

## Analysis plan

- `python3 scripts/c16_summary.py --arm "c18 wc x70" campaign18 three_tier_wc --arm "c18 _oracle_v2_nu x70" campaign18
  three_tier_wc_oracle_v2_nu --pair "c18 _oracle_v2_nu x70" "c18 wc x70"`: TTFT on shared turns with intervals that
  resample users (and, as in campaign 17, 10-min time blocks), time until N users finished, history sources,
  recompute, lost after landing.
- Turns completed and users finished within 5 h, users cut or skipped, and the admission wait (`conv_start.queued_s`),
  which a faster server shortens under a fixed C.
- `scripts/c16_figure.py` (the same `--arm` options), `scripts/srs_summary.py`, `scripts/oracle_eval.py`, cleaner times,
  "reissue cap reached" and re-query counts, and the "Decode batch" state per bin.

## Running it

```
docker exec -d sglang_hicache bash -c 'nohup bash /sgl-workspace/sglang/agent_cache/frozen/oracle_scripts_staging/c18_c19.sh >> /sgl-workspace/sglang/agent_cache/results/campaign18/launcher.log 2>&1'
```

- Stop (all four steps; bash defers a SIGTERM until its child exits, and the brackets keep `pkill -f` from matching the
  `bash -c` that runs it): `docker exec sglang_hicache bash -c 'pkill -f "[c]18_c19.sh"; pkill -f "^bash [r]un_compare.sh"; pkill -f "[r]eplay_agentic.py"; cd /sgl-workspace/sglang/agent_cache/scripts && bash stop_server.sh'`.
- Resume (after a stop or a host reboot: `docker start sglang_hicache`, check `findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3`
  prints `/dev/vdc`): rerun the launch command. It skips campaigns whose manifest holds both arms and reruns only the
  missing arms, each from scratch.
- Do not edit the launcher or `agent_cache/scripts/` while it runs. `python/` is safe (frozen engine).
- Takes ~10.2 h: two arms of 5 h plus a ~5-min wipe and boot each.

## Divergence from the cookbook

As campaigns 13-17: `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` recommends one GPU (TP 1 on H200), the default
attention backend and about 128K context; the study keeps its pinned server (`--attention-backend triton`,
`--context-length 131072`, bf16 KV, HiCache pools as the variable) and adds the v2 flags in the oracle arm.
