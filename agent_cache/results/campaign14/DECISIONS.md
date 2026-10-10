# Campaign 14 — three_tier_wc and three_tier_to with a capacity-driven user cap: do the light phases go away?

Requested 2026-10-07: after the campaign-13 light-region investigation the user asked to "cap by the maximum capacity of
the gpu", then to "fix the light region issue and rerun with l3_wc case only to see if the proposed change resolves this
issue". Campaign 13's hbm_lru arm was cut by a host reboot (23:08Z, 187 of 384 users finished) and is not rerun.

Driver: `../../scripts/run_campaign14.sh` (resumable, like campaign 13's). Manifest `manifest.txt` here, driver logs
`driver.log` and `driver_*.log`, SSD samples `iostat_vdc.log`.

## What campaign 13 showed (three_tier_wc, 128 live users)

- The server alternated between two states with the same request stream (about 107 returning requests per minute and the
  same live working set in both):
  - SSD-bound: reads pinned at about 1.19M tokens/min (about 2 GB/s), about 18 requests waiting on L3 prefetches,
    TTFT median about 14 s;
  - light: most returns 40-60 s after the previous turn hit host memory, queue about 0.3, TTFT median about 0.3 s, with
    the GPU's KV pool about 55% used and the SSD about 24% busy, so the server had spare capacity.
- Light time was 17-30% of the measured window, in phases of 4-15 min that recurred every 35-42 min.
- 128 users hold the live private working set at 0.99 x (L1 + L2): the operating point sits on the boundary between the
  two states.
- The fixed replay order (user i replays conversation i mod 128 with the same gaps) makes the offered load repeat every
  turnover of 128 users (about 42 min). Its dips line up with the light phases (r 0.85-0.87; no shuffled order gets
  close).
- Only the warm-up and the first light phase after it came from the first 128 users finishing together.

The investigation's notes and numbers are in the journal of workflow wf_fca5bc2a-d0b (its scratch outputs were lost with
/tmp in the reboot).

## What changes (client options, all off by default: campaigns 7-13 replay as before)

1. **The cap follows the server backlog** (`--cap-mode backlog`). The backlog is the number of requests sent and still
   waiting for their first token, measured by the client. Every 120 s the cap moves by round(0.5 x (32 - mean backlog)),
   at most 8 users per update, within 16-512, starting at 128. It holds while it limits nobody (the first users are still
   arriving), falls to no more than 8 below the live users and rises to no more than 8 above them. A waiting user starts
   when fewer than the cap are live and at least 5 s after the previous start. In a light phase the backlog falls, so the
   cap rises and more users are admitted until the server is saturated again.
2. **Every user a new conversation** (`NCONV=768 OFFSET=0 --order fresh`, seed 0): users 0-767 replay the trace's 768
   conversations (665 swebench, 94 gaia, 9 wildclaw; 12 system prompts) in a seeded order, each once, with no salted
   copies; no user arrives after the 768th. Campaign 13 replayed 128 swebench conversations about four times each, in a
   fixed order whose load repeated every turnover. Every block of 128 users now mixes the three sources; the largest
   prompt is 98K tokens (validate: 768 users, 0 salted, 20,232 turns, no context action).
3. **Stationary start** (`--start-phase stationary`, seed 0, service 20 s): the first 128 users start mid-session at a
   time-uniform point of their sessions. Their first requests carry a synthetic history (about 3.1M tokens in all, against
   1.2M at turn 0), spread over the first 21 min by the 0.1/s arrivals. These first requests are flagged `mid_start` and
   are not counted as returning turns.

Analysis: one predicate for first and returning requests (`vs_baseline.is_first` / `is_returning`) at every site;
`steady_state.py --window-start`; the cap and the measured backlog in `steady_state.py` and `steady_figures.py`;
`steady_figures.py --label/--setting/--series-label/--series-short` for this campaign's names.

## Same as campaign 13

Trace `lmcache_agentic_trace_262k.json` (now all 768 conversations), lengths min(recorded,
U[30,50]) (`TURNS_RANGE=30:50 TURNS_SEED=120`), gaps x70 capped at 1,200 s, Poisson arrivals at 0.1/s (campaign 13: 0.2/s) from seed 0 (arrivals end with
the 768th user), `NAT160` pools (L1 natural, L2 160 GB write-through, L3 nixl on `/mnt/ssd`, wiped before the arm, cleaner
70/60), Qwen3-30B-A3B-Instruct-2507 bf16 KV, `CTX=131072`, stock template, `K=0 CHECK_IDS=0`, the frozen engine copy of
`2abe005bb2` (`ENGINE_PYTHONPATH`).

## Stop: 5 h of wall time instead of 384 finished users

Campaign 13's three_tier_wc window was 84.6 min, about two light cycles. A fixed `RUN_SECONDS=18000` leaves about 4 h
after the warm-up, room for about six cycles if the light phases persisted. With 768 users in all, the waiting
reservoir empties once the 768th user is admitted (estimate: about 4 h at campaign 13's three_tier_wc rate); after that
the population drains, so the measured window ends at the last admission or at the wall cap, whichever is first.

## What would show the change worked

- Time in the light state in the measured window (minutes with a queue under 1 and a TTFT median under 2 s) near 0,
  against 17-30% in campaign 13.
- No 36-44-min periodicity in the storage share and TTFT (campaign 13: autocorrelation 0.64-0.71).
- The cap settles: cap records show the backlog held near 32 without a sustained oscillation; report the live users,
  throughput and TTFT at that operating point.
- SSD reads stay near the ceiling (about 1.19M tokens/min) and the 40-60 s cohort's host-hit share stays low.

Three changes at once: a clean result says the combination works, not which part did it. The cap is the one expected to
matter (it moves the operating point off the boundary); the shuffle and the stationary start remove the client's own
timing and the first-generation wave.

## Divergence from the cookbook

As campaign 13: `docs/cookbook/autoregressive/Qwen/Qwen3.mdx` recommends one GPU, the default attention backend and about
128K context; this run keeps the study's pinned server (`--attention-backend triton`, `--context-length 131072`, bf16 KV,
HiCache pools as the variable).

## Running it

- Launch (host): `sudo docker start sglang_hicache`, then the `docker exec -d` line at the top of `run_campaign14.sh`.
- `run_compare.sh` runs the client's `--validate` pre-flight first, so a bad knob stops before the boot.
- Analysis: `steady_state.py --manifest <compare dir>/manifest.txt --iostat ../results/campaign14/iostat_vdc.log`;
  `steady_figures.py ... --label "Campaign 14" --setting "x70 gaps, cap by backlog" --series-label "campaign 14 x70"
  --series-short "campaign 14"`; `timeline.py --tier-rule private`.

## Arrival rate and arms (2026-10-08)

The user lowered the arrival rate to 0.1 users/s (360/h) and asked for three_tier_wc and three_tier_to. 0.1/s is still
above both arms' campaign-13 finishing rates (about 182/h and 122/h at 128 users), so users keep waiting to be admitted
while arrivals last; all 768 have arrived after about 128 min. Campaign 13 used 0.2/s, so the warm-up differs from it.
