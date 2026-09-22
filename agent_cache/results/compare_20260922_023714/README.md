# HiCache tiers under a 128-way agentic replay on one H200 (2026-09-22)

Four arms, identical pools and identical client workload, one fresh boot each (`scripts/run_compare.sh`, config in
`config.txt`, sizing chain and run notes in `DECISIONS.md`):

| arm | server | pools |
|---|---|---|
| `hbm_lru` | no HiCache, radix LRU | L1 668,160 tokens (natural pool, bf16 KV = 65.7 GB) |
| `hbm_host` | HiCache, host tier, write-through | L1 668,160 + L2 160 GB (1,627,648 tokens) |
| `three_tier_to` | HiCache, host + nixl L3 on the attached SSD, `timeout` prefetch | L1 + L2 + L3 (2.27 TiB ext4 on `/dev/vdc`, 1.88 GiB/s; wiped before boot; cleaner 70/60 %, never reached) |
| `three_tier_wc` | same, `wait_complete` prefetch | same |

Client: LMCache swebench trace re-converted at a 262K context cut (`traces/lmcache_agentic_trace_262k.json`), 128 conversations
all live (`NCONV=128 C=128`, `OFFSET=32`), up to 40 turns each = 4,676 turns, gaps x70 capped at 1,200 s, no controls, thinking
off, the model's stock chat template (re-renders history byte-identically, so no replay template). Qwen3-30B-A3B-Instruct-2507,
bf16 weights (57 GB) and bf16 KV (98,304 B/token), H200 143 GB, 196 GB host RAM. Final context p50 35.6K tokens, of which
7,808 is the system prompt shared by every conversation: private working set 3.55M tokens against 1.63M of unique L1+L2 capacity
(write-through keeps L2 a superset of L1, so the two tiers do not add). Files: `timeline_<arm>.png`, `compare.png`,
`compare.csv`, `turns_<arm>.csv`, `events_<arm>.csv`, `iostat_vdc.log` (device-side SSD samples, 10 s, from 04:43Z).
Regenerate: `python3 scripts/timeline.py --manifest results/compare_20260922_023714/manifest.txt`.

## Result: the SSD tier turns a 3-minute queue into a 3-second restore

| | hbm_lru | hbm_host | three_tier_to | three_tier_wc |
|---|---|---|---|---|
| turns completed / errors | 4,676 / 0 | 4,676 / 0 | 4,676 / 0 | 4,635 / 2 (a) |
| client wall (s) | 11,429 | 9,155 | 5,327 | 5,148 |
| returning-turn TTFT mean (s) | 97.0 | 61.0 | 1.94 | 1.48 |
| returning-turn TTFT p50 / p90 / p99 (s) | 92.9 / 201 / 224 | 0.98 / 184 / 210 | 0.32 / 6.8 / 17.5 | 0.29 / 4.5 / 16.8 |
| worst 10-min window, p50 (s) | 215 | 178 | 2.8 | 0.6 |
| turns done at 95 % (s) | 8,658 | 6,309 | 2,620 | 2,539 |
| returning turns: device / host / storage / recompute | 985 / 0 / 0 / 3,563 | 1,274 / 1,553 / 0 / 1,721 | 1,669 / 1,921 / 890 / 68 | 1,780 / 1,890 / 837 / 0 |
| uncached tokens per returning turn (mean) | 15,427 | 8,663 | 924 | 558 |
| storage restore TTFT p50 / p90 (s), tokens per restore | – | – | 2.58 / 13.8, 19.8K | 3.34 / 15.1, 20.0K |
| tokens: d2h / h2s / h2d / s2h | – | 40.5M / – / 22.4M / – | 5.27M / 5.27M / 47.8M / 18.1M | 3.57M / 3.57M / 46.4M / 17.2M |
| tokens evicted from device / host | 70.6M / – | 62.2M / 38.8M | 52.4M / 21.8M | 49.3M / 19.1M |
| peak scheduler queue | 69 | 70 | 73 | 79 |
| SSD reads while active, mean / peak (MiB/s); samples at >= 95 % util | – | – | 1,036 / 2,076; 28 % | 999 / 2,046; 26 % |

(a) two conversations aborted by transport resets, see deviations.

Every arm serves the first ~12 minutes identically (device and host hits at 0.2-0.3 s) until the unique live set passes L2's
1.63M tokens. From there the arms diverge: `hbm_host` recomputes every evicted return (~28K tokens, 5-7 s of GPU each), the
closed loop saturates the GPU at 0.2-0.3 turns/s, and TTFT sits at ~180 s p50 for 40 minutes with 55-86 conversations
waiting; the two SSD arms restore the same returns from L3 at 2.6-3.3 s p50 with the queue at 0-7, finish the trace in 58 %
of the wall time, and prefill 9-16x fewer tokens (4.2M / 2.5M vs 39.4M). `hbm_lru`, with nothing behind the 668K-token
device pool, recomputes 78 % of its returns (3,563 of 4,548) at 93 s p50, holds 165-215 s p50 for two hours with 60-90 of the 128 conversations waiting, and needs 11,429 s — 2.1x the SSD arms — for the same trace.

## Why the tiers work here and not in C18 (`compare_20260918_final`)

1. **L1 admits the in-flight set.** 668K tokens hold ~15 requests of 35K at once against 12-42 running; C18's 65K pool
   held 3-5 of 32 and admission queueing (not restores) set TTFT in every arm.
2. **L2 is a real second tier but still too small.** 160 GB holds 46 % of the private working set, so the SSD is reached
   (1.9M tokens beyond L2 at the end of the run) — and because write-through mirrors L1 into L2, the unique capacity of the
   two tiers is L2 alone, 1.63M, not L1+L2 = 2.30M (host evictions began at 02:48Z with the live set at ~1.56M).
3. **Prefetch admission never bound.** The budget is half the host pool = 814K tokens in flight (`cache_controller.py:581`);
   at 12-30 concurrent restores of ~20K it is used to ~60 %. 960 (`timeout`) and 837 (`wait_complete`) prefetches ran, all
   completed in full, none rate-limited, none timed out, against C18's 145 of 354 rate-limited and 167 of 209 empty lookups.
4. **No write amplification.** C18 re-inserted 2.55M tokens of KV for 480K of content because every recompute was backed up
   again; here `hbm_host` still writes 40.5M tokens d2h (its recomputes), but the SSD arms write 3.6-5.3M for 3.6M of content —
   the write-back (81-171 MiB/s) no longer starves the reads.
5. **The SSD is now the wall.** Restores run the device at 1.0 GiB/s mean and 2.0 GiB/s peak, i.e. the measured 1.88 GiB/s
   ceiling, in the overflow phase; the storage p90 of 14-15 s is the queue for the disk, not the GPU. `wait_complete` removes
   the `timeout` arm's 68 post-warm-up recomputes (mean TTFT 1.48 vs 1.94 s) at the price of a slower individual restore
   (3.34 vs 2.58 s p50), because every request now waits for its whole read under that contention.

## Deviations from the plan (all recorded in `DECISIONS.md` as they happened)

- **The 9,000 s per-arm cap did not act** (`timeout --foreground` signals only the `bash start_client.sh` child, which defers
  SIGINT while its pipeline runs). `hbm_host`'s `STAGE CAPPED` line is spurious (all 4,676 turns completed, 9,155 s);
  `hbm_lru` ran uncapped (190 min). Fix for the driver: see RUNBOOK 7.x.
- **`three_tier_wc` lost two conversations** to aiohttp keep-alive resets (conv 6 at turn 6, conv 11 at turn 15; uvicorn
  closes idle pooled connections after `SGLANG_TIMEOUT_KEEP_ALIVE=5` s and `replay_agentic.py` aborts the conversation on
  any exception). 41 of 4,676 turns; the server log is clean at both times. For paired statistics drop those turns from
  every arm.
- **`STAGE FAILED three_tier_wc (client)`** in the driver log is the per-turn table at the end of `start_client.sh` crashing
  on the two error records after the client had finished; the manifest line was added by hand and the table fixed.
- The device-side `iostat_vdc.log` starts at 04:43Z, during arm 1 (which has no SSD traffic), so it covers all L3 activity.
