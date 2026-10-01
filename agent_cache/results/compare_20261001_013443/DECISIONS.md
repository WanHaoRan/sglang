# Campaign 11 — campaign 8 (GAP=10, dense) rerun on the merged engine with upstream #39283

Started 2026-10-01T01:34:43Z, driver log `../compare_driver_c11.log`. Same workload and arms as campaign 8
(`../compare_20260922_165655/`): Qwen3-30B-A3B-Instruct-2507, bf16 KV, L1 natural, L2 160 GB write-through, L3 nixl on
/mnt/ssd; 128 live sessions, 40 turns, OFFSET=32, gaps x10 capped at 1,200 s, 9,000 s per arm; arms hbm_host,
three_tier_to, three_tier_wc, hbm_lru.

## What changed since campaign 8 (all arms run on the new setup)

- **Engine:** the fork merged upstream SGLang main as of 2026-09-24 (commit ace8ac79a7, merged upstream 86b3558b41).
  Our only engine changes on top are the 31-line HICACHE_EVT instrumentation (all 13 event kinds verified present).
  It includes #39283 (scheduler._prefetch_after_device_hit_loss: at admission, if the device+host match shrank since
  arrival, re-issue the SSD lookup and skip the request until it lands) and #40042. Paced miss-retries are now on by
  default (`--hicache-storage-prefetch-retry-poll-interval` 8, `--hicache-storage-prefetch-retry-max-attempts` 8); a
  speculative miss-retry is cancelled for the queue head, so it never delays an admission-ready request.
- **Container:** sglang-kernel upgraded 0.4.6.post1 → 0.4.7 to match the merged pins (same CUDA 13 / torch 2.13 build);
  rollback `pip install --no-deps sglang-kernel==0.4.6.post1`. flashinfer 0.6.18, transformers 5.12.1, torch 2.13 unchanged.
- **Client:** no connection cap (TCPConnector(limit=0)); campaign 8's 100-connection cap queued up to 28 requests inside
  the client. SGLANG_TIMEOUT_KEEP_ALIVE=3600. ReqTimeStats logged on every arm.

So a difference from campaign 8 mixes #39283 with the engine upgrade and the uncapped client. Within this campaign,
hbm_host (no SSD, so no re-query) vs the three-tier arms is the clean comparison. A further arm with the re-query
disabled (`--hicache-storage-prefetch-retry-max-attempts 0`) would isolate #39283 on the same engine; not queued yet.

## Smoke test (01:17-01:33Z, deleted after this note)

three_tier_wc, L1 131,072 / L2 20 GB, 48 sessions at x10, 600 s: booted in 240 s with the requested pools, 760 turns,
0 errors, 13 HICACHE_EVT kinds, ReqTimeStats on every request, 42 "device prefix shrank before admission ... reissuing
storage lookup" and 110 "reissue cap reached" (long-queued requests admitted with the GPU match after 8 attempts).

## Expected

At x10 with the re-query, a returning turn whose L1/L2 context was evicted while queued should be re-fetched from the SSD
instead of recomputed (campaign 8: 49% of returns were admitted with only the shared prompt; the SSD served 0.2%). The
SSD arms should therefore show a large storage share and fewer recomputes, and, because recomputes feed the write-through
pressure that evicts queued contexts, possibly a different load regime. Watch SSD read bandwidth (re-fetching ~20K-token
contexts at the admission rate) and the reissue-cap rate.

## 01:37Z: fifth arm queued — three_tier_wc_norq (re-query off)

Requested by the user. three_tier_wc with `--hicache-storage-prefetch-retry-max-attempts 0`: on the merged engine this
disables both the #39283 admission-time re-query and the paced miss-retries (a request's attempt count, 0, is already at
the cap), i.e. campaign 8's retry behaviour on the new engine (c7-c10 ran 0/4 with the poll interval 0, i.e. retries off; #39283 made
8/8 the default), so three_tier_wc vs three_tier_wc_norq isolates #39283 as shipped: re-query plus paced retries.
Added as a new arm in start_server.sh (existing arms unchanged; the boot facts print the cap). It runs after hbm_lru via
`../chain_c11_norq.sh` (in the container): same knobs, its own compare dir and driver log
(`../compare_driver_c11_norq.log`), and its manifest line is appended to this campaign's manifest.txt. Expected start
~11:30Z, end ~14:00Z.

## 04:07Z: hbm_host done — reproduces campaign 8

Capped at 9000 s as planned, 0 errors. Against campaign 8's hbm_host: 4,667 vs 4,668 turns; returning-turn TTFT mean
115.4 vs 114.8 s, p50 134.3 vs 134.5 s, p99 266.7 vs 265.5 s; admission tiers device / host / recompute 550 / 1,461 /
2,528 vs 558 / 1,456 / 2,526. Neither the engine upgrade (ace8ac79a7) nor removing the client's 100-connection cap
changed the host-only arm: the closed loop is server-bound, so the cap only moved where requests waited. Campaign 8's
SSD arms are therefore a usable pre-#39283 reference, besides three_tier_wc_norq.

## 06:09Z: three_tier_to done — #39283 changes the x10 result

Boot check: retry max attempts 8 (HEAD default). All 128 conversations finished at 6,916 s (not capped; hbm_host and
campaign 8's three_tier_to both hit the 9,000 s cap). Returning turns: TTFT mean 69.2 s (campaign 8 three_tier_to:
115.9), p50 79.3 (134.0), p99 169.8 (266.1); admission tiers device / host / SSD / recompute 603 / 1,434 / 1,150 /
1,361 (campaign 8: 564 / 1,395 / 10 / 2,569). Paired with this campaign's hbm_host on the same (conv, turn), 4,539 pairs:
−46.1 s [−47.5, −44.7]. Server log: 1,158 admission-time re-queries ("reissuing storage lookup"), 2,152 "reissue cap
reached" (admitted with the device match after 8 attempts), 1,208 SSD reads. Window statistics (30-120 min) are not
comparable across arms here, because this arm's load drains from ~6,000 s while the capped arms stay saturated.

## 08:09Z: three_tier_wc done — replicates three_tier_to

Boot check: retry max attempts 8. All conversations finished at 6,961 s (not capped). Returning turns: TTFT mean 68.4 s,
p50 74.7, p99 174.5; device / host / SSD / recompute 594 / 1,457 / 1,158 / 1,339. Paired: vs hbm_host −46.9 s
[−48.3, −45.6] (4,539 pairs); vs three_tier_to −0.8 s [−1.1, −0.6], p50 0.0. Server log: 1,147 admission re-queries,
2,162 "reissue cap reached", 1,199 SSD reads. As in campaigns 8-10, the prefetch policy barely matters at this load.

## 10:44Z: hbm_lru done — reproduces campaign 8; four-arm driver ALL DONE

Capped at 9,000 s, 0 errors. Against campaign 8's hbm_lru: 4,543 vs 4,550 turns; TTFT mean 139.7 vs 138.7 s, p50 146.2
vs 144.4, p99 266.3 vs 263.8; device / recompute 440 / 3,975 vs 440 / 3,982. Both non-SSD arms reproduce campaign 8, so
the x10 change is confined to the SSD arms. three_tier_wc_norq starts next via chain_c11_norq.sh.

## 13:16Z: three_tier_wc_norq done — campaign complete

Boot check: retry max attempts 0. Capped at 9,000 s (2 conversations unfinished), 0 errors; 0 re-queries, 35 SSD reads.
TTFT mean 116.3 s, p50 135.3; device / host / SSD / recompute 528 / 1,435 / 9 / 2,566. Paired: three_tier_wc −
three_tier_wc_norq −47.8 s [−49.3, −46.5]; norq − hbm_host +0.9 s; norq − campaign 8 three_tier_wc +0.2 s. #39283 accounts
for the whole x10 change. Write-up in README.md.
