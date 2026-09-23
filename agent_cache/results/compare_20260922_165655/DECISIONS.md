# Campaign 8 — campaign 7 at GAP=10 instead of GAP=70

Started 2026-09-22T16:56:55Z, driver log `../compare_driver_c8.log`. Everything identical to `../compare_20260922_023714/`
(campaign 7: same model, pools, trace window, 128 live sessions x 40 turns, `GAP_CAP=1200`, same four arms in the same
order) except **`GAP=10`**, and the per-arm wall cap `CLIENT_TIMEOUT=9000` is now enforced (client `--max-seconds`).

## Why

Campaign 7's estimate of the pre-prefetch headroom (storage-hit TTFT 2.7-3.3 s p50 against 0.3 s for a host hit) rests
on gaps of 49 s p50 (x70). The question for this campaign: what happens to the same tiers when the idle gaps are 7x
shorter (7.0 s p50, 12.2 s p90; `../compare_20260922_023714/gap_distribution.png`), i.e. when the L3 read of a
20K-token restore (1.0-1.9 s) is no longer small against the gap and the request rate is up to 7x higher.

## Expected, so a wrong number is recognisable

- Offered load: 128 sessions / (7 s gap + ~5 s service) ~ 10 turns/s against a hit capacity of ~2-8 turns/s (decode
  bound) and a recompute capacity of ~0.2-0.5 turns/s: every arm is oversubscribed; the closed loop settles at the
  server's capacity and the queue holds most of the 128 sessions. The comparison is between failure modes, not
  between an idle and a loaded system.
- L2 turnover shrinks with the arrival rate (campaign 7: ~48 s at 1.2 turns/s), so the share of returns that fall
  through to L3 should stay near campaign 7's ~20 %; the L3 read demand scales with the arrival rate (campaign 7:
  0.9 GiB/s in the overflow phase) and will hit the 1.88 GiB/s ceiling: the SSD arms become disk-bound, storage-hit
  TTFT rises well above campaign 7's 2.7-3.3 s p50 and the queue for the disk sets the tail.
- hbm_host / hbm_lru: same total recompute work as campaign 7 (every evicted return recomputes) at a higher offered
  rate, so the same ~2.5-3.2 h GPU-bound wall time or the 9,000 s cap, whichever comes first; TTFT ~180-220 s p50 in
  the saturated phase as before or worse.
- If the tiered arms' storage hits show TTFT near the host-hit figure, queueing is hiding the read (the mechanism the
  pre-prefetch idea targets); if they show the read plus a disk queue, the gap was not the thing hiding it.

## 22:05Z finding: at GAP=10 the SSD tier gives nothing, because the L2 hit expires while the request queues

hbm_host (capped at 9,000 s, 4,668 turns): mean TTFT 111.8 s; device / host / recompute p50 4.2 / 12.9 / 213 s.
three_tier_to (capped at 9,000 s, 4,666 turns): mean TTFT 113.0 s; 10 storage hits in the whole run; 2,569 recomputes
at 212 s p50; only 68 prefetches ever moved data (1.1M tokens) against 6,724 host evictions. The SSD read side idled
(169 MiB/s mean while active, 0 samples at >= 95 % util; writes 666 MiB/s of write-through).

Mechanism (client turns joined with the server's HICACHE_EVT lines by rid, `three_tier_to`, turns after minute 5):
of 711 recomputed turns, 620 (87 %) arrived with their whole private context in L2 (the L3 query at enqueue asked for
the < 1K-token tail only, hence `storage_hit=0`), and were admitted ~150 s later with 7,808 cached tokens = the shared
system prompt: the chain had been evicted from L2 while the request waited. The 35 that did prefetch >= 8K from L3
got all of it (35/35 hits) and lost it the same way before admission.
Code (python/sglang/srt): the L2 match and the L3 prefetch are decided once at enqueue (`scheduler._add_request_to_queue
-> _prefetch_kvcache`), never at admission; `prefetch_from_storage` protects `last_host_node` only when the missing
tail is >= prefetch_threshold (256) and releases it as soon as the prefetch completes (`_handle_prefetch_result`), so
for essentially the whole queue wait the chain is ordinary LRU material for `evict_host`. At GAP=10 the queue wait
(~150 s) equals L2's turnover (recomputes re-insert ~0.47 turns/s x 24K tokens = 11K tok/s into 1.63M tokens ~ 145 s),
so the hit expires in the queue, the recompute re-inserts it, and that write-through evicts the next waiter's chain.
Campaign 7 (GAP=70) never reached this: its queue wait was 0.2 s.
Consequence for the pre-prefetch idea: the cache state must be right at admission, not at arrival. Candidate controls,
none applied here: admission-lookahead re-match/re-prefetch for the next few queued requests; bounded pinning of a
queued request's matched L2 chain; cache-aware admission order (`--schedule-policy lpm` orders by
`len(prefix_indices) + host_hit_length`; these campaigns run `fcfs`).

## 00:30Z note: `--enable-request-time-stats-logging` added to the launch line from arm 4 (hbm_lru) onward

Logging only (one `queue_duration=..., forward_duration=..., entry_time=...` line per finished request, `observability/
req_time_stats.py`): it gives the exact queue wait per request, which the first three arms lack (their admission time is
known only for host/storage hits through `HICACHE_EVT load_back_init`). Arms 1-3 of this campaign and all of campaign 7
ran without it; campaign 9 runs with it. No other flag changed.

## 03:10Z: campaign complete (ALL DONE 03:10:05Z); artefacts generated

hbm_lru: capped at 9,000 s with 4,550 turns (27 conversations cut at a turn boundary), mean TTFT 135.0 s, device / recompute
p50 5.2 / 164 s; queue wait measured per request (ReqTimeStats): mean 117.9 s, p50 109 s, p90 ~217 s, forward mean 66 s.
Final per-arm means (compare.csv / delay_components.py): hbm_host 114.8 s, three_tier_to 115.9 s, three_tier_wc 116.1 s,
hbm_lru 138.7 s returning-turn TTFT; storage hits 0 / 10 / 9 / 0; cold returns 53 / 54 / 54 / 82 %. Figures:
compare.png, timeline_<arm>.png; per-turn splits in ../analysis_gapscale/turn_components_x10*.csv. Campaign 9 (GAP=1)
was launched by chain_c9.sh at 03:11:16Z into ../compare_20260923_031116/.
