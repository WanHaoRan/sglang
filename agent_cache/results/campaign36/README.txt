Campaign 36: the unified four-action design against the current one at light load with a small host pool, ordered
as part of the author's request of 2026-11-26 (design notes in agent_cache/results/design_unified_1126/).
Setting: campaign 35's with 64 users in session (not 128), an 88 GB host pool (not 160) and 1 h (not 2): the GPU is
mostly unsaturated (about 110 requests a minute against about 140 at saturation), so a turn that waits on a read
mostly waits alone, and about 40 % of the sessions' KV fits in neither GPU nor host memory. Requests pair with campaign 35's
arms only by user, stream and index up to where the smaller cap changes who is in session.
Arms: queue.txt, ec_full first (the engine and flags of campaign 35's). Run by campaign35/chain_multi.sh.
2026-11-27 01:20:19Z: ec_full done (8,314 requests, 0 errors, no cleaner pass; the SSD reached 58 %). Requests sent
before 3,000 s: mean TTFT 0.68 s, p50 0.10 s, p90 1.59 s, p99 10.64 s; prompt tokens from GPU / host / SSD 0.66 / 0.18
/ 0.10; first requests 1.51 s, returning 0.64 s (compare_arms.out).
