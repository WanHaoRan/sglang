Campaign 37: the unified four-action design against the current one at saturation with a tight host pool, ordered
as part of the author's request of 2026-11-26 (design notes in agent_cache/results/design_unified_1126/).
Setting: campaign 35's with a 112 GB host pool (not 160): the host room beyond the GPU's mirror halves (about
0.49 M tokens against 0.97 M), so more history goes through the SSD.
Arms: queue.txt, ec_full first (the engine and flags of campaign 35's). Run by campaign35/chain_multi.sh.
2026-11-27 00:08:21Z: ec_full done (16,883 requests, 0 errors, 3 cleaner passes). Against campaign 35's ec_full (same
users, same GPU unit, 160 GB): mean TTFT 12.82 s against 10.97 s; paired on 15,185 requests +2.26 s [+1.30, +3.25];
p50 0.26 s against 8.21 s, p90 63.3 s against 23.7 s, p99 113 s against 53 s; returning requests that read from the
SSD 23.5 % against 8.5 % in minutes 30-60 and 28.1 % against 25.0 % in minutes 60-110; by window 0.15, 0.54, 7.12 and
24.64 s against 0.14, 0.54, 6.20 and 20.80 s (checks/vs_c35/compare_arms.out). The tight host pool costs 2.3 s of
mean TTFT and splits the requests: half get their first token within 0.3 s and the readers wait far longer.
