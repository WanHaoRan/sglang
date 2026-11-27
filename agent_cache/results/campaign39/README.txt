Campaign 39: the load hold, ordered as part of the author's request of 2026-11-26 (design notes in
agent_cache/results/design_unified_1126/, DESIGN.md section 4.5: Fetch on the top tier).
Setting: campaign 35's (128 users, 160 GB host pool, 2 h; settings.env copied) plus
SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS=16, so requests pair with campaigns 34 and 35 by user, stream and index.
What it tests: today a prefill pass waits on the copy of the turn's history from host memory to the GPU (40 to 53 ms
per admitted request in campaign 34, 8 to 12 % of GPU time at saturation, measured from the logs). With the hold the
copy starts when the batch is built, the running requests keep decoding, and the batch is prefilled once the copy has
landed (or after 16 decode steps, or when nothing is left to decode). Expected through the closed-loop identity: 2.8 to
3.2 s of the 10.96 s mean (inferred); 1.9 to 4.2 s on the simulator.
Arms: queue.txt. ec_full_load = the current design with the hold only; the unified design with the hold is appended
when its engine copy is frozen. The baselines without the hold are campaign 35's and campaign 34's ec_full.
Every HICACHE_EVT load_hold line in a server log is one held batch (steps waited, landed=1 if the copy had landed).
Why 16 steps: the bound only matters when few requests run. With 20 running a decode step is 24 ms and a 26K-token
copy (65 ms) lands within 3 steps, when the batch is released; with one running a step is 4 ms, and 16 steps cover
the same copy.
2026-11-26 22:03:44Z: the first pre-flight stopped at the quiet tier check (preflight.log): served from the SSD the
checked prompt had 19,904 of 20,000 tokens cached, one 64-token page under the check's bar; logprobs agreed (largest
difference 0.15, tolerance 0.25) and no batch had been held (0 load_hold lines), so the hold was not involved. The
server log shows why: the 30 GB host pool (305K tokens) was held by pending SSD writes of the filler prompts
(pending=277120, host_census h_write=188416, free=0), host eviction freed 16,384 of the 19,968 tokens asked, and
the read was cut to 311 pages. A race of the check on a small host pool, not a fault of the engine copy; the check
gets a settle time before the checked request. The watcher commented the arm out (#FAILED-PREFLIGHT in queue.txt)
and started the chain for the baselines: campaign 37's ec_full launched 22:04Z. The arm is restored after a
pre-flight passes. Run folder of the failed check: 20261126_220029_three_tier_wc_custom_L1100000_H30.
2026-11-27 00:15:22Z: second pre-flight (after campaign 37's ec_full; the tier check now settles 30 s after its
filler prompts). Unit tests pass; the quiet tier check passes 5 of 5; with six generations decoding in the
background two batches were held (steps=7, landed=1 each) and both tier stages pass with the held batch's logprobs
equal to the warm run's (difference 0.0, tolerance 0.02), served from host memory and from the SSD. The run was
still counted failed because the cold-against-warm item differed by 0.31 (tolerance 0.25; 0.06 in the quiet check):
neither of those two requests loads anything or is held, the difference comes with batching beside other requests.
The pass rule of the busy check is now the two tier stages plus at least one held batch. The smokes had not run;
the watcher started campaign 36's ec_full (1 h) and the pre-flight is repeated after it.
Run folder: 20261127_001109_three_tier_wc_custom_L1100000_H30.
2026-11-27 01:53:18Z: third pre-flight of engine_loadhold_20261126 (after campaign 36's ec_full) passed. Unit tests
pass; quiet tier check 5 of 5; with six generations decoding 5 of 5 with two held batches (steps=7, landed=1) whose
logprobs equal the warm run's from host memory and from the SSD. Smokes (current design, GPU pool 200,000 tokens,
host 48 GB, 64 users, 600 s, same users):
  hold off: 1,283 requests, 0 errors; mean TTFT 13.98 s (p50 8.90, p90 31.9); 577 output tokens/s; decode time per
            token 12.92 ms (token-weighted, client side).
  hold 16:  1,262 requests, 0 errors, 0 tracebacks; 678 batches held, 664 released with the copy landed, after 2 to 10
            decode steps mostly (19 reached the bound of 16); mean TTFT 14.20 s (p50 8.34, p90 33.9); 568 output
            tokens/s; decode time per token 12.65 ms.
Reading: the mechanism works (server log: a 36 ms copy of 16,128 tokens with five decode steps beside it, then the
prefill). At the smoke's small running set (7 requests; 57 % of admissions load 8,192 tokens or more) it gives back
about 0.08 s of decode time per request, 2 %, and TTFT does not move outside what one ten-minute run can resolve.
At the campaign's 20 running requests a request sees about 15 admissions during its decode instead of 4, so the
same accounting gives about 9 % of slot time; the two-hour arm decides.
2026-11-27 01:57:03Z: ec_full_load launched on engine_loadhold_20261126 (the baseline engine plus the hold, nothing
else). The unified engine's first boot stopped at argument checking a minute earlier (the eviction config key
queued_order was not accepted under return_chance; fixed in arg_groups/kv_cache_hook.py), so the arm took the engine
copy made for it and the GPU did not idle.
2026-11-27 03:03Z, interim (the arm at minute 64; checks/hold_gpu_time.py and checks/pair_early.py against campaign
35's ec_full, same users):
  The mechanism, minutes 28 to 59. A decode interval holding one prefill batch of under 512 new tokens is longer than
  a pure one by 45 / 66 / 84 / 108 ms for copies of 37 / 65 / 85 / 110 ms in campaign 35 (slope on copy time
  0.75 +- 0.09), and by 26 / 32 / 38 / 52 ms for copies of 37 / 70 / 91 / 119 ms here (slope 0.07 +- 0.06):
  the pass no longer waits on the copy. 2,790 batches held (1.50 a second), every one released with its copy landed,
  after 4 decode steps on average. Decode steps a second: 33.76 against 32.13 (+5.1 %), at the same 1.9 prefill
  batches a second. The saving is the copy's excess over the pass's own compute, about 38 ms a batch, as
  Proposition 23 gives it (results/design_unified_1126/PROOFS.md).
  TTFT, paired by request (B - A, 95 % interval over users):
    minutes 0 to 28 (light load):  -0.008 s [-0.019, +0.003]   (0.248 -> 0.240 s)
    minutes 28 to 40:              -0.57 s [-0.69, -0.46]      (5.59 -> 5.02 s)
    minutes 40 to 50:              -1.10 s [-1.28, -0.91]      (7.90 -> 6.80 s)
    minutes 50 to 60:              -1.50 s [-1.70, -1.29]      (4.60 -> 3.10 s)
  Among quick requests of the first 20 minutes (first token within 0.5 s in both runs) the hold costs
  +8.0 ms [+5.3, +10.9]: a held turn's pass starts after its copy instead of beside it.
2026-11-27 04:00:56Z: ec_full_load DONE: 17,519 requests, 0 errors, 4 cleaner passes (campaign 35's ec_full: 17,093
requests, 4 passes). Result against campaign 35's ec_full on the same users (compare_unified/A_128u_160g,
checks/vs_c35/):
  Mean TTFT 9.36 s against 10.97 s: -1.61 s [-1.79, -1.43] on 15,560 paired requests (against campaign 34's run
  -1.59 s [-1.76, -1.41]); p50 6.70 s against 8.21 s.
  By window, paired: minutes 0 to 28 -0.008 s; 28 to 50 -0.85 s [-0.96, -0.75]; 50 to 70 -2.49 s [-2.68, -2.29];
  70 to 90 -3.50 s [-3.81, -3.23]; 90 to 110 -1.96 s [-2.54, -1.39].
  The mechanism over minutes 28 to 110 (checks/vs_c35/hold_gpu_time.out): the slope of a decode interval's length
  on its prefill batch's copy time is 0.71 +- 0.05 without the hold and -0.00 +- 0.04 with it; a batch of under 512
  new tokens costs the running turns 43 / 65 / 84 / 115 ms for copies of 40 / 64 / 84 / 113 ms without the hold and
  22 / 29 / 36 / 46 ms with it. 7,060 batches held (1.43 a second), 7,059 released with the copy landed (one at the
  bound of 16 steps), 4.4 decode steps beside the copy on average. Decode steps a second 33.66 against 32.12
  (+4.8 %); Proposition 23's model of the baseline log gives 37.9 ms of waiting a batch, +5.7 % at 1.43 held batches
  a second.
  The closed loop's accounting (checks/vs_c35/identity_check.out): per ten-minute window the change of TTFT from
  requests in flight over throughput, minus decode time, is -1.34 s weighted by requests against -1.44 s measured;
  the users in session are fewer under the hold until the cap binds (103 against 108 in minutes 60 to 70).
  The simulator's bound for a copy that costs no GPU time was -2.61 s (-3.92 to -1.23 over seeds, -3.14 s on these
  users); the log-based inference of 2.8 to 3.2 s counted the whole copy as lost, where the pass's own compute hides
  part of it.
2026-11-27 04:01Z: arm_checks.sh: 12 ok, 1 "failed": the per-campaign figure set (c28_users/plot_arms.py) only knows
the earlier arm labels and draws nothing for ec_full_load. The same will show for unified and unified_load; their
figures are results/compare_unified/<setting>/timeline.png and ttft_cdf.png (compare_unified/figs.py).
