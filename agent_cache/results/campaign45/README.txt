Campaign 45: a diagnostic arm added on 2026-11-27 after campaign 40. Campaign 40 (everything) was 0.16 s [+0.06,
+0.28] slower than the hold alone (campaign 39) in minutes 50 to 70, before its gain of 1.16 s once the SSD cleaner
deletes. In that window it read 9.0 against 5.4 M tokens from the SSD, its reads landed 6.3 s before admission on
average (1.6 s with the hold alone), staged pins were held 9.2 s on average (1.6 s), and 7 pins were held until the
600 s safety release ("leak"; none with the hold alone). The measured price of host room was 1e-8 to 7e-7 per
token-second: it counts the stall of re-reading a displaced history at c (0.01 to 0.07 once saturated) and not the
recompute a displacement can cause, so staging long before arrival looked free. This arm is campaign 40's with the
current design's fixed price (5e-5): SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES and ..._HOLD_QUANTILE off, everything
else as in campaign 40, same engine copy (frozen/engine_unified_20261126). Setting: campaign 35's.
It runs after campaign 44 (campaign44/after_chain.sh starts the launcher for both folders).

2026-11-27 18:18:03Z: unified_load_fixedprice DONE: 17,642 requests, 0 errors, 3 cleaner passes. Results
(compare_unified/A_128u_160g/):
  Mean TTFT 9.19 s on the 15,560 requests every arm sent: against the current design -1.79 s [-1.99, -1.59];
  against campaign 40 (the measured price) +0.04 s [-0.04, +0.13], a tie; against the hold alone -0.19 s [-0.29,
  -0.09]. Own requests 16,095, 146.3 a minute.
  The fixed price removes the early staging: in the first 70 minutes every pin was released at admission (campaign
  40: 6 at the 600 s safety release, 44 on abort), reads landed 2.1 s before admission on average (14.0 s), and
  minutes 50-70 read 5.6 M tokens from the SSD (9.0 M). But minutes 50-70 cost the same as in campaign 40: 7.41 s
  against 7.39 s, +0.22 s [+0.12, +0.32] against the hold alone in the interim pairing. So the measured price's early
  staging is real and wasteful, and it is not what costs TTFT there; the mid-run cost of the cache rules on top of
  the hold has another cause among them (the read line, price learning off, the order among queued turns and the
  Retain switches are the candidates; no arm separates them).
