### Campaign 34 (fixes on): requests sent before 6,600 s

| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 31.88 | 31.58 | -1 % | 28.06 ± 17.44 [7.45, 54.12] | client.jsonl |
| jit_p50: mean TTFT, s | 11.47 | 11.11 | -3 % | 11.45 ± 5.38 [4.72, 19.36] | client.jsonl |
| jit_exact: mean TTFT, s | 10.92 | 10.58 | -3 % | 11.17 ± 5.34 [4.64, 19.11] | client.jsonl |
| ec_full: mean TTFT, s | 10.96 | 10.62 | -3 % | 11.17 ± 5.31 [4.65, 19.04] | client.jsonl |
| ec_exact: mean TTFT, s | 10.66 | 10.60 | -1 % | 11.13 ± 5.31 [4.60, 18.99] | client.jsonl |
| hicache over the mean of the four hinted arms | 2.90 | 2.94 | +2 % | 2.34 ± 0.56 [1.60, 2.94] | client.jsonl |
| hinted arms: slowest minus fastest, s | 0.81 | 0.53 | -35 % | 0.32 ± 0.15 [0.12, 0.53] | client.jsonl |
| jit_p50 -> ec_full: difference of means, s | -0.51 | -0.49 | -4 % | -0.28 ± 0.15 [-0.49, -0.07] | client.jsonl |
| ec_full -> jit_exact: difference of means, s | -0.05 | -0.04 | -19 % | -0.01 ± 0.04 [-0.04, 0.06] | client.jsonl |
| ec_full -> ec_exact: difference of means, s | -0.30 | -0.02 | -93 % | -0.04 ± 0.01 [-0.05, -0.02] | client.jsonl |
| hicache: p50 TTFT, s | 5.95 | 5.89 | -1 % | 16.70 ± 30.22 [0.13, 70.51] | client.jsonl |
| hicache: p90 TTFT, s | 85.6 | 84.1 | -2 % | 72.9 ± 34.3 [15.6, 106.0] | client.jsonl |
| hicache: p99 TTFT, s | 110.5 | 105.6 | -4 % | 99.6 ± 13.0 [86.2, 117.0] | client.jsonl |
| jit_p50: p50 TTFT, s | 6.38 | 6.22 | -2 % | 5.86 ± 3.85 [0.16, 10.60] | client.jsonl |
| jit_p50: p90 TTFT, s | 22.4 | 25.0 | +12 % | 32.1 ± 19.9 [15.1, 64.5] | client.jsonl |
| jit_p50: p99 TTFT, s | 81.0 | 81.0 | -0 % | 72.7 ± 15.8 [47.6, 89.3] | client.jsonl |
| jit_exact: p50 TTFT, s | 8.24 | 7.80 | -5 % | 7.98 ± 5.96 [0.15, 15.99] | client.jsonl |
| jit_exact: p90 TTFT, s | 22.3 | 22.3 | +0 % | 26.5 ± 10.9 [14.9, 43.9] | client.jsonl |
| jit_exact: p99 TTFT, s | 56.3 | 55.9 | -1 % | 54.5 ± 6.0 [46.8, 62.6] | client.jsonl |
| ec_full: p50 TTFT, s | 8.34 | 7.86 | -6 % | 7.99 ± 5.95 [0.15, 15.93] | client.jsonl |
| ec_full: p90 TTFT, s | 24.0 | 23.2 | -3 % | 26.8 ± 10.8 [14.8, 44.0] | client.jsonl |
| ec_full: p99 TTFT, s | 53.1 | 54.1 | +2 % | 51.4 ± 5.9 [43.7, 59.3] | client.jsonl |
| ec_exact: p50 TTFT, s | 7.87 | 7.82 | -1 % | 8.00 ± 6.02 [0.14, 16.12] | client.jsonl |
| ec_exact: p90 TTFT, s | 22.6 | 22.7 | +1 % | 26.3 ± 9.9 [14.8, 41.4] | client.jsonl |
| ec_exact: p99 TTFT, s | 54.9 | 55.0 | +0 % | 51.9 ± 7.0 [41.5, 59.9] | client.jsonl |

### Campaign 34: by half hour of send time (mean TTFT, s)

| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: minutes 0-30 | 0.41 | 0.40 | -3 % | 1.89 ± 3.21 [0.18, 7.59] | client.jsonl |
| hicache: minutes 30-60 | 30.53 | 30.83 | +1 % | 24.97 ± 31.12 [0.36, 76.12] | client.jsonl |
| hicache: minutes 60-90 | 72.97 | 71.30 | -2 % | 60.48 ± 39.19 [4.23, 104.35] | client.jsonl |
| hicache: minutes 90-110 | 90.14 | 88.25 | -2 % | 78.90 ± 15.25 [55.85, 93.73] | client.jsonl |
| jit_p50: minutes 0-30 | 0.41 | 0.39 | -6 % | 1.24 ± 1.79 [0.17, 4.36] | client.jsonl |
| jit_p50: minutes 30-60 | 6.37 | 6.07 | -5 % | 8.21 ± 9.42 [0.35, 24.01] | client.jsonl |
| jit_p50: minutes 60-90 | 17.80 | 17.37 | -2 % | 15.90 ± 7.97 [4.25, 25.29] | client.jsonl |
| jit_p50: minutes 90-110 | 28.20 | 26.95 | -4 % | 24.39 ± 3.72 [18.43, 28.02] | client.jsonl |
| jit_exact: minutes 0-30 | 0.40 | 0.36 | -10 % | 1.19 ± 1.74 [0.15, 4.25] | client.jsonl |
| jit_exact: minutes 30-60 | 6.13 | 5.86 | -4 % | 7.93 ± 9.11 [0.30, 23.19] | client.jsonl |
| jit_exact: minutes 60-90 | 16.60 | 15.98 | -4 % | 15.30 ± 7.72 [4.16, 24.57] | client.jsonl |
| jit_exact: minutes 90-110 | 27.03 | 26.39 | -2 % | 24.30 ± 3.80 [18.32, 27.81] | client.jsonl |
| ec_full: minutes 0-30 | 0.41 | 0.39 | -6 % | 1.22 ± 1.75 [0.16, 4.29] | client.jsonl |
| ec_full: minutes 30-60 | 6.18 | 5.90 | -5 % | 8.05 ± 9.24 [0.35, 23.55] | client.jsonl |
| ec_full: minutes 60-90 | 16.36 | 15.91 | -3 % | 15.24 ± 7.58 [4.21, 24.16] | client.jsonl |
| ec_full: minutes 90-110 | 27.97 | 26.77 | -4 % | 24.17 ± 3.83 [18.19, 27.61] | client.jsonl |
| ec_exact: minutes 0-30 | 0.39 | 0.36 | -8 % | 1.20 ± 1.76 [0.15, 4.27] | client.jsonl |
| ec_exact: minutes 30-60 | 5.91 | 5.82 | -1 % | 7.91 ± 9.09 [0.30, 23.12] | client.jsonl |
| ec_exact: minutes 60-90 | 15.76 | 16.11 | +2 % | 15.25 ± 7.68 [4.11, 24.39] | client.jsonl |
| ec_exact: minutes 90-110 | 27.44 | 26.48 | -4 % | 24.30 ± 3.80 [18.27, 27.94] | client.jsonl |

### Campaign 34: throughput, tiers, memory, device

| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: requests per minute (to 6,600 s) | 90.6 | 90.9 | +0 % | 96.6 ± 18.1 [72.6, 121.9] | client.jsonl |
| hicache: requests per minute, minutes 90-109 | 55.3 | 55.9 | +1 % | 61.4 ± 7.8 [55.8, 73.3] | client.jsonl |
| jit_p50: requests per minute (to 6,600 s) | 141.9 | 142.6 | +1 % | 136.9 ± 3.9 [133.2, 142.6] | client.jsonl |
| jit_p50: requests per minute, minutes 90-109 | 129.1 | 132.2 | +2 % | 133.5 ± 6.8 [125.0, 144.1] | client.jsonl |
| jit_exact: requests per minute (to 6,600 s) | 141.8 | 142.6 | +1 % | 136.6 ± 3.9 [133.1, 142.6] | client.jsonl |
| jit_exact: requests per minute, minutes 90-109 | 132.1 | 134.4 | +2 % | 133.4 ± 9.2 [125.9, 148.4] | client.jsonl |
| ec_full: requests per minute (to 6,600 s) | 141.5 | 142.3 | +1 % | 136.6 ± 3.8 [132.9, 142.3] | client.jsonl |
| ec_full: requests per minute, minutes 90-109 | 128.3 | 133.3 | +4 % | 132.8 ± 8.6 [125.2, 146.9] | client.jsonl |
| ec_exact: requests per minute (to 6,600 s) | 142.2 | 142.3 | +0 % | 136.4 ± 3.8 [132.9, 142.3] | client.jsonl |
| ec_exact: requests per minute, minutes 90-109 | 132.6 | 134.6 | +1 % | 132.5 ± 8.1 [123.8, 145.0] | client.jsonl |
| hicache: prompt tokens from GPU, share | 0.302 | 0.305 | +1 % | 0.340 ± 0.158 [0.167, 0.557] | client.jsonl |
| hicache: from host, share | 0.241 | 0.236 | -2 % | 0.256 ± 0.102 [0.114, 0.399] | client.jsonl |
| hicache: from SSD, share | 0.121 | 0.121 | -0 % | 0.111 ± 0.043 [0.052, 0.167] | client.jsonl |
| hicache: history recomputed, share | 0.2891 | 0.2913 | +1 % | 0.2427 ± 0.1457 [0.0610, 0.4484] | client.jsonl |
| jit_p50: prompt tokens from GPU, share | 0.186 | 0.191 | +3 % | 0.250 ± 0.161 [0.115, 0.499] | client.jsonl |
| jit_p50: from host, share | 0.603 | 0.596 | -1 % | 0.536 ± 0.118 [0.353, 0.643] | client.jsonl |
| jit_p50: from SSD, share | 0.148 | 0.150 | +1 % | 0.147 ± 0.044 [0.083, 0.201] | client.jsonl |
| jit_p50: history recomputed, share | 0.0053 | 0.0061 | +16 % | 0.0054 ± 0.0015 [0.0036, 0.0070] | client.jsonl |
| jit_exact: prompt tokens from GPU, share | 0.205 | 0.209 | +2 % | 0.271 ± 0.173 [0.128, 0.537] | client.jsonl |
| jit_exact: from host, share | 0.620 | 0.615 | -1 % | 0.543 ± 0.120 [0.357, 0.661] | client.jsonl |
| jit_exact: from SSD, share | 0.113 | 0.114 | +1 % | 0.121 ± 0.064 [0.041, 0.215] | client.jsonl |
| jit_exact: history recomputed, share | 0.0050 | 0.0045 | -10 % | 0.0044 ± 0.0011 [0.0032, 0.0056] | client.jsonl |
| ec_full: prompt tokens from GPU, share | 0.194 | 0.199 | +3 % | 0.257 ± 0.164 [0.119, 0.509] | client.jsonl |
| ec_full: from host, share | 0.606 | 0.600 | -1 % | 0.538 ± 0.109 [0.368, 0.643] | client.jsonl |
| ec_full: from SSD, share | 0.138 | 0.140 | +1 % | 0.140 ± 0.062 [0.058, 0.225] | client.jsonl |
| ec_full: history recomputed, share | 0.0044 | 0.0040 | -8 % | 0.0040 ± 0.0009 [0.0030, 0.0052] | client.jsonl |
| ec_exact: prompt tokens from GPU, share | 0.209 | 0.210 | +1 % | 0.272 ± 0.173 [0.131, 0.537] | client.jsonl |
| ec_exact: from host, share | 0.614 | 0.611 | -0 % | 0.537 ± 0.117 [0.354, 0.650] | client.jsonl |
| ec_exact: from SSD, share | 0.116 | 0.117 | +1 % | 0.125 ± 0.065 [0.043, 0.219] | client.jsonl |
| ec_exact: history recomputed, share | 0.0039 | 0.0046 | +18 % | 0.0044 ± 0.0009 [0.0034, 0.0056] | client.jsonl |
| hicache: from SSD, minutes 30-60 | 0.142 | 0.136 | -4 % | 0.112 ± 0.090 [0.026, 0.251] | client.jsonl |
| hicache: from SSD, minutes 60-90 | 0.242 | 0.251 | +3 % | 0.197 ± 0.085 [0.059, 0.270] | client.jsonl |
| hicache: from SSD, minutes 90-110 | 0.255 | 0.254 | -0 % | 0.261 ± 0.032 [0.209, 0.290] | client.jsonl |
| jit_p50: from SSD, minutes 30-60 | 0.126 | 0.122 | -3 % | 0.115 ± 0.090 [0.024, 0.248] | client.jsonl |
| jit_p50: from SSD, minutes 60-90 | 0.228 | 0.237 | +4 % | 0.205 ± 0.074 [0.074, 0.256] | client.jsonl |
| jit_p50: from SSD, minutes 90-110 | 0.246 | 0.247 | +0 % | 0.260 ± 0.020 [0.239, 0.287] | client.jsonl |
| jit_exact: from SSD, minutes 30-60 | 0.032 | 0.032 | +2 % | 0.076 ± 0.105 [0.015, 0.261] | client.jsonl |
| jit_exact: from SSD, minutes 60-90 | 0.186 | 0.181 | -3 % | 0.163 ± 0.099 [0.027, 0.275] | client.jsonl |
| jit_exact: from SSD, minutes 90-110 | 0.274 | 0.284 | +4 % | 0.263 ± 0.066 [0.150, 0.320] | client.jsonl |
| ec_full: from SSD, minutes 30-60 | 0.079 | 0.078 | -1 % | 0.102 ± 0.097 [0.022, 0.267] | client.jsonl |
| ec_full: from SSD, minutes 60-90 | 0.212 | 0.214 | +1 % | 0.185 ± 0.088 [0.057, 0.272] | client.jsonl |
| ec_full: from SSD, minutes 90-110 | 0.284 | 0.288 | +1 % | 0.273 ± 0.055 [0.181, 0.321] | client.jsonl |
| ec_exact: from SSD, minutes 30-60 | 0.047 | 0.043 | -9 % | 0.082 ± 0.103 [0.011, 0.260] | client.jsonl |
| ec_exact: from SSD, minutes 60-90 | 0.181 | 0.184 | +2 % | 0.168 ± 0.095 [0.036, 0.277] | client.jsonl |
| ec_exact: from SSD, minutes 90-110 | 0.279 | 0.285 | +2 % | 0.269 ± 0.063 [0.162, 0.322] | client.jsonl |
| jit_p50: first cleaner pass, s | 5301 | 5190 | -2 % | 5292 ± 245 [5010, 5550] | campaign34/DECISIONS.md (Run) |
| ec_full: first cleaner pass, s | 5275 | 5190 | -2 % | 5280 ± 242 [5010, 5550] | campaign34/DECISIONS.md (Run) |
| jit_exact: first cleaner pass, s | 5272 | 5190 | -2 % | 5274 ± 247 [5010, 5550] | campaign34/DECISIONS.md (Run) |
| ec_exact: first cleaner pass, s | 5252 | 5190 | -1 % | 5274 ± 247 [5010, 5550] | campaign34/DECISIONS.md (Run) |
| hicache: first cleaner pass, s | none | none |  | 6375 ± 1167 [5550, 7200] | campaign34/DECISIONS.md (Run) |
| jit_p50: reads refused host room before the cut | 0 | 46 |  | 18 ± 19 [3, 46] | campaign34/DECISIONS.md |
| ec_full: reads refused host room before the cut | 0 | 0 |  | 1 ± 2 [0, 4] | campaign34/DECISIONS.md |
| jit_exact: reads refused host room before the cut | 0 | 0 |  | 2 ± 3 [0, 6] | campaign34/DECISIONS.md |
| ec_exact: reads refused host room before the cut | 0 | 0 |  | 1 ± 2 [0, 5] | campaign34/DECISIONS.md |
| jit_p50: pinned host tokens, median of minutes 60-110, K | 136 | 172 | +26 % | 148 ± 29 [104, 176] | campaign34/DECISIONS.md (census: pins 0 to 25 K; jit_p50's early reads 136 K) |
| ec_full: pinned host tokens, median of minutes 60-110, K | 0 | 2 |  | 1 ± 1 [0, 3] | campaign34/DECISIONS.md (census: pins 0 to 25 K; jit_p50's early reads 136 K) |
| jit_exact: pinned host tokens, median of minutes 60-110, K | 0 | 0 |  | 1 ± 1 [0, 3] | campaign34/DECISIONS.md (census: pins 0 to 25 K; jit_p50's early reads 136 K) |
| jit_p50: pinned host tokens, p90, K | 280 | 310 | +11 % | 302 ± 45 [251, 351] | campaign34/DECISIONS.md (census) |
| ec_full: host mirror, median of minutes 60-110, K | 650 | 648 | -0 % | 644 ± 3 [642, 648] | campaign34/DECISIONS.md (census) |
| ec_full: host free, median, K | 17.0 | 11.6 | -31 % | 10.6 ± 2.7 [5.9, 12.4] | campaign34/DECISIONS.md (census: 16 to 18 K) |
| ec_full: pending writes, p90 of minutes 60-110, K | 80 | 33 | -59 % | 33 ± 8 [20, 42] | campaign34/DECISIONS.md (census p90 65 to 94 K) |
| ec_full: running requests, minutes 90-110 (median) | 19.9 | 20.0 | +1 % | 20.6 ± 0.9 [20.0, 22.0] | critique.md (19.7 to 20.1) |
| jit_exact: running requests, minutes 90-110 (median) | 19.9 | 20.0 | +1 % | 20.6 ± 0.9 [20.0, 22.0] | critique.md (19.7 to 20.1) |
| jit_p50: running requests, minutes 90-110 (median) | 19.9 | 19.5 | -2 % | 20.5 ± 1.0 [19.5, 22.0] | critique.md (19.7 to 20.1) |
| ec_full: queued requests, minutes 90-110 (median) | 60 | 59 | -2 % | 52 ± 10 [34, 59] | campaign34/checks/by_minute_ec_full.out |
| ec_full: GPU pool used while queued (median) | 0.974 | 0.980 | +1 % | 0.978 ± 0.002 [0.976, 0.980] | campaign34/checks/by_minute_ec_full.out |
| jit_p50: SSD reads in the run | 3268 | 3405 | +4 % | 3217 ± 777 [2051, 4160] | campaign34/checks/read_thread.out |
| ec_full: SSD reads in the run | 2678 | 2797 | +4 % | 2789 ± 963 [1545, 4180] | campaign34/checks/read_thread.out |
| jit_exact: SSD reads in the run | 2221 | 2368 | +7 % | 2437 ± 964 [1238, 3908] | campaign34/checks/read_thread.out |
| jit_p50: tokens read from the SSD, M | 82.5 | 84.0 | +2 % | 77.6 ± 21.9 [45.6, 103.6] | campaign34/checks/read_thread.out |
| ec_full: tokens read from the SSD, M | 64.1 | 65.2 | +2 % | 64.1 ± 24.7 [31.9, 98.6] | campaign34/checks/read_thread.out |
| jit_exact: tokens read from the SSD, M | 54.4 | 56.4 | +4 % | 57.8 ± 25.3 [26.5, 95.6] | campaign34/checks/read_thread.out |
| jit_p50: reads staged before arrival | 1208 | 1216 | +1 % | 1049 ± 264 [639, 1282] | facts_fetch.md (ntp_stage) |
| ec_full: reads staged before arrival | 1 | 1 | +0 % | 1 ± 1 [0, 2] | facts_fetch.md (ntp_stage) |
| jit_exact: reads staged before arrival | 99 | 103 | +4 % | 107 ± 34 [63, 150] | facts_fetch.md (ntp_stage) |
| jit_p50: write hold switched on, times | 30 | 80 | +167 % | 70 ± 17 [47, 86] | facts_evict.md (srs_write_hold) |
| ec_full: write hold switched on, times | 30 | 51 | +70 % | 61 ± 22 [37, 93] | facts_evict.md (srs_write_hold) |
| jit_exact: write hold switched on, times | 28 | 59 | +111 % | 57 ± 13 [40, 70] | facts_evict.md (srs_write_hold) |
| jit_p50: users with a request before the cut | 617 | 617 | +0 % | 617 ± 14 [602, 640] | campaign34/DECISIONS.md (Against campaign 32) |
| ec_full: users with a request before the cut | 620 | 623 | +0 % | 620 ± 14 [605, 641] | campaign34/DECISIONS.md (Against campaign 32) |
| jit_exact: users with a request before the cut | 622 | 622 | +0 % | 620 ± 13 [606, 640] | campaign34/DECISIONS.md (Against campaign 32) |
| ec_exact: users with a request before the cut | 621 | 624 | +0 % | 620 ± 13 [604, 639] | campaign34/DECISIONS.md (Against campaign 32) |
| hicache: users with a request before the cut | 448 | 448 | +0 % | 484 ± 68 [394, 577] | campaign34/DECISIONS.md (Against campaign 32) |
| ec_full: updates of the learned price in the run | 1451 | 1445 | -0 % | 1445 ± 0 [1445, 1445] | facts_fetch.md (ec_price lines) |
| ec_full: of them, updates that moved the price | 0 | 0 |  | 0 ± 0 [0, 0] | facts_fetch.md (lambda stayed at 5e-5) |

### Campaign 32 (fixes off): requests sent before 6,600 s

| anchor | measured | simulated, seed 0: median of 6 replicas [min, max] | relative error of the median | simulated, seeds 0-4 (no jitter): mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 33.52 | 31.80 [31.16, 32.08] | -5 % | 29.23 ± 18.81 [7.38, 57.73] | client.jsonl |
| hicache: requests per minute | 87.8 | 90.3 [90.1, 91.0] | +3 % | 95.2 ± 19.3 [69.3, 122.2] | client.jsonl |
| hicache: history recomputed, share | 0.314 | 0.293 [0.285, 0.297] | -7 % | 0.255 ± 0.158 [0.060, 0.483] | client.jsonl |
| jit_p50: mean TTFT, s | 17.97 | 15.10 [14.12, 18.67] | -16 % | 18.50 ± 14.43 [5.93, 43.10] | client.jsonl |
| jit_p50: requests per minute | 114.5 | 123.3 [112.7, 127.9] | +8 % | 115.8 ± 18.0 [84.5, 128.0] | client.jsonl |
| jit_p50: history recomputed, share | 0.114 | 0.078 [0.059, 0.129] | -32 % | 0.104 ± 0.110 [0.025, 0.295] | client.jsonl |
| jit_exact: mean TTFT, s | 14.86 | 13.50 [13.35, 13.60] | -9 % | 17.15 ± 13.34 [5.03, 39.80] | client.jsonl |
| jit_exact: requests per minute | 123.2 | 127.9 [127.2, 128.5] | +4 % | 118.4 ± 16.9 [89.3, 130.9] | client.jsonl |
| jit_exact: history recomputed, share | 0.073 | 0.057 [0.055, 0.059] | -21 % | 0.088 ± 0.096 [0.014, 0.256] | client.jsonl |
| ec_full: mean TTFT, s | 14.64 | 13.64 [13.33, 13.74] | -7 % | 16.58 ± 11.91 [5.02, 36.48] | client.jsonl |
| ec_full: requests per minute | 125.0 | 127.5 [127.2, 129.0] | +2 % | 119.0 ± 14.6 [94.4, 131.0] | client.jsonl |
| ec_full: history recomputed, share | 0.063 | 0.059 [0.053, 0.061] | -8 % | 0.082 ± 0.080 [0.013, 0.220] | client.jsonl |
| ec_exact: mean TTFT, s | 14.95 | 13.57 [13.50, 13.62] | -9 % | 15.65 ± 10.36 [4.90, 32.66] | client.jsonl |
| ec_exact: requests per minute | 123.0 | 127.5 [127.2, 128.0] | +4 % | 120.9 ± 12.0 [100.8, 131.7] | client.jsonl |
| ec_exact: history recomputed, share | 0.073 | 0.059 [0.058, 0.061] | -19 % | 0.072 ± 0.063 [0.012, 0.179] | client.jsonl |
| hicache: mean TTFT, minutes 0-30 | 0.42 | 0.40 [0.39, 0.40] | -5 % | 1.85 ± 3.12 [0.18, 7.40] | client.jsonl |
| hicache: mean TTFT, minutes 30-60 | 29.64 | 29.03 [28.17, 29.59] | -2 % | 26.09 ± 33.94 [0.36, 83.01] | client.jsonl |
| hicache: mean TTFT, minutes 60-90 | 83.14 | 73.72 [72.49, 75.38] | -11 % | 65.09 ± 43.05 [4.23, 112.90] | client.jsonl |
| hicache: mean TTFT, minutes 90-110 | 111.82 | 97.43 [95.01, 99.48] | -13 % | 87.10 ± 22.78 [54.30, 110.36] | client.jsonl |
| jit_p50: mean TTFT, minutes 0-30 | 0.42 | 0.39 [0.39, 0.39] | -7 % | 1.29 ± 1.91 [0.17, 4.64] | client.jsonl |
| jit_p50: mean TTFT, minutes 30-60 | 7.27 | 6.22 [6.09, 7.75] | -14 % | 14.19 ± 22.30 [0.35, 53.63] | client.jsonl |
| jit_p50: mean TTFT, minutes 60-90 | 40.08 | 22.09 [17.81, 43.68] | -45 % | 29.64 ± 30.97 [4.22, 82.95] | client.jsonl |
| jit_p50: mean TTFT, minutes 90-110 | 69.30 | 84.39 [70.03, 89.52] | +22 % | 65.39 ± 25.43 [30.56, 84.99] | client.jsonl |
| jit_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 [0.36, 0.37] | -10 % | 1.19 ± 1.74 [0.15, 4.25] | client.jsonl |
| jit_exact: mean TTFT, minutes 30-60 | 6.18 | 5.82 [5.79, 5.86] | -6 % | 11.83 ± 17.53 [0.30, 42.66] | client.jsonl |
| jit_exact: mean TTFT, minutes 60-90 | 21.75 | 16.71 [16.30, 16.91] | -23 % | 27.74 ± 32.19 [4.16, 84.14] | client.jsonl |
| jit_exact: mean TTFT, minutes 90-110 | 79.46 | 80.14 [79.30, 83.50] | +1 % | 63.43 ± 25.76 [22.15, 80.88] | client.jsonl |
| ec_full: mean TTFT, minutes 0-30 | 0.45 | 0.39 [0.39, 0.40] | -13 % | 1.22 ± 1.76 [0.16, 4.29] | client.jsonl |
| ec_full: mean TTFT, minutes 30-60 | 6.42 | 5.92 [5.87, 6.00] | -8 % | 10.07 ± 13.57 [0.35, 33.67] | client.jsonl |
| ec_full: mean TTFT, minutes 60-90 | 20.01 | 16.86 [15.98, 17.12] | -16 % | 27.56 ± 31.30 [4.21, 82.27] | client.jsonl |
| ec_full: mean TTFT, minutes 90-110 | 80.17 | 81.82 [78.21, 83.55] | +2 % | 63.99 ± 25.35 [21.75, 83.56] | client.jsonl |
| ec_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 [0.35, 0.37] | -9 % | 1.20 ± 1.76 [0.15, 4.28] | client.jsonl |
| ec_exact: mean TTFT, minutes 30-60 | 6.39 | 5.82 [5.75, 5.85] | -9 % | 8.47 ± 10.27 [0.30, 25.92] | client.jsonl |
| ec_exact: mean TTFT, minutes 60-90 | 22.09 | 16.79 [16.62, 16.97] | -24 % | 26.45 ± 30.04 [4.11, 79.09] | client.jsonl |
| ec_exact: mean TTFT, minutes 90-110 | 78.47 | 81.55 [80.51, 83.89] | +4 % | 62.33 ± 24.70 [20.97, 81.78] | client.jsonl |
| jit_p50: loop onset, minute (first of 3 in a row with 5 or more refusals) | 42 | 81 [42, 89] | +93 % | 75 ± 29 [26, 97] | campaign34/DECISIONS.md (Arms) |
| ec_exact: loop onset, minute (first of 3 in a row with 5 or more refusals) | 79 | 87 [87, 88] | +10 % | 86 ± 19 [56, 108] | campaign34/DECISIONS.md (Arms) |
| jit_exact: loop onset, minute (first of 3 in a row with 5 or more refusals) | 80 | 88 [87, 90] | +10 % | 82 ± 25 [40, 106] | campaign34/DECISIONS.md (Arms) |
| ec_full: loop onset, minute (first of 3 in a row with 5 or more refusals) | 84 | 88 [87, 90] | +5 % | 84 ± 22 [47, 106] | campaign34/DECISIONS.md (Arms) |
| jit_p50: loop onset, minute (5 in a row) | 67 | 81 [65, 89] | +21 % | 79 ± 25 [37, 97] | campaign34/DECISIONS.md (Arms) |
| jit_p50, fixes on: loop onset | none | none |  | none | campaign34/DECISIONS.md: none |
| jit_exact, fixes on: loop onset | none | none |  | none | campaign34/DECISIONS.md: none |
| ec_full, fixes on: loop onset | none | none |  | none | campaign34/DECISIONS.md: none |
| ec_exact, fixes on: loop onset | none | none |  | none | campaign34/DECISIONS.md: none |
| jit_p50: pending SSD writes at their peak after minute 60, K tokens | 675 | 716 [662, 816] | +6 % | 718 ± 77 [634, 833] | campaign34/DECISIONS.md (campaign 32 column) |
| ec_full: pending SSD writes at their peak after minute 60, K tokens | 610 | 736 [701, 815] | +21 % | 689 ± 43 [638, 742] | campaign34/DECISIONS.md (campaign 32 column) |
| jit_exact: pending SSD writes at their peak after minute 60, K tokens | 523 | 692 [658, 724] | +32 % | 682 ± 53 [604, 740] | campaign34/DECISIONS.md (campaign 32 column) |
| ec_exact: pending SSD writes at their peak after minute 60, K tokens | 576 | 710 [665, 770] | +23 % | 700 ± 56 [610, 749] | campaign34/DECISIONS.md (campaign 32 column) |
| hicache: requests per minute, minutes 90-109 | 46.9 | 51.5 [50.8, 53.1] | +10 % | 58.0 ± 11.1 [48.5, 74.4] | client.jsonl |
| jit_p50: requests per minute, minutes 90-109 | 65.2 | 56.3 [52.9, 63.9] | -14 % | 74.7 ± 23.2 [56.3, 106.2] | client.jsonl |
| jit_exact: requests per minute, minutes 90-109 | 59.2 | 58.3 [56.2, 59.0] | -1 % | 77.2 ± 27.0 [58.7, 122.3] | client.jsonl |
| ec_full: requests per minute, minutes 90-109 | 58.9 | 57.3 [56.7, 59.7] | -3 % | 76.0 ± 27.7 [56.1, 123.8] | client.jsonl |
| ec_exact: requests per minute, minutes 90-109 | 59.9 | 57.4 [55.9, 57.8] | -4 % | 77.1 ± 28.4 [57.2, 126.4] | client.jsonl |

### Campaign 34 minus campaign 32 (the fixes), difference of mean TTFT, s

| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: fixes on minus fixes off (each run's own requests; seed 0 against the median of its 6 replicas) | -1.63 | -0.22 | -86 % | -1.17 ± 1.48 [-3.61, 0.06] | client.jsonl |
| jit_p50: fixes on minus fixes off (each run's own requests; seed 0 against the median of its 6 replicas) | -6.50 | -3.99 | -39 % | -7.05 ± 9.44 [-23.74, -1.21] | client.jsonl |
| jit_exact: fixes on minus fixes off (each run's own requests; seed 0 against the median of its 6 replicas) | -3.94 | -2.92 | -26 % | -5.98 ± 8.31 [-20.69, -0.39] | client.jsonl |
| ec_full: fixes on minus fixes off (each run's own requests; seed 0 against the median of its 6 replicas) | -3.68 | -3.02 | -18 % | -5.40 ± 6.85 [-17.44, -0.36] | client.jsonl |
| ec_exact: fixes on minus fixes off (each run's own requests; seed 0 against the median of its 6 replicas) | -4.29 | -2.97 | -31 % | -4.52 ± 5.24 [-13.67, -0.30] | client.jsonl |

### Campaign 34: paired differences between arms (B - A on the requests both sent before 6,600 s), s

| pair | measured [95 % interval over users] | simulated, seed 0 [interval] | simulated, seeds 0-4: mean [min, max] | sign |
|---|---|---|---|---|
| jit_p50 -> ec_full | -0.62 [-1.02, -0.22] | -0.59 [-1.02, -0.19] | -0.35 [-0.59, -0.06] | same in every seed |
| jit_p50 -> jit_exact | -0.67 [-1.07, -0.28] | -0.65 [-1.04, -0.29] | -0.35 [-0.65, -0.08] | same in every seed |
| ec_full -> jit_exact | -0.06 [-0.19, +0.06] | -0.05 [-0.18, +0.09] | +0.00 [-0.07, +0.10] | changes between seeds |
| jit_p50 -> ec_exact | -1.01 [-1.42, -0.62] | -0.60 [-1.01, -0.19] | -0.39 [-0.60, -0.11] | same in every seed |
| ec_full -> ec_exact | -0.40 [-0.51, -0.28] | +0.01 [-0.11, +0.12] | -0.03 [-0.05, +0.01] | changes between seeds |
| jit_exact -> ec_exact | -0.32 [-0.45, -0.19] | +0.04 [-0.09, +0.16] | -0.03 [-0.14, +0.04] | changes between seeds |
| hicache -> ec_exact | -26.96 [-30.53, -23.61] | -26.63 [-30.26, -23.25] | -21.28 [-39.97, -4.63] | same in every seed |

### Campaign 34: who waits (returning requests)

| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: share of returning requests that read from the SSD, minutes 90-110 | 0.313 | 0.306 | -2 % | 0.303 ± 0.039 [0.243, 0.338] | client.jsonl |
| hicache: their mean TTFT, s | 89.3 | 88.8 | -1 % | 81.7 ± 13.0 [63.8, 95.6] | client.jsonl |
| hicache: mean TTFT of the others, s | 90.6 | 88.2 | -3 % | 77.9 ± 16.0 [53.2, 92.9] | client.jsonl |
| hicache: mean TTFT of requests sent after a wait of 0-1 s | 29.9 | 29.7 | -1 % | 26.8 ± 15.7 [6.9, 49.1] | client.jsonl |
| hicache: mean TTFT of requests sent after a wait of 1-10 s | 31.8 | 31.5 | -1 % | 28.0 ± 19.4 [6.8, 58.3] | client.jsonl |
| hicache: mean TTFT of requests sent after a wait of 10-60 s | 33.9 | 33.6 | -1 % | 29.8 ± 18.3 [9.5, 57.6] | client.jsonl |
| hicache: mean TTFT of requests sent after a wait of 60-300 s | 39.4 | 38.6 | -2 % | 34.1 ± 19.8 [9.9, 62.3] | client.jsonl |
| hicache: mean TTFT of requests sent after a wait of 300-1800 s | 54.1 | 52.7 | -3 % | 44.2 ± 22.6 [15.5, 73.9] | client.jsonl |
| jit_p50: share of returning requests that read from the SSD, minutes 90-110 | 0.261 | 0.268 | +3 % | 0.283 ± 0.012 [0.268, 0.300] | client.jsonl |
| jit_p50: their mean TTFT, s | 69.6 | 66.6 | -4 % | 53.8 ± 14.2 [29.8, 66.6] | client.jsonl |
| jit_p50: mean TTFT of the others, s | 13.1 | 12.0 | -8 % | 12.6 ± 1.1 [11.0, 13.7] | client.jsonl |
| jit_p50: mean TTFT of requests sent after a wait of 0-1 s | 8.0 | 7.7 | -3 % | 7.9 ± 2.6 [4.2, 11.0] | client.jsonl |
| jit_p50: mean TTFT of requests sent after a wait of 1-10 s | 11.8 | 11.6 | -2 % | 11.8 ± 5.9 [4.6, 20.7] | client.jsonl |
| jit_p50: mean TTFT of requests sent after a wait of 10-60 s | 17.7 | 17.4 | -2 % | 18.8 ± 11.4 [6.0, 36.8] | client.jsonl |
| jit_p50: mean TTFT of requests sent after a wait of 60-300 s | 23.0 | 21.5 | -7 % | 22.6 ± 13.4 [7.1, 42.9] | client.jsonl |
| jit_p50: mean TTFT of requests sent after a wait of 300-1800 s | 33.7 | 31.9 | -5 % | 30.3 ± 17.6 [9.2, 56.4] | client.jsonl |
| jit_exact: share of returning requests that read from the SSD, minutes 90-110 | 0.293 | 0.312 | +6 % | 0.285 ± 0.074 [0.154, 0.333] | client.jsonl |
| jit_exact: their mean TTFT, s | 50.4 | 48.8 | -3 % | 41.6 ± 5.0 [35.7, 48.8] | client.jsonl |
| jit_exact: mean TTFT of the others, s | 17.2 | 16.1 | -6 % | 17.1 ± 2.3 [15.0, 19.7] | client.jsonl |
| jit_exact: mean TTFT of requests sent after a wait of 0-1 s | 9.5 | 9.2 | -3 % | 9.4 ± 3.7 [4.3, 14.0] | client.jsonl |
| jit_exact: mean TTFT of requests sent after a wait of 1-10 s | 11.1 | 10.8 | -2 % | 11.3 ± 5.7 [4.5, 20.0] | client.jsonl |
| jit_exact: mean TTFT of requests sent after a wait of 10-60 s | 13.4 | 12.8 | -5 % | 15.5 ± 9.6 [5.6, 31.2] | client.jsonl |
| jit_exact: mean TTFT of requests sent after a wait of 60-300 s | 16.2 | 15.3 | -5 % | 17.0 ± 9.8 [6.2, 32.2] | client.jsonl |
| jit_exact: mean TTFT of requests sent after a wait of 300-1800 s | 19.5 | 19.0 | -2 % | 19.5 ± 10.5 [7.4, 35.6] | client.jsonl |
| ec_full: share of returning requests that read from the SSD, minutes 90-110 | 0.321 | 0.337 | +5 % | 0.303 ± 0.064 [0.192, 0.347] | client.jsonl |
| ec_full: their mean TTFT, s | 46.7 | 46.0 | -2 % | 38.5 ± 5.9 [29.9, 46.0] | client.jsonl |
| ec_full: mean TTFT of the others, s | 19.0 | 16.9 | -11 % | 17.6 ± 2.3 [15.3, 20.3] | client.jsonl |
| ec_full: mean TTFT of requests sent after a wait of 0-1 s | 9.7 | 9.6 | -2 % | 9.8 ± 3.9 [4.3, 14.8] | client.jsonl |
| ec_full: mean TTFT of requests sent after a wait of 1-10 s | 11.1 | 10.6 | -5 % | 11.0 ± 5.4 [4.4, 19.3] | client.jsonl |
| ec_full: mean TTFT of requests sent after a wait of 10-60 s | 12.9 | 12.5 | -3 % | 15.1 ± 9.3 [5.7, 30.3] | client.jsonl |
| ec_full: mean TTFT of requests sent after a wait of 60-300 s | 15.9 | 15.6 | -2 % | 16.8 ± 9.6 [6.1, 31.9] | client.jsonl |
| ec_full: mean TTFT of requests sent after a wait of 300-1800 s | 19.0 | 18.7 | -2 % | 19.4 ± 10.1 [7.9, 35.2] | client.jsonl |
| ec_exact: share of returning requests that read from the SSD, minutes 90-110 | 0.309 | 0.315 | +2 % | 0.303 ± 0.073 [0.177, 0.358] | client.jsonl |
| ec_exact: their mean TTFT, s | 48.5 | 47.5 | -2 % | 39.3 ± 6.0 [31.4, 47.5] | client.jsonl |
| ec_exact: mean TTFT of the others, s | 17.9 | 16.6 | -7 % | 17.5 ± 2.2 [15.3, 20.3] | client.jsonl |
| ec_exact: mean TTFT of requests sent after a wait of 0-1 s | 9.4 | 9.6 | +2 % | 9.9 ± 4.1 [4.3, 15.2] | client.jsonl |
| ec_exact: mean TTFT of requests sent after a wait of 1-10 s | 10.8 | 10.6 | -3 % | 10.9 ± 5.4 [4.4, 19.0] | client.jsonl |
| ec_exact: mean TTFT of requests sent after a wait of 10-60 s | 12.6 | 12.3 | -2 % | 14.8 ± 8.9 [5.4, 29.1] | client.jsonl |
| ec_exact: mean TTFT of requests sent after a wait of 60-300 s | 15.8 | 15.3 | -3 % | 16.5 ± 9.4 [5.9, 31.0] | client.jsonl |
| ec_exact: mean TTFT of requests sent after a wait of 300-1800 s | 18.5 | 18.7 | +2 % | 19.1 ± 10.3 [7.1, 34.9] | client.jsonl |

### Campaign 28 (Triton backend, same users, no fixes): requests sent before 6,600 s; the Triton costs come from the records and were not fitted here

| anchor | measured | simulated, seed 0 | relative error | simulated, seeds 0-4: mean ± sd [min, max] | measured from |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 115.69 | 121.35 | +5 % | 124.31 ± 13.64 [108.20, 142.95] | campaign28/compare_arms.out |
| hicache: p50 TTFT, s | 128.4 | 134.2 | +4 % | 98.9 ± 60.7 [29.5, 165.1] | campaign28/compare_arms.out |
| hicache: p90 TTFT, s | 245.7 | 264.8 | +8 % | 265.3 ± 15.5 [251.3, 291.3] | campaign28/compare_arms.out |
| hicache: requests per minute | 41.3 | 39.3 | -5 % | 37.4 ± 2.6 [33.2, 39.8] | campaign28/compare_arms.out |
| hicache: prompt tokens from SSD, share | 0.160 | 0.159 | -1 % | 0.149 ± 0.015 [0.131, 0.169] | campaign28/compare_arms.out |
| hicache: history recomputed, share | 0.401 | 0.391 | -2 % | 0.412 ± 0.065 [0.350, 0.520] | campaign28/compare_arms.out |
| jit_p50: mean TTFT, s | 49.38 | 50.17 | +2 % | 51.43 ± 5.37 [43.79, 57.82] | campaign28/compare_arms.out |
| jit_p50: p50 TTFT, s | 52.7 | 53.1 | +1 % | 55.4 ± 6.5 [46.9, 64.5] | campaign28/compare_arms.out |
| jit_p50: p90 TTFT, s | 86.2 | 84.4 | -2 % | 86.5 ± 9.4 [75.3, 101.2] | campaign28/compare_arms.out |
| jit_p50: requests per minute | 77.8 | 76.9 | -1 % | 74.2 ± 3.7 [71.2, 79.3] | campaign28/compare_arms.out |
| jit_p50: prompt tokens from SSD, share | 0.420 | 0.461 | +10 % | 0.466 ± 0.029 [0.420, 0.496] | campaign28/compare_arms.out |
| jit_p50: history recomputed, share | 0.012 | 0.004 | -65 % | 0.004 ± 0.001 [0.003, 0.005] | campaign28/compare_arms.out |
| jit_exact: mean TTFT, s | 49.79 | 49.62 | -0 % | 51.30 ± 5.39 [43.68, 57.79] | campaign28/compare_arms.out |
| jit_exact: p50 TTFT, s | 53.9 | 54.0 | +0 % | 56.2 ± 6.7 [47.6, 66.0] | campaign28/compare_arms.out |
| jit_exact: p90 TTFT, s | 85.9 | 81.8 | -5 % | 84.7 ± 9.7 [72.5, 99.2] | campaign28/compare_arms.out |
| jit_exact: requests per minute | 76.9 | 77.2 | +0 % | 74.1 ± 3.8 [70.9, 79.1] | campaign28/compare_arms.out |
| jit_exact: prompt tokens from SSD, share | 0.410 | 0.465 | +13 % | 0.467 ± 0.037 [0.406, 0.502] | campaign28/compare_arms.out |
| jit_exact: history recomputed, share | 0.014 | 0.003 | -76 % | 0.004 ± 0.001 [0.003, 0.005] | campaign28/compare_arms.out |
| ec_full: mean TTFT, s | 48.27 | 49.76 | +3 % | 51.10 ± 5.26 [43.55, 57.21] | campaign28/compare_arms.out |
| ec_full: p50 TTFT, s | 51.7 | 55.8 | +8 % | 57.5 ± 6.3 [48.5, 66.1] | campaign28/compare_arms.out |
| ec_full: p90 TTFT, s | 80.2 | 78.7 | -2 % | 82.2 ± 8.7 [72.0, 95.6] | campaign28/compare_arms.out |
| ec_full: requests per minute | 78.9 | 77.1 | -2 % | 74.3 ± 3.7 [71.4, 79.4] | campaign28/compare_arms.out |
| ec_full: prompt tokens from SSD, share | 0.430 | 0.461 | +7 % | 0.459 ± 0.035 [0.403, 0.500] | campaign28/compare_arms.out |
| ec_full: history recomputed, share | 0.004 | 0.002 | -60 % | 0.002 ± 0.001 [0.002, 0.003] | campaign28/compare_arms.out |
| ec_exact: mean TTFT, s | 48.56 | 50.02 | +3 % | 51.25 ± 5.30 [43.67, 57.58] | campaign28/compare_arms.out |
| ec_exact: p50 TTFT, s | 51.9 | 54.9 | +6 % | 57.4 ± 6.6 [48.9, 67.1] | campaign28/compare_arms.out |
| ec_exact: p90 TTFT, s | 81.8 | 80.9 | -1 % | 83.1 ± 8.5 [72.6, 95.9] | campaign28/compare_arms.out |
| ec_exact: requests per minute | 78.6 | 76.9 | -2 % | 74.1 ± 3.6 [71.0, 79.1] | campaign28/compare_arms.out |
| ec_exact: prompt tokens from SSD, share | 0.440 | 0.468 | +6 % | 0.461 ± 0.035 [0.404, 0.499] | campaign28/compare_arms.out |
| ec_exact: history recomputed, share | 0.005 | 0.003 | -43 % | 0.003 ± 0.001 [0.002, 0.004] | campaign28/compare_arms.out |

### Same users (seed 0), five replicas with up to 5 ms of send jitter: mean TTFT, s

| arm | fixes | replicas | min | max | max - min |
|---|---|---|---|---|---|
| hicache | on | 31.39, 31.76, 31.66, 31.79, 31.73 | 31.39 | 31.79 | 0.40 |
| jit_p50 | on | 11.17, 11.03, 11.14, 11.09, 11.00 | 11.00 | 11.17 | 0.17 |
| jit_exact | on | 10.62, 10.65, 10.61, 10.63, 10.64 | 10.61 | 10.65 | 0.03 |
| ec_full | on | 10.63, 10.56, 10.60, 10.65, 10.58 | 10.56 | 10.65 | 0.09 |
| ec_exact | on | 10.52, 10.56, 10.54, 10.55, 10.55 | 10.52 | 10.56 | 0.04 |
| hicache | off | 31.16, 31.27, 31.79, 31.97, 31.81 | 31.16 | 31.97 | 0.81 |
| jit_p50 | off | 14.85, 17.88, 18.67, 14.28, 14.12 | 14.12 | 18.67 | 4.55 |
| jit_exact | off | 13.35, 13.58, 13.44, 13.55, 13.60 | 13.35 | 13.60 | 0.25 |
| ec_full | off | 13.33, 13.62, 13.74, 13.66, 13.71 | 13.33 | 13.74 | 0.41 |
| ec_exact | off | 13.62, 13.50, 13.59, 13.57, 13.53 | 13.50 | 13.62 | 0.12 |
