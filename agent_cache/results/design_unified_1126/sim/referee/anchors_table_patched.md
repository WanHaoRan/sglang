
### Campaign 34 (fixes on): client side, requests sent before 6,600 s

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 31.88 | 31.57 | -1 % | 26.59 ± 13.31 [11.48, 36.57] | reproduced |
| hicache: mean TTFT, minutes 0-30 | 0.41 | 0.40 | -3 % | 0.81 ± 0.51 [0.23, 1.18] | reproduced |
| hicache: mean TTFT, minutes 30-60 | 30.53 | 30.63 | +0 % | 22.25 ± 21.16 [3.32, 45.10] | reproduced |
| hicache: mean TTFT, minutes 60-90 | 72.97 | 71.67 | -2 % | 60.87 ± 41.76 [12.70, 86.87] | reproduced |
| hicache: mean TTFT, minutes 90-110 | 90.14 | 88.36 | -2 % | 78.26 ± 7.08 [72.66, 86.21] | reproduced |
| hicache: p50 TTFT, s | 5.95 | 5.87 | -1 % | 7.37 ± 4.15 [2.98, 11.23] | reproduced |
| hicache: p90 TTFT, s | 85.62 | 84.64 | -1 % | 79.47 ± 16.07 [60.92, 88.76] | reproduced |
| hicache: requests per minute to 6,600 s | 90.6 | 90.9 | +0 % | 97.6 ± 15.5 [87.2, 115.5] | reproduced |
| hicache: requests per minute, minutes 90-109 | 55.3 | 55.6 | +0 % | 60.5 ± 5.8 [56.0, 67.0] | reproduced |
| hicache: users with a request | 448 | 448 | +0 % | 491 ± 88 [425, 590] | reproduced |
| hicache: prompt tokens from GPU, share | 0.302 | 0.304 | +1 % | 0.264 ± 0.014 [0.250, 0.277] | reproduced |
| hicache: from host, share | 0.241 | 0.238 | -2 % | 0.341 ± 0.130 [0.236, 0.487] | reproduced |
| hicache: from SSD, share | 0.121 | 0.120 | -0 % | 0.114 ± 0.030 [0.080, 0.135] | reproduced |
| hicache: from SSD, minutes 30-60 | 0.142 | 0.139 | -2 % | 0.114 ± 0.055 [0.057, 0.167] | reproduced |
| hicache: from SSD, minutes 60-90 | 0.242 | 0.244 | +1 % | 0.207 ± 0.081 [0.113, 0.254] | reproduced |
| hicache: from SSD, minutes 90-110 | 0.255 | 0.255 | -0 % | 0.262 ± 0.025 [0.236, 0.286] | reproduced |
| hicache: history recomputed, share | 0.2891 | 0.2917 | +1 % | 0.2265 ± 0.1220 [0.0935, 0.3332] | reproduced |
| hicache: history recomputed, M tokens, minutes 30-60 | 18.99 | 19.49 | +3 % | 10.55 ± 11.47 [0.10, 22.83] | reproduced |
| hicache: history recomputed, M tokens, minutes 60-90 | 28.28 | 28.43 | +1 % | 20.94 ± 12.23 [6.82, 28.14] | reproduced |
| hicache: history recomputed, M tokens, minutes 90-110 | 18.73 | 18.85 | +1 % | 20.29 ± 0.75 [19.60, 21.08] | reproduced |
| hicache: returning requests that read from the SSD, minutes 90-110 | 0.313 | 0.307 | -2 % | 0.302 ± 0.031 [0.267, 0.327] | reproduced |
| hicache: their mean TTFT, s | 89.3 | 88.3 | -1 % | 79.7 ± 7.0 [74.0, 87.5] | reproduced |
| hicache: the others' mean TTFT, s | 90.6 | 88.5 | -2 % | 77.7 ± 7.1 [72.1, 85.6] | reproduced |
| jit_p50: mean TTFT, s | 11.47 | 11.22 | -2 % | 12.98 ± 4.73 [7.57, 16.30] | reproduced |
| jit_p50: mean TTFT, minutes 0-30 | 0.41 | 0.39 | -6 % | 0.81 ± 0.53 [0.20, 1.18] | reproduced |
| jit_p50: mean TTFT, minutes 30-60 | 6.37 | 6.12 | -4 % | 10.23 ± 6.36 [3.42, 16.02] | reproduced |
| jit_p50: mean TTFT, minutes 60-90 | 17.80 | 17.26 | -3 % | 20.05 ± 9.15 [9.54, 26.17] | reproduced |
| jit_p50: mean TTFT, minutes 90-110 | 28.20 | 27.71 | -2 % | 23.80 ± 3.56 [20.16, 27.27] | reproduced |
| jit_p50: p50 TTFT, s | 6.38 | 6.24 | -2 % | 7.45 ± 2.51 [4.56, 8.94] | reproduced |
| jit_p50: p90 TTFT, s | 22.42 | 23.28 | +4 % | 41.22 ± 21.89 [17.38, 60.40] | reproduced |
| jit_p50: requests per minute to 6,600 s | 141.9 | 142.6 | +1 % | 136.0 ± 5.6 [130.7, 141.9] | reproduced |
| jit_p50: requests per minute, minutes 90-109 | 129.1 | 130.8 | +1 % | 141.0 ± 12.1 [127.1, 148.9] | reproduced |
| jit_p50: users with a request | 617 | 614 | -0 % | 629 ± 38 [593, 668] | reproduced |
| jit_p50: prompt tokens from GPU, share | 0.186 | 0.190 | +2 % | 0.189 ± 0.041 [0.157, 0.235] | reproduced |
| jit_p50: from host, share | 0.603 | 0.597 | -1 % | 0.580 ± 0.015 [0.565, 0.594] | reproduced |
| jit_p50: from SSD, share | 0.148 | 0.150 | +2 % | 0.162 ± 0.029 [0.128, 0.181] | reproduced |
| jit_p50: from SSD, minutes 30-60 | 0.126 | 0.123 | -2 % | 0.157 ± 0.076 [0.070, 0.211] | reproduced |
| jit_p50: from SSD, minutes 60-90 | 0.228 | 0.232 | +2 % | 0.232 ± 0.036 [0.192, 0.259] | reproduced |
| jit_p50: from SSD, minutes 90-110 | 0.246 | 0.254 | +3 % | 0.246 ± 0.022 [0.221, 0.261] | reproduced |
| jit_p50: history recomputed, share | 0.0053 | 0.0058 | +9 % | 0.0052 ± 0.0002 [0.0050, 0.0053] | reproduced |
| jit_p50: history recomputed, M tokens, minutes 30-60 | 0.26 | 0.25 | -3 % | 0.16 ± 0.02 [0.15, 0.18] | reproduced |
| jit_p50: history recomputed, M tokens, minutes 60-90 | 0.60 | 0.81 | +35 % | 0.75 ± 0.08 [0.66, 0.83] | roughly |
| jit_p50: history recomputed, M tokens, minutes 90-110 | 1.07 | 1.06 | -1 % | 0.81 ± 0.08 [0.72, 0.87] | reproduced |
| jit_p50: returning requests that read from the SSD, minutes 90-110 | 0.261 | 0.258 | -1 % | 0.245 ± 0.025 [0.218, 0.266] | reproduced |
| jit_p50: their mean TTFT, s | 69.6 | 70.0 | +1 % | 61.3 ± 17.3 [41.5, 73.1] | reproduced |
| jit_p50: the others' mean TTFT, s | 13.1 | 12.6 | -4 % | 11.3 ± 1.4 [9.8, 12.2] | reproduced |
| jit_exact: mean TTFT, s | 10.92 | 10.57 | -3 % | 12.73 ± 4.89 [7.15, 16.23] | reproduced |
| jit_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 | -10 % | 0.78 ± 0.51 [0.20, 1.17] | roughly |
| jit_exact: mean TTFT, minutes 30-60 | 6.13 | 5.84 | -5 % | 10.02 ± 6.32 [3.27, 15.78] | reproduced |
| jit_exact: mean TTFT, minutes 60-90 | 16.60 | 15.90 | -4 % | 19.45 ± 9.11 [8.97, 25.41] | reproduced |
| jit_exact: mean TTFT, minutes 90-110 | 27.03 | 26.82 | -1 % | 24.17 ± 4.72 [19.35, 28.79] | reproduced |
| jit_exact: p50 TTFT, s | 8.24 | 7.63 | -7 % | 10.49 ± 5.13 [4.59, 13.99] | reproduced |
| jit_exact: p90 TTFT, s | 22.26 | 22.07 | -1 % | 30.40 ± 10.32 [18.58, 37.66] | reproduced |
| jit_exact: requests per minute to 6,600 s | 141.8 | 142.2 | +0 % | 135.0 ± 5.7 [129.2, 140.6] | reproduced |
| jit_exact: requests per minute, minutes 90-109 | 132.1 | 134.1 | +1 % | 136.6 ± 14.0 [120.6, 146.4] | reproduced |
| jit_exact: users with a request | 622 | 623 | +0 % | 630 ± 38 [595, 671] | reproduced |
| jit_exact: prompt tokens from GPU, share | 0.205 | 0.210 | +3 % | 0.202 ± 0.050 [0.163, 0.258] | reproduced |
| jit_exact: from host, share | 0.620 | 0.614 | -1 % | 0.592 ± 0.009 [0.587, 0.603] | reproduced |
| jit_exact: from SSD, share | 0.113 | 0.113 | +0 % | 0.137 ± 0.061 [0.067, 0.183] | reproduced |
| jit_exact: from SSD, minutes 30-60 | 0.032 | 0.032 | +1 % | 0.081 ± 0.066 [0.022, 0.153] | reproduced |
| jit_exact: from SSD, minutes 60-90 | 0.186 | 0.179 | -4 % | 0.210 ± 0.129 [0.061, 0.288] | reproduced |
| jit_exact: from SSD, minutes 90-110 | 0.274 | 0.289 | +5 % | 0.267 ± 0.047 [0.218, 0.313] | reproduced |
| jit_exact: history recomputed, share | 0.0050 | 0.0049 | -2 % | 0.0048 ± 0.0006 [0.0040, 0.0052] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -6 % | 0.11 ± 0.01 [0.10, 0.11] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 60-90 | 0.64 | 0.59 | -7 % | 0.69 ± 0.24 [0.43, 0.90] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 90-110 | 1.06 | 1.07 | +1 % | 0.78 ± 0.08 [0.71, 0.87] | reproduced |
| jit_exact: returning requests that read from the SSD, minutes 90-110 | 0.293 | 0.318 | +8 % | 0.273 ± 0.042 [0.233, 0.316] | reproduced |
| jit_exact: their mean TTFT, s | 50.4 | 47.5 | -6 % | 41.0 ± 11.8 [27.4, 48.3] | reproduced |
| jit_exact: the others' mean TTFT, s | 17.2 | 17.0 | -1 % | 17.4 ± 2.3 [15.3, 19.9] | reproduced |
| ec_full: mean TTFT, s | 10.96 | 10.63 | -3 % | 12.72 ± 4.81 [7.21, 16.12] | reproduced |
| ec_full: mean TTFT, minutes 0-30 | 0.41 | 0.39 | -5 % | 0.80 ± 0.51 [0.21, 1.16] | reproduced |
| ec_full: mean TTFT, minutes 30-60 | 6.18 | 5.91 | -4 % | 10.10 ± 6.29 [3.35, 15.80] | reproduced |
| ec_full: mean TTFT, minutes 60-90 | 16.36 | 15.90 | -3 % | 19.43 ± 8.95 [9.14, 25.38] | reproduced |
| ec_full: mean TTFT, minutes 90-110 | 27.97 | 26.88 | -4 % | 24.00 ± 4.46 [19.31, 28.18] | reproduced |
| ec_full: p50 TTFT, s | 8.34 | 7.83 | -6 % | 10.55 ± 5.06 [4.74, 14.02] | reproduced |
| ec_full: p90 TTFT, s | 24.01 | 23.63 | -2 % | 29.72 ± 9.74 [18.59, 36.68] | reproduced |
| ec_full: requests per minute to 6,600 s | 141.5 | 142.2 | +1 % | 134.9 ± 5.6 [129.2, 140.4] | reproduced |
| ec_full: requests per minute, minutes 90-109 | 128.3 | 132.5 | +3 % | 136.3 ± 13.5 [120.8, 145.7] | reproduced |
| ec_full: users with a request | 620 | 624 | +1 % | 631 ± 38 [594, 670] | reproduced |
| ec_full: prompt tokens from GPU, share | 0.194 | 0.200 | +3 % | 0.192 ± 0.041 [0.160, 0.238] | reproduced |
| ec_full: from host, share | 0.606 | 0.597 | -2 % | 0.581 ± 0.014 [0.571, 0.597] | reproduced |
| ec_full: from SSD, share | 0.138 | 0.141 | +2 % | 0.157 ± 0.056 [0.093, 0.199] | reproduced |
| ec_full: from SSD, minutes 30-60 | 0.079 | 0.082 | +4 % | 0.126 ± 0.068 [0.059, 0.195] | reproduced |
| ec_full: from SSD, minutes 60-90 | 0.212 | 0.215 | +2 % | 0.224 ± 0.104 [0.104, 0.285] | reproduced |
| ec_full: from SSD, minutes 90-110 | 0.284 | 0.288 | +2 % | 0.270 ± 0.043 [0.224, 0.310] | reproduced |
| ec_full: history recomputed, share | 0.0044 | 0.0048 | +9 % | 0.0043 ± 0.0005 [0.0038, 0.0048] | reproduced |
| ec_full: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -4 % | 0.10 ± 0.01 [0.09, 0.12] | reproduced |
| ec_full: history recomputed, M tokens, minutes 60-90 | 0.46 | 0.45 | -2 % | 0.58 ± 0.16 [0.42, 0.74] | reproduced |
| ec_full: history recomputed, M tokens, minutes 90-110 | 1.00 | 1.17 | +17 % | 0.73 ± 0.14 [0.61, 0.89] | roughly |
| ec_full: returning requests that read from the SSD, minutes 90-110 | 0.321 | 0.330 | +3 % | 0.288 ± 0.045 [0.243, 0.332] | reproduced |
| ec_full: their mean TTFT, s | 46.7 | 44.9 | -4 % | 38.0 ± 10.6 [25.9, 45.7] | reproduced |
| ec_full: the others' mean TTFT, s | 19.0 | 17.9 | -6 % | 18.0 ± 2.6 [15.9, 20.9] | reproduced |
| ec_exact: mean TTFT, s | 10.66 | 10.57 | -1 % | 12.67 ± 4.84 [7.13, 16.11] | reproduced |
| ec_exact: mean TTFT, minutes 0-30 | 0.39 | 0.36 | -8 % | 0.78 ± 0.52 [0.19, 1.14] | reproduced |
| ec_exact: mean TTFT, minutes 30-60 | 5.91 | 5.83 | -1 % | 10.07 ± 6.36 [3.28, 15.89] | reproduced |
| ec_exact: mean TTFT, minutes 60-90 | 15.76 | 15.79 | +0 % | 19.46 ± 9.09 [8.99, 25.42] | reproduced |
| ec_exact: mean TTFT, minutes 90-110 | 27.44 | 26.94 | -2 % | 23.69 ± 4.25 [19.27, 27.74] | reproduced |
| ec_exact: p50 TTFT, s | 7.87 | 7.71 | -2 % | 10.39 ± 4.98 [4.68, 13.84] | reproduced |
| ec_exact: p90 TTFT, s | 22.60 | 22.85 | +1 % | 29.89 ± 9.36 [19.27, 36.89] | reproduced |
| ec_exact: requests per minute to 6,600 s | 142.2 | 142.3 | +0 % | 134.9 ± 5.8 [129.1, 140.6] | reproduced |
| ec_exact: requests per minute, minutes 90-109 | 132.6 | 131.9 | -1 % | 136.4 ± 13.8 [120.6, 146.5] | reproduced |
| ec_exact: users with a request | 621 | 624 | +0 % | 632 ± 37 [596, 669] | reproduced |
| ec_exact: prompt tokens from GPU, share | 0.209 | 0.210 | +1 % | 0.201 ± 0.049 [0.163, 0.257] | reproduced |
| ec_exact: from host, share | 0.614 | 0.611 | -0 % | 0.587 ± 0.010 [0.581, 0.598] | reproduced |
| ec_exact: from SSD, share | 0.116 | 0.117 | +1 % | 0.143 ± 0.061 [0.074, 0.188] | reproduced |
| ec_exact: from SSD, minutes 30-60 | 0.047 | 0.043 | -7 % | 0.103 ± 0.067 [0.036, 0.171] | reproduced |
| ec_exact: from SSD, minutes 60-90 | 0.181 | 0.182 | +0 % | 0.215 ± 0.117 [0.080, 0.287] | reproduced |
| ec_exact: from SSD, minutes 90-110 | 0.279 | 0.288 | +3 % | 0.260 ± 0.050 [0.207, 0.306] | reproduced |
| ec_exact: history recomputed, share | 0.0039 | 0.0043 | +12 % | 0.0044 ± 0.0003 [0.0040, 0.0046] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 30-60 | 0.13 | 0.11 | -12 % | 0.11 ± 0.02 [0.09, 0.13] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 60-90 | 0.47 | 0.46 | -2 % | 0.60 ± 0.14 [0.45, 0.72] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 90-110 | 0.79 | 0.99 | +26 % | 0.73 ± 0.02 [0.72, 0.75] | roughly |
| ec_exact: returning requests that read from the SSD, minutes 90-110 | 0.309 | 0.324 | +5 % | 0.282 ± 0.041 [0.240, 0.321] | reproduced |
| ec_exact: their mean TTFT, s | 48.5 | 47.4 | -2 % | 38.6 ± 11.7 [25.2, 46.5] | reproduced |
| ec_exact: the others' mean TTFT, s | 17.9 | 17.0 | -5 % | 17.4 ± 2.4 [15.0, 19.8] | reproduced |

### Campaign 34 (fixes on): server side

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| hicache: running requests, median, minutes 60-90 | 22.0 | 21.0 | -5 % | 20.7 ± 1.2 [20.0, 22.0] | reproduced |
| hicache: running requests, median, minutes 90-110 | 20.0 | 19.0 | -5 % | 20.8 ± 1.3 [19.5, 22.0] | reproduced |
| hicache: running requests, mean, minutes 90-110 | 19.9 | 19.3 | -3 % | 21.1 ± 1.4 [19.7, 22.5] | reproduced |
| hicache: queued requests, median, minutes 90-110 | 83 | 83 | +0 % | 77 ± 9 [66, 84] | reproduced |
| hicache: queued requests, median, minutes 60-90 | 71 | 70 | -1 % | 60 ± 36 [20, 84] | reproduced |
| hicache: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.978 ± 0.003 [0.976, 0.981] | reproduced |
| hicache: read thread busy share, minutes 30-60 | 0.344 | 0.375 | +9 % | 0.286 ± 0.095 [0.182, 0.369] | reproduced |
| hicache: write thread busy share, minutes 30-60 | 0.282 | 0.277 | -2 % | 0.298 ± 0.043 [0.249, 0.327] | reproduced |
| hicache: read thread busy share, minutes 60-90 | 0.381 | 0.442 | +16 % | 0.400 ± 0.067 [0.326, 0.456] | roughly |
| hicache: write thread busy share, minutes 60-90 | 0.223 | 0.217 | -3 % | 0.247 ± 0.078 [0.186, 0.336] | reproduced |
| hicache: read thread busy share, minutes 90-110 | 0.354 | 0.396 | +12 % | 0.473 ± 0.033 [0.437, 0.501] | roughly |
| hicache: write thread busy share, minutes 90-110 | 0.186 | 0.162 | -13 % | 0.201 ± 0.018 [0.180, 0.215] | roughly |
| hicache: first cleaner pass, s | none | none |  | 5010 ± 0 [5010, 5010] | reproduced |
| hicache: cleaner passes before 6,600 s | 0 | 0 |  | 1 ± 1 [0, 2] | reproduced |
| hicache: SSD reads in the run | 2162 | 2167 | +0 % | 1943 ± 335 [1568, 2215] | reproduced |
| hicache: tokens read from the SSD, M (pages x 64) | 48.3 | 48.3 | +0 % | 44.5 ± 7.6 [35.8, 49.3] | reproduced |
| hicache: tokens written to the SSD, M (submitted, incl. skipped) | 94.7 | 19.9 | -79 % | 22.6 ± 3.8 [19.4, 26.8] | not_reproduced |
| hicache: host mirror, median of minutes 60-110, K | 631 | 614 | -3 % | 619 ± 12 [609, 632] | reproduced |
| hicache: host free, median, K | 5.3 | 12.2 | +130 % | 12.7 ± 1.7 [10.7, 14.1] | not_reproduced |
| hicache: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| hicache: pinned tokens, p90, K | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| hicache: pending writes (census 'pending_write'), p90, K | 8.1 | 10.5 | +30 % | 8.2 ± 1.7 [6.3, 9.5] | roughly |
| hicache: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 0.0 | 0.1 | (measured 0) | 0.2 ± 0.2 [0.0, 0.4] | not_reproduced |
| jit_p50: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| jit_p50: running requests, median, minutes 60-90 | 21.0 | 21.0 | +0 % | 20.7 ± 0.6 [20.0, 21.0] | reproduced |
| jit_p50: running requests, median, minutes 90-110 | 20.0 | 19.0 | -5 % | 20.7 ± 1.2 [20.0, 22.0] | reproduced |
| jit_p50: running requests, mean, minutes 90-110 | 19.7 | 19.6 | -1 % | 20.8 ± 0.7 [20.1, 21.6] | reproduced |
| jit_p50: queued requests, median, minutes 90-110 | 61 | 60 | -2 % | 56 ± 5 [51, 61] | reproduced |
| jit_p50: queued requests, median, minutes 60-90 | 42 | 42 | +0 % | 44 ± 21 [20, 58] | reproduced |
| jit_p50: GPU pool used, median, minutes 90-110 | 0.970 | 0.974 | +0 % | 0.977 ± 0.002 [0.975, 0.979] | reproduced |
| jit_p50: read thread busy share, minutes 30-60 | 0.551 | 0.541 | -2 % | 0.606 ± 0.298 [0.283, 0.872] | reproduced |
| jit_p50: write thread busy share, minutes 30-60 | 0.528 | 0.494 | -6 % | 0.554 ± 0.162 [0.381, 0.702] | reproduced |
| jit_p50: read thread busy share, minutes 60-90 | 0.968 | 0.956 | -1 % | 0.916 ± 0.138 [0.757, 0.996] | reproduced |
| jit_p50: write thread busy share, minutes 60-90 | 0.741 | 0.669 | -10 % | 0.681 ± 0.074 [0.596, 0.732] | reproduced |
| jit_p50: read thread busy share, minutes 90-110 | 0.999 | 0.997 | -0 % | 0.997 ± 0.003 [0.993, 0.999] | reproduced |
| jit_p50: write thread busy share, minutes 90-110 | 0.734 | 0.650 | -11 % | 0.705 ± 0.023 [0.682, 0.729] | roughly |
| jit_p50: first cleaner pass, s | 5301 | 5220 | -2 % | 5120 ± 121 [5010, 5250] | reproduced |
| jit_p50: cleaner passes before 6,600 s | 2 | 3 | +50 % | 3 ± 0 [3, 3] | roughly |
| jit_p50: reads refused host room before 6,600 s | 0 | 21 | (measured 0) | 5 ± 4 [0, 8] | not_reproduced |
| jit_p50: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| jit_p50: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| jit_p50: write hold switched on, times | 30 | 75 | +150 % | 80 ± 8 [71, 87] | not_reproduced |
| jit_p50: reads staged before arrival | 1208 | 1229 | +2 % | 1148 ± 83 [1056, 1216] | reproduced |
| jit_p50: SSD reads in the run | 3268 | 3363 | +3 % | 3390 ± 413 [2936, 3743] | reproduced |
| jit_p50: tokens read from the SSD, M (pages x 64) | 82.5 | 84.0 | +2 % | 83.9 ± 13.8 [68.4, 94.9] | reproduced |
| jit_p50: tokens written to the SSD, M (submitted, incl. skipped) | 31.9 | 30.2 | -5 % | 31.0 ± 0.6 [30.4, 31.6] | reproduced |
| jit_p50: host mirror, median of minutes 60-110, K | 646 | 647 | +0 % | 643 ± 1 [642, 645] | reproduced |
| jit_p50: host free, median, K | 17.7 | 13.5 | -24 % | 11.6 ± 0.9 [10.6, 12.4] | roughly |
| jit_p50: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 153 | 169 | +10 % | 175 ± 56 [141, 239] | reproduced |
| jit_p50: pinned tokens, p90, K | 319 | 303 | -5 % | 316 ± 99 [255, 430] | reproduced |
| jit_p50: pending writes (census 'pending_write'), p90, K | 15.4 | 42.3 | +175 % | 40.1 ± 4.7 [34.9, 43.9] | not_reproduced |
| jit_p50: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 94.6 | 10.2 | -89 % | 11.6 ± 3.5 [7.8, 14.7] | not_reproduced |
| jit_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| jit_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 21.0 ± 1.0 [20.0, 22.0] | reproduced |
| jit_exact: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 20.2 ± 0.8 [19.5, 21.0] | reproduced |
| jit_exact: running requests, mean, minutes 90-110 | 20.0 | 19.8 | -1 % | 20.5 ± 1.1 [19.7, 21.8] | reproduced |
| jit_exact: queued requests, median, minutes 90-110 | 60 | 58 | -3 % | 55 ± 6 [49, 59] | reproduced |
| jit_exact: queued requests, median, minutes 60-90 | 37 | 36 | -3 % | 41 ± 20 [18, 56] | reproduced |
| jit_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.978 ± 0.001 [0.976, 0.978] | reproduced |
| jit_exact: read thread busy share, minutes 30-60 | 0.094 | 0.112 | +19 % | 0.256 ± 0.221 [0.076, 0.503] | roughly |
| jit_exact: write thread busy share, minutes 30-60 | 0.273 | 0.288 | +6 % | 0.369 ± 0.111 [0.284, 0.495] | reproduced |
| jit_exact: read thread busy share, minutes 60-90 | 0.570 | 0.557 | -2 % | 0.692 ± 0.443 [0.182, 0.989] | reproduced |
| jit_exact: write thread busy share, minutes 60-90 | 0.513 | 0.458 | -11 % | 0.564 ± 0.228 [0.302, 0.726] | roughly |
| jit_exact: read thread busy share, minutes 90-110 | 1.000 | 0.996 | -0 % | 0.897 ± 0.168 [0.703, 0.995] | reproduced |
| jit_exact: write thread busy share, minutes 90-110 | 0.737 | 0.658 | -11 % | 0.648 ± 0.055 [0.586, 0.691] | roughly |
| jit_exact: first cleaner pass, s | 5272 | 5190 | -2 % | 5100 ± 131 [5010, 5250] | reproduced |
| jit_exact: cleaner passes before 6,600 s | 2 | 3 | +50 % | 3 ± 0 [3, 3] | roughly |
| jit_exact: reads refused host room before 6,600 s | 0 | 0 |  | 3 ± 5 [0, 9] | reproduced |
| jit_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| jit_exact: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| jit_exact: write hold switched on, times | 28 | 68 | +143 % | 66 ± 14 [52, 79] | not_reproduced |
| jit_exact: reads staged before arrival | 99 | 99 | +0 % | 93 ± 30 [71, 127] | reproduced |
| jit_exact: SSD reads in the run | 2221 | 2344 | +6 % | 2577 ± 895 [1596, 3348] | reproduced |
| jit_exact: tokens read from the SSD, M (pages x 64) | 54.4 | 55.9 | +3 % | 62.1 ± 24.8 [34.3, 81.7] | reproduced |
| jit_exact: tokens written to the SSD, M (submitted, incl. skipped) | 31.8 | 30.5 | -4 % | 31.1 ± 0.7 [30.4, 31.7] | reproduced |
| jit_exact: host mirror, median of minutes 60-110, K | 647 | 647 | +0 % | 643 ± 1 [642, 644] | reproduced |
| jit_exact: host free, median, K | 15.9 | 11.8 | -26 % | 9.0 ± 1.7 [7.0, 10.0] | roughly |
| jit_exact: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| jit_exact: pinned tokens, p90, K | 58 | 57 | -2 % | 51 ± 2 [50, 53] | reproduced |
| jit_exact: pending writes (census 'pending_write'), p90, K | 14.8 | 35.0 | +136 % | 35.3 ± 10.3 [23.5, 42.8] | not_reproduced |
| jit_exact: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 65.1 | 5.5 | -92 % | 6.8 ± 4.8 [1.3, 10.2] | not_reproduced |
| ec_full: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.5 ± 0.5 [21.0, 22.0] | reproduced |
| ec_full: running requests, median, minutes 60-90 | 20.0 | 20.5 | +2 % | 20.7 ± 0.6 [20.0, 21.0] | reproduced |
| ec_full: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 20.3 ± 0.6 [20.0, 21.0] | reproduced |
| ec_full: running requests, mean, minutes 90-110 | 19.7 | 19.9 | +1 % | 20.5 ± 0.8 [19.8, 21.4] | reproduced |
| ec_full: queued requests, median, minutes 90-110 | 60 | 59 | -2 % | 54 ± 5 [49, 58] | reproduced |
| ec_full: queued requests, median, minutes 60-90 | 37 | 36 | -3 % | 42 ± 20 [19, 57] | reproduced |
| ec_full: GPU pool used, median, minutes 90-110 | 0.970 | 0.975 | +1 % | 0.979 ± 0.003 [0.975, 0.981] | reproduced |
| ec_full: read thread busy share, minutes 30-60 | 0.235 | 0.282 | +20 % | 0.391 ± 0.230 [0.189, 0.642] | roughly |
| ec_full: write thread busy share, minutes 30-60 | 0.348 | 0.357 | +3 % | 0.429 ± 0.119 [0.331, 0.561] | reproduced |
| ec_full: read thread busy share, minutes 60-90 | 0.664 | 0.693 | +4 % | 0.734 ± 0.369 [0.310, 0.978] | reproduced |
| ec_full: write thread busy share, minutes 60-90 | 0.572 | 0.536 | -6 % | 0.582 ± 0.202 [0.350, 0.719] | reproduced |
| ec_full: read thread busy share, minutes 90-110 | 0.999 | 0.993 | -1 % | 0.893 ± 0.157 [0.712, 0.993] | reproduced |
| ec_full: write thread busy share, minutes 90-110 | 0.736 | 0.667 | -9 % | 0.637 ± 0.060 [0.569, 0.681] | reproduced |
| ec_full: first cleaner pass, s | 5275 | 5190 | -2 % | 5100 ± 131 [5010, 5250] | reproduced |
| ec_full: cleaner passes before 6,600 s | 3 | 3 | +0 % | 3 ± 0 [3, 3] | reproduced |
| ec_full: reads refused host room before 6,600 s | 0 | 0 |  | 1 ± 2 [0, 3] | reproduced |
| ec_full: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| ec_full: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| ec_full: write hold switched on, times | 30 | 59 | +97 % | 70 ± 15 [54, 82] | not_reproduced |
| ec_full: reads staged before arrival | 1 | 1 | +0 % | 1 ± 1 [0, 1] | reproduced |
| ec_full: SSD reads in the run | 2678 | 2792 | +4 % | 2934 ± 856 [1981, 3640] | reproduced |
| ec_full: tokens read from the SSD, M (pages x 64) | 64.1 | 66.1 | +3 % | 68.6 ± 23.0 [42.8, 86.8] | reproduced |
| ec_full: tokens written to the SSD, M (submitted, incl. skipped) | 31.6 | 30.6 | -3 % | 31.2 ± 0.7 [30.4, 31.8] | reproduced |
| ec_full: host mirror, median of minutes 60-110, K | 647 | 648 | +0 % | 644 ± 0 [643, 644] | reproduced |
| ec_full: host free, median, K | 18.3 | 11.6 | -37 % | 9.6 ± 1.2 [8.9, 11.1] | roughly |
| ec_full: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 5 | (measured 0) | 0 ± 0 [0, 0] | not_reproduced |
| ec_full: pinned tokens, p90, K | 61 | 62 | +2 % | 51 ± 1 [50, 52] | reproduced |
| ec_full: pending writes (census 'pending_write'), p90, K | 12.1 | 34.9 | +189 % | 36.9 ± 11.4 [24.5, 46.8] | not_reproduced |
| ec_full: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 72.7 | 3.8 | -95 % | 7.2 ± 5.1 [1.7, 11.9] | not_reproduced |
| ec_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| ec_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 21.0 ± 1.0 [20.0, 22.0] | reproduced |
| ec_exact: running requests, median, minutes 90-110 | 20.0 | 19.0 | -5 % | 20.3 ± 0.6 [20.0, 21.0] | reproduced |
| ec_exact: running requests, mean, minutes 90-110 | 19.8 | 19.8 | +0 % | 20.5 ± 0.9 [19.8, 21.5] | reproduced |
| ec_exact: queued requests, median, minutes 90-110 | 60 | 59 | -2 % | 54 ± 5 [49, 58] | reproduced |
| ec_exact: queued requests, median, minutes 60-90 | 36 | 36 | +0 % | 42 ± 21 [18, 57] | reproduced |
| ec_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.980 ± 0.002 [0.979, 0.982] | reproduced |
| ec_exact: read thread busy share, minutes 30-60 | 0.143 | 0.151 | +6 % | 0.322 ± 0.222 [0.121, 0.560] | reproduced |
| ec_exact: write thread busy share, minutes 30-60 | 0.299 | 0.306 | +2 % | 0.398 ± 0.113 [0.301, 0.522] | reproduced |
| ec_exact: read thread busy share, minutes 60-90 | 0.555 | 0.572 | +3 % | 0.701 ± 0.400 [0.243, 0.978] | reproduced |
| ec_exact: write thread busy share, minutes 60-90 | 0.525 | 0.462 | -12 % | 0.567 ± 0.212 [0.325, 0.716] | roughly |
| ec_exact: read thread busy share, minutes 90-110 | 1.000 | 0.997 | -0 % | 0.883 ± 0.190 [0.664, 0.993] | reproduced |
| ec_exact: write thread busy share, minutes 90-110 | 0.742 | 0.671 | -10 % | 0.632 ± 0.067 [0.555, 0.675] | reproduced |
| ec_exact: first cleaner pass, s | 5252 | 5190 | -1 % | 5100 ± 131 [5010, 5250] | reproduced |
| ec_exact: cleaner passes before 6,600 s | 3 | 3 | +0 % | 3 ± 0 [3, 3] | reproduced |
| ec_exact: reads refused host room before 6,600 s | 0 | 0 |  | 7 ± 10 [0, 18] | reproduced |
| ec_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| ec_exact: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| ec_exact: write hold switched on, times | 29 | 57 | +97 % | 66 ± 14 [54, 81] | not_reproduced |
| ec_exact: reads staged before arrival | 39 | 48 | +23 % | 51 ± 47 [22, 106] | roughly |
| ec_exact: SSD reads in the run | 2335 | 2393 | +2 % | 2748 ± 891 [1752, 3471] | reproduced |
| ec_exact: tokens read from the SSD, M (pages x 64) | 55.3 | 56.6 | +2 % | 63.8 ± 24.3 [36.4, 82.6] | reproduced |
| ec_exact: tokens written to the SSD, M (submitted, incl. skipped) | 31.7 | 30.6 | -3 % | 31.2 ± 0.7 [30.5, 31.8] | reproduced |
| ec_exact: host mirror, median of minutes 60-110, K | 647 | 647 | -0 % | 644 ± 0 [644, 644] | reproduced |
| ec_exact: host free, median, K | 15.7 | 10.8 | -31 % | 9.7 ± 3.3 [6.1, 12.7] | roughly |
| ec_exact: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| ec_exact: pinned tokens, p90, K | 55 | 59 | +7 % | 50 ± 2 [48, 51] | reproduced |
| ec_exact: pending writes (census 'pending_write'), p90, K | 16.6 | 29.2 | +76 % | 36.8 ± 12.5 [22.6, 45.9] | not_reproduced |
| ec_exact: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 67.1 | 3.3 | -95 % | 8.9 ± 6.5 [1.5, 13.8] | not_reproduced |

### Campaign 32 (fixes off): client side, requests sent before 6,600 s

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 33.52 | 31.96 | -5 % | 28.09 ± 14.53 [11.56, 38.88] | reproduced |
| hicache: mean TTFT, minutes 0-30 | 0.42 | 0.40 | -4 % | 0.81 ± 0.51 [0.23, 1.18] | reproduced |
| hicache: mean TTFT, minutes 30-60 | 29.64 | 29.22 | -1 % | 22.50 ± 21.63 [3.32, 45.95] | reproduced |
| hicache: mean TTFT, minutes 60-90 | 83.14 | 71.56 | -14 % | 71.96 ± 51.93 [12.24, 106.45] | roughly |
| hicache: mean TTFT, minutes 90-110 | 111.82 | 105.19 | -6 % | 87.24 ± 11.79 [79.10, 100.76] | reproduced |
| hicache: p50 TTFT, s | 5.01 | 5.74 | +14 % | 6.35 ± 3.06 [2.95, 8.88] | roughly |
| hicache: p90 TTFT, s | 99.81 | 93.56 | -6 % | 85.81 ± 32.89 [47.84, 105.65] | reproduced |
| hicache: requests per minute to 6,600 s | 87.8 | 90.2 | +3 % | 95.4 ± 17.3 [84.1, 115.2] | reproduced |
| hicache: requests per minute, minutes 90-109 | 46.9 | 49.0 | +4 % | 56.1 ± 5.4 [52.4, 62.4] | reproduced |
| hicache: users with a request | 433 | 445 | +3 % | 483 ± 93 [417, 590] | reproduced |
| hicache: prompt tokens from GPU, share | 0.309 | 0.306 | -1 % | 0.271 ± 0.009 [0.261, 0.278] | reproduced |
| hicache: from host, share | 0.286 | 0.298 | +4 % | 0.372 ± 0.116 [0.283, 0.503] | reproduced |
| hicache: from SSD, share | 0.045 | 0.053 | +18 % | 0.058 ± 0.007 [0.050, 0.062] | roughly |
| hicache: from SSD, minutes 30-60 | 0.077 | 0.077 | +1 % | 0.079 ± 0.020 [0.057, 0.096] | reproduced |
| hicache: from SSD, minutes 60-90 | 0.060 | 0.100 | +68 % | 0.085 ± 0.012 [0.074, 0.098] | not_reproduced |
| hicache: from SSD, minutes 90-110 | 0.059 | 0.064 | +9 % | 0.073 ± 0.015 [0.056, 0.086] | reproduced |
| hicache: history recomputed, share | 0.3141 | 0.2963 | -6 % | 0.2461 ± 0.1372 [0.0948, 0.3624] | reproduced |
| hicache: history recomputed, M tokens, minutes 30-60 | 19.02 | 18.98 | -0 % | 10.73 ± 11.86 [0.10, 23.53] | reproduced |
| hicache: history recomputed, M tokens, minutes 60-90 | 29.81 | 27.49 | -8 % | 22.25 ± 14.07 [6.01, 30.40] | reproduced |
| hicache: history recomputed, M tokens, minutes 90-110 | 20.65 | 20.91 | +1 % | 21.24 ± 0.05 [21.19, 21.30] | reproduced |
| hicache: returning requests that read from the SSD, minutes 90-110 | 0.151 | 0.123 | -18 % | 0.125 ± 0.017 [0.107, 0.138] | roughly |
| hicache: their mean TTFT, s | 112.1 | 106.8 | -5 % | 88.7 ± 12.0 [81.2, 102.6] | reproduced |
| hicache: the others' mean TTFT, s | 111.8 | 105.0 | -6 % | 87.1 ± 11.7 [78.8, 100.4] | reproduced |
| jit_p50: mean TTFT, s | 17.97 | 14.19 | -21 % | 21.05 ± 9.85 [10.75, 30.38] | roughly |
| jit_p50: mean TTFT, minutes 0-30 | 0.42 | 0.39 | -7 % | 0.80 ± 0.53 [0.20, 1.18] | reproduced |
| jit_p50: mean TTFT, minutes 30-60 | 7.27 | 6.16 | -15 % | 14.89 ± 13.43 [3.38, 29.65] | roughly |
| jit_p50: mean TTFT, minutes 60-90 | 40.08 | 18.22 | -55 % | 36.32 ± 24.40 [12.77, 61.50] | not_reproduced |
| jit_p50: mean TTFT, minutes 90-110 | 69.30 | 86.55 | +25 % | 73.16 ± 17.11 [54.30, 87.69] | roughly |
| jit_p50: p50 TTFT, s | 6.47 | 5.42 | -16 % | 8.47 ± 4.59 [3.47, 12.51] | roughly |
| jit_p50: p90 TTFT, s | 63.89 | 55.47 | -13 % | 68.63 ± 20.97 [44.46, 82.06] | roughly |
| jit_p50: requests per minute to 6,600 s | 114.5 | 127.6 | +11 % | 108.5 ± 9.3 [101.0, 118.9] | not_reproduced |
| jit_p50: requests per minute, minutes 90-109 | 65.2 | 54.1 | -17 % | 65.3 ± 7.8 [57.2, 72.7] | not_reproduced |
| jit_p50: users with a request | 522 | 560 | +7 % | 532 ± 74 [464, 611] | roughly |
| jit_p50: prompt tokens from GPU, share | 0.233 | 0.212 | -9 % | 0.233 ± 0.027 [0.212, 0.263] | reproduced |
| jit_p50: from host, share | 0.448 | 0.525 | +17 % | 0.412 ± 0.089 [0.312, 0.481] | roughly |
| jit_p50: from SSD, share | 0.150 | 0.147 | -2 % | 0.161 ± 0.035 [0.120, 0.183] | reproduced |
| jit_p50: from SSD, minutes 30-60 | 0.142 | 0.126 | -11 % | 0.164 ± 0.087 [0.068, 0.237] | roughly |
| jit_p50: from SSD, minutes 60-90 | 0.284 | 0.232 | -18 % | 0.261 ± 0.058 [0.201, 0.316] | roughly |
| jit_p50: from SSD, minutes 90-110 | 0.302 | 0.354 | +17 % | 0.326 ± 0.025 [0.307, 0.353] | roughly |
| jit_p50: history recomputed, share | 0.1138 | 0.0603 | -47 % | 0.1346 ± 0.0836 [0.0711, 0.2293] | not_reproduced |
| jit_p50: history recomputed, M tokens, minutes 30-60 | 1.07 | 0.21 | -81 % | 4.61 ± 7.52 [0.13, 13.29] | not_reproduced |
| jit_p50: history recomputed, M tokens, minutes 60-90 | 16.31 | 3.21 | -80 % | 14.05 ± 9.73 [5.56, 24.68] | not_reproduced |
| jit_p50: history recomputed, M tokens, minutes 90-110 | 16.48 | 16.93 | +3 % | 16.66 ± 1.51 [15.47, 18.36] | reproduced |
| jit_p50: returning requests that read from the SSD, minutes 90-110 | 0.600 | 0.600 | -0 % | 0.538 ± 0.052 [0.487, 0.591] | reproduced |
| jit_p50: their mean TTFT, s | 71.3 | 87.1 | +22 % | 76.0 ± 16.4 [58.3, 90.6] | roughly |
| jit_p50: the others' mean TTFT, s | 66.2 | 86.0 | +30 % | 69.9 ± 17.4 [50.3, 83.7] | roughly |
| jit_exact: mean TTFT, s | 14.86 | 11.87 | -20 % | 18.23 ± 10.01 [7.65, 27.56] | roughly |
| jit_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 | -10 % | 0.78 ± 0.51 [0.20, 1.17] | roughly |
| jit_exact: mean TTFT, minutes 30-60 | 6.18 | 5.84 | -6 % | 11.72 ± 8.81 [3.29, 20.87] | reproduced |
| jit_exact: mean TTFT, minutes 60-90 | 21.75 | 15.97 | -27 % | 31.65 ± 26.11 [8.99, 60.20] | roughly |
| jit_exact: mean TTFT, minutes 90-110 | 79.46 | 43.12 | -46 % | 64.83 ± 35.25 [24.14, 86.08] | not_reproduced |
| jit_exact: p50 TTFT, s | 6.12 | 6.84 | +12 % | 9.97 ± 4.85 [4.39, 13.03] | roughly |
| jit_exact: p90 TTFT, s | 50.53 | 22.45 | -56 % | 53.93 ± 30.96 [18.56, 76.14] | not_reproduced |
| jit_exact: requests per minute to 6,600 s | 123.2 | 135.5 | +10 % | 116.5 ± 13.4 [106.2, 131.6] | not_reproduced |
| jit_exact: requests per minute, minutes 90-109 | 59.2 | 96.8 | +64 % | 80.4 ± 37.6 [56.5, 123.7] | not_reproduced |
| jit_exact: users with a request | 550 | 592 | +8 % | 562 ± 89 [482, 658] | roughly |
| jit_exact: prompt tokens from GPU, share | 0.239 | 0.220 | -8 % | 0.231 ± 0.027 [0.211, 0.262] | reproduced |
| jit_exact: from host, share | 0.535 | 0.587 | +10 % | 0.494 ± 0.103 [0.388, 0.594] | reproduced |
| jit_exact: from SSD, share | 0.098 | 0.106 | +8 % | 0.121 ± 0.053 [0.060, 0.152] | reproduced |
| jit_exact: from SSD, minutes 30-60 | 0.031 | 0.032 | +4 % | 0.075 ± 0.057 [0.022, 0.135] | reproduced |
| jit_exact: from SSD, minutes 60-90 | 0.192 | 0.185 | -4 % | 0.222 ± 0.142 [0.058, 0.312] | reproduced |
| jit_exact: from SSD, minutes 90-110 | 0.331 | 0.284 | -14 % | 0.290 ± 0.078 [0.200, 0.338] | roughly |
| jit_exact: history recomputed, share | 0.0727 | 0.0312 | -57 % | 0.0929 ± 0.0874 [0.0169, 0.1884] | not_reproduced |
| jit_exact: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -4 % | 2.42 ± 4.01 [0.11, 7.05] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 60-90 | 7.69 | 0.73 | -90 % | 9.29 ± 12.74 [0.43, 23.90] | not_reproduced |
| jit_exact: history recomputed, M tokens, minutes 90-110 | 15.72 | 10.35 | -34 % | 13.68 ± 7.52 [4.99, 18.27] | roughly |
| jit_exact: returning requests that read from the SSD, minutes 90-110 | 0.649 | 0.376 | -42 % | 0.466 ± 0.195 [0.240, 0.584] | not_reproduced |
| jit_exact: their mean TTFT, s | 79.7 | 65.0 | -18 % | 69.1 ± 29.5 [35.0, 86.6] | roughly |
| jit_exact: the others' mean TTFT, s | 78.9 | 29.6 | -63 % | 63.0 ± 36.7 [20.7, 85.6] | not_reproduced |
| ec_full: mean TTFT, s | 14.64 | 13.27 | -9 % | 15.57 ± 7.26 [7.19, 20.08] | reproduced |
| ec_full: mean TTFT, minutes 0-30 | 0.45 | 0.39 | -13 % | 0.80 ± 0.51 [0.21, 1.16] | roughly |
| ec_full: mean TTFT, minutes 30-60 | 6.42 | 5.88 | -8 % | 10.10 ± 6.36 [3.32, 15.93] | reproduced |
| ec_full: mean TTFT, minutes 60-90 | 20.01 | 16.05 | -20 % | 20.55 ± 10.00 [9.06, 27.31] | roughly |
| ec_full: mean TTFT, minutes 90-110 | 80.17 | 76.32 | -5 % | 63.86 ± 38.50 [19.41, 86.42] | reproduced |
| ec_full: p50 TTFT, s | 6.51 | 6.47 | -1 % | 9.99 ± 4.61 [4.67, 12.92] | reproduced |
| ec_full: p90 TTFT, s | 51.24 | 27.69 | -46 % | 47.47 ± 25.45 [18.17, 64.08] | not_reproduced |
| ec_full: requests per minute to 6,600 s | 125.0 | 129.5 | +4 % | 124.6 ± 11.8 [111.8, 135.0] | roughly |
| ec_full: requests per minute, minutes 90-109 | 58.9 | 61.0 | +4 % | 84.6 ± 49.7 [55.9, 142.1] | reproduced |
| ec_full: users with a request | 557 | 578 | +4 % | 595 ± 66 [549, 670] | roughly |
| ec_full: prompt tokens from GPU, share | 0.217 | 0.220 | +1 % | 0.208 ± 0.034 [0.175, 0.243] | reproduced |
| ec_full: from host, share | 0.534 | 0.543 | +2 % | 0.529 ± 0.055 [0.486, 0.591] | reproduced |
| ec_full: from SSD, share | 0.130 | 0.129 | -1 % | 0.152 ± 0.051 [0.094, 0.190] | reproduced |
| ec_full: from SSD, minutes 30-60 | 0.082 | 0.081 | -1 % | 0.127 ± 0.069 [0.059, 0.197] | reproduced |
| ec_full: from SSD, minutes 60-90 | 0.230 | 0.207 | -10 % | 0.226 ± 0.101 [0.109, 0.290] | reproduced |
| ec_full: from SSD, minutes 90-110 | 0.321 | 0.349 | +9 % | 0.306 ± 0.075 [0.220, 0.360] | reproduced |
| ec_full: history recomputed, share | 0.0634 | 0.0515 | -19 % | 0.0464 ± 0.0366 [0.0044, 0.0707] | roughly |
| ec_full: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -4 % | 0.11 ± 0.01 [0.09, 0.12] | reproduced |
| ec_full: history recomputed, M tokens, minutes 60-90 | 4.94 | 0.94 | -81 % | 2.55 ± 2.00 [0.42, 4.39] | not_reproduced |
| ec_full: history recomputed, M tokens, minutes 90-110 | 15.81 | 16.55 | +5 % | 11.60 ± 9.32 [0.88, 17.77] | reproduced |
| ec_full: returning requests that read from the SSD, minutes 90-110 | 0.639 | 0.571 | -11 % | 0.470 ± 0.190 [0.251, 0.587] | roughly |
| ec_full: their mean TTFT, s | 80.7 | 79.4 | -2 % | 67.5 ± 33.5 [28.9, 87.0] | reproduced |
| ec_full: the others' mean TTFT, s | 79.6 | 72.0 | -10 % | 62.2 ± 39.8 [16.2, 85.9] | reproduced |
| ec_exact: mean TTFT, s | 14.95 | 13.44 | -10 % | 18.82 ± 10.12 [7.36, 26.56] | roughly |
| ec_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 | -10 % | 0.78 ± 0.52 [0.19, 1.14] | reproduced |
| ec_exact: mean TTFT, minutes 30-60 | 6.39 | 5.83 | -9 % | 10.61 ± 7.13 [3.27, 17.51] | reproduced |
| ec_exact: mean TTFT, minutes 60-90 | 22.09 | 16.49 | -25 % | 37.34 ± 27.27 [8.94, 63.33] | roughly |
| ec_exact: mean TTFT, minutes 90-110 | 78.47 | 79.41 | +1 % | 64.41 ± 37.10 [21.58, 86.96] | reproduced |
| ec_exact: p50 TTFT, s | 6.21 | 6.16 | -1 % | 9.58 ± 4.28 [4.65, 12.41] | reproduced |
| ec_exact: p90 TTFT, s | 50.09 | 29.42 | -41 % | 57.85 ± 34.47 [18.05, 78.37] | not_reproduced |
| ec_exact: requests per minute to 6,600 s | 123.0 | 128.2 | +4 % | 114.8 ± 16.2 [102.8, 133.2] | roughly |
| ec_exact: requests per minute, minutes 90-109 | 59.9 | 59.1 | -1 % | 83.5 ± 43.0 [57.7, 133.1] | reproduced |
| ec_exact: users with a request | 547 | 566 | +3 % | 555 ± 95 [490, 664] | roughly |
| ec_exact: prompt tokens from GPU, share | 0.238 | 0.232 | -2 % | 0.235 ± 0.025 [0.213, 0.262] | reproduced |
| ec_exact: from host, share | 0.528 | 0.552 | +4 % | 0.472 ± 0.102 [0.395, 0.588] | reproduced |
| ec_exact: from SSD, share | 0.105 | 0.104 | -1 % | 0.131 ± 0.051 [0.072, 0.166] | reproduced |
| ec_exact: from SSD, minutes 30-60 | 0.048 | 0.044 | -9 % | 0.099 ± 0.062 [0.036, 0.160] | reproduced |
| ec_exact: from SSD, minutes 60-90 | 0.200 | 0.181 | -10 % | 0.238 ± 0.141 [0.077, 0.339] | reproduced |
| ec_exact: from SSD, minutes 90-110 | 0.330 | 0.351 | +7 % | 0.306 ± 0.080 [0.213, 0.353] | reproduced |
| ec_exact: history recomputed, share | 0.0727 | 0.0561 | -23 % | 0.1010 ± 0.0816 [0.0112, 0.1704] | roughly |
| ec_exact: history recomputed, M tokens, minutes 30-60 | 0.13 | 0.11 | -11 % | 1.37 ± 2.21 [0.09, 3.92] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 60-90 | 7.96 | 2.37 | -70 % | 13.43 ± 12.10 [0.42, 24.36] | not_reproduced |
| ec_exact: history recomputed, M tokens, minutes 90-110 | 15.42 | 16.51 | +7 % | 12.42 ± 8.03 [3.14, 17.31] | reproduced |
| ec_exact: returning requests that read from the SSD, minutes 90-110 | 0.645 | 0.582 | -10 % | 0.480 ± 0.190 [0.262, 0.616] | reproduced |
| ec_exact: their mean TTFT, s | 79.7 | 82.0 | +3 % | 68.8 ± 31.2 [32.7, 87.3] | reproduced |
| ec_exact: the others' mean TTFT, s | 76.9 | 75.8 | -1 % | 62.4 ± 38.9 [17.5, 86.6] | reproduced |

### Campaign 32 (fixes off): server side

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: running requests, median, minutes 30-60 | 22.0 | 21.0 | -5 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| hicache: running requests, median, minutes 60-90 | 22.0 | 21.0 | -5 % | 20.3 ± 1.5 [19.0, 22.0] | reproduced |
| hicache: running requests, median, minutes 90-110 | 20.0 | 19.0 | -5 % | 20.7 ± 1.5 [19.0, 22.0] | reproduced |
| hicache: running requests, mean, minutes 90-110 | 19.9 | 18.9 | -5 % | 20.8 ± 1.3 [19.5, 22.1] | reproduced |
| hicache: queued requests, median, minutes 90-110 | 86 | 84 | -2 % | 78 ± 9 [68, 84] | reproduced |
| hicache: queued requests, median, minutes 60-90 | 72 | 70 | -3 % | 62 ± 37 [20, 87] | reproduced |
| hicache: GPU pool used, median, minutes 90-110 | 0.970 | 0.977 | +1 % | 0.979 ± 0.004 [0.976, 0.983] | reproduced |
| hicache: read thread busy share, minutes 30-60 | 0.374 | 0.414 | +11 % | 0.303 ± 0.118 [0.182, 0.418] | roughly |
| hicache: write thread busy share, minutes 30-60 | 0.743 | 0.752 | +1 % | 0.562 ± 0.260 [0.327, 0.841] | reproduced |
| hicache: read thread busy share, minutes 60-90 | 0.469 | 0.492 | +5 % | 0.440 ± 0.092 [0.333, 0.497] | reproduced |
| hicache: write thread busy share, minutes 60-90 | 1.000 | 1.000 | +0 % | 0.826 ± 0.301 [0.479, 1.000] | reproduced |
| hicache: read thread busy share, minutes 90-110 | 0.404 | 0.503 | +24 % | 0.483 ± 0.035 [0.443, 0.511] | roughly |
| hicache: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 1.000 ± 0.000 [1.000, 1.000] | reproduced |
| hicache: first cleaner pass, s | none | none |  | 5010 ± 0 [5010, 5010] | reproduced |
| hicache: cleaner passes before 6,600 s | 0 | 0 |  | 1 ± 1 [0, 2] | reproduced |
| hicache: SSD reads in the run | 2259 | 2434 | +8 % | 2152 ± 407 [1707, 2505] | reproduced |
| hicache: tokens read from the SSD, M (pages x 64) | 33.3 | 41.1 | +23 % | 37.6 ± 4.5 [32.5, 41.0] | roughly |
| hicache: tokens written to the SSD, M (submitted, incl. skipped) | 83.4 | 69.3 | -17 % | 62.1 ± 11.9 [49.3, 72.8] | roughly |
| jit_p50: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.0 ± 0.0 [21.0, 21.0] | reproduced |
| jit_p50: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.7 ± 0.6 [20.0, 21.0] | reproduced |
| jit_p50: running requests, median, minutes 90-110 | 20.0 | 18.0 | -10 % | 20.3 ± 1.2 [19.0, 21.0] | roughly |
| jit_p50: running requests, mean, minutes 90-110 | 20.3 | 18.5 | -9 % | 20.3 ± 1.5 [18.5, 21.2] | roughly |
| jit_p50: queued requests, median, minutes 90-110 | 73 | 75 | +3 % | 75 ± 12 [61, 83] | reproduced |
| jit_p50: queued requests, median, minutes 60-90 | 57 | 41 | -28 % | 52 ± 27 [22, 76] | roughly |
| jit_p50: GPU pool used, median, minutes 90-110 | 0.970 | 0.976 | +1 % | 0.978 ± 0.006 [0.971, 0.982] | reproduced |
| jit_p50: read thread busy share, minutes 30-60 | 0.636 | 0.573 | -10 % | 0.532 ± 0.224 [0.274, 0.674] | reproduced |
| jit_p50: write thread busy share, minutes 30-60 | 0.710 | 0.528 | -26 % | 0.599 ± 0.227 [0.380, 0.833] | roughly |
| jit_p50: read thread busy share, minutes 60-90 | 0.802 | 0.939 | +17 % | 0.746 ± 0.104 [0.636, 0.843] | roughly |
| jit_p50: write thread busy share, minutes 60-90 | 0.953 | 0.721 | -24 % | 0.875 ± 0.132 [0.737, 1.000] | roughly |
| jit_p50: read thread busy share, minutes 90-110 | 0.757 | 0.628 | -17 % | 0.620 ± 0.033 [0.585, 0.648] | roughly |
| jit_p50: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 1.000 ± 0.000 [1.000, 1.000] | reproduced |
| jit_p50: first cleaner pass, s | 6798 | 5220 | -23 % | 5610 ± 806 [5040, 6180] | not_reproduced |
| jit_p50: cleaner passes before 6,600 s | 0 | 1 | (measured 0) | 1 ± 1 [0, 2] | not_reproduced |
| jit_p50: reads refused host room before 6,600 s | 2414 | 1175 | -51 % | 2539 ± 1490 [1458, 4239] | not_reproduced |
| jit_p50: loop onset minute (3 minutes in a row with 5 or more refusals) | 42 | 88 | +110 % | 62 ± 21 [42, 83] | not_reproduced |
| jit_p50: loop onset minute (5 in a row) | 67 | 88 | +31 % | 67 ± 22 [42, 83] | not_reproduced |
| jit_p50: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| jit_p50: reads staged before arrival | 804 | 906 | +13 % | 630 ± 169 [436, 743] | roughly |
| jit_p50: SSD reads in the run | 3707 | 3239 | -13 % | 3313 ± 451 [2851, 3753] | roughly |
| jit_p50: tokens read from the SSD, M (pages x 64) | 63.1 | 68.4 | +8 % | 58.1 ± 6.5 [52.2, 65.1] | reproduced |
| jit_p50: tokens written to the SSD, M (submitted, incl. skipped) | 52.6 | 43.3 | -18 % | 51.5 ± 8.2 [45.5, 60.8] | roughly |
| jit_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.0 ± 0.0 [21.0, 21.0] | reproduced |
| jit_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.7 ± 0.6 [20.0, 21.0] | reproduced |
| jit_exact: running requests, median, minutes 90-110 | 19.0 | 19.0 | +0 % | 20.0 ± 1.0 [19.0, 21.0] | reproduced |
| jit_exact: running requests, mean, minutes 90-110 | 19.5 | 19.2 | -2 % | 20.2 ± 1.4 [18.6, 21.3] | reproduced |
| jit_exact: queued requests, median, minutes 90-110 | 78 | 64 | -17 % | 71 ± 18 [50, 84] | roughly |
| jit_exact: queued requests, median, minutes 60-90 | 38 | 36 | -6 % | 48 ± 28 [19, 74] | reproduced |
| jit_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.975 | +0 % | 0.979 ± 0.001 [0.977, 0.980] | reproduced |
| jit_exact: read thread busy share, minutes 30-60 | 0.089 | 0.112 | +26 % | 0.220 ± 0.165 [0.076, 0.400] | roughly |
| jit_exact: write thread busy share, minutes 30-60 | 0.274 | 0.288 | +5 % | 0.393 ± 0.155 [0.285, 0.571] | reproduced |
| jit_exact: read thread busy share, minutes 60-90 | 0.578 | 0.591 | +2 % | 0.575 ± 0.383 [0.173, 0.937] | reproduced |
| jit_exact: write thread busy share, minutes 60-90 | 0.699 | 0.511 | -27 % | 0.685 ± 0.355 [0.300, 1.000] | roughly |
| jit_exact: read thread busy share, minutes 90-110 | 0.793 | 0.790 | -0 % | 0.590 ± 0.026 [0.571, 0.619] | reproduced |
| jit_exact: write thread busy share, minutes 90-110 | 1.000 | 0.852 | -15 % | 0.888 ± 0.194 [0.664, 1.000] | roughly |
| jit_exact: first cleaner pass, s | 5779 | 5190 | -10 % | 5145 ± 191 [5010, 5280] | not_reproduced |
| jit_exact: cleaner passes before 6,600 s | 1 | 2 | +100 % | 1 ± 2 [0, 3] | not_reproduced |
| jit_exact: reads refused host room before 6,600 s | 1711 | 688 | -60 % | 1888 ± 1830 [330, 3903] | not_reproduced |
| jit_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | 80 | 99 | +24 % | 72 ± 28 [52, 104] | roughly |
| jit_exact: loop onset minute (5 in a row) | 80 | 99 | +24 % | 81 ± 27 [52, 104] | roughly |
| jit_exact: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| jit_exact: reads staged before arrival | 59 | 51 | -14 % | 45 ± 40 [16, 90] | roughly |
| jit_exact: SSD reads in the run | 2507 | 2216 | -12 % | 2469 ± 822 [1554, 3145] | roughly |
| jit_exact: tokens read from the SSD, M (pages x 64) | 42.5 | 46.0 | +8 % | 43.1 ± 13.6 [27.7, 53.5] | reproduced |
| jit_exact: tokens written to the SSD, M (submitted, incl. skipped) | 46.2 | 39.7 | -14 % | 46.9 ± 10.1 [38.3, 58.1] | roughly |
| ec_full: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| ec_full: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.7 ± 0.6 [20.0, 21.0] | reproduced |
| ec_full: running requests, median, minutes 90-110 | 19.0 | 19.0 | +0 % | 19.3 ± 1.5 [18.0, 21.0] | reproduced |
| ec_full: running requests, mean, minutes 90-110 | 19.4 | 19.2 | -1 % | 19.7 ± 1.6 [18.6, 21.6] | reproduced |
| ec_full: queued requests, median, minutes 90-110 | 78 | 75 | -4 % | 70 ± 17 [50, 80] | reproduced |
| ec_full: queued requests, median, minutes 60-90 | 39 | 37 | -5 % | 43 ± 21 [19, 58] | reproduced |
| ec_full: GPU pool used, median, minutes 90-110 | 0.970 | 0.978 | +1 % | 0.979 ± 0.003 [0.976, 0.982] | reproduced |
| ec_full: read thread busy share, minutes 30-60 | 0.233 | 0.282 | +21 % | 0.395 ± 0.234 [0.189, 0.649] | roughly |
| ec_full: write thread busy share, minutes 30-60 | 0.373 | 0.360 | -4 % | 0.433 ± 0.124 [0.331, 0.571] | reproduced |
| ec_full: read thread busy share, minutes 60-90 | 0.701 | 0.688 | -2 % | 0.737 ± 0.354 [0.328, 0.960] | reproduced |
| ec_full: write thread busy share, minutes 60-90 | 0.767 | 0.567 | -26 % | 0.633 ± 0.229 [0.372, 0.801] | roughly |
| ec_full: read thread busy share, minutes 90-110 | 0.780 | 0.688 | -12 % | 0.642 ± 0.078 [0.586, 0.731] | roughly |
| ec_full: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 0.887 ± 0.195 [0.662, 1.000] | reproduced |
| ec_full: first cleaner pass, s | 5584 | 5190 | -7 % | 5110 ± 148 [5010, 5280] | roughly |
| ec_full: cleaner passes before 6,600 s | 1 | 2 | +100 % | 2 ± 1 [2, 3] | not_reproduced |
| ec_full: reads refused host room before 6,600 s | 1403 | 1082 | -23 % | 910 ± 778 [12, 1366] | roughly |
| ec_full: loop onset minute (3 minutes in a row with 5 or more refusals) | 84 | 90 | +7 % | 86 ± 2 [85, 88] | reproduced |
| ec_full: loop onset minute (5 in a row) | 84 | 90 | +7 % | 86 ± 2 [85, 88] | reproduced |
| ec_full: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| ec_full: reads staged before arrival | 2 | 1 | -50 % | 0 ± 0 [0, 0] | not_reproduced |
| ec_full: SSD reads in the run | 2936 | 2654 | -10 % | 2812 ± 744 [2014, 3488] | reproduced |
| ec_full: tokens read from the SSD, M (pages x 64) | 53.6 | 52.3 | -2 % | 58.1 ± 14.3 [43.1, 71.5] | reproduced |
| ec_full: tokens written to the SSD, M (submitted, incl. skipped) | 45.1 | 42.5 | -6 % | 40.5 ± 6.8 [32.7, 44.6] | reproduced |
| ec_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.3 ± 0.6 [21.0, 22.0] | reproduced |
| ec_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.7 ± 0.6 [20.0, 21.0] | reproduced |
| ec_exact: running requests, median, minutes 90-110 | 19.0 | 19.0 | +0 % | 20.5 ± 1.3 [19.0, 21.5] | reproduced |
| ec_exact: running requests, mean, minutes 90-110 | 19.3 | 19.1 | -1 % | 20.3 ± 1.2 [19.0, 21.4] | reproduced |
| ec_exact: queued requests, median, minutes 90-110 | 77 | 76 | -1 % | 72 ± 19 [50, 85] | reproduced |
| ec_exact: queued requests, median, minutes 60-90 | 38 | 35 | -8 % | 50 ± 29 [18, 73] | reproduced |
| ec_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.979 | +1 % | 0.978 ± 0.003 [0.975, 0.980] | reproduced |
| ec_exact: read thread busy share, minutes 30-60 | 0.137 | 0.151 | +11 % | 0.307 ± 0.196 [0.123, 0.513] | roughly |
| ec_exact: write thread busy share, minutes 30-60 | 0.313 | 0.307 | -2 % | 0.412 ± 0.132 [0.303, 0.559] | reproduced |
| ec_exact: read thread busy share, minutes 60-90 | 0.599 | 0.577 | -4 % | 0.541 ± 0.265 [0.238, 0.732] | reproduced |
| ec_exact: write thread busy share, minutes 60-90 | 0.718 | 0.511 | -29 % | 0.736 ± 0.351 [0.338, 1.000] | roughly |
| ec_exact: read thread busy share, minutes 90-110 | 0.797 | 0.646 | -19 % | 0.632 ± 0.021 [0.618, 0.656] | roughly |
| ec_exact: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 0.890 ± 0.190 [0.671, 1.000] | reproduced |
| ec_exact: first cleaner pass, s | 5814 | 5190 | -11 % | 6180 ± 1115 [5010, 7230] | not_reproduced |
| ec_exact: cleaner passes before 6,600 s | 1 | 2 | +100 % | 1 ± 2 [0, 3] | not_reproduced |
| ec_exact: reads refused host room before 6,600 s | 1636 | 1189 | -27 % | 1975 ± 1609 [220, 3382] | roughly |
| ec_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | 79 | 88 | +11 % | 77 ± 26 [55, 106] | roughly |
| ec_exact: loop onset minute (5 in a row) | 79 | 88 | +11 % | 77 ± 26 [55, 106] | roughly |
| ec_exact: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| ec_exact: reads staged before arrival | 53 | 45 | -15 % | 48 ± 51 [14, 107] | roughly |
| ec_exact: SSD reads in the run | 2643 | 2365 | -11 % | 2701 ± 868 [1761, 3472] | roughly |
| ec_exact: tokens read from the SSD, M (pages x 64) | 44.8 | 43.3 | -3 % | 45.8 ± 11.0 [33.2, 53.8] | reproduced |
| ec_exact: tokens written to the SSD, M (submitted, incl. skipped) | 46.5 | 42.7 | -8 % | 46.9 ± 9.6 [36.4, 55.1] | reproduced |

### Paired differences between arms (B - A on the requests both sent before 6,600 s; interval over users)

| campaign | pair | measured (referee's recomputation) | simulated, seed 0 | simulated, fresh seeds: mean [min, max] | sign over the 9 seeds | verdict |
|---|---|---|---|---|---|---|
| campaign34 | jit_p50 -> ec_full | -0.62 [-1.03, -0.22] on 15416 | -0.70 [-1.10, -0.30] on 15484 | -0.34 [-0.38, -0.27] | same | reproduced |
| campaign34 | jit_p50 -> jit_exact | -0.67 [-1.07, -0.29] on 15458 | -0.74 [-1.12, -0.37] on 15505 | -0.31 [-0.43, -0.14] | same | reproduced |
| campaign34 | ec_full -> jit_exact | -0.06 [-0.19, +0.05] on 15529 | -0.06 [-0.18, +0.07] on 15586 | +0.03 [-0.07, +0.12] | changes | reproduced |
| campaign34 | jit_p50 -> ec_exact | -1.01 [-1.43, -0.60] on 15464 | -0.74 [-1.14, -0.36] on 15486 | -0.38 [-0.45, -0.27] | same | reproduced |
| campaign34 | ec_full -> ec_exact | -0.40 [-0.51, -0.29] on 15544 | -0.04 [-0.15, +0.08] on 15591 | -0.04 [-0.08, +0.00] | changes | not_reproduced |
| campaign34 | jit_exact -> ec_exact | -0.32 [-0.46, -0.20] on 15571 | +0.01 [-0.12, +0.13] on 15588 | -0.06 [-0.11, -0.01] | changes | not_reproduced |
| campaign34 | hicache -> ec_exact | -26.96 [-30.48, -23.61] on 9967 | -26.67 [-30.12, -23.38] on 9996 | -18.26 [-26.49, -6.63] | same | reproduced |
| campaign34 | hicache -> ec_full | -26.71 [-30.21, -23.39] on 9967 | -26.56 [-29.99, -23.28] on 9996 | -18.24 [-26.49, -6.53] | same | reproduced |
| campaign32 | jit_p50 -> ec_full | -8.60 [-9.96, -7.30] on 12589 | -2.10 [-2.81, -1.43] on 13950 | -10.14 [-18.57, -5.53] | same | not_reproduced |
| campaign32 | jit_p50 -> jit_exact | -7.75 [-9.00, -6.59] on 12590 | -5.10 [-6.03, -4.24] on 14028 | -5.89 [-6.19, -5.52] | same | roughly |
| campaign32 | ec_full -> jit_exact | +1.26 [+0.93, +1.61] on 13543 | -3.62 [-4.34, -2.95] on 14244 | +5.28 [+0.16, +14.89] | changes | not_reproduced |
| campaign32 | jit_p50 -> ec_exact | -7.54 [-8.82, -6.37] on 12589 | -1.18 [-1.82, -0.53] on 13914 | -3.87 [-8.26, +2.23] | changes | not_reproduced |
| campaign32 | ec_full -> ec_exact | +1.43 [+1.07, +1.81] on 13530 | +0.97 [+0.72, +1.26] on 14097 | +7.31 [+0.35, +13.49] | same | roughly |
| campaign32 | jit_exact -> ec_exact | +0.16 [+0.02, +0.33] on 13505 | +4.13 [+3.38, +4.92] on 14099 | +1.62 [-2.25, +7.72] | changes | not_reproduced |
| campaign32 | hicache -> ec_exact | -28.28 [-32.38, -24.48] on 9653 | -27.03 [-30.90, -23.45] on 9924 | -17.70 [-24.43, -6.75] | same | reproduced |
| campaign32 | hicache -> ec_full | -28.25 [-32.38, -24.43] on 9653 | -26.98 [-30.84, -23.42] on 9924 | -19.86 [-29.10, -6.69] | same | reproduced |

### The fixes: fixes on minus fixes off, mean TTFT of each run's own requests, s

| arm | measured (34 - 32) | simulated, seed 0 | simulated, fresh seeds: mean [min, max] |
|---|---|---|---|
| hicache | -1.63 | -0.39 | -1.49 [-2.31, -0.08] |
| jit_p50 | -6.50 | -2.97 | -8.07 [-14.08, -3.18] |
| jit_exact | -3.94 | -1.30 | -5.50 [-11.33, -0.50] |
| ec_full | -3.68 | -2.65 | -2.84 [-4.59, +0.02] |
| ec_exact | -4.29 | -2.87 | -6.15 [-10.45, -0.23] |
