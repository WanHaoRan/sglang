
### Campaign 34 (fixes on): client side, requests sent before 6,600 s

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 31.88 | 31.58 | -1 % | 28.64 ± 13.53 [8.62, 45.75] | reproduced |
| hicache: mean TTFT, minutes 0-30 | 0.41 | 0.40 | -3 % | 1.09 ± 0.87 [0.23, 3.03] | reproduced |
| hicache: mean TTFT, minutes 30-60 | 30.53 | 30.83 | +1 % | 28.48 ± 26.11 [1.79, 71.75] | reproduced |
| hicache: mean TTFT, minutes 60-90 | 72.97 | 71.30 | -2 % | 61.93 ± 33.67 [4.57, 87.15] | reproduced |
| hicache: mean TTFT, minutes 90-110 | 90.14 | 88.25 | -2 % | 81.40 ± 6.71 [72.91, 91.98] | reproduced |
| hicache: p50 TTFT, s | 5.95 | 5.89 | -1 % | 13.38 ± 20.74 [0.91, 63.98] | reproduced |
| hicache: p90 TTFT, s | 85.62 | 84.05 | -2 % | 75.21 ± 26.90 [13.80, 95.88] | reproduced |
| hicache: requests per minute to 6,600 s | 90.6 | 90.9 | +0 % | 96.7 ± 15.4 [78.6, 118.2] | reproduced |
| hicache: requests per minute, minutes 90-109 | 55.3 | 55.9 | +1 % | 59.5 ± 3.8 [56.0, 67.4] | reproduced |
| hicache: users with a request | 448 | 448 | +0 % | 481 ± 72 [401, 590] | reproduced |
| hicache: prompt tokens from GPU, share | 0.302 | 0.305 | +1 % | 0.266 ± 0.052 [0.176, 0.337] | reproduced |
| hicache: from host, share | 0.241 | 0.236 | -2 % | 0.329 ± 0.139 [0.183, 0.514] | reproduced |
| hicache: from SSD, share | 0.121 | 0.121 | -0 % | 0.114 ± 0.029 [0.072, 0.156] | reproduced |
| hicache: from SSD, minutes 30-60 | 0.142 | 0.136 | -4 % | 0.118 ± 0.062 [0.043, 0.222] | reproduced |
| hicache: from SSD, minutes 60-90 | 0.242 | 0.251 | +3 % | 0.218 ± 0.073 [0.091, 0.287] | reproduced |
| hicache: from SSD, minutes 90-110 | 0.255 | 0.254 | -0 % | 0.254 ± 0.022 [0.224, 0.289] | reproduced |
| hicache: history recomputed, share | 0.2891 | 0.2913 | +1 % | 0.2394 ± 0.1213 [0.0599, 0.3981] | reproduced |
| hicache: history recomputed, M tokens, minutes 30-60 | 18.99 | 19.58 | +3 % | 12.29 ± 11.76 [0.09, 29.37] | reproduced |
| hicache: history recomputed, M tokens, minutes 60-90 | 28.28 | 28.12 | -1 % | 21.85 ± 11.48 [0.17, 28.75] | reproduced |
| hicache: history recomputed, M tokens, minutes 90-110 | 18.73 | 18.99 | +1 % | 19.54 ± 0.75 [18.71, 20.99] | reproduced |
| hicache: returning requests that read from the SSD, minutes 90-110 | 0.313 | 0.306 | -2 % | 0.302 ± 0.022 [0.267, 0.339] | reproduced |
| hicache: their mean TTFT, s | 89.3 | 88.8 | -1 % | 82.9 ± 6.5 [75.2, 93.9] | reproduced |
| hicache: the others' mean TTFT, s | 90.6 | 88.2 | -3 % | 80.8 ± 6.7 [72.4, 91.1] | reproduced |
| jit_p50: mean TTFT, s | 11.47 | 11.11 | -3 % | 12.83 ± 4.36 [5.60, 16.76] | reproduced |
| jit_p50: mean TTFT, minutes 0-30 | 0.41 | 0.39 | -6 % | 1.06 ± 0.78 [0.20, 2.74] | reproduced |
| jit_p50: mean TTFT, minutes 30-60 | 6.37 | 6.07 | -5 % | 10.87 ± 6.74 [1.85, 18.47] | reproduced |
| jit_p50: mean TTFT, minutes 60-90 | 17.80 | 17.37 | -2 % | 19.13 ± 8.10 [4.84, 27.18] | reproduced |
| jit_p50: mean TTFT, minutes 90-110 | 28.20 | 26.95 | -4 % | 23.19 ± 4.18 [17.81, 28.47] | reproduced |
| jit_p50: p50 TTFT, s | 6.38 | 6.22 | -2 % | 7.70 ± 3.26 [1.65, 10.71] | reproduced |
| jit_p50: p90 TTFT, s | 22.42 | 25.02 | +12 % | 37.13 ± 17.21 [15.39, 59.07] | roughly |
| jit_p50: requests per minute to 6,600 s | 141.9 | 142.6 | +1 % | 138.2 ± 5.5 [129.9, 145.5] | reproduced |
| jit_p50: requests per minute, minutes 90-109 | 129.1 | 132.2 | +2 % | 141.0 ± 14.4 [121.1, 169.8] | reproduced |
| jit_p50: users with a request | 617 | 617 | +0 % | 631 ± 23 [593, 666] | reproduced |
| jit_p50: prompt tokens from GPU, share | 0.186 | 0.191 | +3 % | 0.189 ± 0.058 [0.124, 0.300] | reproduced |
| jit_p50: from host, share | 0.603 | 0.596 | -1 % | 0.582 ± 0.041 [0.525, 0.659] | reproduced |
| jit_p50: from SSD, share | 0.148 | 0.150 | +1 % | 0.162 ± 0.031 [0.111, 0.197] | reproduced |
| jit_p50: from SSD, minutes 30-60 | 0.126 | 0.122 | -3 % | 0.160 ± 0.079 [0.064, 0.256] | reproduced |
| jit_p50: from SSD, minutes 60-90 | 0.228 | 0.237 | +4 % | 0.228 ± 0.042 [0.141, 0.262] | reproduced |
| jit_p50: from SSD, minutes 90-110 | 0.246 | 0.247 | +0 % | 0.253 ± 0.020 [0.221, 0.274] | reproduced |
| jit_p50: history recomputed, share | 0.0053 | 0.0061 | +16 % | 0.0053 ± 0.0008 [0.0042, 0.0063] | roughly |
| jit_p50: history recomputed, M tokens, minutes 30-60 | 0.26 | 0.18 | -30 % | 0.19 ± 0.10 [0.09, 0.37] | roughly |
| jit_p50: history recomputed, M tokens, minutes 60-90 | 0.60 | 0.93 | +54 % | 0.57 ± 0.19 [0.36, 0.83] | not_reproduced |
| jit_p50: history recomputed, M tokens, minutes 90-110 | 1.07 | 1.13 | +6 % | 1.04 ± 0.27 [0.64, 1.53] | reproduced |
| jit_p50: returning requests that read from the SSD, minutes 90-110 | 0.261 | 0.268 | +3 % | 0.266 ± 0.027 [0.226, 0.318] | reproduced |
| jit_p50: their mean TTFT, s | 69.6 | 66.6 | -4 % | 52.7 ± 14.2 [35.2, 71.2] | reproduced |
| jit_p50: the others' mean TTFT, s | 13.1 | 12.0 | -8 % | 12.2 ± 1.4 [9.9, 13.7] | reproduced |
| jit_exact: mean TTFT, s | 10.92 | 10.58 | -3 % | 12.55 ± 4.44 [5.17, 16.56] | reproduced |
| jit_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 | -10 % | 1.03 ± 0.79 [0.20, 2.75] | roughly |
| jit_exact: mean TTFT, minutes 30-60 | 6.13 | 5.86 | -4 % | 10.59 ± 6.59 [1.70, 17.78] | reproduced |
| jit_exact: mean TTFT, minutes 60-90 | 16.60 | 15.98 | -4 % | 18.67 ± 8.06 [4.55, 26.48] | reproduced |
| jit_exact: mean TTFT, minutes 90-110 | 27.03 | 26.39 | -2 % | 22.90 ± 4.44 [17.70, 28.47] | reproduced |
| jit_exact: p50 TTFT, s | 8.24 | 7.80 | -5 % | 10.24 ± 5.10 [1.44, 15.65] | reproduced |
| jit_exact: p90 TTFT, s | 22.26 | 22.27 | +0 % | 29.35 ± 8.64 [16.70, 37.45] | reproduced |
| jit_exact: requests per minute to 6,600 s | 141.8 | 142.6 | +1 % | 137.7 ± 5.5 [128.9, 144.9] | reproduced |
| jit_exact: requests per minute, minutes 90-109 | 132.1 | 134.4 | +2 % | 141.2 ± 14.2 [121.6, 170.8] | reproduced |
| jit_exact: users with a request | 622 | 622 | +0 % | 634 ± 23 [594, 671] | reproduced |
| jit_exact: prompt tokens from GPU, share | 0.205 | 0.209 | +2 % | 0.203 ± 0.066 [0.138, 0.333] | reproduced |
| jit_exact: from host, share | 0.620 | 0.615 | -1 % | 0.596 ± 0.040 [0.551, 0.678] | reproduced |
| jit_exact: from SSD, share | 0.113 | 0.114 | +1 % | 0.134 ± 0.055 [0.051, 0.192] | reproduced |
| jit_exact: from SSD, minutes 30-60 | 0.032 | 0.032 | +2 % | 0.090 ± 0.079 [0.017, 0.204] | reproduced |
| jit_exact: from SSD, minutes 60-90 | 0.186 | 0.181 | -3 % | 0.204 ± 0.104 [0.047, 0.302] | reproduced |
| jit_exact: from SSD, minutes 90-110 | 0.274 | 0.284 | +4 % | 0.259 ± 0.052 [0.171, 0.313] | reproduced |
| jit_exact: history recomputed, share | 0.0050 | 0.0045 | -10 % | 0.0046 ± 0.0010 [0.0029, 0.0059] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -6 % | 0.14 ± 0.05 [0.09, 0.21] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 60-90 | 0.64 | 0.46 | -28 % | 0.47 ± 0.26 [0.12, 0.83] | roughly |
| jit_exact: history recomputed, M tokens, minutes 90-110 | 1.06 | 1.05 | -1 % | 0.94 ± 0.23 [0.71, 1.43] | reproduced |
| jit_exact: returning requests that read from the SSD, minutes 90-110 | 0.293 | 0.312 | +6 % | 0.271 ± 0.056 [0.176, 0.340] | reproduced |
| jit_exact: their mean TTFT, s | 50.4 | 48.8 | -3 % | 36.4 ± 11.4 [23.3, 51.0] | reproduced |
| jit_exact: the others' mean TTFT, s | 17.2 | 16.1 | -6 % | 17.3 ± 1.9 [14.8, 20.0] | reproduced |
| ec_full: mean TTFT, s | 10.96 | 10.62 | -3 % | 12.55 ± 4.40 [5.27, 16.55] | reproduced |
| ec_full: mean TTFT, minutes 0-30 | 0.41 | 0.39 | -6 % | 1.05 ± 0.79 [0.21, 2.76] | reproduced |
| ec_full: mean TTFT, minutes 30-60 | 6.18 | 5.90 | -5 % | 10.58 ± 6.49 [1.82, 17.78] | reproduced |
| ec_full: mean TTFT, minutes 60-90 | 16.36 | 15.91 | -3 % | 18.71 ± 8.06 [4.61, 26.64] | reproduced |
| ec_full: mean TTFT, minutes 90-110 | 27.97 | 26.77 | -4 % | 22.88 ± 4.38 [17.86, 28.41] | reproduced |
| ec_full: p50 TTFT, s | 8.34 | 7.86 | -6 % | 10.28 ± 5.13 [1.67, 16.10] | reproduced |
| ec_full: p90 TTFT, s | 24.01 | 23.18 | -3 % | 29.14 ± 8.24 [16.42, 38.00] | reproduced |
| ec_full: requests per minute to 6,600 s | 141.5 | 142.3 | +1 % | 137.6 ± 5.4 [129.1, 144.7] | reproduced |
| ec_full: requests per minute, minutes 90-109 | 128.3 | 133.3 | +4 % | 141.2 ± 14.1 [122.4, 171.1] | reproduced |
| ec_full: users with a request | 620 | 623 | +0 % | 633 ± 23 [594, 670] | reproduced |
| ec_full: prompt tokens from GPU, share | 0.194 | 0.199 | +3 % | 0.193 ± 0.059 [0.125, 0.307] | reproduced |
| ec_full: from host, share | 0.606 | 0.600 | -1 % | 0.587 ± 0.038 [0.547, 0.663] | reproduced |
| ec_full: from SSD, share | 0.138 | 0.140 | +1 % | 0.153 ± 0.048 [0.083, 0.201] | reproduced |
| ec_full: from SSD, minutes 30-60 | 0.079 | 0.078 | -1 % | 0.127 ± 0.070 [0.046, 0.221] | reproduced |
| ec_full: from SSD, minutes 60-90 | 0.212 | 0.214 | +1 % | 0.218 ± 0.086 [0.085, 0.303] | reproduced |
| ec_full: from SSD, minutes 90-110 | 0.284 | 0.288 | +1 % | 0.264 ± 0.044 [0.199, 0.311] | reproduced |
| ec_full: history recomputed, share | 0.0044 | 0.0040 | -8 % | 0.0043 ± 0.0007 [0.0028, 0.0049] | reproduced |
| ec_full: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -6 % | 0.11 ± 0.01 [0.09, 0.13] | reproduced |
| ec_full: history recomputed, M tokens, minutes 60-90 | 0.46 | 0.47 | +3 % | 0.36 ± 0.19 [0.12, 0.69] | reproduced |
| ec_full: history recomputed, M tokens, minutes 90-110 | 1.00 | 0.85 | -15 % | 0.95 ± 0.21 [0.65, 1.27] | reproduced |
| ec_full: returning requests that read from the SSD, minutes 90-110 | 0.321 | 0.337 | +5 % | 0.290 ± 0.047 [0.217, 0.346] | reproduced |
| ec_full: their mean TTFT, s | 46.7 | 46.0 | -2 % | 34.7 ± 10.5 [23.0, 47.7] | reproduced |
| ec_full: the others' mean TTFT, s | 19.0 | 16.9 | -11 % | 17.6 ± 2.0 [15.0, 20.5] | roughly |
| ec_exact: mean TTFT, s | 10.66 | 10.60 | -1 % | 12.51 ± 4.43 [5.15, 16.48] | reproduced |
| ec_exact: mean TTFT, minutes 0-30 | 0.39 | 0.36 | -8 % | 1.03 ± 0.79 [0.19, 2.73] | reproduced |
| ec_exact: mean TTFT, minutes 30-60 | 5.91 | 5.82 | -1 % | 10.54 ± 6.54 [1.72, 17.70] | reproduced |
| ec_exact: mean TTFT, minutes 60-90 | 15.76 | 16.11 | +2 % | 18.58 ± 8.01 [4.53, 26.32] | reproduced |
| ec_exact: mean TTFT, minutes 90-110 | 27.44 | 26.48 | -4 % | 22.95 ± 4.53 [17.57, 28.79] | reproduced |
| ec_exact: p50 TTFT, s | 7.87 | 7.82 | -1 % | 10.40 ± 5.23 [1.50, 16.21] | reproduced |
| ec_exact: p90 TTFT, s | 22.60 | 22.73 | +1 % | 28.66 ± 7.88 [16.97, 37.83] | reproduced |
| ec_exact: requests per minute to 6,600 s | 142.2 | 142.3 | +0 % | 137.6 ± 5.4 [129.0, 144.9] | reproduced |
| ec_exact: requests per minute, minutes 90-109 | 132.6 | 134.6 | +1 % | 141.1 ± 14.1 [121.1, 170.1] | reproduced |
| ec_exact: users with a request | 621 | 624 | +0 % | 634 ± 23 [594, 669] | reproduced |
| ec_exact: prompt tokens from GPU, share | 0.209 | 0.210 | +1 % | 0.205 ± 0.066 [0.138, 0.337] | reproduced |
| ec_exact: from host, share | 0.614 | 0.611 | -0 % | 0.589 ± 0.040 [0.541, 0.669] | reproduced |
| ec_exact: from SSD, share | 0.116 | 0.117 | +1 % | 0.139 ± 0.054 [0.059, 0.194] | reproduced |
| ec_exact: from SSD, minutes 30-60 | 0.047 | 0.043 | -9 % | 0.107 ± 0.077 [0.029, 0.217] | reproduced |
| ec_exact: from SSD, minutes 60-90 | 0.181 | 0.184 | +2 % | 0.206 ± 0.098 [0.058, 0.302] | reproduced |
| ec_exact: from SSD, minutes 90-110 | 0.279 | 0.285 | +2 % | 0.260 ± 0.054 [0.168, 0.316] | reproduced |
| ec_exact: history recomputed, share | 0.0039 | 0.0046 | +18 % | 0.0043 ± 0.0007 [0.0028, 0.0052] | roughly |
| ec_exact: history recomputed, M tokens, minutes 30-60 | 0.13 | 0.11 | -12 % | 0.12 ± 0.03 [0.09, 0.17] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 60-90 | 0.47 | 0.51 | +9 % | 0.39 ± 0.20 [0.17, 0.68] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 90-110 | 0.79 | 1.02 | +30 % | 0.93 ± 0.18 [0.66, 1.27] | roughly |
| ec_exact: returning requests that read from the SSD, minutes 90-110 | 0.309 | 0.315 | +2 % | 0.284 ± 0.062 [0.175, 0.357] | reproduced |
| ec_exact: their mean TTFT, s | 48.5 | 47.5 | -2 % | 34.9 ± 10.4 [22.3, 49.0] | reproduced |
| ec_exact: the others' mean TTFT, s | 17.9 | 16.6 | -7 % | 17.7 ± 2.2 [14.9, 20.9] | reproduced |

### Campaign 34 (fixes on): server side

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.0 ± 1.4 [19.0, 23.0] | reproduced |
| hicache: running requests, median, minutes 60-90 | 22.0 | 21.0 | -5 % | 20.9 ± 0.8 [20.0, 22.0] | reproduced |
| hicache: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 20.9 ± 1.1 [20.0, 23.0] | reproduced |
| hicache: running requests, mean, minutes 90-110 | 19.9 | 19.6 | -2 % | 20.9 ± 1.0 [19.6, 22.3] | reproduced |
| hicache: queued requests, median, minutes 90-110 | 83 | 83 | +0 % | 79 ± 8 [66, 87] | reproduced |
| hicache: queued requests, median, minutes 60-90 | 71 | 70 | -1 % | 63 ± 30 [10, 84] | reproduced |
| hicache: GPU pool used, median, minutes 90-110 | 0.970 | 0.973 | +0 % | 0.980 ± 0.003 [0.976, 0.984] | reproduced |
| hicache: read thread busy share, minutes 30-60 | 0.344 | 0.376 | +9 % | 0.277 ± 0.089 [0.164, 0.381] | reproduced |
| hicache: write thread busy share, minutes 30-60 | 0.282 | 0.282 | -0 % | 0.276 ± 0.037 [0.219, 0.328] | reproduced |
| hicache: read thread busy share, minutes 60-90 | 0.381 | 0.444 | +16 % | 0.402 ± 0.070 [0.288, 0.492] | roughly |
| hicache: write thread busy share, minutes 60-90 | 0.223 | 0.219 | -2 % | 0.243 ± 0.059 [0.190, 0.337] | reproduced |
| hicache: read thread busy share, minutes 90-110 | 0.354 | 0.417 | +18 % | 0.432 ± 0.041 [0.365, 0.501] | roughly |
| hicache: write thread busy share, minutes 90-110 | 0.186 | 0.174 | -6 % | 0.197 ± 0.011 [0.181, 0.216] | reproduced |
| hicache: first cleaner pass, s | none | none |  | 5870 ± 1191 [5010, 7230] | reproduced |
| hicache: cleaner passes before 6,600 s | 0 | 0 |  | 0 ± 1 [0, 2] | reproduced |
| hicache: SSD reads in the run | 2162 | 2160 | -0 % | 1927 ± 276 [1477, 2249] | reproduced |
| hicache: tokens read from the SSD, M (pages x 64) | 48.3 | 48.4 | +0 % | 43.3 ± 6.1 [34.0, 50.5] | reproduced |
| hicache: tokens written to the SSD, M (submitted, incl. skipped) | 94.7 | 19.9 | -79 % | 21.7 ± 3.2 [17.8, 26.8] | not_reproduced |
| hicache: host mirror, median of minutes 60-110, K | 631 | 612 | -3 % | 618 ± 12 [602, 637] | roughly |
| hicache: host free, median, K | 5.3 | 14.1 | +166 % | 11.5 ± 1.1 [9.3, 12.7] | not_reproduced |
| hicache: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| hicache: pinned tokens, p90, K | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| hicache: pending writes (census 'pending_write'), p90, K | 8.1 | 6.3 | -22 % | 7.2 ± 3.5 [2.6, 12.1] | roughly |
| hicache: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 0.0 | 0.1 | (measured 0) | 0.1 ± 0.1 [0.0, 0.3] | not_reproduced |
| jit_p50: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.0 ± 0.9 [19.0, 22.0] | reproduced |
| jit_p50: running requests, median, minutes 60-90 | 21.0 | 20.0 | -5 % | 20.8 ± 1.0 [19.0, 22.0] | reproduced |
| jit_p50: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 21.1 ± 1.4 [19.5, 24.0] | reproduced |
| jit_p50: running requests, mean, minutes 90-110 | 19.7 | 19.6 | -0 % | 21.2 ± 1.2 [19.8, 23.6] | reproduced |
| jit_p50: queued requests, median, minutes 90-110 | 61 | 60 | -2 % | 54 ± 7 [42, 66] | reproduced |
| jit_p50: queued requests, median, minutes 60-90 | 42 | 41 | -2 % | 43 ± 20 [10, 62] | reproduced |
| jit_p50: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.980 ± 0.002 [0.977, 0.983] | reproduced |
| jit_p50: read thread busy share, minutes 30-60 | 0.551 | 0.541 | -2 % | 0.601 ± 0.288 [0.259, 0.966] | reproduced |
| jit_p50: write thread busy share, minutes 30-60 | 0.528 | 0.498 | -6 % | 0.525 ± 0.157 [0.338, 0.710] | reproduced |
| jit_p50: read thread busy share, minutes 60-90 | 0.968 | 0.949 | -2 % | 0.892 ± 0.155 [0.574, 0.998] | reproduced |
| jit_p50: write thread busy share, minutes 60-90 | 0.741 | 0.663 | -10 % | 0.665 ± 0.097 [0.474, 0.782] | roughly |
| jit_p50: read thread busy share, minutes 90-110 | 0.999 | 0.999 | +0 % | 0.995 ± 0.003 [0.992, 0.999] | reproduced |
| jit_p50: write thread busy share, minutes 90-110 | 0.734 | 0.672 | -8 % | 0.712 ± 0.042 [0.655, 0.772] | reproduced |
| jit_p50: first cleaner pass, s | 5301 | 5190 | -2 % | 5231 ± 152 [5010, 5400] | reproduced |
| jit_p50: cleaner passes before 6,600 s | 2 | 3 | +50 % | 3 ± 0 [2, 3] | roughly |
| jit_p50: reads refused host room before 6,600 s | 0 | 46 | (measured 0) | 13 ± 19 [0, 58] | not_reproduced |
| jit_p50: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| jit_p50: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| jit_p50: write hold switched on, times | 30 | 80 | +167 % | 76 ± 5 [67, 83] | not_reproduced |
| jit_p50: reads staged before arrival | 1208 | 1216 | +1 % | 1114 ± 76 [1012, 1225] | reproduced |
| jit_p50: SSD reads in the run | 3268 | 3405 | +4 % | 3482 ± 526 [2656, 4250] | reproduced |
| jit_p50: tokens read from the SSD, M (pages x 64) | 82.5 | 84.0 | +2 % | 83.5 ± 12.9 [62.9, 99.5] | reproduced |
| jit_p50: tokens written to the SSD, M (submitted, incl. skipped) | 31.9 | 30.1 | -5 % | 30.7 ± 0.7 [29.8, 31.8] | reproduced |
| jit_p50: host mirror, median of minutes 60-110, K | 646 | 646 | +0 % | 643 ± 2 [640, 646] | reproduced |
| jit_p50: host free, median, K | 17.7 | 13.1 | -26 % | 11.6 ± 1.2 [9.7, 12.7] | roughly |
| jit_p50: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 153 | 172 | +12 % | 172 ± 40 [132, 242] | reproduced |
| jit_p50: pinned tokens, p90, K | 319 | 310 | -3 % | 299 ± 79 [226, 438] | reproduced |
| jit_p50: pending writes (census 'pending_write'), p90, K | 15.4 | 41.2 | +168 % | 43.6 ± 5.5 [37.9, 52.7] | not_reproduced |
| jit_p50: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 94.6 | 11.8 | -88 % | 11.5 ± 3.8 [5.8, 15.0] | not_reproduced |
| jit_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.1 ± 1.0 [19.0, 22.0] | reproduced |
| jit_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 21.1 ± 0.8 [20.0, 22.0] | reproduced |
| jit_exact: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 21.2 ± 1.2 [19.5, 23.0] | reproduced |
| jit_exact: running requests, mean, minutes 90-110 | 20.0 | 19.9 | -0 % | 21.2 ± 1.2 [19.6, 23.4] | reproduced |
| jit_exact: queued requests, median, minutes 90-110 | 60 | 59 | -2 % | 54 ± 7 [41, 64] | reproduced |
| jit_exact: queued requests, median, minutes 60-90 | 37 | 36 | -4 % | 42 ± 19 [10, 60] | reproduced |
| jit_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.980 ± 0.002 [0.977, 0.984] | reproduced |
| jit_exact: read thread busy share, minutes 30-60 | 0.094 | 0.115 | +22 % | 0.288 ± 0.253 [0.072, 0.646] | roughly |
| jit_exact: write thread busy share, minutes 30-60 | 0.273 | 0.291 | +7 % | 0.370 ± 0.132 [0.256, 0.567] | reproduced |
| jit_exact: read thread busy share, minutes 60-90 | 0.570 | 0.561 | -1 % | 0.673 ± 0.361 [0.147, 0.997] | reproduced |
| jit_exact: write thread busy share, minutes 60-90 | 0.513 | 0.460 | -10 % | 0.554 ± 0.189 [0.268, 0.764] | roughly |
| jit_exact: read thread busy share, minutes 90-110 | 1.000 | 0.991 | -1 % | 0.866 ± 0.175 [0.523, 0.996] | reproduced |
| jit_exact: write thread busy share, minutes 90-110 | 0.737 | 0.661 | -10 % | 0.647 ± 0.086 [0.470, 0.744] | roughly |
| jit_exact: first cleaner pass, s | 5272 | 5190 | -2 % | 5201 ± 162 [4980, 5400] | reproduced |
| jit_exact: cleaner passes before 6,600 s | 2 | 3 | +50 % | 3 ± 0 [2, 3] | roughly |
| jit_exact: reads refused host room before 6,600 s | 0 | 0 |  | 3 ± 5 [0, 16] | reproduced |
| jit_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| jit_exact: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| jit_exact: write hold switched on, times | 28 | 59 | +111 % | 64 ± 12 [49, 84] | not_reproduced |
| jit_exact: reads staged before arrival | 99 | 103 | +4 % | 83 ± 46 [31, 174] | reproduced |
| jit_exact: SSD reads in the run | 2221 | 2368 | +7 % | 2649 ± 894 [1322, 3719] | reproduced |
| jit_exact: tokens read from the SSD, M (pages x 64) | 54.4 | 56.4 | +4 % | 61.8 ± 22.2 [28.3, 85.1] | reproduced |
| jit_exact: tokens written to the SSD, M (submitted, incl. skipped) | 31.8 | 30.6 | -4 % | 30.8 ± 0.7 [30.0, 31.9] | reproduced |
| jit_exact: host mirror, median of minutes 60-110, K | 647 | 647 | +0 % | 644 ± 2 [639, 647] | reproduced |
| jit_exact: host free, median, K | 15.9 | 11.1 | -30 % | 9.2 ± 2.0 [6.7, 11.4] | roughly |
| jit_exact: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 0 | (measured 0) | 1 ± 1 [0, 4] | not_reproduced |
| jit_exact: pinned tokens, p90, K | 58 | 57 | -2 % | 51 ± 7 [44, 67] | reproduced |
| jit_exact: pending writes (census 'pending_write'), p90, K | 14.8 | 35.6 | +140 % | 35.9 ± 11.0 [16.8, 48.5] | not_reproduced |
| jit_exact: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 65.1 | 5.3 | -92 % | 8.2 ± 5.6 [0.5, 16.4] | not_reproduced |
| ec_full: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.1 ± 1.1 [19.0, 23.0] | reproduced |
| ec_full: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 21.0 ± 0.9 [20.0, 22.0] | reproduced |
| ec_full: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 20.9 ± 1.0 [20.0, 23.0] | reproduced |
| ec_full: running requests, mean, minutes 90-110 | 19.7 | 19.7 | +0 % | 21.1 ± 1.1 [19.9, 23.4] | reproduced |
| ec_full: queued requests, median, minutes 90-110 | 60 | 59 | -2 % | 54 ± 8 [40, 64] | reproduced |
| ec_full: queued requests, median, minutes 60-90 | 37 | 36 | -3 % | 42 ± 19 [10, 60] | reproduced |
| ec_full: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.979 ± 0.001 [0.977, 0.981] | reproduced |
| ec_full: read thread busy share, minutes 30-60 | 0.235 | 0.268 | +14 % | 0.403 ± 0.229 [0.181, 0.707] | roughly |
| ec_full: write thread busy share, minutes 30-60 | 0.348 | 0.350 | +1 % | 0.421 ± 0.124 [0.300, 0.589] | reproduced |
| ec_full: read thread busy share, minutes 60-90 | 0.664 | 0.671 | +1 % | 0.712 ± 0.300 [0.268, 0.988] | reproduced |
| ec_full: write thread busy share, minutes 60-90 | 0.572 | 0.503 | -12 % | 0.570 ± 0.162 [0.320, 0.750] | roughly |
| ec_full: read thread busy share, minutes 90-110 | 0.999 | 0.996 | -0 % | 0.876 ± 0.146 [0.612, 0.995] | reproduced |
| ec_full: write thread busy share, minutes 90-110 | 0.736 | 0.668 | -9 % | 0.645 ± 0.075 [0.497, 0.734] | reproduced |
| ec_full: first cleaner pass, s | 5275 | 5190 | -2 % | 5209 ± 167 [4980, 5400] | reproduced |
| ec_full: cleaner passes before 6,600 s | 3 | 3 | +0 % | 3 ± 0 [2, 3] | reproduced |
| ec_full: reads refused host room before 6,600 s | 0 | 0 |  | 3 ± 6 [0, 17] | reproduced |
| ec_full: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| ec_full: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| ec_full: write hold switched on, times | 30 | 51 | +70 % | 70 ± 13 [54, 93] | not_reproduced |
| ec_full: reads staged before arrival | 1 | 1 | +0 % | 0 ± 0 [0, 1] | reproduced |
| ec_full: SSD reads in the run | 2678 | 2797 | +4 % | 3032 ± 827 [1817, 4015] | reproduced |
| ec_full: tokens read from the SSD, M (pages x 64) | 64.1 | 65.2 | +2 % | 68.3 ± 19.6 [39.3, 87.9] | reproduced |
| ec_full: tokens written to the SSD, M (submitted, incl. skipped) | 31.6 | 30.6 | -3 % | 30.9 ± 0.8 [30.0, 32.0] | reproduced |
| ec_full: host mirror, median of minutes 60-110, K | 647 | 648 | +0 % | 644 ± 2 [639, 647] | reproduced |
| ec_full: host free, median, K | 18.3 | 11.6 | -36 % | 9.6 ± 1.0 [8.2, 11.2] | roughly |
| ec_full: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 2 | (measured 0) | 1 ± 2 [0, 5] | not_reproduced |
| ec_full: pinned tokens, p90, K | 61 | 70 | +14 % | 52 ± 5 [43, 60] | reproduced |
| ec_full: pending writes (census 'pending_write'), p90, K | 12.1 | 32.7 | +170 % | 37.0 ± 9.4 [20.9, 51.1] | not_reproduced |
| ec_full: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 72.7 | 4.0 | -95 % | 7.6 ± 5.2 [1.2, 15.1] | not_reproduced |
| ec_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.1 ± 1.1 [19.0, 23.0] | reproduced |
| ec_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.9 ± 1.0 [19.0, 22.0] | reproduced |
| ec_exact: running requests, median, minutes 90-110 | 20.0 | 20.0 | +0 % | 20.8 ± 1.2 [19.0, 23.0] | reproduced |
| ec_exact: running requests, mean, minutes 90-110 | 19.8 | 19.8 | +0 % | 21.2 ± 1.2 [19.6, 23.3] | reproduced |
| ec_exact: queued requests, median, minutes 90-110 | 60 | 59 | -2 % | 54 ± 8 [40, 65] | reproduced |
| ec_exact: queued requests, median, minutes 60-90 | 36 | 36 | +0 % | 42 ± 19 [10, 60] | reproduced |
| ec_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.977 | +1 % | 0.980 ± 0.003 [0.975, 0.984] | reproduced |
| ec_exact: read thread busy share, minutes 30-60 | 0.143 | 0.148 | +4 % | 0.338 ± 0.247 [0.119, 0.692] | reproduced |
| ec_exact: write thread busy share, minutes 30-60 | 0.299 | 0.306 | +2 % | 0.392 ± 0.132 [0.273, 0.585] | reproduced |
| ec_exact: read thread busy share, minutes 60-90 | 0.555 | 0.573 | +3 % | 0.678 ± 0.338 [0.184, 0.992] | reproduced |
| ec_exact: write thread busy share, minutes 60-90 | 0.525 | 0.469 | -11 % | 0.555 ± 0.179 [0.281, 0.749] | roughly |
| ec_exact: read thread busy share, minutes 90-110 | 1.000 | 0.996 | -0 % | 0.858 ± 0.177 [0.516, 0.997] | reproduced |
| ec_exact: write thread busy share, minutes 90-110 | 0.742 | 0.669 | -10 % | 0.634 ± 0.087 [0.454, 0.714] | reproduced |
| ec_exact: first cleaner pass, s | 5252 | 5190 | -1 % | 5209 ± 156 [5010, 5370] | reproduced |
| ec_exact: cleaner passes before 6,600 s | 3 | 3 | +0 % | 3 ± 0 [2, 3] | reproduced |
| ec_exact: reads refused host room before 6,600 s | 0 | 0 |  | 1 ± 1 [0, 4] | reproduced |
| ec_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | none | none |  | none | reproduced |
| ec_exact: loop onset minute (5 in a row) | none | none |  | none | reproduced |
| ec_exact: write hold switched on, times | 29 | 62 | +114 % | 68 ± 14 [48, 90] | not_reproduced |
| ec_exact: reads staged before arrival | 39 | 46 | +18 % | 52 ± 63 [2, 188] | roughly |
| ec_exact: SSD reads in the run | 2335 | 2387 | +2 % | 2804 ± 921 [1401, 3884] | reproduced |
| ec_exact: tokens read from the SSD, M (pages x 64) | 55.3 | 56.1 | +1 % | 63.1 ± 21.8 [30.6, 85.2] | reproduced |
| ec_exact: tokens written to the SSD, M (submitted, incl. skipped) | 31.7 | 30.6 | -3 % | 30.9 ± 0.7 [30.0, 32.0] | reproduced |
| ec_exact: host mirror, median of minutes 60-110, K | 647 | 647 | -0 % | 644 ± 2 [640, 647] | reproduced |
| ec_exact: host free, median, K | 15.7 | 12.5 | -20 % | 9.4 ± 2.1 [6.6, 12.2] | roughly |
| ec_exact: pinned tokens (census 'pinned': pins of arrived and staged turns), median, K | 0 | 0 | (measured 0) | 1 ± 3 [0, 9] | not_reproduced |
| ec_exact: pinned tokens, p90, K | 55 | 59 | +7 % | 52 ± 8 [39, 66] | reproduced |
| ec_exact: pending writes (census 'pending_write'), p90, K | 16.6 | 33.9 | +104 % | 33.6 ± 9.3 [18.0, 45.6] | not_reproduced |
| ec_exact: host-only room held by pending writes (census 'h_write', with ancestors; sim: dirty host-only tokens), p90, K | 67.1 | 4.3 | -94 % | 6.3 ± 3.9 [0.6, 12.3] | not_reproduced |

### Campaign 32 (fixes off): client side, requests sent before 6,600 s

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: mean TTFT, s | 33.52 | 32.08 | -4 % | 29.87 ± 14.43 [8.65, 48.46] | reproduced |
| hicache: mean TTFT, minutes 0-30 | 0.42 | 0.40 | -4 % | 1.10 ± 0.88 [0.23, 3.05] | reproduced |
| hicache: mean TTFT, minutes 30-60 | 29.64 | 29.59 | -0 % | 29.18 ± 27.16 [1.79, 74.43] | reproduced |
| hicache: mean TTFT, minutes 60-90 | 83.14 | 73.88 | -11 % | 68.76 ± 38.21 [4.57, 97.96] | roughly |
| hicache: mean TTFT, minutes 90-110 | 111.82 | 99.12 | -11 % | 88.52 ± 10.49 [73.65, 102.12] | roughly |
| hicache: p50 TTFT, s | 5.01 | 5.67 | +13 % | 13.07 ± 21.37 [0.91, 65.40] | roughly |
| hicache: p90 TTFT, s | 99.81 | 90.23 | -10 % | 81.13 ± 31.63 [13.62, 104.52] | reproduced |
| hicache: requests per minute to 6,600 s | 87.8 | 90.1 | +3 % | 94.9 ± 16.5 [75.6, 118.2] | reproduced |
| hicache: requests per minute, minutes 90-109 | 46.9 | 51.7 | +10 % | 55.6 ± 4.0 [52.1, 63.7] | roughly |
| hicache: users with a request | 433 | 446 | +3 % | 474 ± 75 [389, 590] | roughly |
| hicache: prompt tokens from GPU, share | 0.309 | 0.308 | -1 % | 0.272 ± 0.051 [0.179, 0.337] | reproduced |
| hicache: from host, share | 0.286 | 0.282 | -1 % | 0.354 ± 0.130 [0.218, 0.535] | reproduced |
| hicache: from SSD, share | 0.045 | 0.068 | +49 % | 0.068 ± 0.007 [0.058, 0.078] | not_reproduced |
| hicache: from SSD, minutes 30-60 | 0.077 | 0.089 | +15 % | 0.082 ± 0.026 [0.043, 0.120] | roughly |
| hicache: from SSD, minutes 60-90 | 0.060 | 0.120 | +102 % | 0.117 ± 0.015 [0.091, 0.132] | not_reproduced |
| hicache: from SSD, minutes 90-110 | 0.059 | 0.130 | +122 % | 0.118 ± 0.011 [0.106, 0.137] | not_reproduced |
| hicache: history recomputed, share | 0.3141 | 0.2968 | -6 % | 0.2559 ± 0.1329 [0.0607, 0.4340] | reproduced |
| hicache: history recomputed, M tokens, minutes 30-60 | 19.02 | 19.00 | -0 % | 12.50 ± 11.95 [0.09, 29.61] | reproduced |
| hicache: history recomputed, M tokens, minutes 60-90 | 29.81 | 28.17 | -6 % | 22.98 ± 12.19 [0.17, 30.77] | reproduced |
| hicache: history recomputed, M tokens, minutes 90-110 | 20.65 | 20.25 | -2 % | 20.23 ± 0.58 [19.44, 21.20] | reproduced |
| hicache: returning requests that read from the SSD, minutes 90-110 | 0.151 | 0.252 | +67 % | 0.232 ± 0.017 [0.210, 0.258] | not_reproduced |
| hicache: their mean TTFT, s | 112.1 | 99.6 | -11 % | 90.0 ± 9.6 [77.2, 101.9] | roughly |
| hicache: the others' mean TTFT, s | 111.8 | 99.0 | -11 % | 88.1 ± 10.6 [73.2, 102.1] | roughly |
| jit_p50: mean TTFT, s | 17.97 | 15.35 | -15 % | 20.86 ± 10.51 [6.44, 33.69] | roughly |
| jit_p50: mean TTFT, minutes 0-30 | 0.42 | 0.39 | -7 % | 1.06 ± 0.78 [0.20, 2.74] | reproduced |
| jit_p50: mean TTFT, minutes 30-60 | 7.27 | 6.17 | -15 % | 15.00 ± 12.39 [1.87, 37.10] | roughly |
| jit_p50: mean TTFT, minutes 60-90 | 40.08 | 23.47 | -41 % | 41.14 ± 28.71 [4.96, 75.35] | not_reproduced |
| jit_p50: mean TTFT, minutes 90-110 | 69.30 | 84.69 | +22 % | 62.62 ± 22.90 [27.63, 84.84] | roughly |
| jit_p50: p50 TTFT, s | 6.47 | 5.33 | -18 % | 8.10 ± 4.00 [1.41, 12.83] | roughly |
| jit_p50: p90 TTFT, s | 63.89 | 55.63 | -13 % | 60.27 ± 27.25 [16.74, 84.12] | roughly |
| jit_p50: requests per minute to 6,600 s | 114.5 | 121.7 | +6 % | 112.0 ± 15.6 [92.7, 130.1] | roughly |
| jit_p50: requests per minute, minutes 90-109 | 65.2 | 56.3 | -14 % | 77.9 ± 23.9 [57.7, 121.4] | roughly |
| jit_p50: users with a request | 522 | 545 | +4 % | 538 ± 76 [449, 654] | roughly |
| jit_p50: prompt tokens from GPU, share | 0.233 | 0.222 | -5 % | 0.227 ± 0.048 [0.154, 0.308] | reproduced |
| jit_p50: from host, share | 0.448 | 0.492 | +10 % | 0.433 ± 0.107 [0.279, 0.580] | reproduced |
| jit_p50: from SSD, share | 0.150 | 0.148 | -2 % | 0.160 ± 0.033 [0.109, 0.202] | reproduced |
| jit_p50: from SSD, minutes 30-60 | 0.142 | 0.123 | -13 % | 0.165 ± 0.085 [0.065, 0.263] | roughly |
| jit_p50: from SSD, minutes 60-90 | 0.284 | 0.252 | -11 % | 0.264 ± 0.071 [0.142, 0.337] | roughly |
| jit_p50: from SSD, minutes 90-110 | 0.302 | 0.352 | +16 % | 0.312 ± 0.028 [0.262, 0.347] | roughly |
| jit_p50: history recomputed, share | 0.1138 | 0.0833 | -27 % | 0.1227 ± 0.0848 [0.0225, 0.2316] | roughly |
| jit_p50: history recomputed, M tokens, minutes 30-60 | 1.07 | 0.19 | -82 % | 4.28 ± 5.81 [0.09, 14.88] | not_reproduced |
| jit_p50: history recomputed, M tokens, minutes 60-90 | 16.31 | 9.67 | -41 % | 13.57 ± 11.20 [0.51, 24.43] | not_reproduced |
| jit_p50: history recomputed, M tokens, minutes 90-110 | 16.48 | 16.75 | +2 % | 14.22 ± 4.78 [5.91, 17.75] | reproduced |
| jit_p50: returning requests that read from the SSD, minutes 90-110 | 0.600 | 0.578 | -4 % | 0.487 ± 0.100 [0.315, 0.579] | reproduced |
| jit_p50: their mean TTFT, s | 71.3 | 87.4 | +23 % | 68.7 ± 16.5 [40.8, 85.6] | roughly |
| jit_p50: the others' mean TTFT, s | 66.2 | 80.6 | +22 % | 58.4 ± 25.8 [17.8, 84.1] | roughly |
| jit_exact: mean TTFT, s | 14.86 | 13.45 | -9 % | 19.46 ± 10.43 [5.78, 32.27] | reproduced |
| jit_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 | -10 % | 1.03 ± 0.79 [0.20, 2.75] | roughly |
| jit_exact: mean TTFT, minutes 30-60 | 6.18 | 5.86 | -5 % | 13.72 ± 11.04 [1.70, 33.29] | reproduced |
| jit_exact: mean TTFT, minutes 60-90 | 21.75 | 16.53 | -24 % | 36.83 ± 27.21 [4.55, 71.72] | roughly |
| jit_exact: mean TTFT, minutes 90-110 | 79.46 | 79.30 | -0 % | 57.98 ± 27.70 [22.95, 87.08] | reproduced |
| jit_exact: p50 TTFT, s | 6.12 | 6.24 | +2 % | 9.92 ± 5.29 [1.25, 16.81] | reproduced |
| jit_exact: p90 TTFT, s | 50.53 | 28.16 | -44 % | 55.02 ± 28.92 [15.64, 79.09] | not_reproduced |
| jit_exact: requests per minute to 6,600 s | 123.2 | 128.2 | +4 % | 115.5 ± 16.7 [95.3, 139.2] | roughly |
| jit_exact: requests per minute, minutes 90-109 | 59.2 | 58.7 | -1 % | 85.1 ± 29.4 [56.1, 128.9] | reproduced |
| jit_exact: users with a request | 550 | 568 | +3 % | 551 ± 78 [453, 660] | roughly |
| jit_exact: prompt tokens from GPU, share | 0.239 | 0.232 | -3 % | 0.238 ± 0.057 [0.147, 0.341] | reproduced |
| jit_exact: from host, share | 0.535 | 0.555 | +4 % | 0.479 ± 0.119 [0.295, 0.658] | reproduced |
| jit_exact: from SSD, share | 0.098 | 0.101 | +4 % | 0.119 ± 0.048 [0.051, 0.174] | reproduced |
| jit_exact: from SSD, minutes 30-60 | 0.031 | 0.032 | +5 % | 0.089 ± 0.078 [0.017, 0.212] | reproduced |
| jit_exact: from SSD, minutes 60-90 | 0.192 | 0.178 | -7 % | 0.216 ± 0.120 [0.047, 0.336] | reproduced |
| jit_exact: from SSD, minutes 90-110 | 0.331 | 0.370 | +12 % | 0.294 ± 0.066 [0.186, 0.354] | roughly |
| jit_exact: history recomputed, share | 0.0727 | 0.0553 | -24 % | 0.1063 ± 0.0833 [0.0149, 0.2149] | roughly |
| jit_exact: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -4 % | 3.56 ± 5.12 [0.09, 13.43] | reproduced |
| jit_exact: history recomputed, M tokens, minutes 60-90 | 7.69 | 2.48 | -68 % | 11.90 ± 11.18 [0.12, 24.30] | not_reproduced |
| jit_exact: history recomputed, M tokens, minutes 90-110 | 15.72 | 16.12 | +3 % | 12.77 ± 5.54 [4.26, 17.80] | reproduced |
| jit_exact: returning requests that read from the SSD, minutes 90-110 | 0.649 | 0.598 | -8 % | 0.442 ± 0.151 [0.230, 0.569] | reproduced |
| jit_exact: their mean TTFT, s | 79.7 | 81.9 | +3 % | 64.1 ± 22.6 [35.5, 88.3] | reproduced |
| jit_exact: the others' mean TTFT, s | 78.9 | 75.4 | -5 % | 54.6 ± 28.8 [19.1, 85.4] | reproduced |
| ec_full: mean TTFT, s | 14.64 | 13.48 | -8 % | 17.54 ± 8.78 [5.86, 31.00] | reproduced |
| ec_full: mean TTFT, minutes 0-30 | 0.45 | 0.39 | -14 % | 1.05 ± 0.79 [0.21, 2.76] | roughly |
| ec_full: mean TTFT, minutes 30-60 | 6.42 | 5.89 | -8 % | 11.46 ± 7.70 [1.82, 22.91] | reproduced |
| ec_full: mean TTFT, minutes 60-90 | 20.01 | 16.62 | -17 % | 30.46 ± 24.64 [4.58, 72.39] | roughly |
| ec_full: mean TTFT, minutes 90-110 | 80.17 | 79.48 | -1 % | 58.28 ± 26.42 [19.72, 87.34] | reproduced |
| ec_full: p50 TTFT, s | 6.51 | 6.31 | -3 % | 9.75 ± 5.04 [1.38, 15.43] | reproduced |
| ec_full: p90 TTFT, s | 51.24 | 30.87 | -40 % | 52.47 ± 26.15 [15.60, 79.97] | not_reproduced |
| ec_full: requests per minute to 6,600 s | 125.0 | 128.1 | +2 % | 120.2 ± 12.8 [99.1, 134.8] | reproduced |
| ec_full: requests per minute, minutes 90-109 | 58.9 | 58.5 | -1 % | 83.7 ± 30.6 [58.5, 141.9] | reproduced |
| ec_full: users with a request | 557 | 569 | +2 % | 572 ± 63 [485, 669] | reproduced |
| ec_full: prompt tokens from GPU, share | 0.217 | 0.220 | +2 % | 0.217 ± 0.051 [0.152, 0.317] | reproduced |
| ec_full: from host, share | 0.534 | 0.539 | +1 % | 0.496 ± 0.080 [0.389, 0.625] | reproduced |
| ec_full: from SSD, share | 0.130 | 0.128 | -1 % | 0.148 ± 0.041 [0.084, 0.196] | reproduced |
| ec_full: from SSD, minutes 30-60 | 0.082 | 0.079 | -3 % | 0.128 ± 0.072 [0.046, 0.234] | reproduced |
| ec_full: from SSD, minutes 60-90 | 0.230 | 0.208 | -10 % | 0.237 ± 0.090 [0.087, 0.356] | reproduced |
| ec_full: from SSD, minutes 90-110 | 0.321 | 0.358 | +11 % | 0.306 ± 0.053 [0.217, 0.349] | roughly |
| ec_full: history recomputed, share | 0.0634 | 0.0564 | -11 % | 0.0798 ± 0.0672 [0.0045, 0.1872] | reproduced |
| ec_full: history recomputed, M tokens, minutes 30-60 | 0.12 | 0.11 | -4 % | 1.51 ± 2.76 [0.09, 7.77] | reproduced |
| ec_full: history recomputed, M tokens, minutes 60-90 | 4.94 | 2.58 | -48 % | 8.17 ± 10.31 [0.17, 25.26] | not_reproduced |
| ec_full: history recomputed, M tokens, minutes 90-110 | 15.81 | 16.35 | +3 % | 12.86 ± 6.40 [0.78, 17.79] | reproduced |
| ec_full: returning requests that read from the SSD, minutes 90-110 | 0.639 | 0.595 | -7 % | 0.473 ± 0.135 [0.268, 0.606] | reproduced |
| ec_full: their mean TTFT, s | 80.7 | 80.5 | -0 % | 62.6 ± 22.6 [28.1, 87.8] | reproduced |
| ec_full: the others' mean TTFT, s | 79.6 | 78.0 | -2 % | 55.6 ± 27.5 [16.6, 86.9] | reproduced |
| ec_exact: mean TTFT, s | 14.95 | 13.58 | -9 % | 19.57 ± 10.62 [5.70, 33.27] | reproduced |
| ec_exact: mean TTFT, minutes 0-30 | 0.40 | 0.36 | -10 % | 1.03 ± 0.79 [0.19, 2.73] | reproduced |
| ec_exact: mean TTFT, minutes 30-60 | 6.39 | 5.82 | -9 % | 13.49 ± 11.12 [1.72, 34.29] | reproduced |
| ec_exact: mean TTFT, minutes 60-90 | 22.09 | 16.77 | -24 % | 38.16 ± 28.87 [4.39, 74.51] | roughly |
| ec_exact: mean TTFT, minutes 90-110 | 78.47 | 81.78 | +4 % | 59.99 ± 27.50 [19.45, 85.69] | reproduced |
| ec_exact: p50 TTFT, s | 6.21 | 6.09 | -2 % | 9.50 ± 4.85 [1.30, 14.64] | reproduced |
| ec_exact: p90 TTFT, s | 50.09 | 28.53 | -43 % | 55.69 ± 28.63 [16.31, 79.88] | not_reproduced |
| ec_exact: requests per minute to 6,600 s | 123.0 | 127.6 | +4 % | 115.0 ± 16.9 [93.0, 135.1] | roughly |
| ec_exact: requests per minute, minutes 90-109 | 59.9 | 57.2 | -4 % | 82.6 ± 31.0 [57.4, 142.3] | reproduced |
| ec_exact: users with a request | 547 | 568 | +4 % | 551 ± 79 [452, 670] | roughly |
| ec_exact: prompt tokens from GPU, share | 0.238 | 0.234 | -2 % | 0.240 ± 0.056 [0.160, 0.344] | reproduced |
| ec_exact: from host, share | 0.528 | 0.550 | +4 % | 0.467 ± 0.122 [0.264, 0.632] | reproduced |
| ec_exact: from SSD, share | 0.105 | 0.102 | -3 % | 0.128 ± 0.048 [0.061, 0.182] | reproduced |
| ec_exact: from SSD, minutes 30-60 | 0.048 | 0.043 | -10 % | 0.107 ± 0.081 [0.029, 0.239] | reproduced |
| ec_exact: from SSD, minutes 60-90 | 0.200 | 0.180 | -10 % | 0.229 ± 0.124 [0.058, 0.350] | roughly |
| ec_exact: from SSD, minutes 90-110 | 0.330 | 0.346 | +5 % | 0.302 ± 0.064 [0.197, 0.360] | reproduced |
| ec_exact: history recomputed, share | 0.0727 | 0.0588 | -19 % | 0.1073 ± 0.0858 [0.0043, 0.2307] | roughly |
| ec_exact: history recomputed, M tokens, minutes 30-60 | 0.13 | 0.11 | -11 % | 3.25 ± 5.19 [0.09, 14.10] | reproduced |
| ec_exact: history recomputed, M tokens, minutes 60-90 | 7.96 | 2.81 | -65 % | 12.09 ± 11.56 [0.12, 25.20] | not_reproduced |
| ec_exact: history recomputed, M tokens, minutes 90-110 | 15.42 | 16.85 | +9 % | 12.96 ± 6.49 [0.85, 17.46] | reproduced |
| ec_exact: returning requests that read from the SSD, minutes 90-110 | 0.645 | 0.572 | -11 % | 0.472 ± 0.148 [0.246, 0.598] | roughly |
| ec_exact: their mean TTFT, s | 79.7 | 83.2 | +4 % | 64.4 ± 22.9 [29.8, 86.4] | reproduced |
| ec_exact: the others' mean TTFT, s | 76.9 | 79.8 | +4 % | 57.5 ± 28.8 [16.0, 84.8] | reproduced |

### Campaign 32 (fixes off): server side

| anchor | measured | simulated, seed 0 | error | simulated, fresh seeds 5-12: mean ± sd [min, max] | verdict |
|---|---|---|---|---|---|
| hicache: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.0 ± 1.4 [19.0, 23.0] | reproduced |
| hicache: running requests, median, minutes 60-90 | 22.0 | 21.0 | -5 % | 20.8 ± 0.7 [20.0, 22.0] | reproduced |
| hicache: running requests, median, minutes 90-110 | 20.0 | 19.0 | -5 % | 20.6 ± 1.2 [19.0, 22.0] | reproduced |
| hicache: running requests, mean, minutes 90-110 | 19.9 | 19.2 | -4 % | 20.8 ± 1.1 [19.3, 22.2] | reproduced |
| hicache: queued requests, median, minutes 90-110 | 86 | 84 | -2 % | 80 ± 8 [68, 87] | reproduced |
| hicache: queued requests, median, minutes 60-90 | 72 | 71 | -1 % | 65 ± 31 [10, 86] | reproduced |
| hicache: GPU pool used, median, minutes 90-110 | 0.970 | 0.980 | +1 % | 0.978 ± 0.002 [0.976, 0.981] | reproduced |
| hicache: read thread busy share, minutes 30-60 | 0.374 | 0.386 | +3 % | 0.296 ± 0.109 [0.164, 0.434] | reproduced |
| hicache: write thread busy share, minutes 30-60 | 0.743 | 0.757 | +2 % | 0.599 ± 0.285 [0.284, 1.000] | reproduced |
| hicache: read thread busy share, minutes 60-90 | 0.469 | 0.454 | -3 % | 0.419 ± 0.066 [0.288, 0.464] | reproduced |
| hicache: write thread busy share, minutes 60-90 | 1.000 | 1.000 | +0 % | 0.845 ± 0.277 [0.325, 1.000] | reproduced |
| hicache: read thread busy share, minutes 90-110 | 0.404 | 0.492 | +22 % | 0.463 ± 0.029 [0.419, 0.507] | roughly |
| hicache: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 0.998 ± 0.005 [0.987, 1.000] | reproduced |
| hicache: first cleaner pass, s | none | none |  | 5190 ± 255 [5010, 5370] | reproduced |
| hicache: cleaner passes before 6,600 s | 0 | 0 |  | 0 ± 1 [0, 2] | reproduced |
| hicache: SSD reads in the run | 2259 | 2311 | +2 % | 2126 ± 379 [1507, 2547] | reproduced |
| hicache: tokens read from the SSD, M (pages x 64) | 33.3 | 39.1 | +17 % | 36.1 ± 3.7 [30.6, 40.2] | roughly |
| hicache: tokens written to the SSD, M (submitted, incl. skipped) | 83.4 | 71.1 | -15 % | 65.0 ± 11.8 [46.3, 78.8] | roughly |
| jit_p50: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 20.9 ± 1.1 [19.0, 23.0] | reproduced |
| jit_p50: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.9 ± 1.2 [19.0, 23.0] | reproduced |
| jit_p50: running requests, median, minutes 90-110 | 20.0 | 19.0 | -5 % | 20.6 ± 0.7 [20.0, 22.0] | reproduced |
| jit_p50: running requests, mean, minutes 90-110 | 20.3 | 19.1 | -6 % | 20.6 ± 0.7 [19.6, 21.6] | roughly |
| jit_p50: queued requests, median, minutes 90-110 | 73 | 78 | +6 % | 72 ± 14 [46, 83] | reproduced |
| jit_p50: queued requests, median, minutes 60-90 | 57 | 44 | -24 % | 53 ± 27 [12, 78] | roughly |
| jit_p50: GPU pool used, median, minutes 90-110 | 0.970 | 0.979 | +1 % | 0.977 ± 0.002 [0.973, 0.980] | reproduced |
| jit_p50: read thread busy share, minutes 30-60 | 0.636 | 0.571 | -10 % | 0.548 ± 0.239 [0.258, 0.892] | roughly |
| jit_p50: write thread busy share, minutes 30-60 | 0.710 | 0.527 | -26 % | 0.585 ± 0.218 [0.341, 0.847] | roughly |
| jit_p50: read thread busy share, minutes 60-90 | 0.802 | 0.836 | +4 % | 0.696 ± 0.105 [0.580, 0.858] | reproduced |
| jit_p50: write thread busy share, minutes 60-90 | 0.953 | 0.796 | -17 % | 0.837 ± 0.193 [0.507, 1.000] | roughly |
| jit_p50: read thread busy share, minutes 90-110 | 0.757 | 0.649 | -14 % | 0.677 ± 0.124 [0.571, 0.878] | roughly |
| jit_p50: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 0.959 ± 0.075 [0.827, 1.000] | reproduced |
| jit_p50: first cleaner pass, s | 6798 | 5730 | -16 % | 5748 ± 832 [5010, 7110] | not_reproduced |
| jit_p50: cleaner passes before 6,600 s | 0 | 1 | (measured 0) | 1 ± 1 [0, 3] | not_reproduced |
| jit_p50: reads refused host room before 6,600 s | 2414 | 1742 | -28 % | 2317 ± 1523 [418, 4158] | roughly |
| jit_p50: loop onset minute (3 minutes in a row with 5 or more refusals) | 42 | 78 | +86 % | 71 ± 26 [40, 105] | not_reproduced |
| jit_p50: loop onset minute (5 in a row) | 67 | 78 | +16 % | 72 ± 25 [40, 105] | roughly |
| jit_p50: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| jit_p50: reads staged before arrival | 804 | 817 | +2 % | 637 ± 195 [303, 876] | reproduced |
| jit_p50: SSD reads in the run | 3707 | 3183 | -14 % | 3364 ± 479 [2538, 3857] | roughly |
| jit_p50: tokens read from the SSD, M (pages x 64) | 63.1 | 64.4 | +2 % | 59.2 ± 4.4 [53.8, 66.8] | reproduced |
| jit_p50: tokens written to the SSD, M (submitted, incl. skipped) | 52.6 | 46.1 | -12 % | 49.6 ± 9.4 [36.9, 60.3] | roughly |
| jit_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 20.8 ± 0.9 [19.0, 22.0] | reproduced |
| jit_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.8 ± 0.7 [20.0, 22.0] | reproduced |
| jit_exact: running requests, median, minutes 90-110 | 19.0 | 19.0 | +0 % | 20.6 ± 0.9 [19.0, 22.0] | reproduced |
| jit_exact: running requests, mean, minutes 90-110 | 19.5 | 19.1 | -2 % | 20.7 ± 0.9 [19.0, 21.8] | reproduced |
| jit_exact: queued requests, median, minutes 90-110 | 78 | 75 | -4 % | 70 ± 17 [45, 83] | reproduced |
| jit_exact: queued requests, median, minutes 60-90 | 38 | 35 | -9 % | 50 ± 27 [10, 79] | reproduced |
| jit_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.969 | -0 % | 0.979 ± 0.002 [0.975, 0.983] | reproduced |
| jit_exact: read thread busy share, minutes 30-60 | 0.089 | 0.115 | +29 % | 0.256 ± 0.209 [0.072, 0.545] | roughly |
| jit_exact: write thread busy share, minutes 30-60 | 0.274 | 0.291 | +6 % | 0.425 ± 0.212 [0.257, 0.795] | reproduced |
| jit_exact: read thread busy share, minutes 60-90 | 0.578 | 0.572 | -1 % | 0.504 ± 0.226 [0.147, 0.731] | reproduced |
| jit_exact: write thread busy share, minutes 60-90 | 0.699 | 0.505 | -28 % | 0.696 ± 0.312 [0.270, 1.000] | roughly |
| jit_exact: read thread busy share, minutes 90-110 | 0.793 | 0.669 | -16 % | 0.623 ± 0.076 [0.529, 0.781] | roughly |
| jit_exact: write thread busy share, minutes 90-110 | 1.000 | 1.000 | +0 % | 0.904 ± 0.150 [0.641, 1.000] | reproduced |
| jit_exact: first cleaner pass, s | 5779 | 5190 | -10 % | 5544 ± 513 [4980, 6330] | not_reproduced |
| jit_exact: cleaner passes before 6,600 s | 1 | 2 | +100 % | 1 ± 1 [0, 3] | not_reproduced |
| jit_exact: reads refused host room before 6,600 s | 1711 | 1292 | -24 % | 2208 ± 1687 [284, 4346] | roughly |
| jit_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | 80 | 88 | +10 % | 75 ± 26 [40, 104] | reproduced |
| jit_exact: loop onset minute (5 in a row) | 80 | 88 | +10 % | 75 ± 26 [40, 104] | reproduced |
| jit_exact: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| jit_exact: reads staged before arrival | 59 | 47 | -20 % | 48 ± 42 [16, 127] | roughly |
| jit_exact: SSD reads in the run | 2507 | 2199 | -12 % | 2558 ± 820 [1454, 3606] | roughly |
| jit_exact: tokens read from the SSD, M (pages x 64) | 42.5 | 41.9 | -2 % | 42.0 ± 10.5 [26.1, 53.2] | reproduced |
| jit_exact: tokens written to the SSD, M (submitted, incl. skipped) | 46.2 | 42.8 | -7 % | 48.0 ± 9.8 [36.0, 59.8] | reproduced |
| ec_full: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.1 ± 1.1 [19.0, 23.0] | reproduced |
| ec_full: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 21.0 ± 1.1 [20.0, 23.0] | reproduced |
| ec_full: running requests, median, minutes 90-110 | 19.0 | 19.0 | +0 % | 20.5 ± 0.9 [19.0, 22.0] | reproduced |
| ec_full: running requests, mean, minutes 90-110 | 19.4 | 19.1 | -1 % | 20.6 ± 1.0 [18.7, 22.0] | reproduced |
| ec_full: queued requests, median, minutes 90-110 | 78 | 76 | -3 % | 70 ± 16 [44, 86] | reproduced |
| ec_full: queued requests, median, minutes 60-90 | 39 | 36 | -8 % | 47 ± 24 [10, 77] | reproduced |
| ec_full: GPU pool used, median, minutes 90-110 | 0.970 | 0.978 | +1 % | 0.979 ± 0.002 [0.976, 0.983] | reproduced |
| ec_full: read thread busy share, minutes 30-60 | 0.233 | 0.276 | +19 % | 0.394 ± 0.220 [0.181, 0.747] | roughly |
| ec_full: write thread busy share, minutes 30-60 | 0.373 | 0.362 | -3 % | 0.445 ± 0.154 [0.300, 0.647] | reproduced |
| ec_full: read thread busy share, minutes 60-90 | 0.701 | 0.680 | -3 % | 0.652 ± 0.215 [0.274, 0.928] | reproduced |
| ec_full: write thread busy share, minutes 60-90 | 0.767 | 0.571 | -26 % | 0.694 ± 0.238 [0.323, 1.000] | roughly |
| ec_full: read thread busy share, minutes 90-110 | 0.780 | 0.648 | -17 % | 0.648 ± 0.061 [0.597, 0.783] | roughly |
| ec_full: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 0.913 ± 0.160 [0.637, 1.000] | reproduced |
| ec_full: first cleaner pass, s | 5584 | 5190 | -7 % | 5571 ± 740 [5010, 7170] | roughly |
| ec_full: cleaner passes before 6,600 s | 1 | 1 | +0 % | 2 ± 1 [0, 3] | reproduced |
| ec_full: reads refused host room before 6,600 s | 1403 | 1207 | -14 % | 1602 ± 1263 [0, 3439] | reproduced |
| ec_full: loop onset minute (3 minutes in a row with 5 or more refusals) | 84 | 88 | +5 % | 79 ± 19 [50, 102] | reproduced |
| ec_full: loop onset minute (5 in a row) | 84 | 88 | +5 % | 79 ± 19 [50, 102] | reproduced |
| ec_full: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| ec_full: reads staged before arrival | 2 | 1 | -50 % | 34 ± 97 [0, 275] | not_reproduced |
| ec_full: SSD reads in the run | 2936 | 2675 | -9 % | 2973 ± 647 [1934, 3818] | reproduced |
| ec_full: tokens read from the SSD, M (pages x 64) | 53.6 | 51.3 | -4 % | 54.3 ± 10.7 [37.7, 73.0] | reproduced |
| ec_full: tokens written to the SSD, M (submitted, incl. skipped) | 45.1 | 43.2 | -4 % | 44.5 ± 8.5 [32.7, 56.7] | reproduced |
| ec_exact: running requests, median, minutes 30-60 | 22.0 | 22.0 | +0 % | 21.0 ± 1.2 [19.0, 23.0] | reproduced |
| ec_exact: running requests, median, minutes 60-90 | 20.0 | 20.0 | +0 % | 20.9 ± 0.6 [20.0, 22.0] | reproduced |
| ec_exact: running requests, median, minutes 90-110 | 19.0 | 19.0 | +0 % | 20.6 ± 0.9 [19.0, 22.0] | reproduced |
| ec_exact: running requests, mean, minutes 90-110 | 19.3 | 19.2 | -1 % | 20.7 ± 0.8 [19.4, 21.7] | reproduced |
| ec_exact: queued requests, median, minutes 90-110 | 77 | 77 | +0 % | 71 ± 16 [44, 85] | reproduced |
| ec_exact: queued requests, median, minutes 60-90 | 38 | 35 | -8 % | 50 ± 27 [10, 79] | reproduced |
| ec_exact: GPU pool used, median, minutes 90-110 | 0.970 | 0.976 | +1 % | 0.979 ± 0.003 [0.972, 0.982] | reproduced |
| ec_exact: read thread busy share, minutes 30-60 | 0.137 | 0.149 | +9 % | 0.311 ± 0.211 [0.119, 0.583] | reproduced |
| ec_exact: write thread busy share, minutes 30-60 | 0.313 | 0.306 | -2 % | 0.440 ± 0.207 [0.273, 0.811] | reproduced |
| ec_exact: read thread busy share, minutes 60-90 | 0.599 | 0.571 | -5 % | 0.524 ± 0.215 [0.189, 0.726] | reproduced |
| ec_exact: write thread busy share, minutes 60-90 | 0.718 | 0.503 | -30 % | 0.703 ± 0.304 [0.286, 1.000] | roughly |
| ec_exact: read thread busy share, minutes 90-110 | 0.797 | 0.623 | -22 % | 0.623 ± 0.047 [0.544, 0.706] | roughly |
| ec_exact: write thread busy share, minutes 90-110 | 0.999 | 1.000 | +0 % | 0.908 ± 0.163 [0.641, 1.000] | reproduced |
| ec_exact: first cleaner pass, s | 5814 | 5190 | -11 % | 5835 ± 840 [4980, 7230] | not_reproduced |
| ec_exact: cleaner passes before 6,600 s | 1 | 2 | +100 % | 1 ± 1 [0, 3] | not_reproduced |
| ec_exact: reads refused host room before 6,600 s | 1636 | 1251 | -24 % | 2060 ± 1558 [0, 4122] | roughly |
| ec_exact: loop onset minute (3 minutes in a row with 5 or more refusals) | 79 | 87 | +10 % | 70 ± 23 [40, 102] | roughly |
| ec_exact: loop onset minute (5 in a row) | 79 | 87 | +10 % | 72 ± 26 [40, 113] | roughly |
| ec_exact: write hold switched on, times | 0 | 0 |  | 0 ± 0 [0, 0] | reproduced |
| ec_exact: reads staged before arrival | 53 | 46 | -13 % | 49 ± 65 [0, 190] | roughly |
| ec_exact: SSD reads in the run | 2643 | 2300 | -13 % | 2695 ± 800 [1585, 3735] | roughly |
| ec_exact: tokens read from the SSD, M (pages x 64) | 44.8 | 42.0 | -6 % | 44.8 ± 9.2 [30.5, 53.5] | reproduced |
| ec_exact: tokens written to the SSD, M (submitted, incl. skipped) | 46.5 | 43.7 | -6 % | 47.2 ± 10.3 [32.7, 59.8] | reproduced |

### Paired differences between arms (B - A on the requests both sent before 6,600 s; interval over users)

| campaign | pair | measured (referee's recomputation) | simulated, seed 0 | simulated, fresh seeds: mean [min, max] | sign over the 9 seeds | verdict |
|---|---|---|---|---|---|---|
| campaign34 | jit_p50 -> ec_full | -0.62 [-1.03, -0.22] on 15416 | -0.59 [-1.02, -0.19] on 15515 | -0.34 [-0.47, -0.20] | same | reproduced |
| campaign34 | jit_p50 -> jit_exact | -0.67 [-1.07, -0.29] on 15458 | -0.65 [-1.03, -0.28] on 15555 | -0.34 [-0.48, -0.20] | same | reproduced |
| campaign34 | ec_full -> jit_exact | -0.06 [-0.19, +0.05] on 15529 | -0.05 [-0.18, +0.08] on 15604 | -0.00 [-0.10, +0.09] | changes | reproduced |
| campaign34 | jit_p50 -> ec_exact | -1.01 [-1.43, -0.60] on 15464 | -0.60 [-1.02, -0.19] on 15514 | -0.39 [-0.50, -0.16] | same | roughly |
| campaign34 | ec_full -> ec_exact | -0.40 [-0.51, -0.29] on 15544 | +0.01 [-0.11, +0.12] on 15593 | -0.04 [-0.12, +0.06] | changes | not_reproduced |
| campaign34 | jit_exact -> ec_exact | -0.32 [-0.46, -0.20] on 15571 | +0.04 [-0.08, +0.17] on 15612 | -0.04 [-0.15, +0.08] | changes | not_reproduced |
| campaign34 | hicache -> ec_exact | -26.96 [-30.48, -23.61] on 9967 | -26.63 [-30.10, -23.35] on 9998 | -20.38 [-34.06, -5.14] | same | reproduced |
| campaign34 | hicache -> ec_full | -26.71 [-30.21, -23.39] on 9967 | -26.63 [-30.10, -23.34] on 9998 | -20.32 [-33.93, -5.04] | same | reproduced |
| campaign32 | jit_p50 -> ec_full | -8.60 [-9.96, -7.30] on 12589 | -5.50 [-6.53, -4.55] on 13390 | -5.85 [-20.86, +1.87] | changes | roughly |
| campaign32 | jit_p50 -> jit_exact | -7.75 [-9.00, -6.59] on 12590 | -5.49 [-6.51, -4.54] on 13390 | -2.48 [-7.73, +1.69] | changes | roughly |
| campaign32 | ec_full -> jit_exact | +1.26 [+0.93, +1.61] on 13543 | -0.04 [-0.15, +0.07] on 14073 | +3.49 [-2.01, +19.13] | changes | not_reproduced |
| campaign32 | jit_p50 -> ec_exact | -7.54 [-8.82, -6.37] on 12589 | -5.23 [-6.20, -4.32] on 13390 | -2.36 [-7.03, +1.82] | changes | roughly |
| campaign32 | ec_full -> ec_exact | +1.43 [+1.07, +1.81] on 13530 | +0.43 [+0.29, +0.57] on 14033 | +3.71 [-0.23, +20.41] | changes | not_reproduced |
| campaign32 | jit_exact -> ec_exact | +0.16 [+0.02, +0.33] on 13505 | +0.47 [+0.31, +0.63] on 14027 | +0.26 [-2.21, +1.89] | changes | not_reproduced |
| campaign32 | hicache -> ec_exact | -28.28 [-32.38, -24.48] on 9653 | -27.14 [-30.93, -23.61] on 9914 | -17.83 [-30.34, -5.15] | same | reproduced |
| campaign32 | hicache -> ec_full | -28.25 [-32.38, -24.43] on 9653 | -27.11 [-30.92, -23.58] on 9914 | -20.25 [-30.99, -5.06] | same | reproduced |

### The fixes: fixes on minus fixes off, mean TTFT of each run's own requests, s

| arm | measured (34 - 32) | simulated, seed 0 | simulated, fresh seeds: mean [min, max] |
|---|---|---|---|
| hicache | -1.63 | -0.50 | -1.23 [-2.70, -0.03] |
| jit_p50 | -6.50 | -4.24 | -8.03 [-17.03, -0.84] |
| jit_exact | -3.94 | -2.86 | -6.91 [-15.90, -0.40] |
| ec_full | -3.68 | -2.85 | -4.98 [-14.45, -0.15] |
| ec_exact | -4.29 | -2.98 | -7.06 [-16.97, -0.01] |
