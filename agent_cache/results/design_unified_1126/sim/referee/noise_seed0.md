seed 0, setting 'noise=0.005', replicas [11, 12, 13, 14, 15, 16]

| arm | mean TTFT of the replicas, s | max - min | same arm, replica against replica: paired difference, mean abs / max abs, s |
|---|---|---|---|
| hicache | 31.50, 31.54, 31.71, 31.35, 31.72, 31.63 | 0.37 | 0.29 / 0.68 |
| jit_p50 | 11.23, 11.21, 11.12, 11.31, 11.12, 11.08 | 0.23 | 0.11 / 0.27 |
| jit_exact | 10.57, 10.58, 10.61, 10.62, 10.62, 10.60 | 0.05 | 0.04 / 0.08 |
| ec_full | 10.61, 10.67, 10.57, 10.58, 10.64, 10.63 | 0.10 | 0.06 / 0.12 |
| ec_exact | 10.58, 10.53, 10.56, 10.55, 10.63, 10.47 | 0.15 | 0.10 / 0.28 |

| pair A -> B | paired B - A over all replica pairs (A's replica x B's replica): mean [min, max], s | share of replica pairs with the sign of the mean |
|---|---|---|
| jit_p50 -> ec_full | -0.66 [-0.87, -0.48] | 100 % of 36 |
| jit_p50 -> jit_exact | -0.67 [-0.87, -0.53] | 100 % of 36 |
| ec_full -> jit_exact | -0.01 [-0.11, +0.09] | 53 % of 36 |
| ec_full -> ec_exact | -0.08 [-0.30, +0.09] | 83 % of 36 |
| jit_exact -> ec_exact | -0.07 [-0.26, +0.08] | 83 % of 36 |
| hicache -> ec_full | -26.57 [-26.76, -26.30] | 100 % of 36 |
