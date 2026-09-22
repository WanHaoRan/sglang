# Old vs new: HiCache evaluation, 70B and 32B with fp8_e5m2 KV, nixl backend

- **old**: A100 SXM4, local NVMe (c5); `20260917_a100_gcp_32b70b_fp8kv`
- **new**: H200, attached SSD (c6); `20260921_h200_nebius_32b70b_fp8kv`
- same image, same sglang commit under `python/`, same server flags and client parameters; differences forced by the box are listed in `DEVIATIONS.md`

### Llama-3.3-70B-AWQ: setup

| metric | old | new | new / old |
|---|---:|---:|---:|
| device pool (L1) (tokens) | 194,816 | 531,008 | 2.73x |
| host pool (L2) (tokens) | 610,368 | 976,576 | 1.60x |
| KV bytes per token (B) | 163,840 | 163,840 | 1.00x |
| attention backend | triton | triton | - |
| KV cache dtype | torch.float8_e5m2 | torch.float8_e5m2 | - |
| weight kernel path | The model is convertible to awq_marlin during runtime. Using awq_marlin kernel. | The model is convertible to awq_marlin during runtime. Using awq_marlin kernel. | - |

### Llama-3.3-70B-AWQ: Exp 0: tier attribution, one 4096-token prompt, n=1

| metric | old | new | new / old |
|---|---:|---:|---:|
| TTFT 1 cold recompute (s) | 3.288 | 2.106 | 0.64x |
| TTFT 3 L1 hit (s) | 0.132 | 0.078 | 0.59x |
| TTFT 4 L2 hit (s) | 0.150 | 0.087 | 0.58x |
| TTFT 5 L3 hit, page cache dropped (s) | 1.006 | 0.426 | 0.42x |
| TTFT 6 L3 hit, page cache warm (s) | 1.037 | 0.342 | 0.33x |

Attribution (device/host/storage tokens) new run: 1: None/None/None; 3: 4032/0/0; 4: 0/4032/0; 5: 0/0/4032; 6: 0/0/4032. L3 files after backup: old 128, new 128. n=1 and the first long prefill after boot pays kernel warm-up: use Exp 1 for timings.

### Llama-3.3-70B-AWQ: Exp 1: tier cost fits over 7 lengths (512 to 32512), medians of n=3

| metric | old | new | new / old |
|---|---:|---:|---:|
| P = marginal prefill rate (tok/s) | 819.691 | 1,382.444 | 1.69x |
| recompute bar b*P (GiB/s) | 0.125 | 0.211 | 1.69x |
| recompute fit intercept (s) | -1.543 | -0.946 | 0.61x |
| L1 delivered rate (fit) (GiB/s) | 12.090 | 18.958 | 1.57x |
| L1 fit intercept (s) | 0.080 | 0.050 | 0.62x |
| L2 delivered rate (fit) (GiB/s) | 9.006 | 18.886 | 2.10x |
| L2 fit intercept (s) | 0.072 | 0.048 | 0.66x |
| L3 delivered rate (fit) (GiB/s) | 0.629 | 2.828 | 4.49x |
| L3 fit intercept (s) | 0.036 | 0.191 | 5.28x |
| host to device, differential (L2 - L1 slopes) (GiB/s) | - | - | - |
| disk read, differential (L3 - L2 slopes) (GiB/s) | 0.677 | 3.325 | 4.91x |
| L2 rate / recompute bar (x) | 72.007 | 89.529 | 1.24x |
| L2 beats recompute above L (fit) (tokens) | 1,342.430 | 1,389.302 | 1.03x |
| L3 rate / recompute bar (x) | 5.033 | 13.404 | 2.66x |
| L3 beats recompute above L (fit) (tokens) | 1,615.465 | 1,698.292 | 1.05x |

A tier pays only when its delivered rate exceeds the recompute bar (rate / bar > 1). `-` for the break-even means the tier's slope is not below recompute's, so it never wins at any length.

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, recompute

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.4065 | 0.2331 | 0.57x |
| L=1024 (s) | 0.7842 | 0.4689 | 0.60x |
| L=2048 (s) | 1.5570 | 0.9166 | 0.59x |
| L=4096 (s) | 3.1767 | 1.8628 | 0.59x |
| L=8192 (s) | 6.7331 | 3.9168 | 0.58x |
| L=16384 (s) | 15.6534 | 9.1370 | 0.58x |
| L=32512 (s) | 39.9024 | 23.6919 | 0.59x |

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, L1

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0887 | 0.0549 | 0.62x |
| L=1024 (s) | 0.0938 | 0.0594 | 0.63x |
| L=2048 (s) | 0.1023 | 0.0701 | 0.68x |
| L=4096 (s) | 0.1419 | 0.0816 | 0.57x |
| L=8192 (s) | 0.1723 | 0.1114 | 0.65x |
| L=16384 (s) | 0.2820 | 0.1773 | 0.63x |
| L=32512 (s) | 0.4936 | 0.3144 | 0.64x |

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, L2

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0936 | 0.0547 | 0.58x |
| L=1024 (s) | 0.0988 | 0.0582 | 0.59x |
| L=2048 (s) | 0.1115 | 0.0649 | 0.58x |
| L=4096 (s) | 0.1313 | 0.0793 | 0.60x |
| L=8192 (s) | 0.1991 | 0.1090 | 0.55x |
| L=16384 (s) | 0.3301 | 0.1773 | 0.54x |
| L=32512 (s) | 0.6357 | 0.3128 | 0.49x |

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, L3

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.1304 | 0.1508 | 1.16x |
| L=1024 (s) | 0.3055 | 0.2381 | 0.78x |
| L=2048 (s) | 0.5738 | 0.3113 | 0.54x |
| L=4096 (s) | 1.0094 | 0.4152 | 0.41x |
| L=8192 (s) | 2.0250 | 0.7564 | 0.37x |
| L=16384 (s) | 3.9770 | 1.0123 | 0.25x |
| L=32512 (s) | 7.9311 | 1.9457 | 0.25x |

### Llama-3.3-70B-AWQ: Exp 1: L3 hit TTFT / recompute TTFT (below 1 = L3 wins)

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (x) | 0.32 | 0.65 | 2.02x |
| L=1024 (x) | 0.39 | 0.51 | 1.30x |
| L=2048 (x) | 0.37 | 0.34 | 0.92x |
| L=4096 (x) | 0.32 | 0.22 | 0.70x |
| L=8192 (x) | 0.30 | 0.19 | 0.64x |
| L=16384 (x) | 0.25 | 0.11 | 0.44x |
| L=32512 (x) | 0.20 | 0.08 | 0.41x |

### Qwen3-32B-FP8: setup

| metric | old | new | new / old |
|---|---:|---:|---:|
| device pool (L1) (tokens) | 281,216 | 705,856 | 2.51x |
| host pool (L2) (tokens) | 762,944 | 1,220,736 | 1.60x |
| KV bytes per token (B) | 131,072 | 131,072 | 1.00x |
| attention backend | triton | triton | - |
| KV cache dtype | torch.float8_e5m2 | torch.float8_e5m2 | - |
| weight kernel path | Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8  | Entering DeepGEMM JIT Pre-Compile session. It may take a long time (typically 10-20 mins) if you have not run  | - |

### Qwen3-32B-FP8: Exp 0: tier attribution, one 4096-token prompt, n=1

| metric | old | new | new / old |
|---|---:|---:|---:|
| TTFT 1 cold recompute (s) | 1.938 | 0.526 | 0.27x |
| TTFT 3 L1 hit (s) | 0.097 | 0.049 | 0.51x |
| TTFT 4 L2 hit (s) | 0.113 | 0.053 | 0.47x |
| TTFT 5 L3 hit, page cache dropped (s) | 0.917 | 0.319 | 0.35x |
| TTFT 6 L3 hit, page cache warm (s) | 0.772 | 0.261 | 0.34x |

Attribution (device/host/storage tokens) new run: 1: None/None/None; 3: 4032/0/0; 4: 0/4032/0; 5: 0/0/4032; 6: 0/0/4032. L3 files after backup: old 128, new 128. n=1 and the first long prefill after boot pays kernel warm-up: use Exp 1 for timings.

### Qwen3-32B-FP8: Exp 1: tier cost fits over 7 lengths (512 to 32512), medians of n=3

| metric | old | new | new / old |
|---|---:|---:|---:|
| P = marginal prefill rate (tok/s) | 1,226.302 | 3,255.222 | 2.65x |
| recompute bar b*P (GiB/s) | 0.150 | 0.397 | 2.65x |
| recompute fit intercept (s) | -1.251 | -0.765 | 0.61x |
| L1 delivered rate (fit) (GiB/s) | 9.808 | 15.465 | 1.58x |
| L1 fit intercept (s) | 0.053 | 0.025 | 0.47x |
| L2 delivered rate (fit) (GiB/s) | 7.748 | 15.637 | 2.02x |
| L2 fit intercept (s) | 0.048 | 0.024 | 0.50x |
| L3 delivered rate (fit) (GiB/s) | 0.623 | 2.994 | 4.81x |
| L3 fit intercept (s) | 0.058 | 0.120 | 2.07x |
| host to device, differential (L2 - L1 slopes) (GiB/s) | - | - | - |
| disk read, differential (L3 - L2 slopes) (GiB/s) | 0.677 | 3.703 | 5.47x |
| L2 rate / recompute bar (x) | 51.758 | 39.352 | 0.76x |
| L2 beats recompute above L (fit) (tokens) | 1,624.399 | 2,634.455 | 1.62x |
| L3 rate / recompute bar (x) | 4.162 | 7.534 | 1.81x |
| L3 beats recompute above L (fit) (tokens) | 2,113.751 | 3,322.201 | 1.57x |

A tier pays only when its delivered rate exceeds the recompute bar (rate / bar > 1). `-` for the break-even means the tier's slope is not below recompute's, so it never wins at any length.

### Qwen3-32B-FP8: Exp 1: median TTFT, recompute

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.2393 | 0.0589 | 0.25x |
| L=1024 (s) | 0.4552 | 0.0918 | 0.20x |
| L=2048 (s) | 0.9021 | 0.1757 | 0.19x |
| L=4096 (s) | 1.8597 | 0.3624 | 0.19x |
| L=8192 (s) | 4.0303 | 0.8803 | 0.22x |
| L=16384 (s) | 9.8764 | 2.8421 | 0.29x |
| L=32512 (s) | 26.6922 | 10.1300 | 0.38x |

### Qwen3-32B-FP8: Exp 1: median TTFT, L1

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0638 | 0.0305 | 0.48x |
| L=1024 (s) | 0.0680 | 0.0345 | 0.51x |
| L=2048 (s) | 0.0758 | 0.0395 | 0.52x |
| L=4096 (s) | 0.1046 | 0.0551 | 0.53x |
| L=8192 (s) | 0.1504 | 0.0877 | 0.58x |
| L=16384 (s) | 0.2552 | 0.1559 | 0.61x |
| L=32512 (s) | 0.4597 | 0.2811 | 0.61x |

### Qwen3-32B-FP8: Exp 1: median TTFT, L2

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0680 | 0.0332 | 0.49x |
| L=1024 (s) | 0.0705 | 0.0333 | 0.47x |
| L=2048 (s) | 0.0811 | 0.0393 | 0.48x |
| L=4096 (s) | 0.1067 | 0.0528 | 0.49x |
| L=8192 (s) | 0.1604 | 0.0845 | 0.53x |
| L=16384 (s) | 0.2992 | 0.1478 | 0.49x |
| L=32512 (s) | 0.5673 | 0.2804 | 0.49x |

### Qwen3-32B-FP8: Exp 1: median TTFT, L3

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0904 | 0.0943 | 1.04x |
| L=1024 (s) | 0.2154 | 0.1897 | 0.88x |
| L=2048 (s) | 0.3985 | 0.1905 | 0.48x |
| L=4096 (s) | 1.0021 | 0.3037 | 0.30x |
| L=8192 (s) | 1.6937 | 0.4498 | 0.27x |
| L=16384 (s) | 3.3073 | 0.8278 | 0.25x |
| L=32512 (s) | 6.3898 | 1.4253 | 0.22x |

### Qwen3-32B-FP8: Exp 1: L3 hit TTFT / recompute TTFT (below 1 = L3 wins)

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (x) | 0.38 | 1.60 | 4.23x |
| L=1024 (x) | 0.47 | 2.07 | 4.37x |
| L=2048 (x) | 0.44 | 1.08 | 2.46x |
| L=4096 (x) | 0.54 | 0.84 | 1.56x |
| L=8192 (x) | 0.42 | 0.51 | 1.22x |
| L=16384 (x) | 0.33 | 0.29 | 0.87x |
| L=32512 (x) | 0.24 | 0.14 | 0.59x |

### Llama-3.3-70B-AWQ: recompute with and without HiCache write-through

| metric | old | new | new / old |
|---|---:|---:|---:|
| P from p_measure (no HiCache) (tok/s) | 819.627 | 1,381.445 | 1.69x |
| P from the Exp 1 server (write_through on) (tok/s) | 819.691 | 1,382.444 | 1.69x |
| Exp 1 P / p_measure P (x) | 1.000 | 1.001 | 1.00x |

A ratio below 1 on the new box means write-through to the slower disk is perturbing prefill.
