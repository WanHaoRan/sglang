# Old vs new: HiCache evaluation, 70B and 32B with fp8_e5m2 KV, nixl backend

- **old**: H100 PCIe, virtio disk (old); `20260908_llama70b_awq_fp8kv`
- **new**: A100 SXM4, local NVMe (new); `20260917_a100_gcp_32b70b_fp8kv`
- same image, same sglang commit under `python/`, same server flags and client parameters; differences forced by the box are listed in `DEVIATIONS.md`

### Llama-3.3-70B-AWQ: setup

| metric | old | new | new / old |
|---|---:|---:|---:|
| device pool (L1) (tokens) | 193,728 | 194,816 | 1.01x |
| host pool (L2) (tokens) | 610,368 | 610,368 | 1.00x |
| KV bytes per token (B) | 163,840 | 163,840 | 1.00x |
| attention backend | triton | triton | - |
| KV cache dtype | torch.float8_e5m2 | torch.float8_e5m2 | - |
| weight kernel path | The model is convertible to awq_marlin during runtime. Using awq_marlin kernel. | The model is convertible to awq_marlin during runtime. Using awq_marlin kernel. | - |

### Llama-3.3-70B-AWQ: Exp 0: tier attribution, one 4096-token prompt, n=1

| metric | old | new | new / old |
|---|---:|---:|---:|
| TTFT 1 cold recompute (s) | 3.031 | 3.288 | 1.08x |
| TTFT 3 L1 hit (s) | 0.116 | 0.132 | 1.14x |
| TTFT 4 L2 hit (s) | 0.123 | 0.150 | 1.21x |
| TTFT 5 L3 hit, page cache dropped (s) | 0.413 | 1.006 | 2.44x |
| TTFT 6 L3 hit, page cache warm (s) | 0.296 | 1.037 | 3.50x |

Attribution (device/host/storage tokens) new run: 1: None/None/None; 3: 4032/0/0; 4: 0/4032/0; 5: 0/0/4032; 6: 0/0/4032. L3 files after backup: old 128, new 128. n=1 and the first long prefill after boot pays kernel warm-up: use Exp 1 for timings.

### Llama-3.3-70B-AWQ: Exp 1: tier cost fits over 7 lengths (512 to 32512), medians of n=3

| metric | old | new | new / old |
|---|---:|---:|---:|
| P = marginal prefill rate (tok/s) | 920.168 | 819.691 | 0.89x |
| recompute bar b*P (GiB/s) | 0.140 | 0.125 | 0.89x |
| recompute fit intercept (s) | -1.308 | -1.543 | 1.18x |
| L1 delivered rate (fit) (GiB/s) | 13.437 | 12.090 | 0.90x |
| L1 fit intercept (s) | 0.070 | 0.080 | 1.13x |
| L2 delivered rate (fit) (GiB/s) | 13.094 | 9.006 | 0.69x |
| L2 fit intercept (s) | 0.074 | 0.072 | 0.97x |
| L3 delivered rate (fit) (GiB/s) | 2.952 | 0.629 | 0.21x |
| L3 fit intercept (s) | 0.118 | 0.036 | 0.31x |
| host to device, differential (L2 - L1 slopes) (GiB/s) | - | - | - |
| disk read, differential (L3 - L2 slopes) (GiB/s) | 3.811 | 0.677 | 0.18x |
| L2 rate / recompute bar (x) | 93.258 | 72.007 | 0.77x |
| L2 beats recompute above L (fit) (tokens) | 1,285.121 | 1,342.430 | 1.04x |
| L3 rate / recompute bar (x) | 21.023 | 5.033 | 0.24x |
| L3 beats recompute above L (fit) (tokens) | 1,377.435 | 1,615.465 | 1.17x |

A tier pays only when its delivered rate exceeds the recompute bar (rate / bar > 1). `-` for the break-even means the tier's slope is not below recompute's, so it never wins at any length.

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, recompute

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.3804 | 0.4065 | 1.07x |
| L=1024 (s) | 0.7378 | 0.7842 | 1.06x |
| L=2048 (s) | 1.4542 | 1.5570 | 1.07x |
| L=4096 (s) | 2.9234 | 3.1767 | 1.09x |
| L=8192 (s) | 6.1015 | 6.7331 | 1.10x |
| L=16384 (s) | 14.0575 | 15.6534 | 1.11x |
| L=32512 (s) | 35.5776 | 39.9024 | 1.12x |

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, L1

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0740 | 0.0887 | 1.20x |
| L=1024 (s) | 0.0792 | 0.0938 | 1.18x |
| L=2048 (s) | 0.0903 | 0.1023 | 1.13x |
| L=4096 (s) | 0.1173 | 0.1419 | 1.21x |
| L=8192 (s) | 0.1677 | 0.1723 | 1.03x |
| L=16384 (s) | 0.2648 | 0.2820 | 1.06x |
| L=32512 (s) | 0.4344 | 0.4936 | 1.14x |

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, L2

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0773 | 0.0936 | 1.21x |
| L=1024 (s) | 0.0784 | 0.0988 | 1.26x |
| L=2048 (s) | 0.0914 | 0.1115 | 1.22x |
| L=4096 (s) | 0.1249 | 0.1313 | 1.05x |
| L=8192 (s) | 0.1791 | 0.1991 | 1.11x |
| L=16384 (s) | 0.2750 | 0.3301 | 1.20x |
| L=32512 (s) | 0.4453 | 0.6357 | 1.43x |

### Llama-3.3-70B-AWQ: Exp 1: median TTFT, L3

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.2588 | 0.1304 | 0.50x |
| L=1024 (s) | 0.2514 | 0.3055 | 1.22x |
| L=2048 (s) | 0.2471 | 0.5738 | 2.32x |
| L=4096 (s) | 0.3043 | 1.0094 | 3.32x |
| L=8192 (s) | 0.4435 | 2.0250 | 4.57x |
| L=16384 (s) | 0.7288 | 3.9770 | 5.46x |
| L=32512 (s) | 1.9396 | 7.9311 | 4.09x |

### Llama-3.3-70B-AWQ: Exp 1: L3 hit TTFT / recompute TTFT (below 1 = L3 wins)

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (x) | 0.68 | 0.32 | 0.47x |
| L=1024 (x) | 0.34 | 0.39 | 1.14x |
| L=2048 (x) | 0.17 | 0.37 | 2.17x |
| L=4096 (x) | 0.10 | 0.32 | 3.05x |
| L=8192 (x) | 0.07 | 0.30 | 4.14x |
| L=16384 (x) | 0.05 | 0.25 | 4.90x |
| L=32512 (x) | 0.05 | 0.20 | 3.65x |

### Qwen3-32B-FP8: setup

| metric | old | new | new / old |
|---|---:|---:|---:|
| device pool (L1) (tokens) | 284,224 | 281,216 | 0.99x |
| host pool (L2) (tokens) | 762,944 | 762,944 | 1.00x |
| KV bytes per token (B) | 131,072 | 131,072 | 1.00x |
| attention backend | triton | triton | - |
| KV cache dtype | torch.float8_e5m2 | torch.float8_e5m2 | - |
| weight kernel path | Entering DeepGEMM JIT Pre-Compile session. It may take a long time (typically 10-20 mins) if you have not run  | Your GPU does not have native support for FP8 computation but FP8 quantization is being used. Weight-only FP8  | - |

### Qwen3-32B-FP8: Exp 0: tier attribution, one 4096-token prompt, n=1

| metric | old | new | new / old |
|---|---:|---:|---:|
| TTFT 1 cold recompute (s) | 0.762 | 1.938 | 2.55x |
| TTFT 3 L1 hit (s) | 0.092 | 0.097 | 1.06x |
| TTFT 4 L2 hit (s) | 0.094 | 0.113 | 1.20x |
| TTFT 5 L3 hit, page cache dropped (s) | 0.250 | 0.917 | 3.66x |
| TTFT 6 L3 hit, page cache warm (s) | 0.234 | 0.772 | 3.30x |

Attribution (device/host/storage tokens) new run: 1: None/None/None; 3: 4032/0/0; 4: 0/4032/0; 5: 0/0/4032; 6: 0/0/4032. L3 files after backup: old 128, new 128. n=1 and the first long prefill after boot pays kernel warm-up: use Exp 1 for timings.

### Qwen3-32B-FP8: Exp 1: tier cost fits over 7 lengths (512 to 32512), medians of n=3

| metric | old | new | new / old |
|---|---:|---:|---:|
| P = marginal prefill rate (tok/s) | 2,335.694 | 1,226.302 | 0.53x |
| recompute bar b*P (GiB/s) | 0.285 | 0.150 | 0.53x |
| recompute fit intercept (s) | -0.984 | -1.251 | 1.27x |
| L1 delivered rate (fit) (GiB/s) | 10.363 | 9.808 | 0.95x |
| L1 fit intercept (s) | 0.050 | 0.053 | 1.05x |
| L2 delivered rate (fit) (GiB/s) | 10.405 | 7.748 | 0.74x |
| L2 fit intercept (s) | 0.045 | 0.048 | 1.06x |
| L3 delivered rate (fit) (GiB/s) | 3.176 | 0.623 | 0.20x |
| L3 fit intercept (s) | 0.139 | 0.058 | 0.42x |
| host to device, differential (L2 - L1 slopes) (GiB/s) | - | - | - |
| disk read, differential (L3 - L2 slopes) (GiB/s) | 4.572 | 0.677 | 0.15x |
| L2 rate / recompute bar (x) | 36.495 | 51.758 | 1.42x |
| L2 beats recompute above L (fit) (tokens) | 2,469.265 | 1,624.399 | 0.66x |
| L3 rate / recompute bar (x) | 11.140 | 4.162 | 0.37x |
| L3 beats recompute above L (fit) (tokens) | 2,880.743 | 2,113.751 | 0.73x |

A tier pays only when its delivered rate exceeds the recompute bar (rate / bar > 1). `-` for the break-even means the tier's slope is not below recompute's, so it never wins at any length.

### Qwen3-32B-FP8: Exp 1: median TTFT, recompute

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.1186 | 0.2393 | 2.02x |
| L=1024 (s) | 0.1439 | 0.4552 | 3.16x |
| L=2048 (s) | 0.2738 | 0.9021 | 3.30x |
| L=4096 (s) | 0.5968 | 1.8597 | 3.12x |
| L=8192 (s) | 1.4037 | 4.0303 | 2.87x |
| L=16384 (s) | 4.2098 | 9.8764 | 2.35x |
| L=32512 (s) | 14.0984 | 26.6922 | 1.89x |

### Qwen3-32B-FP8: Exp 1: median TTFT, L1

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0473 | 0.0638 | 1.35x |
| L=1024 (s) | 0.0544 | 0.0680 | 1.25x |
| L=2048 (s) | 0.0647 | 0.0758 | 1.17x |
| L=4096 (s) | 0.1055 | 0.1046 | 0.99x |
| L=8192 (s) | 0.1660 | 0.1504 | 0.91x |
| L=16384 (s) | 0.2545 | 0.2552 | 1.00x |
| L=32512 (s) | 0.4230 | 0.4597 | 1.09x |

### Qwen3-32B-FP8: Exp 1: median TTFT, L2

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0487 | 0.0680 | 1.39x |
| L=1024 (s) | 0.0505 | 0.0705 | 1.40x |
| L=2048 (s) | 0.0584 | 0.0811 | 1.39x |
| L=4096 (s) | 0.0922 | 0.1067 | 1.16x |
| L=8192 (s) | 0.1564 | 0.1604 | 1.03x |
| L=16384 (s) | 0.2502 | 0.2992 | 1.20x |
| L=32512 (s) | 0.4165 | 0.5673 | 1.36x |

### Qwen3-32B-FP8: Exp 1: median TTFT, L3

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.2135 | 0.0904 | 0.42x |
| L=1024 (s) | 0.1843 | 0.2154 | 1.17x |
| L=2048 (s) | 0.1779 | 0.3985 | 2.24x |
| L=4096 (s) | 0.3666 | 1.0021 | 2.73x |
| L=8192 (s) | 0.4324 | 1.6937 | 3.92x |
| L=16384 (s) | 0.6336 | 3.3073 | 5.22x |
| L=32512 (s) | 1.4549 | 6.3898 | 4.39x |

### Qwen3-32B-FP8: Exp 1: L3 hit TTFT / recompute TTFT (below 1 = L3 wins)

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (x) | 1.80 | 0.38 | 0.21x |
| L=1024 (x) | 1.28 | 0.47 | 0.37x |
| L=2048 (x) | 0.65 | 0.44 | 0.68x |
| L=4096 (x) | 0.61 | 0.54 | 0.88x |
| L=8192 (x) | 0.31 | 0.42 | 1.36x |
| L=16384 (x) | 0.15 | 0.33 | 2.22x |
| L=32512 (x) | 0.10 | 0.24 | 2.32x |

### Llama-3.3-70B-AWQ: recompute with and without HiCache write-through

| metric | old | new | new / old |
|---|---:|---:|---:|
| P from p_measure (no HiCache) (tok/s) | 919.189 | 819.627 | 0.89x |
| P from the Exp 1 server (write_through on) (tok/s) | 920.168 | 819.691 | 0.89x |
| Exp 1 P / p_measure P (x) | 1.001 | 1.000 | 1.00x |

A ratio below 1 on the new box means write-through to the slower disk is perturbing prefill.
