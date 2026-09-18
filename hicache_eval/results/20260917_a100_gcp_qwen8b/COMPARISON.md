# Old vs new: HiCache evaluation, Qwen3-8B, nixl backend

- **old**: H100 PCIe, virtio disk (old); Exp 0/1 + A/B from `20260907_203035`, Exp 2/3/4 from `20260908_nixl_exp234`
- **new**: A100 SXM4, local NVMe (new); `20260917_a100_gcp_qwen8b`
- same image, same sglang commit under `python/`, same server flags and client parameters; differences forced by the box are listed in `DEVIATIONS.md`

### Setup

| metric | old | new | new / old |
|---|---:|---:|---:|
| device pool (L1) (tokens) | 374,784 | 376,000 | 1.00x |
| host pool (L2) (tokens) | 678,208 | 678,208 | 1.00x |
| attention backend | fa3 | flashinfer | - |

### Exp 0: tier attribution, one 4096-token prompt, n=1

| metric | old | new | new / old |
|---|---:|---:|---:|
| TTFT 1 cold recompute (s) | 0.391 | 0.426 | 1.09x |
| TTFT 3 L1 hit (s) | 0.059 | 0.060 | 1.02x |
| TTFT 4 L2 hit (s) | 0.067 | 0.083 | 1.24x |
| TTFT 5 L3 hit, page cache dropped (s) | 0.271 | 1.017 | 3.76x |
| TTFT 6 L3 hit, page cache warm (s) | 0.259 | 0.857 | 3.31x |

Attribution (device/host/storage tokens) new run: 1: None/None/None; 3: 4032/0/0; 4: 0/4032/0; 5: 0/0/4032; 6: 0/0/4032. L3 files after backup: old 0 (old driver looked at the wrong dir), new 128. n=1 and the first long prefill after boot pays kernel warm-up: use Exp 1 for timings.

### Backend A/B, nixl backend, 4096-token probe

| metric | old | new | new / old |
|---|---:|---:|---:|
| L3 hit TTFT (s) | 0.195 | 0.845 | 4.33x |
| L3 read rate (hit bytes / TTFT) (GiB/s) | 2.840 | 0.655 | 0.23x |
| L3 write rate (backup bytes / drain) (GiB/s) | 0.097 | 0.099 | 1.03x |
| backup drain (s) | 5.830 | 5.670 | 0.97x |
| cold recompute TTFT (s) | 0.488 | 0.402 | 0.82x |

### Backend A/B, file backend, 4096-token probe

| metric | old | new | new / old |
|---|---:|---:|---:|
| L3 hit TTFT (s) | 6.693 | 1.610 | 0.24x |
| L3 read rate (hit bytes / TTFT) (GiB/s) | 0.083 | 0.344 | 4.16x |
| L3 write rate (backup bytes / drain) (GiB/s) | 0.052 | 0.099 | 1.91x |
| backup drain (s) | 10.870 | 5.680 | 0.52x |
| cold recompute TTFT (s) | 0.326 | 1.221 | 3.75x |

### Backend A/B, nixl over file

| metric | old | new | new / old |
|---|---:|---:|---:|
| L3 read rate ratio nixl/file (x) | 34.3 | 1.9 | 0.06x |

### Exp 1: tier cost fits over 7 lengths (512 to 32512), medians of n=3

| metric | old | new | new / old |
|---|---:|---:|---:|
| P = marginal prefill rate (tok/s) | 15,010.820 | 8,857.792 | 0.59x |
| recompute bar b*P (GiB/s) | 2.061 | 1.216 | 0.59x |
| recompute fit intercept (s) | -0.033 | -0.125 | 3.79x |
| L1 delivered rate (fit) (GiB/s) | 19.824 | 15.459 | 0.78x |
| L1 fit intercept (s) | 0.036 | 0.036 | 1.01x |
| L2 delivered rate (fit) (GiB/s) | 12.874 | 7.384 | 0.57x |
| L2 fit intercept (s) | 0.031 | 0.017 | 0.55x |
| L3 delivered rate (fit) (GiB/s) | 3.277 | 0.625 | 0.19x |
| L3 fit intercept (s) | 0.192 | -0.006 | -0.03x |
| host to device, differential (L2 - L1 slopes) (GiB/s) | 36.725 | 14.135 | 0.38x |
| disk read, differential (L3 - L2 slopes) (GiB/s) | 4.396 | 0.682 | 0.16x |
| L2 rate / recompute bar (x) | 6.245 | 6.070 | 0.97x |
| L2 beats recompute above L (fit) (tokens) | 1,140.101 | 1,504.966 | 1.32x |
| L3 rate / recompute bar (x) | 1.590 | 0.513 | 0.32x |
| L3 beats recompute above L (fit) (tokens) | 9,085.578 | - | - |

A tier pays only when its delivered rate exceeds the recompute bar (rate / bar > 1). `-` for the break-even means the tier's slope is not below recompute's, so it never wins at any length.

### Exp 1: median TTFT, recompute

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0431 | 0.0642 | 1.49x |
| L=1024 (s) | 0.0897 | 0.0910 | 1.01x |
| L=2048 (s) | 0.1366 | 0.1611 | 1.18x |
| L=4096 (s) | 0.2481 | 0.3069 | 1.24x |
| L=8192 (s) | 0.4279 | 0.6370 | 1.49x |
| L=16384 (s) | 0.9199 | 1.4640 | 1.59x |
| L=32512 (s) | 2.2187 | 3.7131 | 1.67x |

### Exp 1: median TTFT, L1

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0295 | 0.0384 | 1.30x |
| L=1024 (s) | 0.0345 | 0.0418 | 1.21x |
| L=2048 (s) | 0.0353 | 0.0516 | 1.46x |
| L=4096 (s) | 0.0795 | 0.0749 | 0.94x |
| L=8192 (s) | 0.1126 | 0.1173 | 1.04x |
| L=16384 (s) | 0.1590 | 0.1843 | 1.16x |
| L=32512 (s) | 0.2511 | 0.3222 | 1.28x |

### Exp 1: median TTFT, L2

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.0341 | 0.0404 | 1.18x |
| L=1024 (s) | 0.0378 | 0.0424 | 1.12x |
| L=2048 (s) | 0.0462 | 0.0477 | 1.03x |
| L=4096 (s) | 0.0761 | 0.0870 | 1.14x |
| L=8192 (s) | 0.1277 | 0.1626 | 1.27x |
| L=16384 (s) | 0.2129 | 0.3163 | 1.49x |
| L=32512 (s) | 0.3720 | 0.6268 | 1.69x |

### Exp 1: median TTFT, L3

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (s) | 0.1992 | 0.1132 | 0.57x |
| L=1024 (s) | 0.2041 | 0.2424 | 1.19x |
| L=2048 (s) | 0.1587 | 0.4396 | 2.77x |
| L=4096 (s) | 0.4421 | 0.8785 | 1.99x |
| L=8192 (s) | 0.5205 | 1.7919 | 3.44x |
| L=16384 (s) | 1.0722 | 3.5793 | 3.34x |
| L=32512 (s) | 1.4586 | 7.1528 | 4.90x |

### Exp 1: L3 hit TTFT / recompute TTFT (below 1 = L3 wins)

| metric | old | new | new / old |
|---|---:|---:|---:|
| L=512 (x) | 4.62 | 1.76 | 0.38x |
| L=1024 (x) | 2.28 | 2.67 | 1.17x |
| L=2048 (x) | 1.16 | 2.73 | 2.35x |
| L=4096 (x) | 1.78 | 2.86 | 1.61x |
| L=8192 (x) | 1.22 | 2.81 | 2.31x |
| L=16384 (x) | 1.17 | 2.44 | 2.10x |
| L=32512 (x) | 0.66 | 1.93 | 2.93x |
