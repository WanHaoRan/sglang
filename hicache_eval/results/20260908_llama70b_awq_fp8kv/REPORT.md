# HiCache L3 across models — where disk KV offload actually pays

**Hardware** 1× H100 PCIe 80 GB · 221 GB host RAM · L3 on the local virtio-blk disk
**Backend** `nixl` (O_DIRECT, io_uring batches) · `wait_complete` · `write_through` · `page_first`
**sglang** `f3ccd1c0e4` · run `20260908_llama70b_awq_fp8kv`

Follow-up to `results/20260907_203035/`, which established that on **Qwen3-8B**
an L3 hit only beats recompute above ~32 K tokens. The question here: is that a
property of HiCache, or of the model? Answer: **the model.**

---

## 1. Headline

Three models, **identical storage, backend and disk**. Only the model changes.

| model | KV B/token | measured P | **recompute bar b·P** | L3 delivered | **L3 beats recompute from** | max speed-up |
|---|---|---|---|---|---|---|
| Qwen3-8B, bf16 KV | 147,456 | 15,011 tok/s | **2.061 GiB/s** | 3.28 GiB/s | 32,512 tokens | 1.5× |
| Qwen3-32B FP8, fp8 KV | 131,072 | 2,336 tok/s | **0.285 GiB/s** | 3.18 GiB/s | **2,048 tokens** | 9.7× |
| Llama-3.3-70B AWQ-INT4, fp8 KV | 163,840 | 920 tok/s | **0.140 GiB/s** | 2.95 GiB/s | **512 tokens** | **18.3×** |

**The tier delivers the same ~3 GiB/s in all three runs.** What moves is the bar
it has to clear — 2.061 → 0.285 → 0.140 GiB/s, a **14.7× drop** — and the
crossover moves with it, 32,512 → 2,048 → 512 tokens.

This is the `bandwidth(L3) > b·P` criterion behaving exactly as predicted, and it
says the earlier "L3 is not worth it" result was a statement about **Qwen3-8B on
an H100**, not about HiCache.

---

## 2. Why big models invert the result

b·P is the rate at which prefill *manufactures* KV bytes. Storage has to beat
that rate to be worth reading instead of recomputing.

- **Qwen3-8B** prefills at 15,011 tok/s. Recompute is cheap, so the bar is high.
- **Llama-70B** prefills at 920 tok/s — 16× slower — while its KV is only 1.1×
  larger per token. The bar collapses.

Doubling model size raises b roughly linearly with depth but *lowers* P roughly
linearly with parameters, so **b·P falls as models grow**. Large models are the
natural home for disk KV offload; small ones are not.

fp8 KV contributes a second, smaller factor: it halves b. Confirmed end-to-end,
not assumed — `pool_host/base.py:149` sets the host pool dtype from
`device_pool.store_dtype`, and Exp 0's backup wrote exactly **163,840 B/token**
(70B) and **131,072 B/token** (32B) to disk, i.e. the fp8 sizes.

---

## 3. Experiment 0 — tier attribution, both models

| tier | Llama-70B | Qwen3-32B |
|---|---|---|
| recompute (cold) | 3.031 s | 0.762 s |
| L1 device | 0.116 s | 0.092 s |
| L2 host | 0.123 s | 0.094 s |
| **L3, page cache evicted** | **0.413 s** | **0.250 s** |
| L3, page cache warm | 0.296 s | 0.234 s |
| **L3 vs recompute** | **7.3× faster** | **3.0× faster** |

Every attribution correct (`4032/0/0`, `0/4032/0`, `0/0/4032`), and for the first
time in this evaluation the hierarchy is strictly ordered at 4096 tokens:
**L1 < L2 < L3 < recompute**.

Pool sizes: L1 = 193,728 tokens (70B) and 284,224 (32B); L2 = 610,368 (70B).

---

## 4. Experiment 1 — cost curves

**Llama-3.3-70B AWQ-INT4 + fp8 KV** — median TTFT (s), N = 3, 0 discards:

| L | L1 | L2 | **L3** | recompute | **L3 speed-up** |
|---|---|---|---|---|---|
| 512 | 0.074 | 0.077 | 0.259 | 0.380 | 1.5× |
| 1024 | 0.079 | 0.078 | 0.251 | 0.738 | 2.9× |
| 2048 | 0.090 | 0.091 | 0.247 | 1.454 | 5.9× |
| 4096 | 0.117 | 0.125 | 0.304 | 2.923 | 9.6× |
| 8192 | 0.168 | 0.179 | 0.444 | 6.102 | 13.8× |
| 16384 | 0.265 | 0.275 | 0.729 | 14.058 | **19.3×** |
| 32512 | 0.434 | 0.445 | 1.940 | 35.578 | 18.3× |

**Qwen3-32B FP8 + fp8 KV:**

| L | L1 | L2 | **L3** | recompute | **L3 speed-up** |
|---|---|---|---|---|---|
| 512 | 0.047 | 0.049 | 0.214 | 0.119 | 0.6× |
| 1024 | 0.054 | 0.051 | 0.184 | 0.144 | 0.8× |
| 2048 | 0.065 | 0.058 | 0.178 | 0.274 | 1.5× |
| 4096 | 0.106 | 0.092 | 0.367 | 0.597 | 1.6× |
| 8192 | 0.166 | 0.156 | 0.432 | 1.404 | 3.2× |
| 16384 | 0.255 | 0.250 | 0.634 | 4.210 | 6.6× |
| 32512 | 0.423 | 0.417 | 1.455 | 14.098 | **9.7×** |

Figure: `model_comparison.png` (all three models, L3 vs recompute, log-log).

L1 and L2 remain within ~5 % of each other in both models — host DRAM over PCIe
is close enough to HBM that the L1/L2 distinction barely matters for TTFT. The
whole decision is L2-or-better versus L3 versus recompute.

---

## 5. Measured P, and why estimating it was not enough

P was measured before committing to the plan (`p_measure/`), and the first-order
estimate was wrong enough to matter: predicted 1,743 tok/s for the 70B, measured
**1,416 at short lengths falling to 913 at 32.5 K** — 1.2–1.9× optimistic.

| L | TTFT (s) | P = L/TTFT |
|---|---|---|
| 512 | 0.362 | 1,416 |
| 4,096 | 2.919 | 1,403 |
| 16,384 | 14.075 | 1,164 |
| 32,512 | 35.597 | 913 |

P is not constant: the quadratic fit is `1.465e-8·L² + 6.156e-4·L + 0.094`,
where the L² term is attention. So the recompute bar **falls further as context
grows**, which is why L3's advantage widens with length rather than flattening.

---

## 6. Caveats

- **N = 3**, medians reported. The tier separations here are 3–19×, far outside
  the spread, but this is not a tail-latency study.
- **`P` for the 70B is AWQ-INT4.** AWQ dequantizes to fp16 for the GEMM, so
  prefill is no faster than fp16 and possibly slower. A bf16 70B would have a
  similar bar; the quantization buys memory, not speed.
- **The AWQ checkpoint is community-provided** (`casperhansen/llama-3.3-70b-instruct-awq`);
  Meta's own repo is gated. Fine for a systems benchmark measuring bytes and
  latency — do not quote accuracy from it.
- **fp8 KV accuracy was not evaluated.** `fp8_e5m2` was chosen for capacity; its
  effect on output quality is out of scope here.
- **These runs do not repeat Exp 2–4** (write interference, amplification,
  oracle). The dead-write finding from the earlier run is a workload property and
  should carry over, but the byte volumes would differ.
- **Reaching L3 still requires a working set that overflows L1+L2** — ~804 K
  tokens for the 70B. Low b·P makes an L3 hit *worth taking*; it does not make
  one *happen*.

---

## 7. Practical read

If you are deciding whether to enable disk KV offload:

1. **Measure b·P for your model on your GPU** — b from the config, P from a
   recompute sweep. It takes ten minutes and it is the whole decision.
2. **Compare it to your storage's delivered bandwidth**, not its spec sheet. Ours
   delivers ~3 GiB/s end-to-end through `nixl` against a device capable of 22 GiB/s
   concurrent — the gap is software (see the previous run's §2).
3. **Above ~30B parameters, disk offload is likely to pay**; below ~10B on a fast
   GPU it is likely not. The 32B sits on the boundary and wins only above ~2 K
   tokens.
4. **Use `nixl`, not `file`.** On the `file` backend none of this holds — it
   delivers 0.058 GiB/s and loses to recompute for every model tested.

## Artifacts

```
p_measure/          recompute-only sweep + P_fit.json (the 10-minute pre-check)
exp0_70b/           Exp 0, Llama-3.3-70B AWQ + fp8 KV
exp1_70b/           Exp 1, 4 tiers x 7 lengths x 3 reps
exp0_32b/           Exp 0, Qwen3-32B FP8 + fp8 KV
exp1_32b/           Exp 1, same grid
model_comparison.png / .json    all three models, fits and crossovers
```
