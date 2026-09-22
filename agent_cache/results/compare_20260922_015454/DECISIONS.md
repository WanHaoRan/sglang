# Campaign 6 — four arms at natural pools (H200, Qwen3-32B bf16 weights)

Started 2026-09-22T01:54:54Z. Same design as `compare_20260918_final/` (C18, A100) with the pools and the
workload re-sized so that L2 and L3 are actually exercised. Config is in `config.txt`; this file is why.

## Arms — capacity identical across all four

| arm | L1 | L2 | L3 | prefetch |
|---|---|---|---|---|
| hbm_lru | natural (468,288 tok, measured) | — | — | — |
| hbm_host | natural | 64 GB = 488,281 tok | — | — |
| three_tier_to | natural | 64 GB | unlimited (`L3_CLEANER_PCT=70,60` = 1.59 TiB) | timeout |
| three_tier_wc | natural | 64 GB | unlimited | wait_complete |

## Inputs measured on this box before sizing (`../probe_bf16/`)

| | value | source |
|---|---:|---|
| L1 at mem-fraction 0.85, no pin | 468,288 tok (57.2 GB) | boot log; predicted 450,880 |
| weights | 61.04 GB, no Marlin line | boot log |
| P (marginal prefill) | **2,920 tok/s** | recompute sweep, 21 rows, 0 tier hits; FP8 weights: 3,255 |
| decode per sequence, 25.6K ctx | 30.8 (B=1) / 29.0 (B=8) / 20.3 (B=16) tok/s | one_batch_server; FP8: 38.4 / 36.4 / 23.3 |
| uncached tokens per returning turn | 6,471 (median) | C18 turns_*.csv |
| final context at TURNS=24 | 26,100 | trace, offset 29 |

## The sizing chain

```
WS  = c * ctx = 48 * 26,100 = 1,252,800  >  L1+L2 = 956,569   -> 296K tokens (24 %) exist only on L3
GPU-s per turn  = 6,471/2,920 + 182 * ITL(B~12)/12  ~ 2.85 s  -> capacity ~0.33 turns/s; take 0.30
unqueued turn   ~ 2.2 s prefill + 182 * 45 ms decode ~ 10.4 s
f_memory  <= L1/(1.2*ctx)/c = 15.0/48 = 0.311
f_compute <= capacity * turn_time / c = 0.30 * 10.4 / 48 = 0.065     <- BINDING
gap >= turn_time * (1-f)/f = 150 s  ->  GAP = 150 / 1.49 = 100
```

At GAP 100 about 3 sessions are active and 45 parked at any instant; the GPU is fully used.
A-priori ceiling on returning turns that can restore from L3 at once: `0.5 * (1 - L1/WS)` = **31 %**.
Prefetch budget `0.5 * L2 / ctx` = 9.4 concurrent restores vs ~0.3 needed: the budget will not bind.

## Correction made before launch

The first sizing (GAP x21) used the memory constraint only. Checked against GPU throughput it was 3.8x
oversubscribed (offered 1.15 turns/s vs 0.30): the closed loop would have inflated turn time by queueing,
f -> ~0.9, nothing idle, nothing parked — C18 exactly. Compute, not memory, is the binding constraint on
this workload, and the gap scale is where that shows up.

## Other choices

- `OFFSET=29`: the 48-wide window with the most >=24-turn conversations (42/48), so c holds for the run.
- `GAP_CAP=1200`: at x100 the trace's p99 gap (10.25 s) becomes 1,025 s; the cap bites on ~0.8 % of turns.
- `CTX=40960`: the model's native max; no YaRN. Trace is the committed 32K-truncated one (ctx at turn 24 is
  below the cut, so no re-conversion).
- `K=0` (no paired controls), as C18: hbm_lru on the identical workload is the recompute reference.

## What each arm should show

- hbm_lru: sessions evicted from L1 recompute ~26K on return; GPU-bound and likely queue-inflated. The cost of no tiering.
- hbm_host: parked sessions restore from L2 (~0.2 s); L3 absent, so the 24 % that would not fit recompute.
- three_tier_*: the 24 % restore from L3 (~1 s at 2.9 GiB/s). `timeout` vs `wait_complete` is C18 cause 3.
- If the three arms tie again, the event log (`HICACHE_EVT`, RUNBOOK 6.6) says which of the four C18 causes survived.
