# Baselines: other serving systems under our trace replay

Each folder here runs one system from its official code, on the same H200, model, GPU KV pool, host memory and SSD as
our SGLang arms, and is driven by the same trace replay (traces, Poisson arrivals, user cap, recorded waits, token ids).
Started 2026-11-21. `results/` paths are under `agent_cache/results/`.

| folder | system | official code | engine | tiers |
|---|---|---|---|---|
| `sglang_vanilla` | SGLang, radix cache only | our frozen engine `frozen/engine_ec_20261111d` | SGLang | GPU |
| `sglang_hicache` | SGLang + HiCache (stock policy) | the same; arm `three_tier_wc` of campaigns 28 and 29 | SGLang | GPU, host, SSD |
| `vllm_vanilla` | vLLM, prefix cache only | `vllm==0.33.0` from PyPI | vLLM 0.33.0 | GPU |
| `vllm_lmcache` | vLLM + LMCache, multiprocess mode | `vllm==0.33.0`, `lmcache==0.5.7.post1` from PyPI | vLLM 0.33.0 | GPU, CPU, SSD |
| `continuum` | Continuum (released rule: a 2 s pin for short tools) + LMCache | github.com/Hanchenli/vllm-continuum `316a587`, `lmcache==0.3.6` | its vLLM 0.10.2 fork | GPU, CPU, SSD |
| `tokencake` | TokenCake (scheduling + tool-window preservation) | github.com/pku-lemonade/TokenCake `1901ebf` | its vLLM 0.22.0 fork | GPU, CPU |

## 0. The short version

1. Build a system once: `bash <folder>/setup.sh` (host; vLLM-based systems only, minutes; the SGLang ones use the
   `sglang_hicache` container as campaigns 28 and 29 did).
2. Check it: boot with small pools, run `common/check_server.py` (cache correctness per tier) and, for Continuum and
   TokenCake, `common/check_session.py` (does its own mechanism act). Commands and results: each folder's README.
3. Run a comparison: a folder under `results/` with `settings.env` and `queue.txt` (formats in
   `common/run_baselines.sh`), then
   `setsid nohup bash common/run_baselines.sh <folder> >> results/<folder>/runner.log 2>&1 < /dev/null &` (host).
4. Compare: `python3 -I -B frozen/oracle_scripts_staging/c28_users/compare_arms.py results/<folder>` (host), as for
   campaigns 28 and 29; requests pair across systems by (play, stream, k).

## What is the same for every system

- **The workload.** `common/replay_baseline.py` imports `scripts/replay_programs.py` unchanged and replaces only the
  wire format (`common/backends.py`). A test sends the same trace through both and checks that every request carries
  the same prompt and output length under the same id (`common/test_replay_baseline.py`). On the GPU, a 2-minute run
  against vLLM paired 57 of 57 FreeInference requests and 75 of 75 TraceLab requests with campaigns 28 and 29 (same
  session, prompt size, output length and wait).
- **The GPU KV pool**: 668,096 tokens x 98,304 bytes = 65.7 GB (`--kv-cache-memory-bytes` on vLLM; SGLang profiles
  this pool itself). **Batching**: 8,192 prefill tokens a step, 64 running requests. **Context**: 262,144.
- **Host memory**: 160 GB decimal as SGLang's `--hicache-size` (1,627,648 tokens), passed as 149.01 GiB to LMCache and
  vLLM, which size in GiB. **SSD**: `/mnt/ssd`, O_DIRECT where the system offers it, capped at 1,800 GB with LRU
  eviction (LMCache); every arm starts on wiped stores.
- **Sampling**: temperature 0, forced output length (`ignore_eos`), `--generation-config vllm` so the model's
  generation config cannot change it.
- **No telemetry**: `DO_NOT_TRACK=1` (vLLM and LMCache usage reports off), set in `common/lib.sh`.

## What each system is given

- vLLM and vLLM + LMCache: nothing beyond the prompt (they take no client hints).
- Continuum: `job_id` (one per stream), `is_last_step`, and `this_func_call`, the recorded reply's tool in the name
  its own parser would give (`Bash/git` -> `git`). The release reads the tool from the reply text, which a replay's
  synthetic replies never contain, so one patch (`continuum/patches/0001-...`) keeps the request's own field when
  the parser finds none. Label it "Continuum (released, 2 s pin)".
- TokenCake: request ids `tc-<uuid4>` as lifecycle ids, metadata marking every turn the session extends as
  `offload_eligible` with a `reusable_prefix` (as its datasets mark tool nodes), and `stall_started` /
  `stall_finished` around every wait, with our running per-tool median wait as the estimate.

## Version choices

- vLLM 0.33.0, not 0.34.0 (the newest), for both vLLM baselines: with vLLM 0.34.0 and LMCache 0.5.7.post1 neither
  multiprocess connector loads (LMCache's needs `CachedRequestData.resumed_req_ids`, which 0.34.0 removed; vLLM's
  builtin passes `server_url`, which LMCache 0.5.7's adapter rejects). LMCache 0.5.7.post1 shipped when 0.33.0 was
  the newest vLLM, and the two resolve together on torch 2.14.0 with no conflict.
- Continuum and TokenCake run their forks' own vLLM versions, installed editable on the kernels of the official vLLM
  wheel of that version (both forks change Python files only; checked by diff against the upstream tags), resolved
  as PyPI stood at their time (`--exclude-newer`), since today's resolution brings transformers 5.x into vLLM 0.10.2.

## First results: 20 minutes of campaign 28's setting (2026-11-21)

FreeInference, 0.1 users/s, cap 128, 160 GB host (TokenCake 129 GB, below), x1 gaps; requests sent before 1,200 s;
requests pair across systems by (play, stream, k). `results/baseline_smoke_compare/compare_arms.out`; campaign 28's
`hicache` and `jit_p50` arms are counted to the same 1,200 s.

| system | requests | a minute | TTFT mean | p50 | p90 | prompt tokens recomputed |
|---|---|---|---|---|---|---|
| SGLang, radix cache only | 907 | 45.4 | 24.91 s | 1.61 | 88.60 | 50 % |
| SGLang + HiCache (campaign 28 `hicache`) | 1,676 | 83.8 | 9.19 s | 3.02 | 23.81 | 10 % |
| SGLang + HiCache again, through `run_baselines.sh` | 1,670 | 83.5 | 9.22 s | 2.93 | 23.70 | 10 % |
| our JIT p50 (campaign 28 `jit_p50`) | 1,675 | 83.8 | 9.20 s | 2.93 | 24.60 | 10 % |
| vLLM 0.33.0, prefix cache only | 1,855 | 92.8 | 5.06 s | 0.13 | 22.31 | 34 % |
| Continuum's own base: the same build, FCFS | 2,185 | 109.2 | 1.78 s | 0.17 | 6.57 | 8 % |
| Continuum (vLLM 0.10.2 fork) + LMCache 0.3.6 | 2,263 | 113.2 | 1.66 s | 0.12 | 2.64 | 7 % |
| TokenCake's own base: the same build, native | 1,537 | 76.8 | 9.11 s | 0.26 | 34.13 | 42 % |
| TokenCake (vLLM 0.22.0 fork), full system | 2,292 | 114.6 | 1.34 s | 0.24 | 3.98 | 10 % |
| vLLM 0.33.0 + LMCache 0.5.7.post1 | 2,658 | 132.9 | 0.32 s | 0.14 | 0.76 | 7 % |

Paired on the requests both sent (95 % intervals over users):

| A | B | requests | B - A |
|---|---|---|---|
| SGLang + HiCache | the same, rerun | 1,661 | +0.07 s [-0.10, +0.25] |
| SGLang + HiCache | vLLM 0.33.0 | 1,642 | -4.84 s [-5.64, -4.01] |
| SGLang + HiCache | vLLM + LMCache | 1,676 | -8.89 s [-10.34, -7.67] |
| SGLang + HiCache | Continuum | 1,665 | -7.67 s [-8.93, -6.56] |
| SGLang + HiCache | TokenCake | 1,676 | -8.55 s [-9.93, -7.40] |
| vLLM 0.33.0 | vLLM + LMCache | 1,855 | -4.77 s [-6.14, -3.77] |
| vLLM + LMCache | Continuum | 2,263 | +1.35 s [+0.94, +1.88] |
| vLLM + LMCache | TokenCake | 2,292 | +1.03 s [+0.82, +1.27] |
| Continuum | its FCFS base (same build) | 2,133 | -0.06 s [-0.53, +0.33] |
| TokenCake | its native base (same build) | 1,537 | +8.63 s [+6.81, +10.97] |

- The rerun reproduces the campaign's HiCache arm, so these 20-minute comparisons resolve about 0.2 s.
- Against their own stock engines: Continuum's pin ties its FCFS base; TokenCake's preservation and scheduling cut
  8.6 s from its native base, which recomputes 42 % of prompt tokens.
- The largest gap is the engine's speed, not its cache: a cold 20,000-token prefill takes 1.454 s on our SGLang
  (Triton attention, as every campaign), 1.371 s on Continuum's vLLM 0.10.2, 1.290 s on TokenCake's vLLM 0.22.0 and
  0.557 s on vLLM 0.33.0. Stock vLLM 0.33.0 with no lower tier out-serves SGLang with three, and every vLLM-based
  system is still below 2 s mean TTFT where the SGLang arms reach 15.6 s in minutes 10 to 30.
- Before tiering policies are compared across engines, our engine needs its deployed attention backend (FA3 on
  H200), the first item of the paper-strategy document's GPU plan: done below.

## FA3: our SGLang arms on the deployed attention backend (2026-11-22)

Campaign 28's six SGLang arms rerun for the same 20 minutes with `--attention-backend fa3` and nothing else changed
(`results/fa3_smoke/`; server logs show bf16 KV and the 668,096-token pool on both backends). Paired with the Triton
runs and the baselines in `results/fa3_compare/compare_arms.out` (`--cut 1200`).

| system | requests | a minute | TTFT mean | p50 | p90 | prompt tokens recomputed |
|---|---|---|---|---|---|---|
| SGLang + HiCache, Triton (campaign 28 `hicache`) | 1,676 | 83.8 | 9.19 s | 3.02 | 23.81 | 10 % |
| SGLang + HiCache, FA3 | 2,666 | 133.3 | 0.22 s | 0.09 | 0.42 | 7 % |
| our JIT p50, FA3 | 2,665 | 133.2 | 0.20 s | 0.08 | 0.42 | 7 % |
| our four rules (`ec_full`), FA3 | 2,667 | 133.3 | 0.22 s | 0.08 | 0.45 | 7 % |
| JIT exact, FA3 | 2,668 | 133.4 | 0.21 s | 0.08 | 0.42 | 7 % |
| old oracle v2, FA3 | 2,668 | 133.4 | 0.22 s | 0.08 | 0.41 | 7 % |
| SGLang, radix cache only, Triton | 907 | 45.4 | 24.91 s | 1.61 | 88.60 | 50 % |
| SGLang, radix cache only, FA3 | 1,892 | 94.6 | 4.38 s | 0.09 | 18.39 | 33 % |
| vLLM 0.33.0, prefix cache only | 1,855 | 92.8 | 5.06 s | 0.13 | 22.31 | 34 % |
| vLLM 0.33.0 + LMCache 0.5.7.post1 | 2,658 | 132.9 | 0.32 s | 0.14 | 0.76 | 7 % |

| A | B | requests | B - A |
|---|---|---|---|
| SGLang + HiCache, Triton | the same, FA3 | 1,676 | -8.96 s [-10.38, -7.80] |
| vLLM + LMCache | SGLang + HiCache, FA3 | 2,658 | -0.10 s [-0.12, -0.08] |
| vLLM + LMCache | our JIT p50, FA3 | 2,657 | -0.12 s [-0.15, -0.11] |
| TokenCake | SGLang + HiCache, FA3 | 2,292 | -1.11 s [-1.35, -0.89] |
| Continuum | SGLang + HiCache, FA3 | 2,263 | -1.43 s [-2.01, -1.05] |
| vLLM 0.33.0 | SGLang radix cache only, FA3 | 1,851 | -1.08 s [-1.39, -0.80] |
| SGLang + HiCache, FA3 | our JIT p50, FA3 | 2,664 | -0.02 s [-0.04, -0.01] |
| SGLang + HiCache, FA3 | JIT exact, FA3 | 2,665 | -0.01 s [-0.03, -0.00] |
| SGLang + HiCache, FA3 | our four rules, FA3 | 2,666 | -0.00 s [-0.01, +0.01] |
| SGLang + HiCache, FA3 | old oracle v2, FA3 | 2,666 | -0.01 s [-0.02, +0.00] |

Mean TTFT by send time, 5-minute windows:

| minutes | HiCache, Triton | HiCache, FA3 | JIT p50, FA3 | rules, FA3 | vLLM + LMCache | TokenCake | Continuum | vLLM | SGLang radix, FA3 |
|---|---|---|---|---|---|---|---|---|---|
| 0-5 | 0.29 | 0.08 | 0.08 | 0.08 | 0.14 | 0.14 | 0.13 | 0.13 | 0.08 |
| 5-10 | 1.00 | 0.19 | 0.17 | 0.19 | 0.24 | 0.38 | 0.21 | 0.34 | 0.28 |
| 10-15 | 11.26 | 0.33 | 0.29 | 0.34 | 0.42 | 0.90 | 0.95 | 3.43 | 2.98 |
| 15-20 | 20.11 | 0.21 | 0.21 | 0.21 | 0.38 | 3.20 | 4.78 | 22.64 | 19.24 |

- FA3 turns our engine from the slowest system to the fastest: the HiCache arms deliver what vLLM + LMCache does
  (133 requests a minute, 7 % recomputed) at 0.10 s lower mean TTFT, mostly on returning requests (0.19-0.20 s
  against 0.31; a user's first request 0.53-0.65 s against 0.61). Stock SGLang on FA3 beats stock vLLM by 1.08 s.
- On FA3 the box keeps up for these 20 minutes, so the tiering policies have nothing to separate: GPU hits 75-82 % of
  prompt tokens, host 11-18 %, the SSD at most 1 %, and the five arms tie within 0.02 s. Campaigns 28 and 29 measured
  the policies in a queue that Triton's prefill built; whether the full 2 h saturates on FA3 is untested.
- A cold 20,000-token prefill: 0.637 s on FA3, against 1.371-1.454 s on Triton and 0.557 s on vLLM 0.33.0
  (`sglang_hicache/README.md`). The same prompt read back from the SSD takes 2.032 s, 3.2 times its recompute.
- Why every campaign ran Triton: `scripts/start_server.sh` pins `--attention-backend triton` with a note that the
  fp8_e5m2 KV rule rewrites fa3 to triton anyway; campaigns 28 and 29 run bf16 KV (`KV_DTYPE=auto`), where the
  automatic resolution is fa3 (`model_override_base.py`), so the pin no longer matches it.

## The full comparison

Campaign 30 (FreeInference) ran 2026-11-24 at campaign 34's setting, with stock SGLang on FA3, Continuum with and
without LMCache, and every server bound to 127.0.0.1: record `results/campaign30/DECISIONS.md`, paired with campaign
34's arms in `results/campaign30/checks/vs_c34/` (`common/update_compare.sh campaign30/checks/vs_c34 6600 campaign34
campaign30`). `results/campaign31/` (TraceLab) is prepared and not launched.

`results/campaign30/` (FreeInference) and `results/campaign31/` (TraceLab) copy the settings of campaigns 28 and 29
(0.1 users/s, cap 128, 160 GB host, x1 gaps, 2 h) and queue the baselines; each fork's own stock base is a commented
line. About 2 h 10 min an arm:

```bash
# host: both traces, one after the other; the launcher's log names each arm's LAUNCH and DONE
cd /home/wanhr/sglang/agent_cache && : > results/.current_run && setsid nohup bash baselines/common/run_baselines.sh \
  campaign30 campaign31 >> results/campaign30/runner.log 2>&1 < /dev/null &
```

Then `python3 -I -B frozen/oracle_scripts_staging/c28_users/compare_arms.py results/campaign30` with campaign 28's arms
added to its manifest, as `results/baseline_smoke_compare/` does.

## Traps

- Port 5555 was briefly taken by another process on this box; LMCache's server listens on 6555 and 6580 here.
- vLLM 0.33.0 builds a FlashInfer sampling kernel at its first boot and needs the venv's `ninja`: the start scripts
  put the venv's `bin` first on PATH, as `source .venv/bin/activate` would.
- A vLLM server refuses a GPU pool smaller than one request at `--max-model-len`: small-pool checks boot with
  `CTX=65536`.
- One server on the GPU at a time: vLLM's memory profiling aborts when another process allocates during a boot, so
  two runners must never share the box (`common/run_official.sh` refuses a busy GPU and waits for a live launcher).
- A single torch-pinned allocation over 128 GiB fails on this host: torch rounds pinned requests up to a power of two,
  and 256 GiB is more than the host's 196 GiB of RAM. TokenCake (vLLM 0.22's CPU offloading pins its tier as one
  tensor) gets 120 GiB; LMCache pins in pieces and SGLang registers its pool at its size, and both get the full 160 GB.
- TokenCake's experiment driver sets `CUDA_VISIBLE_DEVICES` to the GPU's UUID, which vLLM 0.22 cannot parse;
  `tokencake/official/dag.sh` puts a failing `nvidia-smi` stub first on its PATH so it passes the index.
- TokenCake's first boot builds FlashInfer kernels for about 35 minutes (cached afterwards); its driver keeps vLLM's
  compile caches in its run folder, which `.gitignore` leaves out.
- Continuum's agent runner takes `--port` into a config field its vLLM model class does not have; its config's
  `model.base_url` is set instead (`continuum/official/swebench.sh`).
- `run_baselines.sh` pins its inputs (`inputs.sha256`, including `common/backends.py`): an edit there while a folder
  runs stops its next arm, unless the folder is re-pinned deliberately (and noted in its `launcher.log`).
