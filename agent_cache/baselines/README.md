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

## Traps

- Port 5555 was briefly taken by another process on this box; LMCache's server listens on 6555 and 6580 here.
- vLLM 0.33.0 builds a FlashInfer sampling kernel at its first boot and needs the venv's `ninja`: the start scripts
  put the venv's `bin` first on PATH, as `source .venv/bin/activate` would.
- A vLLM server refuses a GPU pool smaller than one request at `--max-model-len`: small-pool checks boot with
  `CTX=65536`.
- `run_baselines.sh` pins its inputs (`inputs.sha256`, including `common/backends.py`): an edit there while a folder
  runs stops its next arm, unless the folder is re-pinned deliberately (and noted in its `launcher.log`).
