# DFlash / DFlash2 / DSpark / EAGLE3 speculative-decoding comparison in SGLang — runbook

Written 2026-09-09 for this box (1x H100 PCIe 80 GB). Every command and flag below
was checked against the checkout at `2cb739b9a7` and against the live host before it
was written down; the "Evidence" notes give the file:line or command that proves it.
Nothing here has been *executed end to end yet* (no container existed when this was
written), so treat the memory numbers as derived, not measured, and record what you
actually see in `results/`.

Source of truth for DFlash itself: https://github.com/z-lab/dflash (pip package
`dflash` 0.1.0; DFlash paper arXiv:2602.06036; DFlash 2 blog https://inco.ai/blog/dflash2/).

---

## 0. The short version

1. There is **no single target model with public drafts for all four methods.** Run two tracks:
   - **Track A — `Qwen/Qwen3-8B`**: EAGLE3 vs DFlash (v1) vs DSpark. All three drafts exist.
   - **Track B — `Qwen/Qwen3.8-27B-FP8`**: DFlash2 vs DSpark vs the checkpoint's own MTP head. This is the
     model the DFlash 2 release headlines. No EAGLE3 draft exists for it; DFlash v1 exists only as
     unvalidated community drafters.
2. Bring-up: add yourself to the `docker` group, start `sglang_dev` from `lmsysorg/sglang:dev`
   with the repo and HF cache bind-mounted, pre-download ~118 GB of checkpoints.
3. Install the `dflash` client in a **host** venv and apply a one-line patch, otherwise it will
   never print acceptance length against this checkout.
4. For each arm: launch one server, `POST /flush_cache`, run `dflash benchmark openai`, read
   `/server_info`, stop the server. Interleave arms per cell so GPU throttling does not bias one method.
5. Headline comparison is **greedy, non-thinking, concurrency 1 and 8, gsm8k + humaneval**.
   Expand to the 5-dataset matrix and sampling only after that reproduces the published ordering.

---

## 1. What is on this box (verified 2026-09-09)

| | |
|---|---|
| GPU | 1x NVIDIA H100 PCIe, 81559 MiB, SM90, ~2.0 TB/s HBM2e (H200 is 4.8 TB/s: expect ~0.4-0.5x of the model cards' absolute tok/s; ratios and acceptance lengths should carry over) |
| Driver | 580.105.08, CUDA 13.0. The image needs `cuda>=13.0`: OK |
| Host | Ubuntu 24.04, 26 vCPU, 221 GB RAM, 968 GB free on `/`, Python 3.12.3, `python3 -m venv` works, no `uv`, no `jq` |
| Docker | docker-ce 29.2.1 + NVIDIA Container Toolkit 1.18.1 pre-baked in the Lambda image. **The `docker` group has no members**, so every `docker` command needs `sudo` until Step 2.1 is done. No `nvidia` runtime is registered and none is needed: `--gpus all` works through CDI (`/var/run/cdi/nvidia.yaml`, regenerated at boot). `--runtime=nvidia` fails. |
| Containers / images | none (a 30 MB `ubuntu:24.04` was pulled for the GPU smoke test) |
| HF cache | `~/.cache/huggingface` does not exist yet |
| Repo | `/lambda/nfs/MLSys-Learn/sglang` at `2cb739b9a7` (main). The `dev` image was built from upstream `db272201a2` (2026-09-08), 1-2 days newer; `python/pyproject.toml` pins are identical, so bring-up is unaffected. |

SGLang support in this checkout (`python/sglang/srt/speculative/spec_info.py:40-48`): `DFLASH`, `DSPARK`,
`EAGLE`, `EAGLE3`, `UNO`, `FROZEN_KV_MTP`, `STANDALONE`, `NGRAM`.

- **DFlash v1 and DFlash2 both use `--speculative-algorithm DFLASH`.** Which class loads is decided by the
  draft checkpoint's `config.json` `architectures[0]`: `DFlashDraftModel` (v1) or `DFlash2DraftModel`
  (`python/sglang/srt/models/registry.py:55-59`, `models/dflash.py:1065-1083`). DFlash2 needs
  `dflash_config.selector_rank/selector_top_k/conv_kernel_size/conv_group_size` in that config and raises
  if they are missing.
- **DSpark** draft classes: `Qwen3DSparkModel`, `LingDSparkModel`, `DSparkDraftModel` (`models/dspark.py:884`).
- **EAGLE3** with a `LlamaForCausalLMEagle3` draft on a Qwen3 target is supported natively
  (`models/qwen3.py:688-692` captures the aux hidden states; the Tengyunw card's "replace qwen3.py"
  instruction is obsolete).

---

## 2. Bring-up

### 2.1 Docker group (required once)

```bash
sudo usermod -aG docker "$USER"
```

Then open a **new login shell** (SSH again). Existing shells, including any Claude Code / VS Code
session, keep their old credentials forever (README gotcha 1); in those use `sudo docker ...` or
`sg docker -c '...'`. Verify a shell with:

```bash
id -nG | tr ' ' '\n' | grep -x docker && docker info >/dev/null && echo OK
docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

Do **not** run `/lambda/nfs/MLSys-Learn/setup-gpu-docker.sh` for real after the container is up: it
registers the nvidia runtime and does `systemctl restart docker`, which kills every container
(live-restore is off). It is not needed for GPU access here. `--dry-run` is safe. If you want it,
run it before Step 2.2, or with `--skip-toolkit`.

### 2.2 Start the dev container

```bash
mkdir -p "$HOME/.cache/huggingface"       # must exist first, else dockerd creates it root-owned
docker pull lmsysorg/sglang:dev            # 15.6 GB
docker run -itd --name sglang_dev \
  --gpus all --shm-size 32g --ipc=host --network=host --privileged \
  -v /lambda/nfs/MLSys-Learn/sglang:/sgl-workspace/sglang \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  lmsysorg/sglang:dev /bin/zsh
docker exec sglang_dev nvidia-smi
docker exec sglang_dev python3 -c 'import sglang; print(sglang.__file__)'   # must be under /sgl-workspace/sglang/python/
docker exec sglang_dev git config --global --add safe.directory /sgl-workspace/sglang
```

Always pass the real `/lambda/nfs/...` path to `-v`, not `~/MLSys-Learn` (bind mounts are resolved by
the daemon). The image pip-installs sglang in editable mode at `/sgl-workspace/sglang`
(`docker/Dockerfile:638-645`), so the bind-mounted checkout is what runs.

Gotchas from the box README that apply here: files written by the container are `root:root` on the
host (`sudo chown -R ubuntu:ubuntu /lambda/nfs/MLSys-Learn/sglang` and `~/.cache/huggingface` when
they get in the way); an empty host HF cache hides the image's prebuilt FA3 cubins, so the first FA3
load re-downloads them (then they persist).

### 2.3 Pre-download checkpoints (inside the container; none are gated, no token needed)

```bash
docker exec sglang_dev bash -c '
for m in Qwen/Qwen3-8B Tengyunw/qwen3_8b_eagle3 z-lab/Qwen3-8B-DFlash-b16 deepseek-ai/dspark_qwen3_8b_block7 \
         Qwen/Qwen3.8-27B-FP8 RadixArk/Qwen3.8-27B-DSpark incoai/Qwen3.8-27B-DFlash2; do
  hf download "$m"
done'
sudo chown -R ubuntu:ubuntu "$HOME/.cache/huggingface"
```

`hf` is on PATH in the image (huggingface_hub >= 1.5 via transformers 5.12.1). `huggingface-cli` is a
dead shim in hub 1.x; the fallback is
`python3 -c 'from huggingface_hub import snapshot_download; snapshot_download("<id>")'`.

Sizes (HF API, `?blobs=true`): Qwen3-8B 16.4 GB; qwen3_8b_eagle3 0.8 GB (pytorch_model.bin);
Qwen3-8B-DFlash-b16 2.1 GB; dspark_qwen3_8b_block7 4.7 GB; Qwen3.8-27B-FP8 30.9 GB;
Qwen3.8-27B-DSpark 3.7 GB; Qwen3.8-27B-DFlash2 3.85 GB. Total ~63 GB; add `Qwen/Qwen3.8-27B` (55.6 GB)
only if you run the BF16 variant in §6.3.

### 2.4 The DFlash benchmark client (host venv, not in the container)

`dflash` pins `datasets==5.0.1 requests==2.34.2 tqdm==4.70.0` exactly; keep it out of the image's
Python.

```bash
python3 -m venv ~/dflash-venv && source ~/dflash-venv/bin/activate
pip install dflash
bash /lambda/nfs/MLSys-Learn/sglang/dflash_eval/scripts/patch_dflash_client.sh   # see below
dflash benchmark --help
```

**Why the patch is mandatory.** The client reads acceptance length from the *top-level*
`meta_info` of the chat-completion JSON (`dflash/benchmark.py:400`). This checkout returns
`meta_info` *per choice* (`python/sglang/srt/entrypoints/openai/protocol.py:1218`,
`serving_chat.py:2161-2214`); there is no top-level key. Unpatched, only "Throughput" prints. The
patch script changes that one line to
`meta = out.get("meta_info") or ((out.get("choices") or [{}])[0].get("meta_info")) or {}`.

What the client does (`dflash/benchmark.py`): non-streaming `POST /v1/chat/completions` with
`return_meta_info: true`, `chat_template_kwargs` from `--reasoning`, `top_k` if > 0; prompts shuffled
with `random.Random(42)` (same order for every server); a warm-up of `concurrency` requests at
64 tokens; `ThreadPoolExecutor(concurrency)`; prints end-to-end output tok/s
(`sum(completion_tokens) / wall`, prefill and reasoning tokens included), mean per-request
`spec_accept_length`, and `spec_verify_ct`. `--max-samples` is ignored by the openai backend; use
`--num-prompts`. Datasets: `gsm8k`, `math500`, `humaneval`, `mbpp`, `mt-bench` (80 prompts, first
turn only, so `--num-prompts 80` there). No HF token or tokenizer download is needed on the host.

---

## 3. Checkpoint matrix

| Target | EAGLE3 | DFlash v1 | DFlash2 | DSpark | MTP |
|---|---|---|---|---|---|
| `Qwen/Qwen3-8B` (8.19B bf16) | `Tengyunw/qwen3_8b_eagle3` (alt `AngelSlim/Qwen3-8B_eagle3`; both `LlamaForCausalLMEagle3`, 1 layer, draft vocab 32000) | `z-lab/Qwen3-8B-DFlash-b16` (block 16, 1.05B) | none servable (only a raw training dump on HF) | `deepseek-ai/dspark_qwen3_8b_block7` (block 7, 2.37B, `Qwen3DSparkModel`, no model card) | none |
| `Qwen/Qwen3.8-27B-FP8` (27.8B, block-FP8, 30.9 GB) | none | community only (`kstoyanov99/Qwen3.8-27B-Dflash`, `jfan/Qwen3.8-27B-heretic-dflash`; block 16, 1.73B, unvalidated) | `incoai/Qwen3.8-27B-DFlash2` (= `z-lab/Qwen3.8-27B-DFlash2` mirror; block 8, selector top-16, 1.92B) | `RadixArk/Qwen3.8-27B-DSpark` (gamma 7, 1.86B) | in checkpoint (`mtp.*`, FP8) |

Verify windows differ and bound acceptance length: DFlash-b16 = 16, EAGLE3 3/4/16 = 16, DSpark
block 7 = 8, DFlash2 = 8, MTP 3/1/4 = 4. **Print the window next to every acceptance length** and
use the matched-width arms in §5 for like-for-like acceptance comparisons.

Precision asymmetry on Track B: the MTP draft is the FP8 checkpoint's own tensors; DSpark and
DFlash2 drafts load in BF16. Say so when comparing.

---

## 4. Controls that must be identical across arms

- Same target checkpoint, `--dtype`, `--attention-backend` and `--speculative-draft-attention-backend`
  per track; same `--mem-fraction-static`, `--max-running-requests`, `--cuda-graph-max-bs-decode`,
  `--chunked-prefill-size`, same env (`PYTORCH_CUDA_ALLOC_CONF`), same piecewise-CUDA-graph setting
  (default on) for **every** arm including the baseline.
- `--disable-radix-cache` everywhere (no prefix reuse; also what the Ascend DFlash2 comparison did).
- Overlap scheduler stays on for all arms (with radix off the mamba radix pass is a no-op, so nothing
  forces it off: `arg_groups/overrides.py:600-613`).
- Losslessness: `--speculative-accept-threshold-single` / `-acc` stay at their default 1.0; never set
  `--speculative-use-rejection-sampling` (off by default, incompatible with topk 4). With
  `temperature 0` all four methods verify greedily (DFlash `dflash_worker_v2.py:1786-1801`, DSpark
  `dspark_verify.py:725-732`, EAGLE); with sampling DFlash and EAGLE3-topk4 both use target-only
  sampling verify.
- **No presence / frequency / repetition penalties in any arm.** Main's DFlash verify path lacks
  the penalty handling on `upstream/dcw02/dflash-penalty-fix` (`dflash_utils.py:209-296`); the
  client sends none by default. (The Inco blog's presence_penalty 1.5 is only its Qwen3.5-4B table;
  the Qwen3.8-27B table uses the model's default sampling.)
- `SGLANG_RAGGED_VERIFY_MODE`: default is `static` (`environ.py:1256`) = verify the whole block. Keep it
  for the comparison; that is also what the DSpark model-card throughput used. `compact` is the
  confidence-budgeted mode CI exercises; `cap-accept` is never a verify-all control.
- Same prompt set and order (client seed 42), same `--max-new-tokens`, warm-up flushed before measuring.
- Run arms **interleaved per cell** (A0, A1, A2, A3 on gsm8k/c1, then the next cell) and log
  `nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 5` in the
  background; the 350 W PCIe card can throttle across long thinking-mode runs.

---

## 5. Track A — Qwen3-8B: EAGLE3 vs DFlash vs DSpark

All servers run inside the container on port 30000, one at a time. `scripts/launch.sh` wraps these.

```bash
COMMON_A="--model-path Qwen/Qwen3-8B --trust-remote-code --dtype bfloat16 \
  --attention-backend fa3 --speculative-draft-attention-backend fa3 \
  --disable-radix-cache --mem-fraction-static 0.80 \
  --max-running-requests 32 --cuda-graph-max-bs-decode 32 \
  --enable-metrics --decode-log-interval 10 --host 0.0.0.0 --port 30000"
```

| Arm | Extra flags | Window |
|---|---|---|
| A0 baseline | (none) | 1 |
| A1 EAGLE3 | `--speculative-algorithm EAGLE3 --speculative-draft-model-path Tengyunw/qwen3_8b_eagle3 --speculative-num-steps 3 --speculative-eagle-topk 4 --speculative-num-draft-tokens 16` | 16 |
| A1-auto | same but with none of the three numeric flags: this checkout auto-tunes Qwen3ForCausalLM to **3/1/4** (`speculative_hook.py:1125-1162`), not 3/4/16. Setting only one or two of the three flags trips an assert. | 4 |
| A1-w8 | `... --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8` (matched width for DSpark) | 8 |
| A1-card | `... --speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32` (the Tengyunw card's setting) | 32 |
| A2 DFlash | `--speculative-algorithm DFLASH --speculative-draft-model-path z-lab/Qwen3-8B-DFlash-b16 --speculative-num-draft-tokens 16` (16 = the draft's `block_size`; omitting the flag auto-infers it, `speculative_hook.py:254-289`; do not "tune" it like an EAGLE knob) | 16 |
| A2-w8 | `... --speculative-num-draft-tokens 8` (matched width) | 8 |
| A3 DSpark | `--speculative-algorithm DSPARK --speculative-draft-model-path deepseek-ai/dspark_qwen3_8b_block7 --speculative-dspark-block-size 7` (gamma auto-infers to 7 anyway; **never** pass `--speculative-num-draft-tokens` to DSPARK unless it equals gamma+1, `speculative_hook.py:600-616`) | 8 |

Expected after launch (derived, verify in the log): `max_total_num_tokens` roughly 330k / 291k / 275k
for A1 / A2 / A3 at 0.80; the draft weights and draft KV pool live inside the static fraction, and 0.80
leaves ~15.75 GiB for activations and CUDA graphs. The DFlash card uses 0.75 and the repo's CI tests 0.7
on smaller cards; 0.80 is fine on 80 GB.

Wait for `The server is fired up and ready to roll` (`http_server.py:2417`) or poll
`curl -s localhost:30000/health`. Then **grep the boot log** (kept under `results/`) for
`max_total_num_tokens`, `max_running_requests`, the resolved `speculative_num_draft_tokens`, and any
of: `Disable DFLASH draft cuda graph`, `Disable DSpark draft cuda graph`,
`DSpark folded sampling disabled`, `kept eager`, `Non-overlap (synchronous) spec v2`. If a draft-graph
line fires, lower `--mem-fraction-static` for **all** arms and relaunch.

### 5.1 Measuring a cell (host venv; `scripts/bench.sh` wraps this)

```bash
source ~/dflash-venv/bin/activate
curl -s -X POST http://127.0.0.1:30000/flush_cache          # zero the lifetime accept-length accumulator
dflash benchmark openai --base-url http://127.0.0.1:30000 --model Qwen/Qwen3-8B \
  --dataset gsm8k --num-prompts 128 --concurrency 1 --max-new-tokens 2048 \
  --temperature 0 --reasoning off
python3 -c "import json,urllib.request;print(json.load(urllib.request.urlopen('http://127.0.0.1:30000/server_info'))['internal_states'][0].get('avg_spec_accept_length'))"
```

- `/server_info` `avg_spec_accept_length` is a **lifetime** accumulator (`scheduler.py:4970-4976`), reset
  only by `/flush_cache` or `/set_internal_state`. Flush before every cell or it blends the warm-up and all
  previous cells. The client's number is a per-request mean of `completion_tokens/spec_verify_ct`;
  `/server_info` and the periodic decode log (`Decode batch ... accept len: X`) are token-weighted. Record which one you quote.
- `--model` is not validated against the served name; any string works. Avoid `:` (parsed as a LoRA adapter).
- Headline pass: `--temperature 0 --reasoning off`, concurrency 1 and 8, `gsm8k` and `humaneval`, 128 prompts.
  Both Track A drafters were trained for non-thinking chat (DFlash-b16 card: "thinking mode disabled";
  Tengyunw: UltraChat), so thinking-mode sampling is out of distribution for them; report it as secondary.
- Secondary pass: `--temperature 1 --top-p 0.95 --top-k 20 --reasoning on` (the DFlash README's Qwen3.8
  recipe; note Qwen3-8B's own `generation_config.json` is 0.6/0.95/20 in thinking mode). Do 64 prompts x 2
  repeats on gsm8k/c1 first to see run-to-run variance.
- Cross-check with SGLang's own client **inside the container** (it imports torch/aiohttp; the host has neither):
  ```bash
  docker exec sglang_dev python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port 30000 \
    --dataset-name sharegpt --num-prompts 128 --max-concurrency 1 --flush-cache
  ```
  It prints accept length from `/server_info` and gives TPOT/ITL. First use downloads ~673 MB of ShareGPT
  into the bind-mounted HF cache. `python3 -m sglang.bench_serving` still works but warns.
- `--enable-metrics` also exposes `sglang:spec_accept_length` on `/metrics`.

Runtime estimate on this card (bandwidth-bound): non-thinking gsm8k, 128 prompts, c1: ~6-8 min baseline,
2-3 min per spec arm. Thinking with a 2048 cap: ~35 min baseline, 12-15 min per spec arm. The full
draft matrix is 15-25 GPU-hours; the headline pass is ~2 h for Track A.

---

## 6. Track B — Qwen3.8-27B: DFlash2 vs DSpark vs MTP

### 6.1 Memory on 80 GB (why the flags below look the way they do)

Qwen3.8-27B is a hybrid GDN model: besides KV it keeps a per-request SSM state slot (146.8 MiB at
float32 state, 74.8 MiB at bfloat16; `mamba_utils.py:116-125`) and, under speculation, a verify
intermediate of `D x (K+1)` slots where D is the verify window and K = `--max-running-requests`
(`kv_cache_configurator.py:2346-2384`).

- With `--disable-radix-cache` **and** `--max-running-requests K`, the state pool is pinned to K slots and
  `--mamba-full-memory-ratio` is **never read**. Concurrency ceiling = K. Do not derive the ratio for
  these runs (the cookbook calculator applies to radix-on serving).
- The speculative hook defaults `--max-running-requests` to **48** when omitted
  (`speculative_hook.py:319-327`). At 0.80 on this card that makes DSpark/DFlash2 fail with
  `Not enough GPU memory for hybrid (mamba/linear-attention) state cache`. **`--max-running-requests 16` is
  mandatory** for B2/B3 on the FP8 checkpoint.
- FP8 target (28.7 GiB) + a 3.5-3.9 GiB BF16 draft at 0.80, K=16, D=8, float32 state: verify intermediate
  19.5 GiB + pool 2.4 GiB, leaving ~8 GiB of fp8 KV = ~200k tokens = ~12k tokens per request at K=16. Enough
  for gsm8k/humaneval with a 2048-4096 output cap.

### 6.2 Launch commands (FP8 target)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # on EVERY Track B arm, or none (from the DSpark card)
COMMON_B="--model-path Qwen/Qwen3.8-27B-FP8 --trust-remote-code \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 \
  --attention-backend fa3 --speculative-draft-attention-backend fa3 \
  --disable-radix-cache --mem-fraction-static 0.80 \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
  --chunked-prefill-size 8192 --max-prefill-tokens 8192 \
  --reasoning-parser qwen3 --enable-metrics --decode-log-interval 10 --host 0.0.0.0 --port 30000"
```

| Arm | Extra flags | Window |
|---|---|---|
| B0 baseline | (none) | 1 |
| B1 MTP-4 | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (cookbook / DSpark-card baseline) | 4 |
| B1 MTP-8 | `--speculative-algorithm EAGLE --speculative-num-steps 7 --speculative-eagle-topk 1 --speculative-num-draft-tokens 8` (the DFlash2 card's "seven-token MTP" baseline; its GSM8K AL 5.02 is impossible with a 4-token window) | 8 |
| B2 DSpark | `--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark --speculative-dspark-block-size 7 --speculative-num-steps 1` | 8 |
| B3 DFlash2 | `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8` | 8 |

Choices and deviations, stated once:
- `fa3` for target and draft on every arm: the DFlash2 card validated FA3 (target + draft) on one H200;
  the cookbook's H200 cells use `flashinfer` and list `fa3` as a valid, slightly faster bs=1 alternative.
  The cookbook marks **DFlash2 + flashinfer + fp8 KV on H200 as "in progress"** (unvalidated), which is why
  fa3 is the primary here. If any arm fails on fa3, switch **all** arms to `--attention-backend flashinfer
  --speculative-draft-attention-backend flashinfer` rather than mixing.
- `--kv-cache-dtype fp8_e4m3` is the cookbook convention for this model (no calibration scales in the
  BF16/FP8 checkpoints; the DSpark card used it). DFlash2's card ran bf16 KV. If B3 misbehaves, first add
  `--speculative-draft-kv-cache-dtype bf16`, then try `--kv-cache-dtype bfloat16` on all arms.
- `--mamba-ssm-dtype float32` = the checkpoint's declared precision and what both model cards used;
  `bfloat16` halves the state but can move acceptance length.
- `--chunked-prefill-size 8192` instead of the cookbook H200's 32768: deliberate for 80 GB at 0.80.
- `--reasoning-parser qwen3` is valid here (`parser/reasoning_parser.py:2116`) and only affects the
  response split, not throughput or acceptance length.
- Optional: `--language-model-only` skips the ~0.9 GiB vision tower (`fields/disagg.py:126`).

### 6.3 BF16 target variant (`Qwen/Qwen3.8-27B`, 51.75 GiB weights)

Reusing `COMMON_B` with the BF16 path raises the hybrid-state-cache error for B2/B3 (11.2 GiB of state
needed vs ~7.4 GiB left). Working points: B2/B3 at `--mem-fraction-static 0.88 --max-running-requests 8
--cuda-graph-max-bs-decode 8 --chunked-prefill-size 4096` (~190k KV tokens), or concurrency-1 only at
0.80 with `--max-running-requests 1 --cuda-graph-max-bs-decode 1`. B0/B1 fit at 0.80 with K=16. Use it
only if you need BF16 parity with the DFlash2 card; the FP8 checkpoint is the DSpark card's setting.

### 6.4 Measuring Track B

Same procedure as §5.1 with `--model Qwen/Qwen3.8-27B-FP8` and the DFlash README sampling:

```bash
dflash benchmark openai --base-url http://127.0.0.1:30000 --model Qwen/Qwen3.8-27B-FP8 \
  --dataset gsm8k --num-prompts 128 --concurrency 1 --max-new-tokens 2048 \
  --reasoning xhigh --temperature 1 --top-p 0.95 --top-k 20
```

- Valid `--reasoning` values for Qwen3.8 are `on|off|xhigh|medium|low` (the chat template raises on
  anything else, so `high` gives HTTP 400). `on` equals `xhigh` (template default).
- The DFlash2 card used `max_new_tokens 4096`, the DSpark card 2048. Start at 2048; rerun the headline
  cells at 4096 if you want card parity.
- Also run a greedy pass (`--temperature 0 --reasoning off`) so Track B has the same headline control as Track A.

---

## 7. What to record per cell and what "reproduced" means

Per cell (arm x dataset x concurrency x sampling), store under `results/<YYYYMMDD_HHMM>/`:
the boot log, the client stdout, `/server_info` `avg_spec_accept_length` after the cell, the verify
window, the `nvidia-smi` clock/power trace, and the exact command (the scripts do this).

### 7.1 The scripts (`scripts/`; syntax-checked and dry-run, not yet run against a live server)

```bash
cd /lambda/nfs/MLSys-Learn/sglang/dflash_eval
RUN=$(date +%Y%m%d_%H%M); mkdir -p results/$RUN; echo $RUN > .current_results   # env.sh reads this
source ~/dflash-venv/bin/activate
export DOCKER="sudo docker"          # or "docker" in a shell that is in the docker group

scripts/launch.sh print A2           # show the exact server command for an arm
scripts/launch.sh A2                 # start it in the container, wait for /health, grep the boot log
scripts/bench.sh A2 gsm8k 1 greedy   # flush_cache -> dflash benchmark -> /server_info -> summary.csv
scripts/bench.sh A2 humaneval 8 greedy
scripts/launch.sh stop               # kill the server before the next arm
```

Arms: `A0 A1 A1-auto A1-w8 A1-card A2 A2-w8 A3 B0 B1-4 B1-8 B2 B3` (flags in `scripts/env.sh`,
`arm_flags`). Sampling presets in `bench.sh`: `greedy`, `think` (Track A), `xhigh` (Track B).

`scripts/run_track_a.sh` and `scripts/run_track_b.sh` (thin wrappers over `run_track.sh`) run a
whole track unattended: one launch per arm, all cells against it, stop, next arm. They skip cells
already in `summary.csv` (so a `RUN=<dir>` rerun resumes), record launch/bench failures in
`failures.log` and keep going, and always stop the server on exit. Knobs are env vars: `ARMS`,
`CELLS` (`"<dataset> <conc>|..."`), `SAMPLING` (`greedy|think` for A, `greedy|xhigh` for B),
`ORDER=arm|cell` (per-arm blocks, or interleaved per cell for thermal fairness at the cost of a
relaunch per cell), `NUM_PROMPTS`, `MAX_NEW`, `DRY_RUN=1`.

```bash
source ~/dflash-venv/bin/activate
DRY_RUN=1 scripts/run_track_a.sh                       # print the plan
nohup scripts/run_track_a.sh > /tmp/track_a.out 2>&1 &  # ~1 h for the headline pass
tail -f results/$(cat .current_results)/track_a.log

DRY_RUN=1 scripts/run_track_b.sh                                        # B0 B1-4 B1-8 B2 B3, greedy
nohup scripts/run_track_b.sh > /tmp/track_b.out 2>&1 &                  # greedy control pass
nohup env RUN=$(cat .current_results) SAMPLING=xhigh scripts/run_track_b.sh > /tmp/track_b_xhigh.out 2>&1 &   # model-card sampling, same results dir
python3 scripts/plot_speedup.py --with-accept-length                    # greedy chart
python3 scripts/plot_speedup.py --sampling xhigh --with-accept-length   # xhigh chart
```

Measured 2026-09-09, greedy, gsm8k + humaneval, c1 and c8 (charts: `results/<run>/speedup_greedy_accept.png`,
rendered with `python3 scripts/plot_speedup.py --run <run> --with-accept-length`):

| track / run | baseline c1 / c8 | arms, speedup range (acceptance length) |
|---|---|---|
| A `20260909_1855` Qwen3-8B | 104 / ~750 tok/s | EAGLE3 3/4/16 2.0-2.5x (3.3-3.4); DFlash-b16 3.8-4.9x (6.1-6.5); DSpark block7 3.9-4.7x (5.9-6.4) |
| B `20260909_2044` Qwen3.8-27B-FP8 | 55 / ~360 tok/s | MTP 3/1/4 2.1-2.6x (3.6-3.7); MTP 7/1/8 2.4-3.2x (5.6-6.2); DSpark 2.7-4.1x (4.9-6.0); DFlash2 3.4-4.6x (6.0-6.5) |

Track B ordering matches the DFlash2 card (DFlash2 > MTP-8 on acceptance length, DFlash2 > DSpark on
throughput); absolute tok/s is about half the H200 card numbers, as expected for the PCIe part. The
`xhigh` (thinking) pass for Track B has not been run yet.

`results/$RUN/summary.csv` collects one row per cell: arm, window, dataset, concurrency, sampling,
client throughput, client accept length, server accept length, log name.

Published reference points (different hardware; treat ratios and acceptance length within ~10% as
reproduced, absolute tok/s will be ~0.4-0.5x on this PCIe card):

| Source | Setup | Numbers |
|---|---|---|
| DFlash v1 card, Qwen3-8B | H100-class, greedy, non-thinking | "up to 6.17x" lossless, "~2.5x faster than EAGLE-3" |
| DFlash2 card, Qwen3.8-27B | 1x H200, BF16 target, FA3, temp 1/top-p 0.95/top-k 20, xhigh, 4096 tokens | AL: gsm8k 5.46, math500 5.28, humaneval 4.39, mbpp 4.79, mt-bench 4.10; c1 speedup 3.43x / 3.34x / 3.11x / 3.29x / 2.67x |
| Inco DFlash2 blog, Qwen3.8-27B, block 8 | same | mean AL: MTP(7-token) 4.28, DSpark 3.62, DFlash2 4.80; gsm8k: 5.02 / 4.36 / 5.46 |
| DSpark card, throughput | 1x H200, **FP8** target, fp8 KV, float32 SSM, static verify-all, extra_buffer, 128 prompts, 2048 tokens, xhigh | gsm8k c1: AR 94.2, EAGLE(MTP 3/1/4) 179.9, DSpark 297.3 tok/s (3.16x); c8: 602.7 / 1001.1 / 1494.0; c32: 1298.5 / 1969.5 / 2268.5 |
| DSpark card, acceptance | 4x GB300 DP4, **NVFP4** target, c128, 8k tokens | gsm8k 4.52, humaneval 3.85, math500 4.23, mt-bench 3.29 (not the same setup as the throughput row) |
| Tengyunw EAGLE3 card, Qwen3-8B | H200, 6/10/32 | 187 -> 365 tok/s |

---

## 8. Optional / later

- **DSpark budgeted verify at high concurrency.** The LMSYS DSpark blog's gains at c>=32 need
  `SGLANG_RAGGED_VERIFY_MODE=compact` plus an SPS cost table from
  `python3 -m sglang.benchmark.dspark_sps_profiler` (10-30 min, separate server) passed via
  `--speculative-dspark-sps-table-path`. Irrelevant at c=1/c=8 and not part of the model-card numbers.
- **DFlash v1 on Qwen3.8-27B** via the community drafters in §3 (unvalidated; same DFLASH flags with
  `--speculative-num-draft-tokens 16`).
- **Muse-Glimmer-30B** is the other DFlash2 model (`incoai/Muse-Glimmer-30B-DFlash2`, DFlash v1 draft
  `meta-models/Muse-Glimmer-30B-assistant`); it has cookbook coverage in
  `docs/cookbook/autoregressive/Meta/MuseGlimmer.mdx` but no DSpark/EAGLE3 drafts, so it adds nothing
  to the four-way comparison.
- `git fetch upstream` before the runs if you want the checkout to match the image's build commit.

---

## 9. How this runbook was verified

Six independent checks were run against the checkout and the live host before writing (host/container
bring-up including a real `--gpus all` smoke test; every CLI flag and model-registry path; the memory
accounting from `kv_cache_configurator.py`; the DFlash client source vs this checkout's OpenAI server;
methodology/fairness; and a completeness critic). Corrections they forced on the first draft, so you do not
re-learn them: the docker group is empty (not just "this shell"); `setup-gpu-docker.sh` would restart
dockerd; the `dflash` client needs the `meta_info` patch; `/server_info` accept length is lifetime;
`--mamba-full-memory-ratio` is ignored with radix off; `--max-running-requests` must be <= 16 for
DSpark/DFlash2 on 80 GB FP8; EAGLE3 auto-tune is 3/1/4 not 3/4/16; `--cuda-graph-max-bs` is a deprecated
alias of `--cuda-graph-max-bs-decode`; `bench_serving` cannot run on the host; `jq` is not installed;
`SGLANG_RAGGED_VERIFY_MODE=static` is already the default; DFlash2 is selected by `architectures`, not
by config fields; the DSpark card's acceptance and throughput tables come from different hardware.
