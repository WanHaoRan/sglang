# Agent KV tiering evaluation on the A100 box — runbook

Written 2026-09-16 for GCP VM `agent-cache` (a2-ultragpu-1g, 1x A100-SXM4-80GB); **updated 2026-09-17 for `Qwen/Qwen3-32B-FP8`** (`--kv-cache-dtype fp8_e5m2 --attention-backend triton`;
user decision 2026-09-17) after bring-up and the HiCache Exp 0/1 rerun on this box; HiCache Exp 2-4 are dropped. Results dirs, abbreviated below: `R32/` =
`hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv/`, `R8/` = `hicache_eval/results/20260917_a100_gcp_qwen8b/`. Flags, paths and line numbers were checked against this checkout
(`main` @ `6ec32e6b7`) and the live host; "Evidence:" gives the file:line or command. The plan being executed is `agent_cache/agent-kv-tiering-evaluation-starter-kit.md` (cited
against the 0.5.19 wheel; re-verified here, mismatches called out).

**State 2026-09-17 21:06 UTC:** §2.1-§2.4 (except §2.4's last two lines: results stamp + `versions.txt`) and §2.6 are done (two containers, use `sglang_hicache`; NVMe ext4 at `/mnt/nvme`; three models in the HF cache). Not done: §2.5 AIPerf venv
and everything in §3 (data) and §4 (client patch): `/home/wanhr/data`, `/home/wanhr/venv312`, host `uv`, `agent_cache/{scripts,patches,traces,results,.current_results}` do not exist.
The SSD is wiped at every VM stop/start and the VM is stopped overnight: redo §2.2 every morning before any `docker start`. Numbers are tagged **measured** (with source), **derived**
(arithmetic shown) or **estimate**. For the 32B, b, P, pool sizes and tier rates are measured on this box (§5.4). **Decode rate and the cost of extending a long cached prefix are NOT
measured**: every concurrency and time figure in §3.4, §5.3, §7.1 and §7.3 is provisional until they are. A third input comes from §3.5, not the GPU: the mean per-turn
`output_length` of the converted trace (the patched client replays it per turn; 220 is only the unpatched loader default, used as a placeholder here).

Inherited lessons: `hicache_eval/HANDOFF.md` §2-§7 (read it first), style from `dflash_eval/RUNBOOK.md`.

---

## 0. The short version

1. **Characterize (Week 1, CPU-only, runs on this host today, §3.5):** `pre_gap` distributions per
   source/model from the LMCache parquet; `agentic-kv-cache` `make repro` on AgentX + Mooncake
   toolagent; memory-time behind gaps >= {1,5,30} s at concurrency {4,8,12,16} (32/64/128 as an analytic row) using
   b = 131,072 B/token (measured; Qwen3-32B-FP8, fp8_e5m2 KV). Decision: upper bound on what any parking policy can free.
2. **Simulate (Weeks 2-3):** extend the simulator with tiers and per-tier restore cost from the measured constants (§5.4: P, bar,
   L1/L2/L3 rates and intercepts; recompute is superlinear, so use the measured TTFT(L) curve, not L/P). Decision: go/no-go (§9.3).
3. **Serve (Weeks 4-6, needs §2 bring-up):** SGLang + HiCache, patched gap-faithful
   `agentic-trace` client (§4) on LMCache traces, then AIPerf/AgentX; sweep HBM pressure and tiers (§5, §7).
4. **Validate (later):** live agents (mini-swe-agent, BFCL) recording traces through a proxy.

Can start now without touching the GPU: §3.1-§3.5 (downloads, converter, Week-1 stats) and §4 (client patch, reviewable
on host; runtime test needs the container). §2 is done apart from the daily NVMe re-format (§2.2, ~2 min) and the AIPerf
venv (§2.5), so §5 onward is unblocked once §3.3 and §4 exist. First GPU job: the two open constants (§5.4, matrix row 1).

---

## 1. What is on this box (verified 2026-09-17 21:06 UTC; GPU idle, no server running)

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 81920 MiB, SM80, PCIe Gen4 x16. **Not Hopper**: `fa3` needs `is_hopper_with_cuda_12_3` (`arg_groups/model_override_base.py:323-329`) and, when pinned, FAILS here: decode CUDA-graph capture dies with `scheduler_metadata must have shape (metadata_size)` (`R8/preflight/boot_fa3/server.log:100,127`). Default MHA backend on SM80 is `flashinfer` (`model_override_base.py:347-351`; the 8B runs used it); every fp8_e5m2-KV measurement pinned `--attention-backend triton` (flashinfer + fp8_e5m2 KV was never booted here). FP8 checkpoints run as weight-only FP8 Marlin (W8A16), selected automatically, no flag (`R32/exp1_32b/server.log:15`). Campaigns 1-3 ran on an H100; campaigns 4 (8B) and 5 (32B, 70B) on this box. |
| Driver | **580.178.04 loaded**, CUDA 13.0 (nvidia-driver-580-server-open, open kernel module). No reboot needed. Secure Boot disabled (checked 2026-09-16). |
| Docker | docker-ce 29.8.1, **`Live Restore Enabled: false`** (a dockerd restart kills both containers). NVIDIA Container Toolkit 1.20.0 (`/etc/docker/daemon.json` registers the `nvidia` runtime; `docker info` -> `Runtimes: io.containerd.runc.v2 nvidia runc`). Group `docker` is live in new login shells (`id -nG` lists it): plain `docker` works. `docker.service` `LimitMEMLOCK=8388608` (8 MiB). |
| Containers | Two, RestartPolicy `no` (Exited after a VM stop: `docker start` only after the §2.2 gate), identical flags: `--gpus all --ipc=host --network=host --privileged --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576`, binds repo -> `/sgl-workspace/sglang`, HF cache -> `/root/.cache/huggingface`, `/mnt/nvme:/mnt/nvme:rshared`, cmd `/bin/zsh`. **`sglang_hicache`** = `lmsysorg/sglang:nightly-dev-20260907-30705c00`: sglang-kernel 0.4.6.post1 (= this checkout's pin), flashinfer-python 0.6.18, torch 2.13.0+cu130, nixl 1.4.1, triton 3.7.1, transformers 5.12.1, datasets 5.0.1, pyarrow 25.0.1, pandas 3.0.5; ran every 2026-09-17 measurement and holds the warm JIT cache (`/root/.cache/sglang`; survives `docker stop/start`, lost on `docker rm`): **USE THIS ONE**. `sglang_dev` = `lmsysorg/sglang:dev` (built 2026-09-17 00:53Z): sglang-kernel 0.4.7 (newer than the pin), never booted a server: spare only. Both: Python 3.12.3, import `/sgl-workspace/sglang/python/sglang`, have `iostat` and `/opt/sglang/bin/{py-spy,hf,uv}`; no `fio`, no `/opt/aiperf`; **`/home/wanhr/data` is NOT bound** into either. (A 57 GB `vsc-sglang-*` devcontainer image is also on disk, no container.) |
| Host | Ubuntu 26.04.1, kernel 7.0.0-1011-gcp, 12 vCPU, 167 GB RAM (idle: 160 GB available), **no swap**, Python 3.14.4 only (no pip/venv/uv/hf/py-spy); **`fio` and `iostat` (sysstat) are installed**. `/tmp` and `/dev/shm` are 84 GB tmpfs. No `nvidia_fs` module (no GDS). |
| Disks | `/` ext4 on PD-SSD, 741 GB free (3 images + 86 GB of models). **`/dev/nvme0n1` 375 GiB local SSD: ext4, label `localssd`, LBA 4096, mounted `/mnt/nvme` (`rw,noatime,discard`), 368.0 GiB (395,188,764,672 B); `/mnt/nvme/hicache_l3` exists, root-owned, empty**; by-id `/dev/disk/by-id/google-local-nvme-ssd-0`. `/etc/fstab:4` HAS the by-id `nofail` line (`cat -n /etc/fstab`, mtime 2026-09-17 16:41): never append it again. **Ephemeral**: wiped at every VM stop/start, i.e. every morning (§2.2). Device ceilings, measured (iostat, 5 s samples, `R32/exp1_32b/iostat.log`; no fio output was saved): 704 MiB/s = ~0.68 GiB/s read at 97 % util, 391 MiB/s = ~0.38 GiB/s write at 99 % util; nixl L3 delivers 0.62-0.63 GiB/s for every model (`R32/REPORT.md:47-49`). |
| Repo | `/home/wanhr/sglang` @ `6ec32e6b7`, upstream ~2026-09-07. Uncommitted: `scripts/setup-gpu-docker.sh`, 12 files under `hicache_eval/scripts/` (env knobs, §11), `hicache_eval/.current_results`, `HANDOFF.md`; untracked: `agent_cache/` (this file + the starter kit only), `hicache_eval/.frozen_results`, the `R32/` and `R8/` dirs, 4 new scripts. Nothing modified under `python/`. `hicache_eval/.current_results` points at a frozen campaign: every hicache_eval driver refuses to run until `RESULTS` points at a new dir (`hicache_eval/scripts/env.sh:6-12`). |
| HF cache | `~/.cache/huggingface/hub`: **`Qwen/Qwen3-32B-FP8` 32G (snapshot `aa55da1ecc13...`, the evaluation model)**, `Qwen/Qwen3-8B` 16G (`b968826d`), `casperhansen/llama-3.3-70b-instruct-awq` 38G (`64d25562`). The 8B and 32B-FP8 tokenizer files (tokenizer.json, tokenizer_config.json incl. chat template, vocab.json, merges.txt) are byte-identical (sha256 of both snapshots). |

**A100 constants are measured** (`R32/{REPORT,COMPARISON}.md`; table in §5.4). Qwen3-32B-FP8, fp8_e5m2 KV, triton, mem-fraction 0.85: b = 131,072 B/token,
P = 1,226 tok/s (H100 2,336: 0.53x; part of it is the weight kernel, W8A16 Marlin here vs DeepGEMM W8A8 there, `R32/DEVIATIONS.md` D4), bar 0.150 GiB/s, device pool
281,216 tokens (H100 284,224: `hicache_eval/scripts/models.sh:21` still holds the H100 value), L3 0.623 GiB/s = 4.2x the bar, L3 hit faster than recompute at 7 of 7
lengths. **Negative case:** Qwen3-8B here has P 8,858 tok/s, bar 1.22 GiB/s, L3 0.625 GiB/s = 0.51x the bar: an L3 hit loses at 7 of 7 lengths (`R8/COMPARISON.md:123-133`),
so the 8B cannot show an L3 benefit on this box; that is why the 32B is the evaluation model. Llama-70B-AWQ: P 820, bar 0.125, L3 0.629 (5.0x).

---

## 2. Bring-up (§2.1 done 2026-09-16; §2.2-§2.4 (bar §2.4's results-stamp lines) and §2.6 done 2026-09-17; §2.5 AIPerf venv NOT done; §2.2 is repeated after every VM stop/start)

### 2.1 Container toolkit (DONE; re-run is a no-op)

```bash
cd /home/wanhr/sglang && ./scripts/setup-gpu-docker.sh --skip-driver --skip-docker    # idempotent; toolkit stage completed 2026-09-16 18:15 UTC
# expect: "Already installed: NVIDIA Container Toolkit CLI version 1.20.0" (:478-479), "Docker already has the 'nvidia' runtime registered" (:455-457); NO apt repo added, NO dockerd restart, NO reboot request (:266-271)
```
Evidence: toolkit stage `setup-gpu-docker.sh:415-490`. `configure_docker_runtime` restarts dockerd (`:459-466`) only when `daemon.json` lacks the `nvidia` runtime (it has it); with
live-restore off a restart kills both containers, so never re-run this after a hand edit of `daemon.json` or while a server is up. The printed "Next" `docker run` hint is right only with
the **uncommitted** `setup-gpu-docker.sh:179` edit (commit it); use §2.3 anyway. The `docker` group is live in new login shells (`id -nG | tr ' ' '\n' | grep -x docker`).

### 2.2 Local NVMe for L3 (done 2026-09-17; **REDO EVERY MORNING**: the SSD is blank after every VM stop/start)

```bash
# host, after EVERY VM start and BEFORE any docker start (both containers have RestartPolicy=no, so they are Exited)
DEV=/dev/disk/by-id/google-local-nvme-ssd-0
findmnt /mnt/nvme >/dev/null || {
  sudo blkid $DEV >/dev/null || sudo mkfs.ext4 -F -m 0 -L localssd -E lazy_itable_init=0,lazy_journal_init=0,discard $DEV   # mkfs ONLY when blkid finds no filesystem
  sudo mkdir -p /mnt/nvme && sudo mount -o discard,defaults,noatime $DEV /mnt/nvme; }
sudo mkdir -p /mnt/nvme/hicache_l3 && sudo chown wanhr:wanhr /mnt/nvme
findmnt /mnt/nvme && df -h /mnt/nvme                      # expect /dev/nvme0n1, ext4, rw,noatime,discard, 369G
# HARD GATE as a condition of the start (no `exit`: pasted into a login shell it would close the SSH session); expect /dev/nvme0n1 / unlimited
if [ "$(findmnt -n -o SOURCE /mnt/nvme)" = /dev/nvme0n1 ]; then docker start sglang_hicache && docker exec sglang_hicache bash -c 'findmnt -n -o SOURCE -T /mnt/nvme/hicache_l3; ulimit -l'
else echo 'NVMe NOT MOUNTED - redo 2.2; container NOT started'; fi
```
The mkfs / mount / mkdir / chown lines are the 2026-09-16 ones (today's label, mount options and owner match them); the `findmnt`/`blkid` guard around them is new
and not yet exercised after a wipe. **fstab:** `/etc/fstab:4` already holds the by-id line (`... /mnt/nvme ext4 discard,defaults,noatime,nofail 0 2`; verified with
`cat -n /etc/fstab`, although an earlier status note said it had not been added): never `tee -a` it again. It helps only across a plain reboot; after a stop/start (or a
host-maintenance restart: `automaticRestart=TRUE`, `onHostMaintenance=TERMINATE`) the SSD has no filesystem, the `nofail` mount fails silently, `/mnt/nvme` is an empty
dir on the root PD, a container started then bind-mounts it and nixl `os.makedirs` the store there (`nixl_utils.py:229-231`): every "L3" write lands on `/dev/sda` with no
error. Hence the hard gate; reusable form: `gate()` in `hicache_eval/scripts/run_a100_rerun_c2.sh:48-54` (also requires >= 110 GiB RAM available, stops a stale server).
No chown of `hicache_l3`: the server (root in the container) pre-creates 256 root-owned bucket dirs `00..ff` and writes 0o644 files in them
(`nixl_utils.py:257-266,277,298-303`; `nixl_routing.py:5-6`), so the L3 wipe runs as root (§7.2 step 1).
NVMe LBA is 4 KiB (`lsblk -o LOG-SEC` = 4096): O_DIRECT needs every nixl FILE to be a multiple of 4096, and nixl writes K and V as separate files (2 per page).
Qwen3-32B-FP8 + fp8_e5m2 KV, page 64: 64 x 131,072 = 8,388,608 B per page = 2 files x 4,194,304 B (1024 x 4096): fine, and measured (4,096 tokens -> 128 files,
536,870,912 B, `R32/exp0_32b/exp0_results.json` step2_backup; log `O_DIRECT is active ... (POSIX)`). The code falls back to buffered I/O only when `O_DIRECT` is absent,
never on EINVAL (`nixl_utils.py:286-296`), so any other model/page size passes that check or sets `SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0`. HF cache and results stay on `/`. Never `mkfs` `/dev/sda*`.

### 2.3 Containers (DONE: do not re-run `docker run`, the names are taken; daily = `docker start sglang_hicache` after the §2.2 gate)

**Use `sglang_hicache`**: its compiled deps match this checkout's pins (sglang-kernel 0.4.6.post1, `docker/Dockerfile:7`) and it ran every
2026-09-17 measurement. `sglang_dev` (sglang-kernel 0.4.7) never booted a server with this checkout: unproven, not comparable. Never recreate the
container between measured stages: a new one pays a 10.6 min JIT-cold boot and a one-time 9.0 s first long prefill (`R8/DEVIATIONS.md` D9). How it was created:
```bash
mkdir -p "$HOME/.cache/huggingface"                       # else dockerd creates it root-owned (dflash RUNBOOK 2.2)
docker run -itd --name sglang_hicache --gpus all --ipc=host --network=host --privileged --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
  -v /home/wanhr/sglang:/sgl-workspace/sglang -v /home/wanhr/.cache/huggingface:/root/.cache/huggingface -v /mnt/nvme:/mnt/nvme:rshared \
  lmsysorg/sglang:nightly-dev-20260907-30705c00 /bin/zsh   # sglang_dev: same line with lmsysorg/sglang:dev
```
`/home/wanhr/data` is not bound: copy what a container must read under the repo bind (`agent_cache/traces/`, §3). `--ulimit memlock=-1:-1` is mandatory: the daemon's
8 MiB `LimitMEMLOCK` is inherited and the pinned host pool (`cudaHostRegister`) fails otherwise (`hicache_eval/hicache_eval_plan.md:55-62`). `--ipc=host` shares the host's
84 GB `/dev/shm`, so the hint's `--shm-size 32g` is inert (dropped). `:rshared` lets a host re-mount of `/mnt/nvme` propagate into a running container (verified 2026-09-16:
`findmnt -o PROPAGATION /` is `shared` and `docker run -v <dir>:/x:rshared ubuntu:24.04` starts); the §2.2 gate remains the defence. The image's editable install is
shadowed by the bind mount (`docker/Dockerfile:639-646`, WORKDIR `:685`).

### 2.4 In-container checks (passed in `sglang_hicache` on 2026-09-17; the LAST TWO lines, results stamp + `versions.txt`, are NOT run yet: run them before §5.2; re-run the `ulimit`/`findmnt` line after every daily `docker start`)

```bash
docker exec sglang_hicache nvidia-smi -L
docker exec sglang_hicache python3 -c 'import sglang, torch; print(sglang.__file__, torch.version.cuda, torch.cuda.get_device_name(0))'
#   -> /sgl-workspace/sglang/python/sglang/__init__.py 13.0 NVIDIA A100-SXM4-80GB   (path MUST be under /sgl-workspace)
docker exec sglang_hicache python3 -c 'from nixl._api import nixl_agent, nixl_agent_config; a=nixl_agent("probe", nixl_agent_config(backends=[])); print(a.get_plugin_list())'
#   -> list containing POSIX (hicache_nixl.py:29; nixl_utils.py:143). GDS/GDS_MT unusable: no nvidia_fs module here.
docker exec sglang_hicache bash -c 'ulimit -l; findmnt -n -o SOURCE -T /mnt/nvme/hicache_l3; df -h /dev/shm | tail -1; free -g | head -2'   # unlimited / /dev/nvme0n1 / 84G / Mem: 167
docker exec sglang_hicache git config --global --add safe.directory /sgl-workspace/sglang       # already set
docker exec sglang_hicache git -C /sgl-workspace/sglang status --short                # no "dubious ownership"
AC=/home/wanhr/sglang/agent_cache; [ -f $AC/.current_results ] || date -u +%Y%m%d_%H%M > $AC/.current_results; mkdir -p $AC/results/$(cat $AC/.current_results)   # NOT run yet: neither exists (§11); absolute paths, the block has no cd
docker exec sglang_hicache pip list 2>/dev/null | grep -Ei '^sgl|^sglang|flashinfer|^torch |nixl|^triton |transformers' | tee $AC/results/$(cat $AC/.current_results)/versions.txt
```
The package is named `sglang-kernel` (the old `sgl-kernel` grep matched nothing). Expected in `sglang_hicache`: sglang-kernel 0.4.6.post1, flashinfer-python 0.6.18, torch 2.13.0+cu130,
nixl 1.4.1, triton 3.7.1, transformers 5.12.1 = the pins (`docker/Dockerfile:7,17`; `python/pyproject.toml:6`) and `R32/versions.txt`. The bind mount shadows only the `sglang` package;
compiled deps come from the image, which is why `sglang_dev` (sglang-kernel 0.4.7, sgl-deep-gemm 0.2.0) is not used. `sglang.__version__` differs per container (image metadata only).

### 2.5 Python extras (only the AIPerf venv is missing)

```bash
docker exec sglang_hicache bash -c 'python3 -m venv /opt/aiperf && /opt/aiperf/bin/pip install -q -U pip aiperf && /opt/aiperf/bin/aiperf --help | head -20'
```
Already in `sglang_hicache`: datasets 5.0.1, pyarrow 25.0.1, pandas 3.0.5, huggingface_hub 1.30.0 with `hf`, py-spy, `iostat`, and `uv` (`/opt/sglang/bin/uv`; the earlier "image has no
uv" was wrong). `/opt/aiperf` lives in the container's writable layer (survives stop/start, lost on `docker rm`). `aiperf` 0.12.0 declares `requires_python <3.14,>=3.11` (PyPI): not
installable on the host's 3.14, and isolated because its `~=` pins would downgrade image packages. Host venv for CPU-only work: §3.5. `huggingface-cli` is a dead shim; use `hf download`.

### 2.6 Models (DONE: all three are in the HF cache, nothing to download)

```bash
docker exec sglang_hicache bash -c 'hf download Qwen/Qwen3-32B-FP8'     # only if the cache is lost; 32G on disk, snapshot aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
sudo chown -R wanhr:wanhr "$HOME/.cache/huggingface"; du -sh "$HOME"/.cache/huggingface/hub/models--*
```
Primary = `Qwen/Qwen3-32B-FP8`; Qwen3-8B (16G) only for smoke tests (negative case, §1); the 70B AWQ (38G) optional. Qwen3-32B-FP8 +
`--kv-cache-dtype fp8_e5m2 --attention-backend triton` is **verified on this A100** (`R32/preflight/boot_32b/checks.txt`, `R32/DEVIATIONS.md` D4, D6):
weight-only FP8 Marlin (automatic), coherent generations, tier round trip 4032/4032/4032 cached tokens (device/host/storage). Boot, measured:
~633 s (10.6 min) JIT-cold (first boot in a new container; prefill CUDA-graph capture 488 s while Marlin/Triton kernels compile), 278-285 s warm with a 100 GB
host pool (first log line to `fired up`; `tokenizer_e2e` 277-284 s; weights 83-91 s, prefill capture ~100 s, decode capture 12-13 s, host-pool pinning 52-64 s; `Engine startup timings` in `R32/*/server.log`).

---

## 3. Data

Raw downloads go to `/home/wanhr/data/` (root disk, outside git); converted traces to
`agent_cache/traces/` (git-tracked JSON; parquet ignored). Budget: LMCache 2.37 GB + AgentX-256k 0.57 GB
+ AgentX full 1.85 GB + Mooncake 13 MB + converted JSON (65 MiB); models (86 GB) are already on disk, `/` has 741 GB free.
Nothing here has been started: `/home/wanhr/data` does not exist yet and is **not bind-mounted** into either container, so
anything a container must read (converted trace, Mooncake jsonl) is copied under `agent_cache/traces/` (repo bind) or `docker cp`'d.

### 3.1 Downloads (host, plain wget; no HF CLI needed)

```bash
mkdir -p /home/wanhr/data/{lmcache,mooncake,agentx}
cd /home/wanhr/data/lmcache && for i in 0 1 2 3 4; do wget -c "https://huggingface.co/datasets/sammshen/lmcache-agentic-traces/resolve/main/data/train-0000${i}-of-00005.parquet"; done && wget -c https://huggingface.co/datasets/sammshen/lmcache-agentic-traces/raw/main/README.md
cd /home/wanhr/data/mooncake && B=https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release && wget -c $B/traces/toolagent_trace.jsonl $B/traces/conversation_trace.jsonl $B/traces/synthetic_trace.jsonl && wget -c -O mooncake_trace.jsonl $B/arxiv-trace/mooncake_trace.jsonl
cd /home/wanhr/data/agentx && wget -c -O traces-062126-256k.jsonl https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/resolve/main/traces.jsonl
[ "$(stat -c %s traces-062126-256k.jsonl)" -eq 568864747 ] || echo 'AgentX TRUNCATED (HF Content-Length 568864747): re-run wget before 3.5'
```
Mooncake URLs match what `bench_serving` auto-downloads (`python/sglang/benchmark/datasets/common.py:13-18`);
its default cache path is `/tmp/<workload>_trace.jsonl` (`datasets/mooncake.py:33-36`) = tmpfs here, so always pass `--dataset-path`.

### 3.2 What the `agentic-trace` loader actually reads (this checkout)

Evidence: `python/sglang/benchmark/datasets/agentic_trace.py:70-100`.
- `data["conversations"]` (list; empty -> ValueError). `metadata` never read.
- per turn: only `turn["messages"]` (falsy -> turn silently dropped, `:85`); `conversation[0]["prompt_tokens"]` informational (`:92`).
- **Every other per-turn key (`pre_gap`, `output_length`) is ignored** today; `output_len` is one value per
  conversation = `--sharegpt-output-len` or 220 (`:78,98`). Rotation `--dataset-offset`, cap `--agentic-max-turns` (`:74-87`).
- At request time messages are rebuilt as `{role, content}` only (`serving.py:1294`): `tool_calls`,
  `tool_call_id`, `name` are stripped. Multi-turn is detected by `prompt[0]` being a str or a
  `List[{role,content}]` (`serving.py:1377-1381`), so per-turn metadata **cannot** ride inside `prompt`.

### 3.3 Converter: LMCache parquet -> agentic-trace JSON (`agent_cache/scripts/convert_lmcache.py`, BUILT and run 2026-09-18)

Output: `agent_cache/traces/lmcache_agentic_trace.json` (65 MiB, one JSON document) + `.stats.json`. Run inside `sglang_hicache`
(pyarrow, transformers, tokenizer files): `cd /sgl-workspace/sglang/agent_cache/scripts && python3 convert_lmcache.py --parquet-dir /tmp/lmcache --workers 12`
(13.5 min: the two token counts below dominate; then `sudo chown -R wanhr:wanhr agent_cache` on the host). Verified against an independent
re-derivation and loaded with the checkout's `AgenticTraceDataset` + `_normalize_round_messages` (731 conversations, 0 rounds rejected).

**What the parquet really is** (measured, not the dataset card): 24,880 rows = LLM calls, 767 session_ids (665 swebench / 94 gaia / 8 wildclaw;
the card's 669/85/10 is wrong), rows contiguous per session, models minimax-m2.5 (18,540 rows), claude-sonnet-4-6, deepseek-v3.1, claude-opus-4-6.
Each row's `input` is the full cumulative message list of that call. One session_id (`wildclaw__..._arxiv_digest__claude`) bundles two independent
runs with different system prompts; sorting by history length interleaved them in the first version of the converter (caught by review). Tool results
carry `tool_call_id` but no name; OpenHands (minimax) sessions put tool results in plain user messages, the claude/deepseek/gaia/wildclaw ones use
`role: tool`. 101 swebench sessions start mid-run (their first recorded call already holds assistant replies), and 1,164 later turns are harness
re-prompts with no assistant reply in between. No content is null.

**Algorithm as built:** rows of a session_id are split into append-only *chains* (a row joins the chain whose last input it extends, longest match first;
otherwise it starts a chain: 769 chains, the extra two being the second run above and its one-row compaction). Each chain is one replayable session
(`<session_id>__chain<k>` when several). Per row the delta `input[len(prev):]` is emitted without assistant messages (the server regenerates them,
`serving.py:1329-1331`); tool messages become `{"role":"user","content":"[TOOL RESULT <function name>]\n<content>"}` (`--tool-role-mode as_user`,
default; the name comes from the preceding assistant tool_call with that id) or stay `role: tool` (`keep`). A row whose delta has no non-assistant
message is merged forward (0 such rows after the chain split). Where the two runs of the split session_id interleave, the dataset clamps
`pre_gap` to 0.0 (6 turns): unknown, not zero. Two token counts per turn with the Qwen3 chat template (`--tokenizer Qwen/Qwen3-32B-FP8`;
tokenizer files are byte-identical to the 8B's): `prompt_tokens` = the ORIGINAL cumulative input, and `replay_prompt_tokens` = the history the client
will actually send (kept messages + one placeholder reply of `output_length` tokens per earlier turn). They differ because the template strips the
`<think>` block of historical replies while the client re-feeds `reasoning_content + content` verbatim: replayed histories run **4.7 % longer** on
average (up to about 35 % shorter for mid-run starts). **Truncation uses the replay count:** cut at the first turn with
`replay_prompt_tokens + output_length > --max-context (32768) - --margin (512)`; emit chains with >= `--min-turns` (5) turns. `--sources`,
`--max-sessions`, `--seed` (shuffle) select; `--revision` is recorded in metadata.

Per-turn fields: `messages` (the only key the loader reads), `prompt_tokens`, `replay_prompt_tokens`, `output_length`, `pre_gap` (0.0 on turn 0),
`iteration`, `source_rows`, `n_assistant_in_delta`, `merged_output_length`, `orig_model`, `source`. Top level: `metadata`, `session_ids`, `sessions`
(per-chain meta: n_rows, n_turns, truncated_at, midrun_start, final counts), `conversations`.

**Result (`.stats.json`):** 731 chains emitted (38 below 5 turns), 17,887 turns. Truncated at 32K: 396 of 630 emitted swebench chains (6,902 turns
dropped), 19 of 92 gaia, 2 of 9 wildclaw; swebench final contexts p50 33.6K, max 84K. Emitted `replay_prompt_tokens`: mean 19.7K, p50 20.5K,
p95 30.7K overall; swebench 20.6K mean, gaia 9.0K. Raw `pre_gap` over all rows but the first of each session reproduces the README exactly
(p50 0.708 s, mean 2.08 s, p95 3.72 s); over emitted turns p50 0.70 / mean 1.49 / p95 3.59 (truncation drops late turns). Time-weighted: turns with a
gap >= 1 / 5 / 30 s are 20 % / 3.2 % / 0.5 % of turns but hold 80 % / 63 % / 53 % of the idle context-token-seconds; the >= 30 s share sits in about
90 turns (max gap 1,511 s), so report the tail with and without the top 1 %. **Use 20.6K (swebench) or 19.7K (all) as the live context per session in
§5.3, not 25K**, and recompute the WS rows from `sessions[*].final_replay_prompt_tokens` if a subset is replayed.

Replay-fidelity caveats for the analysis: turn 0 of the 101 mid-run-start sessions sends system + user + up to 46 tool results with no assistant turn
between them; at each of the 1,164 re-prompt turns the replay has one extra generated reply the recording did not; every generated reply replaces a
recorded one, so tool results after turn 0 answer calls the replayed model never made. Context lengths and hit patterns stay realistic; the text does not.

### 3.4 Other replays

- **Mooncake toolagent (cross-session sharing only, not gaps):** timestamp replay happens only with
  `--backend sglang` (`serving.py:1502-1508`); rounds of a record fire as a burst with `"story"` placeholder
  replies (`mooncake.py:82-123`); prompts are `hash_id + 128 x "hi"` per 512-token block (`:84-88`), so
  KV footprint is ~1/4 of `input_length`. `--num-prompts` truncates before the timestamp sort (`mooncake.py:47,63`): pass all 23,608.
  ```bash
  python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port 30000 --model Qwen/Qwen3-32B-FP8 --dataset-name mooncake --mooncake-workload toolagent --dataset-path /sgl-workspace/sglang/agent_cache/traces/toolagent_trace.jsonl --num-prompts 23608 --mooncake-num-rounds 1 --mooncake-slowdown-factor 15 --max-concurrency 8 --output-file $OUT/mooncake_toolagent.jsonl
  ```
  (copy the jsonl under `agent_cache/traces/` first: `/home/wanhr/data` is not bound into the container; `--use-trace-timestamps` is a no-op,
  `serving.py:1514`. The trace offers 6.6 req/s (23,608 in 1 h); at P = 1,226 tok/s this server is far slower (estimate 0.3-0.5 turns/s,
  §5.3), so without `--mooncake-slowdown-factor` (`serving.py:2711`; 15 is provisional and stretches the 1 h trace to 15 h: cap by wall time) the run only measures backlog.)
- **AgentX via AIPerf (container, `/opt/aiperf/bin/aiperf`):** `aiperf profile --scenario inferencex-agentx-mvp --url http://127.0.0.1:30000 --model Qwen/Qwen3-32B-FP8 --max-context-length 32768 --endpoint-type chat --public-dataset semianalysis_cc_traces_weka_with_subagents_256k --concurrency 8 --use-server-token-count --streaming --extra-inputs ignore_eos:true --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 --benchmark-duration 1800 --random-seed 20260707 --ui simple`
  (flags per NVIDIA tutorial via data-prep reader; entry-point name and download cache unverified until `--help` in §2.5).
  AgentX median context is 142K: with a 32K-context server (`--context-length 32768`; the 32B's native limit is 40,960) only the 256k-capped corpus with
  `--max-context-length 32768` is usable; expect many truncated trees. `--concurrency 8`, not 32: a 32K recompute costs 26.7 s here (measured), so few trees complete in 1800 s. c = 8 in both commands is provisional (§5.3 saturation estimate): revisit after matrix row 1.

### 3.5 Week-1 CPU-only characterization (host; NOT started: no host `uv`, no `/home/wanhr/venv312`, no `/home/wanhr/data`)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH
uv python install 3.12 && uv venv /home/wanhr/venv312 --python 3.12
uv pip install --python /home/wanhr/venv312/bin/python numpy pandas pyarrow orjson transformers huggingface_hub
git clone https://github.com/gauravapiscean/agentic-kv-cache /home/wanhr/agentic-kv-cache && cd /home/wanhr/agentic-kv-cache && mkdir -p data \
  && ln -sf /home/wanhr/data/mooncake/toolagent_trace.jsonl data/ && ln -sf /home/wanhr/data/mooncake/conversation_trace.jsonl data/ \
  && ln -sf /home/wanhr/data/mooncake/mooncake_trace.jsonl data/mooncake_arxiv.jsonl && ln -sf /home/wanhr/data/agentx/traces-062126-256k.jsonl data/agentx.jsonl \
  && ln -sfn /home/wanhr/venv312 .venv && make data && make repro      # make setup would fail: host python3 -m venv has no ensurepip
```
`make data` skips existing `data/*.jsonl` (`fetch_data.sh` `[ -f ]`), so a truncated AgentX passes and its final bare
`python3 experiments/prep.py` (host 3.14, stdlib-only, fine) crashes on JSON parse: check the size in §3.1 first. `requirements.txt` = `numpy>=2.0`, in `venv312`.
`pre_gap` distribution (no message content loaded):
```bash
/home/wanhr/venv312/bin/python - <<'EOF'
import pyarrow.dataset as ds, numpy as np
t = ds.dataset('/home/wanhr/data/lmcache', format='parquet').to_table(columns=['session_id','model','output_length','pre_gap']).to_pandas()
t['source'] = t.session_id.str.split('__').str[0]; t['iter'] = t.groupby('session_id').cumcount(); g = t[t['iter']>0]
for k,v in g.groupby(['source','model']).pre_gap:
    print(k, len(v), 'p50/p90/p95/p99', np.percentile(v,[50,90,95,99]).round(2), 'frac>=1s', (v>=1).mean().round(3), '>=5s', (v>=5).mean().round(3), '>=30s', (v>=30).mean().round(4))
for k,v in t.groupby('source').output_length: print(k, 'output_length mean/p50/p95', round(v.mean(),1), np.percentile(v,[50,95]), 'mean pre_gap', round(g[g.source==k].pre_gap.mean(),2))
print('sessions', t.session_id.nunique(), t.groupby('source').session_id.nunique().to_dict(), 'rows', len(t), 'turns/session p50', t.groupby('session_id').size().median(), 'mean pre_gap', round(g.pre_gap.mean(),2))
EOF
```
Memory-time upper bound with the client's semantics (closed loop: c live sessions, next starts when one finishes). State the turn-time model,
`turn_time = T_pf + output_length / decode_rate`:
- `T_rec(L) = 0.0626 + 3.752e-4 L + 1.365e-8 L^2` s (derived: LSQ on the 7 measured recompute medians of §5.4, max residual 31 ms; 8.0 s at 14K, 14.0 s at 21K, 18.0 s at 25K).
  Never `prompt_tokens / P`: P = 1,226 tok/s is a marginal slope with a -1.25 s intercept.
- `T_pf = T_rec(prompt_tokens)` on turn 0 (upper variant: every turn, nothing cached); on a returning turn with a resident prefix `T_pf = T_L1(prefix) + T_rec(prompt_tokens) -
  T_rec(prefix)`, `T_L1(L) = 0.053 + 12.4e-6 L` s (measured L1 fit): ~1.4 s for +1K on 25K. **Estimate, not measured.**
- `decode_rate`: **not measured** (§5.4); until it is, report every fraction for both ends of the §5.3 estimate (raw 26-42 tok/s at batch 1, 17-28 at batch 7; or the EFFECTIVE ~10-13 at c=8,
  which already contains the other sessions' prefill stalls: never add those on top of it). `output_length`: the trace's per-turn value (the script prints its mean; 220 is only the loader default).

`fraction(g) = sum(context x pre_gap | pre_gap >= g) / sum(context x (turn_time + pre_gap))` for g = 1/5/30 s at c = 4/8/12/16 (c = 32/64/128 as an analytic row only: the 32B
saturates this GPU near c ~ 8, estimate). Also print mean(pre_gap) and the implied `f = turn_time / (turn_time + pre_gap)`: §5.3 turns them into `--agentic-gap-scale`. Absolute
token-seconds convert with b = 131,072 B/token (8,192 tokens/GiB) and are compared with the §5.3 pools. Never normalize by a pool ("pool-seconds"): a trace has no pool; report the
c at which c x mean context crosses each pool (c=32 x 25K = 800K exceeds L1+L2 of every §5.3 level, but not the 1,044,160 tokens of auto L1 + a 100 GB host pool).
Row order within a session is assumed monotone (verified for 3 rows only; the converter sorts).

---

## 4. Client patch: gap-faithful multi-turn replay

Today `wrap_multi_turn_request_func` loops rounds with no sleep (`serving.py:1311-1333`; the only
`asyncio.sleep` calls are inter-conversation pacing at `:1078,:1093`). Starter kit §3.1 is confirmed.

### 4.1 Diff-level design (7 hunks, ~50 lines; NOT applied)

```
--- python/sglang/benchmark/datasets/common.py   (DatasetRow, after line 31 `extra_request_body`)
+    turn_meta: Optional[List[Dict[str, Any]]] = None      # per-turn {pre_gap, output_len, prompt_tokens}; parallel to prompt

--- python/sglang/benchmark/datasets/agentic_trace.py   (replace lines 85-87, add kwarg at 94-100)
-            prompt = [turn["messages"] for turn in conversation if turn.get("messages")]
-            if self.max_turns:
-                prompt = prompt[: self.max_turns]
+            turns = [t for t in conversation if t.get("messages")][: self.max_turns or None]
+            prompt = [t["messages"] for t in turns]
+            turn_meta = [{"pre_gap": float(t.get("pre_gap") or 0.0), "output_len": t.get("output_length"),
+                          "prompt_tokens": int(t.get("prompt_tokens") or 0)} for t in turns]   # prompt_tokens = control length (hunk 6)
 ...            DatasetRow(prompt=prompt, prompt_len=prompt_len, output_len=output_len,
+                          turn_meta=turn_meta)

--- python/sglang/benchmark/serving.py
@@ RequestFuncInput (after line 95 `routing_key`)
+    turn_meta: Optional[List[Dict[str, Any]]] = None
@@ wrap_multi_turn_request_func.f (between line 1319 `prev_messages.extend(normalized)` and 1321 `inner_input = replace(`)
+            meta = (request_func_input.turn_meta or [{}] * len(prompts))[round_index] or {}
+            gap = float(meta.get("pre_gap") or 0.0) * GAP_SCALE          # GAP_SCALE from --agentic-gap-scale, default 1.0
+            if round_index > 0 and gap > 0:
+                await asyncio.sleep(gap)            # previous round was awaited to completion at 1324-1326 == LMCache pre_gap definition
             inner_input = replace(copy.deepcopy(request_func_input), prompt=copy.deepcopy(prev_messages),
+                                  output_len=int(meta.get("output_len") or request_func_input.output_len))
@@ benchmark() main loop RequestFuncInput(...) at 1549-1560
+            turn_meta=request.turn_meta,
   (leave the warmup test_input at 1431-1440 WITHOUT turn_meta so warmup does not sleep through ~38 gaps)
@@ wrap_multi_turn_request_func.f, hunk 6 "side-channel paired control" (§6.5), right after the sleep, when CONTROL_EVERY and round_index > 0
   and (round_index + conv_idx) % CONTROL_EVERY == 0 (the conv_idx offset spreads controls over turn indices) and meta.get("prompt_tokens", 0) > 0 (no length -> no control, never a KeyError):
+            ctl = asyncio.create_task(request_func(replace(inner_input, prompt=salted_random_chat(n_tokens=meta["prompt_tokens"],
+                       seed=SEED ^ conv_idx ^ round_index ^ RC_SALT), output_len=1)))   # TTFT is all that is compared; NOT via prev_messages, NOT via the semaphore
             ... after the round's await: control_outputs[(conv_idx, round_index)] = await ctl
   (in-slot is impossible: every round extends prev_messages and appends the reply, 1319/1329-1331; RC_SALT per process as exp2.py:28-30)
@@ result_details at 1883-1896: + "start_times" (o.start_time, set at 467, never dumped today), "conv_idx", "round_idx", "control_ttfts", "control_details" keyed (conv, round)
@@ cli_main (near --agentic-max-turns, 2267-2274): --agentic-gap-scale (float, 1.0); --agentic-control-every K (int, 0 = off)
```
The control adds 1/K single-shot requests per turn (report it). With the 32B it is not cheap: each is a full recompute of the turn's `prompt_tokens` (18.0 s of GPU at 25K) and, on a
`write_through` arm, 3.05 GiB to L2 and L3, so K comes from the §5.3 / §7.3 budgets (provisional K = 40 at c <= 8), the same K for every arm of a row; K = 1 only in the §4.4 dry run.
Assert `storage == 0 and host == 0 and device <= 64` on every control row (HANDOFF §3).
Per-turn output length: `output_len` is sent as `max_completion_tokens` (`serving.py:435`) with
`ignore_eos = not args.disable_ignore_eos` (`:443-446`, default True), so each turn generates exactly
`output_len` tokens: deterministic KV growth, good for a systems study; pass `--disable-ignore-eos` only
if you want EOS-faithful replies (then `output_len` in metrics is right only if a usage chunk arrives, `:517-519`).

### 4.2 Per-turn cached-token capture (already there; one optional hunk)

With client `--cache-report` the chat func reads `sglext.cached_tokens_details` on every stream
chunk (`serving.py:245-256, 521-522`) because `--cache-report` injects `return_cached_tokens_details: true`
(`:1990-1995`) and the server emits an `sglext` chunk with `choices=[]` (`entrypoints/openai/serving_chat.py:1892-1900`).
Per-turn `cached_tokens` / `cached_tokens_details` reach the JSONL only with `--output-details` (`:1893-1903`).
**Gotcha:** `output.prompt_len` for every round is turn-0 `prompt_tokens` (`:119-121` + deepcopy at `:1322`),
so the printed hit rate / `result["cache_report"]` denominator is wrong for multi-turn (`:1737-1760`).
Optional hunk for the true per-turn denominator: add `usage_prompt_tokens: int = 0` to `RequestFuncOutput`
(after line 112) and after line 519 `u = data.get("usage") or {}; output.usage_prompt_tokens = u.get("prompt_tokens", output.usage_prompt_tokens)`
(mirror after `:484` non-streaming); needs a usage chunk: server `--stream-response-default-include-usage`
(`arg_groups/fields/serving.py:255-258`) or client `--extra-request-body '{"stream_options":{"include_usage":true}}'` (merged into the chat payload at `serving.py:450`; `entrypoints/openai/utils.py:92-106`).

### 4.3 Where the patch lives, how it is applied and reverted

```bash
# author once, then archive (HANDOFF §7: never leave an eval patch under python/)
cd /home/wanhr/sglang && git diff -- python/sglang/benchmark > agent_cache/patches/0001-agentic-trace-pre-gap.patch && git checkout -- python/
# per session, inside the container (both containers and the host share the same bind-mounted repo: applied once = applied everywhere):
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang && git apply --check agent_cache/patches/0001-agentic-trace-pre-gap.patch && git apply agent_cache/patches/0001-agentic-trace-pre-gap.patch'
# before every commit, on the host:
cd /home/wanhr/sglang && git checkout -- python/ && git status --short | grep '^ M python/' && echo 'PATCH STILL APPLIED' || echo clean
```
Also copy the applied diff into `results/<stamp>/patches/` (precedent: `hicache_eval/results/20260908_nixl_exp234/exp2/C2_backup_skip.patch`).

### 4.4 Validating the patch (2-conversation dry run with a visible gap)

```bash
python3 - <<'EOF'                                            # inside container: make a 2-conv, 3-turn trace with 5 s gaps
import json; sys="You are a helpful assistant."
conv=lambda tag:[{"messages":[{"role":"system","content":sys},{"role":"user","content":f"{tag} turn0 "+"lorem "*300}],"prompt_tokens":0,"pre_gap":0.0,"output_length":16},
                 {"messages":[{"role":"user","content":f"{tag} turn1 "+"ipsum "*300}],"prompt_tokens":700,"pre_gap":5.0,"output_length":16},    # approximate lengths: they only
                 {"messages":[{"role":"user","content":f"{tag} turn2 "+"dolor "*300}],"prompt_tokens":1100,"pre_gap":5.0,"output_length":16}]   # size the controls (hunk 6)
json.dump({"metadata":{},"conversations":[conv("A"),conv("B")]},open("/tmp/gaptest.json","w"))
EOF
python3 -m sglang.benchmark.serving --backend sglang-oai-chat --host 127.0.0.1 --port 30000 --model Qwen/Qwen3-32B-FP8 \
  --dataset-name agentic-trace --dataset-path /tmp/gaptest.json --num-prompts 2 --max-concurrency 2 --warmup-requests 0 \
  --agentic-control-every 1 --cache-report --output-details --output-file /tmp/gaptest.jsonl
```
Send one discarded long probe first (the first long prefill in a fresh container took 9.0 s). Pass criteria: wall time >= 10 s (two 5 s gaps per conversation, conversations
concurrent); the server log shows 6 conversation requests with ~5 s spacing between rounds of the same conversation plus 4 controls; the JSONL has 6 entries in
`ttfts`/`cached_tokens_details`/`start_times` (conversation-major, round-minor: the flatten at `serving.py:1567-1569`), rounds 1-2 show `device > 0`, the 4 `control_details` rows show
`storage == 0, host == 0, device <= 64`; `wall(scale 1) - wall(--agentic-gap-scale 0)` = 10 s +/- 1 s (relative: the old absolute "< 3 s" was an 8B guess; here a ~500-token recompute is
0.24-0.46 s measured and the decode rate is unmeasured). Also confirm the `#Conversations: 2 (... turns/conv min=3 max=3 ...)` load line (`agentic_trace.py:107-112`). Extra check (§5.2
history note): on rounds 1-2 `device` should be ~floor64(previous round's prompt tokens), not ~prompt - 64, if the regenerated reply is indeed never a cache hit. The client sends
`temperature: 0.0` by default (`serving.py:439-441`), overriding the model's generation_config sampling, so replies are greedy and comparable across arms.

---

## 5. Server configurations

### 5.1 Verified flag table (this checkout)

| flag | default | choices / notes | evidence |
|---|---|---|---|
| `--radix-eviction-policy` | `lru` | `lru, lfu, slru, priority`. **`fifo` is NOT a CLI choice** (factory has it, argparse rejects it). `--radix-eviction-policy-config` JSON kwargs, only `slru` reads `protected_threshold`. `--disable-radix-cache` is exclusive with hierarchical cache | `arg_groups/choices.py:188`; `mem_cache/utils.py:59-67`; `fields/memory.py:46-60`; `kv_cache_hook.py:168-172` |
| `--enable-hierarchical-cache` | False | builds `UnifiedRadixCache` + `init_hicache` (**not** `HiRadixCache`, which only tests instantiate) | `mem_cache/registry.py:190-198`; `test/registered/unit/mem_cache/test_hiradix_cache_unit.py:76` |
| `--hicache-ratio` | None -> 2.0 (cache mode) | host pool = device tokens x ratio | `fields/memory.py:104-107`; `hicache_hook.py:62-81` |
| `--hicache-size` | 0 | int, decimal GB; **overrides ratio when > 0**; raises if > host budget. Tokens = `(int(GB x 1e9 // b) // page + 1) x page`: at b = 131,072, 100 -> 762,944 (**measured**; the rule reproduces it), 64 -> 488,320, 48 -> 366,272, 32 -> 244,160, 18 -> 137,344 (derived). Default ratio 2.0 would be int(2 x 281,216) = 562,432 -> 562,496 tok after the same +1-page alignment = 73.7 GB (derived). A host pool <= device pool only logs `L2 cache effectiveness is reduced` (`:163-171`) | `pool_host/base.py:150-180`; `fields/memory.py:108-111`; `R32/exp1_32b/startup_facts.txt:5` |
| `--hicache-write-policy` | `write_through` | `write_back, write_through, write_through_selective` | `fields/memory.py:112-118` |
| `--hicache-io-backend` / `--hicache-mem-layout` | `kernel` / `page_first` | `page_first_direct`+`kernel` is silently rewritten to `direct` (starter kit §3.2 combo does not run as kernel); keep `kernel`+`page_first` | `hicache_hook.py:127-150` |
| `--hicache-storage-backend` | None | `file, sim, mooncake, hf3fs, nixl, aibrix, dynamic, eic, simm, mori, shm` | `fields/memory.py:139-157` |
| `--hicache-storage-prefetch-policy` | `timeout` | `best_effort, wait_complete, timeout` | `fields/memory.py:158-164`; `unified_radix_cache.py:1871-1893` |
| `--hicache-storage-backend-extra-config` | None | JSON or `@file`. **No `--hicache-storage-prefetch-timeout` flag exists.** Live-path keys: `prefetch_threshold` (256), `prefetch_timeout_base` (**1.0** s), `prefetch_timeout_per_ki_token` (**0.25**), `hicache_storage_pass_prefix_keys`; timeout = base + pages x (page_size/1024 x per_ki_token), **no max cap** (the 2.0/0.1/30 `PrefetchTimeoutConfig` in `hicache_storage.py:50-55` is the dead HiRadixCache path). **32B on this box (derived):** the default budget is 1.0 s + 0.25 s per 1,024 tokens = 4,096 tok/s; measured single-stream L3 delivery is 0.623 GiB/s = 5,104 tok/s with the SSD already at its read ceiling, so ONE idle 25K restore fits (~5.0 s needed, 7.1 s allowed) but two overlapping L3 restores share the device and hit the cut-off (partial hit + recompute of the rest). Every L3 number measured here used `wait_complete`; `timeout` is unmeasured. Record `storage_prefetch_unfulfilled_tokens_total{reason}`; if the timeout reason dominates, raise `prefetch_timeout_per_ki_token` (e.g. 1.0) in `l3_extra.json` (§5.2) for the whole row | `hybrid_cache/hybrid_cache_controller.py:181-215`; `unified_cache/storage_attachment.py:241-245`; `unified_radix_cache.py:1864-1869`; `R32/COMPARISON.md:146-147` |
| `--hicache-storage-prefetch-retry-poll-interval` / `-max-attempts` | 0 / 4 | HANDOFF §2 suspects this path for `timeout` > `wait_complete` | `fields/memory.py:169-183` |
| `--page-size` | None -> 1 on CUDA | use 64 (nixl README example; storage hits truncated to page multiples) | `overrides.py:1453-1476` |
| `--max-total-tokens` | None | upper bound: `min(profiled, user)`, floored to the page size; the memory-pressure knob. A value above the profiled pool is ignored with `max_total_tokens=... is larger than the profiled value`: grep for it, it means the pin did not take | `kv_cache_configurator.py:2153-2168` |
| `--mem-fraction-static` | None | auto `(gpu_mem - reserved)/gpu_mem`, reserved >= 10 GiB on >60 GB GPUs. **Pass 0.85**: every pool measured on this box (32B: 281,216 tok, weights 32.59 GB, `available_gpu_mem=7.66 GB` left) used it; the auto value would give a different, unmeasured pool | `memory_hook.py:246-293`; `R32/exp1_32b/server_args.txt`; `R32/exp1_32b/startup_facts.txt:4` |
| `--kv-cache-dtype` | `auto` (bf16) | **`fp8_e5m2` for the 32B**: 1 byte/element, b = 2 x 64 layers x 8 KV heads x 128 = 131,072 B/token (bf16 would be 262,144); measured 536,870,912 B / 4,096 tok. Every byte-derived number (pools, O_DIRECT file size, bar) depends on it | `arg_groups/fields/model.py:196-216`; `R32/exp0_32b/exp0_results.json` step2_backup; server.log `KV Cache is allocated. dtype: torch.float8_e5m2, #tokens: 281216, K size: 17.17 GB, V size: 17.17 GB` |
| `--attention-backend` | None -> `flashinfer` on SM80 MHA | **pin `triton`** with fp8_e5m2 KV. The only automatic rewrite is `fa3` + fp8_e5m2 -> `triton`; SM80 never resolves to `fa3`, so without the flag this box would run flashinfer + fp8_e5m2, **never booted here**: b, P, the pool and every TTFT in §5.4 hold only for triton. `fa3` itself fails at decode CUDA-graph capture on this GPU | `arg_groups/overrides.py:1362`; `model_override_base.py:323-329,347-351`; `R32/DEVIATIONS.md` D3; `R8/preflight/boot_fa3/server.log:100,127` |
| FP8 weights on SM80 (no flag) | auto | weight-only FP8 Marlin (W8A16, bf16 activations) is enabled automatically when the GPU lacks native FP8 (`can_auto_enable_marlin_fp8()`; env `SGLANG_FORCE_FP8_MARLIN` forces it); linear = `apply_fp8_marlin_linear` -> `gptq_marlin_gemm` (JIT module) | `layers/quantization/fp8.py:480-484,1019-1020`; `marlin_utils_fp8.py:58-78`; `R32/exp1_32b/server.log:15` |
| `--chunked-prefill-size` | None (auto) | must be a multiple of page size | `validation_hook.py:104-107` |
| `--enable-cache-report`, `--enable-metrics`, `--stream-response-default-include-usage` | False | usage `cached_tokens`; gates every HiCache Prometheus metric; usage chunk on every stream | `fields/serving.py:181-184,255-258`; `fields/observability.py:77` |
| env `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR` / `SGLANG_HICACHE_NIXL_BACKEND_PLUGIN` / `SGLANG_HICACHE_NIXL_USE_DIRECT_IO` (**no `BACKEND` in the third**; `..._BACKEND_USE_DIRECT_IO` matches nothing and is silently ignored) | None -> `/tmp/hicache_storage` (tmpfs here!) / `auto` (3FS > POSIX > GDS_MT > GDS) / True | dir is a comma list, the only one of the three in `environ.py` (`:738`); plugin read by `os.getenv` (`nixl_utils.py:82`), set `POSIX`; direct-IO: extra-config `use_direct_io` first, then the EnvBool (`environ.py:742`; `nixl_utils.py:38-46`); O_DIRECT **confirmed** on this NVMe for the 32B (`O_DIRECT is active with a file-based backend (POSIX)`, `path-mode FILE registration active`, `R32/exp1_32b/startup_facts.txt:8-10`; POSIX was auto-selected there, the explicit `PLUGIN=POSIX` is untested), not on tmpfs | `environ.py:738-742`; `storage/nixl/hicache_nixl.py:88-90`; `storage/nixl/nixl_utils.py:70-82,122` |
| env `SGLANG_HICACHE_FILE_BACKEND_{STORAGE_DIR,MAX_SIZE,EVICTION_RATIO}`; nixl L3 cleaner | None / None / 0.9; high 80 % / low 70 % | **file backend only** (`lru_file_evictor.py:133-135` is the sole reader of `MAX_SIZE`; `start_server.sh:11`'s `L3_MAX_SIZE` does nothing for nixl). nixl has **no byte cap**: only extra-config `l3_cleaner_enabled`, `l3_cleaner_high_watermark`, `l3_cleaner_low_watermark` (% of the filesystem) | `environ.py:725-729`; `storage/file/lru_file_evictor.py:127-148`; `storage/nixl/nixl_utils.py:49-63`; `nixl_cleaner.py:21-22`; `hicache_nixl.py:167-168` |
| `priority` in request body | None -> 0 | reaches `Req.priority` without `--enable-priority-scheduling` unless `--abort-on-priority-when-disabled`; node priority is max-propagated along the insert path, lower evicted first | `scheduler.py:3172-3195`; `radix_cache.py:769-802`; `evict_policy.py:41-46` |

### 5.2 Launch commands (inside `sglang_hicache`, under `bash`; one server per configuration, never re-attach at runtime)

```bash
# docker exec -it sglang_hicache bash     (bash arrays: the container's default shell is zsh, and the old "string with inline JSON" form was word-split at the JSON's spaces)
MODEL=Qwen/Qwen3-32B-FP8; RUNDIR=/sgl-workspace/sglang/agent_cache/results/$(cat /sgl-workspace/sglang/agent_cache/.current_results)   # RUNDIR = stamp dir; $OUT = one cell's dir (§7.2)
[ -s /sgl-workspace/sglang/agent_cache/.current_results ] && mkdir -p $RUNDIR || echo 'no .current_results: run the last two lines of 2.4 first'
export MODEL KV_BYTES_PER_TOKEN=131072 L3_DIR=/mnt/nvme/hicache_l3 NVME_DEV=nvme0n1 HICACHE_FLUSH_TIMEOUT=1800   # read by hcommon.py:11-14,142 (defaults: the 8B's 147456, /var/hicache_l3, 120 s)
# pressure level (§5.3). The arrays capture $L1/$HSIZE when DEFINED (re-assigning L1/HSIZE afterwards silently boots P0): always go through set_level
set_level() { L1=$1; HSIZE=$2
  COMMON=(--model-path $MODEL --host 0.0.0.0 --port 30000 --page-size 64 --context-length 32768 --chunked-prefill-size 8192
    --mem-fraction-static 0.85 --kv-cache-dtype fp8_e5m2 --attention-backend triton --max-total-tokens $L1
    --max-running-requests 64 --radix-eviction-policy lru --enable-metrics --enable-cache-report --reasoning-parser qwen3)
  HOST=(--enable-hierarchical-cache --hicache-size $HSIZE --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first); }
set_level 262144 64        # P0   |   PH: set_level 131072 48   |   PL: set_level 131072 18
printf '{"prefetch_threshold": 256, "prefetch_timeout_base": 1.0, "prefetch_timeout_per_ki_token": 0.25}\n' > $RUNDIR/l3_extra.json   # budget caveat: §5.1
L3=(--hicache-storage-backend nixl --hicache-storage-prefetch-policy timeout --hicache-storage-backend-extra-config @$RUNDIR/l3_extra.json)
L3WC=(--hicache-storage-backend nixl --hicache-storage-prefetch-policy wait_complete --hicache-storage-backend-extra-config @$RUNDIR/l3_extra.json)
L3ENV=(SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/mnt/nvme/hicache_l3 SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX)
# (a) hbm_lru       P0     : python3 -m sglang.launch_server "${COMMON[@]}"                                            > $RUNDIR/server_hbm_lru.log 2>&1 &
# (b) hbm_host      P0     : python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}"                               > $RUNDIR/server_hbm_host.log 2>&1 &
# (c) three_tier    P0     : env "${L3ENV[@]}" python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" "${L3[@]}"  > $RUNDIR/server_three_tier.log 2>&1 &
# (d) three_tier_p  PH, PL : (c) at the level's L1/HSIZE                  (pressure, §5.3)
# (e) hbm_lru_p     PH=PL  : (a) at L1=131072                             (honest single-tier control; one boot serves both levels: same L1)
# (f) three_tier_wc PL     : (d) with "${L3WC[@]}" instead of "${L3[@]}"  (= the prefetch policy of every 2026-09-17 measurement; "worse at idle" is an H100 8B observation, untested here)
# (g) hbm_host_p    PH, PL : (b) at the level's L1/HSIZE                  (host-only under pressure: at PL it recomputes what (d) reads from L3)
```
COMMON = the measured launch line (`R32/exp1_32b/server_args.txt`) plus `--max-running-requests 64` (measured runs left it unset -> 4096) and the `--max-total-tokens`
pin. **Not yet booted with the 32B:** `--max-running-requests 64`, any `--max-total-tokens` pin, any `--hicache-size` other than 100, the `timeout` policy with the
extra-config (`@file` form: `hybrid_cache_controller.py:186-191`), the explicit `PLUGIN=POSIX`, and arm (a) without HiCache (its profiled pool is unverified, which is why
P0 pins 262,144 < 281,216 for every arm). Assert BOTH pools after every boot, and no `larger than the profiled value` warning, before trusting §5.3:
`grep -aoE 'max_total_num_tokens=[0-9]+|host pool: [0-9]+ tokens' $RUNDIR/server_<arm>.log` must show $L1 and 488320 / 366272 / 137344 (a stale `--hicache-size` would otherwise go unnoticed).
Every arm carries fp8_e5m2 KV + triton + mem-fraction 0.85: record it in `config.json`.

`--reasoning-parser qwen3` stays (it is in the measured launch line). With THIS client it does not keep thinking out of history (derived from source, not yet observed):
`sglang.benchmark.serving` concatenates `reasoning_content + content` into `generated_text` (`serving.py:143-149,531,548`) and the multi-turn wrapper re-feeds that as the assistant
`content` (`:1329-1331`); plan §8.13 described `bench_multiturn.py`. The parser still matters: the re-fed text never contains `</think>`, so the Qwen3 template renders every historical
assistant turn as plain text and history is prefix-stable. But the generated reply began with `<think>\n` and the re-rendered one does not, so the radix match of a returning turn ends at
the previous prompt (page-floored) and the reply is re-prefilled: `recompute >= output_len(prev) + new message tokens` even on a perfect hit. Confirm in the §4.4 dry run.

Readiness: poll `/health` **and** grep `The server is fired up and ready to roll` (`hicache_eval/scripts/start_server.sh:40-53`; use `START_TIMEOUT_S=2400`, not the default 900:
~633 s JIT-cold, 278-285 s warm, §2.6), then record as `startup_facts.txt`:
`grep -E 'max_total_num_tokens|Allocating .* host memory|HiCache|storage backend|Marlin|KV Cache is allocated|Load weight end' server.log`. Expected for this model on this box
(`R32/exp1_32b/{startup_facts.txt,server.log}`): `Weight-only FP8 compression will be used leveraging the Marlin kernel`; `Load weight end. ... quant=fp8, fmt=e4m3, ... mem usage=32.59 GB`;
`KV Cache is allocated. dtype: torch.float8_e5m2, #tokens: 281216` and `max_total_num_tokens=281216` [= $L1 with the pin] `... max_running_requests=4096` [64 with the flag];
`Allocating kv hierarchical KV host pool: 762944 tokens, 100.00 GB host memory.` [scales with HSIZE]; `Backend POSIX was instantiated`; `O_DIRECT is active ... (POSIX)`; `HiCacheL3Cleaner
started: dirs=['/mnt/nvme/hicache_l3'] high=80.0% low=70.0%`; `Tree cache initialized: ... impl=UnifiedRadixCache ... hicache_attached=True`. The two `posix_backend.cpp ...
/nonexistent-nixl-probe` E-lines at nixl init are the path-mode probe: benign.

### 5.3 Memory pressure: make the live working set exceed HBM (Qwen3-32B-FP8, fp8_e5m2 KV, b = 131,072 B/token)

**Provisional:** pool sizes below are measured or derived; every f, saturation point, gap scale and c is an ESTIMATE resting on the two
unmeasured constants (decode rate, prefix-extension cost, §5.4) plus the trace's mean per-turn `output_length` (§3.5; 220 below is the loader-default placeholder). Redo this section's numbers after matrix row 1.

- 1 GiB of KV = 8,192 tokens exactly; 1 GB = 7,629 tokens. Weights 32.59 GB on the GPU (measured, `R32/exp1_32b/server.log:16`).
- Auto device pool, **measured** with the §5.2 flags: `max_total_num_tokens=281216` = 34.33 GiB (K 17.17 + V 17.17 GB; 7.66 GB GPU left, `R32/exp1_32b/startup_facts.txt:4`).
  Seen only on HiCache arms, so every arm pins `--max-total-tokens` for an identical L1.
- Pool arithmetic (source): device = `min(profiled, --max-total-tokens)` floored to the page (`kv_cache_configurator.py:2159-2168`); host = `(int(GB x 1e9 // b) // 64 + 1) x 64`
  (`pool_host/base.py:151-159`; 100 -> 762,944 measured). A host pool <= device pool only logs a warning (`base.py:163-171`); what `write_through` does then was not traced, so every arm keeps L2 >= L1.
- Live working set: LMCache contexts grow ~14K -> ~35K (median input 21K); after the 32K cut the converted trace's replayed context averages **20.6K (swebench) / 19.7K (all), measured in §3.3**, so the 25K below is an upper-side assumption (about 20 % pessimistic for swebench, 2x for gaia): rescale the WS column by 0.8 for a swebench replay. WS = c x 25K (derived): c=4 100K tok (12.2 GiB),
  c=8 200K (24.4), c=12 300K (36.6), c=16 400K (48.8), c=32 800K (97.7 GiB). Ignores cross-session sharing of the agent system prompt; recompute from the converter's real `prompt_tokens`.
- **With default pools L3 is never read:** 281,216 + 762,944 = 1,044,160 tokens = 41.8 sessions of 25K, far above any c this GPU sustains (estimate below); the same holds at
  `--hicache-size 64` (769,536 tokens = 30.8 sessions; 30.0 with L1 pinned to 262,144 as in P0). L3 pressure needs BOTH pools shrunk.

| level | `--max-total-tokens` | `--hicache-size` | L1 tok (GiB) | L2 tok (GiB) | L1+L2 (sessions of 25K) | c | WS / pool | in-flight cap L1/(1.2 x 25K) -> needs f <= | role |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 262144 | 64 | 262,144 (32.0) | 488,320 (59.6) | 750,464 (30.0) | 4 (8) | 0.38 L1 (0.76 L1) | 8.7 -> any f (c=8: f <= 1, no margin) | **no-pressure tie control**: WS < L1, expect all arms within noise, L2/L3 reads ~0; c=8 touches L1 late in sessions (8 x 32.7K = 262K) |
| PH | 131072 | 48 | 131,072 (16.0) | 366,272 (44.7) | 497,344 (19.9) | 8 (12) | 1.53 L1 = 0.40 (L1+L2) (2.29 L1 = 0.60) | 4.37 -> f <= 0.55 (0.36) | **host-restore arm**: L1 < WS <= L1+L2, everything that leaves HBM fits in host, L3 reads ~0 by construction |
| PL | 131072 | 18 | 131,072 (16.0) | 137,344 (16.8) | 268,416 (10.7) | 12 (16) | 1.12 (L1+L2) (1.49) | 4.37 -> f <= 0.36 (0.27) | **L3 arm**: WS > L1+L2, ~32K (c=16: 131K) tokens live only on the SSD |
| (defaults) | auto 281,216 | 100 | 281,216 (34.3) | 762,944 (93.1) | 1,044,160 (41.8) | - | - | 9.4 | not an arm: L3 never read at any sustainable c |

Pools: L1 = the flag (multiples of 64, below the measured auto pool), L2 derived from the host rule, GiB = tokens / 8,192. PH and PL share L1, so the single-tier
control (e) boots once for both. Host pinned RAM 64 / 48 / 18 GB of 167 GB, no swap: `--hicache-size 100` boots (all of 2026-09-17) but leaves ~8-28 GiB `free`
(57-62 GiB available, `free -g` rows in `R32/run_c2.log`), at which point the Claude Code harness kills background waiters (§10): keep <= 64 with a client on the box.
`--hicache-size 16` would give 122,112 < L1 (warning), hence 18. Optional larger host arm if c=12 is wanted at f <= 0.55: `--max-total-tokens 196608` (24.0 GiB) +
`--hicache-size 32` (244,160) = 440,768 tokens (17.6 sessions). HANDOFF §5: without WS > L1+L2 every policy ties `nohicache`; PL is the only level where L3 can show anything.

**Two constraints per cell** (the WS ratio alone is not enough): (i) `L1 >= 1.2 x f x c x context`, because the PrefillAdder admits only within available + evictable
device tokens (`schedule_policy.py:636-658,807-808,1215,1228`) and beyond that TTFT is admission queueing (HANDOFF §4); (ii) `WS = c x context > L1` (host arm) or
`> L1+L2` (L3 arm). With the in-flight fraction `f = turn_time / (turn_time + gap)` they require `f <= L1 / (1.2 x WS)`: < 0.83 for any host arm, < 0.42 for any L3 arm.

Where a 25K prefix lives decides TTFT (derived from the §5.4 fits): L1 0.36 s, L2 0.44 s, L3 4.96 s, recompute 18.0 s (`T_rec`, §3.5); only the recompute occupies
the GPU. **Unmeasured input 1, returning turn on a device-resident prefix (+~1K tokens):** bound >= 0.46 s (a 1,024-token prefill from scratch, measured); model value
~1.4 s = `T_L1(25K) + T_rec(26K) - T_rec(25K)` (marginal 1.06 ms/token at 25K); the trace's uncached delta is nearer 330-550 tokens/turn plus the 220 re-prefilled reply
tokens (§5.2). Planning value 1.4 s. **Unmeasured input 2, decode rate:** first-principles ESTIMATE: weight-only Marlin reads 35.0 GB (the logged 32.59 "GB" is GiB) of weights per step plus
3.28 GB of fp8 KV per 25K-token sequence; at an assumed ~2.0 TB/s HBM bandwidth (spec, from memory) x efficiency 0.5-0.8: RAW decode 26-42 tok/s at batch 1, ~17-28 per sequence at batch 7,
~12-19 at 15 (this is what matrix row 1 measures). In the closed loop each sequence's decode also stalls behind the other sessions' ~1.4 s prefills: EFFECTIVE ~10-13 tok/s per sequence at
c=8, ~6-7 at c=16; the f / X figures below use the effective rates.

Consequence (ESTIMATE, closed-loop model on those inputs, 220 output tokens, no control load): one turn is 6.6-9.9 s at c=1 against a ~2.08 s MEAN gap (README; 0.71 s median; f is a time
fraction, so the mean counts), so **f = 0.76-0.83 at c=1 and ~0.90-0.92 at c=8: in-flight ~ 0.9 c.** Turn throughput reaches 65-71 % of its ceiling (0.47-0.54 turns/s) at c=8 and 89-92 %
at c=32 while turn time goes 19-24 s -> 63-74 s: compute saturates near **c ~ 8**, memory admission at c = 8.7 (P0) / 4.4 (PH, PL). The paired controls add T_rec(context)/K of GPU per turn
(§7.3): K = 40 lowers the ceiling to 0.39-0.43 and stretches a c=8 turn to 22-28 s. At real gaps (i) and (ii) cannot both hold: **every WS > L1 cell at gap scale 1 is an overload cell**
(run once as the honest short-gap case, bin by queue depth). Tiering cells need a mean gap of at least ~13-16 s (PH c=8), ~26-34 s (PH c=12), ~31-39 s (PL c=12), ~46-58 s (PL c=16)
(control load at the K of the cleaner paragraph below included; ~10-15 % less without it): set `--agentic-gap-scale = required_mean_gap / mean(pre_gap)` once §3.5 gives the mean (on the
README mean 2.08 s, unverified until §3.5: x6-8, x13-16, x15-19, x22-28). Record measured f per cell (`num_running_reqs`/c from `memtime.csv`, §6.1); require median queue
depth ~0 in the window or bin by it (§9.1).

**L3 write-through capacity (32B).** SSD write ceiling measured 391 MiB/s = 0.38 GiB/s at 99 % util (`R32/exp1_32b/iostat.log`) = 3,130 tok/s at b = 131,072 (derived).
The GPU cannot make KV faster than it prefills: 2,140-2,270 tok/s for 512-4,096-token prompts, 1,660 at 16K, 1,220 at 32.5K (L / median TTFT, §5.4) = at most 73 % of
the ceiling; steady state far lower (estimate 12-20 %). **The disk keeps up with write-through for the 32B on average** (the 8B could not: 8,858 tok/s x 147,456 B = 1.22 GiB/s =
3.2x the ceiling). But the backup is not spread over the prefill: the log shows each 32.5K prompt (26.7 s prefill) followed by a ~10 s burst AT the ceiling (390 MiB/s, 99 % util,
`R32/exp1_32b/iostat.log` nvme0n1 samples 21, 27, 32); derived at 0.38 GiB/s: ~4.5 s for a 14K turn 0, ~0.4 s for a 1.2K delta. Not measured: L3 restores overlapping such a burst, and
concurrent L3 reads (0.62 GiB/s already runs the device at 97 % util) against writes in general: watch `w_await`/`%util` in PL cells.

**L3 cleaner budget.** nixl deletes from 80 % of the 368.0 GiB filesystem = 294 GiB = 2.41M tokens down to 70 %, oldest mtime first (`nixl_cleaner.py:21-22,148-157,181-184`): the turn-0
prefix pages of the longest-parked sessions, i.e. a second eviction policy in the cell. Everything prefilled is written through, controls included (`R32/exp1_32b/metrics_after.txt`:
197,184 prefilled -> 195,648 tokens backed up to L2 and to L3). Budget per three-tier cell (derived): `N_conv x 4.0 GiB (final context <= 32.7K) + N_ctl x 3.05 GiB (25K control) <= 294 GiB`,
`N_ctl ~ 39 x N_conv / K` at 40 turns. K = 4 would be 9-10 controls = ~258K tok (31.5 GiB) per conversation: watermark crossed after ~9 conversations, ~1.3 TiB for a c=8 cell. c=8 /
40 conv: N_ctl <= 44 -> **K = 40** (~1 control per conversation: 160 + 119 = 279 GiB). c=12 / 60 conv: 240 GiB leaves 17 controls; with `"l3_cleaner_high_watermark": 95,
"l3_cleaner_low_watermark": 85` in `l3_extra.json` (350 GiB; the 18 GiB margin covers the cleaner's 30 s interval at 0.38 GiB/s) 36 -> K = 72. c=16: at most 64 conversations (256 GiB),
30 controls at 95/85 -> K >= 84. Not counted: ~220 tokens of orphaned reply KV per turn if the §5.2 history note holds (<= 1.05 GiB per conversation). K and these bounds are provisional:
check `df /mnt/nvme` and `l3_stats.json` after the first c=8 cell, re-derive GiB per conversation, and record cleaner deletions (§9.1).

### 5.4 Constants for Qwen3-32B-FP8 on this box: measured 2026-09-17, plus TWO STILL TO MEASURE

Setup of every measured row: fp8_e5m2 KV, triton, mem-fraction 0.85, page 64, ctx 32768, `--hicache-size 100`, `wait_complete`, idle server, one in-flight probe with `max_tokens=1`,
n = 3; the `timeout` policy and concurrent load are unmeasured. TTFT source: `R32/exp1_32b/ttft_by_tier.csv` (medians recomputed from the CSV, equal to `R32/COMPARISON.md:157-215`; hit rows have 64 uncached tokens).

| median TTFT (s) at L = | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,512 |
|---|---|---|---|---|---|---|---|
| recompute | 0.2393 | 0.4552 | 0.9021 | 1.8597 | 4.0303 | 9.8764 | 26.6922 |
| L1 hit (device) | 0.0638 | 0.0680 | 0.0758 | 0.1046 | 0.1504 | 0.2552 | 0.4597 |
| L2 hit (host) | 0.0680 | 0.0705 | 0.0811 | 0.1067 | 0.1604 | 0.2992 | 0.5673 |
| L3 hit (NVMe, nixl POSIX O_DIRECT) | 0.0904 | 0.2154 | 0.3985 | 1.0021 | 1.6937 | 3.3073 | 6.3898 |
| L3 / recompute (below 1 = L3 wins) | 0.38 | 0.47 | 0.44 | 0.54 | 0.42 | 0.33 | 0.24 |

| constant | value | status | source |
|---|---|---|---|
| b (KV bytes/token) | 131,072 | measured: 536,870,912 B / 4,096 tok | `R32/exp0_32b/exp0_results.json` step2_backup |
| device pool (L1_TOKENS) | 281,216 tok = 34.3 GiB | measured (HiCache arms only) | `R32/exp1_32b/startup_facts.txt:4` |
| weights on GPU | 32.59 GB, fp8 e4m3, weight-only Marlin | measured | `R32/exp1_32b/server.log:15-16` |
| host pool | 762,944 tok @ 100 GB | measured; 488,320 / 366,272 / 137,344 @ 64 / 48 / 18 GB are derived (§5.1 rule) | `R32/exp1_32b/startup_facts.txt:5` |
| P, marginal prefill (slope fit); recompute bar b x P | 1,226 tok/s, intercept -1.251 s; 0.150 GiB/s | measured | `R32/COMPARISON.md:139-141` |
| tier fits, rate + intercept | L1 9.81 GiB/s + 0.053 s (65x the bar); L2 7.75 + 0.048 s (52x; pass ran while write-through drained); L3 0.623 + 0.058 s (4.2x) | measured | `R32/COMPARISON.md:142-152`; `R32/REPORT.md:82-83` |
| SSD ceiling | ~0.68 GiB/s read (704 MiB/s, 97 % util), ~0.38 GiB/s write (391 MiB/s, 99 % util) | measured (iostat, 5 s samples; no fio saved) | `R32/exp1_32b/iostat.log`; `R32/REPORT.md:84` |
| boot to `fired up` | ~633 s (10.6 min) JIT-cold; 278-285 s warm with the 100 GB pool | measured (`R32/preflight/boot_32b/server.log:6,51`; `R32/exp{0,1}_32b*/server.log`) | §2.6 |
| 25K-token recompute; 25K-token restore | ~18 s (quadratic fit 18.0; linear interpolation 18.9; slope fit 19.1); L3 ~5.0 s, L2 ~0.44 s, L1 ~0.36 s | derived (§3.5 fit; intercept + 25,000 x b / rate) | the rows above |
| **decode rate** (tok/s per sequence at batch 1, 4, 8, ~25K contexts) | **NOT MEASURED** | open constant 1 | command below |
| **TTFT of extending a ~24.5K cached prefix by ~1K tokens** | **NOT MEASURED** | open constant 2 | command below |

Admission rule (HANDOFF §2): a tier pays only if `bandwidth(tier) > b x P`. Here L2 and L3 both clear the bar and an L3 hit is faster than recompute
at 7 of 7 lengths; quote that, not the fit break-even "2,114 tokens" (`R32/COMPARISON.md:153`), an artefact of the superlinear recompute curve. The
L3 rate is the SSD ceiling and is model-independent: the 8B (bar 1.22 GiB/s) loses at 7 of 7 lengths on this box.

```bash
# THE TWO OPEN CONSTANTS (matrix row 1, ~6 min). NOT executed yet: flags checked in benchmark/one_batch_server.py:158-210,321-328, behaviour unverified
# (cli_main calls server_args.resolve_once() even with --base-url, :1494-1495). Run against arm (a) at P0: no HiCache, because the prompts are
# seed-fixed (:662) and /flush_cache never clears L3, so on a nixl arm a second run is an L3 hit. One discarded long probe first.
# HOST command: RUNDIR is defined on this line (the §5.2 definition lives in the container shell; an empty value would write to /decode_delta in the container's writable layer, silently)
RUNDIR=/sgl-workspace/sglang/agent_cache/results/$(cat /home/wanhr/sglang/agent_cache/.current_results)   # container path; the stamp comes from §2.4
docker exec -e RUNDIR="${RUNDIR:?set RUNDIR first}" sglang_hicache bash -lc 'cd /sgl-workspace/sglang && mkdir -p "$RUNDIR/decode_delta" && python3 -m sglang.benchmark.one_batch_server \
  --model-path Qwen/Qwen3-32B-FP8 --base-url http://127.0.0.1:30000 --dataset-name random-ids \
  --batch-size 1 4 8 --input-len 25600 --output-len 220 --cache-hit-rate 0.96 --skip-warmup --show-report \
  --result-filename "$RUNDIR/decode_delta/warm_prefix.jsonl" 2>&1 | tee "$RUNDIR/decode_delta/warm_prefix.log"'   # inside the §5.2 container shell: run the inner command without the wrapper
# per batch size: flush, prefill the first int(25600 x 0.96) = 24,576 tokens with max_new_tokens=1 (:462-524, :621-625), then time 1,024 uncached tokens on
# that prefix + 220 decoded tokens. last_ttft at bs=1 = open constant 2; output_throughput / batch_size = open constant 1 at B = 1, 4, 8. Put both in constants.json.
# Cross-check of the decode rate at short context (median ITL); `random` would download ShareGPT, and probe.py:49 fixes max_tokens=1 so it cannot give a decode rate:
#   python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port 30000 --model Qwen/Qwen3-32B-FP8 --dataset-name random-ids --random-input-len 1024 \
#     --random-output-len 256 --random-range-ratio 1.0 --num-prompts 8 --max-concurrency 1 --warmup-requests 1 --output-details --output-file $RUNDIR/decode_delta/itl_c1.jsonl
# Re-check procedures for the measured constants (container shell of §5.2; reuse hicache_eval/scripts until §11's copies exist). hcommon is imported from cwd, scrape() does not mkdir:
cd /sgl-workspace/sglang/hicache_eval/scripts && mkdir -p $RUNDIR/b_measure $RUNDIR/p_measure
# b: one 4096-token write-through probe, read the D2H counters (exp0.py:90-104 procedure)
python3 cachectl.py scrape $RUNDIR/b_measure/before.txt && python3 probe.py --len 4096 --seed 4242 && python3 cachectl.py drain \
  && python3 cachectl.py scrape $RUNDIR/b_measure/after.txt && python3 cachectl.py delta $RUNDIR/b_measure/before.txt $RUNDIR/b_measure/after.txt | grep -E 'hicache_backup_(bytes|tokens)_total'
# expect bytes/tokens == 131072 exactly (measured: 536,870,912 / 4,096)
# P: recompute-only sweep on server (a), then linear fit of median TTFT vs L (run_p_measure.sh:9-17; analyze_models.py:43-50). exp1.py:22 reads os.environ["RESULTS"] unconditionally
# (KeyError without it) and writes to $RESULTS/$EXP1_OUT; it never sources env.sh, so the frozen-dir guard is not involved:
RESULTS=$RUNDIR EXP1_OUT=p_measure python3 exp1.py --reps 3 --tiers recompute --lengths 512,1024,2048,4096,8192,16384,32512 > $RUNDIR/p_measure/run.log 2>&1
# L1_TOKENS: 281216 was measured with HiCache attached (R32/exp1_32b/cold_check.txt); confirm it for arm (a): grep -aoE 'max_total_num_tokens=[0-9]+' $RUNDIR/server_hbm_lru.log | head -1
```

---

## 6. Instrumentation

### 6.1 Metrics to scrape and cadence (`/metrics`, needs `--enable-metrics`)

| family | type | meaning | refresh |
|---|---|---|---|
| `sglang:kv_used_tokens`, `sglang:kv_evictable_tokens`, `sglang:kv_available_tokens`, `sglang:token_usage` | Gauge | device pool: locked by running reqs / radix-cached idle KV / free. **`token_usage` excludes evictable KV** (`used = total - (available + evictable)`, `pool_stats_observer.py:221-224,141-142`) | pushed per prefill batch and every `--decode-log-interval` decode steps (`metrics_reporter.py:812-822,856-858`, cadence only); frozen while idle |
| `sglang:num_running_reqs`, `sglang:num_queue_reqs` | Gauge | in-flight / waiting requests: the queue-depth axis for TTFT bins (§9.1) | same cadence (`metrics_collector.py:279-286,1365-1366`) |
| `sglang:hicache_host_used_tokens`, `sglang:hicache_host_total_tokens` | Gauge | L2 pool occupancy (`logical_size - available_size()`) | same cadence (`metrics_reporter.py:1151-1165`) |
| `sglang:max_total_num_tokens` | Gauge | device pool size, emitted once | `metrics_collector.py:1040` |
| `sglang:prefill_effective_tokens_total{mode=input\|device_hit\|host_hit\|storage_hit}` | Counter | server-side per-tier split, retracted re-counts excluded; hit rate = rate(sum *_hit)/rate(sum all) | per prefill batch (`metrics_collector.py:900-913`; `metrics_reporter.py:760-765`) |
| `sglang:cached_tokens_total{cache_source=device\|host\|storage\|total}` | Counter | tokenizer-side, per FINISHED request (lags by decode time) | `metrics_collector.py:1638-1641` |
| `sglang:cache_hit_rate` | Gauge | per-prefill-batch first-attempt ratio, not cumulative | `metrics_reporter.py:743-759` |
| `sglang:load_back_tokens_total{pool}`, `sglang:load_back_bytes_total`, `sglang:load_back_duration_seconds` | Counter/Hist | L2->GPU volume and CUDA-event span per merged op (bytes/duration_sum = achieved H2D BW) | `metrics_collector.py:2186-2229` |
| `sglang:hicache_backup_tokens_total{pool}`, `sglang:hicache_backup_bytes_total`, `sglang:hicache_backup_duration_seconds` | Counter/Hist | GPU->L2 (all write policies); the bytes counters carry no extra label | `metrics_collector.py:2202-2239` |
| `sglang:evicted_tokens_total`, `sglang:eviction_duration_seconds`, `sglang:hicache_dropped_tokens_total{reason,pool}` | Counter/Hist | device slots freed; tokens destroyed without backup | `metrics_collector.py:2166-2183,2241-2248` |
| `sglang:prefetched_tokens_total`, `sglang:storage_prefetch_hit_tokens_total`, `sglang:storage_prefetch_unfulfilled_tokens_total{reason}`, `sglang:backuped_tokens_total` | Counter | L3 read volume (before prefix dedup on the unified path), L3 query hits, unfulfilled by reason, host->L3 writes | `metrics_collector.py:1906-1931`; `unified_radix_cache.py:1966` |
| `sglang:prefetch_bandwidth`, `sglang:backup_bandwidth` | Hist | **empty for file/nixl** (only hf3fs/mooncake/sim implement `get_stats`) | `hicache_storage.py:337`; `grep -rn 'def get_stats' mem_cache/` |

Cadence: 1 s sampler for the gauges (memory-time, queue depth), before/after snapshot for the counters. Families
with labels (`dropped{reason,pool}`, `backup_tokens{pool}`, `load_back_tokens{pool}`, `prefetch_unfulfilled{reason}`,
`prefill_effective{mode}`, `cached_tokens{cache_source}`) need `hcommon.parse_metrics_labeled` for per-label reads
(`parse_metrics` sums across labels, totals stay right; `hicache_eval/scripts/hcommon.py:94-123`).
```bash
BASE=http://127.0.0.1:30000; CSV=$OUT/memtime.csv      # $OUT = the cell dir of §7.2 step 0
echo 't,token_usage,num_used_tokens,kv_used_tokens,kv_evictable_tokens,kv_available_tokens,hicache_host_used_tokens,hicache_host_total_tokens,max_total_num_tokens,num_running_reqs,num_queue_reqs' > $CSV
while :; do curl -s $BASE/metrics | awk -v t="$(date +%s.%N)" 'BEGIN{n=split("sglang:token_usage sglang:num_used_tokens sglang:kv_used_tokens sglang:kv_evictable_tokens sglang:kv_available_tokens sglang:hicache_host_used_tokens sglang:hicache_host_total_tokens sglang:max_total_num_tokens sglang:num_running_reqs sglang:num_queue_reqs",k," ")} /^#/{next} {split($1,a,"{"); v[a[1]]=$NF} END{printf "%s",t; for(i=1;i<=n;i++) printf ",%s",(k[i] in v?v[k[i]]:""); printf "\n"}' >> $CSV; sleep 1; done &
echo $! > $OUT/.memtime.pid
```
`/server_info` has no cache/HiCache fields (`scheduler.py:4935-4996`); do not scrape it for hit rate.

### 6.2 Per-request tier split: zero server changes needed

`cached_tokens` counts **device + host(load-back) + storage** hits (accumulated from
`len(req.prefix_indices)` after `init_load_back` spliced host indices in: `schedule_batch.py:2656-2660`;
`schedule_policy.py:1305-1316`). The per-tier split is computed once per request on its first chunk
(`schedule_batch.py:2665-2679`, `split_cached_prefix_by_tier` at `:215-249`) and returned as
`meta_info.cached_tokens_details {device, host[, storage, storage_backend]}` (`output_streamer.py:85-115`;
`tokenizer_manager.py:2331-2336`), or on chat as `sglext.cached_tokens_details` when the body sets
`return_cached_tokens_details: true` (`protocol.py:883`; `serving_chat.py:1892-1900`). A declined/dropped
load-back counts as recompute (`materialized_host_hit_len`, `schedule_batch.py:1357-1361`), which is the right restore-hit semantics.

Restore-hit classes per returning turn (prompt_tokens P, details d):
`device-only` if `d.host==0 and d.storage==0 and cached >= P - 64`; `host-restored` if `d.host>0`;
`storage-restored` if `d.storage>0`; else `recomputed`; `recompute_tokens = P - cached`. If the §5.2 history note holds
(regenerated reply never cached; derived, confirm in §4.4), `cached >= P - 64` can never be true on a returning turn: compare
against floor64 of the PREVIOUS round's prompt tokens instead. The
"MatchResult split" the starter kit asks for is exactly this triple; `MatchResult` itself has no storage
field (`base_prefix_cache.py:186-230`), storage arrives via `pop_prefetch_loaded_span` (`scheduler.py:3817-3832`).

### 6.3 Eval-only server log line (only if a client cannot read `sglext`, e.g. AIPerf)

Site: `python/sglang/srt/managers/schedule_batch.py`, immediately after line 2679
(`req._cache_breakdown_computed = True`): class-agnostic, fires exactly once per request, after load-back
splice and storage-span pop. `schedule_batch.py` has `logger` (`:161`) but **no `import time`** (`:56` is `logging`): add one.
```
+                    logger.info("EVAL_TIER rid=%s t=%.6f prompt_len=%d prefix_len=%d device=%d host=%d storage=%d "
+                                "host_hit_len=%d host_loaded_len=%d storage_hit_len=%d retracted=%d",
+                                req.rid, time.time(), len(req.origin_input_ids), len(req.prefix_indices),
+                                req.cached_tokens_device, req.cached_tokens_host, req.cached_tokens_storage,
+                                req.host_hit_length, req.host_loaded_length, req.storage_hit_length, int(req.retracted_stain))
```
Parse with `grep EVAL_TIER server.log`. Archive as `agent_cache/patches/0002-eval-tier-log.patch`, revert before commit (§4.3).
Do not add a `SGLANG_*` env var to gate it (that would pull in `.claude/skills/env-var-conventions`).

### 6.4 Offline computation

- **Memory-time (token-seconds):** the gauge integral of `kv_used + kv_evictable` (+ `hicache_host_used`) is *pool
  occupancy*: a warm LRU pool stays full, so it is ~pool x wall for every arm and "P0 minus arm" measures
  `--max-total-tokens`, not policy. Report **idle-session memory-time** per returning turn from the tier split (§6.2):
  device-only -> `prompt_tokens x pre_gap` on device; host-restored -> same on host; recomputed -> 0 resident (freed),
  cost `recompute_tokens`; sum per arm, normalize by device pool x client wall, x b (131,072 B/token) for byte-seconds. The gauge integral
  is only an upper-bound check, windowed to [client start, last response], sampler stopped **before** the post-run flush
  (`wait_until_flushable` POSTs `/flush_cache`, `hcommon.py:231`, emptying the pools, `scheduler.py:4909-4911`). "Freed" only between equal-pool arms.
- **Restore-hit rate:** from `--output-details` lists (conversation-major/round-minor: the flatten at `serving.py:1567-1569`;
  explicit `conv_idx`/`round_idx` after hunk 7), or from `EVAL_TIER` lines joined on rid.
- **Queue depth per turn:** nearest `memtime.csv` sample to each `start_times` entry -> `num_queue_reqs`/`num_running_reqs`.
- **Server cross-check:** delta of `prefill_effective_tokens_total{mode}` must match the sum of per-turn details (+ controls); balances only with the warmup handled as in §7.2 step 3.

### 6.5 The paired-control rule (HANDOFF §3, carried over verbatim)

Every "restore" TTFT is compared against a **never-before-sent prompt of the same length sent at the
same instant under the same concurrency** (`hicache_eval/scripts/exp2.py:148-165`), never against an idle
recompute curve. Control seeds are salted per process (`exp2.py:28-30`) and control rows must show
`cached_storage` NaN/0, else the control silently became an L3 hit. In a gap replay the control **cannot** sit in the
conversation slot (every round extends `prev_messages` and appends the reply, `serving.py:1319,1329-1331`, poisoning every
later prefix; `exp2.py:148-165` is single-shot, outside any conversation): it is the side-channel request of patch hunk 6
(`--agentic-control-every K`, §4.1), fired at the returning turn's instant, salted per process, outside the semaphore.
Analysis: paired (restore - control) differences per turn-index bin, 95 % CI over >= 30 pairs; assert `storage == 0, host == 0, device <= 64` per control row. With the 32B a control is a
~25K recompute (18 s of GPU, 3.05 GiB of write-through, ~25K of L1 held outside the semaphore: at PH/PL the 4.37-request admission cap is ~3.4 while one runs, and its KV then sits in the LRU as
never-reused pages), so K is set by the §5.3 / §7.3 budgets (K = 40-84 -> ~30-40 pairs per CELL, spread over turn indices by the conv_idx offset): reach >= 30 pairs per bin by pooling turn
indices into coarse bins or by repeating the cell (new `RC_SALT`, cold L3), never by lowering K or adding conversations on a three-tier arm.

---

## 7. Experiment matrix (execution order) and per-run checklist

### 7.1 Order

| # | arm (§5.2) | pressure | trace | c | purpose |
|---|---|---|---|---|---|
| 0 | (a) | P0 | gap-test (§4.4) | 2 | patch + pipeline smoke |
| 1 | (a) | P0 | `one_batch_server` warm-prefix run (§5.4) | 1, 4, 8 | decode_rate(B) and the prefix-extension TTFT: the two unmeasured inputs of §5.3/§7.3. b, P, L1/L2/L3 rates are already measured (`R32/`) |
| 2 | (a),(b),(c) | P0 | LMCache, real gaps (median 0.71 s) | 4, 8 | **tie control and honest short-gap case**: WS = 100K / 200K < L1 = 262K; expect all arms within noise, L2/L3 reads ~0. f ~ 0.85-0.92 (estimate, on the 2.08 s mean gap): in-flight ~ 0.9 c |
| 3 | (e),(g),(d) | PH | LMCache, real gaps | 8 | **overload reference**, run once: f ~ 0.90-0.92 so in-flight ~ 7.3 x 25K = 182K > L1 = 131K; TTFT is admission queueing: bin by queue depth, not a tiering result |
| 4 | (e),(g),(d) | PH | LMCache, gap scale s_H (mean gap >= ~13-16 s, §5.3) | 8 | **host-restore case**: WS = 1.53 L1 = 0.40 (L1+L2); (g) vs (e) is the L2 effect, (d) should equal (g) (L3 idle) |
| 5 | (e),(g),(d) | PL | LMCache, gap scale s_L (mean gap >= ~31-39 s) | 12 | **L3 case**: WS = 1.12 (L1+L2); (g) recomputes what (d) reads from L3. c=16 (1.49x) only if s can reach a 46-58 s mean gap; it runs with 64 conversations (4 x c, the L3 disk cap of §5.3), so its in-flight == N window is shorter: say so in the report |
| 6 | (d) vs (f), (g) | PL | as row 5 | 12 | `timeout` vs `wait_complete` (HANDOFF §2) with the host-only reference. (f) is the policy with measured constants here; (d) with the default budget may measure the cut-off, not the tier (§5.1) |
| 7 | (d) | PL | AgentX via AIPerf, 1800 s | 8 | long-context standardized view (context-capped). At c=8 WS <= 8 x 32,768 = 262K < L1+L2 = 268K: expect L3 reads ~0 unless subagent trees multiply the live contexts (unverified); record storage-hit tokens and do not read this row as an L3 result |
| 8 | (a) | P0 | Mooncake toolagent, `--backend sglang`, slowdown >= ~15 | 8 | cross-session sharing only; pure backlog without the slowdown factor (§3.4) |

**Provisional:** every c, f, K and gap figure above was sized from ESTIMATES of the decode rate and the prefix-extension cost (§5.3);
s_H / s_L cannot be fixed until §3.5 gives mean(pre_gap) and row 1 gives those two constants. Re-derive rows 2-8 after row 1.
Arm order within a row is interleaved per cell (dflash RUNBOOK 0.4) so thermal/queue drift does not bias one arm.
`--seed` and `--dataset-offset` are **fixed per row** (recorded in `config.json`): every arm and gap scale in a row replays
the same conversations in the same order (the loader rotates by offset and takes the first `--num-prompts`,
`agentic_trace.py:74-76,82-83`; sessions span 14K-130K context, so a per-arm offset confounds arm with sample). Salt only
the control prompts (`RC_SALT`, `exp2.py:30`); rotate by row if at all. Use >= 5 x c conversations (20 at c=4, 40 at c=8, 60 at
c=12, 64 = the cap at c=16) so the in-flight == N window dominates (§9.1), and at most 64 per three-tier cell (L3 disk budget, §5.3). `--agentic-control-every`
K is fixed per row like the seed (the control load is part of the workload): K = 40 for rows 2-4, 72 for rows 5-6 with the 95/85 cleaner watermarks (§5.3; provisional).
`--agentic-max-turns 20` halves cell time (§7.3) but caps contexts near ~25K, which lowers WS: if used, use it for every arm of the row and record it.

### 7.2 Per-run checklist (one cell)

```bash
# 0. inside the container (bash). Two names: RUNDIR = stamp dir (§5.2), OUT = this cell's dir; cwd = the scripts dir, because hcommon/cachectl/probe are imported from cwd:
RUNDIR=/sgl-workspace/sglang/agent_cache/results/$(cat /sgl-workspace/sglang/agent_cache/.current_results); OUT=$RUNDIR/<arm>_<trace>_c<N>; mkdir -p $OUT; cd /sgl-workspace/sglang/hicache_eval/scripts
# 0b. preflight gate (also in launch.sh/bench.sh): the store must be on the NVMe, not the root PD (§2.2); env for every hcommon/probe/cachectl call:
[ "$(findmnt -n -o SOURCE -T /mnt/nvme/hicache_l3)" = /dev/nvme0n1 ] || { echo 'L3 dir not on nvme0n1'; exit 1; }     # `exit` is for bench.sh; pasted by hand, stop here instead
export MODEL=Qwen/Qwen3-32B-FP8 KV_BYTES_PER_TOKEN=131072 L3_DIR=/mnt/nvme/hicache_l3 NVME_DEV=nvme0n1 HICACHE_FLUSH_TIMEOUT=1800   # hcommon.py:11-14,142 default to the 8B / /var/hicache_l3 / 120 s
# 1. cold L3 between arms, as ROOT (bucket dirs 00..ff are root-owned, §2.2), ARG_MAX-safe (HANDOFF §3; env.sh:22), gated:
find /mnt/nvme/hicache_l3 -mindepth 1 -delete; [ "$(find /mnt/nvme/hicache_l3 -type f | wc -l)" -eq 0 ] || { echo 'L3 wipe failed (run inside the container or with sudo)'; exit 1; }
# 2. page cache (container is --privileged; falls back to fadvise otherwise, hcommon.py:154-171):
sync; echo 3 > /proc/sys/vm/drop_caches
# 3. launch the arm (§5.2: `set_level`, log to $RUNDIR/server_<arm>.log; do not re-assign OUT), wait for 'fired up' (budget 300 s warm, 2400 s in a new container), save startup_facts.txt, assert BOTH pool sizes (§5.2);
#    ONE manual salted single-shot warm-up request (probe.py --len 4096 --seed $RANDOM; it also absorbs the one-time first-long-prefill cost, 9.0 s seen):
#    never the client's --warmup-requests, see Notes
# 4. flush via the gated endpoint, never the counter (scheduler.py:4779-4785, 4904-4906):
python3 -c 'import hcommon; print(hcommon.wait_until_flushable(600, verbose=True))'
# 5. telemetry on, 1 s (the 2026-09-17 campaign logs are 5 s averages, telemetry.sh:9,11; iostat exists in the container and on the host); memtime sampler on (§6.1):
#    iostat -x -d -t 1 nvme0n1 > $OUT/iostat.log & echo $! > $OUT/.iostat.pid; nvidia-smi dmon -s t -d 1 > $OUT/pcie.log & echo $! > $OUT/.dmon.pid
# 6. counters before (after the flush, so the warm-up prefill is excluded): python3 cachectl.py scrape $OUT/metrics_before.txt
# 7. client (patched, inside container); C = the row's c; ROW_SEED/ROW_OFFSET and K fixed per matrix row from config.json (§7.1; K from the §5.3 disk budget and the §7.3 GPU budget:
#    40 at c <= 8, 72 at c=12, provisional; K=4 would spend ~70 % of the GPU on controls and write ~1.3 TiB per c=8 cell), no client warmup, no --flush-cache:
python3 -m sglang.benchmark.serving --backend sglang-oai-chat --host 127.0.0.1 --port 30000 --model Qwen/Qwen3-32B-FP8 \
  --dataset-name agentic-trace --dataset-path /sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json \
  --num-prompts $(( 5*C > 64 ? 64 : 5*C )) --dataset-offset $ROW_OFFSET --agentic-max-turns 40 --max-concurrency $C --request-rate inf \
  --warmup-requests 0 --agentic-gap-scale $GAP --agentic-control-every $K --cache-report --output-details --seed $ROW_SEED --tag ${ARM}_c${C} --output-file $OUT/client.jsonl 2>&1 | tee $OUT/client.log
# 8. memtime sampler OFF first (the drain below flushes the pools and would be recorded, §6.4): kill $(cat $OUT/.memtime.pid)
# 9. drain then counters after: python3 -c 'import hcommon; hcommon.wait_until_flushable(5400)'; python3 cachectl.py scrape $OUT/metrics_after.txt
#    python3 cachectl.py delta $OUT/metrics_before.txt $OUT/metrics_after.txt $OUT/metrics_delta.json      (drain should be short: the SSD writes faster than the 32B prefills on average, §5.3; the last requests' backups still burst for ~0.4-10 s)
# 10. telemetry off (kill the .pid files); l3 stats: python3 cachectl.py l3 > $OUT/l3_stats.json      (nixl: files = 2 x pages, K and V separate; 4,194,304 B per file, 8,388,608 B per page)
# 11. stop the server from a SCRIPT FILE or by `kill $(cat server.pid)`; never paste pkill/pgrep -f into `docker exec ... bash -lc '<...>'`: the pattern matches that bash's own
#     command line (a wait loop did exactly that on 2026-09-17 and never ended: `R8/DEVIATIONS.md` D5, "a wait loop matched its own command line"). Long cells: watch a marker file or a log line (Monitor), never pgrep (§10).
bash /sgl-workspace/sglang/hicache_eval/scripts/stop_server.sh; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l   # 0
# 12. append summary.csv row; on the host: chown -R wanhr:wanhr agent_cache/results agent_cache/traces (everything the container writes is root:root)
```
Notes: the client's `--warmup-requests` replays the **whole first conversation** (`serving.py:1423,1431-1451`), which is
also conversation 0 of the main run (`:1549-1551`); with `write_through` its KV reaches L2 and L3 before `--flush-cache`
fires (`:1470-1471`), and `flush_cache` resets tree, host pool and controller queues (`scheduler.py:4904-4914`;
`unified_radix_cache.py:343-377`; `cache_controller.py:742-762`) but **never the storage backend** (`clear_hicache_storage`,
`scheduler.py:4633-4635`): conv 0 turn 0 would be an L3 hit in every three-tier cell and the step-6 snapshot would
include warmup prefill (HANDOFF §3 trap); hence step 3 + `--warmup-requests 0`. `--max-concurrency` is per conversation,
semaphore held across gaps (`:1385-1393`): "N live sessions", not N in-flight requests; `--output-file` appends (`:1899`).

### 7.3 Time budget (rough, first pass)

**Provisional: every cell time below is an ESTIMATE until the decode rate and the prefix-extension cost are measured (row 1, §5.4).**
Bring-up is done; daily overhead = §2.2 (~2 min) + `docker start`. Data + converter + Week-1 stats 0.5 day (CPU). Rows 0-1: ~1 h. Measured inputs: recompute 1.86 s @4K, 4.03 s @8K,
9.88 s @16K, 26.69 s @32.5K (18.0 s @25K, derived); 25K restore L3 ~5.0 s, L2 ~0.44 s, L1 ~0.36 s (derived); boot 278-285 s warm with a 100 GB host pool, ~633 s JIT-cold: budget 5 min per cell.
Cell time is **computed, not guessed**: `cell_s ~ turns / X + boot + drain`, `X = min(c / (turn_time(c) + s x mean_gap), 1 / GPU_s_per_turn)`, `GPU_s_per_turn = (1-m) x t_delta + m x
T_rec(context) + decode share + T_rec(context)/K`, m = fraction of returning turns that recompute (0 in a tiered arm that restores, up to 1 in (e)), t_delta ~ 1.4 s (estimate), K =
`--agentic-control-every` (each control is a full recompute: 18.0 s at 25K), turns = min(5 x c, 64) x <= 40. The control term decides K: K=4 adds 4.5 s/turn to the 1.95 s of the turn itself
(ceiling 0.16 turns/s, ~70 % of the GPU on controls), K=16 adds 1.1 s (0.33), K=40 0.45 s (0.39-0.43), K=72 0.25 s; no controls 0.47-0.54.
ESTIMATES (§5.3 model, mean gap 2.08 s, K = 40; K = 72 at c=12): X ~ 0.20-0.26 turns/s at c=4, 0.27-0.33 at c=8 (scale 1), 0.22-0.28 (PH c=8 at s_H), 0.20-0.25 (PL c=12 at s_L). So row 2
c=4 ~ 55-70 min per cell, c=8 ~ 85-105 min, row 4 ~ 100-125 min per tiered cell, row 5 ~ 165-205 min. A recompute-heavy (e) cell is GPU-bound at 1/(1.4 + 16.6 m + 0.55 + 18.0/K) turns/s:
~3.6 h at m = 0.35, 8.4 h at m = 1 (c=8): cap (e) cells by wall time (e.g. 90 min) and compare over the common window, or run rows 3-6 with `--agentic-max-turns 20`. Store the measured X
and turn_time(c) from rows 1-2 in `constants.json` and re-budget. Rows 2-6 are ~17 cells: **4-5 days** at 40 turns, about half with 20; row 7 = 35 min; row 8 only as a wall-time-capped
slice (§3.4). Reserve a day for re-runs.

---

## 8. Profiling

| tool | recipe | attributes | cannot attribute |
|---|---|---|---|
| torch profiler, manual window | server env `SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/agent_cache/results/traces SGLANG_PROFILE_WITH_STACK=false SGLANG_PROFILE_RECORD_SHAPES=false`; `curl -sS -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' -d '{"output_dir":".../traces/win1","activities":["CPU","GPU"],"with_stack":false,"record_shapes":false,"profile_id":"hicache-window"}'`, fire the returning-turn burst, `curl -sS -X POST .../stop_profile` -> `win1/hicache-window-TP-0.trace.json.gz` | H2D load-back as `transfer_kernel_impl<...>` kernels (io_backend `kernel`, `kernels/aot/csrc/kvcacheio/transfer.cu:261,354-377`), or as `Memcpy HtoD` from that same kernel path's `cudaMemcpyAsync` fallback (`:845-849`) or from the `direct` backend's Python-side `transfer_kv_direct` copies (`memory_pool_host.py:433-448`; not in `transfer.cu`), on the separate `host_to_device_stream` (`l2_transfer.py:52-55`); D2H backup on `device_to_host_stream`; forward-stream waits on per-layer load events. Compute kernels for this model (names read from source, ESTIMATE until the first trace is opened; no FlashAttention / flashinfer / DeepGEMM kernels should appear): linear layers = `sglang::apply_fp8_marlin_linear` -> `gptq_marlin_gemm` (JIT Marlin, `marlin_utils_fp8.py:58-78`; `kernels/ops/quantization/gptq_marlin.py:21-36`); prefill attention = Triton `_fwd_kernel` (`kernels/ops/attention/extend_attention.py:327`); GQA decode = `_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2` (`decode_attention.py:510,904`). Prefill (breakable) and decode (full) CUDA graphs are on, so replayed kernels have no CPU-op parent: search by kernel name | L3 file/NIXL IO (Python daemon threads, no CUDA/torch ops); GIL time |
| batch-count capture | `python -m sglang.profiler --url http://127.0.0.1:30000 --num-steps 40 --output-dir .../traces --profile-prefix hicache-window` (blocks until written; `profiler.py:21-66`); `num_steps` counts scheduler batches, prefill+decode (`scheduler.py:4179-4189`) | same | a request-defined window (use manual start/stop) |
| analysis skill | `Skill llm-torch-profiler-analysis` -> `python3 .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py --framework sglang --input .../win1/hicache-window-TP-0.trace.json.gz`; ask for rows below the 1 % default cutoff so transfer kernels appear (SKILL.md:35-39). **Never** its `--url` live mode: it drives synthetic 4090/2048 workloads that evict the state under study (SKILL.md:190-217) | kernel / overlap / fuse tables | stream timelines (use Perfetto on the same file) |
| `generate-profile` skill | `python3 -m sglang.test.send_one --profile` (SKILL.md:61-81) | one synthetic request | a HiCache window; use only as a sanity check of the profiler path |
| py-spy | inside the container (`/opt/sglang/bin/py-spy`), from a script FILE (pgrep self-match, §10): `SCHED=$(pgrep -f 'sglang::scheduler' \| head -1); for i in $(seq 1 12); do echo "===== $i"; timeout 30 py-spy dump --pid $SCHED --nonblocking; sleep 2; done > dumps.txt; grep -E '^Thread .*\(active\+gil\)' dumps.txt \| sed -E 's/.*: "(.*)"/\1/' \| sort \| uniq -c` (`hicache_eval/scripts/c3_dump.sh:12,21-36`). **Never `py-spy record -s`** on the launcher (hung 4.5 h, HANDOFF §3) | thread state at sample instants: GIL holder, `synchronize (torch/cuda/streams.py)` = waiting on GPU, frames in `prefetch_thread_func`/`backup_thread_func` | durations |
| `nvidia-smi dmon -s t -d 1` / `iostat -x -d -t 1 nvme0n1` | §7.2 step 5; `iostat` is installed (host and both containers); log formats from this box and model: `R32/exp1_32b/{pcie,iostat}.log` (5 s samples). Reference ceilings, measured: nvme0n1 704 MiB/s read at 97 % util, 391 MiB/s write at 99 % util | whole-GPU PCIe rx/tx MB/s; whole-device NVMe r/w kB/s, await, %util | load-back vs other H2D; prefetch vs backup vs writeback (r vs w is the only split); per-turn bytes (HANDOFF §3 `iostat_*` is a window) |
| `/metrics` histograms | §6.1: `load_back_duration_seconds` (CUDA-event span per merged op, excludes queue/fence waits) and `hicache_backup_duration_seconds`; bytes_total/duration_sum = BW while copying | L2<->GPU throughput | L3 durations for nixl (no `get_stats`): derive from `prefetched_tokens_total x 131,072 / wall`, cross-check iostat rkB/s; anything above ~0.68 GiB/s is page cache, not the SSD; the idle single-stream reference is 0.623 GiB/s (measured) |

Traps: `profile_stages` is a no-op unless `SGLANG_PROFILE_V2` (`profiler_manager.py:102,` `environ.py:435`);
`with_stack` defaults True via the tokenizer (`tokenizer_control_mixin.py:384-387`) so pass `false` or
traces balloon; the bench client mkdirs `output_dir/<time>` on the **client** side and sends the absolute
path (`serving.py:853-857`), fine only because client and server share the container filesystem;
`--profile-steps`/`--profile-num-steps` suppress the client's `/stop_profile` (`:1566-1580`). CUPTI capture
of the H2D stream on this A100 is expected but unverified until the first trace is opened.

---

## 9. What to record and report

Per cell (`results/<stamp>/<cell>/`): `startup_facts.txt`, `server.log`, `client.jsonl` (+`client.log`),
`metrics_before/after.txt` + `metrics_delta.json`, `memtime.csv`, `iostat.log`, `pcie.log`, `l3_stats.json`,
`config.json` (exact server + client args, seed, gap scale, patch sha), `summary.csv` row.

### 9.1 Report table per (arm, pressure, gap scale, c)

- TTFT p50/p95/p99 **per turn index**, binned also by queue depth at send time (`start_times` hunk joined to `memtime.csv` `num_queue_reqs`; the stock client never emits a send time, `serving.py:1883-1896`; HANDOFF §3 "rep index is confounded with queue depth"), reported only over the window where in-flight == N (why >= 5 x c conversations, §7.1), plus ITL p50/p99; paired (restore - control) differences with 95 % CI per bin (§6.5).
- Throughput at fixed concurrency of **sessions** (`--max-concurrency` = live conversations): completed turns/s, output tok/s (note input throughput is 0 for multi-turn: `serving.py:1614-1616`).
- Prefix hit rate: client (`cache_report`, but with the corrected per-turn denominator, §4.2) and server (`prefill_effective_tokens_total` deltas per mode).
- HiCache counters: `load_back_*`, `hicache_backup_*`, `evicted_tokens_total`, `hicache_dropped_tokens_total{reason}`, `prefetched/backuped/storage_prefetch_hit/unfulfilled{reason}`, L3 files/bytes, cleaner evictions if any.
- Restore-hit rate: fraction of returning turns device-only / host-restored / storage-restored / recomputed, and recompute tokens (§6.2).
- Memory-time: idle-session token-seconds per tier from the per-turn split (§6.4), normalized by device pool x wall; the gauge integral only as an upper bound; "freed" only between arms with equal pool sizes; and its cost: extra TTFT at p95 vs the paired control.
- Gains stated as throughput at a fixed TTFT SLO and memory-time freed; the short-gap regime (rows 2-3, real 0.71 s median) reported as the honest negative case.

### 9.2 Constants block (`results/<stamp>/constants.json`)

Pre-fill from campaign 5 (`source`: `R32/`; values = the §5.4 tables, measured unless tagged): `model` Qwen/Qwen3-32B-FP8 @ aa55da1e; `kv_cache_dtype` fp8_e5m2; `weight_kernel`
fp8 weight-only Marlin (W8A16); `attention_backend` triton; `mem_fraction_static` 0.85; `b` 131072; `P_marginal` 1226.3 tok/s (intercept -1.251 s); `recompute_ttft_s` {L: median,
7 lengths}; `recompute_quadratic` 0.0626 + 3.752e-4 L + 1.365e-8 L^2 (derived; HANDOFF: P is not constant); `recompute_bar_GiBps` 0.150; `L1_TOKENS` 281216; `host_pool_tokens`
762944 @ hicache-size 100; `L1/L2/L3_GiBps` 9.808 / 7.748 / 0.623 with intercepts 0.053 / 0.048 / 0.058 s (`wait_complete`, idle); `ssd_read/write_ceiling_GiBps` 0.68 / 0.38
(iostat); `page_cache_evict_method` drop_caches; `lba` 4096; **`decode_rate_tok_s` {1, 4, 8} and `extend_1k_on_24k_ttft_s`: null until measured (§5.4)**; then the measured turn
throughput and turn_time(c) of §7.3, and the go/no-go memory-time bar of §9.3.

### 9.3 Go/no-go (starter kit §0, made concrete)

- **Week 1 (characterize):** `fraction(1 s)` from §3.5 (closed loop, c = 8, stated turn-time model, both ends of the decode-rate estimate) plus absolute token-seconds vs the PH/PL device pool (131,072 tokens = 16.0 GiB). No-go if `fraction(1 s) < 10 %` of session-token-seconds: parking cannot free much, stop or change workload. (Not "pool-seconds": a trace-only computation has no pool.) With an estimated 6.6-9.9 s turn against a ~2.08 s mean gap (0.71 s median) f is ~0.8 at c=1 and ~0.9 at c=8: expect a small `fraction(1 s)` at scale 1 unless the gap tail is heavy; that number is the Week-1 result.
- **Weeks 2-3 (simulate):** with the measured tier costs (§5.4: restore(L) = 0.058 s + L x 131,072 / 0.623 GiB/s for L3, 0.048 s + L x 131,072 / 7.75 GiB/s for L2, recompute(L) from the measured curve), in the PL-like regime the deadline policy must free >= X % of idle-session memory-time (X fixed in `constants.json` before the runs; 20 % is the proposed bar) at an unchanged p95 TTFT SLO, over >= 30 simulated sessions per bin. "0.5 pp" is a hit-rate unit from the negative results (starter kit §0) and is not the criterion here; else no-go.
- **Serve (rows 4-5):** paired p50 (restore - control) TTFT < 0 with a 95 % CI excluding 0 over >= 30 pairs per (pooled, §6.5) turn-index bin, at median queue depth ~0 in the window, **and** >= X % idle-session memory-time freed vs the same-pool HBM-only arm; a "crossover" without a CI is a dead heat until shown otherwise (`HANDOFF.md:134`). A tier is admissible only if `bandwidth(tier) > b*P` (§5.4). Measured idle, single-stream for the 32B: L2 clears the bar 52x, L3 4.2x, and an L3 hit takes 0.24-0.54x of the recompute TTFT at all 7 lengths, so both tiers are admissible at idle; the open question for rows 4-6 is whether L3 still pays when concurrent restores share a ~0.68 GiB/s SSD and the `timeout` policy cuts them off. (The 8B is the counter-example on this box: L3 = 0.51x its bar, loses 7/7.)

---

## 10. Traps (one line each)

- Paired control, not idle baseline; salt control seeds per process; assert `cached_storage` is NaN on control rows (HANDOFF §3).
- `rm -rf $L3_DIR/*` silently fails past ~10k files, use `find -mindepth 1 -delete` (a 32B cell writes up to ~33k conversation pages plus ~10k control pages = ~85k files: 64 conv x 32.7K, 2 files per 8 MiB page, §5.3); `flush_cache` 400s until fully idle incl. HiCache queues (`scheduler.py:4779-4785`), poll the flush, never `hicache_backup_tokens_total`.
- `pkill -f` AND `pgrep -f <name>` match the shell that runs them when the pattern is on that shell's command line (`bash -lc '... pgrep -f sglang ...'` never sees zero; a wait loop did that on 2026-09-17): run them from a script file, or use `server.pid`, a marker file or a log line. `py-spy record -s` hangs on a ~100 GB-RSS scheduler (`py-spy dump --nonblocking` only).
- nixl L3 cleaner evicts at 80 %/70 % of the filesystem: sessions parked across long gaps are the oldest entries; there is no byte cap for nixl (`L3_MAX_SIZE` is file-backend only), lower the band via extra-config `l3_cleaner_high_watermark`/`_low_watermark` (§5.1); log `storage` per turn and treat cleaner evictions as a metric.
- NVMe LBA is 4 KiB: O_DIRECT needs every nixl file to be a multiple of 4096, and nixl L3 = 2 files per page (K, V): Qwen3-32B-FP8 fp8 KV, page 64: 8,388,608 B per page = 2 x 4,194,304 B, OK and confirmed (`O_DIRECT is active`); file counts in `l3_stats.json` are 2 x pages. The code never falls back on EINVAL (`nixl_utils.py:286-296`), so any other model/page size either passes that check or sets `SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0`; record `lsblk -o LOG-SEC /dev/nvme0n1` in `constants.json`.
- Working set must exceed L1+L2 or every arm ties `nohicache` (HANDOFF §5); P0 is the negative case, PL the L3 test. With the 32B, auto L1 + `--hicache-size 100` is 1,044,160 tokens (41.8 sessions of 25K): L3 is never read there, by construction (§5.3).
- Turn index == rep index == queue depth in a replay (bin TTFT by `num_queue_reqs` at send time via the `start_times` hunk, never by pool); a fixed `--num-prompts` lets the load drain before late turns (24 conv at c=12 is two waves), so use >= 5 x c conversations and report turn-index bins only over the window where in-flight == N.
- With gaps, in-flight HTTP requests << `--max-concurrency` (the intended "N live sessions" semantics, say so); warmup replays the whole first conversation (`serving.py:1431-1449`), keep `turn_meta` off the warmup input; `--agentic-gap-scale 0` == unpatched client, use it as the "no idle window" control.
- Starter-kit flag mismatches: `--hicache-storage-prefetch-timeout` does not exist; `page_first_direct`+`kernel` becomes `direct`; `--radix-eviction-policy fifo` is rejected; timeout defaults on the live unified path are 1.0/0.25/no max, not the dead HiRadixCache 2.0/0.1/30 (pass them explicitly, §5.1).
- `token_usage` excludes radix-cached KV (`pool_stats_observer.py:221-224`); the gauge integral is pool occupancy, ~full under LRU for every arm, so idle-session memory-time comes from the per-turn tier split (§6.4); gauges freeze while idle and refresh only per prefill batch / `--decode-log-interval` decode steps; stop the sampler before the post-run flush.
- `prefetched_tokens_total` counts raw L3->host volume before prefix dedup (use `prefill_effective_tokens_total{storage_hit}` for tokens used); `prefetch_bandwidth`/`backup_bandwidth` are empty for nixl and file, derive L3 BW from counters + iostat.
- Default nixl dir is `/tmp/hicache_storage` = 84 GB tmpfs here (O_DIRECT falls back to buffered): always set `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR`. The VM is stopped every night, so **the SSD is blank EVERY morning**: redo §2.2 (guarded mkfs + mount; the fstab line is already present, never append it again) before `docker start sglang_hicache`. `/mnt/nvme` is also wiped on host-maintenance restarts (`onHostMaintenance=TERMINATE`, `automaticRestart=TRUE`): on that boot `nofail` fails silently and nixl writes "L3" to the root PD with no error, so run the §2.2 gate before every `docker start` and the step-0b preflight before every cell.
- 167 GB RAM, no swap: `--hicache-size` <= 64. 100 DOES boot (762,944 tokens, all of 2026-09-17) but leaves ~8-28 GiB `free` (57-62 GiB available); the Claude Code harness then kills background Bash waiters ("system is running low on memory"): watch long runs through a marker file / log line (Monitor). With the 32B it also makes L3 unreachable (§5.3). 12 vCPU shared by scheduler, HiCache threads, tokenizer and client (plan §8.17 assumed 26): watch scheduler CPU in `top`, keep client concurrency <= 16 (the 32B saturates near c ~ 8, estimate), run a py-spy census once.
- `fa3` is unusable on this A100 (decode CUDA-graph capture fails: `scheduler_metadata must have shape (metadata_size)`); SM80 default is `flashinfer`; pin `--attention-backend triton` with fp8_e5m2 KV, the only measured combination. FP8 checkpoints run as weight-only FP8 Marlin (W8A16), automatically: P is 0.53x the H100's DeepGEMM path, and the first boot in a new container JIT-compiles for 10.6 min (readiness timeout >= 2400 s, `START_TIMEOUT_S`).
- Qwen3-8B cannot show an L3 benefit on this box (bar 1.22 GiB/s vs 0.625 delivered: an L3 hit is 1.8-2.9x SLOWER than recompute at 7/7 lengths): smoke tests only, never the tiering model.
- The first long prefill in a fresh container took 9.0 s (one-time warm-up): send one discarded long probe after creating a container and never recreate it between measured stages (`docker rm` also drops the warm JIT cache).
- L3 delivery is the SSD ceiling (~0.68 GiB/s read, ~0.38 GiB/s write), shared by all concurrent restores and backups; the default `timeout` budget (1 s + 4,096 tok/s) covers one idle 32B restore (5,104 tok/s) but not two overlapping ones (derived, §5.1).
- `hicache_eval/.current_results` points at a frozen campaign: any reused hicache_eval driver that sources `env.sh` refuses to run until `RESULTS` points at a new dir (`env.sh:6-12`). Never write agent_cache results into `hicache_eval/results/`.
- Docker group is live in new login shells (verified 2026-09-17); the toolkit stage restarts dockerd (kills containers) only when the `nvidia` runtime is not yet registered (it is); the setup script's printed `-v` path is right only with the uncommitted `setup-gpu-docker.sh:179` edit; container-written files are `root:root` on the host: the L3 bucket dirs (wipe as root), `agent_cache/results`, `agent_cache/traces`, `python/**/__pycache__`, `git apply` output and the HF cache (`chown -R wanhr:wanhr`).
- Never leave a patch under `python/` (HANDOFF §7; `91d480573` -> `2cb739b9a`); archive under `agent_cache/patches/` and `results/<stamp>/patches/`.
- `_normalize_round_messages` strips `tool_call_id`/`name` (convert tool messages to user text; bare `role: tool` is accepted at source level only, not run end to end, §3.3); empty `messages` turns are dropped by the loader and desync `pre_gap` (merge assistant-only iterations forward in the converter).
- AIPerf's idle guard compresses gaps > 10 s at low concurrency; keep c high or raise `--system-idle-gap-cap-seconds`.

---

## 11. Directory layout and git hygiene

```
agent_cache/
  agent-kv-tiering-evaluation-starter-kit.md   RUNBOOK.md (this)
  .gitignore            copy of dflash_eval/.gitignore (!*.log !*.csv !*.png !*.jsonl; root ignores them at .gitignore:62,173,182,187) + traces/*.parquet
  .current_results      bare stamp, e.g. 20260917_0900 (dflash convention, dflash_eval/scripts/run_track.sh:44-51); NOT a full path (hicache_eval/.current_results style)
  patches/              0001-agentic-trace-pre-gap.patch, 0002-eval-tier-log.patch (git apply / git checkout -- python/, §4.3)
  traces/               lmcache_agentic_trace.json (+ .stats.json), gaptest.json; converter output only
  scripts/
    env.sh              from dflash env.sh:7-29 (DOCKER/CONTAINER=sglang_hicache/PORT/BASE_URL, stamp -> RESULTS, readlink guard) + hicache env.sh, whose knobs now come from the environment:
                        L3_DIR (:4, default /var/hicache_l3), RESULTS (:5), frozen-dir guard (:6-12), MODEL (:15), NVME_DEV (:16, default vda), KV_BYTES_PER_TOKEN (:17, default 147456),
                        PAGE_SIZE=64 (:18), l3_wipe (:22). Set L3_DIR=/mnt/nvme/hicache_l3 NVME_DEV=nvme0n1 MODEL=Qwen/Qwen3-32B-FP8 KV_BYTES_PER_TOKEN=131072 as
                        run_a100_rerun_c2.sh:24-30,44-45 does; AGENT_CACHE_HOST=/home/wanhr/sglang/agent_cache, _CTR=/sgl-workspace/sglang/agent_cache
    models.sh           one primary entry qwen32b: MODEL=Qwen/Qwen3-32B-FP8, KV_BYTES_PER_TOKEN=131072, MODEL_EXTRA_ARGS="--kv-cache-dtype fp8_e5m2 --attention-backend triton --mem-fraction-static 0.85",
                        REASONING_PARSER=qwen3, L1_TOKENS=281216, P_TOK_S=1226, BAR_GIBPS=0.150, L2_GIBPS=7.75, L3_GIBPS=0.623, HOST_TOKENS_AT_100GB=762944 (all measured on this box, §5.4).
                        Do NOT copy hicache_eval/scripts/models.sh:21 (L1_TOKENS=284224 is the H100 value)
    arms.sh             arm_flags(): hbm_lru | hbm_host | three_tier | three_tier_p | hbm_lru_p | three_tier_wc | hbm_host_p (§5.2); arm_pressure(): P0 | PH | PL -> L1, HSIZE (§5.3)
    launch.sh           dflash launch.sh:15-69 skeleton (docker exec -d, append log, health loop) + 'fired up' grep and startup_facts from hicache start_server.sh:40-63, incl. its knobs
                        ATTENTION_BACKEND (:30-34), START_TIMEOUT_S (:41-43, default 900: use 2400), server_args.txt dump (:35), server.pid (:38); stop via server.pid or a script file
    bench.sh            one cell = §7.2; client runs INSIDE the container (host has no python tooling)
    run_arms.sh         dflash run_track.sh:43-57,75-93,96-134 skeleton: RUN/.current_results, DRY_RUN, preflight, cell_done resume, l3_wipe+droppc per cold arm, failures.log, cleanup trap; reuse
                        gate() (NVMe source + >= 110 GiB RAM + stale-server stop), wipe() (verified zero files), boot() (a failed boot ends the stage), postboot_hicache() (cleaner dir == L3 dir,
                        O_DIRECT line), health(), measured() from hicache_eval/scripts/run_a100_rerun_c2.sh:48-88; progress via marker lines ('STAGE DONE' / 'STAGE FAILED', :142), never via pgrep
    convert_lmcache.py  §3.3        analyze.py  per-turn TTFT bins, restore-hit split, memory-time integral, P fit (analyze_models.py:43-50)
    hcommon.py probe.py cachectl.py exp0.py exp1.py telemetry.sh   copy the CURRENT working-tree versions of hicache_eval/scripts (uncommitted; they carry the env knobs: hcommon.py HICACHE_FLUSH_TIMEOUT
                        :132-151, writeload.py --idx-offset :113 + SIGTERM stop :119, exp2.py NVME_DEV :31), changing only the sys.path / source-path constants (cachectl.py:5, probe.py:14,
                        telemetry.sh:5,9,11 -> 1 s). probe.py:49 fixes max_tokens=1: it cannot measure a decode rate (§5.4)
  results/<stamp>/      versions.txt constants.json b_measure/ p_measure/ traces/ server_<arm>.log <arm>_<trace>_c<N>/{...§9} summary.csv failures.log run.log patches/
```

Git hygiene: commits touch only `agent_cache/` (results tracked through the nested `.gitignore`); before
every commit `git checkout -- python/` and confirm `git status` shows nothing under `python/`; `chown -R wanhr:wanhr
agent_cache/results agent_cache/traces` first (container writes are root-owned, as are `python/**/__pycache__`); raw
datasets stay in `/home/wanhr/data/` (untracked, outside the repo); never commit `/mnt/nvme` contents.
Not reusable as workloads: `exp3.py`/`run_exp3.sh` (`bench_multiturn` synthetic rounds, skeleton only) and dflash `COMMON_*` (fa3, `--disable-radix-cache`).
