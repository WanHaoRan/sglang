# Agent KV tiering evaluation on the H200 box — runbook

Written 2026-09-16 for GCP VM `agent-cache` (a2-ultragpu-1g, 1x A100-SXM4-80GB); **ported 2026-09-21 to Nebius instance
`computeinstance-u00jtv5xqvxttejgvw` (1x H200, 143,771 MiB, SM90)**. The A100 text is archived verbatim at
`agent_cache/archive/RUNBOOK.20260921-a100-box.md` (and the pre-step-format text at `archive/RUNBOOK.20260918-before-step-format.md`).
The evaluation model is unchanged (`Qwen/Qwen3-32B-FP8`, `--kv-cache-dtype fp8_e5m2 --attention-backend triton`) so that the A100 campaign
stays the comparison baseline. Results dirs, abbreviated below: `R32/` = `hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv/` (A100 campaign 5,
2026-09-17), `R8/` = `hicache_eval/results/20260917_a100_gcp_qwen8b/`, `C18/` = `agent_cache/results/compare_20260918_final/` (the A100 three-arm
comparison of 2026-09-18, §7.0). Flags, paths and line numbers were checked against this checkout (`main` @ `d608a20d4`) and the live host on
2026-09-21; "Evidence:" gives the file:line or command. The plan being executed is `agent_cache/agent-kv-tiering-evaluation-starter-kit.md`.

**What the port changes, in one paragraph.** The box is different in four ways that matter and one that does not. (1) **The GPU is Hopper, not
Ampere:** SM90 runs the FP8 checkpoint on native FP8 tensor cores, not the A100's weight-only Marlin W8A16 fallback (`can_auto_enable_marlin_fp8()`
is `80 <= sm < 89`, `layers/quantization/fp8_utils.py:2126-2132`), so **every prefill- and decode-derived constant measured on the A100 is invalid
here** (§5.4) — `P`, the recompute bar `b x P`, the TTFT-vs-length curve, the tier fits, every cell time. (2) **HBM is 143,771 MiB, not 81,920:**
the profiled device pool roughly doubles (~700K tokens estimated, §5.3), so the old pressure pins are still available but no longer near the
ceiling. (3) **The L3 disk is a 2.27 TiB attached SSD at `/mnt/ssd`, not a 375 GiB ephemeral local NVMe at `/mnt/nvme`:** it sustains **1.88 GiB/s in
both directions** against the A100 NVMe's 0.68 read / 0.38 write, i.e. **2.8x on read and 5.0x on write** (measured, §1), is **persistent across
restarts** — so the daily format-and-mount step is gone — and is 6x larger, which makes the nixl cleaner's percentage watermarks nearly inert (§5.3). (4) **The box is bare:** no docker images, no containers, no HF cache, no datasets, no host
venv; §2 and §3 are now first-run procedures, not daily checks. What does **not** change: the model, the converted trace (`traces/`), the replay
client (`scripts/replay_agentic.py`, §4.6), the replay templates (`templates/`), the instrumentation (§6), the experiment design (§7) and every
`path:line` fact that is not GPU- or disk-specific.

**How to read this file.**
- Every step (each `###`, and each `##` that has no `###` children) carries the same fields in the same order: **Status** (done / not started / provisional / redo-every-time, with dates; never in the heading, so `§` references stay stable), **Goal** (the decision the step enables or the artifact it produces), **Runs on** (`host` or `container: <name>`, plus user and working directory when they matter), **Touches** (everything the step changes; `read-only` when nothing; a destructive step names what it destroys), **Takes** (wall time, and whether the GPU is busy), the command blocks, **Expected result** (checkable values, not "it works"), **If it differs** (the likely cause and the fix, when a failure has been seen), **Expected lessons** (what the outcome teaches, or the trap the step prevents).
- The first line of every command block says where it runs: `# host` is a login shell on the VM as `wanhr` (zsh with `interactivecomments` set, so comment lines paste cleanly); `# container: <name>` is a shell inside that container, opened with `docker exec -it <name> bash`; when a container command is shown from the host it uses the `docker exec -i <name> bash -c '...'` form.
- `-> prints:` after a block gives the output that means pass, and what another output means. A block that prints several values labels each one.
- Every number is tagged **measured** (date and results path), **derived** (formula and inputs) or **estimate** (and what would make it wrong). A number measured on the **A100** is tagged `measured-A100` and is a *prior*, never a value for this box: §5.4 lists exactly which ones must be re-measured before they may be used. Code facts carry `path:line` in this checkout (`/home/wanhr/sglang` @ `d608a20d4`).
- Paths: the host directory `/home/wanhr/sglang/agent_cache` is `/sgl-workspace/sglang/agent_cache` inside the container (the whole repo is bind-mounted at `/sgl-workspace/sglang`, so every repo path maps the same way). `/home/wanhr/data` is NOT visible from the container. `/mnt/ssd` is bind-mounted into the container at the same path.
- Placeholders look like `<ANGLE_BRACKETS>` and are defined next to their first use. Cross references are `§N.M`.

**State 2026-09-21 19:30 UTC** (checked with read-only commands; the instance has been up since 2026-09-21 18:17 UTC, GPU idle at 0 MiB, no containers):
- **Done on this box:** §2.2 (the attached SSD: `/dev/vdc` formatted ext4 `label=ssd` and mounted at `/mnt/ssd` 2026-09-21 19:20 UTC, `/mnt/ssd/hicache_l3` created and empty, `/etc/fstab:5` by UUID with `nofail`, throughput measured). Nothing else.
- **Carried over in the checkout (no work needed):** the converted trace `agent_cache/traces/lmcache_agentic_trace.json` (68,429,742 B, 731 conversations / 17,887 turns) and its `.stats.json`; the replay templates `agent_cache/templates/*.jinja`; the replay client `agent_cache/scripts/replay_agentic.py` and the `start_server.sh` / `start_client.sh` / `stop_server.sh` / `run_compare.sh` / `timeline.py` drivers; the HiCache event-log patch, which is now **committed** at `5b881454b` and therefore already in the working tree (§6.6, §11 — this is a change from the A100 box, where it was an unapplied patch file).
- **Not started:** §2.1 docker group; §2.3 image pull and container creation (no images, no containers); §2.4; §2.5; §2.6 (HF cache is empty); §3.1 (`/home/wanhr/data` does not exist); §3.5 (no `uv`, no venv, no `agentic-kv-cache` clone); §4.4 / §4.6 dry run; everything from §5 on. No GPU job has ever run on this box.
- **Daily:** nothing. The SSD is a persistent volume with an `fstab` entry, so there is no morning format-and-mount step; `§2.2` is now a one-line check. This is the single biggest procedural difference from the A100 box.
- **Numbers:** every GPU-derived constant in §5.4 is `measured-A100` and must be re-measured here before any row is sized (§5.4 lists the four commands). The disk constants ARE measured on this box (§1). The trace constants (mean gap 1.49 s, mean `output_length` 182.0, mean replay context 19.7K / 20.6K swebench) are properties of the trace file, not of the box, and carry over unchanged.

Inherited lessons: `hicache_eval/HANDOFF.md` §2-§7 (read it first), the A100 three-arm result `C18/README.md` (read it second: it is the only end-to-end
result this study has, and it is a null result with four named causes, §7.0), style from `dflash_eval/RUNBOOK.md`.

---

## 0. The short version

This section is the whole runbook for a reader who already knows the box: the plan in four phases, then the ordered list of everything that actually gets run, with where it runs and how long it takes. At its end the reader knows which step is next today and what blocks it.

**Status:** current as of 2026-09-21 19:30 UTC (matches the state paragraph above)
**Goal:** let a reader pick the next step and its cost without reading §2-§9.
**Runs on:** none (reading only)
**Touches:** read-only
**Takes:** none; GPU idle

**The plan in four phases** (the starter kit, made concrete for this box):
1. **Characterize (CPU-only, runs on this host today, §3.5):** `pre_gap` distributions per source/model from the LMCache parquet; `agentic-kv-cache` `make repro` on AgentX + Mooncake toolagent; memory-time behind gaps >= {1,5,30} s at concurrency {4,8,12,16} using b = 131,072 B/token (a model property, `measured-A100` but dtype-derived and unchanged here, §5.4). Decision: upper bound on what any parking policy can free.
2. **Simulate:** extend the simulator with tiers and per-tier restore cost from the **re-measured** constants of §5.4 (recompute is superlinear, so use the measured TTFT(L) curve, not L/P). Decision: go/no-go (§9.3).
3. **Serve (needs §2 bring-up):** SGLang + HiCache, the gap-faithful `replay_agentic.py` client (§4.6) on LMCache traces, then AIPerf/AgentX; sweep HBM pressure and tiers (§5, §7).
4. **Validate (later):** live agents (mini-swe-agent, BFCL) recording traces through a proxy.

**Run order** (one line per step; times tagged; "GPU busy" marks the steps that occupy the GPU; every GPU cell in steps 12-18 is one pass of the §7.2 checklist, whose launch block is the arm's §5.2 command and which pays a server boot — boot time on this box is **not yet measured**, budget the A100's 278-285 s warm / ~633 s JIT-cold until it is, §2.6):

Every working day, before anything else:
1. §2.2 one-line SSD check (`findmnt /mnt/ssd`) — host — ~1 s — GPU idle. **No format, no mount, no daily gate:** the volume is persistent.
2. §2.3 `docker start sglang_hicache`, then §2.4 block 4 (memlock / L3 mount source / shm / RAM seen from inside the container) — host — ~5 s (estimate) — GPU idle.

Once, the bring-up (nothing below is done on this box as of 2026-09-21):
3. §2.1 add `wanhr` to the `docker` group so `docker` works without `sudo` — host, sudo, needs a new login shell — ~10 s — GPU idle.
4. §2.3 pull `lmsysorg/sglang:nightly-dev-20260907-30705c00` and create `sglang_hicache` — host — 10-25 min (estimate: a ~20 GB image pull) — GPU idle.
5. §2.4 blocks 1-7 incl. the results stamp and `versions.txt` — host — ~15 s (estimate) — GPU idle.
6. §2.6 `hf download Qwen/Qwen3-32B-FP8` (32 GB onto `/`) — container — 5-30 min (estimate, HF throughput) — GPU idle.
7. §2.5 AIPerf venv at `/opt/aiperf` in the container — container — 2-5 min (estimate) — GPU idle. Only needed for §7.1 row 7.

Data and CPU-only work (GPU idle):
8. §3.1 downloads to `/home/wanhr/data` (2.95 GB) — host — ~3 min (measured-A100; network-bound) — GPU idle.
9. §3.3 converter — **not needed**: `agent_cache/traces/lmcache_agentic_trace.json` is committed and byte-identical to the A100 output; re-run only to change `--max-context`, `--sources`, `--tool-role-mode` or `--min-turns`.
10. §3.5 Week-1 characterization — host, `/home/wanhr/venv312` — `uv` + venv ~2 min, `make data` ~1 min, `make repro` ~40 min (measured-A100) — GPU idle.
11. §4.5 replay templates: already generated and committed (`agent_cache/templates/`); re-run `replay_template.py make && check` only after a model snapshot change — host — ~40 s (estimate) — GPU idle.

GPU jobs, in matrix order (§7.1); c, f, K, gap scales and cell times are re-derived after row 1:
12. §5.4 = **row 1, and the first GPU job on this box**: the full constant re-measurement (device pool from the boot log, b, P sweep, tier fits, decode rate, prefix-extension TTFT) — container, GPU busy — ~25 min + boot (estimate, §5.4). **Nothing after this may be sized until it has run.**
13. §4.6 dry-run block = row 0: gap-test dry run, arm (a) at P0, c=2 — container, GPU busy — ~1 min of client time + boot (estimate).
14. §7.1 row 2: arms (a),(b),(c) at P0, LMCache real gaps, c = 4 and 8 — 6 cells, GPU busy — cell times re-derived after row 1.
15. §7.1 row 3: arms (e),(g),(d) at PH, real gaps, c=8, overload reference — 3 cells, GPU busy — wall-time capped.
16. §7.1 row 4: (e),(g),(d) at PH, gap scale s_H, c=8, host-restore case — 3 cells, GPU busy.
17. §7.1 row 5: (e),(g),(d) at PL, gap scale s_L, c=12, L3 case — 3 cells, GPU busy.
18. §7.1 row 6: (d) vs (f), (g) at PL, `timeout` vs `wait_complete` — 2 more cells, GPU busy.
19. §7.1 row 7 (AgentX via AIPerf, needs step 7) and row 8 (Mooncake toolagent) — GPU busy — 35 min / wall-time-capped slice.
20. §8 profiling windows — optional, inside a cell, GPU busy — minutes each.
21. §9 report, `constants.json`, go/no-go — host, CPU.

**What is unblocked today (2026-09-21):** steps 3, 4, 5, 8, 10 (no GPU, no model needed). Step 6 (the 32 GB model download) can run in parallel with 8 and 10.
The first GPU job is step 12 (§5.4): it needs steps 3-6 and nothing else — not the trace, not the datasets, not AIPerf.

**Expected result:** the reader can say which numbered step is next (today: 3, then 4, with 8 and 10 in parallel) and what it costs.
**Expected lessons:** the GPU is idle for everything before step 12, and step 12 is not optional bookkeeping — it is the step that turns every
provisional figure in §3.5, §5.3, §7.1 and §7.3 from an A100 prior into a number for this box. On the A100 the equivalent step was never run, and
the one campaign that did run (`C18/`) ended in a three-way tie that could not be attributed, because the cell was sized from estimates (§7.0).

---

## 1. What is on this box

This section records the hardware, software and disk state every later step assumes, with the date it was checked and a re-check block per group so a reader on a different day can confirm it in under a minute. At its end the reader knows whether the box still matches the runbook, and what is missing from it.

**Status:** verified 2026-09-21 19:10-19:30 UTC with the blocks below (GPU idle, no server, **no containers and no images**)
**Goal:** confirm the box matches the assumptions behind §5, and make the gap between this box and the A100 explicit, so no `measured-A100` number is used here by accident.
**Runs on:** host, as `wanhr`; container rows through `docker exec` once §2.3 has run
**Touches:** read-only
**Takes:** ~30 s for all six blocks (estimate); GPU idle

| | |
|---|---|
| GPU | **NVIDIA H200, 143,771 MiB, compute capability 9.0 (SM90, Hopper), PCIe Gen5 x16, persistence mode on** (measured 2026-09-21). **Hopper, so unlike the A100:** the default MHA backend resolves to `fa3` (`arg_groups/model_override_base.py:323-329`), not `flashinfer`; and FP8 checkpoints run on **native FP8 tensor cores**, because Marlin auto-enable is gated on `80 <= sm < 89` (`layers/quantization/fp8_utils.py:2126-2132`), so the A100's weight-only FP8 Marlin (W8A16) path is NOT taken here. `fa3` + `fp8_e5m2` is still auto-rewritten to `triton` (`arg_groups/overrides.py:_attention_backend_fa3_fp8_fallback`), so the §5.2 `--attention-backend triton` pin now agrees with the automatic resolution instead of overriding it — keep it anyway, so the log is unambiguous. **Consequence: every prefill/decode constant in §5.4 is A100-only and must be re-measured (§5.4).** |
| Driver | **580.173.02**, CUDA 13.0. Secure Boot: not applicable (`mokutil` reports `EFI variables are not supported on this system`). No `nvidia_fs` module (no GDS). |
| Docker | docker-ce **29.8.0**, `Live Restore Enabled: false` (a dockerd restart kills containers), storage driver `overlay2`, root dir `/var/lib/docker` (on `/`). NVIDIA Container Toolkit 1.20.0 **already installed**, and `/etc/docker/daemon.json` **already registers the `nvidia` runtime** (`docker info` -> `Runtimes: nvidia runc io.containerd.runc.v2`), so §2.1 is a no-op check. `docker.service` **`LimitMEMLOCK=infinity`** (the A100 box had 8 MiB; `--ulimit memlock=-1:-1` in §2.3 is kept anyway, it costs nothing and makes the container independent of the daemon setting). **`wanhr` is NOT in the `docker` group** (`id -nG` -> `wanhr` only; the group exists, gid 986, and is empty): plain `docker` fails with `permission denied ... /var/run/docker.sock` until §2.1 runs. Passwordless `sudo` works. |
| Containers | **None. No images at all** (`docker images` is empty). §2.3 creates `sglang_hicache` from `lmsysorg/sglang:nightly-dev-20260907-30705c00`, the image every A100 measurement used, so the compiled pins stay comparable to `R32/` and `C18/`. There is no `sglang_dev` on this box and none is needed. |
| Host | **Ubuntu 24.04.4 LTS, kernel 6.11.0-1016-nvidia, 16 vCPU** (Intel Xeon Platinum 8468, 1 socket, 8 cores x 2 threads, **1 NUMA node**, GPU NUMA affinity node 0, CPU affinity 0-15), **196 GB RAM** (idle: 193 GB available), **no swap**, **Python 3.12.3 only** — and `python3` has **no `pip` and no `ensurepip`** (`python3 -m venv` exists but cannot bootstrap pip), so the §3.5 `uv` route is mandatory, exactly as on the A100 box but for a different reason. `iostat` is present; **`fio`, `uv`, `pip`, `hf`, `py-spy` are NOT**. `/dev/shm` is a **99 GB** tmpfs. **`/tmp` is NOT a tmpfs here: it is on `/` (`/dev/vda1`, ext4)** — the A100 box's "the nixl default `/tmp/hicache_storage` lands on tmpfs" trap therefore does not apply in the same form, but the default is still the wrong device (the root disk, not the SSD), so `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR` is still mandatory (§5.2). |
| Disks | **`/` = `/dev/vda1`, ext4 on a 1.3 TB virtio volume, 1.2 TB free** (measured 2026-09-21; holds docker images, the HF cache, `/home/wanhr/data` and the results). Single-stream `dd` O_DIRECT on it: 594 MB/s write, 457 MB/s read (measured 2026-09-21) — **use the SSD for L3, not this**. **`/dev/vdc` = the attached 2.27 TiB SSD (2,496,449,740,800 B), ext4 label `ssd`, LBA 512 B logical / 4096 B physical, mounted `/mnt/ssd` (`rw,noatime,discard`), `/mnt/ssd` owned `wanhr:wanhr`, `/mnt/ssd/hicache_l3` exists, root-owned, empty**; by-id `/dev/disk/by-id/virtio-disk-b8`; `/etc/fstab:5` holds the UUID line with `nofail` (`UUID=2feca542-a3be-4758-9855-860d3bbdc295 /mnt/ssd ext4 discard,defaults,noatime,nofail 0 2`). **Persistent** (a network-attached volume, not an ephemeral local SSD): it survives restarts, so there is no daily re-format (this is the A100 box's §2.2, deleted). Device ceilings, **measured 2026-09-21** (4 concurrent `dd` streams, 4 MiB blocks, O_DIRECT, with `iostat -x -d 2` sampling alongside; there is no `fio` on this box): the sustained steady state is **1,973,500 kB/s = 1,927 MiB/s = 1.88 GiB/s at 100 % `%util`, in BOTH directions** (5,300 IOPS of ~372 kB, `aqu-sz` 35-38); single stream reaches 946 MB/s read and 988 MB/s write at 1 MiB blocks, 1.6 GB/s write at 4 MiB. **Take the 1.88 GiB/s steady state as the ceiling, not the short-burst `dd` aggregates** (a 4 GiB-per-stream run reports up to 2,205 MiB/s because the first seconds are not yet at 100 % util). At b = 131,072 B/token that is **15,418 tok/s each way** (derived). For comparison the A100's local NVMe was 0.68 GiB/s read (5,571 tok/s) and 0.38 GiB/s write (3,113 tok/s): **this disk is 2.8x on read and 5.0x on write, and unlike that one it is symmetric**. |
| Repo | `/home/wanhr/sglang` @ **`d608a20d4`**, working tree **clean**. The three commits since the A100 runbook's base (`6ec32e6b7`) are `d4280b4ff` (a100 trials), `5b881454b` (agent evaluation) and `d608a20d4` (ttft figures); under `python/` they contain **only** the HiCache event-log eval patch (31 added lines tagged `# EVAL-PATCH` across `managers/cache_controller.py`, `mem_cache/unified_radix_cache.py`, `mem_cache/hybrid_cache/hybrid_cache_controller.py`, `utils/common.py`), which is therefore **already in the tree and needs no `git apply`** (§6.6, §11). `git apply --check -R agent_cache/patches/0002-hicache-event-log.patch` succeeds, which is how you confirm it. `hicache_eval/.current_results` points at the frozen A100 campaign (`.../20260917_a100_gcp_32b70b_fp8kv`, listed in `hicache_eval/.frozen_results`): every hicache_eval driver refuses to run until `RESULTS` points at a new dir (`hicache_eval/scripts/env.sh:6-12`). |
| HF cache | **Empty: `~/.cache/huggingface` does not exist.** §2.6 downloads `Qwen/Qwen3-32B-FP8` (32 GB) onto `/`. The 8B and the 70B AWQ of the A100 box are not needed (the 8B was the documented negative case there; on this GPU its bar changes too, so it says nothing until re-measured). |
| Datasets | **Absent: `/home/wanhr/data` does not exist**; §3.1 downloads 2.95 GB. The converted trace `agent_cache/traces/lmcache_agentic_trace.json` (68,429,742 B) IS present, committed, and needs no re-run. |
| Network | Outbound HTTPS works (`huggingface.co` and `github.com` both answer 200, checked 2026-09-21). |

**Which A100 constants survive the port.** Three groups:
- **Survive unchanged** (properties of the model, the dtype or the trace file, not of the GPU): `b` = 131,072 B/token (2 x 64 layers x 8 KV heads x 128 x 1 byte); the host-pool sizing rule; the converted trace's gap, output-length and context distributions (§3.3); the page/O_DIRECT arithmetic, except that this disk's logical sector size is 512 B, not 4,096, so the alignment constraint is strictly weaker (§2.2).
- **Survive as an upper-bounded prior:** the disk-side numbers, now superseded by this box's own measurements (§1 Disks row).
- **Do NOT survive, must be re-measured (§5.4):** `P` (1,226 tok/s), the recompute bar (0.150 GiB/s), the whole recompute TTFT-vs-length curve, the L1/L2/L3 tier fits, the device pool (281,216 tokens), the weight-load size, boot time, and both open constants (decode rate, prefix-extension TTFT). **Nine numbers.** Every one of them is Hopper-vs-Ampere sensitive, and three of them (`P`, the bar, the tier fits) decide whether L3 is admissible at all on this box (§9.3).

Re-check blocks (all read-only; values in `-> prints:` are the 2026-09-21 19:10-19:30 UTC readings):

```bash
# host. Block A: GPU, driver, compute capability, PCIe, GDS module
echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,compute_cap,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader)"
echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
echo "gpu mem used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "nvidia_fs module loaded: $(lsmod | grep -c '^nvidia_fs')"
```
-> prints `gpu: NVIDIA H200, 143771 MiB, 9.0, 5, 16`; `driver: 580.173.02`; `gpu mem used: 0 MiB` (a larger value means a server is running: find it before any measurement); `nvidia_fs module loaded: 0`. A `compute_cap` of `9.0` is the load-bearing value: it is what makes §5.4's A100 constants inapplicable and what selects the native-FP8 and `fa3` paths (§5.1).

```bash
# host. Block B: docker daemon, docker group membership, dockerd memlock, the measurement container
echo "docker: $(docker info 2>/dev/null | grep -E 'Server Version|Live Restore|Runtimes' | tr -s ' ' | paste -sd ';')"
echo "docker group in this shell: $(id -nG | tr ' ' '\n' | grep -cx docker)"
echo "dockerd memlock: $(systemctl show docker -p LimitMEMLOCK)"
echo "nvidia runtime in daemon.json: $(sudo grep -c '"nvidia"' /etc/docker/daemon.json)"
docker ps -a --format 'container: {{.Names}} | {{.Status}} | {{.Image}}' 2>/dev/null | grep sglang_hicache || echo "container: sglang_hicache NOT CREATED (§2.3)"
```
-> prints `docker: Server Version: 29.8.0; Runtimes: nvidia runc io.containerd.runc.v2; Live Restore Enabled: false` (the runtime order varies between calls; pass: `nvidia` present); `docker group in this shell: 1` **after §2.1 and a new login shell** (`0` today, and then the `docker info`/`docker ps` lines are empty because the socket is refused: that is the §2.1 symptom, not a broken daemon); `dockerd memlock: LimitMEMLOCK=infinity`; `nvidia runtime in daemon.json: 1`; and `container: sglang_hicache NOT CREATED (§2.3)` until §2.3 has run, then `container: sglang_hicache | Up ... | lmsysorg/sglang:nightly-dev-20260907-30705c00`.

```bash
# host. Block C: OS, RAM, swap, tmpfs sizes, host python and tools
echo "os: $(grep PRETTY_NAME /etc/os-release | cut -d= -f2), kernel $(uname -r), $(nproc) vCPU"
echo "ram total/available GiB: $(free -g | awk '/^Mem:/{print $2"/"$7}')"
echo "swap GiB: $(free -g | awk '/^Swap:/{print $2}')"
echo "shm size: $(df -h --output=size /dev/shm | tail -1 | tr -d ' ')"
echo "tmp device: $(findmnt -n -o SOURCE,FSTYPE -T /tmp | tr -s ' ')"
echo "host python: $(python3 --version)"
echo "host ensurepip: $(python3 -c 'import ensurepip' 2>&1 | tail -1 | cut -c1-40)"
for t in pip uv hf py-spy fio iostat; do echo "tool $t: $(command -v $t || echo none)"; done
```
-> prints `os: "Ubuntu 24.04.4 LTS", kernel 6.11.0-1016-nvidia, 16 vCPU`; `ram total/available GiB: 196/193` (available drops with a large host pool up, §5.3); `swap GiB: 0`; `shm size: 99G`; `tmp device: /dev/vda1 ext4` (**not tmpfs**: different from the A100 box); `host python: Python 3.12.3`; `host ensurepip: ModuleNotFoundError: No module named 'ens` (so `python3 -m venv` cannot make a usable venv: §3.5 uses `uv`); `tool pip: none`, `tool uv: none` (until §3.5), `tool hf: none`, `tool py-spy: none`, **`tool fio: none`** (the A100 box had it; disk numbers here come from `dd` + `iostat`), `tool iostat: /usr/bin/iostat`.

```bash
# host. Block D: root disk, the attached SSD (device, by-id link, mount, free space, L3 dir) and the fstab line
echo "root free: $(df -h --output=avail / | tail -1 | tr -d ' ')"
echo "ssd device: $(lsblk -dn -o NAME,SIZE,FSTYPE,LABEL,LOG-SEC,PHY-SEC /dev/vdc | tr -s ' ')"
echo "ssd by-id resolves to: $(readlink -f /dev/disk/by-id/virtio-disk-b8)"
echo "ssd mount: $(findmnt -n -o SOURCE,FSTYPE,SIZE,OPTIONS /mnt/ssd || echo NOT MOUNTED)"
echo "ssd free: $(df -h --output=avail /mnt/ssd | tail -1 | tr -d ' ')"
echo "l3 dir owner: $(stat -c '%U:%G' /mnt/ssd/hicache_l3 2>/dev/null || echo missing)"
echo "l3 files: $(find /mnt/ssd/hicache_l3 -type f 2>/dev/null | wc -l)"
echo "fstab ssd line: $(grep -n /mnt/ssd /etc/fstab)"
```
-> prints `root free: 1.2T`; `ssd device: vdc 2.3T ext4 ssd 512 4096`; `ssd by-id resolves to: /dev/vdc`; `ssd mount: /dev/vdc ext4 2.3T rw,noatime,discard`; `ssd free: 2.3T` (between cells; a cell leaves 2 files per page, §7.2); `l3 dir owner: root:root`; `l3 files: 0`; `fstab ssd line: 5:UUID=2feca542-a3be-4758-9855-860d3bbdc295  /mnt/ssd  ext4  discard,defaults,noatime,nofail  0  2`. `ssd mount: NOT MOUNTED` means the volume was detached or the `nofail` mount failed: `sudo mount -a` and re-run, and do not start the container until it prints the mount (§2.2). `ssd free` showing `99G` is `/dev/shm`, not the SSD.

```bash
# host. Block E: checkout and study state
echo "head: $(git -C /home/wanhr/sglang rev-parse --short HEAD)"
echo "working tree dirty files: $(git -C /home/wanhr/sglang status --short | wc -l)"
echo "files modified under python/: $(git -C /home/wanhr/sglang status --short -- python/ | wc -l)"
echo "event-log patch in tree: $(git -C /home/wanhr/sglang apply --check -R /home/wanhr/sglang/agent_cache/patches/0002-hicache-event-log.patch >/dev/null 2>&1 && echo yes || echo NO)"
echo "agent_cache stamp: $(cat /home/wanhr/sglang/agent_cache/.current_results 2>/dev/null || echo none)"
echo "trace bytes: $(stat -c %s /home/wanhr/sglang/agent_cache/traces/lmcache_agentic_trace.json 2>/dev/null || echo missing)"
echo "hicache_eval stamp: $(cat /home/wanhr/sglang/hicache_eval/.current_results)"
```
-> prints `head: d608a20d4`; `working tree dirty files: 0`; `files modified under python/: 0`; **`event-log patch in tree: yes`** (it is committed at `5b881454b`, so `git status` is clean *and* the patch is active — on the A100 box those two facts were mutually exclusive, §6.6); `agent_cache stamp: none` (until §2.4 block 6 writes one); `trace bytes: 68429742`; `hicache_eval stamp: /sgl-workspace/sglang/hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv` (frozen: never point an agent_cache run at it).

```bash
# host. Block F: the evaluation model in the HF cache (size, owner, snapshot)
D=/home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8
echo "hf cache dir: $([ -d /home/wanhr/.cache/huggingface/hub ] && echo present || echo MISSING)"
echo "model Qwen3-32B-FP8: $([ -d $D ] && echo "size $(du -sh $D | cut -f1), owner $(stat -c %U:%G $D), snapshot $(ls $D/snapshots | head -1 | cut -c1-8)" || echo MISSING)"
```
-> prints `hf cache dir: MISSING` and `model Qwen3-32B-FP8: MISSING` today; after §2.6, `hf cache dir: present` and `model Qwen3-32B-FP8: size 32G, owner root:root, snapshot aa55da1e` (the snapshot pin is `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`, the one every A100 measurement used).

**Expected result:** every `-> prints:` value above; in particular `compute_cap 9.0`, `driver 580.173.02`, `/mnt/ssd` on `/dev/vdc ext4` with `/mnt/ssd/hicache_l3` empty, `head: d608a20d4` with a clean tree and `event-log patch in tree: yes`, and — today — `docker group in this shell: 0`, no container, no HF cache and no `/home/wanhr/data`.
**If it differs:** `docker: ` empty and `docker ps` silent: §2.1 has not run (or the shell predates it); use `sudo docker` for one-off reads, but do §2.1 before §2.3. `ssd mount: NOT MOUNTED`: `sudo mount -a`; if that fails the volume is detached at the hypervisor and no L3 cell may run. `head` not `d608a20d4`: every `path:line` in this file must be re-checked before it is trusted; the four files the eval patch touches are the ones whose anchors shift first. `gpu:` naming anything but an H200: this runbook's §5.3/§5.4 do not describe that box — go back to the archive and port again.
**Expected lessons:** two facts cannot be recovered after the fact and are therefore checked here: the L3 store's device (a cell that ran against `/` instead of `/mnt/ssd` looks identical in every log, §2.2) and the container's compiled-dependency pins (a run in a different image is not comparable to `R32/` or `C18/`, §2.4 block 7). The third, new on this box, is `compute_cap`: an A100 constant used on SM90 produces a plausible-looking but wrong cell size, and nothing in any log would say so.

---

## 2. Bring-up

This section takes the box from bare to "able to serve the 32B with HiCache on the attached SSD": the docker group and the container toolkit (§2.1),
the SSD check (§2.2), the image and the container (§2.3), the in-container checks and the results stamp (§2.4), the AIPerf venv (§2.5) and the model (§2.6).
At its end `sglang_hicache` is running with the GPU, the repo bind, `memlock unlimited` and `/mnt/ssd/hicache_l3` on `/dev/vdc`, and the 32B is on disk.
**Everything in this section is a first run on this box**, except §2.2, which is already done. Status per step: §2.1 not started (the toolkit half is
already satisfied, the group half is not); §2.2 done 2026-09-21; §2.3-§2.6 not started.

### 2.1 Docker access and container toolkit

**Status:** the toolkit half is **already satisfied** on this box (NVIDIA Container Toolkit 1.20.0 installed, `nvidia` runtime registered in `/etc/docker/daemon.json`, verified 2026-09-21); the group half is **not started** (`wanhr` is not in the `docker` group)
**Goal:** make `docker` usable by `wanhr` without `sudo`, and confirm dockerd can hand the GPU to a container, so `--gpus all` in §2.3 and `nvidia-smi -L` in §2.4 work.
**Runs on:** host, as `wanhr` (needs sudo for the group change)
**Touches:** the `docker` group's member list (`/etc/group`). Nothing else: the toolkit is installed and the runtime is registered, so no `daemon.json` edit and **no dockerd restart**.
**Takes:** ~10 s, plus a new login shell for the group to take effect; GPU idle

Block 1 of 3: look first. These three values decide whether block 2 or block 3 is needed.
```bash
# host. Check both halves: is wanhr in the docker group, and is the nvidia runtime already registered?
echo "wanhr in docker group: $(id -nG wanhr | tr ' ' '\n' | grep -cx docker)"
echo "docker group members:  $(getent group docker | cut -d: -f4)"
echo "nvidia runtime entries in daemon.json: $(sudo grep -c '"nvidia"' /etc/docker/daemon.json)"
echo "nvidia-ctk: $(nvidia-ctk --version 2>/dev/null | head -1 || echo none)"
echo "gpu processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
```
-> prints, today, `wanhr in docker group: 0`, `docker group members:` (empty), `nvidia runtime entries in daemon.json: 1`, `nvidia-ctk: NVIDIA Container Toolkit CLI version 1.20.0`, `gpu processes: 0`. The `1` on the third line is what makes block 3 unnecessary: **do not run block 3 while that line is 1**, it would restart dockerd for nothing.

Block 2 of 3: add the group membership (the only change this step makes).
```bash
# host, sudo. Add wanhr to the docker group; the membership only reaches NEW login shells, so this shell still needs sudo afterwards
sudo usermod -aG docker wanhr \
  && echo "docker group members now: $(getent group docker | cut -d: -f4)" \
  && echo "this shell sees the group: $(id -nG | tr ' ' '\n' | grep -cx docker)   (0 is expected: log out and back in, or start a new shell)"
```
-> prints `docker group members now: wanhr` and `this shell sees the group: 0`. Open a new login shell (or `exec sg docker -c zsh` for a one-off), then `docker ps` must work without `sudo`; until then every `docker` line in this runbook needs a `sudo` prefix.

Block 3 of 3: the toolkit installer. **Not needed on this box** — kept because it is the recovery path if `daemon.json` ever loses the runtime.
```bash
# host. Re-run the toolkit stage (idempotent); the guard refuses when the runtime is ALREADY registered, because a
# re-run in that state is pointless, and the path that is not a no-op restarts dockerd and kills every container
[ "$(sudo grep -c '"nvidia"' /etc/docker/daemon.json)" -eq 0 ] || { echo "STOP: nvidia runtime already registered; nothing to do (this is the normal case on this box)"; false; } \
&& /home/wanhr/sglang/scripts/setup-gpu-docker.sh --skip-driver --skip-docker
```
-> prints `STOP: nvidia runtime already registered; nothing to do (this is the normal case on this box)` and runs nothing. If the guard ever passes, the script prints `Skipped (--skip-driver)` (`scripts/setup-gpu-docker.sh:257`), installs the toolkit and then **restarts dockerd** (`configure_docker_runtime`, `:459-466`): with live-restore off that kills every running container and any server in them.

**Expected result:** after block 2 and a new login shell, `id -nG | grep -c docker` prints `1` and `docker ps` prints a header with no error; `docker info | grep Runtimes` lists `nvidia`; `nvidia-smi` still shows 0 GPU processes (nothing was restarted).
**If it differs:** `docker ps` still says `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`: the shell predates the group change — open a new one; the group is on the *process*, not on the user, once the shell exists. Block 3 actually runs and prints `Registering the 'nvidia' runtime`: `daemon.json` was hand-edited and lost the runtime; dockerd restarts, so re-`docker start` the container afterwards.
**Expected lessons:** on this box the toolkit is pre-provisioned and the only missing piece is a group membership — a one-line change whose failure mode (`permission denied` on the socket) looks like a broken daemon and is not. The destructive path in this step is the dockerd restart, and it is triggered by the *content* of `daemon.json`, not by a flag, which is why block 1 reads that file before block 3 is offered at all.

### 2.2 The attached SSD for L3

**Status:** **done 2026-09-21 19:20 UTC** and **persistent**: `/dev/vdc` (2.27 TiB, hot-plugged 19:17:30 UTC) formatted ext4 label `ssd`, mounted at `/mnt/ssd`, `/mnt/ssd/hicache_l3` created, `/etc/fstab:5` UUID line with `nofail`, throughput measured. **There is no daily step.** This replaces the A100 box's format-and-mount-every-morning procedure, which existed only because that box's local NVMe was wiped at every VM stop.
**Goal:** give HiCache an L3 directory on the fast attached SSD, never on the root disk, and be able to prove in one line which device it is on before any cell.
**Runs on:** host, as `wanhr` (the one-line check needs no sudo; the recreate blocks need sudo)
**Touches:** nothing in the normal case (the check is read-only). The recreate blocks touch `/dev/vdc` (the format block DESTROYS everything on it), `/mnt/ssd`, `/mnt/ssd/hicache_l3` and `/etc/fstab`.
**Takes:** the check ~1 s; a full recreate ~1 min (measured 2026-09-21: `mkfs.ext4` with `lazy_itable_init=0` on 2.27 TiB took **19 s**); GPU idle

The daily reality on this box is one block:
```bash
# host. The only routine SSD check: is /mnt/ssd really /dev/vdc, and is the L3 dir there and empty?
echo "ssd mount:    $(findmnt -n -o SOURCE,FSTYPE,OPTIONS /mnt/ssd || echo NOT MOUNTED)"
echo "l3 dir on:    $(findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3 2>/dev/null || echo missing)"
echo "l3 files:     $(find /mnt/ssd/hicache_l3 -type f 2>/dev/null | wc -l)"
echo "ssd free:     $(df -h --output=avail /mnt/ssd 2>/dev/null | tail -1 | tr -d ' ')"
```
-> prints `ssd mount: /dev/vdc ext4 rw,noatime,discard`, `l3 dir on: /dev/vdc`, `l3 files: 0` (between cells) and `ssd free: 2.3T`. `l3 dir on: /dev/vda1` means `/mnt/ssd` is a plain directory on the root disk and the mount is gone: `sudo mount -a`, and run nothing until this block prints `/dev/vdc`.

Recreate blocks — **only** if the volume is replaced or the filesystem is lost. Block R1 of 3: look before touching anything.
```bash
# host. Check: is /mnt/ssd already mounted, does the by-id link point at /dev/vdc, and does the SSD already hold a filesystem?
DEV=/dev/disk/by-id/virtio-disk-b8
echo "mounted now: $(findmnt -n -o SOURCE /mnt/ssd || echo none)"
echo "by-id resolves to: $(readlink -f $DEV)"
echo "existing filesystem: $(sudo blkid -o value -s TYPE $DEV 2>/dev/null || echo none)"
```
-> prints, today, `mounted now: /dev/vdc`, `by-id resolves to: /dev/vdc`, `existing filesystem: ext4`: **nothing to do, do not run R2 or R3.** `mounted now: none` with `existing filesystem: ext4`: the fstab mount did not happen, run R3 only. `existing filesystem: none`: the volume is blank, run R2 then R3. `by-id resolves to:` anything other than `/dev/vdc`: STOP — the volume letter moved (a second attached disk); find out which device is the SSD before formatting anything, because the R2 guard compares against `/dev/vdc` by name.

Block R2 of 3: the destructive one. Every guard is re-evaluated inside the block, so pasting it on the wrong day does nothing.
```bash
# host, sudo. Format the attached SSD as ext4: DESTROYS everything on /dev/vdc. Runs only when nothing is mounted
# from /dev/vdc, the by-id link is /dev/vdc and blkid finds no filesystem; each failed guard prints STOP and the mkfs does not run
DEV=/dev/disk/by-id/virtio-disk-b8
{ [ -z "$(findmnt -n --source /dev/vdc)" ] || { echo "STOP: something is mounted from /dev/vdc; nothing to format"; false; }; } \
&& { [ "$(readlink -f "$DEV")" = /dev/vdc ] || { echo "STOP: $DEV is not /dev/vdc; refusing to format"; false; }; } \
&& { ! sudo blkid "$DEV" >/dev/null 2>&1 || { echo "STOP: $DEV already holds a filesystem; skip to block R3"; false; }; } \
&& sudo mkfs.ext4 -F -m 0 -L ssd -E lazy_itable_init=0,lazy_journal_init=0,discard "$DEV"
```
-> prints the `mke2fs` progress ending in `Writing superblocks and filesystem accounting information: done`; any `STOP:` line means nothing was formatted (the message says why). Took **19 s** on 2026-09-21 (measured: 609,484,800 4K blocks, 152,371,200 inodes, all inode tables written up front). The flags are the A100 box's, unchanged: label `ssd`, `-m 0` (no 5 % reserve — 116 GiB on a disk this size), no lazy init (so no first-write penalty during a measurement), `discard`.

Block R3 of 3: mount, prepare the L3 directory, and persist the mount (nothing is destroyed).
```bash
# host, sudo. Mount the SSD at /mnt/ssd, create the L3 directory, hand /mnt/ssd to wanhr, and add the fstab line
# by UUID with nofail if it is not already there; then show what is mounted
DEV=/dev/disk/by-id/virtio-disk-b8
sudo mkdir -p /mnt/ssd \
&& sudo mount -o discard,defaults,noatime "$DEV" /mnt/ssd \
&& sudo mkdir -p /mnt/ssd/hicache_l3 \
&& sudo chown wanhr:wanhr /mnt/ssd \
&& UUID=$(sudo blkid -o value -s UUID /dev/vdc) \
&& { grep -q "$UUID" /etc/fstab \
     || echo "UUID=$UUID  /mnt/ssd  ext4  discard,defaults,noatime,nofail  0  2" | sudo tee -a /etc/fstab >/dev/null; } \
&& echo "mounted:   $(findmnt -n -o SOURCE,FSTYPE,OPTIONS /mnt/ssd)" \
&& echo "size/free: $(df -h --output=size,avail /mnt/ssd | tail -1 | tr -s ' ')" \
&& echo "fstab:     $(grep -n /mnt/ssd /etc/fstab)"
```
-> prints `mounted: /dev/vdc ext4 rw,noatime,discard`, `size/free: 2.3T 2.3T` (measured 2026-09-21) and one `fstab:` line, `5:UUID=2feca542-a3be-4758-9855-860d3bbdc295  /mnt/ssd  ext4  discard,defaults,noatime,nofail  0  2`. `mount: ... already mounted` means R1 was misread; `mount: wrong fs type` means R2 did not run. Verify the fstab line with `sudo mount -a` (exit 0, no new mount): a bad line there is the one way this step can break the next boot, and `nofail` is what keeps a detached volume from blocking it.

**Why by-UUID and not by-id.** `/dev/disk/by-id/virtio-disk-b8` encodes the hypervisor's attachment slot, which changes if the volume is detached and re-attached; the filesystem UUID does not. The by-id link is still the right thing for the *format* guard (it is a name the operator chose deliberately), and the UUID is the right thing for `fstab`.

**No chown of `hicache_l3`:** the server (root in the container) pre-creates 256 root-owned bucket dirs `00..ff` and writes 0o644 files in them
(`python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py:257-266,277,298-303`; `nixl_routing.py:5-6`), so the L3 wipe runs as root (§7.2 step 1).
**O_DIRECT alignment is not a constraint on this disk:** logical sector size is **512 B** (`lsblk -o LOG-SEC /dev/vdc`), not the A100 NVMe's 4,096, so
every nixl file size is trivially a multiple of it. For the record the arithmetic still holds with room to spare: Qwen3-32B-FP8 + fp8_e5m2 KV at page 64
gives 64 x 131,072 = 8,388,608 B per page = 2 files (K and V) x 4,194,304 B. The code falls back to buffered I/O only when `O_DIRECT` is absent,
never on `EINVAL` (`nixl_utils.py:286-296`). Record `lsblk -o LOG-SEC /dev/vdc` in `constants.json` (§9.2) anyway: it is the value the claim rests on.

**Throughput, measured 2026-09-21** (`dd` O_DIRECT on `/mnt/ssd`, 4 MiB blocks, with `iostat -x -d 2 vdc` sampling alongside; there is no `fio` on this box):

| | sustained steady state (`iostat`, 100 % `%util`) | derived at b = 131,072 | single stream (`dd`) |
|---|---|---|---|
| read | **1,973,500 kB/s = 1,927 MiB/s = 1.88 GiB/s** | **15,418 tok/s** | 946 MB/s at 1 MiB blocks |
| write | **1,973,500 kB/s = 1,927 MiB/s = 1.88 GiB/s** | **15,418 tok/s** | 988 MB/s at 1 MiB, 1.6 GB/s at 4 MiB |

The device is **symmetric**, reaches 100 % `%util` at 4 concurrent streams (5,300 IOPS of ~372 kB, `aqu-sz` 35-38), and does not go faster at 8.
**Use the `iostat` steady state, not the `dd` aggregate:** a 4 GiB-per-stream `dd` run reports up to 2,205 MiB/s because its first seconds are below
100 % util, and an early measurement here overstated the read ceiling as 1.94 GiB/s for exactly that reason. The A100's local NVMe delivered
5,571 tok/s read and 3,113 tok/s write, so this disk is **2.8x** and **5.0x**.
These two numbers are what §5.1's prefetch-timeout budget and §5.3's write-through capacity argument are re-derived from, and they are the main
mechanical reason to expect a different answer from `C18/` on this box.

**Expected result:** the routine check prints `ssd mount: /dev/vdc ext4 rw,noatime,discard`, `l3 dir on: /dev/vdc`, `l3 files: 0`, `ssd free: 2.3T`; `/mnt/ssd` is `wanhr:wanhr` and `/mnt/ssd/hicache_l3` is `root:root` with 0 files.
**If it differs:** `findmnt /mnt/ssd` prints nothing: `sudo mount -a` (the `fstab` line is `nofail`, so a boot with the volume detached leaves `/mnt/ssd` as an empty directory on `/` with no error anywhere). `l3 dir on: /dev/vda1`: same cause, and it is the dangerous one — a container started in that state bind-mounts a root-disk directory and nixl `os.makedirs` the store there (`nixl_utils.py:229-231`), so every "L3" write lands on the wrong device, 4x slower, with no error in any log. `ssd free` far below 2.3T between cells: a previous cell's L3 store was not wiped (§7.2 step 1), or the cleaner never fired (§5.3).
**Expected lessons:** the A100 box's central daily risk — an ephemeral disk silently replaced by the root filesystem — is **structurally gone** here, because the volume is persistent and `fstab`-mounted; what remains is the same *failure mode* with a much lower probability, so the check survives as one read-only block at the top of §7.2 rather than as a hard gate that starts the container. The disk is also 2.8x (read) to 5.0x (write) faster than the one every A100 L3 number was taken on, which is why §5.4 re-measures the tier fits rather than scaling them.

### 2.3 Image and container

**Status:** **not started** — there are no images and no containers on this box (`docker images` and `docker ps -a` are both empty, checked 2026-09-21). The A100 box's `sglang_dev` spare is not recreated: it was never used for a measurement.
**Goal:** have one long-lived container whose compiled dependencies match this checkout's pins and the A100 campaign's, so that every 32B number stays comparable to `R32/` and `C18/`, and whose warm JIT cache is kept between measured stages.
**Runs on:** host, as `wanhr` (after §2.1; otherwise prefix every `docker` with `sudo`); shells inside the container via `docker exec -it sglang_hicache bash` (cwd `/sgl-workspace/sglang`, user root)
**Touches:** pulls ~20 GB into `/var/lib/docker` (on `/`, 1.2 TB free); creates the container; creates `~/.cache/huggingface` on the host
**Takes:** pull 10-25 min (estimate: image size over this box's network, not yet measured); `docker run` ~10 s (estimate); `docker start` ~2 s. A NEW container pays a JIT-cold first boot and a one-time first-long-prefill cost — **both `measured-A100` (10.6 min and 9.0 s, `R8/DEVIATIONS.md` D9) and not yet measured here**; budget them and record the real values in §2.6

**Which image.** `lmsysorg/sglang:nightly-dev-20260907-30705c00` — the image that ran every 2026-09-17 A100 measurement and the 2026-09-18 comparison. Its compiled deps match this checkout's pin (sglang-kernel 0.4.6.post1, `docker/Dockerfile:7`). The bind mount shadows only the `sglang` Python package; the compiled dependencies come from the image, which is why "which image" is a measured-constant question and not a preference (§2.4 block 7).

```bash
# host. Pull the measurement image (~20 GB into /var/lib/docker on /); skipped when it is already local
IMAGE=lmsysorg/sglang:nightly-dev-20260907-30705c00
docker image inspect "$IMAGE" >/dev/null 2>&1 && echo "image present: skip" \
  || { time docker pull "$IMAGE"; }
echo "image: $(docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep 30705c00 || echo MISSING)"
```
-> prints the pull progress and then `image: lmsysorg/sglang:nightly-dev-20260907-30705c00 <size>`. Record the pull time and size here after the first run (not yet measured on this box). A pull failure on this box is a network problem, not an auth one: the image is public.

```bash
# host, bash or zsh (both accept this array form). Create sglang_hicache; refuses if the name exists.
IMAGE=lmsysorg/sglang:nightly-dev-20260907-30705c00
mkdir -p "$HOME/.cache/huggingface"          # else dockerd creates it root-owned (dflash RUNBOOK 2.2)
RUN_ARGS=(
  # GPU, IPC, network, privileges
  --gpus all --ipc=host --network=host --privileged
  # limits: memlock unlimited for the pinned host pool; nofile for many L3 files
  --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576
  # bind mounts: repo, HF cache, the attached SSD (rshared so a host re-mount propagates)
  -v /home/wanhr/sglang:/sgl-workspace/sglang
  -v /home/wanhr/.cache/huggingface:/root/.cache/huggingface
  -v /mnt/ssd:/mnt/ssd:rshared
)
{ ! docker inspect sglang_hicache >/dev/null 2>&1 || { echo "STOP: sglang_hicache exists; use docker start, never recreate"; false; }; } \
&& docker run -itd --name sglang_hicache "${RUN_ARGS[@]}" "$IMAGE" /bin/zsh \
&& echo "created: $(docker ps --format '{{.Names}} {{.Status}}' | grep sglang_hicache)"
```
-> prints the new container id (64 hex chars) then `created: sglang_hicache Up <N> seconds`, or `STOP: sglang_hicache exists ...` on any later run.

```bash
# host. The daily check before `docker start` (read-only)
docker ps -a --format 'container: {{.Names}} | {{.Status}} | {{.Image}}' | grep sglang_hicache || echo "container: sglang_hicache MISSING"
```
-> prints `container: sglang_hicache | Up ... | lmsysorg/sglang:nightly-dev-20260907-30705c00`. `Exited (...)` after an instance restart is normal: `docker start sglang_hicache` (the container has RestartPolicy `no`). No line at all: the container is gone (`docker rm`), recreate it with the block above and budget the JIT-cold boot.

```bash
# host. Open a shell in the measurement container
docker exec -it sglang_hicache bash
```
-> prints a root prompt with cwd `/sgl-workspace/sglang`; `Error response from daemon: ... is not running` means it needs `docker start`.

`--ulimit memlock=-1:-1` is kept even though this box's dockerd already has `LimitMEMLOCK=infinity` (§1): the pinned host pool (`cudaHostRegister`) fails
without it on a box where the daemon limit is small (`hicache_eval/hicache_eval_plan.md:55-62`), and pinning it on the container makes the arm independent
of the daemon. `--ipc=host` shares the host's 99 GB `/dev/shm`. `:rshared` lets a host re-mount of `/mnt/ssd` propagate into a running container.
`/home/wanhr/data` is **not** bound: copy what the container must read under the repo bind (`agent_cache/traces/`, §3). The image's editable install is
shadowed by the bind mount (`docker/Dockerfile:639-646`, WORKDIR `:685`).

**Expected result:** `docker ps -a` lists `sglang_hicache` `Up`; `docker inspect -f '{{.HostConfig.RestartPolicy.Name}} {{range .HostConfig.Ulimits}}{{.Name}}={{.Soft}}:{{.Hard}} {{end}}' sglang_hicache` prints `no memlock=-1:-1 nofile=1048576:1048576`; inside it `pip list | grep sglang-kernel` shows `0.4.6.post1` (§2.4 block 7); `ls /root/.cache/sglang` is empty on a fresh container and fills on the first boot.
**If it differs:** `docker run` fails with `Conflict. The container name ... is already in use`: the guard was bypassed; nothing happened, use `docker start`. `docker: permission denied ... docker.sock`: §2.1 block 2 has not run, or this shell predates it. A container without `memlock=-1:-1`: the server dies at host-pool pinning; recreate it and pay the JIT-cold boot.
**Expected lessons:** comparability with `R32/` and `C18/` rests on the image's compiled packages, not on the bind-mounted `sglang` source, so the image tag is pinned to the A100 campaign's even though a newer nightly exists. The container is state (JIT cache in `/root/.cache/sglang`, the §2.5 `/opt/aiperf` venv), so `docker rm` is a measured cost per recreation, never a cleanup.

### 2.4 In-container checks

**Status:** **not started** (there is no container yet). Re-run block 4 after every `docker start`; run block 7 before §5.2.
**Goal:** prove that the container imports THIS checkout's `sglang`, sees the H200 through CUDA 13, has the nixl POSIX plugin, unlimited memlock and the SSD-backed L3 directory, and that the results directory every later step writes into exists with its version record.
**Runs on:** host, as `wanhr`, through `docker exec -i sglang_hicache ...`; blocks 6-7 also write on the host under `/home/wanhr/sglang/agent_cache` (= `/sgl-workspace/sglang/agent_cache` in the container)
**Touches:** read-only, except block 5 (writes the container's `/root/.gitconfig`), block 6 (creates `agent_cache/.current_results` only if absent, and `agent_cache/results/<stamp>/`) and block 7 (writes `agent_cache/results/<stamp>/versions.txt`)
**Takes:** ~15 s for all seven (estimate); GPU idle

```bash
# host. Block 1: the container sees the GPU
docker exec -i sglang_hicache nvidia-smi -L
```
-> prints `GPU 0: NVIDIA H200 (UUID: GPU-...)`; an NVML error or no line means the container lacks `--gpus all` (§2.3) or the toolkit is not wired into docker (§2.1).

```bash
# host. Block 2: python inside the container imports the bind-mounted checkout, CUDA 13 torch, and names the H200 with its compute capability
docker exec -i sglang_hicache python3 -c 'import sglang, torch; print("sglang file:", sglang.__file__); print("torch cuda:", torch.version.cuda); print("gpu:", torch.cuda.get_device_name(0)); print("capability:", torch.cuda.get_device_capability(0))'
```
-> prints `sglang file: /sgl-workspace/sglang/python/sglang/__init__.py` (the path MUST be under `/sgl-workspace`: anything under `/usr/lib` or `site-packages` means the bind mount is missing and the image's own copy would be measured), `torch cuda: 13.0`, `gpu: NVIDIA H200`, `capability: (9, 0)`. The `(9, 0)` is the line that selects the native-FP8 and `fa3` code paths (§1, §5.1).

```bash
# host. Block 3: the nixl plugin list contains POSIX (the L3 backend used by every measurement)
docker exec -i sglang_hicache python3 -c 'from nixl._api import nixl_agent, nixl_agent_config; a=nixl_agent("probe", nixl_agent_config(backends=[])); print("nixl plugins:", a.get_plugin_list())'
```
-> prints a list containing `POSIX` (`measured-A100`: `['AZURE_BLOB', 'GDS', 'GDS_MT', 'GPUNETIO', 'GUSLI', 'INFINIA', 'LIBFABRIC', 'OBJ', 'POSIX', 'UCX']`; it is a property of the image, so the same list is expected here). Pass = `POSIX` in the list (`python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py:29`; `nixl_utils.py:143`). GDS/GDS_MT are listed but unusable: no `nvidia_fs` module (§1 block A).

```bash
# host. Block 4: limits and mounts as the server will see them; RE-RUN AFTER EVERY docker start
docker exec -i sglang_hicache bash -c 'echo "memlock: $(ulimit -l)"; echo "l3 dir on: $(findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3)"; echo "shm size: $(df -h --output=size /dev/shm | tail -1 | tr -d " ")"; echo "mem total GiB: $(free -g | awk "/^Mem:/{print \$2}")"'
```
-> prints `memlock: unlimited`, `l3 dir on: /dev/vdc`, `shm size: 99G`, `mem total GiB: 196`. `l3 dir on: /dev/vda1` (or `/dev/root`): the container was started while `/mnt/ssd` was unmounted; `docker stop sglang_hicache`, fix the mount (§2.2), start it again. The `99G` belongs to `/dev/shm`, not to the SSD — each value is labelled for exactly that reason.

```bash
# host. Block 5: git inside the container trusts the bind-mounted repo (safe.directory) and sees the same working tree as the host
docker exec -i sglang_hicache git config --global --add safe.directory /sgl-workspace/sglang
echo "safe.directory: $(docker exec -i sglang_hicache git config --global --get-all safe.directory | paste -sd,)"
echo "status lines container: $(docker exec -i sglang_hicache git -C /sgl-workspace/sglang status --short | wc -l)"
echo "status lines host:      $(git -C /home/wanhr/sglang status --short | wc -l)"
```
-> prints `safe.directory: /sgl-workspace/sglang` (one entry, or the same entry repeated if `--add` ran more than once: harmless) and two equal counts, `0` on a clean tree (today). A `dubious ownership` error means the first line did not take.

```bash
# host. Block 6: create the results stamp ONCE (never overwrites an existing one) and its directory; absolute paths, no cd
AC=/home/wanhr/sglang/agent_cache
[ -s "$AC/.current_results" ] || date -u +%Y%m%d_%H%M > "$AC/.current_results"
mkdir -p "$AC/results/$(cat "$AC/.current_results")"
echo "stamp: $(cat "$AC/.current_results")"
echo "results dir: $(ls -d "$AC/results/$(cat "$AC/.current_results")")"
```
-> prints a fresh `stamp: 20260921_HHMM` on the first run (a bare stamp, not a path: §11) and its `results dir:`. **This box has no stamp yet** (`.current_results` does not exist), so the first run of this block creates one; do not delete it mid-study, every `RUNDIR` in §5.2 and §7.2 derives from it. Note that `agent_cache/results/` already holds the A100 run dirs (`20260918_*`, `compare_*`): those are a different box's results and are never written into again.

```bash
# host. Block 7: record the container's package versions in the results dir (the pins every measurement depends on)
AC=/home/wanhr/sglang/agent_cache
docker exec -i sglang_hicache pip list 2>/dev/null \
  | grep -Ei '^sgl|^sglang|flashinfer|^torch |nixl|^triton |transformers' \
  | tee "$AC/results/$(cat "$AC/.current_results")/versions.txt"
```
-> prints (and writes to `versions.txt`) the pins of the image: `sglang-kernel 0.4.6.post1`, `flashinfer-python 0.6.18`, `torch 2.13.0+cu130`, `nixl 1.4.1`, `triton 3.7.1`, `transformers 5.12.1`, plus `flashinfer-cubin`, `flashinfer-jit-cache`, `nixl-cu13`, `sgl-deep-ep`, `sgl-deep-gemm 0.1.7`, `sglang 0.0.0.dev1+g...`, `sglang-router` (13 lines, `measured-A100` on the same image). Pass = the six pinned versions match `docker/Dockerfile:7,17`, `python/pyproject.toml:6` and `R32/versions.txt` — the image is the same, so they must.

The package is named `sglang-kernel` (an `sgl-kernel` grep matches nothing). `sglang.__version__` differs per container (image metadata only).

**Expected result:** all seven `-> prints:` values; in particular block 2's `capability: (9, 0)`, block 4's `l3 dir on: /dev/vdc` and `memlock: unlimited`, and after block 7 a `versions.txt` whose six pinned versions equal `R32/versions.txt`.
**If it differs:** block 2 shows a path outside `/sgl-workspace`: the repo bind is missing, recreate the container (§2.3). Block 2 shows `capability: (8, 0)`: you are not on this box. Block 4 shows `memlock: 8192` (KiB): the container was created without `--ulimit memlock=-1:-1`; the host pool will fail to pin. Block 7 shows a different `sglang-kernel`: the image tag is not the pinned one and nothing measured is comparable to `R32/` or `C18/`.
**Expected lessons:** the bind mount makes source edits visible instantly but cannot change compiled packages, so "which image" is a measured-constant question; `versions.txt` in the results dir is what lets a later reader tell whether a number is comparable across the two boxes. And because the image is deliberately the A100 campaign's, `versions.txt` should be **identical** across the boxes — every difference between this box's numbers and `R32/` is then attributable to hardware, which is the whole point of pinning it.

### 2.5 Python extras

**Status:** not started (no container yet, so no `/opt/aiperf`). Needed only for §7.1 row 7 (AgentX via AIPerf); everything else in this runbook runs on the image's own Python.
**Goal:** make `aiperf` runnable inside `sglang_hicache` without touching the image's pinned packages.
**Runs on:** `container: sglang_hicache` (shown from the host with `docker exec`); needs network access to PyPI
**Touches:** `/opt/aiperf` in the container's writable layer (survives `docker stop/start`, lost on `docker rm`)
**Takes:** 2-5 min (estimate: pip download and install of aiperf and its dependencies; never yet run on either box); GPU idle

```bash
# host. Create an isolated venv in the container and install aiperf into it, then show its help
docker exec -i sglang_hicache bash -c 'python3 -m venv /opt/aiperf && /opt/aiperf/bin/pip install -q -U pip aiperf && /opt/aiperf/bin/aiperf --help | head -20'
```
-> prints the `aiperf` usage text (first 20 lines); pass = a `profile` subcommand is listed (the entry-point name is unverified until this runs: §3.4 uses `aiperf profile`).

Already in the image: datasets, pyarrow, pandas, huggingface_hub with `hf`, `py-spy`, `iostat`, and `uv` at `/opt/sglang/bin/uv`. `aiperf` declares
`requires_python <3.14,>=3.11`, so unlike on the A100 box (host Python 3.14) **this box's host Python 3.12.3 could host it** — but the host has no `pip`
and no `ensurepip` (§1), so the container venv remains the path of least resistance and keeps the image's pins isolated from aiperf's `~=` constraints.
`huggingface-cli` is a dead shim; use `hf download`.

**Expected result:** `docker exec -i sglang_hicache ls /opt/aiperf/bin/aiperf` prints the path; `aiperf --help` lists `profile`; `docker exec -i sglang_hicache pip list | grep -c aiperf` prints `0` (the image's own environment is untouched). Installed version: not yet measured on either box; record it here after the run.
**If it differs:** `aiperf --help` has no `profile` subcommand: the §3.4 command line was written from NVIDIA's tutorial without running it; read the printed help and fix §3.4 before row 7. The install downgraded something in the image: the venv was created with `--system-site-packages` by mistake; `rm -rf /opt/aiperf` in the container and rerun the block as written.
**Expected lessons:** keeping AIPerf in its own venv is what protects the measured pins of §2.4 block 7; the isolation matters more than which Python hosts it.

### 2.6 Model

**Status:** **not started — the HF cache does not exist on this box.** §2.6 is a 32 GB download.
**Goal:** have the evaluation model (`Qwen/Qwen3-32B-FP8`, snapshot `aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`) on disk, and record what it costs to boot on **this** GPU — a number the A100 measured and this box has not.
**Runs on:** host (`du`, `chown`); the download runs inside `sglang_hicache` (it has `hf`; the host does not)
**Touches:** writes 32 GB under `~/.cache/huggingface/hub` on `/` (1.2 TB free); the `chown` block changes the cache's owner to `wanhr:wanhr`
**Takes:** download 5-30 min (estimate, HF throughput; not yet measured on this box); GPU idle

```bash
# host. Check whether the evaluation model's snapshot is present (read-only)
S=/home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
echo "32B snapshot dir: $([ -d "$S" ] && echo present || echo MISSING)"
echo "32B safetensors files: $(ls "$S"/*.safetensors 2>/dev/null | wc -l)   (expect 7)"
echo "hf cache size: $(du -sh /home/wanhr/.cache/huggingface 2>/dev/null | cut -f1 || echo none)"
```
-> prints `32B snapshot dir: MISSING`, `32B safetensors files: 0   (expect 7)` and `hf cache size: none` today; after the download, `present`, `7` and `32G`.

```bash
# host. Download the 32B ONLY when the snapshot is missing (guarded; 32 GB onto /, inside the container because the host has no hf)
S=/home/wanhr/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df
{ [ ! -d "$S" ] || { echo "STOP: snapshot already present, nothing to download"; false; }; } \
&& time docker exec -i sglang_hicache bash -c 'hf download Qwen/Qwen3-32B-FP8'
```
-> prints the `hf download` progress and finally the snapshot path `/root/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/aa55da1ecc13d006e8b8e4f54579b1ea8c3db2df`. Record the wall time here after the first run. Watch from another host shell with `du -sh ~/.cache/huggingface`; stop with Ctrl-C (`hf download` resumes).

```bash
# host, sudo. Give the HF cache back to wanhr after a container-side download (container writes are root:root on the host)
sudo chown -R wanhr:wanhr "$HOME/.cache/huggingface" \
&& echo "hub owner: $(stat -c %U:%G "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8)"
```
-> prints `hub owner: wanhr:wanhr`. Not required for serving (the server runs as root in the container); required before any host-side write into the cache.

**Why this model, still.** `Qwen/Qwen3-32B-FP8` + `--kv-cache-dtype fp8_e5m2 --attention-backend triton` is the configuration of every A100 number
(`R32/`) and of the only end-to-end comparison this study has (`C18/`), so keeping it is what makes the two boxes comparable. Note what changes
underneath it: on SM90 the checkpoint runs on **native FP8**, not the A100's weight-only FP8 Marlin (W8A16) — the boot log will say so, and it is the
single change most likely to move `P` (§5.4). The A100's model-choice argument ("the 32B clears the L3 admission bar 4.2x, the 8B loses at 7/7 lengths")
is a statement about that GPU's `bar = b x P` against that disk's 0.62 GiB/s: **both sides of it change here** and it must be re-derived in §5.4 before
it is quoted again.

**Expected result:** the check block prints `present` and `7` safetensors, and `hf cache size: 32G`; a §5.2 boot of the 32B reaches `The server is fired up and ready to roll!` (time **not yet measured on this box**; `measured-A100`: 278-285 s warm, ~633 s JIT-cold) and logs a `Load weight end. ... quant=fp8 ...` line. **Record here, after the first boot:** the weight-load size, whether the log says Marlin (it should NOT on SM90), the JIT-cold boot time and the warm boot time. Those four values replace the A100's in §5.4 and §7.3.
**If it differs:** the boot log contains `Weight-only FP8 compression will be used leveraging the Marlin kernel`: something forced it (`SGLANG_FORCE_FP8_MARLIN` in the environment), because `can_auto_enable_marlin_fp8()` is false at SM90 (`fp8_utils.py:2126-2132`); unset it — the run would be measuring the A100's kernel on Hopper silicon. A boot far longer than 633 s: the JIT cache is cold (a new container); expected once per container, and the readiness timeout must be >= 2400 s (`START_TIMEOUT_S`, §5.2).
**Expected lessons:** the model is held fixed across the port precisely so that the GPU is the only variable; the boot is where that variable first shows itself, so the four values above are recorded before any cell rather than after.

---

## 3. Data

This section gets the three trace corpora onto the box, turns the LMCache parquet into the JSON the `agentic-trace`
client replays, and extracts the Week-1 numbers (gap distribution, output lengths, memory-time bound) that size every
later step. At its end the reader has `/home/wanhr/data/` (raw downloads, root disk, outside git),
`agent_cache/traces/lmcache_agentic_trace.json` (+ `.stats.json`; git-tracked JSON, parquet ignored by
`agent_cache/.gitignore`), a host venv for CPU-only analysis, and the `agentic-kv-cache` reproduction results.

Disk budget (measured 2026-09-17/18, `ls -l` / `du -sh` on the host): LMCache parquet 2.37 GB (2,370,186,947 B) +
AgentX-256k 0.57 GB (568,864,747 B) + Mooncake 13 MB + converted JSON 65 MiB (68,429,742 B); the AgentX full corpus
(1.85 GB) is not downloaded. The model (32 GB, §2.6) and the docker image (~20 GB, §2.3) share the same disk; `/` had **1.2 TB free on 2026-09-21** (measured, `df -h /`, §1 Disks row and block D), so none of this is tight here.
`/home/wanhr/data` is **not bind-mounted** into either container (§2.3), so anything a container must read is copied:
the converted trace lives under `agent_cache/traces/` (repo bind = `/sgl-workspace/sglang/agent_cache/traces/`),
the parquet was `docker cp`'d to the container's `/tmp/lmcache` (§3.3), the Mooncake jsonl is copied under
`agent_cache/traces/` before §3.4.

**State 2026-09-21 (this box):** §3.1 **not started** (`/home/wanhr/data` does not exist); §3.2 verified against `d608a20d4`;
§3.3 **needs no run** — its output `agent_cache/traces/lmcache_agentic_trace.json` (68,429,742 B) and `.stats.json` are committed and
carried over unchanged, because the trace is a property of the parquet and the tokenizer, not of the box; §3.4 not run; §3.5 **not started**
(no `uv`, no venv, no `agentic-kv-cache` clone). The A100 box had all of this; none of it survived the port, except the two files under `traces/`.
Only §3.1 and §3.5 actually need running here, and only §3.5 for the Week-1 gate.

### 3.1 Downloads

**Status:** **not started on this box** (`/home/wanhr/data` does not exist, checked 2026-09-21); `measured-A100`: the same four downloads took 3 min there. `wget -c` resumes and skips complete files, so the block is safe to re-run
**Goal:** put the LMCache parquet (the converter's input, §3.3), the Mooncake traces (§3.4, §3.5) and the AgentX-256k corpus (§3.5) on the root disk, verified complete, so no later step waits on the network.
**Runs on:** host, as `wanhr`; plain `wget`, no HF CLI needed
**Touches:** creates `/home/wanhr/data/{lmcache,mooncake,agentx}` (~2.95 GB total, on the root disk `/dev/vda1`, outside git); network
**Takes:** ~3 min (measured 2026-09-17 from mtimes 23:10-23:12 UTC); GPU idle

```bash
# host: create the raw-data tree (root disk, outside the repo; never committed)
mkdir -p /home/wanhr/data/{lmcache,mooncake,agentx}
```
-> prints nothing; `ls -d /home/wanhr/data/*` then lists the three dirs.

```bash
# host: LMCache agentic traces, 5 parquet shards + dataset card (2.37 GB; wget -c resumes/skips)
cd /home/wanhr/data/lmcache \
  && for i in 0 1 2 3 4; do
       wget -c "https://huggingface.co/datasets/sammshen/lmcache-agentic-traces/resolve/main/data/train-0000${i}-of-00005.parquet"
     done \
  && wget -c https://huggingface.co/datasets/sammshen/lmcache-agentic-traces/raw/main/README.md
```
-> prints wget progress; pass when the 5 shards total 2,370,186,947 B (sizes in Expected result).

```bash
# host: Mooncake FAST'25 traces (13 MB); the arxiv trace is renamed mooncake_trace.jsonl
cd /home/wanhr/data/mooncake \
  && B=https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release \
  && wget -c $B/traces/toolagent_trace.jsonl $B/traces/conversation_trace.jsonl $B/traces/synthetic_trace.jsonl \
  && wget -c -O mooncake_trace.jsonl $B/arxiv-trace/mooncake_trace.jsonl
```
-> prints wget progress; pass when `wc -l toolagent_trace.jsonl` is 23,608.

```bash
# host: AgentX 256k-capped corpus (569 MB), then a size gate: HF Content-Length is 568864747 B
cd /home/wanhr/data/agentx \
  && wget -c -O traces-062126-256k.jsonl https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k/resolve/main/traces.jsonl \
  && { [ "$(stat -c %s traces-062126-256k.jsonl)" -eq 568864747 ] \
       || { echo 'STOP: AgentX TRUNCATED (HF Content-Length 568864747): re-run wget -c before 3.5'; false; }; }
```
-> prints wget progress and nothing else; the STOP line means the file is short (see If it differs).

```bash
# host: verify every download by size (labelled), read-only
echo "lmcache parquet bytes: $(ls -l /home/wanhr/data/lmcache/*.parquet | awk '{s+=$5} END{print s}')   (expect 2370186947)"
echo "lmcache shards:        $(ls /home/wanhr/data/lmcache/*.parquet | wc -l)   (expect 5)"
echo "mooncake toolagent:    $(wc -l < /home/wanhr/data/mooncake/toolagent_trace.jsonl) lines   (expect 23608)"
echo "agentx bytes:          $(stat -c %s /home/wanhr/data/agentx/traces-062126-256k.jsonl)   (expect 568864747)"
```
-> prints the four labelled lines; pass when every value equals its `(expect ...)`.

**Expected result:** (measured 2026-09-17, `ls -l` on the host) `/home/wanhr/data/lmcache/`: `train-0000{0..4}-of-00005.parquet` =
502,376,957 / 555,802,662 / 476,154,555 / 437,652,845 / 398,199,928 B (sum 2,370,186,947 B = 2.37 GB) + `README.md` 9,920 B.
`/home/wanhr/data/mooncake/`: `toolagent_trace.jsonl` 4,415,857 B (23,608 lines), `conversation_trace.jsonl` 3,029,533 B,
`synthetic_trace.jsonl` 1,136,313 B, `mooncake_trace.jsonl` 4,446,104 B (23,608 lines). `/home/wanhr/data/agentx/traces-062126-256k.jsonl`
568,864,747 B (393 lines, one session tree per line).
**If it differs:** AgentX smaller than 568,864,747 B: the download was cut (HF serves it in one stream); re-run the `wget -c` line, it resumes.
A truncated AgentX is the one failure that surfaces late: `make data` in §3.5 skips existing files and `prep.py` then crashes on JSON parse.
**Expected lessons:** the Mooncake URLs are exactly what `bench_serving` auto-downloads (`python/sglang/benchmark/datasets/common.py:13-18`),
but its default cache path is `/tmp/<workload>_trace.jsonl` (`datasets/mooncake.py:33-36`); on this box `/tmp` is on the root disk, not a tmpfs
(§1 block C), so that default merely writes to the wrong disk instead of to RAM — pass `--dataset-path` to the copy under `agent_cache/traces/` anyway
(§3.4); and because `/home/wanhr/data` is not bound into the container, every container-side consumer reads a copy, never this directory.

### 3.2 What the `agentic-trace` loader actually reads (this checkout)

**Status:** verified 2026-09-17 against `main` @ `6ec32e6b7`; anchors re-checked 2026-09-21 against `d608a20d4` (all resolve: the three commits in between touch nothing under `python/sglang/benchmark/`)
**Goal:** know exactly which per-turn keys the stock loader and client consume, so the converter (§3.3) emits what is read, the analysis does not rely on a field that is silently ignored, and the §4 patch targets the right lines.
**Runs on:** host, read-only (`sed -n` on the repo)
**Touches:** read-only
**Takes:** ~2 min of reading; GPU idle

```bash
# host: print the loader lines the facts below cite (read-only)
sed -n '70,100p' /home/wanhr/sglang/python/sglang/benchmark/datasets/agentic_trace.py
```
-> prints the `load()` body: line 85 `prompt = [turn["messages"] for turn in conversation if turn.get("messages")]`,
line 92 `prompt_len = int(conversation[0].get("prompt_tokens", 0))`; if those lines moved, re-anchor every `:NN` below.

**Expected result:** the following facts hold at the cited lines (Evidence: `python/sglang/benchmark/datasets/agentic_trace.py:70-100`):
- `data["conversations"]` is a list (empty -> `ValueError`, `:70-72`); `metadata` is never read.
- Per turn only `turn["messages"]` is read (`:85`); a falsy `messages` drops the turn **silently**. `conversation[0]["prompt_tokens"]` is informational only (`:92`).
- **Every other per-turn key (`pre_gap`, `output_length`) is ignored** today: `output_len` is one value per conversation =
  `--sharegpt-output-len` or `DEFAULT_AGENTIC_OUTPUT_LEN` = 220 (`:14`, `:78`, `:98`). Rotation `--dataset-offset` and cap `--agentic-max-turns` (`:74-87`).
- At request time each round is rebuilt as `{role, content}` only (`serving.py:1294`): `tool_calls`, `tool_call_id`, `name` are stripped.
  Multi-turn is detected by `prompt[0]` being a `str` or a `List[{role,content}]` (`serving.py:1377-1381`), so per-turn metadata **cannot** ride inside `prompt`.
- The load line the client prints is `#Conversations: N (offset=..., turns/conv min=... max=... avg=...)` (`:107-112`).
**If it differs:** none known; a moved line means the checkout changed: re-anchor before editing §4.1's hunks, which quote these line numbers.
**Expected lessons:** the stock client has no idle window (gaps ignored) and one output length per conversation, so a gap-faithful replay needs the
§4 patch (a `turn_meta` side channel, not fields inside `prompt`); and because empty `messages` turns vanish silently, the converter must never
emit one, or `pre_gap` desyncs from the turn index (§3.3 checks the invariant `empty_messages_turns = 0`).

### 3.3 Converter: LMCache parquet -> agentic-trace JSON (`agent_cache/scripts/convert_lmcache.py`)

**Status:** built and run on the A100 box 2026-09-18; **its output is committed and carried over, so there is nothing to run here.** `agent_cache/traces/lmcache_agentic_trace.json` (68,429,742 B) and `lmcache_agentic_trace.json.stats.json` (4,276 B) are in the checkout and were verified present 2026-09-21. Re-run only to change `--max-context`, `--sources`, `--tool-role-mode` or `--min-turns` — and note that would require §3.1 first (the parquet is not on this box)
**Goal:** produce the one replayable trace for §7: real message deltas with real per-turn `pre_gap` and `output_length`, truncated to what a 32K-context server can replay, plus a `.stats.json` whose numbers feed §5.3 (live context, mean gap, mean output length).
**Runs on:** `container: sglang_hicache` (pyarrow, transformers and the Qwen tokenizer files are there; the host has no pip); cwd `/sgl-workspace/sglang/agent_cache/scripts` = host `/home/wanhr/sglang/agent_cache/scripts`; the chown runs on the host
**Touches:** reads `/tmp/lmcache` in the container (a `docker cp` of the parquet, 2.37 GB in the container's overlay layer: survives `docker stop/start`, lost on `docker rm`); writes `agent_cache/traces/lmcache_agentic_trace.json` and `.stats.json` (root-owned on the host until the chown); RAM about 10 GiB of Python objects plus the tokenizer (`convert_lmcache.py:33-35`)
**Takes:** 13.5 min (`measured-A100` by the run itself; the two chat-template token counts per turn dominate; 12 fork workers on that box's 12 vCPU — use `--workers 16` here, §1). GPU idle, but do not run it while a measured cell is in flight (the client and scheduler share the same 16 vCPU, §10)

```bash
# host: put the parquet where the converter's default --parquet-dir expects it (2.37 GB into the container's overlay; skipped when already there)
docker exec sglang_hicache test -f /tmp/lmcache/train-00004-of-00005.parquet \
  || { docker exec sglang_hicache mkdir -p /tmp/lmcache \
       && docker cp /home/wanhr/data/lmcache/. sglang_hicache:/tmp/lmcache; }
echo "container parquet shards: $(docker exec sglang_hicache bash -c 'ls /tmp/lmcache/*.parquet | wc -l')   (expect 5)"
```
-> prints `container parquet shards: 5`; verified 2026-09-18 (`/tmp/lmcache` in `sglang_hicache` holds the 5 shards + README, copied 2026-09-17 23:52).

```bash
# container: sglang_hicache, cwd agent_cache/scripts. Convert (13.5 min, foreground; defaults spelled out = the 2026-09-18 run recorded in .stats.json "args").
# Watch: it prints "loaded ... rows -> ... replayable sessions", "converted ... sessions", one line per source, then "wrote ...".
# Stop: from another host shell `docker exec sglang_hicache pkill -f convert_lmcache.py` (direct exec, no wrapping bash: pkill cannot match itself).
docker exec -i sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && python3 convert_lmcache.py \
  --parquet-dir /tmp/lmcache \
  --out /sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json \
  --tokenizer Qwen/Qwen3-32B-FP8 \
  --revision main \
  --tool-role-mode as_user \
  --max-context 32768 \
  --margin 512 \
  --min-turns 5 \
  --workers 12'
```
-> prints `loaded 24880 rows -> 769 replayable sessions in <N>s`, `converted 769 sessions in <N>s`, no `FAILED <session_id>` line on stderr,
the stats JSON (without `per_source`), `swebench: chains 665 emitted 630 truncated 396 ...`, and last
`wrote /sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json (65 MiB) and ....stats.json in <N>s` with N about 810 (13.5 min).

```bash
# host: the container wrote as root; give the files back to wanhr (git and the host venv need to read them)
sudo chown -R wanhr:wanhr /home/wanhr/sglang/agent_cache
```
-> prints nothing; `ls -l /home/wanhr/sglang/agent_cache/traces/` then shows `wanhr wanhr` on both files.

```bash
# host: check the output shape and the headline stats (read-only; host python 3.12.3, stdlib json only)
T=/home/wanhr/sglang/agent_cache/traces/lmcache_agentic_trace.json
echo "trace bytes:     $(stat -c %s $T)   (expect 68429742)"
echo "trace newlines:  $(wc -l < $T)   (expect 0: one JSON document, not JSONL)"
echo "stats bytes:     $(stat -c %s $T.stats.json)   (expect 4276)"
python3 - "$T.stats.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print("chains_emitted:          ", s["chains_emitted"], "  (expect 731)")
print("turns_emitted:           ", s["turns_emitted"], "  (expect 17887)")
print("chains_below_min_turns:  ", s["chains_below_min_turns"], "  (expect 38)")
print("chains_failed:           ", s["invariants"]["chains_failed"], "  (expect 0)")
print("empty_messages_turns:    ", s["invariants"]["empty_messages_turns"], "  (expect 0)")
print("rows_merged_forward:     ", s["invariants"]["rows_merged_forward"], "  (expect 0)")
print("replay_within_context:   ", s["invariants"]["replay_within_context"], "  (expect True)")
print("pre_gap raw mean/p50/p95:", round(s["pre_gap_raw_all_rows_excluding_first"]["mean"], 3),
      s["pre_gap_raw_all_rows_excluding_first"]["p50"], s["pre_gap_raw_all_rows_excluding_first"]["p95"], "  (expect 2.084 0.708 3.724)")
print("pre_gap emitted mean/p50:", round(s["pre_gap_emitted_after_turn0"]["mean"], 3), s["pre_gap_emitted_after_turn0"]["p50"], "  (expect 1.492 0.70195)")
print("output_length mean/p50/p95:", round(s["output_length_emitted"]["mean"], 1), s["output_length_emitted"]["p50"], s["output_length_emitted"]["p95"], "  (expect 182.0 97.0 600.0)")
print("replay_prompt_tokens mean/p50/p95:", round(s["replay_prompt_tokens_emitted"]["mean"]), round(s["replay_prompt_tokens_emitted"]["p50"]),
      round(s["replay_prompt_tokens_emitted"]["p95"]), "  (expect 19665 20494 30697)")
EOF
```
-> prints the labelled lines; pass when every value equals its `(expect ...)` (all measured 2026-09-18, `.stats.json`).

```bash
# container: sglang_hicache. Re-check that the checkout's loader and round normalizer accept every turn (the 2026-09-18 check: 731 conversations, 0 rounds rejected)
docker exec -i sglang_hicache bash -c 'cd /sgl-workspace/sglang && PYTHONDONTWRITEBYTECODE=1 python3 - <<"EOF"
from sglang.benchmark.datasets.agentic_trace import AgenticTraceDataset
from sglang.benchmark.serving import _normalize_round_messages
rows = AgenticTraceDataset(dataset_path="/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json",
                           num_requests=0, fixed_output_len=None, offset=0, max_turns=None).load(None)
rejected = sum(1 for r in rows for turn in r.prompt if _normalize_round_messages(turn) is None)
print("conversations loaded:", len(rows), "  (expect 731)")
print("rounds rejected:     ", rejected, "  (expect 0)")
EOF'
```
-> prints the loader's `#Conversations: 731 (offset=0, turns/conv min=5 ...)` line, then `conversations loaded: 731` and `rounds rejected: 0`.

**Expected result:** `agent_cache/traces/lmcache_agentic_trace.json` = 68,429,742 B (65.3 MiB), **one JSON document** (0 newlines; not JSONL),
top level `metadata`, `session_ids`, `sessions`, `conversations`; `lmcache_agentic_trace.json.stats.json` = 4,276 B. Both owned by `wanhr` after the chown.
Contents (all **measured 2026-09-18**, `.stats.json` unless noted):
- Input: 24,880 parquet rows, 767 session_ids, 769 chains (1 session_id with several chains), 769 selected, 0 failed; invariants
  `turns_plus_merged_equals_rows`, `turn0_pre_gap_zero`, `turn0_starts_with_system`, `replay_within_context` all true; `empty_messages_turns` 0, `rows_merged_forward` 0.
- Emitted: **731 chains** (38 below `--min-turns` 5), **17,887 turns**; 102 mid-run starts emitted (101 swebench + 1 wildclaw); 1,164 turns after turn 0 with no assistant reply in between.
- Truncated at 32K: 396 of 630 emitted swebench chains (6,902 turns dropped), 19 of 92 gaia, 2 of 9 wildclaw; swebench final contexts p50 33.6K (33,586), max 84K (84,314).
- Emitted `replay_prompt_tokens`: mean 19.7K (19,665), p50 20.5K (20,494), p95 30.7K (30,697) overall; swebench mean 20.6K (20,570), gaia 9.0K (8,953), wildclaw 13.9K (13,897).
  Recorded `prompt_tokens` mean 18,740; replay/recorded ratio 1.047 (**4.7 % longer**).
- `output_length` per emitted turn: **mean 182.0, p50 97, p95 600** (swebench 181.5, gaia 178.1, wildclaw 286.7 means). This is the third §0 input; 220 is only the loader default.
- Raw `pre_gap` over all rows but the first of each session (n 24,111) reproduces the README exactly: p50 0.708 s, mean 2.08 s, p95 3.72 s, p99 11.76 s;
  fraction >= 1 / 5 / 30 s = 21.3 % / 3.4 % / 0.66 %. Over emitted turns after turn 0 (n 17,156): p50 0.70, mean 1.49, p95 3.59, p99 10.25 s;
  >= 1 / 5 / 30 s = 20.2 % / 3.2 % / 0.50 % (truncation drops late turns). Per source (emitted): swebench p50 0.70 / mean 1.32 / p95 2.24 s;
  gaia p50 2.15 / mean 3.63 / p95 10.47 s; wildclaw p50 0.14 / mean 2.74 / p95 10.04 s.
- Time-weighted (measured 2026-09-18, converter analysis, not in `.stats.json`): turns with a gap >= 1 / 5 / 30 s are 20 % / 3.2 % / 0.5 % of turns but hold
  **80 % / 63 % / 53 %** of the idle context-token-seconds; the >= 30 s share sits in about 90 turns (max gap 1,511 s): report the tail with and without the top 1 %.
- Loader check: 731 conversations, 0 rounds rejected by `_normalize_round_messages`; verified against an independent re-derivation (2026-09-18).
**If it differs:** `FAILED <session_id>` on stderr or `chains_failed > 0`: a session hit an exception in a worker; the trace is still written without it, so
do not use it until the failure is understood. `(65 MiB)` but a different byte count: a different `--max-context`/`--sources`/`--tool-role-mode` than the
recorded `args`: compare `.stats.json` `args` before replacing the committed file. `PermissionError` on the host: the chown step was skipped.
**Expected lessons:** the trace is a 65 MiB single document while the parquet is 2.37 GB because the parquet repeats history: each of the
24,880 rows' `input` is the **full cumulative** message list of that call, whereas the trace stores each turn's delta `input[len(prev):]` once and
the client rebuilds the cumulative history itself (`serving.py:1319` extends `prev_messages`, `:1329-1331` appends the generated reply). Two consequences
for later steps: (i) the replayed history is not the recorded one (4.7 % longer on average because the client re-feeds `reasoning_content + content`
verbatim while the template strips `<think>` from recorded replies; up to about 35 % shorter for mid-run starts), so **§5.3 must use 20.6K (swebench) or
19.7K (all) as the live context per session, not 25K**, and recompute WS from `sessions[*].final_replay_prompt_tokens` when a subset is replayed;
(ii) the gap and output-length inputs of §5.3 now exist: mean `output_length` 182.0 replaces the 220 placeholder (§5.3's turn-time arithmetic is re-derived
from it after matrix row 1), and the gap-scale denominator `mean(pre_gap)` is the emitted-turn mean 1.49 s (2.08 s is the raw README value over turns the replay
never sends; §5.3's multipliers already use 1.49 s), with a per-source spread (swebench 1.32 s, gaia 3.63 s) that makes the scale subset-dependent.

**What the parquet really is** (measured 2026-09-18, not the dataset card): 24,880 rows = LLM calls, 767 session_ids (665 swebench / 94 gaia / 8 wildclaw;
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
tokenizer files are byte-identical to the 8B's, §1): `prompt_tokens` = the ORIGINAL cumulative input, and `replay_prompt_tokens` = the history the client
will actually send (kept messages + one placeholder reply of `output_length` tokens per earlier turn). They differ because the template strips the
`<think>` block of historical replies while the client re-feeds `reasoning_content + content` verbatim: replayed histories run **4.7 % longer** on
average (up to about 35 % shorter for mid-run starts). **Truncation uses the replay count:** cut at the first turn with
`replay_prompt_tokens + output_length > --max-context (32768) - --margin (512)`; emit chains with >= `--min-turns` (5) turns. `--sources`,
`--max-sessions`, `--seed` (shuffle) select; `--revision` is recorded in metadata.

**Per-turn fields:** `messages` (the only key the loader reads, §3.2), `prompt_tokens`, `replay_prompt_tokens`, `output_length`, `pre_gap` (0.0 on turn 0),
`iteration`, `source_rows`, `n_assistant_in_delta`, `merged_output_length`, `orig_model`, `source`. **Top level:** `metadata`, `session_ids`, `sessions`
(per-chain meta: n_rows, n_turns, truncated_at, midrun_start, final counts incl. `final_replay_prompt_tokens`), `conversations`.

**Replay-fidelity caveats for the analysis:** turn 0 of the 101 mid-run-start sessions sends system + user + up to 46 tool results with no assistant turn
between them; at each of the 1,164 re-prompt turns the replay has one extra generated reply the recording did not; every generated reply replaces a
recorded one, so tool results after turn 0 answer calls the replayed model never made. Context lengths and hit patterns stay realistic; the text does not.

### 3.4 Other replays

**Status:** not run (neither command executed on this box; the AgentX command also needs the §2.5 AIPerf venv, which does not exist)
**Goal:** have the two secondary workloads ready for matrix rows 7-8 (§7.1): Mooncake toolagent for cross-session prefix sharing (not gaps) and AgentX via AIPerf for the standardized long-context view; know before running what each can and cannot show.
**Runs on:** `container: sglang_hicache` (the client runs inside the container, the host has no python tooling); the copy step on the host; both need a server from §5.2 on port 30000
**Touches:** copies `toolagent_trace.jsonl` (4.4 MB) under `agent_cache/traces/`; writes `$OUT/mooncake_toolagent.jsonl` and AIPerf's own output dir; GPU busy for the whole run
**Takes:** Mooncake: the trace spans 1 h at native speed, x15 with the provisional slowdown factor = 15 h, so cap by wall time (row 8 is a wall-time-capped slice); AgentX: `--benchmark-duration 1800` = 30 min (+ boot); GPU busy for the whole of either run (the server is the GPU user), plus one boot (278-285 s warm, measured, §2.6)

**Mooncake toolagent (cross-session sharing only, not gaps).** Timestamp replay happens only with `--backend sglang` (`serving.py:1502-1508`); rounds of a
record fire as a burst with `"story"` placeholder replies (`datasets/mooncake.py:82-123`); prompts are `hash_id + 128 x "hi"` per 512-token block (`:84-88`), so the
KV footprint is ~1/4 of `input_length`. `--num-prompts` truncates **before** the timestamp sort (`mooncake.py:47,63`): pass all 23,608. `--use-trace-timestamps` is a
no-op on this path (`serving.py:1514` calls `get_request` without it). The trace offers 6.6 req/s (23,608 in 1 h); at P = 1,226 tok/s this server is far slower
(estimate 0.3-0.5 turns/s, §5.3), so without `--mooncake-slowdown-factor` (`serving.py:2711`; 15 is provisional and stretches the 1 h trace to 15 h) the run only measures backlog.

```bash
# host: copy the trace under the repo bind (the container cannot see /home/wanhr/data)
cp /home/wanhr/data/mooncake/toolagent_trace.jsonl /home/wanhr/sglang/agent_cache/traces/toolagent_trace.jsonl \
  && echo "toolagent lines: $(wc -l < /home/wanhr/sglang/agent_cache/traces/toolagent_trace.jsonl)   (expect 23608)"
```
-> prints `toolagent lines: 23608`.

```bash
# container: sglang_hicache (bash). Mooncake toolagent replay against a running arm (a) at P0 (§5.2, row 8 of §7.1); c = 8 is provisional (§5.3).
# Long run (hours at slowdown 15): watch with `tail -f $OUT/mooncake_toolagent.log`; stop with Ctrl-C in this shell (the output file keeps what finished).
RUNDIR=/sgl-workspace/sglang/agent_cache/results/$(cat /sgl-workspace/sglang/agent_cache/.current_results)
OUT=$RUNDIR/hbm_lru_mooncake_c8; mkdir -p $OUT      # <arm>_<trace>_c<N>, the §7.2 cell naming
MOONCAKE=(
  # server and backend: only --backend sglang replays the timestamps (serving.py:1502-1508)
  --backend sglang --host 127.0.0.1 --port 30000 --model Qwen/Qwen3-32B-FP8
  # dataset: the copy under the repo bind, all 23,608 records, one round per record
  --dataset-name mooncake --mooncake-workload toolagent
  --dataset-path /sgl-workspace/sglang/agent_cache/traces/toolagent_trace.jsonl
  --num-prompts 23608 --mooncake-num-rounds 1
  # pacing: timestamp replay stretched 15x (provisional); c = 8 live sessions
  --mooncake-slowdown-factor 15 --max-concurrency 8
  --output-file $OUT/mooncake_toolagent.jsonl
)
python3 -m sglang.benchmark.serving "${MOONCAKE[@]}" 2>&1 | tee $OUT/mooncake_toolagent.log
```
-> prints `Using time-based Mooncake request scheduler, ignoring --request-rate.` and
`Starting Mooncake trace replay. Sessions: 23608, Rounds per session: 1. Slowdown factor: 15.0` (`serving.py:1505,1509-1511`), then progress;
the absence of the first line means the backend is not `sglang` and the trace was fired without timestamps.

**AgentX via AIPerf** (container, `/opt/aiperf/bin/aiperf` from §2.5; flags per the NVIDIA tutorial via the data-prep reader; entry-point name and download cache are
unverified until `--help` in §2.5). AgentX median context is 142K (starter kit §1.1 and §3.2, `agent_cache/agent-kv-tiering-evaluation-starter-kit.md:15,60`, from the dataset card; not measured here): with a 32K-context server (`--context-length 32768`; the 32B's native limit is 40,960, `max_position_embeddings` in the snapshot's `config.json:14`) only the
256k-capped corpus with `--max-context-length 32768` is usable; expect many truncated trees. `--concurrency 8`, not 32: a 32K recompute costs 26.7 s here
(measured, §5.4), so few trees complete in 1800 s. c = 8 in both commands is provisional (§5.3 saturation estimate): revisit after matrix row 1.

```bash
# container: sglang_hicache. AgentX standardized run (30 min) against a running three-tier arm at PL (row 7 of §7.1); AIPerf writes its own report dir
AIPERF=(
  profile --scenario inferencex-agentx-mvp
  --url http://127.0.0.1:30000 --model Qwen/Qwen3-32B-FP8
  # context cap = the server's --context-length; chat endpoint, streaming, server token counts
  --max-context-length 32768 --endpoint-type chat --use-server-token-count --streaming
  --public-dataset semianalysis_cc_traces_weka_with_subagents_256k
  # load: 8 live trees, ignore EOS so lengths are deterministic, salt the first-turn prefix
  --concurrency 8 --extra-inputs ignore_eos:true --cache-bust first_turn_prefix
  # AIPerf's idle guard compresses gaps > 10 s at low concurrency (§10); 30 min budget; fixed seed
  --system-idle-gap-cap-seconds 10 --benchmark-duration 1800 --random-seed 20260707
  --ui simple
)
/opt/aiperf/bin/aiperf "${AIPERF[@]}"
```
-> prints AIPerf's simple UI and a final summary; not yet run here, so the exact lines are unverified. Pass condition for row 7: the server-side
`storage_prefetch_hit_tokens_total` delta is recorded (§6.1) and the run is not reported as an L3 result unless it is > 0.

**Expected result:** not yet measured. Mooncake: `$OUT/mooncake_toolagent.jsonl` + `.log` with the two startup lines above and 23,608 requests scheduled by timestamp;
completed turns/s is an estimate (0.3-0.5, §5.3): record the measured value here after the run. AgentX: an AIPerf report for 1800 s at c = 8 with many truncated trees;
record `storage_prefetch_hit_tokens_total` and the completed-tree count here after the run.
**If it differs:** Mooncake without `Using time-based Mooncake request scheduler`: `--backend` is not `sglang`. Mooncake finishing in minutes: `--num-prompts` was smaller than
23,608 or the slowdown factor was dropped, and the run measured backlog. AIPerf `command not found`: §2.5 was not done. AIPerf gaps all <= 10 s: the idle guard
compressed them, raise `--system-idle-gap-cap-seconds` or keep c high (§10).
**Expected lessons:** Mooncake is a burst-per-session, placeholder-content replay: it can only show cross-session prefix sharing and, without the slowdown factor,
measures how fast the 32B falls behind a 6.6 req/s trace, so it is row 8 and wall-time-capped. AgentX at c = 8 has WS <= 8 x 32,768 = 262K < L1+L2 = 268K at PL
(§7.1 row 7): L3 reads are ~0 by construction unless subagent trees multiply the live contexts, so a zero `storage` hit count there is expected, not a negative result.

### 3.5 Week-1 CPU-only characterization

**Status:** **not started on this box** (no `uv`, no `/home/wanhr/venv312`, no `/home/wanhr/agentic-kv-cache`, checked 2026-09-21). All three blocks below must run here; the `-> prints:` values are `measured-A100` from the 2026-09-18 run and are trace properties, so they must reproduce exactly. The memory-time upper bound (the third artifact) was never computed on either box and is still the open Week-1 deliverable.
**Goal:** the Week-1 decision of §9.3: the upper bound on what any parking policy can free, from the traces alone: `fraction(g)` of session-token-seconds idle behind gaps >= 1 / 5 / 30 s at c = 4/8/12/16, plus the mean gap and mean `output_length` that §5.3 turns into `--agentic-gap-scale` and the turn-time model.
**Runs on:** host, as `wanhr`, with `/home/wanhr/venv312/bin/python` (the host's own Python 3.12.3 has no `pip` and no `ensurepip`, so `python3 -m venv` cannot bootstrap one; the venv is symlinked as `/home/wanhr/agentic-kv-cache/.venv`)
**Touches:** creates `~/.local/bin/uv`, `~/.local/share/uv/python/`, `/home/wanhr/venv312`, `/home/wanhr/agentic-kv-cache` (+ `data/agentx.pkl` 433,729,974 B, `results/*.txt`); writes `agent_cache/results/<stamp>/week1_pre_gap.txt`; GPU untouched
**Takes:** install + venv ~2 min (estimate); `make data` ~1 min and `make repro` ~40 min (measured 2026-09-18 from mtimes: 01-03 by 00:14, `04_ablation.txt` at 00:52; the ablation dominates); the `pre_gap` script < 1 min (estimate: 4 columns projected from 2.37 GB of parquet, no message content loaded); GPU idle throughout

```bash
# host: 1. uv + a Python 3.12 venv with the analysis deps (skipped when /home/wanhr/venv312 exists; the host python3 has no ensurepip, so uv is the only route to a venv with pip)
[ -x /home/wanhr/venv312/bin/python ] && echo "venv312 exists: skip" || {
  curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH=$HOME/.local/bin:$PATH \
  && uv python install 3.12 && uv venv /home/wanhr/venv312 --python 3.12 \
  && uv pip install --python /home/wanhr/venv312/bin/python numpy pandas pyarrow orjson transformers huggingface_hub; }
echo "venv python: $(/home/wanhr/venv312/bin/python -c 'import sys, pyarrow, pandas; print(sys.version.split()[0], "pyarrow", pyarrow.__version__, "pandas", pandas.__version__)')"
```
-> on this box the first run installs (no `venv312` yet) and then prints `venv python: 3.12.x pyarrow ... pandas ...`; the A100 run printed `3.12.14 pyarrow 25.0.1 pandas 3.0.6` (`measured-A100`). Newer pins are fine (`requirements.txt` of the clone only needs `numpy>=2.0`).

```bash
# host: 2. clone the simulator, point it at the downloads and the venv, prep AgentX, run the four experiments (~40 min; skipped when results exist).
# Watch: `ls -l /home/wanhr/agentic-kv-cache/results/`, 04_ablation.txt appears last. Stop: Ctrl-C (make stops at the current experiment; re-run to continue).
# `make setup` would fail (host python3 -m venv has no ensurepip, §1 block C): the .venv symlink replaces it. `make data` skips existing data/*.jsonl (fetch_data.sh `[ -f ]`),
# so a truncated AgentX from §3.1 passes and its final bare `python3 experiments/prep.py` (host 3.12, stdlib-only, fine) crashes on JSON parse: check the size in §3.1 first.
[ -f /home/wanhr/agentic-kv-cache/results/04_ablation.txt ] && echo "repro results exist: skip" || {
  git clone https://github.com/gauravapiscean/agentic-kv-cache /home/wanhr/agentic-kv-cache \
  && cd /home/wanhr/agentic-kv-cache && mkdir -p data \
  && ln -sf /home/wanhr/data/mooncake/toolagent_trace.jsonl data/ \
  && ln -sf /home/wanhr/data/mooncake/conversation_trace.jsonl data/ \
  && ln -sf /home/wanhr/data/mooncake/mooncake_trace.jsonl data/mooncake_arxiv.jsonl \
  && ln -sf /home/wanhr/data/agentx/traces-062126-256k.jsonl data/agentx.jsonl \
  && ln -sfn /home/wanhr/venv312 .venv && make data && make repro; }
echo "repro result files: $(ls /home/wanhr/agentic-kv-cache/results/*.txt 2>/dev/null | wc -l)   (expect 4)"
```
-> on this box the first run clones and runs (~40 min, `measured-A100`), printing the four `=== N. ... ===` banners of the Makefile targets `validate`, `characterize`, `gap`, `ablation` and teeing each into `results/0N_*.txt`, then `repro result files: 4`; a later run prints `repro results exist: skip`.

```bash
# host: 3. pre_gap distribution per (source, model) and output_length per source from the parquet (no message content loaded); saved next to the results
#    The 5 shards are listed explicitly: a dataset over the directory would also open README.md (downloaded by §3.1) and die with ArrowInvalid.
STAMP=$(cat /home/wanhr/sglang/agent_cache/.current_results); mkdir -p /home/wanhr/sglang/agent_cache/results/$STAMP
/home/wanhr/venv312/bin/python - <<'EOF' | tee /home/wanhr/sglang/agent_cache/results/$STAMP/week1_pre_gap.txt
import pyarrow.dataset as ds, numpy as np, glob
t = ds.dataset(sorted(glob.glob('/home/wanhr/data/lmcache/*.parquet')), format='parquet').to_table(columns=['session_id','model','output_length','pre_gap']).to_pandas()
t['source'] = t.session_id.str.split('__').str[0]; t['iter'] = t.groupby('session_id').cumcount(); g = t[t['iter']>0]
for k,v in g.groupby(['source','model']).pre_gap:
    print(k, len(v), 'p50/p90/p95/p99', np.percentile(v,[50,90,95,99]).round(2), 'frac>=1s', (v>=1).mean().round(3), '>=5s', (v>=5).mean().round(3), '>=30s', (v>=30).mean().round(4))
for k,v in t.groupby('source').output_length: print(k, 'output_length mean/p50/p95', round(v.mean(),1), np.percentile(v,[50,95]), 'mean pre_gap', round(g[g.source==k].pre_gap.mean(),2))
print('sessions', t.session_id.nunique(), t.groupby('source').session_id.nunique().to_dict(), 'rows', len(t), 'turns/session p50', t.groupby('session_id').size().median(), 'mean pre_gap', round(g.pre_gap.mean(),2))
EOF
```
-> prints five (source, model) lines, three per-source lines, and last `sessions 767 {'gaia': 94, 'swebench': 665, 'wildclaw': 8} rows 24880 turns/session p50 34.0 mean pre_gap 2.08`
(measured 2026-09-18 on the host from the 5 parquet shards, stdout only; the tee'd file is not yet written); pass when that last line shows sessions 767, rows 24880
and mean pre_gap 2.08 (the raw mean over all rows but the first of each of the 767 session_ids, n = 24,113; `.stats.json` reports n = 24,111 because it excludes the
first row of each of the 769 chains, and the two extra rows carry `pre_gap` 0.0, so p50 0.708 / mean 2.08 / p95 3.72 agree to the printed precision, §3.3). The
per-source lines (same run): `gaia output_length mean/p50/p95 180.6 [101.  575.4] mean pre_gap 3.72`; `swebench output_length mean/p50/p95 212.2 [102. 712.] mean pre_gap 1.99`;
`wildclaw output_length mean/p50/p95 338.3 [ 145.  1133.9] mean pre_gap 2.61`. The five (source, model) lines are in Expected result (2).

**Memory-time upper bound** (the analysis itself; no script exists yet, write it against `agent_cache/traces/lmcache_agentic_trace.json` and record the outputs
in `agent_cache/results/<stamp>/week1_memtime.txt` and `constants.json`, §9.2). Use the client's semantics (closed loop: c live sessions, the next starts when one
finishes) and state the turn-time model `turn_time = T_pf + output_length / decode_rate`:
- `T_rec(L) = 0.0626 + 3.752e-4 L + 1.365e-8 L^2` s (derived: LSQ on the 7 measured recompute medians of §5.4, max residual 31 ms; 8.0 s at 14K, 14.0 s at 21K, 18.0 s at 25K).
  Never `prompt_tokens / P`: P = 1,226 tok/s is a marginal slope with a -1.25 s intercept.
- `T_pf = T_rec(prompt_tokens)` on turn 0 (upper variant: every turn, nothing cached); on a returning turn with a resident prefix `T_pf = T_L1(prefix) + T_rec(prompt_tokens) -
  T_rec(prefix)`, `T_L1(L) = 0.053 + 12.4e-6 L` s (measured L1 fit, §5.4): ~1.4 s for +1K on 25K. **Estimate, not measured** (open constant 2, §5.4).
- `decode_rate`: **not measured** (open constant 1, §5.4); until it is, report every fraction for both ends of the §5.3 estimate (raw 26-42 tok/s at batch 1, 17-28 at batch 7;
  or the EFFECTIVE ~10-13 at c=8, which already contains the other sessions' prefill stalls: never add those on top of it). `output_length`: the trace's per-turn value
  (mean 182.0, measured, §3.3; 220 is only the loader default).
- `fraction(g) = sum(context x pre_gap | pre_gap >= g) / sum(context x (turn_time + pre_gap))` for g = 1/5/30 s at c = 4/8/12/16 (c = 32/64/128 as an analytic row only:
  the 32B saturates this GPU near c ~ 8, estimate, §5.3). Also print mean(pre_gap) and the implied `f = turn_time / (turn_time + pre_gap)`: §5.3 turns them into `--agentic-gap-scale`.
- Absolute token-seconds convert with b = 131,072 B/token (8,192 tokens/GiB) and are compared with the §5.3 pools. Never normalize by a pool ("pool-seconds"): a trace has
  no pool; report the c at which c x mean context crosses each pool (c=32 x 25K = 800K exceeds L1+L2 of every §5.3 level, but not the 1,044,160 tokens of auto L1 + a 100 GB host pool).
- Row order within a session is assumed monotone (verified for 3 rows only; the converter sorts).

**Expected result:** three artifacts. (1) `agentic-kv-cache/results/01_validate.txt` (Mooncake reproduction, 23,608 requests), `02_characterize.txt`, `03_gap.txt`,
`04_ablation.txt`: present, 1,223 / 466 / 533 / 1,198 B (measured 2026-09-18 00:13-00:52 UTC); their contents are the simulator's single-tier numbers (starter kit: the
published-curve offset of 4-6 pp is unexplained) and are inputs to Weeks 2-3, not results of this study. (2) `week1_pre_gap.txt`: last line `sessions 767 {'gaia': 94, 'swebench': 665, 'wildclaw': 8} rows 24880 turns/session p50 34.0
mean pre_gap 2.08`, the three per-source lines given under block 3, and these five (source, model) lines (measured 2026-09-18 on the host, block 3 run to stdout; the file itself is not yet written):
`('gaia', 'claude-sonnet-4-6') 1283 p50/p90/p95/p99 [ 2.18  6.58 11.45 31.49] frac>=1s 0.725 >=5s 0.168 >=30s 0.0156`;
`('swebench', 'claude-sonnet-4-6') 2449 p50/p90/p95/p99 [ 0.69  1.37  2.76 11.84] frac>=1s 0.152 >=5s 0.029 >=30s 0.0053`;
`('swebench', 'deepseek-v3.1') 2192 p50/p90/p95/p99 [0.7  1.24 1.75 5.81] frac>=1s 0.129 >=5s 0.021 >=30s 0.0`;
`('swebench', 'minimax-m2.5') 18050 p50/p90/p95/p99 [ 0.71  1.7   2.39 10.23] frac>=1s 0.195 >=5s 0.025 >=30s 0.0068`;
`('wildclaw', 'claude-opus-4-6') 139 p50/p90/p95/p99 [ 0.15  8.28 10.04 44.06] frac>=1s 0.331 >=5s 0.115 >=30s 0.0216`
(the five counts sum to 24,113 = rows minus one per session_id). (3) `week1_memtime.txt`: not yet computed. Its shape: a table of `fraction(g)` for g = 1/5/30 s x
c = 4/8/12/16 (+ 32/64/128 analytic), each for both ends of the decode-rate estimate, plus mean(pre_gap), implied f per c, absolute idle token-seconds (and GiB-seconds
at b = 131,072) and the c at which c x mean context (20.6K swebench / 19.7K all, §3.3) crosses 131,072 (PH/PL L1), 268,416 (PL L1+L2), 497,344 (PH L1+L2) and 750,464
(P0 L1+L2) tokens (pool values from §5.3). Expectation, not a measurement: with an estimated 6.6-9.9 s turn at c=1 against a 2.08 s mean gap (0.71 s median), f is ~0.8
at c=1 and ~0.9 at c=8 (§5.3), so `fraction(1 s)` at gap scale 1 is expected small unless the tail carries it; the trace-only weighting of §3.3 (gaps >= 1 s hold 80 % of
idle context-token-seconds, >= 30 s hold 53 % in ~90 turns) says the tail is heavy, so report every fraction with and without the top 1 % of gaps.
**If it differs:** `python3 experiments/prep.py` dies on JSON parse: AgentX is truncated (§3.1 size gate). `pre_gap` last line shows `sessions` != 767 or `rows` != 24880:
a parquet shard is missing or short (§3.1 sizes). `mean pre_gap` not 2.08: the `iter > 0` filter or the per-session grouping is wrong (turn 0 has `pre_gap` 0.0 by
construction and must be excluded). `ArrowInvalid: Could not read schema from '/home/wanhr/data/lmcache/README.md'`: the dataset was opened on the directory
instead of the shard list (seen 2026-09-18 with the directory form), and `week1_pre_gap.txt` is then empty.
**Expected lessons:** this step is the §9.3 Week-1 gate and it decides the shape of everything after it. If `fraction(1 s)` at c = 8 (closed loop, stated turn-time model,
both decode-rate ends) is below 10 % of session-token-seconds, parking cannot free much at real gaps: stop or change the workload / gap scale before spending GPU days on
§7 rows 4-6. If it clears 10 % only because of the >= 30 s tail (~90 turns), the serving result will hinge on a few sessions: state it with and without the top 1 % and size
the §7 conversation counts (>= 5 x c) so the tail is represented. The mean gap also fixes the client's gap scale, `--gap-scale` (§4.6; §5.3: `s = required_mean_gap / mean(pre_gap)`); with the
emitted mean 1.49 s rather than the README's 2.08 s the multipliers grow (§5.3 now lists x9-11 / x17-23 / x21-26 / x31-39), and per-source means (swebench 1.32 s,
gaia 3.63 s) mean a swebench-only replay needs a larger scale than the mixed trace. Finally, mean `output_length` 182.0 (not 220) shortens every estimated turn time in
§5.3 / §7.3: those are re-derived from it together with the measured decode rate once row 1 exists.

---

## 4. Client patch: gap-faithful multi-turn replay

The stock `sglang.benchmark.serving` client fires the rounds of a multi-turn conversation back to back, so a replay has no idle window and nothing
can ever be parked in a lower tier; this section adds the per-turn `pre_gap` sleep, the per-turn output length and the side-channel paired control
that §6.5 and §7 depend on. At its end the reader has a reviewed ~50-line patch archived under `agent_cache/patches/`, applied to the bind-mounted
checkout, and proven end to end by a two-conversation dry run (matrix row 0, §7.1) before any measured cell is started.

Facts this section rests on (verified against `main @ 6ec32e6b7` on 2026-09-18, re-checked against `d608a20d4` on 2026-09-21 — nothing under
`python/sglang/benchmark/` changed in between, so every anchor in this section still resolves): `wrap_multi_turn_request_func` loops rounds with no sleep
(`python/sglang/benchmark/serving.py:1311-1333`); the only `asyncio.sleep` calls in the file are inter-conversation pacing at `serving.py:1078,1093`.
**Nothing in §4 is box-specific**: the client, the templates and the patch design port unchanged, and §4.6's client is already in the checkout.
Only the *dry run* (§4.4, §4.6) must be repeated here, because it is a measurement against a booted server.
Starter kit §3.1 ("add a per-turn `pre_gap` field and an `await asyncio.sleep(gap)` between rounds") is confirmed.

### 4.1 Diff-level design (7 hunks, ~50 lines)

**Status:** superseded 2026-09-18 by the standalone client of §4.6 (same three functions, no edit under `python/`); kept as the reference design and as the fallback if §4.6 fails its dry run. Not applied, and on this box `git status --short -- python/` is empty **while the §6.6 event-log patch is nevertheless active**, because that one is committed (§1 Repo row) — the two facts are no longer equivalent, so never read "clean tree" as "no eval patch" 
**Goal:** give the client three things the study needs and the stock loader lacks: the recorded gap before each turn, the recorded per-turn output
length, and a paired recompute control fired at the returning turn's instant, so that §7's cells measure tiering and not a no-idle-window burst.
**Runs on:** host, editor on `/home/wanhr/sglang/python/sglang/benchmark/` (= container `/sgl-workspace/sglang/python/sglang/benchmark/`: one bind, one working tree)
**Touches:** working-tree edits to `python/sglang/benchmark/datasets/common.py`, `python/sglang/benchmark/datasets/agentic_trace.py`, `python/sglang/benchmark/serving.py` (never committed: reverted by §4.3)
**Takes:** authoring + review, ~1-2 h (estimate); GPU idle

```text
# host: NOT a shell command. Design sketch of the 7 hunks against main @ 6ec32e6b7; every line number was verified on 2026-09-18.
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
   (leave the warmup test_input at 1431-1440 WITHOUT turn_meta so warmup does not sleep through ~38 gaps
    [estimate: turns of conversation 0 minus one; --agentic-max-turns 40 caps it at 39])
@@ wrap_multi_turn_request_func.f, hunk 6 "side-channel paired control" (§6.5), right after the sleep, when CONTROL_EVERY and round_index > 0
   and (round_index + conv_idx) % CONTROL_EVERY == 0 (the conv_idx offset spreads controls over turn indices) and meta.get("prompt_tokens", 0) > 0 (no length -> no control, never a KeyError):
+            ctl = asyncio.create_task(request_func(replace(inner_input, prompt=salted_random_chat(n_tokens=meta["prompt_tokens"],
+                       seed=SEED ^ conv_idx ^ round_index ^ RC_SALT), output_len=1)))   # TTFT is all that is compared; NOT via prev_messages, NOT via the semaphore
             ... after the round's await: control_outputs[(conv_idx, round_index)] = await ctl
   (in-slot is impossible: every round extends prev_messages and appends the reply, 1319/1329-1331; RC_SALT per process as hicache_eval/scripts/archive/exp2.py:28-30)
@@ result_details at 1883-1896: + "start_times" (o.start_time, set at 467, never dumped today), "conv_idx", "round_idx", "control_ttfts", "control_details" keyed (conv, round) (stringified for JSON)
@@ cli_main (near --agentic-max-turns, 2267-2274): --agentic-gap-scale (float, 1.0); --agentic-control-every K (int, 0 = off)
```

Design notes that the sketch does not spell out:

- `salted_random_chat` does not exist in the checkout: the patch adds it. Precedent for an exact-length, seed-divergent prompt is
  `hicache_eval/scripts/hcommon.py:60-72` (`build_prompt`: distinct seeds pick a distinct first word, so two prompts can never share a radix prefix).
- Control length: the sketch sizes the control from the trace's `prompt_tokens` (the ORIGINAL cumulative input). The converter also emits
  `replay_prompt_tokens`, the history the client actually sends, 4.7 % longer on average (§3.3). Decide which one sizes the control before
  authoring and record the choice in `config.json` (§9); the sketch as written uses `prompt_tokens`.
- Cost of the control: 1/K single-shot requests per turn (report it). With the 32B it is not cheap: each is a full recompute of the turn's
  `prompt_tokens` (18.0 s of GPU at 25K, derived from the §3.5 quadratic fit) and, on a `write_through` arm, 3.05 GiB to L2 and L3
  (derived: 25,000 x 131,072 B / 2^30), so K comes from the §5.3 / §7.3 budgets (provisional K = 40 at c <= 8), the same K for every arm of a row;
  K = 1 only in the §4.4 dry run. Assert `storage == 0 and host == 0 and device <= 64` on every control row (HANDOFF §3).
- Per-turn output length: `output_len` is sent as `max_completion_tokens` (`serving.py:435`) with `ignore_eos = not args.disable_ignore_eos`
  (`serving.py:443-446`, default True), so each turn generates exactly `output_len` tokens: deterministic KV growth, good for a systems study.
  Pass `--disable-ignore-eos` only if you want EOS-faithful replies (then `output_len` in metrics is right only if a usage chunk arrives, `serving.py:517-519`).
- `--agentic-gap-scale 0` reproduces the unpatched client (no sleeps): it is the "no idle window" control (§10).

```bash
# host: after editing, confirm the patch touches exactly the three files and count its lines (read-only; absolute repo path, no cd)
REPO=/home/wanhr/sglang
echo "files touched:"; git -C "$REPO" diff --stat -- python/sglang/benchmark
echo "added lines:   $(git -C "$REPO" diff -- python/sglang/benchmark | grep -c '^+[^+]')"
echo "hunks:         $(git -C "$REPO" diff -- python/sglang/benchmark | grep -c '^@@')"
```
-> prints `datasets/common.py`, `datasets/agentic_trace.py`, `serving.py` and nothing else; added lines ~50 (estimate from the sketch; record the real
count here), hunks 7 (some editors merge adjacent hunks: 5-8 is fine, a file outside `python/sglang/benchmark/` is not).

```bash
# host: confirm the container's client (same bind mount) exposes the two new flags and the new dataclass field (read-only)
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang && \
  echo "agentic flags: $(python3 -m sglang.benchmark.serving --help 2>/dev/null | grep -oE -- "--agentic-[a-z-]+" | sort -u | tr "\n" " ")" && \
  echo "turn_meta field: $(python3 -c "from sglang.benchmark.datasets.common import DatasetRow; print(\"turn_meta\" in DatasetRow.__dataclass_fields__)")"'
```
-> prints `agentic flags: --agentic-control-every --agentic-gap-scale --agentic-max-turns` and `turn_meta field: True`. The unpatched output
(run 2026-09-18) is `agentic flags: --agentic-max-turns` and `turn_meta field: False`: seeing that after editing means the edit is not on the bind
mount or the file failed to parse; run `python3 -m sglang.benchmark.serving --help` without the grep to see the traceback.

**Expected result:** the two blocks above pass; the loader still prints its `#Conversations: N (offset=..., turns/conv min=... max=... avg=...)` line
(`agentic_trace.py:107-112`) unchanged, and a run with `--agentic-gap-scale 0 --agentic-control-every 0` behaves like the unpatched client
(no sleep, no extra requests). Runtime proof is §4.4.
**If it differs:** `--help` shows the flags but the dry run never sleeps: hunk 4 (`turn_meta=request.turn_meta` in the main-loop `RequestFuncInput`, `serving.py:1549-1560`)
was left out, so `turn_meta` is `None` and every gap reads as 0. A `KeyError: 'prompt_tokens'` means the control guard `meta.get("prompt_tokens", 0) > 0` was dropped.
**Expected lessons:** the gap must be slept AFTER the previous round is awaited to completion (`serving.py:1324-1326`), which is exactly the LMCache
`pre_gap` definition (end of one call to start of the next); sleeping before the await would measure a different quantity. The control cannot ride in
the conversation slot because every round extends `prev_messages` and re-feeds the reply (`serving.py:1319,1329-1331`): a control there would poison
every later prefix, which is why it is a separate task outside the semaphore, and why K (not c) sets how many pairs a cell yields (§6.5).

### 4.2 Per-turn cached-token capture (stock path plus one optional hunk)

**Status:** capture path present in this checkout (verified 2026-09-18, anchors re-checked against `d608a20d4` on 2026-09-21); the optional `usage_prompt_tokens` hunk is not applied
**Goal:** establish that per-turn tier splits already reach the client JSONL with no server change, and that the client's printed hit rate has the
wrong denominator for multi-turn, so §6.2, §6.4 and §9.1 compute hit rates from per-turn fields and not from `result["cache_report"]`.
**Runs on:** host, read-only source check (`/home/wanhr/sglang/python/sglang/benchmark/serving.py`); the optional hunk is edited together with §4.1
**Touches:** read-only, unless the optional hunk is taken (then two more sites in `serving.py`, reverted by §4.3)
**Takes:** ~1 min; GPU idle

How the data flows (this checkout): with client `--cache-report` the chat func reads `sglext.cached_tokens_details` on every stream chunk
(`serving.py:245-256, 521-522`) because `--cache-report` injects `return_cached_tokens_details: true` (`serving.py:1990-1995`) and the server
builds the details (`entrypoints/openai/serving_chat.py:1892-1900`) and emits them as an `sglext` chunk with `choices=[]`
(`serving_chat.py:1956-1963`). Per-turn `cached_tokens` / `cached_tokens_details` reach the JSONL only with `--output-details` (`serving.py:1893-1903`).

**Gotcha:** `output.prompt_len` for every round is turn-0 `prompt_tokens` (`serving.py:119-121` + deepcopy at `serving.py:1322`), so the printed hit
rate and the `result["cache_report"]` denominator are wrong for multi-turn (`serving.py:1737-1760`).

Optional hunk for the true per-turn denominator: add `usage_prompt_tokens: int = 0` to `RequestFuncOutput` (after `serving.py:112`) and after
`serving.py:519` add `u = data.get("usage") or {}; output.usage_prompt_tokens = u.get("prompt_tokens", output.usage_prompt_tokens)` (mirror after
`serving.py:484` for the non-streaming path). It needs a usage chunk: server `--stream-response-default-include-usage`
(`arg_groups/fields/serving.py:255-258`) or client `--extra-request-body '{"stream_options":{"include_usage":true}}'` (merged into the chat payload at
`serving.py:450`; server side `entrypoints/openai/utils.py:92-106`).

```bash
# host: re-check the four anchors this step relies on (read-only); each value is a line number in serving.py
F=/home/wanhr/sglang/python/sglang/benchmark/serving.py
echo "sglext parser def:       $(grep -n 'def _extract_cache_from_sglext' $F | cut -d: -f1)"
echo "chat stream-chunk call:  $(grep -n '^ *_extract_cache_from_sglext(data, output)' $F | cut -d: -f1 | tail -1)"
echo "cache-report injection:  $(grep -n 'extra_request_body\[\"return_cached_tokens_details\"\]' $F | cut -d: -f1)"
echo "prompt_len copied from:  $(grep -n 'output.prompt_len = request_func_input.prompt_len' $F | cut -d: -f1)"
```
-> prints `245`, `522`, `1995`, `121` (verified 2026-09-18, unchanged at `d608a20d4` on 2026-09-21; the completions backend has its own call at 332, hence `tail -1`). Different numbers mean the
file moved under this section: re-anchor before editing.

**Expected result:** the block prints the four line numbers above; after a `--cache-report --output-details` run the JSONL carries one
`cached_tokens_details` entry per turn (`{device, host[, storage, storage_backend]}`), which §4.4 checks on real output.
**If it differs:** the JSONL has `cached_tokens` but every `cached_tokens_details` is `null`: the server was not started with `--enable-cache-report`
(§5.2 COMMON has it) or the backend is not `sglang-oai-chat`; the injection at `serving.py:1995` only fires for `sglang-oai` / `sglang-oai-chat`.
**Expected lessons:** the tier split is a client-visible field, so the study needs no server patch for restore-hit accounting (§6.2); but every
aggregate the stock client prints for multi-turn divides by turn-0 length, so hit rates must be recomputed per turn (from `cached_tokens_details`
against `replay_prompt_tokens` from the trace, or against `usage_prompt_tokens` if the optional hunk is taken), never read off the console.

### 4.3 Where the patch lives, how it is applied and reverted

**Status:** not needed while §4.6 is the client (nothing under `python/` to apply or revert); applies only if the §4.1 fallback is taken. `agent_cache/patches/` holds only `0002-hicache-event-log.patch`, which is **committed, not applied** (§6.6), and `python/` is clean (checked 2026-09-21)
**Goal:** keep the eval patch active for the container's client during a session while guaranteeing it never reaches a commit (HANDOFF §7:
`91d480573` committed one by accident, `2cb739b9a` reverted it).
**Runs on:** host as `wanhr` (author, archive, revert check); container `sglang_hicache` for the per-session apply (shown as `docker exec` from the host). Both
containers and the host share the one bind-mounted repo (`/home/wanhr/sglang` = `/sgl-workspace/sglang`): applied once = applied everywhere.
**Touches:** `agent_cache/patches/0001-agentic-trace-pre-gap.patch` (created), the three files of §4.1 under `python/sglang/benchmark/` (applied / reverted),
`agent_cache/results/<STAMP>/patches/` (copy). `<STAMP>` = the bare stamp in `agent_cache/.current_results`, created by §2.4 block 6 (this box has none until that block runs, checked 2026-09-21).
**Takes:** seconds per block; GPU idle

1. Author once, then archive the diff and clean the tree (the archive is the source of truth; the working tree is disposable):
```bash
# host: save the authored diff under agent_cache/patches and revert python/ (destroys uncommitted edits under python/ that are NOT in the diff); absolute paths, no cd
REPO=/home/wanhr/sglang
PATCH=$REPO/agent_cache/patches/0001-agentic-trace-pre-gap.patch
mkdir -p "$REPO/agent_cache/patches"
git -C "$REPO" diff -- python/sglang/benchmark > "$PATCH"
# revert only when the archive is non-empty (an empty diff means the edits are not under python/sglang/benchmark, or were already reverted); a failed checkout is its own STOP
if [ -s "$PATCH" ]; then
  git -C "$REPO" checkout -- python/ || { echo "STOP: git checkout failed (root-owned files? run: sudo chown -R wanhr:wanhr $REPO/python and rerun); the archive WAS written"; false; }
else
  echo "STOP: empty diff, nothing archived, python/ NOT reverted"; false
fi
echo "patch bytes:  $(stat -c %s "$PATCH")"
echo "patch sha256: $(sha256sum "$PATCH" | cut -c1-16)"
echo "python/ dirty files: $(git -C "$REPO" status --short -- python/ | wc -l)"
```
-> prints a non-zero `patch bytes`, a sha (goes into `config.json` as the patch sha, §9), and `python/ dirty files: 0`. `STOP: empty diff` means nothing
was archived and nothing reverted; `STOP: git checkout failed` means the archive exists but the tree is still dirty (`python/ dirty files` > 0): chown and rerun.

2. Per session, apply (idempotent check first; a second apply on an already-patched tree fails the `--check` and stops):
```bash
# host -> container: apply the archived patch to the shared working tree (writes the three files as root: chown before editing them from the host)
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang \
  && git apply --check agent_cache/patches/0001-agentic-trace-pre-gap.patch \
  && git apply agent_cache/patches/0001-agentic-trace-pre-gap.patch \
  && echo "applied: $(git status --short -- python/ | wc -l) files modified under python/"'
```
-> prints `applied: 3 files modified under python/`. `error: patch failed` / `patch does not apply` means either it is already applied (check with
`git status --short -- python/`) or the checkout moved off `d608a20d4` (re-anchor §4.1). Applying from the host as `wanhr` (`git apply` in
`/home/wanhr/sglang`) is equivalent and leaves the files `wanhr`-owned.

3. Before every commit, on the host: run the §11 revert block (the one revert procedure of this runbook; it reverse-applies every archived patch under
`agent_cache/patches/` that is applied, refuses with a `STOP:` line if anything else differs under `python/`, and ends with `python/ modified files: 0`).
If it prints `STOP: reverse-apply ... failed`, the file is root-owned (written by the container): `sudo chown -R wanhr:wanhr /home/wanhr/sglang/python` and rerun.

4. Copy the applied diff next to the results it produced (precedent: `hicache_eval/results/20260908_nixl_exp234/exp2/C2_backup_skip.patch`):
```bash
# host: archive the patch inside the current results stamp dir (absolute paths, no cd)
AC=/home/wanhr/sglang/agent_cache
STAMP=$(cat "$AC/.current_results" 2>/dev/null)     # bare stamp, e.g. 20260917_2303 (§11); written by the §2.4 results-stamp lines
[ -n "$STAMP" ] || { echo "STOP: agent_cache/.current_results is missing or empty (run the §2.4 results-stamp lines)"; false; }
[ -n "$STAMP" ] && mkdir -p "$AC/results/$STAMP/patches" \
  && cp "$AC/patches/0001-agentic-trace-pre-gap.patch" "$AC/results/$STAMP/patches/" \
  && echo "archived: $(ls "$AC/results/$STAMP/patches/")"
```
-> prints `archived: 0001-agentic-trace-pre-gap.patch`.

**Expected result:** `agent_cache/patches/0001-agentic-trace-pre-gap.patch` exists (size and sha recorded); after block 2 the container's `--help` shows the
§4.1 flags; after the §11 revert block `git status --short -- python/` prints nothing; the stamp dir holds a copy of the patch.
**If it differs:** `git apply --check` fails on a clean tree: the patch was taken from a different base (diff against `d608a20d4`). Files under
`python/sglang/benchmark/` show as root-owned on the host after block 2: expected (container writes, §10); `chown -R wanhr:wanhr python/` before editing them from the host or if the §11 revert block stops on them.
**Expected lessons:** the working tree is shared by host and both containers, so there is exactly one apply and one revert per session, and the archived
file (not the tree) is what a commit or a results dir must carry; `git status` under `python/` before every commit is the only guard against repeating `91d480573`.

### 4.4 Validating the patch (2-conversation dry run with a visible gap)

**Status:** **not started on this box** (no server has ever run here). Block 1 (the gap trace) and block 2 (the probe) still apply; blocks 3-5 are the patched-`bench_serving` form and are replaced by the §4.6 dry-run block when §4.6 is the client (matrix row 0 of §7.1 either way; needs an arm (a) P0 server up per §5.2)
**Goal:** prove on real output that the sleep, the per-turn `output_len`, the side-channel control and the new JSONL fields all work, and settle the
§5.2 history note (is a regenerated reply ever a cache hit?) that §5.3 and §6.2 depend on, before any measured cell.
**Runs on:** container `sglang_hicache` (`docker exec -it sglang_hicache bash`), any cwd unless stated; server = arm (a) P0 of §5.2 on `127.0.0.1:30000`
**Touches:** `/tmp/gaptest.json`, `/tmp/gaptest.jsonl`, `/tmp/gaptest_scale0.jsonl` (container filesystem); the server's radix cache (one probe + 16 dry-run requests: 6 rounds + 4 controls in block 3, 6 rounds in block 4; no L2/L3 on arm (a))
**Takes:** ~1 min per run, two runs plus one probe (estimate: the 2 x 5 s gaps in series dominate; the six prefills of ~300-1,100 tokens cost 0.24-0.46 s each
on the A100 [`measured-A100`, §5.4 rows 512 / 1,024] and are expected to be faster here, which does not change the ~1 min); GPU busy for seconds only

1. Build a trace with two conversations of three turns and two 5 s gaps each:
```bash
# container: sglang_hicache. Write a 2-conversation, 3-turn agentic-trace JSON with 5 s gaps to /tmp/gaptest.json
python3 - <<'EOF'
import json; sys="You are a helpful assistant."
conv=lambda tag:[{"messages":[{"role":"system","content":sys},{"role":"user","content":f"{tag} turn0 "+"lorem "*300}],"prompt_tokens":0,"pre_gap":0.0,"output_length":16},
                 {"messages":[{"role":"user","content":f"{tag} turn1 "+"ipsum "*300}],"prompt_tokens":700,"pre_gap":5.0,"output_length":16},    # approximate lengths: they only
                 {"messages":[{"role":"user","content":f"{tag} turn2 "+"dolor "*300}],"prompt_tokens":1100,"pre_gap":5.0,"output_length":16}]   # size the controls (hunk 6)
json.dump({"metadata":{},"conversations":[conv("A"),conv("B")]},open("/tmp/gaptest.json","w"))
print("conversations:", 2, "turns each:", 3, "gaps each (s):", [5.0, 5.0])
EOF
```
-> prints `conversations: 2 turns each: 3 gaps each (s): [5.0, 5.0]` and `/tmp/gaptest.json` exists.

2. Send one discarded long probe (the first long prefill in a fresh container took 9.0 s once, `R8/DEVIATIONS.md` D9; skip if this server already served a long request):
```bash
# container: sglang_hicache, cwd = the hicache_eval scripts dir (probe.py imports hcommon from cwd; MODEL picks the tokenizer). One 4,096-token salted probe, output discarded
cd /sgl-workspace/sglang/hicache_eval/scripts && MODEL=Qwen/Qwen3-32B-FP8 KV_BYTES_PER_TOKEN=131072 python3 probe.py --len 4096 --seed $RANDOM >/dev/null && echo "probe: done"
```
-> prints `probe: done` within ~2 s warm (measured recompute at 4,096 tokens: 1.86 s, §5.4) or up to ~9 s the first time in a new container; a `cd: no such file or directory` error means the scripts dir is missing and nothing was sent (the `&&` chain stops there).

3. Run the dry run at gap scale 1 with a control on every returning turn (K = 1 is allowed only here):
```bash
# container: sglang_hicache, bash (docker exec -it sglang_hicache bash; the container's default shell is zsh, hence arrays, not a flag string).
# Patched client, 2 conversations, both live, real 5 s gaps, one paired control per returning turn (~30 s)
# endpoint
ENDPOINT=(
  --backend sglang-oai-chat
  --host 127.0.0.1
  --port 30000
  --model Qwen/Qwen3-32B-FP8
)
# dataset: the 2-conversation trace from block 1, all of it, both conversations live; no warmup (warmup would replay conversation 0, serving.py:1431-1451)
DATASET=(
  --dataset-name agentic-trace
  --dataset-path /tmp/gaptest.json
  --num-prompts 2
  --max-concurrency 2
  --warmup-requests 0
)
# patch flags: sleep the recorded gaps as recorded (--agentic-gap-scale default 1.0), one control on every returning turn
PATCHFLAGS=(
  --agentic-control-every 1
)
# replay template keys (§4.5): thinking off + keep special tokens, so the re-fed reply is byte-identical to the cached ids (server boots with --chat-template, §5.2)
REPLAY=(
  --extra-request-body '{"chat_template_kwargs":{"enable_thinking":false},"skip_special_tokens":false}'
)
# output: per-turn tier split into the JSONL
OUTPUT=(
  --cache-report
  --output-details
  --output-file /tmp/gaptest.jsonl
)
python3 -m sglang.benchmark.serving "${ENDPOINT[@]}" "${DATASET[@]}" "${PATCHFLAGS[@]}" "${REPLAY[@]}" "${OUTPUT[@]}"
```
-> the load line prints `#Conversations: 2 (offset=0, turns/conv min=3 max=3 avg=3.0)` (`agentic_trace.py:107-112`); the run takes >= 10 s (derived:
two 5 s gaps in series per conversation, conversations concurrent) and appends one line to `/tmp/gaptest.jsonl`.

4. Repeat with the gaps switched off (the "unpatched client" control, §10), to a separate file:
```bash
# container: sglang_hicache, bash. Same run with every gap scaled to 0 (no sleeps) and no controls (~5 s); only PATCHFLAGS and the output file differ from block 3
# endpoint
ENDPOINT=(
  --backend sglang-oai-chat
  --host 127.0.0.1
  --port 30000
  --model Qwen/Qwen3-32B-FP8
)
# dataset: same trace, both conversations live, no warmup
DATASET=(
  --dataset-name agentic-trace
  --dataset-path /tmp/gaptest.json
  --num-prompts 2
  --max-concurrency 2
  --warmup-requests 0
)
# patch flags: no idle window (== unpatched client, §10), no controls
PATCHFLAGS=(
  --agentic-gap-scale 0
  --agentic-control-every 0
)
# replay template keys (§4.5), same as block 3
REPLAY=(
  --extra-request-body '{"chat_template_kwargs":{"enable_thinking":false},"skip_special_tokens":false}'
)
# output: separate file so block 5 can diff the two runs
OUTPUT=(
  --cache-report
  --output-details
  --output-file /tmp/gaptest_scale0.jsonl
)
python3 -m sglang.benchmark.serving "${ENDPOINT[@]}" "${DATASET[@]}" "${PATCHFLAGS[@]}" "${REPLAY[@]}" "${OUTPUT[@]}"
```
-> same load line; the run finishes in a few seconds (six prefills + 96 decoded tokens, no sleep).

5. Read both JSONLs (each `--output-file` appends, `serving.py:1899`: the LAST line is this run) and print every pass condition with a label:
```bash
# container: sglang_hicache. Check counts, ordering, per-round device hits, control rows and the wall-time difference of the two dry runs
python3 - <<'EOF'
import json
last = lambda p: json.loads(open(p).read().splitlines()[-1])
r, r0 = last("/tmp/gaptest.jsonl"), last("/tmp/gaptest_scale0.jsonl")
print("wall scale1 (s):", round(r["duration"], 2), " wall scale0 (s):", round(r0["duration"], 2), " difference (s):", round(r["duration"] - r0["duration"], 2))
for k in ("ttfts", "cached_tokens_details", "start_times"):
    print(f"len({k}):", len(r.get(k) or []))
print("conv_idx:", r.get("conv_idx"), " round_idx:", r.get("round_idx"))
print("device per entry:", [(d or {}).get("device") for d in (r.get("cached_tokens_details") or [])])
print("input_lens per entry (turn-0 length repeated per round, the §4.2 gotcha):", r.get("input_lens"))
st, ci = r.get("start_times") or [], r.get("conv_idx") or []
for c in sorted(set(ci)):
    t = [s for s, i in zip(st, ci) if i == c]
    print(f"conv {c} round-start deltas (s):", [round(b - a, 2) for a, b in zip(t, t[1:])])
cd = r.get("control_details") or {}
print("n_controls:", len(cd))
for key, d in cd.items():
    d = d or {}
    ok = (d.get("storage") or 0) == 0 and (d.get("host") or 0) == 0 and (d.get("device") or 0) <= 64
    print(f"control {key}: device={d.get('device')} host={d.get('host')} storage={d.get('storage')} -> {'OK' if ok else 'FAIL'}")
EOF
```
-> prints: `difference (s)` = 10 +/- 1 (derived: two 5 s gaps per conversation in series; the tolerance is the design's allowance for the 0.24-0.46 s
measured prefill of a ~500-1,000-token prompt [§5.4] plus 16 decoded tokens at the unmeasured decode rate); `len(ttfts)`, `len(cached_tokens_details)`,
`len(start_times)` = 6 each, conversation-major / round-minor (the flatten at `serving.py:1567-1569`), i.e. `conv_idx` = `[0,0,0,1,1,1]` and
`round_idx` = `[0,1,2,0,1,2]`; every round-start delta >= 5 s; `device` = 0 on both round-0 entries and > 0 on rounds 1-2; `n_controls: 4`
(rounds 1-2 of both conversations at K = 1) and every control row `-> OK`.

**Expected result:** all pass conditions of block 5 hold (not yet measured: record the printed wall times and the four `device` values of rounds 1-2 here
after the run). Cross-check in the server log (`server_hbm_lru.log` in the stamp dir, §5.2): 10 requests for the scale-1 run, 6 conversation rounds
with ~5 s spacing between rounds of the same conversation plus 4 controls; the loader also prints `#Output tokens per turn: 220` (the unpatched
default, `agentic_trace.py:14`), while the patched client actually sends the trace's per-turn `output_length` = 16 (hunk 5), so `output_lens` in the
JSONL must be 16 for every conversation entry (derived: `max_completion_tokens` 16 with `ignore_eos` True, `serving.py:435,443-446`). Extra check (§4.5): on rounds 1-2 `device` should be about floor64(previous round's prompt + its 16-token reply), i.e. within one page of
`prompt - delta`; a `device` of about floor64(previous prompt) means the reply was NOT reused (the server booted without `--chat-template`, or `REPLAY` was
dropped from the client line) and the §5.2 history note's pessimistic case is what is being measured. The client sends `temperature: 0.0` by default for this backend (`serving.py:439-441`), overriding
the model's `generation_config` sampling, so replies are greedy and comparable across arms. On arm (a) `host` and `storage` are 0 by construction
(no HiCache): the control assertion first bites on arms (b)/(c) in row 2 (§7.1).
**If it differs:** `difference (s)` about 0: the gaps are not slept (hunk 4 missing, `turn_meta` never reaches the main-loop `RequestFuncInput`; or the
trace's `pre_gap` keys did not survive the loader). `len(ttfts)` = 2: the flatten at `serving.py:1567-1569` was bypassed or `--output-details` was
dropped. `n_controls: 0`: the `(round_index + conv_idx) % K` guard or the `prompt_tokens > 0` guard never fired (turn 0 has `prompt_tokens: 0` by
design, rounds 1-2 must have 700 / 1100). A control row with `device > 64`: `salted_random_chat` does not diverge at token one (borrow
`hcommon.build_prompt`'s first-word rule, `hicache_eval/scripts/hcommon.py:60-72`). The old absolute criterion "wall < 3 s" was an 8B guess: do not
reinstate it, the comparison here is relative (scale 1 minus scale 0).
**Expected lessons:** with the §4.5 template `device` on rounds 1-2 is ~`prompt - delta` (the reply IS cached), so §6.2 keeps `cached >= P - 64` and §5.3
carries no re-prefilled reply term; if instead `device` is ~floor64(previous prompt), every returning turn re-prefills its own previous reply
(`recompute >= output_len(prev) + new message tokens` even on a perfect hit): fix the boot or the client line before any measured cell, since §6.2 would
otherwise have to classify `device-only` against the PREVIOUS round's prompt and every arm's hit ceiling drops by ~220 tokens per turn. A 10 s wall difference is the proof that a gap replay has an idle window at all, which is the one thing
the unpatched client never provides and the reason rows 2-8 of §7.1 can measure parking.

### 4.5 Replay chat template: make the re-fed reply a cache hit

**Status:** templates generated, checked offline and proven on the booted arm (a) server 2026-09-18 (28/28 returning turns reused, see below); `agent_cache/templates/`, `scripts/replay_template.py`
**Goal:** make turn N+1 reuse the KV of the reply generated at turn N, so a returning turn's uncached tokens are only its new messages; without this every turn re-prefills its own previous reply and the hit-rate ceiling of every arm is `prompt - output_len(prev)`.
**Runs on:** host, `/home/wanhr/venv312` (transformers + jinja2), read-only on the HF cache; the server flag runs in `container: sglang_hicache` (§5.2)
**Touches:** writes `agent_cache/templates/qwen3_replay_nothink.jinja` and `qwen3_replay_think.jinja` (host path = container `/sgl-workspace/sglang/agent_cache/templates/`); nothing under `python/`
**Takes:** ~20 s per command (tokenizer load); GPU idle

Why (measured 2026-09-18 with the Qwen3-32B-FP8 tokenizer, `scripts/replay_template.py check`): the radix cache matches token ids, and Qwen3's stock template renders an assistant turn one way when generating and another way once it is history. With `enable_thinking: false` the generation prompt ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`, but the same reply re-fed as history renders as `<|im_start|>assistant\n` + content (the empty think block is never emitted for a turn before the last user/tool message); with thinking on, the generated `<think>...</think>` is stripped from history. Either way the token sequence diverges right after `assistant\n`: reply tokens reused 0/37 (thinking on), 0/25 (thinking off). So turning thinking off is necessary but not sufficient, and `--reasoning-parser` cannot help. The fix is a two-line template change that renders history exactly as it was generated: BOTH variants disable the `'</think>' in content` split (historical assistant content is rendered verbatim), and `nothink` additionally inserts the empty think block into historical assistant turns (offline check: reused 25/25 and 37/37). Verbatim rendering is not optional even with thinking off: under `ignore_eos` the model runs past `<|im_end|>` and hallucinates a new turn whose garbage tail often contains `</think>` and `<think>`; with the stock split the re-fed reply was cut at the last `</think>` (live server 2026-09-18: 7 of 35 returning turns reused 2-185 of their reply tokens, e.g. 2/132). For the same reason the server runs WITHOUT `--reasoning-parser`: a `<think>` in the tail made the parser route the rest of the text into `reasoning_content`, which the client does not re-feed (1 of 20 turns after the template fix). With both fixes, 28 of 28 returning turns reused their full reply on the live arm (a) server (2026-09-18, `--check-ids`, 4 conversations x 8 turns; max `reply_reprefilled` 61 < one page). Both variants leave the next turn's uncached tokens equal to its delta.

```bash
# host: (re)generate the two templates from the model's own chat template, then print the cross-turn reuse for stock vs patched (read-only on the HF cache)
cd /home/wanhr/sglang/agent_cache/scripts \
  && /home/wanhr/venv312/bin/python replay_template.py make 2>&1 | grep -v "PyTorch was not found\|corrupted tree cache" \
  && /home/wanhr/venv312/bin/python replay_template.py check 2>&1 | grep -v "PyTorch was not found\|corrupted tree cache"
```
-> prints `wrote .../templates/qwen3_replay_nothink.jinja: 4191 bytes`, `wrote .../qwen3_replay_think.jinja: 4152 bytes`, then four lines ending
`stock, thinking on : reply tokens reused 0/37`, `stock, thinking off: ... 0/25`, `nothink template   : ... 25/25, uncached on next turn 27`,
`think template     : ... 37/37, uncached on next turn 23` (measured 2026-09-18; the check applies the server's own file transform,
`parser/template_manager.py:281-282`, so it measures what a booted server renders). An `AssertionError: ... anchor found 0x` means the model's
template changed (new snapshot): re-derive the two anchors in `replay_template.py:21-25` against `tokenizer_config.json` before trusting any hit rate.

How it is used (the default of every §5.2 arm is `nothink`; `think` is the alternative, both need the same three client keys):

| | server flag (COMMON, §5.2) | client `--extra-request-body` (§4.4 block 3, §7.2 block 7) | reply that comes back |
|---|---|---|---|
| nothink (default) | `--chat-template /sgl-workspace/sglang/agent_cache/templates/qwen3_replay_nothink.jinja`, **no** `--reasoning-parser` | `{"chat_template_kwargs":{"enable_thinking":false},"skip_special_tokens":false}` | content only, like the recorded replies |
| think | `--chat-template .../qwen3_replay_think.jinja`, **no** `--reasoning-parser` | `{"chat_template_kwargs":{"enable_thinking":true},"skip_special_tokens":false}` | raw `<think>...</think>` + content, re-fed verbatim |

`skip_special_tokens:false` matters because `ignore_eos` lets the model emit `<|im_end|>` mid-reply and keep going: with the default `true` that
token is dropped from the returned text, the re-fed history lacks it, and the match ends there. `--extra-request-body` is merged into the chat payload
(`serving.py:450`); `chat_template_kwargs` is honoured by the server (`entrypoints/openai/protocol.py:932`, `serving_chat.py:563`). The client
re-feeds `generated_text` unchanged (`serving.py:1329-1331`): keep it that way, never trim it.

**Expected result:** the block above prints the four reuse lines; after the §4.4 dry run, `device` on rounds 1-2 is about `floor64(previous prompt + previous reply)`, i.e. the uncached count `usage_prompt_tokens - device` (or `replay_prompt_tokens - device`) equals the turn's delta within one 64-token page (prefix matching is page-floored, `mem_cache/radix_cache.py:150-154`). Thinking on or off changes nothing the study measures: the reply length is pinned by `max_completion_tokens` + `ignore_eos`, so the context trajectory is identical; only the tokens' content differs.
**If it differs:** uncached exceeds the delta by about `output_len(prev)` on every returning turn: the server was booted without `--chat-template` (grep the log for `Detected user specified Jinja chat template`), or the client sent no `chat_template_kwargs` (the stock template then wins on the thinking default), or `skip_special_tokens` was left true (uncached grows only on turns whose reply contained `<|im_end|>`). Retokenisation drift (decoded text re-encoding to different ids) shows as a small, turn-dependent excess of a few tokens at the end of a reply, never a full reply's worth.
**Expected lessons:** "feeding the reply back" is not enough for KV reuse; the template must render history byte-identically to the generation prompt, which no stock reasoning-model template does. The same trap applies to AIPerf's verbatim-history replay and to any harness that re-sends recorded rather than generated text: on those, the reply is a miss every turn by construction, so their hit rates are not comparable with these cells (§3.4, §7.1 row 7).

### 4.6 Standalone replay client (`scripts/replay_agentic.py`, supersedes the §4.1-§4.3 patch)

**Status:** the client is committed and carried over unchanged (`agent_cache/scripts/replay_agentic.py`, 16,028 B). On the A100 box it was mock-tested 2026-09-18 and run against a booted arm (a) at P0: 4 conversations x 8 turns, closed loop c=4, `--check-ids`, 28/28 returning turns reused their full reply after the §4.5 fixes (template verbatim + no reasoning parser); it then drove the whole `C18/` comparison (381 turns x 3 arms, §7.0). **On this box it has never run:** the 5 s-gap dry run below is matrix row 0 and is still owed. It is the cheapest end-to-end proof that the port works.
**Goal:** replay the converted trace with the recorded gaps and with every generated reply re-fed verbatim, so a returning turn's uncached tokens are only its tool results + new prompt, with no edit under `python/` and one JSON line per turn for §6.4.
**Runs on:** `container: sglang_hicache` (aiohttp is in the image); host path `agent_cache/scripts/replay_agentic.py` = container `/sgl-workspace/sglang/agent_cache/scripts/replay_agentic.py`
**Touches:** appends to the `--output` JSONL only; sends chat requests to the booted server (and `/generate` requests for controls)
**Takes:** the trace's wall time (gaps included) + one reply per turn; the dry run below ~15 s of client time on a booted arm (a); GPU busy while it runs

What it does per conversation (closed loop, `--concurrency` conversations live, the next starts when one finishes): turn 0 = its `messages`; stream the
reply; append `{"role":"assistant","content": <reply text unchanged>}`; sleep `pre_gap x --gap-scale` (or a value drawn from `--gap-sample FILE`, then
`--gap-cap`) measured from the END of the reply; turn N = history + the turn's `messages`. Every request carries the §4.5 keys (`chat_template_kwargs.enable_thinking`
from `--thinking`, `skip_special_tokens:false`, `ignore_eos:true`, `max_tokens = output_length`, `temperature 0`, `return_cached_tokens_details:true`); the server must be
booted with `--enable-cache-report` and the §4.5 `--chat-template` (both in §5.2 COMMON). Per turn it writes `kind:turn` with `t_send`, `ttft`, `latency`, `gap_slept`,
`prompt_tokens`, `completion_tokens`, `cached_tokens`, `cached_details{device,host,storage}`, `delta_tokens` (= this prompt minus previous prompt + reply) and
`reply_reprefilled` (= uncached tokens beyond the delta: 0 +- one page when the previous reply was reused, ~`output_length` when it was not). `--control-every K` fires,
outside the semaphore, one `/generate` request of random token ids of the turn's `prompt_tokens` length with 1 output token every K returning turns (`kind:control`,
its TTFT is the recompute reference of §6.5; `cached_tokens` must be <= 64). `--check-ids` additionally asks the server for the prompt and output token ids
(`sglext.input_ids/output_ids`, `entrypoints/openai/protocol.py:444-445`) and writes `reply_ids_reused / reply_ids_total`: the exact count of the previous reply's ids
found on the new prompt's prefix, independent of eviction and of page flooring. `kind:run` (config) and `kind:summary` lines bracket the run.
Load control: closed loop by default (request rate ~ `c / (turn_time + s x mean_gap)`, tuned with `--concurrency` and `--gap-scale`); `--arrival-rate L` switches to open
loop (conversations start as a Poisson process at L per second, `--concurrency` becomes the cap; request rate ~ L x turns per conversation). `kind:conv_start` records
`arrival_s`, `start_s`, `queued_s` per conversation: `queued_s` > 0 is client-side admission waiting on the cap, distinct from server queueing (`num_queue_reqs`, §6.1).
`reply_reprefilled` < 0 is possible and means the turn's delta itself was a hit (cross-session sharing of the prompt), not an error.

```bash
# container: sglang_hicache, bash. Dry run on arm (a) at P0 (§5.2): the 2-conversation gap trace of §4.4 block 1, both live, real 5 s gaps, a control on every returning turn, exact id check
cd /sgl-workspace/sglang/agent_cache/scripts \
  && python3 replay_agentic.py \
       --trace /tmp/gaptest.json \
       --url http://127.0.0.1:30000 --model Qwen/Qwen3-32B-FP8 \
       --concurrency 2 --gap-scale 1.0 --thinking off \
       --control-every 1 --check-ids \
       --tag gaptest --output /tmp/gaptest_replay.jsonl \
  && python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open("/tmp/gaptest_replay.jsonl")]
turns = [r for r in rows if r["kind"] == "turn" and r.get("turn", 0) > 0]
ctl = [r for r in rows if r["kind"] == "control"]
print("returning turns:", len(turns), " gap_slept min:", min(r["gap_slept"] for r in turns))
print("reply ids reused / total per returning turn:", [(r["reply_ids_reused"], r["reply_ids_total"]) for r in turns])
print("reply_reprefilled per returning turn:", [r["reply_reprefilled"] for r in turns], "(< 64 each = reused; ~output_length = NOT reused)")
print("controls:", len(ctl), " max control cached_tokens:", max((r.get("cached_tokens") or 0) for r in ctl))
print("errors:", sum(1 for r in rows if r.get("error")))
EOF
```
-> prints `returning turns: 4  gap_slept min: 5.0`; `reply ids reused / total`: four pairs with equal numbers (e.g. `(16, 16)`), `reply_reprefilled` four values below 64,
`controls: 4  max control cached_tokens: 0` (<= 64 passes), `errors: 0`. The client's own last line, `summary: {...}`, repeats `turns`, `controls`, `errors`,
`mean_ttft_s`. Not yet measured: record the printed values here after the run.

**Expected result:** the block passes as printed above and the run takes >= 10 s (two 5 s gaps in series per conversation, conversations concurrent; the client prints `conv N done` lines). Repeat with `--gap-scale 0` to a second file: same reuse numbers, run time a few seconds (the no-idle-window control of §10).
**If it differs:** `reply_ids_reused` 0 on every turn with `reply_reprefilled` ~ `output_length`: the server is on the stock template (no `Detected user specified Jinja chat template` log line, §5.2) or was booted with `--reasoning-parser` under `--thinking on`. `reply_ids_reused` short by a few ids only at the END of the reply: retokenisation drift of the decoded text (partial multibyte char at the `ignore_eos` cut); acceptable, note it. `cached_details` `null` on every turn: `--enable-cache-report` missing from COMMON. Control `cached_tokens` > 64: two controls drew the same random ids (impossible with distinct `(seed, conv, turn)`): check `--seed` is not reused across runs writing to one file. `HTTP 400 ... skip_special_tokens` or `chat_template_kwargs`: an older image; both fields exist in this checkout (`protocol.py:926,932`).
**Expected lessons:** with 16-token replies (the §4.4 gap trace) `cached_tokens` alone cannot show the reuse, because prefix matching is page-floored (64) and the reply is smaller than a page: `floor64(prompt + 16)` equals `floor64(prompt)` almost always, so the usage-based `reply_reprefilled` reads the same either way (seen on the mock 2026-09-18: 27/46 in both modes). Use `--check-ids` for the proof, or replies >= 128 tokens (then the two cases separate clearly: 15-43 vs 171-206 reprefilled on the mock). In the LMCache cells replies average 182 tokens (§3.3), so `reply_reprefilled` is diagnostic there without `--check-ids`; keep `--check-ids` off in cells (it returns ~20K ids per turn) and rely on `reply_reprefilled < 64` as the per-turn guard in `analyze.py` (§6.4). The §4.1 patch remains a valid alternative, but it needs the apply/revert cycle of §4.3 and its JSONL is one line per run with parallel lists; this client's one-line-per-turn output is what §6.4 and §9.1 consume directly.

---

## 5. Server configurations

Every measured cell of §7 boots exactly one server per arm, and the arms must differ only in their tiering, never in the flags the
measurements were taken with or in the size of the device pool. This section pins the flags that exist in this checkout and the ones that
silently rewrite themselves (§5.1), gives one pasteable launch block per arm with the log lines that prove the intended tiers came up (§5.2),
sizes the three pressure levels so the live working set really exceeds the tier under test (§5.3), and holds the one table of constants every
later step reads from, including the two that are still unmeasured and the command that measures them (§5.4). At its end the reader can boot
any arm at any level and check, from the log alone, that the server is the one the numbers below describe.

### 5.1 Verified flag table (this checkout)

**Status:** flag semantics verified against `main` @ `6ec32e6b7` on 2026-09-17 and re-checked against `d608a20d4` on 2026-09-21 (all anchors resolve; the re-check block prints the load-bearing ones). **Two rows changed with the GPU** and are marked SM90 below: the attention-backend row and the FP8-weight row. One row changed with the disk: the prefetch-timeout budget.
**Goal:** pin the exact flags, defaults and env vars that the §5.2 launch blocks rely on, so a starter-kit flag that does not exist here, or one that is silently rewritten, is caught before a boot instead of after a cell.
**Runs on:** host, read-only. Paths in the table are relative to `/home/wanhr/sglang/python/sglang/srt/` unless they begin with `R32/`, `R8/`, `C18/`, `test/` or `hicache_eval/` (`R32/` = `hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv/`, `R8/` = `.../20260917_a100_gcp_qwen8b/`, `C18/` = `agent_cache/results/compare_20260918_final/`; all three are **A100** results). **One abbreviation to know:** the server-args files (`choices.py`, `overrides.py`, `memory_hook.py`, `hicache_hook.py`, `validation_hook.py`, `model_override_base.py` and everything under `fields/`) all live under `arg_groups/`, and the table writes them without that prefix — so `fields/memory.py:104` is `python/sglang/srt/arg_groups/fields/memory.py:104`. The re-check block below `cd`s into `srt/` and uses the full paths, so it resolves them for you.
**Touches:** read-only
**Takes:** reading time; the re-check block < 1 s; GPU idle

| flag | default | choices / notes | evidence |
|---|---|---|---|
| `--radix-eviction-policy` | `lru` | `lru, lfu, slru, priority`. **`fifo` is NOT a CLI choice** (factory has it, argparse rejects it). `--radix-eviction-policy-config` JSON kwargs, only `slru` reads `protected_threshold`. `--disable-radix-cache` is exclusive with hierarchical cache | `arg_groups/choices.py:188`; `mem_cache/utils.py:59-67`; `fields/memory.py:46-60`; `kv_cache_hook.py:168-172` |
| `--enable-hierarchical-cache` | False | builds `UnifiedRadixCache` + `init_hicache` (**not** `HiRadixCache`, which only tests instantiate) | `mem_cache/registry.py:190-198`; `test/registered/unit/mem_cache/test_hiradix_cache_unit.py:76` |
| `--hicache-ratio` | None -> 2.0 (cache mode) | host pool = device tokens x ratio | `fields/memory.py:104-107`; `hicache_hook.py:62-81` |
| `--hicache-size` | 0 | int, decimal GB; **overrides ratio when > 0**; raises if > host budget. Tokens = `(int(GB x 1e9 // b) // page + 1) x page`: at b = 131,072, 100 -> 762,944 (**measured**; the rule reproduces it), 64 -> 488,320, 48 -> 366,272, 32 -> 244,160, 18 -> 137,344 (derived). Default ratio 2.0 would be int(2 x 281,216) = 562,432 -> 562,496 tok after the same +1-page alignment = 73.7 GB (derived). A host pool <= device pool only logs `L2 cache effectiveness is reduced` (`:163-171`) | `mem_cache/pool_host/base.py:150-166`; `arg_groups/fields/memory.py:108-111`; `R32/exp1_32b/startup_facts.txt:5` |
| `--hicache-write-policy` | `write_through` | `write_back, write_through, write_through_selective` | `fields/memory.py:112-118` |
| `--hicache-io-backend` / `--hicache-mem-layout` | `kernel` / `page_first` | `page_first_direct`+`kernel` is silently rewritten to `direct` (starter kit §3.2 combo does not run as kernel); keep `kernel`+`page_first` | `hicache_hook.py:127-150` |
| `--hicache-storage-backend` | None | `file, sim, mooncake, hf3fs, nixl, aibrix, dynamic, eic, simm, mori, shm` | `fields/memory.py:139-157` |
| `--hicache-storage-prefetch-policy` | `timeout` | `best_effort, wait_complete, timeout` | `fields/memory.py:158-164`; `unified_radix_cache.py:1871-1893` |
| `--hicache-storage-backend-extra-config` | None | JSON or `@file`. **No `--hicache-storage-prefetch-timeout` flag exists.** Live-path keys: `prefetch_threshold` (256), `prefetch_timeout_base` (**1.0** s), `prefetch_timeout_per_ki_token` (**0.25**), `hicache_storage_pass_prefix_keys`; timeout = base + pages x (page_size/1024 x per_ki_token), **no max cap** (the 2.0/0.1/30 `PrefetchTimeoutConfig` in `hicache_storage.py:50-55` is the dead HiRadixCache path). **32B, re-derived for THIS disk (2026-09-21):** the default budget is 1.0 s + 0.25 s per 1,024 tokens = 4,096 tok/s, so a 25K restore is allowed **7.10 s**. This SSD delivers **15,418 tok/s at its sustained read ceiling** (1.88 GiB/s at 100 % util, measured §2.2) against the A100 NVMe's 5,571 tok/s, and a 25K restore moves 3.052 GiB. Alone it takes **1.62 s here vs 4.90 s there** (derived). Sharing the device, each of N concurrent restores takes N x 1.62 s, so **four fit inside the 7.10 s budget here (6.49 s) and a fifth does not (8.11 s)** — where on the A100 exactly one fit (4.90 s) and two did not (9.80 s). That is a prediction from the disk alone, not a measurement: the nixl read path may not reach the device ceiling (§5.4 block D decides). `C18/` showed the A100 failing this three ways at once (167 of 209 lookups returned zero usable tokens, cause 3, §7.0), so this row is a primary thing to re-check. Every A100 L3 number used `wait_complete`; `timeout` is unmeasured on both boxes. Record `storage_prefetch_unfulfilled_tokens_total{reason}`; if the timeout reason dominates, raise `prefetch_timeout_per_ki_token` (e.g. 1.0) in `l3_extra.json` (§5.2) for the whole row | `hybrid_cache/hybrid_cache_controller.py:181-215`; `unified_cache/storage_attachment.py:241-245`; `unified_radix_cache.py:1864-1869`; `R32/COMPARISON.md:146-147` |
| `--hicache-storage-prefetch-retry-poll-interval` / `-max-attempts` | 0 / 4 | HANDOFF §2 suspects this path for `timeout` > `wait_complete` | `fields/memory.py:169-183` |
| `--page-size` | None -> 1 on CUDA | use 64 (nixl README example; storage hits truncated to page multiples) | `overrides.py:1453-1476` |
| `--max-total-tokens` | None | upper bound: `min(profiled, user)`, floored to the page size; the memory-pressure knob. A value above the profiled pool is ignored with `max_total_tokens=... is larger than the profiled value`: grep for it, it means the pin did not take | `kv_cache_configurator.py:2153-2168` |
| `--mem-fraction-static` | None | auto `(gpu_mem - reserved)/gpu_mem`, reserved >= 10 GiB on >60 GB GPUs. **Pass 0.85**: every A100 pool used it (32B: 281,216 tok, weights 32.59 GB, `available_gpu_mem=7.66 GB` left, `measured-A100`), so keeping it is what makes the two boxes' pools comparable; the auto value would give a different, unmeasured pool. The pool it yields **here** is §5.4 constant 1 (estimate ~700,000 tok) | `memory_hook.py:246-293`; `R32/exp1_32b/server_args.txt`; `R32/exp1_32b/startup_facts.txt:4` |
| `--kv-cache-dtype` | `auto` (bf16) | **`fp8_e5m2` for the 32B**: 1 byte/element, b = 2 x 64 layers x 8 KV heads x 128 = 131,072 B/token (bf16 would be 262,144); `measured-A100` 536,870,912 B / 4,096 tok, and a dtype/architecture property, so it is the one constant expected to port unchanged (§5.4 block B re-measures it anyway). Every byte-derived number (pools, O_DIRECT file size, bar) depends on it | `arg_groups/fields/model.py:196-216`; `R32/exp0_32b/exp0_results.json` step2_backup; server.log `KV Cache is allocated. dtype: torch.float8_e5m2, #tokens: 281216, K size: 17.17 GB, V size: 17.17 GB` |
| `--attention-backend` (**SM90: changed**) | None -> **`fa3`** on Hopper MHA | **pin `triton`** with fp8_e5m2 KV, as on the A100 — but for a different reason. On SM90 the default MHA resolution is `fa3` (`model_override_base.py:323-329`), and `fa3` + `fp8_e5m2` is then auto-rewritten to `triton` with a warning (`_attention_backend_fa3_fp8_fallback`), so on this box the pin **agrees with** the automatic resolution instead of overriding it. Keep it explicit anyway: it makes the `server_args=` log line unambiguous and it is what `R32/`/`C18/` ran. The A100's extra hazards do not apply here (SM80 defaulted to `flashinfer`, and `fa3` pinned there died at decode CUDA-graph capture with `scheduler_metadata must have shape (metadata_size)`, `R8/preflight/boot_fa3/server.log:100,127`). **If a future arm wants `fa3` for real, it must also change the KV dtype** — and then b, the pools and every §5.4 constant change with it | `arg_groups/overrides.py` `_attention_backend_fa3_fp8_fallback`; `model_override_base.py:323-329,347-351`; `R32/DEVIATIONS.md` D3 |
| FP8 weights on **SM90** (no flag) (**changed**) | auto | **native FP8, NOT Marlin.** `can_auto_enable_marlin_fp8()` returns `80 <= sm < 89`, which is false at SM90, so the A100's weight-only FP8 Marlin (W8A16, bf16 activations) path is not taken and the `Weight-only FP8 compression will be used leveraging the Marlin kernel` log line must be **absent** (§5.4 block A asserts that). `SGLANG_FORCE_FP8_MARLIN=1` would force it back on: never set it. This is the single change most likely to move `P`, and the reason §5.4 re-measures the whole recompute curve instead of scaling the A100's | `layers/quantization/fp8_utils.py:2126-2132`; `layers/quantization/fp8.py:480-484`; `marlin_utils_fp8.py:58-78`; `R32/exp1_32b/server.log:15` (the A100 line) |
| `--chunked-prefill-size` | None (auto) | must be a multiple of page size | `validation_hook.py:104-107` |
| `--enable-cache-report`, `--enable-metrics`, `--stream-response-default-include-usage` | False | usage `cached_tokens`; gates every HiCache Prometheus metric; usage chunk on every stream | `fields/serving.py:181-184,255-258`; `fields/observability.py:77` |
| env `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR` / `SGLANG_HICACHE_NIXL_BACKEND_PLUGIN` / `SGLANG_HICACHE_NIXL_USE_DIRECT_IO` (**no `BACKEND` in the third**; `..._BACKEND_USE_DIRECT_IO` matches nothing and is silently ignored) | None -> `/tmp/hicache_storage` (tmpfs here!) / `auto` (3FS > POSIX > GDS_MT > GDS) / True | dir is a comma list, the only one of the three in `environ.py` (`:738`); plugin read by `os.getenv` (`nixl_utils.py:82`), set `POSIX`; direct-IO: extra-config `use_direct_io` first, then the EnvBool (`environ.py:742`; `nixl_utils.py:38-46`). **On this box `/tmp` is on the root disk, not a tmpfs** (§1 block C), so the default store dir is the *wrong device* rather than RAM — still fatal to an L3 measurement, still silent, so `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/mnt/ssd/hicache_l3` is mandatory. O_DIRECT was **confirmed on the A100's NVMe** for the 32B (`O_DIRECT is active with a file-based backend (POSIX)`, `path-mode FILE registration active`, `R32/exp1_32b/startup_facts.txt:8-10`); on **this** disk it is expected to hold a fortiori (512 B logical sectors vs 4,096, §2.2) but is **not yet confirmed** — the §5.2 assert block checks for the line on every three-tier boot | `environ.py:738-742`; `storage/nixl/hicache_nixl.py:88-90`; `storage/nixl/nixl_utils.py:70-82,122` |
| env `SGLANG_HICACHE_FILE_BACKEND_{STORAGE_DIR,MAX_SIZE,EVICTION_RATIO}`; nixl L3 cleaner | None / None / 0.9; high 80 % / low 70 % | **file backend only** (`lru_file_evictor.py:133-135` is the sole reader of `MAX_SIZE`; `hicache_eval/scripts/start_server.sh:11`'s `L3_MAX_SIZE` does nothing for nixl). nixl has **no byte cap**: only extra-config `l3_cleaner_enabled`, `l3_cleaner_high_watermark`, `l3_cleaner_low_watermark` (**% of the filesystem** — and this filesystem is 2.27 TiB, so the 80/70 default means 1.82/1.59 TiB and is effectively inert; §5.2 sets 30/20 explicitly, §5.3 derives why) | `environ.py:725-729`; `storage/file/lru_file_evictor.py:127-148`; `storage/nixl/nixl_utils.py:49-63`; `nixl_cleaner.py:21-22`; `hicache_nixl.py:167-168` |
| `priority` in request body | None -> 0 | reaches `Req.priority` without `--enable-priority-scheduling` unless `--abort-on-priority-when-disabled`; node priority is max-propagated along the insert path, lower evicted first | `scheduler.py:3172-3195`; `radix_cache.py:769-802`; `evict_policy.py:41-46` |

```bash
# host, read-only: re-check the seven table facts that the §5.2 launch blocks depend on, straight from this checkout
cd /home/wanhr/sglang/python/sglang/srt
echo "eviction choices: $(sed -n '188p' arg_groups/choices.py)"
echo "prefetch policy default: $(grep -A6 'hicache_storage_prefetch_policy: A\[' arg_groups/fields/memory.py | grep -o '\] = "[a-z_]*"')"
echo "live timeout defaults found: $(grep -c 'prefetch_threshold", 256\|prefetch_timeout_base", 1)\|prefetch_timeout_per_ki_token", 0.25' mem_cache/hybrid_cache/hybrid_cache_controller.py) of 3"
echo "nixl env vars in environ.py: $(grep -c '^ *SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR = \|^ *SGLANG_HICACHE_NIXL_USE_DIRECT_IO = ' environ.py) of 2; PLUGIN via os.getenv: $(grep -c 'os.getenv("SGLANG_HICACHE_NIXL_BACKEND_PLUGIN"' mem_cache/storage/nixl/nixl_utils.py) of 1"
echo "page_first_direct+kernel rewrite: $(grep -c 'hicache_io_backend="direct"' arg_groups/hicache_hook.py) of 1"
echo "max_total_tokens pin warning: $(grep -c 'is larger than the profiled value' mem_cache/kv_cache_configurator.py) of 1"
echo "page_size default on CUDA: $(grep -c 'return {"page_size": 1}' arg_groups/overrides.py) of 1"
# the four rows that changed with the H200 and its disk
echo "marlin gate: $(grep -A4 'def can_auto_enable_marlin_fp8' layers/quantization/fp8_utils.py | grep -o 'return .*')"
echo "hopper MHA default: $(grep -c 'is_hopper_with_cuda_12_3 and is_no_spec_infer_or_topk_one' arg_groups/model_override_base.py) of 1"
echo "fa3+fp8_e5m2 rewrite: $(grep -c 'attention_backend == \"fa3\" and view.kv_cache_dtype == \"fp8_e5m2\"' arg_groups/overrides.py) of 1"
echo "nixl cleaner defaults: $(grep -o '_DEFAULT_\(HIGH\|LOW\)_WATERMARK = [0-9.]*' mem_cache/storage/nixl/nixl_cleaner.py | paste -sd'; ')"
```
-> prints, all **verified by running this block on 2026-09-21 at `d608a20d4`**: `eviction choices: RADIX_EVICTION_POLICY_CHOICES = ["lru", "lfu", "slru", "priority"]`
(no `fifo`); `prefetch policy default: ] = "timeout"`; then `3 of 3`, `2 of 2; ... 1 of 1`, `1 of 1`, `1 of 1`, `1 of 1`; then the four H200 rows:
`marlin gate: return 80 <= sm < 89` (**so SM90 does not take the Marlin path**, §5.4 constant 2), `hopper MHA default: 1 of 1`,
`fa3+fp8_e5m2 rewrite: 1 of 1` (together: the default resolves to `fa3` and is then rewritten to `triton`, which is why the pin is a no-op here),
and `nixl cleaner defaults: _DEFAULT_HIGH_WATERMARK = 80.0; _DEFAULT_LOW_WATERMARK = 70.0` (the values §5.2 overrides to 30/20, §5.3).
Any `0 of N`, a different choices list, or a marlin gate whose upper bound is above 89, means the checkout moved under the table: re-grep the symbol,
fix the anchor and re-verify the row before booting an arm.

**Expected result:** the table matches the checkout (the block prints exactly the values above); in particular `fifo` is absent, the prefetch
policy default is `timeout` with the live-path budget 1.0 s + 0.25 s/Ki-token and no cap, and the nixl store dir/plugin/direct-IO env vars
are the three names spelled in the table.
**If it differs:** a `0 of N` line: the file was refactored since `d608a20d4`; find the symbol with `grep -rn <symbol> python/sglang/srt`, fix
the anchor, and re-check the claim (a moved default is a changed server). A `page_size default` of 0: the default may no longer be 1 on
CUDA, but §5.2 passes `--page-size 64` explicitly so the arms are unaffected.
**Expected lessons:** four starter-kit assumptions do not hold in this checkout (`--hicache-storage-prefetch-timeout` does not exist,
`page_first_direct`+`kernel` becomes `direct`, `fifo` is rejected, the timeout defaults are 1.0/0.25/no cap rather than 2.0/0.1/30); the nixl
store defaults to the wrong device; and **SM90 defaults to `fa3`, which the fp8_e5m2 rule then rewrites to `triton`** — the same end state as the
explicit pin, reached by a different route, which is exactly the kind of agreement that stops being true after an upgrade. Each of these would
silently produce a server other than the one §5.4 measures, which is why every one of them is passed explicitly in §5.2 and asserted from the log
after each boot.

### 5.2 Launch commands

**Status:** **no server has ever been booted on this box** (checked 2026-09-21: GPU idle at 0 MiB, no container). The flag set below is the A100 launch line (`R32/exp1_32b/server_args.txt`) plus `--max-running-requests 64` and the `--max-total-tokens` pin, and it is exactly what the three `C18/` arms ran, so it ports unchanged — only the L3 path (`/mnt/ssd`, was `/mnt/nvme`) and the cleaner watermarks differ. Every **expected log value** below that mentions a token count, a GB figure or a boot time is `measured-A100` and is flagged as such: the pool assertions still work (they compare against `$L1` and the host rule, both arithmetic), but the boot time and the weight line must be recorded fresh (§2.6, §5.4 block A). `agent_cache/.current_results` does not exist yet — run §2.4 block 6 before pasting the variables block.
**Goal:** boot exactly one server per arm (a)-(g) at one pressure level, with the same L1 on every arm and a known L2 and L3, so a §7 cell compares tiering policy and not pool size; then prove from the log that the intended tiers came up before any client runs. Never re-attach or re-configure tiers at runtime: a new configuration is a new boot.
**Runs on:** `container: sglang_hicache`, under `bash` (`docker exec -it sglang_hicache bash`). The container's login shell is zsh, which does not word-split `$COMMON` strings, and the earlier string-with-inline-JSON form was split at the JSON's spaces: the arrays below are bash. Host path `/home/wanhr/sglang/agent_cache` = container `/sgl-workspace/sglang/agent_cache`; `/mnt/ssd` is the same path on both sides.
**Touches:** the GPU (one server: ~32.6 GB weights + the L1 pool); host RAM (the `--hicache-size` GB are pinned: 64 / 48 / 18 / 96 of 196 GB); `$RUNDIR/l3_extra.json`; `$RUNDIR/server_<arm>.log` and `$RUNDIR/server_<arm>.pid` (overwritten by a second boot of the same arm: copy the log into the cell dir as `server.log`, §9, before rebooting the same arm at another level); on arms (c), (d), (f) `/mnt/ssd/hicache_l3` (written through during the run; wiped as root between arms, §7.2 step 1).
**Takes:** to `fired up`: **not yet measured on this box**; `measured-A100` 278-285 s warm and ~633 s (10.6 min) JIT-cold in a new container (§2.6) — use those as the budget and the readiness timeout, and record the real values on the first two boots. The GPU is busy from launch until the server is stopped. Watch: `tail -f $RUNDIR/server_<arm>.log`. Stop: `bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh` (a script file, so its `pkill -f` cannot match its own shell, §10), or `kill $(cat $RUNDIR/server_<arm>.pid)`.

**Variables block.** Paste it first in every container shell; every arm block below uses only names defined here plus its own `set_level` call.
`set_level <L1 tokens> <host GB>` builds `COMMON` and `HOST` for one pressure level (§5.3: P0 = `262144 64`, PH = `131072 48`, PL = `131072 18`, PW = `655360 96`).
The arrays capture `$L1` / `$HSIZE` when they are DEFINED: re-assigning `L1` or `HSIZE` afterwards changes nothing (the previously built arrays boot,
i.e. the old level), so always go through `set_level`. `RUNDIR` is the stamp dir written by §2.4; `$OUT`, one cell's dir, is set in §7.2 step 0.

```bash
# container: sglang_hicache, bash (docker exec -it sglang_hicache bash). §5.2 variables block: paste before any arm block below.
MODEL=Qwen/Qwen3-32B-FP8
# RUNDIR = the stamp dir of §2.4 (host /home/wanhr/sglang/agent_cache/results/<stamp>); stops here, without closing the shell, when the stamp file is missing
[ -s /sgl-workspace/sglang/agent_cache/.current_results ] || { echo "STOP: no agent_cache/.current_results; run §2.4 block 6 (results stamp) first"; false; } \
  && RUNDIR=/sgl-workspace/sglang/agent_cache/results/$(cat /sgl-workspace/sglang/agent_cache/.current_results) \
  && mkdir -p "$RUNDIR" && echo "RUNDIR: $RUNDIR"
# knobs read by hicache_eval/scripts/hcommon.py:11-14,142 (its defaults are the 8B's: 147456 B/token, /var/hicache_l3, 120 s flush timeout)
export MODEL KV_BYTES_PER_TOKEN=131072 L3_DIR=/mnt/ssd/hicache_l3 NVME_DEV=vdc HICACHE_FLUSH_TIMEOUT=1800
# set_level <L1 tokens> <host GB>: one pressure level (§5.3) -> COMMON (every arm) and HOST (HiCache arms). Arrays capture L1/HSIZE now.
set_level() {
  L1=$1; HSIZE=$2
  COMMON=(
    # model; quantization is automatic: native FP8 on this SM90 GPU, NOT the A100's weight-only FP8 Marlin (§5.1); the boot log must show no Marlin line
    --model-path "$MODEL"
    --host 0.0.0.0
    --port 30000
    --context-length 32768
    # NO --reasoning-parser (it is in the measured launch line, §5.4): under ignore_eos the reply tail can contain <think>, and a parser would move the
    # rest of the text into reasoning_content, which the client does not re-feed (seen 2026-09-18, §4.5). Negligible for timing: it is HTTP-layer text post-processing.
    # replay template (§4.5): history renders byte-identically to the generation prompt, so a re-fed reply is a cache hit; pairs with the client's
    # chat_template_kwargs enable_thinking:false. The `think` variant only swaps the file.
    --chat-template /sgl-workspace/sglang/agent_cache/templates/qwen3_replay_nothink.jinja
    # KV dtype + attention backend: the only combination measured on this A100 (§5.1)
    --kv-cache-dtype fp8_e5m2
    --attention-backend triton
    # pools and paging: identical L1 on every arm (the pin), 0.85 = the measured mem fraction, 64 in-flight requests
    --page-size 64
    --chunked-prefill-size 8192
    --mem-fraction-static 0.85
    --max-total-tokens "$L1"
    --max-running-requests 64
    --radix-eviction-policy lru
    # metrics: the /metrics families of §6.1 and usage.cached_tokens in every response
    --enable-metrics
    --enable-cache-report
  )
  HOST=(
    # hicache tiering: L2 host pool of HSIZE GB, write-through, kernel io + page_first layout (the pair that is not rewritten, §5.1)
    --enable-hierarchical-cache
    --hicache-size "$HSIZE"
    --hicache-write-policy write_through
    --hicache-io-backend kernel
    --hicache-mem-layout page_first
  )
  echo "level set: L1=$L1 tokens, host pool=$HSIZE GB"
}
# L3 storage backend: nixl on the attached SSD; the extra-config file spells the live-path timeout defaults explicitly (budget caveat: §5.1)
# AND pins the cleaner watermarks, because the defaults (80 %/70 %) are 1.82/1.59 TiB on this 2.27 TiB filesystem and would never fire (§5.3).
# Every use of RUNDIR below is `${RUNDIR:?}`-guarded: after the STOP above these lines abort too, instead of writing /l3_extra.json at the container root.
[ -n "${RUNDIR:-}" ] && printf '{"prefetch_threshold": 256, "prefetch_timeout_base": 1.0, "prefetch_timeout_per_ki_token": 0.25, "l3_cleaner_high_watermark": 30, "l3_cleaner_low_watermark": 20}\n' > "${RUNDIR:?STOP: RUNDIR unset, see the stamp check above}/l3_extra.json"
L3=(
  --hicache-storage-backend nixl
  --hicache-storage-prefetch-policy timeout
  --hicache-storage-backend-extra-config "@${RUNDIR:?STOP: RUNDIR unset, see the stamp check above}/l3_extra.json"
)
L3WC=(
  --hicache-storage-backend nixl
  --hicache-storage-prefetch-policy wait_complete
  --hicache-storage-backend-extra-config "@${RUNDIR:?STOP: RUNDIR unset, see the stamp check above}/l3_extra.json"
)
# env for the nixl store: dir on the attached SSD (the default /tmp/hicache_storage is the ROOT disk here, 4x slower and silent, §5.1);
# plugin pinned to POSIX (POSIX was auto-selected on the A100; the explicit pin is untested on both boxes)
L3ENV=(SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/mnt/ssd/hicache_l3 SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX)
echo "l3_extra.json: $(cat "${RUNDIR:?STOP: RUNDIR unset, see the stamp check above}/l3_extra.json")"
```
-> prints `RUNDIR: /sgl-workspace/sglang/agent_cache/results/<stamp>` (any stamp is fine, an empty value is not) and
`l3_extra.json: {"prefetch_threshold": 256, "prefetch_timeout_base": 1.0, "prefetch_timeout_per_ki_token": 0.25, "l3_cleaner_high_watermark": 30, "l3_cleaner_low_watermark": 20}`. A `STOP:` line means §2.4's
stamp block 6 was skipped; the guards then skip the `l3_extra.json` write and abort both L3 arrays and the last echo (three more
`bash: RUNDIR: STOP: RUNDIR unset ...` lines, verified 2026-09-18 in an interactive bash; `L3` / `L3WC` stay undefined), so nothing else in this
section may run before §2.4 block 6 has.

COMMON = the A100 launch line (`R32/exp1_32b/server_args.txt`) plus `--max-running-requests 64` (the A100 runs left it unset -> 4096) and the
`--max-total-tokens` pin; it is also exactly what `C18/` ran. Every arm carries fp8_e5m2 KV + triton + mem-fraction 0.85: record it in `config.json`.
The `# KV dtype + attention backend` comment inside the array says "the only combination measured on this A100" — on **this** box read it as: the
combination every prior number was taken with, and the one whose automatic resolution on SM90 happens to agree with the pin (§5.1).

**Arm blocks.** One block per arm; each names its log and its level. Arms (d), (e), (f), (g) exist at two levels: pick the `set_level` line for the
level of the row (§7.1) and delete the other. After the arm block, run the readiness block and then the assert block below it.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (a) hbm_lru at P0: single tier, no HiCache (tie control; also the §5.4 row-1 server)
ARM=hbm_lru; set_level 262144 64
python3 -m sglang.launch_server "${COMMON[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=262144 tokens, host pool=64 GB` (the 64 is unused on this arm: HOST is not passed) and `launched hbm_lru: pid <N>, log ...`.
Log, once ready: `Load weight end ... quant=fp8 ... mem usage=<N> GB` (**no Marlin line on SM90**, §5.1; the A100 printed one and `mem usage=32.59 GB`),
`KV Cache is allocated. dtype: torch.float8_e5m2, #tokens: 262144` and `max_total_num_tokens=262144 ... max_running_requests=64`; NO `Allocating kv hierarchical KV host pool` line,
NO `Backend POSIX` / `O_DIRECT` / `HiCacheL3Cleaner` lines; `Tree cache initialized: ... impl=UnifiedRadixCache ... hicache_attached=False`.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (b) hbm_host at P0: L1 + 64 GB host pool, no L3
ARM=hbm_host; set_level 262144 64
python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=262144 tokens, host pool=64 GB` and the `launched hbm_host` line. Log (expected at these pools; the 100 GB form is `measured-A100`, `R32/exp1_32b/startup_facts.txt:4-5`):
`max_total_num_tokens=262144 ... max_running_requests=64`; `Allocating kv hierarchical KV host pool: 488320 tokens, 64.00 GB host memory.`; `Tree cache initialized: ... hicache_attached=True`;
NO `Backend POSIX` / `O_DIRECT` / `HiCacheL3Cleaner` lines (no storage backend).

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (c) three_tier at P0: L1 + 64 GB host pool + nixl L3 on the attached SSD, prefetch policy timeout
ARM=three_tier; set_level 262144 64
env "${L3ENV[@]}" python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" "${L3[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=262144 tokens, host pool=64 GB` and the `launched three_tier` line. Log (line shapes are `measured-A100`, `R32/exp1_32b/startup_facts.txt:5-12`, with the pool values expected at this level and the cleaner dir on the SSD):
`Allocating kv hierarchical KV host pool: 488320 tokens, 64.00 GB host memory.`; `Creating storage backend 'nixl'`; `Backend POSIX was instantiated`; `HiCacheNixl: path-mode FILE registration active.`;
`HiCacheNixl: O_DIRECT is active with a file-based backend (POSIX).`; `HiCacheL3Cleaner started: dirs=['/mnt/ssd/hicache_l3'] high=30.0% low=20.0%` (**the watermarks come from `l3_extra.json`, §5.3; `high=80.0% low=70.0%` means the extra-config did not reach the cleaner**);
`Tree cache initialized: ... hicache_attached=True`; and in the `server_args=` line `'hicache_storage_prefetch_policy': 'timeout'` with `'hicache_storage_backend_extra_config': '@/sgl-workspace/.../l3_extra.json'`.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (d) three_tier_p: arm (c) at a pressure level (§5.3). Keep ONE set_level line.
ARM=three_tier_p
set_level 131072 48   # PH: host-restore level
# set_level 131072 18   # PL: L3 level
env "${L3ENV[@]}" python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" "${L3[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=131072 tokens, host pool=48 GB` (PH) or `... host pool=18 GB` (PL) and the `launched three_tier_p` line. Log: as arm (c) but
`max_total_num_tokens=131072` and `Allocating kv hierarchical KV host pool: 366272 tokens, 48.00 GB host memory.` (PH) or `137344 tokens, 18.00 GB` (PL) (derived, §5.1 rule).

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (e) hbm_lru_p: arm (a) at L1=131072 (honest single-tier control; PH and PL share L1, so ONE boot serves both levels)
ARM=hbm_lru_p; set_level 131072 48   # the 48 is unused: HOST is not passed
python3 -m sglang.launch_server "${COMMON[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=131072 tokens, host pool=48 GB` and the `launched hbm_lru_p` line. Log: as arm (a) but `max_total_num_tokens=131072` and `#tokens: 131072` (expected).

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (f) three_tier_wc at PL: arm (d) with prefetch policy wait_complete (the policy of every 2026-09-17 measurement)
ARM=three_tier_wc; set_level 131072 18
env "${L3ENV[@]}" python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" "${L3WC[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=131072 tokens, host pool=18 GB` and the `launched three_tier_wc` line. Log: as arm (d) at PL, and the `server_args=` line shows
`'hicache_storage_prefetch_policy': 'wait_complete'` (the only difference from (d)). "Worse at idle" for `wait_complete` is an H100 8B observation, untested here.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Arm (g) hbm_host_p: arm (b) at a pressure level (host-only under pressure: at PL it recomputes what (d) reads from L3). Keep ONE set_level line.
ARM=hbm_host_p
set_level 131072 48   # PH
# set_level 131072 18   # PL
python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=131072 tokens, host pool=48 GB` (PH) or `... 18 GB` (PL) and the `launched hbm_host_p` line. Log: as arm (b) but `max_total_num_tokens=131072`
and `Allocating kv hierarchical KV host pool: 366272 tokens, 48.00 GB` (PH) or `137344 tokens, 18.00 GB` (PL).

**Readiness.** Poll `/health` **and** grep `The server is fired up and ready to roll` (the loop mirrors `agent_cache/scripts/start_server.sh`, which already waits 2400 s; if
`hicache_eval/scripts/start_server.sh:40-53` is used instead, pass `START_TIMEOUT_S=2400`, not its default 900, which is too short for a JIT-cold boot: `measured-A100` ~633 s JIT-cold, 278-285 s warm, §2.6).

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Wait (up to 2400 s) for the arm just launched; ARM = its log name (hbm_lru | hbm_host | three_tier | three_tier_p | hbm_lru_p | three_tier_wc | hbm_host_p)
ARM=<ARM>
for i in $(seq 1 2400); do
  curl -sf -o /dev/null --max-time 3 http://127.0.0.1:30000/health && grep -q 'The server is fired up and ready to roll' "$RUNDIR/server_$ARM.log" \
    && { echo "READY: $ARM after ${i}s"; break; }
  kill -0 "$(cat "$RUNDIR/server_$ARM.pid")" 2>/dev/null || { echo "STOP: $ARM died during startup; last lines:"; tail -20 "$RUNDIR/server_$ARM.log"; break; }
  sleep 1
done
```
-> prints `READY: <arm> after <N>s`. **N is not yet measured on this box**; `measured-A100` was 278-285 s warm (with the 100 GB pool; smaller pools pin
less host memory, so expect the same or a little less) and up to ~633 s JIT-cold. **Record the first warm and the first JIT-cold N here and in
`constants.json`** — they are the per-cell tax §7.3 budgets. A `STOP:` line means the server exited: read the tail (an OOM on `--hicache-size`, or a flag
rejected by argparse, shows there). Watch meanwhile with `tail -f $RUNDIR/server_$ARM.log`.

**Startup facts and the pool assertion.** Assert BOTH pools after every boot, and no `larger than the profiled value` warning, before trusting a §5.3 cell:
a stale `--hicache-size` or a pin that did not take would otherwise go unnoticed.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block and the arm block first (set_level sets L1 and HSIZE). Record startup_facts and assert the pools of the arm ARM.
ARM=<ARM>
if [ -n "${OUT:-}" ]; then FACTS=$OUT/startup_facts.txt; else FACTS=$RUNDIR/startup_facts_$ARM.txt; fi   # cell dir when §7.2 step 0 set OUT (§9 layout), else the stamp dir
grep -aE 'max_total_num_tokens|Allocating .* host memory|HiCache|storage backend|Marlin|KV Cache is allocated|Load weight end' "$RUNDIR/server_$ARM.log" > "$FACTS"
echo "startup facts: $FACTS ($(wc -l < "$FACTS") lines)"
echo "device pool:   $(grep -aoE 'max_total_num_tokens=[0-9]+' "$RUNDIR/server_$ARM.log" | head -1)   (want max_total_num_tokens=${L1:?paste the arm block first: set_level sets L1})"
echo "host pool:     $(grep -aoE 'host pool: [0-9]+ tokens' "$RUNDIR/server_$ARM.log" | head -1 || true)   (want $(( (HSIZE*1000000000/131072/64 + 1)*64 )) tokens at $HSIZE GB by the §5.1 rule; none on arms (a) and (e))"
echo "pin ignored:   $(grep -ac 'larger than the profiled value' "$RUNDIR/server_$ARM.log") warnings   (want 0)"
echo "running reqs:  $(grep -aoE 'max_running_requests=[0-9]+' "$RUNDIR/server_$ARM.log" | head -1)   (want max_running_requests=64)"
echo "storage:       $(grep -ao "'hicache_storage_backend': '[a-z]*'" "$RUNDIR/server_$ARM.log" | head -1); $(grep -ao "'hicache_storage_prefetch_policy': '[a-z_]*'" "$RUNDIR/server_$ARM.log" | head -1)   (want 'nixl' + 'timeout' on (c)(d), 'nixl' + 'wait_complete' on (f), nothing printed = None otherwise)"
echo "nixl lines:    $(grep -ac 'Backend POSIX was instantiated' "$RUNDIR/server_$ARM.log") POSIX, $(grep -ac 'O_DIRECT is active' "$RUNDIR/server_$ARM.log") O_DIRECT, $(grep -ac "HiCacheL3Cleaner started: dirs=\['/mnt/ssd/hicache_l3'\]" "$RUNDIR/server_$ARM.log") cleaner-on-ssd   (want 1 1 1 on (c)(d)(f), 0 0 0 otherwise)"
echo "cleaner marks:  $(grep -aoE 'high=[0-9.]+% low=[0-9.]+%' "$RUNDIR/server_$ARM.log" | head -1)   (want high=30.0% low=20.0% on (c)(d)(f); 80/70 means l3_extra.json did not reach the cleaner, §5.3)"
```
-> prints the labelled lines; pass when every `(want ...)` matches its value: `max_total_num_tokens=$L1`, the host pool equals 488320 / 366272 / 137344 / 732480
at 64 / 48 / 18 / 96 GB (derived, §5.1 rule; 762944 at 100 GB is the `measured-A100` anchor, `R32/exp1_32b/startup_facts.txt:5`), 0 warnings, 64 running requests,
the storage/policy pair of the arm, `1 1 1` nixl lines on the three-tier arms and `high=30.0% low=20.0%` on the cleaner.
Expected line **shapes** (`measured-A100`, `R32/exp1_32b/{startup_facts.txt,server.log}`; the values in brackets are what this box must supply):
`Load weight end. ... quant=fp8, fmt=e4m3, ... mem usage=<N> GB` [32.59 on the A100] and **no** `Weight-only FP8 compression ... Marlin kernel` line (§5.1);
`KV Cache is allocated. dtype: torch.float8_e5m2, #tokens: <L1>` and `max_total_num_tokens=<L1> ... max_running_requests=64`;
`Allocating kv hierarchical KV host pool: <tokens> tokens, <HSIZE>.00 GB host memory.`; `Backend POSIX was instantiated`; `O_DIRECT is active ... (POSIX)`;
`HiCacheL3Cleaner started: dirs=['/mnt/ssd/hicache_l3'] high=30.0% low=20.0%`; `Tree cache initialized: ... impl=UnifiedRadixCache ... hicache_attached=True`.
The two E-lines at nixl init, `posix_backend.cpp:245] POSIX path-mode open failed: nixl::FileFd("/nonexistent-nixl-probe")` followed by
`nixl_agent.cpp:545] registerMem: registration failed for the specified or all potential backends`, are the path-mode probe: benign (one pair per boot,
`measured-A100` in `R32/{exp0_32b,exp1_32b,preflight/boot_32b}/server.log`, e.g. `R32/exp1_32b/server.log:32-33`).

`--reasoning-parser qwen3` is dropped for replay (see the COMMON comment; §4.5 has the evidence). The client re-feeds `content` only, so nothing may be routed to `reasoning_content`; the stock `bench_serving` client would concatenate `reasoning_content + content` into
`generated_text` (`serving.py:143-149,531,548`) and the multi-turn wrapper re-feeds that as the assistant `content` (`:1329-1331`). Without `--chat-template` that re-fed reply is
never a cache hit, thinking on or off (measured on the tokenizer 2026-09-18, §4.5): the stock template renders history differently from the generation prompt, so the radix match
of a returning turn ends at the previous prompt (page-floored) and `recompute >= output_len(prev) + new message tokens` even on a perfect hit. The `--chat-template` line above
closes that gap; the §4.4 dry run confirms it on a booted server (its round 1-2 `device` check). Expected log line: `Detected user specified Jinja chat template with content format: string`.

**Expected result:** one `sglang.launch_server` process on the GPU (`nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l` prints 1), `/health` answers,
`READY: <arm> after <N>s` (N to be recorded on this box), and the assert block passes every `(want ...)`: device pool = the level's L1
(262144 at P0, 131072 at PH/PL, 655360 at PW), host pool = 488320 / 366272 / 137344 / 732480 tokens at 64 / 48 / 18 / 96 GB (derived), 0 pin warnings, 64 running requests,
the arm's storage/policy pair, and the three nixl lines only on (c), (d), (f) with the cleaner dir on `/mnt/ssd/hicache_l3` at `high=30.0% low=20.0%`.
**If it differs:** `max_total_num_tokens` below `$L1` with a `larger than the profiled value` warning: the pin exceeded the profiled pool (possible on arm (a), whose
profiled pool is unverified): lower L1 for every arm of the row, never for one. Host pool at the previous level's value: `set_level` was bypassed (the arrays captured the
old `HSIZE`): re-run the arm block from its `set_level` line. No `Backend POSIX` on a three-tier arm, or `HiCacheL3Cleaner ... dirs=['/tmp/hicache_storage']`: `env "${L3ENV[@]}"`
was dropped from the launch line, so L3 is on the root disk: stop the server (`stop_server.sh`) and relaunch; nothing measured on it counts. A cleaner line reading
`high=80.0% low=70.0%`: `l3_extra.json` did not reach the cleaner — the store will then grow to 1.82 TiB before anything is deleted (§5.3). `STOP: ... died`: read the tail;
`Not enough host memory available` means `--hicache-size` exceeds what `free -g` shows available (196 GB here, so 96 is comfortable and 128 would still boot, §5.3). No `READY`
in 2400 s in a fresh container: the JIT-cold boot was 10.6 min on the A100, wait for the prefill CUDA-graph capture lines before assuming a hang.
**Expected lessons:** the launch line is held identical to the A100's on purpose, so that the assert block's `(want ...)` values are the *same arithmetic* on both boxes and
any difference in a cell's result is attributable to hardware rather than to flags; every deviation an arm introduces (the pin, the smaller host pool, `timeout` with the
`@file` extra-config, the explicit POSIX plugin, no HiCache at all) is exactly the set of things the block checks from the log. Two things the log now has to tell you that it
did not before: that **no Marlin line appears** (§5.1) and that the **cleaner watermarks are 30/20, not 80/70** (§5.3) — both are silent-by-default and both would produce a
server other than the one §5.3 describes. The arrays-capture-at-definition trap still means "I changed L1" is not the same as "the server got a new L1": only the log line is.

### 5.3 Memory pressure: make the live working set exceed HBM

**Status:** **provisional, and more provisional than it was on the A100.** The pool arithmetic below is exact (it is the `pool_host/base.py` rule and a division), and the pins are chosen so that the P0/PH/PL levels are **numerically identical to the A100 box's**, which is what keeps `C18/` a usable comparison. Everything derived from the GPU's speed — f, the saturation point, the required gaps, the gap-scale multipliers, c — is an **A100 prior carried over unchanged and expected to be wrong here**, because `P` and the decode rate are both unmeasured on SM90 (§5.4). The trace inputs (mean per-turn `output_length` 182.0, emitted-turn mean gap 1.49 s, mean replay context 19.7K / 20.6K swebench) are trace properties and do carry over. **Redo every time-derived number in this section after §5.4 has run.**
**Goal:** choose (`--max-total-tokens`, `--hicache-size`) per pressure level so that the live working set exceeds the tier under test (host at PH, SSD at PL) while admission still fits in L1, so §7 rows 3-6 measure tiering and not queueing; and size the concurrency, gap scale, control cadence K and L3 disk budget those cells need.
**Runs on:** none for the arithmetic (the flags are applied through `set_level` in §5.2); the check block runs on the host, read-only.
**Touches:** read-only
**Takes:** reading time; check block < 1 s; GPU idle

Qwen3-32B-FP8, fp8_e5m2 KV, b = 131,072 B/token throughout. `b` is a dtype-and-architecture property (2 x 64 layers x 8 KV heads x 128 x 1 byte) and is the one §5.4 constant that ports without re-measurement — but §5.4 re-measures it anyway, because every byte-derived number below rests on it.

- 1 GiB of KV = 8,192 tokens exactly; 1 GB = 7,629 tokens.
- Weights on the GPU: 32.59 GB on the A100 (`measured-A100`, `R32/exp1_32b/server.log:16`). Expected to be close here (the checkpoint is the same fp8 storage), but the kernel path differs (native FP8, not Marlin repacking), so **read it off the first boot log** (§2.6) rather than assuming it.
- **Auto device pool: NOT measured on this box.** `measured-A100`: `max_total_num_tokens=281216` = 34.33 GiB at `--mem-fraction-static 0.85` on 81,920 MiB. **Estimate for this box: ~700,000 tokens** (derived: the static budget grows from 0.85 x 85.9 GB = 73.0 GB to 0.85 x 150.8 GB = 128.1 GB, i.e. +55.1 GB, and all of it goes to KV if weights and activations are unchanged: 36.86 + 55.1 = 92.0 GB / 131,072 = ~701,760 tokens = ~86 GiB). **This estimate is the first thing §5.4 replaces**, from the `max_total_num_tokens=` line of a boot without `--max-total-tokens`.
- Pool arithmetic (source): device = `min(profiled, --max-total-tokens)` floored to the page (`kv_cache_configurator.py:2153-2168`); host = `(int(GB x 1e9 // b) // 64 + 1) x 64` (`mem_cache/pool_host/base.py:151-160`; 100 GB -> 762,944, `measured-A100`). A host pool <= device pool only logs `L2 cache effectiveness is reduced` (`mem_cache/pool_host/base.py:163-171`); what `write_through` does then was not traced, so every arm keeps L2 >= L1.
- Live working set: the converted trace's replayed context averages **20.6K (swebench) / 19.7K (all)**, measured (§3.3). The tables below keep the A100's 25K upper-side assumption so the two boxes' tables line up; rescale the WS column by 0.8 for a swebench-only replay. WS = c x 25K (derived): c=4 100K tok (12.2 GiB), c=8 200K (24.4), c=12 300K (36.6), c=16 400K (48.8), c=32 800K (97.7 GiB).
- **With default pools L3 is never read, and the margin is much larger here:** at the estimated ~700K device pool plus `--hicache-size 100` (762,944) that is ~1.46M tokens = 58 sessions of 25K, against a GPU that sustains nowhere near c=58. L3 pressure needs BOTH pools shrunk, exactly as on the A100 but from a higher starting point.

| level | `--max-total-tokens` | `--hicache-size` | L1 tok (GiB) | L2 tok (GiB) | L1+L2 (sessions of 25K) | c | WS / pool | in-flight cap L1/(1.2 x 25K) -> needs f <= | role |
|---|---|---|---|---|---|---|---|---|---|
| P0 | 262144 | 64 | 262,144 (32.0) | 488,320 (59.6) | 750,464 (30.0) | 4 (8) | 0.38 L1 (0.76 L1) | 8.7 -> any f (c=8: f <= 1, no margin) | **no-pressure tie control**: WS < L1, expect all arms within noise, L2/L3 reads ~0 |
| PH | 131072 | 48 | 131,072 (16.0) | 366,272 (44.7) | 497,344 (19.9) | 8 (12) | 1.53 L1 = 0.40 (L1+L2) (2.29 L1 = 0.60) | 4.37 -> f <= 0.55 (0.36) | **host-restore arm**: L1 < WS <= L1+L2, everything that leaves HBM fits in host, L3 reads ~0 by construction |
| PL | 131072 | 18 | 131,072 (16.0) | 137,344 (16.8) | 268,416 (10.7) | 12 (16) | 1.12 (L1+L2) (1.49) | 4.37 -> f <= 0.36 (0.27) | **L3 arm**: WS > L1+L2, ~32K (c=16: 131K) tokens live only on the SSD |
| PW (**new, H200-only**) | 655360 | 96 | 655,360 (80.0) | 732,480 (89.4) | 1,387,840 (55.5) | 24-32 | to be derived after §5.4 | 21.8 -> f <= 0.91 at c=24 | **wide level**: the same pressure ratios at 5x the absolute scale, which the A100's 80 GB HBM could not reach. Not sized yet; see below |
| (defaults) | auto ~700,000 (est.) | 100 | ~700,000 (~85) | 762,944 (93.1) | ~1,463,000 (58.5) | - | - | ~23 | not an arm: L3 never read at any sustainable c |

Pools: L1 = the flag (multiples of 64, below the profiled pool), L2 derived from the host rule, GiB = tokens / 8,192. PH and PL share L1, so the
single-tier control (e) boots once for both. Host pinned RAM 64 / 48 / 18 (/ 96) GB of **196 GB**, no swap: the A100's "keep `--hicache-size` <= 64 or the
box runs out of RAM" constraint is **much looser here** — 96 GB still leaves ~100 GB, and even 128 GB (976,576 tokens) would boot. Keep the P0/PH/PL
values as they are anyway: they are what `C18/` used.

**Why P0/PH/PL are deliberately unchanged, and what PW is for.** The pins are *below* the profiled pool on both boxes, so the same three numbers
produce the same three experiments; carrying them over is what lets a row on this box be read against `C18/` and `R32/`. What the H200 adds is
**headroom**, and `C18/README.md` named exactly the constraint that headroom relieves: the A100's three-arm tie happened because L1+L2 = 157K sat
against a live set of ~500K, so admission, not tiering, set TTFT (§7.0). PW is the level that tests the same hypothesis with condition (a) of that
README satisfied — L1 big enough to admit the live in-flight set — while still putting the parked set beyond L1+L2. It is listed, not sized: its c,
gap scale and cell time need the §5.4 constants first, and it must not be run before row 2 has tied at P0.

```bash
# host, read-only: reproduce the derived pool and working-set numbers of the table from the §5.1 host-pool rule (b = 131,072, page 64)
python3 - <<'EOF'
b = 131072; page = 64
host = lambda gb: (int(gb * 1e9 // b) // page + 1) * page
for gb in (100, 96, 64, 48, 32, 18, 16):
    print(f"host pool at --hicache-size {gb}: {host(gb)} tokens = {host(gb)/8192:.1f} GiB")
for c in (4, 8, 12, 16, 24, 32):
    print(f"WS at c={c}: {c*25000} tokens = {c*25000/8192:.1f} GiB")
print(f"L1+L2 at P0 / PH / PL / PW: {262144+host(64)} / {131072+host(48)} / {131072+host(18)} / {655360+host(96)} tokens"
      f" = {(262144+host(64))/25000:.1f} / {(131072+host(48))/25000:.1f} / {(131072+host(18))/25000:.1f} / {(655360+host(96))/25000:.1f} sessions of 25K")
print(f"in-flight cap L1/(1.2 x 25K): P0 {262144/30000:.2f}, PH and PL {131072/30000:.2f}, PW {655360/30000:.2f}")
EOF
```
-> prints `host pool at --hicache-size 100: 762944 tokens = 93.1 GiB` (the A100-measured anchor, and a pure-arithmetic rule so it holds here too), then
732480 / 488320 / 366272 / 244160 / 137344 / 122112 for 96 / 64 / 48 / 32 / 18 / 16, WS 100000 / 200000 / 300000 / 400000 / 600000 / 800000 tokens,
`L1+L2 ... 750464 / 497344 / 268416 / 1387840 tokens = 30.0 / 19.9 / 10.7 / 55.5 sessions`, caps `8.74`, `4.37` and `21.85` (verified 2026-09-21).
Any other value means the rule in `mem_cache/pool_host/base.py:151-160` changed and the table, and the assert block of §5.2, must be re-derived.

**Two constraints per cell** (the WS ratio alone is not enough): (i) `L1 >= 1.2 x f x c x context`, because the PrefillAdder admits only within available +
evictable device tokens (`schedule_policy.py:636-658,807-808,1215,1228`) and beyond that TTFT is admission queueing (HANDOFF §4; and it is what `C18/`
actually measured, §7.0); (ii) `WS = c x context > L1` (host arm) or `> L1+L2` (L3 arm). With the in-flight fraction `f = turn_time / (turn_time + gap)`
they require `f <= L1 / (1.2 x WS)`: **< 0.83 for any host arm and < 0.42 for any L3 arm, at any L1** — the L1 cancels, so a bigger GPU does not by itself
buy this. What a faster GPU buys is a **smaller f at the same gap**, because `turn_time` shrinks: that is the mechanism by which the H200 may reach a
regime the A100 could not, and it is unquantified until §5.4 gives the decode rate.

Where a 25K prefix lives decides TTFT. The A100's numbers (`measured-A100`, derived from the §5.4 fits): L1 0.36 s, L2 0.44 s, L3 4.96 s, recompute 18.0 s.
**None of the first three port** (they are fits through GPU-side copy rates) and the fourth certainly does not. What can be said for this box before
§5.4 runs, from the disk alone: a 25K L3 restore moves 25,000 x 131,072 = 3.28 GB = 3.05 GiB, which at this SSD's measured **1.88 GiB/s sustained
ceiling** takes **>= 1.62 s** if it gets the whole device, versus **>= 4.90 s** on the A100's 0.623 GiB/s (derived, §2.2). That is the single largest
mechanical improvement of the port, and it is the reason to re-run the L3 rows here at all.

**Unmeasured input 1, returning turn on a device-resident prefix (+~1K tokens):** A100 model value ~1.4 s. **Unmeasured input 2, decode rate.** Both are
§5.4 open constants, both were open on the A100 too, and on this box **the whole TTFT-vs-length curve joins them** — so the A100's consequence paragraph
(f = 0.76-0.83 at c=1, ~0.90-0.92 at c=8; saturation near c ~ 8; required mean gaps of ~13-16 s at PH c=8 and ~31-39 s at PL c=12; gap-scale multipliers
x9-11 / x17-23 / x21-26 / x31-39 on the 1.49 s emitted-turn mean) is reproduced here **only as the prior to beat**, not as a plan. Expect every required
gap and every multiplier to come down in proportion to whatever speed-up §5.4 measures. Record measured f per cell (`num_running_reqs`/c from
`memtime.csv`, §6.1); require median queue depth ~0 in the window or bin by it (§9.1).

**L3 write-through capacity (32B, re-derived for this disk).** Write ceiling **measured 1.88 GiB/s sustained at 100 % util** (§2.2) =
**15,418 tok/s** at b = 131,072 (derived). The GPU cannot make KV faster than it prefills, and on the A100 that was 1,220-2,270 tok/s, i.e. at most 15 %
of *this* disk's write ceiling. Even if the H200 prefills 3x faster, write-through stays well inside the disk. **This is a qualitative change from the
A100**, where the same comparison was "at most 73 % of the ceiling, with ~10 s bursts AT the ceiling after each 32.5K prompt" — and where, in `C18/`,
`h2s_io` occupied 835 s of a 2,780 s run at the write ceiling and starved the reads (cause 3 of the null result, §7.0). The headroom does not remove the
write amplification itself (cause 4: every recomputed turn re-inserts its KV and write-through backs it up again — 2.55M tokens written for ~480K tokens
of distinct content), it only stops that amplification from saturating the device: the same 312 GiB of `h2s` traffic occupies **~166 s here against the
835 s measured there** (derived). **Watch `w_await` and `%util` on `vdc` in PL cells anyway** and record them: the claim above is arithmetic, not a
measurement under load, and reads and writes share one 1.88 GiB/s budget rather than two.

**L3 cleaner budget — re-derive, do not carry over.** nixl deletes from 80 % of the filesystem down to 70 %, oldest mtime first
(`nixl_cleaner.py:21-22,148-157,181-184`), and **the filesystem is now 2.27 TiB, not 368 GiB**: 80 % is **1.82 TiB = 14.9M tokens**, against the A100's
294 GiB = 2.41M. Two consequences. (1) The cleaner is effectively **inert** for a single cell — a `C18/`-sized cell wrote 312 GiB, which is 13 % of this
disk — so the A100's careful K-vs-watermark arithmetic (K = 40 at c=8, K = 72 at c=12 with 95/85 watermarks) is **no longer a binding constraint**, and K
can be chosen for the GPU budget alone (§7.3). (2) Across a multi-day campaign the store now accumulates until it hits 1.82 TiB, at which point the
cleaner fires mid-cell and becomes a hidden second eviction policy — the exact failure the A100 sized against. **Therefore: keep wiping L3 between arms
(§7.2 step 1), and set explicit watermarks in `l3_extra.json` so the budget is a stated number rather than an accident of disk size.** The §5.2 variables
block writes `"l3_cleaner_high_watermark": 30, "l3_cleaner_low_watermark": 20` (= 698 GiB high, 465 GiB low; derived: 30 % of 2.27 TiB), which is >= 2x the
biggest observed cell and still leaves 1.5 TiB of the disk untouched. Everything prefilled is written through, controls included (`measured-A100`,
`R32/exp1_32b/metrics_after.txt`: 197,184 prefilled -> 195,648 tokens backed up to L2 and to L3). Budget per three-tier cell (derived):
`N_conv x 4.0 GiB (final context <= 32.7K) + N_ctl x 3.05 GiB (25K control) <= 698 GiB`, `N_ctl ~ 39 x N_conv / K` at 40 turns — at c=12 / 60 conversations
that is 240 GiB of conversations and room for ~150 controls, i.e. **K is unconstrained by the disk at every c this GPU can sustain.** Check `df /mnt/ssd`
and `l3_stats.json` after the first c=8 cell and record cleaner deletions (§9.1); a deletion during a cell means this paragraph is wrong.

**Expected result:** the level table above, applied through `set_level` (P0 `262144 64`, PH `131072 48`, PL `131072 18`, PW `655360 96`), gives
L1+L2 = 750,464 / 497,344 / 268,416 / 1,387,840 tokens (30.0 / 19.9 / 10.7 / 55.5 sessions of 25K; derived, reproduced by the check block) and the
admission caps 8.7 / 4.37 / 4.37 / 21.8. Those numbers are exact. **Every other number in this section is an A100 prior and is expected to change**;
the §5.4 run is what replaces them, and no row past 2 may be sized until it has.
**If it differs:** a booted arm shows pools other than the table's (the §5.2 assert block): fix the boot, never the table. The profiled pool comes back far
from ~700,000: the estimate's assumption (weights and activations unchanged from the A100) is wrong — take the real number and re-derive the PW row, which
is the only row that depends on it. Measured f per cell above the cap column: the cell is queueing, not tiering; raise the gap scale for that row or lower
c, and keep the short-gap result as the overload reference (§7.1 row 3).
**Expected lessons:** the pool ratio alone does not make a tiering experiment: in-flight sessions must fit in L1 (constraint i) while the parked ones must
not (constraint ii), and because the L1 cancels out of `f <= L1/(1.2 x WS)`, **a bigger GPU does not relax that trade-off — only a shorter turn or a longer
gap does.** The H200 helps through speed (smaller `turn_time`, hence smaller f at the same gap) and through the disk (a 25K L3 restore has a 1.62 s floor
here against 4.90 s there), not through HBM size. And the two things the A100 campaign got wrong are now cheap to get right: the disk is no longer the
bottleneck for write-through, and the cleaner is no longer a budget to design around — which leaves the prefetch-admission limit (cause 2 of `C18/`:
`cache_controller.py` caps in-flight prefetch at half the host pool) as the one mechanism from that null result this box does **not** automatically fix.

### 5.4 Constants for Qwen3-32B-FP8 on this box

**Status:** **not started — this is the single most important open step on this box, and it is matrix row 1.** Nine constants that the A100 measured are
invalid on SM90 and are listed below with `measured-A100` values as priors and a command each. The A100's own two open constants (decode rate,
prefix-extension TTFT) were never measured on either box and are open here too. Until this step runs, **every c, f, gap scale, K and cell time in §3.5,
§5.3, §7.1 and §7.3 is an Ampere number being used on Hopper.**
**Goal:** re-measure, on this GPU and this disk, the constants every later step reads: the profiled device pool, `b`, the recompute TTFT-vs-length curve
and its slope `P`, the L1/L2/L3 tier fits, the decode rate per sequence, and the TTFT of extending a long cached prefix — and write them into
`constants.json` (§9.2) so no number is re-typed elsewhere.
**Runs on:** `container: sglang_hicache`, bash, against servers booted per §5.2; the tier-fit and P sweeps run from `cd /sgl-workspace/sglang/hicache_eval/scripts` (`hcommon` is imported from cwd). The open-constants block can also be driven from the host with the `docker exec` wrapper shown.
**Touches:** `$RUNDIR/{decode_delta,b_measure,p_measure,tier_measure}/`, `$RUNDIR/constants.json` (§9.2); the server's radix tree and host pool (the tools flush via `/flush_cache`). Nothing on `/mnt/ssd` for the recompute and decode measurements (arm (a) has no L3); the tier sweep does write L3.
**Takes:** ~25 min of GPU-busy measurement in total (estimate: pool 0 min — it is a log line; b ~1 min; P sweep ~5 min at A100 speed, less here; tier sweep ~10 min; open constants ~6 min; ITL cross-check ~2 min), plus one or two boots

**The nine constants to re-measure, with their A100 priors.** A prior is what the number was on the A100; it is stated so a wildly different result is
recognisable as a mistake rather than a discovery, and it is **never** a value to use.

| # | constant | `measured-A100` prior | why it cannot port | how to measure here |
|---|---|---|---|---|
| 1 | profiled device pool at `--mem-fraction-static 0.85` | 281,216 tok = 34.3 GiB | HBM 143,771 MiB vs 81,920 | boot arm (a) **without** `--max-total-tokens`, read `max_total_num_tokens=` (block A) |
| 2 | weights on GPU | 32.59 GB, fp8 e4m3, weight-only Marlin (W8A16) | SM90 runs **native FP8**, not Marlin (`fp8_utils.py:2126-2132`) | `Load weight end.` line of the same boot (block A) |
| 3 | `b`, KV bytes/token | 131,072 | dtype property; expected identical | block B (one 4,096-token write-through probe, D2H counters) |
| 4 | recompute TTFT at L = 512 … 32,512 | 0.2393 … 26.6922 s | prefill is compute-bound; different SM, different FP8 kernel | block C (`exp1.py --tiers recompute`, 3 reps x 7 lengths) |
| 5 | `P`, marginal prefill slope, and the bar `b x P` | 1,226 tok/s, intercept -1.251 s; 0.150 GiB/s | follows from 4 | linear fit over block C's medians |
| 6 | L1 / L2 tier fits | 9.81 / 7.75 GiB/s, intercepts 0.053 / 0.048 s | device- and host-copy rates, PCIe Gen5 here vs Gen4 | block D (`exp1.py --tiers device,host`) |
| 7 | L3 tier fit | 0.623 GiB/s + 0.058 s | **different disk**: 1.88 GiB/s ceiling vs 0.68 | block D (`--tiers storage`) on a three-tier arm |
| 8 | decode rate per sequence at B = 1, 4, 8 | **never measured on either box** | — | block E (`one_batch_server` warm-prefix run) + block F (ITL cross-check) |
| 9 | TTFT of extending a ~24.5K cached prefix by ~1K | **never measured on either box** | — | block E, `last_ttft` at bs=1 |

Boot time is a tenth value worth recording (`measured-A100`: 278-285 s warm, ~633 s JIT-cold): read it off the `READY:` line of the first boots (§2.6).

**Setup that every measured row must share** (so the rows are comparable to each other and to `R32/`): fp8_e5m2 KV, `--attention-backend triton`,
`--mem-fraction-static 0.85`, page 64, ctx 32768, idle server, one in-flight probe with `max_tokens=1`, n = 3 reps. The `timeout` policy and concurrent
load stay unmeasured here as they were there.

The one boot in this runbook that must NOT carry the `--max-total-tokens` pin: the pin hides the profiled pool, which is the value being
measured. `set_level` always builds the pin into `COMMON`, so this block spells the flags out instead of reusing the array.
```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first (for RUNDIR and MODEL).
# Block A (constants 1 and 2): boot arm (a) WITHOUT --max-total-tokens and read the profiled pool and the weight line off the log
ARM=probe_pool
python3 -m sglang.launch_server \
  --model-path "$MODEL" --host 0.0.0.0 --port 30000 --context-length 32768 \
  --chat-template /sgl-workspace/sglang/agent_cache/templates/qwen3_replay_nothink.jinja \
  --kv-cache-dtype fp8_e5m2 --attention-backend triton \
  --page-size 64 --chunked-prefill-size 8192 --mem-fraction-static 0.85 \
  --max-running-requests 64 --radix-eviction-policy lru \
  --enable-metrics --enable-cache-report \
  > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM: pid $(cat "$RUNDIR/server_$ARM.pid")"
# ... wait with the §5.2 readiness block (ARM=probe_pool), then:
echo "profiled device pool: $(grep -aoE 'max_total_num_tokens=[0-9]+' "$RUNDIR/server_$ARM.log" | head -1)"
echo "kv cache line:        $(grep -a 'KV Cache is allocated' "$RUNDIR/server_$ARM.log" | head -1)"
echo "weight line:          $(grep -a 'Load weight end' "$RUNDIR/server_$ARM.log" | head -1)"
echo "marlin mentioned:     $(grep -ac 'Marlin' "$RUNDIR/server_$ARM.log")   (want 0 on SM90)"
echo "attention backend:    $(grep -ao \"'attention_backend': '[a-z0-9_]*'\" "$RUNDIR/server_$ARM.log" | head -1)"
echo "boot seconds:         (from the READY: line of the readiness block)"
```
-> prints the four labelled lines. **Pass conditions:** `max_total_num_tokens=` some value near the ~700,000 estimate (§5.3) — write the exact number into
`constants.json` as `L1_TOKENS` and into the §5.3 PW row; `KV Cache is allocated. dtype: torch.float8_e5m2, #tokens: <same>`; a `Load weight end. ...
quant=fp8 ... mem usage=<N> GB` line; **`marlin mentioned: 0`** — a non-zero count means the A100's W8A16 path is running on Hopper silicon and every
later number would be wrong (check for `SGLANG_FORCE_FP8_MARLIN` in the environment); `'attention_backend': 'triton'`. Then stop this server
(`kill $(cat "$RUNDIR/server_$ARM.pid")`): every other block boots an arm with the pin.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first. Make the re-check dirs and move to the scripts dir (hcommon/cachectl/probe are imported from cwd)
cd /sgl-workspace/sglang/hicache_eval/scripts && mkdir -p "$RUNDIR"/{b_measure,p_measure,tier_measure,decode_delta} \
  && echo "cwd:  $(pwd)" && echo "dirs: $(ls -d "$RUNDIR"/{b_measure,p_measure,tier_measure,decode_delta} | tr '\n' ' ')"
```
-> prints `cwd: /sgl-workspace/sglang/hicache_eval/scripts` and the four created directories (two labelled lines; a `cd:` error means the scripts dir is missing and nothing was created).

```bash
# container: sglang_hicache, bash, cwd = the scripts dir, RUNDIR from the §5.2 variables block; against a HiCache arm ((b) at P0).
# Block B (constant 3): one 4096-token write-through probe, then the D2H backup counters (exp0.py:90-104 procedure)
python3 cachectl.py scrape "$RUNDIR/b_measure/before.txt" && python3 probe.py --len 4096 --seed 4242 && python3 cachectl.py drain \
  && python3 cachectl.py scrape "$RUNDIR/b_measure/after.txt" \
  && python3 cachectl.py delta "$RUNDIR/b_measure/before.txt" "$RUNDIR/b_measure/after.txt" | grep -E 'hicache_backup_(bytes|tokens)_total'
```
-> prints the two counter deltas; pass when bytes / tokens == **131072 exactly** (`measured-A100`: 536,870,912 / 4,096). This is the one constant expected to be unchanged; any other ratio means a different KV dtype or layer count is being served (§5.1 `--kv-cache-dtype` row) and every byte-derived number in §5.3 is off.

```bash
# container: sglang_hicache, bash, cwd = the scripts dir, RUNDIR from the §5.2 variables block; against arm (a) at P0.
# Block C (constants 4 and 5): recompute-only sweep, 3 reps x 7 lengths (~5 min at A100 speed, less here; GPU busy;
# watch $RUNDIR/p_measure/run.log; Ctrl-C stops it). exp1.py:22 reads os.environ["RESULTS"] unconditionally (KeyError without it)
# and writes to $RESULTS/$EXP1_OUT; it never sources env.sh, so the frozen-dir guard is not involved
RESULTS=$RUNDIR EXP1_OUT=p_measure python3 exp1.py --reps 3 --tiers recompute --lengths 512,1024,2048,4096,8192,16384,32512 > "$RUNDIR/p_measure/run.log" 2>&1 \
  && echo "P sweep done: $(wc -l < "$RUNDIR/p_measure/run.log") log lines in $RUNDIR/p_measure/run.log"
```
-> prints `P sweep done: ...`. **Record the seven medians and the slope fit here.** `measured-A100` for comparison: 0.2393 / 0.4552 / 0.9021 / 1.8597 /
4.0303 / 9.8764 / 26.6922 s, slope P = 1,226 tok/s with intercept -1.251 s, and a quadratic `T_rec(L) = 0.0626 + 3.752e-4 L + 1.365e-8 L^2` (max residual
31 ms) that §3.5 and §7.3 use. **Expect the H200 to be materially faster** — native FP8 GEMMs instead of W8A16 Marlin dequant, on a newer SM — but the
curve must still be re-fitted, not scaled: it is superlinear, and a single speed-up factor would misstate both the slope and the intercept. Re-fit the
quadratic and put it in `constants.json`; §3.5's `T_rec` and §7.3's `GPU_s_per_turn` read it from there.

```bash
# container: sglang_hicache, bash, cwd = the scripts dir, RUNDIR from the §5.2 variables block; against a THREE-TIER arm ((c) at P0).
# Block D (constants 6 and 7): the tier sweep. Same lengths, one pass per tier; the storage pass writes and then reads L3 on /mnt/ssd.
RESULTS=$RUNDIR EXP1_OUT=tier_measure python3 exp1.py --reps 3 --tiers device,host,storage --lengths 512,1024,2048,4096,8192,16384,32512 > "$RUNDIR/tier_measure/run.log" 2>&1 \
  && echo "tier sweep done: $(wc -l < "$RUNDIR/tier_measure/run.log") log lines in $RUNDIR/tier_measure/run.log"
```
-> prints `tier sweep done: ...` and writes a per-tier TTFT table. **Record the three fits (rate + intercept) here.** `measured-A100`: L1 9.81 GiB/s +
0.053 s, L2 7.75 GiB/s + 0.048 s, L3 0.623 GiB/s + 0.058 s. **The L3 row is the one that decides the study**: this disk's contended read ceiling is
1.88 GiB/s (measured, §2.2), so a single idle restore should land far above 0.623 — if it does not, the bottleneck is nixl or the prefetch path, not the
device, and that is itself the finding. Note the sweep runs `wait_complete` at idle and single-stream, exactly as the A100's did; concurrent-restore
behaviour is what §7.1 rows 5-6 measure, not this block.

```bash
# host: Block E (constants 8 and 9): measure the decode rate and the prefix-extension TTFT against arm (a) at P0 (~6 min, estimate; GPU busy).
# RUNDIR is the CONTAINER path built from the host stamp file (§2.4); an empty or bare RUNDIR would make the inner command write to
# /decode_delta or results/decode_delta in the container, silently, hence the guard
STAMP=$(cat /home/wanhr/sglang/agent_cache/.current_results 2>/dev/null); [ -n "$STAMP" ] || { echo "STOP: no agent_cache/.current_results; run §2.4 block 6 (results stamp) first"; false; } \
  && RUNDIR=/sgl-workspace/sglang/agent_cache/results/$STAMP && echo "RUNDIR (container path): $RUNDIR" \
  && docker exec -e RUNDIR="$RUNDIR" sglang_hicache bash -lc 'cd /sgl-workspace/sglang && mkdir -p "$RUNDIR/decode_delta" \
  && python3 -m sglang.benchmark.one_batch_server \
    --model-path Qwen/Qwen3-32B-FP8 \
    --base-url http://127.0.0.1:30000 \
    --dataset-name random-ids \
    --batch-size 1 4 8 \
    --input-len 25600 \
    --output-len 220 \
    --cache-hit-rate 0.96 \
    --skip-warmup \
    --show-report \
    --result-filename "$RUNDIR/decode_delta/warm_prefix.jsonl" 2>&1 | tee "$RUNDIR/decode_delta/warm_prefix.log"'
```
-> prints `RUNDIR (container path): /sgl-workspace/sglang/agent_cache/results/<stamp>` (a `STOP:` line means §2.4's stamp is missing: nothing runs), then
one report block per batch size with `last_ttft` and `output_throughput`, and `warm_prefix.jsonl` gains three lines (batch_size 1, 4, 8) with those fields
(`one_batch_server.py:415-438`). `last_ttft` at bs=1 is constant 9; `output_throughput / batch_size` is constant 8 at B = 1, 4, 8.
**Pass condition:** `last_ttft` at bs=1 must be far below a full 25.6K recompute (whatever block C measured it to be); if it is near that value the 96 %
warm-up did not take (flush or seed mismatch) and the run measured a recompute, not the extension.
Prerequisite: arm (a) at P0 is up and asserted (§5.2), and one discarded long probe has been sent (the first long prefill in a fresh container took 9.0 s
once on the A100, `R8/DEVIATIONS.md` D9): from a §5.2 container shell, `cd /sgl-workspace/sglang/hicache_eval/scripts && python3 probe.py --len 4096 --seed $RANDOM`
(prints one JSON line; discard it). Per batch size the tool flushes, prefills `int(25600 x 0.96)` = 24,576 tokens with `max_new_tokens=1`
(`one_batch_server.py:462-524,621-625`), then times 1,024 uncached tokens on that prefix plus 220 decoded tokens.
Flag groups: target (`--model-path`, `--base-url`: an external server, no launch); prompt shape (`random-ids`, 25,600 tokens of which 96 % are pre-cached,
220 output tokens); batch sizes 1 4 8; `--skip-warmup --show-report`; output files under `$RUNDIR/decode_delta/`.
Watch from another host shell: `tail -f /home/wanhr/sglang/agent_cache/results/$(cat /home/wanhr/sglang/agent_cache/.current_results)/decode_delta/warm_prefix.log`.
Stop: `docker exec sglang_hicache pkill -f sglang.benchmark.one_batch_server` (no wrapping `bash -c`, so the pattern cannot match its own shell, §10).

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first (RUNDIR). Block F: cross-check of the decode rate at short context:
# median ITL at c=1 (~2 min, estimate; GPU busy; Ctrl-C stops it)
mkdir -p "$RUNDIR/decode_delta" && python3 -m sglang.benchmark.serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 30000 \
  --model Qwen/Qwen3-32B-FP8 \
  --dataset-name random-ids \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 1.0 \
  --num-prompts 8 \
  --max-concurrency 1 \
  --warmup-requests 1 \
  --output-details \
  --output-file "$RUNDIR/decode_delta/itl_c1.jsonl"
```
-> prints the serving summary with `Median ITL`; `1 / median ITL` is the batch-1 decode rate at a 1K context, which must be consistent with (and a little
above) block E's bs=1 rate at 25K, since a 1K context reads less KV per step. `random-ids` and not `random`: `random` would download ShareGPT; `probe.py:49`
fixes `max_tokens=1`, so it cannot give a decode rate.

**Admission rule (HANDOFF §2): a tier pays only if `bandwidth(tier) > b x P`.** On the A100 that gave: L2 clears the bar 52x, L3 4.2x, and an L3 hit was
faster than recompute at 7 of 7 lengths — which is why the 32B was chosen as the evaluation model there, and why the 8B (bar 1.22 GiB/s vs 0.625
delivered) was the documented negative case. **Both sides of that inequality move here**, in opposite directions: `P` goes up (native FP8 on a newer SM),
which raises the bar, and the disk goes up 2.8x, which raises L3's side. Which way the ratio lands is exactly what blocks C and D decide, and it is a
go/no-go input (§9.3), not a detail. Quote the measured ratio and the 7-length comparison; never the fit break-even, which is an artefact of the
superlinear recompute curve.

**Expected result:** ten values recorded here and in `constants.json` (§9.2): profiled device pool, weight-load size and kernel (Marlin absent), `b`
(expected 131,072), seven recompute medians plus the re-fitted slope/intercept/quadratic, three tier fits, decode rate at B = 1/4/8, prefix-extension
TTFT, and the warm and JIT-cold boot times. Not one of them exists for this box today.
**If it differs:** `marlin mentioned` non-zero in block A: something forces the A100 kernel; fix it before anything else. Block B's ratio not 131,072:
the dtype is not what the flags say. Block C's medians *slower* than the A100's: implausible on this GPU — check `nvidia-smi` for a second process, `top`
for scheduler CPU starvation (16 vCPU shared by scheduler, HiCache threads, tokenizer and client, §10), and that the pin took, before believing it.
Block D's L3 rate near the A100's 0.623 GiB/s despite a 1.88 GiB/s device: the nixl path, not the disk, is the limit — profile it (§8) before sizing rows
5-6, because it would mean the disk upgrade bought nothing. Block E exits on `resolve_once()` errors with `--base-url`: `one_batch_server.py:1494-1495` is
the unverified path; fall back to block F plus `--random-input-len 25600`.
**Expected lessons:** this step is the port. Everything else in §2-§4 is plumbing that either works or fails loudly; these nine numbers fail *silently*,
by making a cell look correctly sized when it is not — which is how the A100 campaign spent a day producing `C18/`, a three-arm tie that could not be
attributed because the cell had been sized from estimates (§7.0). Two specific outcomes to watch for: if `P` rises faster than the disk did, the L3
admission ratio *falls* despite the better SSD and PL becomes harder to demonstrate, not easier; and if the decode rate rises a lot, `turn_time` and hence
f fall, the required mean gaps of §5.3 shrink, and rows 4-6 get cheaper in both gap scale and wall time. Those two pull in opposite directions, which is
why both must be measured before any row past 2 is sized.

---

## 6. Instrumentation

This section fixes what is recorded while a cell runs and how each recorded quantity is read back, so that every number in the §9.1 report
(TTFT per turn index binned by queue depth, restore-hit rate, memory-time, paired restore-minus-control differences) can be computed offline
from the files in `results/<stamp>/<cell>/` alone. §6.1 is the only step that runs during a cell (the 1 s gauge sampler; the counters are
snapshotted by §7.2 steps 6 and 9); §6.2-§6.5 are source-verified facts and rules that `agent_cache/scripts/analyze.py` (§11, not written yet)
and the client patch (§4) implement. At the end the reader knows which `/metrics` family answers which question, where the per-request tier
split comes from without any server change, what to log when a client cannot read that split, and why every restore TTFT is paired with a control.

### 6.1 Metrics to scrape and cadence (`/metrics`, needs `--enable-metrics`)

**Status:** families, meanings and anchors verified against `6ec32e6b7` on 2026-09-18 and re-checked against `d608a20d4` on 2026-09-21 (the three intervening commits add `HICACHE_EVT` log lines, §6.6, and change no metric definition). The 1 s sampler below ran on the A100 box for the `C18/` cells; **it has never run here** (no cell has run on this box).
**Goal:** record device-pool and host-pool occupancy and the queue depth at 1 s resolution for the whole cell, and name the counter family that
answers each question, so that §6.4 can attach a queue depth to every turn and bound memory-time, and §9.1 can bin TTFT by queue depth at send time.
**Runs on:** `container: sglang_hicache`, `bash`, in the §7.2 step 0 shell (it defines `OUT`, this cell's dir); `curl` and `awk` are in the image.
The server must be up with `--enable-metrics` (part of the §5.2 `COMMON` array; it gates every HiCache Prometheus family, §5.1).
**Touches:** writes `$OUT/memtime.csv` and `$OUT/.memtime.pid` (container `/sgl-workspace/sglang/agent_cache/results/<stamp>/<cell>/` =
host `/home/wanhr/sglang/agent_cache/results/<stamp>/<cell>/`, root-owned until the §7.2 step 12 chown); one `GET /metrics` per second; nothing else.
**Takes:** runs in the background for the whole cell (cell length is re-derived after §5.4; the A100's `C18/` cells were ~46 min each, §7.0); the sampler costs one `curl` per second and no GPU
(the client and the server are the GPU users; the sampler is stopped in §7.2 step 8, before the drain).

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

Cadence: a 1 s sampler for the gauges (memory-time, queue depth), and a before/after snapshot for the counters (`cachectl.py scrape` in §7.2
steps 6 and 9, `cachectl.py delta` for the difference). Families with labels (`dropped{reason,pool}`, `backup_tokens{pool}`,
`load_back_tokens{pool}`, `prefetch_unfulfilled{reason}`, `prefill_effective{mode}`, `cached_tokens{cache_source}`) need
`hcommon.parse_metrics_labeled` for per-label reads (`parse_metrics` sums across labels, so totals stay right;
`hicache_eval/scripts/hcommon.py:94-123`). `/server_info` has no cache/HiCache fields (`scheduler.py:4935-4996`); do not scrape it for hit rate.

The sampler keeps only the family name of each gauge (labels stripped) and writes one CSV row per second; the `hicache_host_*` gauges exist
only on HiCache arms (they are registered under `if self.enable_hierarchical_cache`, `metrics_collector.py:641-649`), so those two columns
are empty on arms (a) and (e).

```bash
# container: sglang_hicache, bash, in the §7.2 step 0 shell (it defines OUT = this cell's dir)
# Start the 1 s gauge sampler in the background: one GET /metrics per second, one CSV row per sample, until killed
# Every line that touches $OUT is guarded: with OUT unset the `:?` lines abort and the loop and pid file are skipped (no /memtime.csv at the container root, no orphan loop)
OUT=${OUT:?paste §7.2 step 0 first: it defines OUT}
BASE=http://127.0.0.1:30000
CSV=${OUT:?paste §7.2 step 0 first: it defines OUT}/memtime.csv
KEYS="sglang:token_usage sglang:num_used_tokens sglang:kv_used_tokens sglang:kv_evictable_tokens sglang:kv_available_tokens sglang:hicache_host_used_tokens sglang:hicache_host_total_tokens sglang:max_total_num_tokens sglang:num_running_reqs sglang:num_queue_reqs"
[ -n "${OUT:-}" ] && echo 't,token_usage,num_used_tokens,kv_used_tokens,kv_evictable_tokens,kv_available_tokens,hicache_host_used_tokens,hicache_host_total_tokens,max_total_num_tokens,num_running_reqs,num_queue_reqs' > "$CSV"
[ -n "${OUT:-}" ] && while :; do
  curl -s "$BASE/metrics" | awk -v t="$(date +%s.%N)" -v keys="$KEYS" '
    BEGIN { n = split(keys, k, " ") }
    /^#/ { next }                                   # skip the HELP / TYPE comment lines
    { split($1, a, "{"); v[a[1]] = $NF }            # family name without its {labels} -> the value on that line
    END { printf "%s", t; for (i = 1; i <= n; i++) printf ",%s", (k[i] in v ? v[k[i]] : ""); printf "\n" }
  ' >> "$CSV"
  sleep 1
done &
[ -n "${OUT:-}" ] && echo $! > "$OUT/.memtime.pid" && echo "sampler pid: $(cat "$OUT/.memtime.pid") -> $CSV"
```
-> prints the job line `[1] <PID>` and `sampler pid: <PID> -> <OUT>/memtime.csv`; the pass condition is checked by the next block. Two
`bash: OUT: paste §7.2 step 0 first` lines and no `sampler pid:` line mean OUT is unset: nothing was written and no loop is running. A row with a
timestamp and ten empty fields means `curl` got nothing (server down, or booted without `--enable-metrics`).

```bash
# container: sglang_hicache, bash, same shell
# Look at the sampler after ~5 s (read-only): is it alive, how many rows, what does a row look like
OUT=${OUT:?paste §7.2 step 0 first: it defines OUT}
PID=$(cat "$OUT/.memtime.pid")
echo "sampler pid: $PID alive: $(kill -0 "$PID" 2>/dev/null && echo yes || echo no)"
echo "rows: $(($(wc -l < "$OUT/memtime.csv") - 1))"
echo "header: $(head -1 "$OUT/memtime.csv")"
echo "last:   $(tail -1 "$OUT/memtime.csv")"
```
-> prints `alive: yes`, `rows:` >= 4 after 5 s, `header:` the 11 names above, `last:` a Unix timestamp followed by ten values (two of them
empty on arms (a)/(e)); `rows:` not growing between two runs of this block means the loop died (it is a child of this shell: closing the shell kills it).

```bash
# container: sglang_hicache, bash, same shell
# Stop the sampler (= §7.2 step 8): do it BEFORE the post-run drain, whose /flush_cache empties the pools and would be recorded (§6.4)
OUT=${OUT:?paste §7.2 step 0 first: it defines OUT}
kill "$(cat "$OUT/.memtime.pid")" && echo "sampler stopped; rows: $(($(wc -l < "$OUT/memtime.csv") - 1))"
```
-> prints `sampler stopped; rows: <N>` with N about equal to the cell's wall seconds (derived: `sleep 1` plus one `curl` round trip per row).

**Expected result:** `$OUT/memtime.csv` has the 11-column header and one row per ~1 s over the cell (row count ~ client wall seconds, derived).
In every row `kv_used_tokens + kv_evictable_tokens + kv_available_tokens == max_total_num_tokens` (derived: the three come from one
`_get_token_info` pass, `used = total - (available + evictable)`, `pool_stats_observer.py:221-229`), and `max_total_num_tokens` equals the
arm's `--max-total-tokens` pin: 262144 at P0, 131072 at PH and PL, 655360 at PW (§5.3). On HiCache arms `hicache_host_total_tokens` is
488320 / 366272 / 137344 / 732480 at `--hicache-size` 64 / 48 / 18 / 96 (derived from the §5.1 host-pool rule; 762,944 at 100 GB is the `measured-A100` anchor, `R32/exp1_32b/startup_facts.txt:5`);
on arms (a) and (e) both `hicache_host_*` columns are empty. With the server idle (before the client starts, after the §7.2 step 4 flush)
`num_running_reqs` and `num_queue_reqs` are 0 and every value repeats unchanged row after row: the gauges are frozen while idle, not broken.
While the client runs `num_running_reqs` is <= the row's c plus the in-flight controls (with gaps, in-flight conversation requests <<
`--max-concurrency`, §10; the §6.5 side-channel controls run outside the client semaphore) and `num_queue_reqs` is the queue depth that §9.1 bins by.
**If it differs:** ten empty fields in every row: the server was booted without `--enable-metrics` or is not up (`curl -s` hides the error);
fix the launch (§5.2) and restart the sampler. `hicache_host_*` empty on arms (b), (c), (d), (f), (g): HiCache is not attached, check
`hicache_attached=True` in `startup_facts.txt` (§5.2). Values that never change while the client is visibly running: the scheduler is not
producing prefill batches or decode steps (gauges are pushed only then); take a py-spy census (§8) before trusting anything from that cell.
`max_total_num_tokens` not equal to the pin: the `larger than the profiled value` case of §5.1, the pin did not take; the cell is not at the
intended pressure level and must be rebooted.
**Expected lessons:** the gauges answer "how full is each pool and how deep is the queue at this instant" and nothing else: `token_usage` excludes
radix-cached KV, a warm LRU pool reads ~full on every arm, and the gauge integral therefore measures `--max-total-tokens`, not policy (§6.4);
idle-session memory-time comes from the per-turn tier split (§6.2). `num_queue_reqs` at send time is the confounder of every per-turn TTFT
(turn index == rep index == queue depth in a replay, §10): a cell whose `memtime.csv` is missing or was stopped early cannot be binned, and its
TTFT rows in §9.1 are not interpretable. `prefetch_bandwidth` / `backup_bandwidth` stay empty on nixl, so L3 bandwidth is derived from
`prefetched_tokens_total x 131,072 / wall` and cross-checked against `iostat -x -d 1 vdc` (§8), never read from a histogram.

### 6.2 Per-request tier split: zero server changes needed

**Status:** source facts verified against `6ec32e6b7` on 2026-09-18, re-checked against `d608a20d4` on 2026-09-21; the classification below is not implemented yet (it
belongs in `agent_cache/scripts/analyze.py`, §11 — note `agent_cache/scripts/timeline.py` now does part of the job, §7.0), and the "compare against the previous round" variant waits on the §4.4 dry run.
**Goal:** define, from what the stock server already returns per request, the restore-hit class of every returning turn (device-only /
host-restored / storage-restored / recomputed) that §6.4 computes and §9.1 reports, so that no server patch is needed for the split.
**Runs on:** none (read-only source facts; the client receives the split with `--cache-report --output-details`, §4.2, and the
classification runs offline on `client.jsonl`, §6.4).
**Touches:** read-only.
**Takes:** none; no GPU.

Commands: none (this step defines the classification; §4.2 is where the client captures the input, §6.4 where it is consumed).

`cached_tokens` counts **device + host(load-back) + storage** hits (accumulated from `len(req.prefix_indices)` after `init_load_back`
spliced host indices in: `schedule_batch.py:2656-2660`; `schedule_policy.py:1305-1316`). The per-tier split is computed once per request on
its first chunk (`schedule_batch.py:2665-2679`, `split_cached_prefix_by_tier` at `:215-249`) and returned as
`meta_info.cached_tokens_details {device, host[, storage, storage_backend]}` (`output_streamer.py:85-115`; `tokenizer_manager.py:2331-2336`),
or on chat as `sglext.cached_tokens_details` when the body sets `return_cached_tokens_details: true` (`protocol.py:883`;
`serving_chat.py:1892-1900`). A declined/dropped load-back counts as recompute (`materialized_host_hit_len`, `schedule_batch.py:1357-1361`),
which is the right restore-hit semantics.

Restore-hit classes per returning turn (prompt_tokens P, details d):
`device-only` if `d.host==0 and d.storage==0 and cached >= P - 64`; `host-restored` if `d.host>0`; `storage-restored` if `d.storage>0`;
else `recomputed`; `recompute_tokens = P - cached`. If the §5.2 history note holds (regenerated reply never cached; derived, confirm in §4.4),
`cached >= P - 64` can never be true on a returning turn: compare against floor64 of the PREVIOUS round's prompt tokens instead. The
"MatchResult split" the starter kit asks for is exactly this triple; `MatchResult` itself has no storage field (`base_prefix_cache.py:186-230`),
storage arrives via `pop_prefetch_loaded_span` (`scheduler.py:3817-3832`).

**Expected result:** every per-turn entry of `client.jsonl` (`cached_tokens_details`, present with `--output-details`, §4.2) has `device` and
`host` keys and, on a storage-attached arm, `storage` and `storage_backend` (`output_streamer.py:85-115`); `device + host + storage` equals the
request's `cached_tokens` (derived: both are `len(req.prefix_indices)` on the first chunk, `device = prefix_len - host - storage`,
`schedule_batch.py:2656-2679,247-248`). On arm (a) `host == 0` on every turn and there is no `storage` key. Every turn falls into exactly one of
the four classes and `recompute_tokens >= 0`. The class fractions per arm are not yet measured: record them per cell in `summary.csv` (§9).
**If it differs:** `host > 0` and `storage == 0` on every turn of a three-tier PL cell: L3 was never read (the working set did not exceed L1+L2,
or the prefetch hit the `timeout` cut-off: read `storage_prefetch_unfulfilled_tokens_total{reason}` from `metrics_delta.json`, §5.1). No
`storage` key on arms (c), (d), (f): the L3 backend did not attach, `Backend POSIX was instantiated` and `hicache_attached=True` are missing
from `startup_facts.txt` (§5.2). `device-only` never observed on a returning turn although `host == 0 and storage == 0`: the §5.2 history note
holds; switch the test to floor64 of the previous round's prompt tokens as written above.
**Expected lessons:** the split is per request, computed once after the load-back splice and the storage-span pop, so it is the restore-hit
truth the starter kit asks for and no `MatchResult` logging is needed; counting a declined or dropped load-back as recompute is the semantics the
report wants (a restore that did not materialize saved nothing). The class fractions are the first-order result of rows 4-5: if a PL cell shows
`storage-restored` near 0 with `recomputed` high, the prefetch budget (§5.1) or the L3 cleaner (§5.3) is cutting restores off, and row 6
(`timeout` vs `wait_complete`) is the place to look before any TTFT comparison is read as a tiering result.

### 6.3 Eval-only server log line (only if a client cannot read `sglext`, e.g. AIPerf)

**Status:** not applied; needed only for a client that cannot read `sglext.cached_tokens_details` (AIPerf, row 7); the insertion site and the
missing import were re-checked against `6ec32e6b7` on 2026-09-18 (site line 2679, `logger` at 161, no `import time`); `schedule_batch.py` is untouched by the three commits up to `d608a20d4`, so they still hold.
**Goal:** give a client that cannot read the per-request `sglext` chunk the same per-tier split (§6.2) from the server log, one line per
request, so that row 7 can still report restore-hit classes.
**Runs on:** host (the repo is bind-mounted into both containers: an edit under `/home/wanhr/sglang/python/` is live at
`/sgl-workspace/sglang/python/` and applies to every container at once); the server must be rebooted after the edit (the scheduler imports
`schedule_batch.py` at boot).
**Touches:** `python/sglang/srt/managers/schedule_batch.py` (two hunks: `import time`, and the `logger.info` after line 2679); the archive
`agent_cache/patches/0002-eval-tier-log.patch` (`agent_cache/patches/` does not exist yet, 2026-09-18); reverted by the §11 revert block
before every commit (§4.3 step 3; HANDOFF §7: never leave an eval patch under `python/`).
**Takes:** ~5 min to edit and archive, GPU idle; plus one server boot to load it (278-285 s warm, ~633 s JIT-cold, measured, §2.6).

Site: `python/sglang/srt/managers/schedule_batch.py`, immediately after line 2679 (`req._cache_breakdown_computed = True`): class-agnostic,
fires exactly once per request, after load-back splice and storage-span pop. `schedule_batch.py` has `logger` (`:161`) but **no `import time`**
(`:56` is `logging`): add one.

```bash
# host (read-only): confirm the insertion site and the missing import before editing
SB=/home/wanhr/sglang/python/sglang/srt/managers/schedule_batch.py
echo "site line: $(grep -n 'req._cache_breakdown_computed = True' $SB | cut -d: -f1)"
echo "import time lines: $(grep -c '^import time' $SB)"
echo "logger line: $(grep -n '^logger = logging.getLogger' $SB | cut -d: -f1)"
```
-> prints `site line: 2679`, `import time lines: 0`, `logger line: 161` (checked 2026-09-18 at `6ec32e6b7`, unchanged at `d608a20d4`); any other site line means the
file moved under you: put the hunk after the line that sets `req._cache_breakdown_computed = True`, not after 2679.

```diff
# host: editor hunks for python/sglang/srt/managers/schedule_batch.py (NOT a shell block, NOT a git-apply patch: the patch is generated from the edited file below)
@@ next to line 56 (`import logging`)
+import time
@@ after line 2679 (`req._cache_breakdown_computed = True`), same 20-space indent
+                    logger.info("EVAL_TIER rid=%s t=%.6f prompt_len=%d prefix_len=%d device=%d host=%d storage=%d "
+                                "host_hit_len=%d host_loaded_len=%d storage_hit_len=%d",
+                                req.rid, time.time(), len(req.origin_input_ids), len(req.prefix_indices),
+                                req.cached_tokens_device, req.cached_tokens_host, req.cached_tokens_storage,
+                                req.host_hit_length, req.host_loaded_length, req.storage_hit_length)
```

```bash
# host: archive the edit as patch 0002 (the patch carries exact context, so `git apply` in §4.3 works for it too)
REPO=/home/wanhr/sglang
PATCH=$REPO/agent_cache/patches/0002-eval-tier-log.patch
mkdir -p "$REPO/agent_cache/patches"
git -C "$REPO" diff -- python/sglang/srt/managers/schedule_batch.py > "$PATCH"
echo "patch lines: $(wc -l < "$PATCH")"
echo "hunks: $(grep -c '^@@' "$PATCH")"
```
-> prints `patch lines:` > 0 and `hunks: 2`; `patch lines: 0` means the edit was not made (or was already reverted) and nothing was archived.
Apply per session exactly as §4.3 block 2 does for `0001-agentic-trace-pre-gap.patch`, with this file name; before every commit the §11 revert
block reverse-applies both archived patches in one pass (one `reverted:` line each). Also copy the applied diff into `results/<stamp>/patches/` (§4.3 block 4).

Parse with `grep EVAL_TIER server.log`; the labelled form:

```bash
# host (read-only): count and preview the per-request tier lines in one arm's server log
# <ARM> = the §5.2 arm name of the log, e.g. three_tier (logs are $RUNDIR/server_<arm>.log)
LOG=/home/wanhr/sglang/agent_cache/results/$(cat /home/wanhr/sglang/agent_cache/.current_results)/server_<ARM>.log
echo "EVAL_TIER lines: $(grep -c EVAL_TIER "$LOG")"
grep EVAL_TIER "$LOG" | head -3
```
-> prints `EVAL_TIER lines: <N>` with N = requests admitted since boot (warm-up probe + client turns + controls; one line per request, chunked
prefills included once), then three lines with the eleven `key=value` fields.

**Expected result:** one `EVAL_TIER` line per request in `server.log`; on every line `device + host + storage == prefix_len` (derived:
`device = prefix_len - host - storage`, `schedule_batch.py:247-248`) and `prefix_len <= prompt_len`; the salted warm-up probe of §7.2 step 3 (the first request after boot, L3
wiped in step 1) shows `prefix_len=0 device=0 host=0 storage=0` (derived: a never-sent prompt against an empty tree and an empty store). Exactly one line per request, on its first
prefill chunk, and none after a retraction (derived: the site sits inside `if not req.retracted_stain:` at `schedule_batch.py:2658` and behind
`_cache_breakdown_computed`, so a retracted request's re-prefill is invisible here: count retractions from the
`sglang:num_retracted_requests_total` counter (`observability/metrics_collector.py:472-476`; scrape it with the §6.1 counters) instead; an earlier draft logged `retracted=%d` from this site, which can only ever print 0). Not yet observed on this box: record the first three lines here after the first patched boot.
**If it differs:** `EVAL_TIER lines: 0` after requests were served: the server was booted before the edit (module imported at boot: reboot), or the
patch was reverted by the §11 revert block (§4.3 step 3). `NameError: time` in `server.log`: the `import time` hunk is missing.
**Expected lessons:** the split already exists on every request; this line only exposes it to clients that cannot read `sglext`, and its fields
line up with `cached_tokens_details` (§6.2) so both paths can be joined on `rid` (§6.4). Do not add a `SGLANG_*` env var to gate it (that would
pull in `.claude/skills/env-var-conventions`); the gate is the patch itself, applied per session and reverted before every commit.

### 6.4 Offline computation

**Status:** not started; `agent_cache/scripts/analyze.py` (§11) does not exist on 2026-09-18, and no cell has produced inputs for it. The four
definitions below are its specification.
**Goal:** define how the per-cell files turn into the §9.1 quantities (idle-session memory-time per tier, restore-hit rate, queue depth per turn,
server-side cross-check) so that the analysis is fixed before the first cell and cannot drift between arms.
**Runs on:** host, CPU only (JSON/CSV inputs under `/home/wanhr/sglang/agent_cache/results/<stamp>/<cell>/`, chown'd in §7.2 step 12); the
container works too (`/sgl-workspace/sglang/agent_cache/results/<stamp>/<cell>/`).
**Touches:** read-only over the cell dir; writes only the analysis output (`summary.csv` row, §9).
**Takes:** seconds per cell, estimate; no GPU.

Commands: none (`analyze.py` is not written; the rules below are what it implements).

- **Memory-time (token-seconds):** the gauge integral of `kv_used + kv_evictable` (+ `hicache_host_used`) is *pool occupancy*: a warm LRU pool
  stays full, so it is ~pool x wall for every arm and "P0 minus arm" measures `--max-total-tokens`, not policy. Report **idle-session
  memory-time** per returning turn from the tier split (§6.2): device-only -> `prompt_tokens x pre_gap` on device; host-restored -> same on host;
  recomputed -> 0 resident (freed), cost `recompute_tokens`; sum per arm, normalize by device pool x client wall, x b (131,072 B/token) for
  byte-seconds. The gauge integral is only an upper-bound check, windowed to [client start, last response], sampler stopped **before** the
  post-run flush (`wait_until_flushable` POSTs `/flush_cache`, `hcommon.py:231`, emptying the pools, `scheduler.py:4909-4911`). "Freed" only
  between equal-pool arms.
- **Restore-hit rate:** from `--output-details` lists (conversation-major/round-minor: the flatten at `serving.py:1567-1569`; explicit
  `conv_idx`/`round_idx` after hunk 7), or from `EVAL_TIER` lines joined on rid.
- **Queue depth per turn:** nearest `memtime.csv` sample to each `start_times` entry -> `num_queue_reqs`/`num_running_reqs`.
- **Server cross-check:** delta of `prefill_effective_tokens_total{mode}` must match the sum of per-turn details (+ controls); balances only with
  the warmup handled as in §7.2 step 3.

**Expected result:** per cell, four outputs: idle-session token-seconds (and byte-seconds via b = 131,072) per tier, normalized by device pool x
client wall, with the gauge integral as an upper bound that the per-turn sum never exceeds; the fraction of returning turns in each of the four
§6.2 classes; one queue depth per turn (a `memtime.csv` sample within 1 s of each `start_times` entry, derived from the 1 s cadence); and a
server-side `prefill_effective_tokens_total{mode}` delta that matches the per-turn details plus controls. No cell has run, so no tolerance for the
cross-check is established: record the first observed discrepancy here.
**If it differs:** the gauge integral is below the per-turn sum: the sampler was stopped early or started late (the window must cover [client
start, last response]). The cross-check does not balance: a client warm-up ran (`--warmup-requests` must be 0, §7.2 step 3), or the step-6
snapshot was taken before the flush and includes the warm-up prefill.
**Expected lessons:** pool occupancy is not memory-time saved: every LRU arm keeps its pool full, so the only quantity that distinguishes a parking
policy is what each idle session holds in which tier, which is why the report is built from the per-turn split and the gauges serve as a bound.
A queue depth per turn is what makes a TTFT comparable at all (§10): the sample-to-turn join is the step that turns a replay's turn-index confound
into a binnable axis.

### 6.5 The paired-control rule (HANDOFF §3, carried over verbatim)

**Status:** rule inherited from `hicache_eval/HANDOFF.md` §3 (the paired control carried the whole Exp 2 conclusion there); in this study it is
implemented by patch hunk 6 (§4.1, not applied yet) and checked in the §4.4 dry run.
**Goal:** make every restore TTFT comparable to a recompute of the same length under the same load at the same instant, so that a tier's benefit
is a paired difference with a CI (§9.3) and not a comparison against an idle curve.
**Runs on:** none (the control is fired by the patched client inside the container, §4.1 hunk 6; the pairing is computed offline, §6.4).
**Touches:** read-only as a rule; each control adds GPU work and L2/L3 write-through to the cell (sizes below).
**Takes:** per control, one full recompute of the turn's `prompt_tokens`: ~18 s of GPU at 25K (derived, §5.4 fit); per cell `T_rec(context)/K`
of GPU per turn (§7.3).

Commands: none (the rule is enforced by the client patch and the analysis).

Every "restore" TTFT is compared against a **never-before-sent prompt of the same length sent at the same instant under the same concurrency**
(`hicache_eval/scripts/archive/exp2.py:148-165`), never against an idle recompute curve. Control seeds are salted per process (`exp2.py:28-30`) and
control rows must show `cached_storage` NaN/0, else the control silently became an L3 hit. In a gap replay the control **cannot** sit in the
conversation slot (every round extends `prev_messages` and appends the reply, `serving.py:1319,1329-1331`, poisoning every later prefix;
`exp2.py:148-165` is single-shot, outside any conversation): it is the side-channel request of patch hunk 6 (`--agentic-control-every K`,
§4.1), fired at the returning turn's instant, salted per process, outside the semaphore.

Analysis: paired (restore - control) differences per turn-index bin, 95 % CI over >= 30 pairs; assert `storage == 0, host == 0, device <= 64`
per control row. With the 32B a control is a ~25K recompute (18 s of GPU, 3.05 GiB of write-through, ~25K of L1 held outside the semaphore: at
PH/PL the 4.37-request admission cap is ~3.4 while one runs, and its KV then sits in the LRU as never-reused pages), so K is set by the §5.3 /
§7.3 budgets (K = 40-84 -> ~30-40 pairs per CELL, spread over turn indices by the conv_idx offset): reach >= 30 pairs per bin by pooling turn
indices into coarse bins or by repeating the cell (new `RC_SALT`, cold L3), never by lowering K or adding conversations on a three-tier arm.

**Expected result:** every `control_details` row in `client.jsonl` shows `storage == 0, host == 0, device <= 64`; a cell at K = 40-84 yields
~30-40 pairs (derived from turns / K), and the report shows paired (restore - control) p50 per pooled turn-index bin with a 95 % CI over >= 30
pairs; the first pass of the rule is the §4.4 dry run (4 control rows at K = 1). Not yet measured on this box.
**If it differs:** a control row with `storage > 0` or `host > 0`: the control prompt was sent before (seed not salted per process, or a repeated
cell without a new `RC_SALT`), and that pair is invalid, as is every later pair with the same seed. `device > 64` on a control: the salted prompt
shares a page-aligned prefix with something in the tree (system prompt reuse); check `salted_random_chat` in hunk 6. Fewer than 30 pairs per bin:
pool bins or repeat the cell with a new salt and a cold L3; never lower K on a three-tier arm (L3 disk budget, §5.3).
**Expected lessons:** comparing a loaded restore against an idle recompute invents crossovers that do not exist (HANDOFF §3): the control is the
line the restore has to beat under the same queue. With the 32B each control costs a 25K recompute plus 3.05 GiB of write-through, so K trades
statistical power against cell time and L3 disk (§5.3, §7.3): a CI that does not exclude 0 over >= 30 pairs is a dead heat, not a small win (§9.3).

### 6.6 HiCache event log: millisecond timestamps and tier-transfer events (eval patch 0002)

**Status:** **committed, not applied.** The patch was written on the A100 box 2026-09-18 and then committed at `5b881454b`, so on this checkout (`d608a20d4`) it is **already in the working tree with `git status` clean** — nothing to `git apply`, and nothing to revert before a commit. It was verified there on a booted `hbm_host` P0 server (64 events over 4 conversations x 4 turns) and then produced the full `C18/` event logs, including the L3 events (`s2h_*`, `h2s_*`) and evictions that the first test could not reach. On **this** box it has never emitted a line (no server has run).
**Goal:** a per-event timeline of the tiers in the server log (when and how much KV is offloaded device->host, backed up host->L3, prefetched L3->host, loaded host->device, evicted), with millisecond timestamps, so a cell's client turns can be joined to what the cache did between them; the stock log has none of this (the controller logs nothing at INFO/DEBUG, the cache only prefetch completion).
**Runs on:** nothing to run — the code is in the checkout; the server reads it at boot (`container: sglang_hicache`, same bind mount). The verify block below is host-side and read-only.
**Touches:** **read-only.** The 31 added lines (every one tagged `# EVAL-PATCH`) live in `managers/cache_controller.py`, `mem_cache/unified_radix_cache.py`, `mem_cache/hybrid_cache/hybrid_cache_controller.py` and `utils/common.py` and are part of `HEAD`. `agent_cache/patches/0002-hicache-event-log.patch` is kept as the record of what was added and as the way to *remove* it (`git apply -R`) if a run ever needs the stock logging.
**Takes:** the verify block < 1 s; GPU idle

Millisecond timestamps need `SGLANG_LOG_MS=1` in the server environment, which turns `[%(asctime)s]` into `[... HH:MM:SS.mmm]` (env `environ.py:339`); `agent_cache/scripts/start_server.sh` sets it on every arm. One of the committed `# EVAL-PATCH` lines extends the same millisecond format to uvicorn's access log (`utils/common.py`, `set_uvicorn_logging_configs`). The events are `logger.info("HICACHE_EVT <event> k=v ...")` lines, one per transfer or eviction, greppable with `HICACHE_EVT`:

| event | emitted where | meaning and fields |
|---|---|---|
| `d2h_submit nodes tokens bytes` | `cache_controller.py` `start_writing` | device->host write-through copy submitted (one line per merged write batch; `nodes` = radix nodes it covers) |
| `d2h_done nodes tokens bytes ms` | `unified_radix_cache.py` `writing_check` | that copy acknowledged; `ms` from the CUDA events (`timing_enabled`, needs `--enable-metrics`; -1 otherwise) |
| `h2d_submit nodes tokens bytes` / `h2d_done ... ms` | `start_loading` / `loading_check` | host->device load-back (a returning turn whose prefix lives in L2) |
| `load_back_init rid tokens host_hit node` | `unified_radix_cache.py` `init_load_back` | the scheduler decided to load a request's host-resident prefix; `rid` is the client's `<tag>-c<conv>-t<turn>` |
| `prefetch_start rid tokens` | `prefetch_from_storage` | L3 prefetch requested for a request (storage arms only) |
| `s2h_query rid tokens_req storage_hit` | `prefetch_thread_func` | the storage lookup result: how many of the requested tokens exist in L3 |
| `s2h_io rid pages tokens ms` | `prefetch_io_aux_func` | the L3->host read itself, wall time |
| `h2s_submit op node tokens` / `h2s_io op pages tokens ms` / `h2s_done op tokens` | `write_backup_storage` / `backup_thread_func` (hybrid) / `_drain_backup` | host->L3 backup: queued, written (wall time), acknowledged (`tokens` at done = completed tokens) |
| `evict_device tokens requested ms` | `_evict` | device eviction for an allocation: tokens freed vs asked |
| `evict_host tokens requested` | `evict_host` | host-pool eviction |

The stock `HiCache prefetch success|dropped req= completed= matched= loaded=` INFO line (`unified_radix_cache.py:2028`) completes the prefetch story and stays as is.

```bash
# host: verify that the committed event-log code is in the tree (read-only). Nothing to apply on this checkout.
echo "EVAL-PATCH lines in tree: $(grep -rc 'EVAL-PATCH' /home/wanhr/sglang/python/sglang/srt/managers/cache_controller.py /home/wanhr/sglang/python/sglang/srt/mem_cache/unified_radix_cache.py /home/wanhr/sglang/python/sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py /home/wanhr/sglang/python/sglang/srt/utils/common.py | paste -sd' ')"
echo "patch is in HEAD:         $(git -C /home/wanhr/sglang apply --check -R /home/wanhr/sglang/agent_cache/patches/0002-hicache-event-log.patch >/dev/null 2>&1 && echo yes || echo NO)"
echo "python/ working diff:     $(git -C /home/wanhr/sglang status --short -- python/ | wc -l) files   (0 = clean, and the patch is STILL active)"
echo "commit that added it:     $(git -C /home/wanhr/sglang log --oneline -1 -S'HICACHE_EVT' -- python/)"
```
-> prints `EVAL-PATCH lines in tree: ...cache_controller.py:5 ...unified_radix_cache.py:8 ...hybrid_cache_controller.py:2 ...common.py:1` (16 tagged lines across 4 files),
`patch is in HEAD: yes`, `python/ working diff: 0 files`, and `commit that added it: 5b881454b agent evaluation` (verified 2026-09-21).
**To remove it for one run:** `git -C /home/wanhr/sglang apply -R agent_cache/patches/0002-hicache-event-log.patch`, reboot the server, and re-apply
(`git apply`) afterwards — the working tree then shows a diff, which is the normal §11 situation and not an error.

```bash
# host: after a cell, count events by type and print the timeline of one conversation's turn (rid from client.jsonl), read-only
R=/home/wanhr/sglang/agent_cache/results/$(cat /home/wanhr/sglang/agent_cache/.current_results); L=$R/$(cat $R/.current_run)/server.log   # the latest boot (scripts layout, §11)
echo "events by type:"; grep -o "HICACHE_EVT [a-z2_]*" "$L" | sort | uniq -c
echo "tier transfers with durations (first 5):"; grep -E "HICACHE_EVT (d2h|h2d|s2h|h2s)_(done|io)" "$L" | head -5
echo "per-request events for rid evt_host-c1-t2:"; grep "rid=evt_host-c1-t2" "$L"
```
-> the type counts (2026-09-18, `hbm_host` P0, 4 conv x 4 turns: `32 d2h_submit`, `32 d2h_done`, nothing else: at P0 nothing is evicted, so no `h2d_*`, and no L3 on arm (b)); a `d2h_done` line reads like `HICACHE_EVT d2h_done nodes=1 tokens=7808 bytes=1023410176 ms=81.2` (the shared system prompt: 1.02 GB in 81 ms = 12.6 GB/s device->host, measured); per-request lines appear only for `load_back_init` / `prefetch_*` / `s2h_*`, so at P0 the last block prints nothing.

**Expected result:** every server log line carries milliseconds; on `hbm_host` at P0 the count of `d2h_done` equals `d2h_submit`, their `tokens` sum equals the prefilled-token volume that was inserted (write-through backs up every inserted node), and each `ms` is consistent with ~10-13 GB/s. On a pressure cell (§5.3 PH/PL): `h2d_*` lines appear before returning turns whose `cached_details.host > 0` in `client.jsonl`, `load_back_init rid=` names those turns, and on three-tier arms `h2s_*` follows every `d2h_done` and `s2h_*` precedes L3 hits.
**If it differs:** no `HICACHE_EVT` at all on a HiCache arm: the server booted from a tree where the code is reverted, or it is an older container image with its own copy of `sglang` shadowing the bind mount (§2.4 block 2); the verify block above must print `patch is in HEAD: yes` and 8 tagged lines in `unified_radix_cache.py` (5 in `cache_controller.py`, 2 in the hybrid controller, 1 in `utils/common.py`). `ms=-1.0` on every `*_done`: `--enable-metrics` missing (the timing events are created only then). `d2h_submit` without a matching `d2h_done` at the end of a run: the ack is polled by the scheduler in `check_hicache_events`, so an idle server drains it on the next scheduling iteration; wait one request or check `/metrics`.
**Expected lessons:** write-through granularity is the radix node, not the request: the first conversation's 7,808-token system prompt is one `d2h` of 1 GB, and later conversations' branches produce 256-3,712-token fragments as nodes split; so backup volume per turn is the new tokens only (once), which is what `hicache_eval` measured as "bar" in aggregate and this log now resolves per node with a wall-clock stamp. Joining `t_send`/`ttft` from `client.jsonl` with `load_back_init`/`h2d_done` stamps gives the restore latency each returning turn actually waited for, which §6.4 could only infer from tier splits before. **This log is what made the A100 null result diagnosable** (§7.0): all four named causes — host churn, prefetch admission, timeout cut-off, write amplification — were read off `h2s_io` / `s2h_query` / `evict_host` counts and durations, not off `/metrics`. Because it is now committed rather than applied, the one way to lose it is to boot a server whose `sglang` is not the bind-mounted checkout.

---

## 7. Experiment matrix (execution order) and per-run checklist

This section turns §5 (server configurations) and §6 (instrumentation) into the ordered list of cells that actually get run, one checklist that every cell follows, and the time each row is expected to cost. At its end the reader knows which arm, pressure level, trace and concurrency each row uses, what each row is supposed to show, how to run one cell without missing a step, and how many days the campaign needs.

### 7.0 What the A100 campaign actually found (read before sizing anything)

**Status:** **measured-A100, complete, and the only end-to-end result this study has.** Three arms, one fresh boot each, 2026-09-18 19:50-22:30 UTC, driven by `agent_cache/scripts/run_compare.sh`; artefacts in `C18/` = `agent_cache/results/compare_20260918_final/` (README, `compare.csv`, `compare.png`, `turns_<arm>.csv`, `events_<arm>.csv`, `timeline_<arm>.png`) and the three boot dirs named in `C18/manifest.txt`. Nothing comparable has been run on this box.
**Goal:** state the prior result, its four diagnosed causes and the three conditions it named, so that a cell on this box is sized to test them rather than to repeat them.
**Runs on:** none (reading). Regenerate the figures with `python3 agent_cache/scripts/timeline.py --manifest agent_cache/results/compare_20260918_final/manifest.txt`.
**Touches:** read-only
**Takes:** ~10 min to read `C18/README.md` and one `timeline_*.png`; GPU idle

**The cell.** `hbm_lru` / `hbm_host` / `three_tier`, identical pools (**L1 65,536 tokens, L2 12 GB = 91,520 tokens**, i.e. *below* even the PL level of §5.3), identical client: the LMCache swebench trace, 32 conversations all live (`NCONV=32 C=32`), 12 turns each, gaps x10, `GAP_CAP=600` on two of the three arms. Qwen3-32B-FP8, fp8 KV, A100 80 GB.

**The result: the three arms are indistinguishable.**

| | hbm_lru | hbm_host | three_tier |
|---|---|---|---|
| returning-turn TTFT p50 / p90 (s) | 141 / 247 | 142 / 248 | 143 / 246 |
| turns done at 95 % (s) | 2163.4 | 2112.5 | 2103.3 |
| returning turns: device / host / storage / recompute | 196 / 0 / 0 / 152 | 156 / 43 / 0 / 150 | 146 / 43 / **11** / 149 |
| uncached tokens per returning turn (mean) | 7,427 | 7,052 | 6,961 |
| tokens d2h / h2s / h2d / s2h | – | 2.59M / – / 106K / – | 2.55M / 2.55M / 150K / 191K |
| peak scheduler queue | 28 | 28 | 28 |

43-44 % of returning turns recomputed their whole private context **in every arm**. The tiers restored only during the first ~100 s, plus 11 storage hits in the whole run. TTFT was set by **admission queueing** (queue ~25 for 35 minutes: 32 live conversations against a 64K-token pool that admits 3-5 requests of 12-20K tokens at a time), not by tiering.

**Four diagnosed causes** (read off the §6.6 event log, not off `/metrics`):
1. **The host pool was smaller than the churn.** L1+L2 = 157K tokens against a live set of ~500K; the host LRU evicted 2.5-2.7M tokens, so a conversation's tail left L2 before the conversation returned. Host hits happened only while total cached KV was still below the pool.
2. **Prefetch admission is capped at half the host pool** (`managers/cache_controller.py`, `prefetch_tokens_occupied` against a budget of 45,760 tokens here), and each prefetch reserves its *requested* length, so ~three 15K prefetches fit at once: **145 of 354 prefetch requests never reached the L3 lookup.**
3. **The `timeout` policy gave up before the SSD could deliver.** Budget ~1 s + 0.25 s/Ki-token (~4.7 s for 15K); the L3 read ran at <= 0.5 GiB/s because the disk was busy writing. **167 of 209 lookups ended with zero usable tokens.**
4. **Write amplification.** Every recomputed turn re-inserts its KV and write-through backs it up again: **2.55M tokens (312 GiB) written to L3 for ~480K tokens of distinct content**; `h2s_io` occupied **835 s of the 2,780 s run** at the 0.37 GiB/s write ceiling, which is what starved the reads in (3).

**The three conditions the README named.** Tiers can only show a benefit when (a) L1 admits the live in-flight set (`f x c x context <= L1 / 1.2`, §5.3 constraint i), (b) L2 is large enough that a conversation's tail survives its gap, and (c) prefetch capacity and the timeout budget cover a full-context restore at the disk's *contended* read rate.

**What this box changes, cause by cause** (all predictions, none measured here):
- **(1)** is a pool-sizing choice, not a hardware limit: that cell ran at L1 65,536 / L2 12 GB, which is below PL. Fixing it costs nothing on either box — run PH/PL as §5.3 specifies, or PW.
- **(2) is untouched by the hardware.** The half-the-host-pool prefetch cap is a code constant; a bigger host pool raises the budget proportionally, which is the only lever, and §7.1 row 6 should record `storage_prefetch_unfulfilled_tokens_total{reason}` specifically to see whether admission or timeout dominates.
- **(3)** improves the most: the sustained read ceiling is **1.88 GiB/s here vs ~0.5-0.68 GiB/s there** (measured, §2.2), so the same budget buys ~2.8x more tokens and four concurrent 25K restores fit where one did (§5.1) — *provided* the nixl path reaches the device, which §5.4 block D is the test of.
- **(4)** the amplification itself is unchanged (it is a consequence of write-through plus recompute), but at a **1.88 GiB/s** write ceiling the same 312 GiB would occupy **~166 s instead of 835 s** (derived), so it should stop starving the reads. Note reads and writes share that one ceiling, so the two effects are not independent.

**Expected result:** a reader can state, before sizing any cell here, which of the four causes that cell is exposed to and which it controls for. A cell that reproduces cause (1) — pools far below the live set — will reproduce the tie on this box too, faster.
**If it differs:** if a cell on this box ties at PH/PL *with* median queue depth ~0 and `storage_prefetch_unfulfilled` near zero, that is a genuine null result for tiering on this workload, not a repeat of `C18/`; say so explicitly, because the two look identical in the summary table and completely different in the event log.
**Expected lessons:** the A100 campaign's one campaign-scale mistake was sizing a cell from estimates and only afterwards discovering it had measured admission queueing (§5.4's Expected lessons). The four causes above are the reason §5.4 comes first here and the reason §7.1 row 2 (the tie control at P0) must run before any pressure row: without a tie at P0 there is no baseline against which a tie at PL means anything.
### 7.0b What campaign 7 found on this box (2026-09-22): the tiers work once the pools admit the live set

**Status:** **measured-H200, complete.** Four arms, one fresh boot each, 2026-09-22 02:37-11:27 UTC, driven by `agent_cache/scripts/run_compare.sh`; artefacts in `C7/` = `agent_cache/results/compare_20260922_023714/` (`README.md`, `DECISIONS.md` with the sizing chain and the run notes, `compare.png`, `compare.csv`, `timeline_<arm>.png`, `turns_<arm>.csv`, `events_<arm>.csv`, `iostat_vdc.log`).
**Goal:** state the first result on this box in the same terms as §7.0, so that the next cell is sized against a measured tiering benefit and its measured limit (the SSD's read ceiling), not against the A100 null result.
**Runs on:** none (reading). Regenerate the figures with `python3 agent_cache/scripts/timeline.py --manifest agent_cache/results/compare_20260922_023714/manifest.txt`.
**Touches:** read-only
**Takes:** ~15 min to read `C7/README.md`, `compare.png` and one `timeline_*.png`; GPU idle

**The cell.** `hbm_lru` / `hbm_host` / `three_tier_to` / `three_tier_wc` (the two prefetch policies), identical pools at their natural size: **L1 668,160 tokens** (the profiled pool, bf16 KV at 98,304 B/token), **L2 160 GB = 1,627,648 tokens**, L3 the attached SSD (cleaner 70/60 %, never reached: 331 GB peak). Client: the LMCache swebench trace re-converted at a 262K cut, **128 conversations all live**, up to 40 turns each = 4,676 turns, gaps x70 capped at 1,200 s, stock template, `OFFSET=32`. Model Qwen3-30B-A3B-Instruct-2507 (bf16 weights 57 GB); chosen for its 262K context so that sessions could run 40 turns (final context p50 35.6K, of which 7,808 is the shared system prompt). Private working set 3.55M tokens.

**The result: the SSD tier turns a 3-minute queue into a 3-second restore.**

| | hbm_lru | hbm_host | three_tier_to | three_tier_wc |
|---|---|---|---|---|
| returning-turn TTFT p50 / p90 (s) | 92.9 / 201 | 0.98 / 184 | 0.32 / 6.8 | 0.29 / 4.5 |
| returning-turn TTFT mean (s) | 97.0 | 61.0 | 1.94 | 1.48 |
| turns done at 95 % (s); client wall (s) | 8,658; 11,429 | 6,309; 9,155 | 2,620; 5,327 | 2,539; 5,148 |
| returning turns: device / host / storage / recompute | 985 / 0 / 0 / 3,563 | 1,274 / 1,553 / 0 / 1,721 | 1,669 / 1,921 / **890** / 68 | 1,780 / 1,890 / **837** / 0 |
| uncached tokens per returning turn (mean) | 15,427 | 8,663 | 924 | 558 |
| storage restore TTFT p50 / p90 (s) | – | – | 2.58 / 13.8 | 3.34 / 15.1 |
| tokens d2h / h2s / h2d / s2h | – | 40.5M / – / 22.4M / – | 5.27M / 5.27M / 47.8M / 18.1M | 3.57M / 3.57M / 46.4M / 17.2M |
| peak scheduler queue | 69 | 70 | 73 | 79 |

Every arm serves the first ~12 min identically (device/host hits at 0.2-0.3 s) until the unique live set passes L2's 1.63M tokens; from there `hbm_host` recomputes every evicted return (~28K tokens, 5-7 s of GPU each), the closed loop saturates at 0.2-0.3 turns/s and TTFT sits at ~180 s p50 for 40 min with 55-86 conversations waiting; `hbm_lru` does the same from minute 10 for two hours. The SSD arms restore the same returns at 2.6-3.3 s p50 with the queue at 0-7 and finish in 58 % of the wall time.

**The four §7.0 causes, measured here:**
1. **L1 admits the in-flight set** (668K holds ~15 requests of 35K against 12-42 running); admission queueing is gone from the tiered arms. Cause (1) was a pool-sizing choice, as §7.0 predicted.
2. **Prefetch admission never bound**: budget 0.5 x host pool = 814K tokens in flight, used to ~60 % at 12-30 concurrent restores of ~20K; 960 / 837 prefetches ran, all complete, none rate-limited, none timed out (`HICACHE_EVT`, RUNBOOK 6.6).
3. **The SSD is the wall now, not the policy.** Restores run the device at 1.0 GiB/s mean / 2.0 GiB/s peak (`iostat_vdc.log`, 26-28 % of active samples at >= 95 % util), i.e. the 1.88 GiB/s ceiling of §2.2; the storage p90 of 14-15 s is the disk queue. `wait_complete` removes the `timeout` arm's 68 post-warm-up recomputes (mean 1.48 vs 1.94 s) at the price of a slower individual restore (3.34 vs 2.58 s p50).
4. **No write amplification in the SSD arms**: 3.6-5.3M tokens backed up for 3.6M of content (write-back 81-171 MiB/s); `hbm_host` still re-inserts 40.5M because every recompute is written through again.

**One structural fact worth remembering:** write-through keeps L2 a superset of L1, so the unique capacity of the two tiers is **L2 alone (1.63M), not L1+L2 (2.30M)** — host evictions began at 02:48Z with the live set at ~1.56M. Size the L3-reachability condition against L2, not L1+L2 (§5.3 constraint ii).

**Deviations (all in `C7/DECISIONS.md`):** (a) the 9,000 s per-arm cap did not act — `timeout --foreground` signals only the `bash start_client.sh` wrapper, which defers SIGINT while its pipeline runs; every `STAGE CAPPED` line of that campaign is spurious and `hbm_lru` ran uncapped (190 min). Fixed 2026-09-22: `CLIENT_TIMEOUT` now reaches the client as `--max-seconds` (conversations stop at their next turn boundary, in-flight requests complete, the summary line carries `"capped": true`, the driver prints `STAGE CAPPED` from that) and was verified with a 4-conversation `CLIENT_TIMEOUT=40` self-test. (b) `three_tier_wc` lost two conversations (41 turns) to aiohttp keep-alive resets (`SGLANG_TIMEOUT_KEEP_ALIVE=5` s vs pooled idle connections); `replay_agentic.py` now retries such a request once on a fresh connection and marks the record `retried`. (c) `STAGE FAILED three_tier_wc (client)` in the driver log was the per-turn table crashing on the two error records after the client had finished; the manifest line was added by hand and the driver now keys the manifest on the summary line, not the wrapper's exit code.

**Expected result:** the next cell can assume tiering works at natural pools on this box and should target the limit instead: the SSD read ceiling (a faster device, or fewer bytes per restore — fp8 KV halves them, §5.3) and the prefetch policy trade-off (`wait_complete` for fewer recomputes, `timeout` for shorter individual restores).
**If it differs:** a cell that shows `storage` hits at 2-4 s but a rising `recompute` share under `wait_complete` has hit the admission budget (§7.0 cause 2): read `prefetch_start` vs `s2h_io` counts before touching anything else.
**Expected lessons:** three things went wrong that a script could have caught: the wall cap was never exercised before the campaign (a 40 s self-test would have shown it), a transport reset ended a conversation instead of being retried, and a cosmetic post-processing step could flip a finished run to FAILED. All three are fixed in the scripts; the self-test recipe is `ARMS=hbm_lru NCONV=4 C=4 TURNS=8 GAP=0 CLIENT_TIMEOUT=40 bash run_compare.sh` (~4 min).

### 7.1 Order

**Status:** **one off-matrix four-arm cell has run on this box** (campaign 7, §7.0b, 2026-09-22: tiering works at natural pools, the SSD read ceiling is the limit); no matrix row has run here. Row 1 is now **the full §5.4 constant re-measurement**, not just the two open constants, and it needs only a §5.2 boot of arm (a) plus one of arm (c); row 0 needs the §4.6 client (already in the checkout); rows 4-6 need `s_H`/`s_L`, which cannot be fixed until row 1 lands; row 7 needs the AIPerf venv (§2.5). **Every c, f, K and gap figure below is an A100 prior** (§5.3): re-derive rows 2-8 after row 1. Row order is unchanged from the A100 plan, because the order is a logical dependency, not a hardware one.
**Goal:** fix the execution order so that the two unmeasured constants are measured before any multi-hour cell is sized, the tie control runs before any pressure cell, and every pressure cell has the same-pool HBM-only reference it is compared against.
**Runs on:** the commands of each row live elsewhere: row 0 in §4.4, row 1 in §5.4, rows 2-6 in §7.2 (`container: sglang_hicache`), rows 7-8 in §3.4. This subsection is the plan, not a command sequence.
**Touches:** read-only (each row's cell dirs are created by §7.2 block 0).
**Takes:** none by itself; per-row durations are in each row block and in §7.3 (all GPU-busy).

| # | arm (§5.2) | pressure (§5.3) | trace | c | purpose |
|---|---|---|---|---|---|
| 0 | (a) | P0 | gap-test (§4.4) | 2 | patch + pipeline smoke |
| 1 | (a), then (c) | P0 | §5.4 blocks A-F | 1, 4, 8 | **the whole constant set for this GPU and this disk**: profiled pool, weight kernel, b, the recompute curve and P, the L1/L2/L3 fits, decode_rate(B), the prefix-extension TTFT. On the A100 only the last two were open; here **nine** are (§5.4) |
| 2 | (a),(b),(c) | P0 | LMCache, real gaps (median 0.71 s) | 4, 8 | **tie control and honest short-gap case**: WS = 100K / 200K < L1 = 262K; expect all arms within noise, L2/L3 reads ~0. f ~ 0.85-0.92 (estimate, on the 2.08 s mean gap): in-flight ~ 0.9 c |
| 3 | (e),(g),(d) | PH | LMCache, real gaps | 8 | **overload reference**, run once: f ~ 0.90-0.92 so in-flight ~ 7.3 x 25K = 182K > L1 = 131K; TTFT is admission queueing: bin by queue depth, not a tiering result |
| 4 | (e),(g),(d) | PH | LMCache, gap scale s_H (mean gap >= ~13-16 s, §5.3) | 8 | **host-restore case**: WS = 1.53 L1 = 0.40 (L1+L2); (g) vs (e) is the L2 effect, (d) should equal (g) (L3 idle) |
| 5 | (e),(g),(d) | PL | LMCache, gap scale s_L (mean gap >= ~31-39 s) | 12 | **L3 case**: WS = 1.12 (L1+L2); (g) recomputes what (d) reads from L3. c=16 (1.49x) only if s can reach a 46-58 s mean gap; it runs with 64 conversations (4 x c, the L3 disk cap of §5.3), so its in-flight == N window is shorter: say so in the report |
| 6 | (d) vs (f), (g) | PL | as row 5 | 12 | `timeout` vs `wait_complete` (HANDOFF §2) with the host-only reference. (f) is the policy with measured constants here; (d) with the default budget may measure the cut-off, not the tier (§5.1) |
| 7 | (d) | PL | AgentX via AIPerf, 1800 s | 8 | long-context standardized view (context-capped). At c=8 WS <= 8 x 32,768 = 262K < L1+L2 = 268K: expect L3 reads ~0 unless subagent trees multiply the live contexts (unverified); record storage-hit tokens and do not read this row as an L3 result |
| 8 | (a) | P0 | Mooncake toolagent, `--backend sglang`, slowdown >= ~15 | 8 | cross-session sharing only; pure backlog without the slowdown factor (§3.4) |
| 9 | (d), (g), (e) | **PW** | LMCache at `s_L` | 24-32 | **optional, H200-only**: the same pressure ratios at ~5x the absolute scale (§5.3 PW row), which the A100's 80 GB HBM could not reach. Run only after rows 2-5 and only if row 5 shows a non-zero storage-hit rate; size it from the row-1 constants |

Arm letters: (a) `hbm_lru`, (b) `hbm_host`, (c) `three_tier`, (d) `three_tier_p`, (e) `hbm_lru_p`, (f) `three_tier_wc`, (g) `hbm_host_p` (§5.2). Pressure levels P0 / PH / PL / PW are the `set_level` rows of §5.3 (L1 = 262,144 / 131,072 / 131,072 / 655,360 tokens; `--hicache-size` 64 / 48 / 18 / 96 GB).

**Row 0 — patch + pipeline smoke.** Arm (a), P0, the 2-conversation gap-test trace of §4.4, c = 2.
- Goal: prove that the §4.1 patch sleeps through `pre_gap`, fires the side-channel controls, and dumps per-turn `ttfts` / `cached_tokens_details` / `start_times` before any GPU hours are spent on it.
- Runs on / takes: `container: sglang_hicache`; ~1 h together with row 1 (estimate, §7.3), of which one warm boot is 278-285 s (measured 2026-09-17, §2.6). One discarded long probe first (the first long prefill in a fresh container took 9.0 s, measured, `R8/DEVIATIONS.md` D9).
- Expected result: the §4.4 pass criteria, all of them: wall time >= 10 s; the server log shows 6 conversation requests with ~5 s spacing between rounds of the same conversation plus 4 controls; the JSONL has 6 entries in `ttfts` / `cached_tokens_details` / `start_times` (conversation-major, round-minor); rounds 1-2 show `device > 0`; the 4 `control_details` rows show `storage == 0, host == 0, device <= 64`; `wall(scale 1) - wall(--agentic-gap-scale 0)` = 10 s +/- 1 s; the load line reads `#Conversations: 2 (... turns/conv min=3 max=3 ...)` (`agentic_trace.py:107-112`).
- Expected lessons: a gap difference far from 10 s means the sleep hunk is not on the awaited path (or warmup sleeps too); a control row with `host > 0` or `storage > 0` means the salt or the side channel is broken and every paired comparison in rows 2-6 would be invalid. The `device` value on rounds 1-2 (~floor64 of the previous round's prompt, not ~prompt - 64) decides whether the §5.2 history note holds and therefore which denominator the §6.2 restore-hit classifier uses.

**Row 1 — the constants for this box.** Arms (a) and (c), P0, §5.4 blocks A-F.
- Goal: replace all nine `measured-A100` constants plus the two never-measured ones with values for this GPU and this disk, because every c, f, gap scale, K and cell time in §3.4, §5.3, §7.1 and §7.3 rests on them. **This is the porting step; nothing past row 2 may be sized before it.**
- Runs on / takes: `container: sglang_hicache`; block A on a pinless arm (a), blocks B/E/F on arm (a) or (b) at P0, blocks C and D on arms (a) and (c) respectively (block D writes L3, so it needs the three-tier arm and a wiped store); ~25 min GPU-busy in total (estimate, §5.4) plus two to three boots.
- Expected result: the ten values of §5.4's table recorded in `constants.json` (§9.2) and in §5.4 itself. The three that decide the rest: the **profiled device pool** (estimate ~700,000 tok, which sizes the PW level), the **recompute curve and P** (which set the admission bar `b x P` and every recompute time in §3.5/§7.3), and the **L3 fit** (which, against the bar, decides whether L3 is admissible at all here, §9.3).
- Expected lessons: the two headline outcomes pull in opposite directions and both matter. If `P` rises more than the disk did (2.8x), the L3 admission ratio **falls** despite the faster SSD and PL gets harder to demonstrate, not easier. If the decode rate rises a lot, `turn_time` and hence f fall, the required mean gaps of §5.3 shrink, and rows 4-6 get cheaper in both gap scale and wall time. A block-D L3 rate near the A100's 0.623 GiB/s despite a 1.88 GiB/s device means the nixl path, not the disk, is the limit — profile it (§8) before sizing rows 5-6.

**Row 2 — tie control and honest short-gap case.** Arms (a), (b), (c), P0, LMCache with real gaps (median 0.71 s), c = 4 and 8: 6 cells.
- Goal: show that with WS = 100K / 200K tokens < L1 = 262K no tier is ever read, so the three arms tie; and report the short-gap regime as the honest negative case (§9.1). It also yields the first measured X (turns/s) and turn_time(c) for re-budgeting §7.3.
- Runs on / takes: `container: sglang_hicache`, §7.2 per cell; estimate 55-70 min per c=4 cell, 85-105 min per c=8 cell (§7.3), all GPU-busy.
- Expected result: per-turn-index TTFT of (a), (b), (c) within noise of each other; `load_back_tokens_total` and `prefetched_tokens_total` deltas ~0 on (b) and (c) (§6.1); measured f ~0.85-0.92 (estimate on the 2.08 s mean gap), i.e. in-flight ~0.9 c from `num_running_reqs`/c in `memtime.csv`. Not yet measured: record the measured X, turn_time(c) and f here after the run.
- Expected lessons: if the arms differ at P0, something other than tiering moves TTFT (write-through cost on (b)/(c), host-pool pinning) and that difference is the correction to subtract in rows 3-6; if measured X or turn_time differ materially from the §7.3 estimates, re-budget §7.3 and re-derive `s_H`/`s_L` before row 4.

**Row 3 — overload reference.** Arms (e), (g), (d), PH, LMCache with real gaps, c = 8; run once.
- Goal: record what the pressure cells look like at gap scale 1, where f ~0.90-0.92 puts ~7.3 x 25K = 182K tokens in flight against L1 = 131K (estimate): admission queueing, not tiering.
- Runs on / takes: `container: sglang_hicache`, §7.2 per cell; estimate 85-105 min per cell (scale 1, c=8, §7.3), GPU-busy; the (e) cell may be GPU-bound and wall-time capped (§7.3).
- Expected result: `num_queue_reqs` > 0 for most of the in-flight == N window (`memtime.csv`), TTFT rising with queue depth at send time (§9.1 bins), every WS > L1 cell an overload cell as §5.3 predicts. Not yet measured.
- Expected lessons: this row is binned by queue depth and reported as such, never as a tiering result; if median queue depth in the window is ~0, the f estimate was too high and rows 4-6 can use a smaller gap scale than §5.3's; if the queue never drains, `s_H` must be raised above the ~13-16 s mean gap before row 4.

**Row 4 — host-restore case.** Arms (e), (g), (d), PH, LMCache at gap scale `s_H` (mean gap >= ~13-16 s, §5.3), c = 8.
- Goal: measure the L2 effect: WS = 1.53 L1 = 0.40 (L1+L2), so everything that leaves HBM fits in the host pool; (g) vs (e) isolates host restore, (d) should equal (g) because L3 is idle by construction.
- Runs on / takes: `container: sglang_hicache`, §7.2 per cell; estimate 100-125 min per tiered cell (§7.3), GPU-busy. `s_H = required_mean_gap / mean(pre_gap)`: x9-11 on the converted trace's emitted-turn mean of 1.49 s (measured 2026-09-18 in `agent_cache/traces/lmcache_agentic_trace.json.stats.json`; x10-12 for a swebench-only replay at 1.32 s; the README mean of 2.08 s gave x6-8). The required gap (13-16 s) is the §5.3 estimate on the 220-token placeholder: re-derive it after row 1, then the scale.
- Expected result: `prefetched_tokens_total` and `storage_hit` deltas ~0 on (d); paired p50 (restore - control) TTFT < 0 with a 95 % CI excluding 0 over >= 30 pairs per pooled turn-index bin for (g) and (d) vs their controls, at median queue depth ~0 in the window, plus >= X % idle-session memory-time freed vs (e) (the §9.3 serve criterion; X fixed in `constants.json` beforehand, 20 % proposed). Not yet measured.
- Expected lessons: if (d) differs from (g) at PH, L3 is being read or its write-through is costing something it should not (check `storage_hit`, cleaner deletions, `w_await`); if (g) does not beat (e) with the CI, the queue, not the tier, is setting TTFT and the gap scale is too small for a tiering result; if (g) wins but frees < X % memory-time, the parking policy has little to gain at this gap distribution (§9.3).

**Row 5 — L3 case.** Arms (e), (g), (d), PL, LMCache at gap scale `s_L` (mean gap >= ~31-39 s), c = 12 (c = 16 only if the scale can reach a 46-58 s mean gap).
- Goal: the only level where L3 can show anything (HANDOFF §5): WS = 1.12 (L1+L2) at c=12 (1.49x at c=16), so ~32K tokens (c=16: 131K) live only on the SSD; (g) recomputes what (d) reads from L3.
- Runs on / takes: `container: sglang_hicache`, §7.2 per cell with K = 72 and the 95/85 cleaner watermarks (§5.3); estimate 165-205 min per cell (§7.3), GPU-busy; 60 conversations at c=12, 64 (the L3 disk cap of §5.3) at c=16, whose in-flight == N window is therefore shorter: say so in the report. `s_L` = x21-26 at c=12 and x31-39 at c=16 on the 1.49 s emitted-turn mean (derived, §5.3; the required gaps are re-derived after row 1).
- Expected result: on (d) `storage_hit` / `prefetched_tokens_total` deltas > 0 and `df /mnt/ssd` growing by ~4.0 GiB per conversation plus ~3.05 GiB per control (derived, §5.3); paired p50 (restore - control) < 0 with the 95 % CI excluding 0 for (d), and (d) ahead of (g) on TTFT and memory-time freed (§9.3). Not yet measured on this box; on the A100 the equivalent cell produced **11 storage hits in a whole run** and a three-way tie (§7.0).
- Expected lessons: whether an L3 hit beats recompute at idle is §5.4 block D's answer, not a given (it was 7 of 7 lengths on the A100). Under load the disk's **1.88 GiB/s contended read ceiling** (measured, §2.2) is shared by every concurrent restore and by write-through, but at ~3x the A100's it should no longer be the binding constraint — which makes the **prefetch-admission cap** (half the host pool, cause 2 of §7.0) the most likely remaining limiter. Record `storage_prefetch_unfulfilled_tokens_total{reason}` for exactly that reason. If (d) loses to (g) here with the queue drained and unfulfilled ~0, disk KV does not pay for this model under load, and that is the go/no-go answer; watch `w_await` and `%util` for `vdc` in `iostat.log`. If the cleaner deletes during the cell (`l3_stats.json`, cleaner log lines), the §5.3 budget was wrong and the row is not trustworthy until it is redone.

**Row 6 — prefetch policy.** (d) vs (f), with (g) as the host-only reference, PL, as row 5, c = 12.
- Goal: decide between `timeout` (d) and `wait_complete` (f) under load: (f) is the policy every measured constant was taken with; (d) with the default budget (1.0 s + 0.25 s per 1,024 tokens = 4,096 tok/s, derived §5.1) may measure the cut-off, not the tier.
- Runs on / takes: `container: sglang_hicache`, §7.2 per cell; estimate 165-205 min per cell (§7.3), GPU-busy; (d) and (g) cells can be the row-5 cells if nothing else changed (same seed, offset, K, scale).
- Expected result: `storage_prefetch_unfulfilled_tokens_total{reason}` per arm (§6.1); on (d) either the timeout reason is negligible and (d) ~ (f), or it dominates and (d) shows partial hits plus recompute of the rest. Not yet measured.
- Expected lessons: HANDOFF §2's "`timeout` beats `wait_complete` at idle" is an H100 8B observation, untested here; if the timeout reason dominates on (d), raise `prefetch_timeout_per_ki_token` (e.g. 1.0) in `l3_extra.json` for the whole row (§5.1) rather than reading the cut-off as a tier result. The winner is the policy every later L3 cell uses.

**Row 7 — AgentX via AIPerf.** Arm (d), PL, `aiperf profile --scenario inferencex-agentx-mvp` for 1800 s (§3.4), c = 8.
- Goal: the long-context standardized view, context-capped at 32K (`--max-context-length 32768`; AgentX median context is 142K, so many trees truncate).
- Runs on / takes: `container: sglang_hicache`, `/opt/aiperf/bin/aiperf` (needs §2.5, not done on 2026-09-18); 1800 s plus boot = ~35 min (estimate, §7.3), GPU-busy.
- Expected result: at c=8 WS <= 8 x 32,768 = 262K < L1+L2 = 268K, so `storage_hit` / `prefetched_tokens_total` ~0 unless subagent trees multiply the live contexts (unverified); AIPerf's per-turn TTFT/ITL report and the recorded storage-hit tokens. Not yet measured.
- Expected lessons: not an L3 result: say so. If storage hits are > 0, subagent trees are multiplying the working set and a c=8 AgentX cell is a real L3 cell: re-derive its WS from the storage counters. AIPerf's idle guard compresses gaps > 10 s at low concurrency (§10): keep c high or raise `--system-idle-gap-cap-seconds`.

**Row 8 — Mooncake toolagent.** Arm (a), P0, `--backend sglang` with `--mooncake-slowdown-factor` >= ~15 (§3.4), c = 8.
- Goal: cross-session prefix sharing only (global `hash_ids`); it has no gaps and rounds fire as a burst.
- Runs on / takes: `container: sglang_hicache`; the trace offers 6.6 req/s (23,608 requests in 1 h) against a server estimated at 0.3-0.5 turns/s (§5.3), so the slowdown factor 15 stretches it to 15 h: run only a wall-time-capped slice (§3.4, §7.3).
- Expected result: a prefix hit rate from cross-session sharing (`prefill_effective_tokens_total{mode=device_hit}`) over the slice; without the slowdown factor only backlog is measured. Not yet measured.
- Expected lessons: this row calibrates the simulator's cross-session sharing (§0 step 2), nothing about parking; if backlog dominates even at slowdown 15 the factor must be raised, not the concurrency.

**Rules that hold for every row.** Arm order within a row is interleaved per cell (dflash RUNBOOK 0.4, `dflash_eval/RUNBOOK.md:194`) so thermal or queue drift does not bias one arm. `agent_cache/scripts/run_compare.sh` is the driver that already does one arm-per-boot sweep with a shared client config (it produced §7.0); prefer extending it over pasting §7.2 by hand once row 2 has run once manually. `--seed` and `--dataset-offset` are fixed per row (recorded in `config.json`): every arm and gap scale in a row replays the same conversations in the same order (the loader rotates by offset and takes the first `--num-prompts`, `agentic_trace.py:74-76,82-83`; sessions span 14K-130K context, so a per-arm offset confounds arm with sample). Salt only the control prompts (`RC_SALT`, `hicache_eval/scripts/archive/exp2.py:30`); rotate by row if at all. Use >= 5 x c conversations (20 at c=4, 40 at c=8, 60 at c=12, 64 at c=16) so the in-flight == N window dominates (§9.1). **The A100's 64-conversation cap was an L3 disk budget and no longer binds** (§5.3: 30 % of 2.27 TiB = 698 GiB against ~4.0 GiB per conversation), so the cap is now a cell-time decision, not a disk one. `--control-every` K is fixed per row like the seed (the control load is part of the workload) and is likewise now a **GPU-budget** choice alone (§7.3), not a disk one: the A100's K = 40 / 72 split existed to stay under its cleaner watermark. `--agentic-max-turns 20` halves cell time (§7.3) but caps contexts near ~25K, which lowers WS: if used, use it for every arm of the row and record it.

**Expected result:** after the campaign every row has one cell dir per (arm, c) under `$RUNDIR/` with the §9 files and a `summary.csv` row; `constants.json` carries the two row-1 constants and the measured X / turn_time(c) of row 2; each row's `config.json` files show one seed, offset, K and gap scale across all its arms; rows 4-5 have >= 30 paired (restore - control) differences per pooled bin (§6.5).
**If it differs:** a row whose arms used different seeds or offsets cannot be compared arm-to-arm: rerun the odd cell with the row's values. A three-tier cell that ran with more than 64 conversations or K = 4 has probably crossed the cleaner watermark: check `l3_stats.json` and the cleaner lines before reading it.
**Expected lessons:** the order exists because rows 2-8 cannot be sized without row 1 and cannot be interpreted without row 2: a pressure cell run before the tie control has no way to separate tier cost from the cost of merely enabling HiCache, and a cell sized on the estimated decode rate may spend a day measuring queueing.

### 7.2 Per-run checklist (one cell)

**Status:** **not started on this box.** The state blocks 0, 0b and 1 assert was checked 2026-09-21: `/mnt/ssd` on `/dev/vdc`, `/mnt/ssd/hicache_l3` empty, **no container and no results stamp yet** (§2.3, §2.4 block 6 must run first). Blocks 3-7 are the §5.2 / §5.4 / §4.6 commands sequenced; block 7 uses the standalone client of §4.6 (`--gap-scale`, `--control-every`), which is in the checkout, so no `git apply` is involved. `agent_cache/scripts/` now supplies `start_server.sh`, `start_client.sh`, `stop_server.sh`, `run_compare.sh` and `timeline.py` (they drove §7.0); `hicache_eval/scripts` still supplies `hcommon.py`, `cachectl.py`, `probe.py`, `telemetry.sh`. **`agent_cache/scripts/start_server.sh` hardcodes the pressure levels and the L3 path: check it matches §5.2/§5.3 before using it instead of the blocks below.**
**Goal:** produce one complete, comparable cell: a cold L3 and page cache, a server whose two pool sizes are asserted, counters that exclude the warm-up, 1 s telemetry that excludes the post-run flush, and the file set of §9, so that any two cells of a row differ only in the arm.
**Runs on:** `container: sglang_hicache`, as root (the container's user), under `bash` (`docker exec -it sglang_hicache bash`; the container's default shell is zsh and the §5.2 launch line uses bash arrays), working directory `/sgl-workspace/sglang/hicache_eval/scripts` (= `/home/wanhr/sglang/hicache_eval/scripts` on the host) because `python3 -c 'import hcommon'` imports from cwd. Block 12's `chown` runs on the host as `wanhr`.
**Touches:** destroys the whole L3 store `/mnt/ssd/hicache_l3` (block 1) and the page cache (block 2); writes `$RUNDIR/server_<ARM>.log`, `$RUNDIR/server_<ARM>.pid`, the cell dir `$OUT` and `$RUNDIR/summary.csv`; starts and stops one server (blocks 3, 11) and three background samplers (block 5).
**Takes:** one boot (**not yet measured here**; `measured-A100` 278-285 s warm, ~633 s JIT-cold in a new container) plus the client time of the row (re-derived after §5.4; §7.3) plus a drain of seconds to minutes; GPU busy from block 3 to block 11.

```bash
# container: sglang_hicache (bash), cwd = /sgl-workspace/sglang/hicache_eval/scripts
# 0. Variables for this cell: paste FIRST, in the same shell as every later block of this checklist.
#    RUNDIR = the stamp dir (same definition as §5.2), OUT = this cell's dir. The exports are read by
#    hcommon.py:11-14 (defaults are the 8B's 147456 B/token and /var/hicache_l3) and hcommon.py:142 (flush timeout, default 120 s).
ARM=<ARM>                # hbm_lru | hbm_host | three_tier | three_tier_p | hbm_lru_p | three_tier_wc | hbm_host_p (arms a-g, §5.2)
TRACE=<TRACE>            # lmcache | gaptest | agentx | mooncake (the row's trace, §7.1)
C=<C>                    # the row's concurrency: 2, 4, 8, 12 or 16
GAP=<GAP>                # --gap-scale (§4.6; --agentic-gap-scale in the §4.1 fallback): 1.0 for rows 2-3, s_H for row 4, s_L for rows 5-6 (§5.3; s_H/s_L not fixed yet)
K=<K>                    # --control-every (§4.6; --agentic-control-every in the §4.1 fallback): 40 for rows 2-4, 72 for rows 5-6 (§5.3 disk budget, §7.3 GPU budget; provisional)
ROW_SEED=<ROW_SEED>      # client --seed: chosen once per matrix row, identical for every arm and gap scale of that row
ROW_OFFSET=<ROW_OFFSET>  # client --dataset-offset: likewise fixed per row
NCONV=$(( 5*C > 64 ? 64 : 5*C ))   # conversations: >= 5 x c, capped at 64 (a cell-time choice here; the A100's L3 disk cap no longer binds, §5.3)
RUNDIR=/sgl-workspace/sglang/agent_cache/results/$(cat /sgl-workspace/sglang/agent_cache/.current_results)
OUT=$RUNDIR/${ARM}_${TRACE}_c${C}
export MODEL=Qwen/Qwen3-32B-FP8 KV_BYTES_PER_TOKEN=131072 L3_DIR=/mnt/ssd/hicache_l3 NVME_DEV=vdc HICACHE_FLUSH_TIMEOUT=1800
mkdir -p "$OUT" && cd /sgl-workspace/sglang/hicache_eval/scripts
# the cell's identifiers, fixed per row (§7.1), into config.json (§9); the exact server args and the patch sha are added by block 12a
printf '{"arm":"%s","trace":"%s","c":%s,"gap_scale":%s,"control_every":%s,"seed":%s,"dataset_offset":%s,"num_prompts":%s,"model":"%s","kv_bytes_per_token":%s,"l3_dir":"%s"}\n' \
  "$ARM" "$TRACE" "$C" "$GAP" "$K" "$ROW_SEED" "$ROW_OFFSET" "$NCONV" "$MODEL" "$KV_BYTES_PER_TOKEN" "$L3_DIR" > $OUT/config.json
echo "RUNDIR: $RUNDIR"; echo "OUT: $OUT"; echo "NCONV: $NCONV"; echo "cwd: $(pwd)"; echo "config: $(cat $OUT/config.json)"
```
-> prints `RUNDIR: /sgl-workspace/sglang/agent_cache/results/<stamp>` (the stamp created by §2.4 block 6), `OUT: .../<ARM>_<TRACE>_c<C>`, `NCONV:` 10 / 20 / 40 / 60 / 64 for c = 2 / 4 / 8 / 12 / 16, `cwd: /sgl-workspace/sglang/hicache_eval/scripts`, and `config:` one JSON object with every field filled (an empty or `<...>` value means a placeholder was not replaced). An empty `RUNDIR` suffix (`results/`) means `.current_results` is missing: run §2.4 block 6 (results stamp) first.

```bash
# container: sglang_hicache (bash; variables from block 0)
# 0b. Preflight gate: the L3 store must be on the attached SSD, not on the root disk (§2.2). Stops the checklist, never the shell
#     (a hard-stopping form of this gate belongs in bench.sh, §11, not in a pasted block). Same check as §2.2's, seen from inside the container.
echo "L3 dir source: $(findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3)"
echo "L3 fs free:    $(df -h --output=avail /mnt/ssd | tail -1 | tr -d ' ')"
[ "$(findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3)" = /dev/vdc ] || { echo "STOP: L3 dir is not on /dev/vdc - fix the mount on the host (§2.2) before any cell"; false; }
```
-> prints `L3 dir source: /dev/vdc` and `L3 fs free: 2.3T` (checked 2026-09-21). `/dev/vda1` or an empty value = `/mnt/ssd` was not mounted when the container started: every "L3" write would land on the root disk, ~4x slower, with no error. Stop, `sudo mount -a` on the host, and note that `docker restart` may not be enough (the bind mount must see the new filesystem: §2.3 `:rshared`). Unlike on the A100 box this is now a rare failure — the volume is persistent and in `fstab` — but it is exactly as silent, so the gate stays.

```bash
# container: sglang_hicache, as root (the container's user; the bucket dirs 00..ff are root-owned, §2.2) (variables from block 0)
# 1. Cold L3 between arms: DESTROYS every file and bucket dir under /mnt/ssd/hicache_l3 (the previous cell's whole L3 store).
#    find -delete is ARG_MAX-safe; `rm -rf $L3_DIR/*` silently fails past ~10k files (HANDOFF §3; hicache_eval/scripts/env.sh:22).
#    Takes seconds to ~1 min (estimate) for the ~85k files a full three-tier cell leaves (§10). Gated on the device check so a root-disk dir is never "cleaned" as if it were the store.
[ "$(findmnt -n -o SOURCE -T /mnt/ssd/hicache_l3)" = /dev/vdc ] || { echo "STOP: L3 dir is not on /dev/vdc (block 0b)"; false; } \
  && find /mnt/ssd/hicache_l3 -mindepth 1 -delete
echo "L3 files left: $(find /mnt/ssd/hicache_l3 -type f | wc -l)"
[ "$(find /mnt/ssd/hicache_l3 -type f | wc -l)" -eq 0 ] || { echo "STOP: L3 wipe failed (run inside the container as root, or with sudo on the host)"; false; }
```
-> prints `L3 files left: 0`. Any other count means the wipe ran without root or the store is on a different path than the server's `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR` (§5.2 `L3ENV`): the next cell would not be cold (HANDOFF §3: a "cold" run that was not cold cost a whole experiment once). The server recreates the 256 bucket dirs at boot.

```bash
# container: sglang_hicache (--privileged, so /proc/sys/vm/drop_caches is writable; hcommon.drop_page_cache falls back to fadvise otherwise, hcommon.py:154-171)
# 2. Drop the page cache so this cell's L3 reads come from the SSD, not from RAM.
sync && echo 3 > /proc/sys/vm/drop_caches && echo "drop_caches: ok, Cached now $(awk '/^Cached:/{print $2}' /proc/meminfo) kB"
```
-> prints `drop_caches: ok, Cached now <N> kB` with N far below the pre-drop value (this host had 2 GB of buff/cache idle on 2026-09-21; it grows to tens of GB during a cell). `Permission denied` or `Read-only file system` means the container is not privileged: use `python3 cachectl.py droppc` (prints `{"method": "fadvise", "cached_kb": ...}`) and record that the weaker method was used, because fadvise only evicts the L3 files, not the trace or the model.

```bash
# container: sglang_hicache (bash; variables from block 0; the §5.2 variables block pasted in this shell: it defines set_level, L3, L3WC, L3ENV)
# 3a. Launch the arm exactly as §5.2 lists it; log to $RUNDIR/server_<ARM>.log (the §5.2 / §5.4 grep lines expect that name) and keep the pid for block 11.
#     Shown for arm (a) hbm_lru. For (b)/(g) use "${COMMON[@]}" "${HOST[@]}"; for (c)/(d) prefix `env "${L3ENV[@]}"` and append "${HOST[@]}" "${L3[@]}";
#     for (f) prefix `env "${L3ENV[@]}"` and append "${HOST[@]}" "${L3WC[@]}" (§5.2). Same redirection. Do not re-assign OUT.
set_level <L1> <HSIZE>     # P0: set_level 262144 64 | PH: set_level 131072 48 | PL: set_level 131072 18 | PW: set_level 655360 96 (§5.3; the same level for every arm of the row)
LOG=$RUNDIR/server_${ARM}.log
python3 -m sglang.launch_server "${COMMON[@]}" > "$LOG" 2>&1 &
echo $! > $RUNDIR/server_${ARM}.pid
echo "server pid: $(cat $RUNDIR/server_${ARM}.pid)"; echo "server log: $LOG"
```
-> prints `level set: L1=<L1> tokens, host pool=<HSIZE> GB`, `server pid: <PID>` (a live process: `kill -0 <PID>` succeeds) and `server log: .../server_<ARM>.log`, which starts growing within seconds. `set_level: command not found` means the §5.2 variables block was not pasted in this shell. A process that dies at once with `the following arguments are required: --model-path/--model` means COMMON was empty (set_level was not run in this shell; verified read-only 2026-09-18 with a bare `python3 -m sglang.launch_server` in `sglang_hicache`); one that dies with `unrecognized arguments` means a stale array from an older paste: re-paste the §5.2 block and rerun from `set_level`.

```bash
# container: sglang_hicache (bash; variables from block 0)
# 3b. Wait for readiness: /health AND the 'fired up' log line (the same two conditions as hicache_eval/scripts/start_server.sh:40-53).
#     Budget 2400 s (300 s is enough for a warm boot, 2400 s covers a new container; the script's default is 900): 278-285 s warm, ~633 s JIT-cold (measured, §2.6). Takes 5-11 min.
#     Watch with: tail -f $RUNDIR/server_${ARM}.log      Stop with: kill $(cat $RUNDIR/server_${ARM}.pid)
LOG=$RUNDIR/server_${ARM}.log; ready=0
for i in $(seq 1 2400); do
  if curl -sf -o /dev/null --max-time 3 http://127.0.0.1:30000/health && grep -q 'The server is fired up and ready to roll' "$LOG"; then ready=1; break; fi
  kill -0 "$(cat $RUNDIR/server_${ARM}.pid)" 2>/dev/null || break
  sleep 1
done
echo "ready: $ready after ${i}s"
[ "$ready" -eq 1 ] || { echo "STOP: server not ready (died or > 2400 s); read: tail -40 $LOG"; false; }
```
-> prints `ready: 1 after <N>s` with N in 278-285 for a warm boot with a 100 GB host pool (measured 2026-09-17, `R32/*/server.log` `Engine startup timings`; smaller host pools pin faster, so N may be lower) or up to ~633 for the first boot in a new container. `ready: 0` with the pid gone: the server died, read the tail (a `larger than the profiled value` line or an OOM at host-pool pinning are the known causes, §5.1, §5.3).

```bash
# container: sglang_hicache (bash; variables from block 0; L1 / HSIZE are the set_level values used in block 3a)
# 3c. Save startup_facts.txt and assert BOTH pool sizes before trusting the cell (§5.2): a stale --hicache-size would otherwise go unnoticed.
LOG=$RUNDIR/server_${ARM}.log
grep -E 'max_total_num_tokens|Allocating .* host memory|HiCache|storage backend|Marlin|KV Cache is allocated|Load weight end' "$LOG" > $OUT/startup_facts.txt
echo "device pool: $(grep -aoE 'max_total_num_tokens=[0-9]+' "$LOG" | head -1)"
echo "host pool: $(grep -aoE 'host pool: [0-9]+ tokens' "$LOG" | head -1)"
echo "pin-ignored warnings: $(grep -c 'larger than the profiled value' "$LOG")"
echo "O_DIRECT lines: $(grep -c 'O_DIRECT is active' "$LOG")"
echo "cleaner: $(grep -ao 'HiCacheL3Cleaner started: dirs=\[[^]]*\] high=[0-9.]*% low=[0-9.]*%' "$LOG" | head -1)"
```
-> prints `device pool: max_total_num_tokens=262144` (P0) or `=131072` (PH, PL) or `=655360` (PW) = the `$L1` of `set_level`; `host pool: host pool: 488320 tokens` / `366272` / `137344` / `732480` for `--hicache-size` 64 / 48 / 18 / 96 (derived from the §5.1 rule; only 762,944 @ 100 GB is `measured-A100`, `R32/exp1_32b/startup_facts.txt:5`), empty on arms (a) and (e); `pin-ignored warnings: 0`; on the nixl arms (c), (d), (f) `O_DIRECT lines: 1` and `cleaner: HiCacheL3Cleaner started: dirs=['/mnt/ssd/hicache_l3'] high=30.0% low=20.0%` (the watermarks come from the §5.2 `l3_extra.json`, §5.3), both empty on (a), (b), (e), (g). A device pool other than `$L1`, a host pool other than the level's, or a warning count > 0 means the pin did not take: stop the server (block 11) and fix the §5.2 arrays. A cleaner dir other than `/mnt/ssd/hicache_l3` means `L3ENV` was not passed: every "L3" write would go to `/tmp/hicache_storage`, which on this box is the **root disk** (§5.1, §10). `high=80.0% low=70.0%` means `l3_extra.json` did not reach the cleaner (§5.3).

```bash
# container: sglang_hicache (bash; variables from block 0; cwd = hicache_eval/scripts)
# 3d. ONE manual salted single-shot warm-up request (never the client's --warmup-requests: it replays the whole first conversation into L2/L3, see Expected lessons).
#     It also absorbs the one-time first-long-prefill cost of a fresh container (9.0 s seen, R8/DEVIATIONS.md D9).
python3 probe.py --len 4096 --seed $RANDOM
```
-> prints one JSON line with `"prompt_tokens": 4096`, `"ttft_s"` ~1.86 s (idle recompute median at 4,096 tokens, measured 2026-09-17, `R32/exp1_32b/ttft_by_tier.csv`; up to ~9 s once in a fresh container), `"cached_device": 0` and `"cached_storage"` 0 or null (the seed is fresh). A large `cached_*` value means the seed collided with an earlier probe: send another one.

```bash
# container: sglang_hicache (bash; variables from block 0; cwd = hicache_eval/scripts)
# 4. Flush L1+L2 through the gated endpoint (scheduler.py:4904-4906 runs only when is_fully_idle, whose HiCache clauses are scheduler.py:4779-4785),
#    never by watching hicache_backup_tokens_total (HANDOFF §3). Waits up to 600 s for the warm-up's write-through to drain. L3 files stay on disk.
python3 -c 'import hcommon; print("flushed after s:", hcommon.wait_until_flushable(600, verbose=True))'
```
-> prints zero or more `waiting for backup drain... <N>s files=<F>` lines and then `flushed after s: <seconds>`; on a nixl arm F ends at 128 (the 4,096-token warm-up = 128 files, 536,870,912 B, measured, `R32/exp0_32b/exp0_results.json` step2_backup), which is ~1.3 s of writing at the 0.38 GiB/s ceiling (derived). A `RuntimeError: cache never became flushable` means a request is still running or the backup queue is stuck: check `curl -s http://127.0.0.1:30000/metrics | grep num_running_reqs` before retrying.

```bash
# container: sglang_hicache (bash; variables from block 0)
# 5. Telemetry on at 1 s (the 2026-09-17 campaign logs are 5 s averages, hicache_eval/scripts/telemetry.sh:9,11; iostat exists in the container and on the host).
iostat -x -d -t 1 vdc > $OUT/iostat.log 2>/dev/null & echo $! > $OUT/.iostat.pid
nvidia-smi dmon -s t -d 1 > $OUT/pcie.log 2>/dev/null & echo $! > $OUT/.dmon.pid
echo "iostat pid: $(cat $OUT/.iostat.pid)"; echo "dmon pid: $(cat $OUT/.dmon.pid)"
# then paste the §6.1 memtime sampler block (it reads $OUT from this shell and writes $OUT/memtime.csv and $OUT/.memtime.pid)
```
-> prints `iostat pid: <PID>` and `dmon pid: <PID>`; within a few seconds `$OUT/iostat.log` has the `Device ... r/s ... w_await ... %util` header and `$OUT/pcie.log` the `# gpu rxpci txpci` header; after the §6.1 block `$OUT/.memtime.pid` exists and `wc -l $OUT/memtime.csv` grows by ~1 line/s. A `pcie.log` that stays empty means `nvidia-smi dmon` is not supported in this container: note it, the PCIe column is auxiliary (§8).

```bash
# container: sglang_hicache (bash; variables from block 0; cwd = hicache_eval/scripts)
# 6. Counters before, taken AFTER the block-4 flush so the warm-up prefill is excluded from the delta.
python3 cachectl.py scrape $OUT/metrics_before.txt && echo "sglang metric lines: $(grep -c '^sglang:' $OUT/metrics_before.txt)"
```
-> prints `{"written": ".../metrics_before.txt"}` and `sglang metric lines: <N>` with N > 0. Zero lines means `--enable-metrics` is missing from the launch line (it is in `COMMON`, §5.2).

```bash
# container: sglang_hicache (bash; variables from block 0; the §4.3 patch applied, else the two --agentic-* flags are rejected)
# 7. The client: standalone gap-faithful replay (§4.6), no warmup, no --flush-cache. Foreground; takes 55-205 min per cell (estimate, §7.3).
#    Watch with (another shell): tail -f $OUT/client.log      Stop cleanly: Ctrl-C here (the server keeps running; continue with block 8 and mark the cell aborted in summary.csv).
# endpoint
ENDPOINT=(--url http://127.0.0.1:30000 --model Qwen/Qwen3-32B-FP8)
# dataset: the converted LMCache trace; the row's fixed offset and seed; at most 40 turns per conversation (20 halves cell time but lowers WS, §7.1)
DATASET=(--trace /sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json
         --num-conversations $NCONV --offset $ROW_OFFSET --max-turns 40 --seed $ROW_SEED)
# load: C live conversations, closed loop ("N live sessions": the slot is held across gaps, not N in-flight requests)
LOAD=(--concurrency $C)
# gaps and paired controls: sleep pre_gap x GAP from the end of each reply; one random-id /generate control every K returning turns (§6.5)
GAPS=(--gap-scale $GAP --control-every $K)
# replay template (§4.5): must match the file in COMMON (nothink -> off, think -> on)
REPLAY=(--thinking off)
# output: one JSON line per turn and per control (appended)
OUTPUT=(--tag ${ARM}_c${C} --output $OUT/client.jsonl)
cd /sgl-workspace/sglang/agent_cache/scripts && python3 replay_agentic.py "${ENDPOINT[@]}" "${DATASET[@]}" "${LOAD[@]}" "${GAPS[@]}" "${REPLAY[@]}" "${OUTPUT[@]}" 2>&1 | tee $OUT/client.log
```
-> `client.log` opens with `#Conversations: <NCONV>  turns: <T>  concurrency: <C>  gap x<GAP>  thinking: off`, then one `conv N done: ... turns, ... s elapsed`
line per finished conversation, and ends with `summary: {... "turns": T, "controls": ~39 x NCONV / K, "errors": 0, "mean_ttft_s": ..., "mean_reply_reprefilled": < 64}`.
`$OUT/client.jsonl` gains one `kind:run` line, T `kind:turn` lines (`t_send`, `ttft`, `cached_details`, `reply_reprefilled`, ...), the `kind:control` lines
(`cached_tokens <= 64` each, §6.5) and one `kind:summary`. `mean_reply_reprefilled` >= ~100 means the replies are not being reused: stop, fix the boot (§4.5), rerun the cell.
A cell run twice into the same `$OUT` appends a second `kind:run ... kind:summary` block: keep the last and say so. (The §4.1 patched `bench_serving` form is the
superseded alternative: its single JSONL line with `ttfts` / `cached_tokens_details` / `start_times` / `control_details` lists maps onto the per-turn `ttft` / `cached_details` / `t_send` fields and the `kind:control` lines.)

```bash
# container: sglang_hicache (bash; variables from block 0)
# 8. memtime sampler OFF first: the block-9 drain flushes the pools and would be recorded as a drop in occupancy (§6.4).
kill $(cat $OUT/.memtime.pid) && rm -f $OUT/.memtime.pid; echo "memtime rows: $(wc -l < $OUT/memtime.csv)"
```
-> prints `memtime rows: <N>` with N ~ client wall seconds + 1 (header), and the count no longer changes on a second `wc -l`. A `kill: no such process` means the sampler died earlier (the harness kills background waiters when RAM runs low, §10): the memory-time integral is truncated, check the last `t` in the csv against the client's end time.

```bash
# container: sglang_hicache (bash; variables from block 0; cwd = hicache_eval/scripts)
# 9. Drain (up to 5400 s), then counters after and the delta. The drain should be short: the SSD writes faster than the 32B prefills on average (§5.3),
#    but the last requests' backups still burst for ~0.4-10 s (derived, §5.3). Watch with: python3 cachectl.py l3   (files growing, then flat)
python3 -c 'import hcommon; print("drain s:", hcommon.wait_until_flushable(5400, verbose=True))' \
  && python3 cachectl.py scrape $OUT/metrics_after.txt \
  && python3 cachectl.py delta $OUT/metrics_before.txt $OUT/metrics_after.txt $OUT/metrics_delta.json | grep -E 'prefill_effective_tokens_total|hicache_backup_tokens_total|load_back_tokens_total|prefetched_tokens_total|storage_prefetch_hit_tokens_total'
```
-> prints `drain s: <seconds>` (seconds to a few minutes, estimate), `{"written": ".../metrics_after.txt"}`, then the nonzero deltas of the listed families (the full delta is in `metrics_delta.json`): on write-through arms `hicache_backup_tokens_total` ~ the sum of `prefill_effective_tokens_total{mode=...}` (2026-09-17 reference: 197,184 prefilled -> 195,648 backed up, `R32/exp1_32b/metrics_after.txt`); `load_back_tokens_total` > 0 only on host arms under pressure (rows 3-6); `prefetched_tokens_total` > 0 only on (d)/(f) at PL. A drain that reaches 5400 s means the backup queue is stuck: take the after-snapshot anyway, mark the cell, and stop the server.

```bash
# container: sglang_hicache (bash; variables from block 0; cwd = hicache_eval/scripts)
# 10. Telemetry off, then the L3 store size (nixl writes K and V as separate files: files = 2 x pages, 4,194,304 B per file, 8,388,608 B per page for this model, §2.2).
kill $(cat $OUT/.iostat.pid) $(cat $OUT/.dmon.pid) 2>/dev/null; rm -f $OUT/.iostat.pid $OUT/.dmon.pid
python3 cachectl.py l3 | tee $OUT/l3_stats.json
echo "ssd used: $(df -h /mnt/ssd | awk 'NR==2{print $3" of "$2" ("$5")"}')"
```
-> prints `{"files": <F>, "bytes": <B>}` with B = F x 4,194,304 exactly and F even on the nixl arms (c), (d), (f); `{"files": 0, "bytes": 0}` on (a), (b), (e), (g); then `nvme used: <used> of 369G (<pct>)`. F odd or B not a multiple of 4,194,304 means a partial page write (a killed backup thread): note it. Used space near 80 % of 368.0 GiB means the cleaner ran during the cell (§5.3 budget): check the cleaner lines in the server log before trusting any L3 hit.

```bash
# container: sglang_hicache (bash; variables from block 0)
# 11. Stop the server from a SCRIPT FILE (stop_server.sh uses pkill/pgrep -f from its own file) or by its pid. NEVER paste pkill/pgrep -f into
#     `docker exec ... bash -lc '<...>'`: the pattern matches that bash's own command line (a wait loop did exactly that on 2026-09-17 and never ended:
#     R8/DEVIATIONS.md D5). Long cells are watched through a marker file or a log line, never pgrep (§10).
bash /sgl-workspace/sglang/hicache_eval/scripts/stop_server.sh        # alternative: kill $(cat $RUNDIR/server_${ARM}.pid)
echo "gpu procs: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
```
-> prints `stopped (gpu procs: 0)` (the script's own line, `stop_server.sh:16`) and `gpu procs: 0`. A count > 0 after the script's 60 s wait means a scheduler subprocess survived: `kill -9 $(cat $RUNDIR/server_${ARM}.pid)` and re-check; never start the next cell's server while it is above 0 (both would fail to allocate).

```bash
# container: sglang_hicache (bash; variables from block 0)
# 12a. Close the cell: keep this cell's server log (the next cell of the same arm overwrites $RUNDIR/server_<ARM>.log), the exact server args and
#      the patch sha next to config.json (§9), and append its summary.csv row.
cp $RUNDIR/server_${ARM}.log $OUT/server.log
grep -m1 -ao 'server_args={.*}' $OUT/server.log > $OUT/server_args.txt
sha256sum /sgl-workspace/sglang/agent_cache/patches/0001-agentic-trace-pre-gap.patch > $OUT/patch.sha256 2>/dev/null
echo "server args chars: $(wc -c < $OUT/server_args.txt)"; echo "patch sha256: $(cut -c1-16 $OUT/patch.sha256)"
echo "cell files: $(ls $OUT | tr '\n' ' ')"
```
-> prints `server args chars: <N>` with N > 0 (the `server_args={...}` line is line 6 of a 2026-09-17 log, `R32/exp1_32b/server.log`), `patch sha256: <16 hex chars>` (empty means `agent_cache/patches/` does not exist yet: §4.3 not done, and then block 7 could not have run), and `cell files:` containing `startup_facts.txt server.log server_args.txt patch.sha256 client.jsonl client.log metrics_before.txt metrics_after.txt metrics_delta.json memtime.csv iostat.log pcie.log l3_stats.json config.json` (the §9 per-cell set). A missing file names the block that was skipped.

```bash
# container: sglang_hicache (bash; variables from block 0)
# 12b. Append this cell's summary.csv row: the identifiers of block 0, wall seconds and turns completed, read from the LAST line of client.jsonl (block 7 appends one line per run).
#      The §9.1 headline columns are added by analyze.py (§11, not written yet); until then the row is identifiers + wall_s + turns_completed.
read -r WALL TURNS <<<"$(python3 -c 'import json,sys; r=json.loads(open(sys.argv[1]).read().splitlines()[-1]); print(round(r["duration"],1), len(r.get("ttfts") or []))' $OUT/client.jsonl)"
[ -s $RUNDIR/summary.csv ] || echo 'ts,arm,trace,c,gap_scale,control_every,seed,dataset_offset,num_prompts,wall_s,turns_completed' > $RUNDIR/summary.csv
echo "$(date -u +%FT%TZ),$ARM,$TRACE,$C,$GAP,$K,$ROW_SEED,$ROW_OFFSET,$NCONV,$WALL,$TURNS" >> $RUNDIR/summary.csv
echo "summary rows: $(($(wc -l < $RUNDIR/summary.csv) - 1))"; echo "last row: $(tail -1 $RUNDIR/summary.csv)"
```
-> prints `summary rows: <N>` (one more than before this cell) and `last row: <ts>,<ARM>,<TRACE>,<C>,...,<wall_s>,<turns_completed>` with this cell's ARM/TRACE/C and a non-zero `wall_s` and `turns_completed` (turns_completed = the length of `ttfts`, one entry per replayed turn, §4.4 block 5; expected NCONV x turns/conv, up to 40 x NCONV). An empty `WALL` / `TURNS` (a row ending in `,,`) means `client.jsonl` is missing or its last line is not JSON: block 7 did not finish; mark the cell aborted in the row by hand.

```bash
# host, as wanhr
# 12c. Everything the container wrote is root:root on the host: take ownership before analysis or a commit (§11).
sudo chown -R wanhr:wanhr /home/wanhr/sglang/agent_cache/results /home/wanhr/sglang/agent_cache/traces
echo "root-owned left: $(find /home/wanhr/sglang/agent_cache/results /home/wanhr/sglang/agent_cache/traces -user root | wc -l)"
```
-> prints `root-owned left: 0`. Anything else: a file was written after the `chown` (a sampler still running in the container: check block 8 and 10's pid files).

**Expected result:** `$OUT/` holds the §9 per-cell set listed under block 12a, `$RUNDIR/summary.csv` has one new row (written by block 12b), `$RUNDIR/server_<ARM>.pid` points at a dead process, `nvidia-smi` shows 0 compute apps, and the L3 store still holds this cell's files (they are the next cell's block 1 to destroy). Every number in the cell was taken with the two pools asserted (block 3c), the warm-up excluded (blocks 4, 6) and the sampler stopped before the drain (block 8).
**If it differs:** the client's `cache_report` hit rate looks wrong for multi-turn: its denominator is turn-0 `prompt_tokens` for every round (§4.2), use the per-turn details. Conversation 0 turn 0 is an L3 hit on a three-tier arm: `--warmup-requests` was not 0 or block 1 was skipped. The delta's `prefill_effective_tokens_total` does not match the sum of the per-turn details: the before-snapshot was taken before the flush (block order 4 then 6), or a control was counted on one side only (§6.4).
**Expected lessons:** the client's `--warmup-requests` replays the **whole first conversation** (`serving.py:1423,1431-1451`), which is also conversation 0 of the main run (`:1549-1551`); with `write_through` its KV reaches L2 and L3 before `--flush-cache` fires (`:1470-1471`), and `flush_cache` resets tree, host pool and controller queues (`scheduler.py:4904-4914`; `unified_radix_cache.py:343-377`; `cache_controller.py:742-762`) but **never the storage backend** (`clear_hicache_storage`, `scheduler.py:4633-4635`): conv 0 turn 0 would be an L3 hit in every three-tier cell and the block-6 snapshot would include warmup prefill (HANDOFF §3 trap); hence block 3d + `--warmup-requests 0`. `--max-concurrency` is per conversation, semaphore held across gaps (`serving.py:1385-1393`): "N live sessions", not N in-flight requests; `--output-file` appends (`:1899`). The order of blocks 4 -> 6 and 8 -> 9 is what makes the counters and the memory-time integral attributable to the client window alone.

### 7.3 Time budget

**Status:** **not derivable yet on this box, by construction.** Cell time is computed from the recompute curve, the decode rate and the prefix-extension cost — all three are §5.4 constants and none is measured here. What this section gives instead is (i) the formula, unchanged, (ii) the **one real measurement** the study has, from `C18/` on the A100 (§7.0), and (iii) the A100's estimate table, kept as a strict upper bound. **Rewrite the row figures from the measured X and turn_time(c) after rows 1-2** (stored in `constants.json`, §9.2).
**Goal:** know before starting a row how many GPU-hours it costs and which cells must be wall-time capped, so the campaign fits in the planned days and K is chosen by budget rather than by habit.
**Runs on:** none (arithmetic on the measured and estimated inputs below).
**Touches:** read-only; the measured X and turn_time(c) go into `results/<stamp>/constants.json` after rows 1-2.
**Takes:** none; the figures are the budget itself.

Fixed costs on this box: no daily overhead at all (the SSD is persistent, §2.2; `docker start` is ~2 s). One-time bring-up ~30-60 min (image pull + model download, §2.3/§2.6, both network-bound and both unmeasured here). Data + Week-1 stats 0.5 day (CPU, §3). Row 1 ~25 min of GPU plus 2-3 boots (§5.4).

**The one measured cell (A100, §7.0).** Three arms x (32 conversations, c = 32, 12 turns, gaps x10, `GAP_CAP=600`): **2,608-2,783 s wall per arm (~44-46 min)**, 380-381 turns each, 95 % of turns done by 2,103-2,163 s. That is **~7.3 s of wall per turn at c = 32**, and it was **admission-bound, not GPU-bound** (peak queue 28 on every arm), so it is a lower bound on how fast a correctly-sized cell runs and an upper bound on how much of that time was useful.

Cell time is **computed, not guessed**: `cell_s ~ turns / X + boot + drain`, with `X = min(c / (turn_time(c) + s x mean_gap), 1 / GPU_s_per_turn)`,
`GPU_s_per_turn = (1-m) x t_delta + m x T_rec(context) + decode share + T_rec(context)/K`, m = fraction of returning turns that recompute (0 in a tiered arm
that restores, up to 1 in (e); **`C18/` measured m = 0.43-0.44 in all three arms**, which is what a mis-sized cell looks like), t_delta = the §5.4
prefix-extension constant, K = `--control-every` (each control is a full recompute), turns = min(5 x c, 64) x <= 40.

The control term decides K, and on this box it is a **GPU-budget decision only** — the A100's L3-disk constraint on K is gone (§5.3). The table below is
the A100's, with `T_rec(25K) = 18.0 s`; **recompute the first column from this box's `T_rec(25K)` as soon as §5.4 block C has run**, and the ceiling
column with it.

| K | control GPU s per turn (`T_rec(25K)` / K) | A100: 18.0 s | this box: `T_rec(25K)` = ? |
|---|---|---|---|
| 4 | `T_rec`/4 | 4.5 s (~70 % of the GPU on controls) | to be filled |
| 16 | `T_rec`/16 | 1.1 s | to be filled |
| 40 | `T_rec`/40 | 0.45 s | to be filled |
| 72 | `T_rec`/72 | 0.25 s | to be filled |
| no controls | 0 | 0 | 0 |

**A100 per-row estimates, kept as an upper bound.** (§5.3 closed-loop model on the README mean gap 2.08 s and 220 output tokens, K = 40; K = 72 at c=12.)
X ~ 0.20-0.26 turns/s at c=4, 0.27-0.33 at c=8 (scale 1), 0.22-0.28 (PH c=8 at s_H), 0.20-0.25 (PL c=12 at s_L).

| row | cell | `measured-A100` estimate | this box |
|---|---|---|---|
| 2 | c=4 | ~55-70 min per cell | <= that (faster GPU); re-derive after row 1 |
| 2 | c=8 | ~85-105 min per cell | <= that; re-derive |
| 4 | tiered cell, PH c=8 at s_H | ~100-125 min per cell | <= that; re-derive |
| 5 (and 6) | PL c=12 at s_L | ~165-205 min per cell | <= that; re-derive |
| 3-6, arm (e) | recompute-heavy, GPU-bound at `1/(t_delta + m x T_rec + decode + T_rec/K)` | ~3.6 h at m = 0.35, 8.4 h at m = 1 (c=8, 40 conv x 40 turns) | scales down with `T_rec`; still the cell to wall-time cap |
| 7 | AIPerf 1800 s + boot | ~35 min | ~35 min (client-duration-bound, so unchanged) |
| 8 | Mooncake | wall-time-capped slice (§3.4) | same |
| 9 | PW, c=24-32 | not applicable on the A100 | to be derived from the row-1 constants |

Rows 2-6 are ~17 cells: **4-5 days on the A100** at 40 turns, about half with 20. Expect less here in proportion to whatever speed-up §5.4 measures;
reserve a day for re-runs either way.

**Expected result:** after row 1, this section is rewritten with `T_rec(25K)`, the decode rate and t_delta for this box, and the K table's third column
is filled; after row 2, the row estimates are replaced by measured X and turn_time(c) from `constants.json` and by the measured cell wall times. Today
none of that exists and the only real datum is the `C18/` 44-46 min per arm at c = 32.
**If it differs:** a cell that takes far longer than the A100 estimate is not a slow GPU — check `num_queue_reqs` in `memtime.csv` first, because an
admission-bound cell (§7.0) looks exactly like a slow one in wall time and completely different in the event log. A cell that finishes far *faster* than
expected with a high `m` (recompute fraction) is the `C18/` failure mode: it is drawing turns through a pool too small to hold them.
**Expected lessons:** the control frequency is a GPU-budget decision, not a statistics one: K = 4 would spend ~70 % of the GPU on controls (on the A100),
so pairs per bin come from pooling turn indices or repeating the cell (§6.5), never from lowering K — but the **disk** half of that argument no longer
applies on this box (§5.3), so K may be lowered further than the A100 allowed if the GPU budget permits. A recompute-heavy (e) cell can still cost a
working day on its own; capping it by wall time and comparing over the common window is what keeps a row to one day. (`CLIENT_TIMEOUT`, enforced inside the client since 2026-09-22 via `--max-seconds`; the earlier `timeout --foreground` never reached the client, §7.0b.) And the single most useful budgeting
fact available today is not an estimate at all: `C18/` shows that **a badly sized cell costs a full 45 minutes per arm and yields nothing attributable**,
which is why §5.4 (25 min) runs first.

---

## 8. Profiling

This section exists so that a TTFT or throughput difference between two arms of a §7.1 row can be attributed to a mechanism (H2D load-back, L3 file IO,
GIL contention, admission queueing) instead of being guessed from aggregate numbers. It gives the recipe for each tool that works on this box, what each
can and cannot attribute, and the profiler traps found in the source. At its end the reader has one torch trace of a returning-turn burst, one py-spy
census of the scheduler and the telemetry logs for the cell being explained; nothing in this section feeds §9.1 directly.

**Status:** not started as of 2026-09-18: no torch trace and no py-spy census has been taken on this box with the 32B. Every recipe below was checked
against the source anchors given; the kernel names and the CUPTI capture of the H2D stream are estimates until the first trace is opened.
**Goal:** explain a §9.1 result, e.g. say whether a slow L3 restore in a PL cell (§7.1 row 5) was the SSD, the `timeout` cut-off (§5.1), the GIL or a
queued admission (§5.3), so that the §9.3 verdict names a mechanism and not only a number.
**Runs on:** `container: sglang_hicache` (`docker exec -it sglang_hicache bash`) for the profiler endpoints, `python -m sglang.profiler`, py-spy
(`/opt/sglang/bin/py-spy`) and the analysis script: the manual window needs client and server on one filesystem, which they are inside the container.
`iostat` and `nvidia-smi dmon` run in the container or on the host (both have `iostat`, §1). `/home/wanhr/sglang/agent_cache` on the host =
`/sgl-workspace/sglang/agent_cache` in the container.
**Touches:** writes trace files under `/sgl-workspace/sglang/agent_cache/results/traces/` (the profiler dir as originally planned; the §11 layout puts
`traces/` under `results/<stamp>/`, so `$RUNDIR/traces` is the layout-conformant value: pick one and record it in the cell's `config.json`),
`dumps.txt`, `$OUT/iostat.log`, `$OUT/pcie.log`; all root-owned on the host until the §11 chown. A profiler window perturbs the server it runs in
(CUPTI overhead; a `with_stack` trace balloons), so profile a dedicated repeat of a cell, never a cell whose numbers go into §9.1.
**Takes:** manual window: the burst being profiled plus the trace export (estimate: up to a few minutes for a burst of returning turns; `stop_profile`
replies before the export is finished); batch-count capture: 40 scheduler batches; py-spy census: 12 samples x 2 s = ~30 s, plus up to 30 s per
sample if the scheduler does not answer; analysis script: ~1 min per trace (estimate). GPU busy with the workload under study.

Paste the §5.2 variables block first (it defines `RUNDIR`, `COMMON`, `HOST`, `L3`, `L3ENV`). The profiler env goes on the launch line because the
scheduler reads `SGLANG_TORCH_PROFILER_DIR` when a window opens (`profiler_manager.py:129`) and the tokenizer reads the two `SGLANG_PROFILE_*` defaults
(`tokenizer_control_mixin.py:384-389`), both in processes that exist only after launch.

```bash
# container: sglang_hicache, bash; paste the §5.2 variables block first (RUNDIR, set_level, HOST, L3, L3ENV). Profiler env for the SERVER: prepend it to the §5.2 launch line of the arm you will profile
PROF_ENV=(SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/agent_cache/results/traces   # fallback output dir; the start_profile body below overrides it
          SGLANG_PROFILE_WITH_STACK=false        # default True via the tokenizer (tokenizer_control_mixin.py:384-387): with stacks the trace balloons
          SGLANG_PROFILE_RECORD_SHAPES=false)    # same reason
# example: arm (c) three_tier at P0 = the §5.2 arm (c) block with the env prefix added; same log and pid names, so the §5.2 readiness and assert blocks apply unchanged
ARM=three_tier; set_level 262144 64
env "${PROF_ENV[@]}" "${L3ENV[@]}" python3 -m sglang.launch_server "${COMMON[@]}" "${HOST[@]}" "${L3[@]}" > "$RUNDIR/server_$ARM.log" 2>&1 &
echo $! > "$RUNDIR/server_$ARM.pid"; echo "launched $ARM (profiled): pid $(cat "$RUNDIR/server_$ARM.pid"), log $RUNDIR/server_$ARM.log"
```
-> prints `level set: L1=262144 tokens, host pool=64 GB` and `launched three_tier (profiled): pid <N>, log .../server_three_tier.log`; then run the
§5.2 readiness block with `ARM=three_tier` and the §5.2 assert block, exactly as for an unprofiled launch (`READY: three_tier after <N>s`, both pool
sizes asserted). `set_level: command not found` means the §5.2 variables block was not pasted in this shell.

```bash
# container: sglang_hicache. Open a profiling window on the running server (CPU + GPU activities, no stacks, no shapes); THEN fire the returning-turn burst to attribute
TRACES=/sgl-workspace/sglang/agent_cache/results/traces     # or $RUNDIR/traces to follow the §11 layout; use the same value in the stop block
mkdir -p $TRACES/win1
curl -sS -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$TRACES/win1\",\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false,\"profile_id\":\"hicache-window\"}"
```
-> prints `Start profiling.`; any other body means the server is not up (§5.2) or a window is already open (close it with the next block first).

```bash
# container: sglang_hicache. Close the window after the burst; the export runs after the reply, so the ls may need a retry
TRACES=/sgl-workspace/sglang/agent_cache/results/traces     # same value as in the start block
curl -sS -X POST http://127.0.0.1:30000/stop_profile
sleep 30
echo "trace: $(ls -l $TRACES/win1/hicache-window-TP-0.trace.json.gz 2>&1)"
```
-> prints `Stop profiling. This will take some time.` and then `trace: -rw-r--r-- ... hicache-window-TP-0.trace.json.gz` with a non-zero size (the
name is `<profile_id>-TP-0.trace.json.gz`, `profiler_manager.py:328-353`). `No such file` means the export is still running (repeat the `ls`) or
the window was never opened.

```bash
# container: sglang_hicache. Alternative window: the next 40 scheduler batches (prefill + decode, counted at scheduler.py:4179-4189) instead of a request-defined window; blocks until the trace is written (profiler.py:21-66)
python -m sglang.profiler --url http://127.0.0.1:30000 --num-steps 40 \
  --output-dir /sgl-workspace/sglang/agent_cache/results/traces --profile-prefix hicache-window
```
-> prints `Dump profiling traces to /sgl-workspace/sglang/agent_cache/results/traces/<time>` and `Waiting for 40 steps ...`, then returns; the file
is `<time>/hicache-window-<profile_id>-TP-0.trace.json.gz` (prefix + id, `profiler_manager.py:328-353`). Without `--output-dir` it writes to
`$SGLANG_TORCH_PROFILER_DIR` of the CLIENT shell, default `/tmp` (`profiler.py:18`). It cannot bracket one request: use the manual window for that.

```bash
# container: sglang_hicache, cwd /sgl-workspace/sglang. Offline analysis of ONE trace: kernel / overlap-opportunity / fuse-pattern tables (Skill llm-torch-profiler-analysis)
cd /sgl-workspace/sglang && python3 .claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --input /sgl-workspace/sglang/agent_cache/results/traces/win1/hicache-window-TP-0.trace.json.gz \
  --kernel-table-limit 0      # 0 = every kernel row; rows under 1 % GPU-time share are still hidden by default (SKILL.md:35-39) and the script has no cutoff flag: ask the skill for a lower cutoff so transfer kernels appear
```
-> prints three tables; the pass condition is a kernel table whose top rows are Marlin / Triton names (expected list in the table below) and no
FlashAttention / flashinfer / DeepGEMM row. **Never** the script's `--url` live mode: it drives synthetic 4090/2048 workloads that evict the state under
study (SKILL.md:190-217).

```bash
# container: sglang_hicache. py-spy GIL census of the scheduler under load: 12 non-blocking dumps 2 s apart, from a SCRIPT FILE (pgrep -f matches the calling shell's own command line, §10): never through `docker exec ... bash -lc '...'`
OUT=${OUT:?paste the §7.2 step 0 block first: it defines OUT}
cat > /tmp/pyspy_census.sh <<'EOF'
#!/bin/bash
# pattern: hicache_eval/scripts/archive/c3_dump.sh:12,21-36
set -e
OUT=${1:?usage: pyspy_census.sh <cell dir>}
SCHED=$(pgrep -f 'sglang::scheduler' | head -1)
[ -n "$SCHED" ] || { echo "STOP: no sglang::scheduler process"; false; }
for i in $(seq 1 12); do echo "===== $i"; timeout 30 py-spy dump --pid $SCHED --nonblocking; sleep 2; done > $OUT/dumps.txt 2>&1
echo "samples: $(grep -c '^===== ' $OUT/dumps.txt)"
echo "gil holders (count, thread):"
grep -E '^Thread .*\(active\+gil\)' $OUT/dumps.txt | sed -E 's/.*: "(.*)"/\1/' | sort | uniq -c | sort -rn
EOF
bash /tmp/pyspy_census.sh $OUT
```
-> prints `samples: 12`, then `gil holders (count, thread):` followed by `<count> <thread name>` lines; a `STOP:` line means no server is running.
**Never `py-spy record -s`** on the launcher: it hung 4.5 h on the ~100 GB-RSS scheduler (HANDOFF §3).

```bash
# container: sglang_hicache (or host: same commands). Whole-device telemetry at 1 s: the SSD vdc (r vs w is the only prefetch/backup split) and GPU PCIe rx/tx; §7.2 step 5 starts this same pair
OUT=${OUT:?paste the §7.2 step 0 block first: it defines OUT}
iostat -x -d -t 1 vdc > $OUT/iostat.log 2>&1 & echo $! > $OUT/.iostat.pid
nvidia-smi dmon -s t -d 1 > $OUT/pcie.log 2>&1 & echo $! > $OUT/.dmon.pid
echo "iostat pid: $(cat $OUT/.iostat.pid)"
echo "dmon pid: $(cat $OUT/.dmon.pid)"
```
-> prints `iostat pid: <n>` and `dmon pid: <n>`; both logs grow by one sample per second. Stop them with `kill $(cat $OUT/.iostat.pid) $(cat $OUT/.dmon.pid)`
(§7.2 step 10). Reference formats from this box and model: `R32/exp1_32b/{pcie,iostat}.log` (5 s samples).

What each tool can and cannot attribute (recipes are the blocks above):

| tool | recipe | attributes | cannot attribute |
|---|---|---|---|
| torch profiler, manual window | blocks 1-3 above: server env at launch, `POST /start_profile` with `output_dir`, `activities`, `with_stack`, `record_shapes`, `profile_id`, then `POST /stop_profile` -> `win1/hicache-window-TP-0.trace.json.gz` | H2D load-back as `transfer_kernel_impl<...>` kernels (io_backend `kernel`, `kernels/aot/csrc/kvcacheio/transfer.cu:261,354-377`), or as `Memcpy HtoD` from that same kernel path's `cudaMemcpyAsync` fallback (`:845-849`) or from the `direct` backend's Python-side `transfer_kv_direct` copies (`memory_pool_host.py:433-448`; not in `transfer.cu`), on the separate `host_to_device_stream` (`l2_transfer.py:52-55`); D2H backup on `device_to_host_stream`; forward-stream waits on per-layer load events. Compute kernels for this model (names read from source, ESTIMATE until the first trace is opened; no FlashAttention / flashinfer / DeepGEMM kernels should appear): linear layers = `sglang::apply_fp8_marlin_linear` -> `gptq_marlin_gemm` (JIT Marlin, `marlin_utils_fp8.py:58-78`; `kernels/ops/quantization/gptq_marlin.py:21-36`); prefill attention = Triton `_fwd_kernel` (`kernels/ops/attention/extend_attention.py:327`); GQA decode = `_fwd_grouped_kernel_stage1` + `_fwd_kernel_stage2` (`decode_attention.py:510,904`). Prefill (breakable) and decode (full) CUDA graphs are on, so replayed kernels have no CPU-op parent: search by kernel name | L3 file/NIXL IO (Python daemon threads, no CUDA/torch ops); GIL time |
| batch-count capture | block 4: `python -m sglang.profiler --url ... --num-steps 40 --output-dir ... --profile-prefix hicache-window` (blocks until written; `profiler.py:21-66`); `num_steps` counts scheduler batches, prefill+decode (`scheduler.py:4179-4189`) | same | a request-defined window (use the manual start/stop) |
| analysis skill | block 5: `Skill llm-torch-profiler-analysis` -> `analyze_llm_torch_profile.py --framework sglang --input <trace>`; ask for rows below the 1 % default cutoff so transfer kernels appear (SKILL.md:35-39). **Never** its `--url` live mode (SKILL.md:190-217) | kernel / overlap / fuse tables | stream timelines (use Perfetto on the same file) |
| `generate-profile` skill | `python3 -m sglang.test.send_one --profile` (SKILL.md:61-81) | one synthetic request | a HiCache window; use only as a sanity check of the profiler path |
| py-spy | block 6, from a script file (`hicache_eval/scripts/archive/c3_dump.sh:12,21-36`): `SCHED=$(pgrep -f 'sglang::scheduler' \| head -1)`, 12 x `timeout 30 py-spy dump --pid $SCHED --nonblocking` 2 s apart, census of `(active+gil)` threads. **Never `py-spy record -s`** on the launcher (hung 4.5 h, HANDOFF §3) | thread state at sample instants: GIL holder, `synchronize (torch/cuda/streams.py)` = waiting on GPU, frames in `prefetch_thread_func`/`backup_thread_func` | durations |
| `nvidia-smi dmon -s t -d 1` / `iostat -x -d -t 1 vdc` | block 7 = §7.2 step 5; `iostat` is installed on the host and in the image (there is **no `fio` on this box**, §1). Log formats from the A100 and the same model: `R32/exp1_32b/{pcie,iostat}.log` (5 s samples). **Reference ceiling for `vdc`, measured 2026-09-21 (§2.2): 1,973,500 kB/s = 1,927 MiB/s = 1.88 GiB/s at 100 % util, symmetric read and write** (the A100's nvme0n1 was 704 MiB/s read / 391 MiB/s write). PCIe is Gen5 x16 here vs Gen4 x16 there, so the H2D/D2H ceiling roughly doubles too | whole-GPU PCIe rx/tx MB/s; whole-device r/w kB/s, await, %util | load-back vs other H2D; prefetch vs backup vs writeback (r vs w is the only split); per-turn bytes (HANDOFF §3: `iostat_*` is a window) |
| `/metrics` histograms | §6.1: `load_back_duration_seconds` (CUDA-event span per merged op, excludes queue/fence waits) and `hicache_backup_duration_seconds`; bytes_total/duration_sum = BW while copying | L2<->GPU throughput | L3 durations for nixl (no `get_stats`): derive from `prefetched_tokens_total x 131,072 / wall`, cross-check iostat rkB/s; anything above ~0.68 GiB/s is page cache, not the SSD; the idle single-stream reference is 0.623 GiB/s (measured, §5.4) |

Profiler traps (source facts): `profile_stages` is a no-op unless `SGLANG_PROFILE_V2` (`profiler_manager.py:102`, `environ.py:435`); `with_stack`
defaults True via the tokenizer (`tokenizer_control_mixin.py:384-387`) so pass `false` or traces balloon; the bench client mkdirs `output_dir/<time>`
on the **client** side and sends the absolute path (`serving.py:853-857`), fine only because client and server share the container filesystem;
`--profile-steps`/`--profile-num-steps` suppress the client's `/stop_profile` (`:1566-1580`). CUPTI capture of the H2D stream on this A100 is expected
but unverified until the first trace is opened.

**Expected result:** not yet measured. `win1/hicache-window-TP-0.trace.json.gz` exists with a non-zero size; the analysis kernel table shows (estimate,
names read from source) `gptq_marlin_gemm` for the linear layers, Triton `_fwd_kernel` for prefill attention, `_fwd_grouped_kernel_stage1` +
`_fwd_kernel_stage2` for decode, `transfer_kernel_impl<...>` rows on a second stream during load-back, and no FlashAttention / flashinfer / DeepGEMM
row; `dumps.txt` holds 12 samples and the census names a GIL holder per sample; `iostat.log` shows rkB/s bursts at or below the measured read ceiling
(704 MiB/s at 97 % util, `R32/exp1_32b/iostat.log`) during L3 restores and wkB/s bursts at 391 MiB/s / 99 % util after long prefills. Record the first
opened trace's kernel table here.
**If it differs:** the trace is huge or the export takes many minutes: `with_stack` was left at its default True (pass `false` in the body and in the
env). `Memcpy HtoD` rows but no `transfer_kernel_impl`: the kernel path fell back to `cudaMemcpyAsync` (`transfer.cu:845-849`) or the arm booted with
`--hicache-io-backend direct` (check `server_args.txt`; §5.1 says `page_first_direct`+`kernel` is silently rewritten to `direct`). No trace after
`stop_profile`: no `Start profiling.` was ever printed, or the `--profile-steps` client flag suppressed the stop. `python -m sglang.profiler` wrote to
`/tmp/<time>`: `--output-dir` was omitted. A py-spy census that never returns: it was run through `bash -lc` with the pattern on its own command line
(§10), or `py-spy record -s` was used.
**Expected lessons:** a profile can say where the GPU time and the GIL went, but the L3 path is Python threads and file IO with no CUDA op, so an L3
result is explained by iostat r/w plus the `/metrics` counters, never by the torch trace alone; if the first trace shows `Memcpy HtoD` instead of
`transfer_kernel_impl`, the measured L2 rate (7.75 GiB/s, §5.4) belongs to the fallback path and the `kernel` io backend must be re-verified before
any L2 claim in §9.3.

---

## 9. What to record and report

This section fixes what every cell leaves on disk, what one report row contains, which file every analysis reads its constants from, and the three
go/no-go thresholds the study answers. It exists so that cells run on different days are comparable and so the study can stop early on a negative
Week-1 result instead of after the serving campaign. At its end the reader knows what a complete cell directory looks like, has `constants.json`
written with the two open constants marked null, and can state the verdict rule before the first measured cell.

Per cell (`results/<stamp>/<cell>/`): `startup_facts.txt`, `server.log` (§7.2 step 3 writes it as `$RUNDIR/server_<arm>.log`, one per boot: link or
copy it into the cell, or name it in `config.json`), `client.jsonl` (+`client.log`), `metrics_before/after.txt` + `metrics_delta.json`, `memtime.csv`,
`iostat.log`, `pcie.log`, `l3_stats.json`, `config.json` (exact server + client args, seed, gap scale, patch sha), `summary.csv` row.

### 9.1 Report table per (arm, pressure, gap scale, c)

**Status:** not started: no cell has been run and `agent_cache/scripts/analyze.py` (§11) does not exist yet.
**Goal:** one row per cell whose numbers are comparable across arms of a §7.1 row and across days, with the queue-depth binning that separates a
tiering effect from admission queueing, so that §9.3 can be decided from the table alone.
**Runs on:** host or `container: sglang_hicache`, offline over `results/<stamp>/<cell>/` (CPU only; the container has pandas/pyarrow, the host has
host Python 3.12.3, stdlib only, §1).
**Touches:** read-only over the cell directory; writes the cell's `summary.csv` row (§7.2 step 12) and the report tables.
**Takes:** minutes per cell (estimate); GPU idle.

```bash
# host. Completeness check of one cell directory before analysis: every file §9 lists must exist and be non-empty
CELL=/home/wanhr/sglang/agent_cache/results/$(cat /home/wanhr/sglang/agent_cache/.current_results)/<CELL>    # <CELL> = <arm>_<trace>_c<N>, the cell dir name of §7.2 step 0
for f in startup_facts.txt server.log client.jsonl client.log metrics_before.txt metrics_after.txt metrics_delta.json memtime.csv iostat.log pcie.log l3_stats.json config.json; do
  [ -s $CELL/$f ] && echo "ok      $f ($(stat -c %s $CELL/$f) B)" || echo "MISSING $f"
done
```
-> prints one `ok <file> (<bytes> B)` line per file; any `MISSING` line means the §7.2 checklist stopped early (a missing `server.log` is the
`$RUNDIR/server_<arm>.log` naming: link it in).

What the row contains:

- TTFT p50/p95/p99 **per turn index**, binned also by queue depth at send time (`start_times` hunk joined to `memtime.csv` `num_queue_reqs`; the stock
  client never emits a send time, `serving.py:1883-1896`; HANDOFF §3 "rep index is confounded with queue depth"), reported only over the window where
  in-flight == N (why >= 5 x c conversations, §7.1), plus ITL p50/p99; paired (restore - control) differences with 95 % CI per bin (§6.5).
- Throughput at fixed concurrency of **sessions** (`--max-concurrency` = live conversations): completed turns/s, output tok/s (input throughput is 0
  for multi-turn: `serving.py:1614-1616`).
- Prefix hit rate: client (`cache_report`, but with the corrected per-turn denominator, §4.2) and server (`prefill_effective_tokens_total` deltas per mode).
- HiCache counters: `load_back_*`, `hicache_backup_*`, `evicted_tokens_total`, `hicache_dropped_tokens_total{reason}`,
  `prefetched/backuped/storage_prefetch_hit/unfulfilled{reason}`, L3 files/bytes, cleaner evictions if any.
- Restore-hit rate: fraction of returning turns device-only / host-restored / storage-restored / recomputed, and recompute tokens (§6.2).
- Memory-time: idle-session token-seconds per tier from the per-turn split (§6.4), normalized by device pool x wall; the gauge integral only as an
  upper bound; "freed" only between arms with equal pool sizes; and its cost: extra TTFT at p95 vs the paired control.
- Gains stated as throughput at a fixed TTFT SLO and memory-time freed; the short-gap regime (rows 2-3, real 0.71 s median) reported as the honest
  negative case.

**Expected result:** no value measured yet. A complete row has every bullet above filled, and two internal checks hold: the server-side delta of
`prefill_effective_tokens_total{mode}` equals the sum of the per-turn `cached_tokens_details` plus the controls (§6.4), and every control row shows
`storage == 0, host == 0, device <= 64` (§6.5). Paired differences are reported only where a bin has >= 30 pairs (pool turn indices until it does).
**If it differs:** the server and client hit totals disagree: the warmup prefill leaked into the before-snapshot (§7.2 step 3 and 6 order) or a
control became an L3 hit (salt, §6.5). Fewer than 30 pairs per bin at K = 40-72: pool the turn-index bins or repeat the cell with a new `RC_SALT` and a
cold L3, never lower K or add conversations on a three-tier arm (§6.5).
**Expected lessons:** a p95 TTFT that is worse only in the queue-depth > 0 bins is a §5.3 admission-cap problem, not a tier result, so the two bins must
never be merged; the short-gap rows 2-3 are expected to be the negative case (f ~ 0.85-0.92, estimate), and reporting them as such is what makes a
positive row 4-5 result credible.

### 9.2 Constants block (`results/<stamp>/constants.json`)

**Status:** **not started on this box**, and `agent_cache/.current_results` does not exist yet (§2.4 block 6 creates it). The template below is
rewritten for this box: **every GPU- and disk-derived value is `null`** and carries the A100 value in a `*_a100_prior` field next to it, so the file
itself shows what is missing and what it is expected to be near. Eighteen top-level nulls, not two.
**Goal:** one file that every analysis script and every budget (§5.3, §7.3) reads its constants from, so that no number is re-typed elsewhere with a
different value and the two still-open constants are visibly null until measured.
**Runs on:** host, as `wanhr`.
**Touches:** writes `/home/wanhr/sglang/agent_cache/results/<stamp>/constants.json` (= `/sgl-workspace/sglang/agent_cache/results/<stamp>/constants.json` in the container).
**Takes:** seconds; GPU idle.

Pre-fill with what is true of **this** box today: the model and flags, `b` (a dtype property), the disk ceilings (measured 2026-09-21, §2.2), the
logical sector size, and the trace constants. Everything else is `null` until §5.4 (matrix row 1) fills it, with the A100 value carried alongside as a
prior. After row 1, also add the measured turn throughput and `turn_time(c)` of §7.3 and the go/no-go memory-time bar X of §9.3.
**18 top-level nulls and 10 nested ones** — against the A100 runbook's two, which is the size of the port in one number.

```bash
# host, as wanhr. Write the pre-filled constants block for the current stamp (refuses if one exists: after row 1 the nulls are edited in place, never re-nulled)
AC=/home/wanhr/sglang/agent_cache
F=$AC/results/$(cat $AC/.current_results)/constants.json
[ ! -s "$F" ] || { echo "STOP: $F exists; edit it instead of overwriting"; false; } && mkdir -p "$(dirname $F)" && cat > "$F" <<'EOF'
{
  "box": "Nebius computeinstance-u00jtv5xqvxttejgvw, 1x NVIDIA H200 143771 MiB, SM90, driver 580.173.02, CUDA 13.0",
  "source": "this box, RUNBOOK §5.4 (H200 port of 2026-09-21). *_a100_prior fields are R32/ = hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv/ (measured 2026-09-17) and are NEVER values for this box",

  "model": "Qwen/Qwen3-32B-FP8 @ aa55da1e",
  "kv_cache_dtype": "fp8_e5m2",
  "attention_backend": "triton",
  "attention_backend_note": "SM90 default is fa3; fa3+fp8_e5m2 is auto-rewritten to triton, so the pin agrees with the automatic resolution (RUNBOOK §5.1)",
  "mem_fraction_static": 0.85,
  "weight_kernel": null,
  "weight_kernel_expected": "native FP8 (NOT Marlin: can_auto_enable_marlin_fp8() is 80 <= sm < 89)",
  "weight_kernel_a100_prior": "fp8 weight-only Marlin (W8A16), mem usage 32.59 GB",
  "weights_gb": null,

  "b": 131072,
  "b_note": "2 x 64 layers x 8 KV heads x 128 x 1 byte; dtype property, expected identical to the A100; §5.4 block B re-measures it",

  "L1_TOKENS": null,
  "L1_TOKENS_estimate": 701760,
  "L1_TOKENS_a100_prior": 281216,
  "L1_TOKENS_note": "profiled device pool at mem-fraction 0.85; read off a boot WITHOUT --max-total-tokens (§5.4 block A)",
  "host_pool_tokens_at_100gb": 762944,
  "host_pool_tokens_note": "arithmetic rule (int(GB*1e9//b)//64+1)*64, box-independent; 64/48/18/96 GB -> 488320/366272/137344/732480",

  "P_marginal": null,
  "P_marginal_unit": "tok/s",
  "P_marginal_intercept_s": null,
  "P_marginal_a100_prior": 1226.3,
  "P_marginal_intercept_s_a100_prior": -1.251,
  "recompute_ttft_s": {"512": null, "1024": null, "2048": null, "4096": null, "8192": null, "16384": null, "32512": null},
  "recompute_ttft_s_a100_prior": {"512": 0.2393, "1024": 0.4552, "2048": 0.9021, "4096": 1.8597, "8192": 4.0303, "16384": 9.8764, "32512": 26.6922},
  "recompute_quadratic": null,
  "recompute_quadratic_a100_prior": "0.0626 + 3.752e-4 L + 1.365e-8 L^2 (derived; HANDOFF: P is not constant)",
  "recompute_bar_GiBps": null,
  "recompute_bar_GiBps_a100_prior": 0.150,

  "L1_GiBps": null, "L1_intercept_s": null,
  "L2_GiBps": null, "L2_intercept_s": null,
  "L3_GiBps": null, "L3_intercept_s": null,
  "tier_fits_a100_prior": {"L1_GiBps": 9.808, "L1_intercept_s": 0.053, "L2_GiBps": 7.748, "L2_intercept_s": 0.048, "L3_GiBps": 0.623, "L3_intercept_s": 0.058},
  "tier_fits_note": "wait_complete, idle server, single stream (§5.4 block D)",

  "l3_device": "/dev/vdc, 2496449740800 B ext4 label=ssd, mounted /mnt/ssd, persistent (fstab UUID + nofail)",
  "ssd_read_ceiling_GiBps": 1.882,
  "ssd_write_ceiling_GiBps": 1.882,
  "ssd_ceiling_tok_s": 15418,
  "ssd_ceiling_note": "measured 2026-09-21: 4 concurrent dd O_DIRECT streams at 4 MiB, iostat -x -d 2 steady state 1973500 kB/s at 100% util in BOTH directions (5300 IOPS of ~372 kB, aqu-sz 35-38); symmetric. Short-burst dd aggregates overstate it (up to 2205 MiB/s) because the first seconds are below 100% util. No fio on this box",
  "ssd_ceiling_a100_prior": {"read_GiBps": 0.68, "write_GiBps": 0.38},
  "lba": 512,
  "lba_a100_prior": 4096,
  "l3_cleaner_watermarks_pct": {"high": 30, "low": 20},
  "l3_cleaner_note": "pinned in l3_extra.json; the 80/70 default would be 1.82/1.59 TiB on this 2.27 TiB filesystem and never fire (§5.3)",
  "page_cache_evict_method": "drop_caches",

  "decode_rate_tok_s": {"1": null, "4": null, "8": null},
  "extend_1k_on_24k_ttft_s": null,
  "boot_s_warm": null,
  "boot_s_jit_cold": null,
  "boot_s_a100_prior": {"warm": 281, "jit_cold": 633},

  "trace_mean_pre_gap_s": 1.49,
  "trace_mean_output_length": 182.0,
  "trace_mean_replay_prompt_tokens": 19665,
  "trace_note": "properties of agent_cache/traces/lmcache_agentic_trace.json, box-independent (§3.3)",

  "open_constants_note": "every null above is a §5.4 (matrix row 1) measurement; until they are filled, no §7.1 row past 2 may be sized",
  "turn_throughput_turns_s": null,
  "turn_time_s_by_c": null,
  "measured_after_row1_note": "turn throughput and turn_time(c) of §7.3: fill from rows 1-2",
  "go_no_go_memory_time_freed_pct_X": 20,
  "go_no_go_note": "X = proposed bar (§9.3); fix it before the first row 4 cell"
}
EOF
echo "constants: $(python3 -c "import json;d=json.load(open('$F'));n=[k for k,v in d.items() if v is None];print(len(d),'keys;',len(n),'null:',','.join(n))")"
```
-> prints `constants: 63 keys; 18 null: weight_kernel,weights_gb,L1_TOKENS,P_marginal,P_marginal_intercept_s,recompute_quadratic,recompute_bar_GiBps,L1_GiBps,L1_intercept_s,L2_GiBps,L2_intercept_s,L3_GiBps,L3_intercept_s,extend_1k_on_24k_ttft_s,boot_s_warm,boot_s_jit_cold,turn_throughput_turns_s,turn_time_s_by_c`
(verified 2026-09-21 by parsing the document above; `recompute_ttft_s` and `decode_rate_tok_s` hold a further 7 and 3 nested nulls, which the key count
does not show). A `STOP:` line means the file already exists and the nulls must be edited in place, never re-nulled.

**Expected result:** `constants.json` exists in the stamp dir with: the box identity; the model, dtype, backend and mem-fraction; `b` = 131,072; the
**measured** disk block (`ssd_read/write_ceiling_GiBps` 1.943 / 2.153, `lba` 512, the cleaner watermarks, the device description); the **trace** block
(mean gap 1.49 s, mean output 182.0, mean replay context 19,665); and **twelve top-level nulls** plus two nested null maps, each paired with an
`*_a100_prior`. Counted exactly: **63 keys, 18 top-level nulls, plus 7 nested nulls in `recompute_ttft_s` and 3 in `decode_rate_tok_s`** (verified
2026-09-21). X = 20 until the user fixes it. After matrix row 1 the nulls hold numbers and `open_constants_note` is rewritten.
**Expected lessons:** the nulls are the reason every c, f, K and cell-time figure in §3.4, §5.3, §7.1 and §7.3 is provisional; a §7.1 row past 2 must not
be sized while they are null. The `*_a100_prior` fields exist so that a wrong value is *recognisable*: a measured `P` near 1,226 on an H200 would mean the
Marlin path is somehow active (§5.2), and a measured `L3_GiBps` near 0.623 on a 1.88 GiB/s disk would mean the nixl path, not the device, is the limit.
Keeping `L1_TOKENS` here is also what stops a stale value from leaking through a copied script — the A100 runbook's warning about the H100's 284,224 in
`hicache_eval/scripts/models.sh:21` now applies to the A100's 281,216 as well.

### 9.3 Go/no-go (starter kit §0, made concrete)

**Status:** rules fixed 2026-09-17 and unchanged by the port (they are statistical rules, not hardware ones); no verdict yet on this box. Week 1 needs
§3.5, the simulation needs Weeks 2-3, the serve verdict needs §7.1 rows 4-5. **The serve gate's *inputs* changed**: the admission-rule numbers below are
`measured-A100` and are re-measured by §5.4. X must be written to `constants.json` (§9.2) before the first row 4 cell.
**Goal:** three explicit stop/continue thresholds so the study can end in a week on a negative Week-1 result instead of after the serving campaign, and
so a serve-phase "win" is a paired, CI-backed statement rather than a crossover read off two curves.
**Runs on:** none: a decision step over the §3.5 output, the simulator output and the §9.1 tables.
**Touches:** `constants.json` (records X and the verdict); no files otherwise.
**Takes:** none; GPU idle.

No commands: the inputs are produced by §3.5, the simulator (short version item 2, §0) and §9.1.

- **Week 1 (characterize):** the quantity is `fraction(1 s)` from §3.5, computed as `sum(context x pre_gap | pre_gap >= 1 s) / sum(context x
  (turn_time + pre_gap))` in the closed-loop model at c = 8 with the stated turn-time model (`turn_time = T_pf + output_length / decode_rate`), for both
  ends of the decode-rate estimate, plus absolute token-seconds compared with the PH/PL device pool (131,072 tokens = 16.0 GiB). **Threshold: no-go if
  `fraction(1 s) < 10 %` of session-token-seconds**: parking cannot free much, stop or change workload. (Not "pool-seconds": a trace-only computation
  has no pool.) With an estimated 6.6-9.9 s turn against a ~2.08 s mean gap (0.71 s median) f is ~0.8 at c=1 and ~0.9 at c=8: expect a small
  `fraction(1 s)` at scale 1 unless the gap tail is heavy; that number is the Week-1 result.
- **Weeks 2-3 (simulate):** with the tier costs **measured on this box** (§5.4 block D: `restore(L) = L3_intercept + L x 131,072 / L3_GiBps`, likewise
  for L2, and `recompute(L)` from this box's re-fitted curve — the A100 forms were `0.058 s + L x 131,072 / 0.623 GiB/s` and `0.048 s + L x 131,072 /
  7.75 GiB/s`), in the PL-like regime the deadline policy must free **>= X % of idle-session memory-time**
  (X fixed in `constants.json` before the runs; 20 % is the proposed bar) **at an unchanged p95 TTFT SLO, over >= 30 simulated sessions per bin**; else
  no-go. "0.5 pp" is a hit-rate unit from the negative results (starter kit §0) and is not the criterion here.
- **Serve (rows 4-5):** go only if BOTH hold: (i) paired p50 (restore - control) TTFT **< 0 with a 95 % CI excluding 0 over >= 30 pairs per (pooled,
  §6.5) turn-index bin, at median queue depth ~0 in the window**, and (ii) **>= X % idle-session memory-time freed vs the same-pool HBM-only arm**; a
  "crossover" without a CI is a dead heat until shown otherwise (`hicache_eval/HANDOFF.md:144`, the "L3 crosses recompute at R=2" row). A tier is
  admissible only if `bandwidth(tier) > b*P` (§5.4), and **on this box neither side of that inequality is measured yet**. `measured-A100`, idle and
  single-stream for the 32B: L2 cleared the bar 52x, L3 4.2x, and an L3 hit took 0.24-0.54x of the recompute TTFT at all 7 lengths. Both sides move
  here and in opposite directions — `P` up (native FP8 on SM90, so the bar rises), the disk up 2.8x — so **§5.4 blocks C and D are a go/no-go input,
  not a detail**. The open question for rows 4-6 is then the same one the A100 could not answer: whether L3 still pays when concurrent restores share
  the device and the `timeout` policy cuts them off — except that on the A100 it demonstrably did not, for four diagnosed reasons (§7.0), three of which
  this box's disk substantially relieves and one of which (the prefetch-admission cap) it does not.

**Expected result:** not yet measured at any of the three gates. Week 1: one `fraction(1 s)` per (source, c) for both decode-rate ends, compared with
10 %. Weeks 2-3: freed memory-time in % vs X at unchanged p95 TTFT. Serve: per row 4 and row 5, the paired p50 difference with its CI and the freed
memory-time in % vs the same-pool HBM-only arm (e). Write each verdict and its inputs into `constants.json`. **Gate zero, new and cheap:** after §5.4
blocks C and D, compute `L3_GiBps / (b x P)` and the 7-length L3-vs-recompute comparison. If L3 is not admissible at idle on this box, rows 5, 6 and 9
are skipped and the study reports that — a one-hour answer to a question that cost the A100 campaign a day.
**Expected lessons:** Week 1 below 10 % means the trace has too little idle context behind >= 1 s gaps for any parking policy to matter, and the
serving weeks are skipped or the workload changed; above it, the simulation says whether a tier's restore cost eats the freed memory-time before any
GPU time is spent. In the serve gate a negative paired difference with a CI that includes 0 is "no effect shown", not "slightly worse", and a freed
memory-time below X with a good TTFT means the tier is admissible but the workload never parks enough: both are reported as such, not as a win.

---

## 10. Traps (one line each)

This section is the list of failure modes that each cost a wrong number or a lost day on this box or its predecessors (HANDOFF §3 for the inherited
ones). It exists so a reader can check, before each step, which trap applies to it; each line ends with the step it protects. At its end the reader
has read every trap once and knows where to look when a number looks too good.

**Status:** living list, last checked 2026-09-21 against the checkout (`d608a20d4`) and this box. Traps that were A100-specific have been rewritten rather than deleted, so the reason each one existed is still readable.
**Goal:** name, for every step of §2-§9, the mistake that would silently invalidate it, so the step's **If it differs** line has a known cause to point at.
**Runs on:** none (read before the step it protects).
**Touches:** read-only.
**Takes:** ~5 min to read; GPU idle.

- Paired control, not idle baseline; salt control seeds per process; assert `cached_storage` is NaN on control rows (HANDOFF §3) (protects §6.5, §7.2 step 7).
- `rm -rf $L3_DIR/*` silently fails past ~10k files, use `find -mindepth 1 -delete` (a 32B cell writes up to ~33k conversation pages plus ~10k control pages = ~85k files: 64 conv x 32.7K, 2 files per 8 MiB page, §5.3; the A100's `C18/` cell wrote 312 GiB = ~78k files); `flush_cache` 400s until fully idle incl. HiCache queues (`scheduler.py:4779-4785`), poll the flush, never `hicache_backup_tokens_total` (protects §7.2 steps 1 and 4).
- `pkill -f` AND `pgrep -f <name>` match the shell that runs them when the pattern is on that shell's command line (`bash -lc '... pgrep -f sglang ...'` never sees zero; a wait loop did that on 2026-09-17): run them from a script file, or use `server.pid`, a marker file or a log line. `py-spy record -s` hangs on a ~100 GB-RSS scheduler (`py-spy dump --nonblocking` only) (protects §7.2 step 11, §8 py-spy block).
- nixl L3 cleaner evicts at 80 %/70 % **of the filesystem**, and on this box that is 1.82 TiB / 1.59 TiB of a 2.27 TiB disk, so the default would **never fire within a cell and then fire mid-campaign**: §5.2 pins `l3_cleaner_high_watermark: 30, l3_cleaner_low_watermark: 20` in `l3_extra.json` and the §5.2 assert block checks the log line says `high=30.0% low=20.0%`. Sessions parked across long gaps are the oldest entries, so a firing cleaner is a second eviction policy inside the cell; there is no byte cap for nixl (`L3_MAX_SIZE` is file-backend only). Log `storage` per turn and treat cleaner evictions as a metric (protects §5.3 cleaner budget, §9.1).
- O_DIRECT needs every nixl file to be a multiple of the logical sector size, and nixl L3 = 2 files per page (K, V): Qwen3-32B-FP8 fp8 KV, page 64: 8,388,608 B per page = 2 x 4,194,304 B. **This disk's LBA is 512 B** (the A100's NVMe was 4 KiB), so the constraint is strictly weaker and the arithmetic holds with room to spare; the `O_DIRECT is active` line is nevertheless asserted on every three-tier boot (§5.2) because it has not yet been seen on `vdc`. File counts in `l3_stats.json` are 2 x pages. The code never falls back on EINVAL (`nixl_utils.py:286-296`), so any other model/page size either passes that check or sets `SGLANG_HICACHE_NIXL_USE_DIRECT_IO=0`; record `lsblk -o LOG-SEC /dev/vdc` in `constants.json` (protects §2.2, §5.2, §9.2).
- Working set must exceed L1+L2 or every arm ties `nohicache` (HANDOFF §5); P0 is the negative case, PL the L3 test. With the 32B, auto L1 + `--hicache-size 100` is ~1.46M tokens here (~58 sessions of 25K; it was 1,044,160 on the A100): L3 is never read there, by construction (§5.3). **The opposite error is just as fatal and is the one that actually happened:** pools far *below* the live set make every arm tie too, because TTFT becomes admission queueing (`C18/`: L1+L2 = 157K against a live set of ~500K, three-way tie, §7.0) (protects §5.3, §7.1 rows 4-5).
- Turn index == rep index == queue depth in a replay (bin TTFT by `num_queue_reqs` at send time via the `start_times` hunk, never by pool); a fixed `--num-prompts` lets the load drain before late turns (24 conv at c=12 is two waves), so use >= 5 x c conversations and report turn-index bins only over the window where in-flight == N (protects §7.1, §9.1).
- With gaps, in-flight HTTP requests << `--max-concurrency` (the intended "N live sessions" semantics, say so); warmup replays the whole first conversation (`serving.py:1431-1449`), keep `turn_meta` off the warmup input; `--agentic-gap-scale 0` == unpatched client, use it as the "no idle window" control (protects §4.1, §7.2 steps 3 and 7).
- Starter-kit flag mismatches: `--hicache-storage-prefetch-timeout` does not exist; `page_first_direct`+`kernel` becomes `direct`; `--radix-eviction-policy fifo` is rejected; timeout defaults on the live unified path are 1.0/0.25/no max, not the dead HiRadixCache 2.0/0.1/30 (pass them explicitly, §5.1) (protects §5.1, §5.2).
- `token_usage` excludes radix-cached KV (`pool_stats_observer.py:221-224`); the gauge integral is pool occupancy, ~full under LRU for every arm, so idle-session memory-time comes from the per-turn tier split (§6.4); gauges freeze while idle and refresh only per prefill batch / `--decode-log-interval` decode steps; stop the sampler before the post-run flush (protects §6.4, §7.2 step 8).
- `prefetched_tokens_total` counts raw L3->host volume before prefix dedup (use `prefill_effective_tokens_total{storage_hit}` for tokens used); `prefetch_bandwidth`/`backup_bandwidth` are empty for nixl and file, derive L3 BW from counters + iostat (protects §6.1, §8 metrics row, §9.1).
- Default nixl dir is `/tmp/hicache_storage`, which on this box is the **root disk** (`/tmp` is not a tmpfs here, §1 block C): ~4x slower than `/mnt/ssd` and completely silent, so always set `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR`. **The A100 box's "the SSD is blank every morning" trap is gone**: `/dev/vdc` is a persistent volume with an `fstab` UUID line and `nofail`, so there is no daily format and no start gate. What survives is the same silent failure with a lower probability — if the volume is detached or the mount fails, `nofail` lets the boot succeed and `/mnt/ssd` is an empty directory on `/` — so the §7.2 step-0b preflight runs before every cell and §2.2's one-line check before every session (protects §2.2, §5.2, §7.2 step 0b).
- **196 GB RAM**, no swap (the A100 box had 167): `--hicache-size 96` leaves ~100 GB and is comfortable; 128 (976,576 tokens) would still boot. The A100's `<= 64` rule was a RAM rule and no longer binds — but a large host pool still makes L3 unreachable by construction (§5.3), which is the reason PL uses 18 GB. Keep enough headroom that background Bash waiters are not killed for memory: watch long runs through a marker file / log line, never a foreground wait. **16 vCPU** (the A100 had 12) shared by scheduler, HiCache threads, tokenizer and client (the hicache_eval plan's caveat 17, `hicache_eval/hicache_eval_plan.md:332`, assumed 26): watch scheduler CPU in `top`, and re-derive the concurrency ceiling from §5.4 rather than reusing the A100's "saturates near c ~ 8"; run a py-spy census once (protects §5.3, §7.2 step 3, §8 py-spy block).
- **This GPU is SM90, and that invalidates nine measured constants** (§5.4). On Hopper the MHA default is `fa3` (not the A100's `flashinfer`), `fa3` + `fp8_e5m2` is auto-rewritten to `triton`, and FP8 checkpoints run on **native FP8**, not weight-only Marlin W8A16 (`can_auto_enable_marlin_fp8()` is `80 <= sm < 89`). Pin `--attention-backend triton` anyway so the log is unambiguous, and assert that **no Marlin line appears** in the boot log (§5.2). Never carry an A100 `P`, bar, TTFT curve or tier fit into a calculation here: they fail silently by making a cell look correctly sized (protects §5.1, §5.2, §5.4, §7.1).
- The A100's model-choice argument ("the 32B clears the L3 bar 4.2x; the 8B loses 7/7 because its bar is 1.22 GiB/s against 0.625 delivered") is a statement about **that** GPU's `b x P` against **that** disk: both sides move here (§2.6, §5.4). The 32B stays the evaluation model for comparability, but the admission ratio must be re-derived from §5.4 blocks C and D before it is quoted (protects §2.6, §7.1, §9.3).
- The first long prefill in a fresh container took 9.0 s on the A100 (one-time warm-up) and the first boot JIT-compiled for 10.6 min: send one discarded long probe after creating a container, keep the readiness timeout at 2400 s, and never recreate the container between measured stages (`docker rm` also drops the warm JIT cache in `/root/.cache/sglang`). Neither cost is measured on this box yet; record both on the first boot (§2.6) (protects §2.3, §7.2 step 3).
- L3 delivery is bounded by the disk ceiling (**1.88 GiB/s = 15,418 tok/s, symmetric, measured 2026-09-21**, §2.2), and reads and writes share that one budget. The default `timeout` budget (1 s + 4,096 tok/s, so 7.10 s for a 25K restore) covered exactly one idle restore on the A100's 0.623 GiB/s and covers four here (§5.1) — a prediction, not a measurement, until §5.4 block D shows the nixl path actually reaches the device. **Measure the ceiling from `iostat` steady state at 100 % util, never from a short `dd` aggregate:** the latter overstated it by 15 % here on 2026-09-21 (protects §5.1, §7.1 row 6).
- `hicache_eval/.current_results` points at a frozen **A100** campaign (`20260917_a100_gcp_32b70b_fp8kv`, listed in `hicache_eval/.frozen_results`): any reused hicache_eval driver that sources `env.sh` refuses to run until `RESULTS` points at a new dir (`env.sh:6-12`). Never write this box's results into `hicache_eval/results/`, and never write them into the `agent_cache/results/20260918_*` or `compare_*` dirs either — those are A100 results (protects §5.4, §11).
- **`wanhr` is not yet in the `docker` group on this box** (§2.1): until `usermod -aG docker wanhr` has run AND a new login shell has been opened, every `docker` command fails with `permission denied ... docker.sock`, which looks like a broken daemon and is not. The toolkit stage restarts dockerd (kills containers) only when the `nvidia` runtime is not yet registered (it already is). Container-written files are `root:root` on the host: the L3 bucket dirs (wipe as root), `agent_cache/results`, `agent_cache/traces`, `python/**/__pycache__` and the HF cache (`chown -R wanhr:wanhr`) (protects §2.1, §2.6, §7.2 step 12, §11).
- Never leave an **unarchived** patch under `python/` (HANDOFF §7; `91d480573` -> `2cb739b9a`); archive under `agent_cache/patches/` and `results/<stamp>/patches/`. **New on this box:** the §6.6 event-log patch is *committed* (`5b881454b`), so a clean `git status -- python/` no longer implies "no eval patch is active" — the two facts were equivalent on the A100 and are not here (protects §4.1, §4.3, §6.6, §11).
- `_normalize_round_messages` strips `tool_call_id`/`name` (convert tool messages to user text; bare `role: tool` is accepted at source level only, not run end to end, §3.3); empty `messages` turns are dropped by the loader and desync `pre_gap` (merge assistant-only iterations forward in the converter) (protects §3.3, §4.4).
- AIPerf's idle guard compresses gaps > 10 s at low concurrency; keep c high or raise `--system-idle-gap-cap-seconds` (protects §3.4, §7.1 row 7).
- A reply shorter than one page (64 tokens) is invisible to `cached_tokens`: prefix matching is page-floored, so `floor64(prompt + reply)` = `floor64(prompt)` and the usage-based reuse check reads the same whether the reply hit or not. The §4.4 gap trace has 16-token replies: prove reuse there with `--check-ids`, or use replies >= 128 tokens (protects §4.6 dry run, §4.4 block 5).
- Re-feeding the generated reply does not make it a cache hit: Qwen3's stock template renders history differently from the generation prompt (empty think block inserted only when generating; `<think>` stripped from history), thinking on or off, so the match ends at the previous prompt and every turn re-prefills ~220 reply tokens. Boot with `--chat-template .../qwen3_replay_nothink.jinja` and send `enable_thinking:false` + `skip_special_tokens:false`; verify `device ~ prompt - delta` on rounds 1-2 (protects §4.5, §4.4 block 5, §5.2, §6.2).

**Expected result:** every trap names the step it protects and every step's **If it differs** line can be traced to one of them; no new trap has been
added without a step reference.
**Expected lessons:** the traps cluster into four families: a control that was not a control (paired, salted, asserted), a store that was not where it
was assumed to be (the root disk instead of `/mnt/ssd`, a frozen results dir), a process that matched itself (pgrep, py-spy), and a gauge that measured
the pool instead of the policy. This port adds a fifth family: **a constant that belonged to another GPU** (§5.4) — the only one of the five that leaves
no trace in any log. A new anomaly is checked against those five before any new mechanism is proposed.

---

## 11. Directory layout and git hygiene

This section says where every file of the study lives on the host and in the container, which of them git tracks, and what must be undone before a
commit. It exists because container-written files are root-owned, the repo root ignores the evidence file types, and an eval patch once reached
`main` (HANDOFF §7). At its end the reader can create the stamp dir for a new campaign, confirm the tracking rules, and make a commit that touches
only `agent_cache/` with `python/` pristine.

**Status:** mostly in place as of 2026-09-21, because it is all committed: `agent_cache/.gitignore`, `patches/0002-hicache-event-log.patch`,
`templates/*.jinja`, `traces/lmcache_agentic_trace.json` (68,429,742 B) + `.stats.json`, and `scripts/{convert_lmcache,replay_agentic,replay_template,timeline}.py`
plus `{start_server,start_client,stop_server,run_compare}.sh` all exist and are `wanhr`-owned. `results/` holds the **A100** run dirs (`20260918_*`,
`compare_*`) and `results/.current_run`. **Missing on this box:** `agent_cache/.current_results` (the stamp pointer; §2.4 block 6 creates it), this box's
stamp dir, `versions.txt`, `constants.json`, and `scripts/analyze.py`.
**Goal:** a layout in which every result is tracked through the nested `.gitignore`, every reused script carries this box's constants, and no commit
can carry a `python/` patch or a root-owned file.
**Runs on:** host, as `wanhr` (`chown` needs sudo). `/home/wanhr/sglang/agent_cache` on the host = `/sgl-workspace/sglang/agent_cache` in the container.
**Touches:** `agent_cache/.current_results`, `agent_cache/results/<stamp>/`, ownership of `agent_cache/results` and `agent_cache/traces`; the
revert block discards the working-tree diff under `python/` (destructive by design, guarded: it refuses unless that diff is archived under
`agent_cache/patches/`). **On this box the revert block normally has nothing to do**, because the only eval patch is committed (§6.6).
**Takes:** seconds each; GPU idle.

Layout (tree as planned; `traces/` and `results/` are what exists today):

```
agent_cache/
  agent-kv-tiering-evaluation-starter-kit.md   RUNBOOK.md (this)
  .gitignore            copy of dflash_eval/.gitignore (!*.log !*.csv !*.png !*.jsonl; root ignores them at .gitignore:62,173,182,187) + traces/*.parquet (+ data/)
  .current_results      bare stamp, e.g. 20260917_0900 (dflash convention, dflash_eval/scripts/run_track.sh:44-51); NOT a full path (hicache_eval/.current_results style)
  patches/              0002-hicache-event-log.patch (§6.6 — COMMITTED at 5b881454b, so this file is the record and the way to remove it, not something to apply);
                        0001-agentic-trace-pre-gap.patch only if the §4.1 fallback is taken; 0003-eval-tier-log.patch only for §6.3
  traces/               lmcache_agentic_trace.json (+ .stats.json, on disk as lmcache_agentic_trace.json.stats.json), gaptest.json; converter output only
  templates/            qwen3_replay_nothink.jinja (the §5.2 default), qwen3_replay_think.jinja; generated by scripts/replay_template.py make (§4.5), tracked
  scripts/
    replay_template.py  §4.5: make (write the two templates from the model's tokenizer_config) | check (print cross-turn reply reuse, stock vs patched); host venv312
    replay_agentic.py   §4.6: the gap-faithful closed-loop replay client (per-turn JSONL, reply re-fed verbatim, /generate controls, --check-ids); container, stdlib + aiohttp
    start_server.sh     [ARM] [LEVEL]: one boot into a fresh results/<stamp>/<time>_<arm>_<level>/ dir, refuses a busy port or a live .current_run, waits for READY, prints the startup facts; SGLANG_LOG_MS=1
    start_client.sh     env knobs NCONV TURNS C GAP K THINKING CHECK_IDS ARRIVAL TAG OFFSET SEED: replay_agentic.py against the .current_run boot, output client_<time>_<tag>/ inside it, per-turn table
    stop_server.sh      kills the .current_run server, waits for the pid to exit and port 30000 to free
    env.sh              from dflash env.sh:7-29 (DOCKER/CONTAINER=sglang_hicache/PORT/BASE_URL, stamp -> RESULTS, readlink guard) + hicache env.sh, whose knobs now come from the environment:
                        L3_DIR (:4, default /var/hicache_l3), RESULTS (:5), frozen-dir guard (:6-12), MODEL (:15), NVME_DEV (:16, default vda), KV_BYTES_PER_TOKEN (:17, default 147456),
                        PAGE_SIZE=64 (:18), l3_wipe (:22). Set L3_DIR=/mnt/ssd/hicache_l3 NVME_DEV=vdc MODEL=Qwen/Qwen3-32B-FP8 KV_BYTES_PER_TOKEN=131072 as
                        run_a100_rerun_c2.sh:24-30,44-45 does for the A100 paths; AGENT_CACHE_HOST=/home/wanhr/sglang/agent_cache, _CTR=/sgl-workspace/sglang/agent_cache
    models.sh           one primary entry qwen32b: MODEL=Qwen/Qwen3-32B-FP8, KV_BYTES_PER_TOKEN=131072, MODEL_EXTRA_ARGS="--kv-cache-dtype fp8_e5m2 --attention-backend triton --mem-fraction-static 0.85",
                        HOST_TOKENS_AT_100GB=762944 (arithmetic, box-independent). **Leave L1_TOKENS, P_TOK_S, BAR_GIBPS, L2_GIBPS and L3_GIBPS UNSET (or null)
                        until §5.4 measures them on this box**, then fill from constants.json (§9.2) — never from the A100's 281216 / 1226 / 0.150 / 7.75 / 0.623,
                        and never from hicache_eval/scripts/models.sh:21 (L1_TOKENS=284224, the H100 value). Three GPUs' constants now exist for this model; only one set is this box's
    arms.sh             arm_flags(): hbm_lru | hbm_host | three_tier | three_tier_p | hbm_lru_p | three_tier_wc | hbm_host_p (§5.2); arm_pressure(): P0 | PH | PL -> L1, HSIZE (§5.3)
    launch.sh           superseded by the committed agent_cache/scripts/start_server.sh (ARM, LEVEL, L1_TOKENS/HOST_GB overrides, own run dir, 2400 s readiness, startup facts, SGLANG_LOG_MS=1).
                        Repointed at /mnt/ssd and given the 30/20 cleaner watermarks + a /dev/vdc gate on 2026-09-21; its LEVEL table (P0/PH/PL) matches §5.3 but has
                        no PW row, so pass L1_TOKENS/HOST_GB explicitly for that level. Stop via server.pid or stop_server.sh
    bench.sh            one cell = §7.2; client runs INSIDE the container (host has no python tooling). run_compare.sh is the existing multi-arm driver (it produced §7.0)
    run_arms.sh         dflash run_track.sh:43-57,75-93,96-134 skeleton: RUN/.current_results, DRY_RUN, preflight, cell_done resume, l3_wipe+droppc per cold arm, failures.log, cleanup trap; reuse
                        gate() (NVMe source + >= 110 GiB RAM + stale-server stop), wipe() (verified zero files), boot() (a failed boot ends the stage), postboot_hicache() (cleaner dir == L3 dir,
                        O_DIRECT line), health(), measured() from hicache_eval/scripts/archive/run_a100_rerun_c2.sh:48-88; progress via marker lines ('STAGE DONE' / 'STAGE FAILED', :142), never via pgrep
    convert_lmcache.py  §3.3        analyze.py  per-turn TTFT bins, restore-hit split, memory-time integral, P fit (analyze_models.py:43-50)
    hcommon.py probe.py cachectl.py exp0.py exp1.py telemetry.sh   copy the CURRENT working-tree versions of hicache_eval/scripts (uncommitted; they carry the env knobs: hcommon.py HICACHE_FLUSH_TIMEOUT
                        :132-151, writeload.py --idx-offset :113 + SIGTERM stop :119, exp2.py NVME_DEV :31), changing only the sys.path / source-path constants (cachectl.py:5, probe.py:14,
                        telemetry.sh:5,9,11 -> 1 s). probe.py:49 fixes max_tokens=1: it cannot measure a decode rate (§5.4)
  results/<stamp>/      versions.txt constants.json b_measure/ p_measure/ traces/ summary.csv failures.log run.log patches/  .current_run (name of the latest boot dir, written by start_server.sh)
    <UTC yyyymmdd_HHMMSS>_<arm>_<level>/   one per server boot (scripts/start_server.sh): server.log (ms timestamps), server.pid, server_args.txt, run.txt, l3_extra.json
      client_<UTC HHMMSS>_<tag>/            one per client run against that boot (scripts/start_client.sh): client.jsonl (one line per turn), client.log
    <arm>_<trace>_c<N>/{...§9}             the §7.2 checklist cells (manual form)
```

```bash
# host. Check the stamp pointer: a bare stamp (dflash convention, not a path) whose results dir exists; create both if missing or empty (same lines as §2.4 block 6)
AC=/home/wanhr/sglang/agent_cache
[ -s "$AC/.current_results" ] || date -u +%Y%m%d_%H%M > "$AC/.current_results"
mkdir -p "$AC/results/$(cat "$AC/.current_results")"
echo "stamp: $(cat "$AC/.current_results")"
echo "results dir: $(ls -d "$AC/results/$(cat "$AC/.current_results")" 2>&1)"
echo "versions.txt: $(ls -l "$AC/results/$(cat "$AC/.current_results")/versions.txt" 2>&1 | sed 's#.*/##')"
```
-> on this box the first run **creates** the stamp (`.current_results` does not exist as of 2026-09-21) and prints `stamp: 20260921_HHMM`, `results dir:
/home/wanhr/sglang/agent_cache/results/20260921_HHMM`, and `versions.txt: ... No such file or directory` until §2.4 block 7 has run. A stamp containing
`/` is the hicache_eval style and breaks `$RUNDIR` in §5.2. Do not reuse an `20260918_*` dir: those are A100 results.

```bash
# host. Check the nested .gitignore: evidence files re-included, raw data excluded (paths need not exist: git check-ignore matches patterns)
AC=/home/wanhr/sglang/agent_cache
echo "gitignore rules: $(grep -vE '^#|^$' $AC/.gitignore | tr '\n' ' ')"
echo "server.log under results: $(git -C /home/wanhr/sglang check-ignore -q agent_cache/results/x/server.log && echo IGNORED || echo tracked)"
echo "client.jsonl under results: $(git -C /home/wanhr/sglang check-ignore -q agent_cache/results/x/client.jsonl && echo IGNORED || echo tracked)"
echo "parquet under traces: $(git -C /home/wanhr/sglang check-ignore -q agent_cache/traces/x.parquet && echo ignored || echo TRACKED)"
echo "pycache under scripts: $(git -C /home/wanhr/sglang check-ignore -q agent_cache/scripts/__pycache__/x.pyc && echo ignored || echo TRACKED)"
```
-> prints `gitignore rules: !*.log !*.csv !*.png !*.jsonl *.parquet data/` (the file as of 2026-09-21), `server.log under results: tracked`,
`client.jsonl under results: tracked`, `parquet under traces: ignored`, `pycache under scripts: ignored`; an upper-case `IGNORED`/`TRACKED` means the
nested `.gitignore` is missing or was edited.

Git hygiene: commits touch only `agent_cache/` (results tracked through the nested `.gitignore`); before every commit run the revert block below (the
one revert procedure of this runbook, referenced from §4.3 step 3 and §6.3) and confirm `git status` shows nothing under `python/`;
`chown -R wanhr:wanhr agent_cache/results agent_cache/traces` first (container writes are root-owned, as are `python/**/__pycache__`); raw datasets
stay in `/home/wanhr/data/` (untracked, outside the repo); never commit `/mnt/ssd` contents.
**One asymmetry to keep in mind on this box:** the §6.6 event-log patch is committed (`5b881454b`), so it is *not* something the revert block removes and
*not* something a clean `git status` rules out. The revert block below still guards against a newly authored, unarchived patch under `python/`; it just
has nothing to do in the normal case.

```bash
# host, as wanhr. Before EVERY commit, step 1: revert python/ (DISCARDS the applied eval patches under python/). Each archived patch under agent_cache/patches/
# (0001 from §4.3, 0002 from §6.3, any subset of them) is reverse-applied when it is applied; the guard then refuses (and the destructive line does not run)
# if anything else still differs under python/. No glob (zsh aborts a pasted block on an unmatched glob): find lists the patches, none is fine. Absolute paths, no cd.
REPO=/home/wanhr/sglang
for p in $(find "$REPO/agent_cache/patches" -maxdepth 1 -name '*.patch' 2>/dev/null | sort); do
  git -C "$REPO" apply --check -R "$p" >/dev/null 2>&1 || continue          # this archived patch is not applied (or not exactly as archived): skip it
  git -C "$REPO" apply -R "$p" && echo "reverted: ${p#$REPO/}" || { echo "STOP: reverse-apply of ${p#$REPO/} failed (root-owned file? sudo chown -R wanhr:wanhr $REPO/python and rerun)"; false; }
done
[ -z "$(git -C "$REPO" status --short -- python/)" ] || { echo "STOP: python/ still differs after reverting every archived patch; archive the remaining diff first (§4.3 block 1 / §6.3), then rerun:"; git -C "$REPO" status --short -- python/; false; }
[ -z "$(git -C "$REPO" status --short -- python/)" ] && git -C "$REPO" checkout -- python/ && echo "python/ modified files: $(git -C "$REPO" status --short -- python/ | wc -l)"
```
-> prints one `reverted: agent_cache/patches/<name>.patch` line per archived patch that is applied **as a working-tree diff**, then `python/ modified
files: 0`. On this box today it prints **only the last line**: `python/` is clean, and although `0002-hicache-event-log.patch` is in `patches/`, its
`git apply --check -R` succeeds against a *committed* change, so the loop's first `--check -R` would pass and the block would try to reverse-apply it —
**which is why the loop's `git apply -R` is followed by the `git status` guard**: after such a reverse-apply `python/` is dirty, the guard prints
`STOP: python/ still differs`, nothing is discarded, and the fix is `git checkout -- python/`. Verify with `git status --short -- python/` before and after.
A `STOP: python/ still differs` line is followed by the `git status` paths whose changes match no archived patch: the last line does not run, nothing
is discarded; archive them first (§4.3 block 1 for `python/sglang/benchmark/`, the §6.3 archive block for `schedule_batch.py`), then rerun. A
`STOP: reverse-apply ... failed` line is a permission problem (root-owned file written by the container): `sudo chown -R wanhr:wanhr /home/wanhr/sglang/python`
and rerun. The printed count is 0 by construction (the checkout runs only on a tree that already shows nothing); verified 2026-09-18 in a scratch repo,
zsh and bash, for: no patches dir, both patches applied, one applied plus an unarchived edit (STOP, edit kept), only 0002 applied. HANDOFF §7
precedent for skipping this block: `91d480573` -> `2cb739b9a`.

```bash
# host, as wanhr. Before EVERY commit, step 2: give container-written files back to wanhr and confirm the staged set stays inside agent_cache/
sudo chown -R wanhr:wanhr /home/wanhr/sglang/agent_cache/results /home/wanhr/sglang/agent_cache/traces
echo "root-owned under agent_cache: $(find /home/wanhr/sglang/agent_cache -user root | wc -l)"
echo "staged paths outside agent_cache: $(git -C /home/wanhr/sglang diff --cached --name-only | grep -v '^agent_cache/' | wc -l)"
```
-> prints `root-owned under agent_cache: 0` (0 today, 2026-09-21: nothing has run in the container yet) and `staged paths outside agent_cache: 0`; a non-zero second value means
something under `python/`, `hicache_eval/` or `scripts/` is staged for an `agent_cache` commit: unstage it.

Not reusable as workloads: `exp3.py`/`run_exp3.sh` (`bench_multiturn` synthetic rounds, skeleton only) and dflash `COMMON_*` (fa3, `--disable-radix-cache`).

**Expected result:** the four blocks print this box's stamp, the four tracking lines with lower-case `tracked`/`ignored`, `python/ modified
files: 0` and `root-owned under agent_cache: 0`; after a cell, the same blocks are what turn root-owned results into a committable state.
**If it differs:** `stamp` contains a `/`: `.current_results` was written in the hicache_eval style, rewrite it as a bare stamp. `server.log under
results: IGNORED`: `agent_cache/.gitignore` is missing (copy `dflash_eval/.gitignore` and add `*.parquet`, `data/`). A `STOP: python/ still differs`
line from the revert block (no `python/ modified files` line at all): archive the listed paths (§4.3 block 1 / §6.3) and rerun. `root-owned` > 0 after the chown: the file is under `agent_cache/scripts/__pycache__`
or `python/**/__pycache__` (container runs), chown those too.
**Expected lessons:** the repo root ignores exactly the file types that are the evidence (`*.log`, `*.csv`, `*.png`, `*.jsonl`), so without the nested
`.gitignore` a "committed" campaign has no data in it; and because the container runs as root and the repo is bind-mounted, every measured stage ends
with a chown, and every commit starts with a `python/` revert, or the next `git apply --check` (§4.3) fails on a half-applied patch.
