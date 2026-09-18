# Deviations from campaigns 1 and 3 (Qwen3-8B, nixl) in this rerun

Rerun on 2026-09-17 on GCP `a2-ultragpu-1g`. Reference runs: `20260907_203035/nixl` (Exp 0/1, backend A/B)
and `20260908_nixl_exp234` (Exp 2/3/4). Same image tag, same commit under `python/` (no diff against
`f3ccd1c0e`), same server flags, same client parameters, same probe seeds. See `versions.txt`.

| # | what differs | why | effect on the comparison |
|---|---|---|---|
| D1 | GPU A100-SXM4-80GB (SM80) instead of H100 PCIe; 12 vCPU / 167 GB RAM instead of 26 / 221; PCIe Gen4 | the box | this is the comparison: recompute and L1 move with the GPU (and the smaller host); L2 moves with the host-to-device path (PCIe Gen4 x16 here), but is confounded with the concurrent write-through of D5 |
| D2 | L3 on a local NVMe (`/mnt/nvme/hicache_l3`, ext4, noatime, O_DIRECT through nixl POSIX) instead of a virtio disk in the container overlay | the box | this SSD saturates at about 0.68 GiB/s read and 0.38 GiB/s write (iostat in `exp1_nixl/` and `exp1_nixl_l2/`: 695 to 704 MiB/s and 391 MiB/s at 97 to 99 % util); an fio run by the operator agreed (about 0.7 / 0.4) but was not saved. The old disk delivered about 3 GiB/s through nixl |
| D3 | attention backend `flashinfer` instead of `fa3` | SM80 default. Pinning `fa3` fails at decode CUDA-graph capture: `scheduler_metadata must have shape (metadata_size)` (`preflight/boot_fa3/server.log`) | recompute and L1 differences are GPU plus attention kernel and were not separated in this run. fa3 captured its prefill graphs on SM80 and failed only in decode CUDA-graph capture; a prefill-only fa3 configuration was not tried |
| D4 | Exp 1 main pass ran `--tiers recompute,L1,L3`; L2 rows come from the dedicated pass | the old combined run's L2 phase produced 0 rows and 63 discards, and its fillers would write about 490 GB here, past the nixl cleaner's 80 % watermark on a 369 GB filesystem | none on emitted rows: the old L2 rows also came from the dedicated pass |
| D5 | the dedicated L2 pass ran against a cold store while about 92 GB (86 GiB) of write-through was draining | the old driver wiped `/var/hicache_nixl` while the server used `/var/hicache_l3`, so the old pass ran against a warm store and wrote nothing | L2 probes here share PCIe and CPU with the L3 backup stream; `exp1_nixl_l2/iostat.log` and `pcie.log` document it. Those two logs carry an idle tail after about 18:12Z (L2 probes ran 18:08:56 to 18:09:00Z, the backup drain ended 18:11:44Z; a wait loop matched its own command line) |
| D6 | `HICACHE_FLUSH_TIMEOUT=1800` for `hcommon.flush_cache` (old: fixed 120 s) | Exp 0 step 5 flushes behind 83 GB of fillers; at 0.4 GiB/s the drain outlasts 120 s | none on timings; the wait is outside every probe |
| D7 | Exp 0 `step2_backup.l3_files_after` is 128 here and 0 in the old run | same dir mismatch as D5: the old harness counted files in a dir the server never wrote to | attribution is otherwise identical (4032 tokens on the intended tier at every step) |
| D8 | Exp 2, Exp 3 and Exp 4 were not run | cancelled on operator request at about 18:08Z, before `exp2_base` started, in favour of Exp 0/1 on the 32B and 70B | `COMPARISON.md` has no Exp 2/3/4 sections. The harness changes prepared for them stay in `scripts/` (`run_a100_rerun.sh`: chunked rate sweep, non-replaying writeload, per-chunk metrics) |
| D9 | the first long prefill in a fresh container took 9.0 s (`preflight/boot_default/probes.jsonl`) | one-time warm-up; the container's JIT caches were populated around that time, but the server log records no cause | absorbed by the unmeasured preflight boots; do not recreate the container between measured stages |

Backend A/B note (both boxes): the value `backend_ab.py` labels `L1` is an L3 hit without `drop_caches`,
because `wait_until_flushable` flushes L1+L2 before that probe (`L1_attr` is `[0, 0, 4032]` in old and new).
