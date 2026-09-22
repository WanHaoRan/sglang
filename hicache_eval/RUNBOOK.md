# hicache_eval runbook

Exp 0 / Exp 1: TTFT of a prefix-cache hit per tier vs prompt length, one campaign per box.
Assumes the box is already set up (container `sglang_hicache`, models in the HF cache, L3 disk mounted).
Background: `HANDOFF.md`. Current campaign: `results/20260921_h200_nebius_32b70b_fp8kv/README.md`.

Everything runs inside the container unless marked `# host`:

```bash
docker exec -it sglang_hicache bash
cd /sgl-workspace/sglang/hicache_eval/scripts
```

---

## 1. Port to a new box

Six values. Discover each, then write it into the two files that hold it.

### Find the GPU, its compute capability and the L3 device

```bash
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
findmnt -n -o SOURCE,FSTYPE,SIZE -T /mnt/ssd/hicache_l3
```

### Measure the L3 disk ceiling (steady state at 100 % util, NOT the dd aggregate)

```bash
# host. 4 streams saturate it; watch iostat, the dd total overstates by ~15 %
D=/mnt/ssd/bench; sudo mkdir -p $D && sudo chown $USER $D
iostat -x -d 2 8 vdc > /tmp/io.txt 2>&1 & IOP=$!
P=(); for i in 0 1 2 3; do dd if=/dev/zero of=$D/w$i bs=4M count=1536 oflag=direct 2>/dev/null & P+=($!); done
wait "${P[@]}"; wait $IOP          # wait on the dds by pid: a bare `wait` also waits on iostat
awk '/^vdc/{print $9" wkB/s  util "$NF}' /tmp/io.txt; sudo rm -rf $D
```

### Pick the cleaner watermarks: nixl cleans at a % OF THE FILESYSTEM, so a big disk never fires

```bash
# host. Aim the high mark at a stated budget (>= 2x the biggest campaign), not at 80 % of the disk.
# Shipped: L3_CLEANER_PCT=30,20 on a 2270 GiB disk = 681/454 GiB.
df -h --output=size /mnt/ssd | tail -1
python3 -c "fs=2270; b=680; p=round(b/fs*100); print(f'{b} GiB of {fs} GiB -> high={p}% low={p-10}%')"
```

### Edit the two config files

```bash
# scripts/env.sh:6,18      L3_DIR and the block device behind it
# scripts/run_h200.sh:41-48  RESULTS, L3_DIR, NVME_DEV, L3_CLEANER_PCT, ATTENTION_BACKEND, START_TIMEOUT_S
sed -n '4,20p' env.sh; sed -n '41,48p' run_h200.sh
```

| knob | where | how to choose |
|---|---|---|
| `L3_DIR` / `NVME_DEV` | `env.sh:6,18` + `run_h200.sh:42,43` | from `findmnt` above; `NVME_DEV` is the bare device name `iostat` takes |
| `RESULTS` | `run_h200.sh:41` | `results/<YYYYMMDD>_<gpu>_<host>_<models>/` |
| `L3_CLEANER_PCT` | `run_h200.sh:44` | from the `df` arithmetic above; `""` keeps the 80/70 default |
| `ATTENTION_BACKEND` | `run_h200.sh:46` | `triton` with `fp8_e5m2` KV on every GPU so far. SM80 defaults to `flashinfer`, SM90 to `fa3` which the fp8 rule rewrites to `triton` |
| `START_TIMEOUT_S` | `run_h200.sh:47` | 2400 covers a JIT-cold first boot; 900 only covers a warm one |

### Point the campaign at a new dir and freeze the old one

```bash
# host
cd /home/wanhr/sglang/hicache_eval
NEW=20260921_h200_nebius_32b70b_fp8kv
mkdir -p results/$NEW && echo "/sgl-workspace/sglang/hicache_eval/results/$NEW" > .current_results
grep -qxF <old-campaign-dirname> .frozen_results || echo <old-campaign-dirname> >> .frozen_results
```

### Add a row to the figure (one dict entry, keyed by the campaign dir)

```bash
# scripts/plot_tier_ttft.py:28,37-46 — H200/BOXES/BOX_ORDER. New box = new key at the front of BOX_ORDER.
sed -n '26,47p' plot_tier_ttft.py
```

### Check nothing still points at the old box

```bash
grep -rn '/mnt/nvme\|nvme0n1\|/var/hicache' env.sh models.sh start_server.sh run_h200.sh hcommon.py
```

---

## 2. Run

Sections 2-4 use `$RESULTS`, `$L3_DIR`, `$NVME_DEV`. The drivers set their own copies; for your shell:

```bash
source env.sh                                 # RESULTS (from ../.current_results), L3_DIR, NVME_DEV, BASE
echo "$RESULTS  $L3_DIR  /dev/$NVME_DEV"      # refuses if .current_results names a frozen campaign
```

### Preflight: does the checkpoint run here, what pool does it profile, which weight kernel

```bash
bash run_h200.sh preflight_32b
grep -aoE 'max_total_num_tokens=[0-9]+' $RESULTS/preflight/boot_32b/server.log | head -1
grep -aciE 'Weight-only FP8 .* Marlin' $RESULTS/preflight/boot_32b/server.log   # want 0 on SM90
```

### Full campaign: p_measure, then Exp 0 / Exp 1 / Exp 1-L2 per model (`all3` adds the 8B)

```bash
bash run_h200.sh all
```

### Watch it, and stop it

```bash
tail -f $RESULTS/run_h200.log                 # 'STAGE DONE' / 'STAGE FAILED' per stage
bash stop_server.sh                           # from a script file: pkill -f matches its own shell
```

### Resume: stages are independent, re-run only the ones that failed

```bash
grep 'STAGE ' $RESULTS/run_h200.log
bash run_h200.sh exp1_32b exp1_32b_l2
```

---

## 3. Report

### Old vs new tables, every statistic computed identically on both sides

```bash
OLD_C2=/sgl-workspace/sglang/hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv \
OLD_LABEL="A100 SXM4, local NVMe (c5)" NEW_LABEL="H200, attached SSD (c6)" \
  python3 compare_campaigns.py                # -> $RESULTS/COMPARISON.md + comparison.csv
```

### The figure: one row per box, one column per model; rows with no data are skipped

```bash
python3 plot_tier_ttft.py                     # -> $RESULTS/ttft_by_tier_3models.png
python3 plot_tier_ttft.py --mode dark
```

### Sanity: the comparison script must reproduce an old campaign's published numbers

```bash
OLD=/sgl-workspace/sglang/hicache_eval/results/20260917_a100_gcp_32b70b_fp8kv
RESULTS=$OLD COMPARE_OUT=/tmp/chk python3 compare_campaigns.py   # pure python, no env.sh, no frozen guard
diff /tmp/chk/COMPARISON.md $OLD/COMPARISON.md    # want no output (verified 2026-09-21)
```

### Hand-write `REPORT.md`, `DEVIATIONS.md`, `versions.txt` in the shape of the previous campaign

```bash
ls ../results/20260917_a100_gcp_32b70b_fp8kv/*.md
docker exec -i sglang_hicache pip list | grep -Ei '^sgl|flashinfer|^torch |nixl|^triton |transformers' > $RESULTS/versions.txt
```

---

## 4. Traps that cost a measurement before

`HANDOFF.md` §3 has all of them. The four that bite during a run:

### L3 was not actually cold — `rm -rf` silently fails past ~10k files

```bash
find $L3_DIR -mindepth 1 -delete && find $L3_DIR -type f | wc -l   # want 0
```

### L3 is not on the disk you think — a detached volume leaves an empty dir on the root disk

```bash
[ "$(findmnt -n -o SOURCE -T $L3_DIR)" = /dev/$NVME_DEV ] && echo ok || echo WRONG DEVICE
```

### A constant from another GPU — `b*P` is GPU-specific, and `models.sh` L1_TOKENS is H100's

```bash
grep -n 'L1_TOKENS' models.sh
grep -aoE 'max_total_num_tokens=[0-9]+' $RESULTS/exp1_32b/server.log | head -1   # this box's value
```

### A "recompute" control that was secretly an L3 hit — assert it hit no tier

```bash
python3 -c "
import pandas as pd,os
d=pd.read_csv(os.path.join(os.environ['RESULTS'],'exp1_32b','ttft_by_tier.csv'))
r=d[d.tier=='recompute']
print('recompute rows with a tier hit:', int((r[['cached_device','cached_host','cached_storage']].fillna(0).sum(axis=1)>0).sum()), '(want 0)')"
```
