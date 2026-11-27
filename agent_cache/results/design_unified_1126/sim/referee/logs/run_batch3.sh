#!/bin/zsh
# Referee batch 3: fixes-off replicas of the patched copy, the D2 ablation, hicache against GPU speed (<= 4 processes).
cd "$(dirname "$0")"
PY=(/home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B)
export CUDA_VISIBLE_DEVICES=
ARMS=(ec_full hicache jit_p50 jit_exact ec_exact)
HINTED=(ec_full jit_p50 jit_exact ec_exact)
NOISE=(--set fixes=off,noise=0.005,noise_seed=1 --set fixes=off,noise=0.005,noise_seed=2 --set fixes=off,noise=0.005,noise_seed=3 --set fixes=off,noise=0.005,noise_seed=4 --set fixes=off,noise=0.005,noise_seed=5)
nice -n 19 $PY ref_run.py --sim kvsim_patched --arms $ARMS --seeds 0 $NOISE --out runs/noise_patched --procs 4 > batch3_noise_patched.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim_only_d2 --arms $HINTED --seeds 0 --set fixes=off,noise=0.005,noise_seed=1 --set fixes=off,noise=0.005,noise_seed=2 --set fixes=off,noise=0.005,noise_seed=3 --out runs/ablate_only_d2 --procs 4 > batch3_only_d2.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim_no_d2 --arms $HINTED --seeds 0 --set fixes=off,noise=0.005,noise_seed=1 --set fixes=off,noise=0.005,noise_seed=2 --set fixes=off,noise=0.005,noise_seed=3 --out runs/ablate_no_d2 --procs 4 > batch3_no_d2.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim --arms hicache --seeds 0 --set gpu_time_scale=0.97 --set gpu_time_scale=0.975 --set gpu_time_scale=0.98 --set gpu_time_scale=0.985 --set gpu_time_scale=0.9975 --set gpu_time_scale=1.0025 --set gpu_time_scale=1.015 --set gpu_time_scale=1.02 --set gpu_time_scale=1.03 --out runs/gpuscale --procs 4 > batch3_gpuscale.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim_patched --arms hicache ec_full --seeds 0 --set gpu_time_scale=0.99 --set gpu_time_scale=0.995 --set gpu_time_scale=1.005 --set gpu_time_scale=1.01 --out runs/gpuscale_patched --procs 4 > batch3_gpuscale_patched.log 2>&1
echo done > batch3.done
