#!/bin/zsh
# Referee batch 2: the patched copy's anchors, noise replicas, GPU speed, Triton, patched invariants (<= 4 processes).
cd "$(dirname "$0")"
PY=(/home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B)
export CUDA_VISIBLE_DEVICES=
ARMS=(ec_full hicache jit_p50 jit_exact ec_exact)
nice -n 19 $PY ref_run.py --sim kvsim_patched --arms $ARMS --seeds 0,5,6,7 --set "" --set fixes=off --out runs/anchors_patched --procs 4 > batch2_patched.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim --arms $ARMS --seeds 0 --set noise=0.005,noise_seed=11 --set noise=0.005,noise_seed=12 --set noise=0.005,noise_seed=13 --set noise=0.005,noise_seed=14 --set noise=0.005,noise_seed=15 --set noise=0.005,noise_seed=16 --out runs/noise --procs 4 > batch2_noise.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim --arms $ARMS --seeds 0 --set gpu_time_scale=0.995 --set gpu_time_scale=1.005 --set gpu_time_scale=0.99 --set gpu_time_scale=1.01 --out runs/gpuscale --procs 4 > batch2_gpuscale.log 2>&1
nice -n 19 $PY ref_run.py --sim kvsim --arms $ARMS --seeds 0 --set backend=triton,fixes=off --out runs/triton --procs 4 > batch2_triton.log 2>&1
(nice -n 19 $PY ref_invariants.py --sim kvsim_patched --arms ec_full,hicache,jit_p50 --seeds 0 --fixes on --mirror off --out invp_mirroroff.json > invp_mirroroff.out 2>&1 &
 nice -n 19 $PY ref_invariants.py --sim kvsim_patched --arms hicache,jit_p50,ec_exact --seeds 5 --fixes on,off --out invp_seed5.json > invp_seed5.out 2>&1 &
 nice -n 19 $PY ref_invariants.py --sim kvsim_patched --arms jit_exact,ec_exact,nohint --seeds 0 --fixes on,off --out invp_seed0b.json > invp_seed0b.out 2>&1 &
 nice -n 19 $PY ref_invariants.py --sim kvsim_patched --arms unified --seeds 0 --fixes on --switches all --out invp_unified_all.json > invp_unified_all.out 2>&1 &
 wait)
echo done > batch2.done
