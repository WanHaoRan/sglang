#!/bin/zsh
# Every run behind CALIBRATION.md section 7 (about 6 minutes, at most 4 processes at nice 19), then the table.
# usage: zsh rerun_calibration.sh [out folder] [--force]
# Finished runs under the out folder are kept and not repeated; --force runs them again and overwrites their files.
cd "$(dirname "$0")"
OUT=${1:-calibration_runs}
FORCE=(${2:+$2})
PY=(/home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B)
export CUDA_VISIBLE_DEVICES=
ARMS=(hicache jit_p50 jit_exact ec_full ec_exact)
mkdir -p $OUT/logs
nice -n 19 $PY kvsim_compare.py $FORCE --arms $ARMS --seeds 0-4 --set "" --set fixes=off --out $OUT/calib > $OUT/logs/calib.out 2> $OUT/logs/calib.err
nice -n 19 $PY kvsim_compare.py --arms ec_full jit_p50 jit_exact ec_exact hicache --seeds 0-4 --set "" --set fixes=off --out $OUT/calib > $OUT/logs/calib_vs_ec_full.out 2> /dev/null
nice -n 19 $PY kvsim_compare.py $FORCE --arms $ARMS --seeds 0-4 --set "backend=triton,fixes=off" --out $OUT/triton > $OUT/logs/triton.out 2> $OUT/logs/triton.err
nice -n 19 $PY kvsim_compare.py $FORCE --arms $ARMS --seeds 0 --set "noise=0.005,noise_seed=1" --set "noise=0.005,noise_seed=2" \
  --set "noise=0.005,noise_seed=3" --set "noise=0.005,noise_seed=4" --set "noise=0.005,noise_seed=5" \
  --out $OUT/noise > $OUT/logs/noise_on.out 2> $OUT/logs/noise_on.err
nice -n 19 $PY kvsim_compare.py $FORCE --arms $ARMS --seeds 0 --set "fixes=off,noise=0.005,noise_seed=1" --set "fixes=off,noise=0.005,noise_seed=2" \
  --set "fixes=off,noise=0.005,noise_seed=3" --set "fixes=off,noise=0.005,noise_seed=4" --set "fixes=off,noise=0.005,noise_seed=5" \
  --out $OUT/noise > $OUT/logs/noise_off.out 2> $OUT/logs/noise_off.err
nice -n 19 $PY kvsim_compare.py $FORCE --arms ec_full hicache --seeds 0 --set gpu_time_scale=0.98 --set gpu_time_scale=1.0 --set gpu_time_scale=1.02 --out $OUT/sens > $OUT/logs/sens.out 2> /dev/null
nice -n 19 $PY kvsim_compare.py $FORCE --arms ec_full "ec_full@mirror=off" "ec_full@mirror=off+d2h_s_per_token=0" "unified@F3" "unified@F4" "unified@PR1" --seeds 0 --out $OUT/notes > $OUT/logs/notes.out 2> /dev/null
nice -n 19 $PY calib_measured.py > $OUT/logs/measured.out
nice -n 19 $PY calib_structure.py $OUT > $OUT/logs/structure.out
nice -n 19 $PY calib_table.py $OUT > /dev/null
echo done
