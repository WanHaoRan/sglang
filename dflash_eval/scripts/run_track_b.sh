#!/usr/bin/env bash
# Track B (Qwen3.8-27B-FP8): DFlash2 vs DSpark vs MTP, sequentially. Thin wrapper over run_track.sh.
#   scripts/run_track_b.sh                     # B0 B1-4 B1-8 B2 B3 x {gsm8k,humaneval} x {c1,c8}, greedy
#   SAMPLING=xhigh scripts/run_track_b.sh      # the DFlash README / model-card sampling (temp 1, top-p 0.95, top-k 20, reasoning xhigh)
#   ARMS="B0 B3" CELLS="gsm8k 1" scripts/run_track_b.sh
#   DRY_RUN=1 scripts/run_track_b.sh
# All knobs are documented in run_track.sh.
TRACK=B exec "$(dirname "${BASH_SOURCE[0]}")/run_track.sh" "$@"
