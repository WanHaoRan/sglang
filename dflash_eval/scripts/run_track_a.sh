#!/usr/bin/env bash
# Track A (Qwen3-8B): EAGLE3 vs DFlash vs DSpark, sequentially. Thin wrapper over run_track.sh.
#   scripts/run_track_a.sh                     # A0 A1 A2 A3 x {gsm8k,humaneval} x {c1,c8}, greedy
#   SAMPLING=think scripts/run_track_a.sh      # thinking mode, temp 1 / top-p 0.95 / top-k 20
#   ARMS="A1-w8 A2-w8" scripts/run_track_a.sh  # matched-window arms
#   RUN=20260909_1855 scripts/run_track_a.sh   # resume into an existing results dir
#   DRY_RUN=1 scripts/run_track_a.sh
# All knobs are documented in run_track.sh.
TRACK=A exec "$(dirname "${BASH_SOURCE[0]}")/run_track.sh" "$@"
