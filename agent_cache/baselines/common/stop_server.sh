#!/usr/bin/env bash
# Stop the baseline server of the current run (agent_cache/results/.current_run) and wait until the port and the GPU
# are free. Host side; a current run booted by our SGLang scripts in the container is left to scripts/stop_server.sh.
#   host: bash /home/wanhr/sglang/agent_cache/baselines/common/stop_server.sh
set -uo pipefail
. "$(cd "$(dirname "$0")" && pwd)/lib.sh"
bl_stop
