#!/usr/bin/env bash
# run_baselines.sh calls this with the run dir before it stops the server.
bash "$(dirname "$0")/../common/save_metrics.sh" "$1" vllm=http://127.0.0.1:30000/metrics
