#!/usr/bin/env bash
# Save the Prometheus metrics of a running system into its run dir, before run_baselines.sh stops it: the run-level
# shares of prompt tokens served by each tier (vLLM: prefix_cache_hits and external_prefix_cache_hits; LMCache:
# lookup_hit_l1/l2_tokens). Host side.
#   host: bash save_metrics.sh <run dir> <name>=<url> [<name>=<url> ...]
set -uo pipefail
RUN=$1; shift
for pair in "$@"; do
  name=${pair%%=*} url=${pair#*=}
  curl -s --max-time 20 "$url" > "$RUN/metrics_end_$name.txt" && echo "saved $RUN/metrics_end_$name.txt ($(wc -l < "$RUN/metrics_end_$name.txt") lines)"
done
