#!/usr/bin/env bash
# Build the environment of stock vLLM: a uv virtual environment with the pinned release from PyPI.
#   host: bash /home/wanhr/sglang/agent_cache/baselines/vllm_vanilla/setup.sh
# Touches: creates .venv here (about 9 GB, git-ignored) and rewrites requirements.lock from what was installed.
# The pin is vLLM 0.33.0, the newest release LMCache 0.5.7.post1 runs with (vllm_lmcache/README.md), so the two
# vLLM baselines differ by LMCache alone.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM_VERSION=${VLLM_VERSION:-0.33.0}
[ -x "$HERE/.venv/bin/python" ] || uv venv --python 3.12 "$HERE/.venv"
uv pip install --python "$HERE/.venv/bin/python" "vllm==$VLLM_VERSION"
uv pip freeze --python "$HERE/.venv/bin/python" > "$HERE/requirements.lock"
"$HERE/.venv/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__, "cuda available", torch.cuda.is_available())'
