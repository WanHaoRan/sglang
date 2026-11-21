#!/usr/bin/env bash
# Build the environment of vLLM + LMCache: a uv virtual environment with the pinned releases from PyPI, resolved
# together (LMCache's quickstart: `uv pip install lmcache vllm`).
#   host: bash /home/wanhr/sglang/agent_cache/baselines/vllm_lmcache/setup.sh
# Touches: creates .venv here (about 9 GB, git-ignored) and rewrites requirements.lock from what was installed.
# Pins: LMCache 0.5.7.post1, the newest LMCache release (2026-11-12), and vLLM 0.33.0, the newest vLLM it runs with:
# with vLLM 0.34.0 neither connector loads (README.md). src/LMCache (optional, git-ignored) is the tag's source.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VLLM_VERSION=${VLLM_VERSION:-0.33.0}
LMCACHE_VERSION=${LMCACHE_VERSION:-0.5.7.post1}
[ -x "$HERE/.venv/bin/python" ] || uv venv --python 3.12 "$HERE/.venv"
uv pip install --python "$HERE/.venv/bin/python" "vllm==$VLLM_VERSION" "lmcache==$LMCACHE_VERSION"
uv pip freeze --python "$HERE/.venv/bin/python" > "$HERE/requirements.lock"
uv pip check --python "$HERE/.venv/bin/python"
"$HERE/.venv/bin/python" -c 'import vllm, lmcache, torch; print("vllm", vllm.__version__, "lmcache", lmcache.__version__, "torch", torch.__version__, "cuda available", torch.cuda.is_available())'
