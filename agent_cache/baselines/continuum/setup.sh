#!/usr/bin/env bash
# Build Continuum: the authors' fork at the pinned commit, installed editable on the kernels of the official vLLM
# 0.10.2 wheel (the fork changes Python files only), with LMCache 0.3.6, resolved as PyPI stood on 2025-09-16 so
# its dependencies are those of its time (today's resolution brings transformers 5.x, which vLLM 0.10.2 predates).
#   host: bash /home/wanhr/sglang/agent_cache/baselines/continuum/setup.sh
# Touches: src/vllm-continuum (clone, patched), wheels/ (the 417 MB vLLM wheel), .venv (about 8 GB), all git-ignored;
# rewrites requirements.lock. PATCH=0 leaves the fork unpatched (patches/: why each one exists).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=https://github.com/Hanchenli/vllm-continuum
COMMIT=316a58794a6ff86b216e579b74fd56ed0c5a911f      # 2026-03-19, the repository's head on 2026-11-21
WHEEL=vllm-0.10.2-cp38-abi3-manylinux1_x86_64.whl
WHEEL_SHA256=e0cba6110483d9bf                        # prefix of PyPI's digest, checked below
SRC=$HERE/src/vllm-continuum
[ -d "$SRC/.git" ] || git clone "$REPO" "$SRC"
git -C "$SRC" fetch -q origin "$COMMIT" && git -C "$SRC" checkout -q --force "$COMMIT"
if [ "${PATCH:-1}" = 1 ]; then
  for p in "$HERE"/patches/*.patch; do git -C "$SRC" apply "$p" && echo "applied $(basename "$p")"; done
fi
mkdir -p "$HERE/wheels"
if [ ! -s "$HERE/wheels/$WHEEL" ]; then
  URL=$(curl -s https://pypi.org/pypi/vllm/0.10.2/json | python3 -c "import json,sys; print([f['url'] for f in json.load(sys.stdin)['urls'] if f['filename']=='$WHEEL'][0])")
  curl -s -L -o "$HERE/wheels/$WHEEL" "$URL"
fi
[ "$(sha256sum "$HERE/wheels/$WHEEL" | cut -c1-16)" = "$WHEEL_SHA256" ] || { echo "STOP: $WHEEL does not match PyPI's digest"; exit 1; }
[ -x "$HERE/.venv/bin/python" ] || uv venv --python 3.12 "$HERE/.venv"
( cd "$SRC" && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$HERE/wheels/$WHEEL" SETUPTOOLS_SCM_PRETEND_VERSION=0.10.2 \
    uv pip install --python "$HERE/.venv/bin/python" --exclude-newer 2025-09-16T00:00:00Z -e . lmcache==0.3.6 )
uv pip check --python "$HERE/.venv/bin/python"
uv pip freeze --python "$HERE/.venv/bin/python" > "$HERE/requirements.lock"
"$HERE/.venv/bin/python" -c 'import torch, importlib.metadata as m; from lmcache import c_ops; from vllm.v1.core.estimate_with_func import FIXED_THRESHOLD_CONTINUUM as T; print("vllm", m.version("vllm"), "lmcache", m.version("lmcache"), "torch", torch.__version__, "pin threshold", T, "s")'
