#!/usr/bin/env bash
# Build TokenCake: the authors' fork at the pinned commit, installed as its README says (VLLM_USE_PRECOMPILED=1,
# uv, --torch-backend=auto) on the kernels of the official vLLM 0.22.0 wheel, which it is based on and whose compiled
# parts it leaves unchanged (csrc/, rust/, cmake/ equal v0.22.0; checked 2026-11-21). Resolved as PyPI stood on its
# commit day, plus the experiment tools' dependencies from tools/tokencake_experiments/README.md.
#   host: bash /home/wanhr/sglang/agent_cache/baselines/tokencake/setup.sh
# Touches: src/TokenCake (clone), wheels/ (the 249 MB vLLM wheel), .venv (about 8 GB), all git-ignored; rewrites
# requirements.lock.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=https://github.com/pku-lemonade/TokenCake
COMMIT=1901ebf31fdf63590148e61d5c2f65e9fb7ec010      # 2026-09-28, the repository's head on 2026-11-21
WHEEL=vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl
WHEEL_SHA256=c387a977e35795e8                        # prefix of PyPI's digest, checked below
SRC=$HERE/src/TokenCake
[ -d "$SRC/.git" ] || git clone --depth 1 "$REPO" "$SRC"
git -C "$SRC" fetch -q --depth 1 origin "$COMMIT" && git -C "$SRC" checkout -q --force "$COMMIT"
mkdir -p "$HERE/wheels"
if [ ! -s "$HERE/wheels/$WHEEL" ]; then
  URL=$(curl -s https://pypi.org/pypi/vllm/0.22.0/json | python3 -c "import json,sys; print([f['url'] for f in json.load(sys.stdin)['urls'] if f['filename']=='$WHEEL'][0])")
  curl -s -L -o "$HERE/wheels/$WHEEL" "$URL"
fi
[ "$(sha256sum "$HERE/wheels/$WHEEL" | cut -c1-16)" = "$WHEEL_SHA256" ] || { echo "STOP: $WHEEL does not match PyPI's digest"; exit 1; }
[ -x "$HERE/.venv/bin/python" ] || uv venv --python 3.12 "$HERE/.venv"
( cd "$SRC" && VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$HERE/wheels/$WHEEL" SETUPTOOLS_SCM_PRETEND_VERSION=0.22.0 \
    uv pip install --python "$HERE/.venv/bin/python" --exclude-newer 2026-09-29T00:00:00Z -e . --torch-backend=auto )
uv pip install --python "$HERE/.venv/bin/python" --exclude-newer 2026-09-29T00:00:00Z httpx aiohttp psutil prometheus-client pydantic numpy matplotlib
uv pip check --python "$HERE/.venv/bin/python"
uv pip freeze --python "$HERE/.venv/bin/python" > "$HERE/requirements.lock"
"$HERE/.venv/bin/python" -c 'import torch, vllm, vllm._C; from vllm.tokencake.config import TokenCakeConfig; print("vllm", vllm.__version__, "torch", torch.__version__, "tokencake ok")'
