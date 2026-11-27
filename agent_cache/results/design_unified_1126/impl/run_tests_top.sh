#!/bin/bash
# The test files directly under test/registered/unit against one python tree (CPU, niced), one line per file.
pypath="$1"; out="$2"; : > "$out"
for f in $(cd /home/wanhr/sglang && ls test/registered/unit/test_*.py); do
  line=$(docker exec -e PYTHONPATH="${pypath}" -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
    "cd /sgl-workspace/sglang && nice -n 12 timeout 900 python3 -m pytest -q -p no:cacheprovider ${f} 2>&1 | tail -1")
  echo "${f#test/registered/unit/}: ${line}" >> "$out"
done
