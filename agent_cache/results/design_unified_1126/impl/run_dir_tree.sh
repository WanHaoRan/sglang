#!/bin/bash
# One pytest run per test file of test/registered/unit/mem_cache (bench files left out) against one python tree,
# CPU only: the GPU is hidden and pytest is niced; four files at a time.
# Usage: run_dir_tree.sh <python tree, container path> <output file>. Writes "<file>: <pytest summary line>" per file.
pypath="$1"; out="$2"
tmp=$(mktemp -d)
run_one() {
  name="$1"; pypath="$2"; tmp="$3"
  line=$(docker exec -e PYTHONPATH="${pypath}" -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
    "cd /sgl-workspace/sglang && nice -n 10 python3 -m pytest -q -p no:cacheprovider test/registered/unit/mem_cache/${name} 2>&1 | tail -1")
  echo "${name}: ${line}" > "${tmp}/${name}.line"
}
export -f run_one
ls /home/wanhr/sglang/test/registered/unit/mem_cache/ | grep '^test_.*\.py$' | grep -v '_bench\.py$' \
  | xargs -P 4 -I{} bash -c 'run_one "$@"' _ {} "${pypath}" "${tmp}"
cat "${tmp}"/*.line | sort > "${out}"
rm -rf "${tmp}"
