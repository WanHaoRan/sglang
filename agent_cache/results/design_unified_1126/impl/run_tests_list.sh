#!/bin/bash
# One pytest run per test file named in a list (paths relative to the repo root) against one python tree, CPU only:
# the GPU is hidden and pytest is niced; three files at a time.
# Usage: run_tests_list.sh <python tree, container path> <list file> <output file>
pypath="$1"; list="$2"; out="$3"
tmp=$(mktemp -d)
run_one() {
  rel="$1"; pypath="$2"; tmp="$3"
  line=$(docker exec -e PYTHONPATH="${pypath}" -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
    "cd /sgl-workspace/sglang && nice -n 12 timeout 900 python3 -m pytest -q -p no:cacheprovider ${rel} 2>&1 | tail -1")
  echo "${rel#test/registered/unit/}: ${line}" > "${tmp}/$(echo "${rel}" | tr '/' '_').line"
}
export -f run_one
xargs -P 3 -I{} bash -c 'run_one "$@"' _ {} "${pypath}" "${tmp}" < "${list}"
cat "${tmp}"/*.line | sort > "${out}"
rm -rf "${tmp}"
