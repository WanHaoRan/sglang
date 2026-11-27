"""Break one line of the engine, run the test meant to catch it, restore the line.

Usage: python3 mutate.py [tree] [name prefix ...]
  tree: the engine copy to mutate (default: agent_cache/frozen/impl_unified_B), as the container sees it
        under /sgl-workspace/sglang. CPU only: the GPU is hidden and pytest is niced.
  name prefix: run only the mutations whose name starts with one of these (default: all).
A mutation that prints SURVIVED is a hole in the tests; BAD MUTATION means its text is no longer in the file.
"""

import subprocess
import sys

from mutations import MUTATIONS

HOST_ROOT = "/home/wanhr/sglang"
CONTAINER_ROOT = "/sgl-workspace/sglang"


def run(tree: str, name: str) -> str:
    file, old, new, test, k = MUTATIONS[name]
    path = f"{HOST_ROOT}/{tree}/{file}"
    original = open(path).read()
    if original.count(old) != 1:
        return f"{name}: BAD MUTATION (its text occurs {original.count(old)} times)"
    command = (
        f"cd {CONTAINER_ROOT}/{tree} && nice -n 10 python3 -m pytest -q -p no:cacheprovider "
        f"test/registered/unit/mem_cache/{test}.py -k '{k}' 2>&1 | tail -1"
    )
    try:
        with open(path, "w") as out:
            out.write(original.replace(old, new))
        result = subprocess.run(
            ["docker", "exec", "-e", f"PYTHONPATH={CONTAINER_ROOT}/{tree}/python",
             "-e", "CUDA_VISIBLE_DEVICES=9", "-e", "PYTHONDONTWRITEBYTECODE=1", "sglang_hicache",
             "bash", "-c", command],
            capture_output=True, text=True,
        ).stdout.strip()
    finally:
        with open(path, "w") as out:
            out.write(original)
    verdict = "killed" if "failed" in result or "error" in result else "SURVIVED"
    return f"{name}: {verdict}  [{result}]"


if __name__ == "__main__":
    args = sys.argv[1:]
    tree = args.pop(0) if args and "/" in args[0] else "agent_cache/frozen/impl_unified_B"
    for name in MUTATIONS:
        if not args or name.startswith(tuple(args)):
            print(run(tree, name), flush=True)
