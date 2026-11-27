"""Cost of the file calls R4 adds, on the filesystem this folder is on (ext4 on this host; the SSD volume is
ext4 too, and is not touched here). 4,000 small files in 256 bucket folders, named like page files; warm inodes.
Usage: nice -n 10 python3 -I -B utime_bench.py"""
import hashlib
import os
import shutil
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "bench_utime")
N = 4000


def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    paths = []
    for i in range(N):
        sub = os.path.join(ROOT, f"{i % 256:02x}")
        os.makedirs(sub, exist_ok=True)
        p = os.path.join(sub, f"{i:064x}_Qwen-Qwen3-30B-A3B-Instruct-2507_0_1_{'k' if i % 2 == 0 else 'v'}")
        with open(p, "wb") as f:
            f.write(b"x")
        paths.append(p)

    def bench(label, fn, reps=3):
        best = None
        for _ in range(reps):
            t0 = time.perf_counter()
            for p in paths:
                fn(p)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        print(f"{label}: {best / N * 1e6:.2f} us per call ({N} files, best of {reps})")

    t = time.time_ns()
    bench("os.utime(path)                     ", lambda p: os.utime(p))
    bench("os.utime(path, ns=(t, t))          ", lambda p: os.utime(p, ns=(t, t)))
    bench("os.utime(path, ns=(0, 0))          ", lambda p: os.utime(p, ns=(0, 0)))
    bench("os.stat(path)                      ", lambda p: os.stat(p))

    def missing(p):
        try:
            os.utime(p + ".gone", ns=(0, 0))
        except OSError:
            pass

    bench("os.utime on a missing file         ", missing)

    def route(p):
        k = os.path.basename(p)
        h = int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest(), "big")
        return os.path.join("/x", f"{h & 255:02x}", k)

    bench("path build (blake2b route and join)", route)
    os.utime(paths[0], ns=(0, 0))
    print("st_mtime after ns=(0, 0):", os.stat(paths[0]).st_mtime, "| the cleaner's fold max(0.0, st_mtime):",
          max(0.0, os.stat(paths[0]).st_mtime))
    os.utime(paths[2], ns=(4_999_997_000,) * 2)
    print("stamp kept to the nanosecond:", os.stat(paths[2]).st_mtime_ns == 4_999_997_000)
    shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
