"""Review measurement (threads lens): CPU the touch thread spends per file, and how
long a busy touch thread delays a Python loop on another thread (the scheduler's GIL).
Temp dirs only.

Throwaway: copied into the copy's test folder under this name, run, deleted.
"""

import hashlib
import os
import time
import unittest

import test_hicache_nixl_storage as nixl_tests


def spin(seconds: float) -> int:
    """Pure-Python work, as a scheduler pass is; returns the iterations done."""
    done = 0
    stop = time.perf_counter() + seconds
    while time.perf_counter() < stop:
        done += 1
    return done


class ReviewGil(nixl_tests.TestNixlTouch):
    def test_cpu_per_file_and_gil_share(self):
        backend = self._backend(touch_interval_s=300)
        backend.is_zero_copy = True
        pages = 2000
        keys = [hashlib.sha256(str(i).encode()).hexdigest() for i in range(pages)]
        started = time.perf_counter()
        page_files = [backend._page_file_paths(key) for key in keys]
        paths_us = (time.perf_counter() - started) / (2 * pages) * 1e6
        for paths in page_files:
            for path in paths:
                open(path, "w").close()
        started = time.perf_counter()
        counts = backend.file_manager.stamp_pages(page_files, newest_ns=time.time_ns())
        stamp_us = (time.perf_counter() - started) / (2 * pages) * 1e6
        self.assertEqual(counts, (2 * pages, 0))
        print(
            "\nper file: path building %.2f us, utime %.2f us; a %d-page chain "
            "(two files a page) takes %.1f ms on the touch thread"
            % (paths_us, stamp_us, pages, (paths_us + stamp_us) * 2 * pages / 1000)
        )

        alone = spin(0.5)
        # Keep the touch thread busy for the whole second measurement.
        for _ in range(60):
            backend.touch_pages(keys, newest_ns=time.time_ns())
        beside = spin(0.5)
        backend.close()
        print(
            "main-thread loop iterations in 0.5 s: %d alone, %d beside a busy touch "
            "thread (%.0f %% kept)" % (alone, beside, 100.0 * beside / alone)
        )


if __name__ == "__main__":
    unittest.main()
