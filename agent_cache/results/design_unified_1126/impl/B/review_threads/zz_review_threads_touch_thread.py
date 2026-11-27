"""Review checks (threads lens) on the real HiCacheNixl touch thread, in temp dirs.

Throwaway: copied into the copy's test folder under this name, run, deleted.
"""

import gc
import os
import threading
import time
import unittest
import weakref

import test_hicache_nixl_storage as nixl_tests

LOGGER = "sglang.srt.mem_cache.storage.nixl.hicache_nixl"
TOUCH_THREAD = "hicache-l3-touch"


def touch_threads():
    return [t for t in threading.enumerate() if t.name == TOUCH_THREAD]


class ReviewTouchThread(nixl_tests.TestNixlTouch):
    def test_a_failing_job_is_logged_and_the_next_job_still_runs(self):
        backend = self._backend(retire_ended_pages=True)
        self._write_pages(backend, ["a", "b"], zero_copy=True)
        real = backend.file_manager.stamp_pages
        seen = []

        def flaky(page_files, *, newest_ns):
            seen.append(page_files)
            if len(seen) == 1:
                raise RuntimeError("boom")
            return real(page_files, newest_ns=newest_ns)

        backend.file_manager.stamp_pages = flaky
        with self.assertLogs(LOGGER, level="INFO") as logs:
            backend.touch_pages(["a"], newest_ns=None)
            backend.touch_pages(["b"], newest_ns=None)
            backend.close()
        a, b = self._page_mtimes_ns(["a", "b"])
        self.assertGreater(min(a), 0)
        self.assertEqual(b, [0, 0])
        self.assertTrue(any("NIXL L3 touch failed" in line for line in logs.output))
        self.assertEqual(touch_threads(), [])

    def test_close_runs_every_job_queued_before_it_and_later_touches_are_dropped(self):
        backend = self._backend(retire_ended_pages=True)
        keys = [f"p{i}" for i in range(4)]
        self._write_pages(backend, keys, zero_copy=True)
        for key in keys:
            backend.touch_pages([key], newest_ns=None)
        started = time.monotonic()
        backend.close()
        took = time.monotonic() - started
        self.assertEqual(self._page_mtimes_ns(keys), [[0, 0]] * 4)
        self.assertEqual(touch_threads(), [])
        self.assertLess(took, 1.0)
        # After close: no queue, no thread, a touch changes nothing, close again is safe.
        backend.touch_pages(keys, newest_ns=123 * 10**9)
        backend.close()
        self.assertEqual(self._page_mtimes_ns(keys), [[0, 0]] * 4)

    def test_a_job_naming_only_missing_files_creates_nothing(self):
        backend = self._backend(touch_interval_s=300)
        backend.register_mem_pool_host(nixl_tests.MockMemPoolHost(True))
        before = sorted(
            os.path.join(folder, name)
            for folder, _, names in os.walk(self.test_dir)
            for name in names
        )
        with self.assertLogs(LOGGER, level="INFO") as logs:
            backend.touch_pages(["gone-1", "gone-2"], newest_ns=time.time_ns())
            backend.touch_pages(["gone-1"], newest_ns=None)
            backend.close()
        after = sorted(
            os.path.join(folder, name)
            for folder, _, names in os.walk(self.test_dir)
            for name in names
        )
        self.assertEqual(before, after)
        self.assertIn("l3_touch files=0 gone=6", logs.output[-1])

    def test_an_idle_thread_blocks_and_does_not_spin(self):
        backend = self._backend(touch_interval_s=300)
        (thread,) = touch_threads()
        cpu_before = time.thread_time()
        process_before = time.process_time()
        time.sleep(1.0)
        # The whole process used almost no CPU while only the touch thread was awake.
        self.assertLess(time.process_time() - process_before, 0.2)
        self.assertTrue(thread.is_alive())
        self.assertTrue(thread.daemon)
        backend.close()
        del cpu_before

    def test_who_keeps_a_backend_that_was_dropped_without_close(self):
        """Informational: prints whether a backend dropped without close() is collected."""
        results = {}
        for name, switches in (("off", {}), ("on", {"touch_interval_s": 300})):
            backend = self._backend(**switches)
            # _backend registered backend.close as a cleanup: drop that reference too.
            self._cleanups[:] = [
                c for c in self._cleanups if getattr(c[0], "__self__", None) is not backend
            ]
            ref = weakref.ref(backend)
            del backend
            gc.collect()
            results[name] = (ref() is not None, len(touch_threads()))
            if ref() is not None:
                ref().close()
        print("\nkept alive after del+gc (alive, touch threads):", results)


if __name__ == "__main__":
    unittest.main()
