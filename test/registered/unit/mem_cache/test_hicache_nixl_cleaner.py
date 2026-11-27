"""Unit tests for the NIXL FILE L3 cleaner, for skipping writes of stored pages, and
for the mtimes set to steer the cleaner's order."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import importlib.util
import os
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.nixl.nixl_cleaner import (
    HiCacheL3Cleaner,
    _parse_group_key,
    _safe_unlink,
)
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    TOUCH_STEP_NS,
    NixlBackendConfig,
    NixlFileManager,
)
from sglang.test.test_utils import CustomTestCase

# hicache_nixl imports the NIXL bindings at module scope;
# the CPU runner has no NIXL wheel.
_HAS_NIXL = importlib.util.find_spec("nixl") is not None

_FILE_BYTES = 16


class _NixlFileTestCase(CustomTestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_nixl_l3_cleaner_")
        self.base_dirs = [os.path.join(self.test_dir, f"disk{i}") for i in range(2)]
        self.file_manager = NixlFileManager(self.base_dirs, use_direct_io=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _write_key(self, key: str, *, mtime: float, size: int = _FILE_BYTES) -> str:
        path = self.file_manager.get_file_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"x" * size)
        os.utime(path, (mtime, mtime))
        return path

    def _run_single_group_cleanup(self) -> None:
        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            recheck_groups=1,
            unlink_workers=1,
        )
        usage_calls: dict[str, int] = {}

        def fake_usage(path: str) -> float:
            usage_calls[path] = usage_calls.get(path, 0) + 1
            return 90.0 if usage_calls[path] == 1 else 60.0

        cleaner._disk_usage_pct = fake_usage
        self.assertTrue(cleaner._tick())


class TestHiCacheL3Cleaner(_NixlFileTestCase):
    """Tests for watermark-driven cleanup over bucketed NIXL FILE layout."""

    def test_parse_group_key_strips_rank_and_kv_suffix(self):
        """Keys for TP ranks and zero-copy K/V files share one cleanup group."""
        self.assertEqual(_parse_group_key("page-a_model_0_8"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_k"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_7_8_v"), "page-a_model")
        self.assertEqual(_parse_group_key("page-a_model_k"), "page-a_model")

    def test_parse_group_key_strips_hybrid_component_suffix(self):
        """All hybrid component shapes share the logical page's cleanup group."""
        names = [
            "page-a_model_7_8_kv_k",
            "page-a_model_7_8_swa_k",
            "page-a_model_7_8_swa_v",
            "page-a_model_7_8_mamba_temporal",
            "page-a_model_7_8_mamba_conv_0",
            "page-a_model_7_8_indexer_2",
            "page-a_model_7_8_draft_swa",
            "page-a_model_deepseek_v4_c4_indexer_state_2",
        ]

        for name in names:
            with self.subTest(name=name):
                self.assertEqual(_parse_group_key(name), "page-a_model")

    def test_tick_deletes_oldest_group_across_bucketed_dirs(self):
        """A cleaner batch deletes all files in the oldest logical key group."""
        old_keys = ["page-old_model_0_2", "page-old_model_1_2"]
        new_keys = ["page-new_model_0_2", "page-new_model_1_2"]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        self._run_single_group_cleanup()
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_a_group_touched_after_the_scan_is_kept_whole_only_under_keep_touched(self):
        """A scan takes seconds; a page whose mtime moves forward before its unlink is in
        use again. One refreshed file keeps the whole page, and the next oldest goes."""
        names = ("old", "mid", "new")
        for keep_touched in (True, False):
            with self.subTest(keep_touched=keep_touched):
                paths = {
                    name: [
                        self._write_key(f"page-{name}_model_{rank}_2", mtime=mtime)
                        for rank in range(2)
                    ]
                    for name, mtime in zip(names, (100.0, 150.0, 200.0))
                }
                cleaner = HiCacheL3Cleaner(
                    self.base_dirs,
                    tp_rank=0,
                    high_watermark=80.0,
                    low_watermark=70.0,
                    recheck_groups=1,
                    unlink_workers=1,
                    keep_touched=keep_touched,
                )
                scan = cleaner._scan_base_dir

                def scan_then_touch(base_dir, groups):
                    scan(base_dir, groups)
                    if base_dir == self.base_dirs[-1]:
                        os.utime(paths["old"][0], (300.0, 300.0))

                cleaner._scan_base_dir = scan_then_touch
                # Full until one page is gone.
                cleaner._disk_usage_pct = lambda path: (
                    90.0
                    if all(os.path.exists(p) for name in names for p in paths[name])
                    else 60.0
                )
                # A tick that stops advancing over kept groups would loop: stop it.
                stopper = threading.Timer(5.0, cleaner._stop.set)
                stopper.start()
                self.addCleanup(stopper.cancel)
                self.assertTrue(cleaner._tick())
                gone = "mid" if keep_touched else "old"
                for name in names:
                    for path in paths[name]:
                        self.assertEqual(os.path.exists(path), name != gone, path)
                shutil.rmtree(self.test_dir)

    def test_tick_deletes_hybrid_components_atomically(self):
        """Evict every pool component and TP rank for one logical page."""
        physical_suffixes = [
            "",
            "_k",
            "_v",
            "_kv_k",
            "_kv_v",
            "_swa_k",
            "_swa_v",
            "_mamba_temporal",
            "_mamba_conv_0",
        ]
        old_keys = [
            f"page-old_model_{rank}_2{suffix}"
            for rank in range(2)
            for suffix in physical_suffixes
        ]
        new_keys = [
            f"page-new_model_{rank}_2{suffix}"
            for rank in range(2)
            for suffix in physical_suffixes
        ]
        old_paths = [self._write_key(key, mtime=100.0) for key in old_keys]
        new_paths = [self._write_key(key, mtime=200.0) for key in new_keys]

        self._run_single_group_cleanup()
        self.assertFalse(any(os.path.exists(path) for path in old_paths))
        self.assertTrue(all(os.path.exists(path) for path in new_paths))

    def test_tick_ignores_non_bucket_directories(self):
        """Only hash-bucket directories are treated as NIXL FILE cache entries."""
        non_bucket = os.path.join(self.base_dirs[0], "not-a-bucket")
        os.makedirs(non_bucket, exist_ok=True)
        unrelated = os.path.join(non_bucket, "page-old_model_0_2")
        with open(unrelated, "wb") as f:
            f.write(b"x")

        cleaner = HiCacheL3Cleaner(
            self.base_dirs,
            tp_rank=0,
            high_watermark=80.0,
            low_watermark=70.0,
            unlink_workers=1,
        )
        cleaner._disk_usage_pct = lambda _path: 90.0

        self.assertFalse(cleaner._tick())
        self.assertTrue(os.path.exists(unrelated))

    def test_safe_unlink_tolerates_missing_and_os_errors(self):
        """Cleanup races should not fail the cleaner tick."""
        missing = os.path.join(self.test_dir, "missing")
        existing = os.path.join(self.test_dir, "existing")
        with open(existing, "wb") as f:
            f.write(b"abc")

        self.assertEqual(_safe_unlink(missing), (False, 0))
        self.assertEqual(_safe_unlink(self.test_dir), (False, 0))
        self.assertEqual(_safe_unlink(existing), (True, 3))
        self.assertFalse(os.path.exists(existing))

    def test_start_only_runs_on_tp_rank_zero(self):
        """Only TP rank 0 owns file cleanup for a shared storage directory."""
        cleaner = HiCacheL3Cleaner(self.base_dirs, tp_rank=1, interval_sec=0.01)
        cleaner.start()
        self.assertIsNone(cleaner._thread)

    def test_nixl_config_parses_l3_cleaner_options(self):
        """Cleaner settings are top-level NIXL config, not plugin init params."""
        cfg = NixlBackendConfig(
            {
                "use_uring": "true",
                "l3_cleaner_enabled": False,
                "l3_cleaner_high_watermark": "85",
                "l3_cleaner_low_watermark": 75,
            }
        )

        cleaner_config = cfg.get_l3_cleaner_config()
        self.assertFalse(cleaner_config["enabled"])
        self.assertEqual(cleaner_config["high_watermark"], 85.0)
        self.assertEqual(cleaner_config["low_watermark"], 75.0)
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {"use_uring": "true"})

        default_config = NixlBackendConfig().get_l3_cleaner_config()
        self.assertTrue(default_config["enabled"])

    def test_nixl_config_rejects_non_boolean_l3_cleaner_enabled(self):
        """Cleaner enablement uses native config booleans only."""
        cfg = NixlBackendConfig({"l3_cleaner_enabled": "false"})

        with self.assertRaises(ValueError):
            cfg.get_l3_cleaner_config()


class TestSkipStoredWrites(_NixlFileTestCase):
    """A page whose files are already stored is touched instead of written again."""

    def test_touch_if_stored_accepts_only_a_full_size_file(self):
        """Only a file of exactly the page's size is stored, and touching it does not
        truncate it; a missing file is not created, and a short (torn) or longer
        (another layout) file is left alone to be rewritten."""
        stored = self._write_key("page-stored_model_0_1_k", mtime=100.0)
        with self.subTest("full size"):
            self.assertTrue(
                self.file_manager.touch_if_stored(file_path=stored, size=_FILE_BYTES)
            )
            self.assertGreater(os.stat(stored).st_mtime, 100.0)
            self.assertEqual(os.stat(stored).st_size, _FILE_BYTES)

        missing = self.file_manager.get_file_path("page-missing_model_0_1_k")
        with self.subTest("missing"):
            self.assertFalse(
                self.file_manager.touch_if_stored(file_path=missing, size=_FILE_BYTES)
            )
            self.assertFalse(os.path.exists(missing))

        for size in (_FILE_BYTES - 1, _FILE_BYTES + 1):
            path = self._write_key(f"page-{size}b_model_0_1_k", mtime=100.0, size=size)
            with self.subTest(size=size):
                self.assertFalse(
                    self.file_manager.touch_if_stored(file_path=path, size=_FILE_BYTES)
                )
                self.assertEqual(os.stat(path).st_size, size)

    def test_touched_page_outlives_an_untouched_newer_one(self):
        """The cleaner deletes the oldest mtime first, so a page skipped instead of
        rewritten must be touched, or it goes before untouched pages written later."""
        reused = [
            self._write_key(f"page-reused_model_0_1{suffix}", mtime=100.0)
            for suffix in ("_k", "_v")
        ]
        idle = [
            self._write_key(f"page-idle_model_0_1{suffix}", mtime=200.0)
            for suffix in ("_k", "_v")
        ]
        for path in reused:
            self.assertTrue(
                self.file_manager.touch_if_stored(file_path=path, size=_FILE_BYTES)
            )

        self._run_single_group_cleanup()
        self.assertTrue(all(os.path.exists(path) for path in reused))
        self.assertFalse(any(os.path.exists(path) for path in idle))

    @unittest.skipUnless(_HAS_NIXL, "nixl is not installed")
    def test_drop_stored_pages_keeps_unstored_pages_aligned(self):
        """A page leaves the write, touched, only when every one of its files is stored;
        the kept (key, buffer) pairs stay aligned and in order, or one page's data
        lands in another page's file."""
        from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl

        backend = HiCacheNixl.__new__(HiCacheNixl)
        backend.file_manager = self.file_manager
        backend._stored_write_skips = 0
        # Zero-copy MHA writes two files per page, K then V.
        key_strs = [
            f"page-{page}_model_0_1{suffix}"
            for page in range(3)
            for suffix in ("_k", "_v")
        ]
        # Distinct addresses make a misaligned or reordered buffer list visible.
        host_buffers = [(0x1000 * (i + 1), _FILE_BYTES) for i in range(len(key_strs))]
        page0_paths = [self._write_key(key, mtime=100.0) for key in key_strs[:2]]
        # Page 1 is torn: K is stored, V was cut short. Page 2 was never written.
        self._write_key(key_strs[2], mtime=100.0)
        self._write_key(key_strs[3], mtime=100.0, size=_FILE_BYTES // 2)

        kept_keys, kept_buffers = backend._drop_stored_pages(
            page_num=3, key_strs=key_strs, host_buffers=host_buffers
        )

        self.assertEqual(kept_keys, key_strs[2:])
        self.assertEqual(kept_buffers, host_buffers[2:])
        self.assertTrue(all(os.stat(path).st_mtime > 100.0 for path in page0_paths))
        self.assertEqual(backend.take_stored_write_skips(), 1)
        self.assertEqual(backend.take_stored_write_skips(), 0)

    def test_nixl_config_resolves_skip_stored_writes(self):
        """Off by default; the top-level key wins over the env var either way, and the
        key is SGLang's own, never forwarded as a NIXL plugin init param."""
        no_key = NixlBackendConfig()
        key_off = NixlBackendConfig({"skip_stored_writes": False})
        key_on = NixlBackendConfig({"skip_stored_writes": True})
        # override() leaves the variable set when its body raises;
        # patch.dict restores the environment even then.
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES", None)
            self.assertFalse(no_key.get_skip_stored_writes())
            with envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES.override(True):
                self.assertTrue(no_key.get_skip_stored_writes())
                self.assertFalse(key_off.get_skip_stored_writes())
            with envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES.override(False):
                self.assertTrue(key_on.get_skip_stored_writes())

        cfg = NixlBackendConfig({"use_uring": "true", "skip_stored_writes": True})
        self.assertEqual(cfg.get_backend_initparams("POSIX"), {"use_uring": "true"})

    def test_nixl_config_rejects_non_boolean_skip_stored_writes(self):
        """A string such as "false" must raise, not turn the skip on as bool() would."""
        cfg = NixlBackendConfig({"skip_stored_writes": "false"})

        with self.assertRaises(ValueError):
            cfg.get_skip_stored_writes()


class TestTouchedPages(_NixlFileTestCase):
    """Mtimes set to steer the unchanged cleaner: a chain memory serves is refreshed
    with its head newest, an ended session's pages are set to zero."""

    def _write_page(self, name: str, *, mtime: float) -> list[str]:
        # Zero-copy MHA keeps two files per page, K and V.
        return [
            self._write_key(f"{name}_model_0_1{suffix}", mtime=mtime)
            for suffix in ("_k", "_v")
        ]

    def _kept(self, pages: list[list[str]]) -> list[bool]:
        return [all(os.path.exists(path) for path in page) for page in pages]

    def test_a_refreshed_chain_loses_its_tail_before_its_head(self):
        """Written head first, a chain loses its head first, and a lookup stops at the
        first missing page; refreshed with the head newest, it loses its tail first."""
        written = [self._write_page(f"page-w{i}", mtime=100.0 + i) for i in range(3)]
        self._run_single_group_cleanup()
        self.assertEqual(self._kept(written), [False, True, True])

        refreshed = [self._write_page(f"page-r{i}", mtime=10.0 + i) for i in range(3)]
        counts = self.file_manager.stamp_pages(refreshed, newest_ns=50 * 10**9)
        self.assertEqual(counts, (6, 0))
        self._run_single_group_cleanup()
        self.assertEqual(self._kept(refreshed), [True, True, False])

    def test_touching_a_deleted_file_creates_nothing_and_is_counted_gone(self):
        """The cleaner may delete a page between its match and its touch: the touch
        creates nothing (a lookup would take an empty file as stored) and is counted."""
        gone = [
            self.file_manager.get_file_path(f"page-gone_model_0_1{suffix}")
            for suffix in ("_k", "_v")
        ]
        kept = self._write_page("page-kept", mtime=100.0)
        newest_ns = 50 * 10**9
        counts = self.file_manager.stamp_pages([gone, kept], newest_ns=newest_ns)
        self.assertEqual(counts, (2, 2))
        self.assertFalse(any(os.path.exists(path) for path in gone))
        self.assertEqual(
            [os.stat(path).st_mtime_ns for path in kept],
            [newest_ns - TOUCH_STEP_NS] * 2,
        )

    def test_a_page_is_retired_only_once_every_file_of_it_is_at_zero(self):
        """A retired page goes before any page still wanted, however old. The cleaner
        ranks a page by its newest file: one file left untouched keeps it in place."""
        wanted = self._write_page("page-wanted", mtime=100.0)
        ended = self._write_page("page-ended", mtime=200.0)
        half = self._write_page("page-half", mtime=300.0)
        counts = self.file_manager.stamp_pages([ended, half[:1]], newest_ns=None)
        self.assertEqual(counts, (3, 0))
        self._run_single_group_cleanup()
        self.assertEqual(self._kept([wanted, ended, half]), [True, False, True])
        self._run_single_group_cleanup()
        self.assertEqual(self._kept([wanted, half]), [False, True])

    def test_nixl_config_resolves_the_touch_switches(self):
        """Both are off by default; a top-level key wins over its env var and is never
        forwarded as a NIXL plugin init param; a value of the wrong type raises."""
        no_keys = NixlBackendConfig()
        keys_off = NixlBackendConfig(
            {"touch_interval_s": 0, "retire_ended_pages": False}
        )
        keys_on = NixlBackendConfig(
            {"use_uring": "true", "touch_interval_s": 60, "retire_ended_pages": True}
        )

        def resolved(cfg):
            return cfg.get_touch_interval_s(), cfg.get_retire_ended_pages()

        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S", None)
            os.environ.pop("SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES", None)
            self.assertEqual(resolved(no_keys), (0.0, False))
            self.assertEqual(resolved(keys_on), (60.0, True))
            with envs.SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S.override(300.0):
                with envs.SGLANG_HICACHE_NIXL_RETIRE_ENDED_PAGES.override(True):
                    self.assertEqual(resolved(no_keys), (300.0, True))
                    self.assertEqual(resolved(keys_off), (0.0, False))
        self.assertEqual(keys_on.get_backend_initparams("POSIX"), {"use_uring": "true"})

        for bad in ("300", True, -1):
            with self.subTest(touch_interval_s=bad):
                with self.assertRaises(ValueError):
                    NixlBackendConfig({"touch_interval_s": bad}).get_touch_interval_s()
        with self.assertRaises(ValueError):
            NixlBackendConfig({"retire_ended_pages": "false"}).get_retire_ended_pages()


if __name__ == "__main__":
    unittest.main()
