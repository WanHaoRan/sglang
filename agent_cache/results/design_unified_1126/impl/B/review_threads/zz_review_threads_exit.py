"""Review check (threads lens): a process that exits without close() while the touch
thread is alive, with jobs still queued. CPU only, temp dirs; the child inherits the
hidden GPU and the nice level.

Throwaway: copied into the copy's test folder under this name, run, deleted.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

CHILD = r"""
import os, sys, threading, time
from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl

test_dir, switch = sys.argv[1], sys.argv[2]
extra = {"plugin": {"posix": {"active": True}}, "use_direct_io": False,
         "l3_cleaner_enabled": False}
if switch == "on":
    extra["touch_interval_s"] = 300
config = HiCacheStorageConfig(
    tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
    is_mla_model=False, is_page_first_layout=False, model_name="m",
    enable_storage_metrics=False, extra_config=extra,
)
with envs.SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR.override(test_dir):
    backend = HiCacheNixl(storage_config=config, file_path=test_dir)
names = [t.name for t in threading.enumerate() if t.name == "hicache-l3-touch"]
print("touch threads:", len(names), flush=True)
for i in range(2000):
    backend.touch_pages([f"k{i}-{j}" for j in range(50)], newest_ns=time.time_ns())
print("queued", flush=True)
# No close(): the interpreter exits with the daemon thread alive and jobs queued.
"""


class ReviewExit(unittest.TestCase):
    def test_a_process_exits_cleanly_without_close(self):
        for switch in ("off", "on"):
            test_dir = tempfile.mkdtemp(prefix="review_touch_exit_")
            self.addCleanup(shutil.rmtree, test_dir, ignore_errors=True)
            started = time.monotonic()
            done = subprocess.run(
                [sys.executable, "-c", CHILD, test_dir, switch],
                capture_output=True,
                text=True,
                timeout=180,
                env=dict(os.environ),
            )
            took = time.monotonic() - started
            tail = [
                line
                for line in done.stderr.splitlines()
                if "Fatal" in line or "Traceback" in line or "Error" in line
            ]
            print(
                f"\nswitch={switch}: returncode={done.returncode} took={took:.1f}s "
                f"stdout={done.stdout.strip().splitlines()} stderr_flags={tail[:5]}"
            )
            self.assertEqual(done.returncode, 0)
            files = sum(len(names) for _, _, names in os.walk(test_dir))
            self.assertEqual(files, 0)


if __name__ == "__main__":
    unittest.main()
