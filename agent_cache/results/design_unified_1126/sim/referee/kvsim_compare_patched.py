"""Referee: kvsim_compare.py on the patched simulator (referee/kvsim_patched.py), same command line.

  python -I -B referee/kvsim_compare_patched.py --arms ec_full hicache "unified@F1" --seeds 0-4 --out <folder>

The module name 'kvsim' is bound to the patched copy before kvsim_compare imports it; nothing else changes.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import kvsim_patched  # noqa: E402

sys.modules["kvsim"] = kvsim_patched
import kvsim_compare  # noqa: E402

if __name__ == "__main__":
    sys.exit(kvsim_compare.main())
