"""kvsim_compare.py on the referee's patched simulator with one more cost set, 'fa3_nocopy': FA3 with no GPU time
charged for the host-to-GPU load (h2d = 0), the upper bound of what decoding beside the copy can give. The hold's
own delay of the held turn is not modelled. Same command line as kvsim_compare.py; use backend=fa3_nocopy in an arm.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.join(SIM, "referee"))
import kvsim_patched  # noqa: E402

kvsim_patched.BACKENDS["fa3_nocopy"] = dict(kvsim_patched.BACKENDS["fa3"], h2d=0.0)
sys.modules["kvsim"] = kvsim_patched
import kvsim_compare  # noqa: E402

if __name__ == "__main__":
    sys.exit(kvsim_compare.main())
