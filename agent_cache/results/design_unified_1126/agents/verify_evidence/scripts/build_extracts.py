import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    t = time.time()
    ev = vlib.load(int(camp), arm, refresh=True)
    print(spec, {k: len(v) for k, v in sorted(ev.items())}, "%.1fs" % (time.time() - t), flush=True)
