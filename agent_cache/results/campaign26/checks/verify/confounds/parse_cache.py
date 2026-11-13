"""Parse the raw client and server logs of the 13 runs once and cache what the tests need.
  nice -n 19 python3 -I -B parse_cache.py [label ...]
Writes cache/<label>.json and prints one line of counts per run."""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

os.makedirs(C.CACHE, exist_ok=True)
labels = sys.argv[1:] or list(C.RUNS)
print("current run (not read):", C.current_run())
for label in labels:
    t = time.time()
    d = C.parse_run(label)
    with open(os.path.join(C.CACHE, label + ".json"), "w") as f:
        json.dump(d, f)
    turns = d["turns"]
    ok = [x for x in turns if x["ttft"] is not None]
    print(f"{label:14s} {d['run']}  turns {len(turns)} with ttft {len(ok)} errors {sum(1 for x in turns if x['err'])} bad_json {d['bad_lines']} "
          f"convs {len(d['conv_start'])} | server lines {d['server_lines']} stamped {d['server_stamped']} req {len(d['req'])} s2h {len(d['s2h'])} "
          f"stage {len(d['stage'])} disp {len(d['disp'])} evd {len(d['evict_device'])} evh {len(d['evict_host'])} dec {len(d['dec'])} pre {len(d['pre'])} "
          f"| {time.time() - t:.1f}s")
