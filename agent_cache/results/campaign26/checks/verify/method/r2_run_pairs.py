"""Run the campaign's own pairs_ec.py, unmodified and read-only, on arms named here, so the repeat's figures come from the
campaign's code as well as from the independent cache. Prints to stdout only.
  nice -n 19 python3 -I -B r2_run_pairs.py arm [arm ...]"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
AC = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
script = os.path.join(AC, "frozen", "oracle_scripts_staging", "c26_ec", "pairs_ec.py")
current = open(os.path.join(AC, "results", ".current_run")).read().strip()
manifest = dict(line.split()[:2] for line in open(os.path.join(AC, "results", "campaign26", "manifest.txt")) if line.strip())
arms = sys.argv[1:]
assert arms, "name the arms"
for a in arms:
    assert manifest.get(a) != current, f"{a} is the run being written"
assert current not in manifest.values(), "the run being written is in the manifest"
os.chdir(AC)
os.environ.pop("STALL_CSV", None)
sys.argv = [script, *arms]
exec(compile(open(script).read(), script, "exec"), {"__name__": "__main__", "__file__": script})
