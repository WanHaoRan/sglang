import sys, unittest
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
import test_gap_hints
prog = unittest.main(module=test_gap_hints, argv=["test_gap_hints", "-v"], exit=False)
r = prog.result
print(f"RESULT run={r.testsRun} failures={len(r.failures)} errors={len(r.errors)} skipped={len(r.skipped)}")
