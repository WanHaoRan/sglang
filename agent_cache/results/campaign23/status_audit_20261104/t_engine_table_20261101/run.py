import sys, unittest
sys.path.insert(0, "/tmp/claude-1001/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d/scratchpad/status_audit/t_engine_table_20261101")
sys.path.insert(1, "/home/wanhr/sglang/agent_cache/scripts")
import test_gap_hints
suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_gap_hints.EngineReaderTest)
r = unittest.TextTestRunner(verbosity=0).run(suite)
print("engine_table_20261101", f"run={r.testsRun} failures={len(r.failures)} errors={len(r.errors)} skipped={len(r.skipped)}")
for t, tb in r.errors + r.failures: print(tb.strip().splitlines()[-1])
