"""The client's whole test suite against HEAD's python/ (its default ENGINE). Read-only."""
import sys, unittest
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
import test_gap_hints
suite = unittest.defaultTestLoader.loadTestsFromModule(test_gap_hints)
r = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w")).run(suite)
print(f"run={r.testsRun} failures={len(r.failures)} errors={len(r.errors)}")
for t, _ in r.errors + r.failures:
    print("  ", t.id())
