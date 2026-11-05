"""Run the client's EngineReaderTest (agent_cache/scripts/test_gap_hints.py) against each engine copy. Read-only."""
import sys, unittest
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
import test_gap_hints
ENGINES = {
    "python/ (HEAD)": "/home/wanhr/sglang/python/sglang/srt",
    "engine_5df71c4bf0_v2": "/home/wanhr/sglang/agent_cache/frozen/engine_5df71c4bf0_v2/python/sglang/srt",
    "engine_table_20261101": "/home/wanhr/sglang/agent_cache/frozen/engine_table_20261101/python/sglang/srt",
    "engine_owned_20261101": "/home/wanhr/sglang/agent_cache/frozen/engine_owned_20261101/python/sglang/srt",
}
for name, path in ENGINES.items():
    test_gap_hints.ENGINE = path
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_gap_hints.EngineReaderTest)
    r = unittest.TextTestRunner(verbosity=0, stream=open("/dev/null", "w")).run(suite)
    last = sorted({tb.strip().splitlines()[-1] for _, tb in r.errors + r.failures})
    print(f"{name}: run={r.testsRun} failures={len(r.failures)} errors={len(r.errors)} {last}")
