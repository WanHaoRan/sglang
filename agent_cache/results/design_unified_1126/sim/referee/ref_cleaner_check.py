"""Referee: defect D4. After a run, every node the SSD holds must still be reachable by the cleaner's order (an
entry (ssd_t[n], n) in the heap). The original pops one node per pass that it does not delete."""
import importlib, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
for name in sys.argv[1:]:
    kvsim = importlib.import_module(name)
    for arm in ("ec_full", "jit_p50"):
        sim = kvsim.Sim(kvsim.Config(arm=arm, seed=0)).run()
        live = {(t, n) for t, n in sim.ssd_heap}
        lost = [n for n in range(sim.wl.n_nodes) if sim.ssd[n] and (sim.ssd_t[n], n) not in live]
        c = sim.cnt
        print(f"{name} {arm}: cleaner passes {len(sim.cleaner_passes)}; stored nodes the cleaner can no longer find: {len(lost)} "
              f"({sum(sim.tok[n] for n in lost)} tokens); replies not cached for lack of GPU room: {c.get('twig_dropped', 'n/a')}; "
              f"tokens of reads that found their pages deleted: {c.get('read_deleted_tokens', 'n/a')}", flush=True)
