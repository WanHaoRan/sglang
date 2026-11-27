Campaign 30: the baselines on FreeInference at campaign 34's setting, ordered by the author on 2026-11-23 ("schedule
the run for all the baselines the same evaluation as in campaign 34"): sglang_vanilla, vllm_vanilla, vllm_lmcache,
tokencake, continuum (GPU only) and continuum_lmcache, 2 h each, after campaign 34's hicache and ec_exact.
Systems, versions and checks: agent_cache/baselines/README.md and each system's README. Arms and flags: queue.txt.
Run by chain_baselines.sh (log chain.log), which starts baselines/common/run_baselines.sh campaign30 (log runner.log,
launcher.log, manifest.txt) once campaign 34's launcher has exited.

2026-11-24: the host was preempted during the first arm (sglang_vanilla, launched 01:24:29Z; the container stopped at
about 02:45Z). The launcher resumed with that arm from its start at 03:42:59Z on the same GPU; the interrupted
attempt's run folder, driver log and log lines were removed at the author's request.
The host was preempted again during the second arm (vllm_vanilla, launched 05:48:19Z; down from about 06:20Z to 14:14Z)
and came back on another H200 (GPU-42ffbe53); the launcher resumed with vllm_vanilla from its start at 14:22:11Z, and
the remaining arms ran on that unit. The interrupted attempt's run folder, driver log and log lines were removed at the
author's request, after its first 10 minutes had served the GPU check in checks/gpu_unit_vllm.out.
