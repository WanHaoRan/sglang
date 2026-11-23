Campaign 34: the verification of the FA3 engine fixes (agent_cache/OPEN_ITEMS.md section 7), ordered by the author on
2026-11-22: "prototype the engine fixes while campaign 33 runs, after campaign 33 finishes, verify the fixes".

The fixes (working tree python/, frozen as frozen/engine_fa3fix_20261122; each off by default, switched by env vars that
settings.env sets, so the launcher exports them to every arm's server):
  SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES=1          a page whose files the SSD store holds is touched, not written again
  SGLANG_HICACHE_STORAGE_WRITE_COALESCE_PAGES=128   queued writes merge into transfers of up to 128 pages
  SGLANG_HICACHE_STORAGE_READ_HOLD_WRITE_TOKENS=50000  timed reads wait while pending writes hold 50,000 host tokens
  SGLANG_HICACHE_NEXT_TURN_ROOM_RETRY=1             a read refused host room is retried without spending MAX_REFETCHES
  SGLANG_LOG_HICACHE_HOST_CENSUS_INTERVAL=10        a host_census line every 10 s and at refusals (logging only)

Setting: campaign 32's (FreeInference on FA3, settings.env adds the switches), so requests pair with campaign 32's arms
by (play, stream, k). Arms (queue.txt): jit_p50, ec_full, jit_exact, the full 2 h each.

Run by chain_fix.sh after campaign 33 (see its header): unit tests on the frozen copy, a tier correctness check with
the write-path fixes on, then the arms. Log: chain.log; unit_tests.log; the check's run folder holds check_fixes.*.
