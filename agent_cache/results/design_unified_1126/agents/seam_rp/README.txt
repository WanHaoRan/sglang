seam_rp: the Retain and Preempt changes of DESIGN.md section 9 mapped onto the engine at HEAD b7074d9cae
(2026-11-26, CPU only; nothing in the repository was edited). The findings are in the workflow's result;
this folder holds what they rest on.

pseudo_diffs/   one pseudo-diff per change against HEAD (not applied), each with its tests
  00_environ.diff             the nine switches, off by default, where each is read
  R3_stored_mark.diff         stored only on a full ack; the mark R1, R4, R5 and P2 read; the heal
  R1_skip_ended_writes.diff   the explicit end at insert; skip the last turn's write
  R4_cleaner_order.diff       file paths, chain enumeration, the touch thread, refresh and retire
  R5_stored_first.diff        stored pages first within a tier
  R2_write_order.diff         ranked write queue (plan A), why TP = 1, plan B
  P1_read_room_order.diff     class bounds, queued slack, room order, wait instead of refusal
  P2_reserve_hold.diff        per-pass quantities, the reserve rule, the paced walk

scripts (stdlib python3 -I -B unless said; each docstring gives its usage)
  extract.sh        builds ext/<campaign>_<arm>.evt.gz from the arms' server.log
  srv_stats.py      write latency and queue, short acks, write holds, re-evictions, what reads ask eviction for
  client_stats.py   last turns of ended sessions, recomputed history, what ended sessions hold at each
                    cleaner pass, utime call counts
  more_stats.py     false_ends | queued_loss | rereads | p2_whatif
  fork_share.py     who shares whose pages in the played trace (shared heads against forks)
  utime_bench.py    cost of os.utime on this host's ext4
  proto_checks.py   CPU checks on the engine's own modules loaded by path: the unchanged cleaner under
                    retire, refresh and the scan race; class bounds on the real strategies; the reserve rule;
                    the ranked queue; the ack walk   (continuum venv python: it needs msgspec)

out/              the scripts' outputs (*.out is git-ignored)
ext/              gzipped event extracts of campaign 34's five arms and campaign 32's ec_full and jit_p50
