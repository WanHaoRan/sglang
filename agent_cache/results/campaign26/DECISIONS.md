# Campaign 26: the expected-cost policy against today's arms

Started 2026-11-11 02:49Z by request: review the deck "Pricing the Room" (the paper's design), build it, and compare it with
JIT p50 and the other methods at the deck's workload. Setting: 44 users, 70 GB of host memory, gaps x10, `ARRIVAL=0.1`,
`USERS=768`, `TURNS=50`, `SEED=0`, 2 h per arm, as campaigns 21 and 23, so turns pair with those runs by (conv, turn). Two settings joined later: 128 users with 160 GB (`_c128` arms) and 44 users with 100 GB (`_h100` arms).
GPU units, a new one after each host outage (each arm's is in `launcher.log`): `GPU-605dfc41` from the 02:34Z boot, `GPU-a7198925` from 20:40Z on 11 Nov, `GPU-6374d407` from 15:36Z on 12 Nov, `GPU-72ebee63` at 17:03Z on 13 Nov (no client started), `GPU-a8a5cbbe` from 19:28Z on 13 Nov.

Launcher `frozen/oracle_scripts_staging/c26_ec/c26.sh`, watcher `checks/watch_all.sh` (since 06:06Z on 13 Nov; `watch.sh` beside the launcher before that, which skips 100 GB runs), queue `queue.txt`, finished arms in
`manifest.txt`, runs outside the manifest that a table may name (cut runs, earlier campaigns) in `extra_runs.txt`. A queue line may end with extra server flags; they reach the server through `SERVER_EXTRA` in
`scripts/start_server.sh` (added before launch), so new arms need no script edits while the campaign runs.

Relaunch after a shutdown: `: > results/.current_run`, `docker start sglang_hicache`, then the `docker exec -d` line at the
top of `c26.sh`, then the watcher from `agent_cache/` (`bash results/campaign26/checks/watch_all.sh`). Finished arms are skipped; an interrupted arm reruns from the start.

## Review of the design, 2026-11-11

The formulation matches `PAPER_PLAN.md` ("Formulation" 1-8, "Design point") and its math holds: I re-derived rule 1's
threshold and the keep test, and the engine's solver agrees with `room_model.expected_cost_start` on 400 of 400 random
cases. Weak spots, with what the build does about each:

| # | Weak spot | In the build |
|---|---|---|
| 1 | The model prices only the host-only room (44K of 712K tokens) and leaves GPU eviction to HiCache, but the replay puts most of the p50's cost in eviction, and the hinted arms already order GPU and host eviction by next use | rule 3 scores every session in GPU and host memory. Since: the runs put the p50's cost in when its reads start, not in eviction (`ec_r1`, `ec_r3` and `fq90` under Results) |
| 2 | Past the p90 the hint says nothing. A lognormal tail makes an overdue session with a sharp key look certain to return, and there is no "never returns" mass; the replay signals every recorded end, which a real agent may not | checked in the eviction replay (below): the plain lognormal tail is as good as a pooled tail while ends are signalled, so nothing was added |
| 3 | With no forecast ending a session, a staged read's pin has no release (600 s backstop only) | the pin is given back once the entry has passed (its last few support points plus 5 s). A release by rent against stall was tried first and dropped before any run: at 5e-5 a 64K-token pin costs 3.2 s per second, so it released almost every read as it landed |
| 4 | The cost leaves out the queue (a recomputed token costs 6-8 times its prefill) and SSD bandwidth; the replay ties reading at arrival once recomputation is weighted x10 | not built |
| 5 | The room is about 90K and moves; "idle room" has no meaning in a host pool that is always full of evictable histories | room = host pool minus device-resident tokens, held = pinned plus in-flight reads. Since: the room measured 54K at the median and 40-46K in the load episodes, not 90K; 44K is the pool minus the GPU's mirror |
| 6 | Rule 4 compares a flow (tokens refused per interval) with a stock (idle tokens); by my arithmetic it settles where refusals are as large as idle room, near JIT p50's regime | built as stated (`price-learning on`, bumped tokens count as refused), plus a second rule (`harm`): the price is the recomputation staged pins left to arrived turns, per token-second pinned. `ec_r1` logs both rules' signals every 5 s |
| 7 | Soft pins touch the host allocation path for 2-6 % in the replay | built behind a flag; ran as `ec_full`, with rules 1 and 3 and the harm price |
| 8 | The asymptotic claim needs many sessions each small against the room; at 70 GB the room holds two or three histories | paper: present the theory as the rules' derivation; add a setting where the room holds ten or more |
| 9 | Evidence is single runs and a replay of costs, not TTFT | same-GPU controls in this campaign |
| 10 | The reference slide omits Continuum, Tokencake, Bidaw and CachedAttention, which `PAPER_INTRO.md` says must be cited; the queue-aware hint-free rival is not built | fixed-quantile rival is a flag and ran as `fq90`; the queue-aware rival with no hints was built as engine e and ran as `nofc_noend` (repeat, 128 users, 100 GB); the references are open |

Three further points came out of the eviction replay (`frozen/ec_policy_20261111/evict_replay/FINDINGS.md`), which replays
eviction on a run's own timeline with the pool sized so that LRU matches the hint-free arm's 19.4 % of returning turns
reading from SSD:

- Rule 3 with a fixed horizon reads on 16.2 % of returning turns, against 17.7 % for today's order from the p50 and
  12.6 % with exact next use. Any horizon from 10 to 120 s does as well; the engine uses 30 s.
- The deck's "exact order" by return rate reads on 29.8 %, worse than LRU: right after a turn ends the rate is near
  zero, so the histories just used go first. It is not built.
- Most of rule 3's gain is the end signal: LRU that evicts ended sessions first reads on 16.9 %. Without the signal the
  three-quantile score reads on 23.8 %, worse than LRU, and no tail fix helps; scoring from every sample of the key
  reads on 15.1 % with the signal and 18.0 % without. So the score should come from samples the engine collects per
  key, and the fair rival is LRU with ended sessions first. Neither was built at the review; both are scores of `return_chance` now (`samples`, `lru`): `lru` ran as `nofc`, `nofc_rep` and `nofc_c128`, `samples` never ran.

## What was built (engines frozen as `frozen/engine_ec_20261111d` and `e`; in the author's commit a5fe6121c8 since 2026-11-13 17:03Z)

- `python/sglang/srt/mem_cache/next_turn_policy.py`: the distribution (two-piece lognormal through p10/p50/p90, a point
  for exact hints), `read_start` (rule 1), `chance_within` (rules 2 and 3), `PriceController`
  (rule 4).
- `--hicache-storage-next-turn-read-timing expected_cost` with `--hicache-storage-next-turn-memory-price` (default 5e-5):
  rule 1. A landed read stays pinned until admission, as in the room model, or until its entry has passed. A plan lives
  until its turn arrives or its session ends (30 min cap as bookkeeping). A plan may stage twice, which leaves one of
  the arrival gate's three lookups for the turn's own read.
- `--hicache-storage-next-turn-read-timing arrival`: never read before arrival; the plan only serves the gate, the hold
  and queued-turn protection. With `score: lru` below it is the rival that uses no gap forecast.
- `--radix-eviction-policy return_chance` (`horizon_s`, default 30): rule 3 for GPU and host eviction. `score` picks
  where the chance comes from: `entry` (the hint's three quantiles), `samples` (the gaps the engine has seen for the
  entry's key, then for every key), `lru` (no forecast: ended sessions first, queued turns last, LRU between), or `queue` (engine e only: `lru` with the end signal ignored too). The score is the chance alone: the plan's cost factor c_i and the deck's keep test are not built (every history read back whole from SSD costs the same, `PAPER_PLAN.md` Formulation 6; turns whose SSD copy was short are 2.7-2.9 % of TTFT seconds at most, `checks/limits/why_recompute.out`).
- `--hicache-storage-next-turn-soft-pins on`: rule 2. It takes room from the staged read least likely to return within its own read time; the deck's comparison of the two due times is not built.
- `--hicache-storage-next-turn-price-learning on | harm`: rule 4's price as the deck states it, or by measured harm.
  The per-key recalibration of rule 4 is not built as such; `score: samples` is the engine learning each key's gaps. Nor are the cold-start bisection (every arm starts at the fitted 5e-5) and the count of evicted histories as refused (rule 3 ranks and reads no price); so "all four rules" (`ec_full`) is rules 1 to 3 with the harm-learned price.
- `--hicache-storage-next-turn-landing-quantile q`: the fixed-quantile rival, under `just_in_time`.
- New log events: `ec_start`, `ec_release`, `ec_bump`, `ec_price`.

Engines d and e were cut from b684fd6464. `python/` and the tests at a5fe6121c8 equal engine e; d is e without the `queue` score (`next_use_eviction.py`, `fields/memory.py`, `kv_cache_hook.py`). `agent_cache/frozen/` is outside git (`.git/info/exclude`), and d's `changes_vs_HEAD.patch` lacks the new file `next_turn_policy.py` (e's has it). Three earlier freezes are deleted; none produced a run. The first had the release by rent. The second was stopped
30 s into `ec_r1` when a two-reviewer code check came back (no crash paths, the defects below), and the hint-free control
ran in its slot. A second review of the fixes cleared the arms to run and found rarer defects, fixed as well. In
`engine_ec_20261111d`:

- **Wait estimate.** The predicted queue wait counted turns blocked on their own SSD read, which admission passes over.
  In the control's log a turn with resident history predicted to wait 1-2 s waited a median 4 ms. Under expected-cost
  timing the queue now counts only while some turn with nothing to read has stayed queued for 0.5 s, the sign that
  the GPU is what the queue waits for.
- **Restaging.** A staged read refused room was staged again half a second later and used up the gate's lookups. It is
  now left to the turn's arrival; only a read that was bumped or released may stage again (twice at most).
- **Rent.** It was charged on the span being read. A pinned tail keeps the session's whole chain out of host eviction,
  so it is now charged on every token of the key not in GPU memory.
- **Soft pins** asked staged reads for the whole read's size instead of the pages it was short of, and could bump a
  turn that had just arrived.
- **Rule 3** scored one node per session. A queued turn whose tail had been evicted, or a span read back from SSD,
  ranked as un-hinted and went first. A registration now covers the session's whole path: nodes follow the session's
  tail from turn to turn, a span read back is adopted when it lands, and a queued registration lapses after 600 s.
- **Smaller:** a landing quantile above the median let the plan expire before its read; flag combinations that did
  nothing are refused; refusals are recorded only when the price reads them; exp() is clamped; a read still parked
  when its entry passes is given up; a fixed price is no longer clamped; the harm price's prior fades.

Known and left: with soft pins, a turn that reached the scheduler within the last pass can have its landed staged read
bumped (one pass wide; needs a scheduler hook to close). `ec_release` and `ec_bump` pop the loaded-span record, so a turn later
served by those still-resident pages counts as a host hit in the logs, not a storage hit.

Unit tests are in `test/registered/unit/mem_cache/` (`test_next_turn_estimators.py`, `test_next_turn_prefetch.py`,
`test_next_use_eviction.py`, `test_storage_read_dispatch.py`, `test_oracle_v2_flags.py`); they pass with the GPU hidden
(`CUDA_VISIBLE_DEVICES=7`). `test_pp_prefetch_ticket.py` has two failures that the untouched engine has too (`test_allocation_error_preserves_ack_sequence_and_next_ticket`, `test_cancel_before_ticket_defers_free_until_final_ack`): its fixture builds the cache and the controller with `__new__` and lacks attributes this branch sets in `__init__` since f1885117c1 (`_storage_read_ms_ewma`, `pinned_prefetch`). By reading, not rerun.

## Arms

| Label | Engine | What it is |
|---|---|---|
| `jit_p50_ctl` | `engine_owned_20261101` | today's JIT p50, a control on this GPU unit |
| `hintfree_ctl` | `engine_5df71c4bf0_v2` | hint-free arm (read pins, no hints), a control on this GPU unit |
| `ec_r13` | `engine_ec_20261111d` | rules 1 and 3: expected-cost start at 5e-5, eviction by chance of return within 30 s (three quantiles) |
| `nofc` | `engine_ec_20261111d` | the rival with no gap forecast: reads at arrival only, eviction by the end signal and the queue |
| `ec_r1` | `engine_ec_20261111d` | rule 1 alone: next-use eviction from the p50 and hard pins as today |
| `nofc_noend` | `engine_ec_20261111e` | `nofc` with the end signal ignored too: the hint-free rival that keeps queued turns' history |
| `ec_r3` | `engine_ec_20261111d` | rule 3 alone: JIT p50's read timing, eviction by chance of return |
| `ec_r13_rep`, `nofc_rep` | `engine_ec_20261111d` | repeats of the two leading arms |
| `jit_p50_d` | `engine_ec_20261111d` | today's JIT p50 on the new engine with no extra flag: the same-build control (the cache changes are all behind the new flags, by diff; the prefetcher was restructured) |
| `ec_r134` | `engine_ec_20261111d` | rules 1, 3 and 4: the price learned from the harm staged pins do, starting at 5e-5 |
| `fq90` | `engine_ec_20261111d` | the fixed-quantile rival: JIT reads timed to land at the entry's p90, next-use eviction |
| `ec_full` | `engine_ec_20261111d` | all four rules: `ec_r134` with soft pins |
| `nofc_noend_c128`, `ec_r134_c128`, `nofc_c128` | `engine_ec_20261111e`, `d`, `d` | the deck's second setting, 128 users and 160 GB: the rival with no hints, rules 1, 3 and 4, then the rival with the end signal |
| `jit_exact_d` | `engine_ec_20261111d` | added 2026-11-12: today's JIT on exact hints, the reference on this build and GPU unit |
| `nofc_noend_rep` | `engine_ec_20261111e` | added 2026-11-12: the rival with no hints, again |
| `nofc_noend_h100`, `jit_p50_h100`, `jit_exact_h100` | `engine_ec_20261111e`, `d`, `d` | added 2026-11-12: 44 users with a 100 GB host pool: the rival with no hints, JIT p50, JIT exact |
| `jit_p50_d_cut`, `nofc_noend_h100_cut`, `oracle_v2`, `oracle_v2_p50`, the four `_c128` references | see `extra_runs.txt` | runs outside the manifest that tables name: two runs cut by outages, and references from campaigns 19, 21 and 23 |

Every new arm is today's JIT p50 arm (`three_tier_wc_oracle_v2_nu_owned`: gate, hold, pin fraction 0.3) with the flags
above replacing its read timing and eviction. The queue was rewritten 11:15Z after `nofc`, reordered 13:20Z after
`ec_r1` and changed 15:16Z after `nofc_noend` (only unstarted lines; the order is under Results) and, after the outage
of 16:32Z, the changes of 01:38Z and 02:38Z on 2026-11-12, the outage of 07:07Z that day and the outages of 07:31Z and 17:06Z on 2026-11-13, ends about 01:41Z on 2026-11-14 with three 100 GB arms (`nofc_noend_h100` on engine e, `jit_p50_h100` and `jit_exact_h100` on engine d; defined under Results, 02:38Z on 12 Nov). `checks/validate_queue.py` parsed every unstarted line's flags with its own engine and ran the
validation hooks: all pass.

- **`engine_ec_20261111e`** (frozen 11:14Z) is engine d plus one eviction score, `"queue"`: an ended session ranks by
  LRU like any other, and a queued turn still goes last. With reads at arrival it uses the hint for nothing an engine
  could not see itself (which request continues which session), so it stands for an engine with no hints.
- **Not queued:** rule 3 from the engine's own samples. With `nofc` level with `ec_r13`, the score is worth less than
  one run can show; the replay put it at one point of read share.
- **The 128-user price is learned, not fixed.** 5e-5 was fitted to 70 GB; the deck puts the price near zero at 160 GB.
  If `ec_r134` shows the harm rule misbehaving at 44 users, the 128-user line changes before it starts. Since: `ec_r134` did show it (the price decays in calm stretches and is lowest just before each load episode), the 128-user line ran unchanged, and it made no difference there: rule 1 started no read early and the price never moved.

## Results

**`jit_p50_ctl` (done 04:55Z, clean).** Today's JIT p50 repeats its mechanisms but not its TTFT to better than about
0.15-0.25 s. Against campaign 23's run on the 9,660 returning turns that it, the hint-free arm and JIT exact all share:
1.46 against 1.60 s (hint-free 1.75, JIT exact 0.72). On the 6,715 turns that original HiCache's two runs also share:
1.35 against 1.60 s (original HiCache 5.47 and 6.62, hint-free 1.61, JIT exact 0.51). Staged reads (2,202 against
2,094), launches refused room (1,732 against 1,743), mean pinned tokens (31.5K against 30.2K) and the share of turns
reading from SSD (23.7 against 23.8 %) all repeat; the difference sits in the heavy stretch (positions 5,500-7,000:
3.92 against 5.24 s), where queueing amplifies it. Hardware speed is the same (decode, prefill and SSD per page).
So a difference between arms under about 0.25 s is not a result on one run each. Tables: `compare_ec_ctl.out`.

**`hintfree_ctl` (done 07:00Z, clean).** The hint-free arm repeats no better: 1.89 against campaign 21's 1.61 s on the
6,715 turns below (heavy stretch 7.43 against 5.96 s), with the same share of turns reading from SSD (20.6 against
19.2 %). With two runs each, JIT p50 (1.35, 1.60) and the hint-free arm (1.61, 1.89) overlap, so the deck's opening tie
(1.51 against 1.47 s) is inside the spread; the means lean 0.27 s toward p50 hints.

**`ec_r13` (done 09:03Z, clean): rules 1 and 3 on three-quantile hints reach the exact-hint arm's level.** Settings
checked in the log (expected_cost, price 5e-5, `return_chance` with `horizon_s` 30, no tracebacks, no client errors).
Mean TTFT on the 6,715 returning turns that all eight runs share (`compare_ec.out`):

| Arm | Mean TTFT | Queue wait | Recomputed per turn | Share of the gain |
|---|---|---|---|---|
| Original HiCache, two runs | 5.47, 6.62 s | 4.53, 5.50 s | 2,962, 3,814 | 0 % |
| Hint-free, two runs | 1.61, 1.89 s | 1.04, 1.29 s | 750, 903 | 72-82 % |
| JIT p50, two runs | 1.60, 1.35 s | 0.89, 0.69 s | 970, 928 | 78-86 % |
| `ec_r13` | 0.66 s | 0.25 s | 214 | 97-98 % |
| JIT exact | 0.51 s | 0.13 s | 120 | 100 % |

The share is of the way from each original-HiCache run to JIT exact.

- **It is outside the spread.** 0.66 s is 0.69 s below the better JIT p50 run and 0.95 s below the better hint-free run;
  arms repeat to about 0.25-0.3 s. One run so far.
- **It keeps pace with exact hints through the run.** Position 7,000 of its own send order at minute 67.6 (exact 66.7,
  JIT p50 77.2, hint-free 78.7), 11,690 turns in 115 min (exact 11,699, JIT p50 10,208). In the heavy stretch
  (positions 5,500-7,000) it is at 1.56 s against exact 1.14, JIT p50 4.01-5.24 and hint-free 5.96-7.43.
- **Fewer reads, almost no pins.** On its own returning turns:

  | | JIT p50 (two runs) | `ec_r13` | JIT exact |
  |---|---|---|---|
  | Turns that read from SSD | 23.7-23.8 % | 14.3 % | 11.3 % |
  | Staged reads | 2,094-2,202 | 629 | 1,202 |
  | Read launches refused room | 1,732-1,743 | 395 | 381 |
  | Mean pinned tokens | 30.2-31.5K | 4.2K | 5.9K |
  | SSD pages read | 563-580K | 437K | 388K |

  The replay had put rule 3 at 16.2 % of turns reading against 17.7 % for today's order; the run gives 14.3 against
  23.7 %, so pins pushing histories out look like the rest of JIT p50's reads.
- **Rule 1 mostly declines to stage.** Of 2,667 start decisions, 535 were "leave it to arrival", 1,503 "wait", and
  629 staged; 38 landed reads were released after their entry passed. Hardware speed matches the controls.

Which rule carries it is not separated yet: `nofc` (no gap forecast), `ec_r1` (rule 1 alone) and `ec_r3` (rule 3 under
JIT timing) follow, then a repeat of `ec_r13`.

**Rule 4 on `ec_r13`'s own signals** (`price_openloop_ec_r13.out`, script `c26_ec/price_openloop.py`). The arm logged,
every 5 s, the tokens refused room, the idle room, the staged pins and the harm charged. Replayed open loop from 5e-5:

- **As the deck states it, the price falls to its floor within a minute and stays there** (at 1e-7 in 77 % of
  intervals). The idle room averages 75K tokens against 7.7K refused per interval, so (refused - idle) / R is above zero
  in 9 % of intervals. At the floor rule 1 reads everything back at once, which is the regime the price was meant to
  leave. Open loop only: at a lower price more would be staged and refused, and the rule would settle where refusals
  are as large as idle room.
- **By harm, the price stays near the calibrated one without being told it.** The recomputation that staged pins left
  to arrived turns, over the token-seconds they were pinned, is 2.9e-5 over the run (median 2.7e-5, p10 to p90 9.6e-6
  to 5e-5). The room model's best fixed price was 5e-5.

So the learned-price arm will use the harm rule; the deck's rule stays in the engine for comparison.

**`nofc` (done 11:05Z, clean): the rival with no gap forecast is level with rules 1 and 3 on the first set of turns and
0.12 s behind on the whole run.** Settings checked in the log (`read_timing=arrival`, `return_chance` with score `lru`,
no staged read, no tracebacks, no client errors). It reads only at arrival and evicts ended sessions first, queued turns
last and by LRU between; from the hint it takes the end signal and which request continues which session.

| Mean TTFT | Turns | JIT exact | `ec_r13` | `nofc` | JIT p50 (ctl) | Hint-free (ctl) |
|---|---|---|---|---|---|---|
| Returning turns all nine runs share | 6,715 | 0.51 s | 0.66 s | 0.69 s | 1.35 s | 1.89 s |
| Returning turns these five share | 9,702 | 0.72 s | 0.79 s | 0.92 s | 1.48 s | 1.95 s |
| Returning turns the first three share | 11,299 | 0.70 s | 0.76 s | 0.88 s | | |
| p90, p99 on the 11,299 | | 0.86, 10.4 s | 1.30, 9.2 s | 1.51, 13.3 s | | |
| Turns over 10 s on the 11,299 | | 118 | 97 | 197 | | |

The first row stops where original HiCache's runs stop (about 7,500 turns); the later rows add the second load episode
(positions 8,500-9,500), where every arm, exact hints included, is at 4-5 s. Tables: `compare_ec.out`, `split_ec.out`,
`split_ec_trio.out` (script `c26_ec/split_ec.py`).

- **Most of the way from today's arms to exact hints needs no gap forecast.** From JIT p50's 1.35-1.60 s and the
  hint-free arm's 1.61-1.89 s, `nofc` is at 0.69 s with 131 recomputed tokens per turn (JIT p50 928-970, hint-free
  750-903) and 14.2 % of turns reading from SSD (23.7 and 20.6 %). It stages nothing, so it pins 2.6K tokens on average
  (JIT p50 31K) and has 277 launches refused room (JIT p50 1,732).
- **Against the hint-free arm it adds four things, not yet separated:** ended sessions evicted first, queued turns'
  history protected, the arrival gate and hold, and re-reads for queued turns. Turns that shrank while queued fall from
  288 to 7. `nofc_noend` takes the end signal away.
- **JIT p50's early reads cost it more than they save.** It has every mechanism `nofc` has, plus reads timed for the
  p10 and eviction by the p50, and is 0.66-0.91 s slower. `ec_r1` and `ec_r3` say which of the two it is.
- **What rule 1 adds over reading at arrival is 0.03 to 0.12 s, inside one run's noise.** The 0.12 s sits in the
  second load episode (0.09 s of it) and in the tail (97 against 197 turns over 10 s). On the 548 shared turns `ec_r13`
  staged, it is at 2.98 s against 3.56 s for the same turns in `nofc` and 2.44 s with exact hints. The slowest 1 % of
  turns carry 23 % of each leading arm's summed TTFT, so the repeats (`ec_r13_rep`, `nofc_rep`) decide whether the
  0.12 s is real.
- **The deck's replay said as much.** It put rule 1 14 % below reading at arrival on stall plus recomputation and
  level with it once recomputation is charged for the queue it causes. Exact hints are 0.06 s ahead of `ec_r13` and
  0.18 s ahead of `nofc` on the 11,299 turns: that is the whole prize a gap forecast can still win at this load.
- **Where the rest to exact hints sits:** in turns that read from SSD in the arm but not with exact hints (760 turns in
  `ec_r13`, 615 in `nofc`, about 0.11 s per shared turn each), partly returned by turns that read only with exact
  hints (420 and 287 turns). Which histories are in memory differs more between arms than when reads start.

**`ec_r1` (done 13:08Z, clean): rule 1 alone is level with rules 1 and 3 and with the rival.** Settings checked in the
log (`expected_cost` at 5e-5, `next_use` eviction with `horizon_s` 300 as in JIT p50, no tracebacks, no client errors).

| Mean TTFT | Turns | JIT exact | `ec_r13` | `ec_r1` | `nofc` | JIT p50, two runs |
|---|---|---|---|---|---|---|
| Returning turns all ten runs share | 6,715 | 0.51 s | 0.66 s | 0.67 s | 0.69 s | 1.35, 1.60 s |
| Of them, positions 5,500-7,000 | 1,374 | 1.14 s | 1.56 s | 1.75 s | 1.83 s | 4.01, 5.24 s |
| Returning turns the four leading arms share | 11,198 | 0.70 s | 0.76 s | 0.86 s | 0.88 s | |
| p90, p99 on the 11,198 | | 0.87, 10.6 s | 1.29, 9.2 s | 1.41, 12.2 s | 1.51, 13.3 s | |
| Turns over 10 s on the 11,198 | | 118 | 97 | 173 | 197 | |

Tables: `compare_ec.out`, `split_ec_lead.out`.

- **JIT p50's loss is when its reads start, not its eviction order.** `ec_r1` keeps JIT p50's eviction (next use from
  the p50) and changes only the read start: 0.67 s against 1.35-1.60 s. It stages 670 reads against 2,094-2,202, pins
  4.8K tokens on average against 30-32K, has 396 launches refused room against 1,732-1,743, and 14.0 % of its turns
  read from SSD against 23.7 %. The two arms run different engine builds; `jit_p50_d` checks that.
- **Rule 1 mostly declines here too.** Of 2,713 start decisions, 605 were "leave it to arrival", 1,439 "wait" and 669
  staged; 32 landed reads were released after their entry passed.
- **Rule 3 adds 0.01 to 0.10 s on top of rule 1, inside one run's noise.** The 0.10 s sits where `nofc`'s deficit
  sits: in the second load episode (positions 8,750-11,000: `ec_r13` 0.02 s per shared turn ahead of exact hints,
  `ec_r1` 0.06 s and `nofc` 0.07 s behind) and in the tail. `ec_r1`'s reading turns recompute more (912 tokens against
  626 in `ec_r13` and 636 in `nofc`).
- **Three ways of not reading early land in the same place.** Rule 1 with either eviction order and reads at arrival
  with a forecast-free order are at 0.66-0.69 s on the first set and 0.76-0.88 s on the whole run. What they share is
  that few or no reads are staged ahead of arrival, so almost nothing is pinned. `ec_r13` is first in both rows, by a
  margin the repeats have to confirm.
- **The turns that need SSD differ between arms.** Of the 605 shared turns `ec_r1` staged, `ec_r13` read 253 from SSD
  and exact hints 249; of the 540 `ec_r13` staged, `ec_r1` read 266. So a staged turn in one arm is mostly a turn whose
  history another arm still had in memory, and its TTFT (2-3 s in every arm) is the load episode's queue.

**Where the wait is (13:20Z; `stall_ec_lead5.out`, `stall_ec_ctl.out`, script `c26_ec/stall_ec.py`).** A turn's queue wait
is split where its last SSD read lands: the part before is its read stall (zero when it read nothing or its reads had
landed by arrival), the part after is the rest. On the 11,198 returning turns the four leading arms share, per turn:

| | JIT exact | `ec_r13` | `ec_r1` | `nofc` |
|---|---|---|---|---|
| Mean TTFT | 0.70 s | 0.76 s | 0.86 s | 0.88 s |
| Read stall | 0.18 s | 0.24 s | 0.28 s | 0.34 s |
| of it, the read was the last wait (admitted within 0.1 s of landing) | 0.05 s | 0.13 s | 0.12 s | 0.17 s |
| of it, the GPU queue held the turn as well | 0.13 s | 0.11 s | 0.15 s | 0.17 s |
| Rest of the queue wait | 0.11 s | 0.11 s | 0.15 s | 0.13 s |
| After admission | 0.41 s | 0.41 s | 0.44 s | 0.41 s |
| Turns reading from SSD; of them landed by arrival | 1,262; 684 | 1,582; 216 | 1,572; 269 | 1,581; 4 |

- **The distance between the leading arms is read stall.** `nofc` is 0.18 s behind exact hints, 0.16 s of it stall;
  `ec_r13` is 0.06 s behind, all of it stall. The order is the same in both load episodes (stall per turn at
  positions 5,500-7,000: 0.37, 0.56, 0.64, 0.88 s; at 8,750-11,000: 0.88, 0.90, 1.08, 1.29 s).
- **What read timing can win here is 0.17 s per turn, a fifth of TTFT.** That is the stall of turns that waited for
  nothing but their read when every read starts at arrival. Rule 1 removes 0.045 s of it under either eviction order;
  exact hints remove 0.12 s. The other half of the stall falls on turns the GPU queue held too; it differs by run
  (0.11-0.17 s) and read timing alone does not remove it.
- **Rule 1 lands few reads early.** 216-269 of about 1,580 reading turns have their history in memory at arrival
  (exact hints: 684 of 1,262). A turn that had not landed stalls 2.0-2.4 s, of which 0.7 s is the read itself.
- **The SSD is not the limit.** The read thread is busy 16-20 % of these runs and at most 55 % of any 5 minutes.
- **The room is.** Host memory beyond the GPU's mirror (the `idle` of the price log, every 5 s) is 54K tokens at the
  median in all three arms that log it, and 40-46K in the two load episodes with a tenth of intervals under 10K,
  while 17-23K tokens of reads are refused per interval there. A read loads 17-19K tokens, so two or three histories
  fit, staged or kept. That is `PAPER_PLAN.md`'s R of 44K, and it bounds what any read timing can hold early here.
- **The arrival hold costs little.** Reads sit parked after arrival for 0.08-0.13 s per turn, half of it with the
  read thread idle, but four in five of those turns waited 3.5-5.5 s more for the GPU after their read landed. The
  turns admitted as their held read landed carry 0.006-0.019 s per turn (`checks/holds.py`, `checks/predwait.py`).
- **`ec_r1`'s extra 0.10 s over `ec_r13` is GPU-side.** Its read-bound stall is the same; it recomputes more on reading
  turns and queues more (after admission 0.44 against 0.41 s, queue outside read-bound stall 0.30 against 0.22 s).
  That is what rule 3's order would be worth under rule 1, if it repeats.
- **JIT p50 loses everywhere except in read-bound stall.** On the 9,702 turns six arms share it is 0.76 s behind exact
  hints: 0.27 s after admission, 0.25 s in the queue and 0.23 s of stall, with 2,259 turns reading from SSD against
  1,065-1,340 in the four leading arms. Its read-bound stall (0.13 s) equals rule 1's: its staged reads land, and the room
  they hold is what costs. The hint-free arm is 1.23 s behind: 0.70 s in the queue, 0.32 s of stall, 0.21 s after
  admission.

**The new build against the control's, by code (13:40Z).** A separate reader traced `engine_owned_20261101` against
`engine_ec_20261111d` under the JIT p50 arm's flags (ten files differ, 478 changed lines in the prefetcher) and found no
reachable difference in behaviour: every new branch is behind `expected_cost`, `arrival`, `return_chance`, soft pins or
the landing quantile, and the statements that time reads, estimate waits, register next use and order eviction are the
same text. The one difference on the path is a stricter hint schema (a `p90` that is not a number rejects the table),
which this client never sends. I spot-checked three of its claims in the code. Static only; `jit_p50_d` is the run.

**Queue reordered 13:20Z** (unstarted lines only, flags unchanged; backup `queue.txt.bak_1320`, all lines re-validated):
`jit_p50_d` runs next because every reading of the new arms rests on that control, then the two repeats, `fq90`,
`ec_r134`, `ec_full`, `ec_r3`, and the two 128-user arms. It still ends about 10:00Z on 2026-11-12.

**`nofc_noend` (done 15:11Z, clean): an engine with no hints is level with the rival that has the end signal, and
0.05 to 0.15 s behind rules 1 and 3.** Settings checked in the log (`read_timing=arrival`, `return_chance` with score
`queue`, no staged read, no tracebacks, no client errors). It reads only at arrival and evicts by LRU with queued turns
last; an ended session ranks like any other.

| Mean TTFT | Turns | JIT exact | `ec_r13` | `ec_r1` | `nofc` | `nofc_noend` | Hint-free (ctl) |
|---|---|---|---|---|---|---|---|
| Returning turns all eleven runs share | 6,715 | 0.51 s | 0.66 s | 0.67 s | 0.69 s | 0.71 s | 1.89 s |
| Of them, positions 5,500-7,000 | 1,374 | 1.14 s | 1.56 s | 1.75 s | 1.83 s | 1.85 s | 7.43 s |
| Returning turns the five leading arms share | 11,198 | 0.70 s | 0.76 s | 0.86 s | 0.88 s | 0.91 s | |
| p90, p99 on the 11,198 | | 0.87, 10.6 s | 1.29, 9.2 s | 1.41, 12.2 s | 1.51, 13.3 s | 1.74, 11.6 s | |
| Turns over 10 s on the 11,198 | | 118 | 97 | 173 | 197 | 147 | |
| Turns reading from SSD on the 11,198 | | 1,262 | 1,582 | 1,572 | 1,581 | 1,898 | |
| SSD pages read, whole run | | 388K | 437K | 448K | 469K | 562K | 517K |

Tables: `compare_ec.out`, `split_ec_lead.out`, `stall_ec.out`, `split_ec_noend.out`, `stall_ec_noend.out`; figure
`../compare_c26_ec/ec_wait_split.png` (seven runs, table view `ec_wait_split.csv`).

- **With no hint at all the engine gets 96 % of the way from original HiCache to exact hints.** 0.71 s on the 6,715
  turns, against 5.47 and 6.62 s for original HiCache's two runs and 0.51 s with exact hints. The earlier hint-free
  arm is at 1.61 and 1.89 s (72-82 %). The 79 % of 2026-11-04 was that arm's figure.
- **The difference between the two hint-free arms is the queue-state mechanisms, about 1.0 s.** `nofc_noend` adds to
  the hint-free arm: queued turns' history evicted last, the arrival gate and hold, re-reads for queued turns. Paired
  on 9,703 shared turns the hint-free control is 1.02 s behind: 0.59 s in 812 turns that read from SSD only there,
  0.28 s in turns that read in both, 0.16 s in turns that read in neither. The two evict as much (67M and 47M GPU and
  host tokens against 66M and 50M), but 9 turns shrank while queued against 288, and reading turns recompute 585 tokens
  against 2,436.
- **The end signal is worth 0.03 s of TTFT, inside one run's noise, and a sixth of the SSD traffic.** Paired on the
  11,295 turns the two rivals share, `nofc_noend` is 0.028 s behind `nofc` (read stall +0.038 s, rest of the queue
  -0.009 s). Without the signal 1,914 turns read from SSD against 1,598 and the run reads 562K pages against 469K.
  Recomputation is level (145 against 131 tokens per turn on the 6,715, 203 against 185 on the 9,702). The eviction
  replay had put the signal at 2.5 points of read share (19.4 to 16.9 %); here it is 2.7 points (16.9 to 14.2 %).
- **What hints add over no hints at this setting, one run each.** Three-quantile hints under rules 1 and 3: 0.05 s on
  the first set and 0.14-0.15 s on the whole run (0.76 against 0.91 s on the 11,198, 0.79 against 0.93 s on 9,703), of
  which 0.10 s per shared turn in the second load episode; 22 % fewer SSD pages. Exact hints under JIT: 0.20 s and 31 %
  fewer pages. The repeats say how much of the 0.05-0.15 s is real.
- **Where it is behind exact hints:** 0.20 s per shared turn of read stall, nothing elsewhere (rest of the queue
  +0.01 s, after admission 0.00 s). 1,447 turns waited for nothing but their read (0.21 s per shared turn), against
  1,183 in `nofc` and 288 with exact hints.

**Queue changed 15:16Z** (unstarted lines only; backup `queue.txt.bak_1516`, all ten unstarted lines re-validated). The
128-user rival is now `nofc_noend_c128` on `engine_ec_20261111e`: it is the rival `PAPER_PLAN.md` says a reviewer will
ask for, and at 44 users it ties the one with the end signal. `nofc_c128` moves to the end, after `ec_r134_c128`, to
split the end signal from the forecast at 128 users; it is the line to drop if the time is wanted. Order from here:
`jit_p50_d` (running, to about 17:14Z), `ec_r13_rep`, `nofc_rep`, `fq90`, `ec_r134` (to about 01:26Z), `ec_full`,
`ec_r3`, `nofc_noend_c128` (from about 05:32Z), `ec_r134_c128`, `nofc_c128` (to about 11:41Z on 2026-11-12).

**How the repeats will be read (15:20Z, written before they run; `pairs_ec.out`, `pairs_ec_ctl.out`, script
`c26_ec/pairs_ec.py`).** The script sets every pair of runs against each other on the turns they share, with a block
bootstrap over 500 consecutive positions beside each difference, and lists what running the same arm again moved.

- **The leading arms differ only in the two load episodes.** On the 11,198 turns, outside positions 5,500-7,000 and
  8,750-11,000 the five are at 0.40-0.46 s. Inside (2,972 turns): exact hints 1.53 s, `ec_r13` 1.66, `ec_r1` 2.02,
  `nofc` 2.12, `nofc_noend` 2.14. So rules 1 and 3 are 0.46-0.48 s per turn ahead of both rivals under load and level
  with them otherwise.
- **Within one pair of runs** `ec_r13` is ahead of `ec_r1`, `nofc` and `nofc_noend` by 0.10, 0.11 and 0.14 s per turn
  (5 % to 95 %: 0.02-0.19, 0.02-0.23, 0.06-0.24), and not separated from exact hints (0.06 s, -0.01 to 0.14). `nofc`
  and `nofc_noend` are not separated (0.03 s, -0.06 to 0.11).
- **That interval is a floor.** It holds the turns' scatter inside two runs, not what a second run changes. Of the
  three controls that ran twice, the same arm moved by 0.25 s (JIT p50), 0.28 s (hint-free) and 1.15 s (original
  HiCache) on the 6,715 turns, all of it in the load episodes (0.005, 0.03 and 0.73 s elsewhere), and for two of the
  three the interval excludes zero.
- **Rule for the repeats.** A difference between two arms counts as shown when every pair of their runs has the same
  sign and the smallest of those differences is larger than the largest move between two runs of either arm.
  Otherwise the arms are not separated at this setting. `ec_r13_rep` and `nofc_rep` are read this way against
  `ec_r13`, `nofc` and `nofc_noend`.

**Outage 16:32Z to 20:39Z; relaunched 20:41Z.** The host went down 80 minutes into `jit_p50_d` and came back on another
GPU unit (`GPU-a7198925`, before `GPU-605dfc41`). The launcher was restarted as written at the top and began `jit_p50_d`
again from its first turn; the queue is unchanged and now ends about 17:10Z on 2026-11-12. Arms from here run on the new
unit, so the rerun is also the control on it.

**`jit_p50_d`, the 80 minutes before the outage (`compare_ec_cut.out`, `pairs_ec_cut.out`, `checks/hw_window.out`): the
new build does not move JIT p50.** The cut run is intact (7,241 turns, no errors, no tracebacks, no broken line) and is
read as `jit_p50_d_cut` through `extra_runs.txt`, which the comparison scripts take only when the label is named. Turns
sent in its first 74.7 minutes are kept, which covers the first load episode.

| On the 6,579 returning turns these runs share | JIT p50 (campaign 23) | `jit_p50_ctl` | `jit_p50_d_cut` | `nofc_noend` | `ec_r13` | JIT exact |
|---|---|---|---|---|---|---|
| Mean TTFT | 1.53 s | 1.35 s | 1.39 s | 0.71 s | 0.66 s | 0.51 s |
| In the first load episode (1,275 turns) | 5.16 s | 4.20 s | 4.46 s | 1.94 s | 1.66 s | 1.20 s |
| Elsewhere (5,304 turns) | 0.66 s | 0.66 s | 0.66 s | 0.42 s | 0.42 s | 0.34 s |
| Turns reading from SSD | 1,562 | 1,535 | 1,462 | 1,040 | 925 | 651 |
| Recomputed tokens per turn | 922 | 913 | 902 | 147 | 211 | 122 |

- **Same build or not, JIT p50 is JIT p50.** The new build is 0.05 s from the old build's run on this GPU unit (5 % to
  95 % within the pair: -0.04 to 0.15 s) and 0.14 s from campaign 23's; the two old-build runs are 0.18 s apart. Outside
  the load episode the three agree to 0.004 s. Its first staged read was for the same turn, 9 min 54 s after the first
  arrival, as in `jit_p50_ctl`.
- **So the new arms' lead over JIT p50 is not the build.** Each of `nofc_noend`, `ec_r13` and exact hints is 0.64 to
  1.02 s ahead of each of the three JIT p50 runs, three to five times what a second JIT p50 run moves. By the rule
  above that difference is shown.
- **Hardware reads the same in the same minutes.** In each run's first 74.7 minutes the decode step is 33.6, 46.0 and
  60.3 ms at 8-15, 16-23 and 24-31 running requests against 34.8, 46.1 and 61.0 ms in `jit_p50_ctl`; first-turn
  prefill 0.091 against 0.092 ms per token. The whole-run table shows the cut run faster only because steps lengthen
  as contexts grow later in a run.

**Figures in the earlier campaigns' styles (21:17Z, by request; all in `../compare_c26_ec/`).**

- `pricing_vs_jit.png`, `.pdf`, `.csv` (script `c26_ec/plot_pricing_vs_jit.py`): the over-time figure in the style of
  `scripts/jit_vs_hicache.py`'s `hicache_vs_jit.png` (it imports that script's helpers). Five runs: original HiCache,
  `jit_p50_ctl`, `nofc_noend`, `ec_r13`, JIT exact; four panels in 5-minute bins: mean TTFT, turns per second, history
  recomputed per returning turn, and the host memory needed to hold SSD-loaded KV until admission against the 44K
  free beyond the GPU's copy. `--arms` and `--panels` choose others (`unused` is the earlier figure's fourth panel).
  Its footnote is over the 7,283 returning turns these five share: 5.48, 1.33, 0.69, 0.64 and 0.51 s.
- Minutes 45-120 the reads waiting for admission need 80K tokens in original HiCache, 91K in JIT p50, 6K with no
  hints, 17K under rules 1 and 3 and 10K with exact hints; they are above the 44K for 68, 80, 5, 13 and 8 % of the
  seconds. The measure is that script's `held_tokens` and gives its 80K and 10K for the two runs both figures show.
- `timeline_<label>.png` for each run, `compare.png`, `compare_steady_1800s.png`, `ttft_over_time_60s.png` and
  `paired_cdf_<label>.png` against `jit_p50_ctl` (script `c26_ec/plot_timeline.py`): `scripts/timeline.py`'s standard
  set, unchanged. The comparison figures hold six runs (the five above and `hintfree_ctl`); a run landing later gets
  its timeline with `plot_timeline.py <label>`.

**`jit_p50_d` (rerun done 22:46Z on 2026-11-11, clean): JIT p50 on the new build and on the new GPU unit is JIT p50.**
1.48 s on the 6,715 turns, between `jit_p50_ctl`'s 1.35 s and campaign 23's 1.60 s. Its mechanisms repeat: 2,149
staged reads (2,094 and 2,202), 1,761 launches refused room (1,743 and 1,732), 28.8K tokens pinned on average (30.2K and
31.5K), 24.3 % of turns reading from SSD (23.8 and 23.7 %). The two GPU units read alike in the same minutes
(`checks/hw_window_units.out`, `hw_window_units60.out`): decode step 35.7, 47.3 and 62.3 ms against 36.8, 48.4 and
64.3 ms for `jit_p50_ctl`, first-turn prefill 0.088 against 0.091 ms per token.

**`ec_r13_rep` (done 00:49Z on 2026-11-12, clean): rules 1 and 3 again are 0.10 to 0.12 s behind their first run, so
their lead over the rivals is not shown.** Settings and events checked as for `ec_r13`. Tables: `compare_ec.out`,
`pairs_ec.out` (11,145 turns six runs share), `pairs_ec_ctl.out` (the 6,715), `stall_ec.out`.

| Mean TTFT | Turns | JIT exact | `ec_r13` | `ec_r13_rep` | `ec_r1` | `nofc` | `nofc_noend` |
|---|---|---|---|---|---|---|---|
| Returning turns all thirteen runs share | 6,715 | 0.51 s | 0.66 s | 0.78 s | 0.67 s | 0.69 s | 0.71 s |
| Of them, positions 5,500-7,000 | 1,374 | 1.14 s | 1.56 s | 2.31 s | 1.75 s | 1.83 s | 1.85 s |
| Returning turns these six share | 11,145 | 0.70 s | 0.76 s | 0.86 s | 0.86 s | 0.88 s | 0.91 s |
| Of them, in the two load episodes | 2,972 | 1.53 s | 1.66 s | 2.10 s | 2.02 s | 2.12 s | 2.14 s |
| Of them, elsewhere | 8,173 | 0.40 s | 0.44 s | 0.41 s | 0.45 s | 0.43 s | 0.46 s |
| p99; turns over 10 s on the 11,145 | | 10.6 s; 118 | 9.2 s; 97 | 13.7 s; 196 | 12.2 s; 173 | 13.3 s; 197 | 11.6 s; 147 |
| Turns reading from SSD; pages read on the 11,145 | | 1,260; 382K | 1,576; 424K | 1,559; 414K | 1,568; 446K | 1,576; 458K | 1,893; 553K |

- **By the rule written before the repeats, rules 1 and 3 are not separated from either rival.** On the 6,715 turns
  their two runs are 0.035 s ahead to 0.085 s behind `nofc`, and 0.056 s ahead to 0.064 s behind `nofc_noend`. On
  the 11,145 they are ahead in every pair (0.02-0.11 s and 0.05-0.15 s), but the arm itself moved 0.10 s between its
  runs. The first run's lead in the load episodes (1.66 against 2.1 s) did not come back (2.10 s).
- **What a second run of one arm moves, on the 6,715 turns:** original HiCache 1.15 s, the hint-free arm 0.28 s, JIT
  p50 0.12, 0.13 and 0.25 s (three pairs), rules 1 and 3 0.12 s. Outside the load episodes the pairs agree to
  0.01-0.08 s, original HiCache's excepted (0.73 s).
- **The policy's own behaviour repeats; the GPU queue in the episodes does not.** Staged reads 560 (629), decisions
  left to arrival 566 (535), launches refused 420 (395), mean pins 4.6K (4.2K), 14.0 % of turns reading (14.3 %).
  The repeat's extra wait is stall on turns the GPU queue held as well (0.17 against 0.11 s per turn) and queue wait
  outside the read (0.16 against 0.11 s); its read-bound stall is no larger (0.11 against 0.13 s).
- **What does repeat is the SSD traffic.** Both runs read 414-424K pages on the shared turns, against 458K with the
  end signal alone and 553K with no hint: 7-10 % and 23-25 % fewer.
- **What is shown at this setting, by the same rule:** every arm against original HiCache (3.6 s or more against a
  rerun move of 1.15 s); `nofc`, `nofc_noend`, rules 1 and 3 and exact hints against the hint-free arm (0.84-1.38 s
  against 0.28 s) and against JIT p50 (0.58-1.09 s against 0.25 s). Not shown: JIT p50 against the hint-free arm,
  and anything among `nofc`, `nofc_noend`, `ec_r1` and rules 1 and 3. Exact hints are 0.15-0.27 s ahead of both
  runs of rules 1 and 3 and 0.20 s ahead of `nofc_noend`, on one run of 2026-11-01; `jit_exact_d` reruns it.

**Turn latency and throughput (`checks/latency_lead.out`, `latency_all.out`, script `checks/latency.py`).** A reviewer
agent raised it and I recomputed it: among the leading arms the TTFT differences do not reach the end of the turn.

| On the 11,145 turns | `nofc_noend` | `nofc` | `ec_r13` | `ec_r13_rep` | `ec_r1` | JIT exact |
|---|---|---|---|---|---|---|
| Mean TTFT | 0.91 s | 0.88 s | 0.76 s | 0.86 s | 0.86 s | 0.70 s |
| Mean turn latency, send to last token | 12.10 s | 12.01 s | 12.05 s | 12.45 s | 12.32 s | 12.06 s |
| Turns sent before minute 115 | 11,680 | 11,710 | 11,690 | 11,527 | 11,580 | 11,699 |

- **Against the arm with no hints, no arm is ahead on latency by more than 0.1 s:** `nofc` -0.09 s, `ec_r13` -0.05 s,
  exact hints -0.04 s, `ec_r1` +0.22 s, `ec_r13_rep` +0.35 s. Throughput is level: 11,527 to 11,710 turns and 2.34 to
  2.37 M completion tokens before minute 115. A turn generates 205 tokens over about 11 s, so 0.1-0.2 s at the first
  token is 1-2 % of the turn.
- **Today's arms lose more at the end of the turn than at its first token.** On the 6,715 turns JIT p50 is 0.64-0.77 s
  behind `nofc_noend` in TTFT and 3.7-4.3 s behind in latency (15.3-15.9 against 11.6 s); the hint-free arm 1.18 s
  and 4.2 s; original HiCache 4.8-5.9 s and 14-18 s (25.8-29.9 s per turn).

**An outside reading of the evidence (`checks/verify/reviewer/result.json`; one agent acting as a programme-committee
reviewer, its scripts beside it).** Its points that change what runs or what can be claimed, in its order:

1. The gain needs no forecast: the engine with no hints should be the baseline of record, and the hint's worth
   stated over it in seconds beside the rerun move. `PAPER_PLAN.md`'s claim and `PAPER_INTRO.md`'s "2.9x from exact
   return times" do not survive that rival (1.4x, exact hints only).
2. TTFT of returning turns is not what the agent gets: latency and throughput (above).
3. One run per arm cannot carry 0.05-0.15 s. Now measured: rules 1 and 3 moved 0.12 s.
4. The 44K-token room is the 70 GB pool minus the GPU's mirror; sessions in progress hold a median 1.15 M tokens. A
   reviewer will ask for a larger pool or no mirror before pricing 44K. Not run: a host-pool sweep at 44 users.
5. Neither setting stresses the SSD (the read thread is 16-21 % busy at 44 users; the GPU queue decides at 128). The
   setting where read timing and eviction order should matter is x70 gaps at 128 users. Not run.
6. No rule is isolated: rule 1 mostly declines to act and rule 3 reads as often as LRU with ended sessions first.
7. The exact-hint reference is one run on another build and unit: `jit_exact_d` is queued.
8. The price was fitted on the turns it is scored on and has no sensitivity run.
9. One trace, one seed.
10. At 128 users the fair rival has not run: `nofc_noend_c128` now runs next.
11. The rival still takes the session id and the reply key from the hint channel: call it "session id only".
12. Rules 1 and 3 have recompute bursts of their own (one block of 500 turns recomputed 183K tokens against 15-18K).

Its numbers that I re-derived agree with mine (the shared sets, the means, latency); the other figures in this
list are its own and unchecked.

**Queue changed 01:38Z** (unstarted lines only; backup `queue.txt.bak_0137`; all ten lines re-validated). Order from here:
`nofc_rep` (running, to about 02:51Z), `nofc_noend_c128` (the number the 128-user claim hangs on), `jit_exact_d`
(added: exact hints on this build and unit), `ec_r134`, `nofc_noend_rep` (added), `ec_r134_c128`, `ec_full`, `fq90`,
`ec_r3`, `nofc_c128`; it ends about 21:20Z on 2026-11-12. `ec_r3` stays: it asks whether eviction order alone repairs
JIT p50, which a single run can answer; the two arms that would complete rule 1 by rule 3 cannot be told apart at
0.1 s per run and are not queued.

**Independent recomputation (2026-11-12 02:23Z; `checks/verify/client/` and `checks/verify/server/`, each with its own
parsers, frozen numbers and `result.json`).** Two agents re-derived the tables from the raw logs before opening any of
my outputs. Every number they could set against a printed one agrees: 157 of 157 client-side (shared sets, means,
percentiles, paired differences, shares of the gain) and 1,946 of 1,946 server-side (reading turns, pages, queue wait,
the stall and its split, evictions). The two that were to audit the scripts and test other explanations were cut off
twice by the usage limit; what they had computed is in `checks/verify/method/` and `checks/verify/confounds/`.

- **The runs are comparable turn by turn.** No duplicate keys, no errors, no turn censored by the 300 s rule; a key's
  prompt is within 1 token in 89 % of cases across the five leading runs (0.13 % at most); every run starts from an
  empty cache; the arms' server arguments differ only in the flags meant to differ and the sampling seed. In the two
  load episodes the arms were offered the same load (85-91 turns per minute in the first, 66-72 in the second) and
  held nearly the same sessions (Jaccard 0.96-0.98). Decode and prefill speed show no drift over the day.
- **Clocks agree.** In the arms that read only at arrival no read starts before arrival (earliest 25 ms after); queue
  wait never exceeds the client's TTFT.
- **Corrected in the scripts.** Three column heads of `compare_ec.out` said less than the column holds ("s2h reads"
  is turns that read; the first staged and refused counts are over the whole log). Turns past the order run's last
  position (1,218 of the 11,198, the leading arms' last 15 minutes) had been one bootstrap block; they now take
  blocks of their own, which moves the intervals by 0.01 s at most. `stall_ec.out` now holds six runs; the five-run
  table the 13:20Z entry cites is `stall_ec_lead5.out`.
- **Readings to hold more loosely.** The split of the stall by "admitted within 0.1 s" moves with that bound: the
  part where the read was the last wait is 0.14 to 0.20 s per turn in `nofc` for bounds of 0.05 to 0.5 s, so "what
  read timing can win" is that range, not 0.17 s. "After admission" is TTFT minus queue wait and includes 0.13 s
  between the client's send and the scheduler's entry. Read I/O after arrival is 21-34 % of the stall; the rest is
  time parked before the launch and waiting for the queue. One or two turns per arm sit exactly on the 0.05 s
  "landed" threshold and change class with float rounding.

**What a read costs when the queue is short, and what a larger pool would change (02:38Z; `checks/calm_reads.out`,
script `checks/calm_reads.py`).** Asked by the author: with a short queue and more host memory, do early reads pay?

| Outside the load episodes, 8,174 shared returning turns | `nofc_noend` | `nofc` | `ec_r13` | `ec_r13_rep` | JIT exact |
|---|---|---|---|---|---|
| Turns reading from SSD | 809 (9.9 %) | 577 (7.1 %) | 678 (8.3 %) | 612 (7.5 %) | 439 (5.4 %) |
| Their TTFT, mean and median | 1.60, 1.18 s | 1.60, 1.19 s | 1.47, 0.99 s | 1.36, 0.97 s | 1.22, 0.34 s |
| TTFT of the other turns | 0.34 s | 0.34 s | 0.34 s | 0.34 s | 0.35 s |
| All | 0.46 s | 0.43 s | 0.44 s | 0.41 s | 0.40 s |

- **With a short queue a read at arrival costs the turn about 1.2 s, and an early read timed right removes it:** the
  median reading turn is at 1.18 s with no hints and at 0.34 s with exact hints, the level of a turn that read
  nothing. Rules 1 and 3 on three quantiles are at 0.97-0.99 s: at this price few of their reads land early.
- **Per turn that is 0.06 s, because 5-10 % of turns read.** The gain of early reads is the number of reads times
  about 1.2 s; more room lets more of them start early but leaves fewer to start.
- **The pool sets how many reads there are.** Sessions in progress hold a median 1,176K tokens (1,050K-1,302K for the
  middle 80 %, 1,443K at most). A 70 GB pool holds 712K, a 100 GB pool 1,017K (below the live context 93 % of the
  time), a 160 GB pool 1,628K (above it always). So at 160 GB no live session's history is on SSD and nothing is
  left to read early; at 100 GB about a third as much is on SSD as at 70 GB, with 349K tokens free beyond the GPU's
  copy against 44K.
- **Appended to the queue (backup `queue.txt.bak_0238`, validated), after everything else: 100 GB at 44 users,**
  `nofc_noend_h100`, `jit_p50_h100`, `jit_exact_h100`: no early reads, early reads from the p50, early reads from
  exact hints. If room is what JIT p50 lacks, it should close on exact hints there. Labels ending `_h<GB>` stay out of
  the 70 GB tables unless named. The setting with many reads, a short queue and room at once is x70 gaps at 128
  users and 160 GB; it is not queued.

**`nofc_rep` (done 02:51Z on 2026-11-12, clean): the rival with no gap forecast again.** 0.80 s on the 6,715 turns
(first run 0.69 s) and 0.86 s on the 11,145 turns the seven leading runs share (first run 0.88 s). Its mechanisms repeat:
14.7 % of turns reading from SSD (14.2 %), 297 launches refused room (277), 2.4K tokens pinned on average (2.6K), 473K
pages read on the shared turns (458K). Tables: `compare_ec.out`, `pairs_ec.out`.

| Mean TTFT | JIT exact | `ec_r13`, repeat | `ec_r1` | `nofc`, repeat | `nofc_noend` |
|---|---|---|---|---|---|
| 6,715 turns | 0.51 s | 0.66, 0.78 s | 0.67 s | 0.69, 0.80 s | 0.71 s |
| 11,145 turns | 0.70 s | 0.76, 0.86 s | 0.86 s | 0.88, 0.86 s | 0.91 s |
| of them in the two load episodes | 1.53 s | 1.66, 2.10 s | 2.02 s | 2.12, 1.95 s | 2.14 s |

- **Two runs of an arm now span as much as the arms do.** On the 6,715 turns rules 1 and 3 moved 0.12 s and the
  rival 0.11 s, and the four runs interleave (0.66, 0.69, 0.78, 0.80 s). On the 11,145 the rival moved 0.01 s and
  rules 1 and 3 0.10 s; over the four pairs rules 1 and 3 are 0.003 to 0.114 s ahead.
- **So the verdict of `ec_r13_rep` stands with both repeats in:** nothing among `nofc`, `nofc_noend`, `ec_r1` and
  rules 1 and 3 is shown in TTFT at this setting; the fewer SSD pages under rules 1 and 3 are (414-424K against
  458-473K and 553K).

**`nofc_noend_c128` (done 04:55Z on 2026-11-12, clean): the rival with no hints at 128 users and 160 GB.** Numbers
only, as for the other 128-user arms. Table: `compare_c128.out` (the same table before this arm:
`compare_c128_pre.out`). The other rows are campaigns 18, 20, 23 and 25, on earlier builds and GPU units.

| First 6,900 s | Turns | Mean TTFT | Queue wait | Turn latency | SSD pages | Turns with over one read |
|---|---|---|---|---|---|---|
| original HiCache | 8,334 | 50.20 s | 49.13 s | 73.96 s | 433K | 1 |
| hint-free, read pins | 10,223 | 38.79 s | 37.95 s | 57.45 s | 1,063K | 0 |
| early reads, exact hints (`oracle`) | 15,525 | 20.58 s | 20.15 s | 32.85 s | 1,438K | 36 |
| early reads, p50 (`p50`) | 15,473 | 20.68 s | 20.26 s | 32.98 s | 1,441K | 162 |
| JIT exact | 15,384 | 20.93 s | 20.50 s | 33.32 s | 998K | 0 |
| JIT p50 | 15,366 | 20.98 s | 20.55 s | 33.39 s | 1,195K | 120 |
| `nofc_noend_c128` | 15,418 | 20.84 s | 20.41 s | 33.19 s | 1,153K | 0 |

- **On every column it lands inside the range of the four hinted arms.** In SSD pages it is 16 % above JIT exact and
  4 % below JIT p50. 25.9 % of its returning turns read from SSD, one read each; 33 tokens recomputed per returning
  turn (33-36 in the hinted arms).
- On the 8,001 returning turns all seven arms sent: mean TTFT 12.07 s (hinted arms 11.73-12.23 s, hint-free 27.74 s,
  original HiCache 50.55 s), turn latency 24.13 s (23.67-24.32 s, 44.72 s, 74.67 s).
- Paired on the returning turns it shares with each hinted arm (14.8K): +0.29 s against `oracle`, +0.17 s against
  `p50`, -0.10 s against JIT exact, -0.16 s against JIT p50. Against the hint-free arm, 23.6 s faster on 9,840.
- One run of this arm. `checks/boot.py` now prints an arm's resolved flags, pools, banners and error counts in one
  call.

**`jit_exact_d` (done 06:59Z on 2026-11-12, clean): exact hints on the new build and GPU unit repeat campaign 23's
run.** This is point 7 of the outside reading. Tables: `compare_ec.out`, `pairs_ec.out` (11,145 turns the eight leading
runs share; the seven-run table earlier entries cite is kept as `pairs_ec_7runs.out`), `pairs_ec_ctl.out` (the 6,715
turns all fifteen runs share), `checks/latency_lead8.out`.

| Mean TTFT | JIT exact, repeat | `ec_r13`, repeat | `ec_r1` | `nofc`, repeat | `nofc_noend` |
|---|---|---|---|---|---|
| 6,715 turns | 0.51, 0.54 s | 0.66, 0.78 s | 0.67 s | 0.69, 0.80 s | 0.71 s |
| 11,145 turns | 0.70, 0.77 s | 0.76, 0.86 s | 0.86 s | 0.88, 0.86 s | 0.91 s |
| of them in the two load episodes | 1.53, 1.88 s | 1.66, 2.10 s | 2.02 s | 2.12, 1.95 s | 2.14 s |
| of them elsewhere | 0.40, 0.37 s | 0.44, 0.41 s | 0.45 s | 0.43, 0.47 s | 0.46 s |

- **The reference holds.** Its two runs are 0.04 s apart on the 6,715 turns and 0.07 s on the 11,145, all of it in the
  load episodes (+0.16 and +0.35 s there, 0.00 and -0.03 s elsewhere). Its mechanisms repeat: 11.5 % of turns
  reading from SSD (11.3 %), 1,207 staged reads (1,202), 376 launches refused room (381), 6.0K tokens pinned on
  average (5.9K), 389K pages on the shared turns (382K). Hardware control as the other arms.
- **Exact hints against the two rivals without a forecast: shown, by the rule written before the repeats.** They are
  ahead in every pair of runs. Of `nofc` by 0.15-0.29 s on the 6,715 turns and 0.09-0.18 s on the 11,145, against
  rerun moves of 0.04-0.11 s and 0.01-0.07 s. Of `nofc_noend` by 0.17-0.21 s and 0.14-0.21 s; that arm has one run,
  and `nofc_noend_rep` is queued.
- **Exact hints against rules 1 and 3 on three quantiles: not shown.** On the 6,715 turns the four pairs are
  0.11-0.27 s apart while rules 1 and 3 moved 0.12 s between their own runs; on the 11,145 one pair reverses (-0.01
  to +0.16 s). `ec_r1`, with one run, is 0.09-0.17 s behind both runs of exact hints.
- **Turn latency and throughput still do not follow TTFT among the leading arms.** On the 11,145 turns 12.04 s (first
  run 12.06 s, `nofc_noend` 12.10 s); 11,703 turns sent before minute 115 (11,699 and 11,680).
- `ec_r134` (rules 1 and 3 with the learned price) booted with its settings at 07:00Z and ends about 09:02Z.

**Outage 07:07Z to 15:31Z on 2026-11-12; relaunched 15:36Z.** The host went down 6.5 minutes into `ec_r134` (461 turns,
no errors; too short to read, so it is not listed in `extra_runs.txt`) and came back on a third GPU unit
(`GPU-6374d407`). The launcher was restarted as written at the top and began `ec_r134` again from its first turn. The
queue is unchanged: `ec_r134`, `nofc_noend_rep`, `ec_r134_c128`, `ec_full`, `fq90`, `ec_r3`, `nofc_c128`,
`nofc_noend_h100`, `jit_p50_h100`, `jit_exact_h100`; it now ends about 12:10Z on 2026-11-13. Arms from here run on the
new unit; each table's hardware control line says whether it reads as the other two.

**A second figure set, `results/compare_c26_designs/` (17:08Z on 2026-11-12, asked for).** 44 users, 70 GB, 0.1 users/s,
five arms: original HiCache, the old oracle (v2, exact hints: campaign 21's run, which is also the order run), JIT p50, JIT
exact and the pricing rules. JIT and the rules are their runs on `engine_ec_20261111d` (`jit_p50_d`, `jit_exact_d`,
`ec_r13`), so the three differ in flags only; one run per arm in the comparisons. The other rule arms that have landed
(`ec_r1`, `ec_r13_rep`) get a timeline and a row in the wait split, and later ones join as they land.

- Same figure kinds and file names as `compare_c26_ec/`. `ec_overview.png` there draws every run of the five arms, and
  the footnote of `pricing_vs_jit.png` gives the other runs' means on the same turns.
- On the 6,715 returning turns every run of the five arms sent: 5.47, 1.53, 1.48, 0.54 and 0.66 s; the other runs 6.62 s
  (HiCache), 1.35 and 1.60 s (JIT p50), 0.51 s (JIT exact), 0.78 s (rules 1 and 3). The old oracle's 1.53 s on this set
  is level with JIT p50. Wait split on the 7,283 turns the seven runs drawn share: `stall_ec_designs.out`.
- One command redraws it: `docker exec sglang_hicache bash .../c26_ec/figs_designs.sh` (about a minute). The plot
  scripts take the set from `FIGSET=designs` (`--set designs` for `plot_pricing_vs_jit.py`); `oracle_v2` is listed in
  `extra_runs.txt`, so the comparison scripts take it when named. The first set still reproduces to the byte (checked
  on `pricing_vs_jit`).

**What still limits the best arms (17:35Z on 2026-11-12, asked for; scripts and outputs in `checks/limits/`).** Whole
runs of `jit_exact_d` and `nofc_noend`, returning turns. Mean TTFT of `jit_exact_d` 0.82 s (median 0.27 s) = 0.14 s
before the scheduler has the request + 0.38 s in its queue + 0.29 s after admission.

- **One turn in a hundred computes its history again, and that is the largest loss left.** 123 returning turns (1.0 %)
  recompute 4K tokens or more: 11.1 s mean TTFT against 0.71 s for the rest, 14 % of all TTFT seconds, and 2.6M
  tokens, 23 % of the run's prefill work. 72 % of them fall in fourteen minutes, eleven of which are among the
  fourteen with the most queue seconds. `nofc_noend`: 113 turns, 2.4M tokens, 22 %.
- **Their history was on SSD and host memory had no room for it.** Of the 123: 58 had the read refused room and read
  nothing (58 % of the tokens; median 20K tokens on SSD, 2.1 refusals), 25 read it and lost it before admission
  (21 %), 31 read less than the history (11 %). `nofc_noend`: 50, 22 and 34. The room beyond the GPU's copy is 44K
  tokens, about two such histories. No staged read is involved in `nofc_noend`, so rule 2 cannot be what frees it:
  staged pins hold 2-6K tokens on average.
- **The queue is fourteen minutes.** They hold 89 % of `jit_exact_d`'s queue seconds (65 % of `nofc_noend`'s): minutes
  53-62, 86-91 and 116-119, GPU pool 0.8-0.96 full, in most of them 3-9 turns a minute with 8K or more tokens to
  prefill against 0.7 otherwise. Outside them `nofc_noend` queues 24 s a minute against 6 s: its reads at arrival, which is what
  exact hints save.
- **Before the scheduler:** median 0.08 s, mean 0.14 s, growing with the prompt (0.04 s under 8K tokens, 0.11 s at
  32-64K, 0.17 s above); 17 % of all TTFT seconds. Not profiled; the server tokenizes the whole history each turn
  with one tokenizer worker.
- **After admission:** 0.14 s with under 128 new tokens, 0.23 s at 512-2K, 0.51 s at 2-8K. A prefix loaded from host
  costs 5-45 ms more than one on the GPU, so staging onto the GPU would buy little.
- **Decode:** 37, 49 and 62 ms a step at 8-15, 16-23 and 24-31 running requests, so going from about 20 to 28 running
  adds about 11 % tokens per second while holding 40 % more of the GPU pool.

Read: the next gain at this setting is room for an arrived turn's read (the host pool mirrors the GPU, so 668K of
its 712K tokens are copies of what the GPU holds), not the timing of early reads. The 100 GB arms (room 349K) are the
first test; `PAPER_PLAN.md` item 4 already names the mirror.

**`ec_r134` (done 17:42Z on 2026-11-12, clean, GPU unit `GPU-6374d407`): the learned price buys nothing over the fixed
one.** Rules 1 and 3 with the price set by measured harm. Tables: `compare_ec.out`, `pairs_ec.out` (now the 11,067
turns the nine leading runs share; the eight-run table is kept as `pairs_ec_8runs.out`), `pairs_ec_ctl.out`,
`stall_ec_designs.out`. Hardware control as the other arms.

| Mean TTFT | JIT exact, repeat | `ec_r13`, repeat | `ec_r134` | `ec_r1` | `nofc`, repeat | `nofc_noend` |
|---|---|---|---|---|---|---|
| 6,715 turns | 0.51, 0.54 s | 0.66, 0.78 s | 0.80 s | 0.67 s | 0.69, 0.80 s | 0.71 s |
| 11,067 turns | 0.70, 0.77 s | 0.77, 0.86 s | 0.89 s | 0.87 s | 0.88, 0.87 s | 0.91 s |
| of them in the two load episodes | 1.53, 1.88 s | 1.66, 2.10 s | 2.08 s | 2.02 s | 2.12, 1.95 s | 2.14 s |
| of them elsewhere | 0.40, 0.37 s | 0.44, 0.41 s | 0.45 s | 0.44 s | 0.42, 0.47 s | 0.46 s |

- **Not separated from the fixed price or from the rivals, by the rule for repeats.** Behind both runs of rules 1 and
  3 (0.02 and 0.14 s on the 6,715 turns, 0.03 and 0.12 s on the 11,067), with the smaller gap inside that arm's own
  rerun move. Against `nofc_noend` 0.08 s behind on the 6,715 and 0.02 s ahead on the 11,067. Exact hints are ahead of
  it in both pairs (0.25-0.29 s and 0.12-0.19 s). One run.
- **The price went down, and the arm spent more room on early reads.** From 5e-5 it moved between 5.7e-7 and 7.5e-5,
  median 3.1e-5, last 2.4e-5 (1,416 updates, 76 % of them below 5e-5). Against the fixed price's two runs: 838 staged reads (629, 560), 8.3K
  tokens pinned on average (4.2K, 4.6K), 654 launches refused room (395, 420), 58 landed reads lost (39, 49), 320
  tokens recomputed per turn on the 6,715 (214, 253). 11,446 turns kept (11,690, 11,527).
- This fits the entry above: what is short is room for an arrived turn's read, and a lower price hands more of it to
  reads that have not been asked for yet.
- `compare_c26_designs/` redrawn with its timeline and its bar in the wait split (0.77 s on the 7,283 turns).
  `nofc_noend_rep` booted with its settings at 17:43Z and ends about 19:45Z.

**`compare_c26_designs/`: `ec_r134` is now a compared arm (17:58Z on 2026-11-12, asked for).** Six arms in `compare.png`,
`compare_steady_1800s.png`, `ttft_over_time_60s.png`, `pricing_vs_jit.png` and `ec_overview.png`, and a
`paired_cdf_ec_r134.png` against JIT p50. The pricing arms share the orange and differ by dash: rules 1 and 3 solid,
with rule 4 dash-dot, all four rules dotted. `ec_full` is wired the same way and joins on the next
`figs_designs.sh` after it lands. The first set still reproduces to the byte.

**Why the learned price does no better than the fixed one (asked; `checks/limits/price_path_ec_r134.out`,
`recompute_rules.out`, `why_recompute_rules.out`).** On TTFT the arm equals the repeat of rules 1 and 3 (0.80 against
0.78 s), so "worse" is not shown; what is outside both fixed-price runs is its behaviour.

- **The price is lowest just before each load episode.** After a calm stretch with no harm to charge it had decayed to
  3e-6 at minutes 48-51 and to 2e-6 to 6e-6 at minutes 78-87, a fifteenth of the fixed 5e-5. It was back near 5e-5 by
  minutes 54-57 and 90-91, after the refusals.
- **So the arm entered the episodes with the room already taken by early reads.** Staged pins at minutes 46-51: 17K,
  35K and 22K tokens against 0.1K, 0.0K and 2.5K under the fixed price; refusals began there (204K, 94K and 432K
  tokens against 0, 0 and 18K). Before the second episode, 8-19K against 0-4K.
- **Whole run against the two fixed-price runs:** 215 turns recomputing 4K tokens or more (136, 148), 111 of them with
  the read refused room (72, 74); 4.0M tokens of history recomputed, a third of all prefill (2.8M, 3.1M); harm charged
  1,081 s (457, 529). What it bought: 206 reads landed before arrival (130, 134), and 0.07 s a turn waiting for the read
  alone (0.09, 0.10).
- **The rule sets the price to average harm per staged token-second, and that ratio does not rise with more staging:**
  2.7e-5 here against 2.9e-5 and 3.9e-5 at the fixed price, with harm doubled and staged pins nearly tripled (5.7K
  against 2.2K and 1.9K tokens). Nothing in it stops the price from falling when the room is idle, and the room is
  idle most of the time (80K tokens free outside minutes 52-64 and 85-92, 25K inside).
- The deck's own form of rule 4 would go further the same way: replayed on `ec_r13`'s log it fell to its floor within
  a minute. A price for this room has to depend on the room as it is now or as forecast, not on the last two minutes.

**`nofc_noend_rep` (done 19:45Z on 2026-11-12, clean, GPU unit `GPU-6374d407`): the rival with no hints repeats to
0.01 s.** Tables: `compare_ec.out`, `pairs_ec.out` (the same 11,067 turns, now shared by ten leading runs; the nine-run
table is kept as `pairs_ec_9runs.out`), `pairs_ec_ctl.out` (the 6,715 turns all seventeen runs share; before:
`pairs_ec_ctl_16runs.out`), `checks/latency_lead10.out`. Hardware control as the other arms.

| Mean TTFT | JIT exact, repeat | `ec_r13`, repeat | `ec_r134` | `ec_r1` | `nofc`, repeat | `nofc_noend`, repeat |
|---|---|---|---|---|---|---|
| 6,715 turns | 0.51, 0.54 s | 0.66, 0.78 s | 0.80 s | 0.67 s | 0.69, 0.80 s | 0.71, 0.72 s |
| 11,067 turns | 0.70, 0.77 s | 0.77, 0.86 s | 0.89 s | 0.87 s | 0.88, 0.87 s | 0.91, 0.91 s |
| of them in the two load episodes | 1.53, 1.88 s | 1.66, 2.10 s | 2.08 s | 2.02 s | 2.12, 1.95 s | 2.14, 2.12 s |
| of them elsewhere | 0.40, 0.37 s | 0.44, 0.41 s | 0.45 s | 0.44 s | 0.42, 0.47 s | 0.46, 0.47 s |

- **This arm repeats more closely than any other.** +0.005 s on the 6,715 turns and +0.001 s on the 11,067 (5 % to
  95 %: -0.03 to +0.03 s), in the load episodes too (-0.01 and -0.02 s). The three hinted arms with a repeat moved
  0.04-0.12 s on the 6,715 and 0.01-0.10 s on the 11,067. Whole run: 11,645 turns (11,680), 559K SSD pages (562K),
  17.0 % of turns reading from SSD (16.9 %), 301 launches refused room (293), 2.5K tokens pinned (2.5K). One pair
  of runs, so how small its move is may be chance.
- **Exact hints against no hints: shown, now with two runs on each side.** Ahead in all four pairs, by 0.17-0.21 s on
  the 6,715 turns and 0.14-0.21 s on the 11,067, against rerun moves of 0.04 and 0.07 s.
- **Rules 1 and 3 against no hints: not shown.** On the 6,715 turns the four pairs run from -0.06 to +0.06 s. On the
  11,067 the rules are ahead in all four (0.05-0.14 s), but the smallest is under their own rerun move (0.10 s).
- **The end signal (`nofc` against `nofc_noend`): not shown on TTFT, shown on SSD traffic.** On the 11,067 turns the
  arm with the signal is ahead in all four pairs by 0.03-0.04 s, over both arms' rerun moves there (0.01 and 0.00 s).
  On the 6,715 the pairs run from -0.09 to +0.03 s, `nofc_rep` being behind both runs of `nofc_noend`. It reads
  14-17 % fewer SSD pages in every pair (456K and 472K against 550K and 548K on the 11,067).
- **Turn latency still does not follow TTFT.** On the 11,067 turns: no hints 12.12 and 12.17 s, exact hints 12.08 and
  12.06 s, rules 1 and 3 12.07 and 12.48 s, `ec_r1` 12.34 s, `ec_r134` 12.67 s. `ec_r134` is the slowest and sent
  the fewest turns before minute 115 (11,446 against 11,527-11,710).
- `ec_r134_c128` booted with its settings at 19:47Z (128 users, 160 GB, learned price; no tracebacks, no client
  errors) and ends about 21:48Z.

**`ec_r134_c128` (done 21:48Z on 2026-11-12, clean, GPU unit `GPU-6374d407`): at 128 users rule 1 starts no read
early, and the arm lands with the old oracle.** Tables: `compare_c128.out` (eight runs; the seven-run table is kept as
`compare_c128_7runs.out`), `stall_ec_designs_c128.out`, `checks/read_starts.out`. Each run's first 6,900 s; one run per
arm, no repeat at 128 users.

| First 6,900 s | HiCache | old oracle | old p50 | JIT exact | JIT p50 | no hints | rules 1, 3, 4 |
|---|---|---|---|---|---|---|---|
| turns | 8,334 | 15,525 | 15,473 | 15,384 | 15,366 | 15,418 | 15,502 |
| mean TTFT, all turns | 50.20 s | 20.58 s | 20.68 s | 20.93 s | 20.98 s | 20.84 s | 20.57 s |
| mean TTFT, 8,001 shared returning turns | 50.55 s | 11.73 s | 11.82 s | 12.12 s | 12.23 s | 12.07 s | 11.70 s |
| turn latency on them | 74.67 s | 23.67 s | 23.78 s | 24.18 s | 24.32 s | 24.13 s | 23.64 s |
| SSD pages | 433K | 1,438K | 1,441K | 998K | 1,195K | 1,153K | 1,049K |
| returning turns reading from SSD | 19.6 % | 33.1 % | 32.7 % | 21.4 % | 25.5 % | 25.9 % | 23.6 % |
| turns with over one read | 1 | 36 | 162 | 0 | 120 | 0 | 0 |
| reads staged before arrival | 0 | 4,643 | 4,556 | 2 | 462 | 0 | 0 |

- **Clean.** 16,227 turns in the 2 h, no client errors, no tracebacks, settings as queued (expected cost, 5e-5, price
  learning `harm`, soft pins off, return chance with a 30 s horizon).
- **Rule 1 started no read early, so rule 4 had nothing to learn from.** 3,700 start decisions from 2,071 s on: 3,670
  "left to arrival" and 30 "wait", those 30 at 7,225-7,239 s, after the client stopped sending. Its estimates when
  deciding: read 4.9 s at the median (p90 6.3 s), queue wait after arrival 28.9 s (p10 21.7 s); the read was the
  shorter in 3,662 of the 3,670. No read staged, no launch refused room, no harm charged; all 1,432 price lines read
  5e-5. Arrived turns' reads pinned at most 264K of the 964K tokens of room.
- **The same rule read early at 44 users.** `ec_r134` there: 838 "now", 1,391 "wait" and 664 "left to arrival". The
  first two expected no queue wait (median 0.0 s), the last a median 3.9 s against a 3.1 s read. So the rule tells the
  two settings apart by the expected wait, with no switch.
- **After arrival the engine's hold times the read to admission, as in the JIT arms.** 3,402 reads held, a median 7.2 s
  (its load estimate, with the reads queued ahead, was a median 16 s against an expected wait of 26 s; the no-hint arm
  and the JIT arms held 2,826-3,203 reads a median 5.3-8.0 s). On the 8,001 shared turns: 838 reading turns, 1 landed
  by arrival, 23.5 s of stall per reading turn (JIT 24.3 s both; old oracle 13.4 s), 3.0 s in the queue after landing
  (JIT 4.6 and 3.2 s; old oracle 14.4 s), 11 turns admitted within 0.1 s of the landing. So nearly all of that stall
  is queue time the turn had anyway.
- **TTFT: level with the old oracle; no order among the leading arms is shown.** Paired by turn: against the old oracle
  -0.01 s (14,940 turns), old p50 -0.14 s, the no-hint arm -0.31 s (faster on 7,884 turns, slower on 6,986), JIT
  exact -0.41 s, JIT p50 -0.47 s. The six arms that protect queued turns span 20.57-20.98 s, and there is no second
  run of any of them to say what a rerun moves.
- **SSD traffic: 1,049K pages,** 27 % under the old oracle, 12 % under JIT p50, 9 % under the no-hint arm, 5 % over JIT
  exact; one read per reading turn, none read twice. Read in the 2 h: 7.0 TB, against 9.6 TB (old oracle), 8.0 TB (JIT
  p50), 6.7 TB (JIT exact) and 2.8 TB (HiCache, which recomputes instead).
- **What the arm tests at 128 users is rule 3.** Its reads start as the no-hint arm's do (at arrival, held to
  admission), so as far as the settings go the difference from that arm is the eviction score: return chance from the
  hints against queue state only. That goes with 9 % fewer pages and no TTFT gain that one run can show.
- **An all-four-rules arm at 128 users would repeat this one:** rule 2 acts on the pins of reads staged early, and none
  was staged. Not queued; the user's call.

**Figure set for 128 users: `results/compare_c26_designs_c128/`,** drawn by `c26_ec/figs_designs_c128.sh` (78 s): original
HiCache and the old oracle (campaign 19, the first 2 h of 5 h runs), JIT p50 and JIT exact (campaign 23, the build
before) and `ec_r134_c128`; the four older runs are named in `extra_runs.txt`. Every run is cut at the smaller of 6,900 s
and 300 s before its last send, as `compare_c128.py`. The figures are `compare_c26_designs`'s, with two differences. The
deck-style figure's fourth panel is the SSD read rate (from `s2h_io` landings; 2-3 % above the earlier iostat numbers),
since the room is not the limit here. Its footnote and the wait split's say that the pricing arm's reads were left to
arrival and held to admission; the first checks the log for a read started early, the second is worded for this arm.
Script changes: `plot_timeline.py` (set `designs_c128`, a time cut, a legend corner), `plot_pricing_vs_jit.py` (the set,
the `ssd` panel, that footnote), `plot_ec.py`, `stall_ec.py` (`STALL_LIMIT_S`, `STALL_ORDER_RUN`), `plot_stall.py` (the
set's footnote). The 44-user sets redraw to the byte after them (46 of 48 files of `compare_c26_designs`; the deck
figure's CSV gains an `ssd` column and its PDF a new date).
- `ec_full` booted with its settings at 21:51Z (soft pins on; no tracebacks, no client errors) and ends about 23:53Z.

**`ec_full` (done 23:53Z on 2026-11-12, clean, GPU unit `GPU-6374d407`): all four rules land at 0.66 s, the best
pricing run's level; rule 2 does what it was built to do and pays for it in SSD traffic.** Tables: `compare_ec.out`
(18 runs), `pairs_ec.out` (11 leading runs; earlier `pairs_ec_10runs.out`), `pairs_ec_ctl.out` (18 runs; earlier
`pairs_ec_ctl_17runs.out`), `stall_ec_designs.out`, `checks/latency_lead11.out`, `checks/read_starts_c44.out`,
`checks/bumps.out`. One run of this arm.

| 6,715 shared returning turns | JIT exact | rules 1, 3 | rule 1 | rules 1, 3, 4 | all four | no gap forecast | no hints |
|---|---|---|---|---|---|---|---|
| mean TTFT | 0.51, 0.54 s | 0.66, 0.78 s | 0.67 s | 0.80 s | 0.66 s | 0.69, 0.80 s | 0.71, 0.72 s |
| in the two load episodes | 1.14, 1.31 s | 1.56, 2.31 s | 1.75 s | 2.22 s | 1.66 s | 1.83, 2.21 s | 1.85, 1.84 s |
| turns over 10 s | 14, 27 | 30, 93 | 55 | 83 | 38 | 53, 66 | 25, 47 |
| recomputed tokens per turn | 120, 151 | 214, 253 | 181 | 320 | 169 | 131, 193 | 145, 135 |
| SSD pages, whole run | 388K, 397K | 437K, 415K | 448K | 420K | 534K | 469K, 482K | 562K, 559K |
| reads staged before arrival | 1,202, 1,207 | 629, 560 | 670 | 838 | 1,569 | 0 | 0 |
| turns sent before minute 115 | 11,699, 11,703 | 11,690, 11,527 | 11,580 | 11,446 | 11,706 | 11,710, 11,654 | 11,680, 11,645 |

- **Clean.** 12,161 turns in the 2 h, no client errors, no tracebacks, settings as queued (expected cost, 5e-5, price
  learning `harm`, soft pins on, return chance with a 30 s horizon). Decode, first-turn prefill and SSD speed in line
  with the other arms (37.2 ms per step at 8-15 running, 0.089 ms per token, 2.34 ms per page).
- **Rule 2 works as built.** With soft pins an arrived turn's read takes room from staged reads instead of being
  refused. The harm charged to early reads fell from 1,081 s (`ec_r134`) to 404 s, so the learned price settled lower
  (median 1.4e-5 against 3.1e-5, from 5e-5), and rule 1 staged 1,569 reads early against 838.
- **It pays in SSD traffic.** 462 of the 1,569 staged reads were bumped (352 turns, 110 of them more than once, for 301
  arrived turns). Their reads had loaded 81K pages, 14 % of all the run read and 47 % of what it staged. The data had
  sat a median 2.4 s; the bumped turn arrived a median 8.4 s later (163 of 462 within 5 s) and 452 read again, 192
  before arrival and 260 at it. Whole run: 534K pages, 27 % over `ec_r134`, 22-29 % over rules 1 and 3, 5 % under the
  no-hint arm (the fixed-price rules were 22-26 % under it). SSD reads used: 72 % (88-89 % in the other pricing arms).
- **No more turns found their read landed.** On the 7,283 turns the designs share: 196 of 1,049 reading turns had
  landed by arrival, against 206 of 988 (`ec_r134`) and 130-134 (rules 1 and 3); JIT exact 442 of 740. Its wait is
  0.65 s = 0.17 read stall + 0.08 other queue wait + 0.39 after admission; `ec_r134` 0.77 = 0.18 + 0.15 + 0.44.
- **TTFT: nothing shown beyond what the pricing arms already had.** Against `ec_r134` -0.14 s on the 6,715 turns (5 %
  to 95 %: -0.25 to -0.04) and -0.09 s on the 11,067 (-0.21 to +0.05), one run each, where a rerun of rules 1 and 3
  moved 0.12 and 0.10 s. The gain is not on the turns the bumps touched: bumped turns +0.06 s (339 turns), turns the
  room went to +0.18 s (217), all other returning turns -0.12 s (10,936). Against rules 1 and 3 it is inside their two
  runs (+0.00 and -0.12 s). Ahead of the no-hint arm in both pairs (0.05-0.06 s; 0.11 s on the 11,067), behind exact
  hints in both (0.12-0.15 s; 0.03-0.10 s); with no repeat of this arm neither counts as shown.
- **Latency and throughput: level with the best.** 12.03 s per turn on the 11,067 (exact hints 12.08 and 12.06 s, no
  hints 12.12 and 12.17 s, `ec_r134` 12.67 s) and 11,706 turns before minute 115.
- **Reading.** At 70 GB the room beyond the GPU's copy is 44K tokens. Rules 2 and 4 move that room between early reads
  and arrived turns; they do not make more of it, and what the early reads gain is taken back by the bumps. Whether
  they pay where there is room to stage (a larger host pool) is not tested: the 100 GB arms queued are the no-hint
  rival and the two JIT arms.

**Figures: `results/compare_c26_designs/` redrawn with `ec_full`** (`figs_designs.sh`, 94 s; 53 files): a seventh series,
dotted orange, in every figure, its own timeline and paired CDFs, and a ninth row in the wait split. The 34 per-run
files of the other runs (timelines, event and turn tables, paired CDFs) came out byte-identical. `plot_pricing_vs_jit.py` now picks the legend's column count so that it ends
inside the figure (seven labels ran past the right edge in three columns; two here); the 128-user figure, the
`compare_c26_ec` one and the six-series one redraw to the byte with it.
- `fq90` booted with its settings at 23:54Z (just-in-time, landing quantile 0.9, next-use eviction; no tracebacks, no
  client errors) and ends about 01:56Z on 2026-11-13.

**`fq90` (done 01:56Z on 2026-11-13, clean, GPU unit `GPU-6374d407`): timing just-in-time reads to land at the gap's p90
instead of its p10 takes three-quantile hints from 1.35-1.60 s to 0.94 s, still behind every leading run; its early
reads hold five times the host room that rule 1's do.** Tables: `compare_ec.out` (19 runs), `pairs_ec_ctl.out` (19
runs; earlier `pairs_ec_ctl_18runs.out`), `pairs_ec_fq90.out` (the 11 leading runs and this one, 10,830 turns),
`stall_ec_fq90.out`, `checks/latency_lead13.out`, `checks/pin_hold.out` (new check `checks/pin_hold.py`). One run of
this arm. It differs from the JIT p50 arm (`jit_p50_d`) and from rule 1 alone (`ec_r1`) in the read timing only: same
build, same next-use eviction, same hints.

| 6,715 shared returning turns | JIT p50 (lands at p10) | `fq90` (lands at p90) | rule 1 | rules 1, 3 | no gap forecast | no hints | JIT exact |
|---|---|---|---|---|---|---|---|
| mean TTFT | 1.60, 1.35, 1.48 s | 0.94 s | 0.67 s | 0.66, 0.78 s | 0.69, 0.80 s | 0.71, 0.72 s | 0.51, 0.54 s |
| in the two load episodes | 5.24, 4.01, 4.33 s | 2.71 s | 1.75 s | 1.56, 2.31 s | 1.83, 2.21 s | 1.85, 1.84 s | 1.14, 1.31 s |
| turns over 10 s | 246, 172, 183 | 98 | 55 | 30, 93 | 53, 66 | 25, 47 | 14, 27 |
| recomputed tokens per turn | 970, 928, 1,007 | 446 | 181 | 214, 253 | 131, 193 | 145, 135 | 120, 151 |
| SSD pages, whole run | 563K, 580K, 569K | 506K | 448K | 437K, 415K | 469K, 482K | 562K, 559K | 388K, 397K |
| reads staged before arrival | 2,094, 2,202, 2,149 | 613 | 670 | 629, 560 | 0 | 0 | 1,202, 1,207 |
| mean tokens pinned by reads | 30.2K, 31.5K, 28.8K | 15.3K | 4.8K | 4.2K, 4.6K | 2.6K, 2.4K | 2.5K, 2.5K | 5.9K, 6.0K |
| turns sent before minute 115 | 10,043, 10,208, 10,050 | 11,204 | 11,580 | 11,690, 11,527 | 11,710, 11,654 | 11,680, 11,645 | 11,699, 11,703 |

- **Clean.** 11,740 turns in the 2 h, no client errors, no tracebacks, settings as queued (just-in-time, landing
  quantile 0.9, next-use eviction, no price learning, no soft pins). The engine times such a read past the p50 on
  purpose (`early_s` negative, plan expiry and read deadline moved with it). Decode, first-turn prefill and SSD speed in
  line with the other arms (36.2 ms per step at 8-15 running, 0.090 ms per token, 2.30 ms per page).
- **Against landing at the p10: 0.41-0.66 s faster in all three pairs** (5 % to 95 % clear of zero in each), where the
  three JIT p50 runs differ among themselves by up to 0.25 s. In the episodes 2.7 s against 4.0-5.2 s, half the
  recompute, and about 1,000-1,160 more turns before minute 115. One run of this arm, so by the repeat rule it is not
  counted as shown; it is the largest move between two settings of one knob in the campaign.
- **Against the leading runs: behind all eleven.** On the 6,715 turns by 0.14-0.29 s (rules, no gap forecast, no
  hints) and 0.40-0.43 s (exact hints), every interval clear of zero. That set ends in the first load episode. On the
  10,830 turns it shares with the leading runs alone: 0.97 s against 0.78-0.92 s and 0.71, 0.78 s (exact), behind by
  0.05-0.20 and 0.19-0.26 s, with two of the eleven intervals clear of zero and a third touching it; outside the
  episodes ten of eleven are (0.04-0.14 s). Rule 1 alone, the arm that differs in timing only: ahead by 0.27 s on the
  6,715 (-0.50 to -0.09) and by 0.09 s on the 10,830 (-0.31 to +0.12), one run each.
- **Latency and throughput: between the two groups.** 12.99 s per turn on the 9,716 turns shared with the leading runs
  and `jit_p50_d` (leading 12.00-12.61 s, `jit_p50_d` 16.24 s); 11,204 turns before minute 115 (leading
  11,446-11,710).
- **Why: its early reads sit on the room** (`checks/pin_hold.out`; an early pin is one held before its turn arrived).
  It lands about as many reads early as rule 1, 419 against 367, but they wait longer for their turn: p90 60 s against
  12 s, 75 over 30 s against 13, and 82 are given up when the plan times out with the turn still not there, against 32
  that rule 1 releases. Before arrival they hold 13.2K tokens on average, of 44K of room, against 2.7K (rules 1 and
  3: 2.4K and 2.0K; exact hints 4.0K with 864 early pins, none over 30 s; landing at the p10 24.6K with 1,182, 160
  over 30 s). The room they hold is refused to the reads of turns that have arrived: 625 launches refused against
  396, and 446 tokens recomputed per turn against 181.
  On the 9,716 turns of `stall_ec_fq90.out` its read stall equals rule 1's (0.29 and 0.30 s per turn); the difference
  is after admission (0.50 against 0.44 s).
- **Reading.** A fixed quantile cannot tell a read whose turn is about to come from one whose turn is a minute out or
  not coming. Moving it from the p10 to the p90 halves the room held and removes 0.41-0.66 s of the 0.68-0.93 s by
  which landing at the p10 trails rule 1; the expected-cost start gets the rest by holding a fifth of the room for
  about the same number of early reads. That is the deck's argument for rule 1 seen in a run, with the size of the
  edge over the better fixed quantile tried at 0.09-0.27 s on one run each. Against the rivals that start no read
  early it changes nothing: they hold no room before arrival and stand where rule 1 does.
- `ec_r3` booted with its settings at 01:57Z (just-in-time landing at the p10, return chance with a 30 s horizon; no
  tracebacks, no client errors) and ends about 03:59Z.

**`ec_r3` (done 03:58Z on 2026-11-13, clean, GPU unit `GPU-6374d407`): rule 3 alone leaves JIT p50 where it was, 1.48 s
against 1.35-1.60 s; the gain of the pricing arms is rule 1's.** Tables: `compare_ec.out` (20 runs),
`pairs_ec_ctl.out` (20 runs; earlier `pairs_ec_ctl_19runs.out`), `pairs_ec_r3.out` (this run and the three JIT p50
runs, 9,690 turns), `stall_ec_designs.out`, `checks/latency_r3.out`, `checks/pin_hold.out` (eight runs). One run of this
arm. It differs from the JIT p50 arm (`jit_p50_d`) in the eviction rule only, chance of return within 30 s from the
three quantiles instead of next use from the p50: same build, same just-in-time reads landing at the p10, same hints.
With `ec_r1` and `ec_r13` it completes the two-by-two of rules 1 and 3.

| 6,715 shared returning turns | neither rule (JIT p50) | rule 3 alone (`ec_r3`) | rule 1 alone | rules 1, 3 |
|---|---|---|---|---|
| mean TTFT | 1.60, 1.35, 1.48 s | 1.48 s | 0.67 s | 0.66, 0.78 s |
| in the two load episodes | 5.24, 4.01, 4.33 s | 4.23 s | 1.75 s | 1.56, 2.31 s |
| turns over 10 s | 246, 172, 183 | 186 | 55 | 30, 93 |
| recomputed tokens per turn | 970, 928, 1,007 | 1,042 | 181 | 214, 253 |
| SSD pages, whole run | 563K, 580K, 569K | 540K | 448K | 437K, 415K |
| tokens evicted, GPU / host (M) | 62.4 / 52.3, 63.4 / 53.5, 63.5 / 52.4 | 60.6 / 51.0 | 52.4 / 40.5 | 51.4 / 39.6, 50.1 / 38.4 |
| reads staged before arrival | 2,094, 2,202, 2,149 | 2,207 | 670 | 629, 560 |
| mean tokens pinned by reads | 30.2K, 31.5K, 28.8K | 32.0K | 4.8K | 4.2K, 4.6K |
| turns sent before minute 115 | 10,043, 10,208, 10,050 | 10,058 | 11,580 | 11,690, 11,527 |

- **Clean.** 10,409 turns in the 2 h, no client errors, no tracebacks, settings as queued (just-in-time landing at the
  p10, return chance with a 30 s horizon, no price learning, no soft pins). Decode and first-turn prefill in line with
  the other arms (36.1 ms per step at 8-15 running, 0.089 ms per token); SSD 2.61 ms per page in its 83 large reads
  against 2.47-2.50 for the JIT p50 runs.
- **Rule 3 without rule 1: no effect on TTFT.** Against the three JIT p50 runs -0.12, +0.13 and +0.00 s on the 6,715
  turns, and -0.04, +0.09 and -0.02 s on the 9,690 turns the four runs share. Five of the six intervals include zero;
  the sixth, against the fastest JIT p50 run on the 6,715, has this arm slower (+0.01 to +0.28). Against `jit_p50_d`,
  the run that differs in the rule alone: +0.004 s (-0.11 to +0.12) and -0.019 s (-0.13 to +0.09). The three JIT p50
  runs differ among themselves by up to 0.25 s. Latency 16.10 s per turn against 15.59-16.19 s, and 10,058 turns before
  minute 115 against 10,043-10,208. The wait splits part for part like `jit_p50_d`'s on the 7,283 turns of the figure
  set: 1.48 = 0.39 read stall + 0.38 rest of the queue wait + 0.71 after admission, against 1.49 = 0.38 + 0.42 + 0.69.
- **Rule 3 with rule 1: no effect either** (earlier runs): rules 1 and 3 against rule 1 alone -0.02 s (-0.06 to
  +0.03) and +0.10 s (-0.02 to +0.32). **Rule 1 is worth the same in both columns:** 0.68-0.93 s under next-use
  eviction (`ec_r1` against the three JIT p50 runs) and 0.71-0.83 s under return chance (rules 1 and 3 against this
  run), every interval clear of zero.
- **SSD traffic: perhaps a few percent less.** 4-7 % fewer pages than each JIT p50 run on every set (540K against
  563-580K in the whole run; 533K against 554-564K on the 9,690 turns, where the three JIT p50 runs span 1.8 %), with
  3-5 % fewer tokens evicted. With rule 1 on, the whole-run counts of rules 1 and 3 are 2-7 % under rule 1's, and on
  the shared turns they are level with it (254K, 251K against 251K). One run here.
- **Why: the eviction rule cannot reach what costs** (`checks/pin_hold.out`). The reads still land at the p10 and
  wait for their turn: 1,175 early pins against 1,182 for `jit_p50_d`, held 45 s at the p90 against 41 s, holding
  27.9K tokens before arrival against 24.6K, of 44K of room. Pinned memory is outside any eviction score, so the reads
  of turns that have arrived are refused that room as before (1,871 launches refused against 1,732-1,761) and the turns
  recompute as before.
- **Reading.** The two-by-two is complete: the 0.7-0.9 s between JIT p50 and the pricing arms is the read start (rule
  1), in both columns; the eviction score moves TTFT in neither row. It joins the earlier result that four eviction
  scores are level once reads hold little room: next use from the p50 (`ec_r1` 0.67 s), chance of return (0.66,
  0.78 s), ended first then LRU (`nofc` 0.69, 0.80 s), queued turns only (`nofc_noend` 0.71, 0.72 s). At this setting
  rule 3 is not a source of the TTFT gain; what is left for it to show is the SSD traffic, and that needs a repeat.
- Figures: `compare_c26_designs/` redrawn at 04:04Z. `timeline_ec_r3.png` is new and `ec_wait_split.png` has the arm's
  row; every other figure and table of the set is byte-identical (`pricing_vs_jit.pdf` is re-saved, its PNG and CSV
  unchanged).
- `nofc_c128` booted with its settings at 04:00Z (128 users, 160 GB host pool, reads at arrival, return chance with
  score `lru`; no tracebacks, no client errors) and ends about 06:02Z.

**`nofc_c128` (done 06:02Z on 2026-11-13, clean, GPU unit `GPU-6374d407`): at 128 users the rival with no gap forecast
lands with the old oracle and the pricing arm, 20.57 s against 20.58 and 20.57 s.** Tables: `compare_c128.out` (nine
runs; the eight-run table is kept as `compare_c128_8runs.out`), `stall_ec_c128_rivals.out` (six of the seven arms
that protect queued turns, 14,801 turns). Each run's first 6,900 s; one run per arm, no repeat at 128 users, so no
order between arms is shown. The arm reads at arrival and evicts ended sessions first, queued turns
last, LRU between; it differs from the no-hint rival (`nofc_noend_c128`) in using the end-of-session signal.

| First 6,900 s | old oracle | old p50 | JIT exact | JIT p50 | rules 1, 3, 4 | no hints | no gap forecast |
|---|---|---|---|---|---|---|---|
| turns | 15,525 | 15,473 | 15,384 | 15,366 | 15,502 | 15,418 | 15,519 |
| mean TTFT, all turns | 20.58 s | 20.68 s | 20.93 s | 20.98 s | 20.57 s | 20.84 s | 20.57 s |
| mean TTFT, 8,001 shared returning turns | 11.73 s | 11.82 s | 12.12 s | 12.23 s | 11.70 s | 12.07 s | 11.75 s |
| turn latency on them | 23.67 s | 23.78 s | 24.18 s | 24.32 s | 23.64 s | 24.13 s | 23.68 s |
| SSD pages | 1,438K | 1,441K | 998K | 1,195K | 1,049K | 1,153K | 1,122K |
| returning turns reading from SSD | 33.1 % | 32.7 % | 21.4 % | 25.5 % | 23.6 % | 25.9 % | 24.9 % |
| turns with over one read | 36 | 162 | 0 | 120 | 0 | 0 | 0 |
| reads staged before arrival | 4,643 | 4,556 | 2 | 462 | 0 | 0 | 0 |

- **Clean.** 16,244 turns in the 2 h, no client errors, no tracebacks, settings as queued (reads at arrival, return
  chance with score `lru`, 160 GB host pool of 1,627,648 tokens). Nothing staged, no launch refused room.
- **TTFT: with the leaders.** The seven arms that protect queued turns now span 20.57-20.98 s. Paired by turn: -0.01 s
  against the old oracle (14,956 turns), +0.00 s against the pricing arm (14,943), -0.30 s against the no-hint rival
  (14,871; faster on 7,842 turns, slower on 7,029), -0.39 s against JIT exact and -0.45 s against JIT p50.
- **SSD traffic: in the middle.** 1,122K pages: 22 % under the old oracle, 6 % under JIT p50 and 3 % under the no-hint
  rival; 7 % over the pricing arm and 12 % over JIT exact. One read per reading turn.
- **The wait is the GPU queue's.** On the 14,801 turns six arms share: 20.60 = 7.08 read stall + 13.10 rest of the
  queue wait + 0.42 after admission, against 20.60 = 5.99 + 14.19 + 0.42 for the old oracle and 20.57 = 6.48 + 13.67 +
  0.42 for the pricing arm. Reading at arrival puts a second more of the wait under the read and takes it off the rest
  (+1.09 and -1.09 s against the old oracle); the turn waits as long either way. After its read lands a reading turn
  waits another 2.4 s (old oracle 10.5 s, pricing arm 2.9 s).
- **The end-of-session signal at 128 users: not shown.** Against the no-hint rival 0.30 s less TTFT (20.61 against
  20.91 s) and 3 % fewer SSD pages, one run each. At 44 users, with two runs a side, the signal showed on SSD pages
  only (14-17 % fewer).
- **Reading.** At 128 users every arm that keeps queued turns' history lands within 0.41 s (2 %) of the others, with
  or without hints, a gap forecast or a price; what differs is SSD traffic (998K-1,441K pages). The setting cannot
  rank read-start rules: the queue wait covers the read.
- Figures: none changed. `compare_c26_designs_c128/` holds the five design series, not the rivals.
- Watcher: `c26_ec/watch.sh` runs its problem checks (tracebacks, client errors, a stalled client) only on run
  folders ending `_NAT160` or `_H70`, so it would have watched the three 100 GB arms for their end only. Replaced at
  06:06Z, before the first of them had a client, by `checks/watch_all.sh`: the same script with the pattern widened to
  every host pool size. `watch.sh` is left as it is.
- `nofc_noend_h100` booted with its settings at 06:07Z (44 users, 100 GB host pool of 1,017,280 tokens, reads at
  arrival, return chance with score `queue`, engine e; no tracebacks, no client errors) and ends about 08:09Z. Its
  server took 4.5 min to start, 212 s of it loading weights from a cold page cache after the 160 GB arm; the client
  starts only once the server is ready.

**Outages of 07:31Z and 17:06Z on 2026-11-13: `nofc_noend_h100` cut at minute 84, relaunched 19:28Z on the arm's third
GPU unit, the campaign's fifth.**

- The host went down at 07:31Z, 84 min into `nofc_noend_h100` (client from 06:06:58Z: 8,800 turns, no client errors,
  no tracebacks). It came back at 17:02Z and the launcher was restarted at 17:03Z on GPU unit `GPU-72ebee63`; the host
  went down again at 17:06Z, before that server had finished starting (no client, nothing to keep). It came back at
  19:27Z and the launcher was restarted at 19:28Z on GPU unit `GPU-a8a5cbbe`. The arm reruns from scratch.
- The 84-minute run is kept as `nofc_noend_h100_cut` in `extra_runs.txt` (run
  `20261113_060157_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H100`, GPU unit `GPU-6374d407`, the unit of every arm
  from `ec_r134` to `nofc_c128`).
- Queue unchanged, three arms: `nofc_noend_h100` to about 21:35Z, `jit_p50_h100` to about 23:38Z, `jit_exact_h100` to
  about 01:41Z on 2026-11-14. Watcher: `checks/watch_all.sh`.
- `nofc_noend_h100` booted with its settings at 19:33Z (44 users, 100 GB host pool of 1,017,280 tokens, reads at
  arrival, return chance with score `queue`, engine e; no tracebacks, no client errors). Server start 4.3 min, 165 s of
  it loading weights from a cold page cache after the reboot.

**`nofc_noend_h100_cut` (the 84 min before the outage, GPU unit `GPU-6374d407`): with a 100 GB host pool the rival
with no hints reads from SSD on 0.7 % of returning turns and reaches 0.37 s, ahead of JIT exact at 70 GB (0.48,
0.51 s).** Tables: `compare_ec_h100cut.out` (ten runs), `pairs_ec_h100cut.out`, `stall_ec_h100cut.out`,
`checks/latency_h100cut.out` (seven runs on the 8,030 returning turns they share: the cut run's first 79 min, which
hold the first load episode and not the second). One run, cut; the rerun in progress is its repeat.

| 8,030 shared returning turns | no hints, 100 GB (cut) | no hints, 70 GB | rules 1, 3, 70 GB | JIT exact, 70 GB |
|---|---|---|---|---|
| mean TTFT | 0.37 s | 0.65, 0.66 s | 0.60, 0.70 s | 0.48, 0.51 s |
| in the first load episode (1,453 turns) | 0.51 s | 1.77, 1.77 s | 1.52, 2.21 s | 1.12, 1.25 s |
| elsewhere (6,577 turns) | 0.34 s | 0.41, 0.41 s | 0.40, 0.37 s | 0.34, 0.34 s |
| p99; turns over 10 s | 2.6 s; 0 | 7.2 s; 25 and 7.9 s; 47 | 6.8 s; 29 and 10.4 s; 94 | 5.0 s; 15 and 6.1 s; 27 |
| read stall + rest of queue wait + after admission | 0.006 + 0.019 + 0.346 | 0.223 + 0.053 + 0.376, 0.230 + 0.060 + 0.367 | 0.141 + 0.072 + 0.390, 0.190 + 0.116 + 0.400 | 0.070 + 0.045 + 0.366, 0.072 + 0.052 + 0.383 |
| turns reading from SSD | 56 | 1,125, 1,138 | 991, 1,006 | 691, 743 |
| SSD pages | 17K | 329K, 328K | 266K, 258K | 210K, 223K |
| recomputed tokens per turn | 34 | 127, 118 | 179, 217 | 106, 131 |
| turn latency | 10.72 s | 11.35, 11.32 s | 11.52, 11.87 s | 11.26, 11.27 s |

- **Clean to the cut.** 8,800 turns in 83.7 min, no client errors, no tracebacks. Decode 36.3 ms per step at 8-15
  running and first-turn prefill 0.089 ms per token, in line with the other arms.
- **Almost nothing is read.** 56 of the 8,030 returning turns read from SSD against 1,125 and 1,138 for the same arm
  at 70 GB; host eviction 7.5 M tokens in 79 min against 47 M in 115 min; no read launch refused room; 396K tokens of
  room idle at the price updates against 76K. The sessions not parked in a long gap hold up to about 1,010K tokens in
  the first episode (`load K` in the position table), just under the pool's 1,017K.
- **TTFT.** 0.28 and 0.29 s under the same arm at 70 GB (intervals -0.48 to -0.11), 1.26 s under it in the episode.
  0.11 and 0.14 s under JIT exact at 70 GB (+0.02 to +0.26), all of it in the episode (0.61 and 0.74 s; elsewhere
  0.00). No turn over 10 s.
- **The wait.** The read stall is 0.006 s per turn (0.22-0.23 s at 70 GB, 0.07 s for JIT exact there). That is all a
  read-start rule could take off at this pool size: the two JIT arms queued at 100 GB can lead this arm by a few
  hundredths of a second at most, unless their eviction order costs or saves reads.
- **Latency.** 0.54-0.63 s per turn under the 70 GB leaders, more than the TTFT gap: the time after the first token
  is 0.3-0.4 s shorter as well.
- **Reading.** At 44 users, 30 GB more host memory removes the reads themselves, with no hint: the larger pool does
  not make room for earlier reads, it leaves almost none to start. Read-start rules have something to decide only
  where the pool is well under the working load (70 GB here).
- **Limits.** One run, its first 79 min. The second episode carries a higher load (about 1,050K tokens in the 70 GB
  runs, above the pool), so the full run should read more there.

**Host-pool sweep on the eviction replay (CPU, `checks/limits/pool_sweep.py` and `.out`): the reads fade by about half
for every 5 GB above 70 GB, and the model fitted at 70 GB gives the 100 GB run's 0.7 %.** The one-pool model of
`frozen/ec_policy_20261111/evict_replay/` (no pins, no GPU and host split, a run's own timeline; read shares, not
TTFT), fitted on `nofc_noend_rep` so that LRU gives its measured 17.0 % of returning turns reading from SSD. Another
host size moves the model's pool by the host pool's ratio or by the tokens added; both are shown, as a range.

| Host pool | 70 GB | 75 | 80 | 85 | 90 | 100 | 110 | 160 |
|---|---|---|---|---|---|---|---|---|
| LRU (no hints) | 17.0 % | 9.4-10.8 | 4.9-6.7 | 2.7-4.2 | 1.5-2.7 | 0.8-1.2 | 0.6-0.8 | 0.2-0.3 |
| LRU, ended sessions first | 13.7 % | 6.7-8.0 | 3.0-4.4 | 1.3-2.4 | 0.6-1.3 | 0.0-0.3 | 0.0 | 0.0 |
| next use from the p50 | 15.5 % | 9.1-10.5 | 4.5-6.6 | 2.2-3.9 | 1.5-2.1 | 0.8-1.1 | 0.6-0.8 | 0.2-0.3 |
| return chance within 30 s (rule 3) | 13.6 % | 8.2-9.5 | 4.3-6.1 | 1.8-3.6 | 0.6-1.8 | 0.0-0.3 | 0.0 | 0.0 |
| exact next use | 9.1 % | 4.8-5.5 | 2.1-3.3 | 0.9-1.7 | 0.4-0.9 | 0.1-0.2 | 0.0-0.1 | 0.0 |

- **Check.** On the 100 GB cut run's own timeline the model gives 0.7 % (by ratio) and 0.9 % (by tokens); measured
  0.7 %. The 70 GB run's first 84 min give 0.7-0.8 % at those pools and 12.6 % at 70 GB, so the fall is the pool's
  and not the timeline's.
- **What a forecast can save in reads shrinks with the pool:** exact next use against LRU is 7.9 points of returning
  turns at 70 GB, about 5 at 75 GB, 3 at 80 GB, 2 at 85 GB and under 1 from 100 GB.
- **Rule 3's score does not beat ended-first LRU at any pool size** (row added 21:25Z; it was computed and left out):
  level at 70 GB (13.6 against 13.7 %, with 13 % fewer tokens read per turn), behind from 75 to 85 GB (8.2-9.5 against
  6.7-8.0 %, 4.3-6.1 against 3.0-4.4 %, 1.8-3.6 against 1.3-2.4 %), level again from 90 GB. So by the model a
  pricing arm at a pool between 70 and 100 GB would not gain from the three-quantile score over the end signal.
- **Reading.** 70 GB, the smallest pool that still mirrors the GPU, is the setting most favourable to hints at 44
  users; a pool between 70 and 100 GB would show a smaller version of the 70 GB result, and a pool above 100 GB (the
  136 GB arm once proposed) next to nothing. Not queued. A setting with more reads and a short GPU queue would need
  more users per GB without the GPU saturating; the model cannot place it, since it replays one run's timeline.

**The old routine on p50 hints at 44 users (campaign 21's run, named `oracle_v2_p50` in `extra_runs.txt`): 2.05 s on
the 6,715 returning turns ten runs share, behind the same routine on exact hints (1.53 s) and JIT p50 on the new
build (1.48 s).** Tables: `compare_ec_oldp50.out`, `pairs_ec_oldp50.out`. The deck's methods table lists the old
routine with exact or p50 hints; the p50 one was in no table of this campaign (found by the audit of 13 Nov).

- On the 9,274 returning turns the eight hinted and no-hint runs share: 1.97 s against 1.68 s with exact hints (+0.28 s,
  interval [-0.02, +0.61], so that order is not separated on this pair), 1.51 s for JIT p50, 0.84 s for JIT exact,
  0.91 and 0.92 s with no hints, 0.79 and 0.92 s for rules 1 and 3. It reads from SSD on 2,326 turns (1,852 with exact
  hints) and 445K pages (355K).
- One run from campaign 21 (its own GPU unit; 1.95 s there on that campaign's 7,768 shared turns). The old routine was
  not rerun on the new build.

**Audit of what is left undone (2026-11-13, asked for): `checks/audit/`.** `audit_20261113.json` holds 102 checked
items with their evidence and 12 more from a completeness pass, `items.tsv` is the index, `scratch/` the readers'
probes (unreviewed). Done from it the same day: the head, the review table, "What was built" and the Arms table above
were corrected in place (additions marked "Since:"; the earlier text is `checks/audit/DECISIONS_before_head_edit.md`),
rule 3's row in the pool sweep, the old routine's p50 row, and `checks/limits/price_path.py` saved beside its output.
At 21:27Z the author stopped the follow-up analyses and at 21:29Z asked for nothing more until the 100 GB arms are
done, so the rest is parked: by owner in `items.tsv`, 17 items the lead could finish, 9 that wait for the arms, 52
decisions for the author and 24 held edits to the paper docs and the deck. The audit's corrections to entries below
(`record_corrections` in the JSON, part B) are not applied; nor is the byte check of the older figure sets after
today's plot-script edits (`checks/figs_md5/check.sh` is ready).

**`nofc_noend_h100` done 21:33Z on 2026-11-13 (rc 0, full 2 h, GPU unit `GPU-a8a5cbbe`); `jit_p50_h100` launched 21:33Z
and booted with its settings** (100 GB host pool of 1,017,280 tokens, just-in-time reads on p50 hints, next-use
eviction, engine d; client from 21:35:19Z, no tracebacks, no client errors; weights loaded in 13 s from the page
cache). By the author's request of 21:29Z the three 100 GB arms are compared once all are done.

**`jit_p50_h100` done 23:35Z on 2026-11-13 (rc 0, full 2 h); `jit_exact_h100` launched 23:35Z and booted with its
settings** (100 GB host pool of 1,017,280 tokens, just-in-time reads on exact hints, next-use eviction, engine d,
GPU unit `GPU-a8a5cbbe`; client from 23:37:34Z, no tracebacks, no client errors). It ends about 01:38Z on 14 Nov.

**100 GB host pool at 44 users, all three arms (done 01:38Z on 2026-11-14, each a clean 2 h on GPU unit
`GPU-a8a5cbbe`): no hints, JIT p50 and JIT exact tie at 0.41 s, and all three beat every 70 GB run.** With 30 GB more
host memory under 1 % of returning turns read from SSD, so there is nothing left for a hint to time. Tables:
`compare_ec_h100.out`, `pairs_ec_h100.out` (the three arms, 11,828 shared returning turns), `pairs_ec_h100_vs70.out`,
`stall_ec_designs_h100.out`, `checks/latency_h100.out` (with the 70 GB runs, 9,716 shared returning turns),
`pairs_ec_h100_rep.out` (with the cut run), `checks/limits/pool_check_h100.out`. Figures:
`results/compare_c26_designs_h100/` (`figs_designs_h100.sh`).

| 9,716 shared returning turns | No hints | JIT p50 | JIT exact |
|---|---|---|---|
| mean TTFT at 100 GB | 0.41 s | 0.41 s | 0.42 s |
| mean TTFT at 70 GB | 0.93, 0.93 s | 1.58 s | 0.72, 0.83 s |
| p99 TTFT at 100 GB (70 GB) | 3.0 s (12.4, 12.5) | 3.0 s (14.5) | 4.1 s (11.5, 12.7) |
| turns reading from SSD at 100 GB (70 GB) | 92 (1,598, 1,610) | 103 (2,361) | 62 (1,061, 1,110) |
| SSD pages at 100 GB (70 GB) | 27K (469K, 470K) | 27K (568K) | 20K (324K, 335K) |
| wait at 100 GB: read stall + rest of queue + after admission | 0.011 + 0.056 + 0.343 | 0.009 + 0.055 + 0.347 | 0.002 + 0.074 + 0.343 |
| turn latency at 100 GB (70 GB: `_rep`, `_d` runs) | 10.90 s (12.16) | 10.89 s (16.24) | 10.90 s (12.11) |
| turns sent before minute 115 at 100 GB (70 GB) | 12,236 (11,645) | 12,237 (10,050) | 12,236 (11,703) |

- **Clean.** 12,861-12,862 turns each, no client errors, no tracebacks, settings as queued. Hardware control in line
  with the 70 GB runs (decode 36.6-37.4 ms per step at 8-15 running, first-turn prefill 0.087-0.089 ms per token).
- **The three arms are level.** On their own 11,828 shared returning turns: JIT p50 minus no hints +0.001 s
  [-0.01, +0.01], JIT exact minus no hints +0.004 s [-0.01, +0.02], JIT exact minus JIT p50 +0.004 s [-0.01, +0.02].
  Same turns served, same latency. No order is shown; the JIT arms have one run each.
- **The no-hint arm repeats.** On the 8,030 turns of the cut run's first 79 min the full run is at 0.37 s against
  0.37 s (-0.003 s [-0.02, +0.01]), 59 reading turns against 56, on another GPU unit. On that set the JIT arms are
  0.002-0.008 s ahead of both no-hint runs, and on the whole 2 h JIT exact is 0.004 s behind: mixed signs, thousandths.
- **Against 70 GB:** no hints -0.52 s on both pairs ([-0.82, -0.26]), JIT p50 -1.17 s [-1.70, -0.67], JIT exact -0.30
  and -0.41 s ([-0.54, -0.10], [-0.73, -0.13]). JIT exact at 70 GB is 0.31-0.42 s behind every 100 GB arm, rules 1 and
  3 at 70 GB 0.37-0.49 s behind. Throughput is 5 % above the 70 GB leaders' and 22 % above JIT p50's.
- **Why hints do nothing here.** Reads are needed on 62-103 of 9,716 turns, and no launch was refused room (385K
  tokens of host memory idle at the median). Exact hints land 55 of their 62 reads before the turn arrives, the p50
  12 of 103, no hints none of 92: that is 0.009 s of read stall per turn between the best and the worst, and one run's
  rest-of-queue wait differs by more (0.055-0.074 s). JIT p50 staged 10 reads early in 2 h, JIT exact 64.
- **The second load episode is the GPU's.** Positions 8,750-11,000 (not in the cut run): 0.69, 0.70 and 0.76 s, of
  which read stall 0.045, 0.031 and 0.014 s and queue 0.27, 0.28 and 0.37 s. The slowest 500-turn bin (minute 80) is
  1.34-1.44 s in all three.
- **The forecast of 02:38Z on 12 Nov was wrong by about six.** It said "about a third as much on SSD as at 70 GB";
  measured is a seventeenth (92 against 1,598 and 1,610 reading turns; 27K against 470K pages). It compared the pool
  with the context of every session in progress (median 1,176K tokens); what has to stay in memory is the context of
  sessions not parked in a long gap, which peaks at 1,012-1,019K and 992-1,003K in the two episodes, at or just under
  the 1,017K pool.
- **The pool model holds on the full runs** (`pool_check_h100.out`): LRU 0.8-1.1 % against 0.9 % measured for no
  hints, next use from the p50 0.7-1.0 % against 1.0 % for JIT p50. Exact next use gives 0.1-0.2 % against 0.6 %
  measured for JIT exact, whose eviction is next use within a 300 s horizon and not the model's exact order.
- **Reading.** At 44 users and x10 gaps the two things early reads need do not occur together: at 70 GB there are
  reads (14 % of turns) and 44K tokens of room; at 100 GB there are 349K of room and no reads; the sweep says reads
  fade by half per 5 GB in between. So "do early reads pay once there is room" has no test at this user count, and
  the deck's ample-room claims are not shown here: the tie is from having nothing to read. What the 30 GB bought is
  larger than anything a hint bought at 70 GB (0.52 s against 0.10-0.21 s for exact hints over no hints).
- **Limits.** One run per JIT arm. One trace, one seed. Original HiCache was not run at 100 GB, so whether the extra
  memory alone would fix it, without the no-hint arm's queue protection and read pins, is open. `pairs_ec.py` now
  passes named arms to the loader (it dropped `_h100` arms before; the earlier file is
  `checks/audit/pairs_ec_before_h100.py`). In `pricing_vs_jit.png` the three 100 GB end labels sit on each other.

**End state, 01:38Z on 2026-11-14.** The queue is done: launcher idle with no `END` file (an appended queue line
starts within a minute; `touch results/campaign26/END` makes it exit), server stopped, GPU idle, container up, no
watcher running. 21 runs in the manifest (15 at 44 users and 70 GB, 3 at 128 users, 3 at 100 GB); the tables also
name five runs of campaigns 21, 23 and 25 and the runs in `extra_runs.txt`. No relaunch is needed after an outage
unless arms are added.
