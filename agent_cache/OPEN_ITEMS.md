# Open items after campaigns 28 and 29: two engine defects and the decisions that wait

Written 2026-11-20, after the last arm of campaign 29 (01:25:51Z). Campaigns 28 (FreeInference) and 29 (TraceLab) are
complete; their records are `results/campaign28/DECISIONS.md` and `results/campaign29/DECISIONS.md`. This file holds
what they left open: two defects found in the engine, what each does to the numbers, the decisions that wait for the
author, and the steps to pick the work up. Nothing described here has been applied: no engine fix, no client change,
no queue change.

Paths: host `/home/wanhr/sglang/agent_cache` is container `/sgl-workspace/sglang/agent_cache` (container
`sglang_hicache`); every command says which side it runs on. Code anchors are `path:line` at commit `fe7d9a10bc`,
relative to `/home/wanhr/sglang`. The working tree's `python/` equals the frozen copy `frozen/engine_ec_20261111e`.
The campaigns ran `frozen/engine_ec_20261111d`, which has the same code at every anchor below (checked by diff: the
two copies differ by one more `return_chance` score, `queue`, that no arm used; only the `ReturnChanceStrategy`
anchors sit one or two lines higher in copy d).

## 0. The short version

1. **Defect 1 (§1), not fixed.** A queued request may look up the SSD three times, and a lookup that finds nothing
   counts. In the replay most returning requests spend all three on text that no tier holds; when their real history
   later leaves memory, the engine refuses to read it back and computes it again.
2. **Defect 2 (§2), not fixed.** The `next_use` eviction ranks a request that has waited over 60 s in the queue as
   "not coming" and evicts its session's tail right after ended sessions. With TraceLab's three-minute queue this
   sets off defect 1 for 250 to 290 requests an arm.
3. **What they change (§3).** On TraceLab 99 % of the history the three just-in-time arms computed again is defect 1,
   and the 16 s lead of the four-rules arm is defect 2 acting through defect 1, not the rules. The ranking against
   original HiCache stands.
4. **Decisions that wait (§4).** D1: fix both before the next evaluation (my recommendation: yes). D2: which arms
   run again on the fixed engine (one, as the trial). D3: the follow-up setting below saturation (a proposal is on
   record). D4: 3 h arms or a warm start for it (my preference: 3 h arms).
5. **Steps for the return (§5).** §5.1 box state (host, 1 min, GPU idle). §5.2 reproduce the evidence (host, 1 min,
   GPU idle). §5.3 CPU tests around a fix (container, 30 s, GPU idle). §5.4 freeze the fixed engine (container,
   1 min, GPU idle). §5.5 run `jit_p50` again on TraceLab (2 h 6 min, GPU busy).
6. **Smaller items (§6)**, one line each.

## 1. Defect 1: the read allowance goes on lookups that find nothing

This section says what the engine does, why that costs a request its history, how often it happened and how it can
be fixed. At its end the reader can find the code, the log lines and the counts. The records call it "the cap" or
"the three-lookup cap".

**Status:** not fixed; the engine is unchanged. The fix is decision D1 (§4).

**What the engine does.**

- A hinted turn gets a plan whose key is the previous prompt plus the previous reply as the model generated it
  (`python/sglang/srt/mem_cache/next_turn_prefetch.py:528`). The plan treats every token of that key that is not in
  GPU or host memory as missing history: at the arrival gate as `chain - matched` (`:1327`), in the queue check as
  `baseline - resident` (`:809`, `:818`, `:873`).
- A lookup, though, can only ask the SSD for the request's own prompt beyond its match in memory
  (`python/sglang/srt/managers/scheduler.py:3157`, `:3171`).
- The replay sends the previous reply back as new ids, and some prompts drop part of the previous one. The next
  prompt then does not contain the key's tail, the plan sees 256 tokens or more missing (the lookup threshold), and
  the lookup asks for text that is in no tier.
- Every lookup counts, whatever it finds. The gate allows three a request (`MAX_REFETCHES = 2`, `:108`; counted at
  `:1332` to `:1339`), the queue check two re-reads (`:692`, `:878`). A lookup's result is known only in the storage
  thread (`python/sglang/srt/managers/cache_controller.py:1285`), and nothing reports it back to the plan.
- The constant's own comment is about reads ("re-evicted pages are fetched again only once", `:106`). It was meant
  to bound reads that keep being evicted, not lookups that read nothing.

**One request** (measured: `jit_p50-c58-t11`, campaign 29; §5.2 prints its log lines):

| after arrival | event                                                                                                                                     |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 0 s           | the prompt matches 19,328 tokens in memory and the chain is 22,144, so the gate allows a read; lookup 1 asks for 3,008 tokens and finds 0 |
| 61.0 s        | 2,496 tokens of the chain's tail leave memory, a second after the registration made at arrival turns overdue (§2); lookup 2 finds 0      |
| 61.5 s        | lookup 3 finds 0                                                                                                                          |
| 62.0 s        | `ntp_reevict ... refetches=2 capped=1`: the allowance is used up                                                                        |
| 185.7 s       | the history itself leaves memory ("device prefix shrank ... current_match=0"); the gate answers `capped`                                |
| 187.1 s       | admitted with `cached_input_len=0`: all 22,338 prompt tokens are computed, no lookup was left to ask the SSD for them                   |

**How often.**

| the three arms with the gate: `jit_exact`, `jit_p50`, `ec_full` | FreeInference (campaign 28)           | TraceLab (campaign 29)                   |
| --------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------- |
| queue wait, median                                                    | 55, 54, 54 s                          | 193, 185, 171 s                          |
| lookups that found nothing, of all lookups                            | 52, 47, 53 %                          | 63, 61, 60 %                             |
| returning requests refused a read at the cap                          | 97 of 8,048; 93 of 8,145; 33 of 8,257 | 333 of 2,924; 313 of 2,982; 175 of 3,228 |
| history computed again, of the history                                | 1.4, 1.2, 0.4 %                       | 11.8, 10.8, 5.7 %                        |
| of it, by the refused requests                                        | 93, 91, 77 %                          | 99.4, 99.3, 98.7 %                       |
| GPU time spent computing history again, minutes 30 to 110             | 5.6, 5.0, 1.9 %                       | 19.2, 19.4, 11.6 %                       |

Measured 2026-11-19 and 20, returning requests sent before 6,600 s: `checks/cap_share_<arm>.out`, `checks/cap_path.out`
and `checks/gpu_work.out` under each campaign's folder; the lookup row from the `s2h_query` lines of each server log
(§5.2). A session's first request has to ask as well: 5 % of `jit_p50`'s lookups on FreeInference and 7 % on TraceLab,
where it finds nothing either.

- It needs a queue long enough for the history to leave memory after the allowance is gone: rare with a 54 s queue,
  common with three minutes.
- The old oracle arm runs without the gate, so only the queue check's limit binds it. `hicache` has no plan at all;
  its own storm (8 lookups used up at arrival) is the baseline's behaviour and not part of this list.

**Fix: two ways, not exclusive (neither written).**

- A. Count only lookups that find pages: count a lookup when its read lands (the plan already hears that,
  `on_read_landed`, `:1154`) instead of when the gate lets it through, and do not ask again until the request's
  match in memory changes (or the queue check asks every 0.5 s).
- B. Remove the cause: at arrival cut the plan's key to the prefix the request's prompt shares with it, so text the
  request does not have is no longer missing history. The gate then needs the request's token ids; today it gets a
  handle and a match length (`python/sglang/srt/mem_cache/unified_radix_cache.py:2155`).
- My preference: B, and A if empty lookups remain. B also drops the lookup at arrival that cannot find anything.
- My reading, not run: a real client is exposed too wherever its next prompt does not repeat the reply token for
  token (a tool call rendered again by the template, reasoning dropped from the history).
- Tests that pin today's behaviour and have to change: `test/registered/unit/mem_cache/test_next_turn_prefetch.py:1568`
  (the gate answers read, covered, reread, reread, capped) and `:765` (re-reads up to the cap).

## 2. Defect 2: `next_use` ranks a turn that waits over 60 s in the queue as overdue

This section says what the eviction order does to a queued turn, where, how often, and how it can be fixed. The two
defects meet in one place: defect 2 decides when the tail of a queued session leaves memory, defect 1 turns that
into a lost history.

**Status:** not fixed; the engine is unchanged. The fix is decision D1 (§4).

**What the engine does.**

- When a hinted turn arrives, the prefetcher registers its session's tail with next use = now
  (`python/sglang/srt/mem_cache/next_turn_prefetch.py:770` to `:776`; log line `nue_register ... next_use_ms=0`).
- `NextUseStrategy` ranks a registration more than `overdue_grace_s` in the past (60 s,
  `python/sglang/srt/mem_cache/next_use_eviction.py:134`) as `TIER_OVERDUE` (`:156`): "the turn is not coming",
  evicted second, right after ended sessions. The grace was set to a plan's minimum expiry (`:143`), for turns that
  never arrive. A turn that has arrived and waits in the queue falls under it too.
- `ReturnChanceStrategy`, the four-rules arm's order, ranks a session registered with a time as queued and evicts it
  last, for up to 600 s (`:240` to `:243`, `:58`). Its registration covers the session's whole path (`:192`); under
  `next_use` only the tail node carries it.
- `oracle_v2`, `jit_exact` and `jit_p50` run `next_use` (`agent_cache/scripts/start_server.sh:115`); `ec_full` runs
  `return_chance`; `hicache` runs `lru`.

**How often.** Requests that used up their allowance on empty lookups 30 s or more before admission ("at risk"), by
the time after arrival at which it was used up:

| `jit_exact`, `jit_p50`, `ec_full`   | FreeInference (campaign 28) | TraceLab (campaign 29) |
| ----------------------------------------- | --------------------------- | ---------------------- |
| at risk                                   | 53, 52, 84                  | 369, 329, 236          |
| of them, used up 55 to 70 s after arrival | 4, 4, 1                     | 291, 252, 29           |
| of those, later refused a read            | 4, 4, 0                     | 275, 243, 14           |
| of them, used up in the first 5 s         | 47, 42, 64                  | 59, 62, 137            |

Measured 2026-11-20: `results/campaign28/checks/cap_path.out`, `results/campaign29/checks/cap_path.out` (§5.2).

- The 55 to 70 s row is defect 2: full on TraceLab under `next_use`, nearly empty under `return_chance`, and nearly
  empty on FreeInference, where the queue wait is under the grace.
- The first-5 s row is defect 1 alone: part of the key was already out of memory when the turn arrived, so the
  queue check asked at once.
- My reading, not measured: the overdue rank also costs the old oracle arm, where a queue wait is about 260 s and
  an unpinned read is gone a second after landing (campaign 29's record, `oracle_v2` entry).

**Fix: two ways (neither applied).**

- (a) In the engine: give `next_use` the rule `return_chance` has. A registration made at arrival ranks last until
  the turn is admitted, with the same 600 s backstop for a turn aborted before it commits. The arrival needs a
  marker of its own, because a plain time is also what a hint registers.
- (b) Without a code change: `--radix-eviction-policy-config {"horizon_s":300,"overdue_grace_s":1200}` on the three
  `next_use` arms; copy d accepts it (`python/sglang/srt/arg_groups/kv_cache_hook.py:400`). It also keeps for
  1,200 s every session whose hinted turn has not come, which changes the table arm: there a session is overdue
  today one minute after the time its p50 hint gave. So (b) serves as a check and (a) as the fix.
- Test that pins today's behaviour: `test/registered/unit/mem_cache/test_next_use_eviction.py:56` (the tiers and
  their order). The queued rule of `return_chance` is tested at `:237` and `:422`.

## 3. What the defects change in the results

This section sorts the findings of the two campaigns into those that stand, those whose size is not clean and those
the defects produce. One run per arm throughout; the findings are the last section of campaign 29's record.

| finding                                                                                                      | with the defects in mind                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| every next-turn design beats original HiCache at saturation (80 s on FreeInference, 98 to 175 s on TraceLab) | stands; on TraceLab the three just-in-time arms are held back by defect 1, so their lead is understated                                     |
| the old oracle ties just-in-time at a 54 s queue (+3.0 s for `jit_exact`)                                  | the +3.0 s is defect 1: 93 % of the history `jit_exact` computed again belongs to its 97 refused requests                                 |
| the old oracle loses 78 s to just-in-time at a three-minute queue                                            | the direction stands (its reads land two minutes early); the size is not clean: the old oracle carries defect 2, just-in-time carries both  |
| exact and table hints tie (-0.9 s and -3.7 s)                                                                | stands; the -3.7 s on TraceLab is defect 1's draw while the queue forms                                                                     |
| the four-rules arm leads `jit_p50` by 15.6 s on TraceLab (1.7 s on FreeInference)                          | produced by the defects: the rules made no decision, and `return_chance` avoids defect 2, so defect 1 refuses 175 requests instead of 313 |
| price, read timing and soft pins decided nothing                                                             | stands; a saturated setting gives them nothing to decide                                                                                    |

Estimate, not run: with defect 1 fixed the three just-in-time arms on TraceLab stop spending 19, 19 and 12 % of GPU
time on refused histories. By the arithmetic that matched `ec_full` against `jit_p50` ((1 - 0.116) / (1 - 0.194) =
+9.7 %, measured +10.2 %), each would send about 37 requests a minute in minutes 30 to 110 (30.2, 30.4 and 33.5
measured) and they would end within a few seconds of each other. It is wrong if another limit binds first; the
storage read thread was 45 to 59 % busy in those arms.

## 4. Decisions that wait

These are the decisions only the author can take, each with its options, what I recommend and what it holds up.
None has been acted on.

**D1. Fix the two defects before the next evaluation?**

- **Status:** open since 2026-11-19.
- Options: fix both (an engine change and a new frozen copy; campaigns 28 and 29 stay as the record of copy d); fix
  defect 1 only; fix nothing and report both as limits of the comparison.
- My recommendation: fix both, defect 1 by way B then A (§1), defect 2 by way (a) (§2). Below saturation the queue
  is short and both bind less, but every overloaded minute brings them back, and the arms differ in how they meet
  them (`return_chance` avoids defect 2), so they would again decide the order of the arms.
- Holds up: D2 and the follow-up campaign. Cost: CPU work on the code and its unit tests, then §5.3 to §5.5.

**D2. Which arms run again on the fixed engine at the saturated setting?**

- **Status:** open; needs D1.
- Options: `jit_p50` on TraceLab alone, as the trial of the fix (§5.5); `ec_full` and `jit_exact` on TraceLab as
  well (4 h 12 min more); `oracle_v2` too (2 h 6 min more; the arm without the gate, a test of fix (a) alone);
  nothing on FreeInference, where the four hinted arms are within 2.1 s.
- My recommendation: the one trial arm. Run the others only if TraceLab's saturated numbers go into the paper; the
  four-rules arm's 16 s lead should not be quoted from copy d.

**D3. The follow-up setting below saturation.**

- **Status:** open since 2026-11-19; proposal in campaign 28's record, "Discussion with the author". Its numbers are
  a model on paper (`results/campaign28/checks/load_model.py`, `start_state.py`), not a run.
- Why: both campaigns ran saturated (about 20 requests running, 77 to 102 queued). The server then times every read
  from the queue position, so neither a hint's precision nor a rule has anything to decide.

| setting   | proposal                                                                                                    | reason                                                                                                             |
| --------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| trace     | FreeInference without the SWE-bench agents, 65,536-token limit                                              | those agents wait 2.3 s a turn and almost never idle; the other harnesses wait 44.6 s (68.4 s under the cap below) |
| waits     | x1, capped at 3,600 s (1,200 s in campaigns 28 and 29)                                                      | idle sessions are what reaches the SSD                                                                             |
| load      | 0.032 users/s, cap 128 kept as a guard                                                                      | 80 % of the GPU by the model; with the cap not binding every arm gets the same users                               |
| host pool | 100 GB                                                                                                      | the session KV settles at 132 to 165 GB by the two models on record, so 17 to 39 % of it lies past the pool        |
| arms      | the five, and the just-in-time engine with no hints                                                         | the sixth says whether hints are needed at all                                                                     |
| first     | a 30-minute trial                                                                                           |                                                                                                                    |
| to report | TTFT after waits over 60 s beside the mean, p99, host room held by early reads, SSD bytes read and not used | 88 % of turns follow a wait under a minute and stay in GPU memory                                                  |

- What to expect (estimate): little on the mean (campaign 26 measured 0.14 to 0.21 s for exact hints) and the read's
  worth on turns after a long wait: a 30 K history reads in 1.5 s and computes in 3.3 s (speeds measured in
  campaign 28). The storage read thread stays under 40 % busy by the model.
- Open inside it: keep or drop the SWE-bench agents (46 % of campaign 28's requests); the gap cap; 100 or 70 GB.
  An earlier, simpler proposal on the same record keeps every harness at 0.034 users/s with 70 GB.

**D4. Start state of the follow-up: 3 h arms or a warm start?**

- **Status:** open since 2026-11-19. My question "do you want me to prepare the warm start?" has no answer yet.
- Why: below saturation the population fills slowly. In the proposal the KV past a 100 GB pool is 0 at minute 30
  and settles between minutes 60 and 90 (model: `results/campaign28/checks/start_state_proposal.out`).
- Options: 3 h arms counting hours 2 and 3 (no code; six arms take about 19 h, derived from 3 h 6 min an arm); or a
  warm start in the arrival mode of `scripts/replay_programs.py` (the settled population seeded mid-session at
  t = 0, drawn in proportion to session length, primed), which makes a 75-minute arm worth a 2 h one (six arms
  about 8 h, derived, plus the client change and its checks).
- My preference: 3 h arms for this follow-up. The primer computes every initial history at once and leaves the
  tiers in priming order, not as an arm's policy would, and its first minutes have to be left out anyway. A warm
  start pays off over several campaigns.
- A client change needs a new campaign folder (Traps).

## 5. Steps for the return

The steps run in this order. §5.1 to §5.3 change nothing and can be run at any time; §5.4 and §5.5 need a fix in the
working tree (D1) and are not started. At the end a fixed engine has been compared with campaign 29's `jit_p50` on
the same users.

### 5.1 Box state

**Status:** checked 2026-11-20 about 03:00Z: container up, GPU 0 MiB, launcher idle (2 processes), weights helper
running, no server, SSD 85 % used (the last arm's store; the next arm wipes it).
**Goal:** know whether the launcher of campaigns 28 and 29 is still alive before touching a queue file. While it is,
a line appended to either `queue.txt` starts a 2 h arm within a minute.
**Runs on:** host, any directory
**Touches:** read-only (the second block writes two empty files)
**Takes:** under 1 min, GPU idle

```bash
# host: container, GPU, launcher, helper, server and SSD, one labelled line each
echo "container: $(docker ps --format '{{.Names}} {{.Status}}' | grep sglang_hicache)"
echo "gpu used MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
echo "launcher processes: $(pgrep -f '[r]un_campaigns.sh' | wc -l)"
echo "weights helper processes: $(pgrep -f '[w]arm_weights.sh' | wc -l)"
echo "server processes: $(docker exec sglang_hicache pgrep -f '[s]glang.launch_server' | wc -l)"
echo "ssd used: $(df --output=pcent /mnt/ssd | tail -1 | tr -d ' ')"
echo "last launcher line: $(tail -1 /home/wanhr/sglang/agent_cache/results/campaign29/launcher.log)"
```

-> prints `sglang_hicache Up ...`, `0 MiB`, `2` launcher processes (the `bash -c` wrapper and the script; `0` after
a host restart), `1` helper (or `0`), `0` server processes, a percentage, and a launcher line with `IDLE`. Pass: no
server process and under 1,000 MiB on the GPU.

```bash
# host: end the idle launcher (optional; it exits within a minute, nothing is deleted)
cd /home/wanhr/sglang/agent_cache && tail -1 results/campaign29/launcher.log | grep -q ' IDLE ' \
  && touch results/campaign28/END results/campaign29/END && echo "END files written" \
  || echo "STOP: the last launcher line is not IDLE (an arm is running, or it has ended); nothing written"
```

-> prints `END files written`; a minute later the first block shows `0` launcher processes and both launcher logs
end with `ALL DONE`. Needed only before another launcher starts (a second instance is refused). §5.5 does not need
it: it uses the idle launcher, or starts it again.

**Expected result:** the first block passes; the launcher is either idle or gone.
**If it differs:** container `Exited`: the host restarted, `docker start sglang_hicache`. A server process with no
arm running: `docker exec sglang_hicache bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh`.
**Expected lessons:** the launcher outlives its queues by design, so an edit of a queue file is a launch.

### 5.2 Reproduce the evidence

**Status:** done 2026-11-20; the outputs equal the stored ones.
**Goal:** see both defects in the logs before any code changes, and have the numbers a fixed run is compared with.
**Runs on:** host, `/home/wanhr/sglang/agent_cache`
**Touches:** read-only (prints to the terminal)
**Takes:** under 1 min, GPU idle

```bash
# host: requests whose allowance went on empty lookups, by the time after arrival it was used up (TraceLab, then FreeInference)
cd /home/wanhr/sglang/agent_cache \
  && python3 -I -B results/campaign29/checks/cap_path.py results/campaign29 jit_exact jit_p50 ec_full \
  && python3 -I -B results/campaign29/checks/cap_path.py results/campaign28 jit_exact jit_p50 ec_full
```

-> prints a header and three rows for each campaign. Pass: the TraceLab rows show `369`, `329`, `236` under "at
risk" and `291, 275`, `252, 243`, `29, 14` under "marked 55-70 s: n, refused" (the table of §2).

```bash
# host: one refused request from arrival to admission (campaign 29, jit_p50, user 58, turn 11)
cd /home/wanhr/sglang/agent_cache && grep -a -E "(rid|req)=jit_p50-c58-t11[ ,]" \
  results/20261119_211511_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160/server.log | cut -c1-200
```

-> prints 24 lines. Pass: three `s2h_query ... storage_hit=0` lines, then `ntp_reevict ... refetches=2 capped=1`,
then `ntp_gate ... decision=capped matched=0`, and a last line with `cached_input_len=0` (the table of §1).

```bash
# host: lookups that found nothing, of all lookups, for the three arms with the gate in campaign 29
cd /home/wanhr/sglang/agent_cache && for a in jit_exact jit_p50 ec_full; do
  log=results/$(awk -v l=$a '$1 == l { print $2 }' results/campaign29/manifest.txt)/server.log
  echo "$a: $(grep -a 's2h_query' $log | grep -a -c 'storage_hit=0$') of $(grep -a -c 's2h_query' $log) lookups found nothing"
done
```

-> prints `jit_exact: 4006 of 6390`, `jit_p50: 3917 of 6409`, `ec_full: 4212 of 6994`. Pass: those numbers.

**Expected result:** the three blocks print the numbers above.
**If it differs:** `No such file`: a run folder named in `results/campaign29/manifest.txt` is missing from this
checkout; the server logs are in git since `fe7d9a10bc`.
**Expected lessons:** the 55 to 70 s column is defect 2 and every other column is defect 1 alone; a fixed run is
judged on the same two outputs.

### 5.3 CPU tests around a fix

**Status:** baseline taken 2026-11-20 on the unfixed tree: `18 passed`, and `67 passed` with `52 subtests passed`.
No fix is written.
**Goal:** show that a fix changes what it means to and nothing else: the two test files that cover the eviction
order and the cap pass before it, and pass after it with the tests it rewrites and adds.
**Runs on:** host, into container `sglang_hicache`
**Touches:** read-only (no bytecode, no pytest cache)
**Takes:** about 30 s, GPU idle

```bash
# host -> container: the unit tests of the eviction order and of the prefetcher, each file in its own process
docker exec -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c 'cd /sgl-workspace/sglang && for t in \
  test/registered/unit/mem_cache/test_next_use_eviction.py test/registered/unit/mem_cache/test_next_turn_prefetch.py; do
  echo "== $t: $(python3 -m pytest -q -p no:cacheprovider "$t" 2>&1 | tail -1)"; done'
```

-> prints two lines. Pass today: `18 passed` and `67 passed ... 52 subtests passed`, with no `failed` or `error`.
After a fix the counts rise by the tests added.

**Expected result:** two passing lines. A fix should add at least three tests: a turn queued over 60 s under
`next_use` outlasts every session that has not arrived; a lookup that finds nothing does not use the allowance; a
history that leaves memory after three empty lookups is still read back.
**If it differs:** a failure before any change means the working tree moved since `fe7d9a10bc`;
`git log -3 -- python/sglang/srt/mem_cache` shows how.
**Expected lessons:** three tests pin today's behaviour (named in §1 and §2). They are the ones to rewrite, so a red
run on exactly those three is the fix at work. The wider gate of eight files is `ORACLE_V2_HANDOFF.md` §H3; run it
before §5.4.

### 5.4 Freeze the fixed engine

**Status:** not started (needs D1 and the fix). The block was tried on 2026-11-20 against a scratch folder in the
container's `/tmp`: it printed `frozen: ..., 0 diff lines against fe7d9a10bc` and refused a second run.
**Goal:** a copy of `python/` for the arms that run the fix, so copy d stays what the campaigns ran and later edits
do not reach a running arm.
**Runs on:** host, into container `sglang_hicache`
**Touches:** creates `agent_cache/frozen/<FIXED_ENGINE>/` (110 MB, measured). `<FIXED_ENGINE>` is a new folder name
of your choice, for example `engine_capfix_20261125`; it is used again in §5.5.
**Takes:** under 1 min, GPU idle

```bash
# host -> container: copy the working tree's engine into a new frozen folder (refuses to overwrite an existing one)
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang && F=agent_cache/frozen/<FIXED_ENGINE> && [ ! -e "$F" ] \
  && mkdir -p "$F" && cp -r python "$F/python" && git -c safe.directory="*" diff HEAD -- python > "$F/changes_vs_HEAD.patch" \
  && echo "frozen: $F, $(wc -l < "$F/changes_vs_HEAD.patch") diff lines against $(git -c safe.directory="*" rev-parse --short HEAD)" \
  || echo "STOP: $F exists or the copy failed; nothing overwritten"'
```

-> prints `frozen: agent_cache/frozen/<FIXED_ENGINE>, N diff lines against <commit>`; N is 0 when the fix is
committed. Pass: the line starts with `frozen:`.

**Expected result:** the folder exists and holds `python/` and `changes_vs_HEAD.patch`.
**If it differs:** `STOP`: the name is taken; pick another, never overwrite a frozen copy.
**Expected lessons:** none beyond the convention: the working tree equals copy e, and e differs from d by one unused
score, so the new copy differs from the campaigns' engine by the fix alone.

### 5.5 Run `jit_p50` again on TraceLab with the fixed engine

**Status:** not started (needs §5.4; a queue change, so the author's call).
**Goal:** show the fix in a run. The arm, trace, users and seed are those of campaign 29's `jit_p50`, so its
requests pair one to one with the five arms on record.
**Runs on:** host, `/home/wanhr/sglang/agent_cache`
**Touches:** appends one line to `results/campaign29/queue.txt`. The launcher then deletes everything under
`/mnt/ssd/hicache_l3` (the last arm's KV pages; nothing else lives there), writes a new run folder under `results/`,
and appends to `results/campaign29/manifest.txt` and `launcher.log`. The block that starts the launcher again also
empties `results/.current_run`.
**Takes:** 2 h 6 min (measured for campaign 29's arms), GPU busy

```bash
# host -> container: campaign 29's pinned inputs are unchanged (an arm refuses to start otherwise)
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts \
  && sha256sum --quiet -c ../results/campaign29/inputs.sha256 && echo "inputs of campaign 29: unchanged"'
```

-> prints `inputs of campaign 29: unchanged`. A `FAILED` line names a changed script: restore it, or run the arm in
a new campaign folder.

```bash
# host: queue the arm; a live launcher starts it within a minute
cd /home/wanhr/sglang/agent_cache && [ -d frozen/<FIXED_ENGINE>/python ] && ! grep -q '^jit_p50_fix ' results/campaign29/queue.txt \
  && echo 'jit_p50_fix three_tier_wc_oracle_v2_nu_owned dist --hint-no-call <FIXED_ENGINE>' >> results/campaign29/queue.txt \
  && echo "queued: $(tail -1 results/campaign29/queue.txt)" \
  || echo "STOP: frozen/<FIXED_ENGINE>/python is missing or the label is already queued; nothing written"
```

-> prints `queued: jit_p50_fix three_tier_wc_oracle_v2_nu_owned dist --hint-no-call <FIXED_ENGINE>`.

```bash
# host: only if §5.1 showed 0 launcher processes: start the launcher again (finished arms are skipped)
cd /home/wanhr/sglang/agent_cache && : > results/.current_run && docker start sglang_hicache && docker exec -d sglang_hicache \
  bash -c 'nohup bash /sgl-workspace/sglang/agent_cache/frozen/oracle_scripts_staging/c28_users/run_campaigns.sh campaign28 campaign29 >> /sgl-workspace/sglang/agent_cache/results/c28_runner.log 2>&1' \
  && echo "launcher started"
```

-> prints `sglang_hicache` and `launcher started`; the launch of 2026-11-19 used the same `docker exec` line.

```bash
# host: only if §5.1 showed 0 helper processes: reread the model weights into the page cache at each launch
cd /home/wanhr/sglang/agent_cache && { setsid nohup bash frozen/oracle_scripts_staging/c28_users/warm_weights.sh \
  >> results/c28_warm.log 2>&1 < /dev/null & } && echo "helper started"
```

-> prints `helper started`. Without it a boot can crawl at 9 MB/s and miss the launcher's 40-minute limit.

```bash
# host: watch the arm (LAUNCH, then DONE or STOP), and count the gate's refusals while it runs
cd /home/wanhr/sglang/agent_cache && echo "launcher: $(tail -1 results/campaign29/launcher.log | cut -c1-160)" \
  && echo "refused so far: $(grep -a 'ntp_gate' results/$(cat results/.current_run)/server.log | grep -a -c 'decision=capped')"
```

-> campaign 29's `jit_p50` had 39, 88 and 145 refusals about 30, 40 and 60 minutes into the client's run and 334 at
its end (measured). Pass: far fewer; under 20 at the end (derived: 332 requests were refused there, 316 of them on
empty lookups). To stop the arm, for errors only (a slowdown is a result):
`docker exec sglang_hicache bash -c 'pkill -f "[r]un_campaigns.sh"; pkill -f "[r]eplay_programs.py"; pkill -f "[i]ostat -dxt vdc"; bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh'`.

```bash
# host: after the DONE line: every per-arm check, paired with the two arms it is compared with, then the cap path
cd /home/wanhr/sglang/agent_cache && bash frozen/oracle_scripts_staging/c28_users/arm_checks.sh campaign29 jit_p50_fix jit_p50 ec_full \
  && python3 -I -B results/campaign29/checks/cap_path.py results/campaign29 jit_p50 jit_p50_fix ec_full
```

-> prints one `ok` line per check, then the cap-path rows of the old and the new run. The figure step may print
`FAIL` (Traps); the checks before it do not depend on it.

**Expected result** (estimates from §3, to be replaced by the run): in `results/campaign29/checks/` the file
`cap_share_jit_p50_fix.out` shows under 20 requests refused (313 of 2,982) and under 1 % of the history computed
again (10.8 %); `cap_path` shows the 55 to 70 s cell near 0 (252); `gpu_work.out` shows about 37 requests a minute
in minutes 30 to 110 (30.4). `results/campaign29/compare_arms.out` then shows a mean TTFT near 140 s (164.2 s; 150.7 s
for `ec_full` on copy d).
**If it differs:** no `LAUNCH` line after two minutes: `tail -3 results/campaign29/launcher.log` shows a `STOP` with
its reason, and the launcher has exited; fix the cause and start it again. Refusals as before: the arm did not run
the fix; `engine=` in the run folder's `run.txt` names the copy it imported.
**Expected lessons:** refusals gone and about 37 requests a minute: the differences between the three just-in-time
arms on TraceLab were the defects, and D2 decides whether `ec_full` runs again. Refusals gone and the rate
unchanged: another limit binds (the read thread, decode) and the estimate of §3 was wrong. Refusals still there:
they are not the empty lookups of §1; rerun §5.2 on the new run.

## 6. Smaller open items

A list to return to, one line each, not evaluated here.

- A no-hint arm on the just-in-time engine at the saturated setting (campaign 26's `nofc` and `nofc_noend`,
  `results/campaign26/DECISIONS.md`, "Arms"). It says whether hints are needed there at all.
- Hint tables are kept per harness and model and learned online; their scope is open since campaign 27.
- Prompt tokens are uniform random ids: no system prompt shared between sessions, and replies come back as new ids.
  A client that sent back the ids the server generated would behave like a real one and take away most of defect
  1's trigger (my reading; a client change, not looked into).
- Which rivals from the literature join the five arms (open since campaign 27).
- A dense second model, where a miss costs more (campaign 28's record, "Discussion with the author").
- One run per arm: the spread between runs is not measured at this setting. On TraceLab only 111 of the 355 and 332
  refused requests of `jit_exact` and `jit_p50` are the same request.
- The storage read thread is single and reads 17 to 20 K tokens a second; it was 83 to 96 % busy in FreeInference's
  hinted arms (`results/campaign28/checks/read_thread.out`). A follow-up should keep it well under full.
- A traceback at server stop (`prefetch_io_aux_func`, `'NoneType' object has no attribute 'data_ptr'`), after the
  client's summary: a harmless race, an engine item of its own.
- `results/_trial_c128/` is a leftover of a trial; a safety check refused my `rm`, so removing it is the author's.
- Campaign 26's parked items: `results/campaign26/checks/audit/items.tsv`.
- Held until asked: bringing `PAPER_PLAN.md`, the findings docs and the deck up to campaigns 28 and 29.

## Traps

- The launcher of campaigns 28 and 29 is alive and idle: appending to a `queue.txt` is a launch (§5.1).
- An arm of campaign 28 or 29 refuses to start once its trace or one of `start_server.sh`, `stop_server.sh`,
  `start_programs.sh`, `replay_programs.py`, `program_trace.py`, `gap_hints.py`, `program_summary.py` has changed
  (`inputs.sha256`). The engine folder is not pinned. So an engine fix can run inside campaign 29 (§5.5); a client
  change, such as the warm start, needs a new campaign folder.
- Frozen copies are records: fix in `python/`, freeze a new copy (§5.4), never edit copy d or e.
- The SSD cleaner starts at 90 % of the volume. Campaign 29's arms ended at 49, 58, 77, 79 and 85 % for 2,090 to
  4,016 requests, about 1.9 points per 100 requests. Estimate: a fixed arm passes 90 % in its last minutes, after
  the 6,600 s cut the comparison counts to. The `DONE` line prints `cleaner passes`; a pass before the cut changes
  what the store holds and has to be reported with the arm (§5.5).
- The figure script styles five labels (`frozen/oracle_scripts_staging/c28_users/plot_arms.py:53`). A sixth label
  such as `jit_p50_fix` may fail the last step of `arm_checks.sh`; not tried (§5.5).
- The host is preemptible and its `/tmp` is wiped on a restart. An interrupted arm leaves no manifest line and runs
  again from its start once the launcher is started (§5.5).
- Stop a run only for errors; a slowdown is a result to report.

## Layout and hygiene

- This file is `agent_cache/OPEN_ITEMS.md`. When an item closes, mark it here with the date and where its result is.
- Records: `results/campaign28/DECISIONS.md`, `results/campaign29/DECISIONS.md` (its last section sets both
  campaigns side by side).
- Checks: `results/campaign28/checks/*.py` and `results/campaign29/checks/cap_path.py`; each prints its usage in its
  docstring. Their `.out` files are not in git (`*.out` is ignored); every one is rebuilt by its script from the run
  folders, which are in git.
- Engine copies: `frozen/engine_ec_20261111d` (campaigns 28 and 29), `frozen/engine_ec_20261111e` (equal to the
  working tree at `fe7d9a10bc`).
- Nothing is left to revert: this file and a pointer to it at the end of each record are the only changes of
  2026-11-20 after the author's commit `fe7d9a10bc`.
