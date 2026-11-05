# Paper plan

Started 2026-11-01 after "speed up and get this project published sooner". "Published" is read as a paper; nothing gets submitted, posted or sent upstream without asking.

## The claim

An end-to-end system where forecasts from the agent drive next-use eviction, deadline-ordered SSD staging and read pins across GPU, host memory and SSD, measured on real hardware. Each part exists elsewhere; I found nothing that combines them (`frozen/hint_realism_20261027/reports/lit.md`, section 8). The results decide how strong the claim is: campaign 22 shows how much of the gain needs hints at all, and campaign 23 shows whether a routine that uses the timing beats one that does not. At 44 users the first question is answered (`CAMPAIGN23_FINDINGS.md`, 2026-11-04): read pins with no hint give 79 % of the gain over original HiCache, realistic p50 hints add nothing measurable on top, and exact hints add 0.96 s, only under the just-in-time routine. At 128 users it may go the other way (campaign 25, 2026-11-04): the hint-free arm keeps about 58 % of the gain (29.19 s against 53.12 s for original HiCache and 12.06 s for the oracle on 8,175 shared turns), and p50 hints keep nearly all the rest (12.14 s, 12.56 s under the just-in-time routine). But that arm also lacks three mechanisms the hinted arms run from queue state alone (a queued turn's history kept, its read paced by queue position, lost history re-read), and its losses go through them (check of 2026-11-05), so 42 % is an upper bound on what the hint's information is worth; a hint-free engine with those three is untested.

Campaign 23's results and the options they open are in `CAMPAIGN23_FINDINGS.md`.

## Contributions and where they stand

| Contribution | Evidence | Status |
|---|---|---|
| The system: forecasts driving eviction, staging and pins across three tiers | campaigns 18-21, 24 | at 44 users and 70 GB over 2 h on 7,768 shared turns: baseline 5.17 s, oracle 1.48 s, p50 1.95 s, just-in-time reads with the arrival gate and hold 0.50 s with exact hints and 1.51 s with p50 hints; turns 8,037, 10,319, 10,021, 12,201, 10,419. One run each, except that campaign 25 repeated the baseline, 1.43 s slower on the turns both runs share, and reran the just-in-time arms at a faster arrival rate, within 0.04-0.12 s. At 128 users and 160 GB, first 2 h on 8,175 shared turns: baseline 53.12 s, oracle 12.06 s, p50 12.14 s, just-in-time with exact hints 12.46 s and with p50 hints 12.56 s; TTFT there is the GPU's queue, so the new routine's gain is I/O only. With exact hints: one read per reading turn, 29 % fewer SSD pages, SSD 47-78 % busy instead of 98 %. With p50 hints: 1.031 reads per reading turn, 15 % fewer pages, SSD 60-92 % busy. Against a hint-free arm with read pins (1.47 s at 44 users), only the just-in-time routine on exact hints gains; at 128 users the same hint-free arm is at 29.19 s against 12.06-12.56 s for every hinted arm, p50 included |
| Latency against load: where the knee sits and how far each routine moves it | `results/compare_c21_arms/load_curve.png`, `results/campaign21/DECISIONS.md` | read from swings in one workload: episodes come and go at the same turns in every arm; at a 5 s target the gain looks like a shift of about 3-5 % for the horizon routine and 11 % for just-in-time reads with the gate and hold on exact hints, 6 % on p50 hints; at 128 users every hinted arm sits far past the knee at about 21 s, and the routine moves nothing; needs the user-count sweep below. The hint-free arm has no curve yet, and by load band it does worse than the oracle near the knee |
| The conditional table: sent with the request, resolved by the server when the reply exists, so no message follows the reply | client, engine, `DIST_HINT_DESIGN.md` | built and committed; table_smoke passed 2026-11-01 and both 2 h p50 arms of campaign 23 ran on it, with the reply key taken from the replay; the engine has never read its own reply on a GPU |
| Where the gain comes from, one change at a time | campaign 22, and the hint-free arm at campaign 21's setting | 44 users done 2026-11-04 (`CAMPAIGN23_FINDINGS.md`): of 5.17 to 0.50 s, read pins with paced retries off and no hint give 79 % (78-83 % against three runs of the baseline), the old routine on exact hints 0 %, just-in-time timing 18 %, the gate and hold 4 % (noise level); 70-90 % of each step is shorter queue wait in one load episode. Campaign 25 separated the two: read pins carry it (75-79 % with paced retries left on, 0.10 s from the hint-free arm), and paced retries off alone gives 7-21 %, inside the baseline's run-to-run spread. 128 users: the hint-free step ran as campaign 25's first arm and keeps about 58 % of the gain, a floor for a hint-free engine since it lacks the hinted arms' queue-state mechanisms; the steps between it and the oracle (`three_tier_wc_oracle_v2` without next-use eviction, `three_tier_wc_poll0`) not run |
| What a realistic forecast costs | campaign 20 (queue), 21 (no queue), 23 (just-in-time reads) | 20 and 21 done: at 44 users and 70 GB the p50 keeps 87 % of the oracle's gain, +1.86 s in the heavy stretch against 0.05 s of repeat noise, measured on a run since deleted. Campaign 23 at 44 users: once the routine uses the timing the forecast costs 1.01 s, not 0.47 s, and keeps 78 % of exact hints' gain; most of the extra recomputation is on reads refused host room, while early-landed reads hold 4.6 times the pinned token-seconds. Pins against eviction not separated. At 128 users and 160 GB the forecast costs 0.07 s of TTFT (0.10 s under the old routine) and no recomputation, with the same forecast error, but about half the I/O saving: 20 % more pages than exact hints, 129 turns read twice and pins twice as heavy. Those double reads and most of the extra pins come from reads sent for turns already past their forecast arrival; a rule against that is not built. Against the hint-free arm at 44 users, p50 hints are worth -0.05 s on 7,768 turns and +0.17 s on 9,688: not separable from zero. At 128 users they are worth up to 16.6-17.1 s per turn on 8,175 turns (campaign 25), an upper bound (see the claim) |
| Failure modes others can reuse: spent retry budgets causing a recompute storm, reads and backups failing near the host mirror's size, evict-then-read-back, an SSD cleaner deleting shared prefixes | `ORACLE_FINDINGS.md` sections 9.4, 9.10, 9.11; campaign 21 | traced; needs writing up |

## Needed before submitting

1. **Repeat runs** of the arms the claims rest on. At 44 users and 70 GB, campaign 25 (2026-11-05, GPU unit `GPU-2c5a8afb`) reran the hint-free arm, new exact and new p50 with users arriving five times faster (same users, same order): they moved +0.01, +0.04 and +0.12 s on the turns each shares with its first run, so they count as near-repeats. It also repeated the baseline at the first rate, and the baseline does not repeat: 7.02 against 5.58 s on 6,812 shared turns, congesting earlier and not recovering by the end, at the same hardware speed. So near its knee original HiCache varies by about 25 % run to run, and the paper should report it as a range or a mean of runs; the hint-free arm's share of the gain is 78-83 % against its three runs. Still missing: the oracle at 44 users (its 0.05-0.14 s spread was measured on runs deleted on 2026-11-01), the old p50, and every arm at 128 users. Record the GPU unit of every run: the VM is preemptible, the unit changed at both 2026-11-01 shutdowns, and campaigns 19-21 have no record of theirs.
2. **Sessions per GPU under a latency target** for the best routine against the baseline. This is the number an operator buys, and it now decides how the claim reads: at 44 users mean TTFT falls 2.7-fold with the p50 and 3.5-fold with exact hints because the setting sits on a steep knee, while the knee itself seems to move only a few percent (`results/campaign21/DECISIONS.md`). Mean TTFT at one setting can overstate the gain as easily as undersell it.
3. **A human-paced workload.** Real use is 92 % human waiting with median contexts of 126K tokens (TraceLab); our trace is batch agents with gaps stretched by a factor. At least a calibrated replay, or state it plainly as a limit.
4. **The hint-free rival.** The model says removing the host mirror cuts SSD reads more than any hint. A reviewer will ask, so measure it or explain why not; write-back's lookup bug (`ORACLE_FINDINGS.md` issue 7) blocks the direct test. A simpler hint-free rival, original HiCache with read pins (the retry setting does not matter, campaign 25), already matches the new routine's p50 arm at 44 users and beats the old routine's by 0.48 s. At 128 users it keeps only about 58 % of the gain (campaign 25), because LRU evicts the history of queued turns and the refused re-reads are never retried; the hinted arms avoid both with mechanisms that use queue state, not the hint. A version of it with those (option I in `CAMPAIGN23_FINDINGS.md`) is the rival a reviewer will ask for.
5. **Recheck prior work** just before submission. SGLang #45128, vLLM #52113 and Dynamo's prefetch hooks are all moving.

## Design point: the hint's form and the engine's policy

Discussed 2026-11-04 and 2026-11-05 to give the paper an algorithmic part. Settled: the client sends a distribution and the engine picks the operating point, and whatever is learned is learned in the engine. Open: the distribution's form. Nothing gets built or run until the author says go. Offline checks: `frozen/hint_policy_20261104/`.

**Why it matters.** Today the engine plans on the p50 as if it were exact: the deadline sits at the entry's p50, the read is timed to land p50 - p10 + 1 s before the forecast admission, and when the plan expires at p50 + max(60 s, p50) the session is marked ended; p90 and n go unused. At 44 users and 70 GB that does no better than no hint (1.51 s against the hint-free arm's 1.47 s), while exact hints under the same routine reach 0.50 s. The forecast's errors are paid in shared host memory:

- **Two-moded entries.** The online table is calibrated overall at x10 (10.9 % of gaps below p10, 45.6 % below p50, 13.3 % above p90), but many entries mix a rejected-call mode near 0.8 s, the 7 s polling grid and a long tail: `python -c` sends p10 0.78 s against p50 7.3 s, `pip` 0.85 s against 22 s. So the early start is a median 6.2 s, and over 3 s on 70 % of staged turns.
- **Long gaps.** Gaps of 60 s or more hold 45 % of idle time, and 79-100 % of them fall above p90.
- **The cost.** Both become reads pinned long before admission, on about 44K tokens of free host room: 1,743 of 5,001 read launches revoked for room against 381 of 2,088 with exact hints, and 88 % of recomputed tokens on turns whose read found its data but was cut or revoked. At 128 users and 160 GB the room is there and the forecast costs 0.07 s, but reads sent for turns already past their forecast arrival cause the 129 double reads.

**Settled: the client sends a distribution, the engine picks the point.** The client knows what the agent is waiting for and its history. Only the engine knows what landing early or late costs at that moment: free host room, and queue wait against read time. One number fixes a trade-off the client cannot see, and the right point moves with server state: with room to spare (128 users, 160 GB, no read revoked) landing early costs nothing, while with 44K tokens of room a rough price (below) puts it near p75-p80. This is the paper's case against sending one number.

**Open: the form.** Sending the distribution and sending all the percentiles carry the same information; what is open is the encoding. My lean is the histogram.

| Form | For | Against |
|---|---|---|
| Quantile list: p10/p50/p90 as today, or more points | what is sent today; easy to read | levels (key, tool, `*`) cannot be merged without the raw gaps; a few points blur two modes; nothing is known past the last point |
| Log-spaced histogram per key on one shared grid (about 24 bins), plus an "ended" bin and n | levels merge by adding counts, so falling back becomes shrinkage; O(1) update, and decay for drift; conditional survival and any quantile read off directly; both modes kept | the grid is fixed in advance; more numbers per key, kept small by sparse counts and a table version |
| A fitted shape: one lognormal, or a mixture | few numbers | one lognormal does not fit these entries; a mixture needs fitting on the client |
| The last gaps per key | no binning | the most bytes; the engine does the estimating |

Independent of the form:
- **Censoring.** Count gaps still running as censored (Kaplan-Meier). That removes today's low bias, where a long gap joins the table only once it ends.
- **Session history** sharpens the tail, not the center: after a same-key gap of 30 s or more, the next same-key gap is that long 35 % of the time against 4 % otherwise, while as a point forecast the session's last gap does worse than the table (mean log error 0.52 against 0.41). Like a learned forecaster on the client (the best offline model reached R² 0.83 against the online table's 0.77), it is a better estimator behind the same form, for later.
- **Wire.** A table version and only the changed entries instead of 3 KB per request. Human waits would need an event kind rather than a tool key.

**The engine's policy: decide by expected cost.** Each decision weighs expected costs under the entry's distribution, conditioned on how long the session has been idle, with host memory charged a price λ per pinned token-second.

1. **When to start a read.** Pick the start time, or "read at arrival", with the lowest λ × tokens × expected time pinned before admission plus expected stall. With no queue the read should land at the quantile q = λ·tokens / (λ·tokens + 1), so large reads and scarce room land later. Reading at arrival wins for wide entries and when the queue wait covers the read. Rechecked every tick against the idle time, so an overdue turn's read moves later instead of going out for a turn already past its forecast arrival.
2. **Soft pins.** A read that landed before its turn arrived can be taken back: an arrived turn's read that needs room takes it from the speculative read least likely to be needed soon, instead of being revoked or cut.
3. **Eviction by conditional survival**: P(return within H | idle so far, key) × reload cost per token. No forecast ends a session; only the end signal does. In the 2026-10-27 trace study this order beats both today's tiers fed point forecasts and LRU (AUC 0.900 against 0.825 and 0.763 at x70; 0.952 against 0.837 and 0.922 at x10).
4. **Learning, in the engine, from outcomes nobody sends.** Raise λ when reads are revoked or cut for room and lower it when room sits idle, so one code path serves both settings. Recalibrate each key from where arrivals fall in the distribution it sent: a key whose arrivals keep landing above its p90 gets a heavier tail.

With exact hints this reduces to today's just-in-time routine, and with an uninformative hint to the hint-free arm. The story for the paper: where memory is short, a point forecast acted on as exact is worth no more than none, because its errors are paid in shared memory; a distribution and a price recover part of that, and the rest is missing information. Prior work to place it against (not yet in `lit.md`): informed prefetching and caching (TIP, SOSP '95) weighs prefetching against caching by cost and benefit, from hints that disclose future reads; integrated prefetching and caching (Cao et al., SIGMETRICS '95) gives the "do no harm" rule that soft pins follow; LHD (NSDI '18) evicts by expected hits per unit of space, as rule 3 does. Here what is read next is known, the session's history, and only when is uncertain, with a wide, two-moded forecast.

**Offline check, first order**, on `jit_p50_c44`'s 1,981 turns staged before arrival, with the distribution built from the sent p10/p50/p90 only. Host room, revokes and the queue are not modelled, so it cannot predict TTFT, and it overstates pins (52K against the 27K the run measured): read the rows against each other.

| Read rule | Mean pinned demand | Total stall |
|---|---|---|
| Today: land at p10 | 52.0K | 40 s |
| Land at p50 | 28.5K | 663 s |
| Land at q = 0.8 | 13.1K | 1,701 s |
| Never before arrival | 1.9K | 2,828 s |
| Expected cost, λ = 5e-6 | 22.1K | 857 s |
| Expected cost, λ = 1e-5 | 12.4K | 1,296 s |
| Expected cost, λ = 2e-5 | 7.7K | 1,561 s |
| Exact arrival | 8.9K | 418 s |

- At the same stall the expected-cost rule needs 14-49 % fewer pinned token-seconds than the fixed quantiles for λ from 5e-6 to 5e-5; at 2e-6 it lands almost everything early and gains nothing. So the distribution's shape is worth something beyond choosing one quantile.
- It cannot reach exact hints: at their pin level it still stalls about 1,100 s more, about 0.1 s per turn over the run.
- If the 1.01 s forecast cost came entirely from the extra 24.7K pinned tokens (30.9K against 6.2K with exact hints), λ is about 4e-5 to 6e-5, which lands a 64K-token read near p75-p80, not p10.
- A crude rule that moves the quantile with current pins (p10 below 15K, p50 below 30K, else p80) does no better than a fixed quantile.

**Next, when the author says go.**
1. CPU, a few hours: add host room and revokes to this model, calibrate it on the pins and revokes of `jit_p50_c44` and `jit_exact_c44`, choose λ, and test whether soft pins pay.
2. Engine, small: rule 1 with a fixed λ, and no forecast ends a session; the plan keeps the whole entry instead of the p50 and the early start. Rules 2 and 3 after, each behind its own flag.
3. GPU, 2 h arms: rule 1 at 44 users / 70 GB and at 128 users / 160 GB, against `jit_p50`, `jit_exact` and the hint-free arm on the same turns.

A probe that needs only the client: send p10 = p50, so reads land at the p50. It shows whether long pins are the lever.

## Decisions for the author

- Venue and deadline, which set how much of the list above fits.
- Which result leads, once campaign 22 and campaign 23 are in.
- Whether the engine work also goes upstream, and when.
- The hint's form, and whether the policy in the design point above gets built.

## When results arrive (UTC)

| Results | Expected |
|---|---|
| Campaign 24: just-in-time probe at 70 GB | about 06:20, 2026-11-01 |
| Campaign 21: p50 | done 08:22 |
| Campaign 21: baseline | done 10:24 |
| Campaign 21: oracle | done 12:26 |
| Hint-free arm at campaign 21's setting | cut at 117 of 120 min when the host went down at 14:25 |
| Campaign 22 | not running since 14:25, not restarted |
| Campaign 23: two smokes | passed 19:49 and 20:21 |
| Campaign 23: `jit_exact_c44` | done 22:24 |
| Campaign 23: `jit_p50_c44` | done 00:26, 2026-11-02 |
| Campaign 23: `jit_exact_c128` | done 02:30, 2026-11-02 |
| Campaign 23: `jit_p50_c128` | done 04:35, 2026-11-02; campaign finished |
