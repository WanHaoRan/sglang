# Paper plan

Started 2026-11-01 after "speed up and get this project published sooner". "Published" is read as a paper; nothing gets submitted, posted or sent upstream without asking.

## The claim

An end-to-end system where forecasts from the agent drive next-use eviction, deadline-ordered SSD staging and read pins across GPU, host memory and SSD, measured on real hardware. Each part exists elsewhere; I found nothing that combines them (`frozen/hint_realism_20261027/reports/lit.md`, section 8). The results decide how strong the claim is: campaign 22 shows how much of the gain needs hints at all, and campaign 23 shows whether a routine that uses the timing beats one that does not.

Campaign 23's results and the options they open are in `CAMPAIGN23_FINDINGS.md`.

## Contributions and where they stand

| Contribution | Evidence | Status |
|---|---|---|
| The system: forecasts driving eviction, staging and pins across three tiers | campaigns 18-21, 24 | at 44 users and 70 GB over 2 h on 7,768 shared turns: baseline 5.17 s, oracle 1.48 s, p50 1.95 s, just-in-time reads with the arrival gate and hold 0.50 s with exact hints and 1.51 s with p50 hints; turns 8,037, 10,319, 10,021, 12,201, 10,419. One run each. At 128 users and 160 GB, first 2 h on 8,175 shared turns: baseline 53.12 s, oracle 12.06 s, p50 12.14 s, just-in-time with exact hints 12.46 s and with p50 hints 12.56 s; TTFT there is the GPU's queue, so the new routine's gain is I/O only. With exact hints: one read per reading turn, 29 % fewer SSD pages, SSD 47-78 % busy instead of 98 %. With p50 hints: 1.031 reads per reading turn, 15 % fewer pages, SSD 60-92 % busy |
| Latency against load: where the knee sits and how far each routine moves it | `results/compare_c21_arms/load_curve.png`, `results/campaign21/DECISIONS.md` | read from swings in one workload: episodes come and go at the same turns in every arm; at a 5 s target the gain looks like a shift of about 3-5 % for the horizon routine and 11 % for just-in-time reads with the gate and hold on exact hints, 6 % on p50 hints; at 128 users every hinted arm sits far past the knee at about 21 s, and the routine moves nothing; needs the user-count sweep below |
| The conditional table: sent with the request, resolved by the server when the reply exists, so no message follows the reply | client, engine, `DIST_HINT_DESIGN.md` | built, unit-tested; GPU smoke queued |
| Where the gain comes from, one change at a time | campaigns 19 and 22, and the hint-free arm at campaign 21's setting | queued |
| What a realistic forecast costs | campaign 20 (queue), 21 (no queue), 23 (just-in-time reads) | 20 and 21 done: at 44 users and 70 GB the p50 keeps 87 % of the oracle's gain, +1.86 s in the heavy stretch against 0.05 s of repeat noise, measured on a run since deleted. Campaign 23 at 44 users: once the routine uses the timing the forecast costs 1.01 s, not 0.47 s, and keeps 78 % of exact hints' gain; most of the extra recomputation is on reads refused host room, while early-landed reads hold 4.6 times the pinned token-seconds. Pins against eviction not separated. At 128 users and 160 GB the forecast costs 0.07 s of TTFT (0.10 s under the old routine) and no recomputation, with the same forecast error, but about half the I/O saving: 20 % more pages than exact hints, 129 turns read twice and pins twice as heavy. Those double reads and most of the extra pins come from reads sent for turns already past their forecast arrival; a rule against that is not built |
| Failure modes others can reuse: spent retry budgets causing a recompute storm, reads and backups failing near the host mirror's size, evict-then-read-back, an SSD cleaner deleting shared prefixes | `ORACLE_FINDINGS.md` sections 9.4, 9.10, 9.11; campaign 21 | traced; needs writing up |

## Needed before submitting

1. **Repeat runs** of the arms the claims rest on. No arm at 44 users and 70 GB has a repeat on disk. The oracle repeated to within 0.05-0.14 s over its first 76 min, but those runs, the two stopped attempts, were deleted on 2026-11-01, so that spread cannot be recomputed and needs a fresh repeat before a paper leans on it. The baseline and the p50 never had one. Run repeats on one GPU unit and record it: the VM is preemptible and the unit's UUID changed at both 2026-11-01 shutdowns, and campaigns 19-21 have no record of theirs. Campaign 23 is in: the just-in-time routine leads at 44 users and ties at 128 with less I/O, so repeat its two 44-user arms and the oracle first.
2. **Sessions per GPU under a latency target** for the best routine against the baseline. This is the number an operator buys, and it now decides how the claim reads: at 44 users mean TTFT falls 2.7-fold with the p50 and 3.5-fold with exact hints because the setting sits on a steep knee, while the knee itself seems to move only a few percent (`results/campaign21/DECISIONS.md`). Mean TTFT at one setting can overstate the gain as easily as undersell it.
3. **A human-paced workload.** Real use is 92 % human waiting with median contexts of 126K tokens (TraceLab); our trace is batch agents with gaps stretched by a factor. At least a calibrated replay, or state it plainly as a limit.
4. **The hint-free rival.** The model says removing the host mirror cuts SSD reads more than any hint. A reviewer will ask, so measure it or explain why not; write-back's lookup bug (`ORACLE_FINDINGS.md` issue 7) blocks the direct test.
5. **Recheck prior work** just before submission. SGLang #45128, vLLM #52113 and Dynamo's prefetch hooks are all moving.

## Decisions for the author

- Venue and deadline, which set how much of the list above fits.
- Which result leads, once campaign 22 and campaign 23 are in.
- Whether the engine work also goes upstream, and when.

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
