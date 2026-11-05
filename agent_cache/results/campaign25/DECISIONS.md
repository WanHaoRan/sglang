# Campaign 25: does the 128-user gain need hints, and how the 44-user arms behave under more load

Requested 2026-11-04: "1) with 128 users, do a hint-free arm to study the hint's effectiveness, 2) in the 44 users scenario,
let's increase the QPS to 0.5 and 1, for the original hicache, hint free, jit exact and jit_p50, each 2 hours."

**Launcher.** `frozen/oracle_scripts_staging/c25_load/c25.sh` runs `queue.txt` line by line and is resumable through
`manifest.txt`; `watch.sh` beside it wakes the session on an arm's end or an error. Each arm wipes the SSD first
(`run_compare.sh`), as every earlier campaign did.

**Arm 1, `hintfree_c128`.** `three_tier_wc_dl_poll0` (read pins, paced retries off, no hints) on campaign 19's workload: 128
users, 160 GB, `USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 SEED=0 ORACLE_HINTS=1`, 2 h, on `frozen/engine_5df71c4bf0_v2` like
campaigns 19-20 and the 44-user hint-free arm. Hints are sent and ignored, so request bodies match campaign 19's. Compared
on the same turns, first 2 h, with campaign 19's original HiCache and oracle, campaign 20's p50 and campaign 23's
`jit_exact_c128` and `jit_p50_c128`. At 44 users the same arm tied the oracle (`../../CAMPAIGN23_FINDINGS.md`).

**Arm 1 result.** Done 2026-11-04 23:40Z, `20261104_213242_three_tier_wc_dl_poll0_NAT160`: 10,454 turns, no client errors,
no tracebacks. On the 8,175 returning turns all six 128-user arms share, mean TTFT is 53.12 s for
original HiCache, 29.19 s hint-free, 12.06 s oracle, 12.14 s old p50, 12.46 s new exact and 12.56 s new p50, so the
hint-free arm keeps about 58 % of the gain. On the 10,066 turns it shares with the oracle it is at 41.04 s against 15.82 s,
and 98 % of that gap is queue wait.

- Same GPU speed as campaign 19 (decode 49.5 against 49.2 ms per token; first-turn prefill 0.108 against 0.111-0.113 ms
  per token), and the arms match for the first 30 min.
- 36 % of its returning turns lose a mean 18.3K tokens of GPU history while queued, against about 2 % in the oracle (whose
  shrink line shows 5, because its prefetcher re-read lost history for 304 more before admission). The engine reads the
  lost part again, but with no hints every read is due now, which skips the read window and budget: reads in flight hold
  a median 494K of the 959K tokens of host room at dispatch, 535 reads (first 2 h) found none and were revoked, and about
  630 lookups were declined at the limiter (reads in flight plus pins at half the host pool; inferred, no line logs it).
  Poll interval 0 cancels the paced retry both refusals arm, and this arm has no prefetcher to re-read.
- 21.2M tokens recomputed in 2 h, over 80 % on those refused or declined turns. The GPU prefills about three times the
  oracle's tokens, so the queue grows.
- **Check, 2026-11-05** (asked whether this confirms the hint is useful). A read-only check of the setup, the engine paths
  and the logs found nothing besides the arm that could move TTFT by seconds. But the arm lacks three mechanisms the
  hinted arms run from queue state alone: a queued turn's history marked as needed now on arrival
  (`next_turn_prefetch.py:439-445`), read deadlines from queue position, and re-reads for queued turns. They live in the
  prefetcher and need a plan, so they went with the hint, and the hint-free losses go through them. The 42 % is therefore
  an upper bound on what the hint's information is worth here. At onset (minutes 35-40) the hint-free arm held less host
  room for reads than the oracle and already lost 106 queued turns to its 15, so host room follows the losses. 305 of 314
  turns first listed as "history in no tier" were limiter declines that never queried the SSD. Findings:
  `verify_hint_gap_20261105.txt`; the check's scripts: `frozen/oracle_scripts_staging/c25_c128/verify_20261105/`.

Tables: `compare_c128.out` (same turns, all six arms), `why_hintfree_c128.out` (speed, queue, GPU work, SSD, history loss)
and `shrink_outcome.out` (what follows a loss). Figures: `../compare_c25_c128/`. Scripts:
`frozen/oracle_scripts_staging/c25_c128/`.

**Part 2, the 44-user arms.** At 44 users the client's cap of 44 sessions is full from about minute 10 at today's
`ARRIVAL=0.1` users/s (up to 420 users wait for a slot), so raising `ARRIVAL` to 0.5 or 1 changes only how fast the slots
fill. Asked how to raise the load (more users, shorter gaps, or the arrival rate as asked), the user chose the arrival rate
as asked. So these arms test a faster start and then repeat today's load; they also act as near-repeats on one GPU unit.

| Label | Arm | Hints | Engine |
|---|---|---|---|
| `base_a05`, `base_a1` | `three_tier_wc` (original HiCache) | sent, ignored | `engine_5df71c4bf0_v2` |
| `hintfree_a05`, `hintfree_a1` | `three_tier_wc_dl_poll0` | sent, ignored | `engine_5df71c4bf0_v2` |
| `jit_exact_a05`, `jit_exact_a1` | `three_tier_wc_oracle_v2_nu_owned` | exact | `engine_owned_20261101` |
| `jit_p50_a05`, `jit_p50_a1` | `three_tier_wc_oracle_v2_nu_owned` | online p50 table | `engine_owned_20261101` |

All at 44 users, 70 GB, `GAP=10`, `USERS=768`, 2 h, `ARRIVAL` 0.5 then 1. **The four `ARRIVAL=1` arms were cancelled for
now by request on 2026-11-04 ("cancel the QPS=1 case for now and only run the QPS=0.5 case"), while arm 1 ran; their lines
stay in `queue.txt` as comments, so restoring them is removing the `#`.** The same seed gives the same users in the same
order at every rate, with arrival times scaled by the rate, so turns pair across rates and with campaigns 21 and 23 at
`ARRIVAL=0.1`. All 768 users arrive within 26 min at 0.5/s and 13 min at 1/s; about 270-400 of them start in 2 h, so the
cap stays full to the end.

**`base_a05`.** Done 2026-11-05 01:46Z, `20261104_234044_three_tier_wc_custom_L1nat_H70`: 7,004 turns (8,037 at
`ARRIVAL=0.1`), no client errors, no tracebacks. Mean TTFT 7.54 s against 5.16 s; on the 6,762 returning turns both
runs share, 7.57 s against 5.54 s. Hardware speed matches the 0.1 run (decode 45.5 against 47.1 ms per step at 16-23
running, first-turn prefill 0.087 against 0.089 ms per token, SSD 2.98 against 2.87 ms per page). By send order the
faster start carries more load from about position 3,500 (923-1,067K tokens against 880-1,010K), so a first congested
stretch comes at minutes 30-55 (2-6 s), and the heavy stretch from position 5,500 lasts to the end of the 2 h instead of
easing by minute 105. Original HiCache's repeat at 0.1 does the same (`base_a01_rep` below), so the faster start is not why.
Script: `frozen/oracle_scripts_staging/c25_a05/compare_a05.py`; the four arms are compared once
all have landed.

**`hintfree_a05`.** Done 2026-11-05 03:48Z, `20261105_014608_three_tier_wc_dl_poll0_custom_L1nat_H70`: 11,063 turns
(10,355 at `ARRIVAL=0.1` in its 117 min), no client errors, no tracebacks. Unlike the baseline it repeats its 0.1 run: 1.84
against 1.83 s on the 10,012 returning turns both share, with the same episodes at the same minutes. On the 6,762 turns it
shares with `base_a05` it is at 1.85 s against 7.57 s (1.61 against 5.54 s at 0.1 on the same turns). Hardware speed
matches (decode 48.6 against 48.5 ms per step at 16-23 running).

**`jit_exact_a05`.** Done 2026-11-05 05:50Z, `20261105_034841_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70`: 12,532
turns (12,201 at `ARRIVAL=0.1`), no client errors, no tracebacks. It also repeats its 0.1 run: 0.76 against 0.72 s on the
11,804 returning turns both share. On the 6,762 turns all three arrival-0.5 arms share it is at 0.66 s, against 1.85 s
hint-free and 7.57 s original HiCache, so the hint-free arm keeps 83 % of the gain at 0.5 (78 % at 0.1 on the same turns).
So far only the baseline moves with the rate. Hardware speed matches (decode 49.2 against 49.6 ms per step at 16-23
running; SSD 2.14 against 2.20 ms per page).

**`jit_p50_a05`.** Done 2026-11-05 07:54Z, `20261105_055051_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70`: 10,566
turns (10,419 at `ARRIVAL=0.1`), no client errors, no tracebacks. 1.77 against 1.65 s on the 10,076 returning turns both
share. Hardware speed matches (decode 47.1 against 48.3 ms per step at 16-23 running).

**Part 2 result.** On the 6,762 returning turns all four arrival-0.5 arms share: original HiCache 7.57 s, hint-free 1.85 s,
new p50 1.65 s, new exact 0.66 s; at 0.1 on the same turns 5.54, 1.61, 1.62 and 0.51 s. The order holds, and so do the
shares of the gain: hint-free 83 % (78 % at 0.1), p50 86 % (78 %), so p50 and hint-free still tie within the +-0.5 s
interval for ties. Against its own 0.1 run each hinted or hint-free arm moves 0.01-0.12 s on the same turns, so these
are near-repeats. Only original HiCache moves, +2.03 s: it congests from minute 30 instead of 45, and its main episode
lasts to the end of the 2 h instead of easing by minute 105. `base_a01_rep` (part 3) shows that is within original
HiCache's run-to-run spread. Tables: `compare_a05.out`. Figure: `../compare_c25_a05/arrival_overview.png` (and `.csv`).

**Part 3, added 2026-11-05 about 04:50Z.** The user asked to extend the run to 8-9 h from then with what is not yet known.
The 128-user hint follow-ups are left out because the user closed that topic. All three arms run at 44 users, 70 GB,
`ARRIVAL=0.1`, on `engine_5df71c4bf0_v2`, with hints sent and ignored:

- `base_a01_rep`, original HiCache, 2 h: the baseline's first repeat. At 0.5 it ran 2.0 s slower than at 0.1 on the same
  turns, while the hint-free arm repeated within 0.01 s. Every 44-user gain is measured against the 0.1 run, so its
  run-to-run spread is the first unknown.
- `poll0_a01` (`three_tier_wc_poll0`: original HiCache with paced miss retries off, no pins), 105 min, and `dl_a01`
  (`three_tier_wc_dl`: read pins and deadline order with paced retries on), 105 min. This is option H: which of the two
  levers carries the hint-free arm's 79 % of the 44-user gain. If turning retries off is enough on its own, a one-flag
  change to original HiCache is the fair baseline. `three_tier_wc_poll0` has never run; its launch branch only adds the two
  retry flags. In 105 min even an arm at the baseline's pace reaches about position 7,000, past the first heavy stretch.
- Script: `frozen/oracle_scripts_staging/c25_h/compare_h.py`, written before the arms ran and checked on the campaign 21 and
  23 runs (it gives the known 79 %). Its mechanism table counts what each lever acts on: in original HiCache 979 returning
  turns had a read land and lost it before admission, 91 % of its recomputation, against none in the hint-free arm; 23
  turns hit the reissue cap and 662 looked up storage more than once.

**`base_a01_rep` result.** Done 2026-11-05 09:57Z, `20261105_075438_three_tier_wc_custom_L1nat_H70`: 7,058 turns (8,037 in
the first run), no client errors, no tracebacks. Same speed as the first run: decode 45.5 against 46.7 ms per step at 16-23
running, first-turn prefill 0.093 against 0.089 ms per token, SSD 3.05 against 2.87 ms per page. It does not repeat: on the
6,812 returning turns both runs share it is at 7.02 s against 5.58 s, +1.43 s, all of it from position 3,000 on. Like
`base_a05` it congests from minute 30 (2.36 s in minutes 30-45, against 0.84 s), and its main episode lasts to the end
(33.1 s in the last 15 min, against 3.2 s). `base_a05` is 0.75 s from it on the 6,760 turns they share, about half the gap
between the two 0.1 runs, so the arrival rate's effect on original HiCache is inside its run-to-run spread. Only original
HiCache varies this much: the other arms moved 0.01-0.12 s. In both its runs 91-92 % of the recomputation is reads that
landed and were lost before admission; the repeat lost one on 17.0 % of the shared returning turns against 13.2 %, and
recomputed 3,814 tokens per turn against 2,962. The shares hold: the hint-free arm keeps 78 % of the gain against the first
run, 82 % against the repeat and 83 % at 0.5. Tables: `base_repeat.out` (all nine runs, pairs, per 15 min; script
`frozen/oracle_scripts_staging/c25_a05/base_repeat.py`) and `compare_h_rep.out` (mechanisms). The repeat is dashed in
`../compare_c25_a05/arrival_overview.png`.

**`poll0_a01`.** Done 2026-11-05 11:45Z, `20261105_095747_three_tier_wc_poll0_custom_L1nat_H70`: 7,040 turns in 105 min, no
client errors, no tracebacks, retry poll interval 0 in the server log. Same speed (decode 46.5 ms per step at 16-23 running,
first-turn prefill 0.087 ms per token). Turning paced retries off on its own leaves original HiCache where it was: on the
6,621 returning turns it shares with both original HiCache runs, the hint-free arm and new exact, it is at 5.01 s against
5.36 and 6.22 s, with the hint-free arm at 1.61 s, so it keeps 7-21 % of the gain, inside the baseline's run-to-run spread.
Its main episode is as deep (22.6 s in positions 5,500-7,000, against 24.8 and 26.4 s, and 6.2 s hint-free). It looks
storage up less often (1.08 lookups per returning turn against 1.17-1.21) and never reaches the reissue cap, but it still
loses landed reads before admission on 10.5 % of the shared turns (12.8 and 16.2 % in the two original HiCache runs, none
in the hint-free arm), 81 % of its recomputation. So retries off is not a one-flag fix; `dl_a01` (pins with retries on)
tells whether pins are enough on their own. Table: `compare_h_poll0.out`.

**`dl_a01`.** Done 2026-11-05 13:32Z, `20261105_114546_three_tier_wc_dl_custom_L1nat_H70`: 9,426 turns in 105 min, no client
errors, no tracebacks; deadline order, pin fraction 0.3 and retry poll interval 8 in the server log. Same speed as the
hint-free arm (decode 48.8 against 48.5 ms per step at 16-23 running, first-turn prefill 0.093 against 0.091 ms per token,
SSD 2.30 against 2.25 ms per page).

**Part 3 result.** On the 6,621 returning turns all six arms share (`compare_h.out`), with shares of the gain from each
original HiCache run to just-in-time reads on exact hints (0.51 s):

| Read pins | Paced retries | Arm | Mean TTFT, s | Share of the gain | Landed reads lost, share of turns | Recomputed tokens per turn |
|---|---|---|---|---|---|---|
| no | on | original HiCache, two runs | 5.36, 6.22 | 0 % | 12.8 %, 16.2 % | 2,848, 3,595 |
| no | off | `poll0_a01` | 5.01 | 7-21 % | 10.5 % | 2,680 |
| yes | on | `dl_a01` | 1.71 | 75-79 % | 0 | 727 |
| yes | off | hint-free (campaign 21) | 1.61 | 77-81 % | 0 | 748 |

The pins carry the hint-free arm's gain. With paced retries left on the pinned arm is 0.10 s from the hint-free arm, inside
the 0.01-0.12 s the other arms repeat within, and follows it closely (6.55 against 6.17 s in positions 5,500-7,000; position
7,000 at minute 77.0 against 76.4). Retries off without pins stays at original HiCache's level. Without pins 81-92 % of the
recomputation is reads that landed and were lost before admission; with pins none are. So the fair hint-free rival is
original HiCache with read pins, whatever the retry setting, not a one-flag change to retries. Figure:
`../compare_c25_h/option_h.png` (and `.csv`; script `frozen/oracle_scripts_staging/c25_h/plot_h.py`).

**Timing.** Arm 1 launched 2026-11-04 21:32Z. With part 3 the queue ends about 13:40Z on 2026-11-05 if the host is not
preempted; a preemption loses only the arm in progress. Finished 2026-11-05 13:32Z with no preemption; all eight arms clean.

**GPU unit.** `GPU-2c5a8afb-36d2-5490-e083-9eecacbd838a`, new since the 2026-11-04 15:36Z reboot (`hardware.txt`). Campaign 23 ran
on `GPU-b59d6a34`, campaigns 19-21 on unrecorded units.

**Stop only for errors**: a traceback, client errors, a dead server, or a client that stops sending. Slowdown is a result.
