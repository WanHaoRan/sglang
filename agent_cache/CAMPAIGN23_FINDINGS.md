# Campaign 23 findings

Campaign 23 tested the request of 2026-10-31: stop the blind match and fetch at arrival and let the scheduler decide when
to fetch, ideally exactly once per turn. Its last arm finished on 2026-11-02, and no arm had client errors. This note
gathers the findings so the next steps can be chosen from them. The evidence arm by arm is in
`results/campaign23/DECISIONS.md`. Every arm is one run, with no repeats.

## In short

- **With realistic hints, when memory is the limit, the new routine cuts TTFT by 23 %.** At 44 users and 70 GB, mean TTFT
  is 1.51 s against the old routine's 1.95 s on the same turns (5.17 s with no hints). That only draws level with the old
  routine on exact hints, 1.48 s.
- **Exact next-turn times reach 0.50 s, so 1.01 s is still on the table.** Under the old routine that gap was 0.47 s, so
  "p50 is enough" was partly a fact about the old routine.
- **When the GPU queue is the limit, the gain is I/O only.** At 128 users and 160 GB every hinted arm sits at
  12.06-12.56 s on the same turns (53.12 s with no hints). The new routine reads 15 % fewer SSD pages with p50 hints and
  29 % fewer with exact hints.
- **The gate and the hold work as asked.** Lookups at arrival that find nothing fall by 90-93 %. No held read went out
  after its turn's admission or was left behind, in any arm.
- **Exactly one fetch per turn holds in one case of four:** 128 users with exact hints. Short host room breaks it at 44
  users, and forecast error breaks it at 128 users.
- **The main problem left looks like reads sent before their turn arrives.** Under p50 hints they hold most of the pinned
  memory at both settings. At 128 users a simple rule should remove them. At 44 users most are needed, and what to do
  about them is open.

The choices and a recommendation are under "Options for next steps".

## What was tested

The old routine reads a turn's history as soon as it has been evicted and admission is within 120 s. At arrival it also
looks up the SSD for whatever part of the prompt is not in memory, even when that part is only new text. The new routine
changes three things:

- **Just-in-time reads.** A read sent before arrival starts at predicted admission minus read time minus 1 s, and with p50
  hints also minus p50 - p10.
- **The gate.** At arrival, a planned turn whose history is in memory, to within 256 tokens, gets no SSD lookup. Otherwise
  it may look up at most 3 times. A turn with no plan looks up as before.
- **The hold.** A read of a queued turn waits until predicted admission minus read time minus 2 s.

Each setting ran with two hints. **Exact** is the true time of the next request, which a real system does not have.
**p50** is the median gap for the tool call the reply names, from a table the agent system keeps online, which a real
system can have.

## Results

**Baseline** is HiCache with no hints, **oracle** the old routine on exact hints, **old p50** the old routine on p50 hints,
and **new exact** and **new p50** the new routine. All use the same trace, seed and arrival rate, with gaps stretched by 10.

**44 users, 70 GB of host memory.** TTFT and recomputation are on the 7,768 returning turns all five arms share, lookups
and reads on the 8,008 turns they share, and the last two rows cover each arm's 2 h.

| | Baseline | Oracle | Old p50 | New exact | New p50 |
|---|---|---|---|---|---|
| Mean TTFT, s | 5.17 | 1.48 | 1.95 | 0.50 | **1.51** |
| Recomputed tokens per turn | 2,870 | 977 | 1,329 | 114 | **940** |
| Turns sent in 2 h | 8,037 | 10,319 | 10,021 | 12,201 | **10,419** |
| Arrival lookups that found nothing | 3,461 | 3,051 | 3,092 | 236 | **294** |
| SSD pages | 592,964 | 293,759 | 375,501 | 215,881 | **430,992** |
| Reads per turn that reads | 1.057 | 1.331 | 1.309 | 1.145 | **1.276** |
| Mean pinned tokens | | 39.3K | 32.3K | 6.2K | **30.9K** |
| SSD busy, minutes 45-120 | 31-53 % | 17-37 % | 15-37 % | 22-32 % | **21-52 %** |

![44 users, all arms over time](results/compare_c23_c44/arms_overview.png)

**128 users, 160 GB of host memory.** Each arm's first 2 h, because the older arms ran 5 h. TTFT and recomputation are on
the 8,175 returning turns all five arms share. Lookups and reads are on the 16,137 turns the four hinted arms share, since
the baseline reached only about 8,500 turns.

| | Baseline | Oracle | Old p50 | New exact | New p50 |
|---|---|---|---|---|---|
| Mean TTFT, s | 53.12 | 12.06 | 12.14 | 12.46 | **12.56** |
| Recomputed tokens per turn | 3,784 | 35 | 34 | 33 | **34** |
| Turns sent in 2 h | 8,511 | 16,228 | 16,210 | 16,101 | **16,082** |
| Arrival lookups that found nothing | | 6,364 | 6,493 | 472 | **512** |
| SSD pages | | 1,518,868 | 1,523,453 | 1,071,057 | **1,290,748** |
| Reads per turn that reads | | 1.008 | 1.033 | 1.000 | **1.031** |
| Mean pinned tokens | | 148.8K | 127.0K | 32.9K | **66.4K** |
| SSD busy, minutes 45-120 | 39-48 % | 97-98 % | 97-98 % | 47-78 % | **60-92 %** |

![128 users, all arms over time](results/compare_c23_c128/arms_overview.png)

## What explains the results

**The forecast costs where host room is short.** The forecast error is the same at both settings: actual minus predicted
arrival is p10 about -5.5 s, median -0.3 s, p90 +4.9 s. Its cost is not. Against exact hints it costs 1.01 s at 44 users,
and 0.07 s at 128 users on 15,503 shared turns (0.10 s under the old routine).

- At 44 users only about 44K tokens of host memory are free beyond the GPU mirror, and a read that lands early stays
  pinned until its turn is admitted. With p50 hints 1,743 of 5,001 read launches are revoked for lack of room, against 381
  of 2,088 with exact hints, and 88 % of recomputed tokens sit on turns whose read found its data but was cut or revoked.
- Across the hinted arms recomputation follows pins: 39.3K pinned and 977 tokens per turn (oracle), 32.3K and 1,329 (old
  p50), 30.9K and 940 (new p50), 6.2K and 114 (new exact).
- At 128 users the room is 959K tokens and pins peak at 401K. No read is revoked while the client is sending, and
  recomputation stays at 34 tokens per turn.

Two limits on this reading: the settings differ in users as well as memory, and eviction also runs on the forecast, so at
44 users pins and eviction are not separated.

**Reads sent before the turn arrives are what a forecast damages.** With exact hints such a read lands a median 1.7 s before
admission (p90 2.5 s, 44 users). With p50 hints it is 4.7 s (p90 24.6 s). Under p50 hints, turns whose read came due before
arrival hold 93 % of pinned token-seconds at 44 users and 66 % at 128 users, for different reasons:

- At 44 users there is little queue (median wait estimate 1.0 s, read time 2.5 s), so most of these reads have to start
  before the turn arrives. 87 % of pinned token-seconds come from reads due ahead of the forecast arrival. Of the long
  pins, about half are true gaps above the key's p90 and half the deliberate early start by p50 - p10.
- At 128 users the queue wait is about 25 s, so almost no read needs to start early: 17 do with exact hints. With p50 hints
  515 do, and 463 of those came due with the forecast arrival already missed, by a median 12.5 s and with the turn still
  away. The routine goes on assuming the turn arrived on time and is queueing. These turns hold 56 % of pinned
  token-seconds, and 508 of the 515 had a wait estimate above read time plus 1 s, so they could have waited for the turn.

**One fetch per turn needs both room and good timing.**

| Reads per turn that reads | Exact | p50 |
|---|---|---|
| 44 users, 70 GB | 1.145 | 1.276 |
| 128 users, 160 GB | 1.000 | 1.031 |

- At 44 users the extra reads are mostly remainders: eviction cannot free the whole span, so the first read is cut short
  and the next fetches the rest (all 224 such turns with exact hints, 646 of 672 with p50). Pages are rarely read twice:
  268 of 434,495 with exact hints, about 14,700 of 610,945 with p50.
- At 128 users reads are almost never cut (3 of 4,298) and every double read has one cause. The read lands before its turn
  arrives, the plan times out 60 s after the forecast arrival and releases the pin, the span has left host memory by the
  time the turn comes, and the turn reads it again. That is 129 turns and 37,070 pages, 2.9 %.

**At 128 users the GPU queue decides TTFT.** Of about 21 s per returning turn, queue wait is 20.5-20.9 s and everything after
admission is 0.42 s in every hinted arm. From minute 45 the GPU pool is 0.98 full, with about 30 requests running and 59-73
queued. Hints already removed what memory was costing, so no read routine can shorten what is left. What the new routine
adds is SSD headroom, and what that buys is unmeasured because the GPU is the limit.

**At 44 users most of the exact-hint gain is read timing.** On the 7,412 turns shared with `jit_70`, which had just-in-time
reads alone for 75 min: oracle 1.52 s, `jit_70` 0.67 s, new exact 0.49 s. The gate and the hold add 0.18 s of the 1.03 s,
not separated from noise. Their clear effect is on I/O and lookups.

**Both new arms at 128 users trail their old-routine rivals by about 0.4 s.** That is +0.40 s with exact hints and +0.38 s
with p50 hints on about 15,500 turns, with 0.8 % fewer tokens generated. I first read this as GPU speed. The p50 arm weakens
that: its early decode is only 0.3 % slower than its rival's, yet it lags as much. Routine and GPU unit arrive together, so
this needs a repeat on one unit.

**The hold is sound, and late reads follow the SSD's ceiling.** Holds per arm were 173, 470, 3,203 and 2,826, dispatched a
median 3.2-4.8 s before admission. Even so, at 128 users a read can land after its turn could have gone.

| Reading turns queued 5 s or more | Oracle | Old p50 | New exact | New p50 |
|---|---|---|---|---|
| Passed by a later arrival | 4.5 % | 25.9 % | 16.7 % | 32.1 % |
| Delay per reading turn, s | 0.18 | 1.73 | 0.61 | 1.62 |

Turns without a read are never passed. The turn's own forecast is not the cause: the share is 31-34 % whether the forecast
was right or off by up to 30 s. It follows the SSD. With reads under 1.6 GB/s around the admission the two new arms match
(4-6 %, 9-10 %, 17 %). At 1.8 GB/s or more new p50 is at 48 % and new exact at 20 %, and 47 % of new p50's reads fall there
against 8 %. The ceiling is about 1.85 GB/s, and new p50's 20 % more pages push it there.

**A rough capacity figure.** Read from swings in one run, the load carried at a mean TTFT of 5 s rises over the baseline by
11 % with exact hints and 6 % with p50 hints (1,055K, 1,176K and 1,117K tokens). The 44-user setting sits on a steep knee, so
mean TTFT there can overstate the gain. At 128 users every hinted arm is far past the knee.

## How far this can be trusted

- **One run per arm.** The oracle's spread at 44 users was once measured at 0.05-0.14 s on runs since deleted, and 128 users
  has no estimate. So the gaps of 0.07-0.4 s at 128 users and the 0.18 s from the gate and hold are not known to be real.
- **Changes arrive together.** The older arms ran on an older engine, `engine_5df71c4bf0_v2`. At 128 users nothing separates
  read timing, the gate and the hold.
- **The settings differ in users and memory at once,** so "room" against "users" is a reading, not a result.
- **GPU units.** The preemptible host changed unit twice on 2026-11-01. All four new arms ran on one unit, `GPU-b59d6a34`, and
  the older arms' units are unrecorded. Speeds match at 44 users. At 128 users early decode differs by up to 2 %.
- **Workload.** A batch-agent trace with gaps stretched by 10, in a closed loop. Real use has long human waits.

## Options for next steps

Each 2 h arm takes about 2 h 5 min, and a preemption loses at most the arm in progress.

| | Step | What it answers | Work | GPU time |
|---|---|---|---|---|
| A | Send a read before arrival only if the queue wait could not cover it; rerun p50 at 128 users, and at 44 users to check it does no harm | Does p50 reach one fetch at 128 users, and how much of the I/O saving comes back? It would touch 99 % of such reads at 128 users and 14 % at 44 | Small engine change | 4 h |
| B | At 44 users, exact gaps for eviction and p50 for reads | Is the 1.01 s read timing and pins, or eviction on the forecast? Decides whether C is worth running | Small engine and client change | 2 h |
| C | Lighter pins at 44 users: no early start by p50 - p10, a time limit on pins, or holding a read until room appears | How much of the 1.01 s comes back | Small each, none built | 2 h each |
| D | Repeats on one recorded GPU unit: new exact, new p50 and oracle at 44 users; new exact and oracle at 128 | Which gaps are real, including the 0.4 s at 128 users | None | 10 h |
| E | User-count sweep, best routine against the baseline | Sessions per GPU under a latency target, the number an operator buys | None | 20 h for 5 counts |
| F | Campaign 22, the step-by-step ladder at 128 users | Which part of the gain needs a hint at all | Check its launcher against today's `start_server.sh` | 6 h |
| G | Analyse the hint-free arm at 44 users, cut at 117 of 120 min | The same question at 44 users | Analysis only | None |

**My recommendation.** Do G now, since it costs no GPU time. Then run A and B as one campaign of about 6 h: they are small,
A answers the one-fetch goal at 128 users, and B says what the 1.01 s is. Run C only if B points at pins. Hold D until the
routine settles, so repeats are not spent twice. E and F are for the paper and come after D.

The engine work behind all of this is uncommitted in `python/`. Committing it before A or B would give those changes a clean
diff. The paper's other open items (human-paced workload, prior-work recheck, venue and deadline) are in `PAPER_PLAN.md`.

## Small issues logged

- A read still in flight at shutdown hit the freed host pool (`AttributeError` at `mha.py:696`), once, after the client ended.
  Not fixed.
- With p50 hints at 128 users 19 % more turns read than with exact hints. Turns that came 5 s or more before the forecast read
  about twice as often (280 to 549), and those that came 1 s or more after read less (1,257 to 1,014), which looks like
  eviction acting on the wrong time. Turns forecast within 1 s also read more (1,532 to 2,097), with no direct evidence why.
- New p50 reads 15 % more pages than old p50 at 44 users, because its reads are larger (244 pages per reading turn against 188).
- On held reads at 128 users the queue estimate runs later than real admission by a mean 1.57 s with p50 hints and 0.46 s with
  exact hints. The read estimate does not differ, and turns passed ahead explain only 0.09 s of that gap. Not explained.
- Under the old routine p50 is passed five to six times as often as the oracle at the same SSD rate, which the ceiling cannot
  explain.
- The lookup cap refused 25 lookups with exact hints and 215 with p50 at 44 users, and none at 128 users.
- The spike near position 8,750 at 44 users appears in every arm and looks like the GPU short of room for what is running.
- Below the knee at 44 users new p50 is slower than the baseline (0.42 s against 0.37 s, 0.94 s against 0.72 s). New exact is not.
- `read_start_timing.py`'s "case 2" row misreads the hold, because it reports the planned trigger and not the dispatch.
- `results/compare_c24_jit_vs_c21/` still draws attempts since deleted, and `c24_probe/jit_table.py`, `plot_jit.py`,
  `c21_x10_c44/plot_overview.py` and `plot_standard.py` no longer run.
- Two tests in `test_pp_prefetch_ticket.py` fail the same way on this engine and on `engine_table_20261101`.

## Where the data is

| Arm | 44 users, 70 GB | 128 users, 160 GB |
|---|---|---|
| Baseline | `20261101_082229_three_tier_wc_custom_L1nat_H70` | `20261018_103544_three_tier_wc_NAT160` |
| Oracle | `20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70` | `20261018_154254_three_tier_wc_oracle_v2_nu_NAT160` |
| Old p50 | `20261101_062014_three_tier_wc_oracle_v2_nu_custom_L1nat_H70` | `20261031_204526_three_tier_wc_oracle_v2_nu_NAT160` |
| New exact | `20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70` | `20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160` |
| New p50 | `20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70` | `20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160` |

Run folders are under `results/`; the hint-free arm is `20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70`. Tables are in `results/campaign23/`, figures in `results/compare_c23_c44/` and
`results/compare_c23_c128/`, scripts in `frozen/oracle_scripts_staging/c23_engine/`, and the engine in
`frozen/engine_owned_20261101/python`. `frozen/` is excluded from git. Mean pinned tokens here use one definition for every
arm, each arm's first 2 h, so new exact at 44 users reads 6.2K where `DECISIONS.md` also has 6.7K, which counts pins after the
client stopped.
