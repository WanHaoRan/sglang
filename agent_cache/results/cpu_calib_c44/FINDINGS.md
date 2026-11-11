# CPU calibration of the room model (44 users, 70 GB)

2026-11-07. Step 1 of "Next" in `PAPER_PLAN.md` (design point): add host room and revokes to the offline model,
calibrate it on `jit_exact_c44` and `jit_p50_c44`, choose λ, test soft pins, and settle the hint's form. No GPU used.
Code: `scripts/room_model.py` (`measure`, `calibrate`, `policies`, `figure`); outputs in this folder.

## The model

It replays one run's reads. Held fixed from the run:

- each turn's arrival and the GPU's queue wait;
- the pages each turn had to read from SSD, so eviction is the run's;
- the time the history went missing.

What changes is when each read starts, and whether a landed read can be taken back:

- **Room.** A read holds its pages in the host room from launch until its turn is admitted. A launch that finds no
  room is revoked, as the engine's host allocation is.
- **SSD.** One FIFO read thread serves reads at the run's measured rate: 10 ms plus the median ms per page of each
  10-minute bin.
- **Read scheduler.** Timed reads dispatch below a 512-page window; reads for arrived turns bypass it.
- **After a revoke.** A waiting turn retries 1 s later, up to three lookups, then recomputes. A turn whose staged read
  was revoked reads at arrival, under the hold.
- **The run's own extra reads.** Re-reads after a landed read (history lost while the turn waited) are replayed at
  their logged times.

Choices the first fit forced:

- **Queue wait.** A reading turn's own measured wait is used, or the local median of such waits where a landing read
  gated its admission. Turns without reads are no guide: they cluster where the GPU is not congested, and their
  median wait is about 0 s.
- **Earliest start.** A staged read never starts before the run's history went missing. With p50 hints the run staged
  a median 1.2 s after its routine's trigger (p90 5.5 s), because the engine stages only what is missing when it
  probes every 0.5 s.
- **Turns that cannot be staged.** Turns whose first read in the run came at arrival (280 with exact hints, 652 with
  p50 hints) read at arrival under every rule.
- **Hints.** The p10/p50/p90 hints are rebuilt per turn from `client.jsonl`. The histogram and raw-sample forms come
  from the client's online table, rebuilt per agent system (source, orig_model). The rebuilt table reproduces all
  10,089 hints the p50 run sent: same n and same p50.

## Calibration: one room fits both runs

The only free parameter is the room. At **90K tokens** the model reproduces each run's own routine within about
±25 % on every count (`calibration.csv`):

| | Exact: measured | Exact: model | p50: measured | p50: model |
|---|---|---|---|---|
| Staged reads | 1,187 | 1,149 | 1,932 | 1,875 |
| Staged reads revoked | 139 | 124 | 614 | 549 |
| Reads for arrived turns | 854 | 899 | 2,770 | 2,903 |
| Of those, revoked | 239 | 299 | 1,053 | 1,338 |
| Mean pinned tokens | 7,273 | 8,588 | 33,701 | 34,620 |
| Pinned before arrival | 4,163 | 4,765 | 24,367 | 24,570 |
| Landing lead, p50 / p90 | 1.77 / 2.99 s | 1.90 / 3.34 s | 5.28 / 25.55 s | 6.42 / 26.51 s |
| Reads landing after arrival | 787 | 701 | 1,936 | 1,668 |

The effective room is about twice the static 44K tokens, because the GPU is not always full and cached host data can
be evicted for a read. It also moves: a fixed threshold on pinned plus in-flight tokens classifies only 85 % (exact)
and 70 % (p50) of launches correctly. So the policy results below are also given at 70K and 110K.

## Read-start rules on the p50 run's reads

The cost below is stall seconds plus recomputed tokens at 9K tokens/s, over the 2,576 turns that read history.
Room 90K (`policies.csv`, `policy_tradeoff.png`):

| Rule | Staged | Revoked: staged / arrived | Mean pinned | Stall | Recomputed | Cost |
|---|---|---|---|---|---|---|
| JIT p50, as run | 1,875 | 549 / 1,338 | 34.6K | 275 s | 13.4M | 1,766 s |
| Read at arrival | 0 | 0 / 558 | 7.0K | 1,069 s | 3.7M | 1,481 s |
| Fixed quantile, q = 0.5 | 1,557 | 347 / 1,075 | 28.0K | 402 s | 9.2M | 1,424 s |
| Fixed quantile, q = 0.9 (best) | 644 | 70 / 715 | 13.4K | 780 s | 5.0M | 1,336 s |
| Expected cost, p10/p50/p90, λ = 5e-5 | 842 | 79 / 671 | 13.5K | 695 s | 5.2M | 1,269 s |
| Expected cost, histogram, λ = 2e-5 | 826 | 60 / 680 | 14.5K | 701 s | 5.4M | 1,296 s |
| Expected cost, all samples, λ = 1e-5 | 1,113 | 157 / 820 | 19.1K | 556 s | 6.8M | 1,309 s |
| Expected cost (samples, 1e-5) + soft pins | 1,087 | 118 / 586 | 15.0K | 795 s | 4.0M | 1,241 s |
| Exact arrival times | 1,505 | 142 / 592 | 13.7K | 382 s | 3.9M | 813 s |

- **The model reproduces the headline without being fitted to it.** The run's p50 rule (land near the p10 for every
  size, expire at p50 + max(60 s, p50)) costs more than reading at arrival: 1,766 against 1,481 s. The GPU runs gave
  1.51 against 1.47 s.
- **Choosing λ.** The best λ for the p10/p50/p90 form is 5e-5 at 90K. That matches the 4-6e-5 implied by the p50's
  1.01 s cost, and lands a 64K-token read near p77. The best λ falls as the room grows:
  - 1e-4 at 70K;
  - 5e-5 at 90K;
  - 2e-5 at 110K (3e-6 for the histogram and sample forms).

  So a fixed λ is right for one room only, which is rule 4's case for learning it from revokes and idle room.
- **Against fixed quantiles.** The expected-cost rule beats the best fixed quantile by 5 % at 90K (1,269 against
  1,336 s), 8 % at 70K and 1 % at 110K. The best quantile itself moves with the room: 0.9 at 70K and 90K, 0.5 at
  110K.
- **The hint's form.** No form wins:

  | Best cost | p10/p50/p90 | Histogram | All samples |
  |---|---|---|---|
  | Room 70K | 1,598 s | 1,625 s | 1,611 s |
  | Room 90K | 1,269 s | 1,296 s | 1,309 s |
  | Room 110K | 1,109 s | 1,097 s | 1,090 s |

  All within 3 %. For timing reads, three numbers carry what a 24-bin histogram or every sample does. Whether a
  histogram pays for eviction (rule 3) is not tested here.
- **Soft pins.** They mostly stand in for good timing. They cut the run's p50 rule by 24 % (1,766 to 1,345 s), but
  the best expected-cost rule by only 2-6 % (2 % at 90K, 3 % at 70K, 6 % at 110K).
- **Timing alone stops well short of exact arrival times.** Exact times cost 813 s on the same reads. The best
  realistic rule (1,241 s) covers 36 % of the way from reading at arrival to exact times.

## How much this depends on the price of recomputation

The cost above charges a recomputed token its own prefill time only. Campaign 23 found that a turn's own extra
recomputation is 11-14 % of the gap between p50 and exact hints, against 71 % for queue wait. That puts the full
price of a recomputed token at roughly 6-8 times its own prefill. At 90K, weighting recomputation:

| Weight on recomputation | ×1 | ×3 | ×10 |
|---|---|---|---|
| JIT p50, as run | 1,766 s | 4,749 s | 15,188 s |
| Read at arrival | 1,481 s | 2,304 s | 5,187 s |
| Best fixed quantile | 1,336 s | 2,448 s | 6,338 s |
| Best expected cost, hard pins | 1,269 s | 2,216 s | 5,321 s |
| Best with soft pins (at the ×1 λ) | 1,241 s | 2,133 s | 5,167 s |
| Exact arrival times | 813 s | 1,675 s | 4,692 s |

At a weight of 3 the best rule still beats reading at arrival by 7 %. At 10 it ties, and every fixed quantile loses
to reading at arrival. Each staged read takes room that an arrived turn's read may then be refused, and the
recomputation that follows is what the GPU queue pays for. At 70K and a weight of 10, reading at arrival
(8,659 s) even beats exact arrival times (9,294 s).

## What it means for the plan

1. **Build rule 1 with soft pins**, and start λ near 5e-5 for p10/p50/p90. Learn λ from revokes and idle room
   (rule 4): the best λ moves 5-30-fold between 70K and 110K, and the room moves within a run.
2. **Keep p10/p50/p90 as the hint for timing reads.** The histogram's case now rests on eviction (rule 3), the
   explicit end, and merging entries.
3. **Better timing cannot close the gap to exact hints at this setting.** With the run's evictions held fixed:
   - the best rules cut the p50 rule's direct cost by 28-30 %, and its recomputation from 13.4M to 4.0-5.2M
     tokens;
   - that would likely bring the p50 arm to about the hint-free arm, perhaps slightly better;
   - it would not bring it to exact hints.

   The rest of the 1.01 s is in what eviction does with wrong times: 2.7 times as many turns read from SSD (1,925
   against 723). That makes option B of `CAMPAIGN23_FINDINGS.md` (exact gaps for eviction, p50 for reads, one 2 h
   GPU arm) the deciding experiment, and rule 3 at least as important as rule 1.

## Limits

- **Eviction is the run's.** A rule that pins less would leave more room for cached histories and so cause fewer
  reads; the model credits none of that, so it understates the gain from pinning less.
- **The room is one number.** The real room moves with the GPU's mirror, and the 70K and 110K rows bound the effect.
- **Queue waits and re-reads are taken from the run,** not produced by the rule being tested.
- **TTFT is out of reach.** The model gives direct costs and counts. The queue effect enters only through the
  recomputation weight above.

## Files

- `calibration.csv`: each run's own routine over rooms 30K-160K, beside the measured row.
- `policies.csv`: every rule at rooms 70K, 90K and 110K.
- `policy_tradeoff.png` / `.pdf`: pinned memory against cost at 90K, and each key rule's cost split into stall and
  recomputation.
- `run_exact.json.gz`, `run_p50.json.gz`: the parsed runs (cache of `load_run`).

Rerun: `cd agent_cache/scripts && python3 room_model.py measure|calibrate|policies|figure`. Policies take about 30 s.
