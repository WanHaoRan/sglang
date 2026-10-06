# Just-in-time, exactly-once SSD reads: ASCII figures for sparse and dense load

These figures show where SGLang HiCache with an SSD tier (the `three_tier_*` arms: L1 GPU, L2 host memory, L3 SSD
written through) loses time at sparse load (gap scale x70) and at dense load (x10, x1), and how reading each
session's context once, just in time, from the agent's gap hint addresses each case. The numbers under each figure
come from the campaigns listed at the end: "measured" means a campaign result, "model" an oracle or model estimate.
The figures are plain ASCII, at most 96 columns wide.

```
*            a turn ends, or a request arrives      //        a long stretch of time, cut
[== Ri ==]   one SSD -> host read of Ri's context   [#]       prefill of the new tokens only
             (one FIFO thread does every read)      [#####]   prefill of the whole context
[......]     Ri waiting in the scheduler queue      xxxx      Ri's host copy is evicted
|<- x ->|    a span of time                         p         one paced retry poll
```

## Figure 1. Sparse load (x70), SGLang today

A session returns after a long gap (every SSD-hit gap in campaign 7 was at least 47 s). By then its context has left
GPU and host memory, and only the SSD copy remains.

**(a) The queue cannot hide the SSD read (minor).**

```
            turn t ends                 turn t+1 arrives
agent i     ----*-----------//----------------*--------------------------------->
                  long gap: its context       | lookup at arrival
                  leaves GPU and host         |
queue                                         * empty: no wait to hide behind
SSD read                                      [== Ri ==]
GPU                                                     [#]-> first token
                                              |<------>|
                                              exposed read
```

Measured: the SSD lookup starts when the request enters the queue (`_add_request_to_queue` → `_prefetch_kvcache`), and
at x70 there is almost no queue to hide it behind. In campaign 12, restores that found the read thread idle reached
their first token in 0.9 s (p50), 0.57-0.58 s of it the read itself. At realistic cadence (campaign 10) the read is
0.62 s of a 1.02 s restore.

**(b) SSD read contention: each returning turn waits for the reads ahead of it.**

```
agent 1     ---//---*R1           R1, R2 and R3 return within seconds,
agent 2     ---//------*R2        each after a long gap
agent 3     ---//---------*R3

SSD read            [==== R1 ====][==== R2 ====][==== R3 ====]
GPU                               [#]R1         [#]R2         [#]R3
                          |<---- R3 waits ----->|<-- read --->|
                          for R1's and R2's reads
```

Measured (campaign 12, x70): returns come in bursts (91-92 % of the SSD loads fall between minutes 20 and 40), and one
FIFO thread serves every SSD read. A restored turn waits behind a median 3 (`timeout`) or 10 (`wait_complete`) other
reads. That wait is 60 % / 88 % of its TTFT and its own read only 11 % / 8 % (TTFT p50 4.1 / 10.4 s, p90 22-23 s).
Campaign 7: 58 % / 80 %.

## Figure 2. Sparse load, ours: read before the turn arrives

```
agent 1     --*-----------------G1----------------------------*R1       context already
agent 2     ----*-----------------G2-----------------------------*R2    in host memory
agent 3     ------*-----------------G3------------------------------*R3

SSD read                  [==== R1 ====][==== R2 ====][==== R3 ====]
                          ^S1           ^S2           ^S3
                          one read each, inside the gaps,
                          earliest predicted arrival first
GPU                                                           [#][#][#]
                                                               R1 R2 R3
```

Each reply carries the client's hint `next_gap_s` (the `kv_hints` action `sglang.next_turn`). The server predicts the
next arrival P = t_end + G + δ and counts back from it the session's own read and every read already queued ahead. So
the burst of Figure 1(b) is read inside the gaps, back to back, earliest predicted arrival first, and each turn finds
its context in host memory. The read is also no earlier than needed: staged pages are evictable host leaves, and
applying campaign 7's host survival curve, a read at the end of the reply would lose ~77 % of them before use.

Model (oracle arrival times, campaign 7): a 10 s lead hides 81-91 % of the read time; a 20 s lead cuts mean
returning-turn TTFT from 1.94 to 1.17 s (`timeout`) and from 1.48 to 0.50 s (`wait_complete`).

## Figure 3. Dense load (x10, x1), SGLang today

A session returns quickly: its context is still in GPU or host memory when the turn arrives, and the turn then waits a
minute or two in the queue.

**(a) Without retries: the matched context is unprotected and evicted before the turn is served.**

```
agent i     --*Ri arrives
              | match: whole context in GPU + host memory -> no SSD read
queue         [.......................................................]admitted
                  waits behind the requests ahead; holds no lock on its pages

host (L2)   [== Ri ==]--------------------------xxxx evicted
                        other requests' prefills fill host memory

SSD         Ri's context is here the whole time (write-through): never read

GPU                                                                    [############]
                                                                       recompute all
```

Measured (retries off: campaigns 8 and 9 on the pre-merge engine, campaign 11's `three_tier_wc_norq`): 93-95 % of all
prefill recomputed context that was in GPU + host memory when its request arrived, and on the SSD the whole time. The
loss grows with the wait (x1): 0 % under 10 s, 29-35 % at 20-30 s, at least 94 % beyond 60 s. A queued request holds
no lock on its pages, and admission re-matches GPU and host memory only. Campaign 11's retries-off arm restored 9 turns
from the SSD, recomputed 57 % of its returns, and matched the arm without an SSD (mean TTFT 116.3 vs 115.4 s).

**(b) With retries: the lost context is re-read at the queue head, and the re-reads queue on the read thread.**

```
agent i     --*Ri arrives
polls          p p p p p p p p  polls for the new tool-output tail (never
                                on the SSD): the 8 attempts are gone in seconds

queue         [.........................................]head: context shrank -> re-query

host (L2)   [== Ri ==]------------xxxx evicted                                    [Ri back]

SSD read                                            [== Rk ==][== Rl ==][== Ri ==]
                                                    the requests just ahead
                                                    re-read at the head too
                                                        |<------ ~4 s at the head ------->|
GPU                                                                                        [#]Ri

or: the polls already spent the budget -> no re-query -> Ri recomputes all
```

Measured (campaign 11, x10, upstream #39283: re-query at admission, paced polls, 8 attempts): 1,141-1,145 of the
~1,150 SSD restores per arm came from the re-query at admission, each holding its turn ~4 s at the queue head (2.1 s
until the data is in host memory, 0.8 s of it the read itself, then ~2 s to admission). SSD reads rose from 0.05 TB
with retries off to 2.3 TB (utilization mean 25 %, p90 58 %, peaks 99 %). The polls spend the shared budget on the new
tail, which can never be on the SSD: 93 % of the remaining recomputes (1,263 of 1,361, `timeout`) had no attempt left
at the head. In campaign 12, 93-98 % of the x10 and x1 SSD reads followed a re-issued lookup, a median 64-92 s after
the request arrived.

## Figure 4. Dense load, ours: one read, just before the turn is served

```
agent i     --*Ri arrives

queue         [.......................................................]admitted at A
                    k = 9             k = 5                 k = 1

host (L2)   [== Ri ==]------xxxx evicted (still on the SSD) [staged]->

SSD read                                          [== Ri ==]
                                                  ^ S: the only read

GPU                                                                    [#]Ri
                                                                       no recompute,
                                                                       no wait at the head
```

Once the turn is queued, its admission time follows from its position: A = now + (k + 0.5) / admission rate. Its
context may be evicted while it waits, which costs nothing, since it is on the SSD. One read is issued when the
remaining wait equals the read time plus the reads queued ahead. It is issued directly, not as a retry, so neither the
polls' budget nor a re-query at the head is involved, and the pages land just before they are used. If they are
evicted again before admission, the read is repeated, at most twice (logged as `ntp_reevict`).

Expected (oracle case 2, not yet measured): it removes the ~4 s at the head and the budget-blocked recomputes. A
just-in-time reload at x1 needs ~0.6-0.7 GiB/s, about a third of the SSD.

## Figure 5. One rule for both regimes

```
             agent turn t ends; its reply carries the hint next_gap_s = G
                                       |
                                       v
             predicted arrival   P = t_end + G + delta   (delta: learned offset)
                                       |
                                       v
   predicted admission   A = P + W   before arrival (W = 0 when the queue is empty)
                         A = now + (k + 0.5) / admission rate   once queued at position k
                                       |
                                       v
           start ONE read at   S = A - (own read + reads queued ahead) - margin
                                       |
                 +---------------------+----------------------+
                 |                                            |
        sparse: S falls in the gap                  dense: S falls in the queue wait
        the read ends before the turn               the read lands just before
        arrives: staged ahead, nothing              admission: read once, no polls,
        exposed, no waiting behind reads            no re-query at the head
                 |                                            |
                 +---------------------+----------------------+
                                       |
                                       v
        at admission: re-match; if the context is still missing, fall back to today's
        lookup, so a late prediction costs no more than today
```

## Summary

| load                       | SGLang today                                                                      | why                                                        | ours                                                                       |
| -------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| sparse (x70), Fig. 1a      | the read is exposed: ~0.6 s per restore (minor)                                   | the read starts when the turn arrives                      | the read ends in the gap, before the turn arrives                          |
| sparse (x70), Fig. 1b      | restored turns wait behind other reads: 60-88 % of their TTFT                     | one FIFO read thread, bursty returns                       | reads start early enough to cover the reads ahead, earliest deadline first |
| dense (x10, x1), Fig. 3a   | context lost while queued, then recomputed: 93-95 % of prefill                    | no lock while queued; admission never asks the SSD         | one read just before admission                                             |
| dense, retries on, Fig. 3b | lost context re-read at the head, ~4 s each; spent budgets still force recomputes | reads issued at admission, one after another on one thread | one read, issued directly, timed to land at admission                      |

## Sources

- `TAKEAWAYS.md`, `CACHEDATTENTION_COMPARISON.md`, `read_backlog.csv` (this directory; campaigns 7-10).
- `../campaign12/README.md` and `../campaign12/x70/read_backlog.csv` (per-restore wait, read time and reads ahead).
- `../compare_20261001_013443/README.md` (campaign 11: retries on and off).
- `../compare_20261001_211629/`, `../compare_20261002_060200/`, `../compare_20261002_200123/` (campaign 12 at x70,
  x10, x1).
- `../../ORACLE_PLAN.md`: §2 (hint, predicted arrival and admission, trigger), §3.1 and §3.5 (read pipeline, retry
  budget), step 7 (refetch while queued, re-read cap).
