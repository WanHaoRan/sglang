#!/usr/bin/env python3
"""Check every number of agent_cache/DESIGN_ITERATIONS.md against the record it was taken from.

Run (CPU, read-only on the repository; prints to stdout):
    nice -n 10 python3 -I -B check_numbers.py > check_numbers.txt

Two checks.
1. Each entry of CLAIMS is (snippet of DESIGN_ITERATIONS.md, [(record, [literal strings])]). The snippet must be in
   the history file, and every literal string must be in its record. A literal string is copied from the record, so a
   mistyped number in the history file shows up as a snippet or a string that is not found.
2. Coverage: after the checked snippets, the code spans and the references that carry no measurement (campaign,
   entry, section and rule numbers, dates, percentile names, gap scales, product names) are taken out of the history
   file, no digit may be left. Whatever is left is printed as "uncovered".
"""
import re
import sys
from pathlib import Path

AC = Path("/home/wanhr/sglang/agent_cache")
DOC = AC / "DESIGN_ITERATIONS.md"

OF = "ORACLE_FINDINGS.md"
OP = "ORACLE_PLAN.md"
OV2 = "ORACLE_V2_PLAN.md"
C23F = "CAMPAIGN23_FINDINGS.md"
DH = "DIST_HINT_DESIGN.md"
RH = "REALISTIC_HINTS.md"
PP = "PAPER_PLAN.md"
OI = "OPEN_ITEMS.md"
DN = "agent_cache_design_discussion_2026-11-23.md"
RR = "REPLAY_REDESIGN.md"
CR = "results/design_review_1123/critique.md"
FF = "results/design_review_1123/facts_fetch.md"
D1 = "results/design_unified_1126/inputs/deck1_pricing_the_room_1106.txt"
D2 = "results/design_unified_1126/inputs/deck2_revised_1123.txt"
DD = "results/design_unified_1126/DESIGN.md"


def c(n):
    return f"results/campaign{n}/DECISIONS.md"


C26_ROW = ("| mean TTFT                    | 0.51, 0.54 s   | 0.66, 0.78 s   | 0.67 s | 0.80 s        | 0.66 s   "
           "| 0.69, 0.80 s    | 0.71, 0.72 s   |")
C29_ROW = "115.7, 47.7, 49.8, 49.4, 48.3 s"
C32_ROW = ("| TTFT mean, p50, p90 | 33.5, 5.0, 99.8 s | 14.6, 6.5, 51.2 s | 15.0, 6.2, 50.1 s | 18.0, 6.5, 63.9 s "
           "| 14.9, 6.1, 50.5 s | 17.2, 5.7, 61.7 s |")
C34_ROWS = ["17.97, 6.47, 63.89 → 11.47, 6.38, 22.41 s", "14.64, 6.51, 51.24 → 10.96, 8.34, 24.01 s",
            "14.86, 6.12, 50.52 → 10.92, 8.24, 22.26 s", "14.95, 6.21, 50.09 → 10.66, 7.87, 22.60 s",
            "33.52, 5.01, 99.81 → 31.88, 5.95, 85.61 s"]
C16_ROW = "| 18.5 s       | 17.3 s          | 13.2 s                     | 11.6 s"
C23_44 = "| Mean TTFT, s | 5.17 | 1.48 | 1.95 | 0.50 | **1.51** |"
C23_128 = "| Mean TTFT, s | 53.12 | 29.19 | 12.06 | 12.14 | 12.46 | **12.56** |"
C15_ROW = "| 22.2 / 10.4 / 40.4 s | 20.8 / 16.8 / 36.1 s   | 53.3 / 10.7 / 155.4 s | 41.1 / 8.7 / 112.9 s   |"

CLAIMS = [
    # ---- what a new reader needs first
    ("a GPU pool of about 668K tokens", [(OF, ["668,160 tokens"]), (c(28), ["GPU pool 668,096 tokens"])]),
    ("a host pool of 160 GB (1.63M tokens)", [(OF, ["160 GB = 1,627,648 tokens"])]),
    ("one SSD of about 2 GB/s", [(OF, ["reads ~2 GB/s when busy"])]),
    ("about 44K tokens at 70 GB and 959K at 160 GB",
     [(PP, ["| 43,968 | 959,488 |", "host-only room, since write-through mirrors the GPU"]),
      (OF, ["a host copy can only be evicted after its GPU copy is gone"])]),
    ("(768 sessions) with the recorded gaps stretched x70 or x10",
     [(OF, ["768 real agent sessions", "recorded gap x 70", "Campaign 17 scales by 10 instead"])]),
    ("at most 128 users in session; their numbers", [(c(28), ["at most 128 in session"])]),
    ("sent in the first 6,600 s", [(c(29), ["requests sent before 6,600 s"]), (c(34), ["requests sent before 6,600 s"])]),
    ("Campaigns 14 to 29 use Triton attention; campaigns 30 and 32 to 35 use FlashAttention 3 (FA3)",
     [(c(32), ["| attention | Triton (`scripts/start_server.sh` pins it) | FA3 (`--attention-backend fa3`"]),
      (c(30), ["arm `hbm_lru`: the radix cache on the GPU only, FA3"]),
      ("results/campaign35/queue.txt", ["--attention-backend fa3"]),
      ("scripts/start_server.sh", ["rewrites to triton anyway"])]),
    ("0.042 to 0.047 ms a token against Triton's 0.11 ms",
     [(OI, ["converts tokens at Triton's 0.11 ms a token", "0.042 to 0.047 ms."])]),
    ("TraceLab saturates it from minute 6 on Triton and never on FA3 (campaigns 29 and 33)",
     [(c(33), ["Triton saturated the box from minute 6 (campaign 29)"]),
      (c(34), ["TraceLab (campaign 33) did not saturate on FA3"])]),
    # ---- 1
    ("reference run campaign 14, 2026-10-08", [("results/campaign14/manifest.txt", ["20261008_204128_three_tier_wc_NAT160"])]),
    ("665 turns recomputed a history the SSD had already delivered, 59 % of all recomputed history",
     [(OF, ["665 returning turns had their SSD read finish", "That was 59 % of all recomputed history"])]),
    ("(mean TTFT of 100 to 180 s for 30 to 50 min)", [(OF, ["storm\" for 30-50 min (mean TTFT 100-180 s"])]),
    ("A request may ask the SSD 8 times (paced retries, 8 scheduler passes apart, and the re-query at admission of upstream #39283)",
     [(OF, ["re-issues it every 8 scheduler passes, up to 8 times", "re-query of upstream sglang #39283"])]),
    ("(95.8 s, campaign 19)", [(OF, ["its mean TTFT was 95.8 s"])]),
    ("on FA3 94 % of its lookups find nothing (campaign 34)", [(c(34), ["94 % of its SSD"])]),
    ("18.5 s at x70 (campaign 14)", [(OF, [C16_ROW])]),
    ("5.17 s at 44 users and 70 GB (campaign 21; a repeat ran 1.43 s slower, campaign 25)",
     [(c(21), ["mean TTFT is 5.17 s for the baseline", "C=44", "HOST_GB=70"]),
      (C23F, ["ran 1.43 s slower than its first run"])]),
    ("115.7 s on FreeInference (campaign 28)", [(c(29), [C29_ROW])]),
    ("33.5 s on FA3 (campaign 32) and 31.88 s with the write fixes (campaign 34)",
     [(c(32), [C32_ROW]), (c(34), [C34_ROWS[4]])]),
    ("(53.3 against 22.2 s over campaign 14's measurement window)",
     [(OF, ["| TTFT mean / p50 / p90                     | 22.2 / 10.4 / 40.4 s   | 53.3 / 10.7 / 155.4 s |"])]),
    # ---- 2
    ("(plan 2026-10-01; campaign 15, 2026-10-09)",
     [(OP, ["Written 2026-10-01"]), ("results/campaign15/manifest.txt", ["20261009_164101_three_tier_wc_oracle_NAT160"])]),
    ("(campaign 15: x70 gaps, 768 sessions, 5 h)", [(OV2, ["recorded gaps x70 capped at 1,200 s, 5 h per arm", "768 fresh"])]),
    ("Under `wait_complete` 20.8 against 22.2 s for the baseline; under `timeout` 41.1 against 53.3 s", [(OF, [C15_ROW])]),
    ("a staged read waited about 30 s behind others",
     [(OF, ["A staged read queued behind a median 9,820 pages (~60 GB, ~30 s)"])]),
    ("queued reads held 39 % of the host pool and the share of returning turns served from host memory fell from 16 % to 10 %",
     [(OF, ["queued reads held 39 % of the host pool", "host hits fell from 16 % to 10 % of"])]),
    ("(509 turns lost one)", [(OF, ["| c15 wc + oracle v1                            | 509 "])]),
    # ---- 3
    ("(plan 2026-10-10; campaigns 16 to 19, 2026-10-11 to 10-18)",
     [(OV2, ["Written 2026-10-10"]), (OF, ["All three arms ran 2026-10-11 19:17Z"]), (c(18), ["(2026-10-18)"]),
      (c(19), ["(2026-10-18)"])]),
    ("On campaign 16's 20,217 shared turns (x70 gaps) the baseline is at 18.5 s and oracle v1 at 17.3 s.",
     [(OF, ["Mean TTFT over the same 20,217", C16_ROW])]),
    ("| 13.2 s |", [(OF, [C16_ROW])]),
    ("within 30 % of the host pool", [(OF, ["within 30 % of the host pool (488K tokens)"])]),
    ("(p99 TTFT from 169 to 46 s)", [(OF, ["p99 TTFT 169 s -> 46 s"])]),
    ("| 11.6 s |", [(OF, [C16_ROW])]),
    ("into a 512-page window, up to 120 s before the predicted admission",
     [(OF, ["while fewer than 512 pages are in flight", "its predicted admission is within 120 s"])]),
    ("the baseline's paced retries are off",
     [(OF, ["| no paced retries   | automatic with the oracle and deadline order"])]),
    ("host memory held by waiting reads fell from 28 % to 2 %", [(OF, ["held by waiting reads from 28 % to 2 %"])]),
    ("| 5.7 s |", [(OF, ["| **5.7 s**"])]),
    ("then sessions over 60 s overdue, then sessions due back after 300 s, then entries with no hint, last sessions due back within 300 s",
     [(OF, ["then sessions more than 60 s overdue, then sessions returning after 300 s (farthest first), then un-hinted "
            "entries (LRU), and last sessions returning within 300 s"])]),
    ("returning turns served from host memory rose from 14 % to 41 % and SSD reads fell 19 %",
     [(OF, ["raised host hits from 14 % to 41 % of returns, cut SSD reads by", "19 % and gave the largest gain"])]),
    ("15.8 to 11.2 s at x10 (campaign 17)", [(OF, ["cut mean TTFT from 15.8 s to 11.2 s"])]),
    ("1.09 to 0.58 s at x70 (campaign 18, within the run-to-run spread)",
     [(OF, ["arm cut mean TTFT from 1.09 s to 0.58 s", "within the run-to-run spread seen so far"])]),
    ("95.8 to 20.3 s at x10 (campaign 19)", [(OF, ["its mean TTFT was 95.8 s", "against 20.3 s in the oracle arm"])]),
    ("(a median 9.8 s before admission at 44 users and 70 GB)",
     [(C23F, ["Its staged reads land a median 9.8 s (p90 48 s) before admission", "**44 users, 70 GB of host memory.**"])]),
    ("(1.52 against 1.51 s, campaign 23's analysis)",
     [(C23F, ["| Hint-free | read pins, paced retries off | 1.51 | -3.82 | 79 % | 709 |",
              "| Oracle | old routine, exact hints, next-use eviction | 1.52 | +0.01 | 0 % | 1,000 |"])]),
    ("(27 % of the arm's SSD reads in campaign 18)", [(OF, ["In campaign 18 that was 27 % of the oracle arm's SSD reads"])]),
    # ---- 4
    ("(2026-10-31; campaigns 20 and 21)", [(DH, ["Written 2026-10-31"])]),
    ("x10 gaps, 128 users (campaign 20): 20.37 s against 20.27 s with exact hints and 95.83 s for the baseline",
     [(c(20), ["| Mean TTFT | 95.83 s | 20.27 s | 20.37 s |", "GAP=10 C=128"])]),
    ("A queue wait of about 26 s", [(c(20), ["| 26.0 / 50.0 / 71.8 s | 26.1 / 52.1 / 66.4 s |"])]),
    ("44 users, 70 GB (campaign 21): 1.95 s against 1.48 s with exact hints and 5.17 s for the baseline, so the p50 keeps 87 % of the gain",
     [(c(21), ["mean TTFT is 5.17 s for the baseline, 1.48 s for the oracle and 1.95 s for the p50, so the p50 keeps "
               "87 % of the oracle's gain"])]),
    ("every gap of 300 s or more was missed and its session marked ended (all 159 in campaign 20)",
     [(c(20), ["expired and ended their sessions in the engine", "all 159 gaps of 300 s or"])]),
    ("(the discussion of 27 and 28 October)", [(RH, ["held on 2026-10-27 and 2026-10-28"])]),
    # ---- 5
    ("(2026-11-01 to 11-05; campaigns 24, 23 and 25)",
     [(c(24), ["Started 2026-11-01 05:02Z"]), (c(25), ["Done 2026-11-05 07:54Z"])]),
    ("minus the read time minus 1 s (on table hints also p50 minus p10 earlier, so it lands near the gap's p10)",
     [(C23F, ["starts at predicted admission minus read time minus 1 s", "hints also minus p50 - p10"]),
      (c(26), ["JIT p50 (lands at p10)"])]),
    ("it took oracle v2 from 1.52 to 0.67 s (44 users, 70 GB, 7,410 shared turns)",
     [(C23F, ["| `jit_70` | just-in-time timing | 0.67 | -0.85 | 18 % | 229 |",
              "| Oracle | old routine, exact hints, next-use eviction | 1.52 |",
              "on the 7,410 returning turns all seven arms share"])]),
    ("otherwise at most 3. Hold", [(C23F, ["it may look up at most 3 times"])]),
    ("minus the read time minus 2 s", [(C23F, ["predicted admission minus read time minus 2 s"])]),
    ("Lookups at arrival that find nothing fell 90 to 93 %", [(C23F, ["fall by 90-93 %"])]),
    ("TTFT gained 0.17 s more at 44 users (noise level)",
     [(C23F, ["| New exact | gate and hold | 0.49 | -0.17 | 4 % | 112 |"]), (PP, ["the gate and hold 4 % (noise level)"])]),
    ("at 128 users the GPU queue sets TTFT (12.46 against 12.06 s) and the gain is 29 % fewer SSD pages",
     [(C23F, [C23_128, "29 % fewer with exact hints", "**128 users, 160 GB of host memory.**"])]),
    ("is about 78 s ahead of oracle v2 (campaign 29",
     [(c(29), ["| `oracle_v2` to `jit_exact` | +3.0 s [+2.6, +3.5] | -77.7 s [-85.1, -70.4] |"]),
      (OI, ["the old oracle loses 78 s to just-in-time at a three-minute queue", "the size is not clean"])]),
    ("(campaign 23: 44 users, 70 GB, x10 gaps, 7,768 shared turns).** Baseline 5.17 s, oracle v2 1.48 s, `jit_exact` 0.50 s, `jit_p50` 1.51 s.",
     [(C23F, [C23_44, "on the 7,768 returning turns all five arms share", "with gaps stretched by 10"])]),
    ("(1.51 against 1.47 s)", [(C23F, ["level with the hint-free arm's 1.47 s"])]),
    ("(30.9K tokens pinned against 6.2K with exact hints, on about 44K of free host room), and 1,743 of 5,001 read launches are refused room",
     [(C23F, ["| Mean pinned tokens | | 39.3K | 32.3K | 6.2K | **30.9K** |",
              "only about 44K tokens of host memory are free beyond the GPU mirror",
              "1,743 of 5,001 read launches are revoked for lack of room"])]),
    ("reaches 0.94 s against 1.35 to 1.60 s (campaign 26)",
     [(c(26), ["takes three-quantile hints from 1.35-1.60 s to 0.94 s"])]),
    # ---- 6
    ("(2026-11-01 to 11-14; campaigns 21, 25 and 26)",
     [(c(26), ["End state, 01:38Z on 2026-11-14"]), (c(21), ["A fourth arm, added 2026-11-01"])]),
    ("At 44 users and 70 GB it gives 79 % of the gain: 1.51 s against 5.33 s for HiCache and 0.49 s for `jit_exact` (7,410 shared turns)",
     [(C23F, ["**44 users, 70 GB of host memory.**", "| Original HiCache | | 5.33 | | | 2,940 |",
              "| Hint-free | read pins, paced retries off | 1.51 | -3.82 | 79 % | 709 |",
              "| New exact | gate and hold | 0.49 | -0.17 | 4 % | 112 |"])]),
    ("At 128 users it keeps about 58 % (29.19 s against 53.12 s for HiCache and 12.06 s for oracle v2, campaign 25)",
     [(c(25), ["29.19 s hint-free, 12.06 s oracle", "mean TTFT is 53.12 s for", "keeps about 58 % of the gain"])]),
    ("of the hint the arm uses only which request continues which session",
     [(c(26), ["it uses the hint for nothing an engine", "(which request continues which session)"])]),
    ("(campaign 26: 44 users, 70 GB, 6,715 shared turns, two runs each).** `nofc_noend` 0.71 and 0.72 s, `nofc` 0.69 and 0.80 s, against HiCache at 5.47 and 6.62 s and `jit_exact` at 0.51 and 0.54 s: 96 % of the way with no hint.",
     [(c(26), [C26_ROW, "| 6,715 shared returning turns | JIT exact      | rules 1, 3     | rule 1 | rules 1, 3, 4 | all four | no gap forecast | no hints       |",
               "With no hint at all the engine gets 96 % of the way",
               "against 5.47 and 6.62 s for original HiCache's two runs", "Setting: 44 users, 70 GB of host memory"])]),
    ("At 128 users the seven arms that protect queued turns span 20.57 to 20.98 s (all turns of each run's first 6,900 s)",
     [(c(26), ["The seven arms that protect queued turns now span 20.57-20.98 s", "| First 6,900 s",
               "| mean TTFT, all turns                    | 20.58 s    | 20.68 s | 20.93 s   | 20.98 s | 20.57 s       | 20.84 s  | 20.57 s         |"])]),
    ("Exact hints beat it by 0.14 to 0.21 s (shown, two runs a side)",
     [(c(26), ["Ahead in all four pairs, by 0.17-0.21 s on", "the 6,715 turns and 0.14-0.21 s on the 11,067",
               "shown, now with two runs on each side"])]),
    ("With 100 GB of host memory the no-hint arm, `jit_p50` and `jit_exact` tie at 0.41 to 0.42 s, because under 1 % of turns read from the SSD: 30 GB more memory bought 0.52 s, more than any hint.",
     [(c(26), ["| mean TTFT at 100 GB                                          | 0.41 s                | 0.41 s                | 0.42 s                |",
               "under 1 % of returning turns read from SSD", "What the 30 GB bought is",
               "(0.52 s against 0.10-0.21 s for exact hints over no hints)"])]),
    # ---- 7
    ("(deck 2026-11-06; built 2026-11-11; campaigns 26, 28, 29, 32 to 34)",
     [(D1, ["6 November 2026"]), (c(26), ["Started 2026-11-11 02:49Z"])]),
    ("λ was fitted at 5e-5", [(c(26), ["every arm starts at the fitted 5e-5"])]),
    ("least likely to return within 30 s", [(c(26), ["eviction by chance of return within 30 s"])]),
    ("(campaign 26: 44 users, 70 GB, 6,715 shared turns).** All four (`ec_full`) 0.66 s; rules 1 and 3 0.66 and 0.78 s; rule 1 alone 0.67 s; rule 3 alone 1.48 s, so the gain is rule 1's.",
     [(c(26), [C26_ROW, "rule 3 alone leaves JIT p50 where it was, 1.48 s", "the gain of the pricing arms is rule 1's"])]),
    ("`jit_p50` 1.35 to 1.60 s, no hints 0.71 and 0.72 s, `jit_exact` 0.51 and 0.54 s",
     [(c(26), [C26_ROW, "| mean TTFT                    | 1.60, 1.35, 1.48 s     | 0.94 s"])]),
    ("soft pins took back 462 of 1,569 staged reads, and 452 of those were read again",
     [(c(26), ["462 of the 1,569 staged reads were bumped", "and 452 read again"])]),
    ("the four hinted arms at 47.7 to 49.8 s against HiCache at 115.7 s", [(c(29), [C29_ROW])]),
    ("`ec_exact` 10.66 s, `jit_exact` 10.92 s, `ec_full` 10.96 s, `jit_p50` 11.47 s, HiCache 31.88 s",
     [(c(34), C34_ROWS)]),
    ("there `ec_full` started 1 read before arrival and its price never moved (review of 23 November)",
     [(FF, ["and staged 1 read", "lambda stayed at 5e-5 in all 1,451 updates"]),
      (CR, ["ec_full staged 1 read before arrival"])]),
    ("(on exact hints, from 2026-11-22)", [(c(28), ["(added 2026-11-22, Results)"])]),
    # ---- 8
    ("(2026-11-19 and 11-20; one fixed 2026-11-22)", [(OI, ["Measured 2026-11-19 and 20", "fixed 2026-11-22"])]),
    ("A queued request may ask the SSD three times", [(OI, ["A queued request may look up the SSD three times"])]),
    ("On TraceLab 175 to 333 requests an arm were refused a read this way and carry about 99 % of the history those arms recomputed (campaign 29)",
     [(OI, ["333 of 2,924; 313 of 2,982; 175 of 3,228", "99.4, 99.3, 98.7 %"])]),
    ("had waited over 60 s in the queue", [(OI, ["has waited over 60 s in the queue"])]),
    ("Run check on TraceLab's first 30 minutes: requests whose lookups ran out 55 to 70 s after arrival fell from 88 to 6, and those of them refused a read from 57 to 1",
     [(OI, ["| allowance used up 55 to 70 s after arrival | 88 | 6 |", "| of them refused a read | 57 | 1 |",
            "30 min of TraceLab's"])]),
    ("The 15.6 s lead of the four rules over `jit_p50` on TraceLab (campaign 29)",
     [(c(29), ["| `jit_p50` to `ec_full` | -1.7 s [-2.1, -1.3] | -15.6 s [-17.0, -14.3] |"]),
      (OI, ["produced by the defects"])]),
    # ---- 9
    ("(found in campaign 32, 2026-11-22; verified in campaign 34, 2026-11-23)",
     [(OI, ["Found on FA3 (campaign 32, 2026-11-22)", "Verified 2026-11-23"])]),
    ("(campaign 32: FreeInference, 128 users, 160 GB)",
     [(c(28), ["at most 128 in session", "host pool 160 GB"]), (c(32), ["Campaign 28's (`settings.env` is a copy"])]),
    ("(74 to 93 % of writes were pages the SSD already held)", [(OI, ["in the loop 74 to 93 % of the writes are such pages"])]),
    ("(515 to 675K tokens)", [(OI, ["under 90 K tokens (515 to 675 K)"])]),
    ("it started, from minute 42 (`jit_p50`; minute 67 by a stricter test) to minute 84 (`ec_full`)",
     [(c(34), ["| minute 42 (67) |", "| minute 84 (84) |", "| minute 80 (80) |", "| minute 79 (79) |",
               "Onset: the first of 3 (in brackets 5) minutes in a row"])]),
    ("The hinted arms ended at 14.6 to 18.0 s, HiCache at 33.5 s.", [(c(32), [C32_ROW])]),
    ("transfers of up to 128 pages", [(c(34), ["queued writes merge into transfers of up to 128 pages"])]),
    ("wait while pending writes hold 50,000 host tokens", [(c(34), ["timed reads wait while pending writes hold 50,000 host tokens"])]),
    ("pending writes stayed under 90K tokens, and the hinted arms were 5.8 to 9.5 s faster (HiCache 3.06 s",
     [(c(34), ["under 90 K tokens", "the fixed engine is 5.8 to 9.5 s faster", "| -3.06 s [-3.67, -2.49], 9,653 requests |",
               "-9.52 s [-11.05, -8.17]", "-5.79 s [-6.68, -4.91]", "-6.18 s [-7.25, -5.23]", "-6.75 s [-7.86, -5.71]"])]),
    ("within about 1 s of each other (10.66 to 11.47 s)", [(c(34), C34_ROWS[:4] + ["1.0 s ahead of `jit_p50`"])]),
    ("five arms sit at 0.63 to 0.66 s (campaign 33)",
     [(c(33), ["| TTFT mean, p50, p90 | 0.66, 0.19, 1.66 s | 0.63, 0.18, 1.58 s | 0.64, 0.18, 1.57 s | 0.64, 0.18, 1.60 s "
               "| 0.63, 0.18, 1.54 s | 0.77, 0.19, 1.69 s |"])]),
    ("From minute 88", [(c(34), ["from minute 87 or 88"])]),
    # ---- 10
    ("(2026-11-23; not built)", [(DN, ["discussion of 2026-11-23"])]),
    ("Revised design of 23 November", [(DN, ["discussion of 2026-11-23"]), (D2, ["23 November 2026"])]),
    ("read pins 3.82 s; reads timed to the queue with queued turns' history kept, 1.02 s more; the write fixes 5.8 to 9.5 s; 30 GB more host memory 0.52 s. Reads before arrival added at most 0.14 to 0.21 s with exact hints and cost 0.6 to 0.9 s on p50 hints; eviction order moved 0.1 s at most.",
     [(D2, ["−3.82 s: 79 % of the gap to exact hints", "−1.02 s more", "−5.8 to −9.5 s on the same users",
            "30 GB more host memory, 70 → 100 GB | −0.52 s", "−0.14 to −0.21 s against no hints",
            "+0.6 to +0.9 s against no early reads", "0.1 s at most"])]),
    ("(1.81 s of cost against 1.10 s for reading at arrival)", [(CR, ["costs 1.81 s against 1.10 s for reading at arrival"])]),
    ("under write-through 0.65M of the 1.63M host tokens only mirror the GPU", [(CR, ["0.65 M of 1.63 M tokens at 160 GB"])]),
    ("about 94 TTFT-seconds (campaigns 32 and 34, minutes 90 to 109)",
     [(CR, ["52.0/0.555 = 94 TTFT-s per GPU-s", "for minutes 90-109"])]),
    ("On 2026-11-26 the author asked", [(DD, ["The author's brief (2026-11-26)"])]),
    # ---- 11
    ("(draft v1 of 2026-11-26; in progress)", [(DD, ["(draft v1, 2026-11-26)", "Nothing here is implemented yet"])]),
    ("Its ten claims (section 7 of the draft)", [(DD, ["## 7. What can be proved", "10. (Many sessions)"])]),
    ("campaign 35 at campaign 34's setting (running), campaign 36 at light load with a small host pool (48 users, 88 GB, 1 h) and campaign 37 at saturation with a tight host pool (128 users, 112 GB, 2 h)",
     [("results/campaign35/README.txt", ["Setting: campaign 34's", "ec_full first, unchanged from campaign 34, as the baseline"]),
      ("results/campaign36/README.txt", ["48 users in session (not 128), an 88 GB host pool (not 160) and 1 h (not 2)", "at light load with a small host pool"]),
      ("results/campaign36/queue.txt", ["(48 users, 88 GB, 1 h)"]),
      ("results/campaign37/README.txt", ["at saturation with a tight host pool", "a 112 GB host pool (not 160)"]),
      ("results/campaign37/queue.txt", ["(128 users, 112 GB, 2 h)"])]),
    # ---- also on record
    ("Rebuilt on 2026-10-17 (`REPLAY_REDESIGN.md`): runs from campaign 18 on do not pair with campaigns 13 to 17.",
     [(RR, ["Replay client redesign (2026-10-17)", "New runs do not pair with campaigns 13-17"])]),
    ("(0 hits in 1,281 lookups)", [(OF, ["(0 hits in 1,281 lookups)"]), (c(16), ["`three_tier_wc_wb` is held"])]),
    # ---- summary table (the same numbers again)
    ("| 18.5 s at x70 (campaign 14); 31.88 s on FA3 (campaign 34) |", [(OF, [C16_ROW]), (c(34), [C34_ROWS[4]])]),
    ("| 20.8 against 22.2 s (campaign 15) |", [(OF, [C15_ROW])]),
    ("| 13.2, 11.6 and 5.7 s against 18.5 s (campaign 16) |", [(OF, [C16_ROW, "| **5.7 s**"])]),
    ("| p50 1.95 s against exact 1.48 s (campaign 21) |", [(c(21), ["1.48 s for the oracle and 1.95 s for the p50"])]),
    ("| 0.50 s exact, 1.51 s p50, baseline 5.17 s (campaign 23) |", [(C23F, [C23_44])]),
    ("| 0.71 s with no hints: 96 % of the way to exact hints (campaign 26) |",
     [(c(26), [C26_ROW, "With no hint at all the engine gets 96 % of the way"])]),
    ("| `ec_full` 0.66 s (campaign 26); 10.96 s on FA3 (campaign 34) |", [(c(26), [C26_ROW]), (c(34), [C34_ROWS[1]])]),
    ("| requests it cost a read: 57 before the fix, 1 after (run check) |",
     [(OI, ["| allowance used up 55 to 70 s after arrival | 88 | 6 |", "| of them refused a read | 57 | 1 |"])]),
    ("| 5.8 to 9.5 s faster on the same users (campaign 34 against 32) |",
     [(c(34), ["the fixed engine is 5.8 to 9.5 s faster"])]),
    ("| 10-08, reference run |", [("results/campaign14/manifest.txt", ["20261008_204128"])]),
    ("| 2 | First oracle | 10-09 |", [("results/campaign15/manifest.txt", ["20261009_164101"])]),
    ("| 3 | Oracle v2 | 10-11 |", [("results/campaign16/manifest.txt", ["20261011_191741_three_tier_wc_dl_NAT160"])]),
    ("| 4 | Gap-table hints | 10-31 |", [(DH, ["Written 2026-10-31"])]),
    ("| 5 | JIT reads, \"owned\" arm | 11-01 |", [(c(23), ["`jit_exact_c44` done, 22:24Z", "Restarted 2026-11-01"])]),
    ("| 11-01 to 11-14 |", [(c(26), ["End state, 01:38Z on 2026-11-14"])]),
    ("| 11-06 deck, 11-11 built |", [(D1, ["6 November 2026"]), (c(26), ["Started 2026-11-11 02:49Z"])]),
    ("| 11-20 found, 11-22 one fixed |", [(OI, ["Written 2026-11-20", "fixed 2026-11-22"])]),
    ("| 11-22 to 11-23 |", [(OI, ["Found on FA3 (campaign 32, 2026-11-22)", "Verified 2026-11-23"])]),
    ("| 10 | T = G + W, five components | 11-23 |", [(DN, ["discussion of 2026-11-23"])]),
    ("| 11-26, in progress |", [(DD, ["(draft v1, 2026-11-26)"])]),
]

# Shares and differences the history file quotes, recomputed from the means it quotes beside them:
# (what, recomputed value, quoted value, tolerance)
ARITH = [
    ("entry 4: p50 keeps 87 % of the gain (campaign 21)", (5.17 - 1.95) / (5.17 - 1.48) * 100, 87, 0.5),
    ("entry 6: pins give 79 % of the gain (campaign 23's analysis)", (5.33 - 1.51) / (5.33 - 0.49) * 100, 79, 0.5),
    ("entry 6: hint-free keeps about 58 % at 128 users (campaign 25)", (53.12 - 29.19) / (53.12 - 12.06) * 100, 58, 0.5),
    ("entry 6: 96 % of the way, HiCache run 1 (campaign 26)", (5.47 - 0.71) / (5.47 - 0.51) * 100, 96, 0.5),
    ("entry 6: 96 % of the way, HiCache run 2 (campaign 26)", (6.62 - 0.71) / (6.62 - 0.51) * 100, 96, 1.0),
    ("entry 10: pins step 3.82 s = 5.33 - 1.51 (campaign 23's analysis)", 5.33 - 1.51, 3.82, 0.005),
    ("entry 5: gate and hold 0.17 s = 0.67 - 0.49, rounded means (campaign 23's analysis)", 0.67 - 0.49, 0.17, 0.011),
    ("host room at 70 GB: 712,128 - 668,160 tokens (PAPER_PLAN)", (712128 - 668160) / 1000, 44, 0.05),
    ("host room at 160 GB: 1,627,648 - 668,160 tokens (PAPER_PLAN)", (1627648 - 668160) / 1000, 959, 0.5),
    ("FA3 fixes: fastest and slowest hinted arm 11.47 - 10.66 s, 'about 1 s' (campaign 34)", 11.47 - 10.66, 1.0, 0.25),
]

# references that carry no measurement; removed before the coverage scan
NEUTRAL = [
    r"`[^`\n]*`",                                   # code spans: arm labels, flags, paths, commits
    r"(?m)^## \d+\.",                               # entry headings
    r"(?m)^\| \d+ \|",                              # row numbers of the summary table
    r"[Cc]ampaigns? \d{1,2}\b(?![.,]\d)(?:(?:, | and | to | on | against )\d{1,2}\b(?![.,]\d))*",   # campaign numbers
    r"entr(?:y|ies) \d+",                           # cross references
    r"sections? \d+(?: to \d+)?",                   # sections of records
    r"[Rr]ules? \d+(?: and \d+)?",                  # rule numbers
    r"[Ff]ix \d+", r"[Dd]efect \d+",
    r"\(\d\)", r"(?m)^  \d\. ",                    # enumerations (1) (2) ... and the numbered rules
    r"Written 2026-11-26", r"Date \(2026\)",        # this file's date; the table's year
    r"\bp(?:10|50|90|99)\b", r"\bx(?:70|10)\b", r"\bv[12]\b",
    r"H200", r"Qwen3-30B-A3B", r"bf16", r"FA3", r"FlashAttention 3", r"λL / \(1 \+ λL\)",
]


def main():
    doc = DOC.read_text()
    cache = {}
    bad = 0
    left = doc
    for snippet, sources in CLAIMS:
        ok = True
        if snippet not in doc:
            ok = False
            print(f"SNIPPET NOT IN THE HISTORY FILE: {snippet[:90]!r}")
        for rel, needles in sources:
            if rel not in cache:
                cache[rel] = (AC / rel).read_text(errors="replace")
            for needle in needles:
                if needle not in cache[rel]:
                    ok = False
                    print(f"NOT FOUND in {rel}: {needle[:110]!r}\n    for: {snippet[:90]!r}")
        bad += not ok
    for snippet in sorted((snippet for snippet, _ in CLAIMS), key=len, reverse=True):
        left = left.replace(snippet, " ")
    for pat in NEUTRAL:
        left = re.sub(pat, " ", left)
    uncovered = []
    for ln, line in enumerate(left.splitlines(), 1):
        for m in re.finditer(r"[^\s|]*\d[^\s|]*", line):
            uncovered.append((ln, m.group(0), line.strip()[:120]))
    print(f"claims: {len(CLAIMS)}, failed: {bad}")
    off = 0
    for what, got, quoted, tol in ARITH:
        ok = abs(got - quoted) <= tol
        off += not ok
        print(f"  {'ok ' if ok else 'OFF'} {what}: recomputed {got:.3f}, quoted {quoted}")
    print(f"recomputed shares and differences: {len(ARITH)}, off: {off}")
    bad += off
    print(f"records opened by the check: {len(cache)}")
    print(f"digit tokens left uncovered: {len(uncovered)}")
    for ln, tok, ctx in uncovered:
        print(f"  uncovered {tok!r} (after removal, line {ln}): {ctx}")
    print(f"lines in {DOC.name}: {len(doc.splitlines())}")
    return 1 if (bad or uncovered) else 0


if __name__ == "__main__":
    sys.exit(main())
