#!/usr/bin/env python3
"""Write final_answer.json: a durable copy of the structured answer of this check
(values from final2_values.txt and final2_add.txt; the other fields are written here).

Run as:  nice -n 19 python3 -I -B final_answer.py
"""
import json
import os

OUT = "/home/wanhr/sglang/agent_cache/results/campaign26/checks/verify/server"
ORDER = ["0", "1", "2", "3", "4", "5a", "5b", "5c", "5x"]
SETS = ["LEAD5", "SEVEN", "ALL11", "CUT7", "ALL13", "SIX", "own"]

values = []
for name in ("final2_values.txt", "final2_add.txt"):
    with open(os.path.join(OUT, name)) as f:
        for line in f:
            if line.strip():
                values.append(json.loads(line))
values = [v for v in values if not (v["metric"] == "share_of_s2h_io_lines_starting_over_0.05s_before_arrival"
                                    and v["arm"] in ("nofc", "nofc_noend"))]
values = [v for v in values if v["item"] != "5x"]   # entry_time - client send: given as ranges in the sanity list
seen = set()
for v in values:
    k = (v["item"], v["arm"], v["arm_set"], v["metric"])
    assert k not in seen, k
    seen.add(k)
idx = {id(v): i for i, v in enumerate(values)}
values.sort(key=lambda v: (ORDER.index(v["item"]), SETS.index(v["arm_set"]), idx[id(v)]))

EDGE = ("has a stall of exactly 0.050 s at the logs' 1 ms resolution. The file subtracts epoch seconds held as floats "
        "(steps of 2.4e-7 s near 1.79e9) and lands just under 0.05, so the turn counts as landed there. "
        "With the same float arithmetic my number equals the file's.")
mismatches = [
    {"file": "stall_ec.out", "what": "arm set and shared returning turns the file is computed on",
     "mine": "LEAD5 (jit_exact ec_r13 ec_r1 nofc nofc_noend), 11198 turns",
     "theirs": "six arms (jit_exact ec_r13 ec_r13_rep ec_r1 nofc nofc_noend), 11145 turns",
     "likely_cause": "The file was regenerated at 01:35:28Z on 2026-11-12, during this check, with the new arm ec_r13_rep. "
                     "All its per-arm numbers are now on 11145 turns, 53 fewer than LEAD5: reading turns are lower by 2 to 6 per arm and "
                     "per-turn seconds differ by up to 0.0021. On the file's own six-arm set all 242 of its numbers equal mine at the "
                     "printed precision (computed after opening). The five-arm version it replaced equalled my frozen LEAD5 numbers on "
                     "all 202 of its numbers (compare.json, 21:07Z 2026-11-11)."},
    {"file": "stall_ec.out", "what": "ec_r1, six-arm set: reading turns landed by arrival, exact microsecond arithmetic",
     "mine": "268", "theirs": "269", "likely_cause": "Turn c46-t1 " + EDGE},
    {"file": "stall_ec.out", "what": "ec_r1, six-arm set: turns where the read was the last wait, exact microsecond arithmetic",
     "mine": "893", "theirs": "892", "likely_cause": "Same turn c46-t1: not landed under exact arithmetic and admitted within 0.1 s of its read."},
    {"file": "stall_ec.out", "what": "nofc_noend, six-arm set: reading turns landed by arrival, exact microsecond arithmetic",
     "mine": "5", "theirs": "7", "likely_cause": "Each of turns c236-t1 and c361-t1 " + EDGE},
    {"file": "stall_ec.out", "what": "nofc_noend, six-arm set: turns where the read was the last wait, exact microsecond arithmetic",
     "mine": "1444", "theirs": "1442", "likely_cause": "Same two turns c236-t1 and c361-t1."},
    {"file": "stall_ec_ctl.out", "what": "ec_r1, SEVEN: reading turns landed by arrival, exact microsecond arithmetic",
     "mine": "209", "theirs": "210", "likely_cause": "Turn c46-t1 " + EDGE},
    {"file": "stall_ec_ctl.out", "what": "ec_r1, SEVEN: turns where the read was the last wait, exact microsecond arithmetic",
     "mine": "748", "theirs": "747", "likely_cause": "Same turn c46-t1."},
    {"file": "stall_ec_ctl.out", "what": "nofc_noend, SEVEN: reading turns landed by arrival, exact microsecond arithmetic",
     "mine": "5", "theirs": "6", "likely_cause": "Turn c236-t1 " + EDGE},
    {"file": "stall_ec_ctl.out", "what": "nofc_noend, SEVEN: turns where the read was the last wait, exact microsecond arithmetic",
     "mine": "1205", "theirs": "1204", "likely_cause": "Same turn c236-t1."},
    {"file": "stall_ec_ctl.out", "what": "jit_p50_ctl, SEVEN: reading turns landed by arrival, exact microsecond arithmetic",
     "mine": "759", "theirs": "760", "likely_cause": "Turn c223-t6 " + EDGE},
    {"file": "stall_ec_ctl.out", "what": "jit_p50_ctl, SEVEN: turns where the read was the last wait, exact microsecond arithmetic",
     "mine": "698", "theirs": "697", "likely_cause": "Same turn c223-t6."},
    {"file": "stall_ec_ctl.out", "what": "hintfree_ctl, SEVEN: reading turns landed by arrival, exact microsecond arithmetic",
     "mine": "4", "theirs": "5", "likely_cause": "Turn c242-t1 " + EDGE},
    {"file": "stall_ec_ctl.out", "what": "hintfree_ctl, SEVEN: turns where the read was the last wait, exact microsecond arithmetic",
     "mine": "911", "theirs": "910", "likely_cause": "Same turn c242-t1."},
    {"file": "stall_ec_ctl.out", "what": "hintfree_ctl, SEVEN: stall per reading turn that had not landed, exact microsecond arithmetic",
     "mine": "2.52", "theirs": "2.53",
     "likely_cause": "Same turn c242-t1 joins the not-landed turns with 0.05 s, which moves the mean from 2.5262 s to just under 2.525 s, across the rounding edge."},
    {"file": "compare_ec.out",
     "what": "column 's2h reads' taken as its label says, s2h_io lines on the arm's kept turns: jit_exact / ec_r13 / ec_r1 / nofc / nofc_noend",
     "mine": "1526 / 1885 / 1854 / 1831 / 2205", "theirs": "1311 / 1652 / 1606 / 1641 / 1950",
     "likely_cause": "The column counts kept turns that read from SSD, first turns included, not reads. Taken as that turn count, "
                     "my numbers equal the file's for all 13 arms."},
    {"file": "compare_ec.out",
     "what": "column 'staged reads' under 'on each arm's own returning turns', taken as ntp_stage lines on the arm's kept returning turns: jit_exact / ec_r13 / ec_r1",
     "mine": "1068 / 550 / 606", "theirs": "1202 / 629 / 670",
     "likely_cause": "The first number of the column is the whole-log line count: it includes reads staged outside the kept window and "
                     "for turns that never arrived. As whole-log counts my numbers equal the file's. The bracketed turn counts "
                     "(1068 / 550 / 605) are on the kept returning turns and equal mine."},
    {"file": "compare_ec.out",
     "what": "column 'read launches refused room' under 'on each arm's own returning turns', taken as srs_dispatch launched=0 lines on the arm's kept returning turns: jit_exact / ec_r13 / ec_r1 / nofc / nofc_noend",
     "mine": "339 / 334 / 370 / 230 / 262", "theirs": "381 / 395 / 396 / 277 / 293",
     "likely_cause": "Same as the staged column: the first number is the whole-log line count, which my whole-log count equals. "
                     "The bracketed turn counts (233 / 227 / 256 / 171 / 185) equal mine."},
]

sanity = [
    "Labels in the value list. arm_set SIX = LEAD5 + ec_r13_rep (11,145 turns, the set of the regenerated stall_ec.out; computed after opening). ALL13 = ALL11 + jit_p50_d + ec_r13_rep (the same 6,715 turns as ALL11; its two entries per new arm were computed after opening). own = the arm's own run: its kept returning turns, kept window or whole log, as the metric says. A metric ending in [exact] is the exact-microsecond value, listed only where it differs from the float value before it. Item 0 holds set sizes and windows.",
    "Runs read: the 13 of the task (12 arms and the order run), plus jit_p50_d and ec_r13_rep after opening. The live run 20261112_004840 was never opened. Each run has 1 client_* folder. The 13 runs hold 135,207 turn records (7,058 to 12,209 per run): ttft null 0, error set 0, duplicate (conv, turn) 0. So the largest t_send is the same over all records and over records with a ttft.",
    "ReqTimeStats: 135,207 keyed lines for 135,207 turn records; 3 unkeyed lines per run (1 warm-up, 2 health checks). Keys with more than one line: 0 in all 15 runs, so keeping the last line (as I do, like the existing analysis) changes nothing. Client turns without a line: 0. attempts=0 on every line.",
    "Key mapping: input_len, output_len and cached_input_len of each ReqTimeStats line equal the client's prompt_tokens, completion_tokens and cached_tokens on 135,207 of 135,207 keys.",
    "Log format: 0 lines of interest without a time stamp and 0 unparsable lines in 15 logs; 177 unstamped lines per log, none of them a line of interest; time stamps step backwards on 26 to 62 lines per log, by 6 ms at most.",
    "Two parsers of mine (regex into a cache; plain string splitting on the raw logs) give identical counts and sums on all 13 runs: 135,207 ReqTimeStats, 30,546 s2h_io, 10,401 ntp_stage, 35,219 srs_dispatch, 81,617 evict_device and 60,234 evict_host lines.",
    "Second from-scratch implementation (exact integer microseconds, raw logs, written and saved before any existing output was opened in this session): 2,895 numbers recomputed. 1,220 of 1,287 integers are identical; the other 67 are the same 1 or 2 turns per arm with a stall of exactly 0.050 s. 1,473 of 1,608 floats agree within 1e-6; the rest differ by at most 9e-6 s per turn (0.05 s per moved turn in the sums).",
    "Re-running the frozen analysis code today reproduces 2,397 of 2,397 frozen values bit for bit (sets SEVEN and CUT7 and all per-run values). The sha256 of the frozen files equals the record written at 21:02:14Z on 2026-11-11, before the existing outputs were first opened.",
    "s2h_io lines with pages=0: 0 in every run, so 'at least one s2h_io line' (mine) and 'pages > 0' (the files) select the same reading turns. s2h_io lines under keys with no client record: jit_exact 4, ec_r13 3, ec_r1 4, nofc 0, nofc_noend 0.",
    "Shared sets: LEAD5 11,198 (all of ec_r1's kept returning turns, 98.9 to 99.1 % of the other four arms'), SEVEN 9,702, ALL11 6,715, CUT7 6,579. LEAD5 by position: 1,453 turns in episode 1, 1,519 in episode 2, 8,226 elsewhere, of which 1,218 are absent from the order run. The order run has 10,319 positions; 12 keys sit in t_send ties and 0 of them change label under another tie-break.",
    "Reads ending after admission: 0 of 9,031 s2h_io lines on LEAD5 shared turns (the closest read end is 2.4 to 4.2 ms before admission) and 0 in the two controls. The upper clamp of the stall binds on 0 turns; the lower clamp binds on 678 / 212 / 260 / 0 / 0 turns (reads that ended before arrival). So slack 0, 0.1 and 0.5 s give identical sums and counts in item 4.",
    "Clock check 5a: in the two arms that read only at arrival no read starts before arrival: nofc minimum +0.027 s over 2,014 lines, nofc_noend +0.025 s over 2,412 lines, 0 lines below -0.05 s. For contrast, lines starting more than 0.05 s before arrival in the arms that stage reads: jit_exact 933 of 1,703, ec_r13 386 of 2,077, ec_r1 418 of 2,011.",
    "Clock check 5b: the server span (ReqTimeStats line time - entry_time) is shorter than the client latency on every one of 60,787 turns of the five runs (maximum of the difference -0.0045 to -0.0061 s), median -0.075 to -0.081 s, 1st percentile -1.26 to -1.46 s. entry_time minus client send time: minimum +0.0033 to +0.0044 s, median 0.070 to 0.075 s, 99th percentile 1.26 to 1.44 s. Time stamps are UTC as stated.",
    "Clock check 5c: queue wait exceeds client TTFT by more than 0.05 s on 0 of 11,198 turns in each LEAD5 arm, and on 0 turns in every arm of all four sets. The minimum of TTFT - queue wait - pre-entry delay is +0.019 to +0.029 s.",
    "Item 6, files as they stood at 01:38Z on 2026-11-12 (copies and sha256 in existing_snapshot/, unchanged at 01:57Z): 1,946 numeric comparisons, 1,946 equal at the printed precision (stall_ec.out 242, stall_ec_ctl.out 282, split_ec_lead.out 260, compare_ec.out 1,162); 17 arm lists and run folders also equal. Origin of my side: 460 frozen before opening, 23 printed in my pre-opening text outputs, 1,463 computed after opening (242 on the six-arm set, 218 rows of the two new arms, 624 per-15-minute rows).",
    "Item 6, earlier versions of the files (21:07Z on 2026-11-11): 1,728 numeric comparisons, 1,728 equal; 574 of them rest on frozen numbers, including 114 of the five-arm stall_ec.out that has since been overwritten.",
    "Which item numbers the four files print now. Item 1: reading turns, turns with a staged read, turns refused room, reads per reading turn and pages per reading turn for LEAD5 (split_ec_lead.out, 23 rows equal); evictions to 0.1 M tokens (compare_ec.out, 22 rows equal); staged and refused line counts on shared turns are printed nowhere. Item 2: SEVEN (stall_ec_ctl.out, 14 rows equal) and ALL11 (compare_ec.out, 22 rows equal); for LEAD5 only split by read class (split_ec_lead.out, 20 rows equal). Item 3: SEVEN (stall_ec_ctl.out, 42 rows equal); LEAD5 only in the overwritten file (30 rows equal then). Items 4 and 5: printed nowhere.",
    "Coverage: every number of stall_ec.out, stall_ec_ctl.out and split_ec_lead.out was compared. Not recomputed in compare_ec.out: landed reads lost, the run-to-run block, the mechanisms columns other than turns, reads per reading turn and evictions, the by-position table, the load K columns, the minute-reached table, and mean pinned tokens, unpins, expected-cost decisions and prices.",
    "Hardware lines of compare_ec.out for the LEAD5 arms, recomputed after opening and equal: decode 37.12 to 37.98 ms per step at 8-15 running requests and 48.51 to 49.55 at 16-23; first-turn prefill 0.087 to 0.090 ms per token; SSD 2.18 to 2.24 ms per page.",
    "Not in the value list: frozen_results_items1to5.json holds all 4,459 frozen numbers (four sets, splits by position, whole-run totals); ext/ext_values.json holds 4,933 on the six-arm and thirteen-arm sets.",
]

concerns = [
    "Two of the four files changed under the check. compare_ec.out (01:33:11Z) and stall_ec.out (01:35:28Z on 2026-11-12) were regenerated with the new arms jit_p50_d and ec_r13_rep. compare_ec.out's all-arms set is still the same 6,715 turns, so the rows of the eleven original arms did not move. stall_ec.out moved from LEAD5 (11,198 turns) to six arms (11,145), so no existing file prints the LEAD5 item 3 numbers any more, while the 13:20Z table of DECISIONS.md still cites stall_ec.out for the 11,198-turn figures.",
    "One repeat moves as much as most gaps between the leading arms. On the six-arm set, ec_r13 against its repeat ec_r13_rep: mean TTFT 0.763 -> 0.861 s (+0.098), mean queue wait 0.354 -> 0.437 (+0.083), read stall 0.241 -> 0.281 (+0.040), stall with the GPU queue holding too 0.111 -> 0.168 (+0.057), stall with the read as the last wait 0.130 -> 0.113 (-0.017), TTFT - queue wait 0.410 -> 0.425 (+0.015), landed by arrival 216 -> 202, reading turns 1,576 -> 1,559, SSD pages 424,388 -> 414,256 (-2.4 %). Among ec_r13, ec_r1, nofc and nofc_noend on LEAD5 the six TTFT gaps are 0.014, 0.029, 0.044, 0.100, 0.114 and 0.143 s, and the repeat itself (0.861 s) lands at ec_r1's level (0.865 s on the same turns); the GPU-queue-held stall spans 0.111 to 0.169 s over the five arms, a range of 0.059 against the repeat's 0.057. It is one pair, so the spread itself is not measured.",
    "The same on ALL11's 6,715 turns, second run over first: original HiCache TTFT +21.0 %, queue wait +21.5 %, stall +19.9 %; hint-free +17.1 %, +23.8 %, +14.3 %; JIT p50 -15.5 %, -22.2 %, -22.6 %; ec_r13 to ec_r13_rep +18.3 % (0.656 -> 0.776 s), +45.6 % (0.247 -> 0.360 s), +37.1 % (0.163 -> 0.223 s).",
    "The 0.050 s landed threshold falls on the logs' 1 ms grid. 0 to 2 reading turns per arm have a stall of exactly 0.050 s, and float rounding decides their class in the existing files and in my frozen numbers. Exact arithmetic changes landed and last-wait counts by 1 or 2 in ec_r1, nofc_noend, jit_p50_ctl and hintfree_ctl (13 printed numbers in the two stall files) and the last-wait sum by 4e-6 to 9e-6 s per turn. Entries marked [exact] are the exact-arithmetic values.",
    "The split of the stall depends on the 0.1 s bound (item 4). From bound 0.05 to 0.5 s the last-wait sum goes 0.040 -> 0.082 s per turn in jit_exact, 0.108 -> 0.157 in ec_r13, 0.103 -> 0.169 in ec_r1, 0.142 -> 0.204 in nofc, 0.183 -> 0.260 in nofc_noend. ec_r13 minus ec_r1 changes sign: +0.005, +0.006, -0.004, -0.012 s at bounds 0.05, 0.1, 0.2, 0.5. nofc minus jit_exact stays 0.102 to 0.122 s and nofc minus ec_r13 0.034 to 0.047 s.",
    "Read stall is mostly not disk time. On LEAD5 shared turns the read I/O after arrival is 21 to 34 % of the stall (426.7 of 2,031.9 s in jit_exact, 911.4 of 2,689.9 in ec_r13, 930.5 of 3,114.0 in ec_r1, 1,172.6 of 3,838.3 in nofc, 1,430.8 of 4,270.0 in nofc_noend); time parked before a launch is another 29 to 40 %; the 109 to 181 turns per arm whose read was held on purpose (ntp_hold) carry 33 to 59 % of the stall. In nofc turn c249-t48 waited 21.99 s before its launch and counts as GPU-queue-held because admission came 0.79 s after its third read.",
    "s2h_io's ms does not cover launch to landing. Read start (line time - ms) minus the launch line (srs_dispatch launched=1) has a median of 1 to 2 ms but exceeds 1 s on 175 of 1,707 reads in jit_exact, 237 of 2,080 in ec_r13, 209 of 2,015 in ec_r1, 264 of 2,014 in nofc and 390 of 2,412 in nofc_noend, up to 2.6 to 3.7 s. So 'read start' in item 5a is more than 1 s later than the launch on 10 to 16 % of reads, and the files' 'inside the read' leaves out the wait between launch and I/O start.",
    "TTFT - queue wait is not only time after admission. It contains the delay from client send to entry_time: mean 0.128 to 0.134 s on LEAD5 shared turns, 31 to 32 % of the 0.408 to 0.438 s; 0.17 to 0.20 s in the two episodes against 0.11 s elsewhere; 99th percentile 1.23 to 1.41 s, maximum 3.0 to 3.2 s, over 0.5 s on 3.8 to 4.6 % of turns.",
    "The means rest on few turns. The two episodes (2,972 turns, 26.5 %) hold 82 to 89 % of each LEAD5 arm's summed queue wait and 80 to 92 % of its stall; episode 2 alone holds 49 to 63 % of the queue wait. The 112 turns with the largest queue wait hold 34 to 51 % of the queue wait and 29 to 49 % of the stall, from 55 to 67 sessions; the 11 largest hold 6 to 9 %.",
    "1,218 of LEAD5's 11,198 shared turns (10.9 %) are absent from the order run. They count as 'elsewhere' by the task's rule but appear in no stretch row of the files. Their mean TTFT is 0.56 to 0.70 s against 0.40 to 0.46 s for all 'elsewhere' turns. The order run has 10,319 positions, so episode 2 (8,750 to 10,999) can hold at most 1,569 of them.",
    "SEVEN cuts the leading arms short. Only 85.7 to 86.6 % of their kept returning turns are in SEVEN, because the two controls are slower: the last shared turn is sent at 6,063 to 6,138 s in the leading arms against 6,836 to 6,900 s on LEAD5, and episode 2 has 1,241 turns instead of 1,519. Their mean queue wait is 0.022 to 0.042 s higher on SEVEN than on LEAD5.",
    "'Landed by arrival' is a threshold effect in the arms that read only at arrival. nofc's 4 and nofc_noend's 7 (5 exact) are reads that ended within 50 ms after arrival; with the threshold at 0 both are 0, at 0.1 s they are 32 and 34. In the arms that stage reads the last read truly ended at or before arrival on 678 of 684 (jit_exact), 212 of 216 (ec_r13) and 260 of 269 (ec_r1) landed turns.",
    "Refused launches without any read are outside the stall. 42 / 51 / 52 / 24 / 36 LEAD5 shared turns had a launch refused and no s2h_io line; their queue wait, 0.015 / 0.012 / 0.021 / 0.016 / 0.019 s per shared turn, sits in the rest of the queue wait. In the controls on SEVEN it is 214 turns and 0.049 s (jit_p50_ctl) and 210 turns and 0.175 s (hintfree_ctl).",
    "Evictions are per run, not per shared turn. The kept windows of the LEAD5 arms are 6,896.8 to 6,900.0 s, so the sums are comparable there; hintfree's window is 6,719.2 s and jit_p50_d_cut's 4,482.5 s. The files print them to 0.1 M tokens only. ec_r13 against its repeat: device 51.43 -> 50.10 M, host 39.64 -> 38.44 M tokens.",
    "What I could not do. The LEAD5 numbers of item 3 and LEAD5's mean queue wait have no printed counterpart in a file that exists now; I compared them with the overwritten version as my earlier comparison quoted it. Items 4 and 5 have no printed counterpart at all. Every number on the six-arm set and of the two new arms was computed after the files were opened, so it is not blind.",
    "Process. This check ran in two sessions: the first froze items 1 to 5 and compared them, then ended before reporting; this one re-derived them with a second implementation before opening anything. Two sub-second inline python one-liners over my own JSON outputs were run without nice and outside a script file; everything touching run data ran as a script under nice -n 19 python3 -I -B, one at a time.",
]

scripts = ["parse_runs.py", "analyze.py", "explore.py", "check_raw.py", "check_stall.py", "supp.py", "compare.py", "posthoc.py",
           "posthoc2.py", "check_clock.py", "final_values.py", "recheck.py", "recheck_diff.py", "parse_new.py", "analyze_ext.py",
           "compare2.py", "exact_cmp.py", "tally2.py", "verify_concerns.py", "spot.py", "final2.py", "final2_add.py", "final_answer.py"]
for s in scripts:
    assert os.path.exists(os.path.join(OUT, s)), s
answer = {"values": values, "mismatches": mismatches, "matched_count": 1946, "sanity": sanity, "concerns": concerns,
          "script_paths": [os.path.join(OUT, s) for s in scripts]}
with open(os.path.join(OUT, "final_answer.json"), "w") as f:
    json.dump(answer, f, indent=1)
with open(os.path.join(OUT, "final_answer_values.txt"), "w") as f:
    for v in values:
        f.write(json.dumps(v, separators=(",", ":")) + "\n")
print("values %d, mismatches %d, sanity %d, concerns %d, scripts %d" % (
    len(values), len(mismatches), len(sanity), len(concerns), len(scripts)))
