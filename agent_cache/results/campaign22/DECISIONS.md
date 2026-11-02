# Campaign 22: where does the x10 gain come from? One change at a time

Requested 2026-11-01, after the discussion of whether the gain comes from next-use eviction: "Just schedule the settlement evaluation after the ongoing run in case it
finishes earlier than I came back." It starts by itself when campaign 21 ends.

**The question.** At x10 the oracle arm cut mean TTFT from 95.8 s to 20.3 s (campaign 19), and p50 hints kept all of it (campaign 20). The arm changes four things at once. Which one
carries the gain, and does any part of it need a hint?

**The ladder.** A dry run of `start_server.sh` confirmed that each arm's flags differ from the previous arm's only by what this table adds.

| Step | Arm | Adds | Needs a hint? | Run |
|---:|---|---|---|---|
| 0 | `three_tier_wc` | original HiCache | no | campaign 19 |
| 1 | `three_tier_wc_poll0` (new) | paced miss retries off | no | here |
| 2 | `three_tier_wc_dl_poll0` (new) | deadline read order, read window and pins | no | here |
| 3 | `three_tier_wc_oracle_v2` | next-turn prefetcher: staged reads, re-reads of queued losses | a plan | here |
| 4 | `three_tier_wc_oracle_v2_nu` | next-use eviction | yes | campaign 19; with the p50, campaign 20 |

**The workload.** Campaign 19's, unchanged: `USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 C=128 SEED=0 ORACLE_HINTS=1 CLIENT_TIMEOUT=7200` on `traces/lmcache_agentic_trace_262k.json`, `LEVEL=NAT160`, on the frozen engine
`frozen/engine_5df71c4bf0_v2/python`, the SSD wiped before each arm. Arms 1 and 2 ignore the hints the client sends, so request bodies match campaign 19's. Cookbook divergences as in campaign 20.

**2 h per arm**, by request ("set the timeout to be 2 hours for all of them"), where campaign 19 ran 5 h. So compare on the turns both arms completed, which means campaign 19's first 2 h. That still holds the baseline's storm, which starts at 38 min. It leaves out
the first SSD cleanup (2.4 h in campaign 19's oracle arm) and the 60 s wait episode after the second.

**One arm first.** The launcher runs campaign 21's fourth arm (`three_tier_wc_dl_poll0` at 44 users and 70 GB) before these three, since it needs the same new arm in `start_server.sh`.

**Reading it.**
- The step where the storm stops (recompute falls from thousands of tokens per returning turn to tens, turns/s rises from about 0.6 to about 2.3) is what keeps the arm out of it.
- Step 3 against step 4 is next-use eviction's own share at x10. If they match, the gain does not come from next-use eviction.
- Steps 1 and 2 show how much needs no hint at all.
- One run per arm, and campaign 19's two ends ran two weeks earlier. The spread between runs at x10 is unmeasured, so only large differences count; a storm's presence or absence is one.

**Launcher.** `frozen/oracle_scripts_staging/c22_ladder/c22.sh` (copy here as `launcher.sh`, log `launcher.log`). It waits for `C21 ALL DONE`, then installs `c22_ladder/start_server.sh`, the current file plus the two new arms and a boot
line for the retry interval, into `scripts/` if that file has not changed since. It does not start after a failed campaign 21 arm. `touch GO` here starts it without that line (used if campaign 21 is stopped under its own rules), and `touch CANCEL` gives up.

**Not running since 2026-11-01 14:25Z.** The host reboot ended the launcher during the first arm, `three_tier_wc_dl_poll0` at campaign 21's setting, at 117 of its 120 min; that run has no summary line and no manifest line (`../campaign21/DECISIONS.md`). None of this ladder's three 128-user arms ran. The user asked to restart campaign 23, so it started with `GO` and this campaign stays stopped. The ladder is still the answer to where the x10 gain comes from.
