# Next-turn hints from gap distributions

How an agent system gives the LLM system next-turn timing using only what a real system knows. This covers the first step;
later steps are listed at the end. Written 2026-10-31. The client is built, the engine is unchanged, and nothing has run on
a GPU yet.

## Why

The oracle campaigns told the engine each session's exact next request time, and it paid off: mean TTFT fell from 1.09 s
to 0.58 s at x70 and from 95.8 s to 20.3 s at x10. A real system does not know that time. It does know this: the LLM's reply
names a tool call, and the tool call mostly sets how long the next request takes to come.

## The design

1. **The agent system keeps the numbers.** It calculates the gap distribution per tool category while the system is in
   use, and refines it with every request.
2. **The LLM system turns them into a deadline.** It reads each user's table from the requests. For now it uses one simple
   number, the p50; better methods come later.
3. **The LLM system looks the tool up after generation.** Once the reply exists it knows the tool call, checks that user's
   table, and gets the next round's time from point 2.
4. **The oracle routine does the rest.** The first step reuses the oracle design unchanged, with this time in place of the
   exact one. We adjust it later.

```
Agent system                                        LLM system
------------                                        ----------
table: tool category -> gap distribution
       refined by every finished gap

request + current table    ------------------->     generates the reply
                                                    reads the tool call from its own reply
                                                    looks it up in this user's table
                                                    next gap = that entry's p50
                                                    runs the oracle routine with that gap
runs the tool              <-------------------     reply
measures the gap, updates the table

next request + updated table  ---------------->     ...
```

## What the agent system does

- **Categories.** The tool plus the first word of its command: `execute_bash/pytest`, `execute_bash/git`,
  `str_replace_editor/view`. The first word is what separates a long test run from a quick `ls`: it picks out gaps of 300 s
  or more far better than the tool alone (AUC 0.84 against 0.66 on this trace). File sizes add almost nothing and stay out.
- **The table.** Each category has p10, p50, p90 and the number of gaps behind them, in seconds. An entry appears after 5
  gaps. A tool-level entry and a default entry `*` cover categories with too few gaps.
- **Online.** A gap runs from the reply's last token to the next request. When it ends, the client adds it to the table, and
  the next request from any session of this agent system carries the new numbers.
- **On the wire.** Every request carries the table in `kv_hints`, as a version 2 `sglang.next_turn` action
  `{session, next_rid, gaps}`. A full table is about 3 KB (38 entries on this trace). `next_rid` is the id the client will
  give its next request, which is how the engine links the two.

## What the LLM system does

- **Reads the table** from the request. Every user has its own table, and they are never mixed.
- **Reads the tool call** from its own reply when the turn commits (`on_turn_committed`, where the oracle hint is read
  today), using the same category rule as the client.
- **Picks the entry:** the category, else its tool, else `*`.
- **Takes the p50 as the next gap.** The p50 rather than the mean, because rare long gaps pull the mean up (at x70, median
  50 s and mean 74 s). This is one line to change.
- **Runs the oracle routine** as if the client had sent that number: next-use registration and eviction, deadline-driven
  reads, pins, expiry and arrival all stay as they are.

## Mapping the real system onto the trace replay

The design assumes a real agent system. These are the points a replay cannot do, and what stands in for them.

| Real system | Trace replay |
|---|---|
| An agent system runs many sessions | Each harness and model in the trace is one agent system. Its table starts empty when the run starts and grows across its sessions |
| The client measures the gap after running the tool | The client sleeps the recorded gap x scale (at most 1,200 s) and adds it to the table when the sleep ends |
| The engine reads the tool call from its reply | The replay's generated reply is not the recorded one, so the request also carries `replay_reply_key`, the recorded reply's category (`traces/<trace>.reply_keys.json`). The engine reads it at commit, in place of its own reply, and only under the server setting `reply_key_source: hint` |
| The table survives restarts | Not modelled: every run starts cold. Campaign metrics start at 45 min, after the table has filled |

## What gets built in the first step

**Client: done, not committed, never yet run against a real server.** `scripts/gap_hints.py` (`OnlineGapTable`),
`scripts/replay_agentic.py --hints dist` (`HINT_SOURCE=online` is the default), `scripts/reply_keys.py` (the recorded
categories) and 60 tests in `scripts/test_gap_hints.py`. Two other sources stay for comparison: `precalc` (the table from the
whole trace, known in advance) and `session` (each session's own gaps alone).

**Engine: not started, about 100 lines plus tests (an estimate).** No edit to `Scheduler`, `TokenizerManager` or
`ModelRunner`. Under `python/sglang/srt/mem_cache/`:
- `next_turn_prefetch.py`: a version 2 reader beside `read_next_turn_hint`. In `on_turn_committed`, pick the entry, take its
  p50 and continue down today's path. Version 1 stays, so the oracle arms keep working.
- New `reply_key.py`: reads the tool and first word from the reply's text, with the same rule as `gap_hints.reply_key`. Text
  form only for now, which is 83 % of the calls in this trace.
- A server setting `reply_key_source`: `reply` (the default) reads the real reply, and `hint` reads `replay_reply_key`.
- A test that runs every recorded text reply through the new reader and compares it with `reply_keys.json`.

**The first measurement does not need the engine work.** The client also sends the entry's p50 as a version 1 action, and
today's engine acts on it exactly as the new engine would: the same number at the same moment. So the first runs need only
the client. The engine work is what makes this usable in a real system, and it does not change the replay's result.

## How we evaluate it

**The question:** how much of the oracle's gain survives when the exact gap is replaced by the p50?

**The runs.** The `three_tier_wc_oracle_v2_nu` server on the same frozen engine (`frozen/engine_5df71c4bf0_v2`) with
`HINTS=dist`: a short smoke, then one 5 h run at x70 and one at x10, about 12 GPU-hours (an estimate). Each is compared on
the same turns with the baseline and oracle arms of campaign 18 (x70) and campaign 19 (x10), which used the same users and
seed. x10 gives the clearer answer because its gain is large (75 s). The x70 gain is 0.51 s and the spread between full runs
has never been measured, so x70 needs repeats if the result lands close.

**How good the numbers are, before any run.** Scored offline on campaign 18's real send times, x70, after the first 45 min:

| Table | Gaps below p10 / p50 / p90 (ideal 10 / 50 / 90 %) | Quantile loss (lower is better) | Gaps of 300 s or more above p90 | AUC for gaps of 300 s or more |
|---|---|---|---|---|
| Built during the run | 10.3 / 44.3 / 87.3 % | 0.180 | 63.9 % | 0.837 |
| Known in advance from the whole trace (best case) | 10.4 / 48.2 / 89.3 % | 0.179 | 62.3 % | 0.843 |

The online table costs little once warm. In the first 15 min it is weak (AUC 0.62, and 4.5 % of gaps have no entry), and it
sits a little low because a long gap only joins once it has finished. Scripts and output are in
`frozen/dist_hint_design/online_table/`.

**What we expect (not measured).**
- **Gaps under 60 s** (78 % at x70, 98 % at x10): the p50 is about right, so these should behave like the oracle.
- **Gaps of 300 s or more** (4.2 % of gaps but 35 % of idle time at x70): the p50 is far too short. The routine treats such a
  session as overdue, then expires its plan and marks it ended; on the known-in-advance table's p50, 98.6 % of these gaps go
  overdue and 90.8 % expire. This is where the gain should shrink, and it is the first thing to fix after the first step.

## Later steps

The fuller engine design for these is in `frozen/dist_hint_design/DIST_HINT_DESIGN_long_20261031.md`.

1. **Better than one p50.** Use the whole distribution: move the deadline as idle time passes, and never let a forecast end
   a session.
2. **One owner of reads.** Arrival stops starting a blind fetch and the scheduler decides when to read, aiming at one read
   per turn. Today's arrival fetch stays until then, because the first step reuses the oracle routine as it is.
3. **Read timing.** Which quantile to stage at, how early to start, and just-in-time reads for arrived turns.
4. **The campaign 18 and 19 engine issues:** landed spans that next-use eviction never registers, pins, and whole-session
   eviction at x10 (`ORACLE_FINDINGS.md`, section 11).

## Logged for later

Not evaluated and not blocking the first step. Full review texts and scripts are in `frozen/dist_hint_design/review_20261031/`.

**Session end**
- How a real engine learns the agent finished: a `finish` call or a reply with no call has no "ended" entry yet. The replay
  uses the key `_end`.
- 53 sessions end in the trace but the agent went on under another session id; the replay says "ended" there.
- 228 sessions stop at exactly 50 calls. The evidence points to the harness's budget (`frozen/dist_hint_design/cap_reading/`),
  so the replay says "ended" there too. 26 sessions running past 50 calls remain unexplained.
- A turn with no table entry counts as a session end today. It only matters during the cold start.
- Human waits cannot be tested on this trace, which holds batch agents only.

**Table and categories**
- Starting from earlier runs' tables, and keeping tables across restarts.
- Old gaps never leave the table, and it goes stale when a harness or model changes.
- Sending only table changes instead of 3 KB per request.
- The online table depends on run timing, so two arms get slightly different tables.
- A leading `# comment`, `export X=y &&` or `timeout N` hides the real command word.
- Rejected calls borrow a key, and a rejected first call can be told from its key on MiniMax sessions.
- The client's table has a `_no_call` category that the planned engine reader does not produce yet.
- Native tool-call formats need per-parser work (about 40 parsers).

**Client housekeeping**
- The `precalc` comparison source only: `task_of` misses `__run2` ids, so a session's own run can enter its list, and leaving
  one task out lets `n` reveal that task's call count.
- Tests are weaker than their count suggests: 22 of 59 deliberate breaks of the older code still pass.
- `hint_eval.py` crashes on very short `--turns` or small `--scales`.
- The launcher pre-flight accepts a stale keys file, `--hint-*` flags without `--hints dist` are silently ignored, and `KNOBS`
  in `gap_hints.py` is unused.
- The run record gained `hint_*` args, so it is the one record that is not byte-identical to before for no-hint and oracle
  runs. Request bodies are identical.
- `--hint-end natural` counts the 50-call stop per session id, `reply_keys.py` cannot detect a lost call, and 56 rejections
  with no tool name get a random tool.

**For the later engine steps** (from the review of the long design)
- Holding a session from p90 to late protects the stalest sessions most, and a forecast can fall below hint-less traffic in
  eviction order.
- Parked lookups hold a lock nobody budgets for.
- Exact and forecast plans follow different rules, so comparing them is not only comparing gaps.
