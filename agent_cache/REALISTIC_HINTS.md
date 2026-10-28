# Realistic next-turn hints: discussion record and open decisions

Written 2026-10-28. This records a discussion held on 2026-10-27 and 2026-10-28: how to replace the oracle next-turn hint
of campaigns 15-19 with one a real system could send. Nothing has been done yet. The engine, the replay scripts and the
results are unchanged, and no GPU job ran. **Status: waiting for the user's decisions (§7).**

This file is meant to be enough on its own to make those decisions. The evidence behind it is in
`agent_cache/frozen/hint_realism_20261027/`: scripts, outputs and full reports. That folder is git-ignored
(`.git/info/exclude`), so it exists only on this machine, and the paths into it below resolve only here.

**Reading the numbers.**
- A number is a measurement unless it is marked *(estimate)*, *(derived)* or *(inferred)*.
- The bracket after a claim names its source; the keys are in §10.
- "Offline" measurements come from two places: the public lmcache agentic trace that our replay uses (768 sessions,
  24,879 turns), and the calibrated offline cache simulation built on the campaign 18/19 timelines.
- Campaign measurements come from the H200.
- Code anchors are `path:line` at HEAD `799bac38ef`. Engine paths are under `python/sglang/srt/`.
- Times are UTC, the machine's clock.

## 0. The short version

- **The question (2026-10-27).** The engine gets its next-turn hint from the trace, which a real system does not have.
  How do we make the improvement real? The user offered three options:
  - a less accurate hint, such as a distribution;
  - just the kind of work, from which the engine infers the distribution;
  - some other kind of feedback.
- **Round 1 answer: the engine forecasts (§2.2).**
  - A distribution is the right form, but the engine should build it, from what it can see: the tool call it
    generated, the session's idle time and its own queue.
  - The client sends only a few facts the engine cannot see.
  - Before building a predictor, measure how much of the gain needs a hint at all, and fix the engine's hint-free
    rules first.
- **The user's direction (2026-10-28).** The agent system on the client (Claude Code or another harness) estimates
  when its next round will come, from its workload type and the system status. It sends that estimate to the serving
  system as the KV hint. The trace evaluation should then use hints only as accurate as a real harness's would be.
- **Round 2 answer: mostly yes; the agent declares, the engine forecasts (§4-§6).**
  - The harness knows exactly what *kind* of wait it is in: a tool, an approval, the user, background work, a
    scheduled wake-up, a compaction, or the session's end. A server can infer little of this. So the harness should
    send these facts, plus its own estimate if it has one.
  - The harness does not know *when* the session will be back, except for waits it schedules or bounds itself.
    Durations come from history, and the server sees more of it: every session it serves, plus its own queue.
  - Claude Code's newest request headers for gateways already send facts, not forecasts.
- **Making the evaluation realistic (§6):**
  - model what the agent would know, with hints learned online from finished turns only, not noise around the oracle;
  - fix the engine first, or we measure its expiry rule, which turns a late session into an ended one;
  - add a human-paced workload built from TraceLab;
  - compare all arms on one fixed engine: the CPU simulation first, GPU only for the decisive pairs, with at least
    3 replicates each.
- **Decisions waiting (§7):**
  1. Adopt "the agent declares, the engine forecasts", with the agent's own estimate tested as one arm?
  2. Add the TraceLab workload?
  3. Run in the order: CPU simulation, engine work, GPU pairs?

  Four questions from round 1 are still open, and this plan's GPU budget needs approval.

Proposed order, if all three are accepted (§8; nothing has started):

| step | what | takes | GPU |
|---|---|---|---|
| §8.1 | CPU simulation of the arms on the c18/c19 timelines | ~1 CPU-day (estimate) | idle |
| §8.2 | TraceLab download, audit and converter | 77.2 MB; ~1 CPU-day + 0.5-1 day (estimate) | idle |
| §8.3 | Engine fixes: (a)-(e), round 1's other fixes, a post-reply hint channel | (a)-(e): ~110-160 engine + 150-200 test lines (estimate); the rest not sized | idle |
| §8.4 | Engine forecaster and reply shim, needed by arms E1-E3 | ~1.1-1.5K lines (estimate) | idle |
| §8.5 | Causal client-hint mode in the replay client | not sized | idle |
| §8.6 | Round 1's GPU checks: restored-KV correctness, whether the x10 win needs hints, x70 headroom without staging | ~31-32 GPU-h (estimate) | busy |
| §8.7 | Load point: sweep the load on the fixed engine, floor vs oracle | 20-60 GPU-h (estimate) | busy |
| §8.8 | GPU decisive pairs, ≥ 3 replicates each | 60-75 GPU-h per workload (derived: 12-15 runs × ≤ 5 h) | busy |
| §8.9 | Later: live demo, Claude Code → SGLang `/v1/messages` | not sized | busy |

GPU total: ~110-170 GPU-h with one workload in §8.8; each further workload adds 60-75 (derived).

## 1. Background: the oracle hint and what it bought

**What the replay sends.**
- With turn k's request, the client attaches a hint about turn k+1 of the same session.
- The hint is one `sglang.next_turn` v1 action, `{session, next_rid, next_gap_s}`.
- `next_gap_s` is exactly the sleep the client then takes: the recorded gap × 70 (c18) or × 10 (c19), capped at
  1,200 s.
- No hint is sent on a session's last turn. None is sent either when the client's 18,000-s timeout will certainly cut
  the next turn: that happened on 21 turns of c18's oracle arm (`agent_cache/scripts/replay_agentic.py:136-156`)
  [replay §0, §1.2].

**When the engine reads it.** Once, when turn k commits (`mem_cache/unified_radix_cache.py:1168-1176` →
`mem_cache/next_turn_prefetch.py:265`). To the engine, this is the same as a message sent right after the reply with
no delay [replay §1.3].

**It is three signals in one** [FP §2]:
- session linkage: which later request continues this session;
- the end: a turn without a hint is treated as the last one, registered ENDED and evicted first;
- the gap value.

A real system has the first. It has the second only partly: through a close or a parsed finish call, never for a
session that just stops [FP §2]. It does not have the gap. The hint is built before reply k exists, and reply k (the
tool it calls) mostly sets the gap (inferred) [engine §0.2].

**What it bought.** These are upper bounds from one run per arm [OF §1]:
- c16 (old client, x70): mean TTFT 18.5 s for the c14 baseline, 5.7 s for `_oracle_v2_nu`.
- c18 (redesigned client, x70): 1.09 s → 0.58 s. The −0.51 s has a block interval of [−0.89, −0.21] [FP §9].
- c19 (x10): the baseline fell into a recompute storm. Mean TTFT was 95.8 s vs 20.3 s on the 15,033 turns it ran. The
  oracle arm finished all 768 users in 4.7 h; the baseline finished 413.

**How much of that the hint itself bought is unknown.**
- The x10 win came from the baseline's retry-budget storm [FP §1].
- Mechanisms that do not use the gap value acted in the same arm and were not separated from it [FP §2; OF §10]:
  - pins and paced retries off (poll interval 0), which need no hint;
  - the queued-turn refetch, which needs the hint's session linkage but not the gap.
- Run-to-run spread at ~1 s TTFT was never measured. The only estimate, ~1 s, came from a smoke run at 13-17 s TTFT
  [FP §9].

## 2. The discussion, in order

### 2.1 The question (user, 2026-10-27 18:07)

> Let's work to make this system improvement real, now the system is receiving the oracle hint from the trace, but in
> the real system, we don't have such priviledge. What's in my mind is that the hint could be 1) less accurate, like a
> distribution, 2) maybe barely work type, so the LLM engine can infer the distribution and make the prediction on the
> next turn time. What do you think? Let's first discuss the idea before taking any actions. Or is there any other
> feedback type that helps?

### 2.2 Round 1 answer (sent 2026-10-28 00:02): the engine forecasts

**How it was reached.**
- Five read-only investigations: the engine, the campaign evidence, how predictable the trace is, the cache simulation,
  and the literature.
- Four design proposals were then judged and merged into one position.
- That position's 123 claims were checked one by one: 102 held, 21 were corrected.
- The result is [FP]. The chat answer condensed it; where the two differ, this section says so.

**Opening line of the answer:** a distribution is the right form, but the engine should build it, not the client. The
client should only send a few facts the engine cannot see. And before building a predictor, we should measure how much
of our gain needs a hint at all; the evidence says less than it looks.

**The oracle hint is three signals** (§1): linkage is realistic, the end partly, the exact gap not.

**Idea 1, a less accurate hint such as a distribution: right form, wrong producer.**
- It adds little information: a client would derive it from the tool call the engine itself just generated. In the
  simulation (x70, 959K room), a per-tool typical gap sent as a hint keeps 50-54 % of the oracle's eviction gain. The
  engine's own per-tool estimate plus an exact session end keeps 48-54 % [sim §0.2].
- Realistic errors are one-sided. For x70 gaps over 300 s, the best realistic predictor is about 3× too short at the
  median, and more than 4× too short 41 % of the time [trace §8]. Today's engine would turn about 60 % of those gaps
  into "ended, evict first" [FP §3].
- Nothing in production sends next-turn times; vendors sell retention times instead [lit §4.1].

**Idea 2, the engine infers the distribution: yes, as the default** [FP §4].
- The predictive signal is in the reply, which the engine wrote. The tool name, together with the source and the
  harness, explains 73.6 % of the variance of the log of the gap. With the command, R² is 0.805, and the AUC for
  picking out x70 gaps over 300 s rises from 0.70 to 0.90 [trace §0.1, §4].
- A table per (source, tool, command class), learned online from earlier sessions, reaches R² 0.766. A session's own
  past reaches at most 0.42 [trace §0.5].
- Ranking idle sessions by their chance of returning soon, given how long they have been idle, beats LRU: AUC 0.900
  vs 0.763 at x70 [trace §8].
- Share of the oracle's gain kept in the simulation [sim §0.2]: idle time only 27-32 %; plus the tool called 32-38 %;
  plus exact session ends 48-54 %. An end the engine has to guess lands toward the low end (inferred).
- Limits: it cannot see ends without a finish call, harness stalls or human pauses; about 75 % of the gaps come from
  one harness family [FP §4]; the replay cannot exercise it, because the replies are generated text, not the recorded
  tool calls.

**Other feedback, ranked by value per bit:**
1. queue state;
2. session ended;
3. "waiting for a human";
4. a "next turn coming" ping a few seconds ahead.

The chat called tool-finished events and client-sent gap distributions "not worth it". [FP §3, §5] is softer: it
accepts distributions as an optional input checked against outcomes, and it ranks two more signals (tool class with
timeout, retention class).

**Fixes needed regardless of hints:**
- a missing hint means UNKNOWN, not ENDED;
- queued requests stay protected until admission;
- expiry never ends a session;
- every queued request gets a refetch;
- paced retries off, pins only for requests that have arrived, and early SSD reads off by default.

**Reality check:**
- The hint-driven share of the gains is unmeasured (§1).
- x70 inflates hint value (§6.3).
- More host memory is the rival: in the model, LRU with 291K more tokens of room (~28.6 GB, derived from 96 KiB per
  token [OF §2]) beats every realistic signal [sim §3].
- At the trace's real pacing (median gap 0.7 s), this machine barely touches host memory or the SSD.

**First steps proposed** [FP §8]:
1. CPU simulation, about a day;
2. restored-KV correctness, 1-2 GPU-h;
3. whether the x10 win needs hints, ~15 GPU-h, no new code;
4. x70 eviction with exact hints but no early reads, ~15 GPU-h.

After those come a TraceLab check, the hint-free fixes under a load sweep, a session-end-only run, the forecaster and
pings. That is about 75 GPU-h at one load level, or ~115 with the sweep (estimate).

**Five questions were asked** (§7 lists the four still open). [FP §10] has a sixth, on the GPU budget and TraceLab.

### 2.3 The user's direction (2026-10-28 01:02)

> The end-to-end system idea is that the agent system running on the client's machine will provide the estimation for
> the next round time, based on its own workload type and the system status. Then, the agent system will provide the
> hint to the llm serving system, so that the serving system can manage its kv cache efficiently like what we've
> explored. Right now, we are running the trace driven evaluations, to make it practicle like the real system, we should
> update the hint's accuracy to mimic the real system and workload. In this direction, what do you think? The agentic
> system (like claude code/ other agent system) should provide the kv hint since it should know its own workload, right?

### 2.4 Round 2 answer (sent 2026-10-28 01:51): the agent declares, the engine forecasts

**How it was reached.** A draft position (claims D1-D12) took the user's framing: the harness produces the estimate.
Five independent investigations then checked it:
- what Claude Code, Codex and OpenHands expose, from their documentation and code;
- what TraceLab records, a public dataset of real Claude Code and Codex sessions;
- how the replay and the engine would carry a realistic hint;
- a skeptical critique of the draft;
- a fact check of every claim.

**What the critique changed:**
- The draft said the harness knows everything the engine knows, and more. That is false: the engine knows its own
  queue and statistics pooled over all sessions [critic O6].
- The draft's ladder of arms compared amounts of information, not who produces the hint. No arm compared a client's
  estimate with an engine's estimate made from the same data [critic O1].
- Most hint columns of `trace/realistic_hints.csv` were fitted out-of-fold, partly on later sessions. A client could
  not compute them online [critic O2].
- On the GPU, a five-arm ladder cannot be resolved inside a −0.51-s effect [critic O4].

**The answer** is §4-§6.
- What changed from round 1: the client's facts move from "other feedback" to the primary input. They are the
  signals only the client has, and in real use the long waits are human ones (§3.4).
- What did not change:
  - durations still come from statistics the engine owns;
  - the hint-free fixes still come first;
  - a client estimate is still used as a prior, checked against outcomes.

### 2.5 Corrections to what the chat said

Found while writing and checking this record on 2026-10-28. The rest of this document already uses the corrected
statements.
- **Round 2: "they score 0.83 at separating long from short gaps (AUC), against 0.70-0.72 for every online
  learner."** The 0.83 belongs to a separate out-of-fold classifier in the same file (`p_gt_30.000`). The out-of-fold
  point hint scores 0.74 (recomputed: `pred_s5_s` 0.736, `pred_online_s` 0.702, `p_gt_30.000` 0.831). The point hint's
  edge over online learners is therefore small (§4.1).
- **Round 2: "My rough estimate is 110-160 lines plus tests."** That covers only fixes (a)-(e). Round 1's other fixes,
  an engine forecaster (~1.1-1.5K lines, estimate [FP §8 step 8]) and a post-reply channel come on top (§8.3, §8.4).
- **Round 2: the GPU plan.** It left out the load sweep [FP §8 step 6], which is needed to find a load at which the
  floor-to-oracle gap exceeds twice the spread (§8.7).
- **Round 2: "Claude Code can already run against this SGLang."** This follows from the documentation and our code.
  It was never tested end to end (§6.5).
- **Round 2: "gateway hint headers ... session id".** The session id is an ordinary request header, sent on every
  request. The hint headers go by default only to the Anthropic API; a custom base URL needs
  `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` (§3.1).
- **Round 2: "TraceLab has 8,058 ... sessions. Human waiting is 92 % of session wall-clock."** The 92.3 % is the
  paper's figure on its 4,265 sessions; the repository's refresh on the 8,058 gives 91.6 % (§3.4).
- **Round 1: "The tool name alone explains 74 %."** That model also includes the source and the harness, which alone
  explain 6.5 % [trace §4].

## 3. What a real agent harness knows

### 3.1 Claude Code

This comes from the official documentation, because the CLI source is not public [harness §0, §2]. Hooks can POST
their JSON payload to a URL, so a separate program (a sidecar) can observe most of this without modifying the harness.
Typing needs a mod.

| what the harness knows | exact? | where Claude Code exposes it |
|---|---|---|
| A tool is running: which one, its timeout, whether it runs in the background | yes | `PreToolUse` hook (`tool_name`; `tool_input` with `timeout`, `run_in_background`) |
| A permission prompt is waiting for a human | yes, with two exceptions (below) | `PermissionRequest` hook, fired when it asks; the `permission_prompt` notification comes only after ~6 s |
| The turn is over: it is waiting for the user, or paused for background work or a scheduled wake-up | yes, with two exceptions (below) | `Stop` hook; its `background_tasks` and `session_crons` fields tell "done" from "paused" |
| A question to the user is open | yes | `AskUserQuestion` (optional timeout of 60 s, 5 min or 10 min) |
| The context was compacted, so the cached conversation prefix is dead (the system-prompt layers can stay cached) | yes | `PreCompact` / `PostCompact` hooks; header `x-claude-code-context-compacted` |
| The session ended | only on an explicit exit; a terminal left open is not an end | `SessionEnd` hook (reason `clear`, `resume`, `logout`, `prompt_input_exit` or `other`) |
| How long the last tools took | after the fact | `PostToolUse` `duration_ms`; header `x-claude-code-prev-tool-durations` |
| The user started typing | yes; the lead is unmeasured | only through a mod (`prompt.edit`); no hook |
| Which session a request belongs to | yes | header `x-claude-code-session-id`, on every request |

**The four exceptions** (`design/client_hint/harness_web/cc/hooks.md:2079, 2089, 2724-2728`):
- A sandboxed command's network-request prompt fires no `PermissionRequest`, only the ~6-s notification.
- In a session that cannot prompt, `PermissionRequest` fires for a call that is then denied automatically, with no
  human involved.
- `Stop` does not run after a user interrupt.
- An API error fires `StopFailure` instead of `Stop`.

**The harness knows an exact time only for waits it schedules or bounds itself:**
- wake-ups and crons (`ScheduleWakeup`: 1 min to 1 h);
- retry delays and usage-limit resets;
- idle compaction, after ~54 min;
- foreground tool timeouts, which are upper bounds. Bash defaults to 120 s with a 600-s maximum. At the timeout the
  command moves to the background instead of being killed, unless it starts with `sleep` or background tasks are
  disabled.

**Its own behaviours run on fixed timers, not on predictions:** 6 s for the permission notification, 60 s for
`idle_prompt`, 3 min for the session recap, ~54 min for idle compaction [harness §1].

**What it already sends to the server is facts** [harness §2.5]:
- **The cache TTL class.** By default it is 1 h on a subscription within plan usage, for the main conversation and a
  few helper requests Anthropic controls, and 5 min otherwise. Settings and environment variables can set either
  bucket to 5 min or 1 h. Billing and the request type choose it, not a predicted gap.
- **The gateway hint headers (v2.1.273+; the prompt id needs v2.1.283+).** They carry request class, agent type,
  compaction, "context compacted", previous tool durations and a prompt id. They are sent by default to the Anthropic
  API, but to a custom base URL only with `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`.
- **Neither describes the coming gap.** The TTL class goes with request k and covers the gap after reply k, but it is
  not chosen by the gap. The hint headers that report on a gap (tool durations, "context compacted") ride on request
  k+1, after it. A hint about the coming gap needs a message sent after the reply.

### 3.2 Other harnesses

- **Codex** sends a session-derived `prompt_cache_key`, routing headers and a WebSocket pre-warm (`generate=false`). It
  sends no retention field [harness §3.5].
- **OpenHands V1** sends `prompt_cache_retention: "24h"` by default, but only to GPT-5+ and GPT-4.1 models. Pointed at
  an open model on SGLang, it sends none [harness §4.3].
- **None of them tells the server "next call in T seconds"** [harness §1].
- **OpenHands V0 (CodeAct) produced the trace's 665 SWE-bench sessions.** The trace's 94 GAIA sessions were run with
  Inspect AI [harness §4.1; CN cap_midrun]. V0's rules explain the SWE-bench timing. It polls every 0.5 s and returns control after 10 s without
  new output [harness §1]:
  - 94.8 % of SWE-bench `execute_bash` gaps in [0.6, 10) s lie within 0.08 s of the grid 0.70 + 0.5·k s;
  - 54 of the 207 gaps of at least 10 s lie in [10, 11) s.

  V0's code does not explain the trace's spike of gaps near 300 s [harness §9].

### 3.3 What a server can never infer on its own

From [harness §7], inferred from what reaches the server:
- whether a human must act on a tool call, which depends on the client's permission mode;
- whether the user is present, focused or typing;
- background tasks and crons pending at the end of a turn, and whether a scheduled wake-up actually fires;
- the client backing off after network errors the server never saw;
- whether the user abandoned the session or exited it;
- an upcoming compaction or model switch;
- harness caps on iterations or budget;
- side requests the harness generates.

### 3.4 What real use looks like: TraceLab

TraceLab is a public dataset of the authors' own Claude Code and Codex use at one lab [tracelab §0, §1, §3].

**The data:**
- The Hugging Face version is v0.0.2 (2026-07-24): 8,058 sessions, 665,453 model calls, 743,819 tool calls, 2,688,829
  timing events and 52 users. It is 77.2 MB of Parquet, under CC BY 4.0. A contributor's issue splits the sessions
  into Claude Code 5,319 and Codex 2,739; that split was not verified.
- The paper (arXiv 2606.30560v2) describes 4,265 sessions from 43 developers. Its counts equal release v0.0.1's, so it
  used v0.0.1 (inferred).

**What real use looks like** (the paper's figures on v0.0.1; the repository's refresh on v0.0.2 in brackets, taken
from its README, not recomputed):
- Human waiting is 92.3 % of session wall-clock, or 64.3 % with idle gaps capped at 1 h [v0.0.2: 91.6 %].
- Human-input waits (Table 7) have a median of 1.4 min, a p90 of 20.6 min, a mean of 46.7 min and a p99 of 13.9 h
  [v0.0.2: p50 1.4 min, p90 15.2 min, p99 11.2 h]. The metric counts every positive gap before any user message,
  including system-written ones.
- Tool calls longer than 1 min are 4 % of calls but 85 % of tool time [v0.0.2: 3.4 % and 83.6 %].
- 9.7 % of sessions contain a compaction [v0.0.2: Claude 4.4 %, Codex 17.4 %].
- Contexts are long: Claude prefixes have a p50 of 126K tokens, a p90 of 467K and a p99 of 918K; the Codex p99 is
  231K. Our replay is cut to 262K.

**What it records and what it can calibrate:**
- It has six event types: tool call, tool result, reasoning, usage report, text and user message.
- It has no events for approvals, permission mode, compaction, session start or end, links to subagents, background
  completion or typing. Start, end and compaction can only be reconstructed, from the first and last rounds and from
  drops in token counts. Tool arguments were removed.
- It can calibrate:
  - foreground tool waits, pessimistically, because the arguments are missing;
  - timer and polling waits;
  - human waits inside a tool (AskUserQuestion, ExitPlanMode);
  - human waits at the end of a turn, once non-human user messages are filtered out.
- It cannot calibrate permission waits or the lead time that typing gives. It calibrates subagent and background waits
  only partly: from the parent's side of blocking calls, because child sessions are not linked.

## 4. Where "the agent knows its own workload" stops

### 4.1 Durations come from history, and the server pools more of it

How well each learner predicts the gap on our trace:
- R² is for the log of the gap.
- AUC measures how well the hint separates gaps longer than 30 s (real time) from shorter ones.
- The last two columns are for x70 gaps longer than 300 s (n = 1,014). The ratio is uncapped. The expiry column
  applies the replay's 1,200-s cap.
- R² ranges run from all sessions to sessions after the first 100.
- Sources: [CN repo_online], [CN far_errors], [critic §4], [trace §0.5], [factcheck D7]; the S5 AUC was recomputed
  from `trace/realistic_hints.csv` on 2026-10-28.

| learns from | causal | R² | AUC > 30 s | median truth / hint | expired under today's rule (§4.4) |
|---|---|---|---|---|---|
| this session's own past (O3) | yes | 0.42 | 0.72 | 7.56 | 83.1 % |
| the same repository and model only | yes | 0.70-0.74 | 0.70 | 3.43 | 68.3 % |
| a table pooled over all earlier sessions (O4) | yes | 0.76-0.77 | 0.70 | 3.93 | 87.2 % |
| the same repository, falling back to the pooled table | yes | 0.77-0.78 | 0.71 | 2.94 | 65.6 % |
| a gradient-boosted model fit on 4/5 of all sessions (S5 point hint) | no (out-of-fold) | 0.82 | 0.74 | 2.94 | 60.3 % |

- "Causal" means the learner sees only earlier turns or sessions, as a running client would.
- By default one harness instance sees only its own user's history, while the server sees every session it serves. A
  harness vendor or a team could pool history too, through a shipped table or a shared log (inferred).
- The out-of-fold S5 point hint ranks long gaps only a little better than the causal learners: 0.74 vs 0.70-0.72. A
  separate out-of-fold S5 classifier in the same file (`p_gt_30.000`) reaches 0.83.

### 4.2 Errors on the long gaps that matter are mostly one-sided

- **x70, gaps over 300 s: realistic hints are too short** [CN far_errors]. The uncapped median truth/hint ratio is
  2.94 (S5), 3.93 (O4) and 7.56 (O3). 41 %, 49 % and 69 % of the hints are more than 4× too short. With truth and
  hint capped at 1,200 s, these become 2.70, 3.64 and 7.26, and 37 %, 48 % and 69 %.
- **x70, gaps of 60-300 s: hints are nearly right.** S5's median ratio is 1.15. SWE-bench bash gaps in this range sit
  on the harness's 0.70 + 0.5·k s polling grid [harness §1], which makes them predictable (inferred).
- **x10, gaps over 300 s (n = 159): every learner misses.** The median ratio is 86-98× with truth and hint capped at
  1,200 s (101-185× uncapped), and still 10-17× without the 300-s spike.
- **Random noise around the oracle hides this.** In the simulation (x70, 959K room), lognormal noise with σ = 0.5 keeps
  86-88 % of the exact hint's gain. A realistic per-tool median hint keeps 50-54 % [sim §0.2].
- **Human waits are worse.** The harness knows that a human is involved, not whether the wait will last one minute or
  14 hours (TraceLab p99 13.9 h, §3.4).

### 4.3 What only the engine knows

- **Its queue and admission.** Even exact gaps left an admission error of −7 to +8 s (p10 to p90) at x10
  [evidence §0.3]. Eviction and staging act on the next *admission*, not on the next send.
- **Statistics pooled over every client** (§4.1), and **its own cache and SSD state.**

So "system status" splits in two:
- The agent's own status is exact and worth sending: back-off, background jobs, wake-ups.
- The server's status (queue, cache, SSD load) only the engine sees.

### 4.4 Today's engine punishes imperfect hints

**Expiry ends the session.**
- A plan expires at the predicted arrival + max(60 s, gap) (`mem_cache/next_turn_prefetch.py:297`, checked at
  `:615-616`).
- `_expire` then aborts any staged read and calls `_end_session` (`:821-835`; the call is at `:831`). The session
  becomes ENDED, i.e. evicted first.
- The late turn finds no plan, so it gets no queued refetch and no deadline [critic O3].

**How often realistic hints hit that rule.** This applies the rule to the gaps alone, with no queue wait and no arrival
offset [replay §5.1]. The exact oracle expired 0 plans in c18 and c19.

| hint sent as the point value | x70, gaps 60-300 s | x70, gaps > 300 s | x10, gaps > 300 s |
|---|---|---|---|
| S5 | 9.1 % | 60.3 % | 97.5 % |
| O4 pooled table | 14.2 % | 87.2 % | 100 % |
| quantile model's p90 | 2.0 % | 28.1 % | 78.0 % |

**Staged reads are wasted** [CN stage_hold]. Staging means reading a session's KV from SSD back into host memory before
its turn returns.
- For x70 gaps over 300 s, S5 hints waste 61 % of staged reads and hold 1.79× the token-seconds of host memory that
  exact hints hold. O4 hints waste 87 % at 1.75×.
- This is a rule-based upper bound, not a cache simulation.

**A missing hint means ENDED.**
- No hint, a malformed one, or a v2-only payload: all of them register the turn ENDED (`:265-277`; version check at
  `:96-101`, `:107`).
- So today "no hints" and "end only" are the same arm.
- With 50 % of hints missing, the ENDED rule does worse than LRU: −49 % to −64 % of the gain at 959K [sim §0.4].

**The simulation has no plan expiry.** Its kept shares (§4.2) therefore compare amounts of information under the
simulation's own rules, not today's engine. Today's engine would probably do somewhat worse; the fact check expects a
modest loss, which is unmeasured (inferred) [factcheck].

## 5. The revised position: the agent declares, the engine forecasts

### 5.1 What the agent sends (a sketch; inferred design)

After each reply, keyed by session [critic §8; harness §9], the agent sends:
- **`wait`, the kind of wait:**
  - `tool` (name, signature, timeout, background or not);
  - `user`;
  - `approval`;
  - `retry` (timer);
  - `background` (job);
  - `scheduled` (time);
  - `compaction` (the prefix is dead);
  - `ended`.
- **Optionally, its own `estimate`**, as quantiles.
- **Later, `resume` events with their lead:** the user is typing, an approval was granted, a job is done, a wake-up is
  due.

The same vocabulary fits Claude Code, Codex and OpenHands: TOOL(bound?), HUMAN(approval | question | next prompt),
BACKGROUND(n), SCHEDULED(t), RETRY(t), INVALIDATED, ENDED.

### 5.2 What the engine does

- **It keeps a return forecast for each idle session:**
  - survival curves per (wait kind, signature, tenant or repository);
  - learned online from all sessions;
  - conditioned on how long the session has been idle and on the engine's own queue.
- **The agent's estimate enters as a prior,** weighted by how well that agent's estimates have matched outcomes so far
  (log score or PIT).
- **Safeguards.** Decisions keep an LRU fallback (a "ghost" cache) and per-tenant budgets. Today any client can send
  hints, and the "trusted orchestrator" rule exists only as a comment (`managers/io_struct.py:362-367`).
- **Optionally, the engine tells the client** where the KV sits and what a restore costs. The client can then time a
  warm-up ping or pick a retention class (inferred).

### 5.3 The research claim, restated

"Which part of the KV-tiering gain needs information only the client has, and what is the smallest protocol that
captures it?" [critic §8]

It is measured with the arms of §6.4:
- engine only (E1);
- \+ client facts (E2);
- \+ client estimate (E3 = E2 + estimate);
- E2 + ping (the ping is added to E2, not to E3);
- the oracle.

Each runs on a tool-paced and a human-paced workload, with causal learners, partial adoption and a client that
overclaims.

The novelty claim must be hedged [critic O8; factcheck D11]:
- Harness timing hints already exist in research and in upstream plans: Tokencake's developer-supplied `predict_time`,
  Dynamo's plans, SGLang #24656 and vLLM #57103.
- KVFlow's client sends an execution order, not a time.
- Continuum predicts on the serving side.
- A27 simulated imperfect prediction with an SSD tier.

## 6. Making the trace-driven evaluation realistic

### 6.1 Model what the agent would know, not noise around the oracle

**Compute hints inside the replay client.**
- Use a learner that sees only finished turns.
- Give it three scopes: the session's own past, the repository or user, and a pooled table.
- Log each hint next to the truth.

**Do not use `trace/realistic_hints.csv` as it is:**
- its S5, quantile-model, per-signature, `p_gt` and `p_end` columns are out-of-fold, i.e. fitted partly on later
  sessions;
- its `pred_online_s` (O4) column is learned online, but in its own random session order, not the replay's;
- its `p_end` uses the turn index and agent step, which encode the 50-call cap;
- `pred_s5_s` and `qgbt_q50_s` are the same predictor;
- its quantiles cross on 1,598 rows [replay §2.3].

**Piggybacking on request k is acceptable**, under two conditions [factcheck D5]:
- the hint uses only what the agent knows after reply k;
- it links turns the way a harness could. Today plans are keyed by `next_rid`, a request id the replay client
  chooses.

**How it compares with a real message sent after the reply.**
- The engine reads the hint at commit. A real message would arrive somewhat later. On the comparable path in c18 /
  c19 the delay was 82 / 94 ms at the median and 0.97 / 1.10 s at p99 [replay §7.2].
- Piggybacking cannot emulate updates during the wait: a human is needed, a ping, a retry, a late end. Those need a
  real side channel.

**The exact end** is realistic for batch agents, which have a finish call and their own cap. For interactive sessions
it is a probability. Both are inferred, and that this trace's final replies are finish calls is unverified
[critic §9].

### 6.2 Fix the engine first

These are the minimal changes, in this order [replay §5.2-§5.3]. They sit in `mem_cache/next_turn_prefetch.py`,
`mem_cache/next_use_eviction.py` and `arg_groups/kv_cache_hook.py`, and none touches a frozen class.
- **(c) Expiry never ends the session,** and the late turn can still be matched.
- **(a) + (b) A missing hint means UNKNOWN (plain LRU), and the client sends `ended` explicitly.** Without both, "no
  hints" and "end only" stay one arm.
- **(e) Queued turns never go overdue.** This already fired for 67 turns in c19, with exact hints.
- **(d) A quantile payload.** Stage from the early quantile, and compute overdue and expiry from the late one.

**Size:** about 110-160 engine lines and 150-200 test lines (estimate).

**From round 1** [FP §6-§7], not sized:
- staging only when a return is likely;
- a cap on pinned token-seconds;
- refetch for queued turns even without a plan;
- plans keyed by session.

**A post-reply hint channel** comes on top. It needs an IO struct, an HTTP route, a TokenizerManager forward, a
dispatcher entry, and commit records in the prefetcher. The scheduler's dispatcher has no hint message today
(`managers/scheduler.py:1729-1836`).

The engine forecaster of §5.2 is a separate piece of work (§8.4). Then re-measure the oracle on the fixed engine; it
becomes the new upper bound.

### 6.3 Add a human-paced workload

**Why the lmcache trace is not enough:**
- It has no humans.
- x70 manufactures hint value. 47 % of x70 returns fall at 45-60 s, where LRU evicts. Most of them are
  `execute_bash`'s ~0.72-s gaps × 70 [sim §1], i.e. OpenHands' polling grid × 70 [trace §0.1; harness §1].
- No learner predicts x10's long gaps (§4.2).
- At real speed the host and SSD tiers are rarely reached: only ~0.35 % of real gaps reach 60 s [critic O5].

**Build a TraceLab-based workload with our own converter** [tracelab §4.2]. TraceLab's own export drops the human waits.
The converter has to:
- group sessions and drop copies left by resumed sessions;
- label each gap by kind;
- synthesize the new tokens of each turn;
- fit the 262K context window;
- choose a policy for multi-hour gaps;
- overlay user-days to load one server, since 52 users over ten months are far too few.

**Keep lmcache x70** as a stress test for tool waits, with controls for the 1,200-s cap, the 300-s spike and the
50-call cap. Keep x1 (real speed) as the honest negative.

### 6.4 Compare on one fixed engine

[critic §5]

| arm | client sends | engine does |
|---|---|---|
| E0 | session id only | forecasts from idle time alone |
| E1 | session id | a per-(tool, signature) forecast from its own parse of the reply. Needs a reply shim, because the replay sets `ignore_eos` and sends no `tools` |
| E2 = E1 + client facts | wait kind, `ended`, compaction | E1 with curves per wait kind |
| E3 = E2 + client estimate | as E2, plus quantiles | uses the estimate as a prior, weighted by its record so far |
| E2 + ping | as E2, plus a resume event with its lead | stages on the ping |
| oracle | the exact gap | the same path, with an exact forecast |

E0 serves as the floor (inferred).

**Each arm is crossed with:**
- **Learning scope:** the session's own past, the repository or user, or a pooled table.
- **Workload:** lmcache x70 with its controls, TraceLab human-paced, or x1.
- **Room:** the simulation's four host rooms (318K, 578K, 959K, 1,250K tokens), plus "+291K tokens of host room" and
  mirror reclaim.
- **Adoption and trust:** 100 %, 70 % or 30 % of clients hinting; one client that always claims "1 s"; one biased
  ×0.5.

**Key contrasts:**
- **E2 − E1:** are the client's facts worth more than the run-to-run spread?
- **E3 − E2:** does a client estimate add anything once the facts are sent?
- **(E2 + ping) − E2:** does the lead from a resume event hide SSD reads?
- **Each against "+room":** is it worth more than adding host memory?

### 6.5 A live demo later

**What should work today.** This comes from the docs and the code; it was not tested end to end
[harness §8.1, §10].
- Claude Code can be pointed at this SGLang with `ANTHROPIC_BASE_URL`.
- SGLang serves `/v1/messages` and `/v1/messages/count_tokens` (`entrypoints/http_server.py:2030-2047`).

**What is missing:**
- The Anthropic adapter builds the chat request without `kv_hints` or a session id
  (`entrypoints/anthropic/serving.py:564-588`).
- There is no way to send a hint after the reply.

**Smallest path** (inferred, not started):
- set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`;
- map `x-claude-code-session-id` to the SGLang session;
- run a sidecar that receives HTTP hook POSTs (`PreToolUse`, `PermissionRequest`, `Stop`, `SessionEnd`, `PreCompact`);
- add a hint endpoint to SGLang.

**A self-hosted run likely exercises fewer harness timers than subscription use** (inferred) [harness §8.1]:
- With a gateway token and no API key, Claude Code fetches no feature flags, so idle compaction is off.
- Separately, the main conversation gets the 5-min TTL class unless `promptCacheTtl=1h` is set.

## 7. Decisions for the user

**From round 2** (asked 2026-10-28 01:51):
1. **The framing.** Adopt "the agent declares, the engine forecasts", with the agent's own estimate as one arm (E3)?
   - The alternative is "the agent estimates and the engine obeys". That is E3 without E1 or E2, so it cannot show
     whether the value comes from the client's estimate or from the client's facts.
   - Recommendation: adopt.
2. **The TraceLab workload.** Add the human-paced workload?
   - Cost: a 77.2-MB download (CC BY 4.0), ~1 CPU-day of analysis and 0.5-1 day for a converter (estimate).
   - Without it, the most valuable client-only signals cannot be tested: human waits, approvals, background work,
     wake-ups. On lmcache only the exact end and the harness's own timeout bounds can be emulated [harness §9].
   - Recommendation: yes.
3. **The order.** CPU simulation, then engine work, then GPU pairs?
   - GPU pairs pay off only once the run-to-run spread is measured (§8.6).
   - Recommendation: yes.

**Still open from round 1** (asked 2026-10-28 00:02). Its question 3, "engine only, or also a small client protocol?",
is answered by the user's direction.

4. **Setting and metric.**
   - Setting: x70-style serving with many idle sessions, or real pacing with human pauses.
   - Metric: mean time to first token, or sessions per GPU under a latency target. The latter needs the load sweep
     (§8.7).
   - [FP §10] also asks which model and host size.
5. **Replay client.** May it change? Round 1 asked about `session_id`, `/close_session` and the recorded tool type;
   [FP §10] adds a cap option.
   - This plan adds `ended`, a client-hint mode and the reply shim (§8.4, §8.5).
   - It breaks comparability with c18/c19, except for an oracle mode that keeps request bodies byte-identical
     [replay §6.7].
6. **Upstreaming.** Is it a goal? If so, these matter now:
   - TP > 1, which the prefetcher refuses today;
   - several replicas;
   - authentication;
   - [FP §10] adds the Rust HTTP server, which drops `kv_hints` and `session_id`.
7. **The hint-free rival.** Are reclaiming host memory or switching to write-back in scope? In the model, 291K more
   tokens of host room (~28.6 GB, derived from 96 KiB per token [OF §2]) beat every realistic signal.

**New for this plan:**

8. **GPU budget.** Are ~110-170 GPU-h acceptable for one workload, with each further workload adding 60-75 (§0, §8)
   and ≥ 3 replicates per decisive arm? [FP §10] asked this for round 1's ladder, at ~75-115 GPU-h.

## 8. Proposed plan, if accepted

Each step follows the runbook template. No commands are given yet: most steps need code or scripts that do not exist
yet, though §8.6's x10 check needs no new code [FP §8 step 3]. Each step gets its commands when it is approved. Do not
edit `agent_cache/scripts/` while a campaign runs.

### 8.1 CPU simulation of the arms

**Status:** not started; waits for decisions 1 and 3 (§7).
**Goal:** rank the arms of §6.4 offline, so GPU time goes only to pairs whose difference can exceed the run-to-run
spread.
**Runs on:** host CPU, with the analysis venv `agent_cache/frozen/hint_realism_20261027/venv/` and `python -I`.
**Touches:** new files under `agent_cache/frozen/hint_realism_20261027/` (the simulation extension and its outputs).
Everything else is read-only.
**Takes:** ~1 CPU-day (estimate, from [FP §8] step 1); GPU idle.
**Expected result:**
- For each arm, room (318K, 578K, 959K and 1,250K, plus ~500K and ~1.9M) and timeline (c18 x70, c19 x10): the share
  of returns served from SSD, and the share of the exact hint's gain kept.
- Both under today's rules, with plan expiry added to the simulation (it has none), and under the fixed rules of §6.2.
- Client learners that see only the past, in three scopes.
- Exact hints sent through the new forecast type reproduce the existing exact-hint result (P1) within 1 point.

**If it differs:** if P1 is not reproduced, the extension is wrong. Fix it before ranking anything.
**Expected lessons:**
- Whether the client's facts (E2) beat the engine alone (E1), and whether the client's estimate (E3) adds anything on
  top.
- What each signal is worth in tokens of host memory.
- Whether the controls (cap, spike, 50-call cap) change the ranking.
- Round 1's thresholds [FP §8 step 1]:
  - the engine forecaster must beat P7 by ≥ 5 points on both timelines at every room, and never fall below LRU, in
    every variant of the simulation model;
  - a client form must beat the forecaster by ≥ 5 points;
  - a best signal worth < 334K tokens means mirror reclaim or write-back comes first.
- The lmcache timelines cannot test human waits. Repeat on the TraceLab-paced timeline once §8.2 exists.

### 8.2 TraceLab human-paced workload

**Status:** not started; waits for decision 2 (§7).
**Goal:** a workload with human waits at the end of turns and inside tools, so those client signals can be tested,
plus error models for the kinds of wait TraceLab supports. Permission waits, the typing lead and background waits stay
untestable [tracelab §0].
**Runs on:** host. Download from Hugging Face `UW-SyFI/TraceLab`, tag v0.0.2, three Parquet files.
**Touches:**
- a new, empty directory for the untrusted data, e.g. `agent_cache/frozen/hint_realism_20261027/data_tracelab/`;
  nothing in it is executed;
- analysis scripts in a separate directory;
- one new converted trace under `agent_cache/traces/`; existing traces are not touched.

**Takes:** a 77.2-MB download, ~1 CPU-day of analysis and 0.5-1 day for the converter (estimates [tracelab §0]); GPU
idle.
**Expected result:**
- The SHA-256 sums equal the dataset card's.
- A gap distribution for each kind of wait:
  - tool, by executable;
  - timer or polling;
  - a human inside a tool;
  - a human at the end of a turn, after filtering out system-written user messages and Codex auto-review requests.
- A converted trace, with user-days overlaid to load one server.

**If it differs:** if human waits carry < 10 % of the KV-token-seconds spent in long gaps, drop the human-wait and
ping channel ([FP §8] step 5).
**Expected lessons:**
- How much of the long-gap KV time in real use follows human waits rather than tools.
- Whether compaction and subagents are common enough that sessions must be modelled as trees.

### 8.3 Engine fixes

**Status:** not started; waits for decisions 1 and 3 (§7).
**Goal:** an engine whose rules do not punish imperfect hints, so the arms measure information, not the expiry rule.
**Runs on:** host (code, plus unit tests on CPU).
**Touches:**
- `python/sglang/srt/mem_cache/next_turn_prefetch.py`, `mem_cache/next_use_eviction.py`,
  `arg_groups/kv_cache_hook.py` and their tests;
- for the post-reply channel, also `managers/io_struct.py`, `entrypoints/http_server.py`,
  `managers/tokenizer_manager.py` and `managers/scheduler.py`. Edits to `Scheduler` and `TokenizerManager` follow the
  `large-class-style` skill.

**Takes:** ~110-160 engine lines and 150-200 test lines for (a)-(e) (estimate). Round 1's other fixes and the channel
come on top and are not sized.
**Expected result:** unit tests pass for these cases, plus a switch that restores c18/c19's rules:
- a turn that returns after its plan expired is still matched and is not ENDED;
- a commit without a hint registers nothing (UNKNOWN);
- `ended` registers ENDED;
- a queued turn never becomes OVERDUE;
- a quantile payload is validated (finite, sorted, capped).

**If it differs:** none.
**Expected lessons:**
- The common floor for every arm of §6.4.
- Whether the oracle still shows its c18/c19 gain on the fixed engine (§8.7, §8.8).

### 8.4 Engine forecaster and reply shim

**Status:** not started; waits for decisions 1 and 3, and for question 5 (the shim changes the replay client) (§7).
**Goal:** the engine side of "the agent declares, the engine forecasts": the return forecast of §5.2, which arms E1-E3
use.
**Runs on:** host (code, plus tests on CPU). Its GPU runs are part of §8.8.
**Touches:**
- new code inside the prefetcher's path (`mem_cache/next_turn_prefetch.py`; placement as in [FP §6]);
- for the shim, `agent_cache/scripts/replay_agentic.py`. The shim sends the recorded tool class with the commit-time
  hint, because the replay's generated replies are not the recorded tool calls.

**Takes:** ~1.1-1.5K lines (estimate [FP §8 step 8]). FP also budgets ~10 GPU-h for a shadow run and 2 seeds; this
plan folds those into §8.8.
**Expected result:**
- In shadow mode, each forecast is logged next to the actual return.
- Exact hints sent through it reproduce the oracle's decisions.

**If it differs:** round 1's stop rule applies: "kill if ghost misses ≥ LRU's or within spread" [FP §8 step 8].
**Expected lessons:** whether per-(tool, signature) curves learned online from all sessions beat idle time alone on
real hardware.

### 8.5 Causal client-hint mode in the replay client

**Status:** not started; waits for decision 1 and question 5 (§7).
**Goal:** for each turn k, send the hint a harness could compute after reply k.
**Runs on:** host (code). The client runs as usual during the GPU steps.
**Touches:** `agent_cache/scripts/replay_agentic.py`: new flags, e.g. a client-hint mode and `ended` on the last turn.
The oracle mode stays byte-identical.
**Takes:** not sized. Replay §5.2's ~60-100 lines size a mode that reads `trace/realistic_hints.csv`. An online learner
also needs per-turn reply features (tool, command signature, repository), which the client does not parse today. The
trace's own feature extraction, `trace/features.py`, is 299 lines.
**Expected result:**
- Every turn record carries the hint sent, next to the truth.
- In oracle mode, request bodies equal c18/c19's.

**If it differs:** none.
**Expected lessons:** the learner's online error for each kind of wait, which is the realistic error model the
evaluation needs.

### 8.6 Round 1's GPU checks

**Status:** not started; waits for decision 3 (§7).
**Goal:** settle three questions before any hint arm runs on GPU, and measure the run-to-run spread.
**Runs on:** the H200, with the campaign 18/19 setup and the cookbook divergences listed in [FP §8].
**Touches:** new result directories under `agent_cache/results/`.
**Takes:** estimates from [FP §8] steps 2-4, ~31-32 GPU-h in all; GPU busy:
- ~1-2 GPU-h for restored-KV correctness;
- ~15 GPU-h for whether the x10 win needs hints (no new code);
- ~15 GPU-h for x70 eviction headroom without staging, including the hint-free floor and a same-seed replicate of the
  c18 baseline.

**Expected result:**
- KV restored from SSD matches recomputed KV.
- `three_tier_wc_oracle_v2` with `ORACLE_HINTS=0` either avoids the recompute storm or does not.
- An exact-hint eviction arm without staging lands at a measured SSD-served share.
- Same-seed replicates give the run-to-run spread.

**If it differs:** any mismatch in restored KV stops everything.
**Expected lessons:**
- If the x10 arm avoids the storm without hints, x10 stops being evidence about hints: queue protection carried the
  win.
- The spread sets how many replicates §8.8 needs.

### 8.7 Load point

**Status:** not started; waits for §8.3 and §8.6.
**Goal:** find a load at which the gap between the floor and the oracle on the fixed engine exceeds 2× the run-to-run
spread, so the contrasts of §8.8 can be resolved.
**Runs on:** the H200.
**Touches:** new result directories under `agent_cache/results/`.
**Takes:** 20-60 GPU-h (estimate [FP §8 step 6]); GPU busy.
**Expected result:**
- The floor and the oracle at client concurrency C from 128 to 224.
- For each: users finished, admission wait, p99 TTFT, session time and decode speed.
- A knee where the floor starts to fall behind.

**If it differs:** if floor → oracle stays below 2× the spread at every load, stop hint-driven eviction work
[FP §8 step 6].
**Expected lessons:** at which load hints can matter at all on this machine. C = 256 likely exceeds the ~2.5 turns/s
this machine completes (inferred [FP §8]).

### 8.8 GPU decisive pairs

**Status:** not started; waits for §8.1, §8.3, §8.4, §8.5 and §8.7.
**Goal:** run on GPU only the contrasts the simulation found decisive: the floor (E0), E1, E2 with and without the
ping, and the oracle.
**Runs on:** the H200, at the load point from §8.7.
**Touches:** new result directories under `agent_cache/results/`.
**Takes:** at least 3 replicates per arm; GPU busy.
- A run lasts up to ~5 h (the client's 18,000-s timeout).
- 4-5 arms × 3 replicates is 60-75 GPU-h per workload (derived).
- The floor and oracle runs of §8.7 at the chosen load can count toward this.

**Expected result:** each contrast of §6.4 measured with a difference larger than 2× the spread from §8.6, or shown to
be within it.
**If it differs:** if E2 − E1 is within 2× the spread on every workload, the client's facts do not pay at this load;
report that as the result.
**Expected lessons:** whether client facts pay off on real hardware, and what each is worth in TTFT and in sessions
finished.

### 8.9 Live demo with Claude Code (later)

**Status:** not started; not proposed until §8.8 shows a gain.
**Goal:** show that a real harness can drive the engine's KV management through the protocol of §5.1.
**Runs on:** SGLang on the H200; Claude Code on a client machine.
**Touches:**
- the SGLang Anthropic adapter and a new hint endpoint;
- a new sidecar;
- Claude Code settings: `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`, HTTP hooks.

**Takes:** not sized.
**Expected result:** hints with the wait kinds of §5.1 reach the prefetcher, keyed by `x-claude-code-session-id`.
**If it differs:** none.
**Expected lessons:** the real lead time of resume events, typing and approval clicks, which no dataset records.

## 9. Traps

Each trap names the step it affects.
- **The simulation has no plan expiry.** Its kept shares probably flatter imperfect hints under today's engine
  (inferred; the loss is expected to be modest and is unmeasured) (§8.1).
- **Two different S5 AUCs are easy to mix up.** For gaps over 30 s, the S5 point hint scores 0.74; a separate S5
  classifier scores 0.83. Compare point hints with point hints (§8.1).
- **Most columns of `trace/realistic_hints.csv` are out-of-fold, not causal.** Only `pred_online_s` (O4) is learned
  online, and in its own random session order. `p_end` encodes the 50-call cap (§8.5).
- **`pred_s5_s` equals `qgbt_q50_s`.** They are one arm, not two (§8.1).
- **The quantile columns cross on 1,598 rows, and `p_gt` is not monotone** on 3,982 / 4,747 rows. Sort them before
  sending (§8.5).
- **"No hints" equals "end only" until fixes (a) + (b).** Today a missing hint, a malformed one and a v2-only payload
  all mean ENDED (§8.3).
- **x70 manufactures hint value.** It puts 47 % of returns at 45-60 s, where LRU evicts (§8.1, §8.8).
- **The 1,200-s cap distorts the gaps.** It removes 49.6 % of x70 idle time and plants a certain return at 1,200 s,
  which forecasts from idle time and the oracle both exploit (§8.1).
- **The 300-s spike cannot be foreseen by any client.** It is 73 gaps: 0.30 % of gaps but 80.5 % of the
  KV-token-seconds in x10 gaps over 300 s. Run it as a control (§8.1).
- **228 of 768 sessions end at the 50-call cap.** Whether that is a harness cap or a dataset window is unresolved, so
  run both readings (§8.1).
- **Run-to-run spread at ~1 s TTFT is unmeasured.** One run per arm cannot rank arms (§8.6, §8.8).
- **TraceLab's human-wait statistics include system-written user messages.** A contributor's analysis, not reproduced
  here, finds Codex auto-review requests are 38 % of user-initiated Codex steps in v0.0.2 [#33] (§8.2).
- **TraceLab v0.0.2 has reported data defects:** a Codex round-boundary bug in 43 % of eligible rounds, and copied rows
  from resumed sessions. Both were reported by contributors and not reproduced here [#32, #33] (§8.2).
- **TraceLab contexts exceed our 262K window** (Claude p90 467K) (§8.2).
- **Every result is TP 1.** The prefetcher refuses TP or PP ≠ 1, DP attention and PD
  (`mem_cache/next_turn_prefetch.py:234-241`) (§8.6).

## 10. Sources

Paths are relative to `agent_cache/frozen/hint_realism_20261027/` unless they start with `agent_cache/` or `python/`.

| key | file | what |
|---|---|---|
| [FP §n] | `design/final_position.md` | round 1's verified position |
| [PV2] | `design/client_hint/position_v2.md` | round 2's position, short form |
| [harness §n] | `design/client_hint/harness.md` | what Claude Code, Codex and OpenHands expose; copies of their docs in `harness_web/` (untrusted, never executed) |
| [tracelab §n] | `design/client_hint/tracelab.md` | TraceLab: releases, schema, numbers, feasibility; copies in `tracelab_evidence/` |
| [#n] | https://github.com/uw-syfi/TraceLab/issues/n | TraceLab issues, as cited in `tracelab.md` |
| [replay §n] | `design/client_hint/replay.md` | today's oracle path, the CSV join, what the engine accepts, minimal engine changes |
| [critic §n], [critic On] | `design/client_hint/critic.md` | skeptical critique of the round-2 draft (objections O1-O14) |
| [CN name] | `design/client_hint/critic_numbers/<name>.out` | the critique's computations; scripts next to them |
| [factcheck] | `design/client_hint/factcheck.md` | claim-by-claim check of the draft (`factcheck_recompute.*`) |
| [engine], [evidence], [trace], [sim], [lit] | `reports/<name>.md` | round 1's five investigations |
| [OF] | `agent_cache/ORACLE_FINDINGS.md` | campaigns 14-19 |
| none | `design/client_hint/draft_position_v1.md` | the round-2 draft (claims D1-D12) |
| none | `README.md` | index of the folder, and how it was rebuilt after the 2026-10-27 reboot |

## 11. Glossary

- **Oracle hint:** the exact next-turn gap from the trace, sent by the replay client.
- **x70, x10, x1:** the factor applied to the trace's recorded gaps. x1 is real time. Gaps are capped at 1,200 s.
- **Room:** the host-memory tokens available for idle sessions in the simulation.
- **Kept share:** in the simulation, the share of the exact next-use policy's gain over LRU that a policy keeps, on
  returns served from SSD, at a given room.
- **P1, P6, P7, P7m, P9:** simulation policies:
  - P1: exact next use;
  - P6: idle time only;
  - P7: per-tool forecast plus idle time, no end signal;
  - P7m: the client sends the pending tool's median gap, with the exact end;
  - P9: P7 plus the exact end.
- **S5:** gradient-boosted models fitted out-of-fold on reply-time features (`trace/realistic_hints.csv`): a point
  hint (`pred_s5_s`) and threshold classifiers (`p_gt_*`).
- **O3, O4:** online learners. O3 learns from the session's own past (same class); O4 is a running median per (source,
  tool, command class), learned from earlier sessions.
- **Causal / out-of-fold:** a causal learner sees only the past. An out-of-fold model is fitted on other sessions,
  including later ones.
- **ENDED, OVERDUE, FAR, UNKNOWN, SOON:** today's engine eviction tiers for idle sessions, evicted in that order
  (`mem_cache/next_use_eviction.py:25-29`). **QUEUED** is the tier proposed by fix (e) (§6.2); today's engine does not
  have it.
- **Expiry:** today, a plan expires at the predicted arrival + max(60 s, gap), and the session becomes ENDED.
- **Staging:** reading a session's KV from SSD into host memory before its turn returns.
- **Floor:** the engine with the hint-free fixes and no hint beyond session linkage (arm E0).
- **E0-E3:** the arms of §6.4.
