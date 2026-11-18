# Our evaluation setup: KV prefix-cache prefetch and retention for agentic sessions (SGLang HiCache, `agent_cache/`), as of 2026-11-17

**Conventions.**
- Paths are relative to `/home/wanhr/Documents/codes/sglang/agent_cache/`, except paths that start with `python/` (repo root) and absolute paths.
- `file:line` is the line or lines that state the fact. "Inference" marks my own reading. "Not stated" means no document I read says it.
- Two sources outside the repository are used, always flagged:
  - the user's memory notes, under `~/.claude/projects/-home-wanhr-Documents-codes-sglang/memory/`;
  - the only copy of the paper's design section I found, `/tmp/claude-1000/-home-wanhr-Documents-codes-sglang/f065b572-320d-4c3c-9365-ec9611418f7c/scratchpad/paper_build/design.tex`, cited as `design.tex (scratch)`. `agent_cache/paper/design.tex` does not exist and has no git history (see Q10).
- A second `design.tex` in `~/Downloads/dbsnxvpwfjfxdjdbctztffwfjvrzmzvp/sections/` is dated Oct 2025 and belongs to an unrelated paper, so it is not used.

**Terminology bridge (paper vs engineering docs).**

| Paper (`design.tex (scratch)`) | Engineering docs | Where |
|---|---|---|
| "prefix cache": a session's stored KV | "history" | design.tex (scratch):9-11 |
| $R$: host memory available to idle prefix caches and to reads in progress; $C_{host}-C_{GPU}$ under write-through | "the room" | design.tex (scratch):80-85; `PAPER_PLAN.md:41` |
| "price" $\lambda$, in s of TTFT per token-second | "rent" | design.tex (scratch):33-35, 160-163; `python/sglang/srt/arg_groups/fields/memory.py:295-302` |
| "staged read": lands before its request | same | design.tex (scratch):259-260 |
| "preemption" | "soft pins" | design.tex (scratch):256-271; `results/campaign26/DECISIONS.md:61` |
| "retention index" | rule 3 / `return_chance` | design.tex (scratch):301-314; `results/campaign26/DECISIONS.md:58-60` |

Other engineering-doc terms: "JIT" (just-in-time reads), "reading turns" (turns that read from SSD), "pins" (a landed read held until admission), "arms".

## 1. Evaluation mode: live agents, trace replay on a real engine, simulation, or synthetic?

### Takeaway
Every TTFT number comes from replaying recorded agent traces on a real engine: SGLang with HiCache on one H200, driven by a closed-loop client that sleeps the recorded tool gaps. No tool runs, and no live agent has been driven against the engine. CPU-only models set the policy parameters and rank the rules: `room_model.py`, `prefill_curve.py`, `toy_floor.py`, an eviction replay, a host-pool sweep and offline hint scoring. They replay or are calibrated on GPU-run logs, and they state that they cannot predict TTFT.

### Cited Findings
**Trace replay on a real engine (all GPU campaigns)**
- The client replays one recorded conversation per user as a closed loop: send turn 0, stream the reply, append it verbatim to the history, sleep the recorded `pre_gap` × `--gap-scale`, then send the history plus the next turn's new messages (tool results, user prompt) — `scripts/replay_agentic.py:7-10`.
- The server is one SGLang `launch_server` per arm, booted by `start_server.sh` — `scripts/start_server.sh:2`, `:209`. The box is the H200 — `RUNBOOK.md:3-4`. "Campaign measurements come from the H200" — `REALISTIC_HINTS.md:16`.
- **GPU-replay numbers the paper documents rely on:**
  - campaign 14 (×70, original HiCache): the Figure 1 failure and "a returning turn reads about 1.6 GB" — `PAPER_INTRO.md:139-141`;
  - campaign 16, the ×70 ablation — `PAPER_INTRO.md:84-86`;
  - campaigns 18-21, 23, 24 and 25 at 44 and 128 users — `PAPER_PLAN.md:15-19`, `CAMPAIGN23_FINDINGS.md:70-99`;
  - campaign 26 (the pricing / expected-cost arms; 21 runs) — `results/campaign26/DECISIONS.md:1211-1214`.
- **Where each paper contribution's evidence comes from:**
  - contributions 1 (the failure mode) and 2 (the system, "1.4-10x") are GPU-replay numbers — `PAPER_INTRO.md:73-92`;
  - contribution 3 (acting on wrong forecasts) was "Status: offline only" on 2026-11-06 — `PAPER_INTRO.md:107-110`;
  - it was then built and run on the GPU in campaign 26 (2026-11-11 to 11-14) — `results/campaign26/DECISIONS.md:47-65`, `:149-185`.

**CPU-only models**
- **Room model** (`scripts/room_model.py`):
  - What it does: it "replays one run's reads". It holds each turn's arrival and GPU queue wait, the pages each turn read from SSD ("so eviction is the run's"), and when the history went missing. It changes only when reads start and whether a landed read can be taken back — `results/cpu_calib_c44/FINDINGS.md:9-25`.
  - No GPU is used — `results/cpu_calib_c44/FINDINGS.md:3-4`.
  - Its one free parameter, the room, is fitted to 90K tokens on the two campaign 23 runs at 44 users — `results/cpu_calib_c44/FINDINGS.md:43-55`.
  - "TTFT is out of reach" — `results/cpu_calib_c44/FINDINGS.md:148-149`.
  - It chose λ = 5e-5 and the hint form — `results/cpu_calib_c44/FINDINGS.md:81-100`; `PAPER_PLAN.md:192-199`. The engine's 5e-5 "was fitted to 70 GB" — `results/campaign26/DECISIONS.md:130`.
- **`prefill_curve.py`**: refits the recomputation cost as quadratic in context, from a 2026-09-22 single-stream sweep, and re-prices the room model's replay — `scripts/prefill_curve.py:1-9`; `results/cpu_calib_c44/FINDINGS.md:151-162`.
- **`toy_floor.py`**: a "Toy check of the 'Pricing the Room' rules against the weak-duality floor. Made-up keys, not trace data". It is a one-session dynamic program in which many sessions share one hard room, with "No queue, no SSD contention, open-loop arrivals" — `scripts/toy_floor.py:1-10`.
- **Eviction replay**: replays eviction "on a run's own timeline with the pool sized so that LRU matches the hint-free arm's 19.4 % of returning turns reading from SSD". It scored rule 3 at 16.2 %, the p50 order at 17.7 % and exact next use at 12.6 % — `results/campaign26/DECISIONS.md:34-45`. The code is in `frozen/ec_policy_20261111/evict_replay/`, which is not in this checkout.
- **Host-pool sweep** on the eviction replay — `results/campaign26/DECISIONS.md:1096`; the script `results/campaign26/checks/limits/pool_sweep.py` exists here.
- **First-order offline check** on `jit_p50_c44`'s 1,981 staged turns: "Host room, revokes and the queue are not modelled, so it cannot predict TTFT" — `PAPER_PLAN.md:174-185`.
- **Offline hint scoring and trace studies:**
  - calibration and AUC on campaign 18's real send times — `DIST_HINT_DESIGN.md:110-119`;
  - `scripts/hint_eval.py` "scores the same picks offline against the trace and replays the engine's rules on them" — `REPLAY_REDESIGN.md:297-298`;
  - the 2026-10-27 trace study's eviction AUCs (0.900 vs 0.825 vs 0.763 at ×70) — `PAPER_PLAN.md:106`;
  - "the calibrated offline cache simulation built on the campaign 18/19 timelines" — `REALISTIC_HINTS.md:14-15`.

**Live agents and synthetic workloads**
- Live-agent validation exists only as a plan:
  - "Later: validate with live agents (mini-swe-agent on SWE-bench Verified ...)" — `agent-kv-tiering-evaluation-starter-kit.md:76`;
  - the Claude Code → SGLang live demo is "inferred, not started" — `REALISTIC_HINTS.md:610-626`.
- A human-paced workload is still listed as needed — `PAPER_PLAN.md:26`; `PAPER_INTRO.md:124`.
- Synthetic content appears in only two places:
  - the toy model's made-up keys — `scripts/toy_floor.py:1`;
  - the replay's generated replies, which are not the recorded ones — `scripts/gap_hints.py:19-20` (see Q3).
- The client was also checked against a mock OpenAI server — `REPLAY_REDESIGN.md:652-666`.

### Inferences
- For an item-by-item comparison, this is a **closed-loop trace replay on a real engine, with recorded tool gaps replayed as sleeps and output lengths forced** (inference).
- The CPU models are replays of measured GPU runs, not independent simulators: arrivals, queue waits, evictions and re-reads come from the run's logs (inference from `results/cpu_calib_c44/FINDINGS.md:9-25`).
- Only `toy_floor.py` is a synthetic workload in the strict sense (inference).

### Gaps
- No document reports a discrete-event simulator that outputs TTFT. The starter kit planned to extend `agentic-kv-cache` with tiers (`agent-kv-tiering-evaluation-starter-kit.md:37`, `:72`); whether that was done is not stated.
- The eviction replay, the hint-realism study and most analysis scripts live in `frozen/`. It is git-excluded and absent from this checkout (`ls agent_cache/frozen` fails; `CAMPAIGN23_FINDINGS.md:369`; `REALISTIC_HINTS.md:8-9`), so those CPU results cannot be re-inspected here.

## 2. Tools: real execution, recorded gaps replayed as delays, or sampled?

### Takeaway
Tools are never executed. Each recorded inter-request gap is replayed as a client-side sleep. The gap is the LMCache trace's `pre_gap`: the time from the previous response completing to the next request being sent. The sleep is the recorded gap × a gap scale, capped at 1,200 s and counted from the end of the previous reply's stream:
- ×10 in campaigns 17, 19-21, 23, 25 and 26;
- ×70 in campaigns 14-16 and 18.

The scale is used because at real speed gaps rarely reach the host or SSD tiers. Tool identity enters only through hint keys taken from the recorded replies.

### Cited Findings
- **Gap definition in the source data:** `pre_gap` is "seconds between the previous iteration's response completing and this iteration's request being sent" (median 0.71 s) — `agent-kv-tiering-evaluation-starter-kit.md:13`. The converter uses the same definition — `scripts/convert_lmcache.py:5-7`.
- **How gaps are converted:**
  - A call whose delta has no non-assistant message is merged forward; "its pre_gap and output_length are carried into the next turn (pre_gap added ...)" — `scripts/convert_lmcache.py:23-25`, `:139-145`.
  - "inside a split session_id the dataset clamps pre_gap to 0.0 where the two runs interleave (6 turns), so those gaps are unknown, not zero" — `scripts/convert_lmcache.py:30-31`.
  - Turn 0's gap is 0 in every emitted conversation — `traces/lmcache_agentic_trace_262k.json.stats.json:25`.
- **How gaps are replayed:**
  - Before turn t > 0 the user sleeps min(1,200, `pre_gap` × `--gap-scale`) s, counted from the end of turn t−1's reply stream; turn 0's gap is not slept — `REPLAY_REDESIGN.md:157-161`. Code: `scripts/gap_hints.py:158-160`; `scripts/replay_agentic.py:384-389`.
  - `GAP_CAP_S = 1200.0`; "at x70, 57 of the 262K trace's 768 conversations hold a sleep over 5 h" — `scripts/gap_hints.py:51-52`.
  - Sampled gaps are gone: the old `--gap-sample` flag was removed — `REPLAY_REDESIGN.md:70`.
- **Gap scale by campaign:**
  - ×70 for campaigns 14-16 and ×10 for campaign 17 — `REPLAY_REDESIGN.md:168-172`.
  - ×70 in campaign 18 and ×10 in campaign 19 — `REALISTIC_HINTS.md:77-78`.
  - ×10 in campaign 20 — `DIST_HINT_DESIGN.md:96-97`.
  - ×10 in the campaign 21/23/25 arms — `CAMPAIGN23_FINDINGS.md:68-69`.
  - ×10 in campaign 26 — `results/campaign26/DECISIONS.md:4`; `results/campaign26/config_ec_r13.txt:1` (`GAP=10`).
- **What the 1,200 s cap removes:** 49.6 % of idle time at ×70 (206 of 24,111 sleeps shortened) and 30.0 % at ×10 (74 sleeps) — `REPLAY_REDESIGN.md:168-175`.
- **Recorded gap statistics** (262K trace, 24,111 gaps after turn 0):
  - p50 0.708 s, p90 1.77 s, p95 3.72 s, p99 11.76 s, mean 2.08 s;
  - 21.3 % ≥ 1 s, 3.4 % ≥ 5 s, 0.66 % ≥ 30 s — `traces/lmcache_agentic_trace_262k.json.stats.json:47-57`;
  - p50 by source: SWE-bench 0.706 s (`:90-100`), GAIA 2.18 s (`:117-127`), WildClaw 0.154 s (`:144-154`);
  - at ×70 the median gap is 50 s and the mean 74 s — `DIST_HINT_DESIGN.md:60-61`.
- **Why gaps are scaled, and what that costs:**
  - "At real speed the host and SSD tiers are rarely reached: only ~0.35 % of real gaps reach 60 s" — `REALISTIC_HINTS.md:567`.
  - "x70 manufactures hint value. 47 % of x70 returns fall at 45-60 s, where LRU evicts", mostly OpenHands' ~0.72 s polling grid × 70 — `REALISTIC_HINTS.md:564-565`.
  - ×1 is kept "as the honest negative" — `REALISTIC_HINTS.md:578-579`.
- **Tool identity:**
  - Tool-result headers map to categories: OpenHands "EXECUTION RESULT of [x]", Inspect AI and WildClaw "[TOOL RESULT x]" — `scripts/gap_hints.py:68-78`.
  - Tool results are rewritten as user messages "[TOOL RESULT <function name>]" — `scripts/convert_lmcache.py:26-27`, `:112-119`.
  - The hint key is the tool plus the first word of its command (`execute_bash/pytest`) — `scripts/gap_hints.py:107-131`; `DIST_HINT_DESIGN.md:43-46`.
  - Recorded replies' keys come from `<trace>.reply_keys.json`, built from the parquet "because the converted trace dropped the replies" — `scripts/gap_hints.py:19-22`. For the 262K trace that file lists 22,296 calls (18,463 text, 3,833 native), 1,779 rejected and 36 with no call — `traces/lmcache_agentic_trace_262k.json.reply_keys.json:1` (single-line JSON, `metadata.checks`).
  - A call the harness rejected takes the first word of a random accepted call of the same tool, source and model, "so that the key cannot reveal the rejection" — `REPLAY_REDESIGN.md:282-284`.
- **Online hint table updates:** the client "sleeps the recorded gap x scale (at most 1,200 s) and adds it to the table when the sleep ends" — `DIST_HINT_DESIGN.md:72`; `scripts/gap_hints.py:11-14`.
- **Gap artefacts:**
  - "The 300-s spike cannot be foreseen by any client. It is 73 gaps: 0.30 % of gaps but 80.5 % of the KV-token-seconds in x10 gaps over 300 s" — `REALISTIC_HINTS.md:897-898`.
  - The cap "plants a certain return at 1,200 s" — `REALISTIC_HINTS.md:895-896`.

### Inferences
- A replayed session's timeline is the replay's own generation time (on the test GPU, at recorded output lengths) plus scaled recorded gaps. The original model's generation time is excluded by the `pre_gap` definition (inference from `scripts/convert_lmcache.py:5-7` and `scripts/replay_agentic.py:387-389`).
- `pre_gap` is measured at the original harness, so it includes harness overhead as well as tool runtime (inference from the definition).

### Gaps
- No document splits `pre_gap` into tool runtime, harness overhead and network time.
- Per-key gap distributions are not tabulated beyond examples, e.g. `pip` p10 0.85 s and p50 22 s at ×10 (`PAPER_PLAN.md:136`, `:145`).
- The gap scale of campaign 24 (a just-in-time probe at 70 GB) is not stated in the documents I read.

## 3. Model outputs: free generation or forced? How does a session stay on its recorded path?

### Takeaway
Outputs are forced in length and meaningless in content. Every request sets `max_tokens` to the recorded `output_length`, with `ignore_eos`, temperature 0 and thinking off. The model therefore produces a reply of the recorded length whose text is not the recorded reply. That text is appended verbatim to the history, so its KV is a prefix hit. The next turn appends the recorded tool results or user messages, so every session follows its recorded path whatever was generated.

Because the generated replies hold no recorded tool call, the hinted just-in-time and pricing arms run with `--hicache-storage-next-turn-reply-key-source hint`. The engine then takes the tool key from the request's `replay_reply_key`, the recorded reply's key, instead of parsing its own reply. No engine-parsed key has been used on a GPU.

### Cited Findings
- **Request body:** `max_tokens` = recorded `output_length`, `ignore_eos: True`, `temperature: 0.0`, `skip_special_tokens: False`, `chat_template_kwargs: {"enable_thinking": False}`, `return_cached_tokens_details: True` — `scripts/replay_agentic.py:258-270`, `:400`. "So a reply has the recorded length" — `REPLAY_REDESIGN.md:141-143`. Thinking is always off — `REPLAY_REDESIGN.md:458-462`.
- **History handling:**
  - The client appends `history += turn["messages"]`; after the reply, `history.append({"role": "assistant", "content": r["text"]})  # verbatim: this is what makes the KV a hit` — `scripts/replay_agentic.py:394`, `:409`.
  - The converter drops assistant messages: "the server regenerates them and the client feeds its reply back" — `scripts/convert_lmcache.py:23`.
- **The generated text is junk past end of turn:** "under ignore_eos the model runs past <|im_end|> and hallucinates '</think>' inside its garbage tail" — `scripts/replay_template.py:31-33`. That is why the server runs without `--reasoning-parser` — `scripts/start_server.sh:81-83`.
- **Reply reuse:**
  - Qwen3 thinking models need a replay chat template; "the 2507 Instruct models have no think block and their stock template already re-renders history identically" — `scripts/start_server.sh:99-101`; `scripts/replay_template.py:1-12`.
  - Reuse is measured per turn as `reply_reprefilled` = prompt − cached − delta, "~0 (+- a page) when the previous reply's KV was reused" — `scripts/replay_agentic.py:363-366`; `REPLAY_REDESIGN.md:315-317`.
- **Prompt-length fidelity:**
  - The converter's `replay_prompt_tokens` assumes "one placeholder reply of output_length tokens per earlier turn" — `scripts/convert_lmcache.py:16-19`, `:52`, `:171-173`.
  - On the 262K trace the replay prompt averages 1.007 × the recorded prompt — `traces/lmcache_agentic_trace_262k.json.stats.json:73`.
  - Across arms, campaign 23's prompts match "within 0.3 tokens on average" — `CAMPAIGN23_FINDINGS.md:154-155`.
  - In campaign 26, "a key's prompt is within 1 token in 89 % of cases across the five leading runs" — `results/campaign26/DECISIONS.md:527-529`.
- **Length clipping:**
  - The server runs without `--allow-auto-truncate`, so a prompt plus `max_tokens` over the context length gets HTTP 400 and ends the user.
  - The largest such sum is 109,430 tokens in the 262K traces, inside CTX = 131,072.
  - `short_reply` marks a reply the server shortened — `REPLAY_REDESIGN.md:143-148`; `scripts/replay_agentic.py:361-362`.
- **What `--hicache-storage-next-turn-reply-key-source` does:**
  - Help text: "'reply' reads the generated reply. 'hint' reads the table's replay_reply_key, for a trace replay whose generated replies are not the recorded ones"; the default is `reply` — `python/sglang/srt/arg_groups/fields/memory.py:254-265`.
  - Code: `python/sglang/srt/mem_cache/next_turn_prefetch.py:164-165` (the field is "read only under reply_key_source 'hint'") and `:575-576` (`if self._reply_key_source == "hint": key = table.replay_reply_key`).
  - The `_table`, `_jit` and `_owned` arms pass it — `scripts/start_server.sh:173`, `:175`, `:177`. The `_owned` server is every JIT and pricing arm of campaigns 23-26 — `CAMPAIGN23_FINDINGS.md:355-361`; `results/campaign26/DECISIONS.md:119`.
- **Client side:**
  - "replay_reply_key stands in for the engine reading its own reply: it is the recorded reply's key (<trace>.reply_keys.json), because the replay's replies are not the recorded ones" — `scripts/replay_agentic.py:22-26`.
  - "a workaround for the replay only" — `scripts/gap_hints.py:19-22`; `REPLAY_REDESIGN.md:278-281`.
  - How the real system maps onto the replay — `DIST_HINT_DESIGN.md:73`.
- **Consequences:**
  - "the engine has never read its own reply on a GPU" — `PAPER_PLAN.md:17`.
  - Offline, the engine's parser `reply_key.py` matches the client's keys "on all 25,617 recorded replies" — `DIST_HINT_DESIGN.md:85`.
  - The paper's evaluation must disclose the `hint` setting — memory note `paper-design-section.md:38` (outside the repo).
- **Exact (oracle) hints** carry the exact next gap in a v1 action and need no key — `scripts/replay_agentic.py:18-20`, `:215-232`.
- **An engine-parsed key on replay would need a shim:** "Needs a reply shim, because the replay sets `ignore_eos` and sends no `tools`" — `REALISTIC_HINTS.md:588`.
- **Caveat stated in advance:** "regenerating replies with a different model changes context lengths and tool-call text; that is fine for a systems study as long as you report it" — `agent-kv-tiering-evaluation-starter-kit.md:49`.

### Inferences
- Decode work per turn is realistic in length but not in content. Anything that depends on output content, such as tool-call parsing or a real stop at end of turn, is not exercised (inference).
- Campaign 26's arms differ in "the sampling seed" (`results/campaign26/DECISIONS.md:529-530`). At temperature 0 with forced lengths this should not change token counts (inference).

### Gaps
- No document compares the replay's generated replies with the recorded ones, for example in time per token. Only the length match is reported.

## 4. Data: which traces and tasks, and how they are converted

### Takeaway
One public trace set is used, the LMCache agentic traces:
- tasks and frameworks: SWE-bench Verified via OpenHands CodeAct, GAIA via Inspect AI, and WildClaw;
- recorded with MiniMax-M2.5, Claude Sonnet 4.6, Claude Opus 4.6 and DeepSeek V3.1;
- converted by `convert_lmcache.py` into per-turn message deltas.

Every campaign since 13 uses the 262K-context conversion: 768 conversations and 24,879 turns, at most 50 turns each. Per-turn prompts have a median of 24.6K tokens and a p95 of 47.1K. Each arm replays a seeded subset of the conversations.

### Cited Findings
**Source and frameworks**
- LMCache agentic traces (`huggingface.co/datasets/sammshen/lmcache-agentic-traces`, CC-BY-4.0):
  - "787 multi-turn agent sessions, 24,881 iterations: SWE-bench Verified via OpenHands CodeAct (669 sessions, median 38 turns), GAIA via Inspect AI (85), WildClaw (10); generated with MiniMax-M2.5, Claude Sonnet 4.6, Claude Opus 4.6, DeepSeek V3.1";
  - fields `session_id`, `model`, `input` (the full cumulative message list), `output_length`, `pre_gap` — `agent-kv-tiering-evaluation-starter-kit.md:13`.
- **The converter's own counts** on the downloaded parquet (revision `main`) disagree with the starter kit's (see Q11):
  - 24,880 rows and 767 session ids, one of which holds two chains, for 769 chains — `traces/lmcache_agentic_trace_262k.json.stats.json:17-21`;
  - by source: 665 SWE-bench chains (23,356 rows), 94 GAIA (1,377) and 10 WildClaw (147) — `:75-78`, `:102-105`, `:129-132`.
- The replay's own description: "the public lmcache agentic trace that our replay uses (768 sessions, 24,879 turns)" — `REALISTIC_HINTS.md:14-15`.
- Batch agents only: "Human waits cannot be tested on this trace" — `DIST_HINT_DESIGN.md:155`.

**Converted traces and what the campaigns use**

| Trace | Conversations | Turns | Used by |
|---|---|---|---|
| 32K cut | 731 | 17,887 | the wrappers' default |
| 262K | 768 | 24,879 | campaigns 13 onward |
| `262k_ge45_tl_s1` | 283 | 13,989 | ORACLE_PLAN E1/E2 |

- At most 50 turns per conversation in each — `REPLAY_REDESIGN.md:118-124`.
- Campaign 26 (and, by pairing, 21 and 23) uses the 262K trace with `TURNS=50` — `results/campaign26/config_ec_r13.txt:1`; `results/campaign26/DECISIONS.md:4-5`.
- **262K conversion settings:**
  - tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507, max_context 262,144, margin 512, min_turns 5, tool_role_mode `as_user` — `traces/lmcache_agentic_trace_262k.json.stats.json:3-14`;
  - generated 2026-09-22T02:33:43Z (`traces/lmcache_agentic_trace_262k.json`, `metadata.generated_at`);
  - no chain truncated (`:79-81`); 768 chains emitted, with 1 below the minimum turns; 137 start mid-run; 1,779 later turns have no assistant message — `:31-35`.
- **The 50-call stop:** 228 of the 768 conversations (all SWE-bench) stop at exactly 50 calls; "whether that is a harness cap or the dataset's window is unresolved" — `scripts/gap_hints.py:55-57`; `DIST_HINT_DESIGN.md:152`.

**Context lengths**
- Per-turn prompt tokens (262K trace): mean 25,525, p50 24,597, p95 47,050. Replay prompt p50: 24,740. Output length: mean 211, p50 102, p95 709 — `traces/lmcache_agentic_trace_262k.json.stats.json:58-72`.
- Final prompt per session: SWE-bench p50 36,653 and max 103,885 (`:84-85`); GAIA p50 20,995 (`:111-112`).
- Shared system prompt: 7,808 tokens; final context p50 35.6K (campaign 7's 128 SWE-bench conversations) — `RUNBOOK.md:2458`.
- History a returning turn reads:
  - ×70 (campaign 14): 1.58 GB per SSD-reading turn on average — `PAPER_INTRO.md:139`;
  - 44 users, ×10: reading turns' histories end at 11K / 26K / 42K tokens (p10 / p50 / p90) — `results/cpu_calib_c44/FINDINGS.md:160-161`;
  - "A read loads 17-19K tokens" — `results/campaign26/DECISIONS.md:294-295`.
- Live contexts in total: sessions in progress hold a median 1,176K tokens at 44 users — `results/campaign26/DECISIONS.md:561-562`.

**How `convert_lmcache.py` converts**
- Rows of a `session_id` are split into append-only chains; a WildClaw session bundles two runs — `scripts/convert_lmcache.py:5-11`, `:72-97`.
- Each turn carries only the new non-assistant messages, plus `prompt_tokens` (recorded cumulative input), `replay_prompt_tokens`, `output_length`, `pre_gap`, `iteration`, `orig_model` and `source` — `scripts/convert_lmcache.py:13-20`, `:143-148`.
- Rules — `scripts/convert_lmcache.py:22-31`, `:166-183`:
  - assistant messages are dropped;
  - calls with no non-assistant message are merged forward;
  - tool results become user messages "[TOOL RESULT name]";
  - a session is truncated at the first turn whose replay prompt plus output exceeds max-context − margin;
  - a session is emitted only with at least min-turns turns.

**Supporting scripts**
- **`replay_template.py`** writes Qwen3 replay templates (`qwen3_replay_nothink.jinja`, `qwen3_replay_think.jinja`) that keep a re-fed reply prefix-identical — `scripts/replay_template.py:1-12`. The campaigns on the 2507 Instruct model ran the stock template instead (`TEMPLATE=`) — `REPLAY_REDESIGN.md:231-233`, `:519`.
- **`reply_keys.py`** builds `<trace>.reply_keys.json` from the parquet and refuses a trace whose gaps differ from the parquet's — `REPLAY_REDESIGN.md:279-284`.
- **User-to-conversation assignment:** a seeded permutation `random.Random(f"{seed}:0")`, never two users on the same conversation — `scripts/replay_agentic.py:156-158`; `REPLAY_REDESIGN.md:132-136`.

### Inferences
- With `USERS=768`, `ARRIVAL=0.1` and 2-hour runs, at most about 720 users can arrive. With `C=44` far fewer start, so each arm replays the same seeded subset, not all 768 conversations (inference: 0.1/s × 7,200 s; identical permutation and arrival times).
- Each (source, model) pair counts as one "agent system" for the online hint table (`scripts/replay_agentic.py:166`; `DIST_HINT_DESIGN.md:71`), so the per-model mix shapes the hints (inference).

### Gaps
- The trace's per-model session counts are not stated.
- How many distinct conversations start in a 2 h arm is not stated; documents report only turns sent.
- No document explains the 787 vs 767 session discrepancy.

## 5. Client–engine interface: requests, hints, and what the engine parses itself

### Takeaway
The client is a custom aiohttp script, not SGLang's `bench_serving`.
- **Requests:** streamed OpenAI `/v1/chat/completions` calls with an `rid` of `<tag>-c<user>-t<turn>`, through an unlimited connection pool.
- **Load control:** a fixed cap on users in session; idle users keep their slots.
- **Hints:** a `kv_hints` field rides on turn k's request and describes turn k+1:
  - a v1 action carries the exact next gap (oracle);
  - a v2 action carries the per-key gap table (p10/p50/p90/n, plus an `_end` entry) and `replay_reply_key`, with the picked p50 also sent as v1.
- **Engine side:** the engine reads hints when the turn commits, links turns by `next_rid`, picks the table entry by reply key, and adds its own queue state. In replays it does not parse its own replies.

### Cited Findings
**API and session identification**
- Streamed POST to `{url}/v1/chat/completions`:
  - `rid` "shows up in the server's --log-requests and HICACHE_EVT lines";
  - `model` is the first model listed by `/v1/models` — `scripts/replay_agentic.py:255-277`, `:458-464`;
  - ids are `<tag>-c<user>-t<turn>` — `scripts/replay_agentic.py:209-210`, `:39`.
- The session id travels only inside `kv_hints`, as `"session": f"{tag}-c{conv}"`, together with `"next_rid"` — `scripts/replay_agentic.py:215-221`. "Today plans are keyed by `next_rid`, a request id the replay client chooses" — `REALISTIC_HINTS.md:521-523`.
- Each response returns a per-tier cached-token split (`device`, `host`, `storage`) — `scripts/replay_agentic.py:269`, `:298-300`, `:355-357`; `REPLAY_REDESIGN.md:309`.
- Connections:
  - no total timeout, a 3,600 s read timeout and no connection limit;
  - the limit was removed because aiohttp's default 100-connection pool queued requests inside the client ("x1 campaign: p50 7.4 s, p95 51 s before server entry") — `scripts/replay_agentic.py:466-472`;
  - one retry on a keep-alive connection the server closed — `scripts/replay_agentic.py:307-324`; `REPLAY_REDESIGN.md:236-246`.

**Gaps and the concurrency cap**
- Sleeps start from the end of the previous reply (Q2) — `scripts/replay_agentic.py:384-389`.
- `--concurrency` caps "the users in session, each counted from its first turn to its last with the idle gaps included: later arrivals wait in arrival order (conv_start.queued_s, outside every TTFT) and a returning turn never waits" — `scripts/replay_agentic.py:14-16`.
- It is an `asyncio.Semaphore` — `scripts/replay_agentic.py:191`, `:330-343`; `REPLAY_REDESIGN.md:177-187`.

**Hints that cross the boundary**
- **v1 (oracle) envelope** `{session, next_rid, next_gap_s}`:
  - `next_gap_s` is the sleep after scaling and capping;
  - it is suppressed when the timeout will certainly cut the next turn, and not sent on a user's last turn — `scripts/replay_agentic.py:215-232`; `REPLAY_REDESIGN.md:211-224`.
- **v2 envelope** `{session, next_rid, gaps, replay_reply_key}`:
  - `gaps` is a list of `{key, p10, p50, p90, n}` or `{key, ended: true}`;
  - a v1 action with the picked entry's p50 is added unless `--hint-point none` — `scripts/gap_hints.py:412-421`;
  - a full example envelope is in `REPLAY_REDESIGN.md:250-264`.
- **Table semantics (the gap-distribution table in `gap_hints.py`):**
  - Key: the tool plus the first word of its command. A leading `cd` and `VAR=value` are skipped, and `python -m x` becomes `python-m-x` — `scripts/gap_hints.py:107-131`.
  - The engine picks the key's entry, else its tool's, else `*` — `scripts/gap_hints.py:144-150`.
  - An entry needs at least 5 gaps — `scripts/gap_hints.py:58-59`, `:303-306`; `DIST_HINT_DESIGN.md:47-48`.
  - Sources — `scripts/gap_hints.py:11-17`, `:333-350`; `scripts/replay_agentic.py:234-245`; `DIST_HINT_DESIGN.md:69-74`:
    - `online`, the default: one table per agent system (harness and model), starting empty each run and updated as each sleep ends;
    - `precalc`: other tasks' recorded gaps;
    - `session`: a lognormal fit to the session's own gaps;
    - `none`.
  - Size: about 3 KB and 38 entries per request — `DIST_HINT_DESIGN.md:50-52`; `REPLAY_REDESIGN.md:272-273`.
- **End-of-session entries:**
  - Key `_end` with `{"ended": true}` marks where the recorded session ends under `--hint-end explicit`, the default. `natural` marks it only below the 50-call cap, and `none` never — `scripts/gap_hints.py:24-27`, `:61-64`, `:384`, `:396-399`.
  - Where the replay stops early (`--turns`, `--timeout`) or the converter cut a conversation, no end is signalled — `REPLAY_REDESIGN.md:285-291`.
  - In the v1-only engine path, a turn without a v1 action "reads as no hint, which it registers ENDED" — `REPLAY_REDESIGN.md:292-293`. "a turn without a hint is treated as the last one, registered ENDED and evicted first" — `REALISTIC_HINTS.md:89-90`.
- **A hint is "three signals in one":** session linkage, the end, and the gap value — `REALISTIC_HINTS.md:87-94`.
- **`next_use` is an engine eviction policy fed by these hints, not a hint field:**
  - "ended sessions first, then sessions returning after 300 s, then LRU" (`--radix-eviction-policy next_use --radix-eviction-policy-config '{"horizon_s": 300}'`) — `scripts/start_server.sh:114-115`.
  - The prefetcher "marks a session's history as needed now when its turn arrives ... so next-use eviction passes over it" — `CAMPAIGN23_FINDINGS.md:172-173`.
  - Campaign 26 adds `return_chance` eviction, with scores `entry`, `samples`, `lru` and `queue` — `results/campaign26/DECISIONS.md:58-60`.

**What the engine reads or parses itself**
- **When hints are read:** "Once, when turn k commits ... To the engine, this is the same as a message sent right after the reply with no delay" — `REALISTIC_HINTS.md:83-85`. A real post-reply message would arrive 82/94 ms later at the median — `REALISTIC_HINTS.md:525-527`.
- **The v2 table:**
  - At commit, `next_turn_prefetch.py` picks the entry and takes its p50; "A table wins over a version 1 action in the same request" — `DIST_HINT_DESIGN.md:84`.
  - `reply_key.py` can read the tool and first word from the reply's text form or from `<tool_call>` JSON — `DIST_HINT_DESIGN.md:85`; `python/sglang/srt/mem_cache/reply_key.py:1-7`. Replays use `replay_reply_key` instead (Q3).
  - Engines before 2026-11-01 read only v1, so "the client does the pick for it" — `REPLAY_REDESIGN.md:292-293`; `DIST_HINT_DESIGN.md:88-90`.
- **Queue state the engine adds on its own:**
  - which turns are queued and what each matched on arrival — `CAMPAIGN23_FINDINGS.md:190-192`;
  - a predicted wait from queue position over the admission rate — `CAMPAIGN23_FINDINGS.md:174-175`;
  - the arrival gate and arrival hold — `python/sglang/srt/arg_groups/fields/memory.py:331-356`; `CAMPAIGN23_FINDINGS.md:52-56`.
- **What the no-hint arms still take from the hint channel:**
  - The `hintfree_ctl` client sends oracle hints, which that server ignores — `results/campaign26/config_hintfree_ctl.txt:1`; `scripts/start_server.sh:23-24`.
  - `nofc` takes "the end signal and which request continues which session" — `results/campaign26/DECISIONS.md:202-203`.
  - `nofc_noend` uses the hint only for session linkage — `results/campaign26/DECISIONS.md:125-127`.
  - The outside reviewer: "The rival still takes the session id and the reply key from the hint channel: call it 'session id only'" — `results/campaign26/DECISIONS.md:507`.

### Inferences
- Hints are piggybacked on the request as an extension of the OpenAI body (`kv_hints`, `rid`). There is no separate control channel: "The scheduler's dispatcher has no hint message today" (`REALISTIC_HINTS.md:553-555`) (inference).

### Gaps
- The wire cost of the ~3 KB table per request is not measured. Sending only changes is logged as later work — `DIST_HINT_DESIGN.md:163`.

## 6. Load: users, arrivals, run length, warm-up, timeouts

### Takeaway
"44 users" and "128 users" are caps on users in session (`C=44`, `C=128`). Idle users hold their slots through their gaps. Behind the cap, 768 users arrive as a Poisson process at 0.1 users/s (seed 0) and wait for a slot outside TTFT. Each arm runs 2 h (`CLIENT_TIMEOUT=7200`) from an empty cache and SSD.

Host memory pairs with the user count:
- 44 users with 70 GB (and 100 GB in three campaign 26 arms);
- 128 users with 160 GB.

Comparisons use the returning turns all compared runs share; no warm-up cut is stated for those tables. Older campaigns ran 5 h at ×70 under a backlog-driven cap that has since been removed.

### Cited Findings
- **Campaign 26 configs:**
  - 44 users: `USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 C=44 SEED=0 CLIENT_TIMEOUT=7200 SGLANG_TIMEOUT_KEEP_ALIVE=3600 ... HINTS=dist` — `results/campaign26/config_ec_r13.txt:1`;
  - 128 users: `C=128`, host `HOST_GB=level` (NAT160) — `results/campaign26/config_ec_r134_c128.txt:1`;
  - 100 GB: `HOST_GB=100` — `results/campaign26/config_jit_p50_h100.txt:1`.
  - In words: "Setting: 44 users, 70 GB of host memory, gaps x10, ARRIVAL=0.1, USERS=768, TURNS=50, SEED=0, 2 h per arm, as campaigns 21 and 23" — `results/campaign26/DECISIONS.md:4-5`.
- **Arrivals:**
  - a Poisson process at `--arrival-rate`, the first user at 0 s, drawn from `random.Random(seed)` — `scripts/replay_agentic.py:134-142`, `:158`;
  - every arm gets the same users, conversations and arrival times, matching campaigns 14-17 at the same seed within 19.9 ms — `REPLAY_REDESIGN.md:401-408`;
  - campaign 25 reran three arms at 0.5 users/s ("same users, same order") — `PAPER_PLAN.md:24`; `CAMPAIGN23_FINDINGS.md:21-23`.
- **What the cap means:**
  - "a cap on users binds much sooner than a cap on requests in flight would";
  - at ×70 with `C=128`, 615-632 of 768 users wait for admission outside every TTFT (derived) — `REPLAY_REDESIGN.md:20-22`, `:179-187`, `:614-616`;
  - at 44 users, "TTFT leaves out a 14-20 minute wait for a session slot" — memory note `c26-100gb-conclusion-audit.md:29` (outside the repo; no repo document gives this number).
- **Engine-side concurrency:**
  - `--max-running-requests 64` — `scripts/start_server.sh:91`; "The 64-request cap never binds" at 44 users — `CAMPAIGN23_FINDINGS.md:140-141`;
  - at 128 users, 59-90 turns are queued from minute 45, and the GPU pool is 0.98 full with about 30 running — `CAMPAIGN23_FINDINGS.md:166-167`, `:242-244`.
- **Turns per arm:**
  - 44 users: 8,037-12,201 sent in 2 h; 128 users: 8,511-16,228 — `CAMPAIGN23_FINDINGS.md:77`, `:94`;
  - 100 GB arms: 12,861-12,862 each — `results/campaign26/DECISIONS.md:1175`;
  - at ×10 and 128 users over 5 h (campaign 19), the oracle arm finished all 768 users and the baseline 413 — `REALISTIC_HINTS.md:99-100`.
- **Load episodes at 44 users:**
  - turn positions 5,500-7,000 and 8,750-11,000 — `results/campaign26/DECISIONS.md:371-374`;
  - offered load 85-91 turns per minute in the first episode and 66-72 in the second — `results/campaign26/DECISIONS.md:530-531`.
- **Run length:**
  - 2 h for campaigns 21, 23, 25 and 26 — `results/campaign26/DECISIONS.md:4-5`; `CAMPAIGN23_FINDINGS.md:64-66`;
  - 5 h (an 18,000 s timeout) for campaigns 14-17 — `REPLAY_REDESIGN.md:417`;
  - campaign 20: one 5 h run — `DIST_HINT_DESIGN.md:96-97`;
  - the 128-user references from campaigns 19-20 are 5 h runs, compared over their first 2 h or 6,900 s — `CAMPAIGN23_FINDINGS.md:86-88`; `results/campaign26/DECISIONS.md:594`; `results/campaign26/extra_runs.txt:4-5`.
- **Timeout semantics:** after T, no user starts and no request is sent; sleeping users end `capped`, waiting users are skipped, and in-flight requests complete — `REPLAY_REDESIGN.md:197-209`. Other timeouts: aiohttp `sock_read` 3,600 s — `scripts/replay_agentic.py:471`; `SGLANG_TIMEOUT_KEEP_ALIVE=3600` in the configs.
- **Warm-up and cold start:**
  - every arm starts with an empty SSD — `CAMPAIGN23_FINDINGS.md:154`; "every run starts from an empty cache" — `results/campaign26/DECISIONS.md:529`;
  - the online hint table starts empty, and "Campaign metrics start at 45 min, after the table has filled" — `DIST_HINT_DESIGN.md:74`;
  - SSD-busy columns cover minutes 45-120 — `CAMPAIGN23_FINDINGS.md:82`;
  - the campaign 23 and 26 TTFT tables cover "returning turns all ... arms share", with no time cut stated — `CAMPAIGN23_FINDINGS.md:70-71`; `results/campaign26/DECISIONS.md:151`;
  - the figure turn set keeps returning turns that every drawn run sent at least 300 s before its last send (about minute 115) — `scripts/c26_pricing.py:14-15`, `:35`;
  - in campaign 26, "no turn censored by the 300 s rule" — `results/campaign26/DECISIONS.md:527`.
- **The removed backlog cap (campaigns 14-17):**
  - it started at 128 and moved by up to 8 users every 120 s to hold the client-measured backlog of requests awaiting their first token at 32 (within 16-512), with at least 5 s between user starts;
  - in campaign 14's `three_tier_wc` arm it ranged 92-196, mean 166.6;
  - those campaigns also had a stationary start (125 of the first 128 users began mid-session) and 30-50-turn draws per user — `REPLAY_REDESIGN.md:189-192`, `:412-417`;
  - campaign 13 used a fixed cap of 128, cycling through 128 conversations — `REPLAY_REDESIGN.md:189`, `:421-424`.

### Inferences
- At 0.1 users/s over 2 h at most about 720 users arrive, and with `C=44` most of them wait for a slot rather than run. The offered load is set mainly by C and the gap scale, not by the arrival rate (inference). This agrees with campaign 25, where arrivals 5× faster moved three arms by only 0.01-0.12 s (`CAMPAIGN23_FINDINGS.md:21-26`).
- Our "users" correspond to papers' "concurrent sessions", but they include idle sessions (inference).

### Gaps
- No repo document reports the campaign 26 slot wait; only the memory note gives 14-20 min.
- C for campaign 18 (×70, redesigned client) is not stated in the documents I read.
- Whether the campaign 23 and 26 TTFT tables drop a warm-up is not stated. The 45-minute statement (`DIST_HINT_DESIGN.md:74`) predates them.

## 7. Stack: engine, model, GPU, KV capacities, tiers

### Takeaway
- **Engine:** SGLang (this fork) with HiCache plus the `agent_cache` prefetcher and eviction changes; each arm boots a fresh server from a frozen engine copy.
- **Hardware:** one NVIDIA H200 (143,771 MiB, TP 1).
- **Model:** Qwen3-30B-A3B-Instruct-2507, bf16 KV (98,304 B/token), 131,072-token context, triton attention.
- **GPU KV pool:** 668,160 tokens (the natural pool at mem-fraction 0.85).
- **Host (L2) pool, write-through from the GPU:** 70 GB = 712,128 tokens, 100 GB = 1,017,280, or 160 GB = 1,627,648. The host-only room is $C_{host}-C_{GPU}$: 44K tokens at 70 GB.
- **L3:** the NIXL POSIX backend on a 2.27 TiB network-attached SSD (about 1.9-2.1 GB/s, shared by reads and writes), prefetch policy `wait_complete`.

### Cited Findings
- **Hardware:**
  - a Nebius instance, "1x H200, 143,771 MiB, SM90" — `RUNBOOK.md:3-4`, `:113`;
  - host: 16 vCPU, 196 GB RAM, no swap — `RUNBOOK.md:117`;
  - container `sglang_hicache` from `lmsysorg/sglang:nightly-dev-20260907-30705c00` — `RUNBOOK.md:116`;
  - "Every result is TP 1. The prefetcher refuses TP or PP ≠ 1, DP attention and PD" — `REALISTIC_HINTS.md:907-908`;
  - the VM is preemptible and the GPU unit changes across restarts — `PAPER_PLAN.md:24`; `results/campaign26/DECISIONS.md:6`.
- **Model and server settings:**
  - campaign runs: `MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto L3_CLEANER_PCT=70,60` — `results/campaign26/config_ec_r13.txt:1`;
  - `TEMPLATE=` (stock) — `REPLAY_REDESIGN.md:519`, `:572`;
  - the model was picked in campaign 7 "for its 262K context" (bf16 weights 57 GB) — `RUNBOOK.md:2458`.
- **Common flags:** `--kv-cache-dtype $KV_DTYPE --attention-backend triton --page-size 64 --chunked-prefill-size 8192 --mem-fraction-static 0.85 --max-running-requests 64 --radix-eviction-policy lru --enable-metrics --enable-cache-report --enable-request-time-stats-logging` — `scripts/start_server.sh:84-97`.
- **Pool levels:** NAT160 sets no `--max-total-tokens` (the server profiles its own pool) and a 160 GB host pool; `HOST_GB` overrides the host size, giving run names like `custom_L1nat_H70` — `scripts/start_server.sh:53-63`.
- **Capacities:**
  - $C_{GPU}$ 668,160 tokens; $C_{host}$ 712,128 (70 GB) and 1,627,648 (160 GB); $R$ = 43,968 and 959,488; $b$ = 98,304 B/token (bf16); $\beta$ = 2.11 GB/s shared — `PAPER_PLAN.md:36-46`;
  - 100 GB = 1,017,280 tokens — `results/campaign26/DECISIONS.md:1148`;
  - L1 668,160 and L2 160 GB = 1,627,648 also appear in `RUNBOOK.md:2458`.
- **Tier configuration:**
  - L2: `--enable-hierarchical-cache --hicache-size $HSIZE --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first` — `scripts/start_server.sh:102-106`.
  - L3: `--hicache-storage-backend nixl --hicache-storage-prefetch-policy wait_complete`, with `l3_extra.json` (`prefetch_threshold` 256, `prefetch_timeout_base` 1.0, `prefetch_timeout_per_ki_token` 0.25, cleaner watermarks), `SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/mnt/ssd/hicache_l3` and plugin POSIX — `scripts/start_server.sh:107-110`, `:129-139`.
  - The boot refuses to start unless `/mnt/ssd/hicache_l3` sits on `/dev/vdc` — `scripts/start_server.sh:130-131`.
- **Agent-cache levers:**
  - deadline read order, a 512-page read window and prefetch pin fraction 0.3 — `scripts/start_server.sh:111-113`;
  - next-use eviction with a 300 s horizon — `:114-115`;
  - the JIT arm: `--hicache-storage-next-turn-prefetch on --hicache-storage-next-turn-reply-key-source hint --hicache-storage-next-turn-read-timing just_in_time --hicache-storage-next-turn-arrival-gate on --hicache-storage-next-turn-arrival-hold on` — `:172-173`;
  - retry flags: poll interval 0 with 8 attempts on the `oracle_v2*` and `dl_poll0` arms, 8/8 otherwise — `:163-166`;
  - campaign 26's flags (expected-cost timing, memory price, `return_chance` eviction, soft pins, price learning, landing quantile) — `results/campaign26/DECISIONS.md:52-64`; help texts in `python/sglang/srt/arg_groups/fields/memory.py:266-330`.
- **SSD (bandwidth figures disagree; see Q11):**
  - a 2.27 TiB network-attached persistent volume, `/dev/vdc`, ext4, `rw,noatime,discard`;
  - a dd ceiling of 1.88 GiB/s at 100 % util "in BOTH directions" (2026-09-21) — `RUNBOOK.md:118`;
  - "2.11 GB/s shared cap measured on 2026-10-13", with reads plus writes at 2.10-2.11 GB/s under load — `PAPER_INTRO.md:140`;
  - a read ceiling of "about 1.85 GB/s" at 128 users in campaign 23 — `CAMPAIGN23_FINDINGS.md:263-264`;
  - big reads take 3.20-3.63 ms per page — `CAMPAIGN23_FINDINGS.md:284-286`.
- **Engine versions:**
  - frozen copies are loaded through `ENGINE_PYTHONPATH` "so edits made while a campaign runs do not reach its later arms" — `scripts/start_server.sh:192-195`;
  - campaign 26 arms ran `engine_owned_20261101`, `engine_5df71c4bf0_v2`, and `engine_ec_20261111d` and `e` — `results/campaign26/DECISIONS.md:99-116`;
  - "python/ and the tests at a5fe6121c8 equal engine e" — `results/campaign26/DECISIONS.md:67`;
  - campaign 23's engine work was committed at 03afa49778 — `CAMPAIGN23_FINDINGS.md:313-314`.
- **Speed constants:**
  - prefill T(L) = 0.092 + 0.84e-6 L + 3.21e-9 L² s, fitting 4K-65K tokens within 7 %, from the 2026-09-22 single-stream sweep (bf16 KV, triton, 8K chunks) — `results/cpu_calib_c44/FINDINGS.md:153-158`;
  - decode step 33.6 / 46.0 / 60.3 ms at 8-15 / 16-23 / 24-31 running requests; first-turn prefill about 0.09 ms per token — `results/campaign26/DECISIONS.md:412-414`.

### Inferences
- The room is small at 44 users because write-through mirrors the GPU pool in host memory: 712K − 668K ≈ 44K tokens, or two to three histories (inference consistent with `results/campaign26/DECISIONS.md:293-296`).

### Gaps
- Host↔GPU transfer throughput for the L2 path is not stated; only "PCIe Gen5 x16" appears (`RUNBOOK.md:113`).

## 8. Metrics: TTFT, reading turns, recomputed tokens, stall, and others

### Takeaway
The headline is mean client-side TTFT over the returning turns that every compared run sent, paired by (conversation, turn). Around it sit:
- queue wait (server log) and a split of the stall;
- history recomputed per returning turn;
- the share of turns that read from SSD ("reading turns");
- SSD pages read and device busy, and reads per reading turn;
- pinned tokens and read launches refused room;
- p90/p99 and turns over 10 s;
- turn latency (send to last token) and throughput (turns sent);
- "share of the gain".

Differences between arms are judged against measured rerun moves, with a 500-position block bootstrap as a within-pair floor.

### Cited Findings
- **TTFT:**
  - measured at the client, from `t_send` (taken just before the connection opens) to the first non-empty content delta — `scripts/replay_agentic.py:274`, `:290-295`; `REPLAY_REDESIGN.md:209`;
  - the slot wait `conv_start.queued_s` is outside TTFT — `scripts/replay_agentic.py:14-16`.
- **Turn latency** runs from send to last token — `scripts/replay_agentic.py:304`; `results/campaign26/DECISIONS.md:479`.
- **Paired turn sets:**
  - tables name the "returning turns all N arms share": 7,768 at 44 users and 8,175 at 128 users in campaign 23; 6,715 and 11,145 in campaign 26 — `CAMPAIGN23_FINDINGS.md:70-71`, `:86-88`; `results/campaign26/DECISIONS.md:151`, `:442`;
  - "turns pair with those runs by (conv, turn)" — `results/campaign26/DECISIONS.md:5`.
- **Queue wait** comes from `--enable-request-time-stats-logging`, one "queue_duration=..., forward_duration=..." line per request — `scripts/start_server.sh:94-96`. "After admission" is TTFT minus queue wait, and includes about 0.13 s between the client's send and the scheduler's entry — `results/campaign26/DECISIONS.md:541-542`.
- **Stall:** "A turn's queue wait is split where its last SSD read lands: the part before is its read stall (zero when it read nothing or its reads had landed by arrival), the part after is the rest" — `results/campaign26/DECISIONS.md:269-271`. The "read was the last wait" share moves with the 0.1 s bound: 0.14-0.20 s for bounds of 0.05-0.5 s — `results/campaign26/DECISIONS.md:539-541`.
- **Recomputation:**
  - per turn, `reply_reprefilled` counts history tokens recomputed beyond the new messages — `scripts/replay_agentic.py:363-366`; `REPLAY_REDESIGN.md:315-317`;
  - campaign tables report "Recomputed tokens per turn" — `CAMPAIGN23_FINDINGS.md:76`;
  - the figures use "history recomputed per returning turn (with the turn's own delta_tokens)" — `scripts/c26_pricing.py:11-13`.
- **Reading turns:** "'s2h reads' is turns that read" — `results/campaign26/DECISIONS.md:534-535`; "Turns that read from SSD" — `:172`; the share of returning turns with an SSD read — `scripts/c26_pricing.py:13`.
- **I/O:**
  - SSD pages; reads per reading turn; arrival lookups that found nothing; SSD busy from iostat over minutes 45-120 — `CAMPAIGN23_FINDINGS.md:78-82`;
  - the read thread is busy 16-20 % of the run at 44 users — `results/campaign26/DECISIONS.md:292`;
  - `results/campaign26/iostat_vdc.log` exists.
- **Host memory:**
  - mean pinned tokens, with one definition for every arm over each arm's first 2 h — `CAMPAIGN23_FINDINGS.md:81`, `:369-371`;
  - read launches revoked or refused for room — `CAMPAIGN23_FINDINGS.md:202-204`;
  - `held_tokens`: SSD-loaded KV held until admission, against the 44K room — `results/campaign26/DECISIONS.md:419-427`.
- **Tails and throughput:**
  - p90, p99 and turns over 10 s — `results/campaign26/DECISIONS.md:210-211`;
  - turns sent in 2 h — `CAMPAIGN23_FINDINGS.md:77`; turns sent before minute 115 and completion tokens — `results/campaign26/DECISIONS.md:480-484`.
- **Share of the gain:** "The share is of the way from each original-HiCache run to JIT exact" — `results/campaign26/DECISIONS.md:161`. Campaign 23 reports per-step shares — `CAMPAIGN23_FINDINGS.md:111-117`.
- **Statistics:**
  - a block bootstrap over 500 consecutive turn positions, reported as 5-95 % — `results/campaign26/DECISIONS.md:367-377`;
  - "That interval is a floor": it is turn scatter within one pair of runs, not rerun noise — `:378-381`;
  - the decision rule: a difference counts when every pair of the two arms' runs has the same sign and the smallest difference exceeds the largest rerun move of either arm — `:382-385`;
  - measured rerun moves on 6,715 turns: original HiCache 1.15 s, hint-free 0.28 s, JIT p50 0.12-0.25 s, rules 1 and 3 0.12 s — `:458-460`;
  - "a difference between arms under about 0.25 s is not a result on one run each" — `:142`.
- **Hardware control per run:** decode step by running count, first-turn prefill, and SSD time per page — `results/campaign26/DECISIONS.md:412-415`, `:437-438`.
- **Independent recomputation:** two agents re-derived 157/157 client-side and 1,946/1,946 server-side numbers from the raw logs — `results/campaign26/DECISIONS.md:520-525`.
- **Client records:** each turn record's fields are listed in `REPLAY_REDESIGN.md:300-313`.
- **CPU model metric:** stall seconds plus recomputed tokens at a flat 9K tokens/s, later priced by position on the quadratic curve — `results/cpu_calib_c44/FINDINGS.md:63-64`, `:160-162`.
- **Planned but unreported metrics:**
  - the starter kit planned restore-hit rate, memory-time, and throughput at a TTFT SLO — `agent-kv-tiering-evaluation-starter-kit.md:66`;
  - "Sessions per GPU under a latency target" is still listed as needed — `PAPER_PLAN.md:25`;
  - the latency-against-load knee is read "from swings in one workload" — `PAPER_PLAN.md:16`; `CAMPAIGN23_FINDINGS.md:266-270`.

### Inferences
- No GPU campaign reports SLO attainment or goodput, so comparing with papers that report SLO metrics needs a conversion (inference).

### Gaps
- How campaign 23's server-side "recomputed tokens per turn" is computed is not spelled out in the documents; the scripts are in `frozen/`.
- The starter kit's restore-hit rate is not reported under that name (not stated).

## 9. Arms and baselines: what each one means

### Takeaway
Every arm shares one workload and differs only in server flags and engine build.
- **Baselines:**
  - original HiCache (`three_tier_wc`);
  - a pins-only "hint-free" arm (`three_tier_wc_dl_poll0`, campaigns 21-26);
  - from campaign 26, queue-aware rivals with no gap forecast (`nofc`, and `nofc_noend` with no end signal either). The outside review says `nofc_noend` should be the baseline of record.
- **Hinted arms:**
  - the old routine: the oracle (exact hints) and old p50;
  - the just-in-time routine: JIT exact and JIT p50;
  - campaign 26's expected-cost ("pricing the room") arms and a fixed-quantile rival.

### Cited Findings
**Baselines**
- **Original HiCache** = `three_tier_wc`: "no hints, no read pins, paced miss retries on" — `CAMPAIGN23_FINDINGS.md:64`; `scripts/start_server.sh:145`.
- **Hint-free (pins only)** = `three_tier_wc_dl_poll0`, run in campaigns 21-25 and as `hintfree_ctl` in 26 — `scripts/start_server.sh:23-24`, `:160-169`; `CAMPAIGN23_FINDINGS.md:64-65`:
  - deadline order and pinned staging, with paced retries off;
  - "Hints the client sends are ignored";
  - "effectively only pins" — memory note `no-hint-arm-mechanics.md:23` (outside the repo).
- **`nofc`:** reads only at arrival; eviction takes ended sessions first, queued turns last, and LRU in between — `results/campaign26/DECISIONS.md:104`, `:200-203`.
- **`nofc_noend`** (engine e): the same with the end signal ignored, "the hint-free rival that keeps queued turns' history"; "it uses the hint for nothing an engine could not see itself (which request continues which session)" — `results/campaign26/DECISIONS.md:106`, `:125-127`.
  - Its mechanisms: pins, plus queue state (queued-last eviction, reads ordered by queue position with a hold, re-reads), plus a session-link gate — memory note `no-hint-arm-mechanics.md:11-21`.
  - The outside review: "the engine with no hints should be the baseline of record" — `results/campaign26/DECISIONS.md:493-495`.

**The old routine**
- **Oracle** = `three_tier_wc_oracle_v2_nu` with exact hints. It "reads a turn's history as soon as it has been evicted and admission is within 120 s", in deadline order, with next-use eviction — `CAMPAIGN23_FINDINGS.md:48-50`, `:66-67`; `scripts/start_server.sh:178-179`. Its reads land a median 9.8 s early — `PAPER_PLAN.md:116`.
- **Old p50** = the same server with `HINTS=dist`, the client sending the picked p50 as a v1 action — `CAMPAIGN23_FINDINGS.md:67`; `DIST_HINT_DESIGN.md:88-90`, `:96-97`.

**The just-in-time routine**
- **JIT exact** ("new exact"; `jit_exact_d`, `jit_exact_h100`) = `three_tier_wc_oracle_v2_nu_owned` with exact hints:
  - a read starts at predicted admission − read time − 1 s;
  - the arrival gate skips an SSD lookup when the history is in memory to within 256 tokens, and caps lookups at 3;
  - the arrival hold waits until predicted admission − read time − 2 s — `CAMPAIGN23_FINDINGS.md:52-56`; `PAPER_PLAN.md:117`.
- **JIT p50** ("new p50"; `jit_p50_ctl`, `jit_p50_d`, `jit_p50_h100`) = the same server with the online p50 table. The deadline is the p50 and reads start p50 − p10 early; the plan expires at p50 + max(60 s, p50), and the session then counts as ended — `PAPER_PLAN.md:118`, `:143`; `CAMPAIGN23_FINDINGS.md:52-53`.
- **`jit_70`:** JIT timing without the gate and hold, a 75-minute run — `CAMPAIGN23_FINDINGS.md:116`, `:361`; `scripts/start_server.sh:174-175`.

**Campaign 26: expected-cost ("pricing the room") arms** (`engine_ec_20261111d`) — `results/campaign26/DECISIONS.md:99-116`

| Arm | What it is |
|---|---|
| `ec_r13` | rule 1 (expected-cost start at λ = 5e-5) and rule 3 (eviction by chance of return within 30 s, from three quantiles) |
| `ec_r1` | rule 1 alone |
| `ec_r3` | rule 3 under JIT timing |
| `ec_r134` | rules 1 and 3 with the price learned from harm |
| `ec_full` | `ec_r134` plus soft pins |
| `fq90` | fixed-quantile rival: JIT reads landing at the p90 |
| `ec_r134_c128` | the 128-user version |

- Every new arm is "today's JIT p50 arm ... with the flags above replacing its read timing and eviction" — `results/campaign26/DECISIONS.md:119`.
- Same-build, same-GPU controls: `jit_p50_ctl`, `hintfree_ctl`, `jit_p50_d`, `jit_exact_d` — `results/campaign26/DECISIONS.md:101-102`, `:109`, `:114`.
- Reference runs from earlier campaigns: `oracle_v2`, `oracle_v2_p50`, `baseline_c128`, `oracle_v2_c128`, `jit_p50_c128`, `jit_exact_c128` — `results/campaign26/extra_runs.txt:1-9`.

**Other arms and mappings**
- Older launcher arms: `hbm_lru` (single tier), `hbm_host` (L1+L2), `three_tier_to` (timeout prefetch policy), `_norq` (admission-time re-query off), `_ntpobs` (prefetcher observe-only), `three_tier_wc_wb` (write-back), `three_tier_wc_dl` and `three_tier_wc_poll0` — `scripts/start_server.sh:7-27`.
- Two method tables:
  - `PAPER_PLAN.md:112-120` gives each method's hint, when its read lands, what it evicts first, and its pins;
  - the paper's design section gives the limit cases: λ → 0, λ → ∞, exact forecasts → Belady order, one shared distribution → LRU, and a fixed quantile — design.tex (scratch):362-371.

**Headline results, for scale (scoped; one run per arm unless noted)**
- **44 users, 70 GB** (campaign 23, 7,768 shared turns): baseline 5.17 s, oracle 1.48, old p50 1.95, JIT exact 0.50, JIT p50 1.51, hint-free 1.47 — `CAMPAIGN23_FINDINGS.md:13-20`, `:75`.
- **128 users, 160 GB** (8,175 shared turns): baseline 53.12 s, hint-free 29.19, hinted arms 12.06-12.56 — `CAMPAIGN23_FINDINGS.md:92`. Over the first 6,900 s, campaign 26's `nofc_noend_c128` lands at 20.84 s, inside the hinted arms' 20.58-20.98 — `results/campaign26/DECISIONS.md:594-606`.
- **Campaign 26, 44 users** (6,715 shared turns; runs separated by commas): original HiCache 5.47, 6.62 s; hint-free 1.61, 1.89; JIT p50 1.35-1.60; `ec_r13` 0.66, 0.78; `nofc` 0.69, 0.80; `nofc_noend` 0.71; JIT exact 0.51, 0.54 — `results/campaign26/DECISIONS.md:155-159`, `:579`, `:621`.
  - "nothing among nofc, nofc_noend, ec_r1 and rules 1 and 3 is shown in TTFT at this setting" — `:586-588`.
- **100 GB at 44 users:** no hints, JIT p50 and JIT exact tie at 0.41 s, with under 1 % of returning turns reading from SSD — `results/campaign26/DECISIONS.md:1156-1189`.

### Inferences
- "Hint-free" names two different arms: campaign 25's pins-only `three_tier_wc_dl_poll0` and campaign 26's queue-aware `nofc_noend`. Any comparison must say which one (inference).

### Gaps
- I did not read the full campaign 26 entries for `ec_full`, `fq90`, `ec_r3` and `nofc_c128` (`results/campaign26/DECISIONS.md:835-1041`), so their results are not summarised here.
- No arm ran at ×70 with 128 users and 160 GB in campaign 26 ("not queued") — `results/campaign26/DECISIONS.md:500-501`, `:569-570`.
- Original HiCache was not run at 100 GB — `results/campaign26/DECISIONS.md:1206-1207`.

## 10. Artifacts: what would let someone reproduce a campaign (and the document register)

### Takeaway
Each boot writes a run directory (server log, args, run facts), and each client run writes `client.jsonl`. Each campaign keeps a folder with its decision log, manifest, per-arm configs, queue, launcher and driver logs, iostat log, and a tarball of the scripts at launch.

A campaign cannot be fully reproduced from this checkout alone:
- the frozen engine copies, the campaign 26 launcher and most comparison and eviction-replay scripts sit in `frozen/`, which is git-excluded and absent here;
- the paper's design section is not in the repository.

### Cited Findings
- **Per-boot run directory** `results/<UTC yyyymmdd_HHMMSS>_<ARM>_<LEVEL>/` with `server.log`, `server.pid`, `server_args.txt`, `run.txt` and `l3_extra.json` — `scripts/start_server.sh:35-36`, `:211-212`, `:228`.
- **Client output** `<run dir>/client_<HHMMSS>_<TAG>/client.jsonl` and `client.log` — `scripts/start_client.sh:29-30`, `:83`, `:88-89`.
- **Campaign 26 folder** — `results/campaign26/manifest.txt:1-21`; `results/campaign26/DECISIONS.md:8-13`; directory listing. It holds:
  - `DECISIONS.md`;
  - `manifest.txt` (21 runs: label, run directory, client directory);
  - `config_*.txt` (one line per arm with every knob);
  - `queue.txt` and its backups, `launcher.log`, `driver_*.log`;
  - `iostat_vdc.log`, `scripts_at_launch.tgz`, `extra_runs.txt`;
  - `checks/`: `boot.py`, `latency.py`, `calm_reads.py`, `holds.py`, `limits/`, `verify/`, `audit/`.
- **Campaign 23:** run directories `CAMPAIGN23_FINDINGS.md:353-361`; tables in `results/campaign23/`; figures in `results/compare_c23_c44/` and `results/compare_c23_c128/` — `:363-366`.
- **Driver and relaunch:** `run_compare.sh` knobs and `config.txt`, and the full command for a campaign-14-sized run — `REPLAY_REDESIGN.md:372-380`, `:568-575`. Relaunching campaign 26 after an outage — `results/campaign26/DECISIONS.md:12-13`.
- **Figures from the repository:** `scripts/c26_pricing.py` "computes every column of the box's CSVs the same way (byte-identical on the box's runs)" — `scripts/c26_pricing.py:1-16`.
- **CPU models:** `cd agent_cache/scripts && python3 room_model.py measure|calibrate|policies|figure`, and `python3 prefill_curve.py fit|replay` — `results/cpu_calib_c44/FINDINGS.md:196-197`. Outputs: `calibration.csv`, `policies.csv`, `prefill_curve.csv` — `:187-194`.
- **Traces:** `traces/lmcache_agentic_trace_262k.json` with its `.stats.json` and `.reply_keys.json`, plus the 32K and `ge45_tl_s1` traces, are in the repo (directory listing). The raw parquet is not — `.gitignore:8-10`.
- **Not in this checkout:** `frozen/`, which holds the engine copies, the launcher `frozen/oracle_scripts_staging/c26_ec/c26.sh`, the comparison scripts, the eviction replay and the hint-realism study — `results/campaign26/DECISIONS.md:8`, `:67`; `CAMPAIGN23_FINDINGS.md:369`; `REALISTIC_HINTS.md:8-9`. The `.out` tables are git-ignored — memory note `c26-100gb-conclusion-audit.md:31`.
- **Engine and tests:** `python/` at a5fe6121c8 equals engine e — `results/campaign26/DECISIONS.md:67`; unit tests are in `test/registered/unit/mem_cache/` — `:93-95`.
- **Box paths:** host `/home/wanhr/sglang/agent_cache` is container `/sgl-workspace/sglang/agent_cache` — `REPLAY_REDESIGN.md:9-10`.
- **The paper's design section is missing from the repository:**
  - the memory note says it is at `agent_cache/paper/design.tex` — `paper-design-section.md:11` (outside the repo);
  - that path does not exist, and `git log --all -- agent_cache/paper` returns nothing;
  - the only copy I found is in the session scratchpad (`design.tex (scratch)`, mtime 2026-11-17 14:24), which is not durable.

**Document register** (the document's own date, the latest date it mentions, and its last commit from `git log -1`):

| Document | Own date | Latest date mentioned | Last commit |
|---|---|---|---|
| `PAPER_PLAN.md` | started 2026-11-01 (`:3`) | CPU calibration 2026-11-07 (`:192`) | 2026-11-10 (b684fd6464) |
| `PAPER_INTRO.md` | started 2026-11-06 (`:3`) | prior work checked 2026-11-06 (`:161`) | 2026-11-10 |
| `REPLAY_REDESIGN.md` | 2026-10-17 (`:1`) | checks of 2026-10-17 (`:650`); the dist-hint section §3.10 was added later | 2026-10-31 |
| `RUNBOOK.md` | written 2026-09-16, ported 2026-09-21 (`:3-4`) | campaign 7, 2026-09-22 (`:2450`) | 2026-09-22 |
| `agent-kv-tiering-evaluation-starter-kit.md` | 14 September 2026 (`:3`) | same | 2026-09-18 |
| `REALISTIC_HINTS.md` | written 2026-10-28 (`:3`) | 2026-10-28 | 2026-10-31 |
| `DIST_HINT_DESIGN.md` | 2026-10-31 (`:4`) | engine built 2026-11-01 (`:83`, `:134`) | 2026-11-02 |
| `CAMPAIGN23_FINDINGS.md` | last arm 2026-11-02 | updates of 2026-11-04 and 2026-11-05 (`:3-9`) | 2026-11-05 |
| `results/cpu_calib_c44/FINDINGS.md` | 2026-11-07 (`:3`) | addendum 2026-11-16 (`:151`) | 2026-11-17 (e96a6fea1c) |
| `results/campaign26/DECISIONS.md` | started 2026-11-11 02:49Z (`:3`) | end state 2026-11-14 01:38Z (`:1211`) | 2026-11-17 |
| `scripts/replay_agentic.py`, `start_client.sh`, `gap_hints.py` | — | — | 2026-10-31 |
| `scripts/start_server.sh` | — | — | 2026-11-13 |
| `scripts/convert_lmcache.py`, `replay_template.py` | — | "seen 2026-09-18" (`scripts/replay_template.py:33`) | 2026-09-18 |
| `design.tex (scratch)` | not in the file; written 2026-11-16 per memory note `paper-design-section.md:11` | — | not in git |

### Inferences
- Most campaign 23-26 tables cannot be regenerated from this laptop's checkout. The comparison scripts and engine freezes are on the box under `frozen/`; the raw run directories exist under `results/` (inference from the absent `frozen/`).

### Gaps
- Whether the box still holds `frozen/` and the `.out` tables is not stated in the repo.
- No durable copy of the paper's design section was found.

## 11. Limitations stated in the documents, and stale or conflicting statements

### Takeaway
The documents name these limits themselves:
- a batch-agent trace with gaps stretched ×10 or ×70 and capped at 1,200 s, replayed in a closed loop with no human waits;
- forced, content-free outputs, with the reply key read from the replay rather than parsed by the engine;
- hints delivered idealised, at commit;
- at 128 users the GPU queue, not the SSD, sets TTFT, and the formulation takes the queue as given;
- the SSD is not stressed in the latest settings;
- one run per arm against run-to-run moves of 0.12-1.15 s on the same turns;
- one trace, one seed, one model, TP 1;
- settings that change users and memory together;
- the client's backlog cap was removed in October, so new runs do not pair with campaigns 13-17;
- the CPU models hold eviction and queue waits fixed.

Several planning documents are stale against campaign 26 and the 2026-11-16 prefill re-pricing.

### Cited Findings
**Workload realism**
- "A batch-agent trace with gaps stretched by 10, in a closed loop. Real use has long human waits" — `CAMPAIGN23_FINDINGS.md:287`.
- "Real use is 92 % human waiting with median contexts of 126K tokens (TraceLab); our trace is batch agents with gaps stretched by a factor" — `PAPER_PLAN.md:26`. TraceLab's numbers — `REALISTIC_HINTS.md:340-347`.
- Reviewers will check "gaps scaled 10x or 70x, and no workload paced like a human user" — `PAPER_INTRO.md:124`; quote replay numbers as ratios or state the gap scale — `PAPER_INTRO.md:157`.
- Distortions from the trace and the replay:
  - "x70 manufactures hint value" — `REALISTIC_HINTS.md:564-565`, `:894`;
  - the 1,200 s cap removes 49.6 % of ×70 idle time — `REALISTIC_HINTS.md:895-896`; `REPLAY_REDESIGN.md:163-175`;
  - the 300 s spike — `REALISTIC_HINTS.md:897-898`;
  - the unresolved 50-call cap — `REALISTIC_HINTS.md:899-900`.
- ×10 "is a forgiving test", because queue waits cover forecast errors — `DIST_HINT_DESIGN.md:104-108`.
- At most one user per conversation, so "An open-loop steady state like campaign 13's ... cannot be run" — `REPLAY_REDESIGN.md:485-489`.
- Thinking is always off — `REPLAY_REDESIGN.md:458-462`.

**Forced outputs and idealised hints**
- The reply key comes from the replay; "the engine has never read its own reply on a GPU" — `PAPER_PLAN.md:17`; `scripts/gap_hints.py:19-22`. An engine-parsed key would need a reply shim — `REALISTIC_HINTS.md:588`.
- Hints are read at commit, as if delivered with no delay — `REALISTIC_HINTS.md:83-85`, `:525-527`. "Piggybacking cannot emulate updates during the wait" — `:527-528`.
- The exact end signal is realistic only for batch agents — `REALISTIC_HINTS.md:531-533`.
- The online table starts cold in every run and is "Not modelled" across restarts — `DIST_HINT_DESIGN.md:74`.
- The no-hint rival "still takes the session id ... from the hint channel" — `results/campaign26/DECISIONS.md:507`.

**The GPU queue and the SSD**
- At 128 users "TTFT there is the GPU's queue" — `PAPER_PLAN.md:15`; "At 128 users the GPU queue decides TTFT" — `CAMPAIGN23_FINDINGS.md:242-246`.
- The formulation takes the queue as given ("The queue is taken as given") — `PAPER_PLAN.md:135`; design.tex (scratch):128-130, 178-181.
- A recomputed token's full price is "roughly 6-8 times its own prefill" once its queue effect counts — `results/cpu_calib_c44/FINDINGS.md:108-111`.
- "Neither setting stresses the SSD (the read thread is 16-21 % busy at 44 users; the GPU queue decides at 128)" — `results/campaign26/DECISIONS.md:500-501`, `:292`.
- At 100 GB the tie "is from having nothing to read" — `results/campaign26/DECISIONS.md:1201-1205`.

**Noise and scope**
- Few repeats — `CAMPAIGN23_FINDINGS.md:274-278`. Original HiCache "varies by about 25 % run to run" — `PAPER_PLAN.md:24`. Differences under about 0.25 s are not results — `results/campaign26/DECISIONS.md:142`. The bootstrap interval is a floor — `:378-381`. "One run per arm cannot carry 0.05-0.15 s" — `:497`.
- One trace, one seed — `results/campaign26/DECISIONS.md:505`, `:1206`. TP 1 only — `REALISTIC_HINTS.md:907-908`.
- GPU units change across restarts, and older runs' units are unrecorded — `CAMPAIGN23_FINDINGS.md:282-286`. Changes arrive together across engine builds — `:279-280`.
- "The settings differ in users and memory at once" — `CAMPAIGN23_FINDINGS.md:281`. The 44K room is the 70 GB pool minus the mirror; a pool sweep at 44 users was not run when the reviewer raised it — `results/campaign26/DECISIONS.md:498-499` (three 100 GB arms ran later, `:1156`).
- "No rule is isolated" — `results/campaign26/DECISIONS.md:502`. "The price was fitted on the turns it is scored on and has no sensitivity run" — `:504`.
- "TTFT of returning turns is not what the agent gets": turn latency and throughput do not follow TTFT among the leading arms — `results/campaign26/DECISIONS.md:496`, `:473-488`.

**The removed client backlog cap**
- Commit e32458cc9f (2026-10-17) cut the client to one design and removed the backlog-driven cap, the stationary start, the per-user turn draws and other knobs — `REPLAY_REDESIGN.md:3-7`, `:24-25`, `:76`.
- "New runs do not pair with campaigns 13-17"; "A new run is a new workload: compare new arms with each other" — `REPLAY_REDESIGN.md:29-33`, `:419-424`.
- PAPER_INTRO's Figure 1 comes from campaign 14, on an older engine with "the old client's backlog cap". There, "lookups are refused" is inferred from code, "since that log has no refusal event", and the figure should be remade — `PAPER_INTRO.md:141`, `:149-150`.
- Before `limit=0`, aiohttp's 100-connection pool queued requests inside the client (an ×1 campaign) — `scripts/replay_agentic.py:468-470`.

**CPU model limits**
- In the room model, eviction, queue waits and re-reads come from the run, the room is one number, and TTFT is out of reach — `results/cpu_calib_c44/FINDINGS.md:142-149`.
- The first-order check models no host room, revokes or queue — `PAPER_PLAN.md:174`.
- `toy_floor.py` has no queue and no SSD contention — `scripts/toy_floor.py:8-9`.
- "The simulation has no plan expiry" — `REALISTIC_HINTS.md:883-884`.

**Rate limits**
- No document states an API or request rate limit for the workload. The only "limit" is the usage limit that cut off two analysis agents — `results/campaign26/DECISIONS.md:524-525`.

**Stale or conflicting statements (flags)**
- **The flat prefill rate and λ = 5e-5 are superseded.**
  - `PAPER_PLAN.md` uses ρ ≈ 9K tokens/s (`:44`, `:46`), a linear TTFT term (Δ + U)/ρ (`:64`) and λ = 5e-5 (`:98-102`, `:189`, `:194`).
  - The 2026-11-16 addendum fits T(L) = 0.092 + 0.84e-6 L + 3.21e-9 L², says "9K tokens/s is the average of a 32K prefill", and puts the GPU arms' best λ at 1e-4 with soft pins: this "replaces 'start λ near 5e-5'" — `results/cpu_calib_c44/FINDINGS.md:151-158`, `:181-183`.
  - The same file's 2026-11-07 tables and plan item 1 still use the flat rate and 5e-5 (`:63-64`, `:128`).
  - The paper's design section uses the quadratic form — design.tex (scratch):96-105.
  - Campaign 26 ran at a fixed 5e-5, or harm-learned from 5e-5 (`results/campaign26/DECISIONS.md:52`, `:103`, `:110`), so it predates the re-pricing. No GPU run at the re-priced λ is documented (inference).
- **`PAPER_PLAN.md` predates campaign 26:**
  - it lists the fixed-quantile rival as "not run" (`:119`), but `fq90` ran (`results/campaign26/DECISIONS.md:111`);
  - its "Next" engine and GPU steps (`:201-206`) were done in campaign 26 (`results/campaign26/DECISIONS.md:47-65`);
  - its claim of 79 % without hints and up to 42 % for hints at 128 users (`:7`, `:18`) is superseded: the queue-aware no-hint arm reaches 96 %, and "The 79 % of 2026-11-04 was that arm's figure" (`results/campaign26/DECISIONS.md:338-340`); at 128 users the no-hint rival lands inside the hinted range (`:604-606`);
  - the outside reviewer: "PAPER_PLAN.md's claim and PAPER_INTRO.md's '2.9x from exact return times' do not survive that rival (1.4x, exact hints only)" — `results/campaign26/DECISIONS.md:493-495`.
- **`PAPER_INTRO.md`** says contribution 3 is "Status: offline only" (`:107-110`), but it was built and run in campaign 26. Its Figure 1 run is old by its own note (`:149-150`).
- **`RUNBOOK.md`** gives the evaluation model as Qwen3-32B-FP8 with fp8_e5m2 KV (`:6`). Campaigns since 7 use Qwen3-30B-A3B-Instruct-2507 with bf16 KV (`RUNBOOK.md:2458`; `results/campaign26/config_ec_r13.txt:1`). Its §4.6 client reference and several sections still show removed flags — `REPLAY_REDESIGN.md:5-6`, `:442-456`.
- **`start_server.sh` defaults** (Qwen3-32B-FP8, CTX 32768, fp8_e5m2 KV "every campaign so far", the nothink template) do not match any recent campaign, which override them through the environment — `scripts/start_server.sh:32-33`, `:43-46`; `results/campaign26/config_ec_r13.txt:1`.
- **`start_client.sh:8`** says "campaigns 13-17 used the 262K one"; campaigns 18-26 used it too (config files).
- **`REPLAY_REDESIGN.md`** has internal drift:
  - "Eight knobs" (`:18`) against "nineteen flags" (`:87`);
  - "393 lines" (`:52`) against `replay_agentic.py`'s 494 lines today (`wc -l`);
  - its smoke and campaign steps say "not yet run with the new client" (`:495`, `:559`), though campaign 18 ran on the redesigned client (`REALISTIC_HINTS.md:98`).
- **`DIST_HINT_DESIGN.md`** says the engine is "built 2026-11-01, not committed, unit-tested, not yet run on a GPU" (`:83`) and "the engine is unchanged" (`:4`). It was committed at 03afa49778 on 2026-11-02 and ran in campaign 23 — `CAMPAIGN23_FINDINGS.md:313-314`.
- **`CAMPAIGN23_FINDINGS.md`** option I (a queue-aware hint-free arm) is "not built" (`:303`); it was built as engine e and run as `nofc_noend` — `results/campaign26/DECISIONS.md:32`, `:106`, `:125-127`.
- **Dataset counts:** the starter kit's 787 sessions / 24,881 iterations / 669 SWE-bench / 85 GAIA (`agent-kv-tiering-evaluation-starter-kit.md:13`) do not match the converter's 767 session ids / 24,880 rows / 665 / 94 chains (`traces/lmcache_agentic_trace_262k.json.stats.json:17-21`, `:76`, `:103`).
- **SSD bandwidth:** 1.88 GiB/s each way by dd (`RUNBOOK.md:118`), 2.11 GB/s shared by reads and writes (`PAPER_PLAN.md:43`; `PAPER_INTRO.md:140`), and a read ceiling near 1.85 GB/s (`CAMPAIGN23_FINDINGS.md:264`). These are different measures.
- **The room at 44 users / 70 GB:** a static R of 43,968 tokens (`PAPER_PLAN.md:41`), a fitted effective room of 90K (`results/cpu_calib_c44/FINDINGS.md:43`, `:57-59`), and a measured median of 54K, 40-46K in the load episodes (`results/campaign26/DECISIONS.md:27`, `:293-296`).
- **Warm-up:** `DIST_HINT_DESIGN.md:74` says campaign metrics start at 45 min. The campaign 23 and 26 TTFT tables state no time cut (`CAMPAIGN23_FINDINGS.md:70-71`; `results/campaign26/DECISIONS.md:151`).
- **The paper's design section vs what ran:**
  - The paper evicts by a return-rate retention index ν with a max over horizons (design.tex (scratch):301-314). The engine ran rule 3 as a fixed-horizon chance of return within 30 s (`results/campaign26/DECISIONS.md:58-60`). The deck's return-rate order "reads on 29.8 %, worse than LRU ... It is not built" (`:40-41`). The memory note says the evaluation must state which form ran (`paper-design-section.md:23`).
  - The paper's price update counts preempted reads and evicted prefix caches in demand, starts by bisection, and recalibrates keys (design.tex (scratch):338-357). The cold-start bisection, per-key recalibration and counting evicted histories as refused were not built (`results/campaign26/DECISIONS.md:62-63`). The built `on` rule moves the price up while reads are refused and down while room sits idle; `harm` is the alternative — `python/sglang/srt/arg_groups/fields/memory.py:303-317`.
- **Corrections inside `results/campaign26/DECISIONS.md`:**
  - the 100 GB forecast "was wrong by about six" — `:1193-1197`;
  - more errata, listed only in a memory note (outside the repo): "385K idle at the median" is a mean (median 360K); "0.31-0.42 s" should read 0.30-0.42; "0.10-0.21 s" should read 0.10-0.22; "14 % of turns" has no traceable source; the read and pin budgets also scale with the 100 GB pool — `c26-100gb-conclusion-audit.md:21-28`.

### Inferences
- For comparison with published papers, the main differences in kind (inference):
  - our "users" include idle sessions under a hard slot cap, and admission waits sit outside TTFT;
  - outputs are length-forced, so decode load matches the trace but content does not;
  - hint quality is realistic only to the extent of an online table on a batch-agent trace at ×10;
  - the newest result is that a queue-aware engine with no gap forecast reaches most of the gain, so hint value is small at the tested settings:
    - exact hints over no hints: 0.10-0.21 s at 44 users / 70 GB (`results/campaign26/DECISIONS.md:1205`; 0.10-0.22 s per the memory note);
    - no separation at 100 GB (`:1177-1179`);
    - at 128 users the no-hint rival lands inside the hinted arms' range (`:604-610`).

### Gaps
- Rate limits: not stated.
- No document quantifies how far the forced, content-free outputs change engine timing compared with free generation.
- The open items `PAPER_PLAN.md:22-28` lists for the paper are not reported as done after campaign 26: sessions per GPU under a latency target, a human-paced workload, and repeats of the 128-user arms.
