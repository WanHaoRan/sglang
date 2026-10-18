# Replay client redesign (2026-10-17)

Commit `e32458cc9f` ("redesign the trace replay scripts") cut the replay client `scripts/replay_agentic.py` and its
wrappers `scripts/start_client.sh` and `scripts/run_compare.sh` down to one workload design. This document says what the
client does now, what was removed, how a run of the new client relates to campaigns 13-17, how the change was checked,
and what is still open. Until RUNBOOK §4.6 is rewritten (§8.2), it is the client's reference. The client before the cut
is `c20643b695:agent_cache/scripts/replay_agentic.py`.

Paths: on the H200 box, host `/home/wanhr/sglang/agent_cache` is container `/sgl-workspace/sglang/agent_cache` (container
`sglang_hicache`). Code anchors `path:line` are relative to `agent_cache/` at `e32458cc9f`. Every number is tagged
measured (date and source), derived (formula) or estimate.

## 0. The short version

- **The workload.** Each user replays one recorded conversation as a closed loop: send a turn, stream the reply, append
  it verbatim to the history, sleep the recorded gap × `GAP` (at most 1,200 s), send the next turn. No two users replay
  the same conversation.
- **Eight knobs:** trace, users, turns per user, arrival rate, gap scale, oracle hints, a hard cap on users in session,
  and a timeout (§2). The other flags are seed, URL, tag and output file.
- **The cap counts idle users.** A user holds its slot through every sleep, so the cap fills long before the server
  does: with campaign 14's arrivals and ×70 gaps, `C=128` keeps 615-632 of the 768 users waiting for admission, outside
  every TTFT (§3.5).
- **Removed:** the backlog-driven cap, the stationary start, per-user turn-count draws, arrivals past one user per
  conversation, task chaining, context policies, recompute controls, the token-id reuse check, `--offset`/`--order`,
  `--stop-after-users`, the `--gap-cap` flag (now fixed at 1,200 s), `--thinking` (now always off), `--model` (now read
  from the server), and the offline `--dry-run`/`--validate` (§1.2).
- **Old knobs fail loudly.** Both wrappers stop before booting anything when a retired knob is set (§6.3), and the
  client rejects a removed flag with argparse's usage error. `run_campaign12/13/14.sh` therefore stop at once.
- **New runs do not pair with campaigns 13-17.** Against campaigns 14-17, at the same seed and rate the users get the
  same conversations and arrival times (§7.1), but there is no stationary start, no 30-50 turn draw and no backlog cap
  (§7.2). Against campaign 13 only the arrival times match (at its 0.2/s): its users cycled through 128 conversations
  (`OFFSET=32 NCONV=128`) under a fixed cap of 128 until 384 had finished. To rerun one of those campaigns, restore the
  old scripts (§9.3).
- **Open decisions:** the fixed gap cap rules out ORACLE_PLAN's sparse regime as written (§8.1); RUNBOOK, ORACLE_PLAN and
  ORACLE_V2_HANDOFF still show removed flags (§8.2).
- **Run it:** first check that the box's checkout has the redesign (§9.1, first block: without it the wrappers are the
  old ones and run a different workload without an error); smoke run §9.1 (~4 min with the boot, GPU busy); compare
  campaign §9.2 (~5 h per arm, GPU busy).

## 1. What changed

The client had grown to 1,462 lines and 53 flags over campaigns 7-17, and `start_client.sh` to 46 environment knobs.
Most flags served one campaign, and several interacted: the backlog cap with the stationary start, the context policies
with task chaining, `--order` with `--arrival-until`. The redesign keeps the knobs of one design (2026-10-17: "the
trace, user numbers, each user's conversation turns, arrival rate, gap scale, oracle hints, a hard concurrency cap ...,
and a timeout"; each user's conversation different), each with one meaning.

### 1.1 Files in `e32458cc9f`

| file | before → after | change |
|---|---|---|
| `scripts/replay_agentic.py` | 1,462 → 393 lines; 53 → 12 flags | rewritten (§2-§4) |
| `scripts/start_client.sh` | 113 → 68 lines; 46 → 10 knobs | new knobs, the retired-knob guard, per-turn view without the `reply_reused` column |
| `scripts/run_compare.sh` | 79 → 71 lines | new client knobs and defaults, a trace pre-flight in place of `VALIDATE_ONLY`, the guard |
| `scripts/timeline.py` | 2 lines | reads `users` records as the cap series (`:106`); with no cap and an arrival rate, the compare figure's first-turn panel says "uncapped sessions" instead of "up to 0 live sessions" (`:461`) |
| `scripts/c16_figure.py` | 1 line | reads the `waiting` count of `users` records as it read that of `cap` records (`:33`) |
| `scripts/steady_state.py` | 1 line | exit message for a run with no cap (`:55`) |

Sizes measured 2026-10-17 with `wc -l` and `grep -c ap.add_argument` on `c20643b695` and `e32458cc9f`.

### 1.2 Removed flags and knobs

| old flag (old `start_client.sh` knob) | what it did | now |
|---|---|---|
| `--num-conversations` (`NCONV`) | conversations to replay: the first N after `--offset` (0 = all), one user each unless `--arrival-until` cycled more users through them (campaign 13: 128 conversations, 511 users started per finished arm) | `--num-users` (`USERS`) |
| `--max-turns` (`TURNS`) | turns per user, at most | `--turns` (`TURNS`), same meaning |
| `--max-seconds` (`MAX_SECONDS`) | wall cap | `--timeout` (`TIMEOUT`); `run_compare.sh` still takes `CLIENT_TIMEOUT` |
| `--offset`, `--order`, `--order-seed` (`OFFSET`, `ORDER`, `ORDER_SEED`) | which conversation each user replays: rotation, cycle, shuffle or fresh | always the old `--order fresh`, seeded by `--seed` (§3.1) |
| `--gap-cap` (`GAP_CAP`) | cap on every sleep, 0 = none | fixed at 1,200 s (`scripts/replay_agentic.py:43`) |
| `--gap-sample` | sleeps drawn from a list | removed |
| `--thinking` (`THINKING`) | `enable_thinking` in the request | always false (§3.8) |
| `--model` (`MODEL`, in `start_client.sh` only) | model name sent | the first model of `/v1/models` |
| `--turns-range`, `--turns-seed` (`TURNS_RANGE`, `TURNS_SEED`) | each selected conversation replayed min(recorded, a seeded draw in LO:HI) turns, one draw per conversation in selection order (users replaying the same conversation shared it) | removed |
| `--arrival-until`, `--progress-every` (`ARRIVAL_UNTIL`) | users keep arriving until T, cycling through the conversations | removed: at most one user per conversation; a progress line every 60 s |
| `--stop-after-users` (`STOP_AFTER_USERS`) | stop once N users finished | removed |
| `--cap-mode`, `--cap-target`, `--cap-interval`, `--cap-gain`, `--cap-step`, `--cap-min`, `--cap-max`, `--admit-spacing` (`CAP_MODE`, ...) | backlog-driven user cap | removed: `--concurrency` is a fixed cap (§3.5) |
| `--start-phase`, `--start-users`, `--start-seed`, `--start-service` (`START_PHASE`, ...) | first users start mid-session | removed: every user starts at turn 0 |
| `--session-tasks`, `--session-seed`, `--task-gap` (`SESSION_TASKS`, ...) | several conversations chained into one user | removed |
| `--context-policy`, `--max-context`, `--context-trigger`, `--context-margin`, `--compact-tokens`, `--compact-keep`, `--compact-mode`, `--truncate-to` (`CONTEXT_POLICY`, ...) | what a session does when its prompt outgrows the window | removed |
| `--control-every` (`K`) | recompute control requests | removed |
| `--check-ids` (`CHECK_IDS`) | exact token-id check of reply reuse | removed; `reply_reprefilled` measures reuse (§4) |
| `--save-text` | reply text in the JSONL | removed |
| `--dry-run`, `--dry-run-max-chars`, `--validate` (`VALIDATE_ONLY`) | offline print or check | removed (§8.4) |

## 2. Flags and knobs

The client takes twelve flags. The wrappers set them from environment variables: `start_client.sh` defaults to a
4-user smoke run, and `run_compare.sh` exports a 48-user default before calling it.

| flag | client default | `start_client.sh` knob, default | `run_compare.sh` default | meaning |
|---|---|---|---|---|
| `--trace` | required | `TRACE`, `traces/lmcache_agentic_trace.json` | same | the converted trace (`convert_lmcache.py`) |
| `--num-users` | 0 = one per conversation | `USERS`, 4 | 48 | users, each on a different conversation |
| `--turns` | 0 = all | `TURNS`, 10 | 24 | each user replays its first min(N, recorded) turns |
| `--arrival-rate` | 0 = all at once | `ARRIVAL`, 0 | 0 | users per second, Poisson, the first at 0 s |
| `--gap-scale` | 1.0 | `GAP`, 1.0 | 21 | sleep before a returning turn = recorded `pre_gap` × this, at most 1,200 s |
| `--oracle-hints` | off | `ORACLE_HINTS`, 0 (1 = on) | 0 | every turn but a user's last sends the next turn's request id and sleep as `kv_hints` |
| `--concurrency` | 0 = no cap | `C`, 4 | 48 | most users in session at once, idle ones included |
| `--timeout` | 0 = none | `TIMEOUT`, 0 | `CLIENT_TIMEOUT`, 0 | seconds after the start; after it no user starts and no request is sent |
| `--seed` | 0 | `SEED`, 0 | 0 | the arrival times and which user replays which conversation |
| `--url` | `http://127.0.0.1:30000` | port 30000, fixed | same | the server |
| `--tag` | `replay` | `TAG`, `smoke` | `cmp_<arm>` | request ids are `<tag>-c<user>-t<turn>` |
| `--output` | required | `<server run dir>/client_<UTC HHMMSS>_<TAG>/client.jsonl` | same | JSONL, appended |

A negative value for any numeric flag but `--seed` exits with `no flag takes a negative value`
(`scripts/replay_agentic.py:78-79`); `--seed` takes any integer.
`--num-users` above the trace's conversation count exits before the server is contacted (`:101-102`).

The traces (measured 2026-10-17 by counting each trace's `conversations`):

| trace | conversations | turns | most turns in one |
|---|---|---|---|
| `traces/lmcache_agentic_trace.json` (the 32K cut; the wrappers' default) | 731 | 17,887 | 50 |
| `traces/lmcache_agentic_trace_262k.json` (campaigns 13-17) | 768 | 24,879 | 50 |
| `traces/lmcache_agentic_trace_262k_ge45_tl_s1.json` (ORACLE_PLAN E1/E2) | 283 | 13,989 | 50 |

## 3. What the client does

These are the rules needed to read a run; each cites the lines that implement it in `scripts/replay_agentic.py`.

### 3.1 Users and conversations

User i (its id is its arrival rank, the `c<i>` of its request ids) replays conversation `pool[perm[i]]`, where `pool`
lists the trace's non-empty conversations and `perm = random.Random(f"{seed}:0").sample(range(P), P)` for
P = len(pool) (`:99-106`). This is the old `--order fresh --order-seed S`, and the arrivals use the old
`random.Random(S)` stream, so at the same seed users get the conversations and arrival times of campaigns 14-17, which
ran S = 0 (§7.1).

### 3.2 Turns

A user replays the first min(`--turns`, recorded) turns of its conversation, from turn 0 (`:106`). Turn t's prompt is
the whole history: the earlier turns' messages, the earlier replies exactly as the server streamed them, then turn t's
new messages (`:296`, `:310`). `max_tokens` is the recorded `output_length` and `ignore_eos` is on, so a reply has the
recorded length. The server does not cut a reply at the context window (`start_server.sh` passes no
`--allow-auto-truncate`): a prompt plus `max_tokens` over the context length is refused with HTTP 400, which ends the
user (§3.9). The largest such sum is 32,255 tokens in the 32K trace and 109,430 in the 262K traces (measured
2026-10-17), inside `CTX=131072`. `short_reply` (`:264-265`) marks a reply the server shortened anyway: the scheduler
clips `max_tokens` when the prompt plus the reply would come within 2 tokens of the window, or would not fit a KV pool
smaller than the window.

### 3.3 Arrivals

With `--arrival-rate R > 0`, user i arrives at the i-th point of a Poisson process of rate R, the first at 0 s
(`:83-91`). With R = 0 every user arrives at 0 s, all in the same event-loop step and in user order (`:351-353`); they
all start then unless the cap holds them: with `C` below the number of users, the first C start at 0 s and the others
wait for slots (§3.5).

### 3.4 Gaps and the 1,200 s cap

Before turn t > 0 the user sleeps min(1,200, `pre_gap` × `--gap-scale`) seconds, counted from the end of turn t-1's reply
stream (`:136-137`, `:289-291`). Turn 0's `pre_gap` is not slept: the arrival places a user's first turn.
`--gap-scale 0` removes every sleep.

The cap is the constant `GAP_CAP_S = 1200.0` (`:43`), the value campaigns 13-17 passed as `GAP_CAP=1200`. The run
record repeats it as `args.gap_cap`, which `gap_distribution.py:93` reads. At ×70, 57 of the 262K trace's 768
conversations hold a sleep over 5 h, the length of a campaign arm. How much idle time the cap removes depends on the
scale:

| trace, scale | returning turns | sleeps the cap shortens | idle time it removes |
|---|---|---|---|
| 262K, ×70 (campaigns 14-16) | 24,111 | 206 | 49.6 % |
| 262K, ×10 (campaign 17) | 24,111 | 74 | 30.0 % |
| `262k_ge45_tl_s1`, ×1 (ORACLE_PLAN E1/E2) | 13,706 | 188 | 13.7 % |

Derived 2026-10-17 over every returning turn of every conversation: idle removed = Σ max(0, `pre_gap` × scale − 1,200)
/ Σ `pre_gap` × scale.

### 3.5 The concurrency cap

`--concurrency C` caps the users in session. A user takes a slot just before it sends turn 0 and holds it, through every
sleep, until it ends (`:116`, `:235-246`). While C users hold slots, arriving users wait, and they are admitted in
arrival order as slots free. A returning turn never waits for the cap: the cap limits the population, not the requests
in flight. The wait is `conv_start.queued_s` and is outside every TTFT. `C=0` means no cap.

Because idle users keep their slots, a cap on users binds much sooner than a cap on requests in flight would. With campaign 14's arrivals
(768 users at 0.1/s, seed 0) and ×70 gaps, an uncapped run would hold 246 users in session at its peak with zero service
time, and 352 at 32 s per turn; under `C=128`, 615-632 of the 768 users wait for a slot (derived 2026-10-17: FIFO
admission over the trace's sleeps with 0-32 s of service per turn; §9.2).

This is campaign 13's fixed cap (the old `--cap-mode fixed`). Campaigns 14-17 used the backlog cap instead: it started at
128 and moved by up to 8 users every 120 s to hold the client-measured backlog of requests awaiting their first token
at 32, within 16-512, with at least 5 s between two user starts. In campaign 14's `three_tier_wc` arm it ranged 92-196,
mean 166.6 (measured: that run's summary record).

Admission in arrival order relies on `asyncio.Semaphore` handing a freed slot to the longest waiter, which Python
3.10.12 does (§8.5 says how to check the container's Python).

### 3.6 The timeout

With `--timeout T` (seconds after the client started):

- No user starts after T. Users waiting for a slot at T, and users arriving after T, get a `conv_skipped` record with
  `reason: "timeout"` (`:236-239`, `:348-350`).
- No request is sent after T. A sleep that would pass T ends at T, and the user ends with `conv_end` `capped: true`
  (`:289-295`). The one connection retry is not made after T (`:221`).
- A request in flight at T completes and is recorded.
- The summary's `capped` is true when a user was cut or skipped (`:335`), and `run_compare.sh` then prints
  `STAGE CAPPED <arm>`.

`t_send` is taken just before the connection opens, so "no request after T" holds to about a millisecond.

### 3.7 Oracle hints

With `--oracle-hints`, every turn but a user's last carries this `kv_hints` envelope (`:139-145`), the one campaigns
15-17 sent:

```json
{"protocol_version": "1", "message_id": "<tag>-c<user>-t<t>",
 "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn", "action_version": "1",
              "payload": {"session": "<tag>-c<user>", "next_rid": "<tag>-c<user>-t<t+1>", "next_gap_s": 12.345}}]}
```

`next_gap_s` is the sleep the client will take, after the scale and the cap. The envelope is left out when the timeout
will certainly cut the next turn (now + gap ≥ T, `:153`). Every turn record that could carry a hint has
`oracle_hint: {next_rid, next_gap_s, suppressed}` (`:154-155`).

### 3.8 The request

Every request is a streamed `/v1/chat/completions` call with `temperature 0`, `ignore_eos`, `max_tokens` = the recorded
`output_length`, `skip_special_tokens false`, `chat_template_kwargs.enable_thinking false`,
`return_cached_tokens_details true`, `stream_options.include_usage`, `rid` = the request id, and `model` = the first
model listed by `/v1/models` (`:161-175`, `:359-365`). Thinking off matches `start_server.sh`'s default template
(`templates/qwen3_replay_nothink.jinja`) and the stock template of a model without a think block (Qwen3-30B-A3B-Instruct-2507
with `TEMPLATE=`, campaigns 14-17). With no server, the client exits with
`STOP: no model at <url>/v1/models (...); is the server up?`.

### 3.9 Connections and errors

One aiohttp session serves every user, with no total timeout, a 3,600 s read timeout (a queued request sends nothing
until its first token) and no connection limit (`:372-373`). A connection error (any `aiohttp.ClientConnectionError`:
a keep-alive connection the server has closed, a refused connection, or 3,600 s without a byte, which aiohttp raises as
`SocketTimeoutError`) is retried once on a fresh connection with the same request id, and the turn record gets
`retried` (`:217-224`). When a retry completes, `t_send`, `ttft` and `latency` time the retry alone. While the server
still holds the first attempt (after a read timeout), it refuses the retry as a duplicate request id
(`python/sglang/srt/managers/tokenizer_manager.py:3613-3614`). Any other error (an HTTP status other than 200, an error
inside the stream, a connection dropped mid-reply) ends the user: the turn record gets `error`, and the `conv_end`
record has `aborted: true` (`:303-308`).

## 4. Output records

The client appends one JSON object per line to `--output`. Times are seconds since the client started, except
`run.t_start` (Unix time).

| kind | written | fields |
|---|---|---|
| `run` | once, first | `tag`, `t_start`, `model`, `args` (every flag, plus `gap_cap: 1200.0`), `users`, `turns` (planned) |
| `conv_start` | a user starts | `conv`, `task_conv` (index in the trace), `session_id`, `source`, `turns` (planned), `arrival_s` (scheduled), `start_s`, `queued_s` |
| `turn` | each request | `conv`, `turn`, `task_conv`, `source`, `gap_slept`, `pre_gap`, `output_length`, `replay_prompt_tokens`, maybe `oracle_hint` and `retried`; when it completed: `t_send`, `ttft`, `latency`, `prompt_tokens`, `completion_tokens`, `cached_tokens`, `cached_details` (the server's `cached_tokens_details` as sent: `device` and `host`, plus `storage` and the string `storage_backend` with an L3 backend; null when the server sends none), maybe `short_reply`, and on a returning turn `delta_tokens` and `reply_reprefilled`; when it failed: `error` |
| `conv_end` | a user ends | `conv`, `turns_done`, `aborted`, `capped` (only when true), `end_s` |
| `conv_skipped` | a user never starts | `conv`, `reason` (`timeout`), `arrival_s` |
| `users` | every 60 s | `t`, `cap` (C, or null), `live`, `waiting`, `in_flight` (requests), `admitted` and `finished` (users since the previous record) |
| `summary` | once, last | `tag`, `elapsed_s`, `users`, `users_started`, `finished_users`, `skipped_users`, `turns`, `errors`, `retries`, `capped_convs`, `capped`, `mean_ttft_s`, `mean_reply_reprefilled` |

`reply_reprefilled` = `prompt_tokens` − `cached_tokens` − `delta_tokens`, where `delta_tokens` = `prompt_tokens` −
(the previous turn's `prompt_tokens` + `completion_tokens`): the history tokens the server recomputed. It is about 0
(within a 64-token page) when the previous turn's KV was reused (`:266-269`). `finished_users` counts the users who
ended on their own, aborted ones included (`:254`).

Changes from the old format, for scripts that read both:

- `users` is new. It carries the live and waiting counts every 60 s, whatever the cap. The old `cap` record was written
  only by the backlog cap, every 120 s, with `prev_cap`, `backlog_mean` and `target`. `steady_state.py` and
  `steady_figures.py` read `cap` records as backlog-cap updates, which is why the new record has another name.
- `conv_start.arrival_s` is the scheduled arrival. The old value was the time the user's coroutine ran, up to ~20 ms
  later (§7.1).
- `conv_start` gains `task_conv`, `session_id` and `source`, and `turn.task_conv` is always written; `task_conv` is the
  conversation's index in the trace. The old client wrote `conv_start.tasks` (with `task_sessions`) and
  `turn.task_conv` only with task chaining or `--arrival-until`, and as positions in the selection after `--offset`:
  the trace index at `OFFSET=0` (campaigns 14-17), but campaign 13 ran `OFFSET=32`, so its index k is trace
  conversation k + 32.
- `conv_skipped.reason` is `timeout` (old: `max_seconds` or `stop_after_users`).
- `summary.capped` is also true when users were skipped (old: only when a running user was cut). `users`,
  `users_started`, `finished_users` and `skipped_users` are always written.
- `run.users` replaces `run.conversations`; `run.model` is new; `run.turns` is always the planned total (old: null with
  task chaining or `--arrival-until`, so in every campaign 13-17 run).
- Gone: the `control`, `compaction`, `truncation` and `cap` records; the turn fields `reply_ids_reused`,
  `reply_ids_total`, `prev_ids`, `prev_ids_on_prefix`, `mismatch_pos_in_reply`, `prev_ctx`, `new_ctx`, `mid_start`,
  `synthetic_turns`, `task`, `task_turn`, `after_context_action`, `predicted_prompt` and `text`; the conv_start fields
  `start_turn`, `start_idle`, `planned`, `tasks`, `task_sessions`, `task_gaps` and `salt`; the conv_end fields `reason`
  (`context`) and `predicted_prompt` (written only under a context policy); the summary fields `controls`,
  `compactions`, `truncations`, `context_ends`, `window`, `stopped_by`, `stop_s`, `cap_mode`, `cap_mean`, `cap_min`,
  `cap_max`, `mid_start_users` and `mid_start_prompt_tokens`.

## 5. Analysis scripts on a new run

What each script that reads client output does with a run of the new client:

| script | on a new run |
|---|---|
| `timeline.py` | with `--ttft-over-time`, draws the `waiting` count of the `users` records in its waiting-users panel (`:106`, `:572-573`); with C = 0 and an arrival rate, `compare.png`'s first-turn panel says "uncapped sessions" (`:461`; at rate 0 it says "cold prefill", `:463`) |
| `c16_figure.py` | draws the `waiting` count of the `users` records (`:33`) |
| `c16_summary.py` | unchanged: reads `run` (`tag`), `turn` and `conv_end` records; its default arms and pairs are campaigns 14-16's, so for a new run pass `--arm LABEL compare_<UTC time> <arm>` for each arm and at least one `--pair LABEL_A LABEL_B` (without `--pair` it looks up the campaign-16 labels and stops with `KeyError: 'c16 _dl'`, `:83`) |
| `steady_state.py`, `steady_figures.py` | the default warm-up is the run's C, so a run with C = 0 exits and asks for `--warmup-users` or `--window-start` (`steady_state.py:55`); their backlog-cap columns and lines stay empty, since they read only `cap` records (`steady_state.py:37`) |
| `gap_distribution.py` | reads `args.gap_cap`, which the run record sets to 1,200 (`:93`) |
| `oracle_eval.py` | rebuilds request ids from `args.tag` (`:49-50`) |
| `plot_token_tiers.py` | unchanged: reads the `run` and `turn` records and the boot's `server.log`; `args.tag` only titles the figure (`:184`) |
| `vs_baseline.py`, `campaign_summary.py`, `live_timeline.py`, `gapclass.py`, `delay_components.py`, `crossscale.py`, `read_backlog.py`, `next_use_replay.py`, `oracle_timeline.py`, `read_shift_replay.py`, `srs_summary.py` | unchanged: they read only fields the new client still writes; `vs_baseline.py`'s first-request test falls back to turn 0, which is every user's first request now (no `mid_start`) |

## 6. Wrappers

### 6.1 `start_client.sh`

It runs inside the container against port 30000 and stops before starting the client when `results/.current_run` is
missing, when the server it names is not running, or when `/health` fails. The output goes to
`<server run dir>/client_<UTC HHMMSS>_<TAG>/` (`client.jsonl`, `client.log`), next to the boot it measured. After the
client exits, it prints one line per turn (`conv turn t_send gap ttft prompt cached reprefill`) and the summary.

### 6.2 `run_compare.sh`

The server knobs are unchanged (`ARMS LEVEL L1_TOKENS HOST_GB MODEL CTX KV_DTYPE TEMPLATE L3_CLEANER_PCT
ENGINE_PYTHONPATH`). The client knobs are exported with the defaults of §2 (`:25-34`). Before any boot it checks only
that `TRACE` loads and holds at least `USERS` conversations (`:35-39`; §8.4). `config.txt` records `TRACE USERS TURNS
ARRIVAL GAP ORACLE_HINTS C SEED CLIENT_TIMEOUT` with the server fields. Each arm runs
`TAG=cmp_<arm> TIMEOUT=$CLIENT_TIMEOUT bash start_client.sh` (`:57`).

The default client workload changed. The old default replayed conversations 29-76 of the 32K trace (`OFFSET=29`, chosen
because 42 of those 48 have at least 24 turns) with `GAP_CAP=600`. The new default replays the first 48 users of the
seed-0 permutation, of which 23 have at least 24 turns (956 turns at `TURNS=24`, against 1,119), with the 1,200 s cap,
so the population shrinks during a default run (measured 2026-10-17 from the trace).

### 6.3 The retired-knob guard

Both wrappers stop before any work when any of `NCONV OFFSET MAX_SECONDS GAP_CAP K THINKING CHECK_IDS TURNS_RANGE
ARRIVAL_UNTIL STOP_AFTER_USERS SESSION_TASKS CONTEXT_POLICY CAP_MODE ORDER START_PHASE VALIDATE_ONLY` is set and
non-empty, and print `STOP: <knob> is not a client knob any more (replay_agentic.py was trimmed; knobs: ...)`
(`scripts/start_client.sh:26-29`, `scripts/run_compare.sh:16-19`). The other old knobs (`TURNS_SEED`, `CAP_TARGET`,
`START_SEED`, `MAX_CONTEXT`, ...) acted only together with one of these, so they are not checked. `MODEL` acted alone
(the old client's `--model`), but it is still the model knob of `start_server.sh` and `run_compare.sh`, and the client
now sends the served model, so it is not checked either. `VALIDATE_ONLY` is checked because without the guard
`VALIDATE_ONLY=1 bash start_client.sh` would now run a live workload.
`run_campaign12.sh`, `run_campaign13.sh` and `run_campaign14.sh` set guarded knobs, so they stop here.

## 7. Comparing with campaigns 13-17

### 7.1 What still matches

At `--seed 0 --arrival-rate 0.1 --num-users 768` on the 262K trace, all 768 users replay the conversation they replayed
in campaign 14's `three_tier_wc` arm, and each arrives within 9.9 ms of its arrival there (mean 0.88 ms; measured
2026-10-17 against `results/20261008_204128_three_tier_wc_NAT160/client_204548_cmp_three_tier_wc/client.jsonl` with
§10.3's check). Campaigns 15-17 ran the same driver (`run_campaign14.sh`) at the same seed: the same check on all nine
arms of campaigns 14-17 gives 768/768, with arrivals within 19.9 ms (measured 2026-10-17). Their oracle arms sent the
envelope of §3.7.

### 7.2 What differs

| | campaigns 14-16 (`run_campaign14.sh`, ×70) | the new client, same seed and rate |
|---|---|---|
| user cap | the backlog cap (§3.5) | fixed `C` |
| first users | 125 of the first 128 started mid-session (stationary start) | every user starts at turn 0 |
| turns per user | min(recorded, a draw in 30-50, seed 120): 22,001 turns, of which the stationary start skipped 1,769, leaving 20,232 to send | min(recorded, `TURNS`): 24,879 at `TURNS=50` |
| gap cap, timeout | 1,200 s, 18,000 s | 1,200 s, `CLIENT_TIMEOUT` |

Counts measured 2026-10-17 from campaign 14's `three_tier_wc` client records and from the trace. Campaign 17 ran the
same at ×10, where 122 users started mid-session and 20,293 turns were left to send (`results/campaign17/DECISIONS.md`).
Campaign 13 cannot be expressed either: its users kept arriving for the whole run, cycling through 128 conversations of
the 262K trace (`OFFSET=32 NCONV=128`; each of the three finished arms started 511 users, so every conversation was
replayed 3-4 times, the reopened ones with a salt), and each arm was set to stop once 384 users had finished. A new run
is a new workload: compare new arms with each other, not with campaigns 13-17.

## 8. Open items

### 8.1 The fixed gap cap and the sparse regime (decision)

ORACLE_PLAN's sparse regime for E1 and E2 (`ORACLE_PLAN.md:692-693`) replays
`traces/lmcache_agentic_trace_262k_ge45_tl_s1.json` with `GAP=1 GAP_CAP=0`, and the plan's decisions 2 and 5 still read
"open; decide after E1" (`ORACLE_PLAN.md:801`, `:804`). Under the new client the 1,200 s cap would shorten 188 of that
trace's 13,706 sleeps and remove 13.7 % of its idle time (§3.4). Two other things already change that workload: its
commands set guarded knobs (`NCONV`, `OFFSET`, `GAP_CAP`), and campaign 10 replayed conversation c as user c, while the
new client always permutes. Against that, the regime's point (long pauses send returning turns to the SSD) survives a
20-min cap.

Options: (a) add `--gap-cap` back to the client (default 1,200, 0 = none) with a `GAP_CAP` knob, and drop `GAP_CAP`
from the guard; (b) mark E1 and E2 superseded in ORACLE_PLAN, since ORACLE_V2_PLAN moved to the ×70 workload with the
1,200 s cap.

### 8.2 Documents that still show the removed flags

These lines name a removed flag or a guarded knob. Commands among them fail loudly (argparse exit 2, or the guard's
STOP) rather than run a different workload. Lines that describe past runs are history and stay as they are.

- `RUNBOOK.md`: §4.5 (:1347), §4.6 (:1379-1428, the old client reference), §7.0 (:2418), §7.0b (:2458, :2483, :2487),
  §7.1 (:2566), §7.2 (:2574-2592, :2713-2730), §7.3 (:2815, :2820, :2860), §10 (:3263), §11 (:3308, :3310). The
  `--model` lines of §3.4 (:814, :838), §4.4 (:1221, :1258) and §5.4 (:2004) belong to `sglang.benchmark.serving` and
  AIPerf, which the trim did not touch.
- `ORACLE_PLAN.md`: steps 1, 2, 5, 6, 7 and 9 (:316-651), and §5 (:688-702).
- `ORACLE_V2_HANDOFF.md`: the §H4 smoke (:162). Its new form is
  `USERS=32 C=32 TURNS=12 GAP=10 ORACLE_HINTS=1 TAG=smoke bash start_client.sh`. Its permuted conversations plan 363
  turns with a longest per-user idle time of 935 s, against 381 turns and 656 s for the old form on conversations 0-31
  (derived from the 262K trace; the 1,200 s cap does not act on the new users, whose longest ×10 sleep is 428 s), so
  plan on ~20 min (estimate: 935 s of sleep plus 12 turns of service).

### 8.3 Thinking is always off

`enable_thinking` is false in every request (`:171`), so a run with thinking on (`templates/qwen3_replay_think.jinja`,
the old `THINKING=on`) cannot be made. Every campaign so far ran with thinking off or with a model that has no think
block. A think run would need the flag back.

### 8.4 No offline check of the knobs

`--dry-run` and `--validate` are gone, and `run_compare.sh` checks only the trace and `USERS` before its first boot. The
client checks the rest when it starts, after the boot: a bad value such as `C=-1` makes every arm boot, then print
`STAGE FAILED <arm> (client, rc=..., no summary line)`. Try new knob values with a short `start_client.sh` run first
(§9.1).

### 8.5 Admission order depends on the Python version

Admission in arrival order (§3.5) needs an `asyncio.Semaphore` whose `release()` hands the slot to the first waiter.
Python 3.10.12 has it (checked 2026-10-17 in its `asyncio/locks.py`). To check the container's Python:

```bash
# host: the container's Python, and whether a released semaphore slot goes straight to the first waiter
docker exec sglang_hicache python3 -c 'import sys, inspect, asyncio.locks as l; print("python:", sys.version.split()[0]); print("handoff to the first waiter:", "_value -= 1" in inspect.getsource(l.Semaphore._wake_up_next))'
```

-> prints `python: <version>` and `handoff to the first waiter: True`. `False` means a user who arrives just as a slot
frees can take it ahead of the users already waiting, so `queued_s` is no longer first come, first served.

### 8.6 One user per conversation

A run has at most as many users as the trace has conversations (731 or 768), and nobody arrives after the last of them.
An open-loop steady state like campaign 13's, with users arriving for hours, cannot be run. `steady_state.py`'s
warm-up and window assume one.

## 9. Running it

### 9.1 Smoke run

**Status:** not yet run on the H200 box with the new client (checked against a mock server only, §10.1)
**Goal:** see the new client finish a small run against a real server before a campaign, and see reply reuse hold
(`reprefill` near 0).
**Runs on:** host for the first block (the box's checkout, `/home/wanhr/sglang`); then container `sglang_hicache`
(`docker exec -it sglang_hicache bash`), in `/sgl-workspace/sglang/agent_cache/scripts`
**Touches:** boots and stops a server (rewriting the tracked `results/.current_run`); writes a new run dir under
`agent_cache/results/` with `client_<HHMMSS>_smoke/` in it; the SSD is not used (`hbm_host`)
**Takes:** boot ~2 min (measured 2026-10-17: 103-191 s from `run.txt`'s `started_utc` to `fired up`, over the nine
`hbm_host` NAT160 boots of this model in `results/`); client ~40-80 s (estimate: its longest user decodes 2,486 of the
6,215 reply tokens at 11-25K-token prompts, at the 9-33 ms per token this model shows with 1-4 requests in flight in
`results/`); GPU busy

```bash
# host: the box's checkout must contain the redesign, or this step and §9.2 run the pre-trim scripts (read-only)
cd /home/wanhr/sglang && echo "HEAD: $(git log -1 --format='%h %s')" && echo "has the redesign: $(git merge-base --is-ancestor e32458cc9f HEAD 2>/dev/null && echo yes || echo NO)"
```

-> prints the HEAD commit and `has the redesign: yes`. `NO`: run `git pull --ff-only` in `/home/wanhr/sglang`, then
this block again. Without the redesign, `start_client.sh` is the pre-trim one (4 conversations from offset 0,
`TAG=observe_c4`), and §9.2's `run_compare.sh` ignores `USERS` and replays `NCONV=48 OFFSET=29 GAP_CAP=600` for up to
5 h without an error.

```bash
# container: sglang_hicache: the box's model settings (as in ORACLE_V2_HANDOFF §H4), then boot an arm without the SSD
export MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= L3_CLEANER_PCT=70,60
cd /sgl-workspace/sglang/agent_cache/scripts && bash start_server.sh hbm_host NAT160
```

-> prints `READY: hbm_host after <N>s`, then five labelled checks. Pass: `template loaded:  0   (want 0: stock
template)`, `marlin lines: 0`, `profiled warning: 0   (want 0)`, and `host pool:` at 160 GB (`1627648 tokens, 160.00
GB` on 2026-10-07, measured: `results/20261007_025943_hbm_host_NAT160/server.log`). `STOP: hbm_host died during
startup` and the log's last 20 lines mean the boot failed.

```bash
# container: sglang_hicache: 4 users x at most 10 turns, cap 4, recorded gaps, the 32K trace (stop early: pkill -f replay_agentic.py)
bash start_client.sh
```

-> prints `client: --trace ... -> <client dir>`, then
`4 users, 39 turns, each user a different conversation of /sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json; arrivals all at once; concurrency 4; gaps x1, at most 1200 s; oracle hints off; timeout none`,
four `conv <i> done` lines (with a `[60 s] users arrived 4/4, ...` progress line among them if the client runs past a
minute), `summary: {...}`, the per-turn table, and the summary again. Pass: `"users_started": 4`,
`"finished_users": 4`, `"skipped_users": 0`, `"errors": 0`, `"capped": false`, and `reprefill` within ±64 on the
returning turns.

```bash
# container: sglang_hicache: stop the server
bash stop_server.sh
```

-> prints `stopped: <run dir> (pid <N>); port 30000 free`.

**Expected result:** the outputs above; `<run dir>/client_<HHMMSS>_smoke/client.jsonl` holds 1 `run`, 4 `conv_start`,
39 `turn`, 4 `conv_end` and 1 `summary` record, plus one `users` record per full minute the client ran (none if it ends
within 60 s, which it may).
**If it differs:** `STOP: the server of ... is not running` or `STOP: no server on port 30000`: the boot failed or was
stopped; read the run dir's `server.log`. `errors` above 0: read the turns' `error` fields; an HTTP 400 is usually a
prompt plus `max_tokens` over `CTX`. `reprefill` in the thousands on every returning turn: the server does not render
the re-fed reply as it generated it (RUNBOOK §4.5); check `TEMPLATE`.
**Expected lessons:** a `mean_reply_reprefilled` far from 0 means every returning turn recomputes its previous reply,
and a campaign would measure the template, not the cache tiers.

### 9.2 A compare campaign

**Status:** not yet run with the new client
**Goal:** run one workload against several arms, each on a fresh boot with identical pools, with the new knobs.
**Runs on:** host (after §9.1's checkout check), starting a detached job in container `sglang_hicache`
**Touches:** boots and stops one server per arm (rewriting the tracked `results/.current_run`); wipes
`/mnt/ssd/hicache_l3` before each three-tier arm (destroys the L3 store); writes
`agent_cache/results/compare_driver_<UTC time>.log`, `agent_cache/results/compare_<UTC time>/` and one run dir per arm
**Takes:** per arm, the boot (~3-4.5 min: 169-258 s over the thirteen `three_tier_wc` NAT160 boots of this model in
`results/`, measured 2026-10-17) plus up to `CLIENT_TIMEOUT` (5 h below); GPU busy throughout

```bash
# host: campaign 14's size on the new client: 768 users at 0.1/s, all recorded turns, x70 gaps, at most 128 users in session, 5 h per arm
docker exec -d sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/scripts && \
  ARMS="three_tier_wc" LEVEL=NAT160 \
  MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 CTX=131072 KV_DTYPE=auto TEMPLATE= L3_CLEANER_PCT=70,60 \
  TRACE=/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json \
  USERS=768 TURNS=50 ARRIVAL=0.1 GAP=70 C=128 SEED=0 CLIENT_TIMEOUT=18000 SGLANG_TIMEOUT_KEEP_ALIVE=3600 \
  nohup bash run_compare.sh > ../results/compare_driver_$(date -u +%Y%m%d_%H%M%S).log 2>&1'
```

-> prints nothing (`-d`); the watch block below shows that it started. The driver log gets its own name because
`results/compare_driver.log` is a tracked file (a 2026-09-18 log) that `>` would overwrite.

For oracle arms, add `ORACLE_HINTS=1` and list them in `ARMS`: every arm's client then sends the hints, so the request
bodies match and only the server mode differs.

```bash
# host: watch it: the newest driver log's STAGE and STOP lines, then the client's latest progress line (none while the arm boots)
docker exec sglang_hicache bash -c 'cd /sgl-workspace/sglang/agent_cache/results && L=$(ls -t compare_driver_*.log | head -1) && echo "log: $L" && grep -aE "STAGE|ALL DONE|STOP" "$L"; grep -ah "^\[" $(cat .current_run)/client_*/client.log 2>/dev/null | tail -1'
```

-> `log: compare_driver_<UTC time>.log`, `STAGE START three_tier_wc ...`, then, once the client runs, a line like
`[600 s] users arrived 51/768, live 49, waiting 0, done 2 (2 finished); turns ..., requests in flight ...` (derived: 51
of the seed-0 arrivals at 0.1/s fall within 600 s, as campaign 14's log shows; `live` 48-51 and `done` 0-3, depending
on the service time). A `STOP:` line instead of `STAGE START` means the driver refused to start (If it differs); a
progress line shown with it is an older run's.

```bash
# host: stop the campaign cleanly: the driver first (so no next arm boots), then the client, then the server
docker exec sglang_hicache bash -c 'pkill -f "^bash run_compare.sh"; pkill -f "^python3 replay_agentic.py"; bash /sgl-workspace/sglang/agent_cache/scripts/stop_server.sh'
```

-> ends with `stopped: <run dir> (pid <N>); port 30000 free` (`already stopped: ...` if the server had died; read its
`server.log`).

**Expected result:** per arm a `STAGE DONE` line and a manifest line `<arm> <run dir> <client dir>`, and a summary with
`"errors": 0`. All 768 users have arrived by ~7,633 s (derived: the last of 768 seed-0 arrivals at 0.1/s), but each
user holds its slot through every sleep (§3.5), so `live` reaches 128 at ~1,500-1,700 s and stays there while
`waiting` climbs to ~300-420 by ~7,600 s. Expect `STAGE CAPPED three_tier_wc` at any server speed: the queue alone
leaves 3 users cut at 18,000 s with zero service time, and at campaign 14's per-turn latency (median 18.2 s, mean
32.2 s) ~60-150 users are cut or never start (derived 2026-10-17: FIFO admission over the trace's ×70 sleeps capped at
1,200 s).
**If it differs:** `STOP: USERS=768, but <trace> has N conversations`: `TRACE` is the 32K trace (731); set the 262K one.
`STOP: <knob> is not a client knob any more`: the knob is set in the launch command or in the container's own
environment (`docker exec sglang_hicache printenv <knob>` prints it there); remove it. A knob exported in the host shell
does not reach `docker exec`.
**Expected lessons:** this is not campaign 14's workload (§7.2): compare its arms with each other. `C` bounds the
population in session, and with ×70 gaps most of that population is idle at any moment, so the admission queue
(`queued_s`, the `users` records) is where the cap shows; no TTFT includes it.

### 9.3 Rerunning a campaign 13-17 workload

**Status:** not needed so far
**Goal:** reproduce a campaign 13-17 client exactly, with its backlog cap, stationary start and turn draws.
**Runs on:** host (the box's checkout, `/home/wanhr/sglang`), then the container as that campaign's
`results/campaign1N/DECISIONS.md` says
**Touches:** the working-tree copies of the three client scripts (overwritten; the block stops first if they hold
uncommitted edits), restored at the end; nothing is committed
**Takes:** under a minute for each swap; the campaign as documented

```bash
# host: put the pre-trim client and wrappers in the working tree (HEAD and the index keep the new ones); stops if the three files have uncommitted edits, which the restore would destroy
cd /home/wanhr/sglang && { git diff --quiet HEAD -- agent_cache/scripts/replay_agentic.py agent_cache/scripts/start_client.sh agent_cache/scripts/run_compare.sh || { echo "STOP: uncommitted edits in the client scripts; commit or stash them first"; false; } } && git restore --source=c20643b695 --worktree -- agent_cache/scripts/replay_agentic.py agent_cache/scripts/start_client.sh agent_cache/scripts/run_compare.sh && git status --short agent_cache/scripts
```

-> lists the three files as ` M`. Then launch that campaign's driver (campaigns 14-17: `run_campaign14.sh`). Every arm
re-reads the scripts, so the old files must stay in place until the last arm ends. Campaigns 15-17 also kept the
scripts they ran in `results/campaign1N/scripts_at_launch.tgz`.

```bash
# host: after the campaign, back to the new client
cd /home/wanhr/sglang && git restore -- agent_cache/scripts/replay_agentic.py agent_cache/scripts/start_client.sh agent_cache/scripts/run_compare.sh && git status --short agent_cache/scripts
```

-> prints nothing.

**Expected result:** the campaign's client records have the old format (`cap` records, `start_turn`, ...);
`timeline.py` and `c16_figure.py` still read them.
**If it differs:** `git status` lists other files: they were changed by hand; keep them out of the restore.
**Expected lessons:** a commit made while the old files are in the working tree would undo the redesign; commit only
after the second block.

## 10. How the change was checked (2026-10-17)

### 10.1 Mock server

The client ran against a mock OpenAI server (a scratch script, not kept) that streams exactly the requested number of
characters and counts one token per character plus three per message, so `reply_reprefilled` is exact. Everything
passed:

- A: each hinted next turn was sent `next_gap_s` after the previous reply ended, within 50 ms; `reply_reprefilled` was
  0 on every returning turn.
- B: with C = 3, never more than 3 users in session, and the waiting users were admitted in arrival order.
- C: with `--timeout`, no request was sent after T; sleeping users ended `capped`; waiting and later users were
  `conv_skipped`; the request in flight at T completed.
- D: an HTTP 500 ended its user `aborted`; a dropped connection was retried once and marked `retried`.
- E: negative values, `--num-users` above the trace's count, and no server each exited with their message.
- F: 40 users at rate 0 with no cap all started within 10 ms (`queued_s`).
- G: a connection dropped after the timeout was not retried.

`steady_state.py`'s loader, `c16_figure.py`'s series and `timeline.py`'s record filter read the mock output: the
`users` records give the waiting counts, and no `cap` records reach `steady_state.py`.

### 10.2 Review

Four reviewers (parity with the old client, analysis scripts, wrappers, edge cases) raised 18 findings, and an
adversarial verifier per finding kept 8. Fixed before the commit: `VALIDATE_ONLY` added to the guard (the only
medium finding); the per-minute record renamed from `cap` to `users`, so `steady_state.py` and `steady_figures.py` no
longer read it as backlog-cap updates (2 findings); "up to 0 live sessions" in `timeline.py` (2 findings);
`finished_users` counting aborted users again (1 finding). Also changed, though judged immaterial: no connection retry
after the timeout, and no one-step stagger between users who arrive together. The other two are open: §8.1 and §8.2.

### 10.3 The campaign 14 recheck

The numbers of §7.1 come from this check. It needs aiohttp (the container has it), the 262K trace and campaign 14's
client records.

```bash
# container: sglang_hicache: does the new client give campaign 14's users the same conversations and arrival times?
cd /sgl-workspace/sglang/agent_cache && python3 - <<'EOF'
import argparse, importlib.util, json
spec = importlib.util.spec_from_file_location("ra", "scripts/replay_agentic.py")
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
users = ra.load_users(argparse.Namespace(trace="traces/lmcache_agentic_trace_262k.json", num_users=768, turns=50,
                                         arrival_rate=0.1, seed=0))
c14 = "results/20261008_204128_three_tier_wc_NAT160/client_204548_cmp_three_tier_wc/client.jsonl"
starts = {d["conv"]: d for d in map(json.loads, open(c14)) if d["kind"] == "conv_start"}
same = sum(starts[u.conv]["tasks"] == [u.task_conv] for u in users)
dt = [abs(starts[u.conv]["arrival_s"] - u.arrival_s) for u in users]
print(f"same conversation: {same}/768; arrival difference max {max(dt) * 1e3:.1f} ms, mean {sum(dt) / len(dt) * 1e3:.2f} ms")
EOF
```

-> prints `same conversation: 768/768; arrival difference max 9.9 ms, mean 0.88 ms`. Fewer than 768 means the
permutation or the trace changed.
