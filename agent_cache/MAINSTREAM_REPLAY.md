# Mainstream trace replay: the literature's traces, its recipes, and a CPU test bench

Written 2026-11-18 on the laptop (no GPU job). Our campaigns replay the LMCache agentic trace with every gap stretched x10
or x70 and capped at 1,200 s (`scripts/replay_agentic.py`); the evaluation survey (`../reports/Agentic KV caching evaluation
setups.md`) found that the comparable papers replay *real-user* traces in a closed loop at x1, with outputs forced to the
recorded length, and that no other paper multiplies recorded gaps. This document describes a second client that replays
the traces those papers use, under their published recipes, and the CPU mock engine that tests it without a GPU. It
leaves `replay_agentic.py` and the campaign scripts untouched: the LMCache campaigns stay the controlled ablation bench.

Every number is tagged **measured** (date, file), **derived** (formula) or **estimate**. "Laptop" is this machine (24
cores, 62 GB RAM, Python 3.10, aiohttp 3.13, pyarrow 15); "the box" is the H200 instance of `RUNBOOK.md`, where this
checkout sits at host `/home/wanhr/sglang` = container `/sgl-workspace/sglang`.

## 0. The short version

| step | what | takes | GPU |
|---|---|---|---|
| §6.1 | download AgentX (full and 256K), TraceLab v0.0.2, one FreeInference week; verify checksums | ~3 min, 2.4 GB | no |
| §6.2 | convert each to a program trace and check the counts against the publishers' | ~3 min | no |
| §6.3 | unit and replay tests against the mock (`test_program_replay`) | 3 s | no |
| §6.4 | CPU smoke runs of the three recipes against `mock_engine.py` | ~2 min each | no |
| §6.5 | smoke run against SGLang on the box (status: not started) | ~20 min | busy |
| §6.6 | the comparison campaign (a proposal; nothing runs until the author says go) | hours | busy |

## 1. What the mainstream setups fix, and what the client does

The survey's 36 papers converge on one loop; three groups publish it precisely enough to reproduce. The client
(`scripts/replay_programs.py`) implements each as a **recipe** (a set of defaults; explicit flags win, and the run record
keeps the resolved settings).

| | InferenceX AgentX MVP (AIPerf `--scenario inferencex-agentx-mvp`) | MORI (arXiv 2606.00866 v2 §6.1) | TraceLab `session_runner` | our campaigns until now |
|---|---|---|---|---|
| trace | SemiAnalysis Claude Code corpus `cc-traces-weka-062126` (or its `_256k` sibling for ~256K servers) | the same corpus ("AgentX", 391 sessions) and TraceLab | TraceLab v0.0.2 via its canonical CSV | LMCache agentic (batch SWE-bench/GAIA/WildClaw agents) |
| schedule | `--concurrency` session trees kept live; a drained tree's lane draws the next | 20-120 programs in flight; a finished program starts the next | Poisson session arrivals, `--max-active-sessions` cap | Poisson users, a cap that counts idle users |
| unit | session tree (root + subagents) | program = session cut at idle > 30 min and at compactions; subagents run beside it | session | conversation |
| timing | recorded end-to-start gaps after each response (not start-to-start) | "sleeps for the recorded tool-call or human-wait duration" | sleep `tool_wait_after_ms` (tool time only) | recorded gap x10 or x70, capped at 1,200 s |
| prompts | synthesized from the recorded `hash_ids` | not stated (content-free trace) | synthetic tokens: `prefix_len` of the context (previous prompt + the server's real reply) + `input_len` new | recorded text, reply appended verbatim |
| outputs | `max_tokens` = recorded `out`, `ignore_eos` locked | recorded output counts, reasoning included | `max_tokens` = recorded output | recorded length, `ignore_eos`, temp 0 |
| start | each initial lane at a random instant of its trace, after one 1-token primer per live stream; recycled plays from turn 0 | first request | first request | first turn |
| idle | system-idle guard: when nothing is in flight, timers shift so the next request is <= 10 s away | none stated | none | none |
| repeats | cache-bust marker on every play | not stated | none | none |
| length | 1,800 s (>= 900 s), seed locked | 2 h | until done | 2 h |
| interface | none (session-affinity headers for routers) | `program_id` per subagent + release endpoint | none | `kv_hints` |

How the client reproduces each column, with the flag that sets it:

- **Closed loop with dependencies.** Every request starts once the requests it depends on have completed and its
  recorded wait has passed (`Play._stream`, `scripts/replay_programs.py`). A stream's request depends on its
  predecessor; a subagent's first request on the request that spawned it (AgentX: the main model's latest finished
  request, so a subagent launched while that request still streams is timed from the one before); a subagent's later
  chains (utility calls, forks) on the subagent's own latest finished request; the parent's next request also on the
  subagent's last one when the trace says it waited (the join). That is AIPerf's SPAWN/SPAWN_JOIN and MORI's
  "subagents run alongside" in one mechanism.
- **Schedules:** `--lanes N` (InferenceX, MORI) or `--arrival-rate R [--max-active M]` (TraceLab).
- **Timing:** recorded waits at x1 (`--gap-scale 1`, no cap). `--wait-field wait_tool` sleeps TraceLab's tool-only wait.
  `--system-idle-cap 10` is InferenceX's guard (`Clock.guard`). `--gap-scale` and `--gap-cap` remain for ablations and
  CPU tests only.
- **Prompts are token ids** on SGLang's native `/generate` (`--api completions` for OpenAI-compatible servers), so
  the prompt length the server counts equals the recorded one exactly (measured on the mock, §5.3; to be confirmed on
  SGLang in §6.5). Block traces map each block id to fixed tokens (scoped as the trace prescribes), so the recorded
  sharing reappears on the wire, across streams and across sessions; length traces follow TraceLab's builder and carry
  the server's own reply forward.
- **Outputs:** `max_new_tokens` = recorded output, `ignore_eos`, temperature 0.
- **Trajectory starts:** `--start-window LO HI --warmup primer` (InferenceX: 0 1).
- **Programs:** `--split-idle 1800 --split-compaction` (MORI), applied at load time to any trace.
- **Repeats:** `--cache-bust recycle` makes a recycled play's prompts start with tokens unique to the play.
- **Interface:** `--hints` (our engine, §4.2); `--program-id-field program_id --release-path /programs/release` for a
  MORI or ThunderAgent server on `--api completions`.

## 2. The traces

All four are public and were downloaded on 2026-11-18 into `agent_cache/data/` (git-ignored by `agent_cache/.gitignore`,
visible inside the box's container because the repo is bind-mounted, unlike `/home/wanhr/data`). Counts below are
**measured** 2026-11-18 by the converters (`data/converted/*.stats.json`) and compared with the publishers' cards.

| trace | license | file(s) | converted (`data/converted/`) | check against the card |
|---|---|---|---|---|
| AgentX `semianalysisai/cc-traces-weka-062126` | Apache-2.0 | `traces.jsonl`, 1,847,151,435 B, sha256 `29b6a19e…9276` = HF etag | `agentx_062126.programs.jsonl.gz`, 17.5 MB | 393 traces, 56,798 main turns, 1,697 subagent groups, 42,029 inner requests, 98,827 requests, 21,635,381,376 input and 106,474,498 output tokens: **all equal to the card**; every hash list chained (0 conflicts) |
| AgentX 256K `…-062126-256k` | Apache-2.0 | `traces.jsonl`, 568.9 MB, sha256 `e39cd2ff…eeef` = HF etag | `agentx_062126_256k.programs.jsonl.gz`, 11.9 MB | 393 traces, 28,444 main turns, 1,697 groups, 39,822 inner, 68,266 requests, 6,891,228,864 input, 58,728,807 output: **all equal** |
| TraceLab `UW-SyFI/TraceLab` v0.0.2 | CC BY 4.0 | three Parquet tables, sha256 = the card's | `tracelab_v0.0.2.programs.jsonl.gz`, 19.1 MB | 8,058 sessions, 665,453 rounds: **equal**; human waits 91.9 % of reconstructed wall-clock (paper: 92.3 % on its 4,265 sessions), median human wait 1.44 min (paper 1.4), median tool latency 0.23 s (paper 0.3) |
| FreeInference `harvardMadsys/freeinference_agentic_trace`, week 2026-07-12 | CC BY 4.0 | 8 day files, each checked against the week's manifest (bytes, lines, sha256) | `freeinference_2026-07-12.programs.jsonl.gz`, 35.2 MB | 699 sessions + 270 subagent sessions = 969 (= the week's `sessions.parquet` rows), 72,025 requests (= `requests.parquet` rows) |

What each converter decides (the module docstrings are the reference):

- **AgentX** (`scripts/convert_agentx.py`). The top-level stream interleaves the main agent with side one-shots,
  forks, retries, context edits and resumes, which AIPerf's Weka loader also splits by hash-prefix evidence. The
  converter does the same: a request continues the chain whose last request it extends (same model, that request
  finished); otherwise the chain it shares a prefix with when that chain's old state never recurs (a compaction or
  edit); otherwise it starts a chain (a fork when it shares a prefix, a side chain when it shares none). Result on the
  full corpus: 393 main, 1,208 side, 195 fork and 7,990 subagent-chain streams (measured). Main-chain waits equal the
  corpus's own `think_time` for 558 of 583 requests in the 6-trace sample (measured; the rest follow interleaved side
  calls). No tool names: keys are null, MORI's "serving without tool information".
- **TraceLab** (`scripts/convert_tracelab.py`). One stream per session. `prefix` is the provider's cached prefix capped
  at the previous input plus output (the card's advice, TraceLab issue #22: 13,787 transitions, 2.1 %, measured),
  `append` the rest, so prompt lengths equal the recorded inputs. Two waits: `wait_tool` (TraceLab's canonical
  `tool_wait_after_ms`) and `wait` = tool time plus TraceLab's human waits (previous event to each `user_message`), the
  MORI-style gap. Keys: tool name and first bash executable.
- **FreeInference** (`scripts/convert_freeinference.py`). One tree per session, nested subagent sessions as streams
  with spawn and join; waits are the gateway gaps (end to next start; equal to `latency_ms` for all 4,870 single-tool
  steps of 2026-07-12, measured); `gap` is the request's `step_trigger` (tool or user). Block ids are scoped to the week
  as the card prescribes: 19.4 % of the week's prompt tokens are prefixes that earlier sessions already sent (system
  prompts, tool definitions; measured, file order). Keys: tool name and first shell program, which is our hint key.

Shape of the waits a replay sleeps (measured 2026-11-18, `*.stats.json`; seconds):

| trace | p50 | p90 | p99 | share over 60 s | over 300 s | over 1,800 s | MORI programs (idle > 30 min, compactions) |
|---|---|---|---|---|---|---|---|
| AgentX full, main chain | 4.84 | 125.2 | 3,546 | 14.7 % | 4.3 % | 1.5 % | 898 (MORI: 790 from 391 sessions) |
| AgentX 256K, main chain | 2.55 | 124.6 | 4,121 | 13.5 % | 4.2 % | 1.6 % | 720 |
| TraceLab, tool + human | 0.56 | 60.0 | 1,457 | 10.1 % | 3.5 % | 0.85 % | 18,492 |
| TraceLab, tool only | 0.20 | 10.7 | 180 | | | | |
| FreeInference, tool / user | 1.55 / 88.0 | 15.2 / 699 | 177 / 23,998 | | | | 1,118 |
| LMCache (ours), for contrast | 0.708 | | 11.76 | | | 0.66 % at >= 30 s | |

The LMCache row is from `traces/lmcache_agentic_trace_262k.json.stats.json:47-57`. The real-user traces put 4 % of
their waits past five minutes without any stretch, which is the regime host and SSD tiers exist for.

## 3. The program-trace format

`scripts/program_trace.py` defines it (module docstring) and holds the shared code: reading and writing, the chained
`lcp`, block deltas, MORI's split, token synthesis and the two prompt builders. One JSONL line per tree; a tree has
streams; a request has `t`, `dur` (recorded start and duration), `deps` and `wait` (the closed loop), `out`, `gap`,
`key`, and its content: `base`/`keep`/`new`/`tail` (blocks) or `prefix`/`append` (lengths). Block lists are stored as
deltas against an earlier request, which shrinks the 1.85 GB AgentX corpus to 17.5 MB (measured).

## 4. The client and its outputs

### 4.1 Running it

**Status:** done 2026-11-18 (CPU only)
**Goal:** know the flags a run needs; §6.4-§6.6 use them.
**Runs on:** laptop or container; read-only apart from the output file
**Touches:** the `--output` JSONL (appended)
**Takes:** as long as `--duration` plus warmup and `--grace`

```bash
# host (laptop): an InferenceX-style run against a server on port 30000 (the mock, §6.4, or SGLang)
cd /home/wanhr/Documents/codes/sglang/agent_cache/scripts
python3 replay_programs.py \
  --trace ../data/converted/agentx_062126_256k.programs.jsonl.gz \
  --recipe agentx-mvp --lanes 32 \
  --hints none \
  --max-context 262144 --seed 0 \
  --output /tmp/agentx_c32.jsonl
```
-> prints `393 trees from ... recipe agentx-mvp: 32 lanes, duration 1800.0, hints none`, a progress line per minute,
and a `summary:` line with `errors 0`.

**Expected result:** `program_summary.py /tmp/agentx_c32.jsonl` prints one row with TTFT percentiles, throughput and
served shares by tier.
**Expected lessons:** the recipe fixes the workload; vary only the server arm and `--hints` between runs, or the rows do
not compare.

### 4.2 Hint arms

**Status:** done 2026-11-18 (CPU only: the envelopes were checked against the mock, not against the engine)
**Goal:** map the literature's interfaces onto our engine's `kv_hints` so one server build serves every arm.
**Runs on:** none (reference)
**Touches:** read-only
**Takes:** none

| `--hints` | what each request carries | literature name | campaign 26 arm |
|---|---|---|---|
| `none` | nothing | no interface (LRU) | the no-hint arms |
| `program` | v2 action `{session, next_rid, gaps: [_end entry], replay_reply_key: null}` | program ID only (MORI, ThunderAgent without release) | `nofc_noend` |
| `release` | the same, and `replay_reply_key: "_end"` on a stream's last request | program ID plus release | `nofc` |
| `dist` | the online per-key gap table of the tree's agent system (harness, model) and the recorded reply's key; v1 median unless `--hint-point none` | ours | the dist arms |
| `oracle` | v1 action with the next request's exact scaled wait | upper bound | the oracle arms |

`dist` keys are the recorded tool calls (FreeInference: tool and first shell program; TraceLab: tool and first
executable); AgentX has none, so every gap feeds the pooled `*` entry. Because the replayed content is synthetic, the
engine must read the key from the hint (`--hicache-storage-next-turn-reply-key-source hint`), as in campaign 26.

**Expected result:** in a `--hints release` run, the last request of every stream has `hint.reply_key == "_end"` and
no other request has it (tested, §5.2).
**Expected lessons:** `program` and `release` let the paper name its rivals in the literature's terms (program ID only;
program ID plus release) on the same engine build, so a hint gain is measured over them, not over LRU alone.

### 4.3 What a run writes

**Status:** done 2026-11-18
**Goal:** say which record answers which question.
**Runs on:** none (reference)
**Touches:** read-only
**Takes:** none

- `run`: resolved settings and the trace header. `phase`: the warmup and profile boundaries. `play`: a tree replay that
  drained (lane, cut, end). `lanes`: per minute, live trees, requests in flight, timers, idle shift. `summary`.
- `req`: one per request: `phase` (warmup primers are excluded from metrics), lane, play, tree, stream, role, `inst`
  and `k` (the rid is `<tag>-c<inst>-t<k>`, the pattern `jit_vs_hicache.py:44-46` parses), the recorded and slept wait,
  `t_send`, `ttft`, `latency`, the server's `prompt_tokens`, `completion_tokens`, `cached_tokens` and
  `cached_details` (device, host, storage), `prompt_sent` (the client's count) and `ideal_cached` (what an infinite
  cache would hold: the prefix already completed in this run).
- `scripts/program_summary.py` reports, over the profile phase: TTFT mean/p50/p90/p99, TPOT, TTFT by gap kind and by
  wait length (our returning-turn view), output and input tokens/s and requests per minute (MORI's throughput,
  ThunderAgent's steps per minute), served shares by tier and the recomputed share (MORI), the ideal share and
  delivered/ideal (FlatCake's delivered hit rate).

**Expected result:** `python3 program_summary.py <run>.jsonl` prints one table row per run file; `--json` writes the
same numbers with the TTFT breakdowns.
**Expected lessons:** `ideal_cached` is taken when a request is sent, so a request that queues can be served more than
it (§5.4); delivered/ideal slightly above 1 is queueing, not an error.

## 5. The CPU test bench

### 5.1 Mock engine

**Status:** done 2026-11-18
**Goal:** test the client end to end without a GPU.
**Runs on:** laptop
**Touches:** a local port
**Takes:** starts in ~1 s

`scripts/mock_engine.py` speaks the wire format the client relies on (native `/generate` with SGLang's streamed
`meta_info`, `/v1/completions` with OpenAI usage, `/v1/models`, `/health`, `/programs/release`) and rejects unknown
`/generate` fields as `GenerateReqInput` does. Its cache is a page-level prefix cache over device, host and storage
tiers (LRU, write-through by default); a request waits until its KV fits in the device pool, prefills one at a time with
cost `a + uncached/prefill_tps + rho (L^2 - c^2)` plus tier loads, then decodes at `--tpot`. The defaults follow the
H200 prefill fit T(L) = 0.092 + 0.84e-6 L + 3.21e-9 L^2 s (fitted on 4K-65K tokens, `results/cpu_calib_c44/FINDINGS.md:156`)
and the box's 2.11 GB/s shared SSD rate over 98,304 bf16 KV bytes per token (21,500 tokens/s, derived). Even so, **its
numbers are not the H200's and belong in no table**: one prefill at a time, no batching of decode, an estimated host
rate. It exists to make queueing, tier misses and their ordering happen.

**Expected result:** `curl -s localhost:<PORT>/mock/stats` (`<PORT>` is its `--port`) prints request, cached-by-tier,
page and eviction counters.
**Expected lessons:** the mock tests the client; a policy question needs the engine or the CPU room model.

### 5.2 Tests

**Status:** done 2026-11-18: 21 tests pass in 2.4 s (measured); `test_gap_hints` still passes (60 tests, 3 skipped)
**Goal:** pin down the converters' and the client's semantics so later edits cannot drift silently.
**Runs on:** laptop or container
**Touches:** temporary files, a local port
**Takes:** 3 s

```bash
# host (laptop): the program-replay tests, then the existing hint tests
cd /home/wanhr/Documents/codes/sglang/agent_cache/scripts && python3 -m unittest test_program_replay test_gap_hints
```
-> prints `Ran 81 tests` and `OK (skipped=3)`; a failure names the behaviour that changed.

**Expected result:** what the tests hold: chained `lcp` against brute force; block deltas round-trip; MORI's split
(idle, compaction, never across a subagent); shared block ids give identical token prefixes within a scope and differ
across scopes; the bust marker leads; TraceLab's builder carries the server's reply; each converter's chains, deps,
spawn and join waits, keys, tails and prefix cap on synthetic inputs; token-pool windows; against the mock: spawn and join order and waits
in the closed loop, release hints end each stream, `dist` learns online, the idle guard compresses whole-system idle
(10 s of replay time in under 3 s of wall time), a trajectory start sends one 1-token primer and its prompt is then a
hit, the deadline wakes sleeping streams, the completions API and the release endpoint.
**Expected lessons:** none beyond the list; a new behaviour gets a test here.

### 5.3 Smoke runs on the real traces

**Status:** done 2026-11-18 on the laptop against the mock (`--time-scale 0.01` on the mock, `--gap-scale 0.01` on the
client: a test of mechanics, not timing)
**Goal:** show that each recipe runs on its trace without errors and that the wire carries what the trace says.
**Runs on:** laptop
**Touches:** scratch JSONL files
**Takes:** ~1 min each

**Expected result (measured 2026-11-18):** AgentX full, `agentx-mvp`, 4 lanes, 6 trees: 186 profile requests and 5
primers, 0 errors, 14 idle jumps, every request's `prompt_sent` = the server's `prompt_tokens`, and the slept gap from
the previous completion within 1.4 ms of the scheduled wait at the median (167 consecutive pairs). TraceLab,
`tracelab`, 12 sessions at 0.5/s: 482 requests, 0 errors, served = ideal (the carried replies are hits), keys such as
`Bash/grep`, `exec_command/cd` and `_end` on the last request. FreeInference, `mori`, 6 lanes, 14 programs: 459
requests, 0 errors, and 18 % of each session's first prompt served from other sessions' cache: the cross-session sharing
is on the wire.
**Expected lessons:** at `--max-context 262144`, whole-tree dropping removes 173 of 393 AgentX traces (measured), 367 of
8,058 TraceLab sessions and 123 FreeInference programs; on AgentX use the 256K corpus, which drops requests instead of
trees, as InferenceX does for ~256K servers.

### 5.4 A 32-lane AgentX stress run

**Status:** done 2026-11-18 (see the result for what it does and does not show)
**Goal:** check the client at the lane count InferenceX and MORI use, with prompts up to 256K tokens, and see how fast
the AgentX working set outgrows campaign 26's pools.
**Runs on:** laptop
**Touches:** scratch files, local port 31002
**Takes:** ~10 min

```bash
# host (laptop): a mock sized like campaign 26's 160 GB setting, everything 10x faster on both sides (a mechanics test)
cd /home/wanhr/Documents/codes/sglang/agent_cache/scripts
(python3 mock_engine.py --port 31002 --time-scale 0.1 --device-tokens 668160 --host-tokens 1630000 --storage-tokens 40000000 --max-running 64 > /tmp/mock_stress.log 2>&1 &)
python3 replay_programs.py --trace ../data/converted/agentx_062126_256k.programs.jsonl.gz --recipe agentx-mvp --lanes 32 --duration 300 --gap-scale 0.1 --system-idle-cap 1.0 --hints release --max-context 262144 --seed 42 --url http://127.0.0.1:31002 --output /tmp/agentx256_c32.jsonl | tail -1
```
-> prints a `summary:` line with `"errors": 0`.

**Expected result (measured 2026-11-18, laptop; the mock ran with its first defaults, a 20K tokens/s linear prefill
and a 50K tokens/s storage rate):** 38 primers (mean 135K, max 253K prompt tokens) took 290.6 s; then 582 profile
requests in 300 s from 32 lanes (269 main, 224 subagent, 53 fork, 36 side), 0 errors, 2,063 requests left unsent at the
end. Profile prompts averaged 108K tokens (max 255K), every one counted by the server as sent. The closed loop held
under that load: the gap from a completion to the next send exceeded the scheduled wait by 1.9 ms at the median and 6 ms
at p99 (477 consecutive pairs). The client peaked at 0.49 GB RSS.
**What it shows about the workload (derived, and preliminary):** 32 AgentX lanes keep about 32 x 108K = 3.5M tokens of
live context. That is 5x campaign 26's 668,160-token device pool, and 2.1x a 160 GB host tier (1.63M tokens at 98,304
bf16 KV bytes per token). On the mock, 68 % of prompt tokens came from storage, 18 % from host and 8 % from device. Read
at the box's 2.11 GB/s SSD rate, a 100K-token context takes 4.7 s. So a 32-lane InferenceX run on the box would be bound
by SSD bandwidth and warmup, not by hints: the first GPU points should be MORI's 20 programs or InferenceX at 8 lanes.
**What it does not show:** any latency. The mock prefills one request at a time and does not batch decode, so its TTFT
(16.7 s mean here) measures its own saturation. delivered/ideal came out at 1.01 because `ideal_cached` is taken at
send time, and a queued request can find pages that other requests completed while it waited.
**Expected lessons:** the AgentX corpus at InferenceX's lane counts is an SSD-tier workload on this box, which the LMCache
trace never was; pick lanes from the derived working set before booking GPU time.

## 6. Steps

### 6.1 Download the traces

**Status:** done 2026-11-18 on the laptop; not started on the box
**Goal:** the four inputs of §2, verified.
**Runs on:** laptop or box host, in `agent_cache/data/` (git-ignored); network needed
**Touches:** `agent_cache/data/` (2.4 GB)
**Takes:** ~3 min (measured on the laptop)

```bash
# host: AgentX full and 256K, then compare each file's sha256 with the Hugging Face etag
cd /home/wanhr/Documents/codes/sglang/agent_cache && mkdir -p data/agentx_cc_traces_weka_062126 data/agentx_cc_traces_weka_062126_256k
for v in 062126 062126-256k; do d=data/agentx_cc_traces_weka_${v//-/_}; u=https://huggingface.co/datasets/semianalysisai/cc-traces-weka-$v/resolve/main/traces.jsonl; curl -sL -o $d/traces.jsonl $u; echo "$v sha256: $(sha256sum $d/traces.jsonl | cut -c1-64)"; echo "$v etag:   $(curl -sI $u | grep -i '^x-linked-etag' | tr -d '\r"' | cut -d' ' -f2)"; done
```
-> prints two pairs of equal hashes (`29b6a19e...` and `e39cd2ff...`); a mismatch means a truncated download, rerun.

```bash
# host: TraceLab v0.0.2 (three Parquet tables) and one FreeInference week (8 days + manifest)
cd /home/wanhr/Documents/codes/sglang/agent_cache/data && mkdir -p tracelab_v0.0.2 && for t in rounds tool_calls timing_events; do mkdir -p tracelab_v0.0.2/$t; curl -sL -o tracelab_v0.0.2/$t/train.parquet https://huggingface.co/datasets/UW-SyFI/TraceLab/resolve/v0.0.2/data/v0.0.2/$t/train.parquet; done; sha256sum tracelab_v0.0.2/*/train.parquet
W=2026-07-12_2026-07-18; F=freeinference_agentic/release/$W; mkdir -p $F/traces; curl -sL -o $F/manifest.json https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/resolve/main/release/$W/manifest.json; for d in 2026-07-11 2026-07-12 2026-07-13 2026-07-14 2026-07-15 2026-07-16 2026-07-17 2026-07-18; do curl -sL -o $F/traces/$d.jsonl.zst https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/resolve/main/release/$W/traces/$d.jsonl.zst; done; echo "freeinference day files: $(ls $F/traces | wc -l)"
```
-> prints the three TraceLab hashes of the card (`cb8c2b1d...` rounds, `7186e0d8...` tool_calls, `a2074999...`
timing_events) and `freeinference day files: 8`. The FreeInference files are checked against the manifest by the converter (§6.2).

**Expected result:** `du -sh agent_cache/data/*` shows 1.8 G, 543 M, 74 M and 24 M (measured).
**If it differs:** the box's HF access goes through the same URLs; a proxy that strips `x-linked-etag` leaves the etag
line empty, so compare with §2's hashes instead.
**Expected lessons:** other FreeInference weeks are a `--weeks` argument away (16 weeks, 12,002 sessions in all).

### 6.2 Convert

**Status:** done 2026-11-18 on the laptop
**Goal:** the program traces the client reads, with counts that match the publishers'.
**Runs on:** laptop or box (container or host with pyarrow and the `zstd` command)
**Touches:** `agent_cache/data/converted/` (85 MB)
**Takes:** 35 s AgentX full, ~15 s 256K, 75 s TraceLab (2 GB RAM), 44 s FreeInference (measured on the laptop)

```bash
# host: the four conversions; each prints its stats and writes <output>.stats.json
cd /home/wanhr/Documents/codes/sglang/agent_cache/scripts
python3 convert_agentx.py --input ../data/agentx_cc_traces_weka_062126/traces.jsonl --output ../data/converted/agentx_062126.programs.jsonl.gz | head -12
python3 convert_agentx.py --input ../data/agentx_cc_traces_weka_062126_256k/traces.jsonl --output ../data/converted/agentx_062126_256k.programs.jsonl.gz | head -12
python3 convert_tracelab.py --input ../data/tracelab_v0.0.2 --output ../data/converted/tracelab_v0.0.2.programs.jsonl.gz | head -8
python3 convert_freeinference.py --input ../data/freeinference_agentic --weeks 2026-07-12_2026-07-18 --output ../data/converted/freeinference_2026-07-12.programs.jsonl.gz | head -8
```
-> prints, in order, `"traces": 393` with `"requests": 98827`; `"traces": 393` with `"requests": 68266`;
`"sessions": 8058` with `"rounds": 665453`; `"files_verified": 8` with `"sessions": 699`.

**Expected result:** the counts of §2's table, `"unchained_ids": 0` on both AgentX files.
**If it differs:** a different count on AgentX means a different corpus drop (the cards are dated; pin by sha256);
`STOP: ... does not match its manifest entry` means a corrupt FreeInference download.
**Expected lessons:** none beyond §2.

### 6.3 Tests

**Status:** done 2026-11-18 (§5.2)
**Goal:** the semantics hold on the machine that will run the replay.
**Runs on:** container on the box (no GPU needed) or laptop
**Touches:** temporary files, a local port
**Takes:** 3 s

```bash
# container: sglang_hicache
cd /sgl-workspace/sglang/agent_cache/scripts && python3 -m unittest test_program_replay test_gap_hints
```
-> prints `OK (skipped=3)`; an import error means the container lacks aiohttp or pyarrow (only the converters need
pyarrow).

**Expected result:** 81 tests, 3 skipped.
**Expected lessons:** none.

### 6.4 CPU smoke runs against the mock

**Status:** done 2026-11-18 (§5.3)
**Goal:** a recipe runs end to end before any GPU time is spent.
**Runs on:** laptop or box container (no GPU)
**Touches:** a local port (31001), `/tmp/smoke_*.jsonl`
**Takes:** ~1 min each

```bash
# host (laptop): start the mock with every engine delay shrunk 100x (stop it with: pkill -f "mock_engine.py --port 31001")
cd /home/wanhr/Documents/codes/sglang/agent_cache/scripts && (python3 mock_engine.py --port 31001 --time-scale 0.01 > /tmp/mock.log 2>&1 &) && sleep 1 && curl -s localhost:31001/health
```
-> prints `ok`.

```bash
# host (laptop): the three recipes on small subsets, time compressed 100x on both sides (a mechanics test, not timing)
cd /home/wanhr/Documents/codes/sglang/agent_cache/scripts
python3 replay_programs.py --trace ../data/converted/agentx_062126_256k.programs.jsonl.gz --recipe agentx-mvp --lanes 4 --trees 6 --duration 30 --gap-scale 0.01 --system-idle-cap 0.1 --hints release --url http://127.0.0.1:31001 --output /tmp/smoke_agentx.jsonl | tail -1
python3 replay_programs.py --trace ../data/converted/tracelab_v0.0.2.programs.jsonl.gz --recipe tracelab --arrival-rate 0.5 --max-active 8 --trees 12 --duration 40 --gap-scale 0.01 --hints dist --max-context 262144 --url http://127.0.0.1:31001 --output /tmp/smoke_tracelab.jsonl | tail -1
python3 replay_programs.py --trace ../data/converted/freeinference_2026-07-12.programs.jsonl.gz --recipe mori --lanes 6 --trees 14 --duration 40 --gap-scale 0.01 --hints oracle --max-context 262144 --url http://127.0.0.1:31001 --output /tmp/smoke_fi.jsonl | tail -1
python3 program_summary.py /tmp/smoke_agentx.jsonl /tmp/smoke_tracelab.jsonl /tmp/smoke_fi.jsonl
```
-> each `summary:` line shows `"errors": 0`; the table shows `deliv/ideal` near 1.0 (the mock's pools are big enough
for six trees).

**Expected result:** §5.3's numbers within a few requests (the mock's timing is real time, so counts move a little).
**Expected lessons:** `--gap-scale` here only compresses the test; every GPU run in §6.5-§6.6 uses 1.

### 6.5 Smoke run against SGLang on the box

**Status:** not started (no GPU); it is the first GPU step and needs the author's go
**Goal:** confirm on the real engine what the mock showed: token-id prompts are counted as sent, `cached_details`
arrives per tier, hints are accepted, and one recipe runs without errors.
**Runs on:** container `sglang_hicache` on the box, server booted by `start_server.sh`
**Touches:** `agent_cache/results/<server run>/programs_<time>_<tag>/`; GPU busy
**Takes:** ~20 min (estimate: §5.4 shows cold AgentX primers are expensive; a 250K-token cold prefill is ~200 s on the
H200 by the fitted T(L), derived, and that is a 4x extrapolation past the fitted 65K)

```bash
# container: sglang_hicache — a 4-lane InferenceX-style run on the 256K corpus, 10 minutes, release hints
cd /sgl-workspace/sglang/agent_cache/scripts && RECIPE=agentx-mvp LANES=4 DURATION=600 HINTS=release TAG=smoke bash start_programs.sh
```
-> prints `393 trees ... recipe agentx-mvp: 4 lanes, duration 600.0, hints release`, then the summary table with
`errors 0`.

**Expected result:** every `req` record has `prompt_tokens == prompt_sent` (`python3 -c` over the JSONL); host and
storage shares appear in `cached_details` on a three-tier arm; the server log shows `kv_hints` actions accepted.
**If it differs:** `prompt_tokens` larger than sent means the server added tokens (a BOS or a chat template): use
`--api generate` (the default) and check that no `--chat-template` applies to `/generate`. HTTP 400 on `/generate`
names the unknown field.
**Expected lessons:** whether a 32-lane AgentX run fits the box at all: if the warmup alone takes most of an hour, run
MORI's recipe (programs start at their first request) or fewer lanes first.

### 6.6 The comparison campaign

**Status:** proposal 2026-11-18; nothing runs until the author says go (PAPER_PLAN)
**Goal:** the rows a reviewer of this literature expects: the published recipes on their traces, with the
literature's interfaces as rivals and ours as the treatment.
**Runs on:** the box
**Touches:** `agent_cache/results/`; GPU busy
**Takes:** estimate: 0.5 h per InferenceX cell plus warmup, 2 h per MORI cell

| row | trace | recipe | load | arms (`--hints`, with the server arm that reads them) |
|---|---|---|---|---|
| 1 | AgentX 256K | `agentx-mvp` | `--lanes` 8, 16, 32 | none (LRU), program, release, dist, oracle |
| 2 | AgentX 256K | `mori` | `--lanes` 20, 40, 80 | the same |
| 3 | FreeInference (more weeks) | `mori` | `--lanes` 32, 64 | the same; the first trace with tool names and real humans |
| 4 | TraceLab | `tracelab` | `--arrival-rate` 0.05, 0.1 with `--max-active` 64 | none, release, dist |

Rows 1-2 give direct comparability with InferenceX submissions and MORI; row 3 is where per-tool distributions can
beat identity plus release, if anywhere; row 4 reproduces TraceLab's own replayer. Repeat any claim under 0.25 s
(seeds 0 and 1), as the survey recommends.

**Expected result:** one `programs_*` directory per cell under the server run it measured, each with `summary.txt`.
**Expected lessons:** if `dist` does not beat `release` on row 3 (real human and tool waits, tool names), the paper's
claim is a regime map, not a blanket gain (the survey's conclusion); if it does, rows 1-2 show how much of the gain
survives without tool names.

## 7. Deviations from the published setups

1. **Prompts are token ids, not decoded text.** AIPerf decodes the synthetic tokens and sends chat messages; we send
   ids to `/generate`, so no chat template is applied and the server's count equals the recorded one. Effect: none on
   cache structure; the chat headers AIPerf adds (a few tokens per message) are absent.
2. **Cache-bust only on recycled plays, at the start of the prompt.** AIPerf marks every play at the first user turn,
   keeping the system prompt shared. First plays are unmarked here so FreeInference's cross-session sharing survives;
   with 393 traces and <= 128 lanes a recycled AgentX play is rare in a 30-minute run.
3. **Block traces do not reuse the generated reply.** The next prompt is rebuilt from the recorded blocks, so the
   reply region is recomputed each turn: 0.85 % of AgentX-256K prompt tokens and 0.30 % of FreeInference's (derived:
   output / input tokens). Length traces (TraceLab) carry the server's reply as TraceLab's runner does.
4. **Chain detection is a simplification of AIPerf's** (no worker-group or sidecar naming); the timing roles (fork,
   side chain, subagent spawn and join) are the same. Program counts differ from MORI's (898 or 720 against 790; MORI
   used a 391-session drop and does not define its compaction test).
5. **TraceLab prefixes are capped** at the previous input plus output (the card's advice); TraceLab's own export does
   not cap. `--wait-field wait_tool` reproduces the export's sleeps; the default `wait` adds human waits (MORI's).
6. **FreeInference lengths are o200k counts** (the card's tokenizer), not the served model's.
7. **Timing is load-independent**, as in every replay of the survey: tool and human waits were recorded on other
   providers' stacks. The live mini-swe-agent run the survey recommends is the cross-check; it is not part of this work.
8. **`dist` needs the `hint` reply-key source**, because the content is synthetic and no reply carries a tool call.

## 8. Traps

- A `--gap-scale` other than 1 is not the mainstream setup; it is for CPU tests and ablations only (§6.4).
- `--max-context` drops whole trees; on AgentX use the 256K corpus (§5.3).
- The AgentX corpus spans days per trace (median 2.8 h, p90 47 h, measured): an InferenceX lane can idle for hours in a
  human wait; that is the benchmark's design, and the system-idle guard only acts when every lane is idle.
- The mock's numbers are not evidence (§5.1).
- `/home/wanhr/data` is not visible in the container; the data lives under `agent_cache/data/` for that reason (§2).

## 9. Layout and hygiene

| path | what | tracked |
|---|---|---|
| `scripts/program_trace.py` | format, prompts, splitting | yes |
| `scripts/convert_agentx.py`, `convert_tracelab.py`, `convert_freeinference.py` | converters | yes |
| `scripts/replay_programs.py`, `start_programs.sh` | client and its launcher | yes |
| `scripts/mock_engine.py` | CPU stand-in for the server | yes |
| `scripts/program_summary.py` | metrics | yes |
| `scripts/make_token_pool.py` | real-text token pool for `--token-pool` | yes |
| `scripts/gap_profile.py` | gap tables of the converted traces (and the LMCache trace) for the gap dashboard | yes |
| `scripts/test_program_replay.py` | tests | yes |
| `data/` (raw downloads, `data/converted/`) | inputs, 2.5 GB | no (`agent_cache/.gitignore`: `data/`) |

Attribution, if results are published: AgentX (SemiAnalysis, Apache-2.0), TraceLab (SyFI Lab, University of
Washington, CC BY 4.0), FreeInference Agentic Trace (CC BY 4.0, cite the dataset's paper).
