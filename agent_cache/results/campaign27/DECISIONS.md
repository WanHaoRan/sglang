# Campaign 27: the campaign 26 arms on the literature's traces

Started 2026-11-18 20:48Z by request: run the trace replay of `MAINSTREAM_REPLAY.md` (client `scripts/replay_programs.py`)
for original HiCache, the old oracle v2, JIT exact, JIT p50 and the four rules. Held at 21:13Z by the author before any
arm ran: first check that TraceLab and FreeInference replay correctly on the engine as well, then discuss what to run.
Nothing is queued; `queue.txt` is empty and the launcher has not been started.

Launcher `frozen/oracle_scripts_staging/c27_programs/c27.sh` (queue line: label, server arm, host GB, hints, recipe,
lanes, duration, seed, engine, extra server flags), watcher `checks/watch.sh`, smoke scripts `smoke.sh` and
`smoke_traces.sh` beside the launcher. Server settings of every smoke: `Qwen/Qwen3-30B-A3B-Instruct-2507`, context
262,144, bf16 KV, 160 GB host pool (1,627,648 tokens), GPU pool 668,096 tokens, frozen engine `engine_ec_20261111d`,
GPU unit `GPU-c30fd37d` (new after the 20:43Z boot).

## Traces on this host (2026-11-18)

Downloaded into `agent_cache/data/` (git-ignored) and converted into `data/converted/`; the checksums equal the
publishers' and the counts equal `MAINSTREAM_REPLAY.md` section 2.

| Trace | Trees | Requests | Requests with a tool key | Prompt tokens p50, p90 | Output tokens, mean |
|---|---|---|---|---|---|
| AgentX 256K | 393 | 68,266 | 0 % | 89K, 204K | 860 |
| TraceLab v0.0.2 | 8,058 | 665,453 | 86.3 % (1,034 keys, 512 with 5 or more gaps) | 132K, 339K | 589 |
| FreeInference, week of 2026-07-12 | 699 | 72,025 | 92.1 % (822 keys, 402 with 5 or more gaps) | 104K, 305K | 440 |

Tables: `checks/trace_keys.out` (keys and gaps by key), `checks/trace_sizes.out` (what is left under a context cap).

- **AgentX names no tools.** The corpus records a request's time, model, token counts and hashed 64-token prompt
  blocks, no content. Every key is empty, so `--hints dist` sends one pooled entry and the end entry per agent system.
  By stream role the gaps differ (over 60 s: subagent 1.9 %, main 13.5 %, fork 16.0 %, side 19.6 %); role is not a key.
- **A reply with no tool call has no key in this client.** In TraceLab and FreeInference those replies are the human
  waits: 83,966 gaps with p50 66 s and p90 814 s, and 4,909 gaps with p50 88 s and p90 696 s, against tool gaps of
  0.03 to 5 s at the median by key. The client files them under the pooled entry only and the engine picks the pooled
  entry for them. The LMCache client gives such replies the key `_no_call` (`scripts/reply_keys.py`).
- **Tables are kept per agent system** (harness and model): 7 systems in AgentX, 30 in TraceLab, 51 in FreeInference.
- **Sessions are long.** Under a 65,536-token cap TraceLab keeps 3,679 sessions and 40,476 requests (prompt p50 31K,
  p90 52K), FreeInference 194 and 3,841, AgentX 8 and 179. Under 131,072: 6,113 and 110,651; 418 and 19,280; 83 and
  3,139.

## Checks before any GPU time

- `scripts/test_program_replay.py`: 22 tests pass in the container.
- `scripts/test_gap_hints.py`: its three engine-reader cases errored against the engine (`_read_action` missing from
  the test's loader; known since 4 Nov). Loader fixed and one case added that the engine decodes the version 2 table as
  the client sends it; 61 pass. Uncommitted.
- The frozen engine's `/generate` takes `kv_hints` (`GenerateReqInput.kv_hints`), as the chat path does.

## Smoke 1: AgentX 256K, InferenceX recipe, 8 lanes, the four-rules arm, table hints (21:02Z to 21:23Z)

Run `20261118_205653_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160`, `checks/smoke_check.py`.

- **It runs correctly.** 25 requests (8 primers, 17 measured), no client errors, no tracebacks, no warnings. The
  server counted every prompt as sent (up to 253,888 tokens) and every output has the recorded length. Cached tokens
  equal what an infinite cache would hold in every request (delivered over ideal 1.000): device 68.4 %, host 26.5 %,
  storage 0 %, recomputed 5.1 %. The engine read every hint and picked what the client expected (pooled entry 16,
  end 2, empty table 7); request ids in the server log are the client's.
- **This box cannot serve the recipe.** The warm-up (8 cold prompts of 52K to 254K tokens) took 634 s; the longest
  alone 633 s. In the 600 s measured, 17 requests finished: 1.6 a minute. Decode runs at 102-109 ms a step whenever 3
  to 6 of these long requests are in the batch (36-63 ms in campaign 26 with about 30K-token contexts), so a mean
  output of 860 tokens takes about 95 s. Mean TTFT 15.1 s (p50 3.4 s). 242 requests were left unsent.
- **Reading.** A 30-minute arm at 8 lanes would hold 50 to 100 requests, served at the decode's pace, with nothing
  read from SSD at 160 GB in this window. The five arms would not be separable. Not run.

## Smoke 2: TraceLab and FreeInference on one boot of the four-rules arm, 160 GB (21:27Z to 21:56Z)

Run `20261118_212351_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160`, table `checks/smoke_traces_check.out`. Sessions
with a request over 65,536 tokens were dropped, so that a few minutes hold enough requests to exercise the hints; long
prompts were covered by smoke 1.

| Run | Trace, recipe | Hints | Requests | Errors | Prompt count equals sent | Delivered over ideal | Plans | Arrivals matched to a plan |
|---|---|---|---|---|---|---|---|---|
| `tl_dist` | TraceLab, its own (0.1 sessions/s, 16 active, tool waits) | table | 237 | 0 | 237 of 237 | 0.999 | 173 | 169 |
| `fi_dist` | FreeInference, MORI (8 lanes) | table | 118 | 0 | 118 of 118 | 0.999 | 74 | 68 |
| `fi_oracle` | FreeInference, MORI (8 lanes) | exact gap | 84 | 0 | 84 of 84 | 1.136 | 75 | 74 |
| `tl_release` | TraceLab, its own | release | 137 | 0 | 137 of 137 | 1.211 | 0 | 0 |

- **Both traces replay correctly.** No client errors, no tracebacks, no warnings in the server log. Every output has
  the recorded length. No request was served more than 256 tokens short of an infinite cache: TraceLab's prompts,
  built from the previous prompt plus the server's own reply, hit as designed, and FreeInference's 16-token blocks do
  too. Delivered over ideal above 1 in the last two runs is the cache still warm from the run before on the same boot.
- **The engine reads the per-tool tables.** For `tl_dist` it picked a named entry 130 times (78 the key itself, such
  as `Read` or `Bash/grep`; 52 the tool's entry, such as `Bash` for `Bash/ls`), the pooled entry 43 times, the end 17
  times and nothing 47 times (tables still empty); the client's own expectation is the same, count for count. For
  `fi_dist`: named 16, pooled 58, end 3, nothing 41; its tables fill slowly because they are kept per harness and
  model (51 systems, the largest table had 4 entries).
- **Exact and release hints work on this path too:** 75 plans from 82 exact gaps, 74 arrivals matched; under release
  the engine saw 129 links with nothing to plan and 8 ends.
- **Speed at this size:** TraceLab 28 to 31 requests a minute with 16 sessions active (TTFT 0.16 to 0.77 s, 36 to 48 ms
  an output token); FreeInference 14 to 24 a minute with 8 lanes (TTFT 0.15 to 0.89 s, 24 to 25 ms an output token).
- **Not exercised:** nothing was read from SSD (0 % of prompt tokens in all four runs), so no read was staged early
  and rule 1 logged no start decision. Loading the TraceLab trace takes about 75 s before the client's clock starts.

## Probes under memory pressure: 70 GB host pool, MORI's recipe, sessions under 65,536 tokens, four-rules arm, table hints

Two 10-minute runs to exercise the SSD path and to see which trace this box can serve at x1
(`checks/smoke_tl_mori_check.out`, `checks/smoke_fi_mori_check.out`, load estimates in `checks/trace_load.out`).

| | TraceLab, 32 lanes (22:00Z) | FreeInference, 40 lanes (22:15Z) |
|---|---|---|
| requests, errors | 341, 0 | 494, 0 |
| lanes with a request in flight | 27 to 32 of 32 | 8 to 18 of 40 after the first two minutes |
| running, queued (median, second half) | 20, 8 | 10, 0 |
| queue wait per request | mean 13.7 s, median 13.5 s | mean 1.83 s, median 0.03 s |
| TTFT mean, p50 | 15.8 s, 15.3 s | 2.66 s, 0.35 s |
| time per output token, p50 | 99 ms | 46 ms |
| requests a minute | 32.5 | 47.4 |
| prompt tokens from device, host, SSD, recomputed | 4.1, 23.3, 41.2, 31.4 % | 80.2, 3.8, 5.9, 10.0 % |
| requests that read from SSD | most | 46 of 494 |
| rule 1 decisions: at arrival, wait, now | 73, 27, 6 | 30, 26, 0 |
| queued turns that lost their GPU prefix | 70 | 0 |

- **Both run correctly under pressure:** no errors, no tracebacks, no warnings; prompt counts and output lengths
  exact; SSD reads, the read scheduler, rule 1's decisions and the price log all appear in the server log.
- **TraceLab at x1 is bound by decode on this box.** Its lanes are being served nearly all the time (output 589 tokens
  a request, tool waits of 0.2 s at the median), so 32 lanes fill the GPU pool with running requests (token usage
  0.97) and every turn queues about 14 s, whatever its wait was. Fewer lanes fit in GPU memory and read nothing.
  This is campaign 26's 128-user regime: the queue decides.
- **FreeInference at x1 leaves sessions waiting.** With 40 lanes 8 to 18 are in flight, the queue is empty at the
  median, and 9 % of requests read from SSD. TTFT by what the request waited on: 0.68 s after a tool (382 requests),
  3.38 s after a person (71), 19.8 s for a program's first request (41, cold prompts of 20K to 30K tokens, most of
  them in the start burst). This is the regime of campaign 26's 44-user setting.
- **Limits.** Ten minutes each, one run, one arm, and both include their start burst. The week holds 287 programs
  under the cap and the probe started 54, so a 2 h run at 40 lanes needs more weeks or would replay programs. The
  per-system tables stayed sparse: a named entry was picked for 161 of 494 hints in FreeInference.

## Where this stands (22:26Z on 2026-11-18)

All three traces replay correctly on the frozen engine through `replay_programs.py`, under table, exact and release
hints. No arm of the comparison has run. Server stopped, GPU idle, launcher not started. Open with the author: which
trace and load to compare the arms on (the probes point to FreeInference under MORI's recipe near 40 lanes and 70 GB),
whether a reply with no tool call gets its own key, how the tables are scoped, and which rivals join the five arms.

**Figures of the test runs (2026-11-19, asked for): `results/compare_c27_tests/`, redrawn by
`c27_programs/figs_tests.sh`.** The campaign figure scripts read the old client's records, so `adapt_runs.py` first
writes each run of `results/campaign27/tests.txt` in that form under `compare_c27_tests/runs/<label>/` (a request is a
turn of its stream, a warm-up primer a mid-session start, recomputed history is what an infinite cache would have
held and the server did not serve; the server log is cut to the client's window, since four clients shared one boot).
`plot_tests.py` then calls `scripts/timeline.py`, `c26_ec/plot_pricing_vs_jit.py` and `c26_ec/plot_stall.py` unchanged
but for a title override in the last.

- Per run: `timeline_<label>.png` with `events_` and `turns_` CSVs, seven runs.
- All seven: `compare.png` and `ttft_over_time_60s.png` (one legend above the panels; TTFT over time leaves out
  AgentX's warm-up primers).
- Deck style, 1-minute bins: `tests_overview.png` (AgentX at 8 lanes, TraceLab at 32 lanes and 70 GB, FreeInference at
  40 lanes and 70 GB) and `tests_overview_160gb.png` (the four runs on one 160 GB boot).
- Wait split by `stall_ec.py`'s rule, each run over its own returning requests: `ec_wait_split.png` (AgentX 16.99 s =
  4.57 after admission + 12.42 queue; TraceLab 15.34 s = 1.80 + 4.36 queue + 8.81 waiting for its SSD read and then the
  GPU + 0.37 read only; FreeInference 0.73 s = 0.56 + 0.08 + 0.02 + 0.07) and `ec_wait_split_160gb.png`.
- The runs are different workloads, so no figure pairs turns across runs. Request counts, rates and tier shares in
  the figures agree with `checks/smoke_*_check.out`; the bars' "70 % from storage" for TraceLab counts returning
  requests by tier, where the check table's 41 % counts prompt tokens.

## Probes of the campaigns' own schedule on the public traces (2026-11-19 02:09Z to 03:07Z, GPU unit `GPU-dcd42ca3`)

Asked for: every recorded session as one arriving user, with the knobs of campaigns 21-26 (users arriving at a rate,
a cap on users in session, a gap scale, the 1,200 s gap cap). The client does this with no recipe (`--recipe none
--arrival-rate --max-active --gap-scale --gap-cap`); `c27_programs/probe_ours.sh` runs it on the four-rules arm at
70 GB, sessions under 65,536 tokens. Estimates from the traces alone: `checks/our_setup.out` (cap 44) and
`checks/our_setup_cap128.out`; per-minute tables: `checks/probe_*_by_minute.out`.

| Probe | Requests, errors | Mean TTFT | Requests a minute | From SSD (prompt tokens) | State |
|---|---|---|---|---|---|
| TraceLab, x10 gaps, cap 44, users at 0.5/s, 10 min | 313, 0 | 12.0 s | 29.8 | 32 % | saturated from minute 5: 16-18 running, 9-12 queued |
| TraceLab, x10 gaps, cap 44, users at 0.1/s, 25 min | 621, 0 | 13.5 s | 24.4 | 36 % | fine to minute 5 (under 1.7 s), saturated once 44 users are in session: TTFT 10-35 s, 4-12 queued, to the end |
| FreeInference, x1 gaps, cap 44, users at 0.5/s, 10 min | 632, 0 | 1.17 s | 61.6 | 8 % | steady: no queue at the median, 12-30 running, 9-31 % of returning requests from SSD |

- **The schedule runs correctly:** no errors, prompt counts and output lengths exact, in every probe.
- **FreeInference at x1 and 44 users is the 44-user regime of campaign 26** with no gap stretch: a short queue, sessions
  parked in memory, some SSD reads.
- **TraceLab does not settle at 44 users even with x10 gaps and the slow ramp.** Its tool waits are 0.2 s at the median,
  so a user between two human waits sends requests back to back, each 20 to 40 s on this box; 20 to 30 of the 44
  users are in flight, the GPU pool is full of running requests, and returning turns find their prefix on SSD.
- **Estimate for the setting ordered next** (0.1 users/s, cap 128, x1, no turn cut): FreeInference asks 626 output
  tokens a second with 106 users in session, 156 % of the about 400 this box sustains; TraceLab 461 with 46 in
  session, 115 %. Both should saturate, as campaign 26's 128-user setting did. Told to the author with the launch.
- The 2026-11-18 author instruction: gaps are x1 by default from now on.
