# How eight agent-aware KV-cache retention and prefetch papers evaluate their systems (read 17 Nov 2026)

How this was read. I downloaded each paper's arXiv abstract page and latest full text once. HTML pages were converted to text locally; KVLearn has no HTML (HTTP 404), so its PDF was converted with pdftotext. For Continuum I also read the v1 and v4 HTML to compare evaluations. I looked at each linked repository only to confirm how its workload driver or replayer works. I used GitHub API tree listings and raw files at the default branch HEAD, fetched 17 Nov 2026. Section, figure and table numbers below refer to the version named in this table. "Inference" marks my own reasoning. "Not stated" means I searched the full text and did not find it.

| Paper (short name used below) | arXiv | Version read | Version date | Venue as stated on arXiv |
|---|---|---|---|---|
| Continuum (the v7 HTML spells it "Continnum"; v4 is titled "CacheTTL") | 2511.02230 | v7, plus v1 and v4 | 8 Sep 2026 (v1 4 Nov 2025; v4 4 May 2026) | none; the v7 HTML carries an unfilled PVLDB template — [abs](https://arxiv.org/abs/2511.02230), [v7](https://arxiv.org/html/2511.02230v7) |
| CacheWise | 2606.16824 | v1 (only version) | 15 Jun 2026 | none — [abs](https://arxiv.org/abs/2606.16824) |
| Ask the Tool, Don't Guess ("Ask the Tool") | 2609.18849 | v1 (only version) | 16 Sep 2026 | none — [abs](https://arxiv.org/abs/2609.18849) |
| Intent-Aware Caching (IAC) | 2611.09715 | v1 (only version) | 6 Nov 2026 | "NeurIPS 2026" (Comments) — [abs](https://arxiv.org/abs/2611.09715) |
| TokenCake | 2510.18586 | v5 (of 5) | 28 Sep 2026 | "Accepted at EuroSys '27" (Comments) — [abs](https://arxiv.org/abs/2510.18586) |
| KVFlow | 2507.07400 | v1 (only version) | 10 Jul 2025 | none on arXiv — [abs](https://arxiv.org/abs/2507.07400) |
| Adaptive KV Retention for LLM Agents at Human-Approval Timescales ("AKR") | 2608.30830 | v1 (only version) | 31 Aug 2026 | none — [abs](https://arxiv.org/abs/2608.30830) |
| KVLearn ("To Keep or Not to Keep") | 2610.28637 | v1 (only version; PDF) | the PDF watermark reads "10 Aug 2026", which is odd for a 2610 identifier | "Accepted to ACM SYSTOR 2026" (Comments) — [abs](https://arxiv.org/abs/2610.28637), [PDF](https://arxiv.org/pdf/2610.28637v1) |

## 1. Evaluation mode: live agents, trace replay on a real engine, simulation, or synthetic workload?

### Takeaway
Only Continuum has an experiment where a real agent runs tasks end to end, executing real tools, against the engine under test. That experiment is Fig. 12 (plus a small RL micro-benchmark, Table 5). Its headline curves are replays of GPT-5 traces. The other papers fall into three groups:
- **Trace replay on a real engine:** CacheWise, IAC and Ask the Tool's KV results. IAC replays chat, not agent sessions.
- **Mixed:** TokenCake runs its own multi-agent applications live, but replays tool outputs with sampled durations. AKR serves a synthetic approval workload live and adds a CPU-only "measured-cost replay".
- **Synthetic or simulated:** KVFlow drives a real engine with synthetic prompts. KVLearn is, by its own limitations paragraph and its released code, not integrated with a live serving engine.

### Cited Findings
**Continuum (v7)**
- **Main results are trace replay.** "We conduct the trace replay experiments for SWE-Bench, BFCL, and OpenHands workloads" (§6.2; Figs. 8–11). "For results other than the real SWE-Bench experiments in Figure 12, we evaluate on two collected workloads running GPT-5 ... and using poisson distribution for the arrival pattern of agent programs" (§6.1 Datasets, footnote 2) — [v7](https://arxiv.org/html/2511.02230v7)
- **One experiment uses live agents.** "We test Continnum running a real SWE agent for 500 tasks in SWE-Bench-Verified in Company A's internal H100 testbed. We set up our agent client environment by adding a job distributor for the SWE-Bench platform that distributes agents according to a Poisson distribution" (§6.2 "Real SWE-Agent in Distributed Setting", Fig. 12) — [v7](https://arxiv.org/html/2511.02230v7)
- **RL micro-benchmark.** "We tested the OpenHands Agent with GLM-4.5-fp8 training on Multi-SWE bench ... for rollout generation. The hardware setup is an 8xH100 node. We compared with the concurrent RL work ThunderAgent ... on inference steps per minute, as reported by the original paper" (§6.4, Table 5) — [v7](https://arxiv.org/html/2511.02230v7)
- **The paper's own labels.** The introduction calls the two kinds "emulated and real cases": "up to 8.18x improvements ... in both emulated and real cases" (§1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Synthetic transformation of traces.** Fig. 14 "simulate[s] more-turn scenarios on SWE-Bench by repeating the trace (1× to 5×) while inversely scaling the token lengths to emulate more turns but make total token fit within the context window" (§6.3) — [v7](https://arxiv.org/html/2511.02230v7)
- **v1 has no live-agent experiment.** It has no OpenHands, Poisson, SSD or Company A results (keyword counts 0 in v1, non-zero in v4 and v7). v4's evaluation matches v7's — [v1](https://arxiv.org/html/2511.02230v1), [v4](https://arxiv.org/html/2511.02230v4)

**CacheWise (v1)**
- "We replay the traces collected in our dataset for our experiments deterministically" (§6.1 Workloads). The engine is the authors' vLLM-based prototype (§5.3) — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool (v1)**
- **Accuracy study (§4) re-executes real tool calls.** "We replayed the same calls on the same machine under four conditions: idle; contended throughout; contended from a third of the way in; and contended until the halfway point" (§4.3). §2.1 replays leaderboard commands on an idle sandbox and with busy neighbours (Fig. 3) — [v1](https://arxiv.org/html/2609.18849v1)
- **Live agent runs, but only to measure the harness.** Table 4 compares harness configurations "On our sample of SWE-bench Verified tasks" (§4.3) — [v1](https://arxiv.org/html/2609.18849v1)
- **KV results (§5) are a replay on a real engine.** "We replay the agent sessions of Section 4 across all methods on a 4×H100 SXM instance in two configurations: GPU memory only, and GPU memory with DRAM as an offload tier" (§5.1) — [v1](https://arxiv.org/html/2609.18849v1)
- **Lying reporters are only simulated.** "In a preliminary simulation, lying raised the memory the liars held by 6%, and the two ledgers took 5% of it back" (§5.3) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC (v1)**
- **Replay against SGLang.** "During replay, all policies use the same request order, cache capacity, model, hardware setup, and arrival process" (App. A.4 "Replay setting") — [v1](https://arxiv.org/html/2611.09715v1)
- **Real timing for two datasets.** For WildChat and ShareChat "we preserve the original inter-turn delays within each conversation", with new conversations started by "a Poisson process at the target arrival rate" (App. A.4) — [v1](https://arxiv.org/html/2611.09715v1)
- **Synthetic timing for the other two.** For ShareGPT and LMSys-Chat, "inter-turn delays are sampled from a fixed long-tailed distribution with mean 4.6 and variance 1" (§5.1; App. B.6). App. B.7 adds ON/OFF, Pareto and diurnal arrivals and a constant 100 s delay — [v1](https://arxiv.org/html/2611.09715v1)
- **The workload is chat, not agents.** It is multi-turn conversations, with no tool calls (§5 Workloads; Table 2) — [v1](https://arxiv.org/html/2611.09715v1)

**TokenCake (v5)**
- **Tools are recorded once, then replayed.** "We first execute each application with live tool endpoints and record both the tool outputs and latency distributions. During evaluation, the tool server replays the recorded outputs while sampling durations from the collected distributions with a fixed seed" (§7.1 Workload Generation) — [v5](https://arxiv.org/html/2510.18586v5)
- **Requests and arrivals.** "User requests are synthesized from the ShareGPT ... and AgentCode ... datasets. Application arrivals follow a Poisson distribution at varying rates" (§7.1) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow (v1)**
- **Synthetic prompts.** "We generate synthetic input prompts by randomly sampling token sequences with controlled lengths for both parts" (§4.1) — [v1](https://arxiv.org/html/2507.07400v1)
- **"Realistic Workflow Simulation".** This is PEER-template workflows whose agent prompts are LLM-generated, run on the same engine (§4.2) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR (v1)**
- **Two modes.** "We use live serving for end-to-end measurements and measured-cost replay for broader workload sweeps. Replay conditions contain 4,000 requests with Poisson arrivals at multiples of λ_crit" (§4.1) — [v1](https://arxiv.org/html/2608.30830v1)
- **The replay is not run on GPUs.** "Replay, trace generation, and plotting are CPU-only and complete within minutes" (App. G) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn (v1)**
- **No live engine.** The paper describes P/D clusters (§5.1 Hardware). Yet its Limitations paragraph says "integration with a live Mooncake or DistServe cluster and validation against closed production traces are reserved for future work" — [PDF](https://arxiv.org/pdf/2610.28637v1)
- **Bandwidth is a simulated parameter.** "We vary fabric bandwidth β ∈ {50, 100, 200, 400} Gbps to simulate heterogeneous cluster configurations ... T(b) is a configuration parameter fed directly into CARS" (§5) — [PDF](https://arxiv.org/pdf/2610.28637v1)
- **The released code is a simulator.** The repo's layout lists "apps/simulate.cpp Zipf multimodal trace vs No-Cache / LRU-Pool / KVLearn". Its README says "Tensor movement (RDMA / NCCL) stays in your data plane; this library only decides keep vs discard" — [repo README](https://github.com/FastLM/KVLearn/blob/HEAD/README.md)

### Inferences
- **Classification of each result.**
  - **Continuum:** (ii) for Figs. 8–11 and 13–16 (with synthetic transforms in Fig. 14); (i) for Fig. 12. Table 5 is (i) for Continuum's own number. Its ThunderAgent comparison is worded "on inference steps per minute, as reported by the original paper". That may mean the ThunderAgent figure was copied from ThunderAgent's paper rather than measured in the same run; the wording is ambiguous.
  - **CacheWise:** (ii).
  - **Ask the Tool:** §4 is a measurement study with real tool executions and live agent runs. §5 is (ii). §5.3 is (iii).
  - **IAC:** (ii), with (iv)-style synthetic timing for ShareGPT and LMSys-Chat. It is not an agent workload.
  - **TokenCake:** a hybrid of (i) and (ii). Live agents and live LLM calls, but tools are replayed with sampled delays.
  - **KVFlow:** (iv) on a real engine.
  - **AKR:** (iv) on a real engine for the live runs, (iii) for the sweeps.
  - **KVLearn:** (iii) with A100-calibrated cost parameters. This is inferred from the stated future work and the released simulator. The paper never calls its main results simulated.
- **KVLearn is out of scope.** It has no agents, tool calls or tool pauses (keyword search for "agent" and "tool" in the PDF text found none). It is a disaggregated prefill/decode retention paper evaluated on ShareGPT and image and video QA streams. The report should flag it as not agent-aware.

### Gaps
- No version of Continuum (v1, v4, v7) says how the replay injects tool time or whether output lengths are forced. A filename search of the released repo found no replay driver (Section 10).
- Ask the Tool does not say how the §5 replay reproduces tool gaps or outputs.
- I did not read Continuum v2, v3, v5 or v6.

## 2. Tools: executed in a sandbox, replayed as sleeps, or sampled? Does tool latency change with load?

### Takeaway
No paper's KV-cache result uses tool latencies that respond to the load the system under test creates. Only Continuum's live experiment (Fig. 12) executes real tools (SWE-bench Docker containers). Only Ask the Tool measures how tool durations change with machine contention, and that is in its accuracy study, not its KV study. The others use durations taken from traces (CacheWise, Continuum's replays as far as can be told), durations sampled from recorded distributions (TokenCake), or synthetic waits (AKR, IAC). KVFlow and KVLearn have no tools.

### Cited Findings
**Continuum**
- **Recorded tool times.** Table 2's "Tool Time(ms)" column gives (mean, SD): SWE-Bench (925, 3,550); BFCL v4 (1,923, 2,133) (§3.1). The table does not say whether this is per call or per program — [v7](https://arxiv.org/html/2511.02230v7)
- **Replay injection not stated.** No text says how these gaps are reproduced in replay — [v7](https://arxiv.org/html/2511.02230v7)
- **Live run uses real Docker tools.** "SWE-Bench's time limit for environment Docker containers to prevent hanging. When the baseline's running time exceeds 15 minutes it will be preempted and treated as a failure case" (§6.2) — [v7](https://arxiv.org/html/2511.02230v7)
- **Released agent config.** It sets `environment_class: docker`, `timeout: 60` and `step_limit: 50` — [swebench.yaml](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml)

**CacheWise**
- **Tool time is left out of the metric.** "We exclude tool execution durations as they are determined by the external environment and independent of the serving system" (§6.1 Metrics) — [v1](https://arxiv.org/html/2606.16824v1)
- **Durations come from the trace.** Fig. 19's legend reads "Hint=Trace replay with ground truth tool durations" (§6.5) — [v1](https://arxiv.org/html/2606.16824v1)
- **Human pauses become session boundaries.** "During replay, we consider human idle periods as session boundaries" (§6.1) — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **Tool time depends on load.** "We replayed the commands recorded in leaderboard runs on an idle sandbox. The same command finished an order of magnitude faster (Figure 3a). Restricting the sandbox's CPU quota changed almost nothing. Adding busy neighbours, however, made most calls several times slower (Figure 3b)" (§2.1) — [v1](https://arxiv.org/html/2609.18849v1)
- **Four load conditions** were applied to the same calls (§4.3). App. C Fig. 16 follows "two real replays of the same editable install" across a load change — [v1](https://arxiv.org/html/2609.18849v1)
- **Not stated for §5.** The KV replay does not say how tool durations are reproduced — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **No tools.** Gaps are human inter-turn delays: original ones (WildChat, ShareChat) or sampled ("mean 4.6 and variance 1") (§5.1; App. A.4) — [v1](https://arxiv.org/html/2611.09715v1)
- **Released replayer samples log-normal gaps.** `inter_turn_gaps` returns `rng.lognormal(synthetic_delay.mu, synthetic_delay.sigma, size=len(conv.turns) - 1)`. Conversation starts use `rng.exponential(1.0 / rate, ...)` — [iac/data.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/iac/data.py)

**TokenCake**
- **Recorded outputs, sampled durations.** Outputs are recorded, and durations are sampled "from the collected distributions with a fixed seed, ensuring that all baselines receive identical tool behaviors and stall times" (§7.1) — [v5](https://arxiv.org/html/2510.18586v5)
- **Tool latency table.** Table 1 lists MCP tool latencies, "compiled from the official MCP documentation and our empirical measurements". For example: File System, CPU, 100 ms, variability 50 ms; Git, CPU, 100 ms, 100 ms–1 s (§2.1) — [v5](https://arxiv.org/html/2510.18586v5)
- **Noise sensitivity.** §7.5 injects multiplicative noise: "actual execution time is drawn from [t·(1−s), t·(1+s)]" — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **No tools.** Workflows are chains of LLM agents (sgl.function per agent) (§3.3, §4) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **The pause is a human approval wait.** Waits come from "a lognormal distribution (σ=1), an exponential distribution, and a mixture of short approvals (exponential, mean 60 s, weight 0.5) and long waits (lognormal, σ=0.7, mean 3,540 s). Each family is scaled to have mean wait W_ref=1,800 s" (§4.1) — [v1](https://arxiv.org/html/2608.30830v1)
- **Stated limitation.** "Approval outcomes are LLM-simulated and wait durations are synthetic" (§5 Limitations) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **No tools** (keyword search of the PDF text) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- Ask the Tool reports that contention makes tool calls several times slower. Benchmarks with load-independent tool delays therefore understate one coupling: tool sandboxes and the inference host competing for CPU, I/O or network. A paper that prefetches the prefix cache from SSD ahead of a tool's return should either run real tools or vary the delays with load.
- TokenCake's fixed-seed duration sampling makes runs comparable across systems. It also removes the tail correlation between a call and its environment that Ask the Tool documents.

### Gaps
- Continuum (replay) and Ask the Tool (§5) do not say whether tool gaps are slept for their recorded length, compressed, or sampled.
- CacheWise does not say whether recorded tool durations are slept in wall-clock time. Its "Hint" legend implies they come from the trace.

## 3. Model outputs: free generation or forced? How is divergence handled?

### Takeaway
Wherever the paper states the mechanism, outputs are forced or tightly controlled:
- AKR fixes 1 and 8 output tokens for gate requests and 64 for background requests.
- KVFlow fixes output lengths with greedy decoding.
- IAC's released replayer sends the recorded conversation with `max_tokens=1`.

The trace replays (Continuum, CacheWise, Ask the Tool) replay traces produced by other models (GPT-5, Claude) on smaller open models and do not say how outputs are handled. Free generation is confirmed only for Continuum's live experiment (through its released client) and Ask the Tool's harness runs. TokenCake's live applications presumably generate freely, but the paper does not say. Continuum handles divergence by reporting pass rate next to delay.

### Cited Findings
**Continuum**
- **Traces come from a different model.** The replayed traces were generated by GPT-5: "We use GPT-5 for the better model capabilities to ensure that the workflow generated are mostly correct. Base small models often fail to accomplish the task" (§6.1, footnote 2). They are served on Llama-3.1-8B/70B and Gemma-3-12B (§6.1) — [v7](https://arxiv.org/html/2511.02230v7)
- **v1 wording.** "We use GPT-5 for the better model capabilities although we cannot evaluate on them" (§6.1, footnote 1) — [v1](https://arxiv.org/html/2511.02230v1)
- **Token lengths are rescaled.** BFCL is "scaled down ... by 0.4 to fit at least 100 request in the context window of llama-3.1 (128k tokens)" (§6.1). Fig. 14 "inversely scal[es] the token lengths" (§6.3) — [v7](https://arxiv.org/html/2511.02230v7)
- **Live run measures divergence as pass rate.** "We measure the per-job finish time and collect the pass rate of each agent program ... Continnum consistently outperforms baselines in terms of average delay when pass rates are equal. Notice that Continnum actually has higher pass rate than baselines" (§6.2, Fig. 12) — [v7](https://arxiv.org/html/2511.02230v7)
- **Released client generates freely.** It sends `extra_body = {"ignore_eos": False, "job_id": ..., "is_last_step": ...}`, with `max_completion_tokens` defaulting to 2048 — [vllm_model.py](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/src/minisweagent/models/vllm_model.py). The config sets `temperature: 0.0` — [swebench.yaml](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml)

**CacheWise**
- **Mechanism not given.** The replay is "deterministic" (§6.1); output handling is not stated. The serving model is Qwen2.5-Coder-32B-Instruct (§6.1) — [v1](https://arxiv.org/html/2606.16824v1)
- **Traces come from Claude.** They come from "users of Claude Code" (§3). The released session I sampled records `"model": "claude-opus-4-1-20250805"` — [paper](https://arxiv.org/html/2606.16824v1), [events.json](https://github.com/cachewise-project/cachewise-coding-traces/blob/HEAD/parsed_traces/project_004/session_0006/events.json)

**Ask the Tool**
- **Not stated for §5.**
- **Harness runs generate freely.** Table 4 reports how each harness configuration changed solved tasks, steps (stock mean 35.9), completion tokens (14,764), tool time (205.1 s) and wall time (495.8 s), with 95% bootstrap intervals (§4.3) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Not stated in the paper.**
- **Released replayer forces the history.** It builds each request from recorded user and assistant turns (`messages += [{"role": "user", ...}, {"role": "assistant", "content": t.assistant}]`) — [iac/data.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/iac/data.py). It sends `"max_tokens": max_tokens` with `--max-tokens` defaulting to 1, `"temperature": 0.0`, and streaming, and measures TTFT — [scripts/replay.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/scripts/replay.py)

**TokenCake**
- **Not stated.** Generation length and divergence handling are not given.
- **Fixed graphs and tools.** "The current evaluation uses static application graphs for reproducibility" (§9), and tools are identical across baselines (§7.1) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Controlled lengths.** "We adopt deterministic decoding (temperature = 0, greedy sampling) to ensure consistent latency measurements" (§4.1). Output lengths are controlled (for example "8192/32/32" fixed/dynamic/output, §4.1; "dynamic and output token lengths are fixed at 256", §4.2) — [v1](https://arxiv.org/html/2507.07400v1)
- **Correctness argument.** KVFlow "only modifies system-level cache management ... it is guaranteed to preserve the semantic correctness of the output" (§4) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Fixed, short outputs.** "All requests use greedy decoding (temperature=0). A gate request generates one token at suspension and eight tokens at resume. Each of the C=48 closed-loop active clients ... generates 64 tokens per request" (App. F.1) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Not stated** — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **Replay papers imply forced lengths.** Continuum, CacheWise and Ask the Tool replay frontier-model traces on 8B–70B open models. A deterministic replay then almost certainly forces prompt contents or token counts, and possibly output lengths. None of them says how. This is inference; the mechanism is not stated.
- **Forced outputs weaken TTFT claims under load.** With `max_tokens=1` (IAC's released code) or 1 and 8 tokens (AKR), decode never competes for GPU memory or batch slots in the way real agent turns do.

### Gaps
- None of the trace-replay papers states its output-length policy, or what happens when the served model's tokenization or context window differs from the trace's.

## 4. Tasks, benchmarks, datasets and traces (with versions and counts; who collected them)

### Takeaway
The agent traces are mostly the authors' own collections:
- Continuum: GPT-5 runs of mini-swe-agent on SWE-bench and of BFCL v4 Web Search, 100 each, plus an OpenHands Go example.
- CacheWise: Claude Code sessions from lab participants; 94 sessions are released.
- Ask the Tool: public leaderboard runs (about 3,000 trajectories), OpenHands evaluation runs, and its own 100 Qwen runs.

The rest use other material:
- TokenCake: its own DAG applications with ShareGPT and AgentCode requests.
- AKR: τ²-bench prefixes.
- IAC: chat datasets.
- KVFlow: synthetic prompts.
- KVLearn: ShareGPT and image and video QA.

No production agent trace is used. IAC uses a cloud trace only to compare against WA, and KVLearn says production traces are future work.

### Cited Findings
**Continuum**
- **v7 traces.** "We collect and analyze 100 mini-swe-agent traces ... running SWE-Bench ... We also collect 100 traces from BFCL V4 Web Search ...; both workloads use GPT-5 as the base model" (§3.1). Table 2 gives (mean, SD): SWE-Bench turns (10.9, 2.1), tokens per program (70,126, 19,732); BFCL v4 turns (6.3, 2.3), tokens (93,256, 68,687) — [v7](https://arxiv.org/html/2511.02230v7)
- **Agent and task sources.** mini-swe-agent is described as "SWE-bench official agent that rank #5 on leaderboard by Apr 13th" (footnote 3). BFCL is "the latest version of BFCL V4 (Web Search category)". OpenHands is "the multi-SWE-bench ... example in the official repo for the Go language" (§6.1). The live run uses "500 tasks in SWE-Bench-Verified" (§6.2). The RL benchmark uses Multi-SWE-bench rollouts (§6.4) — [v7](https://arxiv.org/html/2511.02230v7)
- **v1 gives a different count.** "we collect and analyze 500 traces from mini-swe-agent ... running SWE-Bench ... and BFCL V4 Agentic" (§3.1), and describes mini-swe-agent as "a 100-line agent that rank #8 by Oct 19th" — [v1](https://arxiv.org/html/2511.02230v1)
- **Traces not yet released.** "We will open-source our traces, code, and the agent serving testbed" (§1) — [v7](https://arxiv.org/html/2511.02230v7)

**CacheWise**
- **Collection.** "traces collected from users of Claude Code ... performing real-world software development and engineering tasks on various open source projects". Footnote: "We collected anonymized traces from consenting participants within our lab." The traces include "user instructions, model outputs, tool calls, their results ... number of tokens, timestamps for each message, human interventions" (§3) — [v1](https://arxiv.org/html/2606.16824v1)
- **Shape of the traces.** Sessions last "36 min at median, > 2.6 hours at tail" (Fig. 5). Tool-triggered requests are "20× more frequent than user-initiated requests at the median" (Fig. 4) — [v1](https://arxiv.org/html/2606.16824v1)
- **Train/test split.** "We randomly sample 80% of the sessions for offline training of the predictor, and use the remaining traces for our experiments" (§6.1) — [v1](https://arxiv.org/html/2606.16824v1)
- **Counts are only in the repo.** The paper states no session count. The released manifest (version 1.0) lists 94 sessions, 20,634 LLM calls and 11,617 tool calls across 11 projects — [manifest.json](https://github.com/cachewise-project/cachewise-coding-traces/blob/HEAD/parsed_traces/manifest.json)

**Ask the Tool**
- **Corpora.** "Throughout we read these leaderboard runs (six submissions, about three thousand trajectories), the OpenHands evaluation runs ..., and our own runs on a hundred random SWE-bench Verified tasks using Qwen 3.8 Flash" (§2.1, footnote 1) — [v1](https://arxiv.org/html/2609.18849v1)
- **Census.** The census adds Terminal-Bench and OpenTelemetry traces (Exgentic). Labels come from "two frontier language models; a third model arbitrates their disagreements, and a human-labelled subset checks them". Terminal-Bench and the OpenTelemetry traces do not record durations, so their shares come from "a replay of the labelled calls" (App. A) — [v1](https://arxiv.org/html/2609.18849v1)
- **Base models.** "two open-weights models ... and two closed frontier models ..., each at its highest reasoning budget" were run on the same tasks (§4.3) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Datasets (Table 2: samples, average length, average turns).**

  | Dataset | Samples | Avg. length | Avg. turns | Timestamps |
  |---|---|---|---|---|
  | ShareChat | 142,808 | 1,250 | 4.62 | yes |
  | WildChat | 1,039,785 | 736 | 2.54 | yes |
  | LMSys-Chat | 1,000,000 | 284 | 2.02 | no |
  | ShareGPT | 94,145 | 442 | 3.51 | no |

  — [v1](https://arxiv.org/html/2611.09715v1)
- **Split.** "the first 40,000 conversations are used as the history/training split, and the next 10,000 conversations are used as the replay trace" (App. A.4) — [v1](https://arxiv.org/html/2611.09715v1)
- **Production trace.** "a production trace from a major cloud provider" is used only to compare against WA (§5; App. B.5) — [v1](https://arxiv.org/html/2611.09715v1)

**TokenCake**
- **Applications.** "Code-Writer ... orchestrates 11 agent types with frequent function calls to file I/O, search, and external test tools". "Deep Research ... models a research workflow with fewer agents but deeper dependency chains" (§7.1, Fig. 1) — [v5](https://arxiv.org/html/2510.18586v5)
- **Inputs.** Requests are "synthesized from the ShareGPT ... and AgentCode ... datasets". Figure 9 labels two datasets "D1, D2" (§7.1–7.2) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Synthetic workflows.** A "sequential agentic workflow composed of 10 agents" with random-token prompts (§4.1) — [v1](https://arxiv.org/html/2507.07400v1)
- **PEER workflows.** "each workflow consists of four agents, instantiated using the workflow templates provided by PEER ... We use the Financial QA dataset from PEER as the workflow input" (§4.2) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Gate contexts.** "From the 2,624 released retail and airline runs [of τ²-bench], we retain trajectories containing a tool the environment marks ToolType.WRITE ... and truncate each immediately before the first such call, yielding 2,161 approval-gated prefixes". The median context is 8.4K tokens: 5.4K shared and about 3.0K private (§2) — [v1](https://arxiv.org/html/2608.30830v1)
- **Prompt pools.** A "1,160-prompt background pool" and a "disjoint 1,000-prompt pool" of gate contexts (App. F.1) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Streams.**
  - "ShareGPT ...: 104 requests sampled". "104" is probably 10⁴ with the superscript lost in extraction; the Table 1 header says "N_req = 5000".
  - MM-Session: LLaVA-Instruct-150K, "We replay 2 000 randomly-ordered pairs" over "412 unique images" with Zipf exponent α = 1.08. The Table 1 header says "80 images, Zipf α=1.1".
  - Video-Stream: ActivityNet-QA, "20 videos ... 1 800 QA pairs". The Table 1 header says "10 clips × 60 s".

  (§5.1, Table 1) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **Who collected the agent traces.**
  - Continuum: the authors, with GPT-5.
  - CacheWise: lab participants using Claude Code.
  - Ask the Tool: public leaderboards plus 100 of its own runs.
  - TokenCake: the authors' own applications.
  - AKR: τ²-bench, a public benchmark with simulated users.
- **KVLearn's workload sizes are internally inconsistent** between §5.1 and Table 1.

### Gaps
- **CacheWise:** the held-out session count and how sessions were sampled into each N are not stated.
- **TokenCake:** which of ShareGPT and AgentCode is D1 and which is D2 is not stated.
- **Continuum:** the model used to produce the OpenHands traces is not stated.

## 5. Agent framework and serving–agent integration (API, what crosses the boundary, who parses tool calls)

### Takeaway
Three integration styles appear.

**Engine parses the model's tool calls; client sends a session ID**
- Continuum: program_id from the client; the engine parses the tool name from its own output.
- CacheWise: the engine parses tool_name and tool_args itself.

**Client or framework sends explicit structure**
- TokenCake: a registered DAG, plus call_start and call_finish with time estimates.
- KVFlow: agent identity and steps-to-execution per request.

**An outside controller sends hints**
- Ask the Tool: refresh, release and query naming a context by response ID.

IAC classifies intent on the engine side. AKR's controller needs only the configured offered load. KVLearn's control plane sits at the KV pool.

### Cited Findings
**Continuum**
- **Client side.** "On the client side, we attach a program identifier (program_id) to every inference request" (§5) — [v7](https://arxiv.org/html/2511.02230v7)
- **Engine parses tool calls.** "The handler parses tool calls from LLM outputs, tracks per-tool latency using observed inter-request intervals within the same program_id" (§5). "In SWE-Bench ... We extract the string within the bash block and use the first word afterwards as the tool call name" (§5.1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Agents used.** mini-swe-agent (SWE-bench), the BFCL v4 agent, and OpenHands (§6.1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Released client.** It calls `client.chat.completions.create(..., extra_body={"ignore_eos": False, "job_id": ..., "is_last_step": ...})`, an OpenAI-compatible API carrying a job ID and a last-step flag — [vllm_model.py](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/src/minisweagent/models/vllm_model.py)

**CacheWise**
- **Engine parses tool metadata.** The block manager stores "tool_name, tool_args and T_i (time when the tool call was generated), which can be parsed by the vLLM engine itself" (§5.3) — [v1](https://arxiv.org/html/2606.16824v1)
- **Agent framework.** Claude Code is only the trace source (§3). The replay client is not described — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **A wrapper reads progress.** A harness wrapper (side channel) emits events: "Every event has the same shape: a call id, the work done, the total if known, the phase, and a timestamp" (§3.2) — [v1](https://arxiv.org/html/2609.18849v1)
- **The engine gets one patched route.** "The patch adds one route that names a context by the id of an earlier response and offers three operations: refresh ..., release ..., and query ... The scheduler, its eviction rule, and the model are untouched" (App. D) — [v1](https://arxiv.org/html/2609.18849v1)
- **A controller sends the hints.** "The controller runs beside the harness and reads each session's progress stream" (App. D) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Engine-side intent and logs.** `node.intent = IntentClassifier(Req.conversation)`. Logs store "the session ID, request ID, turn index, timestamp, and intent label" (App. A.2) — [v1](https://arxiv.org/html/2611.09715v1)
- **Released replayer passes more.** It posts to `/v1/chat/completions` with `"custom_params": {"iac_timestamp": req.arrival, "iac_conv_id": req.conv_id}` — [scripts/replay.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/scripts/replay.py)

**TokenCake**
- **DAG registration.** "The frontend extends vLLM's OpenAI Chat Completion endpoint with a stateful graph registration interface. Users register a DAG of agents and function call nodes before submitting inference requests" (§6.1) — [v5](https://arxiv.org/html/2510.18586v5)
- **Tool lifecycle endpoints.** "call_start carrying the request identifier and an initial time estimate ... call_finish carrying the request identifier and the actual elapsed time" (§6.2) — [v5](https://arxiv.org/html/2510.18586v5)
- **Duration hints.** FuncNode stages and `predict_time` (§3.1, Fig. 5) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Metadata in each request.** "we perform a just-in-time substitution of the LLM call to embed workflow metadata into the HTTP request. This metadata includes the identity of the current agent and the steps-to-execution of all agents" (§3.3) — [v1](https://arxiv.org/html/2507.07400v1)
- **Client ID and prefix marker.** "we assign a unique client ID to each application", plus a user primitive or a heuristic to mark the end of the fixed prefix (§3.3) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Load is the only runtime input.** "At runtime the controller requires only the offered load λ to evaluate t_2" (§3.4). "In our experiments, λ is the configured offered rate for the condition" (App. D) — [v1](https://arxiv.org/html/2608.30830v1)
- **vLLM configuration.** "only eviction_policy and cache_policy_module_path differ where applicable" (App. F.1) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Control plane at the KV pool.** "The control plane intercepts two events: (1) block arrive, ..." (§4) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **Who sends hints.** Only TokenCake and KVFlow have the agent side send structured hints (DAG, time estimates, steps-to-execution). Continuum and CacheWise infer tool identity by parsing the model's own output. Their design therefore assumes the engine sees the raw completion, which a gateway that parses tool calls before the engine would break.
- **No end-of-session signal in Continuum's paper.** The released client has one (`is_last_step`); the paper text only says "If the step is not signified to be the last step (ex. parsed to contain a tool call)" (§5.2).

### Gaps
- How AKR's live driver signals a suspension or a resume to the engine is not stated.
- How CacheWise's replay client tells sessions apart is not stated.

## 6. Load: concurrency, arrival process, run length, warmup

### Takeaway
- **Open Poisson arrivals of whole programs or applications:** Continuum, TokenCake, IAC (new conversations).
- **Closed batch** (N sessions started together and run to completion): CacheWise.
- **Fixed concurrency:** KVFlow; AKR's background clients (closed loop, 48 clients).

Run length and warmup are almost never reported. The exceptions are IAC's 10K-conversation replay (about 33 minutes at 5 new conversations/s), KVFlow's prefix warmup, and AKR's 15 live runs in 14 h.

### Cited Findings
**Continuum**
- **Poisson program arrivals** (§6.1). "In Figure 13, we set the job per second to be 0.13" (§6.3) — [v7](https://arxiv.org/html/2511.02230v7)
- **Released runner.** It sweeps `JPS_VALUES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]` for `fcfs` and `continuum` — [run_experiments.py](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/run_experiments.py). The JPS mode pre-computes arrivals with `random.expovariate(jps)`; a fixed-concurrency `--workers` mode also exists — [swebench.py](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/src/minisweagent/run/extra/swebench.py), [README](https://github.com/Hanchenli/vllm-continuum/blob/main/README.md)
- **Not stated:** programs per replay run, run length and warmup — [v7](https://arxiv.org/html/2511.02230v7)

**CacheWise**
- **Closed batch.** "Our experiments launch N independent coding agent sessions concurrently on a single serving instance and runs until all sessions complete ... We repeat each experiment several times" (§6.1) — [v1](https://arxiv.org/html/2606.16824v1)
- **Values of N.** The text discusses N ≤ 10 and N > 10 (§6.2), N = 30 and 40 (Fig. 16), N = 30 (Fig. 18), N = 30, 40, 50 (Fig. 19), and N = 40 (Fig. 20) — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **Not stated:** concurrency, arrivals and run length for §5 — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Example of the replay's size.** "replaying 10,000 new conversations at an arrival rate of 5 new conversations/s takes approximately 2,000 seconds, or 33.3 minutes ... roughly 3 turns on average ... about 30,000 total requests and an average request rate of approximately 5×3=15 requests/s" (App. A.4) — [v1](https://arxiv.org/html/2611.09715v1)
- **Rates used.** TTFT uses "a request rate of 5 req/s" (§5.2); throughput uses 10 and 20 req/s (§5.3) — [v1](https://arxiv.org/html/2611.09715v1)
- **Released replayer.** It defaults to `--time-scale 1.6`, `--start 1800`, `--warmup-end 2100` and `--end 3030` (trace seconds), and excludes requests before `warmup_end` from the summary — [scripts/replay.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/scripts/replay.py)

**TokenCake**
- **Rates.** Poisson application arrivals over "QPS (0.05–1.0 applications/s)" (Fig. 9) — [v5](https://arxiv.org/html/2510.18586v5)
- **Applications per run.** "20 apps" (Fig. 11), "five applications" (Fig. 12); Fig. 9 not stated — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Single workflow.** "batch size = 1 ... We first warm up the cache by executing each agent's fixed prompt multiple times ... We then execute the 10-agent workflow ten times" (§4.1) — [v1](https://arxiv.org/html/2507.07400v1)
- **High concurrency.** "For each setting, we choose a proper concurrency that the GPU can accommodate without exhausting memory for prefix caching", for example "64 concurrent workflows" (§4.2) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Saturating closed loop.** "a closed-loop experiment with 48 concurrent active clients, sufficient to saturate the node" (§2). C = 48 was chosen from a sweep using "a 90 s window, 64-token completions" (App. A) — [v1](https://arxiv.org/html/2608.30830v1)
- **Live operating point.** "We therefore evaluate live serving at 2λ_crit", where λ_crit = C_2/W_ref (§3.3, §4.1) — [v1](https://arxiv.org/html/2608.30830v1)
- **Run budget.** Table 7: "Live end-to-end serving (15 runs)", 14 h wall clock, 55 GPU-hours (App. G) — [v1](https://arxiv.org/html/2608.30830v1)
- **Replay.** "4,000 requests with Poisson arrivals" (§4.1); 12 conditions "spanning 0.5–3×λ_crit at C_2=425" (App. F.3) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Arrival process not stated.** Fig. 4 covers "a 2-hour serving window for MM-Session". The predictor needs "a warm-up of n_0 ≈ 200 requests" (Limitations) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **AKR's live runs may be shorter than the waits.** 14 h over 15 runs is about 56 minutes per run. That is short relative to waits with a 30-minute mean and a lognormal component averaging 3,540 s. The paper does not say whether waits were compressed in live runs, only that metrics are "measured after draining the serving window" (App. F.2).
- **CacheWise's closed batch** starts all sessions at once and has no arrival process. Results mix start-up effects with steady state.

### Gaps
- **Not stated anywhere:** warmup and steady-state windows for Continuum, CacheWise, Ask the Tool, TokenCake and KVLearn.
- **IAC ambiguity:** is the TTFT experiment's "5 req/s" requests or new conversations per second? App. A.4 uses 5 conversations/s ≈ 15 requests/s as its example.

## 7. Stack: engines and versions, models, GPUs, host memory, SSD, KV tiers used

### Takeaway
- **Engines:** five papers build on vLLM (Continuum 0.10.2; TokenCake against a v0.8.6 baseline; AKR v0.27.1; CacheWise and Ask the Tool unversioned). Two build on SGLang (KVFlow v0.4.4; IAC v0.5.3). KVLearn names no engine.
- **Tiers:**
  - Only Continuum touches SSD: LMCache with 400 or 800 GB, in a single sensitivity figure.
  - Host DRAM tiers appear in Continuum, CacheWise (§6.4 only), Ask the Tool, TokenCake, KVFlow and AKR.
  - IAC is GPU-only.
  - KVLearn models a DRAM pool across nodes.

### Cited Findings
**Continuum**
- **Software.** "Vanilla vLLM ... vllm 0.10.2 with default setting, where chunk size is enabled with size 2048". "vllm 0.10.2 with LMCache 0.3.7 ... For A100 GPUs, we set the DRAM size used in offloading to be 100GB; For B200 and H100 GPUs, ... 200GB per GPU" (§6.1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Models and GPUs.** Llama-3.1-8B, Llama-3.1-70B and Gemma-3-12B on "A100-SXM GPU from Runpod, H100 from AWS and Company A, and B200 GPU from on-prem servers" (§6.1). Fig. 8 panels: Llama 70B (4×B200), Llama 8B (1×B200), Llama 8B (1×A100), Gemma 12B (1×A100) — [v7](https://arxiv.org/html/2511.02230v7)
- **SSD.** "We evaluate Continnum with extended SSD storage layer beyond CPU offloading using LMCache on SWE-bench workload with llama-8B on B200" (§6.3, Fig. 15 with SSD 400 GB and 800 GB) — [v7](https://arxiv.org/html/2511.02230v7)
- **Distributed comparisons.** "SGLang 0.5.5.post3 ... with native cache-aware routing and Nvidia Dynamo 0.7.0.post1 ... configured with 1P1D" (§6.1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Implementation size.** "about 1k lines of Python" on vLLM (§5.3) — [v7](https://arxiv.org/html/2511.02230v7)

**CacheWise**
- **Hardware and model.** "a server with 2×H200 GPUs and AMD EPYC 9534 64-core CPU with Tensor Parallelism (TP=2) ..., with a combined GPU memory of 282GB", chunked prefill "max. 512 tokens per chunk", Qwen2.5-Coder-32B-Instruct (§6.1). The prototype is "~2,500 lines of Python code" in vLLM (§5.3) — [v1](https://arxiv.org/html/2606.16824v1)
- **DRAM tier only in §6.4.** "evicted blocks are transferred from GPU memory to CPU DRAM and swapped back on demand" (§6.4); the DRAM size is not stated — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **Engine and tiers.** "The engine is vLLM ... with prefix caching on; the host tier is its own KV offloading to DRAM" (App. D), on "a 4×H100 SXM instance" (§5.1). The served model and DRAM size are not stated — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Software.** "We implement IAC on top of SGLang (v0.5.3)" (App. A.2) — [v1](https://arxiv.org/html/2611.09715v1)
- **Hardware and models.** "Intel Xeon Platinum 8380 CPU, 40 CPU cores, and four NVIDIA A100 (80 GB) GPUs, connected through PCIe 4.0 ×16 links ... five open-weight LLMs ranging from 8B to 110B parameters ... served on single-, two-, and four-GPU configurations" (§5) — [v1](https://arxiv.org/html/2611.09715v1)
- **Named models.** Llama-3-70B and Qwen-1.5-110B are named in §5.1–5.3. Fig. 7(a)'s axis labels in the PDF include "Llama3.1-8B" — [v1](https://arxiv.org/html/2611.09715v1), [PDF](https://arxiv.org/pdf/2611.09715v1)
- **Capacity is GPU KV in tokens.** For example "140K token capacity (43 GB for Llama-3-70B)" and "24.4 GB of KV cache (80K tokens)" (§5.1) — [v1](https://arxiv.org/html/2611.09715v1)

**TokenCake**
- **Hardware and models.** "Qwen2.5-14B on one NVIDIA A100 (80 GB HBM), Qwen2.5-32B on one NVIDIA H20 (96 GB HBM), and Qwen2.5-72B on two NVIDIA H20 GPUs with tensor parallelism degree 2. All configurations reserve 100 GB of CPU memory as the offload destination" (§7.1) — [v5](https://arxiv.org/html/2510.18586v5)
- **Baseline versions.** "vLLM (v0.8.6)", "Mooncake (v0.3.0-beta)", Parrot (§7.1) — [v5](https://arxiv.org/html/2510.18586v5)
- **Memory settings.** `gpu_mem_utilization=0.5` for the Mooncake comparison; "all other end-to-end experiments in §7.2 use the default 0.8" (§7.4) — [v5](https://arxiv.org/html/2510.18586v5)
- **Mooncake comparison model.** Qwen2.5-7B (Fig. 12) — [v5](https://arxiv.org/html/2510.18586v5)
- **Implementation.** "approximately 9k lines of Python code and reuses some components from vLLM" (§6) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Software.** "based on SGLang v0.4.4" (§3.3) — [v1](https://arxiv.org/html/2507.07400v1)
- **Hardware.** "(1) Llama-3.1-8B on an NVIDIA A10G GPU with 24GB memory and 2GB/s PCIe Gen1 bandwidth; and (2) Qwen2.5-32B on an NVIDIA H100 GPU with 80GB memory and 64 GB/s PCIe Gen5 bandwidth" (§4.1). Host memory size is not stated — [v1](https://arxiv.org/html/2507.07400v1)
- **CPU tier.** The CPU tier is SGLang's HiCache (baseline) or KVFlow's prefetch (§4.1) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Software.** "vLLM v0.27.1 with –tensor-parallel-size 4, –max-model-len 32768, and –gpu-memory-utilization 0.92. The host tier uses OffloadingConnector with kv_role=kv_both, offload_prompt_only=false, and cpu_bytes_to_use = 380×10^9 bytes (375×10^9 for the exponential workload)" (App. F.1) — [v1](https://arxiv.org/html/2608.30830v1)
- **Hardware.** "4× NVIDIA H100 NVL GPUs (94 GB HBM each), 128 vCPUs, and 1.5 TiB host DRAM. The available shared-memory budget limits the host tier to 375–380 GB" (App. G) — [v1](https://arxiv.org/html/2608.30830v1)
- **Model and pool.** Llama-3.1-70B BF16, TP4, with a "measured 680,768-token KV pool" (§4.1; App. A) — [v1](https://arxiv.org/html/2608.30830v1)
- **Calibration only on other GPUs.** Cross-platform calibration and replay cover H100 NVL, A100 SXM and L40S (App. C) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Clusters.** "(a) Small: 4 A100-80GB P-nodes + 4 A100-80GB D-nodes + 256 GB DRAM pool, InfiniBand HDR (200 Gbps); (b) Large: 8 A100-80GB P-nodes + 8 A100-80GB D-nodes + 512 GB DRAM pool, InfiniBand HDR 2×400 Gbps (bonded)" (§5.1) — [PDF](https://arxiv.org/pdf/2610.28637v1)
- **Models.** "LLaMA-3-8B and LLaMA-3-70B ...; LLaVA-1.5-13B and Qwen-VL-Chat (7B)" (§5.1) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **No paper evaluates SSD reads on a forecast, or prefetch from SSD.** Continuum's SSD figure only extends LMCache capacity under its GPU pinning policy.
- **Host DRAM is set by configuration, not by the machine.** AKR caps the host tier at 380 GB on a 1.5 TiB node, and TokenCake uses 100 GB. Host scarcity in these papers is therefore an experimental choice.

### Gaps
- **Not stated:** the vLLM version for CacheWise and Ask the Tool; CacheWise's DRAM size; Ask the Tool's model and DRAM size; KVFlow's host memory; three of IAC's five models.

## 8. Metrics

### Takeaway
The session-level metrics differ:
- Continuum: job completion time.
- CacheWise: session completion time without tool time.
- TokenCake: application latency.
- Ask the Tool: post-tool TTFT.
- AKR: active-request goodput plus resume TTFT and percentage of the prefix cache retained.

IAC and KVLearn report hit rate and TTFT. Few report hit rate or recomputed tokens together with session time; CacheWise (evictions, transfer volume) and AKR (retained and recomputed %) come closest.

### Cited Findings
**Continuum**
- **Metrics.** Average delay (job completion time) and throughput (jobs per second), P90 and P95 (Fig. 11), pass rate (Fig. 12), scheduler overhead in ms (Table 4) and RL steps per minute (Table 5) (§6.2–6.4) — [v7](https://arxiv.org/html/2511.02230v7)
- **Headline numbers.** "reduces delay by 1.12x to 3.66x and improves throughput by 1.10x to 3.22x" (§1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Released analysis.** It computes job duration as "last Request_departure_time − first Request_arrival_time" from server-side scheduler timestamps — [analyze.py](https://github.com/Hanchenli/vllm-continuum/blob/main/continuum_exp/analyze.py)

**CacheWise**
- **Metrics.** "(1) token goodput ..., request throughput (2) request completion time, (3) session completion time (the sum of LLM request latencies across a single session without the tool execution durations ...), and (4) KVCache transfer volume" (§6.1). The paper also counts evictions (Fig. 14) and scheduling overhead (Fig. 20) — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **KV metrics.** "post-tool TTFT p90 and ... throughput against LRU", plus "the volume each run reloads from DRAM" (§5.2, Fig. 9), and the gain by call length (Fig. 10) — [v1](https://arxiv.org/html/2609.18849v1)
- **Accuracy metrics.** Median absolute remaining-time error at 50% and 90% of a call (Fig. 7), and the share of timely first triggers under 2 s and 0.5 s recovery budgets (Fig. 8) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Metrics.** Cache hit rate (Fig. 6), mean TTFT (Fig. 7a), average latency against request rate (Fig. 7b), p50 and p90 TTFT (App. B.8) and per-request hit ratio (App. B.9) — [v1](https://arxiv.org/html/2611.09715v1)
- **Hit rate in the released code.** It is `sum(cached_tokens)/sum(prompt_tokens)` — [scripts/replay.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/scripts/replay.py)

**TokenCake**
- **Metrics.** "end-to-end latency (average and tail percentiles) from application submission to final response, and GPU KV Cache utilization" (§7.1). Throughput in req/s and offload counts are also reported (§7.3–7.5) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Metric.** End-to-end latency speedup over SGLang (Figs. 5, 6, 8) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Live metrics.** "(a) KV retained at resume, (b) active-request goodput, and (c) resume TTFT (p50; p95 whiskers)" (Fig. 4). Table 5 adds recomputed %. Example: ours, lognormal: 0.95 req/s goodput, 38.1% retained, 61.4% recomputed, 435 ms resume TTFT (App. F.2) — [v1](https://arxiv.org/html/2608.30830v1)
- **Replay metric.** "GPU opportunity cost per request (GPU-s)" (Table 6) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Metrics.** "TTFT (ms), end-to-end throughput (tokens/sec), hit rate (fraction of requests served from pool), and inter-node KV transfer volume (GiB/hour)" (§5), plus predictor AUC over 5 runs (Fig. 3) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **CacheWise's session metric is not wall time.** It sums LLM request latencies and leaves out tool time. A policy that lets the prefix cache be reloaded during a tool call gets no credit or blame for overlap with tool time.
- **Few papers decompose the gain.** Only AKR and CacheWise report both a retention or eviction quantity and a session-level outcome. A KV prefetch paper should report hit rate, recomputed tokens and bytes read per tier alongside session time.

### Gaps
- IAC's paper does not define hit rate at the token or the request level; only its released code shows token level.

## 9. Baselines and ablations

### Takeaway
Most papers compare against stock vLLM or SGLang plus one or two reimplemented research policies. Reimplementations dominate: Continuum reimplements Autellix and InferCept; CacheWise reimplements InferCept; Ask the Tool reimplements four predictors; AKR ports MORI. Continuum has been used as a baseline in two different forms:
- Ask the Tool reimplemented the paper's per-tool time-to-live (TTL).
- AKR used the public fork's fixed 2 s TTL.

Only Continuum (SGLang, Dynamo), TokenCake (Mooncake, Parrot) and KVLearn (a Mooncake-style policy) compare against other full systems.

### Cited Findings
**Continuum**
- **Baselines.** Vanilla vLLM; vLLM + LMCache; Autellix (PLAS reimplemented on vLLM, and "Autellix+" with LMCache); InferCept ("We implemented the selectively preserve, swap, or evict algorithm ... Since the CPU offloading in LMCache is non-blocking ..., we update the cost estimation accordingly"); SGLang 0.5.5.post3 and Dynamo 0.7.0.post1 for the live run (§6.1) — [v7](https://arxiv.org/html/2511.02230v7)
- **Ablation and sensitivity.** Program-level FCFS and Static TTL (Fig. 16); batch size and chunk size 256–4096 (Fig. 13); turns (Fig. 14); SSD (Fig. 15) (§6.3–6.4) — [v7](https://arxiv.org/html/2511.02230v7)
- **v1 ablation.** It compared "Program Level FCFS", "Simplified Version" and Ours (Fig. 12) — [v1](https://arxiv.org/html/2511.02230v1)

**CacheWise**
- **Baselines.** "(a) vLLM ... with block-level LRU KVCache eviction and FCFS admission, (b) InferCept ... with FCFS scheduling and moving-average-based eviction"; "we did not directly evaluate against SGLang" (§6.1) — [v1](https://arxiv.org/html/2606.16824v1)
- **Ablations.** CacheWise* (ground-truth tool latencies), vLLM(Prefix-aware) and InferCept(Prefix-aware) (§6.1, Figs. 15–16). Predictor variants: Point Estimate, Tool Name Only, C20, C50, C100 and Hint (§6.5, Fig. 19) — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **Predictors.** "four published predictors, each reimplemented as its paper defines it: Continuum's ... per-tool time-to-live ..., CacheWise's ... clusters ..., TokenCake's ... running average ..., and InferCept's ... two rules", plus an in-flight history estimator in the manner of 3Sigma (§4.1) — [v1](https://arxiv.org/html/2609.18849v1)
- **End-to-end baselines.** "the engine's own LRU, and an oracle that knows every exact return time" (§5.1) — [v1](https://arxiv.org/html/2609.18849v1)
- **Courtesy setting.** "As a courtesy, we fix the sandbox configuration for the predictors, where their history matches the current machine" (§4.2) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Baselines.** LRU, LFU, GDSF, LPC, and "TB+S, an offline oracle that knows the next arrival time of every cached entry" (§5). WA appears in App. B.5 (its own trace and an adapted variant) — [v1](https://arxiv.org/html/2611.09715v1)
- **Ablations.** Median versus mean versus mode of the delay model (Fig. 9, §5.4); Full-Horizon (App. B.4); rolling-window online statistics (App. B.3, "+49% to +83%" over LRU per fold) — [v1](https://arxiv.org/html/2611.09715v1)

**TokenCake**
- **Baselines.** vLLM, vLLM-Prefix, Mooncake and Parrot. Parrot "uses a different runtime ..., so it serves as a system-scope check rather than a controlled experiment" (§7.1, §7.4) — [v5](https://arxiv.org/html/2510.18586v5)
- **Components and sensitivity.** Baseline, agent-only, offload-only and full (§7.3); tool-time noise, request selection and watermark (§7.5) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Baselines.** SGLang (GPU-only radix cache) and "SGLang w/ HiCache" (§4.1). I found no component ablation separating eviction from prefetch in §4 — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Baselines.** "unmodified vLLM ..., vLLM with host offload enabled, MORI ..., and Continuum" (§4.2). Footnote: "MORI has no released implementation, so we evaluate its published policy through a port on vLLM. For Continuum, we use the first author's public fork ..., whose README states that it omits the paper's pause-duration estimator" (§4.2) — [v1](https://arxiv.org/html/2608.30830v1)
- **Continuum setting.** "so we use its shipped 2 s pin TTL" (App. F.1) — [v1](https://arxiv.org/html/2608.30830v1)
- **Replay baselines.** Fixed host timers of 10, 30 and 60 minutes and ∞ (App. F.3) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Baselines.** No-Cache, LRU-Pool, LFU-Pool, Mooncake-LRU and an Oracle that "admits a block iff it will actually be reused within horizon H=1000 requests" (§5) — [PDF](https://arxiv.org/pdf/2610.28637v1)
- **Ablations and sweeps.** No ATC, constant P̂=0.5 and no modality features (Table 2); pool size, cluster size, β, Zipf α and H (Fig. 5, Tables 3–4) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **Name the Continuum variant.** The paper's estimator (as reimplemented by Ask the Tool) and the fork's 2 s TTL (used by AKR) behave differently. A new paper using Continuum as a baseline should say which it ran.
- **Oracles bound different things.** KVLearn's H = 1000-request lookahead, IAC's TB+S and Ask the Tool's exact-return oracle each depend on assumptions that should be stated.

### Gaps
- Exact settings of the reimplemented baselines (thresholds, history windows) are mostly not given. Ask the Tool's keep threshold, lead time and hysteresis band have no stated numbers (App. D).

## 10. Released code, traces or scripts: do they match the evaluated method?

### Takeaway
None of the releases reproduces the paper's evaluated system as described:
- **Continuum:** the fork omits the paper's estimator and ships a live mini-swe-agent driver, but no trace replayer or traces.
- **CacheWise:** traces and predictor only, no vLLM integration.
- **IAC:** targets SGLang v0.5.19 (paper: v0.5.3), with a 1-token, time-compressed replayer.
- **TokenCake:** a vLLM v0.22.0 migration without predictive upload, using different workloads.
- **KVLearn:** a simulator.
- **Ask the Tool, KVFlow and AKR:** link no code.

### Cited Findings
**Continuum**
- **Paper.** "The implementation and artifacts are available at: https://github.com/Hanchenli/vllm-continuum" (§9) — [v7](https://arxiv.org/html/2511.02230v7)
- **README.** "a modified version of vLLM with Continuum-style scheduling support (without the estimation in the paper) ... For multi-node support, you should use sticky-session routing" — [README](https://github.com/Hanchenli/vllm-continuum/blob/main/README.md)
- **What the shipped TTL is.** AKR confirms "the public fork released by the first author (vLLM 0.10.2 base) ... implements Continuum scheduling without the paper's pause-duration estimator, so we use its shipped 2 s pin TTL" (App. F.1) — [AKR v1](https://arxiv.org/html/2608.30830v1)
- **Live driver is included.** The repo vendors mini-swe-agent with a Poisson "JPS mode" ([swebench.py](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/src/minisweagent/run/extra/swebench.py)). Its runner launches two TP4 Llama-3.1-70B vLLM servers behind `continuum_exp/router.py` ([run_experiments.py](https://github.com/Hanchenli/vllm-continuum/blob/main/mini-swe-agent/run_experiments.py)).
- **What I could not find.** That router file is not in the repository tree I listed (3,641 entries on 17 Nov 2026). Searching file names for replay, trace, swe and bfcl found no trace-replay driver for Figs. 8–11 and 13–16 — [repo](https://github.com/Hanchenli/vllm-continuum) (my listing)

**CacheWise**
- **Paper.** "CacheWise is open sourced at github.com/cachewise-project/cachewise-coding-traces" (Abstract) — [v1](https://arxiv.org/html/2606.16824v1)
- **Repo contents.** The repo holds `raw_traces/`, `parsed_traces/`, `workload_analysis/` and `tool_duration_prediction/` (train and infer); I found no vLLM code in the tree — [repo](https://github.com/cachewise-project/cachewise-coding-traces), [README](https://github.com/cachewise-project/cachewise-coding-traces/blob/HEAD/README.md)

**Ask the Tool**
- **No code of its own.** Its GitHub links point only to other projects' issues and docs (mini-swe-agent, SGLang, vLLM, Mooncake, Dynamo, TensorRT-LLM) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Install script.** The README says "`install.sh` clones SGLang v0.5.19, applies the IAC patch". The repo ships `patches/sglang-v0.5.19-iac.patch` — [README](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/README.md), [repo](https://github.com/hasanar1f/intent-aware-caching)
- **README example and claim.** The example serves `Qwen/Qwen3-8B` with `--capacity 100000` and runs `replay.py --time-scale 1.6`. The README's abstract says latency falls "by up to 54.6%", against "46.1%" in the arXiv abstract — [README](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/README.md), [v1](https://arxiv.org/html/2611.09715v1)

**TokenCake**
- **Paper.** "TokenCake is publicly available at https://github.com/pku-lemonade/TokenCake" (Abstract). The v5 HTML carries ACM badges for artifacts available, functional and results reproduced — [v5](https://arxiv.org/html/2510.18586v5)
- **README on predictive upload.** "The paper describes predictive uploading to hide transfer latency. The current implementation uses vLLM v0.22.0 and performs event-driven preservation with demand-triggered CPU-to-GPU recovery; predictive prefetching and further policy tuning remain ongoing work" — [README](https://github.com/pku-lemonade/TokenCake/blob/HEAD/README.md)
- **README on experiments.** "The paper experiments and the current migration use different runtimes and workloads." New measurements: 24 static DAGs, and "the first 20 SWE-bench Verified tasks with mini-swe-agent 2.4.6 ... the agents executed real repository edits and tests, so their subsequent requests and total generated work followed different trajectories" — [README](https://github.com/pku-lemonade/TokenCake/blob/HEAD/README.md)

**KVFlow**
- **No code link** in the arXiv v1 text (URL scan) — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **No code link** in the v1 text (URL scan; its GitHub links are references) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Paper.** "Our code implementation of KVLearn is available at https://github.com/FastLM/KVLearn" (Abstract) — [PDF](https://arxiv.org/pdf/2610.28637v1)
- **Repo.** A C++ library plus `apps/simulate.cpp` (Build: `./build/kvlearn_sim [n_requests] [zipf_alpha] [pool_GiB]`). The README says "this library only decides **keep vs discard**" — [README](https://github.com/FastLM/KVLearn/blob/HEAD/README.md)

### Inferences
- **Not reproducible as published.** None of the eight results can be reproduced from the release as it stands. The nearest is Continuum's live SWE-bench driver, which runs the fork's 2 s TTL rather than the paper's estimator.
- **TokenCake's badges.** They refer to the artifact evaluated for EuroSys. The current repository says it differs from the paper's runtime. Which commit was badged is not stated (inference).

### Gaps
- I did not check whether the CacheWise or IAC releases contain the exact held-out split used in the paper beyond `splits/train_sessions.json` (CacheWise).
- I did not check whether earlier commits of TokenCake or Continuum contain the paper-time code.

## 11. Threats to validity: stated by the authors and noticed

### Takeaway
The recurring weaknesses are:
- replaying traces from a different, stronger model on small open models with undisclosed output and tool-gap handling;
- load-independent tool or human delays;
- forced short outputs;
- closed or compressed timing;
- client-supplied session IDs that real deployments may not send;
- a single node and few models;
- releases that differ from the evaluated system.

The authors themselves state few of these. AKR, TokenCake and KVLearn have explicit limitation paragraphs; Ask the Tool has a Limitations section about signal coverage.

### Cited Findings
**Continuum**
- **Stated (v4 only).** v4 App. D: the model "assumes that tool-call distributions and workload characteristics are sufficiently stable", and in volatile workloads "the model may produce suboptimal TTLs". The v7 HTML has no appendix headings — [v4](https://arxiv.org/html/2511.02230v4), [v7](https://arxiv.org/html/2511.02230v7)
- **Noticed, the overhead table changed between versions.** "Ours" scheduler overhead is 1.63 / 2.74 ms (no offload / CPU offload) in v1 Table 1 but 0.96 / 2.30 ms in v7 Table 4. Baseline rows are unchanged (vLLM 0.95 / 2.33 ms) — [v1](https://arxiv.org/html/2511.02230v1), [v7](https://arxiv.org/html/2511.02230v7)
- **Noticed, the trace count changed.** "500 traces" in v1 §3.1 became 100 + 100 in v4 and v7 §3.1 — [v1](https://arxiv.org/html/2511.02230v1), [v7](https://arxiv.org/html/2511.02230v7)
- **Noticed, ThunderAgent's number may not be a same-run baseline.** The comparison is "on inference steps per minute, as reported by the original paper" (§6.4). This reads as if the ThunderAgent number was taken from its own paper (ambiguous) — [v7](https://arxiv.org/html/2511.02230v7)
- **Noticed, the live run conflates quality and delay.** Baseline pass rate drops because runs over "15 minutes" are "treated as a failure case" (§6.2) — [v7](https://arxiv.org/html/2511.02230v7)

**CacheWise**
- **Stated.** "trace distributions may drift over time ... we leave a thorough study of drift robustness to future work"; CacheWise "exhibits higher P99 request latencies" (§6.1–6.2) — [v1](https://arxiv.org/html/2606.16824v1)
- **Noticed.** One lab population, with the 80/20 train/test split drawn from it; one model on one 2×H200 server; human pauses removed by splitting sessions; tool time excluded from the session metric; replay mechanics unstated — [v1](https://arxiv.org/html/2606.16824v1)

**Ask the Tool**
- **Stated.** Some calls hold no information, some output cannot be parsed, and "Drawing the signal out may change the tool ... We measured the cases we used and found them within noise" (Limitations) — [v1](https://arxiv.org/html/2609.18849v1)
- **Noticed.** The KV setup (model, sessions, concurrency, DRAM size) is undisclosed; competitors are reimplemented; the liar defence is only in "a preliminary simulation" (§5.3) — [v1](https://arxiv.org/html/2609.18849v1)

**IAC**
- **Stated.** ShareGPT and LMSys-Chat gains are smaller "because all intent categories share the same synthetic delay distribution" (App. B.6) — [v1](https://arxiv.org/html/2611.09715v1)
- **Noticed.** A chat workload, not agents; GPU-only KV; the released code uses `max_tokens=1`, a 1.6× time scale and a different SGLang version — [v1](https://arxiv.org/html/2611.09715v1), [replay.py](https://github.com/hasanar1f/intent-aware-caching/blob/HEAD/scripts/replay.py)

**TokenCake**
- **Stated.** "TokenCake relies on developers to declare the agent DAG and tool types ... The evaluation covers a single node with up to two tensor-parallel GPUs; cross-node deployment and a controlled comparison with Parrot on a common engine remain future work" (§9). Static graphs are used "for reproducibility" (§9). Intermediate tool-time noise (0.25) makes TokenCake 8.3% slower than agent-only (§7.5) — [v5](https://arxiv.org/html/2510.18586v5)
- **Noticed.** Synthesized user requests; sampled tool durations; and the Mooncake comparison uses a different model (Qwen2.5-7B, five applications) — [v5](https://arxiv.org/html/2510.18586v5)

**KVFlow**
- **Stated.** "If concurrency is too high, all available memory is consumed by active requests, and the system can no longer maintain reusable prefix caches, placing it beyond the scope of our optimization" (§4.2) — [v1](https://arxiv.org/html/2507.07400v1)
- **Noticed.** Random-token prompts, no tools, fixed lengths, and an A10G with "2GB/s PCIe Gen1" — [v1](https://arxiv.org/html/2507.07400v1)

**AKR**
- **Stated.** "Our experiments use the configured offered load λ ... Approval outcomes are LLM-simulated and wait durations are synthetic, and we do not yet have a production trace ... MORI is evaluated through a policy port ... Our Continuum comparison uses the first author's public scheduling fork, which omits the paper's pause-duration estimator ... Our model-level measurements use a single model family" (§5 Limitations) — [v1](https://arxiv.org/html/2608.30830v1)

**KVLearn**
- **Stated.** "Our evaluation uses publicly available academic traces ... rather than proprietary production logs; ... integration with a live Mooncake or DistServe cluster and validation against closed production traces are reserved for future work" (Limitations) — [PDF](https://arxiv.org/pdf/2610.28637v1)
- **Noticed.** Inconsistent workload sizes between §5.1 and Table 1 (Section 4 above) — [PDF](https://arxiv.org/pdf/2610.28637v1)

### Inferences
- **Lessons for a GPU / DRAM / SSD prefix-cache prefetch and retention paper:**
  1. Run at least one experiment with live agents executing real tools (as Continuum's Fig. 12 does) and report pass rate.
  2. In the controlled replay, state exactly how tool gaps, output lengths and model mismatch are handled.
  3. Report whether tool durations depend on load.
  4. Report hit rate, recomputed tokens and bytes moved per tier along with session time.
  5. Name the exact variant of each reimplemented baseline.
- **This set leaves SSD-tier prefetch open.** No paper evaluates prefetching the prefix cache from SSD; only Continuum's Fig. 15 uses an SSD tier at all.

### Gaps
- Most papers do not report run-to-run variance for the main results. Exceptions: KVLearn's AUC over 5 runs, CacheWise's "several times", AKR's 15 live runs and Ask the Tool's bootstrap intervals for Table 4.

## 12. Comparison table across the eight papers

### Takeaway
The table condenses Sections 1–11. Every cell is sourced in those sections; the first column links the version read.

### Cited Findings
**Table A: workload and integration**

| Paper (version) | Mode | Tools | Outputs | Tasks / traces (origin) | Framework and engine interface | Load |
|---|---|---|---|---|---|---|
| [Continuum v7](https://arxiv.org/html/2511.02230v7) | (ii) replay for Figs. 8–11 and 13–16; (i) live for Fig. 12 and Table 5 | Replay: not stated (recorded gaps; Table 2). Live: real Docker tools, 15-min limit | Replay: GPT-5 traces on Llama/Gemma, handling not stated; BFCL ×0.4. Live: free generation, temp 0 (code) | 100 SWE-bench + 100 BFCL v4 web-search traces (authors, GPT-5); OpenHands multi-SWE-bench Go; 500 SWE-bench Verified tasks live | mini-swe-agent, BFCL agent, OpenHands. Client sends program_id (code: job_id, is_last_step) via OpenAI API. Engine parses tool calls | Poisson programs, x-axis in JPS (0.13 JPS in Figs. 13–14). Run length and warmup not stated |
| [CacheWise v1](https://arxiv.org/html/2606.16824v1) | (ii) "deterministic" replay | From trace (implied); human pauses become session boundaries | Not stated; Claude traces on Qwen2.5-Coder-32B | CATraces: Claude Code, lab participants; repo has 94 sessions; 80/20 split | Engine parses tool_name, tool_args and T_i; no client hints | Closed batch of N sessions (up to 50), run to completion, repeated "several times" |
| [Ask the Tool v1](https://arxiv.org/html/2609.18849v1) | §4: real re-execution and live runs; §5: (ii); §5.3: (iii) | §4: real, load varied; §5: not stated | §5 not stated; §4 runs free | Leaderboard mini-SWE-agent runs (6 submissions, ~3,000 trajectories), OpenHands runs, own 100 SWE-bench Verified tasks with Qwen 3.8 Flash | Harness side channel feeds a controller, which sends refresh/release/query (by response ID) to patched vLLM | Not stated |
| [IAC v1](https://arxiv.org/html/2611.09715v1) | (ii) chat replay; synthetic timing for 2 of 4 datasets | None (chat) | Paper not stated. Code: recorded history, max_tokens=1 | WildChat, ShareChat (real timestamps), ShareGPT, LMSys; 40K train / 10K replay; WA cloud trace | No agent framework; engine-side intent classifier. Code passes conv_id and timestamp | Open Poisson conversation starts, original gaps; 5 req/s (TTFT), 10–20 req/s. Code: 1.6× time scale with warmup |
| [TokenCake v5](https://arxiv.org/html/2510.18586v5) | Hybrid (i)/(ii): live apps, replayed tools | Recorded outputs; durations sampled with fixed seed; noise sweep | Not stated; static DAGs | Own Code-Writer and Deep-Research apps; ShareGPT and AgentCode requests | Custom DAG API on vLLM's OpenAI endpoint; call_start/call_finish with estimates; no output parsing | Poisson apps 0.05–1.0 QPS; 20 apps (Fig. 11), 5 (Fig. 12) |
| [KVFlow v1](https://arxiv.org/html/2507.07400v1) | (iv) synthetic on real SGLang | None | Fixed lengths, greedy decoding | Random-token 10-agent chains; PEER templates with Financial QA | SGLang frontend; per-request agent ID, steps-to-execution, client ID | Batch size 1 (10 runs after warmup); fixed concurrency (e.g., 64) |
| [AKR v1](https://arxiv.org/html/2608.30830v1) | Live (iv) on real vLLM plus CPU-only replay (iii) | Human waits, synthetic (mean 1,800 s); approvals LLM-simulated | Forced: 1 token at suspension, 8 at resume, 64 for background | 2,161 τ²-bench approval-gated prefixes; 1,160 background prompts | No agent framework; controller uses configured λ; BlockPool patch | Closed loop of 48 clients plus gates at 2λ_crit; 15 live runs in 14 h; replay of 4,000 Poisson requests |
| [KVLearn v1](https://arxiv.org/pdf/2610.28637v1) | (iii), inferred: no live engine; released simulator | None (not agentic) | Not stated | ShareGPT, LLaVA-Instruct-150K, ActivityNet-QA (counts inconsistent) | KV-pool control plane | Not stated; 2-hour window (Fig. 4) |

**Table B: stack, metrics, baselines, release**

| Paper | Engine and version | Models and GPUs | Host DRAM / SSD / tiers | Main metrics | Baselines | Release vs paper |
|---|---|---|---|---|---|---|
| Continuum | vLLM 0.10.2 + LMCache 0.3.7 | Llama-3.1-8B/70B, Gemma-3-12B, GLM-4.5-fp8; A100, H100, B200 | DRAM 100 or 200 GB per GPU; SSD 400/800 GB (Fig. 15 only); GPU / DRAM / SSD | Average and P90/P95 job completion time, JPS throughput, pass rate | vLLM, LMCache, Autellix(+), InferCept, SGLang 0.5.5.post3, Dynamo 0.7.0.post1, ThunderAgent (possibly numbers from its own paper) | Fork lacks estimator (2 s TTL); live driver only; no replayer or traces |
| CacheWise | vLLM (version not stated), ~2.5k LoC | Qwen2.5-Coder-32B; 2×H200 TP2 | GPU only; CPU DRAM swap in §6.4 (size not stated); no SSD | Session completion time without tool time, goodput, evictions, transfer volume, P50/P90/P99 | vLLM (LRU), InferCept; ablations with prefix-aware variants and CacheWise* | Traces and predictor only; no engine code |
| Ask the Tool | vLLM (version not stated) + few-hundred-line patch | Not stated for §5; 4×H100 SXM | GPU only and GPU + DRAM (size not stated); no SSD | Post-tool TTFT p90, throughput, DRAM reload volume; remaining-time error | LRU, oracle, reimplemented Continuum, CacheWise, TokenCake, InferCept | No code |
| IAC | SGLang v0.5.3 | 5 LLMs 8B–110B (Llama-3-70B and Qwen-1.5-110B named); 4×A100 80 GB | GPU only (capacity in tokens) | Hit rate, mean and tail TTFT, latency vs rate | LRU, LFU, GDSF, LPC, TB+S oracle; WA in appendix | Code targets SGLang v0.5.19; replayer uses max_tokens 1 and time scale 1.6 |
| TokenCake | ~9k-LoC system on vLLM components; baseline vLLM v0.8.6 | Qwen2.5-14B (A100), 32B (H20), 72B (2×H20), 7B (Fig. 12) | 100 GB CPU; GPU utilization 0.8 or 0.5; no SSD | Application end-to-end latency (mean and tail), GPU KV utilization | vLLM, vLLM-Prefix, Mooncake v0.3.0-beta, Parrot (own engine) | vLLM v0.22.0 migration without predictive upload; different workloads |
| KVFlow | SGLang v0.4.4 | Llama-3.1-8B (A10G), Qwen2.5-32B (H100) | CPU tier via HiCache and prefetch (size not stated); no SSD | Latency speedup | SGLang, SGLang + HiCache | No code linked |
| AKR | vLLM v0.27.1 + OffloadingConnector | Llama-3.1-70B, 4×H100 NVL | Host tier 375–380 GB of 1.5 TiB; no SSD | Goodput, % retained and recomputed, resume TTFT; GPU-s cost in replay | vLLM, vLLM + offload, MORI port, Continuum fork (2 s TTL) | No code linked |
| KVLearn | Not named; live integration is future work | LLaMA-3-8B/70B, LLaVA-1.5-13B, Qwen-VL-Chat 7B; A100 P/D clusters (as described) | DRAM pool 256 or 512 GB; no SSD | TTFT, throughput, hit rate, transfer volume | No-Cache, LRU-Pool, LFU-Pool, Mooncake-LRU, Oracle (H=1000) | Simulator plus decision library |

### Inferences
- **Two evaluation designs worth borrowing.** Continuum's pairing of controlled replays with one live-agent run plus pass rate, and AKR's pairing of live serving with a calibrated replay for sweeps, are the strongest models for an evaluation-methods section. The weak spot in both is the undisclosed or synthetic timing of tool and human gaps.

### Gaps
- The table inherits every "not stated" from Sections 1–11. The most consequential are Continuum's and Ask the Tool's replay mechanics, and AKR's live-run arrival process and wait compression.
