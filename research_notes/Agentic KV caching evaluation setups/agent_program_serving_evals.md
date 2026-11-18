# How papers on serving agent programs and intercepted (tool-calling) LLM requests evaluate their systems

**Scope.** This file covers eleven papers. Astraea is included because the earlier notes give its arXiv ID.
- **Read on:** Nov 17, 2026, from the arXiv full text (latest version).
- **Tables and figure labels:** the PDF text was used where the HTML garbled them.
- **Linked repositories:** read only to confirm how the workload driver works.
- **Mode classes:**
  - (i) live agents;
  - (ii) trace replay on a real engine;
  - (iii) simulation;
  - (iv) synthetic workload.
- **Order:** papers appear in the order of their first arXiv version in every section.
- **Our computations:** marked "our computation" (over a released file).
- **Inferences:** marked "(inference)".

**Papers and versions read:**
- **InferCept**
  - arXiv 2402.01869 **v2** (30 May 2024); ICML 2024 (PMLR vol. 235, per the earlier notes, not re-checked).
  - Sources: [HTML v2](https://arxiv.org/html/2402.01869v2) · [PMLR](https://proceedings.mlr.press/v235/abhyankar24a.html) · [code](https://github.com/WukLab/InferCept)
- **Parrot**
  - 2405.19888 **v1** (30 May 2024); arXiv comment: "To appear on USENIX OSDI 2024".
  - Sources: [abs](https://arxiv.org/abs/2405.19888) · [HTML v1](https://arxiv.org/html/2405.19888v1)
- **Conveyor**
  - 2406.00059 **v2** (4 Jun 2024); no venue found.
  - Sources: [HTML v2](https://arxiv.org/html/2406.00059v2) · [code](https://github.com/conveyor-sys/conveyor)
- **LAMPS / MARS**
  - 2410.18248 **v2** (25 Oct 2024).
  - The NeurIPS 2025 camera-ready, under the same title, calls the system MARS and extends the evaluation.
  - Sources: [HTML v2](https://arxiv.org/html/2410.18248v2) · [NeurIPS 2025 PDF](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf) · [code](https://github.com/mars-repository/mars-codebase)
- **SYMPHONY**
  - 2412.16434 **v1** (21 Dec 2024); no venue.
  - Sources: [HTML v1](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - 2502.13965 **v1** (19 Feb 2025); no venue.
  - Sources: [HTML v1](https://arxiv.org/html/2502.13965v1)
- **Hermes**
  - 2506.14851 **v1** (17 Jun 2025). The arXiv title is "Efficient Serving of LLM Applications with Probabilistic Demand Modeling".
  - Journal ref: ACM TACO 23(2), June 2026. The TACO version was not read.
  - Sources: [abs](https://arxiv.org/abs/2506.14851) · [HTML v1](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - 2512.14142 **v1** (16 Dec 2025); no venue.
  - Sources: [HTML v1](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - 2602.13692 **v3** (30 Jun 2026). The repository README says it was "accepted to ICML 2026 as a Spotlight".
  - Sources: [HTML v3](https://arxiv.org/html/2602.13692v3) · [repo](https://github.com/ThunderAgent-org/ThunderAgent)
- **PBKV** (the prediction-based KV-cache paper for agent workflows)
  - 2605.06472 **v1** (7 May 2026); no venue.
  - Sources: [HTML v1](https://arxiv.org/html/2605.06472v1)
- **MORI**
  - 2606.00866 **v2** (11 Nov 2026); no venue. The v2 HTML has no appendices.
  - Sources: [HTML v2](https://arxiv.org/html/2606.00866v2) · [repo](https://github.com/cblmemo/mori)

---

## 1. Evaluation mode: live agents, trace replay on a real engine, simulation, or synthetic workload

### Takeaway
- **Live runs.** Only three papers run agents or LLM applications live against a real engine with real tools:
  - ThunderAgent and PBKV (agent frameworks);
  - Hermes (ported applications).
- Conveyor runs live with real tools, but one request at a time and with no agent framework.
- **Replay.** Four papers replay recorded interception or session traces on a real engine, turning tool time into sleeps: InferCept, LAMPS/MARS, MORI, and Astraea (by inference).
- **Autellix** replays program traces but never says how tool time appears in them.
- **Synthetic.** SYMPHONY and most of Parrot use synthetic request streams.
- **Simulation** produces only one result (Autellix Fig. 18).
- The closest precedent for a KV prefix-cache tiering paper is **MORI**: trace replay on SGLang + HiCache, with recorded tool and human waits as sleeps and forced output lengths.

### Cited Findings
- **InferCept: mode (ii) with (iv) elements, one harness for all results.**
  - "we use a request dataset that merges the six augmentations presented in § 2 by uniformly sampling requests from them" — [InferCept HTML v2 §5](https://arxiv.org/html/2402.01869v2)
  - **Released driver.** It calls vLLM's in-process `LLMEngine` and builds prompts as `prompt_token_ids = [0] * prompt_size`. Every reproduction command passes `--load-format dummy` (random weights) — [final_tput_bench_real.py L266](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py); [exps/6B_bench.sh](https://github.com/WukLab/InferCept/blob/main/exps/6B_bench.sh)
  - **Trace collection used real tools.** "We use the live Wikipedia API endpoint for our benchmark". ALFWorld trajectories came from prompting GPT-4 in a ReAct loop. "we measure the actual Stable Diffusion call" — [InferCept HTML v2, Appendix "API Properties and Dataset Generation"](https://arxiv.org/html/2402.01869v2)
- **Parrot: mixed.**
  - **Application runs, no tools (§8.2, §8.4).** Chain and map-reduce summarization of long documents (Figs. 11–14) and a MetaGPT-style multi-agent programming workflow (Fig. 18) run end to end. They use Parrot's engine against LangChain → FastChat → vLLM/HuggingFace baselines.
  - **Synthetic streams (§8.3, §8.5, Fig. 10).** Fig. 10 says: "Requests are sampled from ShareGPT and their arrival time follows Poisson distributions" — [Parrot HTML v1 §8](https://arxiv.org/html/2405.19888v1)
  - Parrot "only supports cloud-side orchestration of LLM requests without involving dynamic control flow and native functions (e.g., Python Code)" — [Parrot HTML v1 §6](https://arxiv.org/html/2405.19888v1)
- **Conveyor: live single-request runs with real tools, plus an analytic model.**
  - Figs. 4, 6 and 7 are live runs: "We run each workload 100 times and collect the average latency and the standard deviations" — [Conveyor HTML v2 §4.2](https://arxiv.org/html/2406.00059v2)
  - Fig. 8 compares the §3.4 theoretical model with measurements — [§4.4](https://arxiv.org/html/2406.00059v2)
  - **Released driver.** It starts a fresh `python3 task.py <workload> [lazy]` process per measurement, which adds one request to the engine's request pool — [run_eval.py](https://github.com/conveyor-sys/conveyor/blob/main/run_eval.py); [task.py](https://github.com/conveyor-sys/conveyor/blob/main/task.py)
- **LAMPS/MARS: mode (ii) + (iv) on InferCept's harness.**
  - "We implement LAMPS on top of vLLM" — [LAMPS HTML v2 §5](https://arxiv.org/html/2410.18248v2)
  - The released driver is InferCept's: a vLLM 0.2.0 fork, dummy weights, sleeps and exponential inter-arrivals — [mars-codebase exps/6B_bench.sh](https://github.com/mars-repository/mars-codebase/blob/main/exps/6B_bench.sh); [fixed_final_tput_bench_real.py](https://github.com/mars-repository/mars-codebase/blob/main/benchmarks/fixed_final_tput_bench_real.py)
  - **Other results:**
    - prediction-error injection on the same harness — [§6.4, Fig. 11](https://arxiv.org/html/2410.18248v2);
    - offline predictor accuracy on a ToolBench test split — [§6.4, Table 3](https://arxiv.org/html/2410.18248v2).
- **SYMPHONY: mode (iv), synthetic chat sessions on a real vLLM-integrated prototype, plus one agent figure.**
  - "We generate the arrival time of each request within a session by calculating the time to read a response and type a query based on human reading and typing speed … Our trace generator maintains a fixed number of active users" — [SYMPHONY HTML v1 §4, Trace Generation](https://arxiv.org/html/2412.16434v1)
  - **Agent figure (Fig. 15).** "We serve MetaGPT requests using Symphony, utilizing LLaMa-3.1-8b across 8 GPUs" — [§4.4](https://arxiv.org/html/2412.16434v1)
- **Autellix: mode (ii) for Figs. 12–17 and 19; (iii) for Fig. 18; offline batch for Fig. 16.**
  - **Trace synthesis.** "we synthesize a trace by randomly sampling programs, not LLM calls, from the above workloads and generating programs' arrivals using a Poisson process"; "Our experiments replay entire conversations as a program rather than the first turn" — [Autellix HTML v1 §6.1](https://arxiv.org/html/2502.13965v1)
  - **Fig. 18.** "we emulate clairvoyance with a simulator … The simulation only considers scheduling, where each continuous-batching step is identical" — [§6.5.3](https://arxiv.org/html/2502.13965v1)
  - **Fig. 16.** "all programs are submitted at the start" — [§6.5.1](https://arxiv.org/html/2502.13965v1)
- **Hermes: mode (i), live applications on a real engine and real non-LLM backends, with arrivals from a production trace.**
  - "a comprehensive benchmark suite consisting of 11,300 lines of Python code, which includes all of the application types depicted in Fig. 1", on "vLLM 0.4.3" — [Hermes HTML v1 §4](https://arxiv.org/html/2506.14851v1)
  - "we sampled the request arrival time distribution from the trace published by MoonCake … We utilized its arrival time distribution while sampling real inputs from the application families shown in Fig. 1" — [§5.1](https://arxiv.org/html/2506.14851v1)
- **Astraea: mode (ii)-like by inference; the driver is not described.**
  - "We used the dataset released by Infercept for our experiments"; "We implemented the Astraea system on top of vLLM" — [Astraea HTML v1 §5.1](https://arxiv.org/html/2512.14142v1)
  - App. C says "a complete request in the workload averages 5.17 segments" — [App. C](https://arxiv.org/html/2512.14142v1). That equals the mean segment count of the released 397-template InferCept workload file (our computation).
- **ThunderAgent: mode (i) for every result, serving and RL rollout.**
  - "We deploy OpenHands and mini-SWEAgent on the SWEBench-Lite"; "We apply ToolOrchestra on HLE and OpenHands on ScienceAgentBench"; "We apply the same models, workflows, and samples for RL rollout on two 8×H100 nodes"; "agent Docker environments are offloaded to a dedicated CPU cluster" — [ThunderAgent HTML v3 §5.1](https://arxiv.org/html/2602.13692v3)
  - **Appendix runs are also live:** A.2 LMCache and PD disaggregation, A.3 eight nodes, A.4 SGLang HiCache, A.5 A100, App. G ablations — [App. A, G](https://arxiv.org/html/2602.13692v3)
- **PBKV: mode (i), live agent frameworks against a modified SGLang + HiCache.**
  - "three representative multi-agent workloads, each pairing a public benchmark with a widely-adopted agent framework to reflect real-world deployment" — [PBKV HTML v1 §6.1](https://arxiv.org/html/2605.06472v1)
  - **Runs are stochastic.** LRU "crashes in 3/10 trials at concurrency 84", and every setting is repeated 10 times — [§6.3; App. A](https://arxiv.org/html/2605.06472v1)
  - Offline invocation traces are used only to train the predictor (§4.1). Synthetic prediction noise is injected in App. E — [PBKV HTML v1](https://arxiv.org/html/2605.06472v1)
- **MORI: mode (ii), trace replay on a real engine, for all of §6 (Figs. 5–7).**
  - "We replay three coding-agent workloads of real developers' sessions, reproducing the recorded request lengths, tool-call durations, and waits"
  - Each client "sends a request, waits for the response, sleeps for the recorded tool-call or human-wait duration, and sends the next request" — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - The §3 motivation (Fig. 3) is offline trace analysis — [§3.2–3.3](https://arxiv.org/html/2606.00866v2)

### Inferences
- (inference) The evaluation mode follows what the system controls:
  - **Pause and resume inside one request** (InferCept, LAMPS/MARS, Astraea): replay with sleeps.
  - **Sandbox and tool lifecycle** (ThunderAgent, Hermes): live runs, because the backends themselves are managed.
  - **KV retention on SGLang HiCache:** replay (MORI) or live frameworks (PBKV).
- (inference) A GPU/DRAM/SSD prefix-cache retention and prefetch paper can cite MORI's template as accepted recent practice:
  - real-developer traces;
  - recorded waits replayed as sleeps;
  - forced output lengths;
  - closed loop at fixed concurrency;
  - SGLang + HiCache;
  - two-hour runs.
- (inference) A live cross-check in the style of ThunderAgent or PBKV would answer the two standard objections to replay: tool time that does not depend on load, and the absence of trajectory divergence.
- (inference) No paper here bases its headline result on simulation. A simulator-only evaluation would lack precedent in this group.

### Gaps
- **Autellix:** whether BFCL and LATS programs are re-executed or replayed from recorded call logs is not stated, and no code is public.
- **SYMPHONY:** whether MetaGPT runs live or is replayed is not stated.
- **Astraea:** the workload driver is not described, and no code is linked.

---

## 2. Tools and interceptions: executed for real, replayed as sleeps, or sampled? Does tool latency change with load?

### Takeaway
- **Real tool execution in measured runs** appears only in:
  - ThunderAgent: Docker sandboxes on a separate CPU cluster, plus remote search and model APIs;
  - Hermes: Docker and DNN backends, cold starts included;
  - Conveyor: Python, web search, SQLite and a calculator, one request at a time.
- **Recorded durations replayed as sleeps:** InferCept, LAMPS/MARS (per-tool constants for ToolBench) and MORI.
- **Synthetic human think time:** SYMPHONY and InferCept's chatbot.
- **Unspecified:** Autellix and PBKV do not say how tool time is handled.
- No main result samples tool time from a fitted parametric distribution.
- Tool latency depends on load only in the live setups. ThunderAgent measures this directly: environment preparation time grows with parallelism.

### Cited Findings
- **InferCept**
  - **Design.** The engine's API executor "calls the actual API" — [HTML v2, Appendix "Implementation Details"](https://arxiv.org/html/2402.01869v2)
  - **Released benchmark.** A thread runs `time.sleep(api_time)` and then returns zero tokens. Sampling parameters set `use_api_simulator=True` and take `api_exec_time` from the trace — [final_tput_bench_real.py L48, L117–122](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py)
  - **Table 1, interception time as (mean, variance) in seconds.** Math (9e-5, 6e-5); QA (0.69, 0.17); VE (0.09, 0.014); Chatbot (28.6, 15.6)\*; Image (20.03, 7.8)†; TTS (17.24, 7.6)†. The footnotes read "\* estimated behavior. † partially estimated" — [HTML v2 §2.2, Table 1](https://arxiv.org/html/2402.01869v2)
  - **Chatbot think time.** "We estimate human response time (i.e., the interception time) as the summation of human scanning time (Brysbaert, 2019) of the previous chat response and typing time (Ma et al., 2015) of the next prompt" — [§2.2](https://arxiv.org/html/2402.01869v2)
  - **Released workload (our computation).** The file is `merged_exp_uniform.json`, the filename the InferCept scripts pass; we read the copy in the MARS repo. It has 397 request templates and 1,841 API calls. Recorded `api_time` has mean 12.58 s, median 7.43 s and maximum 60.0 s — [mars-codebase merged_exp_uniform.json](https://github.com/mars-repository/mars-codebase/blob/main/merged_exp_uniform.json)
  - **Unused generators.** The driver also contains |N(11,3)|, U(0.1,20), Gamma(shape 0.5, scale 4) and 20−Gamma generators; the call to them is commented out — [final_tput_bench_real.py L147–167, L187](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py)
  - **Load dependence:** none, since each call sleeps a fixed recorded time.
- **Parrot**
  - No tools are executed.
  - "we introduced a random delay of 200∼300 ms to LLM requests to emulate typical network overhead seen over the Internet" — [Parrot HTML v1 §8.1](https://arxiv.org/html/2405.19888v1)
- **Conveyor**
  - **Real execution.** The engine "spawns a new process with an isolated setup to run the tool instance" — [Conveyor HTML v2 §3.3](https://arxiv.org/html/2406.00059v2)
  - **Tools** — [§4.1](https://arxiv.org/html/2406.00059v2):
    - CodeGen: a Python interpreter.
    - Search: online search, keeping StackOverflow results.
    - Planning: search, then calculator, then formatter.
    - Database: "a small SQLite file on disk".
    - Calculator: 200 × 701.
    - Validation\* is timed for detection only: "we only measure the latency needed for detecting if this invocation is problematic, not including tool execution time".
  - Search and Planning "depend on Internet performance" — [§4.4](https://arxiv.org/html/2406.00059v2)
  - **Repository plugins:**
    - The search plugin scrapes `https://www.google.com/search` — [search_plugin.py](https://github.com/conveyor-sys/conveyor/blob/main/conveyor/plugin/search_plugin.py)
    - The planning plugin searches for the dummy queries "hi" and "hello" and uses hard-coded values — [planning_plugin.py](https://github.com/conveyor-sys/conveyor/blob/main/conveyor/plugin/planning_plugin.py)
    - The SQLite plugin always runs `SELECT * FROM employees ORDER BY name` — [sqlite_plugin.py](https://github.com/conveyor-sys/conveyor/blob/main/conveyor/plugin/sqlite_plugin.py)
- **LAMPS/MARS**
  - **InferCept data:** the recorded `api_time` is slept (code above).
  - **ToolBench data:** each tool gets a constant delay from `api_response_data.csv` ("Delay", in ms), looked up by tool name — [toolbench_workload_gen.py L32–47, L109–117](https://github.com/mars-repository/mars-codebase/blob/main/prediction/toolbench_workload_gen.py)
  - **Released ToolBench workload (our computation):** 1,580 conversations and 2,785 calls. Durations have mean 3.03 s, median 0.92 s and maximum 109.5 s. Each entry carries a per-tool `tool_time_dict` — [mars-codebase](https://github.com/mars-repository/mars-codebase)
  - **Paper Table 2:** ToolBench API duration (1.72, 3.33) and calls per request (2.45, 1.81), as (mean, variance). This does not match the released file — [LAMPS HTML v2 §3.2, Table 2](https://arxiv.org/html/2410.18248v2)
  - **Stated premise:** "execution times within the same API type have low variance" — [LAMPS HTML v2 §3.2](https://arxiv.org/html/2410.18248v2)
  - **Load dependence:** none.
- **SYMPHONY**
  - No tools. Gaps are synthetic reading-plus-typing times; the speed parameters are not stated — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
  - **Advisory lead times:**
    - Chat: "advisory requests can be sent on average 11.3 seconds earlier than the actual LLM inference request".
    - MetaGPT on 4 A100s: "advisory requests can be issued on average 5.8 seconds earlier", using lower bounds from offline profiling.
    - — [§1](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - Interrupts are part of the program model: "external interrupts, which include tool calls (i.e. external API calls), generic code execution, or human inputs" — [Autellix HTML v1 §1](https://arxiv.org/html/2502.13965v1)
  - The analysis then sets them aside: "(3) interceptions, time spent waiting for external interrupts such as tool calls or human input. Since component (3) is unrelated to LLM serving, …" — [§3](https://arxiv.org/html/2502.13965v1)
  - How interrupt time is represented in the replay is **not stated**.
- **Hermes**
  - Non-LLM work runs on real backends: "the LLM-generated codes need to be tested on CPU backends (e.g., with a docker container…)" — [Hermes HTML v1 §2.1](https://arxiv.org/html/2506.14851v1)
  - The Docker image is "python:3.10-slim" — [§2.2, Fig. 2](https://arxiv.org/html/2506.14851v1)
  - "CG and PE … respectively demands Docker and DNN backends … PE applications calling a ViT or Diffusion model" — [§5.4](https://arxiv.org/html/2506.14851v1)
  - The HermesLet "employs two dedicated background threads to prefetch KV cache blocks or LoRA adapters as well as to prewarm docker containers" — [§4](https://arxiv.org/html/2506.14851v1)
  - Cold-start costs are measured — [§2.2, Fig. 2](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - Tool execution: **not stated**.
  - The predictor uses "the category's mean latency"; its examples are InferCept's Table 1 values ("9e-5 seconds", "20.03s and 28.6s") — [Astraea HTML v1 §4.2](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - **Sandboxes.** Real Docker sandboxes: OpenHands has "an average disk footprint exceeding 10GB per sandbox", mini-SWE-agent "2GB per sandbox" — [ThunderAgent HTML v3 §5.1](https://arxiv.org/html/2602.13692v3)
  - **ToolOrchestra tools.** These include a remote search service and model-as-a-tool calls — [App. C, Table 6](https://arxiv.org/html/2602.13692v3)
  - **Measured durations.** "the tail can extend to tens or even hundreds of seconds" — [App. C, Fig. 12](https://arxiv.org/html/2602.13692v3)
  - **Load dependence:**
    - Environment preparation time "increases with the parallel workload number" — [§3.3, Fig. 2(c)](https://arxiv.org/html/2602.13692v3)
    - Remote-tool volatility comes from "network jitter, backend load and queuing delays, and rate limiting" — [App. C](https://arxiv.org/html/2602.13692v3)
  - **Repository vs paper.** The repository's ToolOrchestra example sends expert tools to external APIs: OpenAI `gpt-5-mini` and `gpt-5`, and Together `openai/gpt-oss-120b` — [repo ToolOrchestra README](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/ToolOrchestra/README.md). The paper's Table 6 lists HLE-answer as "Local LLM inference".
  - **OSWorld ablation:** "a fixed 5 s interval between screenshots" — [App. G.2](https://arxiv.org/html/2602.13692v3)
  - **mini-SWE-agent config:** command `timeout: 60` and `step_limit: 135` — [repo swebench.yaml](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml)
- **PBKV**
  - Tool execution, timing and delays are **not described**.
  - The paper only motivates idle periods "during tool calls or transitions between agents" — [PBKV HTML v1 §1](https://arxiv.org/html/2605.06472v1)
- **MORI**
  - Recorded tool-call and human-wait durations are replayed as sleeps — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - **AgentX:** "Of the main agents' steps, 11% end in a gap of at least 60 s (median 3 min), which covers user input, permission prompts, and slow tools".
  - **TraceLab:** "about 8% of steps end in a human wait (median one to two minutes)".
  - **Subagents:** the 965 subagents are each replayed "as a program of its own, started and awaited as recorded" — [§6.1](https://arxiv.org/html/2606.00866v2)
  - **Tails:** gaps "reach a p99 of 519–600 s" — [§3.3, Fig. 3](https://arxiv.org/html/2606.00866v2)
  - **Load dependence:** none.

### Inferences
- (inference) The tool-gap distribution is the input that drives keep, offload and prefetch decisions. Only two sources describe realistic agent gaps:
  - MORI's traces: p99 of 519–600 s; 8–11% of steps ending in long human or tool waits.
  - ThunderAgent's measured heavy tails.
- (inference) InferCept's Table 1, which LAMPS and Astraea reuse, mixes near-zero tool times (Math 9e-5 s, VE 0.09 s) with estimated human think time (Chatbot 28.6 s). Its "tool" gaps are mostly not agent tool calls.
- (inference) Sleep replay makes tool latency independent of load. ThunderAgent's Fig. 2(c) shows environment preparation slowing as parallelism grows. A replay-based tiering paper should state this, or add a load-dependent component.
- (inference) For prefetch timing, replayed recorded durations give a duration predictor a stationary workload. Live remote tools add jitter that is not stationary over time (ThunderAgent App. C).

### Gaps
- **SYMPHONY:** reading and typing speeds are not given.
- **Hermes:** whether its web tools (FacTool knowledge-QA search, ReAct FEVER Wikipedia) call live APIs is not stated.
- **PBKV:** tool handling is not described at all.
- **Autellix:** interrupt time in the replay is not described.
- **LAMPS:** the source of its ToolBench delay CSV is not stated.

---

## 3. Model outputs: free generation or forced? How is divergence handled?

### Takeaway
- **Replay papers force outputs:**
  - InferCept and LAMPS/MARS (released drivers): `ignore_eos=True`, temperature 0, recorded `max_tokens`, dummy weights, zero-token prompts and zero-token tool returns.
  - MORI: "exactly its recorded number of output tokens", including the traced frontier models' reasoning tokens.
  - Parrot: output lengths matched to recorded GPT-4 responses.
- **Live papers generate freely:**
  - ThunderAgent (repository: temperature 0, `ignore_eos` False);
  - PBKV (stochastic decoding, 10 runs);
  - Hermes (free generation; its oracle comes from a temperature-0 trial run);
  - Conveyor (temperature 0, deterministic).
- **Divergence.** PBKV is the only paper that treats divergence as a design problem and repeats runs. No paper measures how divergence changes the KV workload compared with the trace.

### Cited Findings
- **InferCept**
  - The driver sets `temperature=0.0` and `ignore_eos=True`, with `max_tokens=min(2048, recorded total output)` — [final_tput_bench_real.py L110–113](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py)
  - The engine pauses when `generation_counter == api_invoke_interval`, i.e. on the recorded pre-call output length — [llm_engine.py L568](https://github.com/WukLab/InferCept/blob/main/vllm/engine/llm_engine.py)
  - **Truncation:** traces were cut "to fit within the context length" (VE) and "to fit within the maximum model sequence length" (QA) — [HTML v2 Appendix](https://arxiv.org/html/2402.01869v2)
  - **Divergence:** impossible by construction, and not discussed.
- **Parrot**
  - "we documented the LLM responses using GPT-4, ensuring the LLaMA models generated text of similar length for system performance analysis" — [Parrot HTML v1 §8.1](https://arxiv.org/html/2405.19888v1)
  - The forcing mechanism is **not stated**.
  - The DAGs are static, so divergence cannot change the workflow (inference from §6).
- **Conveyor**
  - "We set temperature to be 0, so every test for the same workload has the same LLM output" — [Conveyor HTML v2 §4.1](https://arxiv.org/html/2406.00059v2)
  - The released driver caps decoding at 500 iterations — [task.py](https://github.com/conveyor-sys/conveyor/blob/main/task.py)
  - The repository's Planning prompt dictates the plan text: "Please repeat the following text … $1 = search(Microsoft Market Cap) …" — [task.py](https://github.com/conveyor-sys/conveyor/blob/main/task.py)
- **LAMPS/MARS**
  - Forced: `ignore_eos=True`, temperature 0, recorded `max_tokens`, dummy weights — [fixed_final_tput_bench_real.py L103–106](https://github.com/mars-repository/mars-codebase/blob/main/benchmarks/fixed_final_tput_bench_real.py)
  - **What the scheduler sees:**
    - **InferCept data:** ground truth, since the "dataset already includes detailed output length information, making prediction unnecessary" — [LAMPS HTML v2 §5](https://arxiv.org/html/2410.18248v2). Otherwise, truth plus Gaussian noise proportional to the true value: `np.random.normal(0, p × value)` — [driver L118–119](https://github.com/mars-repository/mars-codebase/blob/main/benchmarks/fixed_final_tput_bench_real.py)
    - **ToolBench:** the pre-API length comes from OPT-125M, while the predicted API time equals the replayed time (`predicted_api_exec_time=api_exec_time, # api length. no error`) — [toolbench driver L125](https://github.com/mars-repository/mars-codebase/blob/main/benchmarks/fixed_final_tput_bench_real_toolbench.py)
- **SYMPHONY**
  - Main runs: whether outputs are forced to ShareGPT lengths, and whether later prompts carry recorded or generated responses, is **not stated** — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
  - The prefill-heavy ablation forces shapes: "we replace all requests with a 1024-token prompt and a 1-token response while keeping the original arrival times" — [§4.5, Fig. 16](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - Output-length control and divergence handling: **not stated** — [Autellix HTML v1 §6](https://arxiv.org/html/2502.13965v1)
- **Hermes**
  - Generation is free. Demand is modelled as a distribution: "we execute five LLM applications … for 100 times each, replaying different inputs sampled from the official dataset"; "we profile each application for 1000 times" — [Hermes HTML v1 §2.3, §4](https://arxiv.org/html/2506.14851v1)
  - The oracle ablation fixes outputs: "exact demands are available—by conducting a trial run with temperature set to 0" — [§5.3](https://arxiv.org/html/2506.14851v1)
  - The temperature of normal runs is **not stated**.
- **Astraea**
  - Output forcing: **not stated**.
  - The scheduler gets "a segment-level oracle for the number of generated tokens ($n_{gen}$), using the ground-truth value from our dataset" — [Astraea HTML v1 §4.2](https://arxiv.org/html/2512.14142v1)
  - (inference) A ground-truth length only helps if the replay forces lengths.
- **ThunderAgent**
  - The paper gives no temperature or length control.
  - **Repository:** `temperature: 0.0` — [swebench.yaml](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/mini-swe-agent/src/minisweagent/config/extra/swebench.yaml). The client sends `"ignore_eos": False` — [vllm_model.py](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/mini-swe-agent/src/minisweagent/models/vllm_model.py)
  - Divergence is not discussed. The metric is "steps per minute … where one step includes a reasoning and acting period" — [HTML v3 §5.1](https://arxiv.org/html/2602.13692v3)
- **PBKV**
  - "the stochastic decoding of LLMs makes the workflows diverge within a few tokens" — [PBKV HTML v1 §2](https://arxiv.org/html/2605.06472v1)
  - Every setting runs 10 times, with standard deviations in App. A, Table 2 — [App. A](https://arxiv.org/html/2605.06472v1)
  - Temperature and maximum tokens are **not stated**.
- **MORI**
  - "Every replayed request generates exactly its recorded number of output tokens, which include the reasoning tokens of the traced models (Claude Opus 4.6–4.8, Fable 5, and Sonnet 4.6 with thinking, and GPT-5-family models), regardless of the served model" — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - The forcing mechanism and the source of prompt token content are **not stated**.
  - (inference) The prompt content is probably synthesized. TraceLab's public traces keep token counts and timing, but "Full tool outputs are not stored" — [TraceLab README](https://github.com/uw-syfi/TraceLab)

### Inferences
- (inference) Forcing output lengths keeps each step's KV footprint and timing identical across the systems compared. A cache-policy comparison needs that.
- (inference) Forcing has costs:
  - MORI imposes frontier-model reasoning lengths, and probably foreign tokenizer counts, on Qwen and gpt-oss models.
  - InferCept and LAMPS use dummy weights and all-zero prompts, which removes any content-dependent effect, such as real cross-session prefix matches. Prefix hits on zero-token prompts would be trivially perfect.
- (inference) With free generation in a closed loop (ThunderAgent, PBKV), systems do different work. Steps per minute and completion time then mix scheduling effects with trajectory differences, and none of these papers reports task success.

### Gaps
- **Not stated:** whether outputs are forced in Autellix, SYMPHONY (main runs) and Astraea.
- **MORI:** how recorded token counts (from Claude and OpenAI tokenizers) map to the served models' tokenizers.

---

## 4. Tasks, benchmarks, datasets and traces (versions, counts, provenance)

### Takeaway
- **Public agent benchmarks run live** appear only in the 2026 papers and Hermes:
  - ThunderAgent: SWE-bench Lite, HLE, ScienceAgentBench.
  - PBKV: HoVer, SWE-bench, FinanceBench.
  - Hermes: ten ported applications.
- **Real-user agent traces:** only MORI replays them (AgentX and TraceLab Claude Code and Codex sessions, both public third-party releases).
- **InferCept family.** InferCept built its own six-augmentation workload (397 templates in the released file), and LAMPS/MARS and Astraea reuse it.
- **Chat and synthetic programs:** SYMPHONY and Autellix rely on ShareGPT conversations; Parrot and Conveyor use small self-made sets.
- **Counts.** Task and session counts per run are often missing.

### Cited Findings
- **InferCept**
  - **Workload types** — [HTML v2 §2.2; Appendix](https://arxiv.org/html/2402.01869v2):
    - Math: GSM8K-XL ("8.5K high-quality grade-school math problems") with a calculator.
    - QA: multihop Wikipedia QA (Yang et al., 2018, i.e. HotpotQA) via ReAct and the live Wikipedia API.
    - VE: ALFWorld via GPT-4 ReAct.
    - Chatbot: ShareGPT.
    - Image: ChatGPT- or GPT-4-generated prompt series sent to Stable Diffusion. §2.2 and the Appendix disagree on which model.
    - TTS: generated prompts sent to Bark.
  - **Counts.** The paper gives no per-type counts. The released file has 397 templates with 2–16 segments each, mean 5.17 (our computation).
  - **Sampling.** Each run samples `4*qps*window` requests with replacement — [final_tput_bench_real.py L185, L200](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py)
  - **Provenance:** the authors' own collection. It is partly synthetic: LLM-generated trajectories and prompts, plus estimated human time. The workload is distributed on Google Drive — [exps/README.md](https://github.com/WukLab/InferCept/blob/main/exps/README.md)
- **Parrot** — [Parrot HTML v1 §8](https://arxiv.org/html/2405.19888v1):
  - Long-document analytics: "ten long documents from the Arxiv-March dataset … Each document has over 20,000 tokens".
  - Bing Copilot: "We synthesized 64 requests from the length distribution we measured using Bing Copilot. The system prompt length is about 6000 tokens. The output lengths ranges from 180 to 800 tokens". Only the final request of each Copilot workflow is modelled.
  - GPTs: "four GPTs applications in four popular categories including productivity, programming, image generation, and data analysis".
  - MetaGPT: Architect, then Coders, then Reviewers; "This review-and-revision cycle is iterated three times"; up to 16 files.
  - Chat: ShareGPT-derived scenarios.
- **Conveyor**
  - "there are unfortunately no publicly available realistic datasets of tool-assisted LLM serving scenarios"; "we generate these synthetic workloads with our own prompts" — [Conveyor HTML v2 §4.1](https://arxiv.org/html/2406.00059v2)
  - Six scenarios, each a single prompt (CodeGen, Search, Planning, Database, Calculator, Validation\*).
- **LAMPS/MARS**
  - "(1) a single-API dataset (a subset of the INFERCEPT dataset containing only a single API), (2) the full INFERCEPT dataset, and (3) the ToolBench dataset" — [LAMPS HTML v2 §6.2](https://arxiv.org/html/2410.18248v2)
  - ToolBench is "comprising over 16,000 real-world APIs across 49 categories" — [§6.1](https://arxiv.org/html/2410.18248v2)
  - **Released ToolBench file construction** — [toolbench_workload_gen.py](https://github.com/mars-repository/mars-codebase/blob/main/prediction/toolbench_workload_gen.py):
    - source: `toolllama_G123_dfs_train_new.json`;
    - only single-user-turn conversations without API errors whose tools have a recorded delay;
    - prompts capped at 2048 tokens;
    - 80/20 split, seed 42.
  - **NeurIPS addition:** a mixed workload "where 50% of the requests are set as non-API (i.e., the API call is disabled but the request is kept)" — [MARS NeurIPS 2025 PDF, App. C.2](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf)
- **SYMPHONY**
  - ShareGPT (philschmid/sharegpt raw): "we use 1000 samples of the shareGPT dataset unless otherwise stated" — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
  - MetaGPT software-company workflow ("This review and revision cycle occurs three times"). The task prompts and project count are **not stated** — [§4.4](https://arxiv.org/html/2412.16434v1)
- **Autellix** — [Autellix HTML v1 §6.1, Fig. 11](https://arxiv.org/html/2502.13965v1):
  - ShareGPT: calls per program are "a long-tailed distribution with a mean of 6.66 and a max of 80".
  - BFCL: "The Berkeley Function Calling Leaderboard (BFCLv3) evaluates LLMs on multi-turn, multi-step tool-usage tasks"; "a mean of 10.75 and a maximum of 70 calls per program".
  - LATS: "derived from running MCTS on HotpotQA"; "on average 159.7 LLM calls".
  - Mixed: "We combine all three workloads, sampling equally from each".
  - Not stated: the number of programs per trace, and who produced the BFCL and LATS runs.
- **Hermes**
  - **Applications.** Ten, each with its upstream source pinned — [Hermes HTML v1 §2.1, Fig. 1; refs](https://arxiv.org/html/2506.14851v1):
    - Graph-of-Thoughts doc_merge (DM).
    - LangChain map-reduce summarization (MRS).
    - Certaindex reasoning (LLMR).
    - FacTool math, code and knowledge-QA verification (EV, CC, KBQAV).
    - ReAct FEVER (FEV) and ReAct ALFWorld (ALFWI).
    - AutoGen code generation (CG).
    - HuggingGPT/JARVIS (PE).
  - **Inputs:** "real inputs" from each application's "official dataset". The dataset versions are **not listed**.
  - **Application mix.** "small (EV, FEV, CC, ALFWI and KBQAV…), medium (CG and PE…) and large (DM and MRS…) … 72%, 26%, and 2%" — [§5.1](https://arxiv.org/html/2506.14851v1)
  - **Arrival times** come from the Mooncake production trace — [§5.1](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - The six InferCept sub-tasks.
  - "we artificially increased the sampling proportion of long-latency API requests in our experiments". The new proportions are **not stated** — [Astraea HTML v1 §5.1](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - **Main benchmarks** — [ThunderAgent HTML v3 §5.1](https://arxiv.org/html/2602.13692v3):
    - SWE-bench Lite with mini-SWE-agent and OpenHands;
    - HLE with ToolOrchestra;
    - ScienceAgentBench with OpenHands.
  - **Appendix tasks:** R2E-Gym (A.4, A.6) and OSWorld (G.2) — [App. A, G](https://arxiv.org/html/2602.13692v3)
  - **Task counts** are **not stated**. The repository runs `--subset lite --split test` — [reproduce_glm4.6.sh](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/mini-swe-agent/scripts/reproduce/reproduce_glm4.6.sh). (inference) That is the standard 300-instance test split.
- **PBKV**
  - "(i) fact verification, using the HoVer dataset with the LangChain agent framework; (ii) code generation, using SWE-bench with Microsoft AutoGen; and (iii) document analysis, using FinanceBench with CrewAI, which constitutes a static workflow" — [PBKV HTML v1 §6.1](https://arxiv.org/html/2605.06472v1)
  - **Predictor training:** "training on 1K traces achieves an accuracy of 0.94 at 1-step and 0.77 at 3-step (500-trace test set)" — [§4.1](https://arxiv.org/html/2605.06472v1)
  - Benchmark splits and per-run task counts are **not stated**.
- **MORI**
  - **AgentX:** "391 Claude Code (CC) sessions, which we split into 790 programs at idle periods longer than 30 minutes and at context compactions. A quarter of these programs spawn subagents (965 in total)". The source is the public SemiAnalysis Hugging Face dataset — [MORI HTML v2 §6.1; references](https://arxiv.org/html/2606.00866v2)
  - **TraceLab** (Claude Code and Codex, release v0.0.2): the analysis uses "11.4k sessions … (6.7k and 4.7k, respectively)" — [§3.2](https://arxiv.org/html/2606.00866v2); [TraceLab](https://github.com/uw-syfi/TraceLab)
  - **Context sizes:** median contexts of "108k, 98k, and 124k tokens" — [§6.1](https://arxiv.org/html/2606.00866v2)
  - The number of sessions replayed per run is **not stated**.

### Inferences
- (inference) For a paper about long-lived agent sessions, MORI's coding-agent traces are the only workload here with:
  - realistic context sizes (about 100k-token medians);
  - realistic gap tails;
  - subagent structure.
- (inference) The InferCept-lineage workload has very different shapes: 2048-token contexts and short sessions, with a mean of 5.17 segments.
- (inference) Benchmarks run live (SWE-bench Lite, HLE, HoVer) tie results to one agent framework's prompting and tool conventions. PBKV states this as a limitation (see §11).

### Gaps
- **Counts not stated:** task or session counts per run for ThunderAgent, PBKV, Autellix and MORI's TraceLab replay.
- **Dataset versions not listed:** Hermes's input datasets and PBKV's benchmark splits.
- **SYMPHONY:** the MetaGPT task set is unspecified.

---

## 5. Agent framework and integration: request path to the engine, what crosses the boundary, and whether the engine parses tool calls

### Takeaway
- **The common interface.** Most recent systems use an OpenAI-compatible API plus a program or session ID, plus an explicit end-of-program signal:
  - ThunderAgent: `program_id` and `POST /programs/release`.
  - MORI: `program_id`, with release documented in the README.
  - Autellix: session, program and thread IDs, plus `start_session` and `end_session`.
  - PBKV: workflow tags and termination messages.
- **Richer hints are rare:**
  - SYMPHONY sends timing hints (`expected_arrival`).
  - Parrot (Semantic Variables) and Hermes (per-application PDGraphs) expose application structure.
- **Duration forecasts.** No paper has the agent framework send per-tool duration forecasts. Durations are:
  - estimated engine-side (InferCept: elapsed time);
  - per-class means (LAMPS/MARS, Astraea);
  - or not used at all (ThunderAgent, MORI).
- **Tool-call parsing.** Only Conveyor's engine parses tool calls from its own output in the evaluation. InferCept's engine does so by design, but in its benchmark a token counter triggers the pause.

### Cited Findings
- **InferCept**
  - **Framework.** ReAct was used only to collect traces; the evaluation is an in-process harness with no HTTP API.
  - **Per-segment metadata.** `SamplingParams` carry `api_invoke_interval`, `api_exec_time`, `api_return_length` and `api_max_calls`. On return, the harness calls `resume_request(request_id, tokens, params)`, so the request keeps its identity across interceptions — [final_tput_bench_real.py L108–145](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py)
  - **Design-time parsing.** Users "register your aug-stop token in `vllm/utils.py`" — [README](https://github.com/WukLab/InferCept); [vllm/utils.py](https://github.com/WukLab/InferCept/blob/main/vllm/utils.py)
  - **Duration estimate.**
    - §4.4 estimates the interception duration as T̂ = t_now − t_call, which "achieves 93% of the performance compared with using an oracle" — [InferCept HTML v2 §4.4](https://arxiv.org/html/2402.01869v2)
    - The released waste code uses `api_exec_time - (now - api_call_time)`, the true remaining time — [sequence.py L407](https://github.com/WukLab/InferCept/blob/main/vllm/sequence.py)
    - Which of the two Fig. 2 used is **not stated**.
- **Parrot**
  - **Baseline path.** "The majority of LLM applications used in our baseline comparisons are developed using LangChain … leverage OpenAI-style chat completion APIs as provided by FastChat" — [Parrot HTML v1 §8.1](https://arxiv.org/html/2405.19888v1)
  - **Parrot path.** SemanticFunctions are lowered to "OpenAI-like APIs with the extension of Semantic Variables" over FastAPI.
    - Submit carries prompt placeholders (name, in/out, `semantic_var_id`, transforms) and `session_id`.
    - Get carries `semantic_var_id`, performance criteria and `session_id`.
    - The service derives the request DAG and prefix hashes.
    - The engine API is `Fill`, `Generate` and `FreeContext`, with parent contexts.
    - — [§4.1–4.2, §7](https://arxiv.org/html/2405.19888v1)
  - **Tools.** The service applies output transforms between requests but runs no tools — [§5.1](https://arxiv.org/html/2405.19888v1)
- **Conveyor**
  - There is no agent framework. Tool developers register "a parser and a plugin" with the serving system — [Conveyor HTML v2 §3.2](https://arxiv.org/html/2406.00059v2)
  - **The engine parses its own token stream** for tool-invocation indicators and streams partial arguments to the tool process over IPC while decoding continues — [§3.3, Fig. 5](https://arxiv.org/html/2406.00059v2)
  - In the released driver, requests enter in-process via `engine.request_pool.add_request` — [task.py](https://github.com/conveyor-sys/conveyor/blob/main/task.py)
- **LAMPS/MARS**
  - The in-process harness passes both true and `predicted_*` values in `SamplingParams` — [fixed_final_tput_bench_real.py](https://github.com/mars-repository/mars-codebase/blob/main/benchmarks/fixed_final_tput_bench_real.py)
  - **Design.** The system "predicts pre-API output length and estimates API properties (duration and response length, Table 2) based on input prompts" — [LAMPS HTML v2 §4.1](https://arxiv.org/html/2410.18248v2)
  - **Multi-API requests.** "After an API call completes, the request re-enters the system … treated as a new request" — [§4.2](https://arxiv.org/html/2410.18248v2)
- **SYMPHONY**
  - The scheduler exposes "(i) An interface for accepting LLM inference requests, and (ii) An interface for accepting advisory requests" — [SYMPHONY HTML v1 §3.2](https://arxiv.org/html/2412.16434v1)
  - **Advisory JSON:** `session_id`, `model_id`, `expected_arrival` (None for chat, e.g. 3.2 for an agent), `ordered` and `priority` — [§3.1, Figs. 10–11](https://arxiv.org/html/2412.16434v1)
  - **Hint sources.** Chat hints fire "as soon as the user starts typing"; agent hints are sent by the upstream agent to "the next downstream agent(s)".
  - **Internal transport.** "Symphony uses gRPC" between the scheduler and the node managers — [§4.1](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - It "extends OpenAI's Chat Completion and vLLM's Python APIs … upon program initialization, Autellix automatically issues a start_session request to the backend".
  - "Subsequent LLM calls are transparently annotated with the appropriate session, program, and thread IDs".
  - "When the program completes or encounters an error, Autellix invokes end_session" — [Autellix HTML v1 §5](https://arxiv.org/html/2502.13965v1)
  - "When a program arrives, its execution DAG is initially unknown" — [§4.1](https://arxiv.org/html/2502.13965v1)
- **Hermes**
  - **Submission.** "a user launches a cloud-hosted LLM application with its tasks submitted to the HermesScheduler" — [Hermes HTML v1 §3.1](https://arxiv.org/html/2506.14851v1)
  - **Transport.** "ZeroMQ-based RPC" connects the HermesScheduler and the HermesLet. The HermesLet "reports execution status while receiving updated request priorities and prewarming signals" — [§4](https://arxiv.org/html/2506.14851v1)
  - **Application structure.** The provider holds per-application PDGraphs (structure, backend specification, demand distributions and branch probabilities) and hosts the application code — [§2.1, §3.2](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - Requests are sequences of segments, and the segment count "unknown upon the request's initial arrival" — [Astraea HTML v1 §3.1](https://arxiv.org/html/2512.14142v1)
  - Each segment carries a predicted compute time (an offline prefill profile plus the oracle $n_{gen}$) and the API category's mean latency — [§4.1–4.2](https://arxiv.org/html/2512.14142v1)
  - How API calls are detected is **not stated**.
- **ThunderAgent**
  - "ThunderAgent directly wraps existing LLM engines and tool orchestrators by interfacing with OpenAI-style endpoints. Program IDs allow the system to distinguish requests from different agentic workflows" — [ThunderAgent HTML v3 §4.1](https://arxiv.org/html/2602.13692v3)
  - **Fig. 11's three changes:** `program_id` in `extra_body` of each chat completion; `run_tool(command, sandbox, program_id=…)`; and "Program end (explicit release) POST /programs/release" — [App. B.3, Fig. 11](https://arxiv.org/html/2602.13692v3)
  - **Phase tracking.** In the repository router, phases change at request boundaries: an arriving request means reasoning, and a finished response moves the program to acting. Nothing parses tool calls — [router.py](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/ThunderAgent/scheduler/router.py)
- **PBKV**
  - "We tag each cache node with the workflows that have accessed it. The server continuously listens for and records workflow termination messages from clients" — [PBKV HTML v1 §4.2](https://arxiv.org/html/2605.06472v1)
  - The predictor reads "the post-norm hidden state of the last prefill token" — [§4.1](https://arxiv.org/html/2605.06472v1)
  - The HTTP or API path is **not stated**.
  - Each agent "is invoked as a single LLM request" — [§1](https://arxiv.org/html/2605.06472v1)
- **MORI**
  - "a standard OpenAI-compatible chat/completion API augmented with a program_id field … a subagent uses a distinct one" — [MORI HTML v2 §5](https://arxiv.org/html/2606.00866v2)
  - **Labels to the engine.** Each tick's type-label updates "are batched into a single request, and SGLang stores the labels on the nodes of its radix-tree prefix cache" — [§4.4, §5](https://arxiv.org/html/2606.00866v2)
  - **No tool information.** "MORI needs to know only which program a tool call belongs to" — [§6.2](https://arxiv.org/html/2606.00866v2)
  - **State transitions.** "Each incoming request transitions a program to Reasoning; each completed response transitions it back to Acting" — [§4.2](https://arxiv.org/html/2606.00866v2)
  - **End of program.** The README documents `POST /programs/release` "when the session ends" — [MORI README](https://github.com/cblmemo/mori)

### Inferences
- (inference) If agents send forecast hints (tool name, expected duration, end-of-session), the interface itself goes beyond this group of papers. The nearest precedents are SYMPHONY's `expected_arrival` and Hermes's application-level probability graphs.
- (inference) An evaluation of such hints must show they are obtainable. Replay can derive them from the trace; live runs need the agent framework to emit them.
- (inference) The baseline interface, against which a hint interface's gain should be measured, is a program ID plus a release signal. ThunderAgent and MORI show that it is enough to infer reasoning and acting phases from request boundaries.

### Gaps
- **PBKV:** the transport of workflow tags and termination messages is not stated.
- **InferCept:** whether Fig. 2 used the oracle or the elapsed-time estimator is not stated.
- **Hermes:** how applications are ported (the degree of code change) is not described.

---

## 6. Load: concurrency, arrival process, run length

### Takeaway
- **Two regimes.**
  - **Open-loop Poisson at request or program granularity:** InferCept, LAMPS/MARS, Autellix, Parrot's GPTs. Hermes uses open-loop arrivals shaped by the Mooncake trace.
  - **Closed loop at fixed concurrency:** ThunderAgent (24–120 programs), MORI (20–120 programs in flight, 2-hour runs), PBKV (concurrency caps 24–96 on a finite pool), SYMPHONY (8–1,024 active users).
- **Single requests.** Conveyor measures one request at a time.
- **Reporting gaps.** Run length and repetition counts are frequently missing. Only Conveyor (100 runs) and PBKV (10 runs) report repetitions.

### Cited Findings
- **InferCept**
  - The paper says only: "varying request load (number of requests arrived per second)" — [InferCept HTML v2 §5.1](https://arxiv.org/html/2402.01869v2)
  - **Arrival process (code):** `rng.exponential(1.0 / args.qps)`, i.e. open-loop Poisson — [final_tput_bench_real.py L273](https://github.com/WukLab/InferCept/blob/main/benchmarks/final_tput_bench_real.py)
  - **Fig. 2 x-axes:** 1.0–4.0 req/s (6B, and 13B on 2 GPUs); 0.5–3.0 (13B on 1 GPU); 0.4–1.4 (70B on 4 GPUs) — [InferCept PDF v2, Fig. 2](https://arxiv.org/pdf/2402.01869v2)
  - **Run length:** "Each data point will run for 30min" — [exps/README.md](https://github.com/WukLab/InferCept/blob/main/exps/README.md)
  - **Breakdown load:** "the load of 2 requests per second for the 6B model" — [§5.2, Fig. 3](https://arxiv.org/html/2402.01869v2)
- **Parrot** — [Parrot HTML v1 §8](https://arxiv.org/html/2405.19888v1); [PDF v1](https://arxiv.org/pdf/2405.19888v1):
  - Chain summaries run against background requests "at varying rates", with 25 concurrent applications in Fig. 13.
  - Bing Copilot runs at batch sizes 8–64 (Figs. 15–16).
  - GPTs: "LLM requests arrive at fixed rates following Poisson distribution".
  - Mixed workload: chat at "a rate of 1 req/s" plus map-reduce applications.
- **Conveyor**
  - One request at a time, 100 runs per workload — [Conveyor HTML v2 §4.2](https://arxiv.org/html/2406.00059v2)
  - No concurrency and no arrival process.
- **LAMPS/MARS**
  - **Rates.** Figs. 6, 8 and 10 span 3–6 req/s — [LAMPS PDF v2](https://arxiv.org/pdf/2410.18248v2). Error injection runs at "higher request rates (8-10 req/s)" — [§6.4](https://arxiv.org/html/2410.18248v2)
  - **Run length.** "We measured throughput by limiting each benchmark to 30 minutes and counting completed requests" — [§6.2](https://arxiv.org/html/2410.18248v2)
  - Arrivals are exponential (code).
- **SYMPHONY**
  - Closed loop with a "fixed number of active users" — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
  - The x-axis sweeps 8–1,024 users — [PDF v1, Fig. 12](https://arxiv.org/pdf/2412.16434v1)
  - Figs. 14 and 17 use 256 users — [§4.3, §4.5](https://arxiv.org/html/2412.16434v1)
  - Run length: **not stated**.
- **Autellix**
  - Open-loop Poisson program arrivals, with latency plotted against "Arrival Rate (program/s)" — [Autellix HTML v1 §6.1](https://arxiv.org/html/2502.13965v1); [PDF v1, Figs. 12–14](https://arxiv.org/pdf/2502.13965v1)
  - Offline mode: at "4000 programs, MLFQ fails to complete execution" — [§6.5.1](https://arxiv.org/html/2502.13965v1)
  - A single engine handles "only 0.2 programs per second for MCTS" — [§1](https://arxiv.org/html/2502.13965v1)
  - Run length: **not stated**.
- **Hermes**
  - **Main runs.** Open loop with Mooncake-shaped arrivals: 300 applications for LLaMA2-7B and 3,000 for LLaMA3-70B, submitted over "10, 15, and 30 minutes (corresponding to workload intensity levels of 3×, 2×, and 1×)" — [Hermes HTML v1 §5.2](https://arxiv.org/html/2506.14851v1)
  - **Other runs** — [§5.3–5.4](https://arxiv.org/html/2506.14851v1):
    - deadline runs: 300 applications in 15 min;
    - ablation: 300 in 10 min;
    - KV prewarm: 500 in 15 min;
    - LoRA: 1,000 in 25 min.
- **Astraea**
  - "average JCT across a range of Queries Per Second (QPS) at four distinct GPU memory availability levels: 30%, 50%, 70%, and 90%" — [Astraea HTML v1 §5.2](https://arxiv.org/html/2512.14142v1)
  - QPS 1–5 appear in App. C.
  - Arrival process and run length: **not stated**.
- **ThunderAgent**
  - **Concurrency.** Fixed "parallel workflow number (i.e., batch size)" — [Fig. 1 caption](https://arxiv.org/html/2602.13692v3). "high concurrency levels (e.g., 96 parallel programs)" — [§5.2](https://arxiv.org/html/2602.13692v3)
  - **Ranges.** Fig. 4 ticks span 24–120 across panels — [PDF v3](https://arxiv.org/pdf/2602.13692v3). Rollout uses N = 144 on 2 nodes (Table 2). A.3 uses "72 programs per node", up to 8 nodes, and A.4 sweeps "batch size from 64 to 192" — [App. A](https://arxiv.org/html/2602.13692v3)
  - **Arrivals:** **not stated**. The repository runs a worker pool (`--workers 128`), i.e. closed-loop (inference) — [reproduce_glm4.6.sh](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/mini-swe-agent/scripts/reproduce/reproduce_glm4.6.sh)
  - **Measurement window (repository):** "from 10 minutes after startup to 1 hour 10 minutes after startup" — [mini-swe-agent README](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/mini-swe-agent/README.md)
- **PBKV**
  - "we set the concurrency limit to 72 / 24 / 48" for HoVer, SWE-bench and FinanceBench, "to induce memory pressure"; Fig. 5a sweeps 36–96 — [PBKV HTML v1 §6.1–6.3](https://arxiv.org/html/2605.06472v1)
  - Each run is a finite pool with warm-up and drain: "As the request pool drains … by completion (around 285 s)" — [§6.4, Fig. 6](https://arxiv.org/html/2605.06472v1)
  - Pool size: **not stated**.
- **MORI**
  - "A closed-loop client keeps a fixed number of programs in flight, from 20 to 120 … when a program finishes, the client starts the next one. On AgentX, the subagents of a program run alongside it as additional programs. Each run lasts two hours" — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - Fig. 7 uses 80 programs — [§6.4](https://arxiv.org/html/2606.00866v2)

### Inferences
- (inference) Closed-loop fixed concurrency suits agent sessions: each session's next request depends on the previous response and the tool gap. But faster systems then complete different numbers of programs and tool gaps in the same window (MORI, ThunderAgent).
- (inference) Open-loop Poisson over intercepted requests (InferCept lineage) can push the system into overload, where counting only finished requests biases latency.
- (inference) For a host- and SSD-tier paper, the run length must cover several long gaps per session; MORI's gaps reach p99 519–600 s. MORI's two-hour runs do; PBKV's roughly 285-second runs could not.

### Gaps
- **Arrival process not stated:** ThunderAgent (paper), Astraea, and InferCept/LAMPS in the paper text (only the code shows it).
- **Not reported:** warm-up exclusion and repetitions in most papers.

---

## 7. Stack: engine and version, models, GPUs, host memory, CPU and SSD KV tiers

### Takeaway
- **Engines.**
  - vLLM forks: InferCept and LAMPS/MARS (0.2.0), Hermes (0.4.3), Autellix (0.6.1), SYMPHONY, Astraea and ThunderAgent (versions not stated in the papers).
  - SGLang with HiCache: PBKV (version not stated) and MORI (v0.5.10).
  - Custom engines: Parrot and Conveyor.
- **Models** grow over time:
  - 6B–70B models with 2,048-token contexts in the InferCept lineage;
  - GLM-4.6 (355B), Qwen3-235B and gpt-oss-120b in ThunderAgent and MORI.
- **KV tiers.** GPU plus CPU swap or host DRAM dominates.
  - Only SYMPHONY includes a disk tier, and it never isolates the disk's effect.
  - ThunderAgent's appendix reports LMCache CPU/SSD offloading as harmful on vLLM.
- No paper evaluates SSD reads timed around tool gaps.

### Cited Findings
- **InferCept**
  - **Engine:** "on top of vLLM" — [InferCept HTML v2 §4.5](https://arxiv.org/html/2402.01869v2). The repository's version string is 0.2.0 — [vllm/__init__.py](https://github.com/WukLab/InferCept/blob/main/vllm/__init__.py)
  - **Models and GPUs.** "We run augmented GPT-J on one NVIDIA A100 GPU. For Vicuna, we use two environments: running on a single A100 GPU and distributed on two A100 GPUs with tensor parallelism. For Llama3, we distribute it on four A100 GPUs" — [§5](https://arxiv.org/html/2402.01869v2)
  - **Host memory** is not stated. The scripts set `--swap-space` to 128 GiB (6B) and 64 GiB (13B with TP2).
  - **KV tiers:** GPU plus CPU swap; no SSD.
- **Parrot**
  - **Testbeds.** "24-core AMD-EPYC-7V13 CPUs equipped with one NVIDIA A100 (80GB) GPU" and "64-core EPYC AMD CPU and four NVIDIA A6000 (48GB) GPUs" — [Parrot HTML v1 §8.1](https://arxiv.org/html/2405.19888v1)
  - **Models:** "LLaMA 13B or LLaMA 7B".
  - **Engine:** its own, "based on efficient kernels from vLLM, xFormers, and ourselves" — [§7](https://arxiv.org/html/2405.19888v1)
  - **KV tiers:** GPU only. The FastChat and vLLM versions are not stated.
- **Conveyor**
  - "two Intel 10-core Xeon Gold 5215 CPUs … and two NVIDIA GeForce RTX 3090 GPUs … PyTorch with FlashInfer CUDA kernels … about 1.9K lines of Python" — [Conveyor HTML v2 §4.1](https://arxiv.org/html/2406.00059v2)
  - **Models:** "Mistral-7B-Instruct-v0.2" and "Functionary-Small-v2.2".
  - **KV tiers:** none evaluated.
- **LAMPS/MARS**
  - **Testbed.** "dual AMD EPYC 7313 CPUs … 503 GB of RAM, and two NVIDIA A100 GPUs with 80 GB memory each connected via NVLink. We manually limited the maximum memory usage of each GPU to 40 GB to emulate the setup used in INFERCEPT's experiments" — [LAMPS HTML v2 §6.1](https://arxiv.org/html/2410.18248v2)
  - **Models:** GPT-J 6B and Vicuna 13B. The NeurIPS version adds TinyLlama-v1.1 (1.1B) and Meta-Llama-3-70B ("served using vLLM's default tensor parallelism (set to 2)") — [MARS NeurIPS 2025 PDF §4](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf)
  - **Predictor:** OPT-125M with 50 bins — [§5](https://arxiv.org/html/2410.18248v2)
  - **KV tiers:** GPU plus CPU swap.
- **SYMPHONY**
  - **Testbed.** "2 nodes, with each node having 4 Nvidia A100 GPUs with 80 GB of HBM. Each node is equipped with 256 GB of DRAM and 4 TB of SSDs … 100Gbps Ethernet" — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
  - **Models:** "LLaMa-3.1 8B … with 128K context length and LLaMa-2-13B with 32K context length".
  - **Engine.** vLLM, version not stated: "Symphony integrates with vLLM's scheduler and performs block management" — [§4.1](https://arxiv.org/html/2412.16434v1)
  - **KV tiers:** HBM, DRAM, disk and remote. "we always maintain one copy of the K,V cache in the slowest memory hierarchy (disk)" — [§3.2–3.3](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - **Engine.** "Autellix builds on vLLM v0.6.1 … we modify only the scheduler … and memory swapping kernels" — [Autellix HTML v1 §5](https://arxiv.org/html/2502.13965v1)
  - **Models and hardware.** "LLaMA-3.1-8B, 70B and Falcon-180B, running on 1, 4, and 8 GPUs … a2-ultragpu-8g instance with eight A100-SXM4-80GB GPUs … 1360 GB host memory, PCIe-4.0×16, and 2 TB of disk space" — [§6.2](https://arxiv.org/html/2502.13965v1)
  - **KV tiers.** CPU swap applies to preempted calls only, with "a large swap space (>1.2TB)" — [§6.5.1](https://arxiv.org/html/2502.13965v1)
- **Hermes**
  - **Engine:** "vLLM 0.4.3" — [Hermes HTML v1 §4](https://arxiv.org/html/2506.14851v1)
  - **Servers** — [§5.1](https://arxiv.org/html/2506.14851v1):
    - "one NVIDIA A100-PCIe-40GB GPU", with 128 GB DRAM and 4 TB SSD;
    - "eight NVIDIA H800-80GB GPUs", with 2 TB DRAM and 28 TB SSD.
  - **Models:** LLaMA2-7B and 13B; "two LLaMA3-70B models across eight H800-80GB GPUs using 4-way tensor parallelism".
  - **KV tiers.** The source tier for KV prefetch is **not stated**. Fig. 13a sweeps "cache sizes (8GB, 16GB, and 32GB)" — [§5.4](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - **Testbed.** "two NVIDIA A100 80GB GPUs connected via NVLink, a 28-core Intel Xeon Gold 6330 CPU at 2.0GHz, and 503GB of RAM"; GPT-J 6B on one A100 and Vicuna-13B on two — [Astraea HTML v1 §5.1](https://arxiv.org/html/2512.14142v1)
  - **KV policy.** Below a GPU-memory high watermark, Preserve; above it, the argmin of InferCept's waste over Preserve, Discard and Swap — [§4.4](https://arxiv.org/html/2512.14142v1)
  - The watermark value, swap size and vLLM version are **not stated**.
- **ThunderAgent**
  - **Engine.** "vLLM is employed as our LLM inference engine" — [ThunderAgent HTML v3 §5.1](https://arxiv.org/html/2602.13692v3). The repository pins `vllm==0.10.1` for ToolOrchestra — [ToolOrchestra README](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/ToolOrchestra/README.md)
  - **Models and GPUs.** "GLM-4.6 (355B) and Qwen-3 (235B) … quantized to FP8 with Tensor Parallelism (TP8) on 8×H100 nodes"; "Qwen3-8B with FP16 precision hosted on one RTX 5090" — [§5.1](https://arxiv.org/html/2602.13692v3)
  - **Host memory:** **not stated**.
  - **Offloading (A.2).** "our implementation with vLLM + LMcache reveals a critical bottleneck: the PCIe bandwidth is insufficient to sustain the high-frequency context switching and large-volume data transfers inherent to agentic workloads" — [App. A.2](https://arxiv.org/html/2602.13692v3)
  - **HiCache (A.4).** ThunderAgent is integrated "into SGLang HiCache with the host-memory ratio set to 2.0, serving MiniMax M2.5 on R2E-Gym across 8×H100 GPUs". The baseline's P50 latency "jumps from ∼10 s to ∼30 s at batch 160 and to ∼70 s at batch 192" — [App. A.4, Fig. 9](https://arxiv.org/html/2602.13692v3)
- **PBKV**
  - **Testbed.** "8× NVIDIA A6000 (48 GB) … NVLink, 128 virtual CPU cores, 512 GB of memory, and 20 GB/s PCIe" — [PBKV HTML v1 §6.1](https://arxiv.org/html/2605.06472v1)
  - **Models:** "Qwen3-14B and Qwen3-32B (two-way tensor parallelism)".
  - **Engine:** SGLang + HiCache with a host-to-GPU capacity ratio of 1 ("allocating equal capacity on GPU and host memory"). The version is **not stated**.
  - **KV tiers:** GPU plus host DRAM; no SSD.
- **MORI**
  - **Engine.** "All systems run on the same SGLang build (v0.5.10) and offload to CPU DRAM through HiCache" — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - **Models and GPUs (Table 1):**

    | GPU | Model | TP | GPU KV pool |
    |---|---|---|---|
    | H200 | Qwen2.5-7B-Instruct-1M | 1 | 1.94M tokens |
    | H200 | gpt-oss-120b | 1 | 0.71M tokens |
    | B300 | Qwen3-235B-A22B (FP8) | 2 | 1.29M tokens |

    — [Table 1](https://arxiv.org/html/2606.00866v2)
  - **CPU pool.** The default is GPU:CPU = 1:1. Fig. 7 holds the total at 2.59M tokens and varies the ratio from 1:1 to 1:4 — [§6.1, §6.4](https://arxiv.org/html/2606.00866v2)
  - **Launch flags:** `--hicache-write-policy write_back` and `--hicache-ratio` ("We evaluated 1.0 and 2.0") — [MORI README](https://github.com/cblmemo/mori)
  - **SSD** is future work — [§7](https://arxiv.org/html/2606.00866v2)

### Inferences
- (inference) There is no established methodology to borrow for SSD-tier evaluation:
  - SYMPHONY keeps a disk copy but shows no disk-specific result.
  - ThunderAgent's A.2 result (LMCache offload harmful) concerns PCIe bandwidth under swap churn, not timed prefetch.
  - MORI explicitly defers SSD.
- (inference) An SSD-tier paper therefore has to define and justify its own device configuration: bandwidth, queue depth, and shared-device contention.
- (inference) Physical host DRAM is often unreported (InferCept, ThunderAgent, MORI), so host scarcity is set by pool ratios (HiCache ratio 1–2; GPU:CPU 1:1–1:4) rather than absolute sizes. Reporting both would be stronger.

### Gaps
- **Versions not stated:** vLLM for SYMPHONY, Astraea and ThunderAgent (paper); SGLang for PBKV.
- **Hermes:** the KV prefetch source tier is not stated.

---

## 8. Metrics

### Takeaway
- **Latency per request or program:** TTFT, normalized latency (latency divided by output tokens), and end-to-end completion time of the request, program, job or application.
- **Throughput:** requests/s, steps/min, or output tokens/s.
- **Cache metrics:** InferCept (GB·min waste), PBKV (GPU token hit rate), ThunderAgent (KV hit rate), MORI (cache-served fraction and recompute after a CPU drop), Hermes (prewarm hit ratio).
- **Task success** is reported by none.

### Cited Findings
- **InferCept**
  - "normalized latency (i.e., the median of every request's end-to-end latency divided by its output length)"; "We also remove a request's intercepted time from its end-to-end latency because it is the same across all serving systems"; also throughput and TTFT — [InferCept HTML v2 §5.1](https://arxiv.org/html/2402.01869v2)
  - GPU memory waste (GB·min) — [§3.2, §5.2](https://arxiv.org/html/2402.01869v2)
- **Parrot**
  - End-to-end application latency, latency per output token, sustainable request rate, "normalized latency (measured as request latency per number of output tokens)", and GPU KV memory — [Parrot HTML v1 §8](https://arxiv.org/html/2405.19888v1)
- **Conveyor**
  - Mean request-completion latency ± standard deviation, timelines, "0.6% extra CPU cycles", and "20–40 lines of code" per tool — [Conveyor HTML v2 §4.2–4.5](https://arxiv.org/html/2406.00059v2)
- **LAMPS/MARS**
  - "end-to-end latency, defined as the time from when a request is submitted to the system until its completion, time-to-first-token (TTFT) … and throughput", reported as mean and P99 — [LAMPS HTML v2 §6.1](https://arxiv.org/html/2410.18248v2)
  - Predictor accuracy: "68.5% accuracy for Acc-5 and 78.3% accuracy for Acc-15"; 13.7 ms per prediction — [§6.4](https://arxiv.org/html/2410.18248v2)
- **SYMPHONY**
  - Normalized latency ("each request's end-to-end latency divided by its output length"), TTFT, TPOT, requests served, per-server load imbalance, and MetaGPT completion time — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - "program-level token latency, defined as the total program response time divided by the number of tokens generated", using the critical path for multi-threaded programs — [Autellix HTML v1 §6.2](https://arxiv.org/html/2502.13965v1)
  - Also: average, P95 and P99; maximum rate under an SLO; makespan; a wait/schedule/swap time breakdown; and prefix-hit rates (Fig. 7) — [§6](https://arxiv.org/html/2502.13965v1)
- **Hermes**
  - Average and P95 application completion time, deadline satisfaction ratio, KV and LoRA cache hit ratio, latency reduction versus wastage over the knob K, and policy runtime — [Hermes HTML v1 §5.2–5.5](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - "average Job Completion Time (JCT), which measures the end-to-end latency from request submission to final completion"; JCT includes API time — [Astraea HTML v1 §3.1, §5.1](https://arxiv.org/html/2512.14142v1)
  - Scheduler overhead — [App. C](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - "steps per minute as our throughput metric, where one step includes a reasoning and acting period of the workflow"; KV-cache hit rate (Fig. 5); latency breakdown; "4.2× disk memory savings" — [ThunderAgent HTML v3 §5.1–5.4](https://arxiv.org/html/2602.13692v3)
  - Per-step end-to-end latency — [App. E](https://arxiv.org/html/2602.13692v3)
  - P50 and P90 latency in A.4 — [App. A.4](https://arxiv.org/html/2602.13692v3)
- **PBKV**
  - End-to-end workflow latency ("from workflow submission to final completion"), per-agent TTFT, and "Average KV-Cache hit rate (token-level) on GPU memory" — [PBKV HTML v1 §6.1](https://arxiv.org/html/2605.06472v1)
  - Hit rate over time (Fig. 6) and predictor top-1 accuracy (App. C) — [§6.4; App. C](https://arxiv.org/html/2605.06472v1)
- **MORI**
  - "output throughput (output tokens per second over the run) and the p90 and median time to first token (TTFT) measured at the client, which includes queueing in the scheduler, KV cache reloads, and prefill" — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - Fraction of prompt tokens served from the GPU or CPU cache, and fraction of reusable prefix recomputed after a CPU copy was dropped — [§6.3, Fig. 6](https://arxiv.org/html/2606.00866v2)

### Inferences
- (inference) For a KV prefix-cache tiering paper, the most directly comparable metric set is MORI's:
  - client-side TTFT that includes reloads;
  - output tokens/s;
  - per-tier hit fractions;
  - recomputed-prefix fraction.

  Add PBKV-style hit rate over time and session completion time.
- (inference) Normalized-latency metrics (InferCept, SYMPHONY, Autellix) divide by output tokens. InferCept also removes interception time, which hides the latency of reloading KV after a tool returns.
- (inference) That reload latency is the quantity a prefetch policy targets, so a tiering paper should report post-tool TTFT directly.

### Gaps
- **Error bars:** reported only by Conveyor and PBKV. The MARS NeurIPS checklist says error bars were omitted "because we are calculating the response time in a system in equilibrium (steady-state)" — [MARS NeurIPS 2025 PDF, checklist](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf)

---

## 9. Baselines and ablations

### Takeaway
- **Default engines.** Every paper compares against the default engine policy: vLLM, or SGLang with LRU or longest-prefix-match.
- **Re-implemented or ported baselines are common:**
  - Continuum ported to SGLang (MORI);
  - "LAS" standing in for Autellix (Astraea);
  - Parrot and VTC as policies (Hermes);
  - KVFlow on a workload built for it (PBKV).
- **Ablations** are mostly parameter sweeps. Component ablations exist for InferCept, LAMPS, Astraea, ThunderAgent and PBKV, but not for MORI.

### Cited Findings
- **InferCept**
  - **Baselines:** vLLM (Discard), ImprovedDiscard, Preserve and Swap — [InferCept HTML v2 §5](https://arxiv.org/html/2402.01869v2)
  - **Ablations:**
    - incremental technique breakdown (Fig. 3);
    - QA-only and Chatbot-only workloads (§5.1);
    - dynamic duration estimate versus oracle (§4.4).
- **Parrot**
  - **Baselines:** FastChat with vLLM or HuggingFace, tuned for latency or for throughput, and vLLM with static prefix sharing. Capacity thresholds are set by the authors, e.g. "4096 tokens" for map-reduce.
  - **Ablations:** affinity scheduling off; prompt-structure sharing off; vLLM's PagedAttention kernel in place of Parrot's.
  - — [Parrot HTML v1 §8](https://arxiv.org/html/2405.19888v1)
- **Conveyor**
  - Only a self-baseline: "the same code but with tool partial execution disabled" — [Conveyor HTML v2 §4.1](https://arxiv.org/html/2406.00059v2)
- **LAMPS/MARS**
  - **Baselines:** vLLM and InferCept.
  - **Ablations (v2):** "LAMPS w/o scheduling"; starvation threshold (100 chosen); error injection at 5/10/30/50% — [LAMPS HTML v2 §6.3–6.4](https://arxiv.org/html/2410.18248v2)
  - **NeurIPS additions:** mixed API and non-API workload, TinyLlama, 70B, and memory occupancy over time — [MARS NeurIPS 2025 PDF, App. C](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf)
- **SYMPHONY**
  - **Baselines:** vLLM with recomputation ("vLLM treats each request … as a new request"), and InferCept with sticky routing ("for each successive request within the same session, we route the request to the same node") — [SYMPHONY HTML v1 §4](https://arxiv.org/html/2412.16434v1)
  - **Ablations:** prefill-heavy workload; 0, 10 and 20% missing advisory requests; prioritization — [§4.5](https://arxiv.org/html/2412.16434v1)
- **Autellix**
  - **Baselines:** vLLM v0.6.1 (FCFS); "vLLM-opt", which "enables chunk-prefill, prefix-caching, and multi-step scheduling"; and MLFQ. Round Robin and Least Used serve as load-balancing baselines.
  - **Ablations:** offline makespan; timing breakdown; clairvoyant SRPT in the simulator; swap kernel.
  - — [Autellix HTML v1 §6.2–6.5](https://arxiv.org/html/2502.13965v1)
- **Hermes**
  - **Scheduling baselines:** vLLM (request-level FCFS), Parrot (application-level FCFS), VTC and EDF — [Hermes HTML v1 §5.1](https://arxiv.org/html/2506.14851v1)
  - **Cache baselines:** LRU and EPWQ (Evict/Prefetch-Waiting-Queue) — [§5.4](https://arxiv.org/html/2506.14851v1)
  - **Ablations:** without online refinement, without Gittins, Oracle, and a K sweep — [§5.3–5.4](https://arxiv.org/html/2506.14851v1)
- **Astraea**
  - **Baselines:** "(1) vLLM with FCFS … (2) vLLM with SJF … (3) vLLM with LAS … the core scheduling idea adopted by Autellix … (4) Infercept (with its default FCFS policy)" — [Astraea HTML v1 §5.1](https://arxiv.org/html/2512.14142v1)
  - **Ablation:** the scheduler alone, without swap or discard (Fig. 6).
  - **Sensitivity:** aging and queue thresholds — [App. B](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - **Baselines:** vLLM; Continuum ("predicting tool execution durations and pinning KV cache to HBM"); and "vLLM + SGLang Gateway" for rollout — [ThunderAgent HTML v3 §5.1](https://arxiv.org/html/2602.13692v3)
  - **Appendix comparisons:** SGLang cache-aware routing (A.3), SGLang + HiCache (A.4), and LMCache and PD disaggregation (A.2).
  - **Ablations:** Δt and f(t); global versus local queue; decay form; eviction order; component-wise — [§5.4; App. G](https://arxiv.org/html/2602.13692v3)
- **PBKV**
  - **Baselines:** LRU on SGLang + HiCache, and KVFlow evaluated only on the "static workflow".
  - **Ablations:** variants PBKV-LAE and PBKV-HE; predictor backbones; horizon K; γ; aggressive prefetch and noise.
  - — [PBKV HTML v1 §6; App. C–E](https://arxiv.org/html/2605.06472v1)
- **MORI**
  - **Baselines:** SGLang (longest cached prefix first), ThunderAgent, and "Continuum … ported to SGLang", all offloading through HiCache — [MORI HTML v2 §6.1](https://arxiv.org/html/2606.00866v2)
  - **Ablation:** only the GPU:CPU ratio sweep (Fig. 7) — [§6.4](https://arxiv.org/html/2606.00866v2)

### Inferences
- (inference) The baselines this literature expects for a tiered prefix-cache retention and prefetch paper on SGLang:
  - SGLang + HiCache with its default eviction;
  - a duration-predicting pin policy (Continuum, as ported by MORI);
  - a program-aware scheduler (ThunderAgent);
  - MORI itself (code public);
  - possibly an InferCept-style waste rule.
- (inference) Ablations that separate the forecast's contribution from the policy's (oracle, no-hint, noisy hint) have precedent:
  - InferCept's estimate versus oracle;
  - LAMPS's error injection;
  - SYMPHONY's missing-advisory sweep;
  - PBKV's noise sweep.

### Gaps
- **MORI:** no component ablation (window k, proactive promotion, typed eviction).
- **Undocumented port fidelity:** MORI's Continuum port and PBKV's KVFlow implementation are not documented.

---

## 10. Released artifacts: code, traces, scripts

### Takeaway
- **Code with drivers:** InferCept, MARS (NeurIPS version only), Conveyor, ThunderAgent and MORI.
- **Workload files:** InferCept's workload is on Google Drive, with a copy in the MARS repo; MARS's ToolBench workload is in its repo.
- **Replayed traces:** only MORI uses publicly available traces, published by third parties. Its own replay client is not released.
- **No artifacts linked:** SYMPHONY, Autellix, Hermes, Astraea, Parrot (v1). PBKV mentions only "supplementary code".

### Cited Findings
- **InferCept**
  - Code: [WukLab/InferCept](https://github.com/WukLab/InferCept)
  - Scripts exist only for 6B and 13B-TP2; "The profiler is still under refactoring"; the workload is on Google Drive — [exps/README.md](https://github.com/WukLab/InferCept/blob/main/exps/README.md)
- **Parrot:** no repository link in arXiv v1. All its GitHub links are third-party — [Parrot HTML v1](https://arxiv.org/html/2405.19888v1)
- **Conveyor**
  - "Our source code will be publicly available at https://github.com/conveyor-sys/conveyor" — [Conveyor HTML v2 §1](https://arxiv.org/html/2406.00059v2)
  - The repository contains the `task.py` and `run_eval.py` drivers — [repo](https://github.com/conveyor-sys/conveyor)
- **LAMPS/MARS**
  - arXiv v2 links no code.
  - The NeurIPS version: "Our implementation is available online"; the repository has drivers, workload JSONs and predictor code — [MARS NeurIPS 2025 PDF](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf); [mars-codebase](https://github.com/mars-repository/mars-codebase)
- **SYMPHONY, Autellix, Hermes, Astraea:** no code or trace links. Their GitHub references point to third-party projects or upstream application sources — [SYMPHONY HTML v1](https://arxiv.org/html/2412.16434v1); [Autellix HTML v1](https://arxiv.org/html/2502.13965v1); [Hermes HTML v1](https://arxiv.org/html/2506.14851v1); [Astraea HTML v1](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - MIT-licensed code with reproduce scripts: mini-SWE-agent with GLM-4.6 and Qwen3-235B; ToolOrchestra with Qwen3-8B — [repo](https://github.com/ThunderAgent-org/ThunderAgent); [examples README](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/README.md)
  - No traces, since the runs are live.
- **PBKV:** "full hyperparameters in the supplementary code"; no public URL in v1 — [PBKV HTML v1 §4.1, App. B](https://arxiv.org/html/2605.06472v1)
- **MORI**
  - Code: the scheduler (a ThunderAgent fork) and the engine (an SGLang fork) — [MORI repo](https://github.com/cblmemo/mori)
  - The repository's top level contains only `README.md`, `ThunderAgent/` and `sglang/` (repository tree listing). The replay client and trace preprocessing are therefore not released (inference).
  - The traces are public: AgentX on Hugging Face and TraceLab on GitHub — [MORI HTML v2, references](https://arxiv.org/html/2606.00866v2); [TraceLab](https://github.com/uw-syfi/TraceLab)

### Inferences
- (inference) Releasing the replay client, trace preprocessing and launch scripts would set a tiering paper apart. None of the closest replay-based works releases its replay client:
  - MORI releases its code but not its replay client;
  - InferCept's driver uses dummy weights and is tied to vLLM 0.2.0.

### Gaps
- **Unverified:** whether the released code for Conveyor, InferCept and MARS matches the code used for the paper figures.

---

## 11. Threats to validity

### Takeaway
- **Few authors state threats.** Explicit statements are limited to:
  - PBKV: predictor training per workload; dependence on message conventions.
  - MORI: AgentX has no tool names; SSD and multi-replica routing deferred.
  - MARS: average API response length.
  - Conveyor: tool and model coverage.
  - SYMPHONY: hints arriving late.
- **Threats we noticed recur across papers:**
  - tool time that does not depend on load, from sleep replay;
  - forced output lengths and dummy weights;
  - small or synthetic task sets;
  - no repetitions or error bars;
  - ported baselines;
  - host DRAM unreported;
  - no SSD tier.

### Cited Findings
- **InferCept**
  - **Stated:** no limitations section. The README warns: "Please do not schedule two swap-involved run concurrently as we assume exclusive access to the PCIE bendwidth" [sic] — [exps/README.md](https://github.com/WukLab/InferCept/blob/main/exps/README.md)
  - **Noticed:**
    - The paper says the engine "calls the actual API", while the released runs sleep for recorded times.
    - Dummy weights and zero-token prompts.
    - Normalized latency excludes interception time.
    - The released waste estimator uses the true remaining time — [HTML v2](https://arxiv.org/html/2402.01869v2); [sequence.py L407](https://github.com/WukLab/InferCept/blob/main/vllm/sequence.py)
- **Parrot**
  - **Stated:** no dynamic control flow or native functions; only the final Bing Copilot request is modelled — [Parrot HTML v1 §6, §8.3](https://arxiv.org/html/2405.19888v1)
  - **Noticed:**
    - Chain-summary gains depend on the injected 200–300 ms delay: "The efficiency of Parrot primarily stems from the decreased network latency" — [§8.2](https://arxiv.org/html/2405.19888v1)
    - Engine kernels differ from the baselines'.
- **Conveyor**
  - **Stated:** "limited to the capability of existing open-source models and the existing sets of tools that they support" — [Conveyor HTML v2 §4.4](https://arxiv.org/html/2406.00059v2). Gains are negligible for light tools — [§5](https://arxiv.org/html/2406.00059v2)
  - **Noticed:** single-request, consumer-GPU microbenchmarks; one fixed prompt per workload; some repository tools are stubs.
- **LAMPS/MARS**
  - **Stated:**
    - "MARS currently approximates API output size using average response length. However, actual API responses can be highly variable" — [MARS NeurIPS 2025 PDF §3](https://papers.nips.cc/paper_files/paper/2025/file/679b0e02978404ee569ce2abfb8455cf-Paper-Conference.pdf)
    - ToolBench gains are limited "due to many requests exceeding 2048 tokens" — [LAMPS HTML v2 §6.2](https://arxiv.org/html/2410.18248v2)
  - **Noticed:**
    - The 40 GB GPU cap rests on a guess about InferCept's hardware ("which we inferred to involve A100 GPUs with 40 GB memory based on their use of AWS machines") — [§6.1](https://arxiv.org/html/2410.18248v2)
    - ToolBench durations are per-tool constants, which makes the "low variance within a class" premise true by construction.
    - The released ToolBench statistics do not match Table 2.
- **SYMPHONY**
  - **Stated:** "Symphony assumes that advisory requests arrive early enough… in extreme cases… performance may be impacted" — [SYMPHONY HTML v1 §3.5](https://arxiv.org/html/2412.16434v1)
  - **Noticed:**
    - The workload is chat with synthetic typing gaps, and the hint lead time comes from the same model.
    - MetaGPT is on "4 A100s" in §1 but "8 GPUs" in §4.4.
    - Fig. 12's caption says 4× more users while the text says 8×.
    - The motivation's "each token requires approximately 1.1 MB of K,V cache" — [§2.4](https://arxiv.org/html/2412.16434v1) — looks about 8× too high for Llama-3.1-8B. Our arithmetic: 2 × 32 layers × 8 KV heads × 128 dims × 2 B = 128 KiB per token.
- **Autellix**
  - **Stated:** the research-prototype frontend "lacks safeguards against user modification" — [Autellix HTML v1 §5](https://arxiv.org/html/2502.13965v1)
  - **Noticed:**
    - Interrupt time is not modelled, so KV retention across tool gaps is never exercised.
    - The max batch size, MLFQ quanta and output control are not stated.
    - The load-balancer gain is "up to 1.5x" in §1 but "up to 1.4×" in §6.4.
- **Hermes**
  - **Stated:** no explicit threats; §6 lists PD disaggregation and multi-backend routing as future work — [Hermes HTML v1 §6](https://arxiv.org/html/2506.14851v1)
  - **Noticed:**
    - Application arrivals reuse a chatbot request trace (Mooncake).
    - The application size mix is borrowed from DL-cluster schedulers.
    - Separation between profiling and evaluation inputs is not stated.
    - The engine is old (vLLM 0.4.3).
- **Astraea**
  - **Stated:** the predictor is "idealized yet practical", chosen to isolate scheduling from prediction error — [Astraea HTML v1 §4.2](https://arxiv.org/html/2512.14142v1)
  - **Noticed:**
    - Generation length is an oracle, and the API means are in-sample.
    - The long-latency reweighting is unspecified.
    - Figure captions keep template text ("This is an example image showing …").
    - JCT is dominated by API and think time: 105.77 s to 122.88 s between QPS 1 and 5 — [App. C](https://arxiv.org/html/2512.14142v1)
- **ThunderAgent**
  - **Stated:**
    - Remote-tool latency varies "across requests and over time" — [ThunderAgent HTML v3 App. C](https://arxiv.org/html/2602.13692v3)
    - Exponential decay is optimal only under memoryless tool times (F.1); linear decay wins on OSWorld (G.2).
  - **Noticed:**
    - No repetitions; the repository default is `REP=1` — [reproduce_hle_qwen3_8b.sh](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/examples/inference/ToolOrchestra/scripts/reproduce/reproduce_hle_qwen3_8b.sh)
    - Third-party APIs make runs time-dependent.
    - Table 2 ("GLM-4.6", "vLLM + Gateway") and Table 7 ("GLM-4.5-fp8", "vLLM") label the same 375 → 672 numbers differently.
    - The paper and repository disagree on where HLE-answer runs and on the number of RTX 5090s.
    - The repository router constructor defaults to `use_acting_token_decay: bool = False` — [router.py](https://github.com/ThunderAgent-org/ThunderAgent/blob/HEAD/ThunderAgent/scheduler/router.py). Whether the CLI enables the decay was not checked.
- **PBKV**
  - **Stated:**
    - "(i) The predictor needs to be trained on a specific workload. (ii) Cache-reuse patterns… depend on the client's specific message-passing convention" — [PBKV HTML v1 §5](https://arxiv.org/html/2605.06472v1)
    - "Restricted to the steady-state interior, LRU's hit rate drops to ∼10% while full PBKV's sustains ∼50%" — [§6.2](https://arxiv.org/html/2605.06472v1)
  - **Noticed:**
    - Tools are undescribed.
    - Load is a capped batch tuned to stress memory.
    - The hit rate counts GPU hits only.
    - KVFlow is evaluated only on a workload built for it.
- **MORI**
  - **Stated:**
    - "AgentX's traces do not name the tools, so this workload also emulates serving without tool information" — [MORI HTML v2 §6.2](https://arxiv.org/html/2606.00866v2)
    - SSD and multi-replica routing are future work — [§7](https://arxiv.org/html/2606.00866v2)
  - **Noticed:**
    - Frontier-model output and reasoning lengths are imposed on Qwen2.5-7B and gpt-oss.
    - Tokenizers differ.
    - Prompt content and shared system prompts are unspecified.
    - Median contexts of 98–124k sit near gpt-oss-120b's window.
    - No repetitions.
    - The ThunderAgent baseline is the code MORI builds on.

### Inferences
- (inference) A replay-based KV-tiering evaluation should pre-empt four threats:
  1. **Load-independent tool time.** Add a sensitivity run with load-dependent or jittered gaps, or a live cross-check.
  2. **Forced lengths and tokenizer mismatch.** State the mapping, or serve a model whose tokenizer matches the trace.
  3. **Prompt content.** Reproduce shared prefixes such as system prompts and tool schemas. Otherwise cross-session reuse is either absent or trivially perfect.
  4. **Repetitions.** Repeat runs and report variance. Most of these papers do not.

### Gaps
- **Venue versions mostly unread.** Only the MARS NeurIPS camera-ready was read; its evaluation extends arXiv v2 (see §9). The Hermes TACO version and the ThunderAgent ICML 2026 version may differ from the arXiv versions read.

### Comparison table across papers (part A: workload realism and integration)

| Paper (version read) | Mode | Tasks / traces (provenance) | Tool / interception time | Output control | Framework → engine path | What crosses the boundary |
|---|---|---|---|---|---|---|
| InferCept (v2) | (ii)+(iv): recorded-interception replay on a vLLM fork; dummy weights in released scripts | Six augment types (GSM8K-XL math, HotpotQA ReAct + live Wikipedia, ALFWorld via GPT-4, ShareGPT chat, SD image, Bark TTS); own collection; 397 templates resampled | `time.sleep` of recorded per-call time; Table 1 mean/variance; human time estimated; not load-dependent | Forced: `ignore_eos`, recorded lengths, temp 0, zero tokens | No framework at run time; in-process `LLMEngine` | Per-segment `api_invoke_interval`, `api_exec_time`, `api_return_length`; request ID persists; pause on token counter (design: aug-stop tokens) |
| Parrot (v1) | Mixed: static app DAGs without tools (§8.2, §8.4) + synthetic streams (§8.3, §8.5) | Arxiv-March (10 docs), Bing Copilot (64 synthesized from production lengths), 4 GPTs apps, MetaGPT, ShareGPT | No tools; random 200–300 ms network delay | Lengths matched to recorded GPT-4 responses | Baseline: LangChain → FastChat OpenAI-style API → vLLM/HF; Parrot: SemanticFunction → FastAPI | Placeholders, `semantic_var_id`, `session_id`, latency/throughput criteria → service derives DAG and prefix hashes |
| Conveyor (v2) | Live single-request runs with real tools + analytic model (Fig. 8) | 6 self-made one-prompt scenarios; no dataset | Real execution in separate processes (Python, web search, SQLite, calculator); some repo tools stubbed | Free, temperature 0 (deterministic) | In-process engine API; tool parser + plugin registered | Engine parses its own decode stream; partial arguments streamed to tool over IPC |
| LAMPS / MARS (v2 + NeurIPS 2025) | (ii)+(iv) on InferCept's harness; noise injection; offline predictor accuracy | InferCept single-API and full; ToolBench (16k APIs; 1,580 released conversations) | Sleeps: recorded (InferCept) or per-tool constants (ToolBench) | Forced; scheduler sees truth, truth + Gaussian noise, or OPT-125M predictions | In-process harness | Predicted pre-API length; API class → mean duration and response length |
| SYMPHONY (v1) | (iv) synthetic chat sessions on vLLM-based prototype; one MetaGPT figure (mode not stated) | ShareGPT 1,000 samples; MetaGPT software-company workflow | Synthetic reading + typing gaps (parameters not stated); no tools | Not stated (ablation: 1,024-token prompt, 1-token output) | Custom scheduler with inference + advisory endpoints; gRPC to node managers | Advisory {`session_id`, `model_id`, `expected_arrival`, `ordered`, `priority`} |
| Autellix (v1) | (ii) program replay on modified vLLM; (iii) simulator for Fig. 18 | ShareGPT whole conversations, BFCLv3, LATS (MCTS on HotpotQA), Mixed | Not stated; interception time called "unrelated to LLM serving" | Not stated | Own Python library extending OpenAI Chat Completion + vLLM APIs | Session/program/thread IDs; `start_session`/`end_session` |
| Hermes (v1) | (i) live ported apps on vLLM + Docker/DNN backends; Mooncake-shaped arrivals | 10 apps (FacTool ×3, ReAct FEVER/ALFWorld, AutoGen, GoT, LangChain MRS, HuggingGPT, Certaindex); official-dataset inputs | Real Docker/DNN execution incl. cold starts | Free; oracle = temperature-0 trial run | Custom HermesScheduler/HermesLet over ZeroMQ RPC | App ID + PDGraph (structure, demand distributions, branch probabilities), execution status, priorities, prewarm signals |
| Astraea (v1) | (ii)-like by inference (driver undescribed) | InferCept dataset, long-latency requests over-sampled | Not stated; category-mean prediction | Not stated; oracle $n_{gen}$ for scheduler | vLLM-based engine; no framework | Segments with predicted compute + API-category mean latency |
| ThunderAgent (v3) | (i) live agents, serving and RL rollout | SWE-bench Lite (mini-SWE-agent, OpenHands), HLE (ToolOrchestra), ScienceAgentBench (OpenHands); R2E-Gym, OSWorld in appendices | Real Docker sandboxes on a CPU cluster; remote APIs; heavy-tailed, load-dependent | Free (repo: temp 0, `ignore_eos` False, 135 steps) | Agents → OpenAI-style proxy → vLLM | `program_id` on LLM and tool calls; `POST /programs/release` |
| PBKV (v1) | (i) live LangChain / AutoGen / CrewAI against modified SGLang + HiCache | HoVer, SWE-bench, FinanceBench (counts not stated); 1K traces train predictor | Not described | Free stochastic decoding; 10 runs | Frameworks → SGLang (API path not stated) | Workflow tag, agent identity, workflow-termination messages |
| MORI (v2) | (ii) trace replay on SGLang + HiCache | AgentX (391 CC sessions → 790 programs + 965 subagents); TraceLab CC/Codex v0.0.2; public real-developer traces | Sleeps of recorded tool-call and human-wait durations | Forced to recorded output-token counts (frontier reasoning included) | Replay client → OpenAI-compatible proxy (ThunderAgent fork) → SGLang | `program_id` (distinct per subagent); batched type labels to engine each tick; release endpoint (README) |

### Comparison table across papers (part B: load, stack, metrics, baselines, artifacts)

| Paper | Load | Engine | Models | GPUs / host | KV tiers | Key metrics | Baselines | Artifacts |
|---|---|---|---|---|---|---|---|---|
| InferCept | Open Poisson (code), 0.4–4 req/s; 30-min runs | vLLM 0.2.0 fork | GPT-J 6B, Vicuna-13B, Llama3-70B | A100 ×1/2/4; host not stated | GPU + CPU swap (64–128 GiB) | Normalized latency (interception removed), req/s, TTFT, GB·min waste | vLLM (Discard), ImprovedDiscard, Preserve, Swap | Code + 6B/13B-TP2 scripts; workload on Google Drive |
| Parrot | Rate sweeps; 10–25 concurrent apps; batch 8–64; Poisson GPTs; chat 1 req/s | Own engine; FastChat + vLLM/HF baselines | LLaMA 7B/13B | 1× A100-80GB; 4× A6000-48GB | GPU only | App E2E latency, per-token latency, max rate, KV memory | FastChat/vLLM latency- and throughput-centric, HF, vLLM static prefix sharing | None linked |
| Conveyor | 1 request at a time, 100 runs/workload | Custom PyTorch + FlashInfer | Mistral-7B-Instruct-v0.2, Functionary-Small-v2.2 | 2× RTX 3090 host | None | Mean latency ± std, timelines, CPU overhead | Same system without partial execution | Repo (Apache-2.0) |
| LAMPS / MARS | Poisson (code) 3–6 req/s (8–10 for errors); 30-min windows | vLLM 0.2.0 fork | GPT-J 6B, Vicuna 13B (+ TinyLlama 1.1B, Llama-3-70B TP2) | 2× A100-80GB capped at 40 GB; 503 GB RAM | GPU + CPU swap | Mean/P99 E2E (incl. API time), TTFT, throughput, predictor accuracy | vLLM, InferCept; w/o scheduling; starvation; error injection | NeurIPS repo with workloads |
| SYMPHONY | Closed loop, 8–1,024 users (256 in Figs. 14, 17) | vLLM (version not stated) + Symphony | Llama-3.1-8B, Llama-2-13B | 2 × 4 A100-80GB; 256 GB DRAM + 4 TB SSD per node | HBM + DRAM + disk + remote | Normalized latency, TTFT, TPOT, requests served, imbalance | vLLM recompute; InferCept with sticky routing | None |
| Autellix | Open Poisson program arrivals; offline 1k–4k programs | vLLM v0.6.1 (modified) | Llama-3.1-8B/70B, Falcon-180B | Up to 8× A100-80GB; 1,360 GB host | GPU prefix cache + CPU swap for preemption | Program-level token latency (avg/P95/P99), max rate under SLO, makespan | vLLM, vLLM-opt, MLFQ; RR, Least-Used | None |
| Hermes | Open, Mooncake-shaped; 300 / 3,000 apps over 10–30 min | vLLM 0.4.3 | LLaMA2-7B/13B, LLaMA3-70B (TP4 ×2) | A100-PCIe-40GB (128 GB DRAM, 4 TB SSD); 8× H800 (2 TB DRAM, 28 TB SSD) | KV/LoRA prefetch; source tier not stated | Avg/P95 app completion time, deadline ratio, hit ratio, wastage | vLLM FCFS, Parrot, VTC, EDF; LRU, EPWQ | None |
| Astraea | QPS ~1–5 at 4 GPU-memory levels; arrivals not stated | vLLM (version not stated) | GPT-J 6B, Vicuna-13B | 2× A100-80GB; 503 GB RAM | GPU + CPU swap | Average JCT (incl. API time), overhead | vLLM FCFS/SJF/LAS, InferCept | None |
| ThunderAgent | Fixed concurrency 24–120 (serving), 144 (rollout), up to 576 on 8 nodes; closed-loop workers (repo) | vLLM (repo pins 0.10.1 for ToolOrchestra); SGLang HiCache in A.4 | GLM-4.6 FP8, Qwen3-235B FP8, Qwen3-8B (+ GLM-4.5-FP8, MiniMax M2.5, Qwen3-32B) | 8× H100 nodes (1–8), 8× A100, RTX 5090; host not stated | GPU only (main); HiCache ratio 2.0 (A.4); LMCache CPU/SSD (A.2) | Steps/min, KV hit rate, per-step latency, disk use | vLLM, Continuum, vLLM + SGLang Gateway, SGLang router, SGLang + HiCache | Code + reproduce scripts (MIT) |
| PBKV | Concurrency caps 72/24/48 (sweep 36–96); finite pool (~285 s); 10 runs | SGLang + HiCache (version not stated) | Qwen3-14B, Qwen3-32B (TP2) | 8× A6000-48GB; 512 GB host; 20 GB/s PCIe | GPU + host DRAM (ratio 1); no SSD | Workflow latency, per-agent TTFT, GPU token hit rate | LRU (HiCache), KVFlow (static only) | "Supplementary code" only |
| MORI | Closed loop, 20–120 programs in flight (+ subagents); 2-h runs | SGLang v0.5.10 + HiCache (write_back) | Qwen2.5-7B-Instruct-1M, gpt-oss-120b, Qwen3-235B-A22B FP8 | 1× H200 (two configs), 2× B300 TP2; DRAM size not stated | HBM + DRAM (1:1 default; 1:1–1:4 at 2.59M tokens); no SSD | Output tok/s, p90/median TTFT (client, incl. reloads), cache-served and recompute fractions | SGLang LPM, ThunderAgent, Continuum ported to SGLang | Code (scheduler + engine forks); public traces; replay client not released |
