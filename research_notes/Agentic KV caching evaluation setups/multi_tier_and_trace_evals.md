# How multi-tier KV-cache serving papers evaluate themselves, and which agentic or multi-turn traces can be replayed

How to read these notes.
- **Sources.** Every paper was read in full from arxiv.org/pdf on 2026-11-17. The version used is named in each heading, and section, figure and table numbers refer to that version. Mooncake was also read in its FAST '25 version from usenix.org.
- **Quotes** in double quotes were checked by script against the extracted text, after whitespace normalization.
- **Mode codes:**
  - (i) live agents: a real agent framework runs tasks end to end against the engine;
  - (ii) trace replay on a real engine;
  - (iii) simulation;
  - (iv) synthetic workload, such as chat or QA datasets with Poisson arrivals and synthetic think or tool times. Unless stated otherwise, (iv) here runs on a real engine.
- **Other markers.** "(inference)" marks my own reading. "Not stated" means the paper does not say. "Read from figure" means a value was read off a plotted axis.
- **Dataset pages** (Hugging Face, GitHub, Zenodo) were checked on 2026-11-17. No large dataset was downloaded.

## Q1. How does each multi-tier KV-cache paper evaluate its system? (checklist items 1–11, per paper)

### Takeaway
None of the seven papers runs live agents, and none replays recorded tool-call gaps on a real engine.
- **Synthetic timing on real engines** (mode iv): CachedAttention, Pensieve, LMCache and Strata drive chat or QA datasets with Poisson arrivals and an invented or unstated gap between turns. Pensieve samples an exponential gap with mean 60 s. Strata inserts a fixed 60 s gap on ShareGPT. CachedAttention and LMCache do not state the gap.
- **Recorded timestamps on real engines** (mode ii):
  - Mooncake replays production traces by timestamp, but its traces carry no session IDs, and its cache-policy results are latency-free simulations.
  - The Alibaba study replays a downscaled production trace with parent-chat links on vLLM. It is the only paper here with real sessions and real timestamps on a real engine, and its traffic has almost no agentic share.
- **Simulation only** (mode iii): Marconi is a CPU trace simulation with synthetic gaps. The paper never says so; its artifact does.
- **SSD.** Only CachedAttention measures an SSD tier end to end, and it does not specify the device.
- **Outputs.** Where outputs are controlled, they are forced to recorded lengths (Mooncake FAST '25, Alibaba). Elsewhere output control is not stated.

### Cited Findings

#### CachedAttention (its KV store is AttentionStore): arXiv 2403.19708v3, 30 Jun 2024
Venue: not printed in v3. Pensieve's reference [16] cites it as USENIX ATC '24 ([Pensieve v3](https://arxiv.org/pdf/2312.05516v3)).

1. **Mode: (iv) on the authors' own engine.** Every serving result measures their PyTorch/Transformers engine on one 4×A100 server.
   - The workload is ShareGPT content with synthetic Poisson session arrivals (Figs. 13–16, 21–25).
   - Fig. 17 applies AWS prices to those runs.
   - Figs. 18–20 are fixed-length microbenchmarks, and Tables 1–2 are model-quality benchmarks.
   - There is no simulator and no timestamped replay. The only "simulate" in the paper is an accuracy test ([v3 §4.1–4.3](https://arxiv.org/pdf/2403.19708v3)).
2. **Gaps between turns: not stated.** Only session arrivals are given: "We set the number of different sessions arriving per second according to a Poisson distribution (with λ = 1.0). 9K conversation sessions are used in the experiments." ([v3 §4.1](https://arxiv.org/pdf/2403.19708v3)).
   - Nothing says when a session's next turn arrives, or whether there is think time.
   - Waiting turns form the queue that the prefetch and eviction windows scan: "the job scheduler maintains a job queue, thus having the full knowledge of waiting jobs" ([v3 §3.3](https://arxiv.org/pdf/2403.19708v3)).
   - A per-session TTL ("the KV cache of each session has a TTL (time to live)") is set to 1 h, in one experiment only ([v3 §4.3.6](https://arxiv.org/pdf/2403.19708v3)).
3. **Model outputs: not stated.** The paper gives no max tokens, no ignore_eos, and does not say whether the history holds recorded or generated replies. One microbenchmark fixes 20 decoding steps ([v3 §4.1, §4.3.2](https://arxiv.org/pdf/2403.19708v3)).
4. **Data: ShareGPT** (Hugging Face philschmid/sharegpt-raw, "more than 90K conversations").
   - "we use 9K conversations from ShareGPT [33] and the average number of turns in these conversations is 5.75. Thus the total number of conversation turns is about 52K. We warm up AttentionStore using the first 10K conversation turns and then evaluate the performance on the following 42K turns."
   - How the 9K were selected is not stated ([v3 §4.1](https://arxiv.org/pdf/2403.19708v3)).
5. **Client–engine interface: not stated.**
   - Requests become jobs in the system's own scheduler, so the workload is fed in-process (inference).
   - Hints come from the internal queue, not from a client.
   - The session is the cache unit: "one item corresponds to all KV caches associated with a conversation session" ([v3 §3.1, §3.3.2](https://arxiv.org/pdf/2403.19708v3)).
6. **Load.**
   - Default 1.0 sessions/s; Fig. 25 sweeps 0.5–2.0 sessions/s on LLaMA-13B.
   - Batches of 24 ("LLaMA-13B operates on two GPUs with 24 batches").
   - Concurrency and wall-clock run length are not stated ([v3 §4.1, §4.3.8](https://arxiv.org/pdf/2403.19708v3)).
7. **Stack.**
   - "All our experiments are performed on 4 NVIDIA A100 GPUs, each with 80GB HBM. The system is equipped with 128GB DRAM and 10TB SSDs." PCIe Gen4.
   - SSD type and bandwidth are not stated; the paper says only that disks have "less than 5 GB/s" access bandwidth.
   - Models: LLaMA-1 65B, LLaMA-2 13B and 70B, Falcon-40B, all FP16.
   - Tiers: HBM+DRAM+SSD in the main runs. Fig. 21 varies the SSD over 2, 5 and 10 TB. Fig. 24 compares HBM-only (10 GB), HBM+DRAM (128 GB), and the full system with the 10 TB SSD added ([v3 §2.4, §4.1, §4.3.3, §4.3.7](https://arxiv.org/pdf/2403.19708v3)).
8. **Metrics.**
   - Hit rate, split into DRAM and disk.
   - TTFT, with neither its start point nor the statistic (mean or percentile) stated.
   - Prefill throughput.
   - GPU time in hours.
   - Dollar cost at "$5/hour per A100 GPU, $0.0088/hour/GB for DRAM and $0.000082/hour/GB for SSD".
   - Perplexity and accuracy ([v3 §4.2–4.3, Fig. 17](https://arxiv.org/pdf/2403.19708v3)).
9. **Baselines.**
   - RE, which recomputes and "discards KV caches after serving a conversation".
   - LRU and FIFO eviction.
   - OF, with positional encoding embedded in the stored KV.
   - HBM-only and HBM+DRAM configurations.
   - TT and NKVT truncation.

   All are modes of the authors' own engine; no external engine is run ([v3 §4.1–4.3](https://arxiv.org/pdf/2403.19708v3)).
10. **Released artifacts: none stated.** The 9K-session subset and the arrival generator are not released ([v3](https://arxiv.org/pdf/2403.19708v3)).
11. **Threats to validity.**
    - **Acknowledged by the authors:**
      - "As there is no public request arrival timestamp available in the dataset, we generate request arrival times based on the Poisson distribution".
      - LLaMA-65B's lower hit rate comes from its larger KV per token.
      - There is no limitations section ([v3 §4.1–4.2](https://arxiv.org/pdf/2403.19708v3)).
    - **Inferred:**
      - The unstated gap between turns decides whether a turn hits DRAM, SSD, or nothing.
      - Look-ahead prefetch helps only turns that are already queued, and queue depth is never reported.
      - One node, chat data only, no repeated runs.

#### Pensieve: arXiv 2312.05516v3, 7 Oct 2024
Venue: EuroSys '25, as printed in the paper ([v3](https://arxiv.org/pdf/2312.05516v3)).

1. **Mode: (iv) on real engines.**
   - The workload is ShareGPT and UltraChat content with Poisson request arrivals and exponential think time.
   - It runs on their C++/CUDA prototype, vLLM v0.2.0 and TensorRT-LLM v0.12.0, on Azure A100s (Figs. 10–15).
   - Figs. 3, 4 and 12 are microbenchmarks. There is no simulator ([v3 §6.1–6.7](https://arxiv.org/pdf/2312.05516v3)).
2. **Gaps between turns: sampled.**
   - "we also simulate user think time, aka the time taken for users to generate the next conversation turn, by sampling from an exponential distribution with varying mean". "Our experiments so far use the average user think time of 60 seconds".
   - Fig. 15 sweeps means of 30, 60, 120, 300 and 600 s.
   - Causality is kept: "a new user prompt is only sent to the system after the response to the conversation’s previous request has been received".
   - Arrivals: "we simulate a request’s arrival time by sampling from a Poisson distribution under different request rates".
   - How Poisson arrival and think time combine is not stated ([v3 §6.1, §6.7](https://arxiv.org/pdf/2312.05516v3)).
3. **Model outputs: not stated.**
   - A request ends at end-of-sequence or maximum length, and the maximum is not given.
   - Whether the history holds generated or recorded replies is not stated.
   - Llama 2-13B was modified, with KV heads cut from 40 to 10; how the weights were derived is not stated ([v3 §4.2, §6.1](https://arxiv.org/pdf/2312.05516v3)).
4. **Data.**
   - ShareGPT: 48,159 conversations, 5.56 turns. UltraChat: 1,468,352 conversations, 3.86 turns.
   - Context is capped at 16,384 tokens, which drops 0.57% of ShareGPT.
   - Conversations sampled per run: not stated ([v3 §6.1, Table 2](https://arxiv.org/pdf/2312.05516v3)).
5. **Client–engine interface.**
   - The server is stateful: dropped KV is recomputed "from the conversation history saved in a persistent store".
   - The baselines are stateless, and vLLM is driven by an in-house loop: "we implement such a driver that adds newly arrived requests into vLLM job queue".
   - Scheduling is FCFS with no hints ([v3 §4.2, §4.3.4, §6.1](https://arxiv.org/pdf/2312.05516v3)).
6. **Load.**
   - Poisson request-rate sweeps; plots show achieved throughput of up to about 4–10 req/s depending on model (read from figure).
   - Offered rates, concurrency and duration are not stated.
   - (Inference) With a 60 s mean think time, r req/s implies about 60·r open conversations ([v3 §6.1, Figs. 10–15](https://arxiv.org/pdf/2312.05516v3)).
7. **Stack.**
   - Their own C++/CUDA engine on PyTorch 2.0.
   - Azure NC A100 v4 with 1 or 4 A100-80GB and "220 GB CPU memory per GPU".
   - "we configure it to allocate 40 GB memory on each GPU for KV cache" for every system.
   - Tiers: GPU plus CPU DRAM, no SSD ("Putting it on disk would incur much longer load latency").
   - Models: OPT-13B/66B, Llama 2-13B/70B ([v3 §1, §5, §6.1](https://arxiv.org/pdf/2312.05516v3)).
8. **Metrics.**
   - Throughput against normalized latency (s per output token). The definition is inconsistent: "90-percentile normalized latency" in §6.1 vs "the mean of each request’s end-to-end latency divided by its output length" in §6.2.
   - CPU hit rate and kernel time.
   - No TTFT ([v3 §6.1–6.2, §6.6](https://arxiv.org/pdf/2312.05516v3)).
9. **Baselines.** vLLM v0.2.0 and TensorRT-LLM v0.12.0, both stateless; Pensieve (GPU cache only); Pensieve with LRU ([v3 §6.1, §6.6](https://arxiv.org/pdf/2312.05516v3)).
10. **Released artifacts: none stated** ([v3](https://arxiv.org/pdf/2312.05516v3)).
11. **Threats to validity.**
    - **Acknowledged by the authors:**
      - Timing is synthetic because "the datasets do not provide timestamps".
      - Throughput falls as think time grows, and the gap to vLLM narrows at 600 s.
      - TensorRT-LLM beats Pensieve (GPU cache) in some panels ([v3 §6.1, §6.2, §6.7](https://arxiv.org/pdf/2312.05516v3)).
    - **Inferred:**
      - The exponential think time has no cited source.
      - Baselines are stateless only.
      - No SSD tier, and the offered load cannot be reproduced.

#### Mooncake: arXiv 2407.00079v5, 8 Oct 2026, plus the FAST '25 paper
Venue: v5 is stamped "Preprint Tech Report by Moonshot AI and MadSys @ Tsinghua". The peer-reviewed version is FAST '25 ([PDF](https://www.usenix.org/system/files/fast25-qin.pdf)). The v5 text carries the technical-report evaluation, not the FAST '25 one.

1. **Mode: mixed, on a real cluster.**
   - **v5:**
     - (ii) replay of a production (Kimi) trace: Fig. 8 (scheduling, 8P+8D, 23,000 requests), Fig. 13 (10P+10D vs vLLM-[20M]), and Table 3 (overload at 2× speed).
     - (iv) public and synthetic datasets with Poisson arrivals: Figs. 11–12.
     - (iii) cache policy and capacity only: Table 1 is computed "assuming a single global" cache pool ([v5 §4.2, §8](https://arxiv.org/pdf/2407.00079v5)).
   - **FAST '25:**
     - (ii) replay of the conversation and tool&agent traces: Figs. 1, 5, 6, 10, 11.
     - (iv) a synthetic Poisson workload: Figs. 7, 8, 13, 15.
     - (iii) hit rate vs capacity in Fig. 9, whose simulation does "not account for factors such as prefill computation time" ([FAST '25 §5](https://www.usenix.org/system/files/fast25-qin.pdf)).
   - v5 says "all the experimental results reported in this paper are based on replayed traces of" real workloads. Its own Poisson runs contradict that (inference).
2. **Gaps between turns: recorded timestamps, open loop, no sessions.**
   - Requests are sent "by replaying these requests according to their actual arrival times".
   - The trace has no session IDs, although sampling "prioritized collecting requests within the same session"; sessions are visible only through shared hash_ids.
   - Overload is produced by time compression: "To simulate overload scenarios, we increased the replay" speed to 2× ([v5 §4, §8.1–8.2](https://arxiv.org/pdf/2407.00079v5)).
   - FAST '25's synthetic workload keeps turn order but dispatches "at a defined rate using a Poisson process" ([FAST '25 §5.2.1](https://www.usenix.org/system/files/fast25-qin.pdf)).
   - (Inference) Nothing ties a follow-up request to completion of the previous one.
3. **Model outputs: forced, with a dummy model.**
   - FAST '25: requests "are preemptively terminated once the model output reaches the predetermined length" ([FAST '25 §5.2.1](https://www.usenix.org/system/files/fast25-qin.pdf)).
   - v5 uses "a dummy model that follows the same architecture as LLaMA2-70B" and FAST '25 a dummy LLaMA3-70B ([v5 §4](https://arxiv.org/pdf/2407.00079v5); [FAST '25 §5.1](https://www.usenix.org/system/files/fast25-qin.pdf)).
4. **Data.**
   - **Released v5 trace:** "23,608 entries, with fields including timestamp, input_length, output_length, and hash_ids". It covers 1 h, uses 512-token prefix-chained block hashes, and has mean input 8,590 and output 182 tokens.
   - **End-to-end "Real Data":** "we constructed a dataset consisting of 23,000 real request traces" (mean 7,955 in / 194 out). By these statistics it is not the released trace (inference). The Poisson runs use ArXiv Summarization, L-Eval, and simulated 16k–128k prompts (Table 2) ([v5 §4, Table 2](https://arxiv.org/pdf/2407.00079v5)).
   - **FAST '25 Table 2:** conversation, 12,031 requests in 1 h; tool&agent, 23,608 requests in 1 h; synthetic, 3,993 requests built from ShareGPT, LEval and LooGLE ([FAST '25 §5.2.1, Table 2](https://www.usenix.org/system/files/fast25-qin.pdf)).
5. **Client–engine interface: not stated in either version.**
   - The Conductor returns HTTP 429 when an SLO cannot be met ([v5 §6.1](https://arxiv.org/pdf/2407.00079v5)).
   - (Inference) A replayer must synthesize token sequences consistent with the hash_ids, because Alg. 1 hashes the prompt tokens.
6. **Load.**
   - Trace replay at 1× and 2×.
   - Poisson sweeps (read from figure): ArXiv 0.5–4 req/s; L-Eval about 0.1–2; 128k prompts down to 0.02–0.14 req/s.
   - Open loop; concurrency not stated; no repeated runs ([v5 §8.1–8.2](https://arxiv.org/pdf/2407.00079v5)).
7. **Stack.**
   - **v5:** nodes of "8 NVIDIA-A800SXM4-80GB GPUs" with RDMA of up to 800 Gbps, up to 20 nodes (10P+10D). Tiers are HBM plus a CPU-DRAM pool; SSD appears only in the architecture ([v5 §8.1](https://arxiv.org/pdf/2407.00079v5)).
   - **FAST '25:** 16 nodes per system with 4×200 Gbps RDMA, and a local DRAM cache of "about 3 million tokens" per node. SSD is never exercised ([FAST '25 §5.1](https://www.usenix.org/system/files/fast25-qin.pdf)).
8. **Metrics.**
   - **v5:** "the 90th percentile (P90) values of TTFT and TBT as the ultimate metrics", normalized to SLOs and plotted against RPS; TTFT/TBT CDFs; number of rejected requests; hit ratio (simulated) ([v5 §2, §8](https://arxiv.org/pdf/2407.00079v5)).
   - **FAST '25:** "effective request capacity" under TTFT/TBT thresholds, prefill GPU time, and cache hit rate ([FAST '25 §5](https://www.usenix.org/system/files/fast25-qin.pdf)).
9. **Baselines.**
   - vLLM; FAST '25 names v0.5.1, with prefix caching and with chunked prefill each tested separately.
   - Random, load-balancing and cache-aware scheduling.
   - Rejection policies.
   - LRU, LFU and LengthAwareCache, in the simulation only ([v5 §6.2, §8, Table 1](https://arxiv.org/pdf/2407.00079v5); [FAST '25 §5](https://www.usenix.org/system/files/fast25-qin.pdf)).
10. **Released artifacts.** The traces and code are at github.com/kvcache-ai/Mooncake: FAST25-release/traces holds the conversation, toolagent and synthetic traces, and arxiv-trace/mooncake_trace.jsonl is the v5 trace. Q3 has the field-level verification ([repo](https://github.com/kvcache-ai/Mooncake)).
11. **Threats to validity.**
    - **Acknowledged by the authors:**
      - The trace is a subset.
      - "Theoretically, up to only 55% of the KVCache can be reused in our current workloads".
      - The model is a dummy, and the FAST '25 Fig. 9 simulation ignores compute time ([v5 §4, §8](https://arxiv.org/pdf/2407.00079v5); [FAST '25 §5](https://www.usenix.org/system/files/fast25-qin.pdf)).
    - **Inferred:**
      - With no session IDs, gaps are fixed open-loop timestamps that cannot react to slower responses.
      - The policy and capacity conclusions are latency-free simulations.
      - SSD is never measured, and no variance is reported.

#### LMCache: arXiv 2510.09665v2, 5 Dec 2025
Venue: not stated. The paper calls itself a report.

1. **Mode: mostly (iv) on real vLLM**, plus one SGLang figure.
   - Multi-round QA ("MRQA", Fig. 8).
   - TriviaQA with Poisson arrivals (Fig. 11).
   - A random 8K-in/200-out prefill–decode workload (Fig. 12).
   - Company F and G production traces, time-compressed (Figs. 9–10), sit between (ii) and (iv).
   - Opt-in telemetry (Figs. 1, 3–4) and customer reports (§9) are not experiments.
   - No live agents and no simulation ([v2 §2.2, §8, §9](https://arxiv.org/pdf/2510.09665v2)).
2. **Gaps between turns: not stated.**
   - MRQA says only that a "session begins with 40 users, and additional users join according to a specified arrival rate (QPS)." Rounds and think time are not stated.
   - The traces are compressed: "we stretch the original trace which lasts for several days so that the workload we run completes within one hour." How the trace is rescaled to each QPS point is not stated ([v2 §8.2–8.3](https://arxiv.org/pdf/2510.09665v2)).
3. **Model outputs.**
   - MRQA: "LLM output is a short answer of 100 tokens at max."
   - Prefill–decode workload: 200 output tokens.
   - Company traces: the companies' "distribution of input and output tokens".
   - ignore_eos is not stated ([v2 §8.2, §8.3, §8.5](https://arxiv.org/pdf/2510.09665v2)).
4. **Data.**
   - MRQA: each query has 10K tokens (20K for Llama-3.1-8B), made of a document of about 12 pages plus a short question.
   - LongBench TriviaQA and the vLLM random dataset.
   - The company traces last several days; their sizes are not stated.
   - Models range from Llama-3.1-8B to Qwen3-Coder-480B ([v2 §8.1–8.5](https://arxiv.org/pdf/2510.09665v2)).
5. **Client–engine interface.**
   - §8.4–8.5 use "the official vLLM benchmarking scripts"; an OpenAI-compatible HTTP API is inferred.
   - The MRQA client is not named.
   - No session IDs or hints cross the boundary, and the controller APIs are not exercised ([v2 §8.4–8.5](https://arxiv.org/pdf/2510.09665v2)).
6. **Load.**
   - Open-loop QPS sweeps (read from figure): about 0.5–8 for MRQA and 0.25–3 for TriviaQA.
   - 40 initial users.
   - Run length is not stated, except the 1 h trace runs ([v2 §8.2–8.4](https://arxiv.org/pdf/2510.09665v2)).
7. **Stack.** LMCache v0.3.6 on vLLM, on an "8×H100 server provided by GMI Cloud". Tiers per experiment:
   - Figs. 8–10: GPU plus local CPU DRAM, capped at 500 GB ("CPU memory LMCache can offload KV cache to 500 GB.").
   - Fig. 11: a remote CPU-memory server "with a bandwidth of 15 Gbps".
   - Figs. 12 and 14: GPU to GPU over NVLink.
   - Fig. 16: SGLang CPU offloading on 2×H100.

   No disk, SSD or object-store backend appears in any §8 experiment ([v2 §8.1–8.8](https://arxiv.org/pdf/2510.09665v2)).
8. **Metrics.**
   - Mean TTFT and mean ITL against QPS; throughput at equal TTFT.
   - TTFT/ITL CDFs, latency breakdowns, and loading bandwidth.
   - No hit ratio is reported in §8 ([v2 §8](https://arxiv.org/pdf/2510.09665v2)).
9. **Baselines.**
   - "Basic vLLM: vLLM v0.10.2 which enables prefix" caching, GPU only.
   - vLLM v0.11.0 with its own CPU offloading.
   - Two black-box commercial endpoints.
   - vLLM native prefill–decode disaggregation via NIXL.
   - SGLang with and without its native CPU offloading ([v2 §8.1, §8.4–8.8](https://arxiv.org/pdf/2510.09665v2)).
10. **Released artifacts.** The code is at github.com/LMCache/LMCache; the README exists and cites the paper. There are no paper-reproduction configs, and the traces are confidential ([v2 §10](https://arxiv.org/pdf/2510.09665v2); [repo](https://github.com/LMCache/LMCache)).
11. **Threats to validity.**
    - **Acknowledged by the authors:**
      - The commercial comparison is black-box.
      - Substitute models were used for the proprietary traces.
      - Loading can be slower than prefill for short contexts ([v2 §8.3–8.4, §8.7](https://arxiv.org/pdf/2510.09665v2)).
    - **Inferred:**
      - No disk tier.
      - MRQA rounds and gaps are unspecified, so retention may never be exercised.
      - The main baseline is GPU-only, so the gains largely reflect added capacity.
      - The abstract's "up to 15×" is above the body's "2.3–14×".
      - No error bars.

#### Strata: arXiv 2508.18572v1, 26 Aug 2025
Venue: not stated. TempoKV's reference list cites it as OSDI, which is secondary ([TempoKV HTML](https://arxiv.org/html/2609.35065)).

1. **Mode: (iv) on real engines, on one node.**
   - LooGLE, NarrativeQA, ReviewMT and ShareGPT with Poisson arrivals: the main Fig. 8 and the ablations in Figs. 9–11 and 13.
   - Microbenchmarks: Figs. 3, 5 and 12.
   - "Strata-Oracle" is a simulation that "simulates the TTFT achieved by a system that had infinite" CPU–GPU bandwidth.
   - The production deployment is claimed without data ([v1 §1, §5, §5.4](https://arxiv.org/pdf/2508.18572v1)).
2. **Gaps between turns.**
   - "we simulate query arrivals using a Poisson" process because timestamps are unavailable.
   - ReviewMT and ShareGPT: "we preserve dependencies across conversation rounds".
   - ShareGPT: "we insert a 60-second “thinking time” between an LLM’s response and the user’s subsequent query for ShareGPT." ReviewMT's gap is not stated.
   - Document-QA queries are "randomly sampled" with no sessions ([v1 §5.1](https://arxiv.org/pdf/2508.18572v1)).
3. **Model outputs: not stated.** Dataset mean outputs range from 13 to 261 tokens ([v1 Table 1](https://arxiv.org/pdf/2508.18572v1)).
4. **Data.** Reuse comes from re-queried documents or multi-turn history ([v1 §5.1, Table 1](https://arxiv.org/pdf/2508.18572v1)):

   | Dataset | Contexts | Queries | Average input tokens |
   |---|---|---|---|
   | LooGLE (Wikipedia portion) | 105 | 2,410 | 21,613 |
   | NarrativeQA | 50 | 1,461 | 54,797 |
   | ReviewMT (agents debating as reviewers) | 100 | 1,092 | 17,708 |
   | ShareGPT | — | 200,869 | 680.9 |

5. **Client–engine interface: not stated.** Scheduling uses engine-side state only, with no client hints (inference) ([v1 §4.3](https://arxiv.org/pdf/2508.18572v1)).
6. **Load.**
   - Poisson rate sweeps; "we cap the maximum number of in-flight queries at 128".
   - ShareGPT GPU memory is restricted to about 500K tokens.
   - Run length is not stated ([v1 §5.1](https://arxiv.org/pdf/2508.18572v1)).
7. **Stack.**
   - 8×H200 with 1.6 TB DRAM and PCIe 5.0 ×16 per GPU, plus a GH200.
   - Llama-3.1-8B, Qwen2.5-14B-1M, and Llama-3.1-70B (TP4).
   - Host tier of 1 TB pinned DRAM (400 GB on GH200).
   - "Disk storage is not used in all benchmarks due to limited support in baseline systems." Disk appears only in the Fig. 12 microbenchmark ([v1 §5.1, §5.3.4](https://arxiv.org/pdf/2508.18572v1)).
8. **Metrics.** Average TTFT against output-token throughput, P90 TTFT, hit rate, I/O-stall share, and bandwidth ([v1 §5.2–5.4](https://arxiv.org/pdf/2508.18572v1)).
9. **Baselines.**
   - vLLM v0.8.5, with and without LMCache v0.2.1.
   - TensorRT-LLM v0.17.0, with and without CPU offload.
   - SGLang v0.4.5.
   - An SGLang-HiCache built by the authors: "To enable a direct comparison to Strata, we implemented SGLang-HiCache" ([v1 §5.1](https://arxiv.org/pdf/2508.18572v1)).
10. **Released artifacts: none stated** ([v1](https://arxiv.org/pdf/2508.18572v1)).
11. **Threats to validity.**
    - **Acknowledged by the authors:** Poisson arrivals for lack of timestamps; no disk tier; the in-flight cap; scope limited to single instances and prefill-dominated workloads ([v1 §5.1, §6](https://arxiv.org/pdf/2508.18572v1)).
    - **Inferred:**
      - Host DRAM probably never binds. The document sets amount to roughly 0.3–0.9 TB of KV against a 1 TB pool, and the about 95% hit rate is close to first-touch misses only.
      - So the results test load bandwidth and batching, not retention.
      - The ShareGPT gap is fixed at 60 s.
      - The HiCache baseline is author-built.

#### Marconi: arXiv 2411.19379v3, 10 Apr 2025
Venue: MLSys 2025, as printed in the paper ([v3](https://arxiv.org/pdf/2411.19379v3)).

1. **Mode: (iii) trace-driven cache simulation for every hit-rate result.**
   - The word "simulat" does not appear in the paper. The artifact shows it: "On our 32-core CPU, a single {LMSys, ShareGPT, SWEBench} trace takes an average of about {30s, 5s, 5-10mins} to finish" ([artifact_evaluation.md](https://github.com/ruipeterpan/marconi)).
   - The TTFT results (Figs. 9 and 10b) say "we use Jamba-1.5-Mini", served "using the vLLM implementation on four A100-40GB GPUs" ([v3 §5.1](https://arxiv.org/pdf/2411.19379v3)).
   - The Zenodo artifact ships ttft.py and a 7.7 kB profiled pickle ([Zenodo](https://zenodo.org/records/14970139)). (Inference) The TTFTs come from a latency lookup applied to simulated hits.
2. **Gaps between turns: synthetic, means only.**
   - "We vary the inter-session arrival time and inter-request arrival time to account for environment response time (e.g., human typing, interactions with the coding IDE) and queuing delay in the inference engine."
   - Fig. 13 varies the "average number of request (chat) sessions per second increases from 0.5 to 2" and the "average response time between requests in a session increases from 5 s to 10 s" ([v3 §5.1, §5.4](https://arxiv.org/pdf/2411.19379v3)).
   - The distribution family and the main-sweep values are not stated.
   - Recorded tool or environment durations are not used, even for the SWE-bench trajectories (inference).
3. **Model outputs: forced.** The traces are recorded token IDs (Llama-2-7b tokenizer). "We do not evaluate downstream metrics (e.g., F1 score) as prefix reusing is exact" ([v3 §5.1, App. B](https://arxiv.org/pdf/2411.19379v3)).
4. **Data.**
   - LMSys-Chat-1M, ShareGPT, and SWE-bench agent trajectories ("We also include an agentic workload: SWE-Agent ... on SWE-Bench").
   - The artifact's raw-data link points instead to the SWE-bench/experiments directory evaluation/verified/20240820_honeycomb ([artifact_evaluation.md](https://github.com/ruipeterpan/marconi)).
   - Session and request counts are not stated ([v3 §5.1](https://arxiv.org/pdf/2411.19379v3)).
5. **Client–engine interface: none.** The simulator reads pre-tokenized trace files (inference from the artifact).
6. **Load.**
   - Open-loop synthetic arrivals: 0.5, 1 and 2 sessions/s, and mean gaps of 5, 7.5 and 10 s.
   - "We vary the cache size from 60 GB (high contention) to 140 GB (low contention)" ([v3 §5.4, Figs. 11 and 13](https://arxiv.org/pdf/2411.19379v3)).
7. **Stack.**
   - A Python simulator modeling a 7B hybrid (4 attention, 24 SSM and 28 MLP layers) in FP16.
   - "Experiments were run on a p4d.24xlarge AWS instance with eight A100-40GB GPUs" and 1,152 GB of DRAM.
   - One cache pool with no tiers and no transfer cost (inference) ([v3 §5.1](https://arxiv.org/pdf/2411.19379v3)).
8. **Metrics.** Token hit rate, TTFT at P5/P50/P95, and FLOP saved ([v3 §5.1–5.3](https://arxiv.org/pdf/2411.19379v3)).
9. **Baselines.**
   - Vanilla, with no prefix caching.
   - vLLM+: "We use a token block size of 32, the largest size that vLLM supports".
   - SGLang+: Marconi's admission policy with LRU eviction.

   All are re-implemented inside the simulator (inference) ([v3 §5.1](https://arxiv.org/pdf/2411.19379v3)).
10. **Released artifacts.**
    - GitHub ([repo](https://github.com/ruipeterpan/marconi)) and Zenodo ([record 14970139](https://zenodo.org/records/14970139), traces.tar.gz 718.2 MB), CC BY-NC.
    - The artifact doc says the Fig. 10–13 scripts "either do not exactly reproduce the figures in the paper" or contain hand-picked numbers.
11. **Threats to validity.**
    - **Acknowledged by the authors:** only Mamba-style SSM hybrids; P5 TTFT 6.3% worse than SGLang+; the artifact caveat above ([v3 §5.3, §6](https://arxiv.org/pdf/2411.19379v3)).
    - **Inferred:**
      - Marconi never runs inside an engine.
      - Timing is synthetic, with unreported distributions.
      - Hit rates use the 7B hybrid, while TTFT uses Jamba.
      - Single tier.

#### KVCache Cache in the Wild (Alibaba): arXiv 2506.02634v5, 14 Feb 2026
Venue: the acknowledgments thank "ATC reviewers and our shepherd", and the trace README cites USENIX ATC '25 ([v5](https://arxiv.org/pdf/2506.02634v5)).

1. **Mode.**
   - **§3 characterization:** offline trace analysis. Its cache-dependent figures are (iii) infinite-cache or oracle simulations (Figs. 4, 6–7, 16–18, 22).
   - **§4.3 policy evaluation:** (ii) trace replay on a real engine, Figs. 25–28: "we implement a CPU-GPU KV$ cache on" vLLM, on 8×A800 ([v5 §3, §4.3](https://arxiv.org/pdf/2506.02634v5)).
2. **Gaps between turns: recorded, with explicit sessions, then downscaled.**
   - "By traversing the parent IDs, we can identify all" requests belonging to the same session.
   - The evaluation downscales timestamps with a method "that preserves the temporal pattern of the trace [50]" (TraceSplitter).
   - The scaling factor, and whether a turn waits for the previous response, are not stated ([v5 §3.1, §4.3](https://arxiv.org/pdf/2506.02634v5)).
3. **Model outputs: forced length, replaced content.**
   - "we constrain the model using the trace’s output length."
   - "we replace the modelgenerated output with the constructed tokens to preserve the cache patterns of multi-turn requests" ([v5 §4.3](https://arxiv.org/pdf/2506.02634v5)).
4. **Data.**
   - Aliyun Bailian traces from one cluster. Trace A is to-C traffic (Text 78%, File 4%, Multimodal 3%, Search 15%). Trace B is to-B API traffic, 0.08% multi-turn.
   - Fields: timestamp, chat ID, parent chat ID, user ID, type, token counts, and hashed tokens. The paper "employs SipHash [8] to generate unique hashes for every" four consecutive tokens.
   - Absolute sizes are not stated; counts are normalized ([v5 §3.1–3.2](https://arxiv.org/pdf/2506.02634v5)).
5. **Client–engine interface.** Inputs are synthesized per anonymized block ("we randomly generate a token list"). The request category (type plus turn) is passed to the policy. The driver is not stated ([v5 §4.3](https://arxiv.org/pdf/2506.02634v5)).
6. **Load.** Downscaled trace timestamps; factor, QPS and duration are not stated ([v5 §4.3](https://arxiv.org/pdf/2506.02634v5)).
7. **Stack.**
   - One 8×A800-80GB server with PCIe Gen4, running vLLM (version not stated).
   - Qwen2-7B and Llama2-13B on 1 GPU; Llama3-70B on 4 GPUs.
   - Tiers: HBM plus DRAM at 1×, 2× or 4× the HBM KV space; no SSD ([v5 §4.3](https://arxiv.org/pdf/2506.02634v5)).
8. **Metrics.** Hit rate; QTTFT (TTFT including queueing); and policy overhead of 79 µs per eviction ([v5 §4.2–4.3](https://arxiv.org/pdf/2506.02634v5)).
9. **Baselines.**
   - LRU, FIFO, LFU and S3-FIFO; ablations GDFS, +Distribution and WA (+Life).
   - "our policy WA achieves 8.1–23.9 % higher hit rate" and "WA achieves 28.3–41.9% QTTFT reduction" ([v5 §4.3](https://arxiv.org/pdf/2506.02634v5)).
10. **Released artifacts.**
    - "A sample of our analyzed trace is available at" github.com/alibaba-edu/qwen-bailian-usagetraces-anon; Q3 has the field-level check.
    - The reference vLLM PR #22236 was closed without merging, according to the PR page ([PR](https://github.com/vllm-project/vllm/pull/22236)).
11. **Threats to validity.**
    - **Acknowledged by the authors:** one week of traces; a single instance; traces scaled down because of GPU limits; no raw content; Trace B lacks request types ([v5 §4.3, §5](https://arxiv.org/pdf/2506.02634v5)).
    - **Inferred:**
      - Measured hit rates (about 0.1–0.4, read from figure) sit far below the ideal 62% and 54%, and the effect of downscaling is not analyzed.
      - No agentic workload.

### Inferences
- **Recorded timestamps on a real engine** appear in only two papers. Mooncake replays at 1× or 2× with no sessions and open loop. Alibaba downscales with TraceSplitter and keeps sessions through parent IDs. Neither has tool-call structure. Mooncake's "tool&agent" trace exists, but without session or tool fields.
- **Papers with sessions but no timestamps invent the gap:**
  - Pensieve: exponential, mean 60 s.
  - Strata: fixed 60 s.
  - Marconi: synthetic means of 5–10 s.
  - CachedAttention and LMCache: never stated.

  Because the gap decides whether a prefix is still in DRAM, on SSD, or gone, every reported tier hit rate is conditional on an assumption that is unstated or uncited.
- **Look-ahead prefetch is evaluated only on already-queued requests.** CachedAttention looks ahead over its job queue. LMCache and Strata describe queue-time prefetch but never isolate it. None evaluates a read that starts before the request arrives.
- **No client hints.** None of the seven passes hints or session IDs from the client to the engine.
- **SSD.** Only CachedAttention measures an end-to-end SSD tier, and its device is unspecified. Strata says disk is unused "due to limited support in baseline systems". So a three-tier GPU/DRAM/SSD evaluation has little direct precedent among these seven.

### Gaps
- **Unstated parameters.** Arrival-to-turn mapping (CachedAttention, Pensieve); gap and round count (LMCache MRQA); Marconi's arrival distributions (in its generate_trace.py, not opened); Alibaba's downscaling factor; Mooncake's replay harness.
- **Versions not read.** I used the arXiv versions above, not the ATC '24 (CachedAttention) or ATC '25 (Alibaba) camera-ready PDFs. For Mooncake, v5 and FAST '25 contain different evaluations; both are summarized.
- **Unverifiable claims.** LMCache's production numbers (Companies C, F, G) are customer reports or analyses.

## Q2. How do the seven papers compare? (comparison table)

### Takeaway
Across the checklist, these papers share a template:
- one server or a small cluster;
- open-loop Poisson or timestamp arrivals;
- chat or QA content, or production traces stripped of content;
- outputs that are unstated or forced to recorded lengths;
- TTFT and hit rate as primary metrics;
- baselines that are either stateless engines or LRU-family policies inside the authors' own system.

The cells that matter most for an agentic tiering paper are almost all empty, unstated or synthetic: recorded tool gaps, closed-loop session dependence, an SSD tier under load, and client hints.

### Cited Findings
Each cell summarizes the Q1 bullets for that paper, which carry the section, figure and table citations.

| Paper (version) | Mode | Gaps between turns | Outputs | Data | Interface | Load | Stack / tiers | Metrics | Baselines | Artifacts |
|---|---|---|---|---|---|---|---|---|---|---|
| CachedAttention ([2403.19708v3](https://arxiv.org/pdf/2403.19708v3)) | (iv) on own PyTorch engine; cost by price model | Not stated; Poisson session arrivals at 1.0/s; turns wait in a job queue | Not stated | ShareGPT: 9K sessions, 5.75 turns, about 52K turns (10K warm-up) | Not stated; in-process queue; internal hints | 1.0 sessions/s (0.5–2.0 sweep); batch 24 | 4×A100-80GB, 128 GB DRAM, 10 TB SSD (type unstated); HBM+DRAM+SSD | Hit rate (DRAM/disk), TTFT, prefill tok/s, GPU hours, $ | RE (recompute), LRU, FIFO, OF, HBM-only, HBM+DRAM | None |
| Pensieve ([2312.05516v3](https://arxiv.org/pdf/2312.05516v3)) | (iv) on real engines | Exponential think time, mean 60 s (30–600 s sweep); next turn after previous response | Not stated (EOS or max length) | ShareGPT 48,159 conversations; UltraChat 1.47M; contexts ≤16,384 tokens | Stateful server; stateless baselines; own vLLM driver; FCFS | Poisson rate sweep; achieved ≤ about 4–10 req/s | 1 or 4×A100-80GB, 220 GB RAM per GPU, 40 GB KV per GPU; GPU+DRAM; no SSD | Throughput vs normalized latency (p90 vs mean conflict); CPU hit rate | vLLM 0.2.0, TensorRT-LLM 0.12.0, GPU-only variant, LRU | None |
| Mooncake ([v5](https://arxiv.org/pdf/2407.00079v5); [FAST '25](https://www.usenix.org/system/files/fast25-qin.pdf)) | (ii) replay and (iv) Poisson on real cluster; (iii) for cache policy and capacity | Recorded timestamps; no session IDs; open loop; 2× compression for overload; FAST synthetic uses Poisson turns | Forced (terminated at recorded length); dummy LLaMA2/3-70B | Kimi traces: 23,608 req/1 h (released) and 23,000 "Real Data"; FAST: conversation 12,031, tool&agent 23,608, synthetic 3,993 | Not stated | 1× and 2× replay; RPS sweeps | Up to 20 nodes of 8×A800 with RDMA; HBM + DRAM pool; SSD not measured | P90 TTFT/TBT vs SLO, effective capacity, rejections, hit ratio (sim.) | vLLM (FAST: 0.5.1 ± prefix caching); scheduler and rejection variants; LRU/LFU/LengthAware (sim.) | Traces (timestamp, lengths, 512-token hash_ids) and code |
| LMCache ([2510.09665v2](https://arxiv.org/pdf/2510.09665v2)) | (iv) on vLLM; compressed company traces between (ii) and (iv); telemetry | Not stated (MRQA); traces compressed from days to 1 h | MRQA ≤100 tokens; 200 for prefill–decode; trace length distributions | MRQA 10K/20K-token document + question; TriviaQA; random 8K/200; company F/G traces | vLLM benchmark scripts; no session IDs or hints | QPS sweeps (about 0.25–8); 40 initial users | vLLM + LMCache 0.3.6 on H100s; local DRAM ≤500 GB; remote DRAM over 15 Gbps; NVLink; no SSD | Mean TTFT/ITL; throughput at equal TTFT | vLLM 0.10.2 (GPU), 0.11.0 (offload), 2 commercial endpoints, vLLM-NIXL, SGLang ± offload | Code; traces confidential |
| Strata ([2508.18572v1](https://arxiv.org/pdf/2508.18572v1)) | (iv) on real engines, 1 node; oracle is (iii) | None for document QA; ShareGPT fixed 60 s; ReviewMT order kept, gap not stated | Not stated | LooGLE 105 contexts/2,410 queries; NarrativeQA 50/1,461; ReviewMT 100/1,092; ShareGPT 200,869 queries | Not stated | Poisson sweeps; ≤128 in flight | 8×H200 + 1.6 TB DRAM (1 TB pinned); GH200; GPU+DRAM; disk only in a microbenchmark | Avg TTFT vs output throughput, P90 TTFT, hit rate, stall % | vLLM 0.8.5 ± LMCache 0.2.1; TensorRT-LLM 0.17 ± offload; SGLang 0.4.5 ± author-built HiCache | None |
| Marconi ([2411.19379v3](https://arxiv.org/pdf/2411.19379v3)) | (iii) CPU trace simulation (shown by the artifact); TTFT from a profiled table (inference) | Synthetic means only: 0.5–2 sessions/s, 5–10 s in-session; no recorded tool time | Forced: recorded tokens | LMSys-Chat-1M, ShareGPT, SWE-bench trajectories; counts not stated | None (simulator) | Open loop; cache 60–140 GB | Python simulator; 7B hybrid; one pool, no tiers | Token hit rate, TTFT P5/P50/P95, FLOP saved | Vanilla, vLLM+, SGLang+ (all simulated) | GitHub + Zenodo with traces (CC BY-NC) |
| Alibaba ([2506.02634v5](https://arxiv.org/pdf/2506.02634v5)) | (iii) characterization; (ii) downscaled replay on vLLM | Recorded; sessions via parent IDs; downscaled (TraceSplitter), factor not stated | Forced length; content replaced by constructed tokens | Aliyun Bailian Trace A (to-C) and Trace B (to-B); one day each; normalized sizes | Not stated; synthetic tokens per hash block; category tag | Downscaled; factor and QPS not stated | 8×A800; HBM + DRAM (1–4× HBM); no SSD | Hit rate, QTTFT, policy overhead | LRU, FIFO, LFU, S3-FIFO, GDFS | 2 h trace sample; unmerged vLLM PR |

### Inferences
- Measured against these seven papers, an agentic tiering evaluation would add the following:
  - recorded per-call tool durations and human waits, not Poisson or fixed think times;
  - closed-loop session dependence: the next call is released only after the previous response, plus the recorded gap;
  - an SSD tier that the policy actually exercises;
  - client hints crossing the API boundary.

  No paper here combines these.
- **What a tiering paper needs from these precedents** (inference):
  - **Forced output lengths are standard** whenever traces are replayed (Mooncake FAST '25, Alibaba). Forcing keeps the KV footprint and turn timing comparable across policies.
  - **Policy questions are often settled by simulation:** Mooncake Table 1, FAST '25 Fig. 9, Marconi, and Alibaba §3. A paper that reports both engine runs and a calibrated simulator follows established practice. It should say which claim rests on which mode, which Mooncake v5 does not do cleanly.

### Gaps
- Several cells read "not stated" because the papers are silent. They could only be filled from code, and none of the code except Marconi's and LMCache's is released.

## Q3. Which traces of agentic or multi-turn sessions record per-turn timing, and can they be replayed faithfully?

### Takeaway
Since mid-2026, public traces record per-call tool timing for real agent sessions. Each one is missing something.

**Real users, tool timing recorded:**
- **FreeInference Agentic Trace** (CC BY 4.0) is the closest fit. It has 12,002 real sessions from 14 agent harnesses. Each request has gateway start, end and TTFT times and a user/tool trigger. Each tool call has a name, category, shell programs and its gap to the next request. Prompts are chained 16-token block IDs. It has no text, and block IDs are local to each week.
- **GitHub Copilot Coding Agent Traces 2026** (CC-BY) has 301K production sessions with per-call and per-tool durations. It has no text or hashes, so prefixes must be synthesized.
- **TraceLab v0.0.2** (CC BY 4.0) has 8,058 sessions with per-event timestamps, tool wall latency and bash command words. It has no content, and it ships a closed-loop replayer for vLLM and SGLang.
- **SWE-chat** (ODC-BY, gated) has 17,968 real sessions with redacted text and per-entry timestamps.

**Benchmark runs with full text and tool spans:** Exgentic, mini-SWE-agent v2 SWE-bench submissions, LMCache agentic traces and Terminal-Bench. Their timing comes from benchmark harnesses.

**Request-level gaps with hashes but no tool/user split:** SemiAnalysis Claude Code traces, Alibaba Bailian, and the FreeInference/Chutes block traces.

**No usable inter-turn gaps:**
- Mooncake, which has no sessions.
- Azure LLM traces.
- LMSYS-Chat-1M and ShareGPT.
- The popular SWE-bench trajectory sets (nebius, SWE-Gym, SWE-smith), which have no timing.

**Not released:** the M365 traces (AgenticTraceSim was not found) and the 11.7M-request agent-platform trace.

### Cited Findings

#### A. Production agentic studies
- **GitHub Copilot, "Agentic Coding in the Wild": arXiv 2608.00101v1, 30 Jul 2026.**
  - **Source:** anonymized telemetry from the Copilot coding agent in Visual Studio and VS Code, first week of June 2026.
  - **Fields recorded:** "timestamps, durations, input/output token counts (without including reasoning tokens), model names, tool names, and success or failure status" for every LLM call and tool invocation.
  - **Not collected:** "We do not collect the prompt text or file content, model outputs, source code, tool arguments, or any user-identifiable information" ([v1 §3.1](https://arxiv.org/pdf/2608.00101v1)).
  - **Scale (Table 3):** 13.5M sessions, 95.1M turns, 3.2M users, 760.5M LLM calls and 774.7M tool calls, 27+ models and 45+ tools. Prompt tokens are 44.9T and completion tokens 39.3B. The abstract and introduction instead say "95T tokens" and "13M sessions", an internal inconsistency ([v1 abstract, §1, Table 3](https://arxiv.org/pdf/2608.00101v1)).
  - **Hierarchy:** Session > User-Turn > Step. "the last step in a turn is always an LLM invocation" ([v1 §3.2](https://arxiv.org/pdf/2608.00101v1)).
  - **Availability: released after v1.** v1 said "We plan to release sanitized traces from our characterization data to the public soon" ([v1 §12](https://arxiv.org/pdf/2608.00101v1)). A sample is now public as "GitHub Copilot Coding Agent Traces 2026", a GitHub release of the Azure Public Dataset.
    - **Release and license:** published 2026-10-01; CC-BY; seven per-day .tar.gz files totaling 3,218,886,136 bytes ([release API](https://api.github.com/repos/Azure/AzurePublicDataset/releases/tags/ghcp-coding-agent-2026); [doc](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md)).
    - **Scope:** "uniformly-sampled agentic coding sessions from GitHub Copilot non-enterprise users, covering June 1–7, 2026". 301K sessions, 1.2M user turns, 9.3M LLM calls, 8.7M tool calls, 37 anonymized models. Totals are 631.4B prompt tokens, 4.91B completion tokens and 540.95B cached prompt tokens. "This data is the sample data used in the arXiv paper" ([doc](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md)).
    - **Schema per session** ([schema.json](https://github.com/Azure/AzurePublicDataset/blob/master/analysis/GitHubCopilotCodingAgents/schema.json)):
      - session_id, then turns with a turn_id.
      - Each llm_call has message_id, timestamp, duration_ms ("end-to-end LLM call duration"), initiator_type ('user' or 'agent'), an anonymized model, and result/status_code.
      - Each llm_call also has tokens (total, prompt, completion, cached, cached_pct) and message_metadata, one entry per prompt segment with type, role, sequenceId and token_len.
      - Each tool_batch has timestamp, duration_ms ("batch wall-clock duration"), auto_continue, result, and request_id ("matches with message_id from LLM calls").
      - Each tool_batch's function_calls carry name ("custom tools are anonymized"), status, and duration_ms ("individual tool duration").
    - **What it lacks:** "This release is metadata-only". It has no prompts, responses, tool outputs, file paths, or identifiers, and no prefix hashes ([doc](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md)).
    - **Timestamp semantics (inference from a sample, not from the docs).** The catalog check examined 381 sessions (3 MB) and concluded that LLM-call and tool-batch timestamps are end times, not start times. Read as start times, 34.6% of consecutive LLM calls would overlap; read as end times, none do. The schema does not say which they are, so verify before replaying.
  - **How its own results are produced:** an offline predictor trained on 150K sessions from one week and "50K sessions from the following week for evaluation" ([v1 §9.2](https://arxiv.org/pdf/2608.00101v1)). Its cache hit rates are the production service's own: a median prompt of 68K tokens, of which 63K are cached ([v1 §6](https://arxiv.org/pdf/2608.00101v1)).
- **Microsoft M365, "Understanding KV Residency in Production-Scale Agentic LLM Workloads": arXiv 2610.22863v1, 16 Oct 2026.**
  - **Traces (Table 2):**

    | Trace | Window | Volume | Sessions |
    |---|---|---|---|
    | Research | 8 days | 17,008,715 calls | 691,293 |
    | Assist | 8 days | 217,106 calls | 23,538 |
    | Code | 34-day event span | 1,119,432 events | 8,157 |
    | Chat | 24 hours | 104,013,216 calls | 13,184,080 |

    They were collected between late June and early August 2026 ([v1 §2.2, Table 2](https://arxiv.org/pdf/2610.22863v1)).
  - **Fields per call:** "context tokens xi , cached tokens ci , output tokens oi , timestamp ti and model identity; additional timing and tool-activity fields are available for a subset of traces." Research, Assist and Chat also have serving region and deployment. Code logs tool and session events with a session-cumulative cache_read field. No content ([v1 §2.2](https://arxiv.org/pdf/2610.22863v1)).
  - **Availability:** not released. "Because the production telemetry cannot be released without risking client confidentiality, we plan to release AgenticTraceSim, a parametric trace generator, together with a residency replay" ([v1 §1 contribution 5, §6](https://arxiv.org/pdf/2610.22863v1)).
  - On 2026-11-17 AgenticTraceSim was not in the Azure Public Dataset: not in the README, the repo tree or the releases ([README](https://github.com/Azure/AzurePublicDataset/blob/master/README.md); [releases](https://github.com/Azure/AzurePublicDataset/releases)).
  - **How its own results are produced:** a "trace-driven residency replay" that "processes an ordered call stream C under a pooled memory budget B ... in timestamp order". This is a simulation:
    - KV size uses the Llama-2-70B geometry as a stand-in (β = 327,680 bytes/token).
    - "The replay charges no promotion latency, dequantization, or transfer cost" ([v1 §2.3, §5, §5.3](https://arxiv.org/pdf/2610.22863v1)).
- **Agent-serving platform with more than 10k GPUs: arXiv 2609.34432v1, 28 Sep 2026.**
  - A two-week trace of 11.69M requests ([v1](https://arxiv.org/pdf/2609.34432v1)).
  - "We plan to release sanitized traces from our characterization data to the public soon" ([v1 conclusion](https://arxiv.org/pdf/2609.34432v1)). Not released.

#### B. Released real-user agentic traces with tool timing
- **FreeInference Agentic Trace** (Hugging Face harvardMadsys/freeinference_agentic_trace; CC BY 4.0; not gated; created 2026-10-05) ([card](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace); [HF API](https://huggingface.co/api/datasets/harvardMadsys/freeinference_agentic_trace)).
  - **Scope:** "16 weeks (2026-05-17 to 2026-09-05) of coding and assistant agents interacting with LLMs through the FreeInference gateway: **12,002 agent sessions** from 267 accounts and 14 agent harnesses, with **1,186,582 LLM requests** and **1,213,347 tool calls**." "No message text is released."
  - **Format:** one session per JSONL line. Subagent sessions are nested under the tool call that launched them.
  - **Request fields:**
    - Position and cause: `step`, `turn_index`, `step_trigger` (user or tool), `link_reason`.
    - Timing: `start_ms`, `end_ms`, `ttft_ms` ("Request timing at the gateway"; epoch milliseconds).
    - Model: `model` and an anonymized `provider`.
    - Tokens: `provider_tokens` {prompt, completion, cached}, plus `role_tokens` (system, user, assistant, tool, tool_definitions).
    - Prompt shape: `tool_definition_names` and `mutation` (e.g. "append").
    - Content: `block_ids`, "chained 16-token blocks, tokenized with tiktoken".
  - **Tool-call fields:**
    - `name` and `category`.
    - `programs`: "Shell calls additionally record the programs found in the command".
    - `arguments`, with "sensitive values ... replaced with typed placeholders".
    - `outcome`, `result_tokens`, and `latency_ms`, which the card's schema comment describes as "Tool outcome and the gap until the next LLM request".
    - `subagent`.
  - **Card caveats:**
    - Inputs use `o200k_base`, not the provider's tokenizer.
    - "Block IDs are local to each week."
    - "Sessions are reconstructed by linking related requests through repeated tool-call IDs and assistant messages" and do not cross week boundaries.
    - Rare tool names are replaced with `other`.
  - **Code and paper:** the card names the paper "From Requests to Sessions: A Large-Scale Characterization of Human-Driven Agentic Workloads". Code is at [GitHub](https://github.com/HarvardMadSys/freeinference_agentic_trace).
  - **Catalog-check observations:**
    - In a one-day sample, all 182 single-tool steps satisfied `latency_ms` = next `start_ms` − this `end_ms`. So the tool gap is measured at the gateway and includes client and network overhead (inference).
    - The processed tables hold 1,222,199 request rows, not the 1,186,582 the card states; the difference is unexplained.
- **GitHub Copilot Coding Agent Traces 2026:** see A. Per-call and per-tool durations; no text or hashes.
- **SWE-chat** (Hugging Face SALT-NLP/SWE-chat; ODC-BY; gated, with automatic approval after accepting conditions; last modified 2026-10-03) ([HF API](https://huggingface.co/api/datasets/SALT-NLP/SWE-chat); [card](https://huggingface.co/datasets/SALT-NLP/SWE-chat)).
  - **Tables (from the API):** `conversations`, `sessions`, `session_logs`, `subagent_tasks`, `context_events`, and others.
  - **Catalog-check reading of the card:**
    - Real sessions from public GitHub repos that checkpoint coding-agent sessions: Claude Code, Codex, Gemini CLI, Cursor, Copilot CLI and others. PII is redacted.
    - v2 has 17,968 sessions, 11.6M conversation rows and 47 GB.
    - `conversations` has a `timestamp` per entry, `model`, input, output, cache-creation and cache-read tokens, `tool_name`, `tool_call_id`, `tool_input_json` and `command`. Content has tool results truncated to 10 KB.
    - Raw transcripts are included.
  - I could not open the gated README myself. Field names come from the catalog check.
- **TraceLab: arXiv 2606.30560v2, 30 Jun 2026.**
  - **Source:** client-side Claude Code and Codex logs from 43 developers, Sep 2025 – Jun 2026.
  - **Scale (Table 1):** 4,265 sessions (2,676 Claude, 1,589 Codex), 357,161 LLM steps, 432,510 tool calls, 54.90B input tokens (52.56B prefix, 2.34B append) and 186.9M output tokens ([v2 Table 1](https://arxiv.org/pdf/2606.30560v2)).
  - **Timing:** "Every event is timestamped, and each LLM invocation reports its token usage." The schema has one row per step, i.e., one LLM invocation with its tool calls.
  - **Tool latency:** "Codex reports the tool execution time explicitly, whereas for Claude we derive it from the tool’s call and return timestamps."
  - **Token fields:** taken from the providers: cache_read_input_tokens, input_tokens, cache_creation_input_tokens, cached_input_tokens, output_tokens and reasoning_output_tokens ([v2 §3.1–3.2](https://arxiv.org/pdf/2606.30560v2)).
  - **Sanitization:** "We replace all session, tool-call, project, and user identifiers with stable pseudonyms, and drop the raw user messages and tool input/output text, keeping only their character counts along with token usage, timestamps, and pseudonymous user identifiers" ([v2 §3.3](https://arxiv.org/pdf/2606.30560v2)).
  - **Release:** "We release the dataset, trace collection pipeline, and analysis code at https://github.com/uw-syfi/TraceLab.git" ([v2 abstract](https://arxiv.org/pdf/2606.30560v2)).
  - **Timing statistics:**
    - "human thinking is 92.3% of session wall-clock".
    - Positive human waits have median 1.4 min and p90 20.6 min.
    - Positive tool latencies have median 0.3 s and p90 13.6 s.
    - The p99 session elapsed time is about 206 h ([v2 §4.4, Table 7](https://arxiv.org/pdf/2606.30560v2)).
  - **How its own results are produced:** the cache results are the providers' own server-side prefix-cache hits. The eviction-timeout sweep is analytic, "under an idealized rule: a step is a complete cache miss when its preceding idle gap exceeds the eviction time, and a full hit otherwise" ([v2 §7.1–7.4](https://arxiv.org/pdf/2606.30560v2)).
  - **Limitations:** "our trace is drawn from our own day-to-day use of coding agents" ([v2 §9](https://arxiv.org/pdf/2606.30560v2)).
  - **Released dataset, v0.0.2** (on Hugging Face as UW-SyFI/TraceLab and as GitHub release v0.0.2; CC BY 4.0 data, Apache-2.0 code; not gated):
    - **Scale:** larger than the paper's: "**665,453** agent steps", "**8,058** sessions", "**743,819** normalized tool calls", "**52** pseudonymous users" ([HF card](https://huggingface.co/datasets/UW-SyFI/TraceLab)).
    - **Tables:** `rounds`, `tool_calls` and `timing_events`; the last holds 2,688,829 ordered events ([HF card](https://huggingface.co/datasets/UW-SyFI/TraceLab)).
    - **Tool-call metadata:** tool name, timestamps, character counts, latency, error status, "executable labels, and command skeletons" ([HF card](https://huggingface.co/datasets/UW-SyFI/TraceLab)).
    - **Schema detail:** the catalog check's reading of the schema lists `emitted_at`, `result_at`, `tool_wall_latency_ms` (= result_at − emitted_at) and `tool_internal_latency_ms`, plus ordered bash `executables` ([DB_SCHEMA.md](https://github.com/uw-syfi/TraceLab/blob/main/artifacts/utils/DB_SCHEMA.md)).
    - **What it lacks:** no LLM request-issue time or TTFT. NVIDIA's AIPerf, which can replay the trace, notes: "The corpus records no server-reported latency and no TTFT, so this is a proxy" ([AIPerf doc](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/tracelab-trace.md)).
  - **Bundled replayer:** `session_runner` is "a session-aware closed-loop workload runner for OpenAI-compatible inference servers (vLLM today, SGLang and others via the pluggable backend)" ([replay README](https://github.com/uw-syfi/TraceLab/blob/main/replay/README.md)).
    - Loop: "send round i -> wait for full LLM response -> sleep tool_wait_after_ms -> send round i+1".
    - It uses synthetic text but preserves `prefix_len`, appended length, target decode length (`max_tokens`), session arrival, and tool/user waits.
- **CacheWise coding traces: arXiv 2606.16824v1.**
  - Real Claude Code traces "from researchers actively using coding assistants for research".
  - "We have open sourced the dataset at github.com/cachewise-project/cachewise-coding-traces." CacheWise is "Implemented in vLLM and evaluated on the collected traces" ([HTML v1](https://arxiv.org/html/2606.16824v1)).
  - The earlier notes record per-bash-command P50s from 0.1 s to 22 s in this paper.
  - **Repo check** ([repo](https://github.com/cachewise-project/cachewise-coding-traces)):
    - **Size:** 94 sessions across 11 projects; 20,634 LLM-call events and 11,617 tool calls; 20.2 MB raw.
    - **Raw fields:** sanitized Claude Code JSONL with `timestamp`, `requestId`, `model` and `usage`, including cache creation and read. All text is "<redacted>".
    - **Parsed tool events:** `tool_name`, `timestamp`, `result_timestamp`, `execution_time_ms`.
    - **Arguments:** "Tool arguments in release data are redacted", so the bash command word is absent.
    - **License:** none in the repo.
    - **Catalog-check inferences:** dates appear shifted, with 2020 timestamps next to 2025 model names; `requestId` repeats once per content block.
- **SemiAnalysis "AgentX" Claude Code corpus** (Hugging Face semianalysisai/cc-traces-weka-062126), as described by two papers:
  - **"When Fancy Eviction Fails":** 98.8K requests, 21.6B tokens, 64-token session-scoped hashes. "Because the corpus lacks absolute timestamps, we artificially schedule sessions to maintain a steady concurrency of 16, while preserving the exact timing between requests within each session" ([2609.28870v3 §4, Table 3](https://arxiv.org/pdf/2609.28870v3)).
  - **FlatCake** replays it ("98,827 requests and 21.6B input tokens across 393 multi-turn sessions") "in recorded arrival order across arrival rates from 0.1 to 1.0 QPS without cache warm-up". This discards the recorded gaps ([2610.38355v1 §5](https://arxiv.org/pdf/2610.38355v1)).
  - **The card** confirms the identity: a "WekaTrace corpus derived from SemiAnalysis Claude Code proxy traces", built 2026-06-21, Apache-2.0, not gated. 393 traces, 56,798 main turns, 42,029 subagent inner requests, 98,827 model requests, 21,635,381,376 input tokens ([card](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)).
  - **Count caveat (card):** the `in` field is "(number of 64-token KV-cache prefix blocks) × 64", not a tokenizer count.
  - **Fields (my 300 KB range read of traces.jsonl):**
    - Trace level: `id`, `models`, `block_size`, `hash_id_scope`, `requests`.
    - Request level: `t`, `model`, `in`, `out`, `hash_ids`, `api_time`, `type`, `ttft`, `think_time`.
    - There are no tool names or tool durations.
  - **Field semantics:** AIPerf's loader defines `think_time` as "Client delay in seconds before this request" and `api_time` as "Server processing time in seconds" ([AIPerf loader](https://github.com/ai-dynamo/aiperf/blob/main/src/aiperf/dataset/loader/weka_trace_models.py)). So the gap mixes tool and human time (inference).
  - **Related Weka traces:** a related Apache-2.0 set of "739 traces with 59,204 requests" in the same format, at whole-second resolution, is in the kv-cache-tester repo ([README](https://github.com/callanjfox/kv-cache-tester)).
- **FreeInference and Chutes traces: arXiv 2609.28870v3, 7 Nov 2026.**
  - **Content:** production traces from two public LLM inference services. FreeInference "is heavily dominated by agentic workloads"; Chutes mixes human multi-turn chat, agentic sessions and API use ([v3 §3](https://arxiv.org/pdf/2609.28870v3)).
  - **Scale (Table 1):**

    | Trace | Requests | Share in multi-turn sessions | Span | Tokens |
    |---|---|---|---|---|
    | FreeInference | 327.5K | 34.3% | 7.0 days | 10.5B |
    | Chutes | 515.8K | 26.1% | 130.9 days | 9.7B |

  - **Format:** 16-token block hashes (Qwen3-Coder-30B tokenizer). Released at huggingface.co/datasets/harvardMadsys/2026_prefixcache ([v3 Table 1, §1](https://arxiv.org/pdf/2609.28870v3)).
  - **Table 3 of the paper compares public traces:**

    | Trace | Requests | Span | Block size | Hash scope | Agentic |
    |---|---|---|---|---|---|
    | TraceLab | 357K | 253 days, scattered | none | — | yes |
    | Mooncake tool&agent | 23.6K | 1 h | 512 tokens | global | yes |
    | AgentX | 98.8K | per session | 64 tokens | per session | yes |
    | Qwen Bailian | 10.8–172.8K | 2 h | 16 tokens | global | "very little" |

    Only FreeInference and Chutes "combine fine-grained block identity scoped across sessions, agentic traffic, and a multi-day span" ([v3 Table 3](https://arxiv.org/pdf/2609.28870v3)).
  - **The release card** (harvardMadsys/2026_prefixcache, "LLM Prefix-Cache Block Traces", CC BY 4.0) ([card](https://huggingface.co/datasets/harvardMadsys/2026_prefixcache)):

    | File | Size | Content | Requests | Sessions | Span |
    |---|---|---|---|---|---|
    | `freeinference.lcp.zst` | 76 MB | "Coding and assistant agents", 2026-07-11 to 07-17 | 327,548 | 222,934 | 7 days |
    | `chutes.lcp.zst` | 232 MB | "Chat and roleplay" | 515,817 | 402,221 | 130.9 days |

  - **Request records:** "arrival and completion time (microseconds, relative), session and user id (chutes has no user ids), turn, and input and output token counts". Each block entry stores the position of its next access, for Belady-style offline policies.
  - **Format:** `lcp`, the PrefixSim/libCacheSim binary format.
  - **No tool fields.** The request-level sources are the agentic trace above and `harvardMadsys/chutes_metadata`.
  - **chutes_metadata** has 6.12B requests with start and complete times and tokens, but no sessions, content or hashes ([card](https://huggingface.co/datasets/harvardMadsys/chutes_metadata)).

#### C. Released agent-benchmark runs with timing
- **Exgentic agent runs** ("General Agent Evaluation", arXiv 2602.22953, ICLR 2026 Workshop on Agents in the Wild, per its abstract page; [abs](https://arxiv.org/abs/2602.22953)).
  - The KV storage study 2611.10178 uses released Exgentic runs as raw-text traces: "their timestamps come from agent execution and therefore include real tool latency". The runs combine four to five models ([2611.10178v1 App. A.1](https://arxiv.org/pdf/2611.10178v1)).
  - Its Table 1 sizes them as follows ([2611.10178v1 Table 1](https://arxiv.org/pdf/2611.10178v1)):

    | Benchmark | Sessions | Requests | Mean turns |
    |---|---|---|---|
    | SWE-bench | 1,959 | 92K | 46.8 |
    | AppWorld | 1,500 | 48K | 32.3 |
    | BrowseComp+ | 1,948 | 38K | 19.7 |
    | τ²-bench Airline | 957 | 11K | — |
    | τ²-bench Retail | 1,848 | 22K | — |
    | τ²-bench Telecom | 1,844 | 30K | — |

  - **Release on Hugging Face:** org Exgentic ([traces-v2 card](https://huggingface.co/datasets/Exgentic/traces-v2); [agent-llm-traces-v2 card](https://huggingface.co/datasets/Exgentic/agent-llm-traces-v2)).
    - **traces-v2:** "OpenTelemetry-shaped execution traces for **10,573 agent runs** across **6 benchmarks**, **5 agent harnesses**, and **15+ models**" with "**626,278** (chat + invoke_agent + execute_tool)" spans. Each span has `start_time`/`end_time` (ISO-8601 UTC).
    - **Inclusions:** user-simulator and judge calls, and "Every retry attempt".
    - **agent-llm-traces-v2:** keeps only the agent's chat calls, 10,056 runs and 241,674 spans, with failed-retry groups collapsed.
    - **Further fields (catalog check):** full input and output messages, tool names and arguments, token usage, and time to first chunk. A `timestamps.backfilled` flag shows that some timestamps were rebuilt.
    - **License:** the v2 cards state none. The v1 traces are CDLA-Permissive-2.0.

- **GAIATrace: arXiv 2606.01725v2, 1 Oct 2026.**
  - Request-level execution traces of two agent systems on the GAIA validation set.
    - OWL: 165 tasks, 2,669 Main-LLM and 2,737 Sub-LLM queries.
    - MiroThinker: 103 tasks, 1,491 Main-LLM and 591 Sub-LLM queries.
    - Plus their tool calls, with per-tool execution statistics in Table 2.
  - The models are mixed, open-source and proprietary: gpt-oss-120b, MiroThinker-1.7-mini, gpt-4o and gpt-4o-mini ([v2 §3–4, Tables 1–2](https://arxiv.org/pdf/2606.01725v2)).
  - **Format and license:** the traces ship "in token_id", with "text verbatim files ... left out", on Zenodo (DOI 10.5281/zenodo.22133686): traces CC-BY-4.0, code Apache-2.0.
  - **Replay:** a Vidur-based simulator, Vidur-Agent, replays them with "dependency-released multi-turn sessions" ([v2 artifact appendix](https://arxiv.org/pdf/2606.01725v2)).
- **Continuum's collected agent traces: arXiv 2511.02230v7.**
  - "We collect and analyze 100 mini-swe-agent traces [33] running SWE-Bench [23]. We also collect 100 traces from BFCL V4 Web" search, run with GPT-5.
  - They are replayed on vLLM "using poisson distribution for the arrival pattern of agent programs", plus a real SWE-bench run (Fig. 12).
  - The paper says "We will open-source our collected agent inference traces"; I did not verify a release ([v7 §3, §6.1–6.2](https://arxiv.org/pdf/2511.02230v7)).
- **SWE-bench experiments repo** ([README](https://github.com/SWE-bench/experiments)). "This repository holds entries, not artifacts". Trajectories sit at `s3://swe-bench-submissions/<split>/<entry>/trajs`, or in the submitter's repo, with no required format. The catalog check opened one sample per agent family:
  - **OpenHands submissions** (e.g. `20251127_openhands_claude-opus-4-5`): lists of role/content or OpenAI-format messages, with no timestamps or tokens ([sample](https://swe-bench-submissions.s3.amazonaws.com/verified/20251127_openhands_claude-opus-4-5/trajs/astropy__astropy-14096.json)).
  - **SWE-agent `.traj`:** `execution_time` per step, plus per-instance `model_stats` (tokens_sent, tokens_received, api_calls). No timestamps. The source times each action with `time.perf_counter()` around `env.communicate` ([types.py](https://github.com/SWE-agent/SWE-agent/blob/main/sweagent/types.py)).
  - **mini-SWE-agent v2** (`20260217_mini-v2.0.0_claude-4-6-opus`, bash-only split): every assistant and tool message carries a Unix `timestamp`, and each call carries `usage` with cached tokens. The docs say: "unix timestamp of when this message was created" ([output_files.md](https://github.com/SWE-agent/mini-swe-agent/blob/main/docs/usage/output_files.md)). These are Y-grade fields from a benchmark setup.
- **OpenHands**
  - **Native event format:** events carry `timestamp`, and actions and observations carry `tool_name`/`tool_call_id`, so tool time can be derived as observation time minus action time. Per-call token usage and `response_latencies` live in `llm.metrics` ([SDK event base](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/openhands/sdk/event/base.py); [metrics docs](https://docs.openhands.dev/sdk/guides/metrics.md)).
  - **OpenHands/openhands-feedback:** 275 real-user sessions from 2024, MIT. Per-event timestamps in µs, e.g. a `run_ipython` action at 03:37:06.010 and its observation at 03:37:10.258. No token fields ([card](https://huggingface.co/datasets/OpenHands/openhands-feedback)).
- **LMCache agentic traces** (sammshen/lmcache-agentic-traces; `DiscoPosse/lmcache-agentic-traces` is an identical copy; CC-BY-4.0) ([card](https://huggingface.co/datasets/DiscoPosse/lmcache-agentic-traces)).
  - **Scope:** "**787 multi-turn agentic LLM sessions** (24,881 total LLM iterations)" from SWE-bench Verified (OpenHands CodeAct), GAIA and WildClaw. The per-source table sums to 764.
  - **Fields:** `input` is the full cumulative messages, each a strict prefix-superset of the previous. `output_length`.
  - **`pre_gap`:** "Seconds between the previous iteration's response completing (last streamed token) and this iteration's request being sent. This is the real tool-execution / user-thinking time".
  - **Intended replay:** "fire request N exactly `pre_gap[N]` seconds after receiving the last token of response N-1".
- **Terminal-Bench 2.0 leaderboard trajectories** in the ATIF format: steps carry millisecond timestamps, tool calls and observations, but no tool-result timestamp. Per-step token metrics are optional, and were 0 in the sample. Apache-2.0 ([dataset](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard); [ATIF spec](https://docs.harborframework.com/agents/atif)).
- **AgentTrace** (pagarsky/agent-trace, Apache-2.0): 1,400 traces from Qwen3-0.6B/1.7B on MBPP and NL2Bash, with tool spans carrying start/end in ns and duration. Toy scale ([dataset](https://huggingface.co/datasets/pagarsky/agent-trace)).
- **Trajectory datasets with no timing at all**, all verified on their cards:

  | Dataset | Size | License | Notes |
  |---|---|---|---|
  | nebius/SWE-agent-trajectories ([card](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)) | 80,036 trajectories | CC-BY-4.0 | — |
  | nebius/SWE-rebench-openhands-trajectories ([card](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)) | 67,074 trajectories | CC-BY-4.0 | Qwen3-Coder-480B, OpenHands v0.54.0 |
  | SWE-Gym/OpenHands-SFT-Trajectories ([card](https://huggingface.co/datasets/SWE-Gym/OpenHands-SFT-Trajectories)) | 491 rows | MIT | — |
  | SWE-bench/SWE-smith-trajectories ([card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)) | about 76K rows over three splits | MIT | — |
  | miromind-ai/MiroVerse-v0.1 ([card](https://huggingface.co/datasets/miromind-ai/MiroVerse-v0.1)) | "147K+ samples" | CC-BY-NC-4.0 | Gated; no schema stated |

  All give full text, so prefixes can be rebuilt, but they cannot support replay of tool gaps.

#### D. Serving traces with session linkage (little or no agentic traffic)
- **Alibaba Qwen-Bailian anonymized traces** ([repo](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon); Apache-2.0).
  - **Files:** traceA (to-C), traceB (to-B API, collected 2024.12), thinking and coder. "two-hour sampled". Sizes 56.4, 96.2, 27.9 and 132.1 MB. The catalog check counted about 43,058, 172,800, 10,812 and 43,011 requests from the last `chat_id` (inference).
  - **Fields:** `chat_id`, `parent_chat_id` (−1 for a root), `timestamp` (seconds, millisecond precision), `input_length`, `output_length`, `type`, `turn`, `hash_ids` ("Salted SipHash blocks (16 tokens per block)").
  - **Not included:** user ID, model, completion time, tool records.
  - **Catalog-check inference:** in traceA and traceB, the same hash ID appears after different prefixes, so whole leading sequences should be compared rather than single IDs.
  - **Replay:** gaps are arrival-to-arrival, mixing generation and think time. The repo points to an external replayer, blitz-serving/trace-replayer.
- **Mooncake traces** ([FAST25-release](https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release); repo Apache-2.0).
  - **Files:** `conversation_trace.jsonl` (12,031 requests), `toolagent_trace.jsonl` (23,608), `synthetic_trace.jsonl` (3,993), and `arxiv-trace/mooncake_trace.jsonl`.
  - **Fields:** relative `timestamp` (ms), `input_length`, `output_length`, `hash_ids` over 512-token blocks. No session, turn or tool fields.
  - **Catalog-check inferences from the files:**
    - Effective time resolution is about 1 s: most timestamps are multiples of 1,000 ms.
    - The "toolagent" reuse is mostly shared system prompts: three first blocks cover 23,590 of its 23,608 requests.
- **BurstGPT** ([repo](https://github.com/HPMLL/BurstGPT); CC-BY-4.0).
  - Release v2.0 added "`Session ID`: conversation ID, only conversation mode have this", and "`Elapsed time`: time between the request submission time and system response time".
  - `Timestamp` is "request submission time, seconds from `0:00:00` on the first day". `BurstGPT_3.csv` covers "another 110 days" with "5344.0k lines" (README).
  - **Think time** can be derived as next submit − previous submit − previous elapsed, at 1 s resolution (catalog check, inference).
  - No text, no hashes, not agentic.
- **ServeGen** ([repo](https://github.com/alibaba/ServeGen); Apache-2.0).
  - Per-client arrival and length data are aggregated at 600 s steps.
  - `data/conversations/conversations_hashed.json` (50.5 MB) holds multi-turn conversations with per-turn Unix timestamps, token counts and block-hash-like IDs.
  - The ID scheme is not documented, and per the catalog check it does not chain across turns, so prefix sharing cannot be rebuilt directly (inference).
- **Azure LLM inference traces 2023 and 2024** ([2023 doc](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2023.md); [2024 doc](https://github.com/Azure/AzurePublicDataset/blob/master/AzureLLMInferenceDataset2024.md); CC-BY): only `TIMESTAMP`, `ContextTokens`, `GeneratedTokens`. No sessions.
- **Azure Functions 2019** ([doc](https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md)): per-function invocation counts in 1-minute bins over 14 days. This is the trace behind three of the Q4 papers.

#### E. Chat datasets (with or without timestamps)
- **WildChat-1M and WildChat-4.8M** ([1M](https://huggingface.co/datasets/allenai/WildChat-1M); [4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M); ODC-BY; not gated).
  - Assistant turns carry a `timestamp` for when "the backend server receives the full response". User turns carry none.
  - Sizes: 837,989 and 3,199,860 non-toxic conversations.
  - 4.8M adds `created`, which the catalog check reads as request start (inference; undocumented on the card).
- **ShareChat** ([card](https://huggingface.co/datasets/tucnguyen/ShareChat); CC BY-NC 4.0; gated): 142,808 conversations from shared ChatGPT, Claude, Gemini, Grok and Perplexity links.
  - Only the ChatGPT and Grok subsets have per-message times (`message_create_time`); Gemini has conversation-level `created_at`/`published_at`.
  - The card's YAML schema conflicts with its prose, and rows were not inspected because the dataset is gated.
  - It is cited as arXiv 2512.17843 in [IAC v1](https://arxiv.org/pdf/2611.09715v1).
- **LMSYS-Chat-1M** ([card](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)): no timestamp field; gated.
- **Chatbot Arena conversations** ([card](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)): one `tstamp` per conversation, averaging 1.2 turns.
- **ShareGPT_Vicuna_unfiltered** ([card](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)): `id` and `conversations` only, no timestamps, chunked to 2,048 tokens.
- **sharetrawl** ([card](https://huggingface.co/datasets/DJRHails/sharetrawl)): 256,538 shared conversations with conversation-level times only.
- **None of these chat sets is agentic.**

#### F. Catalog table
"Tool-gap replay": Y = per-call tool durations, or request plus tool-completion times; Partial = request-level gaps only, benchmark-harness timing, or timing without content; N = no usable gaps. Rows are sourced in A–E above.

| Trace | Kind | Timing recorded | Tool names / durations | Tokens | Content / hashes | Size | License / access | Tool-gap replay | Prefix reconstruction |
|---|---|---|---|---|---|---|---|---|---|
| FreeInference Agentic Trace | Real users, 14 harnesses, gateway | Request start/end/TTFT (epoch ms); user/tool trigger | Name, category, shell programs; `latency_ms` = gap to next request | Prompt/completion/cached; per role | 16-token chained block IDs (o200k, local to each week); no text | 12,002 sessions; 1.19M requests | CC BY 4.0; open | **Y** | Yes, within a week |
| Copilot Coding Agent 2026 | Production | Call and tool-batch timestamp + duration (end times, by inference) | Name, per-call duration | Prompt/completion/cached | Segment types and token lengths only | 301K sessions; 9.3M calls; 3.2 GB | CC-BY; open | **Y** | Synthetic only |
| TraceLab v0.0.2 | Real developers (client logs) | Per-event ms timestamps; no LLM issue time | Name, wall and internal latency, bash executables | Prefix/append/output | None (character counts) | 8,058 sessions; 665K steps | CC BY 4.0; open | Y timing, no content | Token-count chains |
| SWE-chat v2 | Real users, 8 agents | Per-entry timestamps | Name, args, command | In/out/cache | Redacted text + raw transcripts | 17,968 sessions; 47 GB | ODC-BY; gated (auto) | Y (derived) | Re-tokenize |
| CacheWise | Real Claude Code users | Per-entry ms (dates shifted) | Name + execution ms; args redacted | In/out/cache | Text redacted | 94 sessions | No license | Y timing, no content | Token counts |
| SemiAnalysis cc-traces-weka-062126 | Claude Code proxy | `t`, `api_time`, `ttft`, `think_time` | None | in (blocks × 64)/out | 64-token hashes, local to each session | 393 sessions; 98,827 requests | Apache-2.0 | Partial (gap not split) | Within a session |
| 2026_prefixcache (FreeInference, Chutes) | Gateway/proxy | Arrival + completion (µs) | None | in/out | 16-token block IDs | 327,548 + 515,817 requests | CC BY 4.0 | Partial | Yes |
| Exgentic traces-v2 | Benchmark runs | Span start/end (some backfilled) | Tool spans | in/out | Full text | 10,573 runs | v2 none stated | Partial (Y fields) | Re-tokenize |
| GAIATrace | Benchmark (GAIA) | Request-level; per-tool statistics | Per-tool | Token IDs | Token IDs | 268 tasks | CC-BY-4.0 (Zenodo) | Partial (not opened) | Yes |
| mini-SWE-agent v2 (SWE-bench) | Benchmark | Unix time per LLM and tool message | bash; derivable | Per call, incl. cached | Full text | Per submission | Not stated | Partial (Y fields) | Re-tokenize |
| SWE-agent `.traj` | Benchmark | `execution_time` per step | Actions | Per-instance totals | Full text | Per submission | Not stated | Partial | Re-tokenize |
| OpenHands feedback | Real users (2024) | Per-event µs | Actions; derivable | None | Text | 275 sessions | MIT | Y (small) | Approximate |
| LMCache agentic traces | Benchmark | `pre_gap` per call | None | Output only | Full cumulative text | 787 sessions | CC-BY-4.0 | Partial (gap not split) | Exact |
| Terminal-Bench 2.0 ATIF | Benchmark | Per-step ms | Name + args; no result time | Optional | Full text | ≥2,115 trajectories | Apache-2.0 | Partial | Re-tokenize |
| nebius / SWE-Gym / SWE-smith / MiroVerse | Benchmark / SFT | None | In text | None | Full text | 491 to 80K | CC-BY / MIT / NC | N | Text |
| Qwen-Bailian (A, B, thinking, coder) | Production | Arrival (ms), 2 h | None | in/out | 16-token salted hashes | 11K–173K requests per file | Apache-2.0 | Partial (arrival only) | Yes (whole sequences) |
| Mooncake (conversation, toolagent, synthetic) | Production | Arrival (~1 s effective), 1 h | None | in/out | 512-token hashes | 12K / 23.6K / 4K | Apache-2.0 (repo) | N (no sessions) | Coarse |
| BurstGPT_3 | Production chat/API | Submit + elapsed (1 s), 110 days | None | Request/response | None | 5.34M rows | CC-BY-4.0 | Think time only | Synthetic |
| ServeGen conversations | Production | Per-turn Unix ms | None | in/out | IDs (scheme unclear) | 50.5 MB | Apache-2.0 | Partial | Unclear |
| Azure LLM 2023/2024 | Production | Arrival | None | Context/generated | None | Hours to a week | CC-BY | N | No |
| WildChat-1M / 4.8M | Chat | Assistant completion (+ `created`) | None | (usage null) | Full text | 0.84M / 3.2M conversations | ODC-BY | Partial | Exact |
| ShareChat | Chat | Per message (GPT/Grok) | None | None | Full text | 142,808 conversations | CC BY-NC; gated | Partial | Exact |
| LMSYS-Chat-1M / ShareGPT | Chat | None | None | None | Text | 1M / ~53K | Gated / apache-2.0 tag | N | Text |

#### G. How recent papers replay recorded timing
- **Intent-Aware Caching** (arXiv 2611.09715v1) says WildChat and ShareChat "are the only two public datasets that contain both conversation content and timestamps" ([v1 §1 fn. 1](https://arxiv.org/pdf/2611.09715v1)). Its replay has three parts ([v1 App. A.4](https://arxiv.org/pdf/2611.09715v1)):
  - "we preserve the original inter-turn delays within each conversation".
  - It "control[s] the relative arrival time between new conversations using a Poisson process at the target arrival rate".
  - For LMSys-Chat and ShareGPT it samples "inter-turn delays from a long-tailed delay distribution".
- **The KV storage study** (arXiv 2611.10178v1) uses a fixed-concurrency replay: "Each of N slots replays randomly chosen recorded sessions from the trace back to back ... Each session keeps its recorded call times and context sizes" ([v1 App.](https://arxiv.org/pdf/2611.10178v1)). It also sorts its 11 traces by evidence class ([v1 §3.3, Table 1](https://arxiv.org/pdf/2611.10178v1)):
  - raw text, re-tokenized into 16-token chained blocks: WildChat, Exgentic, RAGBench, Open Deep Research;
  - pre-hashed: Qwen-Bailian with 16-token blocks, RAGPulse with 512-token blocks;
  - metadata only, with timestamps, token counts and sessions but no block identities: TraceLab and BurstGPT v2.

  Its timestamp origins are "Real" for WildChat (13 months), BurstGPT v2 (110 days), Qwen-Bailian (2 h) and TraceLab (8 months), and "Agent run" for Exgentic.
- **"When Fancy Eviction Fails"** schedules AgentX sessions at a steady concurrency of 16 while keeping within-session timing. It replays its own traces in PrefixSim, a C++ simulator ([2609.28870v3 §3–4](https://arxiv.org/pdf/2609.28870v3)).
- **FlatCake** replays in recorded order at fixed QPS ([2610.38355v1 §5](https://arxiv.org/pdf/2610.38355v1)). **Mooncake** replays at 1× or 2× ([v5 §8](https://arxiv.org/pdf/2407.00079v5)). **Alibaba** downscales with TraceSplitter ([v5 §4.3](https://arxiv.org/pdf/2506.02634v5)).
- **Simulation-based replays:** M365's residency replay ([2610.22863v1 §5](https://arxiv.org/pdf/2610.22863v1)), TraceLab's idealized timeout rule ([2606.30560v2 §7.4](https://arxiv.org/pdf/2606.30560v2)), and GAIATrace's Vidur-Agent ([2606.01725v2](https://arxiv.org/pdf/2606.01725v2)).

### Inferences
- **The replay method this literature supports for a tool-gap paper.** Run closed loop within each session: release request i+1 a recorded gap after response i completes on the engine under test. Re-time session starts with Poisson arrivals or a fixed concurrency to control load.
  - IAC, the KV storage study, "When Fancy Eviction Fails" (AgentX), TraceLab's session_runner and LMCache-agentic's `pre_gap` recipe all use this pattern.
  - Open-loop timestamp replay (Mooncake), fixed-QPS replay in recorded order (FlatCake), and downscaling (Alibaba) either break the gap-to-completion relation or discard the gaps.
- **Candidate primary trace: FreeInference Agentic Trace.** It is the only verified set with all of the following:
  - real users;
  - a per-request user/tool trigger, so tool time and human time separate;
  - per-tool names plus the shell `programs`, which fit forecasts keyed by tool plus command word;
  - chained block IDs, so prefix sharing and SSD-tier footprints can be rebuilt.

  Its tool gap is measured at the gateway, so it includes client and network time. Block IDs are local to each week, so cross-week reuse is invisible.
- **Candidate secondary traces:**
  - **Copilot 2026**, for production-scale distributions of tool durations and idle time across turns. Its prompts must be synthesized from segment token lengths and cached counts.
  - **TraceLab**, for per-tool latency with bash executables, plus a ready closed-loop replayer for vLLM and SGLang.
  - **SWE-chat**, for real-user text.
- **Generation time is never transferable.** In every real-user trace it reflects the providers' serving stack. Replays should force output lengths to the recorded counts and let the engine under test produce its own generation time. Mooncake and Alibaba force lengths this way.
- **Gap semantics differ by trace and must be normalized before fitting per-tool distributions:**
  - FreeInference `latency_ms`: end of response to next request, at the gateway.
  - TraceLab `tool_wall_latency_ms`: tool emitted to result, in client logs.
  - Copilot `duration_ms`: tool-batch wall clock, with timestamps that appear to be end times.
  - SemiAnalysis `think_time` and LMCache-agentic `pre_gap`: tool and human time mixed.
- **Local harness gap.** This repo's agentic benchmark loader (python/sglang/benchmark/datasets/agentic_trace.py) replays OpenHands/SWE-smith message deltas round by round with the server's own replies. It has no gap field, so as written it cannot replay tool gaps (inference from the loader's docstring and fields). The Mooncake loader in python/sglang/benchmark/datasets/common.py points at the Mooncake traces, which lack sessions.

### Gaps
- No dataset was downloaded in full. Counts come from cards or READMEs, and several disagree internally:
  - TraceLab: paper v2 (4,265 sessions) vs release v0.0.2 (8,058).
  - FreeInference agentic: card 1,186,582 requests vs 1,222,199 processed rows.
  - 2026_prefixcache: 222,934 FreeInference sessions in one week vs 12,002 in 16 weeks in the agentic release. Session definitions evidently differ.
  - LMCache agentic: 787 vs 764 sessions.
  - Copilot abstract "95T tokens" vs Table 3 at 44.9T prompt tokens.
- Unverified from documentation:
  - Copilot timestamp semantics (end vs start) come from a 381-session sample.
  - SWE-chat fields come from a card read by the catalog check, since the README is gated.
  - ShareChat rows were not inspected (gated).
  - GAIATrace's Zenodo files were not opened.
- Not found or not released: AgenticTraceSim (M365); the 11.7M-request agent-platform trace (2609.34432); Continuum's collected traces.
- Excluded for lack of a card or license: `wekaio/replay-40d-oct14-337a03810637`, a 3,190-session replay set per its own summary row. Also excluded: `maxbromberg/flux-traces`, a repackaging of other traces.

## Q4. Serverless keep-alive papers: evaluation mode and trace source (one line each)

### Takeaway
All four lean on recorded arrival traces. Three use the Azure Functions 2019 trace, which records invocation counts per function per minute, so no sub-minute gap is real. RobustScaler uses container-registry and cluster job traces. Two papers combine simulation with a real-system check, one is real-system only, and one is simulation only.

### Cited Findings
- **Serverless in the Wild:** arXiv 2003.03423v3, 6 Jun 2020; USENIX ATC '20 ([USENIX page](https://www.usenix.org/conference/atc20/presentation/shahrad)).
  - **Mode: simulation plus a real system.** "We implement our policy in simulation and for the Apache OpenWhisk [34] FaaS platform, both driven with real workload traces."
  - **Simulation:** uses "the first week of the trace from Section 3" (§5.1–5.2, Figs. 14–19).
  - **OpenWhisk:** 19 VMs replay 68 randomly selected mid-range apps for 8 hours, 12,383 invocations, using FaaSProfiler ("to automate trace replay and result analysis") (§5.3).
  - **Trace:** the Azure Functions production trace covering the "entire infrastructure between July 15th and July 28th" 2019, at 1-minute resolution: "our dataset does not allow the precise reconstruction of interarrival times that are smaller than one minute" ([v3 §1, §3.1, §5](https://arxiv.org/pdf/2003.03423v3)).
- **RobustScaler:** arXiv 2204.07197v1, 14 Apr 2022; venue not stated in the arXiv text.
  - **Mode: simulated replay plus one real deployment.** Simulated replay produces all comparisons ("We replay these real-world traces with various autoscalers to evaluate their performance", Figs. 4–10, Tables I–III). One real deployment on an Alibaba Serverless Kubernetes (ASK) cluster uses the CRS trace (Table IV).
  - **Traces:**
    - CRS, a container-registry service: "time information of a total of 21059 queries" with start and end times, over 4 weeks.
    - Google 2019 cluster data: a 24 h subset "which has 20254 job queries in total".
    - Alibaba 2018: "5 days of data with a total of 503850 job records".
    - Synthetic non-homogeneous Poisson traces.
  - **Timing:** recorded timestamps are replayed on the test split ([v1 §VII](https://arxiv.org/pdf/2204.07197v1)).
- **GearUp:** arXiv 2610.38196v2, 1 Nov 2026; preprint.
  - **Mode: real system only.** "Our single-worker testbed uses one AWS EC2 g6e.4xlarge instance" with one L40S; scaling runs use 1, 2 and 4 workers.
  - **Trace:** "We sample a continuous 300-minute segment from the Azure Functions Trace 2019 dataset": 319 root requests. "The median inter-arrival time is 54.5 s, and the coefficient of variation is 0.516."
  - The trace drives two ML DAG applications. How sub-minute arrival times are derived from per-minute bins is not stated ([v2 §7.2](https://arxiv.org/pdf/2610.38196v2)).
- **Keep-Alive Caching for the Hawkes process (Narayana & Kash):** arXiv 2309.03521v1, 7 Sep 2023; UAI 2023.
  - **Mode: simulation only, in two forms.**
    - Synthetic Hawkes arrivals: "We use Ogata’s modified thinning algorithm".
    - A simulation driven by the Azure trace "on a subset of Azure traces released by Shahrad et al. [2020]". Parameters are fitted on day 8, fit is checked on day 7, and evaluation runs on day 9. The fitted policy is applied only "to the 25% of application processes which had the best goodness of fit" (3,694 of 14,788 common apps).
  - **Timing:** evaluation uses recorded 1-minute counts. The Optimized-TTL baseline uses synthetic Hawkes arrivals ("We generate a single simulation of arrivals for each application based on the estimated Hawkes process parameters for 24 hours") ([v1 §5, App.](https://arxiv.org/pdf/2309.03521v1)).

### Inferences
- The keep-alive literature accepts evaluation by trace-driven simulation, with a small real-system confirmation (Shahrad's OpenWhisk run, RobustScaler's ASK run). This is a precedent for pairing a policy simulator with engine runs in a KV-retention paper.
- These papers inherit the Azure trace's 1-minute resolution, so their arrival timing is coarser than tool-call gaps of seconds.

### Gaps
- RobustScaler's venue is not printed in its arXiv text. I did not verify it elsewhere.
