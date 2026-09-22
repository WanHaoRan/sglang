# Campaign 7 — Qwen3-30B-A3B-Instruct-2507, natural pools, full host memory, 128 sessions

Started 2026-09-22T02:37:14Z. Replaces campaign 6 (`compare_20260922_015454/`, Qwen3-32B bf16, stopped at arm 1 because
once the 7,808-token shared prefix is subtracted the working set never exceeded L1+L2 at 64 GB host). Config in `config.txt`.

## Why this model, and what it changes

Chosen for its 262,144 native context so sessions could run more turns. Two consequences the context window does not advertise:
- `b` = 2 x 48 layers x 4 KV heads x 128 = **98,304 B/tok at bf16 KV** (49,152 at fp8): every pool holds more tokens, so the bar to
  spill past L1+L2 rises, not falls. bf16 KV (the `auto` default) is what keeps L3 reachable with the full host pool.
- Its stock chat template has no think block and re-renders assistant history identically to the generation prompt: a direct
  tokenizer check reuses 48/48 tokens across a turn, so **no replay template** (`TEMPLATE=""`). Qwen3-32B's stock template diverges at token 25.
- The trace was re-converted at 262K (`traces/lmcache_agentic_trace_262k.json`, 768 convs / 24,879 turns vs 731 / 17,887 at the 32K cut).
  Sessions reach p50 37K at turn 40; the trace, not the model, caps context (max ~84K).

## Arms (in run order) — capacity identical on all four

| arm | L1 | L2 | L3 | prefetch |
|---|---|---|---|---|
| hbm_host | natural 668,160 tok (measured) | 160 GB = 1,627,604 tok | — | — |
| three_tier_to | natural | 160 GB | unlimited (`70,60` = 1.59 TiB) | timeout |
| three_tier_wc | natural | 160 GB | unlimited | wait_complete |
| hbm_lru | natural | — | — | — (control, last: it is the slow one) |

## Inputs measured on this box (`../probe_moe/`)

| | value |
|---|---:|
| L1, bf16 KV, mem-fraction 0.85, no pin | 668,160 tok |
| weights | 57.02 GB, no Marlin line, boot 104 s |
| P (recompute sweep, 8 lengths to 65K, 0 tier hits) | **4,903 tok/s** (32B bf16: 2,920) |
| decode per sequence at 25.6K, B = 1 / 8 / 16 | 58.6 / 50.6 / 46.1 tok/s (ITL 17.1 / 19.8 / 21.7 ms) |
| uncached tokens per hitting returning turn (campaign-6 arm 1) | p50 376, mean 1,606 |
| final context, offset-32 window, TURNS=40 | 35,580 (85 of 128 sessions hold all 40 turns) |

## The sizing chain

```
private WS = 128 x (35,580 - 7,808) = 3,554,849  vs  L1+L2 = 668,160 + 1,627,604 = 2,295,764   -> 1.26M tokens (35 %) only on L3
GPU-s per hitting turn ~ 1,606/4,903 + 182 x 21.7ms/16 = 0.58 s  -> capacity ~1.7, take 1.4 turns/s
unqueued turn ~ 0.3 prefill + 0.3 restore + 3.95 decode ~ 5 s
f_memory  <= L1/(1.2 x 35.6K)/128 = 15.6/128 = 0.122
f_compute <= 1.4 x 5 / 128 = 0.055                     <- binding
gap >= 5 x (1-0.055)/0.055 = 86 s  -> GAP 58; run at GAP 70 (offered 1.18 turns/s = 85 % of capacity) for margin
```
A-priori ceiling on simultaneous L3 restores `0.5(1 - L1/WS)` = **41 %**. Prefetch budget `0.5 x L2 / 27.8K` = 29 concurrent vs ~0.4 needed.

## Why the non-tiered arms are capped at 9,000 s and run last

A recompute of ~28K private tokens costs ~5.7 s of GPU against ~0.6 s for a hit. hbm_host must recompute the 35 % that
does not fit in L2 (capacity ~0.4 turns/s, 2.9x oversubscribed at this offered load) and hbm_lru ~80 % (capacity ~0.17, 7x).
Both will queue — that is the cost of not tiering, and it is the result — but uncapped they run 3-8 h. `CLIENT_TIMEOUT=9000`
keeps every completed turn; compare over the common window (RUNBOOK 7.3). Expected: tiered arms ~66 min each, hbm_host ~2.5 h
(capped, ~80 % of turns), hbm_lru 2.5 h (capped, ~33 %). Total ~7 h.

## What "tiering works" looks like here

three_tier_* returning-turn TTFT near recompute-free (~1 s L3 restore of 2.7 GB at ~2.9 GiB/s) with `storage` hits on ~35 % of
returning turns and queue depth ~0; hbm_host with the same 35 % recomputing at ~6 s and a growing queue; hbm_lru queue-bound.
If the three_tier arms also queue, read `HICACHE_EVT` (RUNBOOK 6.6) for prefetch rate-limits and timeouts before anything else.

## 05:10Z note: the per-arm wall cap (CLIENT_TIMEOUT=9000) does not stop the client

`timeout --foreground -s INT 9000 bash start_client.sh` delivers SIGINT only to the `bash start_client.sh` child (that is what
`--foreground` means: children of COMMAND are not timed out). bash defers the signal while its foreground pipeline
(`python3 replay_agentic.py | tee`) runs, and the replay client never receives it. Observed on hbm_host: timeout fired at
02:39:02+9000 s = 05:09:02Z, the client kept running (etimes 9069, 4,673/4,676 turns, last conversation gap-sleeping).
Consequences for this campaign:
- hbm_host: any `STAGE CAPPED hbm_host` line is spurious; the arm ran to completion (timeout returns 124 once the client
  exits on its own, the driver treats that as "capped" but keeps the run).
- hbm_lru (arm 4) will therefore run uncapped unless the client is signalled by hand (`kill -INT <replay_agentic.py pid>`
  after 9,000 s of client wall time makes bash exit on the pending SIGINT and the driver proceeds to STAGE CAPPED).
Fix for future campaigns (not applied while the driver is running): drop `--foreground` so the whole process group is
signalled, use `tee -i`, and give replay_agentic.py a SIGINT handler that writes the summary line and exits.

## 07:25Z note: two conversations lost in three_tier_wc to transport resets (not a cache effect)

client.jsonl of 20260922_064515_three_tier_wc_NAT160 has two turn records with no t_send and an `error` field:
conv 6 turn 6 "Server disconnected" (~06:50:50Z) and conv 11 turn 15 "[Errno 104] Connection reset by peer" (~07:04:50Z).
replay_agentic.py records the exception and ends that conversation (no retry), so the arm completes ~4,620 of 4,676 turns.
The server log has nothing at either minute (no abort/exception; the 1 "abort" match is the server_args dump). Both are the
aiohttp keep-alive race: a pooled connection idles through a gap, uvicorn closes it at its keep-alive timeout, the client
reuses it at the same instant. hbm_host / three_tier_to had 0 such errors, so it is random, not policy-related.
Handling: the comparison is per returning turn; drop conv 6 (turns >= 6) and conv 11 (turns >= 15) from all arms when
computing paired/common-turn statistics, and report the unpaired numbers alongside. Client fix for future campaigns:
retry once on aiohttp.ClientConnectionError / ServerDisconnectedError with a fresh connection (not applied mid-campaign).

## 08:20Z note: "STAGE FAILED three_tier_wc (client)" is a bookkeeping failure, the arm is complete

The replay client finished normally (summary: elapsed 5147.6 s, turns 4635, errors 2, mean_ttft 1.666 s). The per-turn
view at the end of start_client.sh then raised KeyError 't_send' on the two error records (see the 07:25Z note), bash
exited non-zero under set -e, and run_compare.sh logged STAGE FAILED and skipped the manifest line. Fixed by hand:
manifest.txt now carries `three_tier_wc 20260922_064515_three_tier_wc_NAT160 client_064821_cmp_three_tier_wc`, and
start_client.sh's per-turn view now prints ERROR rows for such records (atomic replace, so the running hbm_lru client
still executes the old copy; if hbm_lru also records a transport error, add its manifest line the same way).

## 11:37Z note: the three script fixes were verified after the campaign

`run_compare.sh` now passes `CLIENT_TIMEOUT` to the client as `--max-seconds` (no `timeout` wrapper) and prints STAGE CAPPED
from the summary line's `"capped": true`; the manifest line follows the summary line, not the wrapper's exit code;
`replay_agentic.py` stops conversations at their next turn boundary after the deadline (the gap sleep never crosses it) and
retries a request once on `aiohttp.ClientConnectionError`. Self-tests on the idle box (hbm_lru, 4 conversations x 8 turns,
GAP=0): with `CLIENT_TIMEOUT=40` the run finished naturally in 28.6 s (STAGE DONE, manifest written, no cap;
`compare_20260922_113351`); with `CLIENT_TIMEOUT=15` all four conversations stopped after turn 6 (in-flight requests
completed at 15.1-19.7 s), the summary carried capped=true, and the driver printed STAGE CAPPED, STAGE DONE, the manifest
line and ALL DONE (was `compare_20260922_113546`). The retry path was exercised against a closed port (was `results/_selftest/`:
one `ClientConnectorError` retry, then the recorded abort). These self-test directories were deleted afterwards (2026-09-22 14:29Z).
