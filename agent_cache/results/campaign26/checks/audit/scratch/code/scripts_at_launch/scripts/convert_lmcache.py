"""Convert the LMCache agentic traces (parquet) into the agentic-trace JSON that sglang's benchmark client replays.

  python3 convert_lmcache.py --parquet-dir /tmp/lmcache --out /sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json

Input: one parquet row per LLM call of an agent session: session_id, model, input (the full cumulative message list
sent on that call), output_length (completion tokens), pre_gap (seconds from the previous reply finishing to this
request being sent). Rows of a session_id are contiguous. One session_id can hold more than one independent
context (a WildClaw session bundles two runs with different system prompts), so rows are first split into
append-only chains: a row joins the chain whose last input it extends; a row that extends no chain (a first call,
or a call after the harness rewrote history in place) starts a new chain. Each chain becomes one replayable
session, named "<session_id>__chain<k>" when a session_id has several.

Output: {"metadata": {...}, "session_ids": [...], "sessions": [...], "conversations": [[turn, ...], ...]}, where a turn
carries the NEW non-assistant messages of that call as {role, content} (the loader reads nothing else per turn,
python/sglang/benchmark/datasets/agentic_trace.py:85), plus per-turn metadata for the patched client and the analysis:
  prompt_tokens         chat-template token count of the ORIGINAL cumulative input (recorded replies included)
  replay_prompt_tokens  token count of the history the client will actually send at this turn: the kept messages
                        plus one placeholder reply of output_length tokens per earlier turn (the client feeds its
                        generated reply, reasoning included, back as history; serving.py:1329-1331)
  output_length, pre_gap, iteration, source_rows, n_assistant_in_delta, merged_output_length, orig_model, source

Rules (agent_cache/RUNBOOK.md section 3.3):
- assistant messages are dropped: the server regenerates them and the client feeds its reply back;
- a call whose delta has no non-assistant message is merged forward: its pre_gap and output_length are carried
  into the next turn (pre_gap added, output_length reported as merged_output_length);
- tool results become user messages "[TOOL RESULT <function name>]\\n<content>" (--tool-role-mode as_user, default)
  or keep role tool; the name comes from the assistant tool_call with the same id that precedes the result;
- a session is truncated (never dropped) at the first turn whose replay_prompt_tokens + output_length exceeds
  --max-context minus --margin, and emitted only if at least --min-turns turns remain;
- inside a split session_id the dataset clamps pre_gap to 0.0 where the two runs interleave (6 turns), so those
  gaps are unknown, not zero.
Linux only: the rows are read once in the parent and reach the workers through fork, not pickling. Memory: the
parquet expands to about 10 GiB of Python objects plus the tokenizer. Runs inside the sglang container (pyarrow,
transformers and the Qwen tokenizer files are there).
"""
import argparse
import collections
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time
import traceback

import numpy as np
import pyarrow.dataset as ds

_ROWS = None          # parquet rows as dicts; set in the parent before forking
_TOK = None           # tokenizer; loaded in the parent before forking
PLACEHOLDER_TOKEN = " ok"   # one Qwen3 token per repeat: a reply of n tokens is PLACEHOLDER_TOKEN * n


# ------------------------------------------------------------------ loading and grouping ---
def parquet_files(parquet_dir):
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        sys.exit(f"no *.parquet under {parquet_dir}")
    return files


def load_rows(parquet_dir):
    files = parquet_files(parquet_dir)
    rows = ds.dataset(files, format="parquet").to_table().to_pylist()
    for i, r in enumerate(rows):
        r["idx"] = i
        r["source"] = r["session_id"].split("__")[0]
    return rows


def split_chains(rows, idx):
    """Rows of one session_id -> append-only chains of row indices (longest matching chain wins)."""
    chains = []
    for i in sorted(idx, key=lambda i: (len(rows[i]["input"]), i)):
        inp = rows[i]["input"]
        for chain in sorted(chains, key=lambda c: -len(rows[c[-1]]["input"])):
            last = rows[chain[-1]]["input"]
            if len(inp) > len(last) and inp[:len(last)] == last:
                chain.append(i)
                break
        else:
            chains.append([i])
    return chains


def group_sessions(rows):
    """replayable session name -> row indices in replay order."""
    by_sid = {}
    for r in rows:
        by_sid.setdefault(r["session_id"], []).append(r["idx"])
    sessions = {}
    for sid, idx in by_sid.items():
        chains = split_chains(rows=rows, idx=idx)
        for k, chain in enumerate(chains):
            sessions[sid if len(chains) == 1 else f"{sid}__chain{k}"] = chain
    return sessions


# ------------------------------------------------------------------ message handling ---
def clean_msg(m):
    """The original message without None-valued keys, for the chat template."""
    return {k: v for k, v in m.items() if v is not None}


def note_tool_calls(names, m):
    for tc in m["tool_calls"] or []:
        if tc["id"] and tc["function"]["name"]:
            names[tc["id"]] = tc["function"]["name"]


def replay_msg(m, mode, names):
    """The {role, content} the client will send for one recorded message, or None for an assistant message."""
    if m["role"] == "assistant":
        return None
    content = m["content"] or ""
    if m["role"] == "tool" and mode == "as_user":
        return {"role": "user", "content": f"[TOOL RESULT {names.get(m['tool_call_id'] or '', '')}]\n{content}"}
    return {"role": m["role"], "content": content}


def build_turns(rows, idx, mode):
    """Turns of one chain (token counts filled in later)."""
    turns, prev, names = [], [], {}
    carry_gap, carry_out, carry_rows = 0.0, 0, []
    for k, i in enumerate(idx):
        r = rows[i]
        inp = r["input"]
        assert len(inp) > len(prev) and inp[:len(prev)] == prev, f"row {i} does not extend its chain"
        kept = []
        for m in inp[len(prev):]:
            note_tool_calls(names=names, m=m)
            x = replay_msg(m=m, mode=mode, names=names)
            if x is not None:
                kept.append(x)
        n_assistant = sum(1 for m in inp[len(prev):] if m["role"] == "assistant")
        prev = inp
        if not kept:
            carry_gap += r["pre_gap"]
            carry_out += int(r["output_length"])
            carry_rows.append(i)
            continue
        turns.append({
            "messages": kept, "prompt_tokens": None, "replay_prompt_tokens": None,
            "output_length": int(r["output_length"]), "pre_gap": float(r["pre_gap"] + carry_gap),
            "iteration": k, "source_rows": carry_rows + [i], "n_assistant_in_delta": n_assistant,
            "merged_output_length": carry_out, "orig_model": r["model"], "source": r["source"], "_input_idx": i,
        })
        carry_gap, carry_out, carry_rows = 0.0, 0, []
    return turns


# ------------------------------------------------------------------ token counting (workers) ---
def load_tokenizer(name):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # a forked child must not inherit a live Rust thread pool
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def count_tokens(messages):
    text = _TOK.apply_chat_template([clean_msg(m) for m in messages], tokenize=False, add_generation_prompt=True)
    return len(_TOK(text, add_special_tokens=False)["input_ids"])


def convert_session(name, idx, mode, max_context, margin):
    turns = build_turns(rows=_ROWS, idx=idx, mode=mode)
    history = []
    for t in turns:
        t["prompt_tokens"] = count_tokens(_ROWS[t.pop("_input_idx")]["input"])
        history.extend(t["messages"])
        t["replay_prompt_tokens"] = count_tokens(history)
        history.append({"role": "assistant", "content": PLACEHOLDER_TOKEN * t["output_length"]})
    truncated_at = next((k for k, t in enumerate(turns)
                         if t["replay_prompt_tokens"] + t["output_length"] > max_context - margin), None)
    kept = turns if truncated_at is None else turns[:truncated_at]
    r0 = _ROWS[idx[0]]
    meta = {"session_id": name, "source": r0["source"], "orig_model": r0["model"], "n_rows": len(idx),
            "n_turns_total": len(turns), "n_turns": len(kept), "merged_rows": len(idx) - len(turns),
            "truncated_at": truncated_at, "final_prompt_tokens": turns[-1]["prompt_tokens"] if turns else 0,
            "final_replay_prompt_tokens": turns[-1]["replay_prompt_tokens"] if turns else 0,
            "midrun_start": bool(turns) and turns[0]["n_assistant_in_delta"] > 0, "error": None}
    return meta, kept


def _worker(job):
    """One chain -> (meta, turns); a failure is recorded in the meta instead of killing the pool."""
    if _ROWS is None or _TOK is None:
        raise RuntimeError("rows/tokenizer not inherited: create the pool with the fork context after loading them")
    try:
        return convert_session(**job)
    except Exception:
        r0 = _ROWS[job["idx"][0]]
        return ({"session_id": job["name"], "source": r0["source"], "orig_model": r0["model"], "n_rows": len(job["idx"]),
                 "n_turns_total": 0, "n_turns": 0, "merged_rows": 0, "truncated_at": None, "final_prompt_tokens": 0,
                 "final_replay_prompt_tokens": 0, "midrun_start": False, "error": traceback.format_exc()}, [])


# ------------------------------------------------------------------ statistics ---
def mean(values):
    return float(np.mean(values)) if len(values) else None


def pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def gap_stats(gaps):
    g = np.array(gaps, dtype=float)
    return {"n": int(len(g)), "p50": pct(values=g, q=50), "p90": pct(values=g, q=90), "p95": pct(values=g, q=95),
            "p99": pct(values=g, q=99), "mean": mean(g),
            "frac_ge_1s": mean(g >= 1), "frac_ge_5s": mean(g >= 5), "frac_ge_30s": mean(g >= 30)}


def source_stats(metas, emitted, src):
    turns = [t for m, conv in emitted if m["source"] == src for t in conv]
    ok = [m for m in metas if not m["error"]]
    return {
        "chains": len(metas), "chains_emitted": sum(1 for m, _ in emitted if m["source"] == src),
        "rows": sum(m["n_rows"] for m in metas),
        "chains_truncated_all": sum(1 for m in metas if m["truncated_at"] is not None),
        "chains_truncated_emitted": sum(1 for m, _ in emitted if m["source"] == src and m["truncated_at"] is not None),
        "turns_dropped_by_truncation_all": sum(m["n_turns_total"] - m["n_turns"] for m in metas),
        "midrun_starts_emitted": sum(1 for m, _ in emitted if m["source"] == src and m["midrun_start"]),
        "chains_failed": len(metas) - len(ok),
        "final_prompt_tokens_p50": pct(values=[m["final_prompt_tokens"] for m in ok], q=50),
        "final_prompt_tokens_max": max((m["final_prompt_tokens"] for m in ok), default=None),
        "emitted_prompt_tokens_mean": mean([t["prompt_tokens"] for t in turns]),
        "emitted_prompt_tokens_p50": pct(values=[t["prompt_tokens"] for t in turns], q=50),
        "emitted_replay_prompt_tokens_mean": mean([t["replay_prompt_tokens"] for t in turns]),
        "emitted_output_length_mean": mean([t["output_length"] for t in turns]),
        "pre_gap_after_turn0": gap_stats([t["pre_gap"] for t in turns if t["iteration"] > 0]),
    }


def build_stats(rows, sessions, all_meta, emitted, args):
    by_source = collections.defaultdict(list)
    for m in all_meta:
        by_source[m["source"]].append(m)
    all_turns = [t for _, conv in emitted for t in conv]
    later = [t for t in all_turns if t["iteration"] > 0]
    first_rows = {idx[0] for idx in sessions.values()}
    raw_gaps = [r["pre_gap"] for r in rows if r["idx"] not in first_rows]
    sids = collections.Counter(name.split("__chain")[0] for name in sessions)
    invariants = {
        "parquet_rows": len(rows), "parquet_session_ids": len(sids),
        "session_ids_with_several_chains": sum(1 for v in sids.values() if v > 1), "chains": len(sessions),
        "selected_chains": len(all_meta), "selected_rows": sum(m["n_rows"] for m in all_meta),
        "chains_failed": sum(1 for m in all_meta if m["error"]),
        "turns_plus_merged_equals_rows": all(m["n_turns_total"] + m["merged_rows"] == m["n_rows"] for m in all_meta if not m["error"]),
        "turn0_pre_gap_zero": all(conv[0]["pre_gap"] == 0.0 for _, conv in emitted),
        "turn0_starts_with_system": all(conv[0]["messages"][0]["role"] == "system" for _, conv in emitted),
        "empty_messages_turns": sum(1 for t in all_turns if not t["messages"]),
        "rows_merged_forward": sum(m["merged_rows"] for m in all_meta),
        "replay_within_context": all(t["replay_prompt_tokens"] + t["output_length"] <= args.max_context - args.margin for t in all_turns),
    }
    return {
        "args": vars(args), "invariants": invariants,
        "chains_emitted": len(emitted), "turns_emitted": len(all_turns),
        "chains_below_min_turns": sum(1 for m in all_meta if not m["error"] and m["n_turns"] < args.min_turns),
        "midrun_starts_emitted": sum(1 for m, _ in emitted if m["midrun_start"]),
        "turns_without_assistant_after_turn0": sum(1 for t in later if t["n_assistant_in_delta"] == 0),
        "pre_gap_raw_all_rows_excluding_first": gap_stats(raw_gaps),
        "pre_gap_emitted_after_turn0": gap_stats([t["pre_gap"] for t in later]),
        "output_length_emitted": {"mean": mean([t["output_length"] for t in all_turns]),
                                  "p50": pct(values=[t["output_length"] for t in all_turns], q=50),
                                  "p95": pct(values=[t["output_length"] for t in all_turns], q=95)},
        "prompt_tokens_emitted": {"mean": mean([t["prompt_tokens"] for t in all_turns]),
                                  "p50": pct(values=[t["prompt_tokens"] for t in all_turns], q=50),
                                  "p95": pct(values=[t["prompt_tokens"] for t in all_turns], q=95)},
        "replay_prompt_tokens_emitted": {"mean": mean([t["replay_prompt_tokens"] for t in all_turns]),
                                         "p50": pct(values=[t["replay_prompt_tokens"] for t in all_turns], q=50),
                                         "p95": pct(values=[t["replay_prompt_tokens"] for t in all_turns], q=95)},
        "replay_over_recorded_ratio_mean": mean([t["replay_prompt_tokens"] / t["prompt_tokens"] for t in all_turns if t["prompt_tokens"]]),
        "per_source": {src: source_stats(metas=metas, emitted=emitted, src=src) for src, metas in by_source.items()},
    }


# ------------------------------------------------------------------ main ---
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet-dir", default="/tmp/lmcache")
    ap.add_argument("--out", default="/sgl-workspace/sglang/agent_cache/traces/lmcache_agentic_trace.json")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-32B-FP8")
    ap.add_argument("--revision", default="main", help="dataset revision the parquet files were downloaded from")
    ap.add_argument("--tool-role-mode", choices=["as_user", "keep"], default="as_user")
    ap.add_argument("--sources", default=None, help="comma list of session_id prefixes to keep, e.g. swebench,gaia")
    ap.add_argument("--max-sessions", type=int, default=0, help="0 = all")
    ap.add_argument("--max-context", type=int, default=32768, help="the server's --context-length")
    ap.add_argument("--margin", type=int, default=512, help="headroom below --max-context")
    ap.add_argument("--min-turns", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None, help="shuffle session order with this seed (default: file order)")
    ap.add_argument("--workers", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    args = ap.parse_args()
    if args.min_turns < 1:
        sys.exit("--min-turns must be >= 1: the loader drops empty conversations silently")
    return args


def select_sessions(sessions, args):
    selected = list(sessions)
    if args.sources:
        keep = set(args.sources.split(","))
        selected = [s for s in selected if s.split("__")[0] in keep]
    if args.seed is not None:
        random.Random(args.seed).shuffle(selected)
    if args.max_sessions:
        selected = selected[:args.max_sessions]
    if not selected:
        sys.exit(f"no sessions selected (sources={args.sources!r}); known: {sorted({s.split('__')[0] for s in sessions})}")
    return selected


def print_summary(stats, out, t0):
    print(json.dumps({k: v for k, v in stats.items() if k != "per_source"}, indent=1))
    for src, s in stats["per_source"].items():
        g = s["pre_gap_after_turn0"]
        print(f"{src}: chains {s['chains']} emitted {s['chains_emitted']} truncated {s['chains_truncated_emitted']} "
              f"midrun-starts {s['midrun_starts_emitted']} | final prompt_tokens p50 {fmt(s['final_prompt_tokens_p50'], '.0f')} "
              f"max {s['final_prompt_tokens_max']} | emitted prompt_tokens mean {fmt(s['emitted_prompt_tokens_mean'], '.0f')} "
              f"replay {fmt(s['emitted_replay_prompt_tokens_mean'], '.0f')} | pre_gap p50 {fmt(g['p50'], '.2f')}s "
              f"mean {fmt(g['mean'], '.2f')}s frac>=5s {fmt(g['frac_ge_5s'], '.3f')}")
    print(f"wrote {out} ({os.path.getsize(out) / 2**20:.0f} MiB) and {out}.stats.json in {time.time() - t0:.0f}s")


def main():
    global _ROWS, _TOK
    args = parse_args()
    if "fork" not in mp.get_all_start_methods():
        sys.exit("convert_lmcache.py needs the fork start method (Linux): workers inherit the rows through fork")
    t0 = time.time()
    _TOK = load_tokenizer(args.tokenizer)
    rows = load_rows(args.parquet_dir)
    sessions = group_sessions(rows)
    print(f"loaded {len(rows)} rows -> {len(sessions)} replayable sessions in {time.time() - t0:.0f}s", flush=True)
    selected = select_sessions(sessions=sessions, args=args)

    _ROWS = rows
    jobs = [{"name": s, "idx": sessions[s], "mode": args.tool_role_mode, "max_context": args.max_context,
             "margin": args.margin} for s in selected]
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(_worker, jobs, chunksize=4)
    print(f"converted {len(results)} sessions in {time.time() - t0:.0f}s", flush=True)

    all_meta = [m for m, _ in results]
    failed = [m for m in all_meta if m["error"]]
    for m in failed:
        print(f"FAILED {m['session_id']}:\n{m['error']}", file=sys.stderr)
    emitted = [(m, conv) for m, conv in results if len(conv) >= args.min_turns]
    if not emitted:
        sys.exit("no session has >= --min-turns turns; nothing written")
    out = {
        "metadata": {
            "source": "huggingface.co/datasets/sammshen/lmcache-agentic-traces", "revision": args.revision,
            "parquet_dir": args.parquet_dir,
            "parquet_files": {os.path.basename(f): os.path.getsize(f) for f in parquet_files(args.parquet_dir)}, "tokenizer": args.tokenizer, "tool_role_mode": args.tool_role_mode,
            "max_context": args.max_context, "margin": args.margin, "min_turns": args.min_turns, "seed": args.seed,
            "n_sessions": len(emitted), "n_turns": sum(len(c) for _, c in emitted),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_tokens_note": "recorded: chat-template tokens of the original cumulative input; replay_prompt_tokens: "
                                  "tokens of the history the client sends (kept messages + placeholder replies of output_length)",
        },
        "session_ids": [m["session_id"] for m, _ in emitted],
        "sessions": [m for m, _ in emitted],
        "conversations": [conv for _, conv in emitted],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    stats = build_stats(rows=rows, sessions=sessions, all_meta=all_meta, emitted=emitted, args=args)
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print_summary(stats=stats, out=args.out, t0=t0)
    if failed:
        sys.exit(f"{len(failed)} chains failed (see stderr); the trace was written without them")


if __name__ == "__main__":
    main()
