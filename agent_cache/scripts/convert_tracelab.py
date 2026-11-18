"""Convert TraceLab v0.0.2 (Claude Code and Codex client logs) to a program trace.

  python3 convert_tracelab.py --input DIR --output tracelab.programs.jsonl.gz [--provider claude|codex] [--limit N]

Input: the three Parquet tables of UW-SyFI/TraceLab v0.0.2 (CC BY 4.0) under DIR: rounds/train.parquet,
tool_calls/train.parquet and timing_events/train.parquet. Needs pyarrow.

One tree per session, one stream, one request per round in (round_index, round_pk) order. Content follows TraceLab's
session_runner (lengths): "prefix" is the provider's cached prefix, capped at the previous round's input plus output as
the dataset card advises (TraceLab issue #22), and "append" the rest of the round's input, so a prompt is as long as the
recorded input. Two waits are kept, because the literature uses both:
  wait_tool  the round's summed tool wall latency after the previous round, TraceLab's canonical tool_wait_after_ms
             (csv_export); the session_runner sleeps only this.
  wait       wait_tool plus the human waits in the round, TraceLab's definition (previous event of any kind to each
             user_message); MORI sleeps "the recorded tool-call or human-wait duration".
"t" is rebuilt from the waits and the recorded round durations (last input event to last output event). A round's key is
its response's tool call with the longest wall latency: the tool name and the first executable ("Bash/git").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from program_trace import iter_trees, hint_key, make_header, quantiles, request, split_programs, write_trace

INPUT_EVENTS = {"user_message", "tool_result"}


def _epoch_s(ts) -> float | None:
    return None if ts is None else ts.timestamp()


def load_tables(root: str, provider: str | None):
    import pyarrow.parquet as pq

    rounds = pq.read_table(os.path.join(root, "rounds", "train.parquet"),
                           columns=["round_pk", "provider", "session_id", "round_index", "model", "input_tokens_total",
                                    "prefix_tokens", "output_tokens", "first_input_event_type"]).to_pylist()
    if provider:
        rounds = [r for r in rounds if r["provider"] == provider]
    tools = pq.read_table(os.path.join(root, "tool_calls", "train.parquet"),
                          columns=["round_pk", "tool_index", "tool_name", "tool_wall_latency_ms",
                                   "executables"]).to_pydict()
    events = pq.read_table(os.path.join(root, "timing_events", "train.parquet"),
                           columns=["round_pk", "event_index", "event_type", "timestamp"]).to_pydict()
    return rounds, tools, events


def index_tools(tools: dict) -> dict:
    """round_pk -> [(tool_index, name, wall latency s or None, first executable or None)]."""
    out = defaultdict(list)
    for pk, i, name, lat, exe in zip(tools["round_pk"], tools["tool_index"], tools["tool_name"],
                                     tools["tool_wall_latency_ms"], tools["executables"]):
        out[pk].append((i, name, None if lat is None else lat / 1000.0, exe[0] if exe else None))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def index_events(events: dict) -> dict:
    """round_pk -> [(event_index, type, epoch s)] in event order."""
    out = defaultdict(list)
    for pk, i, typ, ts in zip(events["round_pk"], events["event_index"], events["event_type"], events["timestamp"]):
        out[pk].append((i, typ, _epoch_s(ts)))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def round_key(calls: list) -> str | None:
    """The key of the call that sets the gap: the longest wall latency, the first call on ties or without latencies."""
    if not calls:
        return None
    best = max(calls, key=lambda c: (c[2] or 0.0, -c[0]))
    return hint_key(best[1], best[3])


def build_tree(rounds: list, *, tools: dict, events: dict, stats: dict) -> dict:
    rounds = sorted(rounds, key=lambda r: (r["round_index"], r["round_pk"]))
    prev_event_ts = None    # TraceLab's human-wait walk: the previous event of any kind in the session
    reqs, t = [], 0.0
    prev = None
    for k, r in enumerate(rounds):
        evs = events.get(r["round_pk"], [])
        human = 0.0
        # TraceLab's walk (timing.human_waits_from_event_pairs): a round's events in time order, the previous event
        # carried across rounds; every positive previous-event -> user_message gap is a human wait
        for _, typ, ts in sorted((e for e in evs if e[2] is not None), key=lambda e: (e[2], e[0])):
            if typ == "user_message" and prev_event_ts is not None and ts - prev_event_ts > 0:
                human += ts - prev_event_ts
                stats["human_waits"].append(ts - prev_event_ts)
            prev_event_ts = ts
        prev_calls = tools.get(prev["round_pk"], []) if prev is not None else []
        wait_tool = sum(c[2] for c in prev_calls if c[2] is not None and c[2] >= 0)
        stats["tool_latencies"] += [c[2] for c in prev_calls if c[2] is not None and c[2] > 0]
        wait = (wait_tool + human) if k else 0.0
        cap = None if prev is None else prev["input_tokens_total"] + prev["output_tokens"]
        prefix = max(0, int(r["prefix_tokens"] or 0))
        if cap is not None and prefix > cap:
            prefix = cap
            stats["prefix_capped"] += 1
        total = max(1, int(r["input_tokens_total"] or 0))
        prefix = min(prefix, total)
        dur = _round_duration(evs)
        t = t + wait if k else 0.0
        gap = None if k == 0 else ("user" if r["first_input_event_type"] == "user_message" or human > 0 else
                                   ("tool" if r["first_input_event_type"] == "tool_result" else None))
        reqs.append(request(t=t, dur=dur, deps=[] if k == 0 else [[0, k - 1]], wait=wait,
                            out=int(r["output_tokens"] or 1), gap=gap, key=round_key(tools.get(r["round_pk"], [])),
                            prefix=prefix, append=total - prefix,
                            wait_tool=round(wait_tool if k else 0.0, 3), wait_human=round(human if k else 0.0, 3)))
        t += dur
        prev = r
        stats["wall_s"] += dur + (wait if k else 0.0)
        stats["human_s"] += human if k else 0.0
    harness = "claude-code" if rounds[0]["provider"] == "claude" else rounds[0]["provider"]
    return {"kind": "tree", "tree": rounds[0]["session_id"], "source": "tracelab",
            "system": [harness, rounds[0]["model"]], "scope": rounds[0]["session_id"],
            "streams": [{"name": "main", "role": "main", "model": rounds[0]["model"], "requests": reqs}]}


def _round_duration(evs: list) -> float:
    """Last input event to last output event of a round: the recorded LLM time (0 when either is missing)."""
    ins = [ts for _, typ, ts in evs if typ in INPUT_EVENTS and ts is not None]
    outs = [ts for _, typ, ts in evs if typ not in INPUT_EVENTS and ts is not None]
    if not ins or not outs:
        return 0.0
    return max(0.0, max(outs) - max(ins))


def convert(args: argparse.Namespace) -> dict:
    rounds, tools, events = load_tables(args.input, args.provider)
    tool_idx, ev_idx = index_tools(tools), index_events(events)
    by_session = defaultdict(list)
    for r in rounds:
        by_session[r["session_id"]].append(r)
    order = sorted(by_session, key=lambda s: min(r["round_pk"] for r in by_session[s]))   # first appearance
    if args.limit:
        order = order[: args.limit]
    stats = {"sessions": len(order), "rounds": sum(len(by_session[s]) for s in order), "prefix_capped": 0,
             "wall_s": 0.0, "human_s": 0.0, "human_waits": [], "tool_latencies": []}

    def trees():
        for sid in order:
            yield build_tree(by_session[sid], tools=tool_idx, events=ev_idx, stats=stats)

    header = make_header(source="tracelab", content="lengths", block_tokens=None,
                         converter={"script": os.path.basename(__file__), "provider": args.provider,
                                    "limit": args.limit, "created": time.strftime("%Y-%m-%d")})
    write_trace(args.output, header=header, trees=trees())
    hw, tl = stats.pop("human_waits"), stats.pop("tool_latencies")
    stats["human_share_of_wall"] = round(stats["human_s"] / stats["wall_s"], 4) if stats["wall_s"] else None
    stats["human_wait_min"] = {k: round(v / 60, 2) for k, v in quantiles(hw).items()}
    stats["tool_latency_s"] = {k: round(v, 2) for k, v in quantiles(tl).items()}
    stats["wall_s"], stats["human_s"] = round(stats["wall_s"]), round(stats["human_s"])
    return stats


def describe(path: str) -> dict:
    out = {"trees": 0, "requests": 0, "prompt_tokens": 0, "output_tokens": 0, "programs_mori": 0}
    waits, waits_tool = [], []
    for tree in iter_trees(path):
        reqs = tree["streams"][0]["requests"]
        out["trees"] += 1
        out["requests"] += len(reqs)
        out["prompt_tokens"] += sum(r["prefix"] + r["append"] for r in reqs)
        out["output_tokens"] += sum(r["out"] for r in reqs)
        waits += [r["wait"] for r in reqs[1:]]
        waits_tool += [r["wait_tool"] for r in reqs[1:]]
        out["programs_mori"] += len(split_programs(tree, idle_s=1800.0, compaction=True, content="lengths"))
    out["wait_s"] = quantiles(waits)
    out["wait_tool_s"] = quantiles(waits_tool)
    out["waits_over"] = {f"{x}s": round(sum(w > x for w in waits) / max(1, len(waits)), 4) for x in (60, 300, 1800)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="directory with rounds/, tool_calls/, timing_events/")
    ap.add_argument("--output", required=True, help="program trace (.jsonl or .jsonl.gz)")
    ap.add_argument("--provider", choices=("claude", "codex"), default=None, help="keep one provider's sessions")
    ap.add_argument("--limit", type=int, default=0, help="convert the first N sessions only (0 = all)")
    args = ap.parse_args()
    stats = convert(args)
    stats["converted"] = describe(args.output)
    with open(args.output + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
