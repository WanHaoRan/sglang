"""Convert the FreeInference Agentic Trace (real users behind 14 agent harnesses, gateway logs) to a program trace.

  python3 convert_freeinference.py --input DIR --weeks 2026-07-12_2026-07-18 [...] --output fi.programs.jsonl.gz

Input: harvardMadsys/freeinference_agentic_trace (CC BY 4.0) under DIR: release/<week>/traces/<day>.jsonl.zst and
release/<week>/manifest.json. Each line is one session; a subagent's whole session is nested under the tool call that
launched it. Decompresses with the zstd command; every file is checked against its manifest entry (bytes, lines,
sha256).

One tree per session. The session is the root stream; each nested subagent session is a stream whose first request
depends on the request that launched it ("spawn") and which the launching stream's next request waits for when the
subagent ended before it (the join). Requests in a stream depend on their predecessor, and the wait is the gateway gap
from its end to the next start (the card's latency_ms, which equals it for single-tool steps). "gap" is the request's
step_trigger: "tool" (a tool result) or "user" (a human message). Prompts keep the chained 16-token block ids, scoped to
the week as the card prescribes, so sharing across sessions of the week (system prompts, tool definitions) reappears on
the wire; "tail" restores the dropped partial block (the o200k role-token total minus 16 per block). A request's key is
its response's tool call with the longest latency_ms: the tool name and the first shell program ("bash/git").
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from program_trace import (delta, hint_key, iter_trees, make_header, quantiles, request, split_programs,
                           tree_summary, write_trace)

BLOCK_TOKENS = 16


def read_week(root: str, week: str, stats: dict):
    """Sessions of one week, in file order, each file checked against the manifest."""
    wdir = os.path.join(root, "release", week)
    with open(os.path.join(wdir, "manifest.json")) as f:
        manifest = json.load(f)
    if manifest.get("block_tokens") != BLOCK_TOKENS:
        raise SystemExit(f"STOP: {week} manifest block_tokens {manifest.get('block_tokens')}, expected {BLOCK_TOKENS}")
    for name, meta in sorted(manifest["files"].items()):
        path = os.path.join(wdir, name + ".zst")
        if not os.path.exists(path):
            stats["missing_files"].append(f"{week}/{name}.zst")
            continue
        sha, nbytes, nlines = hashlib.sha256(), 0, 0
        proc = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        for raw in proc.stdout:
            sha.update(raw)
            nbytes += len(raw)
            nlines += 1
            yield week, json.loads(raw)
        if proc.wait() != 0:
            raise SystemExit(f"STOP: zstd failed on {path}")
        if (nbytes, nlines, sha.hexdigest()) != (meta["bytes"], meta["lines"], meta["sha256"]):
            raise SystemExit(f"STOP: {path} does not match its manifest entry")
        stats["files_verified"] += 1


def _key(req: dict) -> str | None:
    calls = req.get("tool_calls") or []
    if not calls:
        return None
    best = max(calls, key=lambda c: (c.get("latency_ms") or 0, -int(c.get("call_index") or 0)))
    programs = best.get("programs") or []
    return hint_key(best.get("name"), programs[0] if programs else None)


def _flatten(session: dict, *, streams: list, parent: tuple | None, depth: int) -> None:
    """Append the session's stream, then its subagents' streams, recursively: [{"src": [...], "parent": ...}]."""
    s = len(streams)
    streams.append({"src": session["requests"], "parent": parent, "depth": depth, "joins": {}})
    for k, r in enumerate(session["requests"]):
        for tc in r.get("tool_calls") or []:
            sub = tc.get("subagent")
            if sub and sub.get("requests"):
                child = len(streams)
                _flatten(sub, streams=streams, parent=(s, k), depth=depth + 1)
                streams[s]["joins"].setdefault(k + 1, []).append(child)


def build_tree(week: str, session: dict) -> dict:
    flat: list = []
    _flatten(session, streams=flat, parent=None, depth=0)
    t0 = min(r["start_ms"] for st in flat for r in st["src"])
    order = sorted((r["start_ms"], s, k) for s, st in enumerate(flat) for k, r in enumerate(st["src"]))
    out_streams = [{"name": "main" if s == 0 else f"sa:{s:03d}", "role": "main" if s == 0 else "subagent",
                    "model": st["src"][0].get("model"), "requests": [None] * len(st["src"])}
                   for s, st in enumerate(flat)]
    recent: list = []
    for _, s, k in order:
        r = flat[s]["src"][k]
        deps = _deps(flat, s, k)
        ready = max((flat[ds]["src"][dk]["end_ms"] for ds, dk in deps), default=t0)
        blocks = r.get("block_ids") or []
        cands = [(list(d), flat[d[0]]["src"][d[1]].get("block_ids") or []) for d in deps] + recent[-8:]
        base, keep, new = delta(blocks, cands)
        tail = sum((r.get("role_tokens") or {}).values()) - BLOCK_TOKENS * len(blocks)
        gap = "spawn" if (k == 0 and s > 0) else (r.get("step_trigger") if k > 0 else None)
        pt = r.get("provider_tokens") or {}
        out_streams[s]["requests"][k] = request(
            t=(r["start_ms"] - t0) / 1000, dur=(r["end_ms"] - r["start_ms"]) / 1000, deps=[list(d) for d in deps],
            wait=(r["start_ms"] - ready) / 1000, out=int(pt.get("completion") or 1), gap=gap, key=_key(r),
            base=base, keep=keep, new=new, tail=max(0, min(BLOCK_TOKENS - 1, tail)),
            ttft=None if r.get("ttft_ms") is None else r["ttft_ms"] / 1000,
            mutation=(r.get("mutation") or {}).get("transition"))
        recent.append(([s, k], blocks))
    return {"kind": "tree", "tree": session["session_id"], "source": "freeinference",
            "system": [session.get("harness"), out_streams[0]["model"]], "user": session.get("user_id"),
            "harness": session.get("harness"), "week": week, "scope": week, "setup": 0, "streams": out_streams}


def _deps(flat: list, s: int, k: int) -> list:
    st = flat[s]
    deps = [(s, k - 1)] if k > 0 else ([st["parent"]] if st["parent"] is not None else [])
    start = st["src"][k]["start_ms"]
    for child in st["joins"].get(k, []):
        last = len(flat[child]["src"]) - 1
        if flat[child]["src"][last]["end_ms"] <= start:   # the subagent finished before this request: a join
            deps.append((child, last))
    return sorted(set(deps))


def convert(args: argparse.Namespace) -> dict:
    stats = {"weeks": args.weeks, "files_verified": 0, "missing_files": [], "sessions": 0, "harness": {}}

    def trees():
        for week in args.weeks:
            for wk, session in read_week(args.input, week, stats):
                if args.limit and stats["sessions"] >= args.limit:
                    return
                stats["sessions"] += 1
                h = session.get("harness")
                stats["harness"][h] = stats["harness"].get(h, 0) + 1
                yield build_tree(wk, session)

    header = make_header(source="freeinference", content="blocks", block_tokens=BLOCK_TOKENS,
                         converter={"script": os.path.basename(__file__), "weeks": args.weeks, "limit": args.limit,
                                    "created": time.strftime("%Y-%m-%d")})
    write_trace(args.output, header=header, trees=trees())
    return stats


def describe(path: str) -> dict:
    """Shape of a converted trace, plus the prompt share that earlier sessions of the week already hold (file order):
    with chained ids that share is a prefix, found by binary search on the blocks seen before the session."""
    out = {"trees": 0, "streams": 0, "requests": 0, "prompt_tokens": 0, "output_tokens": 0, "programs_mori": 0}
    waits = {"tool": [], "user": [], "spawn": []}
    seen: dict = {}            # week scope -> block ids of the sessions before
    block_refs = cross_refs = 0
    from program_trace import materialize_blocks
    for tree in iter_trees(path):
        s = tree_summary(tree, content="blocks", block_tokens=BLOCK_TOKENS)
        for key in ("streams", "requests", "prompt_tokens", "output_tokens"):
            out[key] += s[key]
        out["trees"] += 1
        for st in tree["streams"]:
            for r in st["requests"]:
                if r["gap"] in waits:
                    waits[r["gap"]].append(r["wait"])
        before = seen.setdefault(tree["scope"], set())
        for blocks in materialize_blocks(tree).values():
            block_refs += len(blocks)
            lo, hi = 0, len(blocks)
            while lo < hi:   # longest prefix whose last block an earlier session held
                mid = (lo + hi + 1) // 2
                if blocks[mid - 1] in before:
                    lo = mid
                else:
                    hi = mid - 1
            cross_refs += lo
        before.update(b for st in tree["streams"] for r in st["requests"] for b in r["new"])
        out["programs_mori"] += len(split_programs(tree, idle_s=1800.0, compaction=True, content="blocks"))
    out["wait_s"] = {k: quantiles(v) for k, v in waits.items()}
    out["waits_n"] = {k: len(v) for k, v in waits.items()}
    out["prompt_share_held_by_earlier_sessions"] = round(cross_refs / max(1, block_refs), 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="dataset root (holds release/<week>/...)")
    ap.add_argument("--weeks", nargs="+", required=True, help="week folders, e.g. 2026-07-12_2026-07-18")
    ap.add_argument("--output", required=True, help="program trace (.jsonl or .jsonl.gz)")
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
