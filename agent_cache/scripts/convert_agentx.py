"""Convert the SemiAnalysis Claude Code corpus (Weka trace format; "AgentX" in MORI and InferenceX) to a program trace.

  python3 convert_agentx.py --input traces.jsonl --output agentx.programs.jsonl.gz [--limit N]

Input: semianalysisai/cc-traces-weka-062126 traces.jsonl (Apache-2.0): one trace per line, {"id", "block_size": 64,
"hash_id_scope": "local", "requests": [...]}; a request is {"t", "model", "in", "out", "hash_ids", "api_time",
"think_time", "ttft", "type": "s"|"n"} or a subagent entry {"type": "subagent", "agent_id", "requests": [...]} whose
inner requests carry root-absolute "t" (AIPerf's "absolute" basis for this publisher).

The top-level stream interleaves the main agent with side calls (title and summary one-shots, forks of the main context,
retries) and context edits, as AIPerf's Weka loader documents. The same chain detection splits it here, and inside each
subagent entry, by hash-prefix evidence: a request continues the chain whose last request it extends (same model, that
request finished), or, failing that, the chain it shares a prefix with when that chain's old state is never extended
again (a compaction or context edit); anything else starts a chain of its own (a fork when it shares a prefix with an
earlier request, a side chain when it shares none). The chain with the most requests is the root. Every request depends
on its chain's previous one; a chain's first request on the request it forked from, or else the latest request that had
finished when it started. A subagent's chains depend on the latest root request finished at its start (the spawn) and
the first root request at or after its end waits for all of them (the join); a subagent no later root request reaches
runs in the background. Waits are recorded end-to-start gaps, InferenceX's replay delays; prompts keep the recorded
block ids, which are chained and scoped to the trace. The corpus names no tools, so keys are null (MORI: "serving
without tool information").
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from program_trace import (delta, iter_trees, lcp, make_header, materialize_blocks, quantiles, request,
                           split_programs, tree_summary, write_trace)

BLOCK_TOKENS = 64
EPS_S = 0.05   # arbitrary: timestamp jitter tolerated when ordering a request after another's end


def _end(r: dict) -> float:
    return r["t"] + float(r.get("api_time") or 0.0)


def _extends(r: dict, tail: dict) -> bool:
    n = len(tail["hash_ids"])
    return lcp(r["hash_ids"], tail["hash_ids"]) >= max(1, n - 1)


def _extended_later(reqs: list, q: int, after: int) -> bool:
    """Whether a request after index `after` extends request q's state (its prefix up to q's last block but one)."""
    hq, model = reqs[q]["hash_ids"], reqs[q]["model"]
    n = len(hq)
    if n <= 1:
        return True
    pos, bid = n - 2, hq[n - 2]
    return any(len(reqs[j]["hash_ids"]) > pos and reqs[j]["hash_ids"][pos] == bid and reqs[j]["model"] == model
               for j in range(after + 1, len(reqs)))


def detect_chains(reqs: list) -> list:
    """Chains of request indices (reqs in time order) and their fork parents: [(indices, fork_parent or None)]."""
    chains: list = []   # [indices, fork_parent]
    for i, r in enumerate(reqs):
        ready = [c for c in chains if reqs[c[0][-1]]["model"] == r["model"] and _end(reqs[c[0][-1]]) <= r["t"] + EPS_S]
        ext = [c for c in ready if _extends(r, reqs[c[0][-1]])]
        if ext:
            max(ext, key=lambda c: (lcp(r["hash_ids"], reqs[c[0][-1]]["hash_ids"]), _end(reqs[c[0][-1]])))[0].append(i)
            continue
        related = [(lcp(r["hash_ids"], reqs[c[0][-1]]["hash_ids"]), c) for c in ready]
        related = [(n, c) for n, c in related if n > 0]
        if related:
            _, best = max(related, key=lambda x: x[0])
            if not _extended_later(reqs, best[0][-1], i):   # the old state is dead: a compaction or context edit
                best[0].append(i)
                continue
        parent = max(range(i), key=lambda j: lcp(r["hash_ids"], reqs[j]["hash_ids"]), default=None)
        if parent is not None and lcp(r["hash_ids"], reqs[parent]["hash_ids"]) == 0:
            parent = None
        chains.append([[i], parent])
    return [(c[0], c[1]) for c in chains]


class _Builder:
    """Accumulates the streams of one tree: (stream, index) addresses for the source requests and their deps."""

    def __init__(self) -> None:
        self.streams: list = []      # [{"name", "role", "model", "src": [source request dicts], "deps": [...]}]
        self.addr: dict = {}         # id(source request) -> (stream, index)

    def add_stream(self, *, name: str, role: str, src: list) -> int:
        self.streams.append({"name": name, "role": role, "model": src[0]["model"], "src": src,
                             "deps": [[] for _ in src]})
        s = len(self.streams) - 1
        for k, r in enumerate(src):
            self.addr[id(r)] = (s, k)
        return s

    def latest_finished(self, t: float, *, among: list) -> tuple | None:
        """The request among `among` that finished last before time t (started before t, too)."""
        done = [r for r in among if r["t"] < t and _end(r) <= t + EPS_S]
        return self.addr[id(max(done, key=_end))] if done else None


def build_tree(trace: dict) -> dict:
    top = sorted((r for r in trace["requests"] if r.get("type") != "subagent"), key=lambda r: r["t"])
    top = [r for r in top if r.get("hash_ids")]
    subs = [e for e in trace["requests"] if e.get("type") == "subagent"]
    b = _Builder()
    chains = sorted(detect_chains(top), key=lambda c: (-len(c[0]), c[0][0]))
    roles = {}
    for n, (idx, parent) in enumerate(chains):
        role = "main" if n == 0 else ("fork" if parent is not None else "side")
        s = b.add_stream(name="main" if n == 0 else f"{role}:{n:03d}", role=role, src=[top[i] for i in idx])
        roles[s] = (role, None if parent is None else top[parent])
    for s in list(roles):
        _chain_deps(b, s, fork_parent=roles[s][1], trigger_pool=top)
    # spawn and join points: the main model's top-level requests, whichever chain the main thread is on
    parents = [r for r in top if r["model"] == b.streams[0]["model"]]
    for e in sorted(subs, key=lambda e: e["t"]):
        _add_subagent(b, e, parents=parents)
    return _finish(trace, b)


def _chain_deps(b: _Builder, s: int, *, fork_parent: dict | None, trigger_pool: list) -> None:
    st = b.streams[s]
    for k in range(1, len(st["src"])):
        st["deps"][k].append((s, k - 1))
    first = st["src"][0]
    if s == 0:
        return
    if fork_parent is not None and _end(fork_parent) <= first["t"] + EPS_S:
        st["deps"][0].append(b.addr[id(fork_parent)])
        return
    trig = b.latest_finished(first["t"], among=[r for r in trigger_pool if r is not first])
    if trig is not None and trig[0] != s:
        st["deps"][0].append(trig)


def _add_subagent(b: _Builder, entry: dict, *, parents: list) -> None:
    inner = sorted((r for r in entry.get("requests") or [] if r.get("hash_ids")), key=lambda r: r["t"])
    if not inner:
        return
    spawn = b.latest_finished(inner[0]["t"], among=parents)
    ends = []
    for n, (idx, parent) in enumerate(sorted(detect_chains(inner), key=lambda c: c[0][0])):
        name = f"sa:{entry.get('agent_id')}" + ("" if n == 0 else f":{n:03d}")
        s = b.add_stream(name=name, role="subagent", src=[inner[i] for i in idx])
        for k in range(1, len(idx)):
            b.streams[s]["deps"][k].append((s, k - 1))
        first = inner[idx[0]]
        fork = None if parent is None else inner[parent]
        # a later chain follows the subagent's own progress, as a top-level one follows the session's (_chain_deps)
        trig = None if n == 0 else b.latest_finished(
            first["t"], among=[r for r in inner if r is not first and id(r) in b.addr])
        if fork is not None and _end(fork) <= first["t"] + EPS_S and id(fork) in b.addr:
            b.streams[s]["deps"][0].append(b.addr[id(fork)])
        elif trig is not None:
            b.streams[s]["deps"][0].append(trig)
        elif spawn is not None:
            b.streams[s]["deps"][0].append(spawn)
        ends.append((s, len(idx) - 1))
    sub_end = max(_end(r) for r in inner)
    join = next((r for r in parents if r["t"] >= sub_end - EPS_S), None)
    if join is not None:
        js, jk = b.addr[id(join)]
        b.streams[js]["deps"][jk].extend(ends)


def _finish(trace: dict, b: _Builder) -> dict:
    """Times relative to the tree's first request, deps and waits, and block deltas against the best earlier request."""
    t0 = min(r["t"] for st in b.streams for r in st["src"])
    order = sorted(((r["t"], s, k) for s, st in enumerate(b.streams) for k, r in enumerate(st["src"])))
    streams = [{"name": st["name"], "role": st["role"], "model": st["model"], "requests": [None] * len(st["src"])}
               for st in b.streams]
    recent: list = []   # (ref, hash_ids) of the latest requests, for cross-stream sharing
    for _, s, k in order:
        r = b.streams[s]["src"][k]
        deps = sorted(set(b.streams[s]["deps"][k]))
        ready = max((_end(b.streams[ds]["src"][dk]) for ds, dk in deps), default=t0)
        cands = [(list(d), b.streams[d[0]]["src"][d[1]]["hash_ids"]) for d in deps] + recent[-8:]
        base, keep, new = delta(r["hash_ids"], cands)
        streams[s]["requests"][k] = request(
            t=r["t"] - t0, dur=float(r.get("api_time") or 0.0), deps=[list(d) for d in deps], wait=r["t"] - ready,
            out=int(r["out"]), gap=_gap_kind(b, s, deps), key=None, base=base, keep=keep, new=new, tail=0,
            ttft=r.get("ttft"), think=r.get("think_time"))
        recent.append(([s, k], r["hash_ids"]))
    main = [r["hash_ids"] for r in b.streams[0]["src"]]
    setup = min((lcp(main[0], h) for h in main[1:]), default=len(main[0]) - 1)
    models = sorted({st["model"] for st in streams})
    return {"kind": "tree", "tree": trace["id"], "source": "agentx", "system": ["claude-code", streams[0]["model"]],
            "models": models, "scope": trace["id"], "setup": max(0, min(setup, len(main[0]) - 1)),
            "streams": streams}


def _gap_kind(b: _Builder, s: int, deps: list) -> str | None:
    """spawn: a subagent stream waiting on the session; join: a session stream waiting on subagents."""
    sub = b.streams[s]["role"] == "subagent"
    if any((b.streams[ds]["role"] == "subagent") != sub for ds, _ in deps):
        return "spawn" if sub else "join"
    return None


def _check_chained(trace: dict) -> int:
    """Block ids that sit at two different positions (0 for a chained trace)."""
    pos: dict = {}
    bad = 0
    reqs = [r for r in trace["requests"] if r.get("type") != "subagent"]
    reqs += [q for e in trace["requests"] if e.get("type") == "subagent" for q in e.get("requests") or []]
    for r in reqs:
        for p, h in enumerate(r.get("hash_ids") or []):
            bad += pos.setdefault(h, p) != p
    return bad


def convert(args: argparse.Namespace) -> dict:
    stats = {"traces": 0, "top_level_requests": 0, "subagent_groups": 0, "subagent_inner_requests": 0,
             "input_tokens": 0, "output_tokens": 0, "unchained_ids": 0, "roles": {}}

    def trees():
        with open(args.input) as f:
            for n, line in enumerate(f):
                if args.limit and n >= args.limit:
                    break
                trace = json.loads(line)
                stats["traces"] += 1
                for r in trace["requests"]:
                    if r.get("type") == "subagent":
                        stats["subagent_groups"] += 1
                        stats["subagent_inner_requests"] += len(r.get("requests") or [])
                        for q in r.get("requests") or []:
                            stats["input_tokens"] += int(q.get("in") or 0)
                            stats["output_tokens"] += int(q.get("out") or 0)
                    else:
                        stats["top_level_requests"] += 1
                        stats["input_tokens"] += int(r.get("in") or 0)
                        stats["output_tokens"] += int(r.get("out") or 0)
                stats["unchained_ids"] += _check_chained(trace)
                tree = build_tree(trace)
                for st in tree["streams"]:
                    stats["roles"][st["role"]] = stats["roles"].get(st["role"], 0) + 1
                yield tree

    header = make_header(source="agentx", content="blocks", block_tokens=BLOCK_TOKENS,
                         converter={"script": os.path.basename(__file__), "input": os.path.basename(args.input),
                                    "limit": args.limit, "eps_s": EPS_S, "created": time.strftime("%Y-%m-%d")})
    write_trace(args.output, header=header, trees=trees())
    stats["requests"] = stats["top_level_requests"] + stats["subagent_inner_requests"]
    return stats


def describe(path: str) -> dict:
    """Shape of a converted trace: requests, waits, and MORI's programs (idle > 30 min, compactions)."""
    out = {"trees": 0, "streams": 0, "requests": 0, "prompt_tokens": 0, "output_tokens": 0, "programs_mori": 0,
           "programs_with_subagents": 0, "subagent_streams_in_programs": 0}
    waits, root_waits, spans = [], [], []
    for tree in iter_trees(path):
        materialize_blocks(tree)   # every base reference resolves
        s = tree_summary(tree, content="blocks", block_tokens=BLOCK_TOKENS)
        out["trees"] += 1
        out["streams"] += s["streams"]
        out["requests"] += s["requests"]
        out["prompt_tokens"] += s["prompt_tokens"]
        out["output_tokens"] += s["output_tokens"]
        spans.append(s["span_s"])
        for st in tree["streams"]:
            for r in st["requests"][1:]:
                waits.append(r["wait"])
        root_waits += [r["wait"] for r in tree["streams"][0]["requests"][1:]]
        progs = split_programs(tree, idle_s=1800.0, compaction=True, content="blocks")
        out["programs_mori"] += len(progs)
        for p in progs:
            subs = sum(1 for st in p["streams"] if st["role"] == "subagent")
            out["programs_with_subagents"] += subs > 0
            out["subagent_streams_in_programs"] += subs
    out["wait_s"] = quantiles(waits)
    out["root_wait_s"] = quantiles(root_waits)
    out["root_waits_over"] = {f"{x}s": round(sum(w > x for w in root_waits) / max(1, len(root_waits)), 4)
                              for x in (60, 300, 1800)}
    out["tree_span_h"] = {k: round(v / 3600, 2) for k, v in quantiles(spans).items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="traces.jsonl of cc-traces-weka-062126")
    ap.add_argument("--output", required=True, help="program trace (.jsonl or .jsonl.gz)")
    ap.add_argument("--limit", type=int, default=0, help="convert the first N traces only (0 = all)")
    args = ap.parse_args()
    stats = convert(args)
    stats["converted"] = describe(args.output)
    with open(args.output + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
