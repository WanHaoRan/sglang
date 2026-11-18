"""Gap profile of agent traces: how long each request waits on the one before it, by trace, kind and tool.

  cd agent_cache/scripts
  python3 gap_profile.py --out OUT_DIR \
      --program agentx=../data/converted/agentx_062126.programs.jsonl.gz \
      --program tracelab=../data/converted/tracelab_v0.0.2.programs.jsonl.gz \
      --program freeinference=../data/converted/freeinference_2026-07-12.programs.jsonl.gz \
      --lmcache ../traces/lmcache_agentic_trace_262k.json --scales 10,70

A gap is a request's recorded wait: the seconds from the end of the request it waited on (its latest dependency in a
program trace, the previous turn in the LMCache trace) to its own start. Session starts have none. Its kind is the
trace's own label: "tool" (a tool result came back), "user" (a person wrote), "spawn" (a subagent's first request),
"join" (the parent going on after its subagents) or "unlabeled" (the trace does not say). Its tool is the tool call of
the reply it waited on. Its reuse is the prompt prefix it shares with an earlier request: the KV a prefix cache has to
keep through the gap to serve it, so reuse x wait is that cache's memory-time (token-seconds) for the gap.

Writes one JSON file {rows, meta} per table (meta: this script's text and its inputs): gap_traces, gap_kinds, gap_cdf,
gap_residual, gap_tools, gap_cost, gap_sessions. Stdlib only; about a minute for the four traces.
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import os
import sys
from collections import defaultdict

LABELS = {"agentx": "AgentX", "tracelab": "TraceLab", "freeinference": "FreeInference", "lmcache": "LMCache"}
SOURCES = {
    "agentx": "huggingface.co/datasets/semianalysisai/cc-traces-weka-062126 (Claude Code sessions)",
    "tracelab": "huggingface.co/datasets/UW-SyFI/TraceLab v0.0.2 (Claude Code and Codex sessions)",
    "freeinference": "huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace, week 2026-07-12",
    "lmcache": "huggingface.co/datasets/sammshen/lmcache-agentic-traces (SWE-bench, GAIA and WildClaw agent runs)",
}
COST_EDGES = [0.0, 1.0, 10.0, 60.0, 300.0, 1800.0, math.inf]
COST_BINS = ["<1 s", "1-10 s", "10 s-1 min", "1-5 min", "5-30 min", ">=30 min"]
CDF_GRID = [10 ** (e / 20) for e in range(-40, 121)]   # 0.01 s to 10^6 s, 20 points a decade
# idle ages 0.1 s to 10^5 s, 10 a decade, plus the round ones the page quotes
AGE_GRID = sorted({10 ** (e / 10) for e in range(-10, 51)} | {60.0, 300.0, 600.0, 1800.0, 3600.0})
FLOOR_S = 0.01      # log-scale floor: waits below it (zeros included) count as 10 ms in log statistics
MIN_SERIES = 50     # fewest gaps for a CDF or a kind row
MIN_ALIVE = 30      # fewest gaps still open at an idle age for a residual point
MIN_TOOL = 100      # fewest gaps for a tool row
TOP_TOOLS = 15
MIN_GROUP = 20      # keys with fewer gaps are pooled as one group in the variance split


def sig(x: float | None, digits: int = 4) -> float | None:
    return None if x is None else float(f"{x:.{digits}g}")


def quantile(xs: list, q: float) -> float | None:
    """Linear-interpolated quantile of a sorted list."""
    if not xs:
        return None
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


# ===== Reading: every request as {stream, k, role, t, gap, prompt}, gap = (wait, kind, key, reuse) or None =====

def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def program_trees(path: str):
    """(tree id, requests) of each tree of a program trace (program_trace.py's format)."""
    with open_text(path) as f:
        header = json.loads(f.readline())
        blocks = header["content"] == "blocks"
        bt = header.get("block_tokens") or 0
        for line in f:
            tree = json.loads(line)
            streams = tree["streams"]
            out = []
            for s, st in enumerate(streams):
                for k, r in enumerate(st["requests"]):
                    if blocks:
                        prompt = (r["keep"] + len(r["new"])) * bt + r["tail"]
                        reuse = r["keep"] * bt if r["base"] is not None else 0
                    else:
                        prompt = r["prefix"] + r["append"]
                        reuse = r["prefix"]
                    gap = None
                    if r["deps"]:
                        ds, dk = max(r["deps"], key=lambda d: streams[d[0]]["requests"][d[1]]["t"]
                                     + streams[d[0]]["requests"][d[1]]["dur"])
                        key = streams[ds]["requests"][dk]["key"]
                        gap = (r["wait"], r["gap"] or "unlabeled", key, reuse)
                    out.append({"stream": s, "k": k, "role": st["role"], "t": r["t"], "gap": gap, "prompt": prompt})
            yield tree["tree"], out


def lmcache_trees(path: str, keys_path: str | None):
    """(session id, requests) of each conversation of a convert_lmcache.py trace; every gap there is a tool gap."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    convs, sessions = data["conversations"], data["sessions"]
    if len(sessions) != len(convs):
        sys.exit(f"STOP: {path}: {len(sessions)} sessions for {len(convs)} conversations")
    keys = None
    if keys_path:
        with open(keys_path, encoding="utf-8") as f:
            keys = json.load(f)["conversations"]
        if [len(c) for c in keys] != [len(c) for c in convs]:
            sys.exit(f"STOP: {keys_path} does not match {path} turn for turn")
    for c, conv in enumerate(convs):
        out = []
        for k, turn in enumerate(conv):
            prompt = turn["replay_prompt_tokens"]
            gap = None
            if k:
                prev = conv[k - 1]
                reuse = min(prev["replay_prompt_tokens"] + prev["output_length"], prompt)
                gap = (turn["pre_gap"], "tool", keys[c][k - 1] if keys else None, reuse)
            out.append({"stream": 0, "k": k, "role": "main", "t": None, "gap": gap, "prompt": prompt})
        yield sessions[c]["session_id"], out


# ===== Statistics =====

class Trace:
    def __init__(self, name: str, scale: float = 1.0):
        self.name, self.scale = name, scale
        self.sessions = self.requests = self.negative = 0
        self.gaps: list = []        # (wait, kind, key, reuse)
        self.trees: list = []       # (index, id, gaps, long gaps, spawns, user gaps) for the example pick

    def add(self, tree_id: str, reqs: list) -> None:
        self.sessions += 1
        self.requests += len(reqs)
        n = long_ = spawn = user = 0
        for r in reqs:
            g = r["gap"]
            if g is None:
                continue
            wait = g[0] * self.scale
            if wait < 0:
                self.negative += 1
                wait = 0.0
            self.gaps.append((wait, g[1], g[2], g[3]))
            n += 1
            long_ += wait >= 60
            spawn += g[1] == "spawn"
            user += g[1] == "user"
        self.trees.append((len(self.trees), tree_id, n, long_, spawn, user))

    def scaled(self, scale: float) -> "Trace":
        t = Trace(self.name, scale)
        t.sessions, t.requests, t.negative = self.sessions, self.requests, self.negative
        t.gaps = [(w * scale, kind, key, reuse) for w, kind, key, reuse in self.gaps]
        return t

    @property
    def id(self) -> str:
        return self.name if self.scale == 1 else f"{self.name}_x{self.scale:g}"


def series(tr: Trace) -> dict:
    """kind -> sorted waits, with "all"."""
    out = defaultdict(list)
    for w, kind, _, _ in tr.gaps:
        out["all"].append(w)
        out[kind].append(w)
    return {k: sorted(v) for k, v in out.items()}


def eta2(groups: dict) -> float | None:
    """Share of the variance of log10(wait) between groups (keys under MIN_GROUP gaps pooled as one)."""
    pooled: dict = defaultdict(list)
    for g, ys in groups.items():
        pooled[g if len(ys) >= MIN_GROUP else "(rare)"].extend(ys)
    ys = [y for v in pooled.values() for y in v]
    if len(pooled) < 2 or len(ys) < 2:
        return None
    mean = sum(ys) / len(ys)
    total = sum((y - mean) ** 2 for y in ys)
    between = sum(len(v) * (sum(v) / len(v) - mean) ** 2 for v in pooled.values())
    return round(between / total, 3) if total else None


def bin_of(w: float) -> int:
    return bisect.bisect_right(COST_EDGES, w) - 1


def trace_row(tr: Trace, ser: dict) -> dict:
    all_ = ser["all"]
    idle = sum(all_)
    cost = sum(w * reuse for w, _, _, reuse in tr.gaps)
    reuse_all = sum(reuse for _, _, _, reuse in tr.gaps)
    long_ = [(w, reuse) for w, _, _, reuse in tr.gaps if w > 300]
    user = ser.get("user", [])
    log = lambda w: math.log10(max(w, FLOOR_S))  # noqa: E731
    by_key, by_reply = defaultdict(list), defaultdict(list)
    for w, kind, key, _ in tr.gaps:
        if kind == "tool" and key is not None:
            by_key[key].append(log(w))
            by_reply[key].append(log(w))
        elif kind == "user":
            by_reply["(user turn)"].append(log(w))
    n = len(all_)
    return {
        "trace": tr.id, "label": LABELS[tr.name] + ("" if tr.scale == 1 else f" x{tr.scale:g}"), "scale": tr.scale,
        "sessions": tr.sessions, "requests": tr.requests, "gaps": n,
        "p50_s": sig(quantile(all_, 0.5)), "p90_s": sig(quantile(all_, 0.9)), "p99_s": sig(quantile(all_, 0.99)),
        "mean_s": sig(idle / n),
        "over_60s": round(sum(w > 60 for w in all_) / n, 4), "over_300s": round(len(long_) / n, 4),
        "over_1800s": round(sum(w > 1800 for w in all_) / n, 4),
        "idle_h": sig(idle / 3600), "idle_over_300s": round(sum(w for w, _ in long_) / idle, 4) if idle else None,
        "reuse_over_300s": round(sum(r for _, r in long_) / reuse_all, 4) if reuse_all else None,
        "cost_over_300s": round(sum(w * r for w, r in long_) / cost, 4) if cost else None,
        "user_share": round(len(user) / n, 4) if user else None,
        "user_idle_share": round(sum(user) / idle, 4) if user and idle else None,
        "tool_key_eta2": eta2(by_key), "reply_eta2": eta2(by_reply) if user else None,
        "source": SOURCES[tr.name],
    }


def kind_rows(tr: Trace, ser: dict) -> list:
    n, idle = len(ser["all"]), sum(ser["all"])
    cost = defaultdict(float)
    for w, kind, _, reuse in tr.gaps:
        cost[kind] += w * reuse
        cost["all"] += w * reuse
    rows = []
    for kind, ws in sorted(ser.items(), key=lambda kv: -len(kv[1])):
        if len(ws) < MIN_SERIES and kind != "all":
            continue
        rows.append({"trace": tr.id, "kind": kind, "scale": tr.scale, "n": len(ws), "share": round(len(ws) / n, 4),
                     **{f"p{int(q * 100)}_s": sig(quantile(ws, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
                     "mean_s": sig(sum(ws) / len(ws)), "idle_share": round(sum(ws) / idle, 4) if idle else None,
                     "cost_share": round(cost[kind] / cost["all"], 4) if cost["all"] else None})
    return rows


def cdf_rows(tr: Trace, ser: dict) -> list:
    rows = []
    for kind, ws in ser.items():
        if len(ws) < MIN_SERIES:
            continue
        pts = [(x, bisect.bisect_right(ws, x) / len(ws)) for x in CDF_GRID]
        first = max([i for i, (_, c) in enumerate(pts) if c == 0] or [0])     # keep one 0 and one 1
        last = min([i for i, (_, c) in enumerate(pts) if c == 1] or [len(pts) - 1])
        rows += [{"trace": tr.id, "kind": kind, "x_s": sig(x), "cdf": round(c, 5)} for x, c in pts[first:last + 1]]
    return rows


def residual_rows(tr: Trace, ser: dict) -> list:
    rows = []
    for kind, ws in ser.items():
        if len(ws) < MIN_SERIES:
            continue
        for age in AGE_GRID:
            i = bisect.bisect_right(ws, age)
            alive = ws[i:]
            if len(alive) < MIN_ALIVE:
                break
            back = bisect.bisect_right(ws, age + 60) - i
            rows.append({"trace": tr.id, "kind": kind, "idle_s": sig(age), "alive": len(alive),
                         "residual_p25_s": sig(quantile(alive, 0.25) - age),
                         "residual_p50_s": sig(quantile(alive, 0.5) - age),
                         "residual_p75_s": sig(quantile(alive, 0.75) - age),
                         "return_60s": round(back / len(alive), 4)})
    return rows


def tool_rows(tr: Trace) -> list:
    by_key = defaultdict(list)
    for w, kind, key, _ in tr.gaps:
        if kind == "tool" and key is not None:
            by_key[key].append(w)
    total = sum(len(v) for v in by_key.values())
    top = sorted((k for k, v in by_key.items() if len(v) >= MIN_TOOL), key=lambda k: -len(by_key[k]))[:TOP_TOOLS]
    rows = []
    for key in top:
        ws = sorted(by_key[key])
        rows.append({"trace": tr.id, "key": key, "n": len(ws), "share": round(len(ws) / total, 4),
                     **{f"p{int(q * 100)}_s": sig(quantile(ws, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                     "mean_s": sig(sum(ws) / len(ws))})
    return rows


def cost_rows(tr: Trace) -> list:
    n = [0] * len(COST_BINS)
    reuse, cost, idle = [0.0] * len(COST_BINS), [0.0] * len(COST_BINS), [0.0] * len(COST_BINS)
    for w, _, _, r in tr.gaps:
        b = bin_of(w)
        n[b] += 1
        reuse[b] += r
        cost[b] += w * r
        idle[b] += w
    tot = [sum(n), sum(reuse), sum(cost), sum(idle)]
    return [{"trace": tr.id, "bin": COST_BINS[b], "bin_index": b, "lo_s": COST_EDGES[b],
             "hi_s": None if math.isinf(COST_EDGES[b + 1]) else COST_EDGES[b + 1],
             "gap_share": round(n[b] / tot[0], 4), "reuse_share": round(reuse[b] / tot[1], 4) if tot[1] else None,
             "cost_share": round(cost[b] / tot[2], 4) if tot[2] else None,
             "idle_share": round(idle[b] / tot[3], 4) if tot[3] else None} for b in range(len(COST_BINS))]


def pick_example(tr: Trace) -> int | None:
    """The tree to show: 20-250 gaps with two or more of a minute or longer (and a subagent where the trace has
    any, two person-written turns where it labels them), the one whose gap count is nearest the median of those."""
    need_spawn = any(t[4] for t in tr.trees)
    need_user = any(t[5] for t in tr.trees)
    need_long = any(t[3] >= 2 for t in tr.trees)
    cands = [t for t in tr.trees if 20 <= t[2] <= 250 and (t[3] >= 2 or not need_long)
             and (t[4] >= 1 or not need_spawn) and (t[5] >= 2 or not need_user)]
    if not cands:
        return None
    med = sorted(t[2] for t in cands)[len(cands) // 2]
    return min(cands, key=lambda t: (abs(t[2] - med), t[0]))[0]


def session_rows(name: str, tree_id: str, reqs: list) -> list:
    order = sorted(reqs, key=lambda r: (r["t"] if r["t"] is not None else r["k"], r["stream"], r["k"]))
    rows = []
    for i, r in enumerate(order):
        g = r["gap"]
        rows.append({"trace": name, "session": tree_id, "order": i, "stream": r["stream"], "role": r["role"],
                     "k": r["k"], "start_s": sig(r["t"], 6) if r["t"] is not None else None,
                     "wait_s": sig(max(0.0, g[0])) if g else None, "kind": g[1] if g else "start",
                     "key": g[2] if g else None, "prompt_tokens": r["prompt"]})
    return rows


# ===== Main =====

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--program", action="append", default=[], metavar="NAME=PATH",
                    help=f"a converted program trace; NAME one of {sorted(set(LABELS) - {'lmcache'})}")
    ap.add_argument("--lmcache", help="a convert_lmcache.py trace (its .reply_keys.json beside it is read too)")
    ap.add_argument("--scales", default="", help="gap scales of the LMCache trace to add, as in --gap-scale: 10,70")
    ap.add_argument("--out", required=True, help="directory for the JSON files")
    args = ap.parse_args()

    inputs, readers = [], []
    for spec in args.program:
        name, _, path = spec.partition("=")
        if name not in LABELS or name == "lmcache" or not path:
            sys.exit(f"STOP: --program {spec}: NAME=PATH with NAME one of agentx, tracelab, freeinference")
        readers.append((name, lambda p=path: program_trees(p)))
        inputs.append(f"{name}: {os.path.basename(path)}, converted from {SOURCES[name]}")
    if args.lmcache:
        keys = args.lmcache + ".reply_keys.json"
        keys = keys if os.path.exists(keys) else None
        readers.append(("lmcache", lambda: lmcache_trees(args.lmcache, keys)))
        inputs.append(f"lmcache: {os.path.basename(args.lmcache)}"
                      + (f" and {os.path.basename(keys)}" if keys else "") + f", from {SOURCES['lmcache']}")
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    tables: dict = {k: [] for k in ("traces", "kinds", "cdf", "residual", "tools", "cost", "sessions")}
    for name, read in readers:
        tr = Trace(name)
        for tree_id, reqs in read():
            tr.add(tree_id, reqs)
        pick = pick_example(tr)
        if pick is not None:
            for i, (tree_id, reqs) in enumerate(read()):
                if i == pick:
                    tables["sessions"] += session_rows(name, tree_id, reqs)
                    break
        variants = [tr] + ([tr.scaled(s) for s in scales] if name == "lmcache" else [])
        for v in variants:
            ser = series(v)
            tables["traces"].append(trace_row(v, ser))
            tables["kinds"] += kind_rows(v, ser)
            tables["cdf"] += cdf_rows(v, ser)
            tables["cost"] += cost_rows(v)
            if v.scale == 1:
                tables["residual"] += residual_rows(v, ser)
                tables["tools"] += tool_rows(v)
        print(f"{name}: {tr.sessions} sessions, {tr.requests} requests, {len(tr.gaps)} gaps "
              f"({tr.negative} negative waits set to 0); example tree {pick}", file=sys.stderr)

    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        code = f.read()
    meta = [{"label": os.path.basename(__file__), "value": code, "type": "code", "lang": "python"},
            {"label": "Input", "value": "\n".join(inputs)},
            {"label": "Settings", "value": f"LMCache gap scales {scales or 'none'}; waits under 0 set to 0"}]
    os.makedirs(args.out, exist_ok=True)
    for name, rows in tables.items():
        path = os.path.join(args.out, f"gap_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "meta": meta}, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        print(f"wrote {path}: {len(rows)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
