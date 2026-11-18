"""Program traces: recorded agent sessions in one trace-neutral form for closed-loop replay (replay_programs.py).

Stdlib only. Written by convert_agentx.py, convert_tracelab.py and convert_freeinference.py; MAINSTREAM_REPLAY.md says
how each source maps onto it.

A file is JSONL, gzipped when its name ends in .gz. The first line is the header ({"kind": "header", ...}: the format,
the source, the content model and the converter's settings). Every other line is one tree: a recorded session (its root
agent) with every stream it spawned (subagents, side chains, forks). A stream is a sequence of LLM requests. A request
may start once its dependencies have completed ("deps": [stream, index] pairs of earlier requests in the tree) and its
recorded wait has passed ("wait": seconds from the latest dependency's end; from the tree's start when it has none).
That is the closed loop of every replay in the literature: a request follows its predecessor's response, never the
recorded clock. "t" and "dur" are the recorded start (seconds after the tree's first request) and duration; the replay
reads them only to place trajectory starts and to report.

The prompt has one of two content models, named in the header:
  blocks   hashed content: the blocks of request "base" (a [stream, index] pair or null), kept up to "keep", then the
           block ids "new", then "tail" unique tokens (a dropped partial block). Block ids are chained: equal ids at a
           position mean equal prefixes up to it, so the replay maps each id to fixed tokens and the recorded sharing,
           across streams and (within one "scope") across sessions, reappears on the wire.
  lengths  TraceLab's model: "prefix" tokens of the stream's context (its previous prompt and the reply the server
           generated for it), then "append" new tokens.
"out" is the recorded output length, which the replay forces. "gap" says what the wait is ("tool", "user", "spawn",
"join", or null), "key" is the hint key of the recorded response's tool call ("Bash/git"; null without one).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
from array import array
from collections import OrderedDict

FORMAT = "program-trace/1"
CONTENTS = ("blocks", "lengths")
KEY_SEP = "/"   # the hint key separator, as gap_hints.KEY_SEP


# ===== File format =====

def open_text(path: str, mode: str):
    """A text handle on path, through gzip when it ends in .gz."""
    if path.endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8", compresslevel=6)
    return open(path, mode, encoding="utf-8")


def make_header(*, source: str, content: str, block_tokens: int | None, converter: dict) -> dict:
    if content not in CONTENTS:
        raise ValueError(f"content {content}: one of {CONTENTS}")
    return {"kind": "header", "format": FORMAT, "source": source, "content": content, "block_tokens": block_tokens,
            "converter": converter}


def write_trace(path: str, *, header: dict, trees) -> int:
    """Write the header and the trees (an iterable); returns the number of trees written."""
    n = 0
    with open_text(path, "w") as f:
        f.write(json.dumps(header, separators=(",", ":")) + "\n")
        for tree in trees:
            f.write(json.dumps(tree, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_header(path: str) -> dict:
    with open_text(path, "r") as f:
        header = json.loads(f.readline())
    if header.get("kind") != "header" or header.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file (first line {str(header)[:120]})")
    return header


def iter_trees(path: str):
    """The trees of a program trace, in file order (the header is checked and skipped)."""
    read_header(path)
    with open_text(path, "r") as f:
        f.readline()
        for line in f:
            if line.strip():
                yield json.loads(line)


# ===== Requests and streams =====

def request(*, t: float, dur: float, deps: list, wait: float, out: int, gap: str | None = None,
            key: str | None = None, **content) -> dict:
    """One request record; content is base/keep/new/tail (blocks) or prefix/append (lengths), plus any extras."""
    rec = {"t": round(t, 3), "dur": round(max(0.0, dur), 3), "deps": deps, "wait": round(max(0.0, wait), 3),
           "out": max(1, int(out)), "gap": gap, "key": key}
    rec.update(content)
    return rec


def end_of(req: dict) -> float:
    return req["t"] + req["dur"]


def link_deps(streams: list, s: int, k: int, deps: list) -> tuple:
    """(deps, wait) for request k of stream s: the deps as pairs, and the recorded wait from the latest one's end."""
    req = streams[s]["requests"][k]
    if not deps:
        return [], req["t"]
    ready = max(end_of(streams[ds]["requests"][dk]) for ds, dk in deps)
    return [list(d) for d in deps], req["t"] - ready


def lcp(a: list, b: list) -> int:
    """Length of the common prefix of two chained block-id lists. Binary search is exact because ids are chained: equal
    ids at a position mean equal lists up to there (the converters check this on every trace)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[mid - 1] == b[mid - 1]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def delta(blocks: list, candidates: list) -> tuple:
    """(base, keep, new) that rebuilds `blocks` from the candidate (ref, blocks) pair sharing the longest prefix."""
    best_ref, best_keep = None, 0
    for ref, other in candidates:
        n = lcp(blocks, other)
        if n > best_keep:
            best_ref, best_keep = ref, n
    return best_ref, best_keep, blocks[best_keep:]


def materialize_blocks(tree: dict) -> dict:
    """{(stream, index): block list} for every request of a blocks tree (references resolve in stream order)."""
    out: dict = {}
    pending = [(s, k) for s, st in enumerate(tree["streams"]) for k in range(len(st["requests"]))]
    while pending:
        rest = []
        for s, k in pending:
            r = tree["streams"][s]["requests"][k]
            base = r.get("base")
            if base is None:
                out[(s, k)] = list(r["new"])
            elif tuple(base) in out:
                out[(s, k)] = out[tuple(base)][: r["keep"]] + list(r["new"])
            else:
                rest.append((s, k))
        if len(rest) == len(pending):
            raise ValueError(f"tree {tree.get('tree')}: unresolvable base references {rest[:3]}")
        pending = rest
    return out


def stream_order(tree: dict) -> list:
    """Every (stream, index) of the tree in recorded start order."""
    keys = [(r["t"], s, k) for s, st in enumerate(tree["streams"]) for k, r in enumerate(st["requests"])]
    return [(s, k) for _, s, k in sorted(keys)]


def check_tree(tree: dict, *, content: str) -> None:
    """Raise ValueError on a malformed tree: deps must point at existing, earlier-or-same-time requests, and the
    content fields must match the content model."""
    streams = tree["streams"]
    for s, st in enumerate(streams):
        for k, r in enumerate(st["requests"]):
            for ds, dk in r["deps"]:
                if not (0 <= ds < len(streams) and 0 <= dk < len(streams[ds]["requests"])):
                    raise ValueError(f"tree {tree['tree']} stream {s} request {k}: dep {ds},{dk} out of range")
                if (ds, dk) == (s, k):
                    raise ValueError(f"tree {tree['tree']} stream {s} request {k}: depends on itself")
            if content == "blocks" and not {"keep", "new"} <= r.keys():
                raise ValueError(f"tree {tree['tree']} stream {s} request {k}: blocks content missing")
            if content == "lengths" and not {"prefix", "append"} <= r.keys():
                raise ValueError(f"tree {tree['tree']} stream {s} request {k}: lengths content missing")
    _check_acyclic(tree)


def _check_acyclic(tree: dict) -> None:
    streams = tree["streams"]
    state: dict = {}

    def visit(node: tuple) -> None:
        stack = [(node, iter(streams[node[0]]["requests"][node[1]]["deps"]))]
        state[node] = 1
        while stack:
            cur, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[cur] = 2
                stack.pop()
                continue
            nxt = tuple(nxt)
            if state.get(nxt) == 1:
                raise ValueError(f"tree {tree['tree']}: dependency cycle through {nxt}")
            if nxt not in state:
                state[nxt] = 1
                stack.append((nxt, iter(streams[nxt[0]]["requests"][nxt[1]]["deps"])))

    for s, st in enumerate(streams):
        for k in range(len(st["requests"])):
            if (s, k) not in state:
                visit((s, k))


def hint_key(tool: str | None, word: str | None) -> str | None:
    """A hint key "tool/word" in the engine's label grammar [\\w./-]; None without a tool."""
    if not tool:
        return None
    tool = _label(tool)
    word = _label(word) if word else None
    return tool if not word else f"{tool}{KEY_SEP}{word}"


def _label(s: str) -> str:
    out = "".join(c if (c.isalnum() or c in "_.-") else "-" for c in s.strip())
    return out.strip("-")[:48] or "x"


# ===== Programs (MORI's split) =====

def split_programs(tree: dict, *, idle_s: float | None, compaction: bool, content: str) -> list:
    """Cut a tree into programs where the whole tree sits idle longer than idle_s before a root request, and (with
    compaction) where a root request drops more than half of its predecessor's context. A cut is made only where no
    other stream has a request in flight or still to come before the cut, so subagents stay inside one program. The
    program starting at a cut has no deps there and waits 0: the next program is the next session the lane runs."""
    root = tree["streams"][0]["requests"]
    cuts = [k for k in range(1, len(root)) if _is_cut(tree, k, idle_s=idle_s, compaction=compaction, content=content)]
    if not cuts:
        return [tree]
    bounds = [0] + cuts + [len(root)]
    blocks = materialize_blocks(tree) if content == "blocks" else None
    return [_program(tree, lo, hi, j, blocks) for j, (lo, hi) in enumerate(zip(bounds, bounds[1:]))]


def _is_cut(tree: dict, k: int, *, idle_s: float | None, compaction: bool, content: str) -> bool:
    root = tree["streams"][0]["requests"]
    t_k = root[k]["t"]
    for st in tree["streams"][1:]:
        ts = [r["t"] for r in st["requests"]]
        if ts and min(ts) < t_k <= max(ts):   # a stream with requests on both sides of the cut
            return False
        if any(r["t"] < t_k < end_of(r) for r in st["requests"]):   # a request in flight across the cut
            return False
    before = [end_of(r) for st in tree["streams"] for r in st["requests"] if r["t"] < t_k]
    if idle_s is not None and before and t_k - max(before) > idle_s:
        return True
    if compaction:
        size = (lambda r: r["keep"] + len(r["new"])) if content == "blocks" else (lambda r: r["prefix"] + r["append"])
        return size(root[k]) < 0.5 * size(root[k - 1])
    return False


def _program(tree: dict, lo: int, hi: int, j: int, blocks: dict | None) -> dict:
    """Root requests lo..hi-1 and the streams that start in their time span, re-indexed and re-timed from the
    program's first request. Deps into an earlier program are dropped, and the waits re-derived from those left."""
    old = tree["streams"]
    t_lo = old[0]["requests"][lo]["t"]
    t_hi = old[0]["requests"][hi]["t"] if hi < len(old[0]["requests"]) else float("inf")
    keep_streams = [0] + [s for s in range(1, len(old))
                          if old[s]["requests"] and t_lo <= old[s]["requests"][0]["t"] < t_hi]
    kmap = {(0, k): (0, k - lo) for k in range(lo, hi)}
    for i, s in enumerate(keep_streams[1:], start=1):
        kmap.update({(s, k): (i, k) for k in range(len(old[s]["requests"]))})
    streams = []
    for s in keep_streams:
        ks = range(lo, hi) if s == 0 else range(len(old[s]["requests"]))
        reqs = [_reindex(old[s]["requests"][k], key=(s, k), kmap=kmap, blocks=blocks, t_lo=t_lo) for k in ks]
        streams.append({**{x: v for x, v in old[s].items() if x != "requests"}, "requests": reqs})
    for s, st in enumerate(streams):
        for k, r in enumerate(st["requests"]):
            if r.pop("_relink", False):
                r["deps"], wait = link_deps(streams, s, k, r["deps"])
                r["wait"] = round(max(0.0, wait), 3)
    out = {**{x: v for x, v in tree.items() if x != "streams"}, "tree": f"{tree['tree']}#p{j}", "program": j,
           "streams": streams}
    first = streams[0]["requests"][0]
    out["setup"] = min(int(tree.get("setup", 0)), first["keep"] + len(first["new"]) if "new" in first else 0)
    return out


def _reindex(req: dict, *, key: tuple, kmap: dict, blocks: dict | None, t_lo: float) -> dict:
    r = dict(req)
    r["t"] = round(r["t"] - t_lo, 3)
    deps = [list(kmap[tuple(d)]) for d in r["deps"] if tuple(d) in kmap]
    if len(deps) < len(r["deps"]):
        r["_relink"] = True   # the wait was measured from a dep in an earlier program
    r["deps"] = deps
    if blocks is not None and r.get("base") is not None:
        if tuple(r["base"]) in kmap:
            r["base"] = list(kmap[tuple(r["base"])])
        else:   # the base fell in an earlier program: spell the blocks out
            r["base"], r["keep"], r["new"] = None, 0, blocks[key]
    return r


# ===== Synthetic tokens =====

class TokenSynth:
    """Deterministic token ids for block ids and for unique spans. Ids are uniform in [lo, hi), or windows of a token
    pool (uint32 file of real text tokenized for the served model), which keeps MoE routing closer to real text."""

    def __init__(self, *, lo: int = 100, hi: int = 30000, pool_path: str | None = None, cache_blocks: int = 500_000):
        if not 0 <= lo < hi:
            raise ValueError(f"vocab range {lo}:{hi} is empty")
        self.lo, self.span = lo, hi - lo
        self.pool = None
        if pool_path:
            self.pool = array("I")
            with open(pool_path, "rb") as f:
                self.pool.frombytes(f.read())
            if len(self.pool) < 1 << 16:
                raise ValueError(f"token pool {pool_path}: {len(self.pool)} tokens, need at least 65536")
        self._cache: OrderedDict = OrderedDict()
        self._cap = cache_blocks

    def ids(self, seed: str, n: int) -> list:
        """n token ids determined by seed."""
        if n <= 0:
            return []
        raw = seed.encode()
        if self.pool is not None:
            off = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % (len(self.pool) - n)
            return self.pool[off: off + n].tolist()
        out: list = []
        ctr = 0
        while len(out) < n:
            d = hashlib.blake2b(raw + ctr.to_bytes(4, "little"), digest_size=64).digest()
            out.extend(self.lo + x % self.span for x in struct.unpack("<16I", d))
            ctr += 1
        return out[:n]

    def block_json(self, scope: str, block: int, n: int) -> str:
        """The comma-joined ids of one block, cached: most of a request's blocks were in its predecessor."""
        key = (scope, block)
        frag = self._cache.get(key)
        if frag is None:
            frag = ",".join(map(str, self.ids(f"b|{scope}|{block}", n)))
            self._cache[key] = frag
            if len(self._cache) > self._cap:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return frag


class BlockPrompts:
    """The prompts of one play of a blocks tree. A cache-bust marker (tokens unique to the play), when given, leads
    every prompt of the play: a recycled play keeps its own sharing and shares none with earlier plays of the tree."""

    def __init__(self, tree: dict, *, block_tokens: int, synth: TokenSynth, bust: list | None, play_tag: str):
        self.tree, self.B, self.synth, self.play_tag = tree, block_tokens, synth, play_tag
        self.scope = str(tree.get("scope") or tree["tree"])
        self.bust_json = ",".join(map(str, bust)) if bust else None
        self.bust_len = len(bust) if bust else 0
        self._blocks: dict = {}

    def blocks(self, s: int, k: int) -> list:
        got = self._blocks.get((s, k))
        if got is None:
            r = self.tree["streams"][s]["requests"][k]
            base = r.get("base")
            got = list(r["new"]) if base is None else self.blocks(*base)[: r["keep"]] + list(r["new"])
            self._blocks[(s, k)] = got
        return got

    def prompt(self, s: int, k: int) -> tuple:
        """(comma-joined ids, number of tokens, block ids, whether the bust marker leads)."""
        r = self.tree["streams"][s]["requests"][k]
        blocks = self.blocks(s, k)
        frags = [self.bust_json] if self.bust_json is not None else []
        frags += [self.synth.block_json(self.scope, b, self.B) for b in blocks]
        tail = int(r.get("tail") or 0)
        if tail:
            frags.append(",".join(map(str, self.synth.ids(f"t|{self.scope}|{self.play_tag}|{s}|{k}", tail))))
        return ",".join(f for f in frags if f), len(blocks) * self.B + self.bust_len + tail, blocks, \
            self.bust_json is not None


class LengthPrompts:
    """The prompts of one play of a lengths tree, as TraceLab's session_runner builds them: the stream's context is its
    previous prompt plus the output the server generated for it; a prompt is the context's first `prefix` tokens
    (extended with new tokens when the context is shorter) plus `append` new tokens."""

    def __init__(self, tree: dict, *, synth: TokenSynth, play_tag: str):
        self.tree, self.synth, self.play_tag = tree, synth, play_tag
        self._context: dict = {}
        self._sent: dict = {}

    def prompt(self, s: int, k: int) -> tuple:
        """(comma-joined ids, number of tokens, None, None); call commit(s, output_ids) once the reply is complete."""
        r = self.tree["streams"][s]["requests"][k]
        ctx = self._context.get(s, [])
        prefix, append = int(r["prefix"]), int(r["append"])
        if len(ctx) < prefix:
            ctx = ctx + self.synth.ids(f"x|{self.tree['tree']}|{self.play_tag}|{s}|{k}", prefix - len(ctx))
        ids = ctx[:prefix] + self.synth.ids(f"a|{self.tree['tree']}|{self.play_tag}|{s}|{k}", append)
        self._sent[s] = ids
        return ",".join(map(str, ids)), len(ids), None, None

    def commit(self, s: int, output_ids: list) -> None:
        """The stream's next context: the prompt just sent and the reply the server generated for it."""
        self._context[s] = self._sent.pop(s, []) + list(output_ids)

    def held(self, s: int, r: dict) -> int:
        """Prompt tokens of request r that the stream's context already holds (an infinite cache's hit)."""
        return min(int(r["prefix"]), len(self._context.get(s, [])))


# ===== Statistics =====

def quantiles(xs: list, ps=(0.1, 0.5, 0.9, 0.99)) -> dict:
    if not xs:
        return {}
    ys = sorted(xs)
    return {f"p{round(p * 100):g}": ys[min(len(ys) - 1, int(p * (len(ys) - 1) + 0.5))] for p in ps}


def tree_summary(tree: dict, *, content: str, block_tokens: int | None) -> dict:
    """Counts for the converter's stats file: requests, streams, tokens, waits by kind."""
    reqs = [r for st in tree["streams"] for r in st["requests"]]
    if content == "blocks":
        prompt = [(r["keep"] + len(r["new"])) * block_tokens + int(r.get("tail") or 0) for r in reqs]
    else:
        prompt = [int(r["prefix"]) + int(r["append"]) for r in reqs]
    return {"requests": len(reqs), "streams": len(tree["streams"]), "prompt_tokens": sum(prompt),
            "max_prompt": max(prompt, default=0), "output_tokens": sum(r["out"] for r in reqs),
            "span_s": max((end_of(r) for r in reqs), default=0.0)}
