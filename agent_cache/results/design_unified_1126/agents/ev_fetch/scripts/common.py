"""Shared helpers for the ev_fetch checks (CPU only; stdlib + numpy).

- policy(): the engine's next_turn_policy.py imported by path (pure Python + msgspec).
- load_trace(): the compact trace extract (extract_trace.py).
- gap samples per key as the replay client builds them (replay_programs.py hints_for with
  --hint-key command --hint-no-call, gap scale 1, gap cap 1,200 s).
"""
import gzip
import importlib.util
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTRACTS = os.path.join(ROOT, "extracts")
OUT = os.path.join(ROOT, "out")
RESULTS = "/home/wanhr/sglang/agent_cache/results"
ENGINE = "/home/wanhr/sglang/python/sglang/srt"
TRACE_COMPACT = os.path.join(EXTRACTS, "trace_compact.jsonl.gz")

GAP_CAP_S = 1200.0      # campaign settings.env GAP_CAP
GAP_FLOOR_S = 0.01      # gap_hints.GAP_FLOOR_S
MAX_CONTEXT = 65536     # campaign settings.env MAX_CONTEXT
NO_CALL = "_no_call"
MIN_POOLED = 5          # gap_hints.MIN_POOLED

C34 = {
    "jit_p50": ("20261123_073130_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_073338_jit_p50"),
    "ec_full": ("20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_093918_ec_full"),
    "jit_exact": ("20261123_114134_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_114453_jit_exact"),
    "hicache": ("20261123_211144_three_tier_wc_custom_L1nat_H160", "programs_211406_hicache"),
    "ec_exact": ("20261123_231607_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_231919_ec_exact"),
}

_policy = None


def policy():
    """next_turn_policy.py loaded by path, without importing the sglang package."""
    global _policy
    if _policy is None:
        spec = importlib.util.spec_from_file_location(
            "ntp_policy_by_path", os.path.join(ENGINE, "mem_cache", "next_turn_policy.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _policy = mod
    return _policy


def load_trace(max_context=MAX_CONTEXT):
    """Trees of the compact extract; max_context drops trees as the replay's --max-context does (None keeps all)."""
    trees = []
    with gzip.open(TRACE_COMPACT, "rt") as f:
        json.loads(f.readline())
        for line in f:
            t = json.loads(line)
            if max_context is None or t["peak"] <= max_context:
                trees.append(t)
    return trees


# request fields in the compact extract
T, DUR, WAIT, GAP, KEY, OUT_TOK, PROMPT, KEEP, SELF_DEP, NDEPS, MUT = range(11)


def reply_key(req):
    return req[KEY] or NO_CALL


def tool_of(key):
    return key.split("/", 1)[0]


def iter_gaps(trees):
    """(tree, stream index, k, key of reply k-1, gap s (capped), gap kind, history tokens the next prompt keeps,
    prompt tokens of the next request, system) for every request k >= 1: the sample the client's table observes."""
    for t in trees:
        system = tuple(t.get("system") or ())
        for s, st in enumerate(t["streams"]):
            rq = st["reqs"]
            for k in range(1, len(rq)):
                cur = rq[k]
                yield (t, s, k, reply_key(rq[k - 1]), min(GAP_CAP_S, max(0.0, float(cur[WAIT] or 0.0))), cur[GAP],
                       cur[KEEP], cur[PROMPT], system)


def iter_ends(trees):
    """(key of the last reply of each stream, system): the turns after which no request follows."""
    for t in trees:
        system = tuple(t.get("system") or ())
        for st in t["streams"]:
            rq = st["reqs"]
            if rq:
                yield reply_key(rq[-1]), system


def ln_quantile(sorted_ln, p):
    """gap_hints._quantile on ln(gap): linear interpolation between order statistics."""
    pos = p * (len(sorted_ln) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_ln) - 1)
    return sorted_ln[lo] + (sorted_ln[hi] - sorted_ln[lo]) * (pos - lo)


def entry_quantiles(gaps):
    """(p10, p50, p90) as the client's table entry: quantiles of ln(max(gap, floor)), exponentiated."""
    xs = sorted(math.log(max(g, GAP_FLOOR_S)) for g in gaps)
    return tuple(math.exp(ln_quantile(xs, p)) for p in (0.1, 0.5, 0.9))


def q(xs, p):
    """Plain empirical quantile (nearest rank) of a list; nan when empty."""
    if not len(xs):
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "na"
    return f"{x:.{nd}f}"
