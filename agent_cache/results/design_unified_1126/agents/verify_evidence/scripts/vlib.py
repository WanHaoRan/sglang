"""verify_evidence: my own readers for the campaign logs (not the analysts').

Server logs: '[YYYY-MM-DD HH:MM:SS.fff] ...' in UTC wall clock.  Client records: seconds since the
run record's t_start (epoch).  Every time returned here is seconds since the client's t_start.
"""
import calendar
import gzip
import json
import os
import pickle
import re

RES = "/home/wanhr/sglang/agent_cache/results"
ME = RES + "/design_unified_1126/agents/verify_evidence"
EXT = ME + "/extracts"
OUT = ME + "/out"

_TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)(?:\.(\d+))?\] (.*)$")
_KV = re.compile(r"([A-Za-z_0-9]+)=([^ ,)]+)")


def manifest(camp):
    m = {}
    for line in open(f"{RES}/campaign{camp}/manifest.txt"):
        p = line.split()
        if len(p) >= 3:
            m[p[0]] = (p[1], p[2])
    return m


def paths(camp, arm):
    run, prog = manifest(camp)[arm]
    return f"{RES}/{run}/server.log", f"{RES}/{run}/{prog}/client.jsonl"


def client(camp, arm):
    """-> dict(run=..., reqs=[...], plays=[...], lanes=[...], summary=...)"""
    _, cp = paths(camp, arm)
    out = {"run": None, "reqs": [], "plays": [], "lanes": [], "summary": None, "other": []}
    for line in open(cp):
        d = json.loads(line)
        k = d.get("kind")
        if k == "run":
            out["run"] = d
        elif k == "req":
            out["reqs"].append(d)
        elif k == "play":
            out["plays"].append(d)
        elif k == "lanes":
            out["lanes"].append(d)
        elif k == "summary":
            out["summary"] = d
        else:
            out["other"].append(d)
    return out


def t_start(camp, arm):
    _, cp = paths(camp, arm)
    with open(cp) as f:
        d = json.loads(f.readline())
    assert d["kind"] == "run"
    return d["t_start"]


def epoch(m):
    y, mo, d, h, mi, s, frac = m.group(1, 2, 3, 4, 5, 6, 7)
    e = calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s)))
    if frac:
        e += int(frac) / (10 ** len(frac))
    return e


def kv(s):
    return dict(_KV.findall(s))


def stream(camp, arm):
    """Yield (t_rel, name, rest) for the lines of interest of one server log."""
    sp, _ = paths(camp, arm)
    t0 = t_start(camp, arm)
    with open(sp, errors="replace") as f:
        for line in f:
            m = _TS.match(line)
            if not m:
                continue
            rest = m.group(8)
            if rest.startswith("HICACHE_EVT "):
                sp1 = rest.find(" ", 12)
                name = rest[12:sp1] if sp1 > 0 else rest[12:]
                yield epoch(m) - t0, name, rest[sp1 + 1:] if sp1 > 0 else ""
            elif rest.startswith("ReqTimeStats("):
                yield epoch(m) - t0, "ReqTimeStats", rest
            elif rest.startswith("Prefill batch"):
                yield epoch(m) - t0, "Prefill", rest
            elif rest.startswith("Decode batch"):
                yield epoch(m) - t0, "Decode", rest
            elif rest.startswith("NIXL L3 cleanup"):
                yield epoch(m) - t0, "cleanup", rest
            elif rest.startswith("INFO:"):
                continue
            else:
                yield epoch(m) - t0, "other", rest


def load(camp, arm, refresh=False):
    """Cached parse: {name: [(t_rel, rest), ...]} in file order."""
    os.makedirs(EXT, exist_ok=True)
    p = f"{EXT}/ev_c{camp}_{arm}.pkl"
    if os.path.exists(p) and not refresh:
        with open(p, "rb") as f:
            return pickle.load(f)
    d = {}
    for t, name, rest in stream(camp, arm):
        d.setdefault(name, []).append((t, rest))
    with open(p, "wb") as f:
        pickle.dump(d, f, protocol=4)
    return d


_RTS = re.compile(
    r"ReqTimeStats\(rid=([^,]+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
    r"queue_duration=([0-9.]+)ms, forward_duration=([0-9.]+)ms, entry_time=([0-9.]+)"
)


def req_stats(ev, camp, arm):
    """rid -> dict(entry, adm, queue, fwd, fin, input, cached, out) in seconds since t_start."""
    t0 = t_start(camp, arm)
    out = {}
    for t, rest in ev.get("ReqTimeStats", []):
        m = _RTS.match(rest)
        if not m:
            continue
        rid = m.group(1)
        q = float(m.group(5)) / 1e3
        fw = float(m.group(6)) / 1e3
        e = float(m.group(7)) - t0
        out[rid] = dict(entry=e, queue=q, adm=e + q, fwd=fw, fin=t, input=int(m.group(2)),
                        cached=int(m.group(3)), out=int(m.group(4)))
    return out


_PRE = re.compile(r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([0-9.]+), "
                  r"#running-req: (\d+), #queue-req: (\d+)")
_DEC = re.compile(r"#running-req: (\d+), #token: (\d+), token usage: ([0-9.]+),.*#queue-req: (\d+)")


def batches(ev):
    """-> list of (t, kind, running, queue, usage, new_seq, new_tok) for Prefill and Decode lines."""
    out = []
    for t, rest in ev.get("Prefill", []):
        m = _PRE.search(rest)
        if m:
            out.append((t, "P", int(m.group(5)), int(m.group(6)), float(m.group(4)), int(m.group(1)), int(m.group(2))))
    for t, rest in ev.get("Decode", []):
        m = _DEC.search(rest)
        if m:
            out.append((t, "D", int(m.group(1)), int(m.group(4)), float(m.group(3)), 0, 0))
    out.sort()
    return out


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = q * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def trace_path():
    return "/home/wanhr/sglang/agent_cache/data/converted/freeinference_16w.programs.jsonl.gz"
