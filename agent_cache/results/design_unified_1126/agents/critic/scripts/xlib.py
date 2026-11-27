"""critic: one streaming pass over a run's server.log, cached under ../ext/ (read-only on the runs).

load(camp, arm) -> dict with
  t0            client t_start (epoch s); every time below is seconds since t0
  batch         [(t, 'P', new_seq, new_tok, cached_tok, usage, running, queue) | (t, 'D', running, tokens, usage, queue)]
  ev[name]      [(t, fields-dict)] for every HICACHE_EVT name
  rts[rid]      (t_log, input_len, cached_input_len, output_len, queue_s, forward_s, entry_t)
  cleanup       [(t, groups, files, GiB, secs)]
  other         {kind: [(t, text)]}: read failures, prefix-shrank lines, prefetch success/dropped lines
"""
import calendar
import json
import os
import pickle
import re
import sys

RES = "/home/wanhr/sglang/agent_cache/results"
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(os.path.dirname(HERE), "ext")
OUT = os.path.join(os.path.dirname(HERE), "out")
MANIFEST = {
    "c34": f"{RES}/campaign34/manifest.txt",
    "c32": f"{RES}/campaign32/manifest.txt",
    "c33": f"{RES}/campaign33/manifest.txt",
}
POOL_GPU = 668096
POOL_HOST = 1627648
PAGE = 64
_DATE = {}

RTS = re.compile(
    r"ReqTimeStats\(rid=(\S+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
    r"queue_duration=([0-9.]+)(ms|s), forward_duration=([0-9.]+)(ms|s), entry_time=([0-9.]+)"
)
PRE = re.compile(
    r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), "
    r"#running-req: (\d+), #queue-req: (\d+)"
)
DEC = re.compile(r"#running-req: (\d+), #token: (\d+), token usage: ([\d.]+),.*#queue-req: (\d+)")
CLEAN = re.compile(r"NIXL L3 cleanup: deleted (\d+) groups / (\d+) files \(([\d.]+) GiB\) in ([\d.]+)s")


def runs(camp):
    out = {}
    for line in open(MANIFEST[camp]):
        p = line.split()
        if len(p) >= 3 and not p[0].startswith("#"):
            out[p[0]] = (f"{RES}/{p[1]}", f"{RES}/{p[1]}/{p[2]}")
    return out


def ts(s):
    d = s[:10]
    base = _DATE.get(d)
    if base is None:
        base = _DATE[d] = calendar.timegm((int(d[:4]), int(d[5:7]), int(d[8:10]), 0, 0, 0))
    return base + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + float(s[17:])


def num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def fields(rest):
    out = {}
    for tok in rest.split():
        k, _, v = tok.partition("=")
        out[k] = num(v)
    return out


def client_t0(prog):
    with open(f"{prog}/client.jsonl") as f:
        return json.loads(f.readline())["t_start"]


def load(camp, arm, rebuild=False, run=None, prog=None):
    os.makedirs(EXT, exist_ok=True)
    cache = f"{EXT}/{camp}_{arm}.pkl"
    if os.path.exists(cache) and not rebuild:
        with open(cache, "rb") as f:
            return pickle.load(f)
    if run is None:
        run, prog = runs(camp)[arm]
    t0 = client_t0(prog)
    d = {"t0": t0, "batch": [], "ev": {}, "rts": {}, "cleanup": [], "other": {}, "run": run, "prog": prog}
    ev, batch, other = d["ev"], d["batch"], d["other"]
    with open(f"{run}/server.log", errors="replace") as f:
        for line in f:
            if not line.startswith("[20"):
                continue
            close = line.find("]")
            if close < 0:
                continue
            stamp = line[1:close]
            body = line[close + 2:]
            if body.startswith("HICACHE_EVT "):
                name, _, rest = body[12:].rstrip("\n").partition(" ")
                ev.setdefault(name, []).append((ts(stamp) - t0, fields(rest)))
                continue
            if body.startswith("Prefill batch"):
                m = PRE.search(body)
                if m:
                    batch.append((ts(stamp) - t0, "P", int(m[1]), int(m[2]), int(m[3]), float(m[4]), int(m[5]), int(m[6])))
                continue
            if body.startswith("Decode batch"):
                m = DEC.search(body)
                if m:
                    batch.append((ts(stamp) - t0, "D", int(m[1]), int(m[2]), float(m[3]), int(m[4])))
                continue
            if body.startswith("ReqTimeStats("):
                m = RTS.search(body)
                if m:
                    q = float(m[5]) / (1000.0 if m[6] == "ms" else 1.0)
                    fw = float(m[7]) / (1000.0 if m[8] == "ms" else 1.0)
                    d["rts"][m[1]] = (ts(stamp) - t0, int(m[2]), int(m[3]), int(m[4]), q, fw, float(m[9]) - t0)
                continue
            if "NIXL L3 cleanup" in body:
                m = CLEAN.search(body)
                if m:
                    d["cleanup"].append((ts(stamp) - t0, int(m[1]), int(m[2]), float(m[3]), float(m[4])))
                continue
            for kind, needle in (
                ("readfail", "failed to retrieve page"),
                ("shrank", "device prefix shrank"),
                ("pf", "HiCache prefetch "),
                ("capreach", "reissue cap reached"),
                ("openfail", "open failed"),
            ):
                if needle in body:
                    other.setdefault(kind, []).append((ts(stamp) - t0, body.rstrip("\n")[:400]))
                    break
    with open(cache, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def load_client(prog):
    out = {"run": None, "req": [], "play": [], "lanes": [], "summary": None}
    with open(f"{prog}/client.jsonl") as f:
        for line in f:
            r = json.loads(line)
            k = r.get("kind")
            if k == "req":
                out["req"].append(r)
            elif k == "play":
                out["play"].append(r)
            elif k == "lanes":
                out["lanes"].append(r)
            elif k == "run":
                out["run"] = r
            elif k == "summary":
                out["summary"] = r
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


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        d = load(camp, arm, rebuild=True)
        print(spec, "batch", len(d["batch"]), "rts", len(d["rts"]), "events", {k: len(v) for k, v in sorted(d["ev"].items())},
              "cleanup", len(d["cleanup"]), {k: len(v) for k, v in d["other"].items()}, flush=True)
