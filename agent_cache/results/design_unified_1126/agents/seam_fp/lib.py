"""Shared loader for the seam_fp checks (CPU only, stdlib). Reads the extracts written by extract.sh
(evt/<label>.evt.gz, a filtered copy of a run's server.log) and the run's client.jsonl.

Times: log stamps are UTC wall clock and are converted to epoch seconds; ReqTimeStats.entry_time is already epoch
(the request's wait-queue entry). `t0` of a run is the client's t_start (client.jsonl, kind=run).
"""

from __future__ import annotations

import calendar
import collections
import gzip
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = "/home/wanhr/sglang/agent_cache/results"

_STAMP = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:[.,](\d+))?\] (.*)$")
_KV = re.compile(r"(\w+)=([^\s,)]+)")
_RTS = re.compile(
    r"ReqTimeStats\(rid=(\S+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
    r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)"
)
_BATCH = re.compile(
    r"(Prefill|Decode) batch,.*?token usage: ([\d.]+).*?#running-req: (\d+).*?#queue-req: (\d+)"
)
_DECODE = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*?#queue-req: (\d+)"
)
_PREFILL = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), "
    r"#running-req: (\d+), #queue-req: (\d+)"
)
_SUCCESS = re.compile(
    r"HiCache prefetch (success|dropped) req=(\S+) completed=(\d+) matched=(\d+) loaded=(\d+)"
)
_SHRANK = re.compile(
    r"device prefix shrank before admission req=(\S+) lookup_match=(\d+) current_match=(\d+)"
)


def runs() -> dict[str, tuple[str, str]]:
    out = {}
    with open(os.path.join(HERE, "runs.tsv")) as f:
        for line in f:
            label, run, prog = line.rstrip("\n").split("\t")
            out[label] = (run, prog)
    return out


def epoch(stamp: str, frac: str | None) -> float:
    t = calendar.timegm(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    return t + (float("0." + frac) if frac else 0.0)


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


class Run:
    """One run's events, by request and by kind."""

    def __init__(self, label: str):
        self.label = label
        self.run, self.prog = runs()[label]
        self.req: dict[str, dict] = collections.defaultdict(lambda: collections.defaultdict(list))
        self.kind: dict[str, list] = collections.defaultdict(list)
        self.batches: list[tuple] = []  # (t, kind, usage, running, queue, new_tokens|None)
        self.client: dict[str, dict] = {}
        self.t0: float | None = None
        self.tag: str | None = None
        self._load_client()
        self._load_evt()

    def _load_client(self) -> None:
        path = os.path.join(RESULTS, self.run, self.prog, "client.jsonl")
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                kind = rec.get("kind")
                if kind == "run":
                    self.t0 = rec["t_start"]
                    self.tag = rec.get("tag")
                elif kind == "req" and rec.get("rid"):
                    self.client[rec["rid"]] = rec
                elif kind == "turn" and "conv" in rec:
                    # replay_agentic.py records (campaign 26): the rid is <tag>-c<conv>-t<turn>.
                    rec["rid"] = f"{self.tag}-c{rec['conv']}-t{rec['turn']}"
                    rec["wait_s"] = rec.get("gap_slept")
                    self.client[rec["rid"]] = rec

    def _load_evt(self) -> None:
        path = os.path.join(HERE, "evt", f"{self.label}.evt")
        opener = open
        if not os.path.exists(path):
            path += ".gz"
            opener = gzip.open
        with opener(path, "rt", errors="replace") as f:
            for line in f:
                m = _STAMP.match(line)
                if not m:
                    continue
                t = epoch(m[1], m[2])
                body = m[3]
                if body.startswith("HICACHE_EVT "):
                    parts = body.split(" ", 2)
                    kind = parts[1]
                    fields = {k: _num(v) for k, v in _KV.findall(parts[2] if len(parts) > 2 else "")}
                    fields["t"] = t
                    self.kind[kind].append(fields)
                    rid = fields.get("rid")
                    if isinstance(rid, str):
                        self.req[rid][kind].append(fields)
                    continue
                r = _RTS.search(body)
                if r:
                    rec = self.req[r[1]]
                    rec["rts"] = {
                        "t": t,
                        "input_len": int(r[2]),
                        "cached_input_len": int(r[3]),
                        "output_len": int(r[4]),
                        "queue_s": float(r[5]) / 1e3,
                        "forward_s": float(r[6]) / 1e3,
                        "entry": float(r[7]),
                    }
                    continue
                p = _PREFILL.search(body)
                if p:
                    self.batches.append(
                        (t, "P", float(p[4]), int(p[5]), int(p[6]), int(p[2]), int(p[1]), int(p[3]))
                    )
                    continue
                d = _DECODE.search(body)
                if d:
                    self.batches.append((t, "D", float(d[3]), int(d[1]), int(d[4]), 0, 0, 0))
                    continue
                s = _SUCCESS.search(body)
                if s:
                    self.req[s[2]]["landed"].append(
                        {"t": t, "ok": s[1] == "success", "completed": int(s[3]), "matched": int(s[4]), "loaded": int(s[5])}
                    )
                    continue
                h = _SHRANK.search(body)
                if h:
                    self.req[h[1]]["shrank"].append({"t": t, "lookup_match": int(h[2]), "current_match": int(h[3])})
                    continue
                if "NIXL L3 cleanup" in body:
                    self.kind["cleanup"].append({"t": t, "body": body})
        if self.t0 is None:
            firsts = [r["rts"]["entry"] for r in self.req.values() if "rts" in r and isinstance(r["rts"], dict)]
            self.t0 = min(firsts) if firsts else 0.0

    # --- helpers -----------------------------------------------------------------------------------
    def requests(self):
        """(rid, rec) of client requests that the server finished (has ReqTimeStats), skipping probes."""
        for rid, rec in self.req.items():
            rts = rec.get("rts")
            if not isinstance(rts, dict):
                continue
            if rid.startswith("HEALTH_CHECK") or "-c" not in rid:
                continue
            yield rid, rec

    def minute(self, t: float) -> float:
        return (t - self.t0) / 60.0


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, max(0, int(p * len(xs))))]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")
