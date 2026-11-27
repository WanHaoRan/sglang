"""Shared loaders for the ev_prices checks (read-only on the run folders).

load_server(camp, arm): streams <run>/server.log once, keeps what the checks need and caches it as a
pickle under ../ext/. Times are seconds since the client's t_start (the client.jsonl run record), so they
line up with the client's t_send.
load_client(camp, arm): the client.jsonl records.
requests(camp, arm): one joined record per client request (client fields + the server's arrival, admission,
reads, lookups and gate decisions).
"""
import calendar
import gzip
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

PER_RID = {
    "prefetch_start", "s2h_query", "srs_park", "srs_dispatch", "s2h_io", "srs_pin", "srs_pin_skipped",
    "srs_unpin", "ntp_gate", "ntp_arrive", "ntp_admit", "ntp_reevict", "ntp_refetch", "ntp_due", "ntp_hold",
    "ntp_stage", "ntp_room", "ntp_plan", "ntp_pick", "ntp_expire", "ec_start", "ec_release", "ec_bump",
    "load_back_init", "nue_register",
}
GLOBAL = {
    "evict_host", "evict_device", "host_census", "ec_price", "srs_write_hold", "srs_rate", "h2s_io",
    "h2s_group", "h2d_done",
}
_DATE = {}


def runs(camp):
    out = {}
    for line in open(MANIFEST[camp]):
        p = line.split()
        if len(p) >= 3:
            out[p[0]] = (f"{RES}/{p[1]}", f"{RES}/{p[1]}/{p[2]}")
    return out


def _ts(s):
    """'YYYY-MM-DD HH:MM:SS.mmm' (UTC) -> epoch seconds."""
    d = s[:10]
    base = _DATE.get(d)
    if base is None:
        base = _DATE[d] = calendar.timegm((int(d[:4]), int(d[5:7]), int(d[8:10]), 0, 0, 0))
    return base + int(s[11:13]) * 3600 + int(s[14:16]) * 60 + float(s[17:])


def _num(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _fields(rest):
    out = {}
    for tok in rest.split():
        k, _, v = tok.partition("=")
        out[k] = _num(v)
    return out


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
CAP = re.compile(r"reissue cap reached req=(\S+) attempts=(\d+)")
SHRANK = re.compile(r"device prefix shrank before admission req=(\S+) lookup_match=(\d+) current_match=(\d+)")
FAIL = re.compile(r"Prefetch operation (\S+) failed to retrieve page")
PFOK = re.compile(r"HiCache prefetch (success|dropped) req=(\S+) completed=(\d+) matched=(\d+) loaded=(\d+) occupied=(\d+)")


def client_t0(prog):
    with open(f"{prog}/client.jsonl") as f:
        return json.loads(f.readline())["t_start"]


def load_server(camp, arm, rebuild=False):
    os.makedirs(EXT, exist_ok=True)
    cache = f"{EXT}/{camp}_{arm}.server.pkl"
    if os.path.exists(cache) and not rebuild:
        with open(cache, "rb") as f:
            return pickle.load(f)
    run, prog = runs(camp)[arm]
    t0 = client_t0(prog)
    d = {
        "t0": t0, "batch": [], "rts": {}, "ev": {}, "glob": {k: [] for k in GLOBAL}, "cleanup": [],
        "capreach": {}, "shrank": {}, "readfail": [], "run": run, "prog": prog,
    }
    ev, glob, batch = d["ev"], d["glob"], d["batch"]
    with open(f"{run}/server.log", errors="replace") as f:
        for line in f:
            if not line.startswith("[20"):
                continue
            body = line[26:]
            if body.startswith("HICACHE_EVT "):
                name, _, rest = body[12:].rstrip("\n").partition(" ")
                if name in PER_RID:
                    fl = _fields(rest)
                    rid = fl.pop("rid", None)
                    if rid is None:
                        continue
                    ev.setdefault(str(rid), []).append((_ts(line[1:24]) - t0, name, fl))
                elif name in GLOBAL:
                    glob[name].append((_ts(line[1:24]) - t0, _fields(rest)))
                continue
            if body.startswith("Prefill batch"):
                m = PRE.search(body)
                if m:
                    batch.append((_ts(line[1:24]) - t0, "P", int(m[5]), int(m[6]), float(m[4]), int(m[1]), int(m[2]), int(m[3])))
                continue
            if body.startswith("Decode batch"):
                m = DEC.search(body)
                if m:
                    batch.append((_ts(line[1:24]) - t0, "D", int(m[1]), int(m[4]), float(m[3]), 0, int(m[2]), 0))
                continue
            if body.startswith("ReqTimeStats("):
                m = RTS.search(body)
                if m:
                    q = float(m[5]) / (1000.0 if m[6] == "ms" else 1.0)
                    fw = float(m[7]) / (1000.0 if m[8] == "ms" else 1.0)
                    d["rts"][m[1]] = (_ts(line[1:24]) - t0, int(m[2]), int(m[3]), int(m[4]), q, fw, float(m[9]) - t0)
                continue
            if "NIXL L3 cleanup" in body:
                m = CLEAN.search(body)
                if m:
                    d["cleanup"].append((_ts(line[1:24]) - t0, int(m[1]), float(m[3]), float(m[4])))
                continue
            if "reissue cap reached" in body:
                m = CAP.search(body)
                if m:
                    d["capreach"].setdefault(m[1], _ts(line[1:24]) - t0)
                continue
            if "device prefix shrank" in body:
                m = SHRANK.search(body)
                if m:
                    d["shrank"].setdefault(m[1], []).append((_ts(line[1:24]) - t0, int(m[2]), int(m[3])))
                continue
            if "failed to retrieve page" in body:
                m = FAIL.search(body)
                if m:
                    d["readfail"].append((_ts(line[1:24]) - t0, m[1]))
                continue
            if body.startswith("HiCache prefetch "):
                m = PFOK.search(body)
                if m:
                    ev.setdefault(m[2], []).append((_ts(line[1:24]) - t0, "pf_" + m[1], {
                        "completed": int(m[3]), "matched": int(m[4]), "loaded": int(m[5])}))
    with open(cache, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def load_client(camp, arm):
    run, prog = runs(camp)[arm]
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


def requests(camp, arm):
    """Joined per-request records (dicts), in client order. Server-side fields are None when the server
    logged nothing for the rid."""
    cache = f"{EXT}/{camp}_{arm}.reqs.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    srv = load_server(camp, arm)
    cl = load_client(camp, arm)
    rows = []
    for r in cl["req"]:
        rid = r["rid"]
        rts = srv["rts"].get(rid)
        evs = srv["ev"].get(rid, [])
        row = {
            "rid": rid, "play": r["play"], "tree": r.get("tree"), "stream": r["stream"], "k": r["k"],
            "role": r.get("role"), "oracle_gap_s": r.get("oracle_gap_s"),
            "t_send": r["t_send"], "ttft": r.get("ttft"), "latency": r.get("latency"),
            "prompt": r.get("prompt_tokens"), "out": r.get("completion_tokens"),
            "cached": r.get("cached_tokens") or 0, "ideal": r.get("ideal_cached") or 0,
            "cd": r.get("cached_details") or {}, "gap": r.get("gap"), "wait_s": r.get("wait_s"),
            "key": r.get("key"), "hint": r.get("hint"),
        }
        if rts:
            row.update(fin=rts[0], in_len=rts[1], cached_srv=rts[2], out_len=rts[3], queue=rts[4], fwd=rts[5],
                       arr=rts[6], adm=rts[6] + rts[4])
        else:
            row.update(fin=None, in_len=None, cached_srv=None, out_len=None, queue=None, fwd=None, arr=None, adm=None)
        # lookups: (t_issue, t_answer, tokens_req, storage_hit). reads: one dict per lookup that hit:
        #   t0 lookup issue, park, disp, land (s2h_io line), done (pin / prefetch success line), pages,
        #   loaded (tokens the read put in host memory), ms (I/O time), launched, deadline (ms from park).
        # busy: [start, end] spans in which the request had a lookup or read under way (admission passes
        #   over it); end None = still open at the end of the log.
        lookups, reads, gates, reev, busy = [], [], [], [], []
        cur = None
        ps = None
        arrive = admit = hold = None
        staged = False
        for t, name, fl in evs:
            if name == "prefetch_start":
                ps = t
            elif name == "s2h_query":
                hit = fl.get("storage_hit", 0)
                t_issue = ps if ps is not None and t - ps < 30.0 else t
                lookups.append((t_issue, t, fl.get("tokens_req", 0), hit))
                if hit > 0:
                    cur = {"t0": t_issue, "hit": hit, "park": None, "pages": hit // PAGE, "deadline": None,
                           "disp": None, "land": None, "done": None, "loaded": None, "ms": None,
                           "launched": None, "waited_ms": None, "failed": False}
                    reads.append(cur)
                else:
                    busy.append([t_issue, t])
            elif name == "srs_park":
                if cur is None or cur["park"] is not None:
                    cur = {"t0": t, "hit": fl.get("pages", 0) * PAGE, "park": None, "pages": fl.get("pages", 0),
                           "deadline": None, "disp": None, "land": None, "done": None, "loaded": None, "ms": None,
                           "launched": None, "waited_ms": None, "failed": False}
                    reads.append(cur)
                cur["park"] = t
                cur["pages"] = fl.get("pages", 0)
                cur["deadline"] = fl.get("deadline_ms")
            elif name == "srs_dispatch":
                if cur is not None and cur["disp"] is None:
                    cur["disp"] = t
                    cur["launched"] = fl.get("launched")
                    cur["waited_ms"] = fl.get("waited_ms")
                    if not fl.get("launched"):
                        cur["done"] = t
                        cur["loaded"] = 0
            elif name == "s2h_io":
                if cur is not None and cur["land"] is None:
                    cur["land"] = t
                    cur["ms"] = fl.get("ms", 0)
                    cur["pages"] = cur["pages"] or fl.get("pages", 0)
            elif name in ("pf_success", "pf_dropped"):
                if cur is not None and cur["done"] is None:
                    cur["done"] = t
                    cur["loaded"] = fl.get("loaded", 0) if name == "pf_success" else 0
                    cur["completed"] = fl.get("completed", 0)
            elif name == "ntp_gate":
                gates.append((t, fl.get("decision"), fl.get("matched"), fl.get("chain"), fl.get("lookups")))
            elif name == "ntp_reevict":
                reev.append((t, fl.get("evicted"), fl.get("refetches"), fl.get("capped")))
            elif name == "ntp_arrive":
                arrive = (t, fl.get("pos"), fl.get("qlen"))
            elif name == "ntp_admit":
                admit = (t, fl.get("wait_ms"), fl.get("pred_wait_ql_ms"))
            elif name == "ntp_hold" and hold is None:
                hold = (t, fl.get("pos"), fl.get("hold_ms"), fl.get("wait_ms"), fl.get("load_ms"))
            elif name == "ntp_stage":
                staged = True
        for rd in reads:
            end = rd["done"] if rd["done"] is not None else rd["land"]
            busy.append([rd["t0"], end])
        busy.sort(key=lambda b: b[0])
        row.update(lookups=lookups, reads=reads, gates=gates, reev=reev, busy=busy, ntp_arrive=arrive,
                   ntp_admit=admit, hold=hold, staged=staged, capreach=srv["capreach"].get(rid),
                   shrank=srv["shrank"].get(rid, []))
        rows.append(row)
    with open(cache, "wb") as f:
        pickle.dump(rows, f, protocol=pickle.HIGHEST_PROTOCOL)
    return rows


def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = p * (len(xs) - 1)
    lo = int(i)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


if __name__ == "__main__":
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        d = load_server(camp, arm, rebuild=True)
        print(spec, "batch", len(d["batch"]), "rts", len(d["rts"]), "rids", len(d["ev"]),
              {k: len(v) for k, v in d["glob"].items()}, "cleanup", len(d["cleanup"]),
              "capreach", len(d["capreach"]), "readfail", len(d["readfail"]), flush=True)
