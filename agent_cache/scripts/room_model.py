"""CPU model of staged SSD reads against the host room, calibrated on the 44-user just-in-time runs
(PAPER_PLAN.md, "Formulation" and "Design point", step 1 of "Next").

  python3 room_model.py measure     # each run's launches, revokes, pinned tokens and landing leads, from its logs
  python3 room_model.py calibrate   # replay each run's own read timing through the model; sweep the room
  python3 room_model.py policies    # read-timing rules on the p50 run's reads: quantiles, expected cost, forms, pins
  python3 room_model.py figure      # policy_tradeoff.png from policies.csv

The model keeps the run's turns, arrivals, queue waits and the pages each turn had to read from SSD (so eviction is
the run's), and changes only when each read starts and whether a landed read can be taken back. A read holds its
pages in the host room from launch until its turn is admitted; a launch that finds no room is revoked, as the
engine's host allocation is. One FIFO read thread serves dispatched reads at the run's measured rate.

Inputs: server.log (HICACHE_EVT lines) and client.jsonl (per-turn hints and true gaps) of each run below. Times are
seconds since the client's t_start, the server log read as UTC (as srs_summary.py does). Outputs: results/cpu_calib_c44/.
"""
import argparse
import bisect
import collections
import datetime as dt
import gzip
import heapq
import json
import math
import os
import re
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(RES, "cpu_calib_c44")
RUNS = {   # campaign 23, 44 users, 70 GB of host memory, gaps x10
    "exact": "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "p50": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
}
PAGE = 64
KV_RE = re.compile(r"(\w+)=(-?[\w./*:-]+)")
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")
RID_RE = re.compile(r"-c(\d+)-t(\d+)$")
KINDS = {"ntp_plan", "ntp_pick", "ntp_arrive", "ntp_admit", "ntp_due", "ntp_stage", "ntp_expire", "ntp_hold",
         "ntp_refetch", "ntp_gate", "srs_park", "srs_dispatch", "srs_pin", "srs_unpin", "s2h_io"}


def parse_ts(line: str) -> float:
    return (dt.datetime.strptime(TS_RE.match(line)[1], "%Y-%m-%d %H:%M:%S.%f")
            .replace(tzinfo=dt.timezone.utc).timestamp())


def client_dir(run: str) -> str:
    return os.path.join(run, sorted(d for d in os.listdir(run) if d.startswith("client_"))[-1])


def load_run(name: str) -> dict:
    """Per-turn records joined from the client and the server, cached as JSON."""
    cache = os.path.join(OUT, f"run_{name}.json.gz")
    if os.path.exists(cache):
        with gzip.open(cache, "rt") as f:
            return json.load(f)
    run = os.path.join(RES, RUNS[name])
    rows = [json.loads(line) for line in open(os.path.join(client_dir(run), "client.jsonl"))]
    t0 = next(r for r in rows if r["kind"] == "run")["t_start"]
    turns = {}
    for r in rows:
        if r["kind"] != "turn":
            continue
        hint = r.get("dist_hint") or r.get("oracle_hint")
        turns[(r["conv"], r["turn"])] = dict(
            conv=r["conv"], turn=r["turn"], t_send=r["t_send"], ttft=r.get("ttft"), latency=r.get("latency"),
            prompt=r.get("prompt_tokens"), delta=r.get("delta_tokens"), cached=r.get("cached_details"),
            gap_slept=r.get("gap_slept"), next_hint=hint)
    events = collections.defaultdict(list)
    pinned, io = [], []
    for line in open(os.path.join(run, "server.log"), errors="replace"):
        if "HICACHE_EVT" not in line:
            continue
        kind = line.split("HICACHE_EVT ", 1)[1].split(" ", 1)[0]
        if kind not in KINDS:
            continue
        t = parse_ts(line) - t0
        kv = dict(KV_RE.findall(line.split(kind, 1)[1]))
        if kind in ("srs_pin", "srs_unpin"):
            pinned.append((t, int(kv["pinned"])))
        if kind == "s2h_io":
            io.append((t, int(kv["pages"]), float(kv["ms"])))
        m = RID_RE.search(kv.get("rid", ""))
        if m:
            events[(int(m[1]), int(m[2]))].append((t, kind, kv))
    for key, evs in events.items():
        if key in turns:
            turns[key]["events"] = evs
    out = dict(name=name, turns=[turns[k] for k in sorted(turns)], pinned=pinned, io=io)
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(cache, "wt") as f:
        json.dump(out, f)
    return out


def first(evs: list, kind: str):
    return next(((t, kv) for t, k, kv in evs if k == kind), (None, None))


def reads_of(evs: list) -> list:
    """The turn's SSD reads in order: park (time, pages, timed or now), dispatch (time, launched), landing time."""
    reads, pending = [], collections.deque()
    for t, k, kv in evs:
        if k == "srs_park":
            pending.append(dict(park=t, pages=int(kv["pages"]), timed=kv["deadline_ms"] != "now"))
        elif k == "srs_dispatch" and pending:
            r = pending.popleft()
            r.update(dispatch=t, launched=kv["launched"] == "1")
            reads.append(r)
        elif k == "s2h_io":
            for r in reversed(reads):
                if r.get("launched") and "land" not in r:
                    r["land"] = t
                    break
    return reads


def time_average(series: list, t_end: float) -> float:
    """Mean of a step series [(t, value)] over [first t, t_end]."""
    if not series:
        return 0.0
    area, (t_prev, v_prev) = 0.0, series[0]
    for t, v in series[1:]:
        area += v_prev * (t - t_prev)
        t_prev, v_prev = t, v
    area += v_prev * max(0.0, t_end - t_prev)
    return area / max(1e-9, t_end - series[0][0])


def pct(values: list, q: float):
    values = sorted(values)
    return round(values[int(q * (len(values) - 1))], 2) if values else None


def measure(name: str) -> dict:
    """The run's own numbers, with the model's definitions: a read is staged when parked before its turn arrives."""
    run = load_run(name)
    turns, t_end = run["turns"], max(t for t, _ in run["pinned"])
    m = collections.Counter()
    leads, early_pin_ts = [], 0.0
    for tr in turns:
        evs = tr.get("events", [])
        t_arr, _ = first(evs, "ntp_arrive")
        t_adm, _ = first(evs, "ntp_admit")
        if t_arr is None or t_adm is None:
            continue
        for r in reads_of(evs):
            kind = "staged" if r["park"] < t_arr else "arrival"
            m[f"{kind}_reads"] += 1
            m[f"{kind}_revoked"] += not r["launched"]
            land = r.get("land")
            if land is None:
                continue
            m["landed_after_arrival"] += land > t_arr
            if land <= t_adm:
                if kind == "staged":
                    leads.append(t_adm - land)
                early_pin_ts += max(0.0, min(t_arr, t_adm) - land) * r["pages"] * PAGE
    span = t_end - min(tr["t_send"] for tr in turns)
    return dict(
        name=name, staged_reads=m["staged_reads"], staged_revoked=m["staged_revoked"],
        arrival_reads=m["arrival_reads"], arrival_revoked=m["arrival_revoked"],
        mean_pinned=round(time_average(run["pinned"], t_end)), pinned_before_arrival=round(early_pin_ts / span),
        lead_p50=pct(leads, 0.5), lead_p90=pct(leads, 0.9), landed_after_arrival=m["landed_after_arrival"],
        span_s=round(span))


# ===== Inputs of the model =====

SSD_FIXED_S = 0.010        # small reads take 9-11 ms in both runs' s2h_io lines
RATE_BIN_S = 600
WAIT_HALF_WINDOW_S = 30.0
GAP_CAP_S = 1200.0         # the replay client caps scaled gaps here (run args gap_cap)
LN_GRID = (math.log(0.05), math.log(GAP_CAP_S), 24)   # the histogram form's shared grid: 24 log-spaced bins
SPLIT_POINTS = 100         # support points of the three-quantile form
SAMPLE_POINTS = 200        # cap on support points of the raw-sample form


def ms_per_page(run: dict):
    """Step function: the median read-thread ms per page of reads of 32+ pages in each 10-minute bin."""
    bins = collections.defaultdict(list)
    for t, pages, ms in run["io"]:
        if pages >= 32:
            bins[int(t // RATE_BIN_S)].append(ms / pages)
    table = {b: st.median(v) / 1000 for b, v in bins.items()}
    overall = st.median([v for vs in bins.values() for v in vs]) / 1000
    return lambda t: table.get(int(t // RATE_BIN_S), overall)


def read_gated(evs: list, t_adm: float) -> bool:
    """Admitted within 150 ms of one of its reads landing: the read, not the GPU, set the admission."""
    return any(r.get("land") is not None and 0 <= t_adm - r["land"] < 0.15 for r in reads_of(evs))


def queue_wait(turns: list):
    """Queue wait of a reading turn arriving at t: the median measured wait of reading turns whose admission no read
    gated, arriving within 30 s. Turns without reads are no guide: they cluster where the GPU is not congested."""
    pts = []
    for tr in turns:
        evs = tr.get("events", [])
        t_arr, _ = first(evs, "ntp_arrive")
        t_adm, _ = first(evs, "ntp_admit")
        if t_arr is not None and t_adm is not None and reads_of(evs) and not read_gated(evs, t_adm):
            pts.append((t_arr, t_adm - t_arr))
    pts.sort()
    times = [t for t, _ in pts]

    def wait_at(t: float) -> float:
        lo = bisect.bisect_left(times, t - WAIT_HALF_WINDOW_S)
        hi = bisect.bisect_right(times, t + WAIT_HALF_WINDOW_S)
        return st.median(w for _, w in pts[lo:hi]) if hi > lo else 0.0
    return wait_at


def online_samples(name: str, run: dict) -> dict:
    """(conv, turn) -> sorted ln-gaps of the entry the client matched, as its online table held them when the
    previous turn was sent. Rebuilt per agent system (source, orig_model); it reproduces every hint the run sent."""
    sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "traces", "lmcache_agentic_trace_262k.json")
    model_of = {s["session_id"]: (s.get("source"), s.get("orig_model")) for s in json.load(open(sys_path))["sessions"]}
    rows = [json.loads(line) for line in open(os.path.join(client_dir(os.path.join(RES, RUNS[name])), "client.jsonl"))]
    system = {r["conv"]: model_of[r["session_id"]] for r in rows if r["kind"] == "conv_start"}
    from gap_hints import key_levels, GAP_FLOOR_S
    by_conv = collections.defaultdict(dict)
    for tr in run["turns"]:
        by_conv[tr["conv"]][tr["turn"]] = tr
    seen = []
    for conv, ts in by_conv.items():
        for k, tr in ts.items():
            prev = ts.get(k - 1)
            hint = (prev or {}).get("next_hint") or {}
            if hint.get("reply_key") not in (None, "_end") and tr.get("gap_slept") is not None:
                seen.append((tr["t_send"], system[conv], hint["reply_key"], tr["gap_slept"]))
    seen.sort()
    table, out, i = collections.defaultdict(list), {}, 0
    for tr in sorted(run["turns"], key=lambda r: r["t_send"]):
        while i < len(seen) and seen[i][0] <= tr["t_send"]:
            _, sysk, key, gap = seen[i]
            for level in key_levels(key, "command"):
                bisect.insort(table[(sysk, level)], math.log(max(gap, GAP_FLOOR_S)))
            i += 1
        hint = tr.get("next_hint") or {}
        if hint.get("matched") not in (None, "_end") and hint.get("p50") is not None:
            out[(tr["conv"], tr["turn"] + 1)] = list(table[(system[tr["conv"]], hint["matched"])])
    return out


def split_lognormal(p10: float, p50: float, p90: float):
    """Support points of the two-piece lognormal through p10, p50 and p90: the three-quantile form."""
    z90 = 1.2815515655446004
    lo, hi = math.log(p50 / p10) / z90, math.log(p90 / p50) / z90
    xs = []
    for k in range(SPLIT_POINTS):
        u = (k + 0.5) / SPLIT_POINTS
        z = st.NormalDist().inv_cdf(u)
        xs.append(min(GAP_CAP_S, p50 * math.exp((lo if z < 0 else hi) * z)))
    return xs, [1.0 / SPLIT_POINTS] * SPLIT_POINTS


def histogram_form(ln_samples: list):
    """Counts on the shared 24-bin log grid, spread over four points per bin: the histogram form."""
    lo, hi, n = LN_GRID
    width = (hi - lo) / n
    counts = [0] * n
    for x in ln_samples:
        counts[min(n - 1, max(0, int((x - lo) / width)))] += 1
    total = sum(counts)
    xs, ws = [], []
    for b, c in enumerate(counts):
        if c:
            for j in range(4):
                xs.append(math.exp(lo + width * (b + (j + 0.5) / 4)))
                ws.append(c / total / 4)
    return xs, ws


def sample_form(ln_samples: list):
    """The empirical distribution of every gap the entry has seen, as at most 200 quantile points."""
    n = len(ln_samples)
    k = min(n, SAMPLE_POINTS)
    xs = [math.exp(ln_samples[min(n - 1, int((j + 0.5) * n / k))]) for j in range(k)]
    return xs, [1.0 / k] * k


def build_jobs(name: str) -> dict:
    """The turns that read history from SSD in the run, with what each policy needs to time their reads."""
    run = load_run(name)
    turns = {(tr["conv"], tr["turn"]): tr for tr in run["turns"]}
    rate, wait_at = ms_per_page(run), queue_wait(run["turns"])
    samples = online_samples(name, run) if name == "p50" else {}
    offsets, jobs = [], []
    for (conv, turn), tr in sorted(turns.items()):
        evs = tr.get("events", [])
        t_arr, _ = first(evs, "ntp_arrive")
        t_plan, plan = first(evs, "ntp_plan")
        prev = turns.get((conv, turn - 1)) or {}
        hint = prev.get("next_hint") or {}
        true_gap = hint.get("true_gap_s", hint.get("next_gap_s"))
        if t_arr is not None and t_plan is not None and true_gap is not None:
            offsets.append(t_arr - t_plan - true_gap)
        reads = reads_of(evs)
        if t_arr is None or first(evs, "ntp_admit")[0] is None or not reads:
            continue
        pages = reads[0]["pages"]
        # Re-reads after a landed read: history the run lost while the turn waited (its eviction, not the timing).
        rereads = [(r["park"], r["pages"]) for prev_r, r in zip(reads, reads[1:]) if prev_r["launched"]]
        t_adm, _ = first(evs, "ntp_admit")
        # The GPU's queue wait: measured, unless a landing read set the admission; then the local estimate.
        wait = min(t_adm - t_arr, wait_at(t_arr)) if read_gated(evs, t_adm) else t_adm - t_arr
        job = dict(conv=conv, turn=turn, A=t_arr, W=max(0.0, wait), pages=pages, tokens=pages * PAGE,
                   plan=plan is not None, stageable=reads[0]["park"] < t_arr, rereads=rereads)
        if plan is not None:
            _, due = first(evs, "ntp_due")   # the engine's own load and wait estimates when it timed the read
            r_pred = float(due["load_ms"]) / 1000 if due else SSD_FIXED_S + pages * rate(t_plan)
            w_pred = float(due["wait_ms"]) / 1000 if due else wait_at(t_plan)
            job.update(e=t_plan, pred_arr=t_plan + float(plan["pred_arr_ms"]) / 1000, p50=float(plan["gap_ms"]) / 1000,
                       W_pred=w_pred, r_pred=r_pred, gap=true_gap)
            if "p10" in hint:
                job.update(p10=hint["p10"], p90=hint["p90"], n=hint["n"], ln_samples=samples.get((conv, turn)))
        job["first_park"] = reads[0]["park"]
        jobs.append(job)
    delta = st.median(offsets)
    for job in jobs:
        # The history goes missing when the run evicts it; the engine probes every 0.5 s and stages only what is
        # missing. A staged first read parked well after the routine's trigger marks that time; no rule reads earlier.
        start, _ = plan_start(job, dict(rule="asrun"), delta)
        job["avail"] = job["first_park"] if start is not None and job["first_park"] - start > PROBE_S else None
    return dict(jobs=jobs, rate=rate, delta=delta, name=name)


def support(job: dict, form: str):
    """The gap distribution's support points under one form; cached on the job."""
    key = f"_support_{form}"
    if key not in job:
        if form == "exact":
            job[key] = ([job["gap"]], [1.0])
        elif form == "split":
            job[key] = split_lognormal(job["p10"], job["p50"], job["p90"])
        elif form == "hist":
            job[key] = histogram_form(job["ln_samples"])
        elif form == "samples":
            job[key] = sample_form(job["ln_samples"])
        else:
            raise ValueError(form)
    return job[key]


# ===== Read-start rules =====

JIT_MARGIN_S = 1.0     # NTP_JIT_MARGIN_S: reads are timed to land 1 s before the forecast admission
HOLD_MARGIN_S = 2.0    # NTP_HOLD_MARGIN_S: a queued turn's read waits until admission - read time - 2 s
MIN_EXPIRY_S = 60.0
PROBE_S = 0.5          # PROBE_INTERVAL_S: a pending plan rechecks what is missing this often


def quantile_of(xs: list, ws: list, q: float) -> float:
    acc = 0.0
    for x, w in sorted(zip(xs, ws)):
        acc += w
        if acc >= q:
            return x
    return max(xs)


def expected_cost_start(job: dict, *, lam: float, form: str, delta: float):
    """Rule 1 at the gap's start: the start offset minimizing E[stall] + lam * tokens * E[seconds pinned], with
    the turn's own arrival starting the read if it comes first; None = read at arrival."""
    import numpy as np
    xs, ws = support(job, form)
    x, w = np.asarray(xs), np.asarray(ws)
    r = job["r_pred"] + JIT_MARGIN_S
    wq = job["W_pred"]
    arrive = delta + x
    due = arrive + wq
    cands = np.unique(np.concatenate(([0.0], np.maximum(0.0, due - r), arrive)))
    s = cands[:, None]
    first_arrival = arrive[None, :] <= s
    land = s + r
    stall = np.where(first_arrival, max(0.0, r - wq), np.maximum(0.0, land - due[None, :]))
    pinned = np.where(first_arrival, 0.0, np.maximum(0.0, due[None, :] - land))
    cost = (stall + lam * job["tokens"] * pinned) @ w
    at_arrival = float(w.sum()) * max(0.0, r - wq)
    best = int(np.argmin(cost))
    return None if at_arrival <= cost[best] else float(cands[best])


def plan_start(job: dict, policy: dict, delta: float):
    """(start time of the staged read or None, expiry time or None) under one policy."""
    if not job["plan"] or not job["stageable"] or policy["rule"] == "arrival":
        return None, None
    rule, r = policy["rule"], job["r_pred"]
    if rule == "asrun":   # the run's own routine: deadline at the p50, early by p50 - p10, expiry
        early = max(0.0, job["p50"] - job["p10"]) if "p10" in job else 0.0
        start = job["pred_arr"] + job["W_pred"] - r - JIT_MARGIN_S - early
        expiry = job["pred_arr"] + max(MIN_EXPIRY_S, job["p50"]) if policy.get("expiry", True) else None
        return max(job["e"], start), expiry
    if rule == "oracle":
        return max(job["e"], job["A"] + job["W"] - r - JIT_MARGIN_S), None
    if "p10" not in job:
        return None, None
    if rule == "quantile":
        target = job["e"] + delta + quantile_of(*support(job, policy["form"]), policy["q"])
        return max(job["e"], target + job["W_pred"] - r - JIT_MARGIN_S), None
    if rule == "cost":
        s = expected_cost_start(job, lam=policy["lam"], form=policy["form"], delta=delta)
        return (None if s is None else job["e"] + s), None
    raise ValueError(rule)


def chance_soon(job: dict, t: float, horizon: float, form: str, delta: float) -> float:
    """F(horizon | t): the chance the turn arrives within `horizon` given it has not arrived by t."""
    xs, ws = support(job, form)
    idle = t - job["e"] - delta
    alive = sum(w for x, w in zip(xs, ws) if x > idle)
    soon = sum(w for x, w in zip(xs, ws) if idle < x <= idle + horizon)
    return soon / alive if alive > 0 else 1.0


# ===== The simulation =====

WINDOW_PAGES = 512     # the read scheduler's window: timed reads dispatch only below it
RHO = 9000.0           # GPU prefill, tokens/s (first-turn prefill 0.108-0.113 ms per token)
RETRY_S = 1.0          # median delay from a revoked read to its retry in both runs (0.54 and 1.08 s)
MAX_LOOKUPS = 3        # MAX_REFETCHES = 2 re-reads after the first lookup


def simulate(data: dict, policy: dict, *, room, per_job: bool = False) -> dict:
    """Replay the run's reads under one read-start policy. `room` is tokens, or a function of time.

    Each turn has a primary read (staged before arrival, or at arrival under the hold) and the run's re-reads after a
    landed read (history lost while the turn waited), replayed at their logged times. A launch that finds no room is
    revoked; a turn retries 1 s later while it waits, up to three lookups, then recomputes. Soft pins let an arrived
    turn's read take room from the landed staged reads least likely to be needed within their read time."""
    jobs, rate, delta = data["jobs"], data["rate"], data["delta"]
    room_at = room if callable(room) else (lambda t: room)
    soft, f_form = policy.get("soft", False), policy.get("form", "split")
    heap, seq = [], [0]

    def push(t, kind, x):
        heapq.heappush(heap, (t, seq[0], kind, x))
        seq[0] += 1

    st_ = [dict(primary=None, extra=set(), arrived=False, admitted=False, waiting=False, lookups=0,
                recompute=0, stall=0.0) for _ in jobs]
    reads, landed_before, leads = [], set(), []
    m = collections.Counter()
    occ = dict(flight=0, pinned=0, pages=0, before=0)
    area = dict(total=0.0, before=0.0)
    clock, server_free = [None], [0.0]
    now_q, timed_q = collections.deque(), []

    def due(j):
        return jobs[j]["A"] + jobs[j]["W"]

    for j, job in enumerate(jobs):
        start, expiry = plan_start(job, policy, delta)
        if start is not None and job.get("avail") is not None:
            start = max(start, job["avail"])
        if start is not None and start < job["A"]:
            push(start, "stage", j)
            if expiry is not None and expiry < job["A"]:
                push(expiry, "expire", j)
        push(job["A"], "arrive", j)
        push(due(j), "admit", j)
        for t_b, pages in job["rereads"]:
            push(t_b, "reread", (j, pages))

    def new_read(j, *, staged, extra=False, pages=None, t=None):
        pages = jobs[j]["pages"] if pages is None else pages
        reads.append(dict(job=j, pages=pages, tokens=pages * PAGE, staged=staged, extra=extra, status="parked",
                          tries=1))
        i = len(reads) - 1
        if extra:
            st_[j]["extra"].add(i)
        else:
            st_[j]["primary"] = i
            st_[j]["lookups"] += 1
        if staged:
            heapq.heappush(timed_q, (jobs[j]["pred_arr"] + jobs[j]["W_pred"], i))
        else:
            now_q.append(i)
        return i

    def release(i):
        rd = reads[i]
        if rd["status"] == "landed":
            occ["pinned"] -= rd["tokens"]
            if i in landed_before:
                landed_before.discard(i)
                occ["before"] -= rd["tokens"]
        rd["status"] = "done"

    def fits(need, t):
        return occ["flight"] + occ["pinned"] + need <= room_at(t)

    def reclaim(need, t):
        """Soft pins: take room from landed staged reads, least likely needed within their read time first."""
        victims = sorted(landed_before, key=lambda i2: chance_soon(jobs[reads[i2]["job"]], t,
                                                                  jobs[reads[i2]["job"]]["r_pred"], f_form, delta))
        for i2 in victims:
            if fits(need, t):
                return
            j2 = reads[i2]["job"]
            release(i2)
            st_[j2]["primary"] = None
            m["reclaimed"] += 1
            m["wasted_pages"] += reads[i2]["pages"]

    def maybe_admit(j, t):
        """A waiting turn is admitted once none of its reads is parked or in flight."""
        s = st_[j]
        if not s["waiting"] or s["admitted"]:
            return
        live = [i for i in ([s["primary"]] if s["primary"] is not None else []) + list(s["extra"])
                if reads[i]["status"] in ("parked", "flight")]
        if not live:
            push(t, "admit", j)

    def revoked(i, t):
        rd = reads[i]
        j, s = rd["job"], st_[rd["job"]]
        kind = "staged" if rd["staged"] else ("reread" if rd["extra"] else "arrival")
        m[f"{kind}_revoked"] += 1
        rd["status"] = "done"
        retry = s["arrived"] and not s["admitted"] and t + RETRY_S < max(due(j), t) and \
            (rd["tries"] if rd["extra"] else s["lookups"]) < MAX_LOOKUPS
        if rd["extra"]:
            s["extra"].discard(i)
            if retry:
                push(t + RETRY_S, "retry_extra", (j, rd["pages"], rd["tries"] + 1))
            else:
                s["recompute"] += rd["tokens"]
        else:
            s["primary"] = None
            if retry:
                push(t + RETRY_S, "now_read", j)
            elif s["arrived"]:
                s["recompute"] += rd["tokens"]
                s["gave_up"] = True
        maybe_admit(j, t)

    def launch(i, t):
        rd = reads[i]
        kind = "staged" if rd["staged"] else ("reread" if rd["extra"] else "arrival")
        m[f"{kind}_reads"] += 1
        if not fits(rd["tokens"], t) and soft and not rd["staged"]:
            reclaim(rd["tokens"], t)
        if not fits(rd["tokens"], t):
            revoked(i, t)
            return
        begin = max(t, server_free[0])
        end = begin + SSD_FIXED_S + rd["pages"] * rate(begin)
        server_free[0] = end
        occ["flight"] += rd["tokens"]
        occ["pages"] += rd["pages"]
        rd["status"] = "flight"
        push(end, "land", i)

    def dispatch(t):
        while now_q:
            i = now_q.popleft()
            if reads[i]["status"] == "parked":
                launch(i, t)
        while timed_q and occ["pages"] < WINDOW_PAGES:
            _, i = heapq.heappop(timed_q)
            if reads[i]["status"] == "parked":
                launch(i, t)

    def arrival_read(j, t):
        job = jobs[j]
        hold = due(j) - (SSD_FIXED_S + job["pages"] * rate(t)) - HOLD_MARGIN_S
        push(max(t, hold), "now_read", j)

    while heap:
        t, _, kind, x = heapq.heappop(heap)
        if clock[0] is not None and t > clock[0]:
            area["total"] += occ["pinned"] * (t - clock[0])
            area["before"] += occ["before"] * (t - clock[0])
        clock[0] = t
        if kind == "stage":
            if not st_[x]["arrived"] and st_[x]["primary"] is None:
                new_read(x, staged=True)
        elif kind == "expire":
            i = st_[x]["primary"]
            if not st_[x]["arrived"] and i is not None:
                if reads[i]["status"] == "landed":
                    release(i)
                    m["expired"] += 1
                    m["wasted_pages"] += reads[i]["pages"]
                    st_[x]["primary"] = None
                elif reads[i]["status"] == "flight":
                    reads[i]["abort"] = True
                else:
                    reads[i]["status"] = "done"
                    st_[x]["primary"] = None
        elif kind == "arrive":
            s = st_[x]
            s["arrived"] = True
            i = s["primary"]
            if i is not None and reads[i]["status"] == "landed" and i in landed_before:
                landed_before.discard(i)
                occ["before"] -= reads[i]["tokens"]
            elif i is not None and reads[i]["status"] == "parked":
                reads[i]["status"] = "done"
                s["primary"] = None
                s["lookups"] -= 1
                new_read(x, staged=False)
            elif i is None:
                arrival_read(x, t)
        elif kind == "now_read":
            s = st_[x]
            if not s["admitted"] and s["primary"] is None and not s.get("gave_up"):
                new_read(x, staged=False)
        elif kind == "reread":
            j, pages = x
            if not st_[j]["admitted"] and st_[j]["arrived"]:
                new_read(j, staged=False, extra=True, pages=pages)
        elif kind == "retry_extra":
            j, pages, tries = x
            if not st_[j]["admitted"]:
                reads[new_read(j, staged=False, extra=True, pages=pages)]["tries"] = tries
        elif kind == "land":
            rd = reads[x]
            j = rd["job"]
            occ["flight"] -= rd["tokens"]
            occ["pages"] -= rd["pages"]
            if rd.get("abort"):
                rd["status"] = "done"
                m["expired"] += 1
                m["wasted_pages"] += rd["pages"]
                if st_[j]["primary"] == x:
                    st_[j]["primary"] = None
                    if st_[j]["arrived"] and not st_[j]["admitted"]:
                        arrival_read(j, t)
            else:
                rd["status"], rd["land"] = "landed", t
                occ["pinned"] += rd["tokens"]
                m["landed_after_arrival"] += st_[j]["arrived"]
                if not st_[j]["arrived"] and not rd["extra"]:
                    landed_before.add(x)
                    occ["before"] += rd["tokens"]
                maybe_admit(j, t)
        elif kind == "admit":
            s = st_[x]
            if s["admitted"]:
                continue
            ids = ([s["primary"]] if s["primary"] is not None else []) + list(s["extra"])
            live = [i for i in ids if reads[i]["status"] in ("parked", "flight")]
            if live:
                s["waiting"] = True
                for i in live:
                    if reads[i]["status"] == "parked" and reads[i]["staged"]:
                        reads[i]["staged"] = False
                        now_q.append(i)
            else:
                s["admitted"] = True
                s["stall"] = max(0.0, t - due(x))
                i = s["primary"]
                if i is not None and reads[i]["status"] == "landed":
                    if reads[i]["staged"] and t <= due(x) + 1e-9:
                        leads.append(t - reads[i]["land"])
                elif not s.get("gave_up"):
                    s["recompute"] += jobs[x]["tokens"]
                for i2 in ids:
                    release(i2)
        dispatch(t)

    span = max(job["A"] for job in jobs) - min(job["A"] for job in jobs)
    stall = [s["stall"] for s in st_]
    recompute = sum(s["recompute"] for s in st_)
    out = dict(
        staged_reads=m["staged_reads"], staged_revoked=m["staged_revoked"],
        arrival_reads=m["arrival_reads"] + m["reread_reads"],
        arrival_revoked=m["arrival_revoked"] + m["reread_revoked"], reclaimed=m["reclaimed"], expired=m["expired"],
        mean_pinned=round(area["total"] / span), pinned_before_arrival=round(area["before"] / span),
        lead_p50=pct(leads, 0.5), lead_p90=pct(leads, 0.9), landed_after_arrival=m["landed_after_arrival"],
        stall_s=round(sum(stall)), late_turns=sum(1 for s in stall if s > 0), recompute_tokens=recompute,
        cost_s=round(sum(stall) + recompute / RHO), wasted_pages=m["wasted_pages"], jobs=len(jobs))
    if per_job:   # tokens each reading turn recomputed, in job order: prefill_curve.py prices them by position
        out["recompute_by_job"] = [s["recompute"] for s in st_]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("measure", "calibrate", "policies", "figure"))
    a = ap.parse_args()
    if a.step == "measure":
        for name in RUNS:
            print(json.dumps(measure(name)))
    elif a.step == "calibrate":
        calibrate()
    elif a.step == "policies":
        policies()
    elif a.step == "figure":
        figure()


SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COLORS = {"as run": "#a0307a", "arrival": "#7d5ae0", "exact": "#1baf7a", "quantile": "#2a78d6", "cost": "#d9741f"}


def figure() -> None:
    """policy_tradeoff.png: (a) mean pinned tokens against cost for the fixed-quantile and expected-cost rules,
    (b) each key rule's cost split into stall and recomputation, at the calibrated room."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    rows = [r for r in csv.DictReader(open(os.path.join(OUT, "policies.csv"))) if r["room"] == str(ROOM)]
    by = {r["policy"]: r for r in rows}
    cost = lambda r: float(r["cost_s"])
    pin = lambda r: float(r["mean_pinned"]) / 1000
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15.5, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, bx):
        a.set_facecolor(SURFACE)
        a.grid(True, color=GRID, linewidth=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    quant = sorted((r for r in rows if r["rule"] == "quantile" and not r["soft"]), key=lambda r: float(r["q"]))
    ec = sorted((r for r in rows if r["rule"] == "cost" and r["form"] == "split" and not r["soft"]),
                key=lambda r: float(r["lam"]))
    ax.plot([pin(r) for r in quant], [cost(r) for r in quant], "-o", color=COLORS["quantile"], lw=2, ms=6,
            label="Land at a fixed quantile, q = 0.1 to 0.9")
    ax.plot([pin(r) for r in ec], [cost(r) for r in ec], "-o", color=COLORS["cost"], lw=2, ms=6,
            label="Expected cost (rule 1), λ = 1e-6 to 1e-3")
    for r in quant:
        if float(r["q"]) in (0.1, 0.5, 0.9):
            ax.annotate(f"q={float(r['q']):g}", (pin(r), cost(r)), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, color=INK2)
    for r in ec:
        if float(r["lam"]) in (1e-6, 1e-5, 5e-5, 1e-3):
            ax.annotate(f"λ={float(r['lam']):g}", (pin(r), cost(r)), textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=9, color=INK2)
    for name, key, text, off, ha in (("JIT p50 as run", "as run", "JIT p50, as run", (-12, -4), "right"),
                                     ("at arrival", "arrival", "Read at arrival", (12, -4), "left"),
                                     ("exact arrival", "exact", "Exact arrival times", (12, -4), "left")):
        r = by[name]
        ax.plot(pin(r), cost(r), "s", color=COLORS[key], ms=10, mec=SURFACE, mew=2, label=text)
        ax.annotate(text, (pin(r), cost(r)), textcoords="offset points", xytext=off, ha=ha, fontsize=10, color=INK)
    ax.set_xlabel("Mean tokens pinned in host memory (thousands)")
    ax.set_ylabel("Cost: stall + recomputation, seconds (2,576 turns)")
    ax.set_title("(a) Pinned memory against cost, one point per setting of each rule", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    ax.set_xlim(4, 38)
    ax.set_ylim(0, max(cost(r) for r in rows) * 1.08)

    picks = (("JIT p50 as run", "as run", "JIT p50,\nas run"), ("at arrival", "arrival", "Read at\narrival"),
             ("land at q=0.9 (split)", "quantile", "Best fixed\nquantile (0.9)"),
             ("expected cost lam=5e-05 (split)", "cost", "Expected cost\nλ=5e-5"),
             ("expected cost lam=1e-05 (samples), soft pins", "cost", "Expected cost\n+ soft pins"),
             ("exact arrival", "exact", "Exact arrival\ntimes"))
    for k, (name, key, text) in enumerate(picks):
        r = by[name]
        stall, rec = float(r["stall_s"]), float(r["recompute_tokens"]) / RHO
        bx.bar(k, stall, color=COLORS[key], width=0.62, edgecolor=SURFACE, linewidth=2)
        bx.bar(k, rec, bottom=stall, color=COLORS[key], alpha=0.45, hatch="//", width=0.62, edgecolor=SURFACE,
               linewidth=2)
        bx.text(k, stall + rec + 25, f"{stall + rec:,.0f} s", ha="center", fontsize=10, color=INK)
    bx.set_xticks(range(len(picks)), [p[2] for p in picks], fontsize=9.5)
    bx.set_ylabel("Seconds (2,576 turns)")
    bx.set_title("(b) Where each rule's cost goes", loc="left", fontsize=12)
    bx.legend(handles=[Patch(color=INK2, label="Stall: read lands after the turn is due"),
                       Patch(facecolor=INK2, alpha=0.45, hatch="//", label="Recomputation of revoked reads, at 9K tokens/s")],
              frameon=False, fontsize=9.5, loc="upper right")
    fig.suptitle("44 users, 70 GB: on the p50 run's reads, the expected-cost start time costs 28 % less than the run's "
                 "p50 rule and 14 % less than reading at arrival", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.005, f"CPU model calibrated on both campaign 23 runs (room {ROOM // 1000}K tokens); the run's evictions "
             "and queue waits held fixed. Recomputation is charged at its own prefill time only: it also lengthens other "
             "turns' queue wait, which the model leaves out.", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"policy_tradeoff.{ext}"), dpi=150, facecolor=SURFACE)
    print("wrote", os.path.join(OUT, "policy_tradeoff.png"))


ROOM = 90000   # calibrated: one room fits both runs' revokes, re-reads and pins (calibration.csv)
LAMS = (1e-6, 3e-6, 1e-5, 2e-5, 5e-5, 1e-4, 3e-4, 1e-3)
FORMS = ("split", "hist", "samples")
POLICY_KEYS = ("staged_reads", "staged_revoked", "arrival_reads", "arrival_revoked", "reclaimed", "expired",
               "mean_pinned", "pinned_before_arrival", "stall_s", "late_turns", "recompute_tokens", "cost_s",
               "wasted_pages")


def label(p: dict) -> str:
    rule = p["rule"]
    name = {"asrun": "JIT p50 as run", "arrival": "at arrival", "oracle": "exact arrival",
            "asrun_noexp": "JIT p50, no expiry"}.get(rule)
    if rule == "quantile":
        name = f"land at q={p['q']:g} ({p['form']})"
    if rule == "cost":
        name = f"expected cost lam={p['lam']:g} ({p['form']})"
    return name + (", soft pins" if p.get("soft") else "")


def run_policy(data: dict, p: dict, room) -> dict:
    if p["rule"] == "asrun_noexp":   # the run's timing, but no forecast ends a session
        p = dict(p, rule="asrun", expiry=False)
    return simulate(data, p, room=room)


def policies() -> None:
    """Read-start rules on the p50 run's reads, at the calibrated room and either side of it."""
    data = build_jobs("p50")
    base = [dict(rule="asrun"), dict(rule="asrun_noexp"), dict(rule="arrival"), dict(rule="oracle")]
    quant = [dict(rule="quantile", q=q, form="split") for q in (0.1, 0.25, 0.5, 0.75, 0.9)]
    cost = [dict(rule="cost", lam=lam, form=form) for form in FORMS for lam in LAMS]
    rows = []
    for room in (ROOM, 70000, 110000):
        plist = base + quant + cost
        out = []
        for p in plist:
            sim = run_policy(data, p, room)
            out.append((p, sim))
        # soft pins on the run's routine, the best fixed quantile, and the best expected-cost rule of each form
        best = lambda sel: min((x for x in out if sel(x[0])), key=lambda x: x[1]["cost_s"])[0]
        softs = [dict(rule="asrun", soft=True), dict(best(lambda p: p["rule"] == "quantile"), soft=True)]
        softs += [dict(best(lambda p, f=f: p["rule"] == "cost" and p["form"] == f), soft=True) for f in FORMS]
        for p in softs:
            out.append((p, run_policy(data, p, room)))
        print(f"\n== room {room} tokens, {len(data['jobs'])} reading turns of the p50 run")
        print(f"{'policy':44s} " + " ".join(f"{k[:9]:>9s}" for k in POLICY_KEYS))
        for p, sim in out:
            print(f"{label(p):44s} " + " ".join(f"{sim[k]:>9}" for k in POLICY_KEYS))
            rows.append(dict(room=room, policy=label(p), **{k: p.get(k) for k in ("rule", "q", "lam", "form", "soft")},
                             **sim))
    write_csv("policies.csv", rows)


CAL_KEYS = ("staged_reads", "staged_revoked", "arrival_reads", "arrival_revoked", "mean_pinned",
            "pinned_before_arrival", "lead_p50", "lead_p90", "landed_after_arrival")


def calibrate() -> None:
    """Each run's own routine through the model, over a sweep of the room, beside the run's measured numbers."""
    rows = []
    for name in RUNS:
        data, real = build_jobs(name), measure(name)
        print(f"\n== {name}: {len(data['jobs'])} turns read history from SSD; arrival offset {data['delta']:.3f} s")
        print(f"{'room':>8s} " + " ".join(f"{k[:14]:>14s}" for k in CAL_KEYS))
        print(f"{'measured':>8s} " + " ".join(f"{str(real[k]):>14s}" for k in CAL_KEYS))
        for room in range(30000, 160001, 10000):
            sim = simulate(data, dict(rule="asrun"), room=room)
            print(f"{room:8d} " + " ".join(f"{str(sim[k]):>14s}" for k in CAL_KEYS))
            rows.append(dict(run=name, room=room, **sim))
        rows.append(dict(run=name, room="measured", **{k: real[k] for k in CAL_KEYS}))
    write_csv("calibration.csv", rows)


def write_csv(fname: str, rows: list) -> None:
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(OUT, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
