"""Helpers for the tg_* scripts (checks added 2026-11-12, after jit_p50_d and ec_r13_rep finished).
Run every script as:  nice -n 19 python3 -I -B <script>   (standard library only)."""
import ast, heapq, math, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BUCKETS = ((8, 15), (16, 23), (24, 31))


def med(v):
    v = sorted(v)
    n = len(v)
    return float("nan") if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def p2(k, n):
    """Two-sided exact binomial p-value for k of n at 0.5."""
    if not n:
        return float("nan")
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def run_dir(label):
    return os.path.join(C.RES, C.RUNS[label] if label in C.RUNS else C.EXTRA[label])


def server_args(label):
    s = open(os.path.join(run_dir(label), "server_args.txt")).read()
    return ast.literal_eval(s[s.index("server_args=") + len("server_args="):].strip())


def gpu_units():
    """label -> GPU unit of each LAUNCH line in the campaign's launcher.log (read only); a relaunched label keeps a list."""
    out = {}
    pat = re.compile(r"^C26 (\S+) LAUNCH (\S+) .* gpu (\S+) extra")
    with open(os.path.join(C.RES, "campaign26", "launcher.log")) as f:
        for line in f:
            m = pat.match(line)
            if m:
                out.setdefault(m[1], []).append((m[2], m[3]))
    return out


def read_pages(d):
    """(conv, turn) -> SSD pages read for it (s2h_io lines)."""
    out = {}
    for t, c, n, pages, tok, ms in d["s2h"]:
        out[(c, n)] = out.get((c, n), 0) + pages
    return out


def load_state(d, censor_fix=True):
    """(conv, turn) -> (sessions in progress, live context, requests in flight, in-flight prompt tokens) at the turn's send.
    A session is in progress from its first send to its last send + that turn's latency (the task's definition). With
    censor_fix a session the client cut at the run's end (conv_end.capped, or no conv_end record) stays in progress."""
    turns = sorted(d["turns"], key=lambda t: t["t_send"])
    last = {}
    for t in turns:
        last[t["conv"]] = t
    s_end = {c: t["t_send"] + (t["latency"] or 0.0) for c, t in last.items()}
    if censor_fix:
        for c in s_end:
            ce = d["conv_end"].get(str(c))
            if ce is None or ce.get("capped"):
                s_end[c] = float("inf")
    live, live_sum = {}, 0
    sess_heap, req_heap = [], []
    n_flight = flight_sum = 0
    out = {}
    for t in turns:
        now = t["t_send"]
        while sess_heap and sess_heap[0][0] <= now:
            _, c = heapq.heappop(sess_heap)
            live_sum -= live.pop(c)
        while req_heap and req_heap[0][0] <= now:
            _, p = heapq.heappop(req_heap)
            n_flight -= 1
            flight_sum -= p
        c = t["conv"]
        size = (t["pt"] or 0) + (t["ct"] or 0)
        if c not in live and s_end[c] > now:
            heapq.heappush(sess_heap, (s_end[c], c))
            live[c] = 0
        if c in live:
            live_sum += size - live[c]
            live[c] = size
        n_flight += 1
        flight_sum += t["pt"] or 0
        heapq.heappush(req_heap, (now + (t["latency"] or 0.0), t["pt"] or 0))
        out[(c, t["turn"])] = (len(live), live_sum, n_flight, flight_sum)
    return out


def pos_bins(keys, rank, npos, width, send_time):
    """Bins of `width` positions of the order run; keys absent from it go into bins of `width` turns by send_time(key).
    Returns [(label, lo or None, [keys])]."""
    present = [k for k in keys if k in rank]
    absent = sorted((k for k in keys if k not in rank), key=send_time)
    cell = {}
    for k in present:
        cell.setdefault(rank[k] // width * width, []).append(k)
    out = [(f"{lo}", lo, cell[lo]) for lo in sorted(cell)]
    out += [(f"abs{i // width:02d}", None, absent[i:i + width]) for i in range(0, len(absent), width)]
    return out


def block_boot(d, bins, draws=4000, seed=0):
    """Block bootstrap of the mean of d[key] over bins (lists of keys): (5 %, 95 %, share of draws with the sign opposite
    to the mean)."""
    cells = [(sum(d[k] for k in b), len(b)) for b in bins if b]
    tot = sum(c[0] for c in cells)
    rng = random.Random(seed)
    v = []
    for _ in range(draws):
        pick = rng.choices(cells, k=len(cells))
        v.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    opp = sum(1 for x in v if (x < 0) == (tot > 0) and x != 0) / len(v)
    return C.pct(v, .05), C.pct(v, .95), opp


def session_boot(d, draws=2000, seed=0):
    sess = {}
    for k, x in d.items():
        s = sess.setdefault(k[0], [0.0, 0])
        s[0] += x
        s[1] += 1
    cells = list(sess.values())
    rng = random.Random(seed)
    v = []
    for _ in range(draws):
        pick = rng.choices(cells, k=len(cells))
        v.append(sum(c[0] for c in pick) / sum(c[1] for c in pick))
    return C.pct(v, .05), C.pct(v, .95)


def decode_steps(d, t_lo, t_hi, lo, hi):
    """ms per decode step (1000 * running / gen throughput) on Decode lines with no queue, t in [t_lo, t_hi) s after the
    client's start, running requests in [lo, hi]; returns [(seconds after start, ms per step, #token)]."""
    t0 = d["t0"]
    return [(t - t0, 1000.0 * run / thr, token) for t, run, token, usage, thr, q in d["dec"]
            if t_lo <= t - t0 < t_hi and lo <= run <= hi and q == 0 and thr > 0]
