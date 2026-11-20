"""Campaign 28 or 29: is the GPU short of memory or of computation? How long a decode step takes against the number
of running requests and the context tokens they hold. On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/decode_scale.py <server run> <programs dir> [<server run> <programs dir> ...] > results/campaign28/checks/decode_scale.out
Uses the server's "Decode batch" lines (one per 40 decode steps, with the output rate since the last one) and keeps
the intervals with no "Prefill batch" line inside, so the rate is decode alone; the context in decode comes from the
client's records. Prints the step time by running requests, a least-squares fit (fixed + per request + per context
token), what the fit gives for other pool sizes, and the share of a full pool's time left to prefill. Stdlib only."""
import bisect, json, re, datetime as dt, statistics as st, collections, sys
LINE = re.compile(r"^\[(.{23})\] (Prefill|Decode) batch")
DEC = re.compile(r"#running-req: (\d+), #token: (\d+), token usage: ([\d.]+),.*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PRE = re.compile(r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
clean, all_iv = [], []
for run, programs in ((a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])):
    with open(f"results/{run}/{programs}/client.jsonl") as f:
        t0 = json.loads(f.readline())["t_start"]
        reqs = [r for r in (json.loads(l) for l in f) if r["kind"] == "req" and r.get("ttft") is not None]
    # Context tokens held by requests in decode, every 5 s of the client clock.
    grid = list(range(0, 7300, 5))
    ctx = []
    for t in grid:
        ctx.append(sum(r["prompt_tokens"] + r["completion_tokens"] / 2 for r in reqs if r["t_send"] + r["ttft"] <= t < r["t_send"] + r["latency"]))
    prev_t, pre_tokens, pre_n = None, 0, 0
    with open(f"results/{run}/server.log", errors="replace") as f:
        for line in f:
            if " batch" not in line or not (m := LINE.match(line)):
                continue
            t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() - t0
            if m[2] == "Prefill":
                p = PRE.search(line)
                pre_tokens += int(p[2]); pre_n += 1
                continue
            d = DEC.search(line)
            if d and prev_t is not None and 30 <= t < 6600 and t > prev_t:
                B, thr = int(d[1]), float(d[4])
                T = ctx[min(len(ctx) - 1, bisect.bisect_left(grid, t))]
                row = (B, T, thr, t - prev_t, pre_tokens, float(d[3]))
                all_iv.append(row)
                if pre_n == 0 and thr > 0:
                    clean.append(row)
            prev_t, pre_tokens, pre_n = t, 0, 0
print(f"decode log intervals: {len(all_iv)}; with no prefill batch inside: {len(clean)}")
print("no prefill inside: running requests | intervals | context in decode K (median) | output tokens/s | ms per step")
groups = collections.defaultdict(list)
for B, T, thr, dtv, _, use in clean:
    groups[B if B <= 6 else (7 if B <= 9 else 10 if B <= 13 else 14 if B <= 17 else 18 if B <= 20 else 21 if B <= 23 else 24 if B <= 26 else 27)].append((B, T, thr))
names = {7: "7-9", 10: "10-13", 14: "14-17", 18: "18-20", 21: "21-23", 24: "24-26", 27: "27+"}
for g in sorted(groups):
    v = groups[g]
    if len(v) >= 3:
        print(f"  {names.get(g, str(g)):>6s} | {len(v):4d} | {st.median(x[1] for x in v) / 1e3:6.0f} | {st.median(x[2] for x in v):6.0f} | {st.median(1e3 * x[0] / x[2] for x in v):6.1f}")
# Least squares on the clean intervals: ms per step = c0 + cB * running + cT * context K
import itertools
X = [(1.0, B, T / 1e3) for B, T, thr, *_ in clean]
y = [1e3 * B / thr for B, T, thr, *_ in clean]
def solve(X, y):
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    for i in range(k):
        p = A[i][i]
        A[i] = [a / p for a in A[i]]; b[i] /= p
        for j in range(k):
            if j != i:
                f = A[j][i]
                A[j] = [a - f * c for a, c in zip(A[j], A[i])]; b[j] -= f * b[i]
    return b
c = solve(X, y)
my = sum(y) / len(y)
r2 = 1 - sum((v - sum(a * b for a, b in zip(r, c))) ** 2 for r, v in zip(X, y)) / sum((v - my) ** 2 for v in y)
print(f"fit on them: ms per step = {c[0]:.1f} + {c[1]:.2f} x running + {c[2]:.4f} x context K; R2 {r2:.2f}")
for B, T in ((1, 5), (11, 320), (22, 635), (44, 1270), (44, 635)):
    s = c[0] + c[1] * B + c[2] * T
    print(f"  {B:2d} running, {T:5d} K context: {s:5.1f} ms per step, {1e3 * B / s:5.0f} output tokens/s with no prefill")
# All intervals at a full pool: share of time in prefill, from the fit's decode time
full = [x for x in all_iv if x[5] >= 0.95]
dec_t = sum(40 * (c[0] + c[1] * B + c[2] * T / 1e3) / 1e3 for B, T, thr, dtv, pt, use in full)
wall = sum(x[3] for x in full)
print(f"full pool: {len(full)} intervals, {wall:.0f} s; 40 decode steps each by the fit: {dec_t:.0f} s ({100 * dec_t / wall:.0f} %); "
      f"prefilled {sum(x[4] for x in full) / 1e6:.1f} M tokens in the other {wall - dec_t:.0f} s: {1e3 * (wall - dec_t) / max(1, sum(x[4] for x in full)):.3f} ms a token")
