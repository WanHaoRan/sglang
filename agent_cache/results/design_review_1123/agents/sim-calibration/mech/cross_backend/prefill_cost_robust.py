"""Robustness of the prefill cost fit (prefill_cost.py): the same decode-log intervals, four model forms; prints the
cost of a whole 20 K and 30 K history computed again under each. From agent_cache/:
  python3 -I -B prefill_cost_robust.py <manifest> <suffix> [label ...]"""
import datetime as dt, json, re, sys
manifest, suffix = sys.argv[1], sys.argv[2]
want = set(sys.argv[3:])
rows = [l.split() for l in open(manifest) if l.strip() and l.split()[0].endswith(suffix)]
LINE = re.compile(r"^\[(.{23})\] (Prefill|Decode) batch")
DEC = re.compile(r"#running-req: (\d+), #token: (\d+), token usage: ([\d.]+),.*#queue-req: (\d+)")
PRE = re.compile(r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
iv = []
for lab, run, prog in rows:
    if want and lab not in want:
        continue
    with open(f"results/{run}/{prog}/client.jsonl") as f:
        t0 = json.loads(f.readline())["t_start"]
    prev, pre = None, []
    for line in open(f"results/{run}/server.log", errors="replace"):
        if " batch" not in line or not (m := LINE.match(line)):
            continue
        t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() - t0
        if m[2] == "Prefill":
            p = PRE.search(line); pre.append((int(p[2]), int(p[3]))); continue
        d = DEC.search(line)
        if not d:
            continue
        if prev is not None and 60 <= t < 6600 and t > prev:
            iv.append((int(d[1]), int(d[2]) / 1e3, sum(a for a, _ in pre), len(pre), sum(a * (c + a / 2) for a, c in pre) / 1e6, (t - prev) * 1e3))
        prev, pre = t, []
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
y = [r[5] for r in iv]
forms = {
    "steps(1,B,T) + tok + batch + attn": lambda r: (40, 40 * r[0], 40 * r[1], r[2], r[3], r[4]),
    "steps(1,B) + tok + batch + attn": lambda r: (40, 40 * r[0], r[2], r[3], r[4]),
    "steps(1,T) + tok + batch + attn": lambda r: (40, 40 * r[1], r[2], r[3], r[4]),
    "steps(1,B,T) + tok + batch": lambda r: (40, 40 * r[0], 40 * r[1], r[2], r[3]),
}
print(f"{suffix}: {len(iv)} intervals")
for name, f in forms.items():
    X = [f(r) for r in iv]
    c = solve(X, y)
    my = sum(y) / len(y)
    r2 = 1 - sum((v - sum(a * b for a, b in zip(x, c))) ** 2 for x, v in zip(X, y)) / sum((v - my) ** 2 for v in y)
    has_attn = "attn" in name
    tok, bat = (c[-3], c[-2]) if has_attn else (c[-2], c[-1])
    att = c[-1] if has_attn else 0.0
    out = []
    for n in (20000, 30000):
        ms = sum(tok * a + bat + att * a * (s + a / 2) / 1e6 for a, s in [(min(8192, n - s0), s0) for s0 in range(0, n, 8192)])
        out.append(f"{n // 1000} K: {ms / 1e3:.3f} s ({ms / n:.4f} ms/token)")
    print(f"  {name:36s} R2 {r2:.3f} | per token {tok:.5f} ms, per batch {bat:.1f} ms, attn {att:.5f} | " + "; ".join(out))
