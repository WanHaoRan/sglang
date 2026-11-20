"""Campaign 28 or 29: what the GPU computed in each arm, to see why two arms that queue alike serve at different rates.
On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/gpu_work.py results/campaign28 [from_s to_s] > results/campaign28/checks/gpu_work.out
Over the requests sent in [from_s, to_s) of the client clock (default 1,800 to 6,600 s, the saturated part):
prompt tokens computed, split into new tokens (no cache could hold them) and history computed again; the GPU seconds
that history costs at the engine's measured prefill rate (RECOMPUTE_S_PER_TOKEN, 0.11 ms a token); output tokens;
seconds per output token after the first; and from the server's "Prefill batch" lines the tokens it prefilled.
Stdlib only."""
import datetime as dt
import json
import re
import sys

cdir = sys.argv[1]
lo, hi = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (1800.0, 6600.0)
S_PER_TOKEN = 1.1e-4
PREFILL = re.compile(r"^\[(.{23})\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")


def cached(r):
    d = r.get("cached_details") or {}
    return sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)


with open(f"{cdir}/manifest.txt") as f:
    arms = [line.split()[:3] for line in f if line.strip()]
print(f"{cdir.split('/')[-1]}: requests sent in [{lo:.0f}, {hi:.0f}) s; GPU seconds for history at {1e3 * S_PER_TOKEN:.2f} ms a token")
print(f"{'arm':10s} | {'requests':>8s} {'a minute':>8s} | {'prompt M':>8s} {'new M':>7s} {'again M':>7s} {'again %':>7s} | {'GPU s again':>11s} {'of window':>9s} | "
      f"{'output K':>8s} {'tok/s':>6s} | {'ms per output token: mean':>25s} {'p50':>6s} | {'server prefilled M':>18s}")
for label, run, programs in arms:
    with open(f"results/{run}/{programs}/client.jsonl") as f:
        t0 = json.loads(f.readline())["t_start"]
        reqs = [json.loads(line) for line in f if '"kind":"req"' in line]
    rs = [r for r in reqs if r["phase"] == "profile" and lo <= r["t_send"] < hi and r.get("ttft") is not None and not r.get("error")]
    prompt = sum(r["prompt_tokens"] for r in rs)
    ideal = sum(r["ideal_cached"] for r in rs)
    again = sum(max(0, r["ideal_cached"] - cached(r)) for r in rs)
    out = sum(r["completion_tokens"] for r in rs)
    per = sorted(1e3 * (r["latency"] - r["ttft"]) / (r["completion_tokens"] - 1) for r in rs if r["completion_tokens"] > 20)
    prefilled = 0
    with open(f"results/{run}/server.log", errors="replace") as f:
        for line in f:
            if "Prefill batch" in line and (m := PREFILL.match(line)):
                t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() - t0
                if lo <= t < hi:
                    prefilled += int(m[3])
    print(f"{label:10s} | {len(rs):8d} {60 * len(rs) / (hi - lo):8.1f} | {prompt / 1e6:8.1f} {(prompt - ideal) / 1e6:7.2f} {again / 1e6:7.2f} {100 * again / max(1, ideal):6.2f}% | "
          f"{again * S_PER_TOKEN:11.0f} {100 * again * S_PER_TOKEN / (hi - lo):8.1f}% | {out / 1e3:8.0f} {out / (hi - lo):6.0f} | "
          f"{sum(per) / len(per):25.1f} {per[len(per) // 2]:6.1f} | {prefilled / 1e6:18.2f}")
