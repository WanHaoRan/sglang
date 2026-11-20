"""Campaign 28 or 29: the returning requests of an arm that got under 2 % of their history from the cache, with
the server's lines for the first few. On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/missed_requests.py results/<server run> <programs dir> [how many to trace]
Stdlib only."""
import json
import sys

run, programs = sys.argv[1], sys.argv[2]
show = int(sys.argv[3]) if len(sys.argv) > 3 else 3
with open(f"{run}/{programs}/client.jsonl") as f:
    reqs = [json.loads(line) for line in f if '"kind":"req"' in line]


def cached(r):
    d = r.get("cached_details") or {}
    return sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)


miss = [r for r in reqs if 3600 <= r["t_send"] < 6600 and r["k"] > 0 and r["ideal_cached"] > 0 and r.get("ttft") is not None
        and cached(r) / r["ideal_cached"] < 0.02]
print(f"{len(miss)} returning requests sent in [3600, 6600) s got under 2 % of their history")
print("rid, k, stream, t_send, gap before it (recorded wait s), ttft, ideal_cached, prompt_tokens")
for r in miss:
    print(f"  {r['rid']:28s} k={r['k']:3d} stream={r['stream']} t={r['t_send']:7.0f} wait={r.get('wait_s')} ttft={r['ttft']:6.1f} "
          f"ideal={r['ideal_cached']} prompt={r['prompt_tokens']}")
want = {r["rid"] for r in miss[:show]}
lines = {rid: [] for rid in want}
with open(f"{run}/server.log", errors="replace") as f:
    for line in f:
        for rid in want:
            if rid in line:
                lines[rid].append(line.rstrip()[:330])
for rid, ls in lines.items():
    print(f"\n----- {rid}: {len(ls)} server lines")
    for line in ls[:40]:
        print("  " + line)
