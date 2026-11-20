"""One line per finished request of a replay_programs.py run: sizes, cached tokens by tier, timings.
From agent_cache/:  python3 -I -B results/campaign27/checks/reqs.py <client.jsonl> [max rows]"""
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
reqs = [r for r in rows if r["kind"] == "req"]
print(f"{len(reqs)} requests; phases {sorted({r['phase'] for r in reqs})}; errors {sum(1 for r in reqs if r.get('error'))}")
print("phase   role     rid               wait_s  t_send  prompt  ideal  device    host storage recomp   out   ttft latency")
for r in sorted(reqs, key=lambda r: r.get("t_send") or 0)[:limit]:
    d = r.get("cached_details") or {}
    cached = sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)
    p = r.get("prompt_tokens") or 0
    print(f"{r['phase']:7s} {str(r['role']):8s} {r['rid']:16s} {r['wait_s']:7.1f} {r.get('t_send') or 0:7.1f} {p:7d} {r['ideal_cached']:6d} "
          f"{int(d.get('device') or 0):6d} {int(d.get('host') or 0):7d} {int(d.get('storage') or 0):7d} {p - cached:6d} {r.get('completion_tokens') or 0:5d} "
          f"{r.get('ttft') or -1:6.1f} {r.get('latency') or -1:7.1f}" + (f"  ERROR {r['error'][:60]}" if r.get("error") else "")
          + ("" if p == r["prompt_sent"] else f"  SENT {r['prompt_sent']}"))
