"""How much of a run's prefill is history computed again, when, and what refused the read."""
import collections, glob, json, os, re, statistics as st, sys

RES = "results"
for run in sys.argv[1:]:
    path = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    turns = [r for r in rows if r["kind"] == "turn" and r.get("ttft") is not None]
    ret = [r for r in turns if r["turn"] > 0 and not r.get("mid_start")]
    rec = lambda r: max(r.get("reply_reprefilled") or 0, 0)
    unc = lambda r: int(r.get("prompt_tokens") or 0) - int(r.get("cached_tokens") or 0)
    total_unc = sum(unc(r) for r in turns)
    total_rec = sum(rec(r) for r in ret)
    first_unc = sum(unc(r) for r in turns if r["turn"] == 0 or r.get("mid_start"))
    big = [r for r in ret if rec(r) >= 4096]
    print(f"\n{run}: {len(turns)} turns; prefill tokens {total_unc / 1e6:.2f}M = first turns {first_unc / 1e6:.2f}M + new text of returning turns {(total_unc - first_unc - total_rec) / 1e6:.2f}M "
          f"+ history computed again {total_rec / 1e6:.2f}M ({total_rec / total_unc:.0%})")
    print(f"  returning turns recomputing 4K+ tokens: {len(big)} ({len(big) / len(ret):.1%}); they hold {sum(rec(r) for r in big) / max(1, total_rec):.0%} of the recompute; "
          f"their mean TTFT {st.mean(r['ttft'] for r in big):.2f} s against {st.mean(r['ttft'] for r in ret if rec(r) < 4096):.2f} s for the rest; "
          f"their share of all returning TTFT seconds {sum(r['ttft'] for r in big) / sum(r['ttft'] for r in ret):.0%}")
    per_min = collections.Counter()
    for r in big:
        per_min[int(r["t_send"] // 60)] += 1
    top = sorted(per_min, key=per_min.get, reverse=True)[:14]
    print(f"  the 14 minutes with most such turns hold {sum(per_min[m] for m in top) / max(1, len(big)):.0%} of them: " + " ".join(f"{m}:{per_min[m]}" for m in sorted(top)))
    log = open(os.path.join(RES, run, "server.log"), errors="replace").read()
    refused = len(re.findall(r"HICACHE_EVT srs_dispatch \S+ pages=\d+ .*launched=0", log))
    print(f"  server log: read launches refused {refused}; 'reissue cap reached' {log.count('reissue cap reached')}; prefetch dropped {len(re.findall(r'HiCache prefetch .*dropped', log))}; "
          f"evict_then_recompute markers: shrank {log.count('device prefix shrank before admission')}")
