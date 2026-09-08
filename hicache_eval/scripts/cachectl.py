"""CLI around the cache-state controls in hcommon."""
import json
import sys

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "flush":
        hcommon.flush_cache(float(sys.argv[2]) if len(sys.argv) > 2 else 0.0)
        print(json.dumps({"flushed": True, "l3": hcommon.l3_stats()}))
    elif cmd == "droppc":
        m = hcommon.drop_page_cache()
        print(json.dumps({"method": m, "cached_kb": hcommon.cached_meminfo_kb()}))
    elif cmd == "drain":
        v = hcommon.wait_backup_drain()
        print(json.dumps({"backup_tokens_total": v}))
    elif cmd == "l3":
        print(json.dumps(hcommon.l3_stats()))
    elif cmd == "scrape":
        hcommon.scrape(sys.argv[2])
        print(json.dumps({"written": sys.argv[2]}))
    elif cmd == "delta":
        b = open(sys.argv[2]).read()
        a = open(sys.argv[3]).read()
        d = hcommon.metrics_delta(b, a)
        keep = {k: v for k, v in d.items() if k.startswith("sglang:") and v != 0}
        out = sys.argv[4] if len(sys.argv) > 4 else None
        js = json.dumps(keep, indent=2, sort_keys=True)
        if out:
            open(out, "w").write(js)
        print(js)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
