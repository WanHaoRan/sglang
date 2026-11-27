"""History lost per request (ideal_cached - cached_tokens) by size class, from the client records; tier shares
of first requests; requests by role."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xlib

for spec in sys.argv[1:]:
    camp, arm = spec.split(":")
    run, prog = xlib.runs(camp)[arm]
    c = xlib.load_client(prog)
    reqs = [r for r in c["req"] if r.get("t_send", 1e9) < 6600 and r.get("ttft") is not None]
    print("==", spec, "requests before 6,600 s:", len(reqs))
    bins = [(0, 1), (1, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 10**9)]
    print("lost history per request (ideal_cached - cached_tokens): bin, requests, tokens")
    tot = 0
    for lo, hi in bins:
        sel = [r for r in reqs if lo <= max(0, r["ideal_cached"] - r["cached_tokens"]) < hi]
        tok = sum(max(0, r["ideal_cached"] - r["cached_tokens"]) for r in sel)
        tot += tok
        print(f"   [{lo:5d},{hi if hi < 10**9 else 'inf':>5}) {len(sel):6d} {tok/1e3:9.1f} K")
    print(f"   total {tot/1e3:.1f} K tokens; prompt tokens {sum(r['prompt_tokens'] for r in reqs)/1e6:.1f} M; new text {sum(r['prompt_tokens']-r['ideal_cached'] for r in reqs)/1e6:.2f} M")
    # the [64,256) class by window
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        sel = [r for r in reqs if lo * 60 <= r["t_send"] < hi * 60 and 64 <= r["ideal_cached"] - r["cached_tokens"] < 256]
        allr = [r for r in reqs if lo * 60 <= r["t_send"] < hi * 60]
        print(f"   minutes {lo}-{hi}: {len(sel)} of {len(allr)} requests lose 64-255 tokens, {sum(r['ideal_cached']-r['cached_tokens'] for r in sel)/1e3:.1f} K tokens")
    # first requests of a play: where their shared prefix came from
    first = [r for r in reqs if r.get("k") == 0 and r.get("stream") == 0]
    dev = sum((r.get("cached_details") or {}).get("device", 0) for r in first)
    host = sum((r.get("cached_details") or {}).get("host", 0) for r in first)
    sto = sum((r.get("cached_details") or {}).get("storage", 0) for r in first)
    ideal = sum(r["ideal_cached"] for r in first)
    prm = sum(r["prompt_tokens"] for r in first)
    print(f"first requests of a session: {len(first)}; prompt {prm/1e6:.2f} M, ideal cached {ideal/1e6:.2f} M ({100*ideal/prm:.1f}%), got device {dev/1e6:.2f} M host {host/1e6:.2f} M storage {sto/1e6:.2f} M, lost {(ideal-dev-host-sto)/1e6:.2f} M")
    for lo, hi in [(0, 28), (28, 70), (70, 90), (90, 110)]:
        f = [r for r in first if lo * 60 <= r["t_send"] < hi * 60]
        if not f:
            continue
        idl = sum(r["ideal_cached"] for r in f)
        d_ = sum((r.get("cached_details") or {}).get("device", 0) for r in f)
        h_ = sum((r.get("cached_details") or {}).get("host", 0) for r in f)
        s_ = sum((r.get("cached_details") or {}).get("storage", 0) for r in f)
        ns = sum(1 for r in f if (r.get("cached_details") or {}).get("storage", 0) > 0)
        nshare = sum(1 for r in f if r["ideal_cached"] >= 256)
        print(f"   minutes {lo}-{hi}: {len(f)} first requests, {nshare} share >= 256 tokens with earlier sessions; ideal {idl/1e3:.0f} K: device {d_/1e3:.0f} K, host {h_/1e3:.0f} K, storage {s_/1e3:.0f} K ({ns} requests read), lost {(idl-d_-h_-s_)/1e3:.0f} K"
              f"; mean TTFT {xlib.mean(r['ttft'] for r in f):.2f} s; readers' TTFT {xlib.mean(r['ttft'] for r in f if (r.get('cached_details') or {}).get('storage',0)>0):.2f} s")
    roles = {}
    for r in reqs:
        roles.setdefault((r.get("role"), r.get("stream", 0) > 0), []).append(r)
    print("requests by (role, stream > 0):", {str(k): len(v) for k, v in roles.items()})
    gaps = {}
    for r in reqs:
        gaps.setdefault(r.get("gap"), []).append(r)
    print("requests by gap kind:", {str(k): len(v) for k, v in gaps.items()})
