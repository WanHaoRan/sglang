import re, sys
from datetime import datetime
ts_re = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] (.*)$")
def ts(s): return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timestamp()
def pct(a, q): a = sorted(a); return a[min(len(a)-1, int(q*len(a)))] if a else float('nan')
for run in sys.argv[1:]:
    sub = {}; io = {}; d2h_tok = 0; h2s_sub_tok = 0; h2s_ok_tok = 0; d2h_done_ms = []
    for line in open(f"/home/wanhr/sglang/agent_cache/results/{run}/server.log", errors="ignore"):
        m = ts_re.match(line)
        if not m or "HICACHE_EVT" not in line: continue
        b = m.group(2)
        if "h2s_submit" in b:
            op = int(re.search(r"op=(\d+)", b).group(1)); tok = int(re.search(r"tokens=(\d+)", b).group(1))
            sub[op] = (ts(m.group(1)), tok); h2s_sub_tok += tok
        elif "h2s_io" in b:
            op = int(re.search(r"op=(\d+)", b).group(1)); tok = int(re.search(r"tokens=(\d+)", b).group(1))
            ms = float(re.search(r"ms=(\d+)", b).group(1)); io[op] = (ts(m.group(1)), ms, tok); h2s_ok_tok += tok
        elif "d2h_submit" in b:
            d2h_tok += int(re.search(r"tokens=(\d+)", b).group(1))
        elif "d2h_done" in b:
            d2h_done_ms.append(float(re.search(r"ms=([\d.]+)", b).group(1)))
    lag = [io[o][0] - sub[o][0] for o in sub if o in io]
    wait = [io[o][0] - io[o][1]/1000 - sub[o][0] for o in sub if o in io]
    print(f"{run}: d2h_tokens={d2h_tok} h2s_submit_tokens={h2s_sub_tok} h2s_written_tokens={h2s_ok_tok} ops={len(sub)} io_logged={len(io)}")
    print(f"   h2s submit->io-complete lag s: p50={pct(lag,.5):.3f} p90={pct(lag,.9):.3f} p99={pct(lag,.99):.3f} max={max(lag):.1f}; queue-wait part p50={pct(wait,.5):.3f} p99={pct(wait,.99):.3f} max={max(wait):.1f}")
    print(f"   d2h_done gpu ms: p50={pct(d2h_done_ms,.5)} p99={pct(d2h_done_ms,.99)}")
