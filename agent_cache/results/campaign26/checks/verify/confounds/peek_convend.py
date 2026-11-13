"""How sessions end in the client log: conv_end.end_s against last send + latency; capped sessions (cut by the run's end).
  nice -n 19 python3 -I -B peek_convend.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

for a in C.LEAD5 + ["jit_p50_d_cut", "hintfree"]:
    d = C.load(a)
    last = {}
    first = {}
    for t in sorted(d["turns"], key=lambda t: t["t_send"]):
        last[t["conv"]] = t
        first.setdefault(t["conv"], t)
    ce = d["conv_end"]
    capped = [c for c in ce if ce[c].get("capped")]
    normal = [c for c in ce if not ce[c].get("capped")]
    dif = [ce[c]["end_s"] - (last[int(c)]["t_send"] + last[int(c)]["latency"]) for c in normal if int(c) in last]
    difc = [ce[c]["end_s"] - (last[int(c)]["t_send"] + last[int(c)]["latency"]) for c in capped if int(c) in last]
    st = [d["conv_start"][str(c)]["start_s"] - first[c]["t_send"] for c in first]
    print(f"{a:14s} conv_end {len(ce)} capped {len(capped)} aborted {sum(1 for c in ce if ce[c].get('aborted'))} | normal: end_s - (last send + latency): mean {C.mean(dif):.3f} max {max(dif) if dif else 0:.3f} min {min(dif) if dif else 0:.3f} | "
          f"capped: n {len(difc)} mean {C.mean(difc):.1f} max {max(difc) if difc else 0:.1f} end_s {sorted(set(round(ce[c]['end_s']) for c in capped))[:6]} | start_s - first send: mean {C.mean(st):.3f} max abs {max(abs(x) for x in st):.3f} | "
          f"last t_send {max(t['t_send'] for t in d['turns']):.1f} max latency {max(t['latency'] for t in d['turns'] if t['latency'] is not None):.1f} max ttft {max(t['ttft'] for t in d['turns'] if t['ttft'] is not None):.1f}")
    # turns of capped sessions: how long before the end was their last send
    lag = sorted(max(t['t_send'] for t in d['turns']) - last[int(c)]["t_send"] for c in capped if int(c) in last)
    print("     capped sessions: seconds between their last send and the run's last send:", [round(x) for x in lag])
