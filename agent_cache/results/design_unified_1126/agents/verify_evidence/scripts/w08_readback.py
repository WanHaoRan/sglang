"""ev_write 1a/1c: what the SSD writes bought, on my page model with the logged reads: share of written tokens
that are prompt pages read back at least once, never read back, outputs, final requests; reads per page."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay, chains
for arm in sys.argv[1:] or ["ec_full"]:
    D = ssd_replay.Data(34, arm); ssd_replay.first_sets(D); ssd_replay.uses_index(D)
    ch = chains.load_chains(34, arm)
    res = ssd_replay.replay(D, policy="today")
    deliv = res["deliv"]
    reqs = D.reqs
    tot = 0; cls = {"prompt, read back": 0, "prompt, never read back, held by a later request": 0, "prompt, never read back, no later request holds it": 0,
                    "prompt of a stream's final request": 0, "output": 0, "written again (second submit)": 0}
    seen = set(); nread = 0; nrb = 0
    by_win = {}
    for rid, q in sorted(reqs.items(), key=lambda x: x[1]["t_send"]):
        if q["ttft"] is None: continue
        ks = D.keys[rid]
        w = int((q["t_send"] + q["ttft"]) // 1800)
        bw = by_win.setdefault(w, [0, 0])
        for j in range(q["cached"] // 64, len(ks)):
            i = ks[j]; tot += 1; bw[1] += 1
            if i in seen:
                cls["written again (second submit)"] += 1; continue
            seen.add(i)
            if deliv.get(i, 0) > 0:
                cls["prompt, read back"] += 1; nrb += 1; nread += deliv[i]; bw[0] += 1
            elif ch[rid]["last"]:
                cls["prompt of a stream's final request"] += 1
            elif D.next_use(i, q["t_send"]) < float("inf"):
                cls["prompt, never read back, held by a later request"] += 1
            else:
                cls["prompt, never read back, no later request holds it"] += 1
        n_out = len(D.out_pages[rid]); tot += n_out; cls["output"] += n_out; bw[1] += n_out
    print(f"c34 {arm}: written {tot*64/1e6:.2f} M tokens (model); pages read back at least once {nrb*64/1e6:.2f} M, each read {nread/max(nrb,1):.2f} times")
    for k, v in cls.items():
        print(f"   {k:52s} {v*64/1e6:6.2f} M  {100*v/tot:5.1f} %")
    live = cls["prompt, read back"] + cls["prompt, never read back, held by a later request"] + cls["prompt, never read back, no later request holds it"]
    print(f"   of live sessions' prompt pages, read back: {100*cls['prompt, read back']/live:.1f} %; by half hour of the write (read back / written): " +
          ", ".join(f"{w*30}-{w*30+30}: {100*a/max(b,1):.0f} %" for w, (a, b) in sorted(by_win.items())))
    need = nrb
    print(f"   an oracle writing only what is read back writes {need*64/1e6:.2f} M and saves {100*(1-need/tot):.1f} % of submitted tokens")
