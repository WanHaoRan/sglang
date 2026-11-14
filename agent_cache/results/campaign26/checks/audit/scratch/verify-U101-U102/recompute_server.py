"""Verifier's own server-side spot check for the ten arms landed after the 12 Nov recomputation (read-only).
Own parser. Per run: SSD pages and reading turns from 's2h_io' lines, queue wait from 'ReqTimeStats' lines,
recomputed tokens from the client's reply_reprefilled; kept-turn rule as stated in compare_ec.py's docstring."""
import glob, json, os, re, statistics as st, time

RES = "/home/wanhr/sglang/agent_cache/results"
C26 = os.path.join(RES, "campaign26")
IO = re.compile(r"HICACHE_EVT s2h_io rid=\S+?-c(\d+)-t(\d+) pages=(\d+) ")
QD = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), .*?queue_duration=([\d.]+)ms")
WANT = ["ec_r13", "nofc_rep", "jit_exact_d", "ec_r134", "nofc_noend_rep", "ec_full", "fq90", "ec_r3", "nofc_noend_c128", "ec_r134_c128", "nofc_c128"]
man = {f[0]: (f[1], f[2]) for f in (l.split() for l in open(os.path.join(C26, "manifest.txt"))) if f}
mean = lambda v: st.mean(v) if v else float("nan")
res = {}
t00 = time.time()
for arm in WANT:
    run, cl = man[arm]
    turns = []
    for line in open(os.path.join(RES, run, cl, "client.jsonl"), errors="replace"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("kind") == "turn" and r.get("ttft") is not None:
                turns.append(r)
    last = max(r["t_send"] for r in turns)
    kept = {(r["conv"], r["turn"]): r for r in turns if r["t_send"] < last - 300}
    pages, nreads, queue = {}, {}, {}
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            if "s2h_io" in line:
                m = IO.search(line)
                if m:
                    k = (int(m[1]), int(m[2]))
                    pages[k] = pages.get(k, 0) + int(m[3]); nreads[k] = nreads.get(k, 0) + 1
            elif "ReqTimeStats(" in line:
                m = QD.search(line)
                if m:
                    queue[(int(m[1]), int(m[2]))] = float(m[3]) / 1000
    rec = lambda r: max(r.get("reply_reprefilled") or 0, 0)
    if arm.endswith("_c128"):
        T = {k: r for k, r in kept.items() if r["t_send"] < 6900}
        ks = [k for k in T if k[1] > 0]
        reading = [k for k in ks if pages.get(k, 0) > 0]
        res[arm] = dict(turns=len(T), queue=f"{mean([queue.get(k, 0.0) for k in T]):.2f}", rec=f"{mean([rec(T[k]) for k in ks]):.0f}",
                        share=f"{100 * len(reading) / len(ks):.1f}", pages=sum(pages.get(k, 0) for k in ks),
                        rpr=f"{mean([nreads[k] for k in reading]):.2f}", multi=sum(nreads[k] > 1 for k in reading))
    else:
        res[arm] = dict(turns=len(kept), queue=f"{mean([queue.get(k, 0.0) for k in kept]):.2f}", rec=f"{mean([rec(r) for k, r in kept.items() if k[1] > 0]):.0f}",
                        pages=sum(pages.get(k, 0) for k in kept), reading=sum(1 for k in kept if pages.get(k, 0) > 0))
    print(arm, res[arm], f"[{time.time() - t00:.0f} s]", flush=True)
json.dump(res, open("recompute_server.json", "w"), indent=1)

# set against the printed tables
ok = bad = 0
def cmp(what, a, b):
    global ok, bad
    if str(a) == str(b): ok += 1
    else:
        bad += 1; print("  MISMATCH", what, "mine", a, "printed", b)
txt = open(os.path.join(C26, "compare_ec.out")).read().split("\n")
i = next(n for n, l in enumerate(txt) if l.startswith("per arm, turns sent at least"))
for l in txt[i + 1:]:
    f = l.split()
    if not f: break
    if f[0] in res and not f[0].endswith("_c128"):
        m = res[f[0]]
        cmp(f"{f[0]} queue", m["queue"], f[9]); cmp(f"{f[0]} recomputed", m["rec"], f[10]); cmp(f"{f[0]} pages", m["pages"], f[11]); cmp(f"{f[0]} reading turns", m["reading"], f[12])
txt = open(os.path.join(C26, "compare_c128.out")).read().split("\n")
i = next(n for n, l in enumerate(txt) if l.startswith("per arm, turns sent in its first"))
for l in txt[i + 1:]:
    if not l.strip(): break
    left, right = l.split("|")[0].split(), l.split("|")[1].split()
    if left[0] in res:
        m = res[left[0]]
        cmp(f"{left[0]} queue", m["queue"], left[5]); cmp(f"{left[0]} recomputed", m["rec"], left[7]); cmp(f"{left[0]} share reading", m["share"], right[0])
        cmp(f"{left[0]} pages", m["pages"], right[2]); cmp(f"{left[0]} reads per reading turn", m["rpr"], right[3]); cmp(f"{left[0]} multi-read turns", m["multi"], right[4])
print(f"server-side spot check: matched {ok} of {ok + bad} printed numbers")
