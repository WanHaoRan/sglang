"""For the returning turns that computed 4K+ tokens of history again: what the server did for them before admission."""
import collections, glob, json, os, re, statistics as st, sys

RES = "results"
RID = r"(?:rid|req)=\S+?-c(\d+)-t(\d+)"
PATS = {"query": re.compile(r"HICACHE_EVT s2h_query " + RID + r" tokens_req=(\d+) storage_hit=(\d+)"),
        "dispatch": re.compile(r"HICACHE_EVT srs_dispatch " + RID + r" pages=(\d+) .*launched=(\d)"),
        "io": re.compile(r"HICACHE_EVT s2h_io " + RID + r" pages=(\d+)"),
        "success": re.compile(r"HiCache prefetch success " + RID + r" completed=(\d+) matched=(\d+) loaded=(\d+)"),
        "shrank": re.compile(r"device prefix shrank before admission " + RID + r" lookup_match=(\d+) current_match=(\d+)"),
        "start": re.compile(r"HICACHE_EVT prefetch_start " + RID + r" tokens=(\d+)"),
        "unpin": re.compile(r"HICACHE_EVT srs_unpin " + RID + r" tokens=(\d+) reason=(\w+)")}
for run in sys.argv[1:]:
    path = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))[-1]
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rec = lambda r: max(r.get("reply_reprefilled") or 0, 0)
    big = {(r["conv"], r["turn"]): r for r in rows if r["kind"] == "turn" and r.get("ttft") is not None and r["turn"] > 0 and not r.get("mid_start") and rec(r) >= 4096}
    ev = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(os.path.join(RES, run, "server.log"), errors="replace"):
        if "-c" not in line:
            continue
        for name, pat in PATS.items():
            if m := pat.search(line):
                k = (int(m[1]), int(m[2]))
                if k in big:
                    ev[k][name].append(m.groups()[2:])
                break
    cats = collections.Counter()
    tok = collections.Counter()
    detail = []
    for k, r in big.items():
        e = ev[k]
        hit = max((int(h) for _, h in e["query"]), default=0)
        refused = sum(1 for _, launched in e["dispatch"] if launched == "0")
        read_tokens = 64 * sum(int(p[0]) for p in e["io"])
        loaded = sum(int(l) for _, _, l in e["success"])
        if not e["query"] and not e["start"]:
            cat = "no SSD lookup at all"
        elif hit == 0:
            cat = "looked up, nothing on SSD"
        elif e["shrank"] or any(reason != "admit" for _, reason in e["unpin"]):
            cat = "read, then lost before admission"
        elif refused and read_tokens < 0.5 * rec(r):
            cat = "read refused room"
        elif read_tokens < 0.5 * hit:
            cat = "found on SSD, read little"
        else:
            cat = "read most of what SSD had; SSD had less than the history"
        cats[cat] += 1
        tok[cat] += rec(r)
        detail.append((cat, rec(r), hit, read_tokens, refused, r["ttft"]))
    n, total = len(big), sum(tok.values())
    print(f"\n{run}: {n} returning turns recomputed 4K+ tokens ({total / 1e6:.2f}M tokens)")
    for cat, c in cats.most_common():
        d = [x for x in detail if x[0] == cat]
        print(f"  {cat:58s} {c:4d} turns, {tok[cat] / total:4.0%} of those tokens; recomputed median {st.median(x[1] for x in d):6.0f}, SSD hit median {st.median(x[2] for x in d):6.0f}, "
              f"read median {st.median(x[3] for x in d):6.0f}, refusals mean {st.mean(x[4] for x in d):.1f}, TTFT mean {st.mean(x[5] for x in d):5.1f} s")
