"""Does my SSD replay reproduce the run?  Lookups, store size at the passes, losses after the first pass."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib, ssd_replay
from vlib import kv

def losses(D, base, res, lo, hi, thr=1):
    """per request: pages delivered with nothing deleted minus pages delivered; requests sent in [lo, hi)"""
    n = 0; tok = 0; rows = []
    for rid, need in base["got"].items():
        q = D.reqs[rid]
        if not (lo <= q["t_send"] < hi): continue
        g = res["got"].get(rid, {})
        own = D.keys[rid]
        miss = sum(1 for j, i in need.items() if j < len(own) and own[j] == i and g.get(j) != i)
        if miss >= thr:
            n += 1; tok += miss * 64; rows.append((rid, miss * 64))
    return n, tok, rows

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":"); camp = int(camp)
        t = time.time()
        D = ssd_replay.Data(camp, arm)
        ssd_replay.first_sets(D)
        ev = vlib.load(camp, arm)
        base = ssd_replay.replay(D, delete=False)
        res = ssd_replay.replay(D, policy="today")
        L = D.lookups
        n = len(L)
        ok = sum(1 for i in range(n) if res["hits"][i] == L[i][4])
        ok0 = sum(1 for i in range(n) if base["hits"][i] == L[i][4])
        p1 = D.passes[0][0]
        pre = [i for i in range(n) if L[i][0] < p1]; post = [i for i in range(n) if L[i][0] >= p1]
        print(f"=== c{camp} {arm} ({time.time()-t:.0f} s): lookups placed {n} of {len(ev['s2h_query'])} logged")
        print(f"   logged hit == replayed hit: with the logged passes {ok} of {n}; with no deletion {ok0}; before the first pass {sum(1 for i in pre if res['hits'][i]==L[i][4])} of {len(pre)}; after {sum(1 for i in post if res['hits'][i]==L[i][4])} of {len(post)}")
        bad = [(round(L[i][0]), L[i][1], L[i][2], L[i][3], L[i][4], res["hits"][i]) for i in range(n) if res["hits"][i] != L[i][4]]
        print("   first mismatches (t, rid, start page, pages asked, logged, replayed):", bad[:6], "; staged lookups", D.n_staged_lookups, "; replayed failed reads", len(res["failed"]), sorted(res["failed"]) == sorted(D.failed))
        # after the first pass: nothing / part / all / only new text
        nothing = part = full = 0
        for i in post:
            if L[i][0] >= 6600: continue
            b = base["hits"][i]; h = L[i][4]
            if b == 0: continue
            if h == 0: nothing += 1
            elif h < b: part += 1
            else: full += 1
        print(f"   lookups from the first pass to 6,600 s that a no-deletion store answers: logged found nothing {nothing}, part {part}, all {full}")
        sub = sum(int(kv(r)['tokens']) for t_, r in ev['h2s_submit'])
        wp = 0
        for rid, q in D.reqs.items():
            wp += (len(D.keys[rid]) - q["cached"] // 64) + len(D.out_pages[rid])
        print(f"   pages written by my model {wp} = {wp*64/1e6:.2f} M tokens; h2s_submit {sub/1e6:.2f} M")
        for (t0, pages, nd), (a, b, n_) in zip(res["sizes"], D.passes):
            print(f"   pass at {t0:.0f}-{b:.0f} s: store {pages} pages = {pages*64/1e6:.2f} M tokens before; deletes {nd}  -> implied volume at 100 % if 90.3 % full: {pages/0.903:.0f} pages")
        nl, tl, rows = losses(D, base, res, p1, 6600)
        act = sum(max(q["ideal"] - q["cached"], 0) for q in D.reqs.values() if p1 <= q["t_send"] < 6600 and q["ideal"] - q["cached"] >= 256)
        print(f"   first pass to 6,600 s: replay loses {tl/1e6:.3f} M tokens over {nl} requests (>= 1 page), {sum(1 for r in rows if r[1] >= 1024)} with >= 1,024; the run's records: {act/1e6:.3f} M lost (all causes, >= 256)")
        nl2, tl2, _ = losses(D, base, res, 5400, 6600)
        print(f"   minutes 90-110: replay loses {tl2/1e6:.3f} M over {nl2} requests")
        fr = [r for r in rows if D.reqs[r[0]]["k"] == 0]
        print(f"   of them first requests of a session: {len(fr)} requests, {sum(r[1] for r in fr)/1e3:.0f} K; logged failed reads {len(D.failed)}")
