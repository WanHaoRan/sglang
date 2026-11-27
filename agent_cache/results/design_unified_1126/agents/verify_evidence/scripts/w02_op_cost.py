"""ev_write 3a: cost of a write op alone against with a read in progress; fix-1 skipped pages; device shares."""
import sys, os, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib
from vlib import kv, pct, mean

def intervals(ev, name):
    out = []
    for t, r in ev.get(name, []):
        d = kv(r)
        ms = float(d["ms"])
        out.append((t - ms / 1e3, t, d))
    out.sort(key=lambda x: x[0])
    return out

def overlap_len(a, b, ivs, starts):
    # total length of [a,b] covered by the (non-overlapping, FIFO thread) intervals
    i = bisect.bisect_left(starts, a) - 1
    i = max(i, 0)
    tot = 0.0
    while i < len(ivs) and ivs[i][0] < b:
        lo = max(a, ivs[i][0]); hi = min(b, ivs[i][1])
        if hi > lo:
            tot += hi - lo
        i += 1
    return tot

for camp, arm in [(34, "ec_full"), (34, "jit_exact"), (34, "jit_p50"), (34, "ec_exact"), (32, "ec_full")]:
    ev = vlib.load(camp, arm)
    reads = intervals(ev, "s2h_io")
    rstarts = [x[0] for x in reads]
    w = intervals(ev, "h2s_io")
    grp = intervals(ev, "h2s_group")
    sk = sum(int(d.get("skipped", 0)) for _, _, d in w if "group" not in d) + sum(int(d["skipped"]) for _, _, d in grp)
    print(f"== c{camp} {arm}: fix-1 skipped pages {sk} = {sk*64/1e6:.2f} M tokens; write ops {len(w)} of which merged {sum(1 for x in w if 'group' in x[2])}; groups {len(grp)}")
    buckets = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 128)]
    print("   unmerged write ops, ms per op:   pages | alone: n median mean | read covers >=50% of the op: n median mean | any overlap: n median")
    for lo, hi in buckets:
        alone = []; withr = []; anyr = []
        for a, b, d in w:
            if "group" in d:
                continue
            pg = int(d["pages"])
            if not (lo <= pg <= hi):
                continue
            ms = float(d["ms"])
            ov = overlap_len(a, b, reads, rstarts)
            if ov <= 0:
                alone.append(ms)
            else:
                anyr.append(ms)
                if ov >= 0.5 * (b - a):
                    withr.append(ms)
        print(f"   {lo:>3}-{hi:<3} | {len(alone):5d} {pct(alone,.5):7.1f} {mean(alone):7.1f} | {len(withr):5d} {pct(withr,.5):7.1f} {mean(withr):7.1f} | {len(anyr):5d} {pct(anyr,.5):7.1f}")
    # ms per page over all write I/O (unmerged ops + groups), alone vs with a read
    def per_page(items):
        al_ms = al_pg = wr_ms = wr_pg = 0.0
        for a, b, d in items:
            pg = int(d["pages"]) - int(d.get("skipped", 0))
            if pg <= 0:
                continue
            ov = overlap_len(a, b, reads, rstarts)
            if ov <= 0:
                al_ms += float(d["ms"]); al_pg += pg
            elif ov >= 0.5 * (b - a):
                wr_ms += float(d["ms"]); wr_pg += pg
        return al_ms / max(al_pg, 1), wr_ms / max(wr_pg, 1), al_pg, wr_pg
    items = [x for x in w if "group" not in x[2]] + grp
    a1, a2, n1, n2 = per_page(items)
    print(f"   all writes (unmerged ops + groups): {a1:.2f} ms/page alone ({n1:.0f} pages), {a2:.2f} ms/page with a read over >=50% ({n2:.0f} pages)")
    # reads: ms per page alone vs with a write overlapping >= half
    wall = sorted([(a, b) for a, b, d in items], key=lambda x: x[0])
    wst = [x[0] for x in wall]
    al_ms = al_pg = wr_ms = wr_pg = 0.0
    for a, b, d in reads:
        pg = int(d["pages"])
        ov = overlap_len(a, b, [(x[0], x[1], None) for x in wall], wst)
        if ov <= 0:
            al_ms += float(d["ms"]); al_pg += pg
        elif ov >= 0.5 * (b - a):
            wr_ms += float(d["ms"]); wr_pg += pg
    print(f"   reads: {al_ms/max(al_pg,1):.2f} ms/page alone ({al_pg:.0f} pages), {wr_ms/max(wr_pg,1):.2f} ms/page with writes over >=50% ({wr_pg:.0f} pages)")
    # device share at saturation: tokens moved per second, minutes 90-110
    for lo, hi in [(3600, 4200), (4200, 4800), (4800, 5400), (5400, 6600)]:
        rt = sum(int(d["tokens"]) for a, b, d in reads if lo <= b < hi)
        rbusy = sum(min(b, hi) - max(a, lo) for a, b, d in reads if b > lo and a < hi)
        wt = sum(int(d["tokens"]) for a, b, d in w if lo <= b < hi)
        wbusy = sum(min(b, hi) - max(a, lo) for a, b in wall if b > lo and a < hi)
        print(f"   minutes {lo//60}-{hi//60}: reads {rt/(hi-lo)/1e3:.1f} K tok/s, read thread busy {100*rbusy/(hi-lo):.1f} %; writes {wt/(hi-lo)/1e3:.1f} K tok/s, write thread busy {100*wbusy/(hi-lo):.1f} %")
