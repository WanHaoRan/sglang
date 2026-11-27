"""By 10-minute window of one run: load, the two queues, how reads landed against their GPU turn, and the lateness
weight c (share of stall seconds with no other ready turn waiting).

  python regimes.py <label>
"""
import bisect
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I drops the script folder
import common as C  # noqa: E402
import runlib as R  # noqa: E402

LAG_S = 0.4   # landing to admission of a turn that waited on its read (measured median 0.40-0.43 s)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    label = sys.argv[1]
    d = R.load(label)
    t0 = d["run_rec"]["t_start"]
    reads = json.load(open(os.path.join(C.EXTRACTS, f"{label}.reads.json")))["rows"]
    reqs = json.load(open(os.path.join(C.EXTRACTS, f"{label}.requests.json")))["rows"]
    ready = json.load(open(os.path.join(C.EXTRACTS, f"{label}.ready.json")))
    rt, rc = ready["t"], ready["ready"]
    lines = sorted([(p[0] - t0, p[4], p[6], p[5]) for p in d["prefill"]] + [(x[0] - t0, x[3], x[4], x[1]) for x in d["decode"]])
    census = [(t - t0, f) for t, _, f in d["evts"].get("host_census", [])]
    print(f"== {label}: by 10-minute window (time since the client's start) ==")
    print("min | req adm | readers % | late % of readers | stall s mean (late) | pin s mean (early) | pool<0.90 % of batch lines "
          "with a queue | queue med | running med | no-ready-turn % of time | c = idle share of stall s | parked med | inflight pages med | "
          "park->disp s med | disp->land s med | read ms/page")
    for lo in range(0, 120, 10):
        a, b = lo * 60.0, (lo + 10) * 60.0
        rq = [r for r in reqs if a <= r["adm"] < b and not str(r["rid"]).startswith("HEALTH")]
        rd = [r for r in rq if r["reader"]]
        late = [r for r in rd if r["adm"] - r["D"] > 0.05]
        stall = [r["adm"] - r["D"] for r in late]
        early = [r["D"] - r["last_land"] for r in rd if r["D"] >= r["last_land"]]
        ll = [x for x in lines if a <= x[0] < b]
        withq = [x for x in ll if x[2] > 0]
        # time share with no ready turn
        i = bisect.bisect_right(rt, a) - 1
        t, idle = a, 0.0
        while t < b and i + 1 < len(rt):
            end = min(b, rt[i + 1])
            if i >= 0 and rc[i] <= 0:
                idle += end - t
            elif i < 0:
                idle += end - t
            t = end
            i += 1
        rds = [r for r in reads if a <= r["park"] < b and not r["staged"]]
        num = sum((r["adm"] - r["D"]) * r["idle_share_stall"] for r in rds if "idle_share_stall" in r)
        den = sum((r["adm"] - r["D"]) for r in rds if "idle_share_stall" in r)
        cs = [f for t_, f in census if a <= t_ < b]
        print(f"{lo:3d}-{lo + 10:<3d} | {len(rq):5d} | {100 * len(rd) / max(1, len(rq)):5.1f} | {100 * len(late) / max(1, len(rd)):5.1f} | "
              f"{mean(stall):6.2f} | {mean(early):5.2f} | {100 * sum(1 for x in withq if x[1] < 0.90) / max(1, len(withq)):5.1f} (n {len(withq)}) | "
              f"{C.q([x[2] for x in ll], .5):3.0f} | {C.q([x[3] for x in ll], .5):3.0f} | {100 * idle / (b - a):5.1f} | "
              f"{(num / den if den > 0 else float('nan')):5.2f} | {C.q([f['n_parked'] for f in cs], .5):3.0f} | "
              f"{C.q([f['staging'] // 64 for f in cs], .5):5.0f} | {C.q([r['disp'] - r['park'] for r in rds], .5):6.2f} | "
              f"{C.q([r['land'] - r['disp'] for r in rds], .5):5.2f} | {1000 * C.q([r['s_per_page_park'] for r in rds], .5):4.2f}")


if __name__ == "__main__":
    main()
