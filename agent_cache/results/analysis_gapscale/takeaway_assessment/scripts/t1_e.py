import pickle, numpy as np, bisect
S = "/tmp/claude-1001/-home-wanhr-sglang/454133c8-c6f1-4d1f-8d97-1e8cda557a0d/scratchpad/assess/"
D = pickle.load(open(S + "t1.pkl", "rb")); R = pickle.load(open(S + "t1_rows.pkl", "rb"))
POOL = 1627648
def q(a, p): return float(np.percentile(a, p)) if len(a) else float('nan')
def summ(a): a = np.asarray(a, float); return "mean %.3f p10 %.3f p50 %.3f p90 %.3f p99 %.3f max %.3f" % (a.mean(), q(a,10), q(a,50), q(a,90), q(a,99), a.max()) if len(a) else "n 0"
for arm in ("three_tier_to", "three_tier_wc"):
    rows = R[arm]; sh = [x for x in rows if x["tier"] == "storage"]
    print("=====", arm, "n", len(sh))
    gap = np.array([x["gap"] for x in sh]); tt = np.array([x["ttft"] for x in sh])
    Fw = np.array([x["io_end"] - x["ps"] for x in sh]); Fio = np.array([x["io_ms"] / 1000 for x in sh])
    blocked = np.array([min(x["queue"], x["io_end"] - x["send"]) for x in sh])
    lead = np.array([x["ps"] - x["send"] for x in sh])
    print(" gap min %.1f; gap<Fw: %d; end_prev present %d" % (gap.min(), (gap < Fw).sum(), sum(1 for x in sh if x["end_prev"])))
    print(" baseline TTFT", summ(tt))
    for Fname, F in (("F=observed window ps->io_end", Fw), ("F=own IO time only", Fio)):
        print(" --", Fname, "total fetch time %.0f s" % F.sum())
        # oracle: start at arrival - F, but not before end_prev
        hid = np.minimum(F, gap); exp_ = F - hid
        tt_new = tt - blocked + np.minimum(blocked, exp_)
        print("   (2) oracle hidden share %.1f%% ; TTFT' %s ; TTFT' mean reduction %.2f s (%.0f%%)" % (100 * hid.sum() / F.sum(), summ(tt_new), tt.mean() - tt_new.mean(), 100 * (1 - tt_new.mean() / tt.mean())))
        for m in (0.0, 2.0, 10.0):
            for e in (-0.5, -0.25, 0.25, 0.5):
                ghat = gap * (1 + e)
                start_rel = np.maximum(0, ghat - F - m)          # relative to end_prev
                done_rel = start_rel + F
                late = done_rel - gap                            # >0: completes after arrival
                exposed = np.where(start_rel >= gap, F, np.clip(late, 0, F))   # starts after arrival -> reactive baseline
                hidden = F - exposed
                resid = np.clip(gap - done_rel, 0, None)
                tt_n = tt - blocked + np.minimum(blocked, exposed)
                print("   (3) m=%4.1fs e=%+4.0f%%: hidden %5.1f%% of fetch time; fully hidden turns %5.1f%%; TTFT' mean %.2f p50 %.2f p90 %.2f; residency before arrival p50 %.1f s p90 %.1f s max %.1f s" % (
                    m, 100 * e, 100 * hidden.sum() / F.sum(), 100 * (exposed <= 1e-9).mean(), tt_n.mean(), q(tt_n, 50), q(tt_n, 90), q(resid, 50), q(resid, 90), resid.max()))
