"""Parts 4 and 5: where the prefill tokens go, and what empty lookups cost.

Every request sent before 6,600 s. Prefill tokens = input - cached (server). Split:
  new text            input - ideal_cached (no cache policy changes it); first requests and returning apart
  lost history        ideal_cached - cached, by cause, from the request's own log lines:
    rounding            under 256 tokens (pages are 64 tokens, the lookup threshold is 256)
    tail                256 to 1,023 tokens that a lookup starting at the cached length did not find, or that the
                        gate called covered: text the SSD does not hold (reply tails, prompts that drop text)
    cleaner_read        a read of the request failed on a deleted file (lookup hit, file gone)
    cleaner_miss        a lookup starting at (or before) the cached length found nothing beyond it, after the
                        first cleaner pass (inferred from the timing; the logs do not name deleted pages)
    ssd_miss            the same before any cleaner pass (never written, or not yet)
    reevicted           a lookup found pages beyond the cached length and the read landed, yet they were not
                        there at admission
    refused             ... and the read was refused host room (launched=0)
    cap                 no lookup covered the lost span and the turn was capped: gate 'capped', ntp_reevict
                        capped=1, or (hicache) 'reissue cap reached'      -> defect 1 / F5
    covered_evicted     no lookup covered it, the gate had said 'covered' (history resident at arrival, gone at
                        admission, no lookup made)
    no_lookup           no lookup covered it, none of the above
  first requests' shared prefix (ideal_cached of k = 0) is listed apart.
GPU seconds: tokens x (c_pre + c_x x prefix K) + extra batches x c_b, the run's loaded fit (prefill_fit.json).
TTFT at stake: GPU seconds x mu, with mu = the busy-period count of part 1 (1 at light load, N - B once the
ready queue never empties) in the minute of admission.
usage: part45_recompute.py camp:arm [...]"""
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402
import part1_prices  # noqa: E402

FIT = json.load(open(f"{lib.OUT}/prefill_fit.json"))
CAUSES = ["rounding", "tail", "cleaner_read", "cleaner_miss", "ssd_miss", "reevicted", "refused", "cap",
          "covered_evicted", "no_lookup"]
BINS = [(0, 30), (30, 60), (60, 90), (90, 110)]


def classify(r, fail_rids, t_clean):
    lost = r["ideal"] - r["cached_srv"]
    if lost < 256:
        return "rounding"
    C = r["cached_srv"]
    base = (r["in_len"] // 64) * 64
    cov = [(t1, base - req, hit) for (t0, t1, req, hit) in r["lookups"] if base - req <= C + 64]
    capped = any(g[1] == "capped" for g in r["gates"]) or any(x[3] == 1 for x in r["reev"]) or r["capreach"] is not None
    if r["rid"] in fail_rids and lost >= 1024:
        return "cleaner_read"
    beyond = [(t1, s, h) for (t1, s, h) in cov if s + h > C + 64]
    if beyond:
        if any(rd.get("launched") == 0 for rd in r["reads"]):
            return "refused"
        return "reevicted"
    if cov:
        if lost < 1024:
            return "tail"
        t_last = max(t1 for t1, _, _ in cov)
        return "cleaner_miss" if t_clean is not None and t_last > t_clean else "ssd_miss"
    if lost < 1024 and any(g[1] == "covered" for g in r["gates"]):
        return "tail"
    if capped:
        return "cap"
    if any(g[1] == "covered" for g in r["gates"]):
        return "covered_evicted"
    return "no_lookup"


def gpu_s(fit, tokens, prefix, new_other):
    if tokens <= 0:
        return 0.0
    batches = math.ceil((tokens + new_other) / 8192) - math.ceil(max(new_other, 1) / 8192)
    return (tokens * (fit["ms_per_tok_prefix0"] + fit["ms_per_tok_per_Kprefix"] * (prefix + tokens / 2) / 1e3)
            + batches * fit["ms_per_batch"]) / 1e3


def main():
    os.makedirs(lib.OUT, exist_ok=True)
    for spec in sys.argv[1:]:
        camp, arm = spec.split(":")
        fit = FIT[spec]["loaded"]
        out, st, rows = part1_prices.per_minute(camp, arm)
        srv = st["srv"]
        fail_rids = {rid for _, rid in srv["readfail"]}
        t_clean = min((c[0] for c in srv["cleanup"]), default=None)
        mu_min = [o["mu_bp"] if o["mu_bp"] == o["mu_bp"] else 1.0 for o in out]
        rs = [r for r in rows if r["t_send"] < 6600 and r["in_len"] is not None]
        acc = {}
        per_req = []
        for r in rs:
            m = min(119, int(r["adm"] // 60))
            mu = mu_min[m]
            b = next((i for i, (lo, hi) in enumerate(BINS) if lo * 60 <= r["t_send"] < hi * 60), None)
            if b is None:
                continue
            new = max(0, r["in_len"] - max(r["ideal"], r["cached_srv"]))
            lost = max(0, r["ideal"] - r["cached_srv"])
            first = r["k"] == 0
            g_new = gpu_s(fit, new, max(r["ideal"], r["cached_srv"]), 0)
            g_lost = gpu_s(fit, lost, r["cached_srv"], new)
            key_new = "new_first" if first else ("new_rewrite" if new >= 8192 else "new_returning")
            cause = ("first_shared" if lost >= 256 else "rounding") if first else classify(r, fail_rids, t_clean)
            for key, tok, g in ((key_new, new, g_new), (cause, lost, g_lost)):
                a = acc.setdefault((key, b), [0, 0, 0.0, 0.0])
                a[0] += 1 if tok > 0 else 0
                a[1] += tok
                a[2] += g
                a[3] += g * mu
            a = acc.setdefault(("_all", b), [0, 0, 0.0, 0.0])
            a[0] += 1
            a[1] += r["in_len"] - r["cached_srv"]
            a[2] += 0.0
            a[3] += r["ttft"] or 0.0
            if not first and lost >= 256:
                per_req.append((r["rid"], round(r["t_send"], 1), cause, lost, r["ideal"], r["cached_srv"], round(g_lost, 3), round(mu, 1)))
        with open(f"{lib.OUT}/lost_requests_{camp}_{arm}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rid", "t_send", "cause", "lost_tokens", "ideal", "cached", "gpu_s", "mu"])
            w.writerows(per_req)
        n_req = [acc.get(("_all", b), [0, 0, 0, 0])[0] for b in range(len(BINS))]
        ttft_sum = [acc.get(("_all", b), [0, 0, 0, 0])[3] for b in range(len(BINS))]
        print(f"== {spec}: requests sent before 6,600 s: {sum(n_req)}; loaded cost {fit['ms_per_tok_prefix0']:.4f} ms/tok + "
              f"{fit['ms_per_tok_per_Kprefix']*1e3:.3f} us per K of prefix + {fit['ms_per_batch']:.1f} ms a batch; first cleaner pass "
              f"{'none' if t_clean is None else f'{t_clean:.0f} s'}")
        print("   requests per window " + " / ".join(f"{lo}-{hi}: {n}" for (lo, hi), n in zip(BINS, n_req)) +
              "; mean TTFT " + " / ".join(f"{t/n:.2f}" if n else "-" for t, n in zip(ttft_sum, n_req)) +
              "; mean mu " + " / ".join(f"{lib.mean(mu_min[lo:hi]):.0f}" for lo, hi in BINS))
        keys = ["new_first", "new_returning", "new_rewrite", "first_shared"] + CAUSES
        print(f"   {'class':16s} | " + " | ".join(f"{lo:3d}-{hi:3d}: req  tokens K   GPU s  TTFT-s at mu" for lo, hi in BINS) + " | all: tokens K   GPU s  TTFT-s  s/request")
        tot_lost = [0, 0.0, 0.0]
        for key in keys:
            cells = []
            s_tok = s_g = s_t = 0.0
            for b in range(len(BINS)):
                a = acc.get((key, b), [0, 0, 0.0, 0.0])
                cells.append(f"{a[0]:9d} {a[1]/1e3:9.1f} {a[2]:7.1f} {a[3]:13.0f}")
                s_tok += a[1]
                s_g += a[2]
                s_t += a[3]
            if key in CAUSES or key == "first_shared":
                tot_lost[0] += s_tok
                tot_lost[1] += s_g
                tot_lost[2] += s_t
            print(f"   {key:16s} | " + " | ".join(cells) + f" | {s_tok/1e3:11.1f} {s_g:8.1f} {s_t:8.0f} {s_t/sum(n_req):9.3f}")
        print(f"   lost history in all: {tot_lost[0]/1e3:.1f} K tokens, {tot_lost[1]:.1f} GPU s, {tot_lost[2]:.0f} TTFT-s at mu = "
              f"{tot_lost[2]/sum(n_req):.3f} s per request (mean TTFT {sum(ttft_sum)/sum(n_req):.2f} s)")
        # Part 4: lookups and the cap
        print("   -- lookups (s2h_query) and the allowance, by window of the request's send time")
        print("      window | lookups, found nothing | turns with a gate decision: read/reread, covered, capped | turns whose counted lookups all found nothing |"
              " turns at the cap (gate capped / reevict capped=1 / reissue cap) | of them lost >= 256 tokens: by cause")
        for b, (lo, hi) in enumerate(BINS):
            w = [r for r in rs if lo * 60 <= r["t_send"] < hi * 60 and r["k"] > 0]
            lk = sum(len(r["lookups"]) for r in w)
            lk0 = sum(1 for r in w for x in r["lookups"] if x[3] == 0)
            g_read = sum(1 for r in w if any(g[1] in ("read", "reread") for g in r["gates"]))
            g_cov = sum(1 for r in w if r["gates"] and all(g[1] == "covered" for g in r["gates"]))
            g_cap = sum(1 for r in w if any(g[1] == "capped" for g in r["gates"]))
            allempty = sum(1 for r in w if r["lookups"] and all(x[3] == 0 for x in r["lookups"]))
            capped = [r for r in w if any(g[1] == "capped" for g in r["gates"]) or any(x[3] == 1 for x in r["reev"]) or r["capreach"] is not None]
            cap_empty = sum(1 for r in capped if r["lookups"] and all(x[3] == 0 for x in r["lookups"]))
            lostc = {}
            for r in capped:
                if r["ideal"] - r["cached_srv"] >= 256:
                    c = classify(r, fail_rids, t_clean)
                    v = lostc.setdefault(c, [0, 0])
                    v[0] += 1
                    v[1] += r["ideal"] - r["cached_srv"]
            print(f"      {lo:3d}-{hi:3d} | {lk:6d}, {lk0:6d} ({100*lk0/max(1,lk):4.1f} %) | {g_read:5d}, {g_cov:5d}, {g_cap:4d} | {allempty:5d} | "
                  f"{len(capped):5d} (all lookups empty {cap_empty}) | " + ", ".join(f"{c} {v[0]} ({v[1]/1e3:.1f} K)" for c, v in sorted(lostc.items())))


if __name__ == "__main__":
    main()
