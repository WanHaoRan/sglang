"""Referee: the measured side, read from the campaign records with the referee's own code (read-only, streaming).

  python -I -B ref_measured.py            -> referee/measured.json and a printout

Per arm of campaigns 34 and 32, from <run>/<programs>/client.jsonl: requests sent before 6,600 s, TTFT by window,
requests per minute, tier shares, history computed again, who waits; from <run>/server.log (line by line): running
and queued requests and pool use (Decode batch lines), the host census (campaign 34 only), reads, refusals, read and
write thread busy shares, write holds, staged reads, cleaner passes.
"""
from __future__ import annotations

import calendar
import json
import os
import re
import sys
import time

import numpy as np

RESULTS = "/home/wanhr/sglang/agent_cache/results"
HERE = os.path.dirname(os.path.abspath(__file__))
CUT = 6600.0
WINDOWS = ((0, 1800), (1800, 3600), (3600, 5400), (5400, 6600))
SRV_WINDOWS = ((1800, 3600), (3600, 5400), (5400, 6600), (3600, 6600))


def manifest(campaign: str) -> dict:
    out = {}
    with open(os.path.join(RESULTS, campaign, "manifest.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and not line.startswith("#"):
                out[p[0]] = (os.path.join(RESULTS, p[1]), p[2])
    return out


def read_client(path: str):
    """t_start and one tuple per finished request."""
    t_start = None
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('{"kind":"run"'):
                t_start = json.loads(line)["t_start"]
            elif line.startswith('{"kind":"req"'):
                d = json.loads(line)
                if d.get("ttft") is None or d.get("error"):
                    continue
                det = d.get("cached_details") or {}
                rows.append((d["play"], d["stream"], d["k"], d["t_send"], d["ttft"], d.get("latency") or 0.0,
                             d.get("prompt_tokens") or d["prompt_sent"], d.get("completion_tokens") or 0,
                             d.get("cached_tokens") or 0, det.get("device", 0), det.get("host", 0),
                             det.get("storage", 0), d.get("ideal_cached") or 0, d.get("wait_s") or 0.0))
    return t_start, rows


def client_stats(rows: list) -> dict:
    a = np.array(rows, float)
    play, stream, k, t, ttft = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    prompt, out, cached, dev, host, ssd, ideal = a[:, 6], a[:, 7], a[:, 8], a[:, 9], a[:, 10], a[:, 11], a[:, 12]
    first = (stream == 0) & (k == 0)
    m = t < CUT
    res = {
        "requests_all": int(len(a)), "requests": int(m.sum()), "req_per_min": float(m.sum() / (CUT / 60)),
        "users": int(len(set(play[m].astype(int)))),
        "ttft_mean": float(ttft[m].mean()), "ttft_p50": float(np.percentile(ttft[m], 50)),
        "ttft_p90": float(np.percentile(ttft[m], 90)), "ttft_p99": float(np.percentile(ttft[m], 99)),
        "share_gpu": float(dev[m].sum() / prompt[m].sum()), "share_host": float(host[m].sum() / prompt[m].sum()),
        "share_ssd": float(ssd[m].sum() / prompt[m].sum()),
        "history_recomputed": float(1 - cached[m].sum() / ideal[m].sum()),
        "prompt_mean": float(prompt[m].mean()), "out_mean": float(out[m].mean()),
        "new_mean": float((prompt[m] - ideal[m]).mean()),
        "latency_mean": float(a[m, 5].mean()),
        "windows": [],
    }
    for lo, hi in WINDOWS:
        w = m & (t >= lo) & (t < hi)
        ret = w & ~first
        res["windows"].append({
            "window": f"{lo // 60}-{hi // 60}", "requests": int(w.sum()),
            "req_per_min": float(w.sum() / ((hi - lo) / 60)), "ttft_mean": float(ttft[w].mean()),
            "ttft_p50": float(np.percentile(ttft[w], 50)), "ttft_p90": float(np.percentile(ttft[w], 90)),
            "share_gpu": float(dev[w].sum() / prompt[w].sum()), "share_host": float(host[w].sum() / prompt[w].sum()),
            "share_ssd": float(ssd[w].sum() / prompt[w].sum()),
            "history_recomputed": float(1 - cached[w].sum() / max(1.0, ideal[w].sum())),
            "recomputed_M": float((ideal[w] - cached[w]).clip(min=0).sum() / 1e6),
            "returning_reading": float((ssd[ret] > 0).mean()) if ret.any() else None,
            "ttft_readers": float(ttft[ret & (ssd > 0)].mean()) if (ret & (ssd > 0)).any() else None,
            "ttft_others": float(ttft[ret & (ssd == 0)].mean()) if (ret & (ssd == 0)).any() else None,
        })
    sat = (t >= 5400) & (t < 6540)
    res["req_per_min_90_109"] = float(sat.sum() / 19.0)
    return res


TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d+)\] (.*)")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*#queue-req: (\d+)")
PRE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), "
                 r"#running-req: (\d+), #queue-req: (\d+)")


def kv(text: str) -> dict:
    return dict(x.split("=", 1) for x in text.split() if "=" in x)


def server_stats(path: str, t_start: float) -> dict:
    dec = []            # (t, running, tokens, usage, queue)
    census = []         # (t, dict subset)
    reads = []          # (t_land, ms, pages, tokens)
    writes = []         # (t, ms, pages, tokens, skipped)
    disp = []           # (t, launched, pages)
    holds = []
    cleanups = []
    pre = []            # (t, new tokens, cached tokens, seqs)
    cnt = dict.fromkeys(("ntp_stage", "srs_pin", "ec_price", "reevict", "reevict_capped", "s2h_query", "s2h_miss",
                         "ntp_room", "gate_retry", "ntp_gate", "gate_capped", "prefetch_reissue_cap"), 0)
    prices = set()
    with open(path, errors="replace") as f:
        for line in f:
            m = TS.match(line)
            if not m:
                continue
            g = m.groups()
            t = calendar.timegm((int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]))) + float("0." + g[6])
            t -= t_start
            body = g[7]
            if body.startswith("Decode batch"):
                d = DEC.match(body)
                if d:
                    dec.append((t, int(d.group(1)), int(d.group(2)), float(d.group(3)), int(d.group(4))))
            elif body.startswith("Prefill batch"):
                d = PRE.match(body)
                if d:
                    pre.append((t, int(d.group(2)), int(d.group(3)), int(d.group(1))))
            elif body.startswith("HICACHE_EVT "):
                ev, _, rest = body[12:].partition(" ")
                if ev == "host_census":
                    d = kv(rest)
                    census.append((t, {k: int(d[k]) for k in ("free", "mirror", "host_only", "evict_leaf", "evict_casc",
                                                              "h_write", "h_pin", "h_staged", "h_read", "h_parked",
                                                              "h_lookup", "pending_write", "pinned", "h_other",
                                                              "h_multi", "anc", "refused", "cut") if k in d}))
                elif ev == "s2h_io":
                    d = kv(rest)
                    reads.append((t, int(d["ms"]), int(d["pages"]), int(d["pages"]) * 64))
                elif ev == "h2s_io":
                    d = kv(rest)
                    writes.append((t, int(d["ms"]), int(d["pages"]), int(d["tokens"]), int(d.get("skipped", 0))))
                elif ev == "srs_dispatch":
                    d = kv(rest)
                    disp.append((t, int(d["launched"]), int(d["pages"])))
                elif ev == "srs_write_hold":
                    if "state=on" in rest:
                        holds.append(t)
                elif ev == "ntp_stage":
                    cnt["ntp_stage"] += 1
                elif ev == "srs_pin":
                    cnt["srs_pin"] += 1
                elif ev == "ec_price":
                    cnt["ec_price"] += 1
                    prices.add(kv(rest).get("price"))
                elif ev == "ntp_reevict":
                    cnt["reevict"] += 1
                    if "capped=1" in rest:
                        cnt["reevict_capped"] += 1
                elif ev == "s2h_query":
                    cnt["s2h_query"] += 1
                    if "storage_hit=0" in rest:
                        cnt["s2h_miss"] += 1
                elif ev == "ntp_room":
                    cnt["ntp_room"] += 1
                elif ev == "ntp_gate":
                    cnt["ntp_gate"] += 1
                    if "decision=retry" in rest:
                        cnt["gate_retry"] += 1
                    if "decision=capped" in rest or "decision=cap" in rest:
                        cnt["gate_capped"] += 1
            elif "NIXL L3 cleanup: deleted" in body:
                mm = re.search(r"deleted (\d+) groups .*\(([\d.]+) GiB\) in ([\d.]+)s", body)
                cleanups.append((round(t, 1), int(mm.group(1)), float(mm.group(2)), float(mm.group(3))))
            elif "prefetch reissue cap reached" in body:
                cnt["prefetch_reissue_cap"] += 1
    out = {"counts": cnt, "prices": sorted(p for p in prices if p), "cleaner_passes": cleanups,
           "write_holds_on": len(holds), "write_holds_on_before_cut": sum(1 for t in holds if t < CUT)}
    d = np.array(dec, float)
    out["decode"] = {}
    for lo, hi in SRV_WINDOWS:
        w = d[(d[:, 0] >= lo) & (d[:, 0] < hi)]
        if len(w):
            out["decode"][f"{lo // 60}-{hi // 60}"] = {
                "running_median": float(np.median(w[:, 1])), "running_mean": float(w[:, 1].mean()),
                "queue_median": float(np.median(w[:, 4])), "usage_median": float(np.median(w[:, 3])),
                "ctx_K_median": float(np.median(w[:, 2]) / 1e3), "lines": int(len(w))}
    p = np.array(pre, float)
    out["prefill"] = {}
    for lo, hi in SRV_WINDOWS:
        w = p[(p[:, 0] >= lo) & (p[:, 0] < hi)]
        if len(w):
            out["prefill"][f"{lo // 60}-{hi // 60}"] = {
                "batches": int(len(w)), "new_tokens_M": float(w[:, 1].sum() / 1e6),
                "cached_tokens_M": float(w[:, 2].sum() / 1e6), "seqs": int(w[:, 3].sum()),
                "new_per_batch": float(w[:, 1].mean())}
    r = np.array(reads, float) if reads else np.zeros((0, 4))
    wr = np.array(writes, float) if writes else np.zeros((0, 5))
    out["reads"] = {"n": int(len(r)), "tokens_M": float(r[:, 3].sum() / 1e6), "io_s": float(r[:, 1].sum() / 1e3)}
    out["writes"] = {"n": int(len(wr)), "tokens_M": float(wr[:, 3].sum() / 1e6), "io_s": float(wr[:, 1].sum() / 1e3)}
    out["io"] = {}
    for lo, hi in SRV_WINDOWS:
        a = r[(r[:, 0] >= lo) & (r[:, 0] < hi)]
        b = wr[(wr[:, 0] >= lo) & (wr[:, 0] < hi)]
        out["io"][f"{lo // 60}-{hi // 60}"] = {
            "read_busy": float(a[:, 1].sum() / 1e3 / (hi - lo)), "write_busy": float(b[:, 1].sum() / 1e3 / (hi - lo)),
            "reads": int(len(a)), "read_tokens_M": float(a[:, 3].sum() / 1e6),
            "write_tokens_M": float(b[:, 3].sum() / 1e6),
            "read_ms_page": float(a[:, 1].sum() / max(1.0, a[:, 2].sum())),
            "write_ms_page": float(b[:, 1].sum() / max(1.0, b[:, 2].sum()))}
    dd = np.array(disp, float) if disp else np.zeros((0, 3))
    out["dispatch"] = {"n": int(len(dd)), "refused": int((dd[:, 1] == 0).sum()),
                       "refused_before_cut": int(((dd[:, 1] == 0) & (dd[:, 0] < CUT)).sum()),
                       "refused_before_deadline": int(((dd[:, 1] == 0) & (dd[:, 0] < 7200)).sum())}
    # refusals per minute, for the loop onset rule (first of 3 / 5 minutes in a row with 5 or more)
    per_min = np.zeros(125, int)
    for t, launched, _ in disp:
        if launched == 0 and 0 <= t < 7500:
            per_min[int(t // 60)] += 1

    def onset(run: int):
        for i in range(0, 125 - run):
            if (per_min[i:i + run] >= 5).all():
                return i
        return None
    out["loop_onset_min_3"] = onset(3)
    out["loop_onset_min_5"] = onset(5)
    out["refused_by_window"] = {f"{lo // 60}-{hi // 60}": int(per_min[lo // 60:hi // 60].sum()) for lo, hi in SRV_WINDOWS}
    if census:
        out["census"] = {}
        for lo, hi in ((3600, 6600), (5400, 6600)):
            w = [c for t, c in census if lo <= t < hi]
            if not w:
                continue
            keys = w[0].keys()
            out["census"][f"{lo // 60}-{hi // 60}"] = {
                "lines": len(w),
                **{k: float(np.median([c[k] for c in w])) for k in keys},
                **{k + "_p90": float(np.percentile([c[k] for c in w], 90)) for k in ("pending_write", "pinned", "h_pin",
                                                                                      "h_staged", "h_write")},
                "pin_plus_staged_median": float(np.median([c["h_pin"] + c["h_staged"] for c in w])),
                "pin_plus_staged_p90": float(np.percentile([c["h_pin"] + c["h_staged"] for c in w], 90)),
            }
    return out


def main() -> int:
    out = {}
    t0 = time.time()
    for campaign in ("campaign34", "campaign32"):
        for arm, (run, prog) in manifest(campaign).items():
            cpath = os.path.join(run, prog, "client.jsonl")
            t_start, rows = read_client(cpath)
            s = client_stats(rows)
            s["t_start"] = t_start
            s["server"] = server_stats(os.path.join(run, "server.log"), t_start)
            out[f"{campaign}/{arm}"] = s
            w = s["windows"]
            sv = s["server"]
            print(f"{campaign}/{arm}: n {s['requests']} ({s['req_per_min']:.1f}/min; 90-109 {s['req_per_min_90_109']:.1f}) "
                  f"users {s['users']} TTFT {s['ttft_mean']:.2f} p50 {s['ttft_p50']:.2f} p90 {s['ttft_p90']:.2f} "
                  f"p99 {s['ttft_p99']:.2f} | windows {' / '.join(format(x['ttft_mean'], '.2f') for x in w)} | "
                  f"gpu/host/ssd {s['share_gpu']:.3f}/{s['share_host']:.3f}/{s['share_ssd']:.3f} "
                  f"hist recomputed {s['history_recomputed']:.4f}", flush=True)
            print(f"    running median {[(k, v['running_median']) for k, v in sv['decode'].items()]} queue "
                  f"{[(k, v['queue_median']) for k, v in sv['decode'].items()]} usage "
                  f"{[(k, v['usage_median']) for k, v in sv['decode'].items()]}")
            print(f"    reads {sv['reads']}, dispatch {sv['dispatch']}, onset {sv['loop_onset_min_3']}/{sv['loop_onset_min_5']}, "
                  f"holds on {sv['write_holds_on']}, cleaner {[c[0] for c in sv['cleaner_passes']]}, counts {sv['counts']}")
            print(f"    io {json.dumps(sv['io'])}")
            if "census" in sv:
                print(f"    census {json.dumps(sv['census'])}")
            print(f"    [{time.time() - t0:.0f} s]", flush=True)
    with open(os.path.join(HERE, "measured.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
