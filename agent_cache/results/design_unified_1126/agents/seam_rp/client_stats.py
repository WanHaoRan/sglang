"""Client-side sizes behind R1 (ended sessions' last turn), R4 (what the SSD holds for ended sessions,
utime calls) and P1 (history recomputed), from one client.jsonl and the arm's event extract.

Usage: python3 -I -B client_stats.py <client.jsonl> <ext/arm.evt> [cleaner pass minutes ...]
A stream instance ("-c<inst>-") is one hinted session. A request whose hint names the reply key "_end" is the
stream's last turn in the trace (the explicit end signal). Times are seconds since the client's t_start.
"""
import gzip
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

PAGE = 64
RID = re.compile(r"-c(\d+)-t(\d+)$")
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")


def open_text(path):
    """The extract, gzipped or not."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, errors="replace")


def q(values, p):
    if not values:
        return float("nan")
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


def pages(tokens):
    return tokens // PAGE


def main(client_path, evt_path, pass_minutes):
    run = None
    streams = defaultdict(list)   # inst -> [req record]
    for line in open(client_path):
        if '"kind":"run"' in line:
            run = json.loads(line)
            continue
        if '"kind":"req"' not in line:
            continue
        r = json.loads(line)
        if r.get("ttft") is None or r.get("prompt_tokens") is None:
            continue
        m = RID.search(r["rid"])
        r["_k"] = int(m.group(2))
        streams[int(m.group(1))].append(r)
    t_start = run["t_start"]
    n_req = sum(len(v) for v in streams.values())
    print(f"client {client_path}")
    print(f"streams (sessions) {len(streams)}, requests answered {n_req}, hints {run['args']['hints']}")

    # ---- R1: the last turn of sessions that end explicitly ----
    tot_new = tot_out = tot_recomp = tot_ideal = 0
    end_new = end_out = 0
    n_end = n_end_single = 0
    end_done_at = {}      # inst -> time the last turn finished
    first_prompt = {}
    final_ctx = {}
    outs = {}
    recomp_by_win = defaultdict(int)
    ideal_by_win = defaultdict(int)
    ret_by_win = defaultdict(int)
    for inst, lst in streams.items():
        lst.sort(key=lambda r: r["_k"])
        first_prompt[inst] = lst[0]["prompt_tokens"] if lst[0]["_k"] == 0 else None
        final_ctx[inst] = max(r["prompt_tokens"] for r in lst)
        outs[inst] = sum(r["completion_tokens"] or 0 for r in lst)
        for r in lst:
            cached = r.get("cached_tokens") or 0
            ideal = r.get("ideal_cached") or 0
            new = r["prompt_tokens"] - cached      # computed on the GPU, inserted, backed up and written
            out = r["completion_tokens"] or 0
            tot_new += new
            tot_out += out
            recomp = max(0, ideal - cached)       # history an unbounded cache holds but this turn recomputed
            tot_recomp += recomp
            tot_ideal += ideal
            w = int(r["t_send"] // 1800)
            if r["_k"] > 0:
                recomp_by_win[w] += recomp
                ideal_by_win[w] += ideal
                ret_by_win[w] += 1
            if (r.get("hint") or {}).get("reply_key") == "_end":
                n_end += 1
                n_end_single += r["_k"] == 0
                end_new += new
                end_out += out
                end_done_at[inst] = r["t_send"] + (r.get("latency") or 0.0)
    written = tot_new + tot_out
    print(f"[R1] tokens computed and inserted (prompt - cached + output): {written / 1e6:.2f} M "
          f"(new prompt {tot_new / 1e6:.2f} M, outputs {tot_out / 1e6:.2f} M)")
    print(f"     sessions with an explicit end: {n_end} ({n_end_single} of one request); their last turn: "
          f"new prompt {end_new / 1e6:.3f} M + output {end_out / 1e6:.3f} M = {(end_new + end_out) / 1e6:.3f} M tokens "
          f"= {(end_new + end_out) / written:.2%} of what is inserted; output part alone {end_out / written:.2%}")
    print(f"[P1] history recomputed (ideal_cached - cached_tokens, returning turns): {tot_recomp / 1e6:.2f} M of "
          f"{tot_ideal / 1e6:.2f} M ideal = {tot_recomp / max(tot_ideal, 1):.2%}")
    for w in sorted(ret_by_win):
        print(f"     min {w * 30:3d}-{w * 30 + 30:3d}: returning turns {ret_by_win[w]:5d}, recomputed "
              f"{recomp_by_win[w] / 1e6:5.2f} M of {ideal_by_win[w] / 1e6:6.2f} M ideal = "
              f"{recomp_by_win[w] / max(ideal_by_win[w], 1):.2%}")

    # ---- server: which requests' sessions were registered as ended, and did they continue? ----
    ended_rids = {}
    plan_rids = set()
    for line in open_text(evt_path):
        if "nue_register" in line and "next_use_ms=ended" in line:
            m = TS.match(line)
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
            rid = re.search(r"rid=(\S+)", line).group(1)
            ended_rids.setdefault(rid, t - t_start)
    answered = {r["rid"]: r for lst in streams.values() for r in lst}
    true_end = false_end = unknown = 0
    false_examples = []
    for rid, t in ended_rids.items():
        m = RID.search(rid)
        if m is None:
            unknown += 1
            continue
        inst, k = int(m.group(1)), int(m.group(2))
        nxt = re.sub(r"-t\d+$", f"-t{k + 1}", rid)
        rec = answered.get(rid)
        if rec is not None and (rec.get("hint") or {}).get("reply_key") == "_end":
            true_end += 1
        elif nxt in answered:
            false_end += 1
            if len(false_examples) < 3:
                false_examples.append((rid, (rec or {}).get("hint")))
        else:
            unknown += 1
    print(f"[R1] server registrations 'ended' by rid: {len(ended_rids)}; the trace's explicit end {true_end}; "
          f"the session sent a next turn anyway (false end) {false_end}; neither (run end, abort, no record) {unknown}")
    for ex in false_examples:
        print(f"     false end e.g. {ex}")

    # ---- R4: what ended sessions hold on the SSD at each cleaner pass; utime calls ----
    excl = {}
    for inst in streams:
        fp = first_prompt[inst]
        body = final_ctx[inst] - (fp or 0)
        excl[inst] = (fp or 0, max(0, body), outs[inst])
    for minute in pass_minutes:
        T = minute * 60.0
        ended = [i for i, t in end_done_at.items() if t <= T]
        fp = sum(excl[i][0] for i in ended)
        body = sum(excl[i][1] for i in ended)
        out = sum(excl[i][2] for i in ended)
        live = [i for i, lst in streams.items() if i not in end_done_at or end_done_at[i] > T]
        live = [i for i in live if streams[i][0]["t_send"] <= T]
        live_tok = sum(max((r["prompt_tokens"] for r in streams[i] if r["t_send"] <= T), default=0) for i in live)
        print(f"[R4] pass at minute {minute:5.1f}: sessions ended {len(ended)}; their first prompts {fp / 1e6:5.2f} M, "
              f"pages after the first prompt {body / 1e6:5.2f} M, outputs {out / 1e6:5.2f} M tokens "
              f"(a pass deletes about 2.6 M); live sessions {len(live)} hold {live_tok / 1e6:5.2f} M prompt tokens")
    # utime calls at an end: two files a page, over the session's pages after its first prompt plus outputs
    calls = sorted(2 * pages(excl[i][1] + excl[i][2]) for i in end_done_at)
    calls_all = sorted(2 * pages(sum(excl[i])) for i in end_done_at)
    span = max(end_done_at.values()) - min(end_done_at.values()) if end_done_at else 1.0
    print(f"[R4] utime calls to zero an ended session (2 files a page): after its first prompt p50 {q(calls, .5)}, "
          f"p90 {q(calls, .9)}, max {calls[-1] if calls else 0}, sum {sum(calls)} = {sum(calls) / span:.0f}/s over the run; "
          f"whole chain p50 {q(calls_all, .5)}, p90 {q(calls_all, .9)}, max {calls_all[-1] if calls_all else 0}")
    # refresh on memory serve: per request the pages memory served (device + host), each page at most once per interval
    for interval in (60.0, 300.0, 900.0):
        last = {}      # (inst, page) -> last touch time; streams treated as disjoint (an upper bound)
        total = 0
        per_sec = defaultdict(int)
        peak_req = 0
        for inst, lst in streams.items():
            for r in lst:
                d = r.get("cached_details") or {}
                served = (d.get("device", 0) or 0) + (d.get("host", 0) or 0) + (d.get("storage", 0) or 0)
                t = r["t_send"] + (r.get("ttft") or 0.0)
                n = 0
                for p in range(pages(served)):
                    key = (inst, p)
                    if t - last.get(key, -1e18) >= interval:
                        last[key] = t
                        n += 1
                total += 2 * n
                per_sec[int(t)] += 2 * n
                peak_req = max(peak_req, 2 * n)
        secs = sorted(per_sec.values())
        print(f"[R4] refresh at most once per {interval:5.0f} s a page: {total} utime calls in the run = "
              f"{total / 7200:.0f}/s mean, busiest second {secs[-1] if secs else 0}, p99 second {q(secs, .99)}, "
              f"largest single request {peak_req}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3:]])
