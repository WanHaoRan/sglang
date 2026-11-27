"""Q1: what the SSD writes of one arm bought. Every submitted token gets one class:

  out_last   output node of a session's final request (the session then ended)
  out_live   output node of any other request (the replay never re-sends generated tokens)
  last       prompt pages first written by a session's final request
  rewrite    prompt pages submitted again (the key was submitted before in this run)
  read_back  prompt pages of other turns, read from the SSD at least once after the write
  lost       ... never read back, and a later request that held the page had to compute it again
  mem_only   ... never read back, later requests got it from GPU or host memory every time
  no_use     ... never read back, no later request in the run held the page
  unmatched  ops the join could not tie to a request

  python3 -I -B q1_writes.py <campaign> <label> [<label> ...]
"""
from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import BASE, PAGE, load_trees  # noqa: E402
from model import load  # noqa: E402
from pages import build_pages, served  # noqa: E402

CLASSES = ["read_back", "mem_only", "lost", "no_use", "last", "rewrite", "out_live", "out_last", "unmatched"]
WIN = float(os.environ.get("EV_WIN", "1800"))   # window of the by-time table, seconds


def analyse(campaign: str, label: str, trees) -> dict:
    m = load(campaign, label, trees)
    p = build_pages(m)
    rs, idx = p.rs, p.idx
    tok = collections.Counter()
    ms = collections.Counter()
    by_win = collections.defaultdict(collections.Counter)
    ms_win = collections.defaultdict(collections.Counter)
    sub = collections.Counter()
    first_write_done: dict = {}
    # the engine's own view of ends: the commit's registration says "ended"
    ended_reg = {}
    for rid, st in m.reqstat.items():
        if rid not in m.reqs:
            continue   # in flight when the client stopped: no client record
        for t, node, due in m.reg.get(rid, []):
            if abs(t - st["t"]) < 0.05:
                ended_reg[rid] = due == "ended"
    # sessions cut by the end of the run: the last request sent is not the stream's last
    cut_sessions = {inst for inst, lst in m.sessions.items() if not lst[-1].last}
    last_sent = {lst[-1].rid for lst in m.sessions.values()}
    next_write_t: dict = {}
    for kid, ws in enumerate(p.writes):
        if ws and len(ws) > 1:
            for a, b in zip(ws, ws[1:]):
                next_write_t[(kid, a[2])] = b[0]
    total_ms = 0.0
    out_read_pages = sum(info["out_pages"] for info in p.read_info)
    staged_pages = sum(rd["pages"] for rd in m.reads if rd["staged"])
    for op in m.order:
        o = m.ops[op]
        n_pages = o["tokens"] // PAGE
        io_ms = o["io_ms"] or 0.0
        total_ms += io_ms
        w = int(o["t_sub"] // WIN)
        if o["part"] is None:
            tok["unmatched"] += o["tokens"]
            ms["unmatched"] += io_ms
            by_win[w]["unmatched"] += o["tokens"]
            ms_win[w]["unmatched"] += io_ms
            continue
        r = m.reqs[o["rid"]]
        if o["part"] == "out":
            cls = "out_last" if r.last else "out_live"
            tok[cls] += o["tokens"]
            ms[cls] += io_ms
            by_win[w][cls] += o["tokens"]
            ms_win[w][cls] += io_ms
            continue
        i = idx[o["rid"]]
        a = p.kids_of[i]
        per_page_ms = io_ms / max(n_pages, 1)
        for j in range(o["p0"], o["p0"] + n_pages):
            if j >= len(a):
                tok["unmatched"] += PAGE
                continue
            kid = a[j]
            ws = p.writes[kid]
            first = ws[0][2] == op
            t_w = o["t_sub"]
            t_next = next_write_t.get((kid, op), float("inf"))
            rds = p.reads[kid] or ()
            rb = any(t_w <= t0 < t_next for t0, _, _ in rds)
            if not first:
                cls = "rewrite"
                sub["rewrite_read_back" if rb else "rewrite_not_read"] += PAGE
            elif r.last:
                cls = "last"
                if rb:
                    sub["last_read_back_by_other"] += PAGE
                elif any(p.t_arr[u] > t_w and u != i for u in p.uses[kid]):
                    sub["last_used_from_memory_by_other"] += PAGE
            elif rb:
                cls = "read_back"
                sub["read_back_count"] += sum(1 for t0, _, _ in rds if t_w <= t0 < t_next) * PAGE
            else:
                kinds = set()
                for u in p.uses[kid]:
                    if u != i and p.t_arr[u] > t_w:
                        kinds.add(served(p, u, kid))
                if "rec" in kinds:
                    cls = "lost"
                elif "mem" in kinds:
                    cls = "mem_only"
                else:
                    cls = "no_use"
                    if r.inst in cut_sessions and r.rid in last_sent:
                        sub["no_use_run_ended_first"] += PAGE
                    elif r.rid in last_sent:
                        sub["no_use_other_last_sent"] += PAGE
                    else:
                        sub["no_use_dropped_by_next_prompt"] += PAGE
            if first and cls in ("read_back", "mem_only", "lost", "no_use") and ended_reg.get(o["rid"]):
                sub[f"engine_said_ended_but_live_{cls}"] += PAGE
            tok[cls] += PAGE
            ms[cls] += per_page_ms
            by_win[w][cls] += PAGE
            ms_win[w][cls] += per_page_ms
    total = sum(tok.values())
    read_ms = sum(rd["ms"] for rd in m.reads)
    read_tok = sum(rd["pages"] for rd in m.reads) * PAGE
    skipped_pages = sum(o["skipped"] or 0 for o in m.ops.values() if o["group"] is None) + \
        sum(g["skipped"] for g in m.groups)
    dur = max(o["t_sub"] for o in m.ops.values())
    out = {"campaign": campaign, "label": label, "submitted_tokens": total, "write_ms": total_ms, "read_ms": read_ms,
           "read_tokens": read_tok, "skipped_pages": skipped_pages, "classes": dict(tok), "ms": dict(ms),
           "sub": dict(sub), "join": m.join, "staged_read_tokens": staged_pages * PAGE,
           "staged_read_output_tokens": out_read_pages * PAGE}
    print(f"\n=== {campaign} {label}: {total / 1e6:.2f} M tokens submitted for SSD writes in {len(m.ops)} ops; pages "
          f"skipped as stored {skipped_pages} ({skipped_pages * PAGE / 1e6:.2f} M tokens, "
          f"{skipped_pages * PAGE / total:.1%}); write thread busy {total_ms / 1e3:.0f} s; SSD reads "
          f"{read_tok / 1e6:.2f} M tokens, read thread busy {read_ms / 1e3:.0f} s; run {dur:.0f} s")
    print(f"{'class':<10} {'M tokens':>9} {'share':>7} {'write s':>8} {'of write s':>10} {'of read+write tokens':>21}")
    for c in CLASSES:
        print(f"{c:<10} {tok[c] / 1e6:9.3f} {tok[c] / total:7.1%} {ms[c] / 1e3:8.0f} {ms[c] / max(total_ms, 1):10.1%} "
              f"{tok[c] / (total + read_tok):21.1%}")
    wins = sorted(by_win)
    print(f"by {WIN / 60:g}-minute window of the submit time (M tokens; share of the window's submitted tokens):")
    print(f"{'window':<10}" + "".join(f"{c:>16}" for c in CLASSES) + f"{'all':>9}")
    for w in wins:
        tw = sum(by_win[w].values())
        print(f"{int(w * WIN / 60):3d}-{int((w + 1) * WIN / 60):<6d}" +
              "".join(f"{by_win[w][c] / 1e6:8.2f} {by_win[w][c] / tw:6.1%} " for c in CLASSES) + f"{tw / 1e6:9.2f}")
    out["by_window"] = {int(w * WIN / 60): dict(by_win[w]) for w in wins}
    out["ms_by_window"] = {int(w * WIN / 60): dict(ms_win[w]) for w in wins}
    print("details (M tokens):")
    for k, v in sorted(sub.items()):
        print(f"  {k}: {v / 1e6:.3f}")
    rb_reads = sub["read_back_count"]
    if tok["read_back"]:
        print(f"  pages read back were read {rb_reads / tok['read_back']:.2f} times each on average")
    print(f"  staged (pre-arrival) reads: {staged_pages * PAGE / 1e6:.2f} M tokens, of which the previous turn's "
          f"output node (never part of the next prompt): {out_read_pages * PAGE / 1e6:.3f} M")
    r1_prefill = tok["last"] + tok["out_last"]
    oracle_need = tok["read_back"] + sub["rewrite_read_back"]
    print(f"R1 (skip an ended session's last write, end known when the prompt is inserted): "
          f"{r1_prefill / 1e6:.2f} M = {r1_prefill / total:.1%} of submitted "
          f"({(ms['last'] + ms['out_last']) / 1e3:.0f} s = {(ms['last'] + ms['out_last']) / max(total_ms, 1):.1%} of the "
          f"write thread's time); with the end known only at the reply: {tok['out_last'] / 1e6:.2f} M = "
          f"{tok['out_last'] / total:.1%}")
    print(f"oracle (write only what is read back from the SSD later): needs {oracle_need / 1e6:.2f} M = "
          f"{oracle_need / total:.1%}; saves {(total - oracle_need) / 1e6:.2f} M = {(total - oracle_need) / total:.1%}; "
          f"if pages that had to be computed again also count as needed: saves "
          f"{(total - oracle_need - tok['lost']) / total:.1%}")
    out["r1_tokens"], out["oracle_need"] = r1_prefill, oracle_need
    # false ends: the engine registered 'ended' at a commit that was not the session's last
    fe = [rid for rid, e in ended_reg.items() if e and not m.reqs[rid].last]
    te = [rid for rid, e in ended_reg.items() if e and m.reqs[rid].last]
    missed = [rid for rid, e in ended_reg.items() if not e and m.reqs[rid].last]
    print(f"engine's end signal at commit: {len(te)} true ends, {len(fe)} commits registered ended that were not the "
          f"session's last request, {len(missed)} last requests not registered ended")
    out["ends"] = {"true": len(te), "false": len(fe), "missed": len(missed)}
    return out


def main() -> None:
    campaign, labels = sys.argv[1], sys.argv[2:]
    trees = load_trees()
    res = [analyse(campaign, label, trees) for label in labels]
    path = f"{BASE}/out/q1_{campaign.replace('campaign', 'c')}{'' if WIN == 1800.0 else '_w%d' % WIN}.json"
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    for r in res:
        old[r["label"]] = r
    with open(path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
