"""Page-level record of one arm: every SSD write op tied to the request that made its pages, every SSD lookup and
read tied to the chain positions it covered, and what each request found in memory when it arrived.

  python3 -I -B model.py <campaign> <label>      # builds extracts/model_<c>_<label>.pkl and prints the join's quality

How ops are tied to requests (checked on campaign 34 ec_full: 99.6 % of submitted tokens):
  - a request's output node (the pages past floor64(prompt), holding the prompt's last partial page and the
    generated tokens) is the tail node registered at its commit (nue_register next to the request's ReqTimeStats
    line), when floor64(prompt + output) > floor64(prompt); arms without the registry (hicache) match by time and size;
  - its prompt pages [cached, floor64(prompt)) are a run of consecutive ops (one per <= 8,192-token node of a
    chunked prefill) submitted at its first token, whose tokens sum to floor64(prompt) - cached_input_len.
How lookups are placed: prefetch_start logs the tokens looked up; the key is the request's prompt less one token after
arrival, and the committed chain of the session's previous turn (ntp_plan chain) for a read staged before arrival; the
start is key length - tokens, and the hypothesis that gives a page-aligned start >= 0 is taken (arrival first when the
event is not before the request's entry time).
"""
from __future__ import annotations

import bisect
import collections
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import BASE, PAGE, RID, Run, events, kv, load_trees  # noqa: E402


class Model:
    pass


def floor_page(n: int) -> int:
    return n // PAGE * PAGE


def build(campaign: str, label: str, trees=None, verbose: bool = True) -> Model:
    run = Run(campaign, label, trees)
    m = Model()
    m.campaign, m.label = campaign, label
    m.t_start, m.args = run.t_start, run.args
    m.reqs, m.sessions, m.plays, m.lanes = run.reqs, run.sessions, run.plays, run.lanes
    log = lambda *a: print(*a) if verbose else None  # noqa: E731

    reqstat, reg, ops, order = {}, collections.defaultdict(list), {}, []
    groups = []
    plan_chain, plan_t, gate, ec_start, due, arrive_evt, admit_evt = {}, {}, collections.defaultdict(list), \
        collections.defaultdict(list), collections.defaultdict(list), {}, {}
    ps_all = collections.defaultdict(list)       # rid -> [(t, tokens)]
    lookups = []                                 # dict per s2h_query
    last_ps, last_q, park, disp = {}, {}, {}, {}
    reads_raw = []
    census, holds, evict_host, evict_dev, d2h, cleans, warns, prefill, decode = [], [], [], [], [], [], [], [], []
    pins, unpins, stage, refetch, reevict, hold_evt, rates, load_back, room = [], [], [], [], [], [], [], [], []
    step_items, d2h_fifo, pend_ops, transfers = [], collections.deque(), [], []
    pend_tok, pend_series = 0, []
    for t, ev, rest in events(run.extract):
        if ev == "CLEAN" or ev == "WARN":
            (cleans if ev == "CLEAN" else warns).append((t, rest))
            continue
        d = kv(rest)
        rid = d.get("rid")
        if ev == "REQSTAT":
            reqstat[rid] = {"t": t, "input_len": int(d["input_len"]), "cached": int(d["cached"]),
                            "output_len": int(d["output_len"]), "queue_ms": float(d["queue_ms"]),
                            "forward_ms": float(d["forward_ms"]), "entry": float(d["entry"])}
            step_items.append((t, "R", rid))
        elif ev == "nue_register":
            reg[rid].append((t, int(d["node"]), d["next_use_ms"]))
        elif ev == "h2s_submit":
            op = int(d["op"])
            pend_tok += int(d["tokens"])   # campaign 32's lines carry no pending=: count it here
            ops[op] = {"op": op, "node": int(d["node"]), "tokens": int(d["tokens"]), "t_sub": t,
                       "pending": int(d["pending"]) if "pending" in d else pend_tok, "t_done": None, "io_ms": None,
                       "t_io": None, "skipped": None, "group": None, "rid": None, "part": None, "p0": None}
            order.append(op)
            pend_ops.append(op)
            pend_series.append((t, ops[op]["pending"]))
        elif ev == "h2s_done":
            op = int(d["op"])
            if op in ops:
                pend_tok -= ops[op]["tokens"]
                ops[op]["t_done"] = t
                ops[op]["done_tokens"] = int(d["tokens"])
                ops[op]["pending_after"] = int(d["pending"]) if "pending" in d else pend_tok
                pend_series.append((t, ops[op]["pending_after"]))
        elif ev == "h2s_io":
            op = int(d["op"])
            if op in ops:
                o = ops[op]
                o["t_io"], o["io_ms"], o["io_pages"] = t, float(d["ms"]), int(d["pages"])
                if "skipped" in d:
                    o["skipped"] = int(d["skipped"])
                if "group" in d:
                    o["group"] = int(d["group"])
        elif ev == "h2s_group":
            groups.append({"t": t, "op": int(d["op"]), "ops": int(d["ops"]), "pages": int(d["pages"]),
                           "skipped": int(d["skipped"]), "ms": float(d["ms"]), "failed": int(d["failed"])})
        elif ev == "ntp_plan":
            plan_chain[rid] = int(d["chain"])
            plan_t[rid] = (t, int(d["gap_ms"]) if d["gap_ms"].lstrip("-").isdigit() else None)
        elif ev == "ntp_gate":
            gate[rid].append((t, d.get("decision"), int(d.get("matched", -1)), int(d.get("chain", -1)),
                              int(d.get("lookups", -1))))
        elif ev == "ec_start":
            ec_start[rid].append((t, d["decision"], int(d["tokens"]), int(d["idle_ms"]), d.get("read_ms"),
                                  d.get("wait_ms")))
        elif ev == "ntp_due":
            due[rid].append((t, int(d["case"]), int(d["l3_pages"]), d.get("load_ms"), d.get("wait_ms")))
        elif ev == "ntp_arrive":
            arrive_evt[rid] = (t, int(d["pos"]), int(d["qlen"]))
        elif ev == "ntp_admit":
            admit_evt[rid] = (t, d.get("wait_ms"), d.get("pred_wait_ql_ms"), d.get("case"))
        elif ev == "prefetch_start":
            last_ps[rid] = (t, int(d["tokens"]))
            ps_all[rid].append((t, int(d["tokens"])))
        elif ev == "s2h_query":
            q = {"t": t, "rid": rid, "tokens_req": int(d["tokens_req"]), "hit": int(d["storage_hit"]),
                 "ps": last_ps.get(rid)}
            last_q[rid] = q
            lookups.append(q)
        elif ev == "srs_park":
            park[rid] = {"t": t, "pages": int(d["pages"]), "deadline": d["deadline_ms"], "q": last_q.get(rid)}
        elif ev == "srs_dispatch":
            disp[rid] = {"t": t, "pages": int(d["pages"]), "waited_ms": int(d["waited_ms"]),
                         "inflight_pages": int(d["inflight_pages"]), "pinned_tokens": int(d["pinned_tokens"]),
                         "launched": int(d["launched"]), "pending_write": int(d.get("pending_write_tokens", -1)),
                         "park": park.get(rid), "rid": rid}
            reads_raw.append(("disp", disp[rid]))
        elif ev == "s2h_io":
            reads_raw.append(("io", {"t": t, "rid": rid, "pages": int(d["pages"]), "ms": float(d["ms"]),
                                     "disp": disp.get(rid), "q": last_q.get(rid)}))
        elif ev == "host_census":
            c = {k: int(v) for k, v in d.items() if k != "why"}
            c["t"], c["why"] = t, d["why"]
            census.append(c)
        elif ev == "srs_write_hold":
            holds.append((t, d["state"], int(d["pending_write_tokens"]), int(d["parked"]), int(d["parked_pages"])))
        elif ev == "evict_host":
            evict_host.append((t, int(d["tokens"]), int(d["requested"])))
        elif ev == "evict_device":
            evict_dev.append((t, int(d["tokens"]), int(d["requested"])))
        elif ev == "d2h_submit":
            d2h.append((t, int(d["nodes"]), int(d["tokens"])))
            # the step's lines come within a millisecond or two before its copy is submitted
            d2h_fifo.append({"t": t, "k": int(d["nodes"]), "T": int(d["tokens"]),
                             "items": [x for x in step_items if x[0] >= t - 0.01]})
            step_items = []
        elif ev == "d2h_done":
            # copies are acked in order, and the ack's SSD submits are logged just before its d2h_done line
            if d2h_fifo:
                tr = d2h_fifo.popleft()
                tr["ops"], tr["t_ack"] = pend_ops, t
                tr["ok"] = sum(ops[o]["tokens"] for o in pend_ops) == tr["T"] == int(d["tokens"])
                transfers.append(tr)
            pend_ops = []
        elif ev == "PREFILL":
            prefill.append((t, int(d["new_seq"]), int(d["new_token"]), int(d["cached_token"]), float(d["usage"]),
                            int(d["running"]), int(d["queue"])))
            step_items.append((t, "P", (int(d["new_seq"]), int(d["new_token"]), int(d["cached_token"]))))
        elif ev == "DECODE":
            decode.append((t, int(d["running"]), int(d["tokens"]), float(d["usage"]), int(d["queue"])))
        elif ev == "srs_pin":
            pins.append((t, rid, int(d["tokens"]), d.get("node"), int(d["pinned"])))
        elif ev == "srs_unpin":
            unpins.append((t, rid, int(d["tokens"]), d.get("reason"), int(d["pinned"])))
        elif ev == "ntp_stage":
            stage.append((t, rid, int(d["tokens"]), int(d["matched"]), int(d["gate"]), int(d["inflight"])))
        elif ev == "ntp_refetch":
            refetch.append((t, rid, int(d["issued"]), int(d["shrink"]), int(d["pos"])))
        elif ev == "ntp_reevict":
            reevict.append((t, rid, int(d["evicted"]), int(d["refetches"]), int(d["capped"])))
        elif ev == "ntp_hold":
            hold_evt.append((t, rid, d))
        elif ev == "srs_rate":
            rates.append((t, float(d["ms_per_page"])))
        elif ev == "load_back_init":
            load_back.append((t, rid, int(d["tokens"]), int(d["host_hit"]), int(d["node"])))
        elif ev == "ntp_room":
            room.append((t, rid, d))

    # ----- write ops -> requests -----
    # Structure of the log (checked: every transfer of campaign 34): a scheduler step prints its lines (the finished
    # requests' ReqTimeStats, or the Prefill batch line), then one d2h_submit for the nodes it inserted; copies are
    # acked in order, and each ack's h2s_submit lines sit just before its d2h_done line. So every write op belongs
    # to one step. In a finish step an op is a finished request's output node (the commit's tail node; by size and
    # order in arms without the registry). In a prefill step the ops are, in chain order, the prompt nodes of the
    # requests whose first token came with that step: the ops are split into runs whose tokens equal each
    # request's floor64(prompt) - cached_input_len.
    by_node = {ops[op]["node"]: op for op in order}
    commit_tail = {}
    for rid, st in reqstat.items():
        for t, node, _due in reg.get(rid, []):
            if abs(t - st["t"]) < 0.05:
                commit_tail[rid] = node
    expected = {}
    for rid, r in run.reqs.items():
        st = reqstat.get(rid)
        cached = st["cached"] if st else r.cached
        e = floor_page(r.P) - cached
        if e > 0 and r.t_first is not None:
            expected[rid] = (e, cached)
    admitted = {rid: st["entry"] + st["queue_ms"] / 1000.0 for rid, st in reqstat.items()}
    by_first = sorted((run.reqs[rid].t_first, rid) for rid in expected)
    first_ts = [x[0] for x in by_first]
    matched, prompt_ops = {}, {}
    n_out_exp = sum(1 for rid, r in run.reqs.items()
                    if rid in reqstat and floor_page(r.P + r.out) > floor_page(r.P))
    n_out = n_struct = n_bad_transfer = n_late = 0

    def split_runs(toks: list, cands: list):
        """Assign consecutive runs of toks to candidates (rid, e): a list of (rid, start, end), or None."""
        prefix = [0]
        for x in toks:
            prefix.append(prefix[-1] + x)
        ends = {v: i for i, v in enumerate(prefix)}

        def dfs(pos: int, used: tuple):
            if pos == len(toks):
                return []
            for ci, (rid, e) in enumerate(cands):
                if ci in used:
                    continue
                end = ends.get(prefix[pos] + e)
                if end is None or end <= pos:
                    continue
                rest = dfs(end, used + (ci,))
                if rest is not None:
                    return [(rid, pos, end)] + rest
            return None

        return dfs(0, ())

    for tr in transfers:
        if not tr["ok"]:
            n_bad_transfer += 1
            continue
        ops_tr = list(tr["ops"])
        for _t, kind, rid in tr["items"]:
            if kind != "R":
                continue
            r = run.reqs.get(rid)
            if r is None:
                continue
            e_out = floor_page(r.P + r.out) - floor_page(r.P)
            if e_out <= 0:
                continue
            tail = commit_tail.get(rid)
            pick = None
            for op in ops_tr:
                if ops[op]["tokens"] == e_out and (tail is None or ops[op]["node"] == tail):
                    pick = op
                    break
            if pick is not None:
                matched[pick] = (rid, "out")
                ops_tr.remove(pick)
                n_out += 1
        if not ops_tr:
            continue
        p_items = [x for x in tr["items"] if x[1] == "P"]
        t_b = p_items[-1][0] if p_items else tr["t"]
        # a finishing request whose prompt copy found no host room at its prefill is copied (and written) now,
        # with its output node (campaign 32's loop)
        cands = [(rid, expected[rid][0]) for _t, kind, rid in tr["items"]
                 if kind == "R" and rid in expected and rid not in prompt_ops]
        late = {rid for rid, _e in cands}
        if p_items:
            lo, hi = bisect.bisect_left(first_ts, t_b - 0.05), bisect.bisect_right(first_ts, t_b + 1.0)
            # a request cannot have inserted its prompt before it was admitted
            near = [(abs(tf - t_b - 0.003), rid) for tf, rid in by_first[lo:hi]
                    if rid not in prompt_ops and rid not in late and admitted.get(rid, -1e18) <= t_b + 0.01]
            near.sort()
            cands = [(rid, expected[rid][0]) for _c, rid in near[:14]] + cands
        got = split_runs([ops[o]["tokens"] for o in ops_tr], cands)
        if got is None:
            continue
        for rid, a_, b_ in got:
            prompt_ops[rid] = ops_tr[a_:b_]
            n_struct += 1
            n_late += 1 if rid in late else 0
    # what the steps did not settle: by time and size, cheapest pair first
    free_ops = [op for op in order if op not in matched and not any(op in v for v in ())]
    in_prompt = {op for v in prompt_ops.values() for op in v}
    free_ops = [op for op in free_ops if op not in in_prompt]
    ts_free = [ops[op]["t_sub"] for op in free_ops]
    cands = []
    for rid, (e, _cached) in expected.items():
        if rid in prompt_ops:
            continue
        r = run.reqs[rid]
        lo, hi = bisect.bisect_left(ts_free, r.t_first - 0.3), bisect.bisect_right(ts_free, r.t_first + 3.0)
        for i in range(lo, hi):
            first = free_ops[i]
            s_, run_ops, j = 0, [], i
            while s_ < e and j < len(free_ops) and ops[free_ops[j]]["t_sub"] - ops[first]["t_sub"] <= 0.05:
                s_ += ops[free_ops[j]]["tokens"]
                run_ops.append(free_ops[j])
                j += 1
            if s_ == e and admitted.get(rid, -1e18) <= ops[first]["t_sub"] + 0.01:
                cands.append((abs(ops[first]["t_sub"] - r.t_first - 0.02), rid, tuple(run_ops)))
    cands.sort()
    taken_ops, n_fallback = set(), 0
    for cost, rid, run_ops in cands:
        if rid in prompt_ops or any(op in taken_ops for op in run_ops) or cost > 2.5:
            continue
        prompt_ops[rid] = list(run_ops)
        taken_ops.update(run_ops)
        n_fallback += 1
    for rid, run_ops in prompt_ops.items():
        p0 = expected[rid][1] // PAGE
        for op in run_ops:
            matched[op] = (rid, "prompt")
            ops[op]["p0"] = p0
            p0 += ops[op]["tokens"] // PAGE
    # output nodes the steps did not settle (a few): nearest op of that size at the finish
    for rid, r in run.reqs.items():
        st = reqstat.get(rid)
        e_out = floor_page(r.P + r.out) - floor_page(r.P)
        if st is None or e_out <= 0 or any(matched.get(op) == (rid, "out") for op in ()):
            continue
    truth = [rid for rid in prompt_ops if rid in commit_tail and commit_tail[rid] in by_node
             and floor_page(run.reqs[rid].P + run.reqs[rid].out) == floor_page(run.reqs[rid].P)]
    right = sum(1 for rid in truth if ops[prompt_ops[rid][-1]]["node"] == commit_tail[rid])
    m.match_check = {"by_step": n_struct, "by_time": n_fallback, "scored": len(truth), "right": right,
                     "bad_transfers": n_bad_transfer, "prompt_written_at_finish": n_late}
    n_p_exp, n_p = len(expected), len(prompt_ops)
    unmatched_reqs = [(rid, e, run.reqs[rid].t_first) for rid, (e, _c) in expected.items() if rid not in prompt_ops]
    log(f"  prompt-write join: {n_struct} by scheduler step, {n_fallback} by time and size; check on the "
        f"{len(truth)} requests whose commit names their prompt node (no output page): {right} right "
        f"({right / max(len(truth), 1):.2%}); transfers whose ops did not sum: {n_bad_transfer}; prompt nodes "
        f"copied to host only at the request's finish: {n_late}")
    for op, (rid, part) in matched.items():
        ops[op]["rid"], ops[op]["part"] = rid, part
    left = [op for op in order if op not in matched]
    tot = sum(o["tokens"] for o in ops.values())
    m.join = {"ops": len(ops), "tokens": tot, "out_expected": n_out_exp, "out_matched": n_out,
              "prompt_expected": n_p_exp, "prompt_matched": n_p, "ops_left": len(left),
              "tokens_left": sum(ops[o]["tokens"] for o in left),
              "reqs_unmatched": len(unmatched_reqs), "reqs_unmatched_tokens": sum(x[1] for x in unmatched_reqs)}
    log(f"{campaign} {label}: {len(ops)} write ops, {tot / 1e6:.2f} M tokens submitted; output ops {n_out} of "
        f"{n_out_exp} expected; prompt writes {n_p} of {n_p_exp}; left {len(left)} ops "
        f"{m.join['tokens_left'] / 1e6:.3f} M tokens ({m.join['tokens_left'] / max(tot, 1):.2%}); requests with an "
        f"expected prompt write and none found {len(unmatched_reqs)} ({m.join['reqs_unmatched_tokens'] / 1e6:.3f} M)")

    # ----- lookups and reads -> chain positions -----
    def place(rid: str, t_ps: float, tok: int):
        """(owner rid, staged, start token) of a lookup of `tok` tokens issued at t_ps for rid."""
        r = run.reqs.get(rid)
        st = reqstat.get(rid)
        arr = st["entry"] if st else (r.t_send if r is not None else None)
        hyp = []
        if r is not None:
            hyp.append((False, r.P - 1 - tok))
        chain = plan_chain.get(rid)
        if chain is not None:
            hyp.append((True, chain - tok))
        if arr is None or t_ps < arr - 0.005:
            hyp.sort(key=lambda h: not h[0])     # before arrival: the staged reading first
        for staged, start in hyp:
            if start >= 0 and start % PAGE == 0:
                return staged, start
        return None, None

    for q in lookups:
        ps = q["ps"]
        q["staged"], q["start"] = place(q["rid"], ps[0], ps[1]) if ps else (None, None)
    reads, dispatches = [], []
    for kind, x in reads_raw:
        if kind == "disp":
            dispatches.append(x)
            continue
        q = x["q"]
        if x["disp"] is not None and x["disp"].get("park") is not None and x["disp"]["park"].get("q") is not None:
            q = x["disp"]["park"]["q"]
        staged, start = (q["staged"], q["start"]) if q else (None, None)
        reads.append({"t1": x["t"], "t0": x["t"] - x["ms"] / 1000.0, "ms": x["ms"], "rid": x["rid"],
                      "pages": x["pages"], "staged": staged, "start": start, "disp": x["disp"], "q": q})
    n_placed = sum(1 for r_ in reads if r_["start"] is not None)
    log(f"  SSD reads {len(reads)} ({sum(r_['pages'] for r_ in reads) * PAGE / 1e6:.2f} M tokens), placed {n_placed}; "
        f"staged {sum(1 for r_ in reads if r_['staged'])}; lookups {len(lookups)}")

    # ----- what each request found in memory at arrival -----
    arrival = {}
    for rid, r in run.reqs.items():
        st = reqstat.get(rid)
        if st is None:
            continue
        got = None
        for t_ps, tok in ps_all.get(rid, []):
            if t_ps >= st["entry"] - 0.005:
                start = r.P - 1 - tok
                if start >= 0 and start % PAGE == 0:
                    got = (t_ps, start)
                break
        arrival[rid] = got
    m.ops, m.order, m.groups, m.prompt_ops, m.reqstat, m.reg = ops, order, groups, prompt_ops, reqstat, dict(reg)
    m.plan_chain, m.plan_t, m.gate, m.ec_start, m.due = plan_chain, plan_t, dict(gate), dict(ec_start), dict(due)
    m.arrive_evt, m.admit_evt, m.lookups, m.reads, m.dispatches = arrive_evt, admit_evt, lookups, reads, dispatches
    m.arrival, m.ps_all = arrival, dict(ps_all)
    m.census, m.holds, m.evict_host, m.evict_dev, m.d2h = census, holds, evict_host, evict_dev, d2h
    m.cleans, m.warns, m.prefill, m.decode = cleans, warns, prefill, decode
    m.pins, m.unpins, m.stage, m.refetch, m.reevict, m.hold_evt, m.rates = pins, unpins, stage, refetch, reevict, \
        hold_evt, rates
    m.load_back, m.room = load_back, room
    m.pend_series, m.transfers = pend_series, [{k: v for k, v in tr.items() if k != "items"} for tr in transfers]
    m.unmatched_reqs = unmatched_reqs
    return m


def load(campaign: str, label: str, trees=None, rebuild: bool = False) -> Model:
    path = f"{BASE}/extracts/model_{campaign.replace('campaign', 'c')}_{label}.pkl"
    if os.path.exists(path) and not rebuild:
        with open(path, "rb") as f:
            return pickle.load(f)
    m = build(campaign, label, trees)
    with open(path, "wb") as f:
        pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    return m


def prev_req(m: Model, rid: str):
    """The session's previous request (the turn whose commit made the plan for rid)."""
    mm = RID.search(rid)
    if not mm:
        return None
    inst, k = int(mm[1]), int(mm[2])
    lst = m.sessions.get(inst)
    if not lst or k - 1 < 0:
        return None
    for r in lst:
        if r.k == k - 1:
            return r
    return None


if __name__ == "__main__":
    import model as _self   # so the pickles name model.Model, not __main__.Model

    trees = load_trees()
    for label in sys.argv[2:]:
        _self.load(sys.argv[1], label, trees, rebuild=True)
