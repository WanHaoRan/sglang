#!/usr/bin/env python3
"""oracle_eval.py (--manifest compare_<t>/manifest.txt | --run <server run dir>) [--out DIR]

How well the oracle next-turn prefetch (agent_cache/ORACLE_PLAN.md) predicted and acted, per arm, from the
HICACHE_EVT ntp_* lines of server.log and the client's client.jsonl. Arms without ntp lines are skipped.
  arrival      ntp_arrive: arr_err = arrival - predicted arrival P; delta = raw offset (arrival - commit - gap)
  waits        ntp_admit: measured queue wait vs the two estimates made at arrival (queue length, EWMA)
  triggers     ntp_due per case; admission error (admission - predicted admission) and trigger lead (admission - T)
  reads        per staged (ntp_stage inflight=1) or re-issued (ntp_refetch issued=1) read: its end (the rid's first
               s2h_io after the action), read time vs the load estimate, and admission - read end (<= 0.05 s: the
               admission waited on the read, i.e. late)
  waste        staged reads whose turn never arrived (ntp_expire staged=1) or that the turn did not use (client
               cached_details.storage == 0); tokens re-fetched (ntp_refetch shrink) vs credited from storage; re-evictions
Writes oracle_eval.csv (one row per arm) and oracle_rids_<arm>.csv (one row per planned rid, for hand checks).
Admission time = the ntp_arrive stamp + the measured wait (ntp_admit wait_ms).
"""
import argparse, calendar, csv, json, os, re, statistics as st

EVT = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\.(\d{3})\] HICACHE_EVT (ntp_\w+|s2h_io) rid=(\S+)(.*)")
LATE_S = 0.05   # admission within 50 ms of the read's end: it waited on the read (one scheduler pass is ~ms-1 s)


def ts(day, ms):
    return calendar.timegm(tuple(map(int, re.split(r"[- :]", day))) + (0, 0, 0)) + int(ms) / 1000.0


def num(x):
    return None if x in (None, "na") else float(x)


def pct(xs, f):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(f * len(xs)))] if xs else None


def fmt(x, nd=0):
    return "-" if x is None else f"{x:.{nd}f}"


def load(run, client):
    ev = {}   # rid -> {kind: [(t, fields)]}
    for line in open(os.path.join(run, "server.log"), errors="replace"):
        m = EVT.match(line)
        if m:
            fields = dict(x.split("=", 1) for x in m[5].split() if "=" in x)
            ev.setdefault(m[4], {}).setdefault(m[3], []).append((ts(m[1], m[2]), fields))
    rows = [json.loads(l) for l in open(os.path.join(run, client, "client.jsonl")) if l.strip()]
    run_rec = next(r for r in rows if r.get("kind") == "run")
    tag = run_rec["args"].get("tag") or "replay"
    turns = {f"{tag}-c{r['conv']}-t{r['turn']}": r for r in rows if r.get("kind") == "turn" and "t_send" in r}
    return ev, turns


def per_rid(ev, turns):
    out = []
    for rid, e in ev.items():
        if "ntp_plan" not in e:
            continue
        first = lambda k: e[k][0] if k in e else (None, {})
        (t_plan, plan), (t_arr, arr), (t_due, due) = first("ntp_plan"), first("ntp_arrive"), first("ntp_due")
        (t_stage, stage), (t_adm, adm), (t_exp, exp) = first("ntp_stage"), first("ntp_admit"), first("ntp_expire")
        refetches = [(t, f) for t, f in e.get("ntp_refetch", ()) if f.get("issued") == "1"]
        # Each read the oracle started, with its own load estimate: a staged read (estimate from ntp_due), then re-issues.
        actions = ([(t_stage, num(due.get("load_ms")))] if stage.get("inflight") == "1" else []) + [
            (t, num(f.get("load_ms"))) for t, f in refetches]
        read_end = lambda start: next((t for t, _ in e.get("s2h_io", ()) if t >= start), None)
        reads = [(read_end(t) - t, load) for t, load in actions if read_end(t) is not None]
        last_end = read_end(actions[-1][0]) if actions else None   # the read an admission could wait on
        wait = num(adm.get("wait_ms"))
        admission = t_arr + wait / 1000 if t_arr is not None and wait is not None else None
        turn = turns.get(rid) or {}
        storage = int(((turn.get("cached_details") or {}).get("storage")) or 0)
        out.append(dict(
            rid=rid, gap_ms=num(plan.get("gap_ms")), chain=num(plan.get("chain")),
            arr_err_ms=num(arr.get("arr_err_ms")), delta_ms=num(arr.get("delta_ms")), pos=num(arr.get("pos")),
            due_case=num(due.get("case")), l3_pages=num(due.get("l3_pages")), backlog_pages=num(due.get("backlog_pages")),
            load_ms=num(due.get("load_ms")), est=due.get("est"),
            staged=stage.get("inflight") == "1", stage_gate=stage.get("gate"),
            refetches=len(refetches), refetch_tokens=sum(int(f.get("shrink", 0)) for _, f in refetches),
            reevicts=len(e.get("ntp_reevict", ())), capped=any(f.get("capped") == "1" for _, f in e.get("ntp_reevict", ())),
            read_ms=";".join(str(round(s * 1000)) for s, _ in reads),
            read_minus_load_ms=[round(s * 1000 - load) for s, load in reads if load is not None],
            admission_minus_read_end_ms=round((admission - last_end) * 1000) if admission is not None and last_end is not None else None,
            wait_ms=wait, pred_wait_ql_ms=num(adm.get("pred_wait_ql_ms")), pred_wait_ewma_ms=num(adm.get("pred_wait_ewma_ms")),
            adm_err_ms=num(adm.get("adm_err_ms")), trig_lead_ms=num(adm.get("trig_lead_ms")),
            expired=exp.get("reason"), expired_staged=exp.get("staged") == "1",
            client_storage=storage if turn else None, client_ttft=turn.get("ttft")))
    return sorted(out, key=lambda r: r["rid"])


def summarize(arm, R):
    absol = lambda k: [abs(r[k]) for r in R if r[k] is not None]
    err = lambda k: [r[k] - r["wait_ms"] for r in R if r[k] is not None and r["wait_ms"] is not None]
    actioned = [r for r in R if r["staged"] or r["refetches"]]
    slack = [r["admission_minus_read_end_ms"] for r in actioned if r["admission_minus_read_end_ms"] is not None]
    load_err = [x for r in actioned for x in r["read_minus_load_ms"]]
    staged = [r for r in R if r["staged"]]
    refetched = [r for r in R if r["refetches"]]
    s = dict(arm=arm, plans=len(R), arrived=sum(r["delta_ms"] is not None for r in R),
             admitted=sum(r["wait_ms"] is not None for r in R),
             expired_timeout=sum(r["expired"] == "timeout" for r in R), expired_abort=sum(r["expired"] == "abort" for r in R),
             arr_err_abs_p50_ms=pct(absol("arr_err_ms"), .5), arr_err_abs_p90_ms=pct(absol("arr_err_ms"), .9),
             delta_p50_ms=pct([r["delta_ms"] for r in R if r["delta_ms"] is not None], .5),
             wait_ql_mae_ms=st.mean(abs(x) for x in err("pred_wait_ql_ms")) if err("pred_wait_ql_ms") else None,
             wait_ql_bias_ms=st.mean(err("pred_wait_ql_ms")) if err("pred_wait_ql_ms") else None,
             wait_ewma_mae_ms=st.mean(abs(x) for x in err("pred_wait_ewma_ms")) if err("pred_wait_ewma_ms") else None,
             wait_ewma_bias_ms=st.mean(err("pred_wait_ewma_ms")) if err("pred_wait_ewma_ms") else None,
             due_case1=sum(r["due_case"] == 1 for r in R), due_case2=sum(r["due_case"] == 2 for r in R),
             adm_err_abs_p50_ms=pct(absol("adm_err_ms"), .5),
             trig_lead_p50_ms=pct([r["trig_lead_ms"] for r in R if r["trig_lead_ms"] is not None], .5),
             staged=len(staged), refetched=len(refetched), actioned=len(actioned),
             late=sum(x <= LATE_S * 1000 for x in slack), slack_p50_ms=pct(slack, .5), slack_p90_ms=pct(slack, .9),
             load_err_p50_ms=pct(load_err, .5),
             wasted_staged_expired=sum(r["expired_staged"] for r in R),
             wasted_staged_unused=sum(r["staged"] and r["client_storage"] == 0 for r in R),
             refetch_tokens=sum(r["refetch_tokens"] for r in R),
             refetch_storage_credited=sum(r["client_storage"] or 0 for r in refetched),
             reevicts=sum(r["reevicts"] for r in R), capped=sum(r["capped"] for r in R))
    print(f"{arm}: {s['plans']} plans, {s['arrived']} arrived, {s['admitted']} admitted, expired {s['expired_timeout']} "
          f"(+{s['expired_abort']} aborted)")
    print(f"  arrival |err| p50/p90 {fmt(s['arr_err_abs_p50_ms'])}/{fmt(s['arr_err_abs_p90_ms'])} ms, raw offset p50 "
          f"{fmt(s['delta_p50_ms'])} ms; wait MAE (bias) by queue length {fmt(s['wait_ql_mae_ms'])} ({fmt(s['wait_ql_bias_ms'])}) ms, "
          f"by EWMA {fmt(s['wait_ewma_mae_ms'])} ({fmt(s['wait_ewma_bias_ms'])}) ms")
    print(f"  due case 1/2 {s['due_case1']}/{s['due_case2']}; admission |err| p50 {fmt(s['adm_err_abs_p50_ms'])} ms, trigger "
          f"lead p50 {fmt(s['trig_lead_p50_ms'])} ms")
    print(f"  reads: staged {s['staged']}, re-issued {s['refetched']}; late {s['late']} of {len(slack)} with an end; admission - "
          f"read end p50/p90 {fmt(s['slack_p50_ms'])}/{fmt(s['slack_p90_ms'])} ms; read - load estimate p50 {fmt(s['load_err_p50_ms'])} ms")
    print(f"  waste: staged, turn never came {s['wasted_staged_expired']}, staged but unused {s['wasted_staged_unused']}; "
          f"re-fetched {s['refetch_tokens']} tokens, credited from storage {s['refetch_storage_credited']}; "
          f"re-evictions {s['reevicts']} (capped {s['capped']})")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest"); ap.add_argument("--run"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.manifest:
        root = os.path.dirname(os.path.dirname(os.path.abspath(a.manifest)))
        jobs = [(arm, os.path.join(root, run), client) for arm, run, client in (l.split()[:3] for l in open(a.manifest) if l.strip())]
        out = a.out or os.path.dirname(os.path.abspath(a.manifest))
    elif a.run:
        client = sorted(d for d in os.listdir(a.run) if d.startswith("client_"))[-1]
        arm = next((l.split("=", 1)[1].strip() for l in open(os.path.join(a.run, "run.txt")) if l.startswith("arm=")), "run")
        jobs, out = [(arm, a.run, client)], a.out or a.run
    else:
        ap.error("--manifest or --run is required")
    os.makedirs(out, exist_ok=True)
    rows = []
    for arm, run, client in jobs:
        R = per_rid(*load(run, client))
        if not R:
            print(f"{arm}: no ntp_plan lines (mode off or no hints), skipped")
            continue
        with open(os.path.join(out, f"oracle_rids_{arm}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(R[0].keys())); w.writeheader()
            w.writerows({k: ";".join(map(str, v)) if isinstance(v, list) else v for k, v in r.items()} for r in R)
        rows.append(summarize(arm, R))
    if rows:
        with open(os.path.join(out, "oracle_eval.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print("wrote", os.path.join(out, "oracle_eval.csv"))


if __name__ == "__main__":
    main()
