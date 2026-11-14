"""Real turns for the failure slides of the "Pricing the Room" deck, from campaign 23's 44-user runs (70 GB of host
memory, gaps x10). Each case is one turn drawn on a slide in each of its runs, and the run-wide counts the turn stands
for.

  python3 deck_cases.py   # prints each case; writes results/deck_cases_c44/cases.json

- late:    original HiCache starts a read when its turn arrives, so a turn with nothing ahead of it waits for the read;
           hinted runs start it before the turn arrives.
- early:   JIT p50 stages a read that lands long before its turn is admitted; the read holds host room meanwhile.
- nopin:   original HiCache reads at arrival and pins nothing, so a queued turn's read lands early and is lost; its
           retry reads the history again and can lose it again.
- lost:    original HiCache has no read pins: a read lands, its turn waits in the queue, and LRU evicts the read first.
           Pinned runs keep it.
- nextuse: JIT p50 evicts by the p50 as next use, so a session that returns earlier than its p50 has lost its history;
           LRU keeps a session it has just used.

Times are seconds since the client's t_start, the server log read as UTC (as room_model.py does). Arrival and
admission are the engine's own: ReqTimeStats entry_time and entry_time + queue_duration.
"""
import bisect
import collections
import gzip
import json
import os
import re
import statistics as st

from room_model import KV_RE, PAGE, RES, RID_RE, client_dir, parse_ts, pct, reads_of

OUT = os.path.join(RES, "deck_cases_c44")
RUNS = {   # campaign 23 at 44 users (CAMPAIGN23_FINDINGS.md, run table)
    "base": "20261101_082229_three_tier_wc_custom_L1nat_H70",
    "hintfree": "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "exact": "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "p50": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
}
RUNS_C128 = {   # 128 users, 160 GB: HiCache from campaign 19, the JIT arms from campaign 23 (as jit_vs_hicache.py)
    "c128_base": "20261018_103544_three_tier_wc_NAT160",
    "c128_exact": "20261102_002657_three_tier_wc_oracle_v2_nu_owned_NAT160",
    "c128_p50": "20261102_023031_three_tier_wc_oracle_v2_nu_owned_NAT160",
}
# The turn each slide draws, chosen from the printed candidates, and the runs it is drawn in.
EXAMPLES = {"late": ("169-29", ("base", "p50", "exact")), "early": ("228-15", ("p50", "exact")),
            "nopin": ("90-49", ("base", "exact")), "lost": ("180-46", ("base", "p50", "exact")),
            "nextuse": ("113-44", ("base", "p50", "exact"))}
EXAMPLES_C128 = {"gpu": ("239-4", ("base", "p50", "exact"))}
RTS_RE = re.compile(r"ReqTimeStats\(rid=([^,]+),.*queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, "
                    r"entry_time=([\d.]+)")
# Plain log lines kept as events: a read inserted into host memory, and a queued turn's match found shorter than at
# its last lookup (the engine then looks the history up on the SSD again).
LOG_EVENTS = {"HiCache prefetch success": "loaded", "HiCache device prefix shrank before admission": "shrank"}
GATED_S = 0.05   # admitted this soon after its read landed: the read set the admission


def load(name: str) -> dict:
    """Per-turn client records and server events of one run, cached as JSON."""
    cache = os.path.join(OUT, f"run_{name}_v2.json.gz")
    if os.path.exists(cache):
        with gzip.open(cache, "rt") as f:
            return json.load(f)
    run = os.path.join(RES, {**RUNS, **RUNS_C128}[name])
    rows = [json.loads(line) for line in open(os.path.join(client_dir(run), "client.jsonl"))]
    t0 = next(r for r in rows if r["kind"] == "run")["t_start"]
    turns = {}
    for r in rows:
        if r["kind"] != "turn":
            continue
        cd = r.get("cached_details") or {}
        turns[f"{r['conv']}-{r['turn']}"] = dict(
            conv=r["conv"], turn=r["turn"], t_send=r["t_send"], ttft=r.get("ttft"), prompt=r.get("prompt_tokens"),
            delta=r.get("delta_tokens"), dev=cd.get("device", 0), host=cd.get("host", 0), ssd=cd.get("storage", 0),
            gap=r.get("gap_slept"), hint=r.get("dist_hint") or r.get("oracle_hint"), events=[])
    waits, revokes = [], []
    for line in open(os.path.join(run, "server.log"), errors="replace"):
        if "ReqTimeStats(rid=" in line:
            m = RTS_RE.search(line)
            entry, queue = float(m[4]) - t0, float(m[2]) / 1000
            waits.append([entry, entry + queue])
            rid = RID_RE.search(m[1])
            if rid and f"{rid[1]}-{rid[2]}" in turns:
                turns[f"{rid[1]}-{rid[2]}"].update(entry=entry, admit=entry + queue)
            continue
        if "HICACHE_EVT" in line:
            kind = line.split("HICACHE_EVT ", 1)[1].split(" ", 1)[0]
            kv = dict(KV_RE.findall(line.split(kind, 1)[1]))
        else:
            kind = next((k for msg, k in LOG_EVENTS.items() if msg in line), None)
            if kind is None:
                continue
            kv = dict(KV_RE.findall(line.split("req=", 1)[1]), rid=line.split("req=", 1)[1].split()[0])
        t = parse_ts(line) - t0
        if kind == "srs_dispatch" and kv["launched"] == "0":
            revokes.append((t, kv["rid"], int(kv["pages"])))
        rid = RID_RE.search(kv.get("rid", ""))
        if rid and f"{rid[1]}-{rid[2]}" in turns:
            turns[f"{rid[1]}-{rid[2]}"]["events"].append((t, kind, kv))
    out = dict(name=name, turns=turns, waits=sorted(waits), revokes=revokes)
    os.makedirs(OUT, exist_ok=True)
    with gzip.open(cache, "wt") as f:
        json.dump(out, f)
    return out


def served(tr: dict) -> bool:
    """A returning turn the run admitted and answered."""
    return tr["turn"] > 0 and tr.get("entry") is not None and tr.get("ttft") is not None


def loads_of(tr: dict) -> list:
    """(time, tokens) of each read inserted into host memory by the turn's admission."""
    return [(t, int(kv["loaded"])) for t, k, kv in tr["events"]
            if k == "loaded" and t <= tr["admit"] + GATED_S and int(kv["loaded"])]


def history(tr: dict) -> int:
    return (tr["prompt"] or 0) - (tr["delta"] or 0)


def recomputed(tr: dict) -> int:
    return max(0, history(tr) - tr["dev"] - tr["host"] - tr["ssd"])


def hint_of(run: dict, tr: dict) -> dict:
    """The hint the previous turn sent for this turn's gap."""
    return (run["turns"].get(f"{tr['conv']}-{tr['turn'] - 1}") or {}).get("hint") or {}


def ahead_at(run: dict, t: float) -> int:
    """Requests that entered the waiting queue before t and were still waiting at t."""
    i = bisect.bisect_left(run["waits"], [t - 1e-6, float("inf")])
    return sum(1 for entry, adm in run["waits"][:i] if adm > t + 1e-6)


def ssd_reads(tr: dict) -> list:
    """The turn's SSD reads: from the read scheduler's lines, or, in original HiCache, from s2h_io alone."""
    if any(k == "srs_park" for _, k, _ in tr["events"]):
        return reads_of(tr["events"])
    return [dict(park=t - float(kv["ms"]) / 1000, dispatch=t - float(kv["ms"]) / 1000, launched=True, land=t,
                 pages=int(kv["pages"])) for t, k, kv in tr["events"] if k == "s2h_io"]


def staged_reads(tr: dict) -> list:
    """Reads parked before the turn arrived that landed before it was admitted, with their lead over admission."""
    t_arr, t_adm = tr.get("entry"), tr.get("admit")
    if t_arr is None:
        return []
    return [dict(park=r["park"], land=r["land"], tokens=r["pages"] * PAGE, lead=t_adm - r["land"])
            for r in reads_of(tr["events"]) if r["park"] < t_arr and r.get("land") is not None and r["land"] <= t_adm]


def shared_returning(runs: list) -> list:
    """Returning turns every run served, by key."""
    keys = set.intersection(*({k for k, tr in run["turns"].items() if served(tr)} for run in runs))
    return sorted(keys)


def timeline(run: dict, key: str) -> dict:
    """One turn's events in seconds from its arrival, for drawing."""
    tr = run["turns"][key]
    e = tr["entry"]
    prev = run["turns"].get(f"{tr['conv']}-{tr['turn'] - 1}") or {}
    regs = [kv["next_use_ms"] for _, k, kv in prev.get("events", []) if k == "nue_register"]
    h = hint_of(run, tr)
    reads = [dict(start=round(r["dispatch"] - e, 3), land=None if r.get("land") is None else round(r["land"] - e, 3),
                  tokens=r["pages"] * PAGE, launched=r["launched"]) for r in ssd_reads(tr)]
    out = dict(turn=key, arm=run["name"], gap=tr["gap"], admit=round(tr["admit"] - e, 3), ttft=round(tr["ttft"], 3),
               history=history(tr), new=tr["delta"], dev=tr["dev"], host=tr["host"], ssd=tr["ssd"],
               recomputed=recomputed(tr),
               ahead=ahead_at(run, e), reads=reads,
               pins=[(round(t - e, 3), int(kv["tokens"])) for t, k, kv in tr["events"] if k == "srs_pin"],
               lookups=[(round(t - e, 3), int(kv["tokens"])) for t, k, kv in tr["events"] if k == "prefetch_start"],
               loaded=[(round(t - e, 3), int(kv["loaded"])) for t, k, kv in tr["events"] if k == "loaded"],
               shrank=[(round(t - e, 3), int(kv["lookup_match"]), int(kv["current_match"]))
                       for t, k, kv in tr["events"] if k == "shrank"],
               hint={k: h.get(k) for k in ("reply_key", "p10", "p50", "p90", "n")} if h.get("p50") else None,
               next_use_s=float(regs[-1]) / 1000 if regs and regs[-1] not in ("0", "ended") else None)
    first_land = next((r["land"] for r in ssd_reads(tr) if r.get("land") is not None), None)
    if first_land is not None and first_land < e:
        # Reads refused host room while this one sat landed and pinned, and how many were for turns already waiting.
        refused = [rid for t, rid, _ in run["revokes"] if first_land <= t <= tr["admit"]]
        waiting = set()
        for t, rid, _ in run["revokes"]:
            m = RID_RE.search(rid)
            other = run["turns"].get(f"{m[1]}-{m[2]}") if m else None
            if first_land <= t <= tr["admit"] and other and other.get("entry") is not None and other["entry"] < t:
                waiting.add(f"{m[1]}-{m[2]}")
        out.update(refused_while_pinned=len(refused), refused_waiting_turns=len(waiting),
                   refused_waiting_recomputed=sum(recomputed(run["turns"][k]) for k in waiting))
    return out


# ===== Cases =====

def case_late(runs: dict) -> dict:
    """Turns admitted the moment their own read was inserted into host memory: the read set their admission."""
    summary = {}
    for name, run in runs.items():
        reading, gated, alone = 0, [], []
        for tr in run["turns"].values():
            mine = loads_of(tr) if served(tr) else []
            if not mine:
                continue
            reading += 1
            if 0 <= tr["admit"] - mine[-1][0] <= GATED_S:
                gated.append(tr["admit"] - tr["entry"])
                if ahead_at(run, tr["entry"]) == 0:
                    alone.append(tr["admit"] - tr["entry"])
        summary[name] = dict(returning=sum(map(served, run["turns"].values())), reading_turns=reading,
                             gated=len(gated), gated_alone=len(alone), wait_gated_mean=round(st.mean(gated), 2),
                             wait_alone_mean=round(st.mean(alone), 2))
    base, p50, exact = runs["base"], runs["p50"], runs["exact"]
    picks = []
    for key, tr in base["turns"].items():
        mine = loads_of(tr) if served(tr) else []
        tp, te = p50["turns"].get(key), exact["turns"].get(key)
        if len(mine) != 1 or not (tp and te and served(tp) and served(te)):
            continue
        sp, se = staged_reads(tp), staged_reads(te)
        one_read = all(len(ssd_reads(t)) == 1 and t["ssd"] >= 0.9 * mine[0][1] for t in (tr, tp, te))
        if (0 <= tr["admit"] - mine[0][0] <= GATED_S and ahead_at(base, tr["entry"]) == 0 and mine[0][1] >= 20000
                and one_read and len(sp) == 1 and len(se) == 1):
            picks.append(dict(turn=key, tokens=mine[0][1], wait=round(tr["admit"] - tr["entry"], 2),
                              ttft=round(tr["ttft"], 2), p50_lead=round(sp[0]["lead"], 1), p50_ttft=round(tp["ttft"], 2),
                              exact_lead=round(se[0]["lead"], 1), exact_ttft=round(te["ttft"], 2)))
    # Turns whose TTFT falls from original HiCache to p50 to exact hints first: their rows tell one story.
    picks.sort(key=lambda p: (not p["ttft"] > p["p50_ttft"] >= p["exact_ttft"], -p["tokens"]))
    return dict(summary=summary, picks=picks[:12])


def case_early(p50: dict, exact: dict) -> dict:
    """Staged reads that land early pin host room until admission."""
    summary = {}
    for name, run in (("p50", p50), ("exact", exact)):
        reads = [r for tr in run["turns"].values() for r in staged_reads(tr)]
        summary[name] = dict(
            staged=len(reads), lead_p50=pct([r["lead"] for r in reads], 0.5),
            lead_p90=pct([r["lead"] for r in reads], 0.9),
            pin_token_s=round(sum(r["lead"] * r["tokens"] for r in reads)),
            pin_token_s_over_10s=round(sum(r["lead"] * r["tokens"] for r in reads if r["lead"] > 10)),
            revoked=len(run["revokes"]))
    picks = []
    for key, tr in p50["turns"].items():
        twin = exact["turns"].get(key)
        mine, theirs = staged_reads(tr), staged_reads(twin or dict(events=[]))
        if not mine or not theirs:
            continue
        rd = max(mine, key=lambda r: r["lead"] * r["tokens"])
        if rd["tokens"] >= 24000 and 15 <= rd["lead"] <= 45 and theirs[0]["lead"] <= 3:
            h = hint_of(p50, tr)
            picks.append(dict(turn=key, key=h.get("reply_key"), p10=h.get("p10"), p50=h.get("p50"), gap=tr["gap"],
                              tokens=rd["tokens"], lead=round(rd["lead"], 2), exact_lead=round(theirs[0]["lead"], 2),
                              ttft=round(tr["ttft"], 2), exact_ttft=round(twin["ttft"], 2)))
    picks.sort(key=lambda p: -p["lead"] * p["tokens"])
    return dict(summary=summary, picks=picks[:12])


def case_nopin(base: dict, hintfree: dict, exact: dict) -> dict:
    """Tokens reads loaded into host memory before admission that the turn no longer found there, with and without pins."""
    summary = {}
    for name, run in (("base", base), ("hintfree", hintfree)):
        reading = twice = loaded = used = 0
        for tr in run["turns"].values():
            mine = [n for _, n in loads_of(tr)] if served(tr) else []
            if not mine:
                continue
            reading += 1
            twice += len(mine) >= 2
            loaded += sum(mine)
            used += min(tr["ssd"], sum(mine))
        summary[name] = dict(reading_turns=reading, read_twice=twice, loaded=loaded, used=used,
                             lost_share=round(1 - used / loaded, 3))
    picks = []
    for key, tr in base["turns"].items():
        twin = exact["turns"].get(key)
        if tr.get("admit") is None or tr["turn"] == 0 or not twin or twin.get("ttft") is None:
            continue
        mine = [(t, int(kv["loaded"])) for t, k, kv in tr["events"] if k == "loaded" and t <= tr["admit"] + GATED_S]
        if len(mine) == 2 and tr["ssd"] == 0 and recomputed(tr) >= 30000:
            picks.append(dict(turn=key, loaded=[n for _, n in mine], leads=[round(tr["admit"] - t, 1) for t, _ in mine],
                              admit=round(tr["admit"] - tr["entry"], 1), recomputed=recomputed(tr),
                              ahead=ahead_at(base, tr["entry"]), ttft=round(tr["ttft"], 2),
                              exact_ttft=round(twin["ttft"], 2)))
    picks.sort(key=lambda p: -p["recomputed"])
    return dict(summary=summary, picks=picks[:12])


def case_lost(runs: dict) -> dict:
    """Without pins a landed read can be evicted before its turn is admitted; the turn then recomputes it."""
    bins = [(0, 0.5), (0.5, 2), (2, 5), (5, 15), (15, 60), (60, 1e9)]
    summary = {}
    for name, run in runs.items():
        # Each turn's last read before admission, by the wait from its insertion into host memory to admission.
        tally = collections.defaultdict(lambda: [0, 0])
        for tr in run["turns"].values():
            mine = loads_of(tr) if served(tr) else []
            if not mine:
                continue
            t, tokens = mine[-1]
            b = next(b for b in bins if b[0] <= max(0.0, tr["admit"] - t) < b[1])
            tally[b][0] += 1
            tally[b][1] += tr["ssd"] < 0.5 * tokens and recomputed(tr) >= 0.5 * tokens
        summary[name] = [dict(wait=[lo, hi], reads=n, lost=k) for (lo, hi), (n, k) in sorted(tally.items())]
    base, p50, exact = runs["base"], runs["p50"], runs["exact"]
    picks = []
    for key, tr in base["turns"].items():
        mine = loads_of(tr) if served(tr) else []
        tp, te = p50["turns"].get(key), exact["turns"].get(key)
        if len(mine) != 1 or not (tp and te and served(tp) and served(te)):
            continue
        (t, tokens), theirs = mine[0], loads_of(tp)
        if (tr["ssd"] < 0.5 * tokens and recomputed(tr) >= 0.5 * tokens and tokens >= 24000 and tr["admit"] - t >= 8
                and len(theirs) == 1 and tp["ssd"] >= 0.9 * theirs[0][1] and tp["admit"] - theirs[0][0] >= 3
                and 1 <= tp["admit"] - tp["entry"] <= 8):
            picks.append(dict(turn=key, tokens=tokens, wait=round(tr["admit"] - t, 1), recomputed=recomputed(tr),
                              ttft=round(tr["ttft"], 2), p50_held=round(tp["admit"] - theirs[0][0], 1),
                              p50_queue=round(tp["admit"] - tp["entry"], 2), p50_ttft=round(tp["ttft"], 2),
                              exact_ttft=round(te["ttft"], 2)))
    picks.sort(key=lambda p: -p["recomputed"])
    return dict(summary=summary, picks=picks[:12])


def case_nextuse(runs: dict) -> dict:
    """Turns that had to read history from SSD, by where the true gap fell in the p50 hint the turn was sent with."""
    arms = ("base", "hintfree", "p50", "exact")
    groups = collections.OrderedDict((g, collections.Counter())
                                     for g in ("before p10", "p10 to p50", "p50 to p90", "past p90"))
    picks = []
    keys = shared_returning([runs[a] for a in arms])
    for key in keys:
        trs = {a: runs[a]["turns"][key] for a in arms}
        h, g = hint_of(runs["p50"], trs["p50"]), trs["p50"]["gap"]
        if h.get("p50") is None or g is None:
            continue
        group = ("before p10" if g < h["p10"] else "p10 to p50" if g < h["p50"] else
                 "p50 to p90" if g < h["p90"] else "past p90")
        read = {a: bool(ssd_reads(trs[a])) for a in arms}
        groups[group]["turns"] += 1
        groups[group].update(a for a in arms if read[a])
        kept = all(trs[a]["dev"] + trs[a]["host"] >= 0.9 * history(trs[a]) for a in ("base", "exact"))
        if g < h["p50"] and read["p50"] and not read["base"] and not read["exact"] and kept:
            tokens = ssd_reads(trs["p50"])[0]["pages"] * PAGE
            if tokens >= 16000:
                picks.append(dict(turn=key, key=h.get("reply_key"), p10=h["p10"], p50=h["p50"], gap=g, tokens=tokens,
                                  ttft=round(trs["p50"]["ttft"], 2), base_ttft=round(trs["base"]["ttft"], 2),
                                  exact_ttft=round(trs["exact"]["ttft"], 2)))
    summary = dict(groups={g: dict(turns=c["turns"], **{a: c[a] for a in arms}) for g, c in groups.items()},
                   shared_turns=len(keys),
                   turns_that_read={a: sum(1 for k in keys if ssd_reads(runs[a]["turns"][k])) for a in arms})
    picks.sort(key=lambda p: -p["tokens"])
    return dict(summary=summary, picks=picks[:12])


def gpu_steady() -> dict:
    """Minutes 40-120 at 128 users as hicache_vs_jit_gpu.png reports them, from jit_vs_hicache.py's own code."""
    import jit_vs_hicache as jv
    setting = jv.SETTINGS["c128"]
    data, shared, new, _ = jv.load(setting)
    steady, secs = slice(jv.STEADY_S // jv.TREND_S, None), jv.END_S - jv.STEADY_S
    out = dict(shared_turns=len(shared))
    for arm, name in zip(setting["arms"], ("base", "exact", "p50")):
        tr, d = jv.gpu_trends(data[arm.overview], new), data[arm.overview]
        ttft = [d["returning"][k]["ttft"] for k in shared]
        out[name] = dict(read_gb_s=round(st.mean(tr["read"][steady]), 2), write_gb_s=round(st.mean(tr["write"][steady]), 2),
                         prefill_tok_s=round(sum(tr["prefill"][steady]) / secs),
                         recomputed_share=round(sum(tr["recomputed"][steady]) / sum(tr["prefill"][steady]), 3),
                         ttft_p50=round(jv.q(ttft, 0.5), 2), ttft_p90=round(jv.q(ttft, 0.9), 1))
    return out


def case_gpu(runs: dict) -> dict:
    """128 users, 160 GB: why original HiCache recomputes history in minutes 40-120, and turns the JIT runs read instead."""
    base, p50, exact = runs["base"], runs["p50"], runs["exact"]
    causes, turns = collections.Counter(), collections.Counter()
    picks = []
    for key in shared_returning([base, p50, exact]):
        b, p, e = base["turns"][key], p50["turns"][key], exact["turns"][key]
        if not 2400 <= b["t_send"] < 7200 or recomputed(b) < PAGE:
            continue
        beyond = history(b) - b["dev"]   # what survives at admission is the shared system prompt
        first = next((int(kv["tokens"]) for _, k, kv in b["events"] if k == "prefetch_start"), None)
        cause = ("a read landed and was lost" if loads_of(b) else
                 "in memory at arrival, gone by admission" if first is not None and first < 0.5 * beyond else
                 "neither in memory nor read")
        causes[cause] += recomputed(b)
        turns[cause] += 1
        if (cause.startswith("in memory") and recomputed(b) >= 0.8 * beyond >= 25000
                and all(t["ssd"] >= 0.9 * beyond and recomputed(t) < 1000 and len(ssd_reads(t)) == 1 for t in (p, e))):
            picks.append(dict(turn=key, minute=round(b["t_send"] / 60, 1), tokens=beyond, wait=round(b["admit"] - b["entry"], 1),
                              ttft=round(b["ttft"], 1), p50_ttft=round(p["ttft"], 1), exact_ttft=round(e["ttft"], 1)))
    summary = dict(recomputed_by_cause=dict(causes), turns_by_cause=dict(turns), steady=gpu_steady())
    picks.sort(key=lambda p: (p["wait"], -p["tokens"]))
    return dict(summary=summary, picks=picks[:12])


def main() -> None:
    runs = {name: load(name) for name in RUNS}
    c128 = {name.split("_", 1)[1]: load(name) for name in RUNS_C128}
    cases = dict(late=case_late(runs), early=case_early(runs["p50"], runs["exact"]),
                 nopin=case_nopin(runs["base"], runs["hintfree"], runs["exact"]), lost=case_lost(runs),
                 nextuse=case_nextuse(runs), gpu=case_gpu(c128))
    for name, (key, arms) in EXAMPLES.items():
        cases[name]["example"] = {arm: timeline(runs[arm], key) for arm in arms}
    for name, (key, arms) in EXAMPLES_C128.items():
        cases[name]["example"] = {arm: timeline(c128[arm], key) for arm in arms}
    with open(os.path.join(OUT, "cases.json"), "w") as f:
        json.dump(cases, f, indent=1)
    for name, case in cases.items():
        print(f"===== {name}\n" + json.dumps(case["summary"], indent=1))
        for p in case["picks"][:5]:
            print(json.dumps(p))
        for side, tl in case["example"].items():
            print(f"{side}: " + json.dumps(tl))


if __name__ == "__main__":
    main()
