"""Offline check of the list of gap distributions (replay_agentic.py --hints dist) and of the engine's rules on it; no
server needed.

  python3 -I hint_eval.py --trace TRACE [--turns 50] [--scales 70 10] [--out FILE]

Every conversation of the trace is stepped through gap_hints.SessionHinter exactly as the replay client steps its user
(the hints depend only on the conversation, not on which user replays it or when). For every gap the replay sleeps it
takes the entry the engine would pick for the recorded reply and scores it against that gap.

Part 1, the list, for each source and key:
- calibration: the share of gaps at or below the entry's p10, p50 and p90 (a calibrated list gives 10 / 50 / 90 %);
- pinball: the mean quantile loss of ln(gap) over the three quantiles (lower is better);
- by true gap: the share above p90 and the median true gap / p50;
- AUC of the sent p50 for picking out long gaps.

Part 2, the ladder rules (frozen/dist_hint_design/DIST_HINT_DESIGN_long_20261031.md 3-4) on a list, with no queue wait or arrival offset unless stated:
- where arrivals fall on the ladder; how late the late ones are; what registers SOON;
- what today's engine would do with the p50 as a point;
- how long a staged span is resident before its turn, and which clock lets a read out before arrival under a queue wait.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_hints import (ANY_KEY, END_KEY, GAP_CAP_S, GAP_FLOOR_S, KEYS, GapHistory, GapTable,  # noqa: E402
                       SessionHinter, load_reply_keys, pick_entry, recorded_sessions, scaled_gap)

# 60 s and 300 s: the next-use eviction's overdue grace and horizon (mem_cache/next_use_eviction.py NextUseStrategy)
BANDS = ((0, 60, "< 60 s"), (60, 300, "60-300 s"), (300, math.inf, ">= 300 s"))
TAUS = (0.1, 0.5, 0.9)
LEAD_S = 15.0        # the long design document, 4: dispatch_lead_s
ARRIVAL_OFFSET_S = 0.1   # the engine's seed for the arrival offset (mem_cache/next_turn_estimators.py)
QUEUE_WAITS_S = (0.0, 5.0, 10.0, 14.0, 15.0, 30.0, 45.0)
# The online list depends on when gaps finish, so it needs a run's timeline: frozen/dist_hint_design/online_table/.
GRID = [(source, key) for source in ("precalc", "session") for key in KEYS]


def late_of(hi: float) -> float:
    """The late step: today's expiry rule with p90 as the point."""
    return hi + max(60.0, hi)


def soon_seconds(entry: dict, gap: float) -> float:
    """Idle seconds the ladder keeps a session in SOON (registered next use within 300 s), up to its arrival or late."""
    total, start = 0.0, 0.0
    for registered_at in (entry["p50"], entry["p90"], late_of(entry["p90"])):
        end = min(registered_at, gap)
        total += max(0.0, end - max(start, registered_at - 300.0))
        start = registered_at
    return total


def share(n: float, total: float) -> float:
    return 100.0 * n / total if total else float("nan")


def auc(scores: list, labels: list) -> float:
    """Rank-sum AUC with ties averaged."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    return (sum(r for r, y in zip(ranks, labels) if y) - pos * (pos + 1) / 2) / (pos * neg)


def run(data: dict, keys: list, *, turns: int, scale: float, source: str, key_mode: str) -> list:
    """(picked entry or None, true gap, turn) for every gap the replay sleeps: between two of its first `turns` turns."""
    def gap_of(turn: dict) -> float:
        return scaled_gap(turn, scale=scale)
    recorded = recorded_sessions(data, keys=keys, gap_of=gap_of)
    table = GapTable(recorded, key_mode=key_mode) if source == "precalc" else None
    out = []
    for meta, conv, rec, conv_keys in zip(data["sessions"], data["conversations"], recorded, keys):
        if not conv:
            continue
        hinter = SessionHinter(conv, keys=conv_keys, gap_of=gap_of, calls=meta.get("n_turns_total"),
                               entries=None if table is None else table.entries(source=rec.source, model=rec.model,
                                                                                task=rec.task),
                               history=GapHistory(key_mode=key_mode) if source == "session" else None)
        n = len(conv[: turns or None])
        for t in range(n):
            h = hinter.step(t)
            if t + 1 < n:
                out.append((h.picked, h.true_gap_s, t))
    return out


def ln_q(entry: dict, tau: float) -> float:
    return math.log(max(entry[f"p{round(tau * 100)}"], GAP_FLOOR_S))


def summarize(rows: list) -> str:
    est = [(e, g) for e, g, _ in rows if e is not None]
    n = len(est)
    below = [share(sum(g <= e[f"p{round(t * 100)}"] or g < GAP_FLOOR_S for e, g in est), n) for t in TAUS]
    pinball = 0.0
    for e, g in est:
        y = math.log(max(g, GAP_FLOOR_S))
        for t in TAUS:
            d = y - ln_q(e, t)
            pinball += max(t * d, (t - 1) * d)
    parts = [f"{share(len(rows) - n, len(rows)):5.1f} % | {below[0]:5.1f} {below[1]:5.1f} {below[2]:5.1f} | "
             f"{pinball / (3 * n):6.3f}"]
    for lo, hi, _ in reversed(BANDS):
        sel = [(e, g) for e, g in est if lo <= g < hi]
        parts.append(f"{share(sum(g > e['p90'] for e, g in sel), len(sel)):5.1f} % "
                     f"{statistics.median(g / max(e['p50'], 1e-9) for e, g in sel):6.2f}")
    p50 = [e["p50"] for e, _ in est]
    for threshold in (300, 60):
        parts.append(f"{auc(p50, [g >= threshold for _, g in est]):.3f}")
    return " | ".join(parts)


def list_part(data: dict, keys: list, *, turns: int, scale: float, lines: list) -> dict:
    lines.append(f"\n== Part 1, gap scale x{scale:g}: the entry the engine picks for the recorded reply")
    lines.append(f"  {'source':8s} {'key':8s} | no pick | <=p10  <=p50  <=p90 | pinball | >=300 s: >p90, true/p50 | "
                 "60-300 s: >p90, true/p50 | <60 s: >p90, true/p50 | AUC >=300 s | AUC >=60 s")
    results = {}
    for source, key_mode in GRID:
        rows = run(data, keys, turns=turns, scale=scale, source=source, key_mode=key_mode)
        results[source, key_mode] = rows
        lines.append(f"  {source:8s} {key_mode:8s} | {summarize(rows)}")
    common = [r[2] > 1 for r in results[GRID[0]]]
    lines.append(f"  on the {sum(common):,d} gaps every row estimates (each session's first two have no 'session' entry):")
    for source, key_mode in GRID[:3]:
        rows = [r for r, c in zip(results[source, key_mode], common) if c]
        lines.append(f"  {source:8s} {key_mode:8s} | {summarize(rows)}")
    lines.append(f"  on each session's first two gaps ({len(common) - sum(common):,d}):")
    for source, key_mode in GRID[:3]:
        rows = [r for r, c in zip(results[source, key_mode], common) if not c]
        lines.append(f"  {source:8s} {key_mode:8s} | {summarize(rows)}")
    return results


def arrival_bands(est: list) -> str:
    def row(sel: list) -> tuple:
        n = len(sel)
        before = sum(g < e["p50"] for e, g in sel)
        to_hi = sum(e["p50"] <= g < e["p90"] for e, g in sel)
        to_late = sum(e["p90"] <= g < late_of(e["p90"]) for e, g in sel)
        return share(before, n), share(to_hi, n), share(to_late, n), share(n - before - to_hi - to_late, n)
    cells = [row(est), row([x for x in est if 60 <= x[1] < 300]), row([x for x in est if x[1] >= 300])]
    return " | ".join(" / ".join(f"{c[i]:.1f}" for c in cells) for i in range(4))


def rules_part(rows: list, *, name: str, lines: list) -> None:
    est = [(e, g) for e, g, _ in rows if e is not None]
    lines.append(f"\n  -- {name}: {len(est):,d} gaps with an entry")
    lines.append(f"  arrivals, % (all / true 60-300 s / true >= 300 s): before p50 | p50 to p90 | p90 to late | late: "
                 f"{arrival_bands(est)}")
    late = [g - late_of(e["p90"]) for e, g in est if g >= late_of(e["p90"])]
    if late:
        late.sort()
        lines.append(f"  late turns, seconds after late: median {statistics.median(late):.0f}, "
                     f"p90 {late[int(0.9 * len(late))]:.0f}, max {late[-1]:.0f}")
    for lo, hi, band in BANDS[1:]:
        sel = [(e, g) for e, g in est if lo <= g < hi]
        lines.append(f"  true {band}: {len(sel):,d} gaps; late under hi + 60 s {share(sum(g >= e['p90'] + 60 for e, g in sel), len(sel)):.1f} %; "
                     f"today's engine with the p50 as a point: OVERDUE {share(sum(g >= e['p50'] + 60 for e, g in sel), len(sel)):.1f} %, "
                     f"expired {share(sum(g >= e['p50'] + max(60.0, e['p50']) for e, g in sel), len(sel)):.1f} %")
    lines.append(f"  p50 <= 300 s, so SOON at commit: all {share(sum(e['p50'] <= 300 for e, _ in est), len(est)):.1f} %, "
                 f"of true >= 300 s {share(sum(e['p50'] <= 300 for e, g in est if g >= 300), sum(g >= 300 for _, g in est)):.1f} %")
    soon = [soon_seconds(e, g) / g for e, g in est if g >= 300 and e["p50"] <= 300]
    if soon:
        lines.append(f"  true >= 300 s registered SOON: in SOON a median {100 * statistics.median(soon):.0f} % of the idle time")
    lines.append(f"  gaps above the sent p10: {share(sum(g > e['p10'] for e, g in est), len(est)):.1f} %")
    long_est = [(e, g) for e, g in est if g >= 45]
    for tau in TAUS[:2]:
        name_q = f"p{round(tau * 100)}"
        landed = [(g - max(e[name_q] - LEAD_S, 0.0)) for e, g in long_est]
        resident = [x for x in landed if x >= 0]
        lines.append(f"  staged at {name_q} with a {LEAD_S:.0f}-s lead, gaps >= 45 s ({len(long_est):,d}): resident before the turn "
                     f"median {statistics.median(resident):.0f} s, mean {statistics.fmean(resident):.0f} s; turns before the read "
                     f"{share(len(landed) - len(resident), len(landed)):.1f} %")
    for lo, hi, band in (((45, 300, "45-300 s"), (300, math.inf, ">= 300 s"))):
        for tau in TAUS[:2]:
            name_q = f"p{round(tau * 100)}"
            staged = [(e, g) for e, g in est if lo <= g < hi and g >= max(e[name_q] - LEAD_S, 0.0)]
            lines.append(f"  reads staged at {name_q} for true {band} that serve a plan turning late: "
                         f"{share(sum(g >= late_of(e['p90']) for e, g in staged), len(staged)):.1f} %")
    for wait in QUEUE_WAITS_S:
        cells = []
        for steps in (("p50", "p90"), ("p10", "p50", "p90")):
            by_admission = by_arrival = 0
            for e, g in long_est:
                arrival = g + ARRIVAL_OFFSET_S
                previous = 0.0
                for q in (e[step] for step in steps):
                    t = max(q + ARRIVAL_OFFSET_S + wait - LEAD_S, previous)
                    if t < q:
                        by_admission += t < arrival
                        break
                    previous = q
                by_arrival += max(e[steps[0]] + ARRIVAL_OFFSET_S - LEAD_S, 0.0) < arrival
            cells.append(f"staging at {steps[0]}: lead on admission {share(by_admission, len(long_est)):5.1f} %, "
                         f"on arrival {share(by_arrival, len(long_est)):5.1f} %")
        lines.append(f"  queue wait {wait:4.0f} s, gaps >= 45 s, read out before arrival: {'; '.join(cells)}")


def facts_part(data: dict, keys: list, *, scale: float, lines: list) -> None:
    gaps = [scaled_gap(t, scale=scale) for conv in data["conversations"] for t in conv[1:]]
    long_gaps = [g for g in gaps if g >= 300]
    lines.append(f"\n  gaps >= 300 s: {len(long_gaps):,d} ({share(len(long_gaps), len(gaps)):.1f} % of gaps, "
                 f"{share(sum(long_gaps), sum(gaps)):.1f} % of idle seconds), of which at the {GAP_CAP_S:.0f}-s cap "
                 f"{sum(g >= GAP_CAP_S for g in long_gaps):,d}")


def size_part(data: dict, keys: list, *, scale: float, lines: list) -> None:
    def gap_of(turn: dict) -> float:
        return scaled_gap(turn, scale=scale)
    recorded = recorded_sessions(data, keys=keys, gap_of=gap_of)
    table = GapTable(recorded, key_mode="command")
    lens, sizes = [], []
    kinds = {"word": 0, "tool": 0, "default": 0, "none": 0}
    spike = {}
    for rec, conv_keys in zip(recorded, keys):
        if rec is None:
            continue
        entries = table.entries(source=rec.source, model=rec.model, task=rec.task)
        by_key = {e["key"] for e in entries}
        by_entry = {e["key"]: e for e in entries}
        counts = spike.setdefault(rec.model, [0, 0])
        for key in conv_keys[:-1]:
            e = pick_entry(by_entry, key)
            counts[0] += e["p10"] < e["p50"] / 5
            counts[1] += 1
        lens.append(len(entries))
        sizes.append(len(json.dumps(entries)))
        for key in conv_keys[:-1]:
            if key in by_key:
                kinds["word" if "/" in key else "tool"] += 1
            elif key is not None and key.split("/", 1)[0] in by_key:
                kinds["tool"] += 1
            elif ANY_KEY in by_key:
                kinds["default"] += 1
            else:
                kinds["none"] += 1
    lens.sort()
    sizes.sort()
    total = sum(kinds.values())
    lines.append(f"\n== The list on the wire (precalc, command, x{scale:g}): entries median {lens[len(lens) // 2]}, max {lens[-1]}; "
                 f"bytes median {sizes[len(sizes) // 2]:,d}, max {sizes[-1]:,d}")
    lines.append(f"  the engine's pick: tool + first word {share(kinds['word'], total):.1f} %, tool {share(kinds['tool'], total):.1f} %, "
                 f"default {share(kinds['default'], total):.2f} %, nothing {share(kinds['none'], total):.2f} %")
    lines.append("  gaps whose picked entry has p10 < p50 / 5, by model: "
                 + ", ".join(f"{m} {share(n, t):.1f} % of {t:,d}" for m, (n, t) in sorted(spike.items())))


def end_part(data: dict, keys: list, *, lines: list) -> None:
    def gap_of(turn: dict) -> float:
        return float(turn.get("pre_gap") or 0.0)
    line = []
    for end in ("explicit", "natural", "none"):
        ended = 0
        for meta, conv, conv_keys in zip(data["sessions"], data["conversations"], keys):
            if conv:
                hinter = SessionHinter(conv, keys=conv_keys, gap_of=gap_of, end=end, calls=meta.get("n_turns_total"))
                ended += [hinter.step(t) for t in range(len(conv))][-1].reply_key == END_KEY
        line.append(f"{end} {ended}")
    lines.append(f"\n== Sessions that say they ended at their last recorded turn: {', '.join(line)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", required=True)
    ap.add_argument("--turns", type=int, default=50, help="per conversation, as the replay's --turns (0 = all)")
    ap.add_argument("--scales", type=float, nargs="+", default=[70.0, 10.0])
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args()
    with open(args.trace) as f:
        data = json.load(f)
    keys = load_reply_keys(args.trace, data["conversations"])
    lines = [f"# {os.path.basename(args.trace)}: {sum(1 for c in data['conversations'] if c)} conversations, first "
             f"{args.turns or 'all'} turns each; gaps x scale, at most {GAP_CAP_S:.0f} s",
             "# precalc: quantiles from other tasks' sessions by the same harness and model; session: a Gaussian of "
             "ln(gap) on the session's own past gaps"]
    for scale in args.scales:
        results = list_part(data, keys, turns=args.turns, scale=scale, lines=lines)
        lines.append(f"\n== Part 2, gap scale x{scale:g}: the rules")
        rules_part(results["precalc", "command"], name="precalc, command (the default list)", lines=lines)
        rules_part(results["session", "none"], name="session, none (no tool types)", lines=lines)
        facts_part(data, keys, scale=scale, lines=lines)
    size_part(data, keys, scale=args.scales[0], lines=lines)
    end_part(data, keys, lines=lines)
    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
