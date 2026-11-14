"""Summarize the Oracle v2 events of one server.log (agent_cache/ORACLE_V2_PLAN.md 4.6).

  python3 srs_summary.py --run results/<boot dir> [--window START END]

Counts every HICACHE_EVT kind the v2 levers emit (srs_park, srs_dispatch, srs_pin, srs_pin_skipped, srs_unpin, srs_rate,
nue_register) next to the read events they must agree with (s2h_query hits, s2h_io), and prints the distributions the
go/no-go table of ORACLE_V2_PLAN.md section 6 reads: parked wait before dispatch, pages in flight at dispatch, pinned
tokens, unpin reasons, and the I/O thread's ms-per-page rate. Times are seconds since the client's t_start when a
client directory is present, else since the first log line; --window keeps only events inside [START, END].
"""
import argparse
import collections
import datetime as dt
import json
import os
import re

KV_RE = re.compile(r"(\w+)=(-?[\w./-]+)")
TS_RE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3})\]")
V2_KINDS = ("srs_park", "srs_dispatch", "srs_pin", "srs_pin_skipped", "srs_unpin", "srs_rate", "nue_register")


def parse_ts(line: str) -> float:
    return (
        dt.datetime.strptime(TS_RE.match(line)[1], "%Y-%m-%d %H:%M:%S.%f")
        .replace(tzinfo=dt.timezone.utc)
        .timestamp()
    )


def t_start_of(run: str) -> float | None:
    clients = sorted(d for d in os.listdir(run) if d.startswith("client_"))
    if not clients:
        return None
    with open(os.path.join(run, clients[-1], "client.jsonl")) as f:
        return next(json.loads(l) for l in f if '"kind": "run"' in l)["t_start"]


def quantiles(values: list, qs=(0.5, 0.9, 0.99)) -> str:
    if not values:
        return "n/a"
    values = sorted(values)
    return " ".join(f"p{int(q * 100)}={values[min(len(values) - 1, int(q * len(values)))]:.0f}" for q in qs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--window", type=float, nargs=2, default=None, metavar=("START", "END"))
    a = ap.parse_args()
    t0 = t_start_of(a.run)
    counts = collections.Counter()
    waited, inflight, pinned, rates, unpin_reasons = [], [], [], [], collections.Counter()
    deadlines_now = 0
    launched_fail = 0
    for line in open(os.path.join(a.run, "server.log"), errors="replace"):
        if "HICACHE_EVT" not in line:
            continue
        if t0 is None:
            t0 = parse_ts(line)
        t = parse_ts(line) - t0
        if a.window and not (a.window[0] <= t <= a.window[1]):
            continue
        kind = line.split("HICACHE_EVT ", 1)[1].split()[0]
        kv = dict(KV_RE.findall(line))
        if kind == "s2h_query":
            counts["s2h_query hit" if int(kv.get("storage_hit", 0)) > 0 else "s2h_query miss"] += 1
            continue
        counts[kind] += 1
        if kind == "srs_park" and kv.get("deadline_ms") == "now":
            deadlines_now += 1
        elif kind == "srs_dispatch":
            waited.append(float(kv["waited_ms"]))
            inflight.append(float(kv["inflight_pages"]))
            pinned.append(float(kv["pinned_tokens"]))
            launched_fail += kv.get("launched") == "0"
        elif kind == "srs_unpin":
            unpin_reasons[kv.get("reason", "?")] += 1
        elif kind == "srs_rate":
            rates.append(float(kv["ms_per_page"]))
    print(f"run {a.run}" + (f", window {a.window[0]:.0f}-{a.window[1]:.0f} s" if a.window else ""))
    for kind in ("s2h_query hit", "s2h_query miss", "s2h_io") + V2_KINDS:
        print(f"  {kind:<16} {counts.get(kind, 0):>8,}")
    print(f"  srs_park with deadline=now: {deadlines_now:,} of {counts.get('srs_park', 0):,}")
    print(f"  srs_dispatch launched=0:    {launched_fail:,}")
    print(f"  waited_ms at dispatch:      {quantiles(waited)}")
    print(f"  inflight_pages at dispatch: {quantiles(inflight)}")
    print(f"  pinned_tokens at dispatch:  {quantiles(pinned)}")
    print(f"  unpin reasons:              {dict(unpin_reasons) or 'none'}")
    print(f"  ms_per_page (srs_rate):     {quantiles(rates, (0.1, 0.5, 0.9))}")
    agree = counts.get("srs_dispatch", 0) == counts.get("s2h_io", 0)
    print(f"  dispatch count equals s2h_io count: {'yes' if agree else 'NO'} "
          f"(a FIFO-order run has 0 dispatches; under deadline order every read is dispatched once)")


if __name__ == "__main__":
    main()
