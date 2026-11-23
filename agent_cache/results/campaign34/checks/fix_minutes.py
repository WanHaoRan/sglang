"""Campaign 34: the refused-read loop minute by minute, with and without the engine fixes (OPEN_ITEMS.md section 7).
Stdlib only. On the host, from agent_cache/results:
  python3 -I -B campaign34/checks/fix_minutes.py c32_jit_p50:campaign32:jit_p50 c34_jit_p50:campaign34:jit_p50 ...
  python3 -I -B campaign34/checks/fix_minutes.py --table c34_jit_p50:campaign34:jit_p50
An arm is <name>:<campaign folder>:<label in its manifest.txt>, or <name>:<run folder>:<programs folder> for an arm
still running. Minutes are client minutes (server log time minus the client's t_start).

Per minute: returning requests sent and their mean TTFT; history computed again (ideal_cached - cached, K tokens);
storage reads dispatched and refused for host room (srs_dispatch launched=0); ntp_room marks and gate retries (fix 4);
K tokens submitted for SSD writes, pages skipped as already stored (fix 1), merged groups (fix 2); pending write tokens
at their peak and the median submit-to-done latency of the minute's writes; write-hold switches (fix 3); and the last
host census of the minute (K tokens: free, host-only, held by pending writes, pins, reads, ancestors of held nodes).
Onset: the first minute of 3 (and of 5) in a row with 5 or more refused dispatches."""

import collections
import datetime as dt
import json
import os
import re
import statistics
import sys

LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\] HICACHE_EVT (\w+) (.*)$")
FIELD = re.compile(r"(\w+)=([\w./-]+)")
CENSUS = ("free", "host_only", "h_write", "h_pin", "h_read", "h_lookup", "h_parked", "anc", "pending_write")
WINDOWS = ((30, 60), (60, 90), (90, 110))


def arm_paths(spec):
    name, where, what = spec.split(":")
    folder = os.path.join(where)
    manifest = os.path.join(folder, "manifest.txt")
    if os.path.exists(manifest):
        for line in open(manifest):
            parts = line.split()
            if parts and parts[0] == what:
                return name, parts[1], parts[2]
        raise SystemExit(f"{spec}: label {what} not in {manifest}")
    return name, where, what


def client(run, prog):
    with open(os.path.join(run, prog, "client.jsonl")) as f:
        t0 = json.loads(f.readline())["t_start"]
        reqs = [r for r in map(json.loads, f) if r.get("kind") == "req" and r.get("ttft") is not None]
    return t0, reqs


def cached(r):
    details = r.get("cached_details") or {}
    if details:
        return sum(int(details.get(k) or 0) for k in ("device", "host", "storage"))
    return int(r.get("cached_tokens") or 0)


def scan(run, t0):
    per = collections.defaultdict(collections.Counter)
    census = {}
    submitted, latencies = {}, collections.defaultdict(list)
    pending_peak = collections.defaultdict(int)
    pending = 0
    has_pending_field = None
    with open(os.path.join(run, "server.log"), errors="replace") as f:
        for line in f:
            if "HICACHE_EVT" not in line:
                continue
            m = LINE.match(line)
            if not m:
                continue
            t = dt.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() - t0
            if t < 0:
                continue
            minute = int(t // 60)
            kind, fields = m[2], dict(FIELD.findall(m[3]))
            c = per[minute]
            if kind == "srs_dispatch":
                c["disp"] += 1
                c["ref"] += fields.get("launched") == "0"
            elif kind == "ntp_room":
                c["room"] += 1
            elif kind == "ntp_gate":
                c["gate_" + fields.get("decision", "?")] += 1
            elif kind == "ntp_reevict" and fields.get("capped") == "1":
                c["reevict_capped"] += 1
            elif kind == "h2s_submit":
                tokens = int(fields["tokens"])
                c["wr_tok"] += tokens
                submitted[fields["op"]] = (t, minute)
                if has_pending_field is None:
                    has_pending_field = "pending" in fields
                pending = int(fields["pending"]) if has_pending_field else pending + tokens
                pending_peak[minute] = max(pending_peak[minute], pending)
            elif kind == "h2s_done":
                start = submitted.pop(fields["op"], None)
                if start is not None:
                    latencies[start[1]].append(t - start[0])
                pending = int(fields["pending"]) if has_pending_field else pending - int(fields["tokens"])
                pending_peak[minute] = max(pending_peak[minute], pending)
            elif kind == "h2s_io":
                c["wr_pages"] += int(fields["pages"])
                c["skip_pages"] += int(fields.get("skipped", 0))
            elif kind == "h2s_group":
                c["groups"] += 1
                c["skip_pages"] += int(fields["skipped"])
            elif kind == "s2h_io":
                c["rd_tok"] += int(fields["tokens"])
            elif kind == "srs_write_hold":
                c["hold_" + fields["state"]] += 1
            elif kind == "host_census" and fields.get("why") == "tick":
                census[minute] = {k: int(fields[k]) for k in CENSUS if k in fields}
    return per, census, pending_peak, latencies


def series(spec):
    name, run, prog = arm_paths(spec)
    t0, reqs = client(run, prog)
    per, census, pending_peak, latencies = scan(run, t0)
    for r in reqs:
        if r["k"] > 0:
            c = per[int(r["t_send"] // 60)]
            c["ret"] += 1
            c["ret_ttft_ms"] += round(1000 * r["ttft"])
            c["lost_tok"] += max(0, int(r.get("ideal_cached") or 0) - cached(r))
    rows = {}
    for minute in range(0, 121):
        c = per.get(minute, collections.Counter())
        lat = latencies.get(minute)
        rows[minute] = dict(
            c,
            pend_peak=pending_peak.get(minute, 0),
            wlat=statistics.median(lat) if lat else None,
            census=census.get(minute),
        )
    return name, rows


def onset(rows, run_length):
    for minute in range(0, 111):
        if all(rows[m].get("ref", 0) >= 5 for m in range(minute, minute + run_length)):
            return minute
    return None


def summary(name, rows):
    out = [f"{name}: loop onset (5+ refused a minute) 3 min {onset(rows, 3)}, 5 min {onset(rows, 5)}"]
    for lo, hi in WINDOWS:
        tot = collections.Counter()
        peak, lats = 0, []
        for m in range(lo, hi):
            tot.update({k: v for k, v in rows[m].items() if isinstance(v, int)})
            peak = max(peak, rows[m]["pend_peak"])
            if rows[m]["wlat"] is not None:
                lats.append(rows[m]["wlat"])
        ret = tot["ret"] or 1
        gates = {k[5:]: v for k, v in sorted(tot.items()) if k.startswith("gate_")}
        out.append(
            f"  minutes {lo}-{hi}: returning {tot['ret']} TTFT {tot['ret_ttft_ms'] / 1000 / ret:.2f} s,"
            f" history again {tot['lost_tok'] / 1e6:.2f} M ({tot['lost_tok'] / ret / 1000:.2f} K a request);"
            f" dispatched {tot['disp']}, refused {tot['ref']}, ntp_room {tot['room']},"
            f" reevict capped {tot['reevict_capped']}, gate {gates};"
            f" written {tot['wr_tok'] / 1e6:.1f} M tokens, pages skipped {tot['skip_pages']} of"
            f" {tot['wr_pages'] + tot['skip_pages']} ({100 * tot['skip_pages'] / max(1, tot['wr_pages'] + tot['skip_pages']):.0f} %),"
            f" groups {tot['groups']}; read {tot['rd_tok'] / 1e6:.1f} M; pending peak {peak / 1e3:.0f} K,"
            f" write latency median of minutes {statistics.median(lats) if lats else float('nan'):.1f} s;"
            f" hold on {tot['hold_on']}"
        )
    return "\n".join(out)


def table(name, rows):
    head = (
        f"{name}\nmin | ret ttft  | again K | disp ref room retry cap | wr K  skip%  grp | pend K  wlat s | hold"
        " | census K: free host_only write pin read anc"
    )
    lines = [head]
    for m in range(0, 121):
        r = rows[m]
        if not r.get("ret") and not r.get("wr_tok"):
            continue
        ret = r.get("ret", 0)
        pages = r.get("wr_pages", 0) + r.get("skip_pages", 0)
        cen = r["census"]
        cen_s = (
            " ".join(f"{cen.get(k, 0) / 1e3:5.0f}" for k in ("free", "host_only", "h_write", "h_pin", "h_read", "anc"))
            if cen
            else "-"
        )
        lines.append(
            f"{m:3d} | {ret:3d} {r.get('ret_ttft_ms', 0) / 1000 / max(1, ret):5.2f} | {r.get('lost_tok', 0) / 1e3:7.0f} |"
            f" {r.get('disp', 0):4d} {r.get('ref', 0):3d} {r.get('room', 0):4d} {r.get('gate_retry', 0):5d}"
            f" {r.get('gate_capped', 0) + r.get('reevict_capped', 0):3d} |"
            f" {r.get('wr_tok', 0) / 1e3:5.0f} {100 * r.get('skip_pages', 0) / max(1, pages):5.1f} {r.get('groups', 0):4d} |"
            f" {r['pend_peak'] / 1e3:6.0f} {r['wlat'] if r['wlat'] is not None else float('nan'):6.1f} |"
            f" {r.get('hold_on', 0)}/{r.get('hold_off', 0)} | {cen_s}"
        )
    return "\n".join(lines)


def main():
    args = sys.argv[1:]
    show_table = "--table" in args
    for spec in (a for a in args if a != "--table"):
        name, rows = series(spec)
        print(summary(name, rows))
        if show_table:
            print(table(name, rows))
        print()


if __name__ == "__main__":
    main()
