"""Cut a server.log down to the lines the write-path checks read, with times in seconds since the client's t_start.

  python3 -I -B extract.py <run folder> <programs folder name> <out.tsv>

Output lines: "<t_rel>\t<event>\t<rest>" where event is a HICACHE_EVT name, or REQSTAT (the ReqTimeStats line of a
finished request, rest = "rid=.. input_len=.. cached=.. output_len=.. queue_ms=.. forward_ms=.. entry=<t_rel>"),
CLEAN (an 'NIXL L3 cleanup' line, raw), PREFILL / DECODE (batch lines: running, queue, usage), WARN (lines with
'failed' or 'No such file'). Streams the log; nothing is held in memory.
"""
import calendar
import json
import re
import sys
import time

KEEP = {
    "h2s_submit", "h2s_io", "h2s_done", "h2s_group", "d2h_submit", "d2h_done", "evict_device", "evict_host",
    "prefetch_start", "s2h_query", "s2h_io", "srs_park", "srs_dispatch", "srs_pin", "srs_unpin", "srs_pin_skipped",
    "srs_write_hold", "srs_rate", "host_census", "nue_register", "ntp_plan", "ntp_arrive", "ntp_admit", "ntp_gate",
    "load_back_init", "ntp_refetch", "ntp_due", "ntp_reevict", "ntp_hold", "ec_start", "ec_release", "ec_bump",
    "ntp_stage", "ntp_room", "ntp_expire", "h2d_submit", "h2d_done", "ntp_pick",
}
TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d+)\] (.*)$")
REQSTAT = re.compile(
    r"ReqTimeStats\(rid=([^,]+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
    r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")
PREFILL = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), token usage: ([\d.]+), "
                     r"#running-req: (\d+), #queue-req: (\d+)")
DECODE = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+),.*#queue-req: (\d+)")


def main() -> None:
    run, programs, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(f"{run}/{programs}/client.jsonl") as f:
        t0 = json.loads(f.readline())["t_start"]
    n = kept = 0
    last_decode = -1e9
    with open(f"{run}/server.log", errors="replace") as f, open(out_path, "w") as out:
        out.write(f"#t0\t{t0!r}\n")
        for line in f:
            n += 1
            m = TS.match(line)
            if not m:
                continue
            y, mo, d, h, mi, s, frac, rest = m.groups()
            t = calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), int(s))) + float("0." + frac) - t0
            if rest.startswith("HICACHE_EVT "):
                parts = rest.split(" ", 2)
                ev = parts[1]
                if ev in KEEP:
                    out.write(f"{t:.3f}\t{ev}\t{parts[2] if len(parts) > 2 else ''}\n")
                    kept += 1
                continue
            if rest.startswith("ReqTimeStats("):
                r = REQSTAT.search(rest)
                if r:
                    rid, il, cl, ol, q, fw, entry = r.groups()
                    out.write(f"{t:.3f}\tREQSTAT\trid={rid} input_len={il} cached={cl} output_len={ol} queue_ms={q} "
                              f"forward_ms={fw} entry={float(entry) - t0:.3f}\n")
                    kept += 1
                continue
            if rest.startswith("Prefill batch"):
                r = PREFILL.search(rest)
                if r:
                    out.write(f"{t:.3f}\tPREFILL\tnew_seq={r[1]} new_token={r[2]} cached_token={r[3]} usage={r[4]} "
                              f"running={r[5]} queue={r[6]}\n")
                    kept += 1
                continue
            if rest.startswith("Decode batch"):
                if t - last_decode >= 1.0:   # one a second is enough for the queue and pool use over time
                    r = DECODE.search(rest)
                    if r:
                        out.write(f"{t:.3f}\tDECODE\trunning={r[1]} tokens={r[2]} usage={r[3]} queue={r[4]}\n")
                        kept += 1
                        last_decode = t
                continue
            if "NIXL L3 cleanup" in rest or "HiCacheL3Cleaner" in rest:
                out.write(f"{t:.3f}\tCLEAN\t{rest.strip()}\n")
                kept += 1
                continue
            if "No such file" in rest or "failed to retrieve" in rest or "pages failed" in rest:
                out.write(f"{t:.3f}\tWARN\t{rest.strip()[:300]}\n")
                kept += 1
    print(f"{run}: {n} lines read, {kept} kept -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    t_begin = time.time()
    main()
    print(f"{time.time() - t_begin:.1f} s", file=sys.stderr)
