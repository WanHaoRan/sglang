"""Verifier's rough count (read-only): how many of ec_full's soft-pin bumps a due-time guard would have stopped.

From agent_cache/:  nice -n 10 python3 -I -B <this file> <run folder name>
Reads results/<run>/server.log only. Not one of the lead's checks; numbers are a first estimate.

For each ec_bump (a staged read given up for the read of an arrived turn):
  t        time of the bump
  arr_b    when the bumped turn arrived (ntp_arrive)          adm_b   when it was admitted (ntp_admit)
  arr_f    when the turn the room went to arrived            adm_f   when it was admitted
  chance   the engine's own forecast that the bumped turn is back within its re-read time
  r_b      re-read time of the bumped span: pages * MS_PER_PAGE (+ 1 s margin, as the engine's horizon)
"""
import collections
import datetime as dt
import re
import statistics as st
import sys

MS_PER_PAGE = 2.34  # DECISIONS.md, ec_full entry: SSD 2.34 ms per page
PAGE = 64           # ntp_stage tokens=512 with s2h_io pages=8 in the same log
MARGIN_S = 1.0      # NTP_JIT_MARGIN_S in _read_seconds

EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_bump|ntp_arrive|ntp_admit) rid=\S+?-c(\d+)-t(\d+)(.*)")
FOR = re.compile(r"for_rid=\S+?-c(\d+)-t(\d+) tokens=(\d+) chance=([\d.]+)")


def stamp(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def three(v):
    return "%.2f %.2f %.2f" % (q(v, .1), q(v, .5), q(v, .9))


run = sys.argv[1]
bumps, arrive, admit = [], {}, {}
for line in open("results/%s/server.log" % run, errors="replace"):
    m = EVT.match(line)
    if not m:
        continue
    t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
    if kind == "ec_bump":
        f = FOR.search(m.group(5))
        bumps.append((t, key, (int(f.group(1)), int(f.group(2))), int(f.group(3)), float(f.group(4))))
    elif kind == "ntp_arrive":
        arrive.setdefault(key, t)
    else:
        admit.setdefault(key, t)

print("run", run)
print("bumps", len(bumps), "| bumped turns with an arrival", sum(b[1] in arrive for b in bumps),
      "| helped turns with an admission", sum(b[2] in admit for b in bumps))

# How far off was the helped turn's own admission when it took the room?
lead_f = [admit[b[2]] - b[0] for b in bumps if b[2] in admit]
wait_f = [b[0] - arrive[b[2]] for b in bumps if b[2] in arrive]
print("helped turn: bump to its admission p10 p50 p90 %s s; over 2 s %d, over 5 s %d, over 10 s %d of %d"
      % (three(lead_f), sum(x > 2 for x in lead_f), sum(x > 5 for x in lead_f), sum(x > 10 for x in lead_f), len(lead_f)))
print("helped turn: its arrival to the bump p10 p50 p90 %s s" % three(wait_f))

# A. Hindsight, the deck's strict form: the bumped history was needed sooner than the turn it made room for.
both = [b for b in bumps if b[1] in arrive and b[2] in admit]
a_arr = [b for b in both if arrive[b[1]] < admit[b[2]]]
a_adm = [b for b in both if b[1] in admit and admit[b[1]] < admit[b[2]]]
print("A. bumped turn ARRIVED before the helped turn was admitted: %d of %d (tokens %d of %d)"
      % (len(a_arr), len(both), sum(b[3] for b in a_arr), sum(b[3] for b in both)))
print("A'. bumped turn was ADMITTED before the helped turn was admitted: %d of %d" % (len(a_adm), len(both)))

# B. Hindsight, the toy's guard: the bumped turn came back before its own re-read could land.
have = [b for b in bumps if b[1] in arrive]
gap = lambda b: arrive[b[1]] - b[0]
r_io = lambda b: b[3] / PAGE * MS_PER_PAGE / 1000.0
b_io = [b for b in have if gap(b) < r_io(b)]
b_mg = [b for b in have if gap(b) < r_io(b) + MARGIN_S]
print("B. bumped turn back within its re-read time: I/O only %d, with the 1 s margin %d of %d; re-read I/O p10 p50 p90 %s s"
      % (len(b_io), len(b_mg), len(have), three([r_io(b) for b in have])))
print("   bumped turn had already arrived at the bump (gap <= 0): %d; within 0.25 s after: %d"
      % (sum(gap(b) <= 0 for b in have), sum(0 < gap(b) <= 0.25 for b in have)))

# C. What the engine knew at the bump: its own chance of return within the re-read time.
for thr in (0.25, 0.5, 0.75):
    sel = [b for b in bumps if b[4] >= thr]
    back = [b for b in sel if b[1] in arrive and gap(b) < r_io(b) + MARGIN_S]
    print("C. chance >= %.2f: %d of %d bumps (%d tokens); of them back within re-read + margin: %d"
          % (thr, len(sel), len(bumps), sum(b[3] for b in sel), len(back)))
lo = [b for b in have if b[4] < 0.25]
print("   chance < 0.25: %d bumps, of them back within re-read + margin: %d"
      % (len(lo), sum(gap(b) < r_io(b) + MARGIN_S for b in lo)))

# D. Both tests of a strict rule together, in hindsight: needed sooner AND not just a short re-read away.
d = [b for b in both if arrive[b[1]] < admit[b[2]] and gap(b) < r_io(b) + MARGIN_S]
print("D. arrived before the helped turn's admission and within its own re-read + margin: %d of %d" % (len(d), len(both)))
per_for = collections.Counter(b[2] for b in bumps)
print("helped turns: %d, bumps per helped turn p50 p90 max %d %d %d"
      % (len(per_for), q(list(per_for.values()), .5), q(list(per_for.values()), .9), max(per_for.values())))
