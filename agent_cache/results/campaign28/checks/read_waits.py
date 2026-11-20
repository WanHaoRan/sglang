"""Campaign 28 or 29, a deadline-order arm: how long a storage read waits parked before its dispatch, how long the
read itself takes, and how far ahead of the admission it lands. On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/read_waits.py results/<server run> [<server run> ...] > results/<campaign>/checks/read_waits.out
From the server's lines: "srs_park" (a read asked, with its deadline), "srs_dispatch" (handed to the read thread,
with the wait since the park), "prefetch success" (landed) and "ntp_admit" (the request admitted, with its queue wait).
Stdlib only."""
import datetime
import re
import sys

TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
RID = re.compile(r"(?:rid|req)=(\S+)")


def stamp(line):
    return datetime.datetime.strptime(TS.match(line)[1], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def q(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))] if v else float("nan")


def row(name, values):
    print(f"  {name:58s} {len(values):6d} | mean {sum(values) / max(1, len(values)):7.1f} | p10 {q(values, .1):7.1f} | p50 {q(values, .5):7.1f} | p90 {q(values, .9):7.1f}")


for run in sys.argv[1:]:
    deadline, parked_wait, read_s, lead, queue_wait = [], [], [], [], []
    dispatched, landed = {}, {}
    with open(f"{run}/server.log", errors="replace") as f:
        for line in f:
            if " srs_park " in line:
                if (m := re.search(r"deadline_ms=(-?\d+)", line)):
                    deadline.append(int(m[1]) / 1e3)
            elif " srs_dispatch " in line:
                parked_wait.append(int(re.search(r"waited_ms=(\d+)", line)[1]) / 1e3)
                dispatched[RID.search(line)[1]] = stamp(line)
            elif "prefetch success" in line and int(re.search(r"loaded=(\d+)", line)[1]) > 0:
                rid, t = RID.search(line)[1], stamp(line)
                landed[rid] = t
                if rid in dispatched:
                    read_s.append(t - dispatched.pop(rid))
            elif " ntp_admit " in line:
                rid, t = RID.search(line)[1], stamp(line)
                queue_wait.append(int(re.search(r"wait_ms=(\d+)", line)[1]) / 1e3)
                if rid in landed:
                    lead.append(t - landed[rid])
    print(f"{run.split('/')[-1]}: seconds")
    row("deadline of a read when it is asked (srs_park)", deadline)
    row("parked before dispatch (srs_dispatch waited)", parked_wait)
    row("dispatch to landing", read_s)
    row("last landing to admission", lead)
    row("queue wait of an admitted request with a plan (ntp_admit)", queue_wait)
