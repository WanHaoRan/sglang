"""Campaign 28 or 29, an arm with next-turn reads: what became of the SSD reads issued for requests in the queue.
On the host, from agent_cache/:
  python3 -I -B results/campaign28/checks/reread_fate.py results/<server run> <programs dir> [from_s to_s] > results/<campaign>/checks/reread_fate_<label>.out
Over the returning requests sent in [from_s, to_s) (default 1,800 to 6,600 s), by the history delivered at admission
(nothing, part, all): the reads that landed for the request, whether its last read got a pin in the host pool
("srs_pin") or found the pin budget full ("srs_pin_skipped"), whether the engine saw the pages evicted again and
stopped re-reading ("ntp_reevict ... capped=1"), and the time from the last read's landing to the admission.
Then the pins of the whole run, how long one was held, and how soon a read that landed was found gone again
(only when at least 90 % of what it loaded is missing: a tail the SSD never had also logs "ntp_reevict").
Stdlib only."""
import collections
import datetime
import json
import re
import sys

run, programs = sys.argv[1], sys.argv[2]
lo, hi = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (1800.0, 6600.0)
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)\]")
RID = re.compile(r"(?:rid|req)=(\S+)")


def stamp(line):
    m = TS.match(line)
    return datetime.datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S.%f").timestamp() if m else None


arrive, admit, due, landed, last_pin = {}, {}, {}, collections.defaultdict(list), {}
reevicts, capped, gate = collections.Counter(), set(), collections.defaultdict(collections.Counter)
pins, skips, pin_at, held, unpin_reasons = 0, 0, {}, [], collections.Counter()
lost_after, tail_only, loaded_tokens, last_loaded = [], 0, 0, {}
budget = None
with open(f"{run}/server.log", errors="replace") as f:
    for line in f:
        if "HICACHE_EVT" not in line and "prefetch success" not in line:
            continue
        m = RID.search(line)
        if not m:
            continue
        rid, t = m[1], stamp(line)
        if " ntp_arrive " in line:
            arrive[rid] = t
        elif " ntp_admit " in line:
            admit[rid] = t
        elif " ntp_due " in line:
            due.setdefault(rid, (t, int(re.search(r"case=(\d)", line)[1])))
        elif "prefetch success" in line:
            if (n := int(re.search(r"loaded=(\d+)", line)[1])) > 0:
                landed[rid].append(t)
                loaded_tokens += n
                last_loaded[rid] = n
        elif " srs_pin_skipped " in line:
            skips += 1
            last_pin[rid] = "skipped"
            budget = int(re.search(r"budget=(\d+)", line)[1])
        elif " srs_pin " in line:
            pins += 1
            last_pin[rid] = "pinned"
            pin_at[rid] = t
        elif " srs_unpin " in line:
            reason = re.search(r"reason=(\w+)", line)[1]
            unpin_reasons[reason] += 1
            if rid in pin_at:
                held.append(t - pin_at.pop(rid))
        elif " ntp_reevict " in line:
            reevicts[rid] += 1
            if landed[rid]:
                if int(re.search(r"evicted=(\d+)", line)[1]) >= 0.9 * last_loaded[rid]:
                    lost_after.append(t - landed[rid][-1])
                else:
                    tail_only += 1
            if "capped=1" in line:
                capped.add(rid)
        elif " ntp_gate " in line:
            gate[rid][re.search(r"decision=(\w+)", line)[1]] += 1
with open(f"{run}/{programs}/client.jsonl") as f:
    reqs = [json.loads(line) for line in f if '"kind":"req"' in line]


def cached(r):
    d = r.get("cached_details") or {}
    return sum(int(d.get(k) or 0) for k in ("device", "host", "storage")) if d else int(r.get("cached_tokens") or 0)


def med(values):
    v = sorted(values)
    return v[len(v) // 2] if v else float("nan")


back = [r for r in reqs if lo <= r["t_send"] < hi and r["k"] > 0 and r["ideal_cached"] > 0 and r.get("ttft") is not None]
groups = (("nothing (under 2 %)", lambda s: s < 0.02), ("part", lambda s: 0.02 <= s < 0.95), ("all (95 % or more)", lambda s: s >= 0.95))
print(f"{run.split('/')[-1]}: {len(back)} returning requests sent in [{lo:.0f}, {hi:.0f}) s")
print(f"{'history delivered':20s} {'requests':>8s} {'TTFT s':>7s} | {'with a read':>11s} {'reads each':>10s} | last read: {'pinned':>6s} {'no room to pin':>14s} | "
      f"{'evicted again':>13s} {'re-reads stopped':>16s} | {'arrival to first read due, s':>28s} | {'last landing to admission, s (median, p90)':>42s}")
for name, test in groups:
    rs = [r for r in back if test(cached(r) / r["ideal_cached"])]
    if not rs:
        continue
    ids = [r["rid"] for r in rs]
    read = [i for i in ids if landed[i]]
    lead = sorted(admit[i] - landed[i][-1] for i in read if i in admit)
    first = [due[i][0] - arrive[i] for i in ids if i in due and i in arrive]
    print(f"{name:20s} {len(rs):8d} {sum(r['ttft'] for r in rs) / len(rs):7.1f} | {len(read):11d} {sum(len(landed[i]) for i in read) / max(1, len(read)):10.2f} | "
          f"{'':10s} {sum(last_pin.get(i) == 'pinned' for i in ids):6d} {sum(last_pin.get(i) == 'skipped' for i in ids):14d} | "
          f"{sum(reevicts[i] > 0 for i in ids):13d} {sum(i in capped for i in ids):16d} | {med(first):28.1f} | "
          f"{med(lead):30.1f} {lead[int(0.9 * (len(lead) - 1))] if lead else float('nan'):11.1f}")
gates = collections.Counter()
for r in back:
    gates.update(gate[r["rid"]])
if gates:
    print(f"arrival gate decisions of these requests: {dict(gates)}")
print(f"whole run: {pins} reads pinned, {skips} found no room to pin (budget {budget if budget is not None else 'not logged'} tokens); "
      f"a pin was held {med(held):.1f} s at the median, {sum(held) / max(1, len(held)):.1f} s on average; released on {dict(unpin_reasons)}")
lost_after.sort()
print(f"whole run: reads loaded {loaded_tokens / 1e6:.1f} M tokens into the host pool; {len(lost_after)} times a read that had landed was found gone again, "
      f"{med(lost_after):.1f} s after it landed at the median (p10 {lost_after[int(0.1 * (len(lost_after) - 1))] if lost_after else float('nan'):.1f}, "
      f"p90 {lost_after[int(0.9 * (len(lost_after) - 1))] if lost_after else float('nan'):.1f}); {tail_only} times only a tail was missing")
