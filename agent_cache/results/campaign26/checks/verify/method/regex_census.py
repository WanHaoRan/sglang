"""Regex coverage: for each run's server.log, count the lines that contain each event by plain substring and the
lines the campaign's regexes (copied verbatim from compare_ec.py, stall_ec.py, checks/holds.py, checks/predwait.py)
match under the same control flow as the loaders. Prints every mismatch with sample lines.
  nice -n 19 python3 -I -B regex_census.py"""
import collections, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, HERE)
RUNS = {
    "baseline": "20261101_082229_three_tier_wc_custom_L1nat_H70",
    "base_rep": "20261105_075438_three_tier_wc_custom_L1nat_H70",
    "hintfree": "20261101_122705_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "hintfree_ctl": "20261111_045808_three_tier_wc_dl_poll0_custom_L1nat_H70",
    "jit_p50": "20261101_222451_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_ctl": "20261111_024927_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_p50_d_cut": "20261111_151101_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "jit_exact": "20261101_202142_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r13": "20261111_070012_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "ec_r1": "20261111_110558_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "nofc": "20261111_090311_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
    "nofc_noend": "20261111_130801_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
}
CURRENT = open(os.path.join(RES, ".current_run")).read().strip()
# verbatim from compare_ec.py
TS = re.compile(r"^\[(.{23})\] ")
RID = r"rid=\S+?-c(\d+)-t(\d+)"
RTS = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), .*queue_duration=([\d.]+)ms")
DEC = re.compile(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), .*gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)")
PRE = re.compile(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")
IO = re.compile(r"HICACHE_EVT s2h_io " + RID + r" pages=(\d+) tokens=\d+ ms=(\d+)")
DISP = re.compile(r"HICACHE_EVT srs_dispatch " + RID + r" pages=(\d+) .*pinned_tokens=(\d+) launched=(\d)")
STAGE = re.compile(r"HICACHE_EVT ntp_stage " + RID)
ECS = re.compile(r"HICACHE_EVT ec_start " + RID + r" decision=(\w+)")
ECR = re.compile(r"HICACHE_EVT (ec_release|ec_bump) " + RID + r" .*?tokens=(\d+)")
ECP = re.compile(r"HICACHE_EVT ec_price price=(\S+) refused=(\d+) idle=(\d+)")
UNPIN = re.compile(r"HICACHE_EVT srs_unpin " + RID + r" tokens=(\d+) reason=(\w+) pinned=(\d+)")
PINNED = re.compile(r"HICACHE_EVT srs_(?:pin|unpin) .*pinned=(\d+)")
START = re.compile(r"HICACHE_EVT prefetch_start " + RID)
PIN = re.compile(r"HICACHE_EVT srs_pin " + RID + r" tokens=(\d+)")
EVICT = re.compile(r"HICACHE_EVT (evict_device|evict_host) tokens=(\d+)")
SUCC = re.compile(r"HiCache prefetch success req=\S+?-c(\d+)-t(\d+) completed=\d+ matched=\d+ loaded=(\d+)")
SHR = re.compile(r"device prefix shrank before admission req=\S+?-c(\d+)-t(\d+) lookup_match=(\d+) current_match=(\d+)")
CAP = re.compile(r"reissue cap reached req=\S+?-c(\d+)-t(\d+)")
# verbatim from stall_ec.py / holds.py / predwait.py
ENTRY = re.compile(r"ReqTimeStats\(rid=\S+?-c(\d+)-t(\d+), .*queue_duration=([\d.]+)ms.*entry_time=([\d.]+)")
IO2 = re.compile(r"HICACHE_EVT s2h_io rid=\S+?-c(\d+)-t(\d+) pages=(\d+) tokens=\d+ ms=(\d+)")
EV = re.compile(r"HICACHE_EVT (srs_park|srs_dispatch|s2h_io) rid=\S+?-c(\d+)-t(\d+) (.*)")
ARR = re.compile(r"HICACHE_EVT ntp_arrive rid=\S+?-c(\d+)-t(\d+) .*pos=(\d+) qlen=(\d+)")
ADM = re.compile(r"HICACHE_EVT ntp_admit rid=\S+?-c(\d+)-t(\d+) wait_ms=(\d+) pred_wait_ql_ms=(\S+) ")
DISP2 = re.compile(r"HICACHE_EVT srs_dispatch rid=\S+?-c(\d+)-t(\d+) pages=\d+ waited_ms=(\d+) .*launched=(\d)")

SUBSTR = {  # what a plain grep would count -> the loader's regex for it, tried the way the loader tries it
    "ReqTimeStats(": ("startswith:ReqTimeStats", RTS), "ReqTimeStats( [stall]": ("startswith:ReqTimeStats", ENTRY),
    "Decode batch": ("startswith:Decode batch", DEC), "Prefill batch": ("startswith:Prefill batch", PRE),
    "HICACHE_EVT s2h_io ": ("startswith:HICACHE_EVT", IO), "HICACHE_EVT s2h_io  [stall]": ("startswith:HICACHE_EVT s2h_io", IO2),
    "HICACHE_EVT prefetch_start ": ("startswith:HICACHE_EVT", START), "HICACHE_EVT srs_dispatch ": ("startswith:HICACHE_EVT", DISP),
    "HICACHE_EVT ntp_stage ": ("startswith:HICACHE_EVT", STAGE), "HICACHE_EVT ec_start ": ("startswith:HICACHE_EVT", ECS),
    "HICACHE_EVT ec_release ": ("startswith:HICACHE_EVT", ECR), "HICACHE_EVT ec_bump ": ("startswith:HICACHE_EVT", ECR),
    "HICACHE_EVT ec_price ": ("startswith:HICACHE_EVT", ECP), "HICACHE_EVT srs_unpin ": ("startswith:HICACHE_EVT", UNPIN),
    "HICACHE_EVT srs_pin ": ("startswith:HICACHE_EVT", PIN), "HICACHE_EVT evict_device ": ("startswith:HICACHE_EVT", EVICT),
    "HICACHE_EVT evict_host ": ("startswith:HICACHE_EVT", EVICT), "HiCache prefetch success": ("startswith:HiCache prefetch success", SUCC),
    "device prefix shrank": ("search", SHR), "reissue cap reached": ("search", CAP),
    "HICACHE_EVT srs_park ": ("startswith:HICACHE_EVT s", EV), "HICACHE_EVT srs_dispatch  [holds]": ("startswith:HICACHE_EVT s", EV),
    "HICACHE_EVT s2h_io  [holds]": ("startswith:HICACHE_EVT s", EV), "HICACHE_EVT ntp_arrive ": ("search", ARR), "HICACHE_EVT ntp_admit ": ("search", ADM),
    "HICACHE_EVT srs_dispatch  [predwait]": ("search", DISP2),
}
print("per run: event | lines containing it | lines the loader's regex takes | missed; then samples of missed lines")
for label, run in RUNS.items():
    assert run != CURRENT
    seen, took, missed = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    rid_shapes = collections.Counter()
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            m = TS.match(line)
            body = line[m.end():] if m else None
            for key, (how, rx) in SUBSTR.items():
                word = key.split("  [")[0].split(" [")[0]
                if word not in line:
                    continue
                seen[key] += 1
                ok = False
                if body is not None:
                    if how.startswith("startswith:"):
                        ok = body.startswith(how.split(":", 1)[1]) and rx.match(body) is not None
                    else:
                        ok = rx.search(body if how == "search" and key in ("device prefix shrank", "reissue cap reached") else line) is not None
                if ok:
                    took[key] += 1
                elif len(missed[key]) < 3:
                    missed[key].append(line.rstrip("\n")[:260])
            if "rid=" in line:
                for r in re.findall(r"rid=([^\s,)]+)", line):
                    shape = "c-t" if re.search(r"-c\d+-t\d+$", r) else re.sub(r"[0-9a-f]{8,}", "<hex>", r)
                    rid_shapes[shape] += 1
    print(f"\n== {label}")
    for key in SUBSTR:
        if seen[key]:
            print(f"  {key:40s} {seen[key]:7d} {took[key]:7d} {seen[key] - took[key]:7d}")
            for s in missed[key]:
                print(f"      missed: {s}")
    print("  rid shapes: " + ", ".join(f"{k}: {v}" for k, v in rid_shapes.most_common(8)))
