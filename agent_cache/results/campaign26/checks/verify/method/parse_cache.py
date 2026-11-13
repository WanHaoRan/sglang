"""Independent parse of each run's client.jsonl and server.log into a pickle under cache/.
Run from anywhere:  nice -n 19 python3 -I -B parse_cache.py [label ...]
Does not use the campaign's regexes: lines are split on the timestamp and fields are read as key=value tokens.
A census of line shapes is written to cache/<label>.census.json so that regex coverage can be compared later."""
import calendar, collections, glob, json, os, pickle, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CACHE = os.path.join(HERE, "cache")
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
    "ORDER": "20261101_102434_three_tier_wc_oracle_v2_nu_custom_L1nat_H70",
}
CURRENT = open(os.path.join(RES, ".current_run")).read().strip()
TS = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})\] ")
TS_ANY = re.compile(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d{3}\] ")
RID_CT = re.compile(r"-c(\d+)-t(\d+)$")
KV = re.compile(r"(\w+)=(\S+)")
DAY = {}


def epoch(m):
    d = (m[1], m[2], m[3])
    if d not in DAY:
        DAY[d] = calendar.timegm((int(m[1]), int(m[2]), int(m[3]), 0, 0, 0))
    return DAY[d] + int(m[4]) * 3600 + int(m[5]) * 60 + int(m[6]) + int(m[7]) / 1000


def num(s):
    s = s.rstrip(",)")
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def parse_client(run):
    paths = sorted(glob.glob(os.path.join(RES, run, "client_*", "client.jsonl")))
    path = paths[-1]
    kinds, bad, turns, run_rec = collections.Counter(), 0, [], None
    raw = open(path, "rb").read()
    ends_nl = raw.endswith(b"\n")
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        kinds[r.get("kind")] += 1
        if r.get("kind") == "run":
            run_rec = r
        elif r.get("kind") == "turn":
            cd = r.get("cached_details") or {}
            turns.append(dict(conv=r["conv"], turn=r["turn"], t_send=r.get("t_send"), ttft=r.get("ttft"), latency=r.get("latency"),
                              prompt_tokens=r.get("prompt_tokens"), completion_tokens=r.get("completion_tokens"), cached_tokens=r.get("cached_tokens"),
                              reply_reprefilled=r.get("reply_reprefilled"), storage=cd.get("storage"), host=cd.get("host"), device=cd.get("device"),
                              has_cd=r.get("cached_details") is not None, gap_slept=r.get("gap_slept"), error=r.get("error"), task_conv=r.get("task_conv"),
                              replay_prompt_tokens=r.get("replay_prompt_tokens"), output_length=r.get("output_length"),
                              true_gap=(r.get("dist_hint") or {}).get("true_gap_s") if isinstance(r.get("dist_hint"), dict) else None))
    return dict(path=path, n_client_dirs=len(paths), kinds=dict(kinds), bad_lines=bad, ends_with_newline=ends_nl, run=run_rec, turns=turns)


def parse_server(run):
    census = collections.Counter()
    evt_names = collections.Counter()
    rts, evts, dec, pre, misc = [], collections.defaultdict(list), [], [], collections.defaultdict(list)
    odd = collections.defaultdict(list)
    with open(os.path.join(RES, run, "server.log"), errors="replace") as f:
        for line in f:
            census["lines"] += 1
            m = TS.match(line)
            n_ts = len(TS_ANY.findall(line))
            if n_ts > 1:
                census["lines_with_several_timestamps"] += 1
                if len(odd["several_ts"]) < 5:
                    odd["several_ts"].append(line[:300])
            if not m:
                census["lines_without_leading_timestamp"] += 1
                for word in ("ReqTimeStats", "HICACHE_EVT", "Decode batch", "Prefill batch", "Traceback"):
                    if word in line:
                        census["no_ts_but_has_" + word] += 1
                        if len(odd["no_ts_" + word]) < 5:
                            odd["no_ts_" + word].append(line[:300])
                continue
            t = epoch(m)
            body = line[m.end():].rstrip("\n")
            if "Traceback" in body or "Exception" in body or "Error" in body:
                census["ts_lines_with_Traceback_Exception_Error"] += 1
                if len(odd["errors"]) < 8:
                    odd["errors"].append(line[:300])
            i = body.find("ReqTimeStats(")
            if i >= 0:
                census["ReqTimeStats_any"] += 1
                if i > 0:
                    census["ReqTimeStats_not_at_start"] += 1
                    if len(odd["rts_not_at_start"]) < 5:
                        odd["rts_not_at_start"].append(line[:300])
                seg = body[i:]
                rid = re.match(r"ReqTimeStats\(rid=([^,]*),", seg)
                kv = {k: num(v.rstrip("ms").rstrip(",")) if k.endswith("_duration") else num(v) for k, v in KV.findall(seg)}
                # durations carry a "ms" suffix; parse them explicitly so nothing is truncated
                dur = {k: v for k, v in re.findall(r"(\w+_duration)=(-?[\d.eE+-]+)ms", seg)}
                c = RID_CT.search(rid[1]) if rid else None
                rts.append(dict(t=t, rid=rid[1] if rid else None, conv=int(c[1]) if c else None, turn=int(c[2]) if c else None,
                                queue_ms=float(dur["queue_duration"]) if "queue_duration" in dur else None,
                                forward_ms=float(dur["forward_duration"]) if "forward_duration" in dur else None,
                                entry_time=kv.get("entry_time"), input_len=kv.get("input_len"), cached_input_len=kv.get("cached_input_len"),
                                output_len=kv.get("output_len"), attempts=kv.get("attempts"), raw_queue=re.search(r"queue_duration=(\S+)", seg)[1] if "queue_duration=" in seg else None))
                continue
            i = body.find("HICACHE_EVT ")
            if i >= 0:
                census["HICACHE_EVT_any"] += 1
                if i > 0:
                    census["HICACHE_EVT_not_at_start"] += 1
                    if len(odd["evt_not_at_start"]) < 5:
                        odd["evt_not_at_start"].append(line[:300])
                seg = body[i + len("HICACHE_EVT "):]
                name = seg.split(" ", 1)[0]
                evt_names[name] += 1
                kv = dict(KV.findall(seg))
                rid = kv.get("rid")
                c = RID_CT.search(rid) if rid else None
                rec = dict(t=t, rid=rid, conv=int(c[1]) if c else None, turn=int(c[2]) if c else None, raw=seg)
                for k, v in kv.items():
                    if k != "rid":
                        rec[k] = num(v)
                evts[name].append(rec)
                continue
            if body.startswith("Decode batch"):
                census["Decode_batch"] += 1
                d = re.match(r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), (.*)$", body)
                thr = re.search(r"gen throughput \(token/s\): ([\d.]+)", body)
                q = re.search(r"#queue-req: (\d+)", body)
                if d and thr and q:
                    dec.append((t, int(d[1]), int(d[2]), float(d[3]), float(thr[1]), int(q[1])))
                else:
                    census["Decode_batch_unparsed"] += 1
                    if len(odd["dec_unparsed"]) < 5:
                        odd["dec_unparsed"].append(line[:300])
                continue
            if "Decode batch" in body:
                census["Decode_batch_not_at_start"] += 1
            if body.startswith("Prefill batch"):
                census["Prefill_batch"] += 1
                p = re.match(r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)", body)
                if p:
                    pre.append((t, int(p[1]), int(p[2]), int(p[3])))
                else:
                    census["Prefill_batch_unparsed"] += 1
                    if len(odd["pre_unparsed"]) < 5:
                        odd["pre_unparsed"].append(line[:300])
                continue
            if "Prefill batch" in body:
                census["Prefill_batch_not_at_start"] += 1
            for word, key in (("HiCache prefetch success", "succ"), ("device prefix shrank", "shrank"), ("reissue cap reached", "cap")):
                if word in body:
                    census[key] += 1
                    misc[key].append((t, body))
    return dict(census=dict(census), evt_names=dict(evt_names), rts=rts, evts=dict(evts), dec=dec, pre=pre, misc=dict(misc), odd=dict(odd))


def main():
    os.makedirs(CACHE, exist_ok=True)
    labels = sys.argv[1:] or list(RUNS)
    for label in labels:
        run = RUNS[label]
        assert run != CURRENT, "refusing to read the run being written"
        t0 = time.time()
        out = dict(label=label, run=run, client=parse_client(run))
        if label != "ORDER":
            out["server"] = parse_server(run)
        with open(os.path.join(CACHE, label + ".pkl"), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        c = out["client"]
        census = dict(label=label, run=run, client_kinds=c["kinds"], client_bad_lines=c["bad_lines"], client_ends_with_newline=c["ends_with_newline"],
                      n_client_dirs=c["n_client_dirs"])
        if "server" in out:
            s = out["server"]
            census.update(server=s["census"], evt_names=s["evt_names"], odd=s["odd"], n_rts=len(s["rts"]),
                          n_rts_with_conv=sum(1 for r in s["rts"] if r["conv"] is not None))
        with open(os.path.join(CACHE, label + ".census.json"), "w") as f:
            json.dump(census, f, indent=1, sort_keys=True)
        print(f"{label:14s} {run}  client turns {c['kinds'].get('turn', 0):6d}  bad lines {c['bad_lines']}  "
              + (f"server lines {out['server']['census']['lines']:7d}  ReqTimeStats {len(out['server']['rts']):6d}  " if 'server' in out else "")
              + f"{time.time() - t0:5.1f} s", flush=True)


main()
