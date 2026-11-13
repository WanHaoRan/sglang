"""Parse the two runs that landed after the write-up's last entry (manifest.txt: jit_p50_d, ec_r13_rep) into cache/, with
parse_cache.py's functions. They are finished runs; the run named in results/.current_run is refused.
  nice -n 19 python3 -I -B r2_parse_extra.py"""
import json, os, pickle, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "parse_cache.py")).read()
ns = {"__name__": "parse_cache_functions", "__file__": os.path.join(HERE, "parse_cache.py")}
exec(compile(src[:src.index("\ndef main():")], "parse_cache.py", "exec"), ns)
EXTRA = {"jit_p50_d": "20261111_204059_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70",
         "ec_r13_rep": "20261111_224624_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H70"}
manifest = dict(line.split()[:2] for line in open(os.path.join(ns["RES"], "campaign26", "manifest.txt")) if line.strip())
for label, run in EXTRA.items():
    assert manifest[label] == run, (label, manifest.get(label))
    assert run != ns["CURRENT"], "refusing to read the run being written"
    t0 = time.time()
    out = dict(label=label, run=run, client=ns["parse_client"](run), server=ns["parse_server"](run))
    with open(os.path.join(ns["CACHE"], label + ".pkl"), "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    c, s = out["client"], out["server"]
    print(f"{label:12s} {run}  client kinds {c['kinds']}  bad lines {c['bad_lines']}  ends with newline {c['ends_with_newline']}  server lines {s['census']['lines']}  "
          f"ReqTimeStats {len(s['rts'])}  events {({k: len(v) for k, v in sorted(s['evts'].items()) if k in ('s2h_io', 'ntp_stage', 'srs_dispatch', 'ec_start', 'ec_release', 'ec_price')})}  "
          f"census {s['census']}  {time.time() - t0:.1f} s")
    print("   run.txt: " + " | ".join(l.strip() for l in open(os.path.join(ns["RES"], run, "run.txt")) if l.split("=")[0] in ("engine", "started_utc", "server_extra")))
