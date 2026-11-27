#!/bin/sh
# Rebuilds every extract and table of the ev_fetch checks (CPU only; about 5 minutes; run from anywhere).
# Reads the trace, the run folders' server.log and client.jsonl, and python/sglang/srt/mem_cache/next_turn_policy.py
# (imported by path). Writes only under this folder's extracts/ and out/.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
RES=/home/wanhr/sglang/agent_cache/results
PY="/home/wanhr/sglang/agent_cache/baselines/continuum/.venv/bin/python -I -B"
export CUDA_VISIBLE_DEVICES=
run() { nice -n 10 $PY "$@"; }
mkdir -p "$ROOT/extracts" "$ROOT/out"

# 1. hint shapes (trace)
nice -n 10 python3 -I -B "$HERE/extract_trace.py" /home/wanhr/sglang/agent_cache/data/converted/freeinference_16w.programs.jsonl.gz \
    "$ROOT/extracts/trace_compact.jsonl.gz" > "$ROOT/out/extract_trace.log"
for pop in A A_sys B; do
    run "$HERE/hint_shapes.py" $pop > "$ROOT/out/hint_shapes_$pop.log"
    run "$HERE/hint_shapes_report.py" $pop > "$ROOT/out/hint_shapes_$pop.txt"
done

# 2. value of reading before arrival (trace)
for mode in censored replay; do
    run "$HERE/fetch_value.py" A $mode
    run "$HERE/fetch_value_report.py" A $mode > "$ROOT/out/fetch_value_A_$mode.txt"
done
run "$HERE/fetch_value.py" B censored
run "$HERE/fetch_value_report.py" B censored > "$ROOT/out/fetch_value_B_censored.txt"

# runs: label, manifest, arm, "ec" when the arm stages by expected cost
: > "$ROOT/out/extract_run.log"
extract() {  # <label prefix> <campaign> <arm> [ec]
    set -- "$1" "$2" "$3" "$4" $(grep "^$3 " "$RES/$2/manifest.txt")
    run "$HERE/extract_run.py" "$1_$3" "$6" "$7" >> "$ROOT/out/extract_run.log"
    run "$HERE/read_table.py" "$1_$3" $4
}
extract c34 campaign34 ec_full ec
extract c34 campaign34 jit_exact
extract c34 campaign34 ec_exact ec
extract c34 campaign34 jit_p50
extract c26 campaign26 nofc_noend ec
extract c26 campaign26 nofc ec
extract c26 campaign26 ec_r13 ec
extract c26 campaign26 jit_exact_d
extract c26 campaign26 jit_p50_d
extract c33 campaign33 ec_full ec
extract c33 campaign33 jit_exact

# 2 on the runs, and the entries the client sent
run "$HERE/ceiling.py" c26_nofc_noend c26_nofc c26_ec_r13 c26_jit_exact_d c26_jit_p50_d c34_ec_full c34_jit_exact c34_ec_exact \
    c34_jit_p50 c33_ec_full c33_jit_exact > "$ROOT/out/ceiling.txt"
run "$HERE/hints_sent.py" c34_ec_full c34_jit_p50 c26_nofc_noend > "$ROOT/out/hints_sent.txt"
for x in c34_ec_full c34_jit_exact c26_nofc_noend c33_ec_full; do run "$HERE/regimes.py" $x > "$ROOT/out/regimes_$x.txt"; done

# 3. landing quantile (F1) and 4. the line (F3)
for x in "c34_ec_full ec" "c34_jit_exact" "c34_ec_exact ec" "c34_jit_p50"; do
    set -- $x
    run "$HERE/f1_landing.py" $x > "$ROOT/out/f1_landing_$1.txt"
    run "$HERE/f3_line.py" $x > "$ROOT/out/f3_line_$1.txt"
done
run "$HERE/f1_landing.py" c26_nofc_noend ec > "$ROOT/out/f1_landing_c26_nofc_noend.txt"
run "$HERE/f1_summary.py" c34_ec_full c34_jit_exact c34_ec_exact c34_jit_p50 > "$ROOT/out/f1_summary.txt"
run "$HERE/d_estimators.py" c34_ec_full c34_jit_exact > "$ROOT/out/d_estimators.txt"
