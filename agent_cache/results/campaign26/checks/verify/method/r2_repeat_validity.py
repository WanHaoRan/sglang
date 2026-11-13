"""Is ec_r13_rep a like-for-like repeat of ec_r13 (and jit_p50_d of jit_p50_ctl)? Server arguments, client arguments,
request pairing, errors, and the calm stretches of the run.
  nice -n 19 python3 -I -B r2_repeat_validity.py"""
import ast, json, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import raw, arm, rank, mean
from parse_cache_names import RES, CURRENT

RUN = {a: raw(a)["run"] for a in ("ec_r13", "ec_r13_rep", "jit_p50_ctl", "jit_p50_d", "jit_p50_d_cut")}


def sargs(run):
    text = open(os.path.join(RES, run, "server_args.txt"), errors="replace").read()
    i = text.index("server_args=") + len("server_args=")
    j = text.index("}", i)
    while True:
        try:
            return ast.literal_eval(text[i:j + 1])
        except (SyntaxError, ValueError):
            j = text.index("}", j + 1)


rk = rank()
for x, y in (("ec_r13", "ec_r13_rep"), ("jit_p50_ctl", "jit_p50_d"), ("jit_p50_d_cut", "jit_p50_d")):
    assert CURRENT not in (RUN[x], RUN[y])
    A, B = sargs(RUN[x]), sargs(RUN[y])
    diff = sorted(k for k in set(A) | set(B) if A.get(k, "<absent>") != B.get(k, "<absent>"))
    print(f"== {x} vs {y}")
    print(f"   server args that differ: {[(k, str(A.get(k, '<absent>'))[-40:], str(B.get(k, '<absent>'))[-40:]) for k in diff]}")
    ca, cb = raw(x)["client"], raw(y)["client"]
    da = {k: v for k, v in ca["run"]["args"].items()}
    db = {k: v for k, v in cb["run"]["args"].items()}
    print(f"   client args that differ: {[(k, da.get(k), db.get(k)) for k in sorted(set(da) | set(db)) if da.get(k) != db.get(k)]}")
    ta = {(r['conv'], r['turn']): r for r in ca["turns"]}
    tb = {(r['conv'], r['turn']): r for r in cb["turns"]}
    both = [k for k in ta if k in tb]
    print(f"   turns {len(ta)} and {len(tb)}, keys in both {len(both)}; errors {sum(bool(r['error']) for r in ca['turns'])} and {sum(bool(r['error']) for r in cb['turns'])}; ttft null {sum(r['ttft'] is None for r in ca['turns'])} and {sum(r['ttft'] is None for r in cb['turns'])}; "
          f"task_conv differs {sum(ta[k]['task_conv'] != tb[k]['task_conv'] for k in both)}, replay_prompt_tokens differs {sum(ta[k]['replay_prompt_tokens'] != tb[k]['replay_prompt_tokens'] for k in both)}, "
          f"output_length differs {sum(ta[k]['output_length'] != tb[k]['output_length'] for k in both)}, completion_tokens differs {sum(ta[k]['completion_tokens'] != tb[k]['completion_tokens'] for k in both)}, "
          f"gap_slept differs by over 0.05 s {sum(abs((ta[k]['gap_slept'] or 0) - (tb[k]['gap_slept'] or 0)) > 0.05 for k in both)}, max |prompt_tokens diff| {max(abs(ta[k]['prompt_tokens'] - tb[k]['prompt_tokens']) for k in both)}")
    sa, sb = raw(x)["server"], raw(y)["server"]
    print(f"   server log lines with Traceback/Exception/Error: {sa['census'].get('ts_lines_with_Traceback_Exception_Error', 0)} and {sb['census'].get('ts_lines_with_Traceback_Exception_Error', 0)}; "
          f"ReqTimeStats keys with several lines: {len(sa['rts']) - len({r['rid'] for r in sa['rts']})} and {len(sb['rts']) - len({r['rid'] for r in sb['rts']})}")
    a_, b_ = arm(x), arm(y)
    keys = [k for k in a_["by"] if k in b_["by"] and k[1] > 0]
    for lo, hi in ((0, 3000), (3000, 5500), (5500, 7000), (7000, 8750), (8750, 11000)):
        ks = [k for k in keys if lo <= rk.get(k, -1) < hi]
        if ks:
            dd = [b_["by"][k]["ttft"] - a_["by"][k]["ttft"] for k in ks]
            print(f"   positions {lo:5d}-{hi:5d}: {len(ks):5d} turns, {x} {mean(a_['by'][k]['ttft'] for k in ks):.3f} {y} {mean(b_['by'][k]['ttft'] for k in ks):.3f}, paired change {mean(dd):+.3f}, median |change| {st.median(abs(v) for v in dd):.3f}, "
                  f"median TTFT {st.median(a_['by'][k]['ttft'] for k in ks):.3f} and {st.median(b_['by'][k]['ttft'] for k in ks):.3f}")
