"""Server arguments recorded by each run (server_args.txt): which keys differ between runs that the write-up reads as the
same arm, and between an arm and its control.
  nice -n 19 python3 -I -B r2_args_diff.py"""
import ast, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_cache_names import RUNS, RES, CURRENT


def args(a):
    assert RUNS[a] != CURRENT
    text = open(os.path.join(RES, RUNS[a], "server_args.txt"), errors="replace").read()
    i = text.index("server_args=") + len("server_args=")
    j = text.index("}", i)
    # the dict may hold nested braces in strings: find the matching close by literal_eval on growing prefixes
    while True:
        try:
            return ast.literal_eval(text[i:j + 1])
        except (SyntaxError, ValueError):
            j = text.index("}", j + 1)


A = {a: args(a) for a in RUNS}
SKIP = {"port", "api_key", "log_requests"}
for x, y in (("jit_p50", "jit_p50_ctl"), ("jit_p50_ctl", "jit_p50_d_cut"), ("jit_p50", "jit_exact"), ("baseline", "base_rep"), ("hintfree", "hintfree_ctl"), ("baseline", "hintfree"),
             ("jit_p50_d_cut", "ec_r1"), ("ec_r1", "ec_r13"), ("ec_r13", "nofc"), ("nofc", "nofc_noend"), ("ORDER", "jit_exact"), ("hintfree_ctl", "nofc_noend")):
    keys = sorted(set(A[x]) | set(A[y]))
    diff = [(k, A[x].get(k, "<absent>"), A[y].get(k, "<absent>")) for k in keys if A[x].get(k, "<absent>") != A[y].get(k, "<absent>") and k not in SKIP]
    print(f"\n== {x} vs {y}: {len(A[x])} and {len(A[y])} keys, {len(diff)} differ")
    for k, u, v in diff:
        print(f"   {k}: {str(u)[:150]}  |  {str(v)[:150]}")
