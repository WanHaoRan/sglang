"""Is (conv, turn) the same request in every run? Compares, key by key against jit_exact: the trace session behind the
conv, the replayed prompt length, the output length asked for, the measured prompt tokens, the gap slept before the turn.
  nice -n 19 python3 -I -B pairing.py"""
import collections, os, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *

LABELS = ["baseline", "base_rep", "hintfree", "hintfree_ctl", "jit_p50", "jit_p50_ctl", "jit_p50_d_cut", "jit_exact", "ec_r13", "ec_r1", "nofc", "nofc_noend", "ORDER"]
ref = {(r["conv"], r["turn"]): r for r in raw("jit_exact")["client"]["turns"]}
print("arm: keys | in jit_exact too | task_conv differs | replay_prompt_tokens differs | output_length differs | prompt_tokens differs (max |diff|) | completion_tokens differs (max |diff|) | "
      "gap_slept differs by over 0.05 s (max |diff|) | turns whose t_send order inside the session is broken")
for a in LABELS:
    t = raw(a)["client"]["turns"]
    both = [(r, ref[(r["conv"], r["turn"])]) for r in t if (r["conv"], r["turn"]) in ref]
    dp = [abs((r["prompt_tokens"] or 0) - (q["prompt_tokens"] or 0)) for r, q in both]
    dc = [abs((r["completion_tokens"] or 0) - (q["completion_tokens"] or 0)) for r, q in both]
    dg = [abs((r["gap_slept"] or 0) - (q["gap_slept"] or 0)) for r, q in both]
    print(f"  {a:14s} {len(t):6d} | {len(both):6d} | {sum(r['task_conv'] != q['task_conv'] for r, q in both):4d} | {sum(r['replay_prompt_tokens'] != q['replay_prompt_tokens'] for r, q in both):4d} | "
          f"{sum(r['output_length'] != q['output_length'] for r, q in both):4d} | {sum(x > 0 for x in dp):5d} ({max(dp)}) | {sum(x > 0 for x in dc):5d} ({max(dc)}) | {sum(x > 0.05 for x in dg):5d} ({max(dg):.2f})")
    big = sorted(((abs((r["gap_slept"] or 0) - (q["gap_slept"] or 0)), r["conv"], r["turn"], r["gap_slept"], q["gap_slept"]) for r, q in both), reverse=True)[:3]
    if big[0][0] > 0.05:
        print("       largest gap differences (|diff|, conv, turn, here, jit_exact): " + "; ".join(f"{x[0]:.1f} c{x[1]}-t{x[2]} {x[3]} vs {x[4]}" for x in big))

print("\nsummary records and retries")
import json, glob
RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for a in LABELS:
    path = raw(a)["client"]["path"]
    summ = None
    retried = 0
    with open(path) as f:
        for line in f:
            if '"kind": "summary"' in line:
                summ = json.loads(line)
            elif '"retried"' in line or '"retry"' in line:
                retried += 1
    print(f"  {a:14s} summary: {({k: summ[k] for k in summ if k in ('elapsed_s', 'turns', 'errors', 'retries', 'capped_convs', 'mean_ttft_s')} if summ else None)}; turn lines mentioning a retry: {retried}")
