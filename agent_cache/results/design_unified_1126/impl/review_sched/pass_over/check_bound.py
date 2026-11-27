"""CPU check of the AdmissionPassOver bound: how many times one blocked request is passed.

Mirrors the scheduler walk's calls (scheduler.py _get_new_batch_prefill_raw): exhausted at the
top, admitted() for an added candidate, refused() for a NO_TOKEN refusal not added.
"""

import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    MAX_PASSES,
    AdmissionPassOver,
)


def walk(pass_over, queue, fits):
    admitted = []
    pass_over.begin()
    for rid in queue:
        if pass_over.exhausted:
            break
        if rid in fits:
            admitted.append(rid)
            pass_over.admitted(rid)
            continue
        if not pass_over.refused(rid):
            break
    pass_over.end()
    return admitted


for lookahead in (1, 3, 4, 5, 7, 8, 16):
    po = AdmissionPassOver(lookahead=lookahead)
    total = 0
    n = 0
    while True:
        small = [f"s{n}_{i}" for i in range(lookahead + 4)]
        got = walk(po, ["big", *small], fits=set(small))
        n += 1
        total += len(got)
        if not got:
            break
    print(
        f"lookahead={lookahead:2d} MAX_PASSES={MAX_PASSES}: big passed {total} times "
        f"(passes[big]={po.passes.get('big')}) over {n - 1} walks before the walk stopped at it"
    )

# A request refused behind the blocked one is passed without a count of its own.
po = AdmissionPassOver(lookahead=4)
for k in range(3):
    walk(po, ["big", "big2", f"a{k}", f"b{k}", f"c{k}"], fits={f"a{k}", f"b{k}", f"c{k}"})
print("refused behind the head, passed 9 times:", {"big2": po.passes.get("big2", 0)}, po.passes)
