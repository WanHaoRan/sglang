"""Sanity checks of the page-key model against one arm's client records."""
import sys, collections
sys.path.insert(0, __import__("os").path.dirname(__file__))
from lib import Run, PAGE, load_trees

camp, label = sys.argv[1], sys.argv[2]
run = Run(camp, label)
n = len(run.reqs)
print(f"{camp} {label}: {n} requests, {len(run.sessions)} sessions, {len(run.plays)} plays")
bad_len = 0; tails = collections.Counter()
for r in run.reqs.values():
    tree_pages = len(r.pages)
    if r.P // PAGE != tree_pages:
        bad_len += 1
    tails[r.P - tree_pages * PAGE >= 0] += 1
print("requests whose floor64(prompt_tokens)/64 != whole block pages:", bad_len)
# cached must be page aligned and <= pages
na = sum(1 for r in run.reqs.values() if r.cached % PAGE)
print("cached_tokens not page aligned:", na)
# ideal_cached vs model: prefix pages already written by any earlier-finished request in same scope
errs = sum(1 for r in run.reqs.values() if r.error)
print("errors:", errs, " without ttft:", sum(1 for r in run.reqs.values() if r.ttft is None))
# sessions: last flags
last_sent = sum(1 for s in run.sessions.values() if s[-1].last)
print(f"sessions whose final request (of the trace stream) was sent: {last_sent} of {len(run.sessions)}")
ks = [len(s) for s in run.sessions.values()]
print("requests per session: mean %.1f max %d" % (sum(ks)/len(ks), max(ks)))
roles = collections.Counter(s[0].role for s in run.sessions.values())
print("session roles:", dict(roles))
# keep < previous blocks?  (prompt truncation inside a session)
trunc = 0; tot = 0; lost_pages = 0
for s in run.sessions.values():
    for a, b in zip(s, s[1:]):
        tot += 1
        # common page prefix
        i = 0
        la, lb = a.pages, b.pages
        m = min(len(la), len(lb))
        lo, hi = 0, m
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if la[mid-1] == lb[mid-1]: lo = mid
            else: hi = mid - 1
        if lo < len(la):
            trunc += 1; lost_pages += len(la) - lo
print(f"consecutive pairs where the next prompt drops pages of the previous prompt: {trunc} of {tot}; pages dropped {lost_pages} ({lost_pages*64/1e6:.2f} M tokens)")
print("gap kinds:", dict(collections.Counter(r.gap for r in run.reqs.values())))
print("t_send max %.0f; t_done max %.0f" % (max(r.t_send for r in run.reqs.values()), max(r.t_done or 0 for r in run.reqs.values())))
