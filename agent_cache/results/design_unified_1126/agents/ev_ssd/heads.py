"""Workload facts behind the SSD-tier orders: where pages are shared across sessions.
Usage: python3 -I -B heads.py <arm>"""
import sys, collections, math
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/results/design_unified_1126/agents/ev_ssd")
from replay import *
arm = sys.argv[1]
d = load_arm(arm)
reqs = d["reqs"]; npp = d["n_prompt_pages"]
sharing = page_sharing(d); first = first_requests(d)
by_conv = collections.defaultdict(list)
for r in reqs: by_conv[r.conv].append(r)
n_conv = len(by_conv)
# pages by class
in_first = bytearray(d["n_ids"])
for c, r0 in first.items():
    for p in r0.pages: in_first[p] = 1
tot_prompt = npp; tot_out = d["n_ids"] - npp
first_pages = sum(in_first)
shared_pages = sum(1 for p in range(npp) if sharing[p] >= 2)
shared_in_first = sum(1 for p in range(npp) if sharing[p] >= 2 and in_first[p])
print(f"{arm}: sessions {n_conv}; distinct pages: prompt {tot_prompt} ({tot_prompt*64/1e6:.1f} M tokens), output {tot_out} ({tot_out*64/1e6:.1f} M)")
print(f"  pages in some session's first prompt: {first_pages} ({first_pages/(tot_prompt+tot_out):.1%} of all pages); "
      f"pages held by >= 2 sessions: {shared_pages} ({shared_pages/(tot_prompt+tot_out):.1%}); of them in a first prompt: {shared_in_first} ({shared_in_first/max(1,shared_pages):.1%})")
# first requests: how much is shared with earlier sessions, and were those sessions over when this one started?
end_of = {c: max(r.t_end for r in rs) for c, rs in by_conv.items()}
ended_flag = {c: any(r.ended for r in rs) for c, rs in by_conv.items()}
start_of = {c: min(r.t_send for r in rs) for c, rs in by_conv.items()}
page_users = collections.defaultdict(set)
for r in reqs:
    pass
# users of each page so far, built in send order
users = [None] * npp
n_sh = tok_sh = tok_first = 0; tok_sh_all_ended = 0; n_sh_all_ended = 0; n_first = 0
frac = []
for r in reqs:   # send order
    if r is first[r.conv]:
        n_first += 1
        tok_first += r.q * 64
        h = min(r.H, r.q)
        if h >= 1:
            n_sh += 1; tok_sh += h * 64
            frac.append(h / max(1, r.q))
            # sessions that used page h-1 (the deepest shared page) before: any still open at this send?
            us = users[r.pages[h - 1]] or ()
            open_now = [u for u in us if end_of[u] > r.t_send]
            if not open_now:
                n_sh_all_ended += 1; tok_sh_all_ended += h * 64
    for p in r.pages:
        u = users[p]
        if u is None: users[p] = {r.conv}
        elif r.conv not in u: u.add(r.conv)
frac.sort()
print(f"  first requests: {n_first}, {tok_first/1e6:.2f} M prompt tokens; {n_sh} share >= 1 page with earlier sessions ({tok_sh/1e6:.2f} M tokens, "
      f"median share of the first prompt {frac[len(frac)//2]:.0%}); for {n_sh_all_ended} of them ({tok_sh_all_ended/1e6:.2f} M tokens) every earlier "
      f"session holding the deepest shared page was over when the new one started")
# what the SSD actually gave first requests in the recorded run
st_first = sum(r.storage_act for r in reqs if r is first[r.conv]) * 64
st_ret = sum(r.storage_act for r in reqs if r is not first[r.conv]) * 64
print(f"  recorded: tokens served from the SSD to first requests {st_first/1e6:.2f} M, to returning requests {st_ret/1e6:.2f} M")
# does sharing reach past the earlier session's first prompt?  pages shared by >= 2 sessions that are not in any first prompt
print(f"  shared pages outside every first prompt: {shared_pages - shared_in_first} ({(shared_pages - shared_in_first)*64/1e6:.2f} M tokens)")
# sessions: first prompt pages vs all pages a session's requests hold
fp = sorted(first[c].q for c in by_conv); allp = sorted(max(r.q for r in rs) for c, rs in by_conv.items())
print(f"  per session: first prompt pages p50 {fp[len(fp)//2]}, mean {sum(fp)/len(fp):.0f}; longest prompt pages p50 {allp[len(allp)//2]}, mean {sum(allp)/len(allp):.0f}")
# live pages over time: pages with a future use, and pages of open sessions' latest chains
for T in (1800, 3600, 5263, 6000, 6589):
    nu = set()
    for r in reqs:
        if r.t_pre > T:
            nu.update(r.pages)
    open_latest = set(); n_open = 0
    for c, rs in by_conv.items():
        if start_of[c] <= T < end_of[c]:
            n_open += 1
            last = max((r for r in rs if r.t_send <= T), key=lambda r: r.t_send)
            open_latest.update(last.pages)
    print(f"  t={T:5d} s: pages some later request holds {len(nu)} ({len(nu)*64/1e6:.2f} M tokens); open sessions {n_open}, their latest chains {len(open_latest)} pages ({len(open_latest)*64/1e6:.2f} M)")
