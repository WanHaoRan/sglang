"""From one client.jsonl: (1) is the previous request's generated output ever part of the next
request's cached prefix (ideal_cached > previous prompt)? (2) per conversation, prompt positions ever
read from SSD (union of [device+host, device+host+storage)) against positions written once
(max prompt over the conversation, plus each request's output, page-aligned roughly).
Usage: python client_writes.py client.jsonl"""
import json, sys, collections, re
reqs = collections.defaultdict(list)
for line in open(sys.argv[1]):
    if '"kind":"req"' not in line: continue
    r = json.loads(line)
    m = re.match(r".*-c(\d+)-t(\d+)$", r["rid"])
    reqs[int(m[1])].append((int(m[2]), r))
out_reused = out_not = 0
pos_written = pos_read = 0
out_tok = 0
first_prompt = 0
read_tok_total = 0
convs_with_read = 0
for c, lst in reqs.items():
    lst.sort()
    prev = None
    ranges = []
    maxp = 0
    for t, r in lst:
        pt, ct = r["prompt_tokens"] or 0, r["completion_tokens"] or 0
        out_tok += ct
        maxp = max(maxp, pt)
        if t == 0: first_prompt += pt
        if prev is not None and r["ideal_cached"] is not None:
            if r["ideal_cached"] > prev["prompt_tokens"] + 16: out_reused += 1
            else: out_not += 1
        d = r.get("cached_details") or {}
        s = d.get("storage", 0) or 0
        if s > 0:
            a = (d.get("device", 0) or 0) + (d.get("host", 0) or 0)
            ranges.append((a, a + s)); read_tok_total += s
        prev = r
    pos_written += maxp
    if ranges:
        convs_with_read += 1
        ranges.sort(); cur_s, cur_e = ranges[0]; u = 0
        for s_, e_ in ranges[1:]:
            if s_ > cur_e: u += cur_e - cur_s; cur_s, cur_e = s_, e_
            else: cur_e = max(cur_e, e_)
        u += cur_e - cur_s
        pos_read += u
n = sum(len(v) for v in reqs.values())
print(f"conversations {len(reqs)}, requests {n}, with any SSD read {convs_with_read}")
print(f"next prompt's cached prefix covers the previous output: {out_reused} of {out_reused+out_not}")
print(f"output tokens generated (never in a later prompt if the line above is ~0): {out_tok/1e6:.2f} M")
print(f"prompt positions per conversation (max prompt, summed): {pos_written/1e6:.2f} M; first-request prompts {first_prompt/1e6:.2f} M")
print(f"distinct prompt positions ever read from SSD (union per conversation): {pos_read/1e6:.2f} M = {pos_read/pos_written:.1%} of positions")
print(f"SSD tokens delivered in total (with repeats): {read_tok_total/1e6:.2f} M = {read_tok_total/max(pos_read,1):.2f} x the distinct positions")
