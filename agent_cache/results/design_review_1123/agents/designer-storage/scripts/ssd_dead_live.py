"""At given times, split the KV each conversation has created so far (max prompt so far + outputs, a proxy
for what write-through has sent to the SSD) into conversations that had sent their last request and
finished (dead for the SSD) and those still live; and the age of live conversations' first write.
Usage: python ssd_dead_live.py client.jsonl t1 t2 ..."""
import json, sys, re, collections, statistics as st
conv = collections.defaultdict(list)
for line in open(sys.argv[1]):
    if '"kind":"req"' not in line: continue
    r = json.loads(line); m = re.match(r".*-c(\d+)-t(\d+)$", r["rid"])
    if r.get("ttft") is None: continue
    conv[int(m[1])].append((r["t_send"], r["t_send"] + (r.get("latency") or 0.0), r["prompt_tokens"] or 0, r["completion_tokens"] or 0))
for T in map(float, sys.argv[2:]):
    dead = live = 0; ages = []; live_n = dead_n = 0
    for c, lst in conv.items():
        lst.sort()
        done = [x for x in lst if x[1] <= T]
        if not done: continue
        created = max(x[2] for x in done) + sum(x[3] for x in done)
        last_end = max(x[1] for x in lst)
        if last_end <= T: dead += created; dead_n += 1
        else: live += created; live_n += 1; ages.append(T - lst[0][0])
    print(f"t={T:6.0f} s: dead conversations {dead_n} hold {dead/1e6:5.2f} M tokens; live {live_n} hold {live/1e6:5.2f} M; "
          f"live first-write age p50 {st.median(ages)/60:5.1f} min, p90 {sorted(ages)[int(.9*len(ages))]/60:5.1f}, max {max(ages)/60:5.1f}")
