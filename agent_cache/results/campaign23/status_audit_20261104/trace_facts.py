import sys, json, collections, re
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
from gap_hints import task_of, tool_category, command_word, CALL_CAP, scaled_gap
trace = "/home/wanhr/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json"
data = json.load(open(trace))
keys = json.load(open(trace + ".reply_keys.json"))["conversations"]
sess = data["sessions"]; convs = data["conversations"]
print("conversations", len(convs), "non-empty", sum(1 for c in convs if c))
systems = collections.Counter((m["source"], m["orig_model"]) for m, c in zip(sess, convs) if c)
print("agent systems (source, model):", dict(systems))
# keys
kc = collections.Counter(k for ks in keys for k in ks if k is not None)
print("distinct keys", len(kc), "top 25:", kc.most_common(25))
print("keys containing pytest:", {k: v for k, v in kc.items() if "pytest" in k})
# call caps
calls = [m.get("n_turns_total") or len(c) for m, c in zip(sess, convs) if c]
lens = [len(c) for c in convs if c]
print("sessions with n_turns_total == 50:", sum(1 for n in calls if n == CALL_CAP), "> 50:", sum(1 for n in calls if n > CALL_CAP),
      "cut by converter (n_turns_total > len):", sum(1 for m, c in zip(sess, convs) if c and (m.get("n_turns_total") or len(c)) > len(c)))
# ids
ids = [m["session_id"] for m, c in zip(sess, convs) if c]
print("ids with __run:", [i for i in ids if "__run" in i][:10], sum("__run" in i for i in ids))
print("ids with __chain:", sum("__chain" in i for i in ids))
tasks = collections.Counter(task_of(i) for i in ids)
print("tasks:", len(tasks), "tasks with >1 session:", sum(1 for v in tasks.values() if v > 1))
# examples of id shapes
shapes = collections.Counter(len(i.split("__")) for i in ids)
print("id part counts:", dict(shapes))
for n in sorted(shapes):
    print("  e.g.", n, [i for i in ids if len(i.split("__")) == n][:3])
# gap stats at x70 and x10
for scale in (70, 10):
    gs = [scaled_gap(t, scale=scale) for c in convs for t in c[1:]]
    gs.sort()
    print(f"x{scale}: gaps {len(gs)} median {gs[len(gs)//2]:.1f} mean {sum(gs)/len(gs):.1f} share>=300 {sum(g>=300 for g in gs)/len(gs):.3%} share<60 {sum(g<60 for g in gs)/len(gs):.3%} idle share >=300 {sum(g for g in gs if g>=300)/sum(gs):.3%}")
