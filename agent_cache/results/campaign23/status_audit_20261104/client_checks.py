import sys, json, collections
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
from gap_hints import command_word, reply_key, tool_category
for c in ["export X=1 && python -m pytest -q", "timeout 60 python -m pytest", "# run the tests\npython -m pytest", "cd /repo && git status", "FOO=1 pytest"]:
    print(repr(c), "->", command_word(c), "->", reply_key("execute_bash", {"command": c}))
trace = "/home/wanhr/sglang/agent_cache/traces/lmcache_agentic_trace_262k.json"
data = json.load(open(trace))
keys = json.load(open(trace + ".reply_keys.json"))["conversations"]
kc = collections.Counter(k for ks in keys for k in ks if k is not None)
print("bare tool keys:", {k: v for k, v in kc.items() if "/" not in k and k in ("execute_bash", "bash", "exec", "execute_ipython_cell")})
print("export/timeout keys:", {k: v for k, v in kc.items() if k.split("/")[-1] in ("export", "timeout", "#")})
# rejections whose header names no tool: category exactly "error"
n_err_no_tool = 0; got = collections.Counter()
for ci, conv in enumerate(data["conversations"]):
    for t in range(1, len(conv)):
        if tool_category(conv[t]) == "error":
            n_err_no_tool += 1
            got[(keys[ci][t-1] or "None").split("/")[0]] += 1
print("rejections with no tool in the header:", n_err_no_tool, "keys given, by tool:", dict(got))
