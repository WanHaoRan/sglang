import sys, os, json
sys.path.insert(0, "/home/wanhr/sglang/agent_cache/scripts")
import test_gap_hints as t
from gap_hints import check_knobs, load_reply_keys, reply_keys_path
d = sys.argv[1]
trace = os.path.join(d, "trace.json")
json.dump({"conversations": [[{}, {}]], "sessions": [{}]}, open(trace, "w"))
json.dump({"conversations": [["a", None, None]]}, open(reply_keys_path(trace), "w"))   # wrong shape
check_knobs({}, trace=trace)
print("launcher pre-flight (check_knobs) passes a wrong-shaped keys file")
try:
    load_reply_keys(trace, [[{}, {}]])
except ValueError as e:
    print("the client then stops at load:", e)
replay = t.import_replay()
args = t.parse(replay, ["--trace", "x", "--output", "y", "--hint-source", "session", "--hint-end", "none"])
print("--hint-* without --hints dist:", "hints =", args.hints, "hint_source =", args.hint_source, "(no error)")
