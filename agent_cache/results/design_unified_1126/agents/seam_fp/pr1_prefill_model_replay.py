"""PR1 check (CPU, read-only): the prototype PrefillCostModel fed with a recorded run's admitted turns, the way the
engine would feed it at commit: new tokens = input_len - cached_input_len, total = input_len, seconds = client TTFT
minus the queue wait (the engine has prefill_finished_time - forward_entry_time). Three feeds:
  all        every finished request
  light      only requests admitted with no other request waiting (queue_duration < 50 ms)
  light,dev  light, and no host or storage load-back in the request (cached tokens all from the GPU)
Reported: the fitted a (us/token) and b (ns/token^2), and the model's cold-prefill time at 8K/24K/48K tokens
against the direct fit on cold prefills at an empty queue (f4_read_vs_recompute.py).
Usage: <venv python> pr1_prefill_model_replay.py <label> [minute_hi]
"""
import importlib.util, os, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import lib
spec = importlib.util.spec_from_file_location("next_turn_prices", os.path.join(lib.HERE, "proto", "next_turn_prices.py"))
prices = importlib.util.module_from_spec(spec); sys.modules["next_turn_prices"] = prices; spec.loader.exec_module(prices)

label = sys.argv[1]; hi = float(sys.argv[2]) if len(sys.argv) > 2 else 110.0
r = lib.Run(label)
rows = []
for rid, rec in r.requests():
    rts = rec["rts"]; cl = r.client.get(rid)
    if cl is None or cl.get("ttft") is None or r.minute(rts["entry"]) >= hi:
        continue
    det = cl.get("cached_details") or {}
    rows.append((rts["entry"], rts["input_len"] - rts["cached_input_len"], rts["input_len"], cl["ttft"] - rts["queue_s"],
                 rts["queue_s"], (det.get("host", 0) or 0) + (det.get("storage", 0) or 0)))
rows.sort()
print(f"== {label} to minute {hi:.0f}: {len(rows)} requests")
for name, keep in (("all", lambda x: True), ("light", lambda x: x[4] < 0.05), ("light,dev", lambda x: x[4] < 0.05 and x[5] == 0)):
    m = prices.PrefillCostModel(seed_a=1.1e-4)
    n = 0
    for _, new, total, sec, qd, hs in rows:
        if keep((_, new, total, sec, qd, hs)) and new > 0 and sec > 0:
            m.observe(new_tokens=new, total_tokens=total, seconds=sec); n += 1
    a, b = m.coefficients()
    print(f"   {name:10s} n {n:6d}: a {a*1e6:6.1f} us/token, b {b*1e9:6.3f} ns/token^2; cold 8K {m.seconds(new_tokens=8000,total_tokens=8000):.2f} s, "
          f"24K {m.seconds(new_tokens=24000,total_tokens=24000):.2f} s, 48K {m.seconds(new_tokens=48000,total_tokens=48000):.2f} s; "
          f"recompute 20K under a 4K new prompt {m.seconds(new_tokens=24000,total_tokens=24000)-m.seconds(new_tokens=4000,total_tokens=24000):.2f} s")
